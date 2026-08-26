import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from itertools import combinations
import math

st.title("DFS Lineup Duplication Calculator")

# ==========================================
# Initialize session state for stored results
# ==========================================
if "df_out" not in st.session_state:
    st.session_state.df_out = None
if "is_showdown" not in st.session_state:
    st.session_state.is_showdown = None
if "fighter_cols" not in st.session_state:
    st.session_state.fighter_cols = None
if "own_df" not in st.session_state:
    st.session_state.own_df = None
if "id_to_name" not in st.session_state:
    st.session_state.id_to_name = {}


# ======================
# File Uploads & Inputs
# ======================
lineup_file = st.file_uploader("Upload Lineups CSV", type=["csv"])
own_file = st.file_uploader("Upload Ownership CSV", type=["csv"])
salary_file = st.file_uploader(
    "Upload DK Salaries CSV (required for PGA and Showdown)",
    type=["csv"],
)
contest_size = st.number_input("Contest Size", min_value=1, value=73529)

pga_concentration = st.slider(
    "PGA optimizer concentration",
    min_value=0.0,
    max_value=1.5,
    value=0.65,
    step=0.05,
    help=(
        "Controls how strongly the PGA field is concentrated around high-projection, "
        "salary-efficient lineups after matching individual golfer ownership. "
        "0 = maximum-entropy field; 0.65 is the recommended large-field default."
    ),
)


# -------------
# Helpers
# -------------
def salary_multiplier(s):
    try:
        s = float(s)
    except Exception:
        return 1.0
    if s >= 50000:
        return 1.75
    if s >= 49900:
        return 1.30
    if s >= 49800:
        return 1.00
    if s >= 49700:
        return 0.80
    return 0.60


def extract_id(x):
    s = str(x)
    m = re.search(r"\((\d+)\)", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", s)
    if m2:
        return int(m2.group(1))
    return np.nan


def build_pga_field_model(
    lineups,
    golfer_cols,
    own,
    salary_df,
    contest_size,
    concentration=0.65,
    exact_combo_limit=800000,
    sampled_candidate_limit=180000,
    random_seed=20260826,
):
    """
    Estimate exact-lineup probabilities for PGA.

    Model:
      1) Build the legal DK lineup universe exactly when practical. On larger
         slates, build a large ownership-weighted candidate universe and always
         include every uploaded lineup.
      2) Give high-projection / salary-efficient lineups a modest base preference.
      3) Iteratively rake lineup probabilities so final golfer marginals match
         the supplied ownership projections.
      4) Projected copies = contest_size * exact-lineup probability.

    This avoids the incorrect independence assumption of multiplying six golfer
    ownership percentages together.
    """

    if "DFS ID" not in own.columns or "Ownership" not in own.columns:
        raise ValueError("PGA ownership file needs 'DFS ID' and 'Ownership' columns.")

    if "ID" not in salary_df.columns or "Salary" not in salary_df.columns:
        raise ValueError("PGA DK Salaries file needs 'ID' and 'Salary' columns.")

    own_work = own.copy()
    own_work["DFS ID"] = pd.to_numeric(own_work["DFS ID"], errors="coerce")
    own_work["Ownership"] = pd.to_numeric(own_work["Ownership"], errors="coerce")
    own_work = own_work.dropna(subset=["DFS ID", "Ownership"]).copy()
    own_work["DFS ID"] = own_work["DFS ID"].astype(int)
    own_work = own_work.drop_duplicates("DFS ID", keep="first")

    sal_work = salary_df.copy()
    sal_work["ID"] = pd.to_numeric(sal_work["ID"], errors="coerce")
    sal_work["Salary"] = pd.to_numeric(sal_work["Salary"], errors="coerce")
    sal_work = sal_work.dropna(subset=["ID", "Salary"]).copy()
    sal_work["ID"] = sal_work["ID"].astype(int)
    sal_work = sal_work.drop_duplicates("ID", keep="first")

    pool = own_work.merge(
        sal_work[["ID", "Salary"]].rename(columns={"ID": "DFS ID"}),
        on="DFS ID",
        how="inner",
    )

    if len(pool) < 6:
        raise ValueError(
            "Could not match at least 6 golfers between ownership and DK Salaries files."
        )

    # Prefer the projection supplied with the ownership file.
    proj_col = None
    for candidate in ["fpts", "FPTS", "Proj", "Projection", "Proj Score", "Median"]:
        if candidate in pool.columns:
            proj_col = candidate
            break

    if proj_col is None:
        # Projection is useful for concentration but not required.
        pool["_pga_proj"] = 0.0
        proj_col = "_pga_proj"
    else:
        pool[proj_col] = pd.to_numeric(pool[proj_col], errors="coerce").fillna(0.0)

    player_ids = pool["DFS ID"].astype(int).to_numpy()
    salaries = pool["Salary"].astype(float).to_numpy()
    player_proj = pool[proj_col].astype(float).to_numpy()
    targets = np.clip(pool["Ownership"].astype(float).to_numpy() / 100.0, 1e-6, 0.999999)

    # Ownership files sometimes total 599.9% or 600.1% because of rounding.
    # Normalize to exactly six roster spots.
    if targets.sum() > 0:
        targets = np.clip(targets * (6.0 / targets.sum()), 1e-6, 0.999999)

    id_to_idx = {int(pid): i for i, pid in enumerate(player_ids)}
    n_players = len(player_ids)

    # Convert uploaded lineups to canonical player-index tuples so they are
    # guaranteed to exist in the sampled universe.
    uploaded_idx_keys = []
    for _, row in lineups.iterrows():
        try:
            ids = [int(row[c]) for c in golfer_cols]
            idxs = tuple(sorted(id_to_idx[x] for x in ids))
            if len(set(idxs)) == 6:
                uploaded_idx_keys.append(idxs)
            else:
                uploaded_idx_keys.append(None)
        except Exception:
            uploaded_idx_keys.append(None)

    total_combos = math.comb(n_players, 6)
    candidates = []
    exact_universe = total_combos <= exact_combo_limit

    if exact_universe:
        # Exact enumeration is practical for short fields such as the TOUR Championship.
        for c in combinations(range(n_players), 6):
            if salaries[list(c)].sum() <= 50000:
                candidates.append(c)
    else:
        # For full PGA fields, build a large plausible candidate universe.
        # Gumbel top-k lets us sample six unique golfers efficiently.
        candidate_set = set()
        for key in uploaded_idx_keys:
            if key is not None and salaries[list(key)].sum() <= 50000:
                candidate_set.add(key)

        rng = np.random.default_rng(random_seed)
        sample_weight = np.maximum(targets, 1e-6).copy()

        # Add a mild player-level projection tilt when generating candidates.
        if np.std(player_proj) > 1e-9:
            pz = (player_proj - np.mean(player_proj)) / np.std(player_proj)
            sample_weight *= np.exp(0.15 * float(concentration) * pz)

        sample_prob = sample_weight / sample_weight.sum()
        logp = np.log(np.maximum(sample_prob, 1e-12))

        target_count = int(
            min(
                sampled_candidate_limit,
                max(60000, min(sampled_candidate_limit, int(contest_size) * 2)),
            )
        )

        max_batches = 120
        batch_size = 3000

        for _ in range(max_batches):
            if len(candidate_set) >= target_count:
                break

            g = rng.gumbel(size=(batch_size, n_players))
            scores = g + logp[None, :]
            picks = np.argpartition(scores, -6, axis=1)[:, -6:]
            picks.sort(axis=1)

            legal = salaries[picks].sum(axis=1) <= 50000
            for arr in picks[legal]:
                candidate_set.add(tuple(int(x) for x in arr))
                if len(candidate_set) >= target_count:
                    break

        candidates = list(candidate_set)

    if not candidates:
        raise ValueError("No legal PGA lineups could be generated from these files.")

    C = np.asarray(candidates, dtype=np.int32)
    cand_salary = salaries[C].sum(axis=1)
    cand_proj = player_proj[C].sum(axis=1)

    # Base field preference. Ownership raking below is the hard constraint;
    # these features determine which COMBINATIONS are more likely among lineups
    # that collectively satisfy the same individual golfer ownership.
    if np.std(cand_proj) > 1e-9:
        proj_z = (cand_proj - np.median(cand_proj)) / np.std(cand_proj)
    else:
        proj_z = np.zeros(len(C), dtype=float)

    # Full salary gets a modest preference, not the old 1.75x cliff.
    salary_feature = np.clip((cand_salary - 49000.0) / 1000.0, -2.5, 1.0)

    logw = float(concentration) * (0.90 * proj_z + 0.45 * salary_feature)
    logw -= np.max(logw)
    weights = np.exp(logw)

    # Membership indices make iterative proportional fitting fast and memory-light.
    members = [[] for _ in range(n_players)]
    for r, combo in enumerate(C):
        for j in combo:
            members[int(j)].append(r)
    members = [np.asarray(x, dtype=np.int32) for x in members]

    total_w = float(weights.sum())
    max_rounds = 400 if exact_universe else 140
    max_error = np.inf
    rounds_used = 0

    for it in range(max_rounds):
        for j, target in enumerate(targets):
            idx = members[j]
            if len(idx) == 0:
                continue

            inside = float(weights[idx].sum())
            outside = total_w - inside
            if inside <= 0 or outside <= 0:
                continue

            factor = (target * outside) / (inside * (1.0 - target))
            factor = float(np.clip(factor, 1e-8, 1e8))
            weights[idx] *= factor
            total_w = outside + factor * inside

        rounds_used = it + 1

        if (it + 1) % 10 == 0:
            marginals = np.array(
                [weights[idx].sum() / total_w if len(idx) else 0.0 for idx in members]
            )
            max_error = float(np.max(np.abs(marginals - targets)))
            if max_error < 0.00075:
                break

    weights = weights / weights.sum()

    # Lookup candidate probability for every uploaded lineup.
    candidate_pos = {tuple(map(int, combo)): i for i, combo in enumerate(C)}
    probabilities = np.zeros(len(lineups), dtype=float)

    for r, key in enumerate(uploaded_idx_keys):
        if key is not None:
            pos = candidate_pos.get(key)
            if pos is not None:
                probabilities[r] = float(weights[pos])

    projected_copies = probabilities * float(contest_size)

    diagnostics = {
        "candidate_count": int(len(C)),
        "exact_universe": bool(exact_universe),
        "total_possible_combos": int(total_combos),
        "calibration_error": float(max_error),
        "rounds": int(rounds_used),
    }

    return projected_copies, probabilities, diagnostics



# ======================
# RUN DUPES BUTTON
# ======================
if st.button("Run Dupes"):

    if lineup_file is None or own_file is None:
        st.error("Upload BOTH lineup CSV and ownership CSV.")
        st.stop()

    # Load files
    lineups = pd.read_csv(lineup_file)
    own = pd.read_csv(own_file)
    st.session_state.own_df = own.copy()

    # Detect format
    is_showdown = "CPT" in lineups.columns
    st.session_state.is_showdown = is_showdown

    if is_showdown:
        sport_type = "showdown"
        st.write("✅ Detected NFL Showdown (CPT + 5 FLEX).")
        fighter_cols = ["CPT", "FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]
        gamma = 0.12
    else:
        mma_cols = ["F", "F.1", "F.2", "F.3", "F.4", "F.5"]
        pga_cols = ["G", "G.1", "G.2", "G.3", "G.4", "G.5"]

        if all(c in lineups.columns for c in mma_cols):
            sport_type = "mma"
            fighter_cols = mma_cols
            st.write("✅ Detected MMA (6 fighters).")
        elif all(c in lineups.columns for c in pga_cols):
            sport_type = "pga"
            fighter_cols = pga_cols
            st.write("✅ Detected PGA (6 golfers) — using ownership-calibrated field model.")
        else:
            f_like = [c for c in lineups.columns if re.fullmatch(r"F(?:\.\d+)?", str(c))]
            g_like = [c for c in lineups.columns if re.fullmatch(r"G(?:\.\d+)?", str(c))]

            if len(f_like) >= 6:
                sport_type = "mma"
                fighter_cols = f_like[:6]
                st.write("✅ Detected MMA (6 fighters).")
            elif len(g_like) >= 6:
                sport_type = "pga"
                fighter_cols = g_like[:6]
                st.write("✅ Detected PGA (6 golfers) — using ownership-calibrated field model.")
            else:
                st.error(
                    "Could not detect the 6 lineup player columns. "
                    "Expected MMA columns F/F.1/.../F.5 or PGA columns G/G.1/.../G.5. "
                    f"Columns found: {list(lineups.columns)}"
                )
                st.stop()

        gamma = 0.10

    st.session_state.fighter_cols = fighter_cols

    # Ownership maps
    if "DFS ID" not in own.columns:
        st.error("Ownership file must include 'DFS ID' column.")
        st.stop()

    own["DFS ID"] = own["DFS ID"].astype(int)

    if is_showdown:
        if "Ownership" not in own.columns or "CPTOwnership" not in own.columns:
            st.error("Showdown ownership file must have 'Ownership' and 'CPTOwnership'.")
            st.stop()
        flex_map = dict(zip(own["DFS ID"], own["Ownership"] / 100.0))
        cpt_map = dict(zip(own["DFS ID"], own["CPTOwnership"] / 100.0))
    else:
        if "Ownership" not in own.columns:
            st.error("MMA/PGA ownership file must have 'Ownership'.")
            st.stop()
        own_map = dict(zip(own["DFS ID"], own["Ownership"] / 100.0))

    # Detect projection column
    proj_col = None
    for c in lineups.columns:
        if any(k in c.upper() for k in ["PROJ", "SCORE", "MEDIAN", "FPTS"]):
            proj_col = c
            break
    if proj_col is None:
        st.error("Could not detect projection column (need PROJ, SCORE, MEDIAN, or FPTS).")
        st.stop()

    # Detect salary column
    sal_col = None
    for c in lineups.columns:
        if "SAL" in c.upper():
            sal_col = c
            break
    if sal_col is None and "Salary" in lineups.columns:
        sal_col = "Salary"
    if sal_col is None:
        st.error("Could not detect salary column.")
        st.stop()

    st.write(f"Using projection column: **{proj_col}**")
    st.write(f"Using salary column: **{sal_col}**")

    # ========================
    # DK SALARIES / CPT-FLEX MAP
    # ========================
    id_to_name = {}

    pga_salary_df = None
    if sport_type == "pga":
        if salary_file is None:
            st.error("For PGA, please also upload the DK Salaries CSV.")
            st.stop()
        pga_salary_df = pd.read_csv(salary_file)
        required_pga_salary_cols = {"ID", "Salary"}
        if not required_pga_salary_cols.issubset(set(pga_salary_df.columns)):
            st.error("PGA DK Salaries file must contain 'ID' and 'Salary' columns.")
            st.stop()

    cpt_to_flex = {}
    if is_showdown:
        if salary_file is None:
            st.error("For Showdown, please also upload the DK Salaries CSV.")
            st.stop()

        sal = pd.read_csv(salary_file)

        required_cols = {"Name", "ID", "Roster Position"}
        if not required_cols.issubset(set(sal.columns)):
            st.error(
                "DK Salaries file must contain columns: 'Name', 'ID', 'Roster Position'."
            )
            st.stop()

        # Group by player name; each should have CPT and FLEX rows
        for name, group in sal.groupby("Name"):
            name = str(name).strip()
            # FLEX row(s)
            flex_rows = group[group["Roster Position"].astype(str).str.upper() == "FLEX"]
            # CPT row(s)
            cpt_rows = group[group["Roster Position"].astype(str).str.upper() == "CPT"]

            if not flex_rows.empty:
                flex_id = int(flex_rows["ID"].iloc[0])
                id_to_name[flex_id] = name

                if not cpt_rows.empty:
                    cpt_id = int(cpt_rows["ID"].iloc[0])
                    cpt_to_flex[cpt_id] = flex_id
                    id_to_name[cpt_id] = name
            else:
                # If there's only CPT, still record name but skip mapping
                if not cpt_rows.empty:
                    cpt_id = int(cpt_rows["ID"].iloc[0])
                    id_to_name[cpt_id] = name

        st.session_state.id_to_name = id_to_name
    else:
        # Non-showdown: we can still try to use ownership names if present
        if "Name" in own.columns:
            for _, r in own.iterrows():
                pid = int(r["DFS ID"])
                id_to_name[pid] = str(r["Name"]).strip()
            st.session_state.id_to_name = id_to_name

    # ============================
    # Convert lineup fighter IDs
    # ============================
    # Map names -> IDs from ownership as a fallback
    name_to_id = {}
    if "Name" in own.columns:
        own["Name_clean"] = own["Name"].astype(str).str.upper().str.strip()
        name_to_id = dict(zip(own["Name_clean"], own["DFS ID"]))

    for col in fighter_cols:
        tmp = lineups[col]
        if not np.issubdtype(tmp.dtype, np.number):
            ids = tmp.apply(extract_id)
            # Fallback to name matching if many NAs
            if ids.isna().mean() > 0.3 and name_to_id:
                ids = (
                    tmp.astype(str)
                    .apply(lambda x: x.split("(")[0].strip().upper())
                    .map(name_to_id)
                )
            lineups[col] = ids

    # Drop bad rows
    lineups = lineups.dropna(subset=fighter_cols).copy()
    for col in fighter_cols:
        lineups[col] = lineups[col].astype(int)

    # Normalize CPT IDs to base FLEX ID using DK salaries map
    if is_showdown and cpt_to_flex:
        def normalize_cpt(pid):
            try:
                pid = int(pid)
            except Exception:
                return pid
            return cpt_to_flex.get(pid, pid)

        lineups["CPT"] = lineups["CPT"].apply(normalize_cpt)

    P_opt = lineups[proj_col].max()

    # Dupes functions
    def expected_showdown(row):
        try:
            cpt = int(row["CPT"])
            flex_ids = [int(row[c]) for c in fighter_cols[1:]]
        except Exception:
            return 0.0

        p = (cpt_map.get(cpt, 0.0001) ** 1.4)
        for f in flex_ids:
            p *= flex_map.get(f, 0.0001)

        return contest_size * p * salary_multiplier(row[sal_col]) * np.exp(
            -gamma * (P_opt - row[proj_col])
        )

    def expected_mma(row):
        try:
            ids = [int(row[c]) for c in fighter_cols]
        except Exception:
            return 0.0

        p = 1.0
        for f in ids:
            p *= own_map.get(f, 0.0001)

        return contest_size * p * salary_multiplier(row[sal_col]) * np.exp(
            -gamma * (P_opt - row[proj_col])
        )

    # Apply sport-specific dupe model.
    if sport_type == "showdown":
        lineups["Projected Dupes"] = lineups.apply(expected_showdown, axis=1)

        # Preserve the existing Showdown behavior.
        total_raw = lineups["Projected Dupes"].sum()
        scale = contest_size / total_raw if total_raw > 0 else 1.0
        lineups["Projected Dupes"] *= scale

    elif sport_type == "mma":
        lineups["Projected Dupes"] = lineups.apply(expected_mma, axis=1)

        # Preserve the existing MMA behavior for now.
        total_raw = lineups["Projected Dupes"].sum()
        scale = contest_size / total_raw if total_raw > 0 else 1.0
        lineups["Projected Dupes"] *= scale

    else:
        with st.spinner("Building PGA field model and calibrating golfer ownership..."):
            try:
                pga_copies, pga_probs, pga_diag = build_pga_field_model(
                    lineups=lineups,
                    golfer_cols=fighter_cols,
                    own=own,
                    salary_df=pga_salary_df,
                    contest_size=contest_size,
                    concentration=pga_concentration,
                )
            except Exception as e:
                st.error(f"PGA dupe model error: {e}")
                st.stop()

        lineups["PGA Exact Lineup Probability"] = pga_probs
        lineups["Projected Dupes"] = pga_copies

        model_label = (
            "exact legal-lineup universe"
            if pga_diag["exact_universe"]
            else "large sampled legal-lineup universe"
        )
        st.write(
            f"**PGA field model:** {model_label}; "
            f"{pga_diag['candidate_count']:,} legal candidate lineups; "
            f"ownership calibration max error ≈ "
            f"{100 * pga_diag['calibration_error']:.2f} percentage points."
        )
        st.caption(
            "PGA Projected Dupes are now contest-size × estimated probability of the "
            "exact six-golfer combination. The model matches individual golfer ownership "
            "while allowing realistic concentration around strong, salary-efficient lineups. "
            "The old six-way ownership multiplication and forced contest-size rescaling are "
            "not used for PGA."
        )

    # Store in session state
    st.session_state.df_out = lineups.copy()

    st.success("Dupes calculated! Scroll down to filter and split lineups.")


# ===================================================
# FILTER PANEL (only after dupes are computed)
# ===================================================
if st.session_state.df_out is not None:

    df_out = st.session_state.df_out
    fighter_cols = st.session_state.fighter_cols
    own = st.session_state.own_df
    id_to_name = st.session_state.id_to_name or {}

    st.header("Filter Lineups by ROI & Projected Dupes")

    # ROI column selector
    roi_col = st.selectbox(
        "Select ROI Column:",
        options=df_out.columns,
        help="Choose the ROI column (it can change each slate).",
    )

    max_dupes = st.number_input(
        "Maximum allowed Projected Dupes:", min_value=0.0, value=50.0
    )
    min_roi = st.number_input("Minimum required ROI:", value=0.0)

    # Apply filters
    filtered_df = df_out[
        (df_out["Projected Dupes"] <= max_dupes) & (df_out[roi_col] >= min_roi)
    ].copy()

    filtered_df = filtered_df.sort_values(by=roi_col, ascending=False)

    st.write(f"### Lineups that match your criteria: {len(filtered_df)}")
    st.dataframe(filtered_df.head(50))

    st.download_button(
        label="Download Filtered Lineups",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_lineups.csv",
    )

    st.download_button(
        label="Download All Lineups With Projected Dupes",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="lineups_with_projected_dupes.csv",
    )

    # ==============================================
    # TOP 300 → SPLIT INTO TWO BALANCED SETS
    # ==============================================
    st.subheader("Build Top 300 and Split into Two Balanced Sets")

    def exposure_summary(df, label):
        """Return player name + exposure table for set A or B."""
        rows = []
        for _, r in df.iterrows():
            for c in fighter_cols:
                rows.append(int(r[c]))

        counts = Counter(rows)
        total = len(df)

        exp_df = pd.DataFrame(
            {
                "Player ID": list(counts.keys()),
                f"{label} Times Used": list(counts.values()),
                f"{label} Exposure %": [
                    v * 100.0 / total for v in counts.values()
                ],
            }
        )

        # Add player names if available
        if id_to_name:
            exp_df["Name"] = exp_df["Player ID"].map(
                lambda pid: id_to_name.get(pid, "")
            )
            exp_df = exp_df[
                ["Name", "Player ID", f"{label} Times Used", f"{label} Exposure %"]
            ]

        return exp_df.sort_values(f"{label} Exposure %", ascending=False)

    if st.button("Build Top 300 and Split into Two Sets"):

        if len(filtered_df) < 2:
            st.error("Need at least 2 filtered lineups to split.")
        else:
            # Step 1: take up to top 300 by ROI
            top_n = min(300, len(filtered_df))
            top_df = (
                filtered_df.sort_values(by=roi_col, ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

            # Greedy exposure-balanced split
            exp_diff = Counter()  # exposureA - exposureB per player
            idxA, idxB = [], []

            for i, row in top_df.iterrows():
                players = [int(row[c]) for c in fighter_cols]

                # assign to A
                diffA = exp_diff.copy()
                for p in players:
                    diffA[p] += 1
                scoreA = sum(v * v for v in diffA.values()) + (
                    len(idxA) + 1 - len(idxB)
                ) ** 2

                # assign to B
                diffB = exp_diff.copy()
                for p in players:
                    diffB[p] -= 1
                scoreB = sum(v * v for v in diffB.values()) + (
                    len(idxA) - (len(idxB) + 1)
                ) ** 2

                if scoreA <= scoreB:
                    idxA.append(i)
                    exp_diff = diffA
                else:
                    idxB.append(i)
                    exp_diff = diffB

            setA = top_df.iloc[idxA].reset_index(drop=True)
            setB = top_df.iloc[idxB].reset_index(drop=True)

            st.write(f"Set A size: {len(setA)} lineups")
            st.write(f"Set B size: {len(setB)} lineups")

            st.write("### Exposure Summary — Set A")
            st.dataframe(exposure_summary(setA, "Set A").head(20))

            st.write("### Exposure Summary — Set B")
            st.dataframe(exposure_summary(setB, "Set B").head(20))

            st.download_button(
                label="Download Set A (CSV)",
                data=setA.to_csv(index=False).encode("utf-8"),
                file_name="top150_setA.csv",
            )

            st.download_button(
                label="Download Set B (CSV)",
                data=setB.to_csv(index=False).encode("utf-8"),
                file_name="top150_setB.csv",
            )
