
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st


st.title("DFS Lineup Duplication Calculator")
st.caption(
    "PGA/MMA use a field-construction model. NFL Showdown keeps the existing model."
)

# ==========================================
# Session state
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
if "sport_type" not in st.session_state:
    st.session_state.sport_type = None


# ==========================================
# Inputs
# ==========================================
lineup_file = st.file_uploader("Upload Lineups CSV", type=["csv"])
own_file = st.file_uploader("Upload Ownership CSV", type=["csv"])
salary_file = st.file_uploader(
    "Upload DK Salaries CSV",
    type=["csv"],
)
contest_size = st.number_input("Contest Size", min_value=2, value=20000, step=100)

with st.expander("Advanced dupe-model settings"):
    candidate_field_share = st.slider(
        "Share of the field represented by your uploaded candidate set",
        min_value=0.40,
        max_value=1.00,
        value=0.80,
        step=0.05,
        help=(
            "The uploaded optimizer lineups are the modeled candidate universe. "
            "Some real entries will use lineups outside that universe. 0.80 means "
            "the uploaded candidate set is assumed to capture 80% of field lineup mass."
        ),
    )
    field_concentration = st.slider(
        "Optimizer / field concentration",
        min_value=0.25,
        max_value=1.50,
        value=0.85,
        step=0.05,
        help=(
            "Higher values make the field converge more strongly on high-projection, "
            "chalky, salary-efficient constructions. The ownership calibration is still enforced."
        ),
    )
    use_saber_signal = st.checkbox(
        "Use any uploaded 'Sim Dupes' columns as a relative calibration signal",
        value=True,
        help=(
            "When SaberSim-style Sim Dupes columns are present, their relative ranking "
            "helps the model distinguish combinations that optimizers tend to repeat. "
            "Their raw absolute values are not used."
        ),
    )


# ==========================================
# Helpers
# ==========================================
def extract_id(x):
    s = str(x)
    m = re.search(r"\((\d+)\)", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", s)
    if m2:
        return int(m2.group(1))
    return np.nan


def salary_multiplier_showdown(s):
    """Preserve the old Showdown salary behavior."""
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


def zscore(x):
    arr = np.asarray(x, dtype=float)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros(len(arr), dtype=float)
    return (arr - np.nanmean(arr)) / sd


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x)
    e = np.exp(np.clip(x, -50, 50))
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones(len(x), dtype=float) / max(len(x), 1)
    return e / s


def canonical_key(row, cols):
    try:
        vals = sorted(int(row[c]) for c in cols)
        if len(vals) != 6 or len(set(vals)) != 6:
            return None
        return tuple(vals)
    except Exception:
        return None


def detect_saber_signal(df):
    cols = [c for c in df.columns if "SIM DUPES" in str(c).upper()]
    if not cols:
        return None, []
    sig = []
    usable = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        if s.max() > 0:
            usable.append(c)
            sig.append(np.log1p(s.to_numpy(dtype=float)))
    if not sig:
        return None, []
    mat = np.vstack(sig)
    return np.nanmean(mat, axis=0), usable


def build_field_model(
    lineups,
    player_cols,
    own,
    salary_df,
    proj_col,
    sal_col,
    sport_type,
    contest_size,
    candidate_field_share=0.80,
    concentration=0.85,
    use_saber_signal=True,
):
    """
    Field-construction dupe model for PGA/MMA.

    Core idea:
      - The optimizer lineup CSV is the candidate set of plausible field lineups.
      - Build several entrant archetype distributions over those exact combinations.
      - Blend archetypes.
      - Iteratively calibrate exact-lineup probabilities so player marginals match
        supplied ownership projections.
      - Reserve some mass for unseen lineups outside the uploaded candidate set.
      - For a lineup YOU enter:
          expected other copies = (contest_size - 1) * exact lineup probability
          unique probability = (1 - p) ** (contest_size - 1)
    """

    if "DFS ID" not in own.columns or "Ownership" not in own.columns:
        raise ValueError("Ownership file must include 'DFS ID' and 'Ownership'.")

    own_work = own.copy()
    own_work["DFS ID"] = pd.to_numeric(own_work["DFS ID"], errors="coerce")
    own_work["Ownership"] = pd.to_numeric(own_work["Ownership"], errors="coerce")
    own_work = own_work.dropna(subset=["DFS ID", "Ownership"]).copy()
    own_work["DFS ID"] = own_work["DFS ID"].astype(int)
    own_work = own_work.drop_duplicates("DFS ID", keep="first")

    own_map = dict(zip(own_work["DFS ID"], own_work["Ownership"] / 100.0))
    name_map = {}
    if "Name" in own_work.columns:
        name_map = dict(zip(own_work["DFS ID"], own_work["Name"].astype(str)))

    # Build one row per exact lineup combination.
    work = lineups.copy()
    work["_combo_key"] = work.apply(lambda r: canonical_key(r, player_cols), axis=1)
    work = work[work["_combo_key"].notna()].copy()
    if work.empty:
        raise ValueError("No valid six-player lineups were found.")

    # Numeric lineup features.
    work["_proj"] = pd.to_numeric(work[proj_col], errors="coerce")
    work["_salary"] = pd.to_numeric(work[sal_col], errors="coerce")

    # If the lineup file's total ownership exists, still recompute from the ownership file
    # so everything uses the same source.
    def lineup_own(key):
        return sum(float(own_map.get(pid, 0.0)) for pid in key) * 100.0

    work["_total_own"] = work["_combo_key"].map(lineup_own)

    saber_signal, saber_cols = detect_saber_signal(work) if use_saber_signal else (None, [])
    if saber_signal is not None:
        work["_saber_signal"] = saber_signal
    else:
        work["_saber_signal"] = 0.0

    # Multiple optimizer rows can represent the exact same combination.
    # Collapse to one exact-combination candidate.
    agg = {
        "_proj": "max",
        "_salary": "max",
        "_total_own": "max",
        "_saber_signal": "max",
    }
    combos = work.groupby("_combo_key", as_index=False).agg(agg)

    # Player universe is the ownership file. Keep only players that occur in candidates.
    present = sorted({pid for key in combos["_combo_key"] for pid in key})
    target_df = own_work[own_work["DFS ID"].isin(present)].copy()
    if len(target_df) < 6:
        raise ValueError("Ownership IDs do not match the lineup player IDs well enough.")

    target_ids = target_df["DFS ID"].astype(int).to_numpy()
    targets = target_df["Ownership"].astype(float).to_numpy() / 100.0

    # Candidate sets may omit some low-frequency lineups. Player marginal targets still
    # need to sum to six roster slots inside the modeled candidate share.
    # Normalize only tiny rounding error; otherwise retain relative ownership structure.
    if targets.sum() > 0:
        targets = targets * (6.0 / targets.sum())
    targets = np.clip(targets, 1e-6, 0.999)

    id_to_j = {int(pid): j for j, pid in enumerate(target_ids)}

    # Drop combinations containing IDs that cannot be calibrated.
    valid_mask = combos["_combo_key"].map(lambda k: all(pid in id_to_j for pid in k))
    combos = combos[valid_mask].reset_index(drop=True)
    if len(combos) < 10:
        raise ValueError("Too few candidate lineups remain after matching ownership IDs.")

    # Feature z-scores.
    zp = zscore(combos["_proj"].to_numpy())
    zs = zscore(combos["_salary"].to_numpy())
    zo = zscore(combos["_total_own"].to_numpy())
    zd = zscore(combos["_saber_signal"].to_numpy())

    # Entrant archetypes. Each creates a full probability distribution across candidate
    # lineups. Mixture weights differ slightly by sport.
    if sport_type == "pga":
        archetypes = [
            # name, mix, projection, salary, ownership, saber
            ("optimizer", 0.36, 1.30, 0.50, 0.30, 0.55),
            ("chalk optimizer", 0.22, 0.90, 0.35, 1.00, 0.65),
            ("balanced GPP", 0.24, 0.95, 0.20, 0.15, 0.45),
            ("contrarian", 0.12, 0.80, 0.05, -0.65, 0.25),
            ("recreational", 0.06, 0.25, 0.10, 0.45, 0.15),
        ]
    else:
        archetypes = [
            ("optimizer", 0.42, 1.25, 0.60, 0.35, 0.45),
            ("chalk optimizer", 0.24, 0.85, 0.45, 1.05, 0.55),
            ("balanced GPP", 0.20, 0.95, 0.30, 0.10, 0.35),
            ("contrarian", 0.09, 0.80, 0.10, -0.55, 0.20),
            ("recreational", 0.05, 0.25, 0.15, 0.50, 0.10),
        ]

    base = np.zeros(len(combos), dtype=float)
    for _, mix, bp, bs, bo, bd in archetypes:
        score = concentration * (bp * zp + bs * zs + bo * zo + bd * zd)
        base += mix * softmax(score)

    base = np.maximum(base, 1e-15)
    base /= base.sum()

    # Build candidate membership indices for ownership calibration.
    members = [[] for _ in range(len(target_ids))]
    for i, key in enumerate(combos["_combo_key"]):
        for pid in key:
            members[id_to_j[pid]].append(i)
    members = [np.asarray(v, dtype=np.int32) for v in members]

    # Rake exact-lineup probabilities to match individual ownership marginals while
    # preserving as much of the archetype-induced joint structure as possible.
    weights = base.copy()
    total_w = float(weights.sum())
    max_error = np.inf
    rounds = 0

    for it in range(400):
        for j, target in enumerate(targets):
            idx = members[j]
            if len(idx) == 0:
                continue
            inside = float(weights[idx].sum())
            outside = total_w - inside
            if inside <= 0 or outside <= 0:
                continue

            factor = (target * outside) / (inside * (1.0 - target))
            factor = float(np.clip(factor, 1e-6, 1e6))
            weights[idx] *= factor
            total_w = outside + factor * inside

        rounds = it + 1
        if rounds % 10 == 0:
            marg = np.array([
                weights[idx].sum() / total_w if len(idx) else 0.0
                for idx in members
            ])
            max_error = float(np.max(np.abs(marg - targets)))
            if max_error < 0.001:
                break

    weights /= weights.sum()

    # Keep part of total field mass for unseen combinations not present in uploaded optimizer set.
    # This is the honest correction for incomplete candidate coverage.
    candidate_mass = float(candidate_field_share)
    exact_p = weights * candidate_mass

    prob_map = dict(zip(combos["_combo_key"], exact_p))
    base_map = dict(zip(combos["_combo_key"], base))

    # Results mapped back to every uploaded row.
    p = np.array([prob_map.get(k, 0.0) for k in lineups.apply(lambda r: canonical_key(r, player_cols), axis=1)])
    lam = (float(contest_size) - 1.0) * p  # expected OTHER copies

    # Stable probability calculations.
    unique_prob = np.exp((float(contest_size) - 1.0) * np.log1p(-np.clip(p, 0, 1 - 1e-15)))
    p_two_plus_others = 1.0 - unique_prob - lam * unique_prob
    p_two_plus_others = np.clip(p_two_plus_others, 0.0, 1.0)

    # Approximate upper-tail total copies with Poisson quantile using simulation lookup.
    # This is only for risk display, not core model fitting.
    rng = np.random.default_rng(20260826)
    # Vectorized Monte Carlo would be too large for huge files; use normal/Poisson approximation.
    p90_other = np.where(
        lam < 25,
        np.ceil(lam + 1.282 * np.sqrt(np.maximum(lam, 1e-9))),
        np.ceil(lam + 1.282 * np.sqrt(lam)),
    )
    p90_total = 1.0 + np.maximum(p90_other, 0.0)

    out = {
        "exact_probability": p,
        "projected_dupes": lam,
        "expected_total_copies": 1.0 + lam,
        "unique_probability": unique_prob,
        "prob_2plus_other": p_two_plus_others,
        "p90_total_copies": p90_total,
    }

    diagnostics = {
        "unique_candidate_lineups": int(len(combos)),
        "uploaded_rows": int(len(lineups)),
        "candidate_field_share": candidate_mass,
        "calibration_error": float(max_error),
        "calibration_rounds": int(rounds),
        "saber_columns_used": saber_cols,
        "candidate_mass_average_copies": float(contest_size * candidate_mass / len(combos)),
    }

    return out, diagnostics


# ==========================================
# Run Dupes
# ==========================================
if st.button("Run Dupes"):

    if lineup_file is None or own_file is None or salary_file is None:
        st.error("Upload lineup CSV, ownership CSV, and DK Salaries CSV.")
        st.stop()

    lineups = pd.read_csv(lineup_file)
    own = pd.read_csv(own_file)
    sal = pd.read_csv(salary_file)

    st.session_state.own_df = own.copy()

    # Detect sport / lineup columns.
    is_showdown = "CPT" in lineups.columns
    st.session_state.is_showdown = is_showdown

    if is_showdown:
        sport_type = "showdown"
        fighter_cols = ["CPT", "FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]
        st.write("✅ Detected NFL Showdown (CPT + 5 FLEX).")
    else:
        mma_cols = ["F", "F.1", "F.2", "F.3", "F.4", "F.5"]
        pga_cols = ["G", "G.1", "G.2", "G.3", "G.4", "G.5"]

        if all(c in lineups.columns for c in pga_cols):
            sport_type = "pga"
            fighter_cols = pga_cols
            st.write("✅ Detected PGA (6 golfers).")
        elif all(c in lineups.columns for c in mma_cols):
            sport_type = "mma"
            fighter_cols = mma_cols
            st.write("✅ Detected MMA (6 fighters).")
        else:
            f_like = [c for c in lineups.columns if re.fullmatch(r"F(?:\.\d+)?", str(c))]
            g_like = [c for c in lineups.columns if re.fullmatch(r"G(?:\.\d+)?", str(c))]
            if len(g_like) >= 6:
                sport_type = "pga"
                fighter_cols = g_like[:6]
                st.write("✅ Detected PGA (6 golfers).")
            elif len(f_like) >= 6:
                sport_type = "mma"
                fighter_cols = f_like[:6]
                st.write("✅ Detected MMA (6 fighters).")
            else:
                st.error(
                    "Could not detect six lineup-player columns. "
                    "Expected G/G.1/.../G.5 for PGA or F/F.1/.../F.5 for MMA."
                )
                st.stop()

    st.session_state.fighter_cols = fighter_cols
    st.session_state.sport_type = sport_type

    # Detect projection and salary columns in lineup file.
    proj_col = None
    for c in lineups.columns:
        if any(k in str(c).upper() for k in ["PROJ", "SCORE", "MEDIAN", "FPTS"]):
            proj_col = c
            break
    if proj_col is None:
        st.error("Could not detect projection column.")
        st.stop()

    sal_col = None
    for c in lineups.columns:
        if "SAL" in str(c).upper():
            sal_col = c
            break
    if sal_col is None:
        st.error("Could not detect salary column.")
        st.stop()

    st.write(f"Using projection column: **{proj_col}**")
    st.write(f"Using salary column: **{sal_col}**")
    st.write(f"Contest size used by model: **{int(contest_size):,}**")

    # Build player-name map.
    id_to_name = {}
    if "DFS ID" in own.columns and "Name" in own.columns:
        own_ids = pd.to_numeric(own["DFS ID"], errors="coerce")
        for pid, name in zip(own_ids, own["Name"]):
            if pd.notna(pid):
                id_to_name[int(pid)] = str(name).strip()
    st.session_state.id_to_name = id_to_name

    # Convert lineup cells to IDs.
    name_to_id = {}
    if "DFS ID" in own.columns and "Name" in own.columns:
        tmp_own = own.copy()
        tmp_own["DFS ID"] = pd.to_numeric(tmp_own["DFS ID"], errors="coerce")
        tmp_own = tmp_own.dropna(subset=["DFS ID"])
        tmp_own["Name_clean"] = tmp_own["Name"].astype(str).str.upper().str.strip()
        name_to_id = dict(zip(tmp_own["Name_clean"], tmp_own["DFS ID"].astype(int)))

    for col in fighter_cols:
        if col not in lineups.columns:
            st.error(f"Missing lineup column: {col}")
            st.stop()
        tmp = lineups[col]
        if not np.issubdtype(tmp.dtype, np.number):
            ids = tmp.apply(extract_id)
            if ids.isna().mean() > 0.3 and name_to_id:
                ids = (
                    tmp.astype(str)
                    .apply(lambda x: x.split("(")[0].strip().upper())
                    .map(name_to_id)
                )
            lineups[col] = ids

    lineups = lineups.dropna(subset=fighter_cols).copy()
    for col in fighter_cols:
        lineups[col] = lineups[col].astype(int)

    # ------------------------------
    # Showdown: preserve old behavior
    # ------------------------------
    if sport_type == "showdown":
        if "DFS ID" not in own.columns:
            st.error("Ownership file must include DFS ID.")
            st.stop()
        if "Ownership" not in own.columns or "CPTOwnership" not in own.columns:
            st.error("Showdown ownership file needs Ownership and CPTOwnership.")
            st.stop()

        own["DFS ID"] = pd.to_numeric(own["DFS ID"], errors="coerce")
        own = own.dropna(subset=["DFS ID"]).copy()
        own["DFS ID"] = own["DFS ID"].astype(int)

        flex_map = dict(zip(own["DFS ID"], pd.to_numeric(own["Ownership"], errors="coerce").fillna(0) / 100.0))
        cpt_map = dict(zip(own["DFS ID"], pd.to_numeric(own["CPTOwnership"], errors="coerce").fillna(0) / 100.0))

        cpt_to_flex = {}
        if {"Name", "ID", "Roster Position"}.issubset(set(sal.columns)):
            for name, group in sal.groupby("Name"):
                flex_rows = group[group["Roster Position"].astype(str).str.upper() == "FLEX"]
                cpt_rows = group[group["Roster Position"].astype(str).str.upper() == "CPT"]
                if not flex_rows.empty and not cpt_rows.empty:
                    flex_id = int(flex_rows["ID"].iloc[0])
                    cpt_id = int(cpt_rows["ID"].iloc[0])
                    cpt_to_flex[cpt_id] = flex_id
                    id_to_name[flex_id] = str(name)
                    id_to_name[cpt_id] = str(name)

        if cpt_to_flex:
            lineups["CPT"] = lineups["CPT"].map(lambda x: cpt_to_flex.get(int(x), int(x)))

        p_opt = pd.to_numeric(lineups[proj_col], errors="coerce").max()
        gamma = 0.12

        def expected_showdown(row):
            try:
                cpt = int(row["CPT"])
                flex_ids = [int(row[c]) for c in fighter_cols[1:]]
            except Exception:
                return 0.0
            p = cpt_map.get(cpt, 0.0001) ** 1.4
            for f in flex_ids:
                p *= flex_map.get(f, 0.0001)
            return (
                float(contest_size)
                * p
                * salary_multiplier_showdown(row[sal_col])
                * np.exp(-gamma * (p_opt - float(row[proj_col])))
            )

        lineups["Projected Dupes"] = lineups.apply(expected_showdown, axis=1)
        total_raw = lineups["Projected Dupes"].sum()
        scale = float(contest_size) / total_raw if total_raw > 0 else 1.0
        lineups["Projected Dupes"] *= scale
        lineups["Expected Total Copies"] = 1.0 + lineups["Projected Dupes"]
        lineups["Unique Probability"] = np.nan
        lineups["P(2+ Other Copies)"] = np.nan
        lineups["90th %ile Total Copies"] = np.nan

    # ------------------------------
    # PGA / MMA: rebuilt field model
    # ------------------------------
    else:
        with st.spinner("Building and calibrating the field lineup model..."):
            try:
                model, diag = build_field_model(
                    lineups=lineups,
                    player_cols=fighter_cols,
                    own=own,
                    salary_df=sal,
                    proj_col=proj_col,
                    sal_col=sal_col,
                    sport_type=sport_type,
                    contest_size=int(contest_size),
                    candidate_field_share=float(candidate_field_share),
                    concentration=float(field_concentration),
                    use_saber_signal=bool(use_saber_signal),
                )
            except Exception as e:
                st.error(f"Field model error: {e}")
                st.stop()

        lineups["Exact Lineup Probability"] = model["exact_probability"]
        lineups["Projected Dupes"] = model["projected_dupes"]
        lineups["Expected Total Copies"] = model["expected_total_copies"]
        lineups["Unique Probability"] = model["unique_probability"]
        lineups["P(2+ Other Copies)"] = model["prob_2plus_other"]
        lineups["90th %ile Total Copies"] = model["p90_total_copies"]

        st.success("Field model calibrated.")
        st.write(
            f"Unique candidate combinations: **{diag['unique_candidate_lineups']:,}**  |  "
            f"Candidate field share: **{100 * diag['candidate_field_share']:.0f}%**  |  "
            f"Ownership calibration max error: **{100 * diag['calibration_error']:.2f} percentage points**"
        )
        if diag["saber_columns_used"]:
            st.write(
                f"Relative SaberSim dupe signal used from **{len(diag['saber_columns_used'])}** Sim Dupes column(s)."
            )

        st.info(
            "**Projected Dupes now means expected OTHER entries with your exact lineup.** "
            "So 0.50 means about half an expected duplicate; Expected Total Copies would be 1.50 "
            "including your own entry."
        )

    st.session_state.df_out = lineups.copy()
    st.session_state.id_to_name = id_to_name
    st.success("Dupes calculated. Scroll down to filter and split lineups.")


# ==========================================
# Filter panel
# ==========================================
if st.session_state.df_out is not None:

    df_out = st.session_state.df_out
    fighter_cols = st.session_state.fighter_cols
    id_to_name = st.session_state.id_to_name or {}

    st.header("Filter Lineups by ROI & Projected Dupes")

    roi_col = st.selectbox(
        "Select ROI Column:",
        options=df_out.columns,
        help="Choose the ROI column for this contest.",
    )

    max_dupes = st.number_input(
        "Maximum allowed Projected Dupes:",
        min_value=0.0,
        value=50.0,
        step=0.25,
    )
    min_roi = st.number_input("Minimum required ROI:", value=0.0)

    roi_values = pd.to_numeric(df_out[roi_col], errors="coerce")
    filtered_df = df_out[
        (df_out["Projected Dupes"] <= max_dupes)
        & (roi_values >= min_roi)
    ].copy()

    filtered_df = filtered_df.sort_values(by=roi_col, ascending=False)

    st.write(f"### Lineups that match your criteria: {len(filtered_df)}")
    preview_cols = [
        c for c in [
            *fighter_cols,
            proj_col if "proj_col" in globals() else None,
            sal_col if "sal_col" in globals() else None,
            roi_col,
            "Projected Dupes",
            "Expected Total Copies",
            "Unique Probability",
            "P(2+ Other Copies)",
            "90th %ile Total Copies",
        ]
        if c is not None and c in filtered_df.columns
    ]

    # Streamlit/Arrow errors if the preview requests the same column twice
    # (for example, if the selected ROI column is already one of the columns above).
    preview_cols = list(dict.fromkeys(preview_cols))

    st.dataframe(filtered_df[preview_cols].head(50))

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

    # ==========================================
    # Top 300 -> balanced sets
    # ==========================================
    st.subheader("Build Top 300 and Split into Two Balanced Sets")

    def exposure_summary(df, label):
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
            top_n = min(300, len(filtered_df))
            top_df = (
                filtered_df.sort_values(by=roi_col, ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

            exp_diff = Counter()
            idxA, idxB = [], []

            for i, row in top_df.iterrows():
                players = [int(row[c]) for c in fighter_cols]

                diffA = exp_diff.copy()
                for p in players:
                    diffA[p] += 1
                scoreA = sum(v * v for v in diffA.values()) + (
                    len(idxA) + 1 - len(idxB)
                ) ** 2

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
