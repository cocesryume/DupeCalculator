
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st


st.title("DFS Lineup Duplication Calculator")
st.caption(
    "PGA, MMA, and NFL Showdown use sport-specific field-construction dupe models."
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



def showdown_key(row, cpt_col, flex_cols):
    """Canonical Showdown key: CPT role matters; FLEX order does not."""
    try:
        cpt = int(row[cpt_col])
        flex = tuple(sorted(int(row[c]) for c in flex_cols))
        if len(flex) != 5 or len(set(flex)) != 5 or cpt in flex:
            return None
        return (cpt,) + flex
    except Exception:
        return None


def build_showdown_maps(salary_df):
    """
    Map DK CPT IDs and FLEX IDs to one base/FLEX player ID while preserving role.
    Also return team/position/base salary metadata keyed by base ID.
    """
    required = {"Name", "ID", "Roster Position", "Salary"}
    if not required.issubset(set(salary_df.columns)):
        raise ValueError(
            "Showdown DK Salaries file must include Name, ID, Roster Position, and Salary."
        )

    sal = salary_df.copy()
    sal["ID"] = pd.to_numeric(sal["ID"], errors="coerce")
    sal["Salary"] = pd.to_numeric(sal["Salary"], errors="coerce")
    sal = sal.dropna(subset=["ID", "Salary"]).copy()
    sal["ID"] = sal["ID"].astype(int)

    id_to_base = {}
    base_meta = {}

    for name, grp in sal.groupby("Name", dropna=False):
        grp = grp.copy()
        rp = grp["Roster Position"].astype(str).str.upper()

        flex_rows = grp[rp == "FLEX"]
        cpt_rows = grp[rp == "CPT"]

        # Some salary exports can have combined eligibility text. Prefer explicit FLEX.
        if flex_rows.empty:
            flex_rows = grp[rp.str.contains("FLEX", regex=False)]

        if flex_rows.empty:
            continue

        flex_row = flex_rows.iloc[0]
        base_id = int(flex_row["ID"])
        base_salary = float(flex_row["Salary"])

        team = ""
        if "TeamAbbrev" in grp.columns:
            team = str(flex_row.get("TeamAbbrev", "")).strip()

        position = ""
        if "Position" in grp.columns:
            position = str(flex_row.get("Position", "")).strip().upper()

        base_meta[base_id] = {
            "name": str(name),
            "team": team,
            "position": position,
            "salary": base_salary,
        }
        id_to_base[base_id] = base_id

        for _, r in flex_rows.iterrows():
            id_to_base[int(r["ID"])] = base_id
        for _, r in cpt_rows.iterrows():
            id_to_base[int(r["ID"])] = base_id

    return id_to_base, base_meta


def normalize_showdown_ownership(own, base_meta):
    """
    Infer whether Ownership is FLEX-only (~500% sum) or TOTAL roster ownership (~600% sum).
    Returns CPT and FLEX target maps keyed by base player ID.
    """
    if "DFS ID" not in own.columns:
        raise ValueError("Showdown ownership file must include 'DFS ID'.")
    if "Ownership" not in own.columns or "CPTOwnership" not in own.columns:
        raise ValueError(
            "Showdown ownership file must include 'Ownership' and 'CPTOwnership'."
        )

    o = own.copy()
    o["DFS ID"] = pd.to_numeric(o["DFS ID"], errors="coerce")
    o["Ownership"] = pd.to_numeric(o["Ownership"], errors="coerce")
    o["CPTOwnership"] = pd.to_numeric(o["CPTOwnership"], errors="coerce")
    o = o.dropna(subset=["DFS ID"]).copy()
    o["DFS ID"] = o["DFS ID"].astype(int)
    o = o.drop_duplicates("DFS ID", keep="first")

    valid_ids = set(base_meta)
    o = o[o["DFS ID"].isin(valid_ids)].copy()
    if o.empty:
        raise ValueError(
            "Could not match ownership DFS IDs to FLEX/base IDs from the DK Salaries file."
        )

    own_pct = o["Ownership"].fillna(0.0).clip(lower=0.0)
    cpt_pct = o["CPTOwnership"].fillna(0.0).clip(lower=0.0)

    own_sum = float(own_pct.sum())
    cpt_sum = float(cpt_pct.sum())

    # Auto-detect semantics.
    # FLEX-only ownership should sum near 500%; total roster ownership near 600%.
    if own_sum >= 550.0:
        flex_pct = (own_pct - cpt_pct).clip(lower=0.0)
        ownership_semantics = "total ownership (FLEX target = Ownership - CPT)"
    else:
        flex_pct = own_pct.copy()
        ownership_semantics = "FLEX-only ownership"

    # Normalize rounding error to exactly 100% CPT and 500% FLEX.
    cpt_arr = cpt_pct.to_numpy(dtype=float) / 100.0
    flex_arr = flex_pct.to_numpy(dtype=float) / 100.0

    if cpt_arr.sum() <= 0 or flex_arr.sum() <= 0:
        raise ValueError("Showdown CPT/FLEX ownership projections are empty or invalid.")

    cpt_arr *= 1.0 / cpt_arr.sum()
    flex_arr *= 5.0 / flex_arr.sum()

    # Individual role probabilities cannot exceed 100%.
    cpt_arr = np.clip(cpt_arr, 1e-8, 0.999999)
    flex_arr = np.clip(flex_arr, 1e-8, 0.999999)

    cpt_map = dict(zip(o["DFS ID"].astype(int), cpt_arr))
    flex_map = dict(zip(o["DFS ID"].astype(int), flex_arr))

    return cpt_map, flex_map, {
        "ownership_semantics": ownership_semantics,
        "raw_ownership_sum": own_sum,
        "raw_cpt_sum": cpt_sum,
    }


def showdown_structure_features(key, base_meta):
    """
    Showdown-specific construction features.
    Returns team split, QB/pass-catcher correlation, RB-DST correlation, etc.
    """
    cpt = key[0]
    flex = list(key[1:])
    ids = [cpt] + flex

    teams = [base_meta.get(pid, {}).get("team", "") for pid in ids]
    pos = [base_meta.get(pid, {}).get("position", "") for pid in ids]

    team_counts = Counter(t for t in teams if t)
    counts = sorted(team_counts.values(), reverse=True)
    max_team = counts[0] if counts else 0

    # Balanced 3-3 / 4-2 constructions are the dominant optimizer patterns.
    if max_team == 3:
        team_balance = 1.0
    elif max_team == 4:
        team_balance = 0.7
    elif max_team == 5:
        team_balance = -0.25
    else:
        team_balance = -1.0

    qb_ids = [pid for pid in ids if base_meta.get(pid, {}).get("position", "") == "QB"]
    qb_count = len(qb_ids)

    qb_stack_score = 0.0
    for qid in qb_ids:
        qteam = base_meta.get(qid, {}).get("team", "")
        mates = 0
        bringbacks = 0
        for pid in ids:
            if pid == qid:
                continue
            meta = base_meta.get(pid, {})
            ppos = meta.get("position", "")
            pteam = meta.get("team", "")
            if ppos in {"WR", "TE"}:
                if pteam == qteam:
                    mates += 1
                elif pteam and qteam and pteam != qteam:
                    bringbacks += 1
        qb_stack_score += min(mates, 2) * 0.55 + min(bringbacks, 2) * 0.25

    rb_dst_score = 0.0
    for pid in ids:
        meta = base_meta.get(pid, {})
        if meta.get("position") == "RB":
            team = meta.get("team", "")
            if any(
                base_meta.get(x, {}).get("position") in {"DST", "D"}
                and base_meta.get(x, {}).get("team") == team
                for x in ids
            ):
                rb_dst_score += 0.35

    # Two QBs are common in many Showdown fields; zero-QB is much less common.
    if qb_count == 2:
        qb_count_pref = 0.65
    elif qb_count == 1:
        qb_count_pref = 0.25
    elif qb_count == 0:
        qb_count_pref = -0.75
    else:
        qb_count_pref = -0.50

    return team_balance, qb_stack_score, rb_dst_score, qb_count_pref


def build_showdown_field_model(
    lineups,
    own,
    salary_df,
    proj_col,
    sal_col,
    contest_size,
    candidate_field_share=0.90,
    concentration=0.95,
    use_saber_signal=True,
):
    """
    Rebuilt NFL Showdown dupe model.

    Key improvements:
      * exact lineup identity preserves CPT role;
      * separate CPT and FLEX ownership marginals are calibrated simultaneously;
      * ownership semantics auto-detect total vs FLEX-only projections;
      * smooth salary-left behavior replaces hard salary cliffs;
      * lineup projection, joint ownership, team split, QB stacks, bring-backs,
        RB+DST correlation, and optional SaberSim Sim Dupes shape joint probability;
      * no forced 'sum projected dupes = contest size' normalization;
      * projected dupes = expected OTHER entries with the exact lineup.
    """
    id_to_base, base_meta = build_showdown_maps(salary_df)
    cpt_target_map, flex_target_map, own_diag = normalize_showdown_ownership(
        own, base_meta
    )

    work = lineups.copy()

    # Normalize role IDs to base/FLEX player IDs, but KEEP CPT role separate.
    cpt_raw = pd.to_numeric(work["CPT"], errors="coerce")
    work["_cpt_base"] = cpt_raw.map(
        lambda x: id_to_base.get(int(x), np.nan) if pd.notna(x) else np.nan
    )

    flex_cols = ["FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]
    for c in flex_cols:
        vals = pd.to_numeric(work[c], errors="coerce")
        work[f"_{c}_base"] = vals.map(
            lambda x: id_to_base.get(int(x), np.nan) if pd.notna(x) else np.nan
        )

    base_cols = ["_cpt_base"] + [f"_{c}_base" for c in flex_cols]
    work = work.dropna(subset=base_cols).copy()
    for c in base_cols:
        work[c] = work[c].astype(int)

    def make_key(r):
        cpt = int(r["_cpt_base"])
        flex = tuple(sorted(int(r[f"_{c}_base"]) for c in flex_cols))
        if len(set(flex)) != 5 or cpt in flex:
            return None
        return (cpt,) + flex

    work["_combo_key"] = work.apply(make_key, axis=1)
    work = work[work["_combo_key"].notna()].copy()
    if work.empty:
        raise ValueError("No valid Showdown lineups remained after salary-ID mapping.")

    work["_proj"] = pd.to_numeric(work[proj_col], errors="coerce")
    work["_salary"] = pd.to_numeric(work[sal_col], errors="coerce")

    # Recompute salary from role-aware DK salaries when possible.
    # CPT salary is 1.5x base salary; FLEX is base salary.
    def role_salary(key):
        cpt = key[0]
        flex = key[1:]
        cpt_sal = 1.5 * float(base_meta.get(cpt, {}).get("salary", 0.0))
        flex_sal = sum(float(base_meta.get(pid, {}).get("salary", 0.0)) for pid in flex)
        return cpt_sal + flex_sal

    work["_role_salary"] = work["_combo_key"].map(role_salary)
    # Prefer the uploaded lineup salary if it is valid; otherwise reconstructed salary.
    work["_salary"] = work["_salary"].where(work["_salary"].notna(), work["_role_salary"])
    work["_salary_left"] = 50000.0 - work["_salary"]

    def joint_log_own(key):
        cpt = key[0]
        flex = key[1:]
        p = max(cpt_target_map.get(cpt, 1e-8), 1e-8)
        for pid in flex:
            p *= max(flex_target_map.get(pid, 1e-8), 1e-8)
        return math.log(max(p, 1e-30))

    work["_joint_log_own"] = work["_combo_key"].map(joint_log_own)

    saber_signal, saber_cols = (
        detect_saber_signal(work) if use_saber_signal else (None, [])
    )
    work["_saber_signal"] = saber_signal if saber_signal is not None else 0.0

    structure = work["_combo_key"].map(
        lambda k: showdown_structure_features(k, base_meta)
    )
    work["_team_balance"] = structure.map(lambda x: x[0])
    work["_qb_stack"] = structure.map(lambda x: x[1])
    work["_rb_dst"] = structure.map(lambda x: x[2])
    work["_qb_count_pref"] = structure.map(lambda x: x[3])

    # Collapse duplicate optimizer rows representing the same exact lineup.
    combos = (
        work.groupby("_combo_key", as_index=False)
        .agg(
            {
                "_proj": "max",
                "_salary": "max",
                "_salary_left": "min",
                "_joint_log_own": "max",
                "_saber_signal": "max",
                "_team_balance": "max",
                "_qb_stack": "max",
                "_rb_dst": "max",
                "_qb_count_pref": "max",
            }
        )
        .reset_index(drop=True)
    )

    if len(combos) < 10:
        raise ValueError("Too few unique Showdown candidate lineups.")

    # Build target player universe from players actually present in candidate lineups.
    present = sorted({pid for k in combos["_combo_key"] for pid in k})
    target_ids = [
        pid
        for pid in present
        if pid in cpt_target_map and pid in flex_target_map
    ]
    if len(target_ids) < 6:
        raise ValueError(
            "Could not match enough Showdown players between ownership, salaries, and lineups."
        )

    id_to_j = {pid: j for j, pid in enumerate(target_ids)}

    # Remove combos containing unmatched players.
    mask = combos["_combo_key"].map(
        lambda k: all(pid in id_to_j for pid in k)
    )
    combos = combos[mask].reset_index(drop=True)
    if len(combos) < 10:
        raise ValueError("Too few Showdown candidates after ownership matching.")

    # Smooth construction features.
    zp = zscore(combos["_proj"])
    zj = zscore(combos["_joint_log_own"])
    zd = zscore(combos["_saber_signal"])
    zstack = zscore(
        combos["_team_balance"]
        + combos["_qb_stack"]
        + combos["_rb_dst"]
        + combos["_qb_count_pref"]
    )

    # Salary behavior is deliberately smooth. Full salary is common but not a 1.75x cliff.
    left = combos["_salary_left"].to_numpy(dtype=float)
    salary_score = (
        -0.0012 * np.clip(left, 0, 2500)
        -0.00035 * np.clip(left - 2500, 0, None)
    )
    zsal = zscore(salary_score)

    # Mixture of plausible field-builder archetypes.
    # name, mix, projection, joint-own, salary, structure, saber
    archetypes = [
        ("projection optimizer", 0.28, 1.35, 0.35, 0.55, 0.45, 0.45),
        ("chalk optimizer",      0.26, 0.95, 1.10, 0.65, 0.40, 0.60),
        ("stack optimizer",      0.18, 1.05, 0.35, 0.35, 1.00, 0.45),
        ("balanced GPP",         0.16, 0.95, 0.05, 0.20, 0.55, 0.35),
        ("contrarian GPP",       0.08, 0.85,-0.70, 0.05, 0.45, 0.20),
        ("recreational",         0.04, 0.35, 0.70, 0.45, 0.10, 0.10),
    ]

    base = np.zeros(len(combos), dtype=float)
    for _, mix, bp, bj, bs, bst, bd in archetypes:
        score = float(concentration) * (
            bp * zp + bj * zj + bs * zsal + bst * zstack + bd * zd
        )
        base += mix * softmax(score)

    base = np.maximum(base, 1e-18)
    base /= base.sum()

    n_players = len(target_ids)
    cpt_members = [[] for _ in range(n_players)]
    flex_members = [[] for _ in range(n_players)]

    for i, key in enumerate(combos["_combo_key"]):
        cpt_members[id_to_j[key[0]]].append(i)
        for pid in key[1:]:
            flex_members[id_to_j[pid]].append(i)

    cpt_members = [np.asarray(x, dtype=np.int32) for x in cpt_members]
    flex_members = [np.asarray(x, dtype=np.int32) for x in flex_members]

    cpt_targets = np.array(
        [cpt_target_map[pid] for pid in target_ids], dtype=float
    )
    flex_targets = np.array(
        [flex_target_map[pid] for pid in target_ids], dtype=float
    )

    # Normalize any tiny rounding / dropped-player error after restricting to present IDs.
    cpt_targets *= 1.0 / cpt_targets.sum()
    flex_targets *= 5.0 / flex_targets.sum()
    cpt_targets = np.clip(cpt_targets, 1e-8, 0.999999)
    flex_targets = np.clip(flex_targets, 1e-8, 0.999999)

    weights = base.copy()
    total_w = float(weights.sum())
    max_cpt_err = np.inf
    max_flex_err = np.inf
    rounds = 0

    # Sequential binary raking across role-specific marginals.
    for it in range(600):
        for members, targets in (
            (cpt_members, cpt_targets),
            (flex_members, flex_targets),
        ):
            for j, target in enumerate(targets):
                idx = members[j]
                if len(idx) == 0:
                    continue
                inside = float(weights[idx].sum())
                outside = total_w - inside
                if inside <= 0 or outside <= 0:
                    continue

                factor = (target * outside) / (inside * (1.0 - target))
                factor = float(np.clip(factor, 1e-7, 1e7))
                weights[idx] *= factor
                total_w = outside + factor * inside

        rounds = it + 1

        if rounds % 10 == 0:
            cpt_marg = np.array(
                [
                    weights[idx].sum() / total_w if len(idx) else 0.0
                    for idx in cpt_members
                ]
            )
            flex_marg = np.array(
                [
                    weights[idx].sum() / total_w if len(idx) else 0.0
                    for idx in flex_members
                ]
            )
            max_cpt_err = float(np.max(np.abs(cpt_marg - cpt_targets)))
            max_flex_err = float(np.max(np.abs(flex_marg - flex_targets)))

            if max(max_cpt_err, max_flex_err) < 0.001:
                break

    weights /= weights.sum()

    # Reserve some field mass for exact combinations not represented in the uploaded pool.
    exact_p = weights * float(candidate_field_share)
    prob_map = dict(zip(combos["_combo_key"], exact_p))

    # Map normalized keys back to ORIGINAL lineup row order.
    def original_key(r):
        try:
            cpt_raw = int(r["CPT"])
            cpt = id_to_base.get(cpt_raw)
            flex = []
            for c in ["FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]:
                raw = int(r[c])
                base = id_to_base.get(raw)
                if base is None:
                    return None
                flex.append(base)
            if cpt is None or cpt in flex or len(set(flex)) != 5:
                return None
            return (cpt,) + tuple(sorted(flex))
        except Exception:
            return None

    original_keys = lineups.apply(original_key, axis=1)
    p = np.array([prob_map.get(k, 0.0) for k in original_keys], dtype=float)

    lam = (float(contest_size) - 1.0) * p
    unique_prob = np.exp(
        (float(contest_size) - 1.0)
        * np.log1p(-np.clip(p, 0.0, 1.0 - 1e-15))
    )
    p_two_plus_others = 1.0 - unique_prob - lam * unique_prob
    p_two_plus_others = np.clip(p_two_plus_others, 0.0, 1.0)

    p90_other = np.ceil(
        lam + 1.282 * np.sqrt(np.maximum(lam, 1e-9))
    )
    p90_total = 1.0 + np.maximum(p90_other, 0.0)

    return {
        "exact_probability": p,
        "projected_dupes": lam,
        "expected_total_copies": 1.0 + lam,
        "unique_probability": unique_prob,
        "prob_2plus_other": p_two_plus_others,
        "p90_total_copies": p90_total,
    }, {
        "unique_candidate_lineups": int(len(combos)),
        "candidate_field_share": float(candidate_field_share),
        "cpt_calibration_error": float(max_cpt_err),
        "flex_calibration_error": float(max_flex_err),
        "calibration_rounds": int(rounds),
        "saber_columns_used": saber_cols,
        **own_diag,
    }


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
    # NFL Showdown: rebuilt role-aware field model
    # ------------------------------
    if sport_type == "showdown":
        with st.spinner("Building and calibrating the NFL Showdown field model..."):
            try:
                model, diag = build_showdown_field_model(
                    lineups=lineups,
                    own=own,
                    salary_df=sal,
                    proj_col=proj_col,
                    sal_col=sal_col,
                    contest_size=int(contest_size),
                    candidate_field_share=max(float(candidate_field_share), 0.90),
                    concentration=max(float(field_concentration), 0.95),
                    use_saber_signal=bool(use_saber_signal),
                )
            except Exception as e:
                st.error(f"Showdown field model error: {e}")
                st.stop()

        lineups["Exact Lineup Probability"] = model["exact_probability"]
        lineups["Projected Dupes"] = model["projected_dupes"]
        lineups["Expected Total Copies"] = model["expected_total_copies"]
        lineups["Unique Probability"] = model["unique_probability"]
        lineups["P(2+ Other Copies)"] = model["prob_2plus_other"]
        lineups["90th %ile Total Copies"] = model["p90_total_copies"]

        st.success("NFL Showdown field model calibrated.")
        st.write(
            f"Unique candidate combinations: **{diag['unique_candidate_lineups']:,}**  |  "
            f"Candidate field share: **{100 * diag['candidate_field_share']:.0f}%**  |  "
            f"CPT calibration max error: **{100 * diag['cpt_calibration_error']:.2f} pts**  |  "
            f"FLEX calibration max error: **{100 * diag['flex_calibration_error']:.2f} pts**"
        )
        st.write(
            f"Ownership interpretation: **{diag['ownership_semantics']}** "
            f"(raw Ownership sum {diag['raw_ownership_sum']:.1f}%, "
            f"CPT sum {diag['raw_cpt_sum']:.1f}%)."
        )
        if diag["saber_columns_used"]:
            st.write(
                f"Relative SaberSim dupe signal used from **{len(diag['saber_columns_used'])}** "
                "Sim Dupes column(s)."
            )

        st.info(
            "**Projected Dupes = expected OTHER entries with your exact CPT + 5 FLEX lineup.** "
            "The new Showdown model calibrates CPT and FLEX ownership separately and models "
            "salary left, lineup projection, chalk concentration, team split, QB stacks/bring-backs, "
            "RB+DST correlation, and optional SaberSim dupe signal. It does not use the old "
            "CPT-own^1.4 independence formula or forced contest-size rescaling."
        )

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
