import io
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DFS Lineup Duplication Calculator V3", layout="wide")
st.title("DFS Lineup Duplication Calculator — Showdown V3")
st.caption(
    "NFL Showdown V3 is calibrated from the NE@SEA and SF@LAR backtests. "
    "It adds a conservative safety estimate designed to reduce high-dupe false negatives. "
    "MMA/PGA continues to use the original ownership-product model."
)

# =========================================================
# V3 CALIBRATION CONSTANTS
# =========================================================
# Both V3 models are trained on dupes normalized to a 100,000-entry field,
# then scaled linearly to the contest size entered by the user.
V3_REFERENCE_FIELD = 100000.0

V3_FEATURES = [
    "salary_left", "proj_gap", "cpt_own", "flex_sum", "flex_min", "flex_max",
    "own_geom", "n_under10", "n_under5", "team_count_cpt", "stack_max",
    "qb_count", "dst_count", "k_count", "salary_near_max", "salary_under500",
    "salary_under1000", "proj_near_opt", "proj_under3", "proj_under6",
    "own_sum_all", "cpt_x_flexsum", "geom_sq", "log_own_geom",
    "stack4plus", "stack5plus", "cpt_stack4plus",
]

V3_MEAN = np.array([
    1700.3344481605352, 10.867190635451506, 0.10887257525083613,
    1.461275585284281, 0.13001103678929765, 0.4780086956521739,
    0.20577999837388156, 0.30434782608695654, 0.17725752508361203,
    3.769230769230769, 4.023411371237458, 1.2240802675585285,
    0.47491638795986624, 0.38127090301003347, 0.2279065137730765,
    0.18729096989966554, 0.35785953177257523, 0.0628771317522307,
    0.020066889632107024, 0.11705685618729098, 1.570148160535117,
    0.1494368558528428, 0.04306512687969519, -1.5948274923140746,
    0.7224080267558528, 0.3010033444816054, 0.6187290969899666,
])

V3_SCALE = np.array([
    1241.492745033379, 3.9966691985604283, 0.07081308939139738,
    0.22990635604031123, 0.06332339763115866, 0.0491207541295789,
    0.026827581869048604, 0.4951414372647597, 0.38188649476623904,
    1.0167114063830052, 0.7602941703199719, 0.5299921816599412,
    0.5190738888049822, 0.5253924064850588, 0.24463268907449587,
    0.39014492498776415, 0.47937051149615834, 0.10996768693757135,
    0.14022913239623164, 0.3214880224935897, 0.196365642718957,
    0.09244984807653436, 0.010195154129274934, 0.20787775841805928,
    0.44781097533955977, 0.45869415855501516, 0.48569887947981427,
])

# Base model: regularized log-dupe-rate regression, pooled across both slates.
V3_BASE_COEF = np.array([
    -0.2143554851569733, -0.12138155959354249, -0.029492851621936076,
    0.02654593266209481, 0.1119282857969469, -0.0796264437272928,
    0.08616988136398765, 0.1495732923567049, -0.04558242986600804,
    0.12382517900714159, 0.053755647254518375, 0.01673604248873483,
    0.026056115499628347, -0.0026576298842182263, 0.18749868342588025,
    0.07111887958532763, -0.029846520927040317, 0.031745224724403914,
    -0.01658673798746816, 0.11789110479063244, 0.020444506748402567,
    -0.011936377877006119, 0.1617628537453664, -0.07261195037868634,
    0.0124940956565399, 0.07690333833457465, 0.03920568794494617,
])
V3_BASE_INTERCEPT = 1.951173138427397

# Safety model: approximate 70th-percentile log-dupe-rate model.
# This is deliberately conservative for filtering. On the two historical
# cross-slate tests it reduced the number of false-safe <=9 selections.
V3_SAFE_COEF = np.array([
    -0.22349313289778663, -0.034369128909243284, 0.0,
    0.0, 0.006825158980829517, 0.0,
    0.0, 0.0, 0.0,
    0.04329246387065732, 0.10006439622461902, 0.0,
    0.0, 0.0, 0.2364755123584252,
    0.0, 0.0, 0.0,
    0.0, 0.06628293693527161, 0.09099398720106762,
    0.0, 0.1029574497590659, 0.0,
    0.0, 0.030429404578099103, 0.0,
])
V3_SAFE_INTERCEPT = 2.318064816612671

DEFAULT_STATE = {
    "df_out": None,
    "parsed_names": None,
    "fighter_cols": None,
    "is_showdown": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# Uploads / inputs
# =========================================================
lineup_file = st.file_uploader("Upload Lineups CSV", type=["csv"])
own_file = st.file_uploader("Upload Ownership CSV", type=["csv"])
salary_file = st.file_uploader("Upload DK Salaries CSV", type=["csv"])
contest_size = st.number_input("Contest Size", min_value=1, value=73529, step=1)

# =========================================================
# Helpers
# =========================================================
def read_csv(uploaded):
    return pd.read_csv(io.BytesIO(uploaded.getvalue()))


def normalize_name(value):
    if pd.isna(value):
        return ""
    s = str(value).replace("\u00a0", " ")
    s = s.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    return " ".join(s.strip().split()).lower()


def clean_percent(series):
    s = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    out = pd.to_numeric(s, errors="coerce")
    valid = out.dropna()
    if not valid.empty and valid.max() > 1.5:
        out = out / 100.0
    return out


def extract_id(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return int(value)
    text = str(value).strip()
    m = re.search(r"\((\d+)\)", text)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return int(float(text))
    m = re.search(r"(?<!\d)(\d{5,})(?!\d)", text)
    return int(m.group(1)) if m else np.nan


def detect_projection_column(df):
    for preferred in ["proj score", "median", "projection", "proj", "fpts"]:
        for c in df.columns:
            if str(c).strip().lower() == preferred:
                return c
    for c in df.columns:
        if any(k in str(c).upper() for k in ["PROJ", "MEDIAN", "FPTS", "SCORE"]):
            return c
    return None


def detect_salary_column(df):
    for c in df.columns:
        if str(c).strip().lower() == "salary":
            return c
    for c in df.columns:
        if "SAL" in str(c).upper():
            return c
    return None


def find_col(df, exact=(), contains=(), exclude=()):
    exact = [x.lower() for x in exact]
    for c in df.columns:
        if str(c).strip().lower() in exact:
            return c
    for c in df.columns:
        lc = str(c).strip().lower()
        if exclude and any(x.lower() in lc for x in exclude):
            continue
        if any(x.lower() in lc for x in contains):
            return c
    return None


def build_salary_maps(sal):
    if not {"Name", "ID"}.issubset(sal.columns):
        raise ValueError("DK Salaries must contain Name and ID columns.")
    s = sal.copy()
    s["_ID"] = pd.to_numeric(s["ID"], errors="coerce")
    s = s.dropna(subset=["_ID"]).copy()
    s["_ID"] = s["_ID"].astype(int)
    s["_NAME"] = s["Name"].map(normalize_name)
    s["_TEAM"] = s["TeamAbbrev"].astype(str).str.upper().str.strip() if "TeamAbbrev" in s.columns else ""
    s["_POS"] = s["Position"].astype(str).str.upper().str.strip() if "Position" in s.columns else ""

    id_to_name = dict(zip(s["_ID"], s["_NAME"]))
    id_to_team = dict(zip(s["_ID"], s["_TEAM"]))
    id_to_pos = dict(zip(s["_ID"], s["_POS"]))
    cpt_to_flex = {}
    flex_by_name = {}
    cpt_by_name = {}

    if "Roster Position" in s.columns:
        for nm, g in s.groupby("_NAME"):
            roles = g["Roster Position"].astype(str).str.upper().str.strip()
            flex = g[roles == "FLEX"]
            cpt = g[roles == "CPT"]
            if not flex.empty:
                fid = int(flex["_ID"].iloc[0])
                flex_by_name[nm] = fid
                if not cpt.empty:
                    cid = int(cpt["_ID"].iloc[0])
                    cpt_by_name[nm] = cid
                    cpt_to_flex[cid] = fid
    else:
        for _, r in s.iterrows():
            flex_by_name[r["_NAME"]] = int(r["_ID"])

    for cid, fid in cpt_to_flex.items():
        if fid not in id_to_team and cid in id_to_team:
            id_to_team[fid] = id_to_team[cid]
        if fid not in id_to_pos and cid in id_to_pos:
            id_to_pos[fid] = id_to_pos[cid]
        if fid not in id_to_name and cid in id_to_name:
            id_to_name[fid] = id_to_name[cid]

    return {
        "id_to_name": id_to_name,
        "id_to_team": id_to_team,
        "id_to_pos": id_to_pos,
        "cpt_to_flex": cpt_to_flex,
        "flex_by_name": flex_by_name,
        "cpt_by_name": cpt_by_name,
    }


def build_showdown_ownership(own):
    name_col = find_col(own, exact=["name", "player", "player name"])
    flex_col = find_col(
        own,
        exact=["ownership", "own", "flex ownership", "flexown"],
        contains=["ownership"],
        exclude=["cpt", "captain"],
    )
    cpt_col = find_col(
        own,
        exact=["cptown", "cptownership", "cpt ownership", "captain ownership"],
        contains=["cpt", "captain"],
    )
    id_col = find_col(own, exact=["dfs id", "dfsid", "id", "playerid", "player id"])
    if flex_col is None or cpt_col is None:
        raise ValueError("Showdown ownership file needs FLEX ownership and CPT ownership (e.g. ownership + cptOwn).")

    x = own.copy()
    x["_FLEX"] = clean_percent(x[flex_col])
    x["_CPT"] = clean_percent(x[cpt_col])

    name_flex, name_cpt, id_flex, id_cpt = {}, {}, {}, {}
    if name_col is not None:
        x["_NAME"] = x[name_col].map(normalize_name)
        g = x.dropna(subset=["_FLEX", "_CPT"])
        name_flex = dict(zip(g["_NAME"], g["_FLEX"]))
        name_cpt = dict(zip(g["_NAME"], g["_CPT"]))
    if id_col is not None:
        x["_ID"] = pd.to_numeric(x[id_col], errors="coerce")
        g = x.dropna(subset=["_ID", "_FLEX", "_CPT"]).copy()
        g["_ID"] = g["_ID"].astype(int)
        id_flex = dict(zip(g["_ID"], g["_FLEX"]))
        id_cpt = dict(zip(g["_ID"], g["_CPT"]))

    if not name_flex and not id_flex:
        raise ValueError("Could not build valid Showdown ownership mappings.")
    return name_flex, name_cpt, id_flex, id_cpt


def build_classic_ownership(own):
    id_col = find_col(own, exact=["dfs id", "dfsid", "id", "playerid", "player id"])
    own_col = find_col(own, exact=["ownership", "own", "proj ownership"], contains=["ownership"], exclude=["cpt", "captain"])
    name_col = find_col(own, exact=["name", "player", "fighter", "player name", "fighter name"])
    if own_col is None:
        raise ValueError("MMA/PGA ownership file must contain an ownership column.")
    x = own.copy()
    x["_OWN"] = clean_percent(x[own_col])
    id_map, name_map = {}, {}
    if id_col is not None:
        x["_ID"] = pd.to_numeric(x[id_col], errors="coerce")
        g = x.dropna(subset=["_ID", "_OWN"]).copy()
        g["_ID"] = g["_ID"].astype(int)
        id_map = dict(zip(g["_ID"], g["_OWN"]))
    if name_col is not None:
        x["_NAME"] = x[name_col].map(normalize_name)
        g = x.dropna(subset=["_OWN"])
        name_map = dict(zip(g["_NAME"], g["_OWN"]))
    return id_map, name_map


def v3_feature_vector(base14):
    """Expand the 14 core showdown features into the 27-feature V3 vector."""
    (
        salary_left, proj_gap, cpt_own, flex_sum, flex_min, flex_max,
        own_geom, n_under10, n_under5, team_count_cpt, stack_max,
        qb_count, dst_count, k_count,
    ) = [float(x) for x in base14]

    return np.array([
        salary_left,
        proj_gap,
        cpt_own,
        flex_sum,
        flex_min,
        flex_max,
        own_geom,
        n_under10,
        n_under5,
        team_count_cpt,
        stack_max,
        qb_count,
        dst_count,
        k_count,
        math.exp(-salary_left / 700.0),
        1.0 if salary_left <= 500 else 0.0,
        1.0 if salary_left <= 1000 else 0.0,
        math.exp(-proj_gap / 3.0),
        1.0 if proj_gap <= 3 else 0.0,
        1.0 if proj_gap <= 6 else 0.0,
        flex_sum + cpt_own,
        cpt_own * flex_sum,
        own_geom ** 2,
        math.log(max(own_geom, 1e-6)),
        1.0 if stack_max >= 4 else 0.0,
        1.0 if stack_max >= 5 else 0.0,
        1.0 if team_count_cpt >= 4 else 0.0,
    ], dtype=float)


def v3_predict(base14, field_size, conservative=False):
    x = v3_feature_vector(base14)
    z = (x - V3_MEAN) / V3_SCALE
    # Prevent one weird future slate feature from exploding the log model.
    z = np.clip(z, -4.0, 4.0)
    if conservative:
        log_rate = V3_SAFE_INTERCEPT + float(np.dot(V3_SAFE_COEF, z))
    else:
        log_rate = V3_BASE_INTERCEPT + float(np.dot(V3_BASE_COEF, z))
    rate_per_100k = max(0.0, math.expm1(log_rate))
    return rate_per_100k * (float(field_size) / V3_REFERENCE_FIELD)

# =========================================================
# Run model
# =========================================================
if st.button("Run Dupes"):
    if lineup_file is None or own_file is None:
        st.error("Please upload both the lineups CSV and ownership CSV.")
        st.stop()

    try:
        original = read_csv(lineup_file)
        own = read_csv(own_file)
        sal = read_csv(salary_file) if salary_file is not None else None
        is_showdown = "CPT" in original.columns

        if is_showdown:
            fighter_cols = ["CPT", "FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]
            if sal is None:
                raise ValueError("NFL Showdown V3 requires the DK Salaries CSV.")
        else:
            fighter_cols = ["F", "F.1", "F.2", "F.3", "F.4", "F.5"]

        missing = [c for c in fighter_cols if c not in original.columns]
        if missing:
            raise ValueError(f"Missing lineup columns: {missing}")

        proj_col = detect_projection_column(original)
        salary_col = detect_salary_column(original)
        if proj_col is None or salary_col is None:
            raise ValueError(f"Could not detect projection/salary columns. Found: {list(original.columns)}")

        lineups = original.copy()
        lineups[proj_col] = pd.to_numeric(lineups[proj_col], errors="coerce")
        lineups[salary_col] = pd.to_numeric(lineups[salary_col], errors="coerce")
        parsed_names = pd.DataFrame(index=lineups.index, columns=fighter_cols, dtype=object)
        parsed_base_ids = pd.DataFrame(index=lineups.index, columns=fighter_cols, dtype=float)
        missing_own = []

        if is_showdown:
            maps = build_salary_maps(sal)
            name_flex, name_cpt, id_flex, id_cpt = build_showdown_ownership(own)

            for c in fighter_cols:
                for idx, val in lineups[c].items():
                    pid = extract_id(val)
                    nm = ""
                    base = np.nan
                    if not pd.isna(pid):
                        pid = int(pid)
                        nm = maps["id_to_name"].get(pid, "")
                        base = maps["cpt_to_flex"].get(pid, pid) if c == "CPT" else pid
                        if not nm and not pd.isna(base):
                            nm = maps["id_to_name"].get(int(base), "")
                    else:
                        nm = normalize_name(re.sub(r"\s*\(\d+\)\s*$", "", str(val)))
                        if c == "CPT":
                            cid = maps["cpt_by_name"].get(nm)
                            if cid is not None:
                                base = maps["cpt_to_flex"].get(cid, np.nan)
                        else:
                            base = maps["flex_by_name"].get(nm, np.nan)
                    parsed_names.at[idx, c] = nm
                    parsed_base_ids.at[idx, c] = base

            p_opt = lineups[proj_col].max()
            base_preds = []
            safe_preds = []

            for idx, row in lineups.iterrows():
                cpt_nm = parsed_names.at[idx, "CPT"]
                cpt_id = parsed_base_ids.at[idx, "CPT"]
                cpt_own = name_cpt.get(cpt_nm)
                if cpt_own is None and not pd.isna(cpt_id):
                    cpt_own = id_cpt.get(int(cpt_id))
                if cpt_own is None:
                    missing_own.append((idx, "CPT", cpt_nm or cpt_id))
                    cpt_own = 0.0001

                flex_owns = []
                teams = []
                positions = []
                cpt_team = ""

                if not pd.isna(cpt_id):
                    cpt_team = maps["id_to_team"].get(int(cpt_id), "")
                    cpt_pos = maps["id_to_pos"].get(int(cpt_id), "")
                    if cpt_team:
                        teams.append(cpt_team)
                    if cpt_pos:
                        positions.append(cpt_pos)

                for c in fighter_cols[1:]:
                    nm = parsed_names.at[idx, c]
                    bid = parsed_base_ids.at[idx, c]
                    val = name_flex.get(nm)
                    if val is None and not pd.isna(bid):
                        val = id_flex.get(int(bid))
                    if val is None:
                        missing_own.append((idx, c, nm or bid))
                        val = 0.0001
                    flex_owns.append(float(val))
                    if not pd.isna(bid):
                        tm = maps["id_to_team"].get(int(bid), "")
                        pos = maps["id_to_pos"].get(int(bid), "")
                        if tm:
                            teams.append(tm)
                        if pos:
                            positions.append(pos)

                all_owns = [float(cpt_own)] + flex_owns
                geom = math.exp(sum(math.log(max(x, 1e-9)) for x in all_owns) / len(all_owns))
                team_counts = Counter(teams)
                stack_max = max(team_counts.values()) if team_counts else 0
                team_count_cpt = team_counts.get(cpt_team, 0) if cpt_team else 0

                core = [
                    max(0.0, 50000.0 - float(row[salary_col])),
                    max(0.0, float(p_opt) - float(row[proj_col])),
                    float(cpt_own),
                    float(sum(flex_owns)),
                    float(min(flex_owns)),
                    float(max(flex_owns)),
                    float(geom),
                    float(sum(x < 0.10 for x in flex_owns)),
                    float(sum(x < 0.05 for x in flex_owns)),
                    float(team_count_cpt),
                    float(stack_max),
                    float(sum(p == "QB" for p in positions)),
                    float(sum(p in ["DST", "D"] for p in positions)),
                    float(sum(p == "K" for p in positions)),
                ]
                base_preds.append(v3_predict(core, contest_size, conservative=False))
                safe_preds.append(v3_predict(core, contest_size, conservative=True))

            lineups["Projected Dupes Base"] = base_preds
            lineups["Projected Dupes"] = safe_preds

        else:
            # Preserve the original MMA/PGA model in V3.
            id_map, name_map = build_classic_ownership(own)
            salary_maps = build_salary_maps(sal) if sal is not None else None
            p_opt = lineups[proj_col].max()
            gamma = 0.10
            preds = []

            for idx, row in lineups.iterrows():
                p = 1.0
                for c in fighter_cols:
                    pid = extract_id(row[c])
                    nm = ""
                    if not pd.isna(pid):
                        pid = int(pid)
                        if salary_maps is not None:
                            nm = salary_maps["id_to_name"].get(pid, "")
                    else:
                        nm = normalize_name(re.sub(r"\s*\(\d+\)\s*$", "", str(row[c])))
                    parsed_names.at[idx, c] = nm
                    own_val = id_map.get(pid) if not pd.isna(pid) else None
                    if own_val is None and nm:
                        own_val = name_map.get(nm)
                    if own_val is None:
                        missing_own.append((idx, c, nm or pid))
                        own_val = 0.0001
                    p *= own_val

                salary_mult = (
                    1.75 if row[salary_col] >= 50000 else
                    1.30 if row[salary_col] >= 49900 else
                    1.00 if row[salary_col] >= 49800 else
                    0.80 if row[salary_col] >= 49700 else
                    0.60
                )
                preds.append(contest_size * p * salary_mult * math.exp(-gamma * (p_opt - row[proj_col])))

            lineups["Projected Dupes"] = preds
            raw_sum = lineups["Projected Dupes"].sum()
            if raw_sum > 0:
                lineups["Projected Dupes"] *= contest_size / raw_sum

        if is_showdown:
            output_cols = [
                c for c in original.columns
                if c not in ["Projected Dupes", "Projected Dupes Base"]
            ] + ["Projected Dupes Base", "Projected Dupes"]
        else:
            output_cols = [c for c in original.columns if c != "Projected Dupes"] + ["Projected Dupes"]

        df_out = lineups[output_cols].copy()
        st.session_state.df_out = df_out
        st.session_state.parsed_names = parsed_names
        st.session_state.fighter_cols = fighter_cols
        st.session_state.is_showdown = is_showdown

        if is_showdown:
            st.success("Showdown V3 projected dupes calculated.")
            st.info(
                "For Showdown, 'Projected Dupes Base' is the central V3 estimate. "
                "'Projected Dupes' is the more conservative safety estimate and is the default metric used by the filter below. "
                "Both scale directly to the contest size you entered."
            )
        else:
            st.success("MMA/PGA projected dupes calculated using the original model.")

        st.write(f"Projection column: **{proj_col}**")
        st.write(f"Salary column: **{salary_col}**")
        if missing_own:
            st.warning(f"{len(missing_own):,} player slots used fallback ownership because no ownership match was found.")
        else:
            st.info("Ownership coverage check: 100% of lineup player slots matched.")

    except Exception as exc:
        st.error(f"Could not run dupes: {exc}")
        st.stop()

# =========================================================
# Filter / downloads / balanced split
# =========================================================
if st.session_state.df_out is not None:
    df_out = st.session_state.df_out
    parsed_names = st.session_state.parsed_names
    fighter_cols = st.session_state.fighter_cols
    is_showdown = st.session_state.is_showdown

    st.divider()
    st.header("Filter Lineups by ROI & Dupes")

    roi_col = st.selectbox("Select ROI Column", options=list(df_out.columns))

    if is_showdown:
        dupe_metric = st.selectbox(
            "Dupe metric to filter on",
            options=["Projected Dupes", "Projected Dupes Base"],
            index=0,
            help=(
                "Projected Dupes is V3's conservative safety estimate and is recommended for max-dupe filtering. "
                "Projected Dupes Base is the central estimate."
            ),
        )
    else:
        dupe_metric = "Projected Dupes"

    max_dupes = st.number_input("Maximum allowed Projected Dupes", min_value=0.0, value=50.0, step=0.1)
    min_roi = st.number_input("Minimum required ROI", value=0.0, step=0.01)

    roi_numeric = pd.to_numeric(df_out[roi_col], errors="coerce")
    if roi_numeric.notna().sum() == 0:
        st.warning("Selected ROI column is not numeric.")
        filtered = df_out.iloc[0:0].copy()
    else:
        filtered = df_out[(df_out[dupe_metric] <= max_dupes) & (roi_numeric >= min_roi)].copy()
        filtered["_ROI_SORT"] = pd.to_numeric(filtered[roi_col], errors="coerce")
        filtered = filtered.sort_values("_ROI_SORT", ascending=False).drop(columns=["_ROI_SORT"])

    st.write(f"### {len(filtered):,} lineups match your criteria")
    st.dataframe(filtered.head(100), use_container_width=True)

    st.download_button(
        "Download Filtered Lineups",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_lineups_v3.csv",
        "text/csv",
    )
    st.download_button(
        "Download All Lineups With Projected Dupes",
        df_out.to_csv(index=False).encode("utf-8"),
        "lineups_with_projected_dupes_v3.csv",
        "text/csv",
    )

    st.subheader("Build Top 300 and Split into Two Balanced Sets")
    if st.button("Build Top 300 and Split into Two Sets"):
        if len(filtered) < 2:
            st.error("Need at least 2 filtered lineups to split.")
        else:
            top_n = min(300, len(filtered))
            top_df = filtered.head(top_n).copy()
            target_a, target_b = (top_n + 1) // 2, top_n // 2
            diff = Counter()
            a, b = [], []

            for idx in top_df.index:
                names = [parsed_names.at[idx, c] for c in fighter_cols if parsed_names.at[idx, c]]
                if len(a) >= target_a:
                    choose_a = False
                elif len(b) >= target_b:
                    choose_a = True
                else:
                    da = diff.copy()
                    db = diff.copy()
                    for n in names:
                        da[n] += 1
                        db[n] -= 1
                    sa = sum(v * v for v in da.values()) + 3 * (len(a) + 1 - len(b)) ** 2
                    sb = sum(v * v for v in db.values()) + 3 * (len(a) - (len(b) + 1)) ** 2
                    choose_a = sa <= sb

                if choose_a:
                    a.append(idx)
                    for n in names:
                        diff[n] += 1
                else:
                    b.append(idx)
                    for n in names:
                        diff[n] -= 1

            set_a, set_b = top_df.loc[a].copy(), top_df.loc[b].copy()

            def exposure(df_set, label):
                counts, cpts = Counter(), Counter()
                for idx in df_set.index:
                    for c in fighter_cols:
                        nm = parsed_names.at[idx, c]
                        if nm:
                            counts[nm] += 1
                            if is_showdown and c == "CPT":
                                cpts[nm] += 1
                rows = []
                total = len(df_set)
                for nm, ct in counts.items():
                    row = {
                        "Player": nm,
                        f"{label} Times Used": ct,
                        f"{label} Exposure %": 100 * ct / total if total else 0,
                    }
                    if is_showdown:
                        row[f"{label} CPT Times"] = cpts.get(nm, 0)
                        row[f"{label} CPT Exposure %"] = 100 * cpts.get(nm, 0) / total if total else 0
                    rows.append(row)
                return pd.DataFrame(rows).sort_values(f"{label} Exposure %", ascending=False)

            ca, cb = st.columns(2)
            with ca:
                st.write(f"### Set A — {len(set_a)}")
                st.dataframe(exposure(set_a, "Set A"), use_container_width=True)
            with cb:
                st.write(f"### Set B — {len(set_b)}")
                st.dataframe(exposure(set_b, "Set B"), use_container_width=True)

            st.write(
                f"Average ROI — Set A: **{pd.to_numeric(set_a[roi_col], errors='coerce').mean():.4f}** | "
                f"Set B: **{pd.to_numeric(set_b[roi_col], errors='coerce').mean():.4f}**"
            )
            st.write(
                f"Average {dupe_metric} — Set A: **{set_a[dupe_metric].mean():.3f}** | "
                f"Set B: **{set_b[dupe_metric].mean():.3f}**"
            )

            st.download_button(
                "Download Set A",
                set_a.to_csv(index=False).encode("utf-8"),
                "top_setA_v3.csv",
                "text/csv",
            )
            st.download_button(
                "Download Set B",
                set_b.to_csv(index=False).encode("utf-8"),
                "top_setB_v3.csv",
                "text/csv",
            )
