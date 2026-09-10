import io
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DFS Lineup Duplication Calculator V2", layout="wide")
st.title("DFS Lineup Duplication Calculator — Showdown V2")
st.caption("Experimental NFL Showdown model calibrated from the NE@SEA backtest. MMA/PGA continues to use the original ownership-product model.")

# =========================
# V2 CALIBRATION CONSTANTS
# =========================
TRAINING_FIELD_SIZE = 125798.46
V2_FEATURES = [
    "salary_left", "proj_gap", "cpt_own", "flex_sum", "flex_min", "flex_max",
    "own_geom", "n_under10", "n_under5", "team_count_cpt", "stack_max",
    "qb_count", "dst_count", "k_count",
]
V2_MEAN = np.array([
    2225.503355704698, 9.158322147651011, 0.09653691275167786,
    1.6036402684563758, 0.1354758389261745, 0.5134738255033556,
    0.21552479197757113, 0.3691275167785235, 0.21476510067114093,
    3.7651006711409396, 3.953020134228188, 1.2885906040268456,
    0.5234899328859061, 0.3288590604026846,
])
V2_SCALE = np.array([
    1348.7829145788733, 3.355348810658488, 0.0645274764315944,
    0.20677785405789223, 0.07399568520978321, 0.03970284657685291,
    0.02069918045447222, 0.5477102218282729, 0.41065928968532495,
    0.9368608281606353, 0.7448757427100403, 0.5591856736629047,
    0.5256366241396315, 0.4697986577181208,
])
V2_COEF = np.array([
    -0.5243458255665547, 0.016880761524195657, -0.14682382581151138,
    0.15684159598874794, 0.22065450812397164, -0.056145584129113954,
    0.18728867964751236, 0.5754326352588564, -0.032589730999052886,
    0.13604625311974833, 0.16319816468811724, 0.03365670809438466,
    0.233661945457879, 0.1579052771345828,
])
V2_INTERCEPT = 2.175078466685659

DEFAULT_STATE = {
    "df_out": None,
    "parsed_names": None,
    "fighter_cols": None,
    "is_showdown": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

lineup_file = st.file_uploader("Upload Lineups CSV", type=["csv"])
own_file = st.file_uploader("Upload Ownership CSV", type=["csv"])
salary_file = st.file_uploader("Upload DK Salaries CSV", type=["csv"])
contest_size = st.number_input("Contest Size", min_value=1, value=73529, step=1)


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

    # ensure base FLEX IDs also have team/position/name info
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
    flex_col = find_col(own, exact=["ownership", "own", "flex ownership", "flexown"], contains=["ownership"], exclude=["cpt", "captain"])
    cpt_col = find_col(own, exact=["cptown", "cptownership", "cpt ownership", "captain ownership"], contains=["cpt", "captain"])
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


def v2_predict(feature_vector, field_size):
    x = np.asarray(feature_vector, dtype=float)
    z = (x - V2_MEAN) / V2_SCALE
    mu_training_field = math.exp(V2_INTERCEPT + float(np.dot(V2_COEF, z)))
    return mu_training_field * (float(field_size) / TRAINING_FIELD_SIZE)


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
                raise ValueError("NFL Showdown V2 requires the DK Salaries CSV.")
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
            preds = []

            for idx, row in lineups.iterrows():
                # ownership values by role
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

                features = [
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
                preds.append(v2_predict(features, contest_size))

            lineups["Projected Dupes"] = preds

        else:
            # Preserve the original MMA/PGA model in this V2 app.
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
                salary_mult = 1.75 if row[salary_col] >= 50000 else 1.30 if row[salary_col] >= 49900 else 1.0 if row[salary_col] >= 49800 else 0.80 if row[salary_col] >= 49700 else 0.60
                preds.append(contest_size * p * salary_mult * math.exp(-gamma * (p_opt - row[proj_col])))

            lineups["Projected Dupes"] = preds
            raw_sum = lineups["Projected Dupes"].sum()
            if raw_sum > 0:
                lineups["Projected Dupes"] *= contest_size / raw_sum

        output_cols = [c for c in original.columns if c != "Projected Dupes"] + ["Projected Dupes"]
        df_out = lineups[output_cols].copy()
        st.session_state.df_out = df_out
        st.session_state.parsed_names = parsed_names
        st.session_state.fighter_cols = fighter_cols
        st.session_state.is_showdown = is_showdown

        if is_showdown:
            st.success("Showdown V2 projected dupes calculated.")
            st.info("V2 is experimental and calibrated from one historical Showdown slate. It intentionally does NOT use the old global candidate-pool normalization.")
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


if st.session_state.df_out is not None:
    df_out = st.session_state.df_out
    parsed_names = st.session_state.parsed_names
    fighter_cols = st.session_state.fighter_cols
    is_showdown = st.session_state.is_showdown

    st.divider()
    st.header("Filter Lineups by ROI & Projected Dupes")

    roi_col = st.selectbox("Select ROI Column", options=list(df_out.columns))
    max_dupes = st.number_input("Maximum allowed Projected Dupes", min_value=0.0, value=50.0, step=0.1)
    min_roi = st.number_input("Minimum required ROI", value=0.0, step=0.01)

    roi_numeric = pd.to_numeric(df_out[roi_col], errors="coerce")
    if roi_numeric.notna().sum() == 0:
        st.warning("Selected ROI column is not numeric.")
        filtered = df_out.iloc[0:0].copy()
    else:
        filtered = df_out[(df_out["Projected Dupes"] <= max_dupes) & (roi_numeric >= min_roi)].copy()
        filtered["_ROI_SORT"] = pd.to_numeric(filtered[roi_col], errors="coerce")
        filtered = filtered.sort_values("_ROI_SORT", ascending=False).drop(columns=["_ROI_SORT"])

    st.write(f"### {len(filtered):,} lineups match your criteria")
    st.dataframe(filtered.head(100), use_container_width=True)

    st.download_button("Download Filtered Lineups", filtered.to_csv(index=False).encode("utf-8"), "filtered_lineups_v2.csv", "text/csv")
    st.download_button("Download All Lineups With Projected Dupes", df_out.to_csv(index=False).encode("utf-8"), "lineups_with_projected_dupes_v2.csv", "text/csv")

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
                    sa = sum(v*v for v in da.values()) + 3*(len(a)+1-len(b))**2
                    sb = sum(v*v for v in db.values()) + 3*(len(a)-(len(b)+1))**2
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
                    r = {"Player": nm, f"{label} Times Used": ct, f"{label} Exposure %": 100*ct/total if total else 0}
                    if is_showdown:
                        r[f"{label} CPT Times"] = cpts.get(nm, 0)
                        r[f"{label} CPT Exposure %"] = 100*cpts.get(nm, 0)/total if total else 0
                    rows.append(r)
                return pd.DataFrame(rows).sort_values(f"{label} Exposure %", ascending=False)

            ca, cb = st.columns(2)
            with ca:
                st.write(f"### Set A — {len(set_a)}")
                st.dataframe(exposure(set_a, "Set A"), use_container_width=True)
            with cb:
                st.write(f"### Set B — {len(set_b)}")
                st.dataframe(exposure(set_b, "Set B"), use_container_width=True)

            st.download_button("Download Set A", set_a.to_csv(index=False).encode("utf-8"), "top_setA_v2.csv", "text/csv")
            st.download_button("Download Set B", set_b.to_csv(index=False).encode("utf-8"), "top_setB_v2.csv", "text/csv")
