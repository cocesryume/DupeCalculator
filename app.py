import io
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DFS Lineup Duplication Calculator", layout="wide")
st.title("DFS Lineup Duplication Calculator")
st.caption("Estimate field-level projected dupes, filter by ROI, and optionally split the best lineups into two balanced sets.")

# =========================================================
# Session state
# =========================================================
DEFAULT_STATE = {
    "df_out": None,
    "parsed_names": None,
    "parsed_ids": None,
    "fighter_cols": None,
    "is_showdown": None,
    "id_to_name": {},
    "ownership_mode": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# Uploads / inputs
# =========================================================
lineup_file = st.file_uploader("Upload Lineups CSV", type=["csv"])
own_file = st.file_uploader("Upload Ownership CSV", type=["csv"])
salary_file = st.file_uploader(
    "Upload DK Salaries CSV (required for NFL Showdown; optional for MMA/PGA)",
    type=["csv"],
)
contest_size = st.number_input("Contest Size", min_value=1, value=73529, step=1)

# =========================================================
# Helpers
# =========================================================
def read_uploaded_csv(uploaded_file):
    return pd.read_csv(io.BytesIO(uploaded_file.getvalue()))


def normalize_name(value):
    if pd.isna(value):
        return ""
    s = str(value)
    s = s.replace("\u00a0", " ")
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    s = " ".join(s.strip().split())
    return s.lower()


def clean_percent_series(series):
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    out = pd.to_numeric(s, errors="coerce")
    valid = out.dropna()
    if not valid.empty and valid.max() > 1.5:
        out = out / 100.0
    return out


def clean_id_series(series):
    out = pd.to_numeric(series, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def extract_id(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return int(value)
        return np.nan

    text = str(value).strip()
    m = re.search(r"\((\d+)\)", text)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return int(float(text))
    # Last-resort long integer token, useful for "Name - 12345678".
    m = re.search(r"(?<!\d)(\d{5,})(?!\d)", text)
    if m:
        return int(m.group(1))
    return np.nan


def extract_name_from_cell(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s*\(\d+\)\s*$", "", text)
    return normalize_name(text)


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


def find_column(df, exact_names=None, contains_all=None, contains_any=None, exclude_contains=None):
    exact_names = [x.lower() for x in (exact_names or [])]
    contains_all = [x.lower() for x in (contains_all or [])]
    contains_any = [x.lower() for x in (contains_any or [])]
    exclude_contains = [x.lower() for x in (exclude_contains or [])]

    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in exact_names:
            return c
    for c in df.columns:
        lc = str(c).strip().lower()
        if exclude_contains and any(x in lc for x in exclude_contains):
            continue
        if contains_all and all(x in lc for x in contains_all):
            return c
        if contains_any and any(x in lc for x in contains_any):
            return c
    return None


def detect_projection_column(df):
    preferred = ["proj score", "median", "projection", "proj", "fpts"]
    for p in preferred:
        for c in df.columns:
            if str(c).strip().lower() == p:
                return c
    for c in df.columns:
        cu = str(c).upper()
        if any(k in cu for k in ["PROJ", "MEDIAN", "FPTS", "SCORE"]):
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


def build_salary_maps(salary_df):
    required = {"Name", "ID"}
    if not required.issubset(set(salary_df.columns)):
        raise ValueError("DK Salaries must contain 'Name' and 'ID' columns.")

    sal = salary_df.copy()
    sal["_ID"] = clean_id_series(sal["ID"])
    sal = sal.dropna(subset=["_ID"]).copy()
    sal["_ID"] = sal["_ID"].astype(int)
    sal["_NAME_KEY"] = sal["Name"].map(normalize_name)

    id_to_name = dict(zip(sal["_ID"], sal["Name"].astype(str).str.strip()))
    id_to_name_key = dict(zip(sal["_ID"], sal["_NAME_KEY"]))

    cpt_to_flex = {}
    flex_id_by_name = {}
    cpt_id_by_name = {}

    if "Roster Position" in sal.columns:
        role = sal["Roster Position"].astype(str).str.upper().str.strip()
        for name_key, group in sal.groupby("_NAME_KEY"):
            roles = group["Roster Position"].astype(str).str.upper().str.strip()
            flex_rows = group[roles == "FLEX"]
            cpt_rows = group[roles == "CPT"]
            if not flex_rows.empty:
                flex_id = int(flex_rows["_ID"].iloc[0])
                flex_id_by_name[name_key] = flex_id
                if not cpt_rows.empty:
                    cpt_id = int(cpt_rows["_ID"].iloc[0])
                    cpt_id_by_name[name_key] = cpt_id
                    cpt_to_flex[cpt_id] = flex_id
    else:
        # Classic slates usually have one row per player.
        for _, r in sal.iterrows():
            flex_id_by_name[r["_NAME_KEY"]] = int(r["_ID"])

    return {
        "id_to_name": id_to_name,
        "id_to_name_key": id_to_name_key,
        "cpt_to_flex": cpt_to_flex,
        "flex_id_by_name": flex_id_by_name,
        "cpt_id_by_name": cpt_id_by_name,
    }


def build_showdown_ownership(own):
    """
    Supports BOTH:
      A) old ID-based format: Name / Ownership / CPTOwnership / DFS ID
      B) SaberSim name-based format: name / fpts / ownership / cptOwn
    Returns maps by normalized name and (when available) by FLEX DFS ID.
    """
    own = own.copy()

    name_col = find_column(own, exact_names=["name", "player", "player name"])
    flex_col = find_column(
        own,
        exact_names=["ownership", "own", "flex ownership", "flexown"],
        contains_any=["ownership"],
        exclude_contains=["cpt", "captain"],
    )
    cpt_col = find_column(
        own,
        exact_names=["cptown", "cptownership", "cpt ownership", "captain ownership"],
        contains_any=["cpt", "captain"],
    )
    id_col = find_column(own, exact_names=["dfs id", "dfsid", "id", "playerid", "player id"])

    if flex_col is None or cpt_col is None:
        raise ValueError(
            "Showdown ownership file must include FLEX ownership and CPT ownership. "
            "Supported examples: 'ownership' + 'cptOwn', or 'Ownership' + 'CPTOwnership'."
        )

    own["_FLEX_OWN"] = clean_percent_series(own[flex_col])
    own["_CPT_OWN"] = clean_percent_series(own[cpt_col])

    name_flex = {}
    name_cpt = {}
    if name_col is not None:
        own["_NAME_KEY"] = own[name_col].map(normalize_name)
        good = own.dropna(subset=["_FLEX_OWN", "_CPT_OWN"]).copy()
        good = good[good["_NAME_KEY"] != ""]
        name_flex = dict(zip(good["_NAME_KEY"], good["_FLEX_OWN"]))
        name_cpt = dict(zip(good["_NAME_KEY"], good["_CPT_OWN"]))

    id_flex = {}
    id_cpt = {}
    if id_col is not None:
        own["_ID"] = clean_id_series(own[id_col])
        good = own.dropna(subset=["_ID", "_FLEX_OWN", "_CPT_OWN"]).copy()
        good["_ID"] = good["_ID"].astype(int)
        id_flex = dict(zip(good["_ID"], good["_FLEX_OWN"]))
        id_cpt = dict(zip(good["_ID"], good["_CPT_OWN"]))

    if name_flex:
        mode = "name-based"
    elif id_flex:
        mode = "DFS-ID-based"
    else:
        raise ValueError("Could not build any valid ownership mappings from the Showdown ownership file.")

    return {
        "mode": mode,
        "name_flex": name_flex,
        "name_cpt": name_cpt,
        "id_flex": id_flex,
        "id_cpt": id_cpt,
    }


def build_classic_ownership(own):
    own = own.copy()
    id_col = find_column(own, exact_names=["dfs id", "dfsid", "id", "playerid", "player id"])
    flex_col = find_column(
        own,
        exact_names=["ownership", "own", "proj ownership"],
        contains_any=["ownership"],
        exclude_contains=["cpt", "captain"],
    )
    name_col = find_column(own, exact_names=["name", "player", "fighter", "player name", "fighter name"])

    if flex_col is None:
        raise ValueError("MMA/PGA ownership file must contain an ownership column.")

    own["_OWN"] = clean_percent_series(own[flex_col])
    id_map = {}
    name_map = {}

    if id_col is not None:
        own["_ID"] = clean_id_series(own[id_col])
        good = own.dropna(subset=["_ID", "_OWN"]).copy()
        good["_ID"] = good["_ID"].astype(int)
        id_map = dict(zip(good["_ID"], good["_OWN"]))

    if name_col is not None:
        own["_NAME_KEY"] = own[name_col].map(normalize_name)
        good = own.dropna(subset=["_OWN"]).copy()
        good = good[good["_NAME_KEY"] != ""]
        name_map = dict(zip(good["_NAME_KEY"], good["_OWN"]))

    if not id_map and not name_map:
        raise ValueError("Could not build any valid ownership mappings from the MMA/PGA ownership file.")

    return {"id": id_map, "name": name_map}


# =========================================================
# Run model
# =========================================================
if st.button("Run Dupes"):
    if lineup_file is None or own_file is None:
        st.error("Please upload both the lineups CSV and ownership CSV.")
        st.stop()

    try:
        original_lineups = read_uploaded_csv(lineup_file)
        own = read_uploaded_csv(own_file)
        salary_df = read_uploaded_csv(salary_file) if salary_file is not None else None

        is_showdown = "CPT" in original_lineups.columns
        if is_showdown:
            fighter_cols = ["CPT", "FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]
            gamma = 0.12
            missing_cols = [c for c in fighter_cols if c not in original_lineups.columns]
            if missing_cols:
                raise ValueError(f"Missing Showdown lineup columns: {missing_cols}")
            if salary_df is None:
                raise ValueError("NFL Showdown requires the DK Salaries CSV so CPT/FLEX IDs can be mapped correctly.")
        else:
            fighter_cols = ["F", "F.1", "F.2", "F.3", "F.4", "F.5"]
            gamma = 0.10
            missing_cols = [c for c in fighter_cols if c not in original_lineups.columns]
            if missing_cols:
                raise ValueError(f"Missing MMA/PGA lineup columns: {missing_cols}")

        proj_col = detect_projection_column(original_lineups)
        sal_col = detect_salary_column(original_lineups)
        if proj_col is None:
            raise ValueError(f"Could not detect projection column. Columns: {list(original_lineups.columns)}")
        if sal_col is None:
            raise ValueError(f"Could not detect salary column. Columns: {list(original_lineups.columns)}")

        # Keep original lineup columns untouched for the downloadable output.
        lineups = original_lineups.copy()
        lineups[proj_col] = pd.to_numeric(lineups[proj_col], errors="coerce")
        lineups[sal_col] = pd.to_numeric(lineups[sal_col], errors="coerce")

        parsed_ids = pd.DataFrame(index=lineups.index, columns=fighter_cols, dtype="float64")
        parsed_names = pd.DataFrame(index=lineups.index, columns=fighter_cols, dtype="object")

        salary_maps = None
        if salary_df is not None:
            salary_maps = build_salary_maps(salary_df)

        missing_ownership_slots = []

        if is_showdown:
            own_maps = build_showdown_ownership(own)
            st.session_state.ownership_mode = own_maps["mode"]

            # Parse each slot and map to canonical player name / FLEX base ID.
            for col in fighter_cols:
                for idx, value in lineups[col].items():
                    pid = extract_id(value)
                    name_key = ""
                    base_flex_id = np.nan

                    if not pd.isna(pid):
                        pid = int(pid)
                        name_key = salary_maps["id_to_name_key"].get(pid, "")
                        if col == "CPT":
                            base_flex_id = salary_maps["cpt_to_flex"].get(pid, pid)
                        else:
                            base_flex_id = pid
                    else:
                        # Name-only cells are also supported.
                        name_key = extract_name_from_cell(value)
                        if col == "CPT":
                            cpt_id = salary_maps["cpt_id_by_name"].get(name_key)
                            if cpt_id is not None:
                                pid = cpt_id
                                base_flex_id = salary_maps["cpt_to_flex"].get(cpt_id, np.nan)
                        else:
                            flex_id = salary_maps["flex_id_by_name"].get(name_key)
                            if flex_id is not None:
                                pid = flex_id
                                base_flex_id = flex_id

                    parsed_ids.at[idx, col] = base_flex_id
                    parsed_names.at[idx, col] = name_key

            P_opt = lineups[proj_col].max()

            def showdown_dupes(row):
                idx = row.name
                p = 1.0

                # CPT: always use CPT ownership.
                cpt_name = parsed_names.at[idx, "CPT"]
                cpt_base_id = parsed_ids.at[idx, "CPT"]
                cpt_own = own_maps["name_cpt"].get(cpt_name)
                if cpt_own is None and not pd.isna(cpt_base_id):
                    cpt_own = own_maps["id_cpt"].get(int(cpt_base_id))
                if cpt_own is None:
                    missing_ownership_slots.append((idx, "CPT", cpt_name or cpt_base_id))
                    cpt_own = 0.0001
                p *= cpt_own ** 1.4

                # FLEX: always use FLEX ownership.
                for col in fighter_cols[1:]:
                    nm = parsed_names.at[idx, col]
                    base_id = parsed_ids.at[idx, col]
                    own_value = own_maps["name_flex"].get(nm)
                    if own_value is None and not pd.isna(base_id):
                        own_value = own_maps["id_flex"].get(int(base_id))
                    if own_value is None:
                        missing_ownership_slots.append((idx, col, nm or base_id))
                        own_value = 0.0001
                    p *= own_value

                f_salary = salary_multiplier(row[sal_col])
                f_proj = math.exp(-gamma * (P_opt - row[proj_col]))
                return contest_size * p * f_salary * f_proj

            lineups["Projected Dupes"] = lineups.apply(showdown_dupes, axis=1)

        else:
            own_maps = build_classic_ownership(own)
            st.session_state.ownership_mode = "classic"

            # Build names from salary file and/or ownership names when possible.
            own_name_by_id = {}
            own_name_col = find_column(own, exact_names=["name", "player", "fighter", "player name", "fighter name"])
            own_id_col = find_column(own, exact_names=["dfs id", "dfsid", "id", "playerid", "player id"])
            if own_name_col is not None and own_id_col is not None:
                temp = own[[own_id_col, own_name_col]].copy()
                temp["_ID"] = clean_id_series(temp[own_id_col])
                temp = temp.dropna(subset=["_ID"])
                temp["_ID"] = temp["_ID"].astype(int)
                own_name_by_id = dict(zip(temp["_ID"], temp[own_name_col].map(normalize_name)))

            for col in fighter_cols:
                for idx, value in lineups[col].items():
                    pid = extract_id(value)
                    name_key = ""
                    if not pd.isna(pid):
                        pid = int(pid)
                        name_key = (
                            salary_maps["id_to_name_key"].get(pid, "")
                            if salary_maps is not None
                            else ""
                        )
                        if not name_key:
                            name_key = own_name_by_id.get(pid, "")
                    else:
                        name_key = extract_name_from_cell(value)
                        # If name-only, try to resolve its ID from salaries.
                        if salary_maps is not None:
                            pid = salary_maps["flex_id_by_name"].get(name_key, np.nan)

                    parsed_ids.at[idx, col] = pid
                    parsed_names.at[idx, col] = name_key

            P_opt = lineups[proj_col].max()

            def classic_dupes(row):
                idx = row.name
                p = 1.0
                for col in fighter_cols:
                    pid = parsed_ids.at[idx, col]
                    nm = parsed_names.at[idx, col]
                    own_value = None
                    if not pd.isna(pid):
                        own_value = own_maps["id"].get(int(pid))
                    if own_value is None and nm:
                        own_value = own_maps["name"].get(nm)
                    if own_value is None:
                        missing_ownership_slots.append((idx, col, nm or pid))
                        own_value = 0.0001
                    p *= own_value

                f_salary = salary_multiplier(row[sal_col])
                f_proj = math.exp(-gamma * (P_opt - row[proj_col]))
                return contest_size * p * f_salary * f_proj

            lineups["Projected Dupes"] = lineups.apply(classic_dupes, axis=1)

        # Preserve the current app's field-concentration scaling behavior.
        raw_sum = lineups["Projected Dupes"].sum()
        scale = contest_size / raw_sum if raw_sum > 0 else 1.0
        lineups["Projected Dupes"] = lineups["Projected Dupes"] * scale

        # Save only original columns + Projected Dupes (no internal keys/IDs).
        output_cols = [c for c in original_lineups.columns if c != "Projected Dupes"] + ["Projected Dupes"]
        df_out = lineups[output_cols].copy()

        st.session_state.df_out = df_out
        st.session_state.parsed_names = parsed_names
        st.session_state.parsed_ids = parsed_ids
        st.session_state.fighter_cols = fighter_cols
        st.session_state.is_showdown = is_showdown
        if salary_maps is not None:
            st.session_state.id_to_name = salary_maps["id_to_name"]
        else:
            st.session_state.id_to_name = {}

        if is_showdown:
            st.success(
                f"Dupes calculated. Detected NFL Showdown ownership format: {st.session_state.ownership_mode}."
            )
        else:
            st.success("Dupes calculated. Detected MMA/PGA classic format.")

        st.write(f"Projection column: **{proj_col}**")
        st.write(f"Salary column: **{sal_col}**")
        st.write(f"Scaling factor: **{scale:.3e}**")

        if missing_ownership_slots:
            unique_missing = []
            seen = set()
            for item in missing_ownership_slots:
                key = (item[1], str(item[2]))
                if key not in seen:
                    unique_missing.append(item)
                    seen.add(key)
            st.warning(
                f"{len(missing_ownership_slots):,} lineup slots used the fallback ownership because no match was found. "
                f"There are {len(unique_missing):,} unique missing slot/player combinations."
            )
            with st.expander("Show missing ownership matches"):
                st.dataframe(
                    pd.DataFrame(unique_missing, columns=["Row", "Slot", "Player / ID"]).head(100),
                    use_container_width=True,
                )
        else:
            st.info("Ownership coverage check: 100% of lineup player slots matched the ownership file.")

    except Exception as exc:
        st.error(f"Could not run dupes: {exc}")
        st.stop()

# =========================================================
# Filter / download / split panel
# =========================================================
if st.session_state.df_out is not None:
    df_out = st.session_state.df_out
    parsed_names = st.session_state.parsed_names
    fighter_cols = st.session_state.fighter_cols
    is_showdown = st.session_state.is_showdown

    st.divider()
    st.header("Filter Lineups by ROI & Projected Dupes")

    roi_col = st.selectbox(
        "Select ROI Column",
        options=list(df_out.columns),
        help="Choose whichever column represents ROI for this contest/slate.",
    )

    max_dupes = st.number_input(
        "Maximum allowed Projected Dupes",
        min_value=0.0,
        value=50.0,
        step=0.1,
    )
    min_roi = st.number_input("Minimum required ROI", value=0.0, step=0.01)

    roi_numeric = pd.to_numeric(df_out[roi_col], errors="coerce")
    if roi_numeric.notna().sum() == 0:
        st.warning("The selected ROI column is not numeric. Choose a numeric ROI column to filter/rank lineups.")
        filtered_df = df_out.iloc[0:0].copy()
    else:
        filtered_df = df_out[
            (df_out["Projected Dupes"] <= max_dupes)
            & (roi_numeric >= min_roi)
        ].copy()
        filtered_df["_ROI_SORT"] = pd.to_numeric(filtered_df[roi_col], errors="coerce")
        filtered_df = filtered_df.sort_values("_ROI_SORT", ascending=False).drop(columns=["_ROI_SORT"])

    st.write(f"### {len(filtered_df):,} lineups match your criteria")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    st.download_button(
        "Download Filtered Lineups",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_lineups.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download All Lineups With Projected Dupes",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="lineups_with_projected_dupes.csv",
        mime="text/csv",
    )

    # ---------------------------------------------------------
    # Optional Top 300 -> two balanced sets
    # ---------------------------------------------------------
    st.subheader("Build Top 300 and Split into Two Balanced Sets")
    st.caption("This runs only when you click the button. It first uses the current ROI/dupes filters, then takes up to the top 300 by ROI.")

    if st.button("Build Top 300 and Split into Two Sets"):
        if len(filtered_df) < 2:
            st.error("Need at least 2 filtered lineups to split.")
        else:
            top_n = min(300, len(filtered_df))
            top_df = filtered_df.head(top_n).copy()
            target_a = (top_n + 1) // 2
            target_b = top_n // 2

            exp_diff = Counter()
            idx_a = []
            idx_b = []

            for idx, row in top_df.iterrows():
                names = [parsed_names.at[idx, c] for c in fighter_cols]
                names = [n for n in names if n]

                if len(idx_a) >= target_a:
                    choose_a = False
                elif len(idx_b) >= target_b:
                    choose_a = True
                else:
                    diff_a = exp_diff.copy()
                    for n in names:
                        diff_a[n] += 1
                    score_a = sum(v * v for v in diff_a.values()) + 3 * (len(idx_a) + 1 - len(idx_b)) ** 2

                    diff_b = exp_diff.copy()
                    for n in names:
                        diff_b[n] -= 1
                    score_b = sum(v * v for v in diff_b.values()) + 3 * (len(idx_a) - (len(idx_b) + 1)) ** 2
                    choose_a = score_a <= score_b

                if choose_a:
                    idx_a.append(idx)
                    for n in names:
                        exp_diff[n] += 1
                else:
                    idx_b.append(idx)
                    for n in names:
                        exp_diff[n] -= 1

            set_a = top_df.loc[idx_a].copy()
            set_b = top_df.loc[idx_b].copy()

            def exposure_summary(df_set, label):
                counts = Counter()
                cpt_counts = Counter()
                for idx in df_set.index:
                    for c in fighter_cols:
                        nm = parsed_names.at[idx, c]
                        if nm:
                            counts[nm] += 1
                            if is_showdown and c == "CPT":
                                cpt_counts[nm] += 1

                total = len(df_set)
                rows = []
                for nm, count in counts.items():
                    row = {
                        "Player": nm,
                        f"{label} Times Used": count,
                        f"{label} Exposure %": 100.0 * count / total if total else 0.0,
                    }
                    if is_showdown:
                        row[f"{label} CPT Times"] = cpt_counts.get(nm, 0)
                        row[f"{label} CPT Exposure %"] = 100.0 * cpt_counts.get(nm, 0) / total if total else 0.0
                    rows.append(row)
                return pd.DataFrame(rows).sort_values(f"{label} Exposure %", ascending=False)

            exp_a = exposure_summary(set_a, "Set A")
            exp_b = exposure_summary(set_b, "Set B")

            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"### Set A — {len(set_a)} lineups")
                st.dataframe(exp_a, use_container_width=True)
            with col_b:
                st.write(f"### Set B — {len(set_b)} lineups")
                st.dataframe(exp_b, use_container_width=True)

            st.write(
                f"Average ROI — Set A: **{pd.to_numeric(set_a[roi_col], errors='coerce').mean():.4f}** | "
                f"Set B: **{pd.to_numeric(set_b[roi_col], errors='coerce').mean():.4f}**"
            )
            st.write(
                f"Average Projected Dupes — Set A: **{set_a['Projected Dupes'].mean():.3f}** | "
                f"Set B: **{set_b['Projected Dupes'].mean():.3f}**"
            )

            st.download_button(
                "Download Set A",
                data=set_a.to_csv(index=False).encode("utf-8"),
                file_name=f"top{len(set_a)}_setA.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download Set B",
                data=set_b.to_csv(index=False).encode("utf-8"),
                file_name=f"top{len(set_b)}_setB.csv",
                mime="text/csv",
            )
