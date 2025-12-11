import streamlit as st
import pandas as pd
import numpy as np
import re

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


# ======================
# File Uploads & Inputs
# ======================
lineup_file = st.file_uploader("Upload Lineups CSV", type=["csv"])
own_file = st.file_uploader("Upload Ownership CSV", type=["csv"])
contest_size = st.number_input("Contest Size", min_value=1, value=73529)

# -------------
# Helpers
# -------------
def salary_multiplier(s):
    try: s = float(s)
    except: return 1.0
    if s >= 50000: return 1.75
    if s >= 49900: return 1.30
    if s >= 49800: return 1.00
    if s >= 49700: return 0.80
    return 0.60

def extract_id(x):
    s = str(x)
    m = re.search(r"\((\d+)\)", s)
    if m: return int(m.group(1))
    m2 = re.search(r"(\d+)", s)
    if m2: return int(m2.group(1))
    return np.nan


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

    # Detect format
    is_showdown = "CPT" in lineups.columns
    st.session_state.is_showdown = is_showdown

    if is_showdown:
        fighter_cols = ["CPT", "FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]
        gamma = 0.12
    else:
        fighter_cols = ["F", "F.1", "F.2", "F.3", "F.4", "F.5"]
        gamma = 0.10

    st.session_state.fighter_cols = fighter_cols

    # Ownership data
    own["DFS ID"] = own["DFS ID"].astype(int)

    if is_showdown:
        flex_map = dict(zip(own["DFS ID"], own["Ownership"] / 100))
        cpt_map = dict(zip(own["DFS ID"], own["CPTOwnership"] / 100))
    else:
        own_map = dict(zip(own["DFS ID"], own["Ownership"] / 100))

    # Detect projection column
    proj_col = None
    for c in lineups.columns:
        if any(k in c.upper() for k in ["PROJ", "SCORE", "MEDIAN", "FPTS"]):
            proj_col = c
            break
    if proj_col is None:
        st.error("Could not detect projection column.")
        st.stop()

    # Salary column
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

    # Convert names → DFS IDs
    name_to_id = {}
    if "Name" in own.columns:
        own["Name_clean"] = own["Name"].astype(str).str.upper().str.strip()
        name_to_id = dict(zip(own["Name_clean"], own["DFS ID"]))

    for col in fighter_cols:
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

    # Drop bad rows
    lineups = lineups.dropna(subset=fighter_cols).copy()
    for col in fighter_cols:
        lineups[col] = lineups[col].astype(int)

    P_opt = lineups[proj_col].max()

    # Dupes functions
    def expected_showdown(row):
        try:
            cpt = int(row["CPT"])
            flex_ids = [int(row[c]) for c in fighter_cols[1:]]
        except:
            return 0.0

        p = (cpt_map.get(cpt, 0.0001) ** 1.4)
        for f in flex_ids:
            p *= flex_map.get(f, 0.0001)

        return contest_size * p * salary_multiplier(row[sal_col]) * np.exp(-gamma * (P_opt - row[proj_col]))

    def expected_mma(row):
        try:
            ids = [int(row[c]) for c in fighter_cols]
        except:
            return 0.0

        p = 1.0
        for f in ids:
            p *= own_map.get(f, 0.0001)

        return contest_size * p * salary_multiplier(row[sal_col]) * np.exp(-gamma * (P_opt - row[proj_col]))

    # Apply dupes
    if is_showdown:
        lineups["Projected Dupes"] = lineups.apply(expected_showdown, axis=1)
    else:
        lineups["Projected Dupes"] = lineups.apply(expected_mma, axis=1)

    scale = contest_size / max(lineups["Projected Dupes"].sum(), 1e-12)
    lineups["Projected Dupes"] *= scale

    # Store output in session state
    st.session_state.df_out = lineups.copy()

    st.success("Dupes calculated! Scroll down to filter results.")


# ===================================================
# FILTER PANEL (only appears AFTER dupes are computed)
# ===================================================
if st.session_state.df_out is not None:

    df_out = st.session_state.df_out
    fighter_cols = st.session_state.fighter_cols

    st.header("Filter Lineups by ROI & Projected Dupes")

    # ROI column selector
    roi_col = st.selectbox(
        "Select ROI Column:",
        options=df_out.columns,
        help="Choose the ROI column (changes every slate)."
    )

    max_dupes = st.number_input(
        "Maximum allowed Projected Dupes:",
        min_value=0.0,
        value=50.0
    )

    min_roi = st.number_input(
        "Minimum required ROI:",
        value=0.0
    )

    # Apply filters
    filtered_df = df_out[
        (df_out["Projected Dupes"] <= max_dupes) &
        (df_out[roi_col] >= min_roi)
    ].copy()

    filtered_df = filtered_df.sort_values(by=roi_col, ascending=False)

    st.write(f"### Lineups that match your criteria: {len(filtered_df)}")
    st.dataframe(filtered_df.head(50))

    st.download_button(
        label="Download Filtered Lineups",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_lineups.csv"
    )

    # Download full results
    st.download_button(
        label="Download All Lineups With Projected Dupes",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="lineups_with_projected_dupes.csv"
    )
