import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter

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
    "Upload DK Salaries CSV (for Showdown CPT/FLEX mapping)",
    type=["csv"],
)
contest_size = st.number_input("Contest Size", min_value=1, value=73529)


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
        st.write("✅ Detected NFL Showdown (CPT + 5 FLEX).")
        fighter_cols = ["CPT", "FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]
        gamma = 0.12
    else:
        st.write("✅ Detected MMA / PGA (6 fighters).")
        fighter_cols = ["F", "F.1", "F.2", "F.3", "F.4", "F.5"]
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
    # DK SALARIES CPT/FLEX MAP
    # ========================
    id_to_name = {}

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

    # Apply dupes
    if is_showdown:
        lineups["Projected Dupes"] = lineups.apply(expected_showdown, axis=1)
    else:
        lineups["Projected Dupes"] = lineups.apply(expected_mma, axis=1)

    total_raw = lineups["Projected Dupes"].sum()
    scale = contest_size / total_raw if total_raw > 0 else 1.0
    lineups["Projected Dupes"] *= scale

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

