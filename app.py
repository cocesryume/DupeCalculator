import streamlit as st
import pandas as pd
import numpy as np
import re

st.title("DFS Lineup Duplication Calculator")
st.write("Upload a Lineups CSV and Ownership CSV to estimate field-level expected dupes.")

# ======================
# File Uploads & Inputs
# ======================
lineup_file = st.file_uploader("Upload Lineups CSV", type=["csv"])
own_file = st.file_uploader("Upload Ownership CSV", type=["csv"])
contest_size = st.number_input("Contest Size (total entries in the contest)", min_value=1, value=73529)

# -------------
# Helpers
# -------------
def salary_multiplier(s):
    try:
        s = float(s)
    except:
        return 1.0
    if s >= 50000: return 1.75
    if s >= 49900: return 1.30
    if s >= 49800: return 1.00
    if s >= 49700: return 0.80
    return 0.60

def extract_id_from_string(x):
    """
    Handle cases like:
    - 'Player Name (41081548)'
    - '41081548'
    - 'Player Name - 41081548'
    """
    s = str(x)
    m = re.search(r"\((\d+)\)", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", s)
    if m2:
        return int(m2.group(1))
    return np.nan

# ======================
# RUN BUTTON
# ======================
if st.button("Run Dupes"):

    if lineup_file is None or own_file is None:
        st.error("Please upload BOTH a lineups CSV and an ownership CSV.")
        st.stop()

    # Load CSVs
    lineups = pd.read_csv(lineup_file)
    own = pd.read_csv(own_file)

    st.write("Lineups columns:", list(lineups.columns))
    st.write("Ownership columns:", list(own.columns))

    # ===== Detect format (Showdown vs MMA/PGA) =====
    is_showdown = "CPT" in lineups.columns
    if is_showdown:
        st.write("✅ Detected format: NFL Showdown (CPT + 5 FLEX).")
        fighter_cols = ["CPT", "FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]
        gamma = 0.12
    else:
        st.write("✅ Detected format: MMA / PGA (6 fighters).")
        fighter_cols = ["F", "F.1", "F.2", "F.3", "F.4", "F.5"]
        gamma = 0.10

    # ========= Ownership mapping =========
    if "DFS ID" not in own.columns:
        st.error("Ownership file must include a 'DFS ID' column.")
        st.stop()

    own["DFS ID"] = own["DFS ID"].astype(int)

    if is_showdown:
        flex_map = dict(zip(own["DFS ID"], own["Ownership"] / 100.0))
        cpt_map = dict(zip(own["DFS ID"], own["CPTOwnership"] / 100.0))
    else:
        own_map = dict(zip(own["DFS ID"], own["Ownership"] / 100.0))

    # ========= Projection & salary columns detection =========
    proj_col = None
    for c in lineups.columns:
        cu = c.upper()
        if "PROJ" in cu or "MEDIAN" in cu or "FPTS" in cu or "SCORE" in cu:
            proj_col = c
            break

    if proj_col is None:
        st.error("Could not detect a projection column.")
        st.stop()

    sal_col = None
    for c in lineups.columns:
        if "SAL" in c.upper():
            sal_col = c
            break
    if sal_col is None and "Salary" in lineups.columns:
        sal_col = "Salary"

    if sal_col is None:
        st.error("Could not detect a salary column.")
        st.stop()

    st.write(f"Using projection column: **{proj_col}**")
    st.write(f"Using salary column: **{sal_col}**")

    # ========= Convert fighter columns to DFS IDs =========
    if "Name" in own.columns:
        own["Name_clean"] = own["Name"].astype(str).str.upper().str.strip()
        name_to_id = dict(zip(own["Name_clean"], own["DFS ID"]))
    else:
        name_to_id = {}

    for col in fighter_cols:
        if col not in lineups.columns:
            st.error(f"Expected lineup column '{col}' not found.")
            st.stop()

        if not np.issubdtype(lineups[col].dtype, np.number):
            tmp_ids = lineups[col].apply(extract_id_from_string)

            # fallback: name-based mapping
            if tmp_ids.isna().mean() > 0.5 and name_to_id:
                tmp_ids = lineups[col].astype(str)\
                                      .apply(lambda x: x.split("(")[0].strip().upper())\
                                      .map(name_to_id)

            lineups[col] = tmp_ids

    # Drop unmappable lineups
    lineups = lineups.dropna(subset=fighter_cols).copy()
    for col in fighter_cols:
        lineups[col] = lineups[col].astype(int)

    # ========= Prepare DUPES functions =========
    P_opt = lineups[proj_col].max()

    def expected_dupes_showdown(row):
        try:
            cpt = int(row["CPT"])
            flex_ids = [int(row[c]) for c in ["FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4"]]
        except:
            return 0.0

        prob = (cpt_map.get(cpt, 0.0001) ** 1.4)
        for f in flex_ids:
            prob *= flex_map.get(f, 0.0001)

        fS = salary_multiplier(row[sal_col])
        fP = float(np.exp(-gamma * (P_opt - row[proj_col])))

        return contest_size * prob * fS * fP

    def expected_dupes_mma(row):
        try:
            ids = [int(row[c]) for c in ["F", "F.1", "F.2", "F.3", "F.4", "F.5"]]
        except:
            return 0.0

        p = 1.0
        for f in ids:
            p *= own_map.get(f, 0.0001)

        fS = salary_multiplier(row[sal_col])
        fP = float(np.exp(-gamma * (P_opt - row[proj_col])))

        return contest_size * p * fS * fP

    # ========= Apply DUPES =========
    if is_showdown:
        lineups["Projected Dupes"] = lineups.apply(expected_dupes_showdown, axis=1)
    else:
        lineups["Projected Dupes"] = lineups.apply(expected_dupes_mma, axis=1)

    # ========= Scale to contest size =========
    raw_sum = lineups["Projected Dupes"].sum()
    scale = contest_size / raw_sum if raw_sum > 0 else 1.0

    lineups["Projected Dupes"] = lineups["Projected Dupes"] * scale

    # ========= Final output (remove extras) =========
    df_out = lineups.copy()

    # Remove internal-only columns if present
    for col in ["expected_dupes", "expected_dupes_scaled", "combo_key"]:
        if col in df_out.columns:
            df_out.drop(columns=[col], inplace=True)

    # ========= Combo-level summary (without combo_key in output) =========
    def combo_key(row):
        try:
            return "-".join(map(str, sorted([int(row[c]) for c in fighter_cols])))
        except:
            return "NA"

    df_out["combo_key"] = df_out.apply(combo_key, axis=1)
    valid = df_out[df_out["combo_key"] != "NA"].copy()

    summary = valid.groupby("combo_key").agg({
        "Projected Dupes": "sum",
        proj_col: "mean",
        sal_col: "mean"
    }).reset_index()

    # Remove combo_key from the downloadable summary
    summary = summary.drop(columns=["combo_key"])

    st.success("Duplication modeling complete.")
    st.write(f"Scaling factor used: {scale:.3e}")
    st.dataframe(summary.head(10))

    # Downloads
    st.download_button(
        label="Download Lineups With Projected Dupes CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="lineups_with_projected_dupes.csv"
    )

    st.download_button(
        label="Download Combo Summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="combo_summary.csv"
    )

