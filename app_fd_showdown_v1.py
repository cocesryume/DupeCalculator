import io
import math
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='FanDuel NFL Showdown Dupe Calculator', layout='wide')
st.title('FanDuel NFL Showdown Dupe Calculator — V1')
st.caption('Experimental FanDuel single-game dupe estimator. Uses MVP/UTIL ownership, salary usage and projection. Save results so we can calibrate against actual dupes after contests.')

# -----------------------------
# Session state
# -----------------------------
for k, v in {
    'df_out': None,
    'parsed_names': None,
    'player_cols': None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Inputs
# -----------------------------
lineup_file = st.file_uploader('Upload FanDuel Showdown lineups CSV', type=['csv'])
player_file = st.file_uploader(
    'Upload FanDuel player/projection CSV (DFS ID, Name, Salary, MVP Own, AnyFLEX Own)',
    type=['csv'],
)
contest_size = st.number_input('Contest Size', min_value=1, value=50000, step=1)
salary_cap = st.number_input('FanDuel Salary Cap', min_value=1, value=60000, step=100)


def read_csv(upload):
    return pd.read_csv(io.BytesIO(upload.getvalue()))


def norm_id(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def pct_to_decimal(series):
    s = series.astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
    x = pd.to_numeric(s, errors='coerce')
    valid = x.dropna()
    if not valid.empty and valid.max() > 1.5:
        x = x / 100.0
    return x


def detect_col(df, exact=(), contains=()):
    lowers = {str(c).strip().lower(): c for c in df.columns}
    for e in exact:
        if e.lower() in lowers:
            return lowers[e.lower()]
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(x.lower() in lc for x in contains):
            return c
    return None


def salary_factor(salary, cap):
    """Modest 'optimizer-naturalness' bump for using most of the cap."""
    left = max(float(cap) - float(salary), 0.0)
    if left <= 100:
        return 1.35
    if left <= 500:
        return 1.25
    if left <= 1000:
        return 1.15
    if left <= 2000:
        return 1.05
    if left <= 4000:
        return 0.92
    return 0.82


def projection_factor(proj, best_proj):
    # Keep this intentionally mild until we have FD actual-dupe history.
    gap = max(float(best_proj) - float(proj), 0.0)
    return math.exp(-0.035 * gap)


if st.button('Run FanDuel Dupes'):
    if lineup_file is None or player_file is None:
        st.error('Please upload both the lineup CSV and FanDuel player/projection CSV.')
        st.stop()

    try:
        lineups = read_csv(lineup_file)
        players = read_csv(player_file)

        lineup_cols = ['MVP - 1.5X Points', 'AnyFLEX', 'AnyFLEX.1', 'AnyFLEX.2', 'AnyFLEX.3', 'AnyFLEX.4']
        missing = [c for c in lineup_cols if c not in lineups.columns]
        if missing:
            raise ValueError(f'Missing expected FanDuel lineup columns: {missing}')

        proj_col = detect_col(lineups, exact=['Proj Score', 'Projection', 'Proj'])
        sal_col = detect_col(lineups, exact=['Salary'], contains=['salary'])
        if proj_col is None or sal_col is None:
            raise ValueError(f'Could not detect Proj Score / Salary columns. Found: {list(lineups.columns)}')

        id_col = detect_col(players, exact=['DFS ID', 'ID'])
        name_col = detect_col(players, exact=['Name', 'Player', 'Player Name'])
        mvp_own_col = detect_col(players, exact=['MVP Own', 'mvp ownership'], contains=['mvp own', 'mvp ownership'])
        util_own_col = detect_col(players, exact=['AnyFLEX Own', 'util ownership'], contains=['anyflex own', 'util ownership'])
        if None in [id_col, name_col, mvp_own_col, util_own_col]:
            raise ValueError(
                'Player file must contain DFS ID, Name, MVP ownership and AnyFLEX/UTIL ownership. '
                f'Found columns: {list(players.columns)}'
            )

        p = players.copy()
        p['_ID'] = p[id_col].map(norm_id)
        p['_MVP_OWN'] = pct_to_decimal(p[mvp_own_col])
        p['_UTIL_OWN'] = pct_to_decimal(p[util_own_col])
        p = p.dropna(subset=['_MVP_OWN', '_UTIL_OWN'])
        p = p[p['_ID'] != '']

        id_to_name = dict(zip(p['_ID'], p[name_col].astype(str).str.strip()))
        mvp_map = dict(zip(p['_ID'], p['_MVP_OWN']))
        util_map = dict(zip(p['_ID'], p['_UTIL_OWN']))

        out = lineups.copy()
        out[proj_col] = pd.to_numeric(out[proj_col], errors='coerce')
        out[sal_col] = pd.to_numeric(out[sal_col], errors='coerce')
        best_proj = out[proj_col].max()

        parsed_names = pd.DataFrame(index=out.index, columns=lineup_cols, dtype='object')
        missing_slots = []

        # Precompute slot ownership / names
        slot_owns = pd.DataFrame(index=out.index, columns=lineup_cols, dtype='float64')
        for c in lineup_cols:
            is_mvp = c == 'MVP - 1.5X Points'
            lookup = mvp_map if is_mvp else util_map
            for idx, value in out[c].items():
                pid = norm_id(value)
                parsed_names.at[idx, c] = id_to_name.get(pid, pid)
                own = lookup.get(pid)
                if own is None or pd.isna(own):
                    missing_slots.append((idx, c, pid))
                    own = 0.0001
                slot_owns.at[idx, c] = float(own)

        # Build features and predicted dupes
        expected = []
        own_sum = []
        own_gmean = []
        salary_left = []
        for idx, r in out.iterrows():
            vals = [float(slot_owns.at[idx, c]) for c in lineup_cols]
            base_prob = np.prod(vals)
            sf = salary_factor(r[sal_col], salary_cap)
            pf = projection_factor(r[proj_col], best_proj)
            pred = float(contest_size) * base_prob * sf * pf
            expected.append(pred)
            own_sum.append(100.0 * sum(vals))
            own_gmean.append(100.0 * float(np.prod(vals) ** (1.0 / len(vals))))
            salary_left.append(float(salary_cap) - float(r[sal_col]))

        out['Projected Dupes'] = expected
        out['FD Own Sum'] = own_sum
        out['FD Own GeoMean'] = own_gmean
        out['Salary Left'] = salary_left

        st.session_state.df_out = out
        st.session_state.parsed_names = parsed_names
        st.session_state.player_cols = lineup_cols

        st.success(f'Calculated FanDuel projected dupes for {len(out):,} lineups.')
        st.write(f'Projection column: **{proj_col}** | Salary column: **{sal_col}**')
        st.write(f'Player mapping coverage: **{100*(1-len(missing_slots)/(len(out)*6)):.2f}%** of lineup slots matched ownership.')

        if missing_slots:
            st.warning(f'{len(missing_slots):,} lineup slots used fallback ownership because no player match was found.')
            with st.expander('Show missing player/ownership matches'):
                st.dataframe(pd.DataFrame(missing_slots, columns=['Row', 'Slot', 'FD ID']).head(100), use_container_width=True)

        sim_dupe_cols = [c for c in out.columns if 'Sim Dupes' in str(c)]
        if sim_dupe_cols:
            st.info('SaberSim Sim Dupes columns were detected and preserved in the output so you can compare them with this experimental FD model.')

    except Exception as exc:
        st.error(f'Could not run FanDuel dupes: {exc}')
        st.stop()


# -----------------------------
# Filtering / downloads
# -----------------------------
if st.session_state.df_out is not None:
    df = st.session_state.df_out
    parsed_names = st.session_state.parsed_names
    lineup_cols = st.session_state.player_cols

    st.divider()
    st.header('Filter FanDuel Lineups')

    numeric_candidates = []
    for c in df.columns:
        if pd.to_numeric(df[c], errors='coerce').notna().sum() > 0:
            numeric_candidates.append(c)

    default_roi_idx = 0
    roi_like = [c for c in numeric_candidates if 'ROI' in str(c).upper()]
    if roi_like:
        default_roi_idx = numeric_candidates.index(roi_like[-1])

    roi_col = st.selectbox('Select ROI / ranking column', numeric_candidates, index=default_roi_idx)
    max_dupes = st.number_input('Maximum allowed Projected Dupes', min_value=0.0, value=9.0, step=0.1)
    min_roi = st.number_input('Minimum required ROI', value=0.0, step=0.01)

    roi_num = pd.to_numeric(df[roi_col], errors='coerce')
    filtered = df[(df['Projected Dupes'] <= max_dupes) & (roi_num >= min_roi)].copy()
    filtered['_ROI_SORT'] = pd.to_numeric(filtered[roi_col], errors='coerce')
    filtered = filtered.sort_values('_ROI_SORT', ascending=False).drop(columns=['_ROI_SORT'])

    st.write(f'### {len(filtered):,} lineups match your criteria')
    st.dataframe(filtered.head(100), use_container_width=True)

    st.download_button(
        'Download Filtered FanDuel Lineups',
        data=filtered.to_csv(index=False).encode('utf-8'),
        file_name='filtered_fd_showdown_lineups.csv',
        mime='text/csv',
    )
    st.download_button(
        'Download All FanDuel Lineups With Projected Dupes',
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='fd_showdown_all_with_projected_dupes.csv',
        mime='text/csv',
    )

    # Optional 150-lineup portfolio split, same spirit as DK app
    st.subheader('Optional: Top 300 → Two Balanced 150-Lineup Sets')
    if st.button('Build Balanced Sets'):
        if len(filtered) < 2:
            st.error('Need at least 2 filtered lineups.')
        else:
            top = filtered.head(min(300, len(filtered))).copy()
            n = len(top)
            target_a = (n + 1) // 2
            target_b = n // 2
            diff = Counter()
            a, b = [], []

            for idx in top.index:
                names = [parsed_names.at[idx, c] for c in lineup_cols if parsed_names.at[idx, c]]
                if len(a) >= target_a:
                    choose_a = False
                elif len(b) >= target_b:
                    choose_a = True
                else:
                    da = diff.copy()
                    db = diff.copy()
                    for nm in names:
                        da[nm] += 1
                        db[nm] -= 1
                    sa = sum(v*v for v in da.values()) + 3*(len(a)+1-len(b))**2
                    sb = sum(v*v for v in db.values()) + 3*(len(a)-(len(b)+1))**2
                    choose_a = sa <= sb
                if choose_a:
                    a.append(idx)
                    for nm in names:
                        diff[nm] += 1
                else:
                    b.append(idx)
                    for nm in names:
                        diff[nm] -= 1

            set_a = top.loc[a]
            set_b = top.loc[b]
            st.write(f'Set A: **{len(set_a)}** | Set B: **{len(set_b)}**')
            st.download_button('Download Set A', set_a.to_csv(index=False).encode('utf-8'), 'fd_setA.csv', 'text/csv')
            st.download_button('Download Set B', set_b.to_csv(index=False).encode('utf-8'), 'fd_setB.csv', 'text/csv')

st.divider()
st.caption('FD V1 is intentionally experimental. After the contest, send the actual FanDuel dupe counts plus the full pre-lock output and we can calibrate V2 on real FanDuel field behavior.')
