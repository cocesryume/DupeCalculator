import io
import math
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='FanDuel NFL Showdown Dupe Calculator', layout='wide')
st.title('FanDuel NFL Showdown Dupe Calculator — V1.2')
st.caption(
    'Experimental FanDuel single-game dupe estimator. Uses MVP/UTIL ownership, salary usage and projection. '
    'Ownership file may be the simple SaberSim name/fpts/util ownership/mvp ownership format.'
)

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
ownership_file = st.file_uploader(
    'Upload FanDuel ownership CSV (supports name / fpts / util ownership / mvp ownership)',
    type=['csv'],
)
mapping_file = st.file_uploader(
    'Upload FanDuel player mapping CSV (DFS ID + Name; required when ownership file has no DFS ID)',
    type=['csv'],
)
contest_size = st.number_input('Contest Size', min_value=1, value=50000, step=1)
salary_cap = st.number_input('FanDuel Salary Cap', min_value=1, value=60000, step=100)


def read_csv(upload):
    """Read a Streamlit-uploaded CSV with a couple of safe fallbacks."""
    raw = upload.getvalue()
    last_exc = None
    for encoding in ('utf-8-sig', 'utf-8', 'latin1'):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            last_exc = exc
    # Final fallback: Python parser can tolerate some unusual quoting better.
    try:
        return pd.read_csv(io.BytesIO(raw), engine='python')
    except Exception:
        raise last_exc


def norm_id(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def normalize_name(x):
    if pd.isna(x):
        return ''
    s = str(x).replace('\u00a0', ' ').strip().lower()
    s = ' '.join(s.split())
    return s


def pct_to_decimal(series):
    s = (
        series.astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    x = pd.to_numeric(s, errors='coerce')
    valid = x.dropna()
    if not valid.empty and valid.max() > 1.5:
        x = x / 100.0
    return x


def detect_col(df, exact=(), contains=(), exclude_contains=()):
    lowers = {str(c).strip().lower(): c for c in df.columns}
    for e in exact:
        if e.lower() in lowers:
            return lowers[e.lower()]
    for c in df.columns:
        lc = str(c).strip().lower()
        if exclude_contains and any(x.lower() in lc for x in exclude_contains):
            continue
        if any(x.lower() in lc for x in contains):
            return c
    return None


def salary_factor(salary, cap):
    """Modest optimizer-naturalness bump for using most of the cap."""
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
    # Intentionally mild until we have more FanDuel actual-dupe history.
    gap = max(float(best_proj) - float(proj), 0.0)
    return math.exp(-0.035 * gap)


if st.button('Run FanDuel Dupes'):
    if lineup_file is None or ownership_file is None:
        st.error('Please upload both the lineup CSV and FanDuel ownership CSV.')
        st.stop()

    try:
        lineups = read_csv(lineup_file)
        own = read_csv(ownership_file)
        mapping = read_csv(mapping_file) if mapping_file is not None else None

        lineup_cols = [
            'MVP - 1.5X Points',
            'AnyFLEX',
            'AnyFLEX.1',
            'AnyFLEX.2',
            'AnyFLEX.3',
            'AnyFLEX.4',
        ]
        missing = [c for c in lineup_cols if c not in lineups.columns]
        if missing:
            raise ValueError(f'Missing expected FanDuel lineup columns: {missing}')

        proj_col = detect_col(lineups, exact=['Proj Score', 'Projection', 'Proj'])
        sal_col = detect_col(lineups, exact=['Salary'], contains=['salary'])
        if proj_col is None or sal_col is None:
            raise ValueError(f'Could not detect Proj Score / Salary columns. Found: {list(lineups.columns)}')

        # -----------------------------
        # Ownership file: allow the simple SaberSim format
        # name, fpts, util ownership, mvp ownership
        # -----------------------------
        own_name_col = detect_col(own, exact=['Name', 'Player', 'Player Name'])
        own_id_col = detect_col(own, exact=['DFS ID', 'ID'])
        mvp_own_col = detect_col(
            own,
            exact=['MVP Own', 'mvp ownership'],
            contains=['mvp own', 'mvp ownership'],
        )
        util_own_col = detect_col(
            own,
            exact=['AnyFLEX Own', 'util ownership'],
            contains=['anyflex own', 'util ownership', 'flex ownership'],
            exclude_contains=['mvp'],
        )

        if own_name_col is None or mvp_own_col is None or util_own_col is None:
            raise ValueError(
                'Ownership file must contain Name, MVP ownership, and AnyFLEX/UTIL ownership. '
                f'Found columns: {list(own.columns)}'
            )

        own2 = own.copy()
        own2['_NAME_KEY'] = own2[own_name_col].map(normalize_name)
        own2['_MVP_OWN'] = pct_to_decimal(own2[mvp_own_col])
        own2['_UTIL_OWN'] = pct_to_decimal(own2[util_own_col])

        # Blank ownerships are treated as zero-ish instead of dropping the player.
        own2['_MVP_OWN'] = own2['_MVP_OWN'].fillna(0.0)
        own2['_UTIL_OWN'] = own2['_UTIL_OWN'].fillna(0.0)
        own2 = own2[own2['_NAME_KEY'] != '']

        name_to_mvp = dict(zip(own2['_NAME_KEY'], own2['_MVP_OWN']))
        name_to_util = dict(zip(own2['_NAME_KEY'], own2['_UTIL_OWN']))

        # Optional direct ID mappings if ownership file happens to include IDs.
        id_to_mvp = {}
        id_to_util = {}
        id_to_name = {}
        if own_id_col is not None:
            own2['_ID'] = own2[own_id_col].map(norm_id)
            own_id_good = own2[own2['_ID'] != '']
            id_to_mvp = dict(zip(own_id_good['_ID'], own_id_good['_MVP_OWN']))
            id_to_util = dict(zip(own_id_good['_ID'], own_id_good['_UTIL_OWN']))
            id_to_name = dict(zip(own_id_good['_ID'], own_id_good['_NAME_KEY']))

        # If ownership is name-based (your normal format), use the mapping file as ID -> Name bridge.
        if not id_to_name:
            if mapping is None:
                raise ValueError(
                    'This ownership file is name-based and has no DFS ID. '
                    'Please upload the FanDuel player mapping CSV containing DFS ID and Name.'
                )
            map_id_col = detect_col(mapping, exact=['DFS ID', 'ID'])
            map_name_col = detect_col(mapping, exact=['Name', 'Player', 'Player Name'])
            if map_id_col is None or map_name_col is None:
                raise ValueError(
                    'Player mapping file must contain DFS ID and Name. '
                    f'Found columns: {list(mapping.columns)}'
                )
            map2 = mapping.copy()
            map2['_ID'] = map2[map_id_col].map(norm_id)
            map2['_NAME_KEY'] = map2[map_name_col].map(normalize_name)
            map2 = map2[(map2['_ID'] != '') & (map2['_NAME_KEY'] != '')]
            id_to_name = dict(zip(map2['_ID'], map2['_NAME_KEY']))

        out = lineups.copy()
        out[proj_col] = pd.to_numeric(out[proj_col], errors='coerce')
        out[sal_col] = pd.to_numeric(out[sal_col], errors='coerce')
        best_proj = out[proj_col].max()

        parsed_names = pd.DataFrame(index=out.index, columns=lineup_cols, dtype='object')
        slot_owns = pd.DataFrame(index=out.index, columns=lineup_cols, dtype='float64')
        missing_slots = []

        for c in lineup_cols:
            is_mvp = c == 'MVP - 1.5X Points'
            for idx, value in out[c].items():
                pid = norm_id(value)
                name_key = id_to_name.get(pid, '')
                parsed_names.at[idx, c] = name_key or pid

                if is_mvp:
                    own_value = id_to_mvp.get(pid) if id_to_mvp else None
                    if own_value is None and name_key:
                        own_value = name_to_mvp.get(name_key)
                else:
                    own_value = id_to_util.get(pid) if id_to_util else None
                    if own_value is None and name_key:
                        own_value = name_to_util.get(name_key)

                if own_value is None or pd.isna(own_value):
                    missing_slots.append((idx, c, pid, name_key))
                    own_value = 0.0001
                else:
                    # Keep true 0% ownership tiny-but-nonzero for numerical stability.
                    own_value = max(float(own_value), 0.000001)

                slot_owns.at[idx, c] = float(own_value)

        expected = []
        own_sum = []
        own_gmean = []
        salary_left = []

        for idx, r in out.iterrows():
            vals = [float(slot_owns.at[idx, c]) for c in lineup_cols]
            base_prob = float(np.prod(vals))
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

        total_slots = len(out) * len(lineup_cols)
        matched_slots = total_slots - len(missing_slots)
        coverage = 100.0 * matched_slots / total_slots if total_slots else 0.0

        st.success(f'Calculated FanDuel projected dupes for {len(out):,} lineups.')
        st.write(f'Projection column: **{proj_col}** | Salary column: **{sal_col}**')
        st.write(f'Ownership mapping coverage: **{coverage:.2f}%** of lineup slots matched.')

        if missing_slots:
            st.warning(f'{len(missing_slots):,} lineup slots used fallback ownership because no player match was found.')
            with st.expander('Show missing player/ownership matches'):
                st.dataframe(
                    pd.DataFrame(missing_slots, columns=['Row', 'Slot', 'FD ID', 'Mapped Name']).head(200),
                    use_container_width=True,
                )
        else:
            st.info('Ownership coverage check: 100% of lineup player slots matched the ownership file.')

        sim_dupe_cols = [c for c in out.columns if 'Sim Dupes' in str(c)]
        if sim_dupe_cols:
            st.info(
                'SaberSim Sim Dupes columns were detected and preserved in the output so you can compare them '
                'with this experimental FanDuel model.'
            )

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
    filtered = df[(df['Projected Dupes'] < max_dupes) & (roi_num >= min_roi)].copy()
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
                    sa = sum(v * v for v in da.values()) + 3 * (len(a) + 1 - len(b)) ** 2
                    sb = sum(v * v for v in db.values()) + 3 * (len(a) - (len(b) + 1)) ** 2
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
st.caption(
    'FD V1.2 supports SaberSim ownership files with name / fpts / util ownership / mvp ownership. '
    'After the contest, send the actual FanDuel dupe counts plus the full pre-lock output so we can calibrate V2.'
)
