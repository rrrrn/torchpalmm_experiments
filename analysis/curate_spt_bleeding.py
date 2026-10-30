"""Curate the long-format dataset for the bleeding-percentage additive mixed model (f1..f12).

One row per supportive periodontal therapy (SPT) visit. Visit 1 has no previous visit: its interval is 0
and its lagged predictors are NaN. Reads the PhysioNet files in analysis/spt_data/ and writes curated_spt_bleeding.csv there.
Usage: python3 -I curate_spt_bleeding.py
"""
import numpy as np
import pandas as pd

import os
src = os.path.dirname(os.path.abspath(__file__)) + '/spt_data/'
os.chdir(src)
apt = pd.read_csv(src + '01_initial_periodontal_therapy_data.csv')
spt = pd.read_csv(src + '02_supportive_periodontal_therapy_data.csv')
sup = pd.read_csv(src + '03_supplementary_data.csv')

spt = spt.sort_values(['pat_id', 'spt_id']).reset_index(drop=True)
## pat 4237 has no visit 9: its visit 10 takes visit 8 as the previous visit (interval 217 d spans the missing visit)
spt['after_missing_visit'] = (spt.groupby('pat_id').spt_id.diff() > 1).astype(int)

## visit-level quantities (six sites per tooth for probing depth, four for bleeding)
d6 = spt.n_teeth_spt * 6
spt['bleeding_pct'] = spt.n_bop_pos_spt / (spt.n_teeth_spt * 4) * 100                       # Y
spt['percent_from5mm_spt'] = (spt.n_5mm_spt + spt.n_6mm_spt + spt.n_from7mm_spt) / d6 * 100
spt['percent_from6mm_spt'] = (spt.n_6mm_spt + spt.n_from7mm_spt) / d6 * 100
## agree with the published derived file
chk = spt.merge(sup[['pat_id', 'spt_id', 'percent_from5mm_spt', 'percent_from6mm_spt']], on=['pat_id', 'spt_id'], suffixes=('', '_pub'))
assert np.allclose(chk.percent_from5mm_spt, chk.percent_from5mm_spt_pub, equal_nan=True)
assert np.allclose(chk.percent_from6mm_spt, chk.percent_from6mm_spt_pub, equal_nan=True)

g = spt.groupby('pat_id')
spt['cum_followup_years'] = g.true_interval_days.cumsum() / 365.25                          # f1 (first visit = 0)
spt['prev_percent_from5mm'] = g.percent_from5mm_spt.shift(1)                                # f4
spt['prev_percent_from6mm'] = g.percent_from6mm_spt.shift(1)                                # f5
spt['prev_bleeding_pct'] = g.bleeding_pct.shift(1)                                          # f10
spt['n_previous_visits'] = spt.spt_id - 1                                                   # f12

## patient-level quantities
apt['pct_from4mm_after_apt'] = (apt[['n_4mm_follow_up', 'n_5mm_follow_up', 'n_6mm_follow_up', 'n_from7mm_follow_up']].sum(axis=1)
                                / (apt.n_teeth_follow_up * 6) * 100)                       # f8
apt['pct_from5mm_baseline'] = (apt.n_5mm_baseline + apt.n_6mm_baseline + apt.n_from7mm_baseline) / (apt.n_teeth_baseline * 6) * 100
apt['pct_from6mm_baseline'] = (apt.n_6mm_baseline + apt.n_from7mm_baseline) / (apt.n_teeth_baseline * 6) * 100  # f9
apt = apt.rename(columns={'gender': 'sex', 'diagnosis': 'baseline_diagnosis', 'smoking': 'smoking_status'})
pat_cols = ['pat_id', 'sex', 'age_baseline', 'smoking_status', 'baseline_diagnosis', 'diagnosis_source', 'med_history',
            'n_teeth_baseline', 'apt_duration_days', 'pct_from4mm_after_apt', 'pct_from5mm_baseline', 'pct_from6mm_baseline',
            'spt_duration_years', 'subsample_5years']

df = spt.merge(apt[pat_cols], on='pat_id', how='left', validate='m:1')
df = df.rename(columns={'true_interval_days': 'days_since_prev_visit', 'spt_year': 'calendar_year'})
df['sex'] = df.sex.astype('category')
df['smoking_status'] = pd.Categorical(df.smoking_status, ['non-smoker', 'former smoker', 'smoker'])
df['baseline_diagnosis'] = df.baseline_diagnosis.astype('Int64').astype('category')

cols = ['pat_id', 'spt_id', 'age_at_spt', 'bleeding_pct',
        'cum_followup_years', 'age_baseline', 'days_since_prev_visit', 'prev_percent_from5mm', 'prev_percent_from6mm',
        'n_teeth_baseline', 'apt_duration_days', 'pct_from4mm_after_apt', 'pct_from5mm_baseline', 'pct_from6mm_baseline', 'prev_bleeding_pct',
        'calendar_year', 'n_previous_visits',
        'smoking_status', 'sex', 'baseline_diagnosis', 'diagnosis_source', 'med_history', 'subsample_5years',
        'n_teeth_spt', 'imputed', 'corrected', 'after_missing_visit']
df = df[cols].reset_index(drop=True)

df.to_csv('curated_spt_bleeding.csv', index=False)
df[df.subsample_5years == 1].to_csv('curated_spt_bleeding_subsample445.csv', index=False)

print('rows', len(df), 'patients', df.pat_id.nunique(), '| subsample rows', int((df.subsample_5years == 1).sum()))
print('complete cases (current-visit smooths + covariates):',
      len(df.dropna(subset=cols[2:9] + cols[9:10] + cols[11:12] + ['smoking_status', 'sex', 'baseline_diagnosis', 'bleeding_pct'])))
print(df.isna().sum()[df.isna().sum() > 0])
print(df.describe().T[['count', 'mean', 'min', 'max']])
