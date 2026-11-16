"""Reference analysis for Paper 1 (IJMECS). Input: merged_wide CSV. Output: results.json + tables/*.csv
Reference implementation of the aggregation rule is recomputed from D1..D5 so every number is reproducible.
"""
import sys, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, mannwhitneyu, kruskal, chi2_contingency
from sklearn.metrics import cohen_kappa_score, roc_auc_score
try:                                   # optional: used only for a supplementary statistic that the paper does not report
    import krippendorff
except ImportError:
    krippendorff = None

rng = np.random.default_rng(20261018)
NBOOT = 2000
src = sys.argv[1]; outdir = sys.argv[2]
os.makedirs(f'{outdir}/tables', exist_ok=True)
df = pd.read_csv(src)
D = ['D1', 'D2', 'D3', 'D4', 'D5']
R = {}

# ---------- reference implementation of the rule ----------
def band_exact(s, n):
    # Bands are decided in integer arithmetic (same rule as risk_engine.py):
    # mean >= 9/5 <=> 5*S >= 9*n, etc., so a mean of exactly 1.8 is always level 2.
    if n == 0: return np.nan
    return 1 + sum(5 * s >= e * n for e in (9, 13, 17, 21))
df['Rmean'] = df[D].mean(axis=1)             # mean of available dimensions (one case has 4)
df['ref_base'] = [band_exact(int(s), int(n)) for s, n in zip(df[D].sum(axis=1), df[D].notna().sum(axis=1))]
crit = (df[['D2', 'D3']] == 5).any(axis=1)
df['ref_final'] = np.where(crit, 5, df.ref_base)
df['override'] = crit & (df.ref_base < 5)
R['n'] = int(len(df))
R['n_override'] = int(df.override.sum())

def stage(g):
    if g.startswith('อ.'): return 'Kindergarten (K1-K3)'
    if g in ('ป.1', 'ป.2', 'ป.3'): return 'Lower primary (G1-G3)'
    if g in ('ป.4', 'ป.5', 'ป.6'): return 'Upper primary (G4-G6)'
    return 'Lower secondary (G7-G9)'
order = ['Kindergarten (K1-K3)', 'Lower primary (G1-G3)', 'Upper primary (G4-G6)', 'Lower secondary (G7-G9)']
df['stage'] = pd.Categorical(df.grade.map(stage), order, ordered=True)

# ---------- Table: sample ----------
intact = df.marital.isin(['อยู่ด้วยกันจดทะเบียนสมรส', 'อยู่ด้วยกันไม่ได้จดทะเบียนสมรส'])
R['sample'] = dict(
    n=int(len(df)), male=int((df.sex == 'ช').sum()), female=int((df.sex == 'ญ').sum()),
    stage_counts=df.stage.value_counts().reindex(order).astype(int).to_dict(),
    age_mean=float(df.age.mean()), age_sd=float(df.age.std()), age_min=float(df.age.min()), age_max=float(df.age.max()),
    poverty_pct=float(df.poverty.mean() * 100), poverty_n_known=int(df.poverty.notna().sum()),
    income_median=float(df.income_dmc.median()), income_lt5000_pct=float((df.income_dmc.dropna() < 5000).mean() * 100), income_n_known=int(df.income_dmc.notna().sum()),
    non_intact_pct=float((~intact).mean() * 100), not_with_parents_pct=float((df.living != 'พ่อแม่').mean() * 100),
    guardian_mother_pct=float((df.guardian == 'มารดา').mean() * 100), grandparent_guardian_pct=float(df.guardian.isin(['ยาย', 'ย่า', 'ตา']).mean() * 100),
    free_transport_pct=float((df.travel_mode == 'พาหนะไม่เสียค่าโดยสาร').mean() * 100), walk_pct=float((df.travel_mode == 'เดินเท้า').mean() * 100))
desc_rows = []
for col, label in [('att_pct', 'Attendance rate (%)'), ('days_absent', 'Days absent'), ('days_leave', 'Days on leave'), ('days_late', 'Days late'),
                   ('ded_n', 'Conduct infractions (count)'), ('ded_pts', 'Conduct demerit points'), ('income_dmc', 'Household income (THB/month)'),
                   ('dist_km_dmc', 'Home-school distance (km)'), ('travel_min', 'Travel time (min)')]:
    s = df[col].dropna(); desc_rows.append(dict(Variable=label, n=len(s), Mean=s.mean(), SD=s.std(), Median=s.median(), Min=s.min(), Max=s.max()))
for sc, label in [('PCT', 'Course score, K1-G6 (%)'), ('GPA', 'Grade point average, G7-G9 (0-4)')]:
    s = df[df.acad_scale == sc].acad_value; desc_rows.append(dict(Variable=label, n=len(s), Mean=s.mean(), SD=s.std(), Median=s.median(), Min=s.min(), Max=s.max()))
pd.DataFrame(desc_rows).round(2).to_csv(f'{outdir}/tables/T_descriptives.csv', index=False)
R['descriptives'] = pd.DataFrame(desc_rows).round(2).to_dict(orient='records')

# ---------- Dimension scores ----------
dist = df[D + ['H']].apply(lambda s: s.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int))
dist.loc['Mean'] = df[D + ['H']].mean().round(2)
dist.loc['SD'] = df[D + ['H']].std().round(2)
dist.to_csv(f'{outdir}/tables/T_dimension_distribution.csv')
R['dim_dist'] = {c: df[c].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist() for c in D + ['H']}
R['dim_mean'] = df[D + ['H']].mean().round(2).to_dict(); R['dim_sd'] = df[D + ['H']].std().round(2).to_dict()
rho = df[D].corr(method='spearman'); rho.round(2).to_csv(f'{outdir}/tables/T_dimension_spearman.csv')
R['dim_rho'] = rho.round(2).to_dict()
off = rho.values[~np.eye(5, dtype=bool)]
R['dim_rho_range'] = [float(off.min()), float(off.max())]; R['dim_rho_mean'] = float(off.mean())
mat = df[D].to_numpy().T
R['krippendorff_ordinal'] = float(krippendorff.alpha(reliability_data=mat, level_of_measurement='ordinal')) if krippendorff else None
# dimension-data anchoring (Spearman of each dimension with its own data field)
anchor = {'D1': ('acad_value', 'course score'), 'D2': ('att_pct', 'attendance rate'), 'D3': ('ded_pts', 'demerit points'),
          'D4': ('poverty', 'poverty flag'), 'D5': ('travel_min', 'travel time')}
R['anchor'] = {}
for k, (f, lab) in anchor.items():
    s = df[[k, f]].dropna(); r = spearmanr(s[k], s[f]); R['anchor'][k] = dict(field=f, label=lab, rho=float(r.correlation), p=float(r.pvalue), n=len(s))
# D1 within scale
for sc in ['PCT', 'GPA']:
    s = df[df.acad_scale == sc][['D1', 'acad_value']].dropna(); R['anchor'][f'D1_{sc}'] = dict(rho=float(spearmanr(s.D1, s.acad_value).correlation), n=len(s))
# D4 with income
s = df[['D4', 'income_dmc']].dropna(); R['anchor']['D4_income'] = dict(rho=float(spearmanr(s.D4, s.income_dmc).correlation), n=len(s))
s = df[['D5', 'dist_km_dmc']].dropna(); R['anchor']['D5_dist'] = dict(rho=float(spearmanr(s.D5, s.dist_km_dmc).correlation), n=len(s))
s = df[['D5', 'poverty']].dropna(); R['anchor']['D5_poverty'] = dict(rho=float(spearmanr(s.D5, s.poverty).correlation), n=len(s))
# cross-dimension homeroom check
R['d2_d3_qwk'] = float(cohen_kappa_score(df[['D2', 'D3']].dropna().D2.astype(int), df[['D2', 'D3']].dropna().D3.astype(int), weights='quadratic'))

# ---------- Agreement with H ----------
def metrics(pred, truth):
    s = pd.concat([pred, truth], axis=1).dropna().astype(int); x, y = s.iloc[:, 0].to_numpy(), s.iloc[:, 1].to_numpy()
    return dict(qwk=cohen_kappa_score(x, y, weights='quadratic'), lwk=cohen_kappa_score(x, y, weights='linear'),
                rho=spearmanr(x, y).correlation, exact=(x == y).mean(), within1=(np.abs(x - y) <= 1).mean(), n=len(s))
def boot(pred, truth, key):
    s = pd.concat([pred, truth], axis=1).dropna().astype(int).to_numpy(); vals = []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(s), len(s)); x, y = s[idx, 0], s[idx, 1]
        if key == 'qwk': vals.append(cohen_kappa_score(x, y, weights='quadratic'))
        elif key == 'rho': vals.append(spearmanr(x, y).correlation)
        elif key == 'exact': vals.append((x == y).mean())
        elif key == 'within1': vals.append((np.abs(x - y) <= 1).mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
R['agree'] = {}
for name, col in [('ref_final', 'ref_final'), ('ref_base', 'ref_base')] + [(k, k) for k in D]:
    if name == 'D1':   # two rows of an earlier draft of this table were dropped; their draws are kept so the intervals reproduce exactly
        for _ in range(2 * 4 * NBOOT): rng.integers(0, len(df), len(df))
    m = metrics(df[col], df.H); m = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in m.items()}
    m['qwk_ci'] = boot(df[col], df.H, 'qwk'); m['within1_ci'] = boot(df[col], df.H, 'within1'); m['exact_ci'] = boot(df[col], df.H, 'exact'); m['rho_ci'] = boot(df[col], df.H, 'rho')
    R['agree'][name] = m
R['agree']['Rmean_rho'] = float(spearmanr(df.Rmean, df.H).correlation)
cm = pd.crosstab(df.ref_final, df.H).reindex(index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5]).fillna(0).astype(int)
cm.to_csv(f'{outdir}/tables/T_confusion_final_vs_H.csv'); R['confusion'] = cm.values.tolist()
R['level_dist_ref'] = df.ref_final.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist()
R['level_dist_base'] = df.ref_base.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist()
R['H_dist'] = df.H.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist()
# direction of disagreement
diff = (df.ref_final - df.H); R['diff'] = dict(system_lower=int((diff < 0).sum()), equal=int((diff == 0).sum()), system_higher=int((diff > 0).sum()), mean_diff=float(diff.mean()))

# ---------- Screening ----------
def screen(score, truth, thr_pred, thr_truth=3):
    s = pd.concat([score, truth], axis=1).dropna(); p = (s.iloc[:, 0] >= thr_pred).astype(int); t = (s.iloc[:, 1] >= thr_truth).astype(int)
    tp = int(((p == 1) & (t == 1)).sum()); fp = int(((p == 1) & (t == 0)).sum()); fn = int(((p == 0) & (t == 1)).sum()); tn = int(((p == 0) & (t == 0)).sum())
    return dict(thr=float(thr_pred), flagged=int(p.sum()), flagged_pct=float(p.mean() * 100), se=tp / (tp + fn), sp=tn / (tn + fp), ppv=tp / (tp + fp) if tp + fp else np.nan, npv=tn / (tn + fn), tp=tp, fp=fp, fn=fn, tn=tn,
                balanced_acc=(tp / (tp + fn) + tn / (tn + fp)) / 2, youden=tp / (tp + fn) + tn / (tn + fp) - 1)
def boot_screen(score, truth, thr_pred, thr_truth=3):
    s = pd.concat([score, truth], axis=1).dropna().to_numpy(); se, sp = [], []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(s), len(s)); p = s[idx, 0] >= thr_pred; t = s[idx, 1] >= thr_truth
        se.append((p & t).sum() / t.sum()); sp.append((~p & ~t).sum() / (~t).sum())
    return [float(np.percentile(se, 2.5)), float(np.percentile(se, 97.5))], [float(np.percentile(sp, 2.5)), float(np.percentile(sp, 97.5))]
R['screen_level3'] = screen(df.ref_final, df.H, 3); R['screen_level3']['se_ci'], R['screen_level3']['sp_ci'] = boot_screen(df.ref_final, df.H, 3)
R['screen_base3'] = screen(df.ref_base, df.H, 3)
R['screen_level4_H4'] = screen(df.ref_final, df.H, 4, 4)
R['H_ge3_n'] = int((df.H >= 3).sum()); R['H_ge3_pct'] = float((df.H >= 3).mean() * 100)
s = df[['Rmean', 'H']].dropna()
def boot_auc(score, y):
    vals = []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() in (0, len(y)): continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
R['auc_H3'] = float(roc_auc_score((s.H >= 3).astype(int), s.Rmean)); R['auc_H3_ci'] = boot_auc(s.Rmean.to_numpy(), (s.H >= 3).astype(int).to_numpy())
R['auc_H4'] = float(roc_auc_score((s.H >= 4).astype(int), s.Rmean)); R['auc_H4_ci'] = boot_auc(s.Rmean.to_numpy(), (s.H >= 4).astype(int).to_numpy())
# per-dimension AUC for H>=3
R['auc_dim_H3'] = {k: float(roc_auc_score((df.loc[df[k].notna(), 'H'] >= 3).astype(int), df[k].dropna())) for k in D}
# calibration sweep on mean score
sweep = []
for thr in np.round(np.arange(1.4, 3.61, 0.2), 1):
    m = screen(df.Rmean, df.H, thr); m['overridden_added'] = 0; sweep.append(m)
# add override to sweep variant
sweep2 = []
for thr in np.round(np.arange(1.4, 3.61, 0.2), 1):
    pred = pd.Series(np.where(crit, 5.0, df.Rmean), index=df.index); sweep2.append(screen(pred, df.H, thr))
pd.DataFrame(sweep).round(3).to_csv(f'{outdir}/tables/T_threshold_sweep.csv', index=False)
pd.DataFrame(sweep2).round(3).to_csv(f'{outdir}/tables/T_threshold_sweep_with_override.csv', index=False)
R['sweep'] = pd.DataFrame(sweep).round(3).to_dict(orient='records'); R['sweep_override'] = pd.DataFrame(sweep2).round(3).to_dict(orient='records')
best = max(sweep2, key=lambda m: m['youden']); R['best_youden'] = best

# ---------- Override audit ----------
ov = df[df.override]
R['override_audit'] = dict(n=int(len(ov)), H_dist=ov.H.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist(),
                           H_ge3=int((ov.H >= 3).sum()), H_le2=int((ov.H <= 2).sum()), trig_D2=int((ov.D2 == 5).sum()), trig_D3=int((ov.D3 == 5).sum()), trig_both=int(((ov.D2 == 5) & (ov.D3 == 5)).sum()),
                           base_dist=ov.ref_base.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist(),
                           att_range=[float(ov[ov.D2 == 5].att_pct.min()), float(ov[ov.D2 == 5].att_pct.max())], ded_range=[float(ov[ov.D3 == 5].ded_pts.min()), float(ov[ov.D3 == 5].ded_pts.max())])
ov[['anon', 'grade', 'att_pct', 'ded_pts', 'acad_value'] + D + ['H', 'Rmean', 'ref_base', 'ref_final']].round(2).to_csv(f'{outdir}/tables/T_override_cases.csv', index=False)
# D2==5 overall attendance; D3==5 overall ded
R['D2eq5_att'] = [float(df[df.D2 == 5].att_pct.min()), float(df[df.D2 == 5].att_pct.max())]; R['D2eq5_n'] = int((df.D2 == 5).sum())
R['D3eq5_ded'] = [float(df[df.D3 == 5].ded_pts.min()), float(df[df.D3 == 5].ded_pts.max())]; R['D3eq5_n'] = int((df.D3 == 5).sum())
R['att_lt50_n'] = int((df.att_pct < 50).sum()); R['att_lt50_D2eq5'] = int(((df.att_pct < 50) & (df.D2 == 5)).sum())

# ---------- Stage ----------
st = pd.crosstab(df.stage, df.ref_final).reindex(columns=[1, 2, 3, 4, 5]).fillna(0).astype(int); st['n'] = st.sum(axis=1)
st['pct_ge3'] = (df.groupby('stage', observed=False).ref_final.apply(lambda s: (s >= 3).mean() * 100)).round(1)
st['mean_level'] = df.groupby('stage', observed=False).ref_final.mean().round(2)
st['H_pct_ge3'] = (df.groupby('stage', observed=False).H.apply(lambda s: (s >= 3).mean() * 100)).round(1)
st.to_csv(f'{outdir}/tables/T_stage.csv'); R['stage'] = st.reset_index().astype(str).to_dict(orient='records')
R['stage_kw_p'] = float(kruskal(*[g.ref_final for _, g in df.groupby('stage', observed=False)]).pvalue)
R['stage_kw_H_p'] = float(kruskal(*[g.H for _, g in df.groupby('stage', observed=False)]).pvalue)
# agreement by stage
R['agree_by_stage'] = {str(k): {kk: float(v) for kk, v in metrics(g.ref_final, g.H).items()} for k, g in df.groupby('stage', observed=False)}

# ---------- Factor associations ----------
feats = [('att_pct', 'Attendance rate'), ('days_absent', 'Days absent'), ('days_leave', 'Days on leave'), ('days_late', 'Days late'), ('ded_n', 'Conduct infractions'), ('ded_pts', 'Demerit points'),
         ('acad_value', 'Course performance (within scale)'), ('fail_n', 'Failed subjects'), ('income_dmc', 'Household income'), ('poverty', 'Poverty flag'), ('dist_km_dmc', 'Home-school distance'), ('travel_min', 'Travel time'), ('age', 'Age')]
rows = []
for f, lab in feats:
    if f == 'acad_value':
        # pooled within-scale rank: rank within scale group
        s = df[['acad_value', 'acad_scale', 'ref_final', 'H']].dropna(); s['rk'] = s.groupby('acad_scale').acad_value.rank(pct=True)
        a = spearmanr(s.rk, s.ref_final); b = spearmanr(s.rk, s.H); n = len(s)
    else:
        s = df[[f, 'ref_final', 'H']].dropna(); a = spearmanr(s[f], s.ref_final); b = spearmanr(s[f], s.H); n = len(s)
    rows.append(dict(Factor=lab, n=n, rho_system=a.correlation, p_system=a.pvalue, rho_H=b.correlation, p_H=b.pvalue))
fa = pd.DataFrame(rows); fa.round(3).to_csv(f'{outdir}/tables/T_factor_associations.csv', index=False); R['factors'] = fa.round(3).to_dict(orient='records')
def grp(mask, label):
    a, b = df.ref_final[mask], df.ref_final[~mask]; ha, hb = df.H[mask], df.H[~mask]
    return dict(label=label, n1=int(mask.sum()), n0=int((~mask).sum()), mean1=float(a.mean()), mean0=float(b.mean()), p=float(mannwhitneyu(a, b).pvalue), ge3_1=float((a >= 3).mean() * 100), ge3_0=float((b >= 3).mean() * 100),
                H_mean1=float(ha.mean()), H_mean0=float(hb.mean()), H_p=float(mannwhitneyu(ha, hb).pvalue))
R['groups'] = [grp(df.poverty == 1, 'Poverty flag (vs not)'), grp(~intact, 'Non-intact family (vs intact)'), grp(df.living != 'พ่อแม่', 'Not living with both parents'), grp(df.sex == 'ช', 'Male (vs female)')]
# note poverty mask includes NaN as not; handle
pm = df.poverty.notna(); a, b = df.ref_final[(df.poverty == 1)], df.ref_final[(df.poverty == 0)]
R['groups'][0].update(n1=int((df.poverty == 1).sum()), n0=int((df.poverty == 0).sum()), mean1=float(a.mean()), mean0=float(b.mean()), p=float(mannwhitneyu(a, b).pvalue), ge3_1=float((a >= 3).mean() * 100), ge3_0=float((b >= 3).mean() * 100),
                      H_mean1=float(df.H[df.poverty == 1].mean()), H_mean0=float(df.H[df.poverty == 0].mean()), H_p=float(mannwhitneyu(df.H[df.poverty == 1], df.H[df.poverty == 0]).pvalue))

# ---------- Missingness ----------
R['missing'] = {k: int(v) for k, v in df.isna().sum().items() if v > 0 and k not in ('H_PANEL', 'SCREENING')}


# ---------- Sensitivity analyses (added 19 Oct 2026) ----------
# (a) cluster bootstrap: resample the 12 classes with replacement, keeping students within class
classes = df.grade.unique(); yH3 = (df.H >= 3).astype(int)
cb_qwk, cb_auc = [], []
for _ in range(NBOOT):
    pick = rng.choice(classes, len(classes), replace=True)
    idx = np.concatenate([np.where(df.grade.to_numpy() == c)[0] for c in pick])
    cb_qwk.append(cohen_kappa_score(df.ref_final.to_numpy()[idx], df.H.to_numpy().astype(int)[idx], weights='quadratic'))
    yy = yH3.to_numpy()[idx]
    if 0 < yy.sum() < len(yy): cb_auc.append(roc_auc_score(yy, df.Rmean.to_numpy()[idx]))
R['cluster_boot'] = dict(n_classes=int(len(classes)), qwk_ci=[float(np.percentile(cb_qwk, 2.5)), float(np.percentile(cb_qwk, 97.5))],
                         auc_H3_ci=[float(np.percentile(cb_auc, 2.5)), float(np.percentile(cb_auc, 97.5))])
# (b) random dimension weights (Dirichlet(1,...,1)), same bands and override
Dm = df[D].to_numpy(); wk, ws = [], []
def level_w(w):
    m = np.nansum(Dm * w, axis=1) / np.nansum(np.where(np.isnan(Dm), 0, w), axis=1)
    b = 1 + (m[:, None] >= np.array([1.8, 2.6, 3.4, 4.2])[None, :] - 1e-9).sum(axis=1)
    return np.where(crit, 5, b)
for _ in range(NBOOT):
    w = rng.dirichlet(np.ones(5)); lv = level_w(w)
    wk.append(cohen_kappa_score(lv, df.H.astype(int), weights='quadratic')); ws.append(((lv >= 3) & (df.H >= 3)).sum() / (df.H >= 3).sum())
eq = R['agree']['ref_final']['qwk']
R['weight_sens'] = dict(qwk_p5=float(np.percentile(wk, 5)), qwk_p50=float(np.percentile(wk, 50)), qwk_p95=float(np.percentile(wk, 95)),
                        qwk_min=float(min(wk)), qwk_max=float(max(wk)), share_below_equal=float(np.mean(np.array(wk) < eq)),
                        se_p5=float(np.percentile(ws, 5)), se_p50=float(np.percentile(ws, 50)), se_p95=float(np.percentile(ws, 95)))
# (c) agreement within each class (one homeroom teacher each)
pc = []
for g, x in df.groupby('grade'):
    pc.append(dict(grade=g, n=int(len(x)), rho=float(spearmanr(x.ref_final, x.H).correlation), H_mean=float(x.H.mean()), sys_mean=float(x.ref_final.mean())))
pcd = pd.DataFrame(pc); pcd.round(3).to_csv(f'{outdir}/tables/T_per_class.csv', index=False)
R['per_class'] = dict(rho_min=float(pcd.rho.min()), rho_max=float(pcd.rho.max()), rho_median=float(pcd.rho.median()),
                      H_mean_min=float(pcd.H_mean.min()), H_mean_max=float(pcd.H_mean.max()), rows=pcd.round(3).to_dict(orient='records'))
# (d) discrimination of the mean of D1..D4 (without the director's dimension)
m4 = df[['D1', 'D2', 'D3', 'D4']].mean(axis=1)
R['auc_D1to4_H3'] = float(roc_auc_score(yH3, m4)); R['auc_D1to4_H3_ci'] = boot_auc(m4.to_numpy(), yH3.to_numpy())

# ---------- Revision analyses (added Nov 2026, first revision PAPER024920) ----------
from scipy.stats import fisher_exact, beta as _beta
def cp(k, n, a=0.05):   # Clopper-Pearson exact interval
    lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1)); hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]
Lf = df.ref_final.to_numpy(); Hh = df.H.to_numpy().astype(int); gr = df.grade.to_numpy()
# (e) leave-one-class-out agreement: how much any single homeroom rater drives kappa
loo = [float(cohen_kappa_score(Lf[gr != c], Hh[gr != c], weights='quadratic')) for c in classes]
# within-class association: both the mean score and H are ranked within their own class, so differences in level
# between classes and in severity between raters are removed from both sides
hr = df.groupby('grade').H.rank(pct=True); rr = df.groupby('grade').Rmean.rank(pct=True)
R['loco'] = dict(qwk_min=min(loo), qwk_max=max(loo), rho_within=float(spearmanr(rr, hr).correlation), rho_pooled=float(spearmanr(df.Rmean, df.H).correlation))
# (f) small cells at levels 4-5: exact test and exact intervals
def two(a, b):
    t = pd.crosstab(Lf >= a, Hh >= b).reindex(index=[False, True], columns=[False, True]).fillna(0).astype(int).to_numpy()
    tn, fp, fn, tp = int(t[0, 0]), int(t[1, 0]), int(t[0, 1]), int(t[1, 1])
    orr, p = fisher_exact([[tp, fp], [fn, tn]])
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, odds_ratio=float(orr), fisher_p=float(p), se=tp / (tp + fn), sp=tn / (tn + fp), se_cp=cp(tp, tp + fn), sp_cp=cp(tn, tn + fp), ppv=tp / (tp + fp), ppv_cp=cp(tp, tp + fp))
R['exact'] = dict(L3_H3=two(3, 3), L4_H4=two(4, 4), L5_H4=two(5, 4))
l5 = df[df.ref_final == 5]
R['level5'] = dict(n=int(len(l5)), by_override=int(l5.override.sum()), by_mean=int((~l5.override).sum()), H_ge3=int((l5.H >= 3).sum()), H_ge4=int((l5.H >= 4).sum()), H_le2=int((l5.H <= 2).sum()),
                   H_dist=l5.H.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist(), stage=l5.stage.value_counts().reindex(order).astype(int).tolist())
l4 = df[df.ref_final == 4]; R['level4_desc'] = dict(rmin=float(l4.Rmean.min()), rmax=float(l4.Rmean.max()), H=l4.H.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist(), stage=l4.stage.value_counts().reindex(order).astype(int).tolist())
R['level4_n'] = int((df.ref_final == 4).sum()); R['H4_n'] = int((df.H == 4).sum()); R['H5_n'] = int((df.H == 5).sum())
R['H_ge4_sys'] = df[df.H >= 4].ref_final.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int).tolist()
l5[['anon', 'stage'] + D + ['Rmean', 'ref_base', 'override', 'H', 'att_pct', 'ded_pts']].round(2).to_csv(f'{outdir}/tables/T_level5_cases.csv', index=False)
# (g) boundary chosen out of sample: leave one class out, pick the Youden-optimal boundary on the other eleven, apply it to the class left out
thr_grid = np.round(np.arange(1.4, 3.61, 0.2), 1); score_o = np.where(crit, 5.0, df.Rmean.to_numpy()); y3 = (Hh >= 3)
picked = []; pred_cv = np.zeros(len(df), bool)
for c in classes:
    tr = gr != c
    def you(t):
        p = score_o[tr] >= t; return (p & y3[tr]).sum() / y3[tr].sum() + (~p & ~y3[tr]).sum() / (~y3[tr]).sum() - 1
    t_best = float(max(thr_grid, key=you)); picked.append(t_best); pred_cv[~tr] = score_o[~tr] >= t_best
tp = int((pred_cv & y3).sum()); fn = int((~pred_cv & y3).sum()); tn = int((~pred_cv & ~y3).sum()); fp = int((pred_cv & ~y3).sum())
R['cv_boundary'] = dict(picked=sorted(set(picked)), picked_counts={str(k): int(picked.count(k)) for k in sorted(set(picked))}, se=tp / (tp + fn), sp=tn / (tn + fp), flagged_pct=float(pred_cv.mean() * 100), se_cp=cp(tp, tp + fn), sp_cp=cp(tn, tn + fp))
# per-class performance of the fixed 2.0 boundary
pcb = []
for c in classes:
    m = gr == c; p = score_o[m] >= 2.0; y = y3[m]
    pcb.append(dict(grade=c, n=int(m.sum()), pos=int(y.sum()), se=float((p & y).sum() / y.sum()) if y.sum() else np.nan, sp=float((~p & ~y).sum() / (~y).sum()) if (~y).sum() else np.nan))
pd.DataFrame(pcb).round(3).to_csv(f'{outdir}/tables/T_boundary2_per_class.csv', index=False)
b2 = pd.DataFrame(pcb); R['boundary2_per_class'] = dict(se_min=float(b2.se.min()), se_max=float(b2.se.max()), sp_min=float(b2.sp.min()), sp_max=float(b2.sp.max()))
# (h) where the downward difference sits: by stage and by dimension
bs = []
for k, g in df.groupby('stage', observed=False):
    d_ = g.ref_final - g.H; bs.append(dict(stage=str(k), n=int(len(g)), lower=int((d_ < 0).sum()), equal=int((d_ == 0).sum()), higher=int((d_ > 0).sum()), mean_diff=float(d_.mean()), mean_sys=float(g.ref_final.mean()), mean_H=float(g.H.mean())))
pd.DataFrame(bs).round(3).to_csv(f'{outdir}/tables/T_bias_by_stage.csv', index=False); R['bias_stage'] = bs
R['bias_dim'] = {k: dict(mean=float(df[k].mean()), mean_minus_H=float((df[k] - df.H).mean()), lower=int((df[k] < df.H).sum()), higher=int((df[k] > df.H).sum())) for k in D}
R['H_mean'] = float(df.H.mean()); R['Rmean_mean'] = float(df.Rmean.mean())
# direction of error once the boundary is moved to 2.0 (dichotomous): missed vs extra
p2 = score_o >= 2.0; R['dir_at_2'] = dict(missed=int((~p2 & y3).sum()), extra=int((p2 & ~y3).sum()))
p26 = Lf >= 3; R['dir_at_26'] = dict(missed=int((~p26 & y3).sum()), extra=int((p26 & ~y3).sum()))

json.dump(R, open(f'{outdir}/results.json', 'w'), indent=1, ensure_ascii=False, default=float)
df.to_csv(f'{outdir}/analytic_dataset.csv', index=False)
print(json.dumps({k: R[k] for k in ['krippendorff_ordinal', 'dim_rho_range', 'd2_d3_qwk', 'diff', 'auc_H3', 'auc_H3_ci', 'auc_H4', 'best_youden', 'override_audit', 'stage_kw_p', 'level_dist_ref', 'H_dist']}, indent=1, ensure_ascii=False, default=float))
for k in ['ref_final', 'ref_base', 'D1', 'D2', 'D3', 'D4', 'D5']: print(k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in R['agree'][k].items()})
print('screen', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R['screen_level3'].items()})
print(pd.DataFrame(R['sweep_override'])[['thr', 'flagged_pct', 'se', 'sp', 'ppv', 'youden']].to_string())
print(st.to_string()); print(fa.round(3).to_string()); print(pd.DataFrame(R['groups']).round(3).to_string())
print('anchor', {k: round(v['rho'], 2) for k, v in R['anchor'].items()})
