"""Check that the key numbers written in the manuscript match results.json.

Usage: python verify_numbers.py RESULTS.json
Prints OK / XX for each value and a final summary. Any XX means the manuscript text must be updated.
"""
import json, sys

R = json.load(open(sys.argv[1], encoding='utf-8'))
A = R['agree']; ok = []

def chk(label, claim, actual, tol=0.006):
    good = abs(claim - actual) <= tol; ok.append(good)
    print(('OK ' if good else 'XX '), f'{label:28s} manuscript={claim:<8} computed={actual:.4f}')

F, B, S = A['ref_final'], A['ref_base'], R['screen_level3']
chk('QWK system vs H', 0.47, F['qwk']); chk('QWK CI low', 0.38, F['qwk_ci'][0]); chk('QWK CI high', 0.56, F['qwk_ci'][1])
chk('Spearman rho', 0.52, F['rho']); chk('exact agreement', 0.44, F['exact']); chk('within one level', 0.85, F['within1'])
chk('QWK no override', 0.50, B['qwk'])
for k, v in {'D1': 0.33, 'D2': 0.20, 'D3': 0.29, 'D4': 0.29, 'D5': 0.69}.items(): chk(f'QWK {k} vs H', v, A[k]['qwk'])
chk('sensitivity (level>=3)', 0.36, S['se']); chk('specificity (level>=3)', 0.94, S['sp']); chk('PPV', 0.77, S['ppv'])
chk('flagged students', 52, S['flagged'], 0)
chk('AUC H>=3', 0.80, R['auc_H3']); chk('AUC CI low', 0.75, R['auc_H3_ci'][0]); chk('AUC CI high', 0.85, R['auc_H3_ci'][1])
chk('AUC H>=4', 0.85, R['auc_H4'])
Y = R['best_youden']; chk('Youden threshold', 2.0, Y['thr'], 0); chk('Youden sensitivity', 0.75, Y['se']); chk('Youden specificity', 0.73, Y['sp'])
for lv, n in zip([1, 2, 3, 4, 5], [139, 111, 31, 6, 15]): chk(f'students at level {lv}', n, R['level_dist_ref'][lv - 1], 0)
chk('system lower than H', 119, R['diff']['system_lower'], 0); chk('system higher than H', 51, R['diff']['system_higher'], 0)
O = R['override_audit']; chk('override cases', 11, O['n'], 0); chk('override H>=3', 6, O['H_ge3'], 0)
chk('D2 vs attendance rho', -0.89, R['anchor']['D2']['rho'])
chk('D3 vs demerit rho', 0.89, R['anchor']['D3']['rho'])
C = R['cluster_boot']; chk('cluster QWK CI low', 0.34, C['qwk_ci'][0]); chk('cluster QWK CI high', 0.56, C['qwk_ci'][1])
chk('cluster AUC CI low', 0.76, C['auc_H3_ci'][0]); chk('cluster AUC CI high', 0.84, C['auc_H3_ci'][1])
Wt = R['weight_sens']; chk('weights QWK p5', 0.32, Wt['qwk_p5']); chk('weights QWK p50', 0.41, Wt['qwk_p50']); chk('weights QWK p95', 0.60, Wt['qwk_p95'])
chk('weights share below equal', 0.69, Wt['share_below_equal']); chk('weights Se p5', 0.26, Wt['se_p5']); chk('weights Se p50', 0.32, Wt['se_p50']); chk('weights Se p95', 0.59, Wt['se_p95'])
Pc = R['per_class']; chk('per-class rho min', 0.15, Pc['rho_min']); chk('per-class rho max', 0.84, Pc['rho_max']); chk('per-class rho median', 0.49, Pc['rho_median'])
chk('per-class H mean min', 1.6, Pc['H_mean_min'], 0.05); chk('per-class H mean max', 2.7, Pc['H_mean_max'], 0.05)
chk('AUC D1-D4', 0.70, R['auc_D1to4_H3']); chk('AUC D1-D4 CI low', 0.64, R['auc_D1to4_H3_ci'][0]); chk('AUC D1-D4 CI high', 0.76, R['auc_D1to4_H3_ci'][1])
SW = {m['thr']: m for m in R['sweep_override']}; chk('sweep Sp at 2.8', 0.96, SW[2.8]['sp']); chk('sweep Sp at 3.2', 0.97, SW[3.2]['sp'])
# ---- first revision (Nov 2026)
Lc = R['loco']; chk('leave-one-class-out QWK min', 0.43, Lc['qwk_min']); chk('leave-one-class-out QWK max', 0.49, Lc['qwk_max']); chk('within-class rho', 0.56, Lc['rho_within']); chk('pooled rho mean score', 0.58, Lc['rho_pooled'])
E4 = R['exact']['L4_H4']; chk('L>=4 vs H>=4 TP', 7, E4['tp'], 0); chk('H>=4 n', 30, E4['tp'] + E4['fn'], 0); chk('Se level 4', 0.23, E4['se']); chk('Se L4 exact low', 0.10, E4['se_cp'][0]); chk('Se L4 exact high', 0.42, E4['se_cp'][1])
chk('Sp level 4', 0.95, E4['sp']); chk('Sp L4 exact low', 0.92, E4['sp_cp'][0]); chk('Sp L4 exact high', 0.97, E4['sp_cp'][1]); chk('PPV level 4', 0.33, E4['ppv']); chk('PPV L4 low', 0.15, E4['ppv_cp'][0]); chk('PPV L4 high', 0.57, E4['ppv_cp'][1])
chk('Fisher OR level 4', 5.6, E4['odds_ratio'], 0.05); chk('Fisher p level 4', 0.002, E4['fisher_p'], 0.0005); chk('Fisher OR level 3', 8.4, R['exact']['L3_H3']['odds_ratio'], 0.05)
L5 = R['level5']; chk('level 5 n', 15, L5['n'], 0); chk('level 5 by override', 11, L5['by_override'], 0); chk('level 5 by mean', 4, L5['by_mean'], 0); chk('level 5 H>=3', 10, L5['H_ge3'], 0); chk('level 5 H>=4', 6, L5['H_ge4'], 0)
chk('system level 4 n', 6, R['level4_n'], 0); chk('H=4 n', 28, R['H4_n'], 0); chk('H=5 n', 2, R['H5_n'], 0)
G = R['H_ge4_sys']; chk('H>=4 at system 1-2', 13, G[0] + G[1], 0); chk('H>=4 at system 3', 10, G[2], 0); chk('H>=4 at system 4-5', 7, G[3] + G[4], 0)
CV = R['cv_boundary']; chk('CV boundary picked', 2.0, CV['picked'][0], 0); chk('CV picked in all folds', 12, CV['picked_counts']['2.0'], 0); chk('CV Se', 0.75, CV['se']); chk('CV Sp', 0.73, CV['sp'])
chk('CV Se low', 0.66, CV['se_cp'][0]); chk('CV Se high', 0.83, CV['se_cp'][1]); chk('CV Sp low', 0.66, CV['sp_cp'][0]); chk('CV Sp high', 0.79, CV['sp_cp'][1])
bs = R['bias_stage']; chk('bias kindergarten', -0.16, bs[0]['mean_diff']); chk('bias G1-3', -0.52, bs[1]['mean_diff']); chk('bias G4-6', -0.20, bs[2]['mean_diff']); chk('bias G7-9', -0.22, bs[3]['mean_diff'])
chk('G1-3 lower', 39, bs[1]['lower'], 0); chk('G1-3 higher', 6, bs[1]['higher'], 0); chk('G1-3 n', 85, bs[1]['n'], 0); chk('G7-9 lower', 32, bs[3]['lower'], 0); chk('G7-9 higher', 20, bs[3]['higher'], 0)
for k, v in {'D1': -0.26, 'D2': -0.20, 'D3': -0.27, 'D4': -0.49, 'D5': 0.26}.items(): chk(f'{k} minus H', v, R['bias_dim'][k]['mean_minus_H'])
chk('missed at 2.6', 71, R['dir_at_26']['missed'], 0); chk('extra at 2.6', 12, R['dir_at_26']['extra'], 0); chk('missed at 2.0', 28, R['dir_at_2']['missed'], 0); chk('extra at 2.0', 52, R['dir_at_2']['extra'], 0)
chk('level 4 mean min', 3.4, R['level4_desc']['rmin']); chk('level 4 mean max', 3.6, R['level4_desc']['rmax']); chk('level 4 H=4', 1, R['level4_desc']['H'][3], 0); chk('level 4 H=3', 4, R['level4_desc']['H'][2], 0); chk('level 4 H=2', 1, R['level4_desc']['H'][1], 0); chk('level 4 lower secondary', 4, R['level4_desc']['stage'][3], 0)
print('\nALL OK' if all(ok) else '\nMISMATCH FOUND: update the manuscript text', f'({sum(ok)}/{len(ok)})')
sys.exit(0 if all(ok) else 1)
