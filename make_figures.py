import json, sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, roc_auc_score

out = sys.argv[1]; FIG = sys.argv[2] if len(sys.argv) > 2 else f'{out}/figures'; os.makedirs(FIG, exist_ok=True)
R = json.load(open(f'{out}/results.json')); df = pd.read_csv(f'{out}/analytic_dataset.csv')
D = ['D1', 'D2', 'D3', 'D4', 'D5']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': False})
SEQ = ['#dbe7f6', '#a8c4ea', '#6f9ddc', '#2a78d6', '#163f73']     # single-hue blue ramp, light -> dark (levels 1..5)
S1, S2, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
LVL = ['1 Very low', '2 Low', '3 Moderate', '4 High', '5 Critical']
W = 6.7  # inches, text width

def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.close(fig)

# ---- Fig 1: architecture ----
FH = 3.12
fig, ax = plt.subplots(figsize=(W, FH)); fig.subplots_adjust(0, 0, 1, 1); ax.set_xlim(0, W); ax.set_ylim(0, FH); ax.axis('off')
FS = 8.4; LH = 0.158
BOXES = []   # (patch, [text artists]) for the overflow check below
def box(x, y, w, h, title, lines, fc='#f3f6fb', ec=S1):
    p = ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=ec, lw=1.1, zorder=2))
    ts = [ax.text(x + w / 2, y + h - 0.14, title, ha='center', va='center', fontsize=8.4, fontweight='bold', color=INK)]
    for i, t in enumerate(lines): ts.append(ax.text(x + 0.07, y + h - 0.36 - i * LH, t, ha='left', va='center', fontsize=FS, color=MUTED))
    BOXES.append(((x, y, w, h), ts))
def check_inside(fig):
    # every text line must sit inside its box with a small margin; stop the build otherwise
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    for (x, y, w, h), ts in BOXES:
        for t in ts:
            (x0, y0), (x1, y1) = inv.transform(t.get_window_extent(rend).get_points())
            if x0 < x + 0.03 or x1 > x + w - 0.03 or y0 < y + 0.02 or y1 > y + h - 0.02:
                raise SystemExit(f'Fig1 text overflows its box: "{t.get_text()}"')
def arrow(x1, y1, x2, y2, label=None, color=MUTED, dy=0.09):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, lw=1.0, color=color, shrinkA=0, shrinkB=0, zorder=1))
    if label: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha='center', fontsize=8.4, color=MUTED)
bw, gap, y0, bh = 1.535, 0.14, 1.32, 1.78
xs = [0.03 + i * (bw + gap) for i in range(4)]
box(xs[0], y0, bw, bh, 'A. Platform data', ['Attendance, absence,', '  leave, lateness', 'Conduct demerit points', 'Course score or GPA', 'Household income,', '  poverty flag,', '  family structure', 'Home-school distance,', '  travel time and mode'])
box(xs[1], y0, bw, bh, 'B. Dimension ratings', ['Web form shows A;', 'raters score 1 to 5:', 'D1 Learning:', '  academic head', 'D2 Attendance and', 'D3 Behaviour: level', '  classroom teacher', 'D4 Family: counsellor', 'D5 Travel: director'])
box(xs[2], y0, bw, bh, 'C. Aggregation engine', ['Mean of D1 to D5', '  (equal weights)', 'Five equal-width bands', '  with edges at 1.8,', '  2.6, 3.4 and 4.2', 'Max-risk override:', '  D2 = 5 or D3 = 5', '  forces level 5'])
box(xs[3], y0, bw, bh, 'D. Output', ['Priority level 1 to 5', '  for each student', 'Dimension profile', 'Class and school', '  dashboard', 'Routing to support', '  team actions'])
PAD = 0.02   # FancyBboxPatch pad: the drawn border lies PAD outside (x, y, w, h)
IN = 0.012   # arrows run slightly under the box borders (boxes are drawn on top), so no gap can show at any zoom
for i in range(3): arrow(xs[i] + bw + PAD - IN, y0 + bh / 2, xs[i + 1] - PAD + IN, y0 + bh / 2)
box(xs[0], 0.08, bw * 4 + gap * 3, 0.92, 'E. Independent criterion and validation (this study)', ['Holistic 1 to 5 rating of every student by the class homeroom teacher (12 teachers),', '  given independently of B and C', 'Agreement with C: weighted kappa, within-one-level agreement, sensitivity,', '  specificity and area under the ROC curve'], fc='#fff4ee', ec=S2)
E_TOP = 0.08 + 0.92 + PAD
arrow(xs[2] + bw / 2, y0 - PAD + IN, xs[2] + bw / 2, E_TOP - IN, color=S2)
arrow(xs[0] + bw / 2, y0 - PAD + IN, xs[0] + bw / 2, E_TOP - IN, color=S2)
check_inside(fig)
save(fig, 'Fig1_architecture')

# ---- Fig 2: distributions ----
fig, ax = plt.subplots(figsize=(W, 2.6))
labels = ['D1 Learning', 'D2 Attendance', 'D3 Behaviour', 'D4 Family / economy', 'D5 Travel / support', 'H Homeroom (criterion)']
cols = D + ['H']
left = np.zeros(len(cols))
for lv in range(5):
    vals = np.array([R['dim_dist'][c][lv] for c in cols]) / 302 * 100
    ax.barh(range(len(cols)), vals, left=left, color=SEQ[lv], edgecolor='white', linewidth=1.5, height=0.72, label=LVL[lv])
    for i, v in enumerate(vals):
        if v >= 7: ax.text(left[i] + v / 2, i, f'{v:.0f}', ha='center', va='center', fontsize=8.4, color='white' if lv >= 3 else INK)
    left += vals
ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Share of students (%)')
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=8.4, handlelength=1.2)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
save(fig, 'Fig2_score_distributions')

# ---- Fig 3: Spearman heatmap ----
cols = D + ['H']; rho = df[cols].corr(method='spearman')
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(rho.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('blue', ['#ffffff', '#163f73']), vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        v = rho.values[i, j]; ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8.4, color='white' if v > 0.55 else INK)
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(cols); ax.set_yticklabels(cols)
ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('Spearman rank correlation', fontsize=8.4); cb.ax.tick_params(labelsize=8.4)
save(fig, 'Fig4_dimension_correlation')

# ---- Fig 4: confusion ----
cm = np.array(R['confusion']); fig, ax = plt.subplots(figsize=(4.4, 3.8))
im = ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('blue', ['#ffffff', '#163f73']), vmin=0, vmax=cm.max())
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8.4, color='white' if cm[i, j] > cm.max() * 0.55 else INK, fontweight='bold' if i == j else 'normal')
for k in range(5): ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec=S2, lw=1.4))
ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(['1', '2', '3', '4', '5']); ax.set_yticklabels(['1', '2', '3', '4', '5'])
ax.set_xlabel('Homeroom teacher holistic rating (H)'); ax.set_ylabel('System priority level')
ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.tick_params(length=0)
for i in range(5): ax.text(5.0, i, f'n={cm[i].sum()}', va='center', fontsize=8.4, color=MUTED)
for j in range(5): ax.text(j, 4.85, f'n={cm[:, j].sum()}', ha='center', fontsize=8.4, color=MUTED)
ax.set_xlim(-0.5, 5.6); ax.set_ylim(5.1, -0.5)
save(fig, 'Fig6_confusion')

# ---- Fig 5: calibration + ROC ----
sw = pd.DataFrame(R['sweep_override'])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={'width_ratios': [1.35, 1]})
a1.plot(sw.thr, sw.se, color=S1, lw=2, marker='o', ms=4, label='Sensitivity (H >= 3)')
a1.plot(sw.thr, sw.sp, color=S2, lw=2, marker='s', ms=4, label='Specificity')
a1.plot(sw.thr, sw.ppv, color=MUTED, lw=1.5, ls='--', marker='^', ms=4, label='Positive predictive value')
a1.axvline(2.6, color='#9a9a9a', lw=1, ls=':'); a1.text(2.66, 0.44, 'current band\nboundary 2.6', fontsize=8.4, color=MUTED)
a1.axvline(2.0, color='#9a9a9a', lw=1, ls=':'); a1.text(1.68, 0.03, 'alternative\n2.0', fontsize=8.4, color=MUTED, ha='center')
a1.set_xlabel('Mean dimension score threshold for "action required"'); a1.set_ylabel('Proportion'); a1.set_ylim(0, 1.02); a1.set_xlim(1.35, 3.65)
a1.legend(frameon=False, fontsize=8.4, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
s = df[['Rmean', 'H']].dropna(); y = (s.H >= 3).astype(int)
fpr, tpr, _ = roc_curve(y, s.Rmean); a2.plot(fpr, tpr, color=S1, lw=2)
a2.text(0.98, 0.04, f'AUC = {R["auc_H3"]:.2f}\n[{R["auc_H3_ci"][0]:.2f}, {R["auc_H3_ci"][1]:.2f}]', transform=a2.transAxes, ha='right', va='bottom', fontsize=8.4, color=INK)
a2.plot([0, 1], [0, 1], color='#bbbbbb', lw=1, ls='--')
pt = R['screen_level3']; a2.plot(1 - pt['sp'], pt['se'], 'o', color=S2, ms=6); a2.annotate('level >= 3\n(current)', (1 - pt['sp'], pt['se']), xytext=(8, -4), textcoords='offset points', fontsize=8.4, color=MUTED)
b = R['best_youden']; a2.plot(1 - b['sp'], b['se'], 's', color=S2, ms=6); a2.annotate('mean >= 2.0', (1 - b['sp'], b['se']), xytext=(8, -4), textcoords='offset points', fontsize=8.4, color=MUTED)
a2.set_xlabel('1 - specificity'); a2.set_ylabel('Sensitivity')
a1.text(-0.12, 1.04, '(a)', transform=a1.transAxes, fontweight='bold'); a2.text(-0.18, 1.04, '(b)', transform=a2.transAxes, fontweight='bold')
fig.tight_layout(w_pad=2.5); save(fig, 'Fig7_calibration_roc')

# ---- Fig 6: stage ----
st = pd.DataFrame(R['stage']); x = np.arange(4); w = 0.36
fig, ax = plt.subplots(figsize=(W, 2.7))
b1 = ax.bar(x - w / 2, st.pct_ge3.astype(float), w, color=S1, label='System level >= 3')
b2 = ax.bar(x + w / 2, st.H_pct_ge3.astype(float), w, color=S2, label='Homeroom rating >= 3')
for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f'{b.get_height():.0f}%', ha='center', fontsize=8.4, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f'{s.replace(" (", chr(10)+"(")}\n(n = {n})' for s, n in zip(st.stage, st.n)]); ax.set_ylabel('Students rated moderate or higher (%)'); ax.set_ylim(0, 60)
ax.legend(frameon=False, fontsize=8.4, loc='upper left'); ax.yaxis.grid(True, color='#e6e6e6', lw=0.6); ax.set_axisbelow(True)
save(fig, 'Fig5_stage')

# ---- Fig 7: factor associations ----
fa = pd.DataFrame(R['factors']).sort_values('rho_system', key=lambda s: s.abs())
fig, ax = plt.subplots(figsize=(5.75, 3.4)); yy = np.arange(len(fa))
ax.hlines(yy, fa.rho_H, fa.rho_system, color='#d0d0d0', lw=1.2, zorder=1)
ax.scatter(fa.rho_system, yy, color=S1, s=36, zorder=3, label='with system level'); ax.scatter(fa.rho_H, yy, color=S2, s=36, marker='s', zorder=3, label='with homeroom rating')
ax.axvline(0, color='#9a9a9a', lw=0.8); ax.set_yticks(yy); ax.set_yticklabels(fa.Factor); ax.set_xlabel('Spearman rank correlation'); ax.set_xlim(-0.65, 0.65)
ax.legend(frameon=False, fontsize=8.4, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2); ax.xaxis.grid(True, color='#e6e6e6', lw=0.6); ax.set_axisbelow(True); ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
save(fig, 'Fig8_factor_associations')

# ---- Fig 8: anchoring ----
fig, axs = plt.subplots(1, 3, figsize=(W, 2.6))
def bx(ax, dcol, ycol, ylabel, title, data=None):
    data = df if data is None else data
    groups = [data.loc[data[dcol] == k, ycol].dropna() for k in range(1, 6)]
    bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=True, flierprops=dict(marker='o', ms=2.5, mfc=MUTED, mec='none', alpha=0.6), medianprops=dict(color=INK, lw=1.3))
    for p, c in zip(bp['boxes'], SEQ): p.set(facecolor=c, edgecolor=MUTED, lw=0.8)
    for el in ['whiskers', 'caps']: plt.setp(bp[el], color=MUTED, lw=0.8)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([str(k) for k in range(1, 6)], fontsize=8.4); ax.set_xlabel(f'{dcol} score  (n = ' + ', '.join(str(len(g)) for g in groups) + ')', fontsize=8.4); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=8.8, loc='left')
bx(axs[0], 'D2', 'att_pct', 'Attendance rate (%)', '(a) D2 vs attendance')
bx(axs[1], 'D3', 'ded_pts', 'Conduct demerit points', '(b) D3 vs demerits')
s = df.copy(); s['rk'] = s.groupby('acad_scale').acad_value.rank(pct=True) * 100
bx(axs[2], 'D1', 'rk', 'Course performance percentile\n(within scale)', '(c) D1 vs performance', data=s)
fig.tight_layout(w_pad=1.5); save(fig, 'Fig3_anchoring')
print('figures written:', sorted(os.listdir(FIG)))
