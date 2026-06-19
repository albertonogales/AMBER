import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
sys.path.insert(0, '.')
import AMBER
from aeon.datasets import load_classification

X_raw, y = load_classification('CBF')
X = X_raw[:, 0, :]
classes = np.unique(y)
cnames  = {c: n for c, n in zip(classes, ['Cylinder','Bell','Funnel'])}
colors  = {c: col for c, col in zip(classes, ['steelblue','firebrick','forestgreen'])}

print("Training…")
som = AMBER.Map(data=X, size=4, period=5000, distance='dtw',
                normalization='zscore', initial_lr=0.5,
                initial_neighbourhood=3, use_decay=True,
                dtw_band=20, random_seed=42)
print("Done training.")

cls = AMBER.Classification(som, X)
cm  = cls.classification_map
cm['labels'] = y  # assign class labels after classification

majority, counts = {}, {}
for (r, c), grp in cm.groupby(['x','y']):
    maj = grp['labels'].value_counts().idxmax()
    majority[(r,c)] = maj
    counts[(r,c)]   = len(grp)

k = 4
fig, axes = plt.subplots(k, k, figsize=(10, 8), sharex=True, sharey=True,
                          gridspec_kw=dict(hspace=0.08, wspace=0.08))
T, ts = X.shape[1], np.arange(X.shape[1])

for r in range(k):
    for c in range(k):
        ax = axes[r][c]
        w  = som.weights[r, c]
        maj = majority.get((r, c), None)
        n   = counts.get((r, c), 0)
        if maj is not None:
            bg = list(to_rgba(colors[maj])); bg[3] = 0.18
            ax.set_facecolor(bg); col = colors[maj]
        else:
            ax.set_facecolor('#f0f0f0'); col = 'gray'
        ax.plot(ts, w, color=col, lw=1.6, zorder=3)
        ax.text(0.03, 0.93, f'[{r},{c}]', transform=ax.transAxes, fontsize=7, va='top', color='#333')
        if n > 0:
            ax.text(0.97, 0.93, f'n={n}', transform=ax.transAxes, fontsize=7, va='top', ha='right', color='#333')
        ax.tick_params(labelsize=6); ax.set_xlim(0, T-1)

fig.text(0.5, 0.02, 'Time step', ha='center', fontsize=10)
fig.text(0.02, 0.5, 'Amplitude (z-scored)', va='center', rotation='vertical', fontsize=10)
patches = [mpatches.Patch(color=colors[c], label=cnames[c]) for c in classes]
fig.legend(handles=patches, loc='upper center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.suptitle('CBF SOM weight map — DTW distance, 4×4 grid', y=1.04, fontsize=12)
OUT = 'examples/cbf_weight_map.png'
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Saved {OUT}")
