"""
RSOM vs Standard SOM -- Sleep-EDF temporal experiment.

Dataset : PhysioNet Sleep-EDF (Cassette subset, Fpz-Cz + Pz-Oz EEG).
          10 whole-night recordings (SC4001-SC4091, one per subject).
Features: per-epoch band powers (delta, theta, alpha, sigma, beta) from
          both EEG channels -> 10 features.
Labels  : W=0, N1=1, N2=2, N3/N4=3, REM=4  (unknown '?' discarded).

Training:
  Standard SOM -- AMBER.Map (random epoch order).
  RSOM         -- AMBER.TemporalMap.train_sequential():
                  temporal order within each night, context reset
                  between nights (Voegtlin 2002).

Output: Table (QE, TE, NMI, TC) + two figures.
"""

import sys, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import welch
from scipy.stats import mode
from sklearn.metrics import normalized_mutual_info_score
import mne

_repo = pathlib.Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import AMBER
from AMBER.distances import SIGNAL_DISTANCE_MATRIX

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# -- Config -------------------------------------------------------------------
DATA_DIR      = pathlib.Path('/Users/albertonogales/mne_data/physionet-sleep-data')
FS            = 100
EPOCH_SEC     = 30
MAP_SIZE      = 5
N_PASSES      = 10
INITIAL_LR    = 0.5
DISTANCE      = 'euclidean'
SEED          = 42

STAGE_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}
STAGE_NAMES  = ['Wake', 'N1', 'N2', 'N3', 'REM']
STAGE_COLORS = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
BANDS = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,12),
         'sigma': (12,16), 'beta': (16,30)}

# -- Feature extraction -------------------------------------------------------
def band_powers(epoch, fs):
    f, psd = welch(epoch, fs=fs, nperseg=min(4*fs, len(epoch)))
    return np.array([np.log1p(np.trapz(psd[(f>=lo)&(f<hi)], f[(f>=lo)&(f<hi)]))
                     for lo, hi in BANDS.values()])

def load_recording(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    raw.resample(FS, verbose=False)
    raw.set_annotations(mne.read_annotations(str(hyp_path)), verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAP, verbose=False)
    n_samp = EPOCH_SEC * FS
    data   = raw.get_data(picks=['EEG Fpz-Cz', 'EEG Pz-Oz'])
    X_list, y_list = [], []
    for ev in events:
        onset, label = ev[0], ev[2]
        if onset + n_samp > data.shape[1]: continue
        epoch = data[:, onset:onset+n_samp]
        X_list.append(np.concatenate([band_powers(epoch[0], FS),
                                       band_powers(epoch[1], FS)]))
        y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)

# -- Load data ----------------------------------------------------------------
print("Loading Sleep-EDF recordings ...")
psg_files = sorted(DATA_DIR.glob('*PSG.edf'))
hyp_files = sorted(DATA_DIR.glob('*Hypnogram.edf'))

recordings_X, recordings_y = [], []
for psg, hyp in zip(psg_files, hyp_files):
    X, y = load_recording(psg, hyp)
    recordings_X.append(X); recordings_y.append(y)
    print(f"  {psg.name}: {len(y)} epochs  stages={np.bincount(y, minlength=5)}")

X_all = np.concatenate(recordings_X)
y_all = np.concatenate(recordings_y)
print(f"\nTotal: {len(y_all)} epochs, {len(recordings_X)} recordings")

mu  = X_all.mean(axis=0)
std = X_all.std(axis=0) + 1e-8
recordings_norm = [(r - mu) / std for r in recordings_X]
X_norm = np.concatenate(recordings_norm)

# -- Evaluation ---------------------------------------------------------------
def evaluate(tm, recordings_norm, recordings_y, alpha, beta):
    """Classify with temporal context reset between recordings; compute metrics."""
    dist_fn = SIGNAL_DISTANCE_MATRIX[DISTANCE]
    k       = tm.map_size
    n_feat  = tm.weights.shape[2]
    bmus = []
    for rec in recordings_norm:
        tm.reset_context()
        for x in rec:
            d_sig = dist_fn(tm.weights, x)
            d_ctx = dist_fn(tm.weights, tm._context if tm._context is not None
                            else np.zeros(n_feat))
            d_eff = (1 - beta) * d_sig + beta * d_ctx
            bf    = int(np.argmin(d_eff))
            bmus.append(bf)
            bmu_rc = (bf // k, bf % k)
            if tm._context is None:
                tm._context = tm.weights[bmu_rc].copy()
            else:
                tm._context = (alpha * tm._context
                               + (1 - alpha) * tm.weights[bmu_rc])
    bmus  = np.array(bmus)
    y_all = np.concatenate(recordings_y)
    X_all = np.concatenate(recordings_norm)

    bmu_w = tm.weights.reshape(-1, n_feat)[bmus]
    qe    = float(np.mean(np.linalg.norm(X_all - bmu_w, axis=1)))

    te_count = 0
    for x in X_all:
        d = dist_fn(tm.weights, x).ravel()
        o = np.argsort(d)
        r1, c1 = divmod(int(o[0]), k)
        r2, c2 = divmod(int(o[1]), k)
        if max(abs(r1-r2), abs(c1-c2)) > 1:
            te_count += 1
    te = te_count / len(X_all)

    neuron_labels = {}
    for b, lbl in zip(bmus, y_all):
        neuron_labels.setdefault(b, []).append(lbl)
    assigned = {b: int(mode(v, keepdims=False).mode) for b,v in neuron_labels.items()}
    pred   = np.array([assigned.get(b,-1) for b in bmus])
    purity = float(np.mean(pred == y_all))
    nmi    = float(normalized_mutual_info_score(y_all, bmus))

    cls_obj = AMBER.Classification(tm, np.concatenate(recordings_norm))
    tc = AMBER.TemporalAnalysis(cls_obj).temporal_coherence

    return dict(QE=qe, TE=te, Purity=purity, NMI=nmi, TC=tc, bmus=bmus)

# -- Train Standard SOM -------------------------------------------------------
PERIOD_SOM = len(X_norm) * N_PASSES
print(f"\nTraining Standard SOM (period={PERIOD_SOM}) ...")
som = AMBER.Map(
    data=X_norm, size=MAP_SIZE, period=PERIOD_SOM,
    distance=DISTANCE, normalization='none',
    weights='PCA', use_decay=True, random_seed=SEED,
)
# Wrap in a TemporalMap shell for unified evaluate() — no context effect (beta=0)
tm_som = AMBER.TemporalMap(
    size=MAP_SIZE, period=2, context_weight=0.0, context_influence=0.0,
    normalization='none', confirm=True, random_seed=SEED,
)
tm_som.weights = som.weights
tm_som.input_data_dimension = som.input_data_dimension
tm_som._Map__trained = True  # type: ignore[attr-defined]

res_som = evaluate(tm_som, recordings_norm, recordings_y, alpha=0.0, beta=0.0)
print(f"  QE={res_som['QE']:.3f}  TE={res_som['TE']:.3f}  "
      f"NMI={res_som['NMI']:.3f}  TC={res_som['TC']:.3f}")

# -- RSOM ablation via train_sequential ---------------------------------------
ALPHAS = [0.2, 0.5, 0.8]
BETAS  = [0.1, 0.2, 0.3, 0.5]

print(f"\nTraining RSOM ablation ({len(ALPHAS)}x{len(BETAS)} grid) ...")
rows = []
for alpha in ALPHAS:
    for beta in BETAS:
        tm = AMBER.TemporalMap(
            size=MAP_SIZE,
            period=2,                   # overridden by train_sequential
            initial_lr=INITIAL_LR,
            distance=DISTANCE,
            normalization='none',
            weights='PCA',
            use_decay=True,
            context_weight=alpha,
            context_influence=beta,
            confirm=True,
            random_seed=SEED,
        )
        tm.train_sequential(recordings_norm, n_passes=N_PASSES)
        res = evaluate(tm, recordings_norm, recordings_y, alpha=alpha, beta=beta)
        rows.append(dict(alpha=alpha, beta=beta,
                         QE=res['QE'], TE=res['TE'],
                         NMI=res['NMI'], TC=res['TC'],
                         _tm=tm, _bmus=res['bmus']))
        print(f"  alpha={alpha} beta={beta}  NMI={res['NMI']:.3f}  TC={res['TC']:.3f}")

ablation = pd.DataFrame(rows)

# -- Results table ------------------------------------------------------------
best_nmi = ablation.loc[ablation['NMI'].idxmax()]
valid    = ablation[ablation['NMI'] > 0.1]
best_tc  = valid.loc[valid['TC'].idxmax()]

print("\n--- Results ---")
tbl = pd.DataFrame([
    {'Model': 'Standard SOM',    'alpha': '--', 'beta': '--',
     'QE': res_som['QE'], 'TE': res_som['TE'],
     'NMI': res_som['NMI'], 'TC': res_som['TC']},
    {'Model': 'RSOM (best NMI)', 'alpha': best_nmi.alpha, 'beta': best_nmi.beta,
     'QE': best_nmi.QE, 'TE': best_nmi.TE,
     'NMI': best_nmi.NMI, 'TC': best_nmi.TC},
    {'Model': 'RSOM (best TC)',  'alpha': best_tc.alpha,  'beta': best_tc.beta,
     'QE': best_tc.QE,  'TE': best_tc.TE,
     'NMI': best_tc.NMI, 'TC': best_tc.TC},
]).set_index('Model')
print(tbl.to_string(float_format=lambda x: f'{x:.3f}' if isinstance(x, float) else x))
print(f"\nDelta NMI (best) = {best_nmi.NMI - res_som['NMI']:+.3f}")
print(f"Delta TC  (best) = {best_tc.TC  - res_som['TC']:+.3f}")

# -- Figure 1: Trajectory comparison (first 80 epochs, subject SC4001) --------
k       = MAP_SIZE
n_show  = 80
rec_y   = recordings_y[0]

som_bmus  = res_som['bmus'][:n_show]
som_rows  = som_bmus // k;  som_cols = som_bmus % k

# Best-TC RSOM trajectory for the same recording
tc_tm   = best_tc['_tm']
dist_fn = SIGNAL_DISTANCE_MATRIX[DISTANCE]
rsom_bmus_r = []
tc_tm.reset_context()
for x in recordings_norm[0]:
    d_sig  = dist_fn(tc_tm.weights, x)
    d_ctx  = dist_fn(tc_tm.weights,
                     tc_tm._context if tc_tm._context is not None
                     else np.zeros(tc_tm.weights.shape[2]))
    d_eff  = (1 - best_tc.beta) * d_sig + best_tc.beta * d_ctx
    bf     = int(np.argmin(d_eff))
    rsom_bmus_r.append(bf)
    bmu_rc = (bf // k, bf % k)
    if tc_tm._context is None:
        tc_tm._context = tc_tm.weights[bmu_rc].copy()
    else:
        tc_tm._context = (best_tc.alpha * tc_tm._context
                          + (1 - best_tc.alpha) * tc_tm.weights[bmu_rc])
rsom_bmus_r = np.array(rsom_bmus_r[:n_show])
rsom_rows = rsom_bmus_r // k;  rsom_cols = rsom_bmus_r % k

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, brows, bcols, title in [
    (axes[0], som_rows,  som_cols,  'Standard SOM'),
    (axes[1], rsom_rows, rsom_cols,
     f'RSOM (alpha={best_tc.alpha}, beta={best_tc.beta})'),
]:
    ax.set_xlim(-0.5, k-0.5); ax.set_ylim(-0.5, k-0.5)
    ax.set_xticks(range(k));   ax.set_yticks(range(k))
    ax.grid(True, alpha=0.3);  ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Column', fontsize=8); ax.set_ylabel('Row', fontsize=8)
    ax.invert_yaxis()
    for t in range(len(brows)-1):
        ax.annotate('', xy=(bcols[t+1], brows[t+1]),
                    xytext=(bcols[t], brows[t]),
                    arrowprops=dict(arrowstyle='->',
                                    color=STAGE_COLORS[rec_y[t]],
                                    lw=1.2, alpha=0.65))
    for t in range(len(brows)):
        ax.scatter(bcols[t], brows[t], c=STAGE_COLORS[rec_y[t]],
                   s=45, zorder=5, edgecolors='white', linewidths=0.5)

patches = [mpatches.Patch(color=STAGE_COLORS[i], label=STAGE_NAMES[i])
           for i in range(5)]
fig.legend(handles=patches, loc='lower center', ncol=5,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5,-0.02))
fig.suptitle('BMU trajectory -- Sleep-EDF (first 80 epochs, subject SC4001)',
             fontsize=10, fontweight='bold')
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig1_path = pathlib.Path(__file__).parent / 'rsom_trajectory_comparison.png'
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"\nFigure 1 -> {fig1_path.name}")

# -- Figure 2: Ablation heatmap -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, metric, label, cmap in [
    (axes[0], 'NMI', 'NMI (up)',                 'YlOrRd'),
    (axes[1], 'TC',  'Trajectory Coherence (up)', 'YlGn'),
]:
    pivot = ablation.pivot(index='alpha', columns='beta', values=metric)
    im    = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(BETAS)));  ax.set_xticklabels(BETAS, fontsize=8)
    ax.set_yticks(range(len(ALPHAS))); ax.set_yticklabels(ALPHAS, fontsize=8)
    ax.set_xlabel('context_influence (beta)', fontsize=9)
    ax.set_ylabel('context_weight (alpha)',   fontsize=9)
    baseline = res_som[metric]
    ax.set_title(f'{label}  (SOM baseline = {baseline:.2f})',
                 fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax)
    for i in range(len(ALPHAS)):
        for j in range(len(BETAS)):
            v = pivot.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if v > 0.65 else 'black')

fig.suptitle('RSOM ablation -- Sleep-EDF: alpha x beta grid',
             fontsize=10, fontweight='bold')
plt.tight_layout()
fig2_path = pathlib.Path(__file__).parent / 'rsom_ablation_heatmap.png'
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Figure 2 -> {fig2_path.name}")

plt.show()
print("\nDone.")
