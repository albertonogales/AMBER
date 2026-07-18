"""
Static plotting helpers for trained SOMs.

All methods are @staticmethod — no Visualization instance needed.
Every matplotlib method accepts filename=None (show) or a path (save)
and returns (fig, ax) so callers can further customise the figure.
Plotly methods return the Figure object.
"""

import logging

import matplotlib.patches as mpatches
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

logger = logging.getLogger(__name__)

try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):  # type: ignore[misc]
        pass  # no-op outside Jupyter


def _majority_counts(cm_df):
    """Return (majority, counts) dicts keyed by (row, col) BMU.

    Requires cm_df columns: 'x', 'y', 'labels'.
    """
    majority, counts = {}, {}
    for (r, c), grp in cm_df.groupby(['x', 'y']):
        vc = grp['labels'].value_counts()
        majority[(r, c)] = int(vc.idxmax())
        counts[(r, c)]   = len(grp)
    return majority, counts


def _class_palette(labels, class_names, palette):
    """Resolve classes, names, colours and return (classes, col_map, name_map).

    :param labels: 1-D array of integer class codes
    :param class_names: list of names in class-code order, or None → 'Class N'
    :param palette: list of colour strings in class-code order, or None → tab10
    """
    classes = sorted(np.unique(np.asarray(labels, dtype=int)).tolist())
    n_cls   = len(classes)

    if class_names is None:
        class_names = [f'Class {c}' for c in classes]
    if palette is None:
        cmap_ = plt.colormaps.get_cmap('tab10')
        palette = [cmap_(i / max(n_cls, 1)) for i in range(n_cls)]

    col_map  = {c: palette[i]     for i, c in enumerate(classes)}
    name_map = {c: class_names[i] for i, c in enumerate(classes)}
    return classes, col_map, name_map


def _save_or_show(fig, filename):
    """Save figure to filename (if given) or display it; then return fig."""
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return fig


class Visualization:
    """Static plotting helpers for trained SOMs; no instance needed."""

    # ------------------------------------------------------------------
    # Activation / distribution plots
    # ------------------------------------------------------------------

    @staticmethod
    def heat_map(classification, colorscale='Reds', cmax=0):
        """Annotated heatmap of BMU activation counts.

        :param cmax: colour scale max; 0 = auto-scale to data maximum.
        :return: Plotly Figure
        """
        map_rot = np.transpose(classification.activations_map)
        cmax = np.max(classification.activations_map) if cmax == 0 else cmax
        fig = ff.create_annotated_heatmap(
            map_rot, showscale=True, colorscale=colorscale, zmin=0, zmax=cmax
        )
        pio.show(fig)
        return fig

    @staticmethod
    def elevation_map(classification, title='Elevation Map'):
        """3-D surface plot of BMU activation counts.

        :return: Plotly Figure
        """
        map_rot = np.rot90(classification.activations_map, k=-1)
        fig = go.Figure(
            data=[go.Surface(z=np.fliplr(map_rot), colorscale='Reds')],
            layout=go.Layout(
                title=title, autosize=False, width=1000, height=1000,
                margin=dict(l=65, r=50, b=65, t=90),
            ),
        )
        pio.show(fig)
        return fig

    @staticmethod
    def bar_chart(data):
        """Interactive Plotly bar chart of a 1-D data array.

        :return: Plotly Figure
        """
        data_np = np.asarray(data).reshape(-1)
        fig = go.Figure(
            data=[go.Bar(y=data_np)],
            layout={
                'xaxis': {'title': 'Times Activated'},
                'yaxis': {'title': 'Number of Neurons'},
                'barmode': 'relative',
            },
        )
        pio.show(fig)
        return fig

    @staticmethod
    def neurons_per_num_activations_map(classification):
        """Bar chart of neurons activated exactly k times, for k = 0…max.

        :return: Plotly Figure
        """
        n_max = np.max(classification.activations_map) + 1
        counts = np.array([
            np.count_nonzero(classification.activations_map == i)
            for i in range(n_max)
        ])
        return Visualization.bar_chart(counts)

    # ------------------------------------------------------------------
    # Codebook / weight plots
    # ------------------------------------------------------------------

    @staticmethod
    def codebook_vector(map, index=0, header='none'):
        """Annotated heatmap of feature index across all neurons.

        :param index: feature index to display
        :param header: plot title; 'none' suppresses title
        :return: Plotly Figure
        """
        map_rot = np.transpose(np.around(map.weights[:, :, index], decimals=2))
        fig = ff.create_annotated_heatmap(map_rot, showscale=True)
        if header != 'none':
            fig.layout.title = header
        for ann in fig.layout.annotations:
            ann.font.size = 7
        pio.show(fig)
        return fig

    @staticmethod
    def codebook_vectors(map, headers=np.array([])):
        """Plot codebook_vector for every input dimension of the map."""
        if headers.size < 1:
            headers = np.arange(map.input_data_dimension)
        for i in range(map.input_data_dimension):
            Visualization.codebook_vector(map, i, str(headers[i]))

    @staticmethod
    def characteristics_graph(map, row, column, labels=np.array([]),
                               size_x=10, size_y=10, angle=45, filename=None):
        """Line plot of the weight vector for neuron [row, column].

        :param labels: optional feature-name labels for the x-axis
        :param angle: x-tick label rotation in degrees
        :param filename: save path; None → display
        :return: (fig, ax)
        """
        data = np.array(map.weights[row][column])
        fig, ax = plt.subplots(figsize=(size_x, size_y))
        ax.plot(data, label=f'[{row},{column}]')
        if labels.size > 0:
            ax.set_xticks(np.arange(map.input_data_dimension))
            ax.set_xticklabels(labels, rotation=angle)
        ax.legend()
        return _save_or_show(fig, filename), ax

    @staticmethod
    def characteristics_bargraph(map, row, column, labels=np.array([]),
                                  size_x=10, size_y=10, angle=45, filename=None):
        """Colour-coded bar chart of the weight vector for neuron [row, column].

        :param labels: optional feature-name labels for the x-axis
        :param angle: x-tick label rotation in degrees
        :param filename: save path; None → display
        :return: (fig, ax)
        """
        data = np.array(map.weights[row][column])
        rainbow = plt.colormaps.get_cmap('tab20').resampled(data.shape[0])
        fig, ax = plt.subplots(figsize=(size_x, size_y))
        ax.bar(np.arange(data.shape[0]), data,
               label=f'[{row},{column}]',
               color=rainbow(np.linspace(0, 1, data.shape[0])))
        if labels.size > 0:
            ax.set_xticks(np.arange(map.input_data_dimension))
            ax.set_xticklabels(labels, rotation=angle)
        ax.legend()
        return _save_or_show(fig, filename), ax

    @staticmethod
    def umatrix(classification, colorscale='binary', filename=None):
        """Display the U-matrix; dark regions indicate cluster boundaries.

        :return: (fig, ax)
        """
        fig, ax = plt.subplots()
        im = ax.imshow(np.rot90(classification.umatriz), cmap=colorscale)
        plt.colorbar(im, ax=ax)
        return _save_or_show(fig, filename), ax

    @staticmethod
    def full_map_weights(map, labels=np.array([]), size_x=25, size_y=30, filename=None):
        """Grid of weight-vector line plots.

        :param filename: save path; None → display
        :return: (fig, axes)
        """
        k  = map.map_size
        xs = np.arange(map.input_data_dimension)
        fig, axes = plt.subplots(k, k, sharex='col', sharey='row',
                                 figsize=(size_x, size_y))
        for i in range(k):
            for j in range(k):
                axes[i, j].plot(map.weights[i, j], label=f'[{i},{j}]')
                if labels.size > 0:
                    axes[i, j].set_xticks(xs)
                    axes[i, j].set_xticklabels(labels, rotation=45,
                                                ha='right', fontsize=5)
        return _save_or_show(fig, filename), axes

    # ------------------------------------------------------------------
    # Class-labelled plots
    # ------------------------------------------------------------------

    @staticmethod
    def umatrix_labeled(classification, labels, class_names=None, palette=None,
                        figsize=(8, 9), title=None, filename=None):
        """U-matrix with majority-class circle markers per neuron.

        :param classification: Classification with tagged=True
        :param labels: 1-D array of integer class codes, one per sample
        :param class_names: human-readable names in class-code order; defaults to 'Class N'
        :param palette: matplotlib colour strings in class-code order; defaults to tab10
        :param filename: save path; None → display
        :return: (fig, ax)
        """
        cm_df = classification.classification_map
        k     = classification.activations_map.shape[0]
        umat  = classification.umatriz

        if 'labels' not in cm_df.columns or cm_df['labels'].nunique() <= 1:
            logger.warning(
                "umatrix_labeled: classification_map has no meaningful 'labels' column. "
                "Run Classification with tagged=True to get per-sample class labels."
            )

        classes, col_map, name_map = _class_palette(labels, class_names, palette)
        majority, counts = _majority_counts(cm_df)

        fig = plt.figure(figsize=figsize)

        handles = [mpatches.Patch(color=col_map[c], label=name_map[c]) for c in classes]
        fig.legend(handles=handles, loc='upper center', ncol=len(classes),
                   fontsize=10, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.45, 0.95))

        if title is None:
            title = ('U-matrix\nColoured markers: majority class per neuron  '
                     '·  Number: sample count')
        fig.suptitle(title, fontsize=11, y=1.00)

        ax  = fig.add_axes([0.08, 0.05, 0.78, 0.80])
        cax = fig.add_axes([0.89, 0.05, 0.03, 0.80])
        im  = ax.imshow(umat, cmap='binary', origin='upper',
                        extent=[-0.5, umat.shape[1] - 0.5,
                                 umat.shape[0] - 0.5, -0.5])
        plt.colorbar(im, cax=cax, label='Mean distance to neighbours')

        for (r, c), maj in majority.items():
            uy, ux = r * 2, c * 2
            ax.plot(ux, uy, 'o', color=col_map[maj], markersize=22,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=5)
            ax.text(ux, uy, str(counts[(r, c)]),
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold', zorder=6)

        ax.set_xlabel('Neuron column', fontsize=10)
        ax.set_ylabel('Neuron row',    fontsize=10)
        ax.set_xticks(range(0, umat.shape[1], 2))
        ax.set_xticklabels(range(k))
        ax.set_yticks(range(0, umat.shape[0], 2))
        ax.set_yticklabels(range(k))

        return _save_or_show(fig, filename), ax

    @staticmethod
    def hit_map(classification, labels, class_names=None, palette=None,
                figsize=(10, 9), title=None, filename=None):
        """Hit map: cell size encodes sample count, colour encodes majority class.

        Cell side scales with sqrt(n/n_max); dead neurons shown as empty grey cells.

        :param classification: Classification with tagged=True
        :param labels: 1-D array of integer class codes, one per sample
        :param class_names: human-readable names; defaults to 'Class N'
        :param palette: matplotlib colours in class-code order; defaults to tab10
        :param filename: save path; None → display
        :return: (fig, ax)
        """
        cm_df = classification.classification_map
        k     = classification.activations_map.shape[0]

        classes, col_map, name_map = _class_palette(labels, class_names, palette)
        majority, counts = _majority_counts(cm_df)
        max_n = max(counts.values()) if counts else 1

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(k - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f8f8')

        for r in range(k):
            for c in range(k):
                if (r, c) in majority:
                    col = col_map[majority[(r, c)]]
                    n   = counts[(r, c)]
                    bg  = (*to_rgba(col)[:3], 0.18)
                    hw  = 0.18 + 0.72 * np.sqrt(n / max_n)
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               color=bg, zorder=1))
                    ax.add_patch(plt.Rectangle((c - hw / 2, r - hw / 2),
                                               hw, hw, color=col, zorder=2))
                    ax.text(c, r, str(n), ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold', zorder=3)
                else:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               color='#e8e8e8', zorder=1))

        ax.set_xticks(range(k))
        ax.set_xlabel('Neuron column', fontsize=11)
        ax.set_yticks(range(k))
        ax.set_ylabel('Neuron row',    fontsize=11)
        ax.grid(True, linewidth=0.5, color='white', zorder=0)

        handles = [mpatches.Patch(color=col_map[c], label=name_map[c]) for c in classes]
        fig.legend(handles=handles, loc='upper center', ncol=min(len(classes), 5),
                   fontsize=9, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.00))

        if title is None:
            title = 'Hit Map\nColour: majority class  ·  Cell size: sample count'
        fig.suptitle(title, fontsize=11, y=1.06)

        return _save_or_show(fig, filename), ax

    @staticmethod
    def weight_map_grid(som, classification, labels, class_names=None,
                        palette=None, figsize=None, title=None, filename=None):
        """Grid of weight-vector profiles (one subplot per neuron) coloured by majority class.

        Dead neurons show a neutral grey background.

        :param som: trained Map instance
        :param classification: Classification with tagged=True
        :param labels: 1-D array of integer class codes, one per sample
        :param class_names: human-readable names; defaults to 'Class N'
        :param palette: matplotlib colours in class-code order; defaults to tab10
        :param figsize: defaults to (2.4*map_size, 1.9*map_size)
        :param filename: save path; None → display
        :return: (fig, axes)
        """
        cm_df = classification.classification_map
        k     = som.map_size
        if figsize is None:
            figsize = (2.4 * k, 1.9 * k)

        classes, col_map, name_map = _class_palette(labels, class_names, palette)
        majority, counts = _majority_counts(cm_df)

        T  = som.input_data_dimension
        ts = np.arange(T)

        fig, axes = plt.subplots(k, k, figsize=figsize, sharex=True, sharey=True,
                                 gridspec_kw=dict(hspace=0.06, wspace=0.06))
        for r in range(k):
            for c in range(k):
                ax = axes[r][c]
                w  = som.weights[r, c]
                if (r, c) in majority:
                    col = col_map[majority[(r, c)]]
                    bg  = list(to_rgba(col))
                    bg[3] = 0.18
                    ax.set_facecolor(bg)
                    n = counts[(r, c)]
                else:
                    col = '#888888'
                    ax.set_facecolor('#f0f0f0')
                    n = 0
                ax.plot(ts, w, color=col, lw=1.5, zorder=3)
                ax.text(0.03, 0.95, f'[{r},{c}]', transform=ax.transAxes,
                        fontsize=6.5, va='top', color='#333333')
                if n > 0:
                    ax.text(0.97, 0.95, f'n={n}', transform=ax.transAxes,
                            fontsize=6.5, va='top', ha='right', color='#333333')
                ax.tick_params(labelsize=5.5)
                ax.set_xlim(0, T - 1)

        fig.text(0.5,  0.01, 'Time step',            ha='center', fontsize=11)
        fig.text(0.01, 0.5,  'Amplitude (z-scored)', va='center',
                 rotation='vertical', fontsize=11)

        handles = [mpatches.Patch(color=col_map[c], label=name_map[c]) for c in classes]
        fig.legend(handles=handles, loc='upper center', ncol=len(classes),
                   fontsize=9, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.00))

        if title is None:
            title = f'SOM weight map — {k}×{k} grid'
        fig.suptitle(title, y=1.05, fontsize=12)

        return _save_or_show(fig, filename), axes

    # ------------------------------------------------------------------
    # Temporal visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def trajectory(classification, temporal_analysis,
                   background='activations',
                   cmap_path='plasma', cmap_bg='YlOrRd',
                   figsize=(7, 7), title='BMU Trajectory',
                   random_seed=None, filename=None):
        """Time-ordered BMU path on the SOM grid; early = dark, late = bright.

        :param background: 'activations' or 'umatrix'
        :param random_seed: seed for jitter that offsets overlapping points
        :param filename: save path; None → display
        :return: (fig, ax)
        """
        traj = temporal_analysis.trajectory
        if len(traj) < 2:
            logger.warning("Trajectory too short to plot.")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)

        bg = (classification.umatriz if background == 'umatrix'
              else classification.activations_map.astype(float))
        ax.imshow(bg, cmap=cmap_bg, origin='upper',
                  extent=[-0.5, bg.shape[1] - 0.5, bg.shape[0] - 0.5, -0.5])

        k = classification.activations_map.shape[0]
        for g in range(k + 1):
            ax.axhline(g - 0.5, color='white', linewidth=0.4, alpha=0.4)
            ax.axvline(g - 0.5, color='white', linewidth=0.4, alpha=0.4)

        rows = np.array([p[0] for p in traj], dtype=float)
        cols = np.array([p[1] for p in traj], dtype=float)
        n    = len(traj)
        cmap = plt.colormaps.get_cmap(cmap_path)
        colors = cmap(np.linspace(0.15, 1.0, n - 1))

        _rng    = np.random.default_rng(random_seed)
        offsets = _rng.uniform(-0.12, 0.12, size=(n, 2))

        for t in range(n - 1):
            ax.annotate(
                '', xy=(cols[t+1] + offsets[t+1, 1], rows[t+1] + offsets[t+1, 0]),
                xytext=(cols[t] + offsets[t, 1], rows[t] + offsets[t, 0]),
                arrowprops=dict(arrowstyle='->', color=colors[t],
                                lw=1.5, mutation_scale=10),
            )

        ax.plot(cols[0],  rows[0],  'o', color='lime',  markersize=10,
                zorder=5, label='start')
        ax.plot(cols[-1], rows[-1], 's', color='white', markersize=10,
                zorder=5, label='end')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n - 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Time step')

        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()

        return _save_or_show(fig, filename), ax

    @staticmethod
    def transition_matrix_plot(temporal_analysis,
                               normalised=True,
                               cmap='Blues',
                               figsize=(8, 7),
                               title='Transition Matrix',
                               filename=None):
        """Heatmap of neuron-to-neuron transition frequencies or probabilities.

        :param normalised: True = row-normalised probabilities; False = raw counts
        :param filename: save path; None → display
        :return: (fig, ax)
        """
        T = (temporal_analysis.transition_matrix_norm
             if normalised else temporal_analysis.transition_matrix)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(T, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Probability' if normalised else 'Count')

        k = temporal_analysis.map_size
        for g in range(0, k ** 2 + 1, k):
            ax.axhline(g - 0.5, color='grey', linewidth=0.5, alpha=0.6)
            ax.axvline(g - 0.5, color='grey', linewidth=0.5, alpha=0.6)

        ax.set_xlabel('To neuron index')
        ax.set_ylabel('From neuron index')
        ax.set_title(title)
        plt.tight_layout()

        return _save_or_show(fig, filename), ax

    @staticmethod
    def dwell_time_map(temporal_analysis, classification,
                       cmap='Blues', figsize=(6, 5),
                       title='Mean Dwell Time per Neuron',
                       filename=None):
        """Heatmap of mean consecutive dwell time per BMU.

        :param filename: save path; None → display
        :return: (fig, ax)
        """
        k = temporal_analysis.map_size
        dwell_grid = np.zeros((k, k))
        for (row, col), mean_dwell in temporal_analysis.dwell_times().items():
            dwell_grid[row, col] = mean_dwell

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(dwell_grid, cmap=cmap, origin='upper')
        plt.colorbar(im, ax=ax, label='Mean dwell time (steps)')
        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        for r in range(k):
            for c in range(k):
                if dwell_grid[r, c] > 0:
                    ax.text(c, r, f'{dwell_grid[r, c]:.1f}',
                            ha='center', va='center', fontsize=7, color='black')
        plt.tight_layout()

        return _save_or_show(fig, filename), ax
