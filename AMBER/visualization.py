import matplotlib.patches as mpatches
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):  # type: ignore[misc]
        pass  # no-op outside Jupyter
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from plotly.offline import iplot


class Visualization:
    """Static collection of plotting helpers for trained SOMs.

    All methods are ``@staticmethod`` — no instance is needed::

        Visualization.heat_map(classification)
        Visualization.umatrix(classification)
    """

    # HEAT MAP
    @staticmethod
    def heat_map(classification, filename='heat_map', colorscale='Reds', cmax=0):
        """Annotated heatmap of BMU activation counts across the SOM grid.

        :param classification: a completed :class:`~AMBER.Classification` instance
        :param filename: Plotly filename / title (default ``'heat_map'``)
        :param colorscale: Plotly colorscale name (default ``'Reds'``)
        :param cmax: maximum value for the colour scale; 0 = auto-scale to data maximum
        """
        # MODIFIED. Activation map rotated 90º so it matches with the Heat Map visualisation
        map_rot = np.transpose(classification.activations_map)
        cmax = np.max(classification.activations_map) if cmax == 0 else cmax
        fig = ff.create_annotated_heatmap(map_rot, showscale=True, colorscale=colorscale, zmin=0, zmax=cmax)

        iplot(fig, filename=filename)

    # ELEVATION MAP
    @staticmethod
    def elevation_map(classification, filename='elevation_map'):
        """3-D surface plot of BMU activation counts (elevation = activation frequency).

        :param classification: a completed :class:`~AMBER.Classification` instance
        :param filename: Plotly filename / title (default ``'elevation_map'``)
        """
        # MODIFIED. Activation map rotated 90º so it matches with the Elevation Map visualisation
        map_rot = np.rot90(classification.activations_map, k=-1)

        data = [
            go.Surface(
                z=np.fliplr(map_rot),
                colorscale='Reds'
            )
        ]
        layout = go.Layout(
            title=filename,
            autosize=False,
            width=1000,
            height=1000,
            margin=dict(
                l=65,
                r=50,
                b=65,
                t=90
            )
        )
        fig = go.Figure(data=data, layout=layout)
        iplot(fig, filename=filename)

    # CHARACTERISTICS GRAPH
    @staticmethod
    def characteristics_graph(map, row, column, labels=np.array([]), size_x=10, size_y=10, angle=45):
        """Line plot of the weight vector for a single neuron.

        :param map: a trained :class:`~AMBER.Map` instance
        :param row: row index of the neuron
        :param column: column index of the neuron
        :param labels: feature-name labels for the x-axis (optional)
        :param size_x: figure width in inches (default 10)
        :param size_y: figure height in inches (default 10)
        :param angle: x-tick label rotation in degrees (default 45)
        """
        map.characteristics_data_labels = labels

        data = np.array(map.weights[row][column])
        plt.figure(figsize=(size_x, size_y))
        if map.characteristics_data_labels.size > 0:
            plt.xticks(np.arange(map.input_data_dimension), map.characteristics_data_labels, rotation=angle)
        display(plt.plot(data, label='[' + str(row) + ',' + str(column) + ']'))

    # CHARACTERISTICS BAR GRAPH
    @staticmethod
    def characteristics_bargraph(map, row, column, labels=np.array([]), size_x=10, size_y=10, angle=45):
        """Colour-coded bar chart of the weight vector for a single neuron.

        Each bar corresponds to one input feature; bars are coloured with the
        ``tab20`` colormap for easy visual discrimination.

        :param map: a trained :class:`~AMBER.Map` instance
        :param row: row index of the neuron
        :param column: column index of the neuron
        :param labels: feature-name labels for the x-axis (optional)
        :param size_x: figure width in inches (default 10)
        :param size_y: figure height in inches (default 10)
        :param angle: x-tick label rotation in degrees (default 45)
        """
        map.characteristics_data_labels = labels

        data = np.array(map.weights[row][column])
        plt.figure(figsize=(size_x, size_y))
        if map.characteristics_data_labels.size > 0:
            plt.xticks(np.arange(map.input_data_dimension), map.characteristics_data_labels, rotation=angle)
        rainbow = plt.colormaps.get_cmap('tab20').resampled(data.shape[0])
        display(plt.bar(np.arange(data.shape[0]), data, label='[' + str(row) + ',' + str(column) + ']',
                        color=rainbow(np.linspace(0, 1, data.shape[0]))))

    # BAR CHART
    @staticmethod
    def bar_chart(data, filename='bar_chart'):
        """Interactive bar chart of an arbitrary 1-D data array (Plotly).

        :param data: array-like of values to plot
        :param filename: Plotly filename / title (default ``'bar_chart'``)
        """
        data_np = np.asarray(data).reshape(-1)
        data_bar = [go.Bar(y=data_np)]
        layout = {
            'xaxis': {'title': 'Times Activated'},
            'yaxis': {'title': 'Number of Neurons'},
            'barmode': 'relative'
        }
        iplot({'data': data_bar, 'layout': layout}, filename=filename)

    # NEURONS PER NUM ACTIVATIONS
    @staticmethod
    def neurons_per_num_activations_map(classification, filename='neurons_per_num_activations_map', save=False):
        """Bar chart of the number of neurons activated exactly k times, for k = 0 … max.

        Useful for diagnosing dead neurons (activated 0 times) and over-used neurons.

        :param classification: a completed :class:`~AMBER.Classification` instance
        :param filename: Plotly filename / title
        :param save: unused (reserved for future file-export support)
        """
        num_max_activations = np.max(classification.activations_map) + 1
        neurons_per_num_activations = np.zeros(num_max_activations)

        for i in range(0, num_max_activations):
            neurons_per_num_activations[i] = np.count_nonzero(classification.activations_map == i)

        Visualization.bar_chart(data=neurons_per_num_activations, filename=filename)

    @staticmethod
    def codebook_vector(map, index=0, header='none', filename='codebook_vector'):
        """Annotated heatmap of a single codebook (weight) dimension across all neurons.

        Displays the value of feature ``index`` for every neuron in the grid,
        useful for understanding how a particular input dimension is distributed
        across the map.

        :param map: a trained :class:`~AMBER.Map` instance
        :param index: feature index to display (default 0)
        :param header: plot title; ``'none'`` suppresses the title
        :param filename: Plotly filename (default ``'codebook_vector'``)
        """
        map_rot = np.transpose(np.around(map.weights[:, :, index], decimals=2))

        fig = ff.create_annotated_heatmap(map_rot, showscale=True)
        if header != 'none':
            fig.layout.title = header

        # Make text size smaller
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 7

        iplot(fig, filename=filename)

    @staticmethod
    def codebook_vectors(map, headers=np.array([])):
        """Plot :meth:`codebook_vector` for every input dimension of the map.

        :param map: a trained :class:`~AMBER.Map` instance
        :param headers: feature names used as plot titles; defaults to ``0, 1, …, D-1``
        """
        if headers.size < 1:
            headers = np.arange(map.input_data_dimension)
        for i in range(0, map.input_data_dimension):
            Visualization.codebook_vector(map, i, str(headers[i]))

    @staticmethod
    def umatrix(classification, colorscale='binary'):
        """Display the U-matrix (unified distance matrix) of the trained map.

        Each cell in the U-matrix encodes the mean distance between a neuron and
        its neighbours; dark regions indicate cluster boundaries.

        :param classification: a completed :class:`~AMBER.Classification` instance
        :param colorscale: matplotlib colormap name (default ``'binary'``)
        """
        plt.imshow(np.rot90(classification.umatriz), cmap=colorscale)
        plt.colorbar()

    @staticmethod
    def umatrix_labeled(classification, labels, class_names=None, palette=None,
                        figsize=(8, 9), title=None, filename=None):
        """U-matrix overlaid with majority-class markers for each neuron.

        Combines topology (greyscale U-matrix background) with semantics
        (coloured circle markers showing the majority class and sample count
        of each active neuron).  The legend is placed between the title and
        the axes so it never overlaps neuron markers.

        :param classification: a completed :class:`~AMBER.Classification` instance.
            Must have been created with ``tagged=True`` so that
            ``classification_map['labels']`` contains integer class codes.
        :param labels: 1-D array-like of integer class codes, one per sample,
            used to build the legend (values must match those stored in
            ``classification_map['labels']``).
        :param class_names: list of human-readable class names in class-code
            order.  If ``None``, names default to ``'Class 0'``, ``'Class 1'``, …
        :param palette: list of matplotlib colour strings in class-code order.
            If ``None``, the ``tab10`` colormap is used.
        :param figsize: figure size in inches (default ``(8, 9)``).
        :param title: figure suptitle.  If ``None``, a generic title is used.
        :param filename: if given, save the figure to this path (PNG/PDF/…).
            If ``None``, the figure is displayed with ``plt.show()``.
        """
        cm_df   = classification.classification_map
        k       = classification.activations_map.shape[0]
        umat    = classification.umatriz
        classes = sorted(np.unique(np.asarray(labels, dtype=int)).tolist())
        n_cls   = len(classes)

        if class_names is None:
            class_names = [f'Class {c}' for c in classes]
        if palette is None:
            cmap_ = plt.colormaps.get_cmap('tab10')
            palette = [cmap_(i / max(n_cls, 1)) for i in range(n_cls)]

        col_map  = {c: palette[i]     for i, c in enumerate(classes)}
        name_map = {c: class_names[i] for i, c in enumerate(classes)}

        # majority class and count per neuron
        majority, counts = {}, {}
        for (r, c), grp in cm_df.groupby(['x', 'y']):
            vc = grp['labels'].value_counts()
            majority[(r, c)] = int(vc.idxmax())
            counts[(r, c)]   = len(grp)

        fig = plt.figure(figsize=figsize)

        # legend between title and axes
        handles = [mpatches.Patch(color=col_map[c], label=name_map[c])
                   for c in classes]
        fig.legend(handles=handles, loc='upper center', ncol=n_cls,
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
            color  = col_map[maj]
            ax.plot(ux, uy, 'o', color=color, markersize=22,
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

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def hit_map(classification, labels, class_names=None, palette=None,
                figsize=(10, 9), title=None, filename=None):
        """Hit map where cell size encodes sample count and colour encodes majority class.

        Each neuron cell is drawn as a coloured square whose side length scales
        with ``sqrt(n / n_max)``, so high-load neurons appear larger and dead
        neurons (no samples) are shown as an empty grey background cell.  A
        light-tinted background fills the full cell area with the majority-class
        colour, providing an additional visual cue.

        :param classification: a completed :class:`~AMBER.Classification` instance.
            Must have been created with ``tagged=True``.
        :param labels: 1-D array-like of integer class codes, one per sample.
        :param class_names: list of human-readable class names in class-code
            order.  Defaults to ``'Class 0'``, ``'Class 1'``, …
        :param palette: list of matplotlib colour strings in class-code order.
            Defaults to the ``tab10`` colormap.
        :param figsize: figure size in inches (default ``(10, 9)``).
        :param title: figure suptitle.  If ``None``, a generic title is used.
        :param filename: if given, save the figure to this path; otherwise
            ``plt.show()`` is called.
        """
        cm_df   = classification.classification_map
        k       = classification.activations_map.shape[0]
        classes = sorted(np.unique(np.asarray(labels, dtype=int)).tolist())
        n_cls   = len(classes)

        if class_names is None:
            class_names = [f'Class {c}' for c in classes]
        if palette is None:
            cmap_ = plt.colormaps.get_cmap('tab10')
            palette = [cmap_(i / max(n_cls, 1)) for i in range(n_cls)]

        col_map  = {c: palette[i]     for i, c in enumerate(classes)}
        name_map = {c: class_names[i] for i, c in enumerate(classes)}

        majority, counts = {}, {}
        for (r, c), grp in cm_df.groupby(['x', 'y']):
            vc = grp['labels'].value_counts()
            majority[(r, c)] = int(vc.idxmax())
            counts[(r, c)]   = len(grp)

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
                            fontsize=7, color='white', fontweight='bold',
                            zorder=3)
                else:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               color='#e8e8e8', zorder=1))

        ax.set_xticks(range(k))
        ax.set_xlabel('Neuron column', fontsize=11)
        ax.set_yticks(range(k))
        ax.set_ylabel('Neuron row',    fontsize=11)
        ax.grid(True, linewidth=0.5, color='white', zorder=0)

        # legend above axes (2 rows if many classes)
        ncol_leg = min(n_cls, 5)
        handles  = [mpatches.Patch(color=col_map[c], label=name_map[c])
                    for c in classes]
        fig.legend(handles=handles, loc='upper center', ncol=ncol_leg,
                   fontsize=9, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.00))

        if title is None:
            title = ('Hit Map\nColour: majority class  ·  Cell size: sample count')
        fig.suptitle(title, fontsize=11, y=1.06)

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def weight_map_grid(som, classification, labels, class_names=None,
                        palette=None, figsize=None, title=None, filename=None):
        """Grid of weight-vector profiles coloured by majority class.

        Produces a ``map_size × map_size`` panel of subplots.  Each subplot
        shows the weight vector of one neuron as a line plot.  The subplot
        background is tinted with the majority-class colour; dead neurons
        (no assigned samples) are shown with a neutral grey background.  The
        neuron address ``[row, col]`` and sample count ``n=…`` are annotated
        inside each cell.

        :param som: a trained :class:`~AMBER.Map` instance.
        :param classification: a completed :class:`~AMBER.Classification`
            instance created with ``tagged=True``.
        :param labels: 1-D array-like of integer class codes, one per sample.
        :param class_names: list of human-readable class names in class-code
            order.  Defaults to ``'Class 0'``, ``'Class 1'``, …
        :param palette: list of matplotlib colour strings in class-code order.
            Defaults to the ``tab10`` colormap.
        :param figsize: figure size in inches.  Defaults to
            ``(2.4 * map_size, 1.9 * map_size)``.
        :param title: figure suptitle.  Defaults to a generic title.
        :param filename: if given, save to this path; otherwise ``plt.show()``.
        """
        cm_df   = classification.classification_map
        k       = som.map_size
        classes = sorted(np.unique(np.asarray(labels, dtype=int)).tolist())
        n_cls   = len(classes)

        if class_names is None:
            class_names = [f'Class {c}' for c in classes]
        if palette is None:
            cmap_ = plt.colormaps.get_cmap('tab10')
            palette = [cmap_(i / max(n_cls, 1)) for i in range(n_cls)]
        if figsize is None:
            figsize = (2.4 * k, 1.9 * k)

        col_map  = {c: palette[i]     for i, c in enumerate(classes)}
        name_map = {c: class_names[i] for i, c in enumerate(classes)}

        majority, counts = {}, {}
        for (r, c), grp in cm_df.groupby(['x', 'y']):
            vc = grp['labels'].value_counts()
            majority[(r, c)] = int(vc.idxmax())
            counts[(r, c)]   = len(grp)

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
                    bg = list(to_rgba(col))
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

        fig.text(0.5, 0.01, 'Time step',           ha='center',  fontsize=11)
        fig.text(0.01, 0.5, 'Amplitude (z-scored)', va='center',
                 rotation='vertical', fontsize=11)

        handles = [mpatches.Patch(color=col_map[c], label=name_map[c])
                   for c in classes]
        fig.legend(handles=handles, loc='upper center', ncol=n_cls,
                   fontsize=9, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.00))

        if title is None:
            title = f'SOM weight map — {k}×{k} grid'
        fig.suptitle(title, y=1.05, fontsize=12)

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def full_map_weights(map, labels=np.array([]), size_x=25, size_y=30, filename='full_map_weights'):
        """Grid of weight-vector line plots — one subplot per neuron.

        Produces a ``map_size × map_size`` panel of weight profiles and saves
        it to disk as an image file.

        :param map: a trained :class:`~AMBER.Map` instance
        :param labels: feature-name labels for the x-axis of each subplot
        :param size_x: total figure width in inches (default 25)
        :param size_y: total figure height in inches (default 30)
        :param filename: output file path (no extension); saved via ``fig.savefig``
        """
        fig, ax = plt.subplots(map.map_size, map.map_size, sharex='col', sharey='row', figsize=(size_x, size_y))
        for i in range(map.map_size):
            for j in range(map.map_size):
                weights = np.rot90(map.weights)
                ax[i, j].xticks = (np.arange(map.input_data_dimension), labels)
                ax[i, j].plot(weights[i, j], label='[' + str(j) + ',' + str(i) + ']')
        fig.savefig(filename)

    # ------------------------------------------------------------------
    # Temporal visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def trajectory(classification, temporal_analysis,
                   background='activations',
                   cmap_path='plasma', cmap_bg='YlOrRd',
                   figsize=(7, 7), title='BMU Trajectory',
                   random_seed=None):
        """Plot the time-ordered sequence of BMU positions on the SOM grid.

        The path is drawn as a colour-coded line (early = dark, late = bright)
        with arrows indicating direction.  The background shows either the
        activation counts or the U-matrix.

        :param classification:   a completed Classification instance
        :param temporal_analysis: the matching TemporalAnalysis instance
        :param background: 'activations' or 'umatrix'
        :param cmap_path: matplotlib colormap for the trajectory line
        :param cmap_bg:   matplotlib colormap for the background heatmap
        :param figsize:   figure size in inches
        :param title:     plot title
        :param random_seed: seed for the jitter RNG that slightly offsets overlapping
            trajectory points; pass an integer for a reproducible figure, None for
            a different jitter each call
        """
        traj = temporal_analysis.trajectory
        if len(traj) < 2:
            print("Trajectory too short to plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Background
        if background == 'umatrix':
            bg = classification.umatriz
        else:
            bg = classification.activations_map.astype(float)

        ax.imshow(bg, cmap=cmap_bg, origin='upper',
                  extent=[-0.5, bg.shape[1] - 0.5,
                           bg.shape[0] - 0.5, -0.5])

        # Draw grid lines
        k = classification.activations_map.shape[0]
        for g in range(k + 1):
            ax.axhline(g - 0.5, color='white', linewidth=0.4, alpha=0.4)
            ax.axvline(g - 0.5, color='white', linewidth=0.4, alpha=0.4)

        # Colour-coded path
        rows = np.array([p[0] for p in traj], dtype=float)
        cols = np.array([p[1] for p in traj], dtype=float)
        n    = len(traj)
        cmap = plt.colormaps.get_cmap(cmap_path)
        colors = cmap(np.linspace(0.15, 1.0, n - 1))

        jitter = 0.12
        _rng = np.random.default_rng(random_seed)
        offsets = _rng.uniform(-jitter, jitter, size=(n, 2))

        for t in range(n - 1):
            c0 = cols[t]     + offsets[t,   1]
            r0 = rows[t]     + offsets[t,   0]
            c1 = cols[t + 1] + offsets[t+1, 1]
            r1 = rows[t + 1] + offsets[t+1, 0]
            ax.annotate(
                '', xy=(c1, r1), xytext=(c0, r0),
                arrowprops=dict(
                    arrowstyle='->', color=colors[t],
                    lw=1.5, mutation_scale=10
                )
            )

        # Start / end markers
        ax.plot(cols[0],  rows[0],  'o', color='lime',   markersize=10,
                zorder=5, label='start')
        ax.plot(cols[-1], rows[-1], 's', color='white',  markersize=10,
                zorder=5, label='end')

        # Colourbar for time
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=n - 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Time step')

        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def transition_matrix_plot(temporal_analysis,
                               normalised=True,
                               cmap='Blues',
                               figsize=(8, 7),
                               title='Transition Matrix'):
        """Heatmap of neuron-to-neuron transition frequencies.

        :param temporal_analysis: a TemporalAnalysis instance
        :param normalised: if True, shows row-normalised probabilities;
                           if False, shows raw counts
        :param cmap:    matplotlib colormap
        :param figsize: figure size in inches
        :param title:   plot title
        """
        T = (temporal_analysis.transition_matrix_norm
             if normalised else temporal_analysis.transition_matrix)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(T, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax,
                     label='Probability' if normalised else 'Count')

        k = temporal_analysis.map_size
        # Mark neuron grid boundaries
        for g in range(0, k ** 2 + 1, k):
            ax.axhline(g - 0.5, color='grey', linewidth=0.5, alpha=0.6)
            ax.axvline(g - 0.5, color='grey', linewidth=0.5, alpha=0.6)

        ax.set_xlabel('To neuron index')
        ax.set_ylabel('From neuron index')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def dwell_time_map(temporal_analysis, classification,
                       cmap='Blues', figsize=(6, 5),
                       title='Mean Dwell Time per Neuron'):
        """Heatmap showing how long the signal dwells on each BMU on average.

        :param temporal_analysis: a TemporalAnalysis instance
        :param classification:    the matching Classification instance
        :param cmap:   matplotlib colormap
        :param figsize: figure size in inches
        :param title:   plot title
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

        # Annotate cells with values
        for r in range(k):
            for c in range(k):
                if dwell_grid[r, c] > 0:
                    ax.text(c, r, f'{dwell_grid[r, c]:.1f}',
                            ha='center', va='center',
                            fontsize=7, color='black')
        plt.tight_layout()
        plt.show()
