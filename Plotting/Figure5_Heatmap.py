import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# =========================
# USER OPTIONS
# =========================
SHOW_CONTOURS = False

data_dir = "Output_Gamma"


# =========================
# LOAD DATA
# =========================
gamma_mean = np.loadtxt(
    os.path.join(data_dir, "gamma_mean.csv"),
    delimiter=","
)

sync_mean = np.loadtxt(
    os.path.join(data_dir, "sync_mean.csv"),
    delimiter=","
)

e_gaba_values = np.loadtxt(
    os.path.join(data_dir, "e_gaba_values.csv"),
    delimiter=","
)

I_ext_values = np.loadtxt(
    os.path.join(data_dir, "I_ext_values.csv"),
    delimiter=","
)
I_ext_values=I_ext_values/100

# =========================
# CLASSIFY STATES
# =========================
state_map = np.zeros_like(gamma_mean, dtype=int)

for i in range(gamma_mean.shape[0]):
    for j in range(gamma_mean.shape[1]):

        g = gamma_mean[i, j]
        s = sync_mean[i, j]

        # Missing data
        if np.isnan(g) or np.isnan(s):
            state_map[i, j] = -1

        # High synchrony
        elif s >= 0.75:

            if g == 0:
                state_map[i, j] = 5

            elif g > 0.25:
                state_map[i, j] = 3

            else:
                state_map[i, j] = 4

        # Moderate synchrony
        elif s >= 0.4:

            if g > 0.25:
                state_map[i, j] = 2

            else:
                state_map[i, j] = 0

        # Low synchrony
        else:

            if g > 0.25:
                state_map[i, j] = 1

            else:
                state_map[i, j] = 0


# =========================
# HEATMAP COLORMAP
# =========================
cmap = ListedColormap([
    "#F28E2B",  # State 0 : orange
    "#00441B",  # State 1 : dark green
    "#238B45",  # State 2 : medium green
    "#66C2A4",  # State 3 : light green
    "#FB6A4A",  # State 4 : light red
    "#67000D"   # State 5 : dark red
])

bounds = np.arange(-0.5, 6.5, 1)
norm = BoundaryNorm(bounds, cmap.N)


# =========================
# FIGURE LAYOUT
# =========================
FIG_WIDTH_CM = 17.8
FIG_WIDTH_IN = FIG_WIDTH_CM / 2.54

fig = plt.figure(
    figsize=(FIG_WIDTH_IN, 4),
    constrained_layout=True
)

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[4, 1],
    wspace=0.05
)

ax = fig.add_subplot(gs[0, 0])
ax_scheme = fig.add_subplot(gs[0, 1])


# =========================
# MAIN HEATMAP
# =========================
x_edges = np.arange(len(I_ext_values) + 1)
y_edges = np.arange(len(e_gaba_values) + 1)

mesh = ax.pcolormesh(
    x_edges,
    y_edges,
    state_map,
    cmap=cmap,
    norm=norm,
    shading="flat"
)

# ax.set_aspect("equal")
ax.set_aspect("auto")

x_centers = np.arange(len(I_ext_values)) + 0.5
y_centers = np.arange(len(e_gaba_values)) + 0.5

ax.set_xticks(x_centers)

ax.set_xticklabels(I_ext_values, ha='center', fontsize=12)

ax.set_xlabel(
    r"$I_{ext}$ ($\mu A$)",
    fontsize=14,
    labelpad=5
)

ax.set_yticks(y_centers)
ax.set_yticklabels(
    e_gaba_values,
    fontsize=12
)

ax.set_ylabel(
    r"$E_{\mathrm{GABA}}$ (mV)",
    fontsize=14,
    labelpad=5
)

ax.tick_params(axis="x", pad=5)
ax.tick_params(axis="y", pad=5)

if SHOW_CONTOURS:

    X, Y = np.meshgrid(
        x_centers,
        y_centers
    )

    ax.contour(
        X,
        Y,
        state_map,
        levels=np.arange(0.5, 5.5, 1),
        colors="black",
        linewidths=2
    )



# ======================================================
# CLASSIFICATION SCHEME SUBPLOT
# ======================================================

scheme_matrix = np.full((3, 3), -1, dtype=int)

# Bottom row: infrequent stable oscillations
scheme_matrix[0, 1] = 0
scheme_matrix[0, 2] = 1

# Middle row: some stable oscillations
scheme_matrix[1, 2] = 2

# Top row: frequent stable oscillations
scheme_matrix[2, 2] = 3
scheme_matrix[2, 1] = 4
scheme_matrix[2, 0] = 5

scheme_cmap = ListedColormap([
    "white",    # Unused combinations
    "#F28E2B",  # State 0
    "#00441B",  # State 1
    "#238B45",  # State 2
    "#66C2A4",  # State 3
    "#FB6A4A",  # State 4
    "#67000D"   # State 5
])

scheme_plot = scheme_matrix + 1

scheme_norm = BoundaryNorm(
    np.arange(-0.5, 7.5, 1),
    scheme_cmap.N
)

ax_scheme.pcolormesh(
    np.arange(4),
    np.arange(4),
    scheme_plot,
    cmap=scheme_cmap,
    norm=scheme_norm,
    shading="flat"
)

ax_scheme.set_aspect("equal")

# Tick positions
ax_scheme.set_xticks(np.arange(3) + 0.5)
ax_scheme.set_yticks(np.arange(3) + 0.5)

# Tick labels
ax_scheme.set_xticklabels(
    ["none", "rare", "frequent"],
    rotation=45,
    ha="right",
    rotation_mode="anchor",
    fontsize=6
)

ax_scheme.set_yticklabels(
    ["infrequent", "some", "frequent"],
    rotation=45,
    ha="right",
    va="center",
    fontsize=6
)

# Axis labels
ax_scheme.set_xlabel(
    "Transient gamma events",
    fontsize=8
)

ax_scheme.set_ylabel(
    "Sustained oscillations",
    fontsize=8
)

# Grid lines
for x in range(4):
    ax_scheme.axvline(
        x,
        color="black",
        linewidth=1
    )

for y in range(4):
    ax_scheme.axhline(
        y,
        color="black",
        linewidth=1
    )

ax_scheme.set_title(
    "Classification Scheme",
    fontsize=10,
    pad=25
)

# plt.tight_layout()

plt.savefig(
    os.path.join(data_dir, "state_heatmap.png"),
    dpi=600, bbox_inches='tight'
)

plt.savefig(
    os.path.join(data_dir, "state_heatmap.svg")
)

plt.show()