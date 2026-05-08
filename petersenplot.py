import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

# -------------------------
# DATA
# -------------------------
def add_confidence_ellipse(ax, x, y, n_std=2.0, **kwargs):
    """
    Draw covariance-based confidence ellipse.
    n_std = 1 (68%), 2 (95% approx for Gaussian)
    """
    if len(x) < 2:
        return

    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # angle of ellipse
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # width/height
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        (mean_x, mean_y),
        width,
        height,
        angle=theta,
        fill=False,
        **kwargs
    )

    ax.add_patch(ellipse)

def plot_kde(ax, x, y, cmap="Blues"):
    """
    2D KDE contour in Petersen space
    """
    if len(x) < 3:
        return

    values = np.vstack([x, y])
    kde = gaussian_kde(values)

    # grid
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1

    Xg, Yg = np.mgrid[
        x_min:x_max:200j,
        y_min:y_max:200j
    ]

    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    ax.contourf(Xg, Yg, Z, levels=8, cmap=cmap, alpha=0.25)
    ax.contour(Xg, Yg, Z, levels=6, colors='black', linewidths=0.5, alpha=0.4)

stars = np.array([
    "ASAS J071842-5947.7", "DH Peg", "AE Boo", "BX Leo", "HY Com",
    "AP Ser", "TV Lyn", "CS Eri", "RU Psc", "EV Psc"
])

f0 = np.array([
    3.917573, 3.913785, 3.17570, 2.755466, 2.22927,
    2.933814, 4.155439, 3.21204, 2.561068, 3.2652273
])

f0_err = np.array([
    5.1e-6, 1.4e-6, 1.7e-5, 9.8e-6, 1.3e-5,
    7.6e-6, 3.5e-6, 1.5e-5, 5.2e-6, 1.6e-7
])

fx_err = np.array([
    7.5e-5, 3.4e-4, 3.2e-4, 3.3e-4, 1.0e-3,
    7.1e-4, 4.7e-4, 4.7e-4, 2.3e-4, 5.9e-6
])

fx = np.array([
    6.39062, 6.2752, 5.1824, 4.4979, 3.552,
    4.6663, 6.7590, 5.2080, 4.1658, 5.322257
])

# -------------------------
# PERIODS
# -------------------------
P0 = 1 / f0
Px = 1 / fx

P0_err = f0_err / (f0 ** 2)
Px_err = fx_err / (fx ** 2)

ratio = Px / P0
ratio_err = ratio * np.sqrt(
    (Px_err / Px) ** 2 +
    (P0_err / P0) ** 2
)

# =========================================================
# 1) NORMAL PETERSEN DIAGRAM (clean thesis figure)
# =========================================================
# UNIQUE COLORS FOR EACH STAR (independent of clustering)
cmap_points = plt.colormaps["tab20"].resampled(len(stars))
colors = cmap_points(np.arange(len(stars)))

fig, ax = plt.subplots(figsize=(4, 3))

for i in range(len(stars)):
    ax.errorbar(
        P0[i], ratio[i],
        xerr=P0_err[i],
        yerr=ratio_err[i],
        fmt='o',
        color=colors[i],
        markeredgecolor='black',
        capsize=2
    )
    ax.text(P0[i], ratio[i], stars[i], fontsize=7, alpha=0.7)

ax.set_xlabel(r'$P_0$ (days)')
ax.set_ylabel(r'$P_x / P_0$')
ax.set_title("Petersen Diagram")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("petersen_normal.jpg", dpi=150, bbox_inches = "tight")
plt.show()
plt.close()

# =========================================================
# 2) SILHOUETTE + ELBOW (OPTIMAL k)
# =========================================================
X = np.column_stack([P0, ratio])
X_scaled = StandardScaler().fit_transform(X)

ks_sil = range(2, 10)
ks_elbow = range(1, 10)

sil_scores = []
inertias = []

for k in ks_sil:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    labels = gmm.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

for k in ks_elbow:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax1 = plt.subplots(figsize=(7, 4.5))

ax1.plot(list(ks_sil), sil_scores, marker='o', color='tab:blue', label='Silhouette score')
ax1.set_xlabel("k")
ax1.set_ylabel("Silhouette score", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(list(ks_elbow), inertias, marker='s', color='tab:red', label='Inertia')
ax2.set_ylabel("Inertia", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Cluster Validation in Petersen Space")
plt.tight_layout()
plt.savefig("petersen_space_val.jpg", dpi=150, bbox_inches = "tight")
plt.show()
plt.close()

# =========================================================
# 3) FINAL CLUSTERING (FIXED: k = 2)
# =========================================================
k = 2

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

cmap = plt.colormaps["Set2"].resampled(k)
cluster_colors = cmap(clusters)

# =========================================================
# 4) CLUSTERED PETERSEN DIAGRAM
# =========================================================
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(len(stars)):
    ax.errorbar(
        P0[i], ratio[i],
        xerr=P0_err[i],
        yerr=ratio_err[i],
        fmt='o',
        color=cluster_colors[i],
        markeredgecolor='black',
        capsize=2
    )
    ax.text(P0[i], ratio[i], stars[i], fontsize=7, alpha=0.7)

# -------------------------
# CENTROIDS (cluster-colored crosses, NOT red)
# -------------------------
for c in range(k):
    idx = clusters == c

    ax.scatter(
        np.mean(P0[idx]),
        np.mean(ratio[idx]),
        marker='X',
        s=220,
        color=cmap(c),
        edgecolor='black',
        linewidth=1.3,
        zorder=5
    )

# -------------------------
# CONFIDENCE ELLIPSES
# -------------------------
for c in range(k):
    idx = clusters == c
    
    if np.sum(idx) < 3:
        continue

    add_confidence_ellipse(
        ax,
        P0[idx],
        ratio[idx],
        n_std=2,
        edgecolor=cmap(c),
        linewidth=1.5,
        linestyle='--',
        alpha=0.7
    )

# -------------------------
# LEGEND
# -------------------------
legend_handles = [
    Line2D([0],[0],
        marker='o',
        linestyle='',
        label=f'Cluster {i+1}',
        markerfacecolor=cmap(i),
        markeredgecolor='black',
        markersize=7
    )
    for i in range(k)
]

ax.legend(handles=legend_handles, title="Clusters", loc="lower right")

ax.set_xlabel(r'$P_0$ (days)')
ax.set_ylabel(r'$P_x / P_0$')
ax.set_title("Petersen Diagram (clustering variant)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("petersen_clustered_k2.jpg", dpi=150, bbox_inches = "tight")
plt.show()
plt.close()

# =========================================================
# OUTPUT
# =========================================================
print("\nCluster assignment:")
for i, s in enumerate(stars):
    print(f"{s:25s} -> Cluster {clusters[i]+1}")
    
# =========================================================
# LOG PETERSEN DIAGRAM
# =========================================================

logP0 = np.log10(P0)
logP0_err = P0_err / (P0 * np.log(10))

fig, ax = plt.subplots(figsize=(4,3))

for i in range(len(stars)):
    ax.errorbar(
        logP0[i],
        ratio[i],
        xerr=logP0_err[i],
        yerr=ratio_err[i],
        fmt='o',
        color=colors[i],
        markeredgecolor='black',
        capsize=2
    )

    ax.text(
        logP0[i],
        ratio[i],
        stars[i],
        fontsize=7,
        alpha=0.7
    )

ax.set_xlabel(r'$\log P_0$ (days)')
ax.set_ylabel(r'$P_x/P_0$')
ax.set_title("Log-Petersen Diagram")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("log_petersen.jpg", dpi=150, bbox_inches="tight")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(5,4))

for i in range(len(stars)):
    # only plot if fx exists (skip EF Cnc)
    if np.isnan(fx[i]):
        continue

    ax.errorbar(
        f0[i], fx[i],
        xerr=f0_err[i],
        yerr=fx_err[i],
        fmt='o',
        color='tab:blue',
        markeredgecolor='black',
        capsize=3,
        alpha=0.9
    )

    ax.text(
        f0[i], fx[i],
        stars[i],
        fontsize=7,
        alpha=0.7
    )

ax.set_xlabel(r'$f_0$ (d$^{-1}$)')
ax.set_ylabel(r'$f_x$ (d$^{-1}$)')
ax.set_title(r'$f_0$ vs $f_x$ (frequency space check)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("f0_vs_fx.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

# =========================
# LEFT: PETERSEN DIAGRAM
# =========================
for i in range(len(stars)):

    ax1.errorbar(
        P0[i], ratio[i],
        xerr=P0_err[i],
        yerr=ratio_err[i],
        fmt='o',
        color='tab:blue',
        markeredgecolor='black',
        capsize=2,
        alpha=0.9
    )

    # highlight FX stars
    if not np.isnan(fx[i]):
        ax1.scatter(P0[i], ratio[i],
                    s=120,
                    facecolors='none',
                    edgecolors='red',
                    linewidth=1.5)

    ax1.text(P0[i], ratio[i], stars[i], fontsize=7, alpha=0.7)

ax1.set_xlabel(r'$P_0$ (days)')
ax1.set_ylabel(r'$P_x/P_0$')
ax1.set_title("Petersen Diagram")
ax1.grid(alpha=0.3)

# =========================
# RIGHT: FREQUENCY SPACE
# =========================
for i in range(len(stars)):

    if np.isnan(fx[i]):
        continue

    ax2.errorbar(
        f0[i], fx[i],
        xerr=f0_err[i],
        yerr=fx_err[i],
        fmt='o',
        color='tab:blue',
        markeredgecolor='black',
        capsize=2,
        alpha=0.9
    )

    # highlight FX stars
    ax2.scatter(f0[i], fx[i],
                s=120,
                facecolors='none',
                edgecolors='red',
                linewidth=1.5)

    ax2.text(f0[i], fx[i], stars[i], fontsize=7, alpha=0.7)

ax2.set_xlabel(r'$f_0$ (d$^{-1}$)')
ax2.set_ylabel(r'$f_x$ (d$^{-1}$)')
ax2.set_title(r'$f_0$ vs $f_x$')
ax2.grid(alpha=0.3)

# =========================
# GLOBAL TITLE
# =========================
plt.suptitle("Petersen vs Frequency Space: FX-mode diagnostics", fontweight='bold')

plt.tight_layout()
plt.savefig("petersen_vs_frequency_combined.jpg", dpi=250, bbox_inches="tight")
plt.show()
plt.close()
