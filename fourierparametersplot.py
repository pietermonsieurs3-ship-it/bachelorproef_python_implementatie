import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------
# DATA
# -------------------------

def add_cov_ellipse(ax, x, y, n_std=1.5, facecolor='none', **kwargs):
    if len(x) < 2:
        return

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * n_std * np.sqrt(vals)

    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        **kwargs
    )

    ax.add_patch(ell)

stars = [
    "ASAS J071842-5947.7", "DH Peg", "AE Boo", "BX Leo", "HY Com",
    "AP Ser", "EF Cnc", "TV Lyn", "CS Eri", "RU Psc", "EV Psc"
]

R21 = np.array([
    0.0679, 0.18489, 0.0961, 0.0705, 0.0733,
    0.1329, 0.2379, 0.1467, 0.180, 0.0704, 0.1081
])

R31 = np.array([
    0.0457, 0.05535, 0.0672, 0.0678, 0.0682,
    0.0884, 0.0864, 0.0844, 0.094, 0.0658, 0.0738
])

phi21 = np.array([
    0.4383, -0.28825, 0.521, -0.826, -0.999,
    0.5613, -0.5093, -0.5118, -0.5070, -0.7598, 0.5324
])

phi31 = np.array([
    0.1717, -0.7265, 0.146, -1.104, -2.590,
    0.1117, -1.0764, -0.9941, -1.040, -1.5894, 0.0875
])

# -------------------------
# ERRORS (RESTORED)
# -------------------------
R21_err = np.array([2.5e-4, 6.8e-5, 6.3e-4, 4.7e-4, 5.7e-4,
                    3.4e-4, 2.2e-4, 1.9e-4, 1.5e-3, 4.7e-4, 2.3e-4])

R31_err = R21_err

phi21_err = np.array([5.9e-4, 6.2e-5, 1.0e-3, 1.1e-3, 2.6e-3,
                      4.2e-4, 1.3e-4, 2.2e-4, 7.8e-4, 6.1e-4, 3.4e-4])

phi31_err = np.array([5.9e-4, 1.9e-4, 1.4e-3, 1.1e-3, 1.3e-3,
                      6.2e-4, 3.4e-4, 3.7e-4, 1.4e-3, 6.5e-4, 5.0e-4])

# -------------------------
# COLORS
# -------------------------
cmap = plt.colormaps["tab20"].resampled(len(stars))
colors = cmap(np.arange(len(stars)))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

for i in range(len(stars)):

    ax0.errorbar(phi21[i], R21[i],
                 xerr=phi21_err[i], yerr=R21_err[i],
                 fmt='o', color=colors[i], ecolor=colors[i],
                 capsize=2, markeredgecolor='black', alpha=0.9)

    ax0.text(phi21[i], R21[i], stars[i], fontsize=7, alpha=0.7)

    ax1.errorbar(phi31[i], R31[i],
                 xerr=phi31_err[i], yerr=R31_err[i],
                 fmt='o', color=colors[i], ecolor=colors[i],
                 capsize=2, markeredgecolor='black', alpha=0.9)

    ax1.text(phi31[i], R31[i], stars[i], fontsize=7, alpha=0.7)

for ax in (ax0, ax1):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=9)

ax0.set_xlabel(r'$\phi_{21}$')
ax0.set_ylabel(r'$R_{21}$')
ax0.set_title(r'$R_{21}$ vs $\phi_{21}$')

ax1.set_xlabel(r'$\phi_{31}$')
ax1.set_ylabel(r'$R_{31}$')
ax1.set_title(r'$R_{31}$ vs $\phi_{31}$')

plt.suptitle("Fourier Parameter Space of RR Lyrae Stars (standard)", fontweight="bold")
plt.tight_layout()
plt.savefig("fp_space_normal.jpg",dpi=300, bbox_inches = "tight")
plt.show()
plt.close(fig)

X = np.column_stack([phi21, R21, phi31, R31])
X_scaled = StandardScaler().fit_transform(X)

ks_sil = range(2, 10)
ks_elbow = range(1, 10)

sil_scores = []
inertias = []

for k in ks_sil:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

for k in ks_elbow:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

l1 = ax1.plot(list(ks_sil), sil_scores, marker='o', color='tab:blue', label='Silhouette score')
ax1.set_xlabel("k")
ax1.set_ylabel("Silhouette", color='tab:blue')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
l2 = ax2.plot(list(ks_elbow), inertias, marker='s', color='tab:red', label='Inertia')
ax2.set_ylabel("Inertia", color='tab:red')

# -------------------------
# RESTORED LEGEND
# -------------------------
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("Cluster Validation in Fourier Space", fontweight="bold")
plt.tight_layout()
plt.savefig("fp_cluster_val.jpg",dpi=300, bbox_inches = "tight")
plt.show()
plt.close(fig)

k = 3

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

cmap = plt.colormaps["Set2"].resampled(k)
cluster_colors = cmap(clusters)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

for i in range(len(stars)):

    ax0.scatter(phi21[i], R21[i], color=cluster_colors[i], edgecolor='black')
    ax0.text(phi21[i], R21[i], stars[i], fontsize=7, alpha=0.6)

    ax1.scatter(phi31[i], R31[i], color=cluster_colors[i], edgecolor='black')
    ax1.text(phi31[i], R31[i], stars[i], fontsize=7, alpha=0.6)

for c in range(k):
    idx = clusters == c
    if np.sum(idx) == 0:
        continue

    ax0.scatter(np.mean(phi21[idx]), np.mean(R21[idx]),
                marker='X', s=180, color=cmap(c), edgecolor='black')

    ax1.scatter(np.mean(phi31[idx]), np.mean(R31[idx]),
                marker='X', s=180, color=cmap(c), edgecolor='black')
                
    add_cov_ellipse(ax0, phi21[idx], R21[idx],
                    n_std=1.5,
                    edgecolor=cmap(c),
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.8)

    add_cov_ellipse(ax1, phi31[idx], R31[idx],
                    n_std=1.5,
                    edgecolor=cmap(c),
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.8)

ax0.set_title("Clustered φ21 vs R21")
ax1.set_title("Clustered φ31 vs R31")

for ax in (ax0, ax1):
    ax.grid(alpha=0.25)

legend_handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='',
        label=f'Cluster {i+1}',
        markerfacecolor=cmap(i),
        markeredgecolor='black',
        markersize=7
    )
    for i in range(k)
]

ax1.legend(
    handles=legend_handles,
    title="Clusters",
    loc="upper left",
    fontsize=8,
    frameon=True
)

ax0.set_title("Clustered φ21 vs R21")
ax1.set_title("Clustered φ31 vs R31")

ax0.set_xlabel(r'$\phi_{21}$')
ax0.set_ylabel(r'$R_{21}$')

ax1.set_xlabel(r'$\phi_{31}$')
ax1.set_ylabel(r'$R_{31}$')

for ax in (ax0, ax1):
    ax.grid(alpha=0.25)

plt.suptitle("Fourier Parameter Space of RR Lyrae Stars (cluster variant)", fontweight="bold")

fig.tight_layout()
fig.subplots_adjust(top=0.88, bottom=0.12)
plt.savefig("fp_space_clustered.jpg",dpi=300, bbox_inches = "tight")
plt.show()
plt.close(fig)

# -------------------------
# PCA
# -------------------------
# PCA on the same standardized feature space used for clustering

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance:", pca.explained_variance_ratio_)

cmap = plt.colormaps["Set2"].resampled(k)

fig, ax = plt.subplots(figsize=(6, 5))

for c in range(k):
    idx = clusters == c

    # points
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
               s=80,
               color=cmap(c),
               edgecolor='black',
               alpha=0.85,
               label=f"Cluster {c+1}")

    # centroid
    centroid = X_pca[idx].mean(axis=0)
    ax.scatter(centroid[0], centroid[1],
               marker='X',
               s=200,
               color=cmap(c),
               edgecolor='black',
               linewidth=1.2)

    # ellipse
    add_cov_ellipse(ax,
                    X_pca[idx, 0],
                    X_pca[idx, 1],
                    n_std=1.5,
                    edgecolor=cmap(c),
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.8)

    # -------------------------
    # STAR LABELS (KEY ADDITION)
    # -------------------------
    for i in np.where(idx)[0]:
        ax.annotate(
            stars[i],
            (X_pca[i, 0], X_pca[i, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            alpha=0.75
        )

# -------------------------
# STYLE
# -------------------------
ax.set_xlabel(f"PC1")
ax.set_ylabel(f"PC2")
ax.set_title(
    "PCA Projection of Fourier Parameter Space",
    fontweight="bold"
)

total_var = pca.explained_variance_ratio_.sum()

textstr = (
    f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%\n"
    f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%\n"
    f"Total: {total_var*100:.1f}%"
)

ax.text(
    0.02, 0.98, textstr,
    transform=ax.transAxes,   # <-- key: axes-relative coords
    fontsize=9,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.9,
        edgecolor='black'
    )
)

ax.grid(alpha=0.25)
ax.legend(title="Clusters", fontsize=8)

plt.tight_layout()
plt.savefig("fp_pca_plot.jpg",dpi=300, bbox_inches = "tight")
plt.show()
plt.close(fig)

# -------------------------
# PCA LOADINGS PLOT
# -------------------------

features = ["phi21", "R21", "phi31", "R31"]

loadings = pca.components_.T  # shape: (features, PCs)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# PC1
axes[0].bar(features, loadings[:, 0])
axes[0].set_title("PC1 Loadings")
axes[0].set_ylabel("Contribution weight")
axes[0].grid(alpha=0.3)

# PC2
axes[1].bar(features, loadings[:, 1])
axes[1].set_title("PC2 Loadings")
axes[1].grid(alpha=0.3)

plt.suptitle("PCA Loadings: Fourier Parameter Contributions", fontweight="bold")
plt.tight_layout()
plt.savefig("pca_loadings_plot.jpg",dpi=300, bbox_inches = "tight")
plt.show()
plt.close(fig)

# Feedback on screen
print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"Total: {total_var*100:.1f}%")

from scipy.stats import pointbiserialr, spearmanr
import numpy as np

# -------------------------
# FX FLAG (your ground truth)
# -------------------------
fx_flag = np.array([
    1, 1, 1, 1, 1,
    1, 0, 1, 1, 1, 1
])

features = {
    "R21": R21,
    "phi21": phi21,
    "R31": R31,
    "phi31": phi31
}

print("\n=== POINT-BISERIAL CORRELATION (fx vs parameters) ===")

for name, x in features.items():
    r, p = pointbiserialr(fx_flag, x)
    print(f"{name:6s} | r = {r:+.3f} | p = {p:.4f}")

print("\n=== SPEARMAN CORRELATION (robust monotonic test) ===")

for name, x in features.items():
    rho, p = spearmanr(fx_flag, x)
    print(f"{name:6s} | rho = {rho:+.3f} | p = {p:.4f}")

def permutation_test(x, y, n_perm=10000):
    observed = np.abs(np.corrcoef(x, y)[0,1])

    count = 0
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        stat = np.abs(np.corrcoef(x, y_perm)[0,1])
        if stat >= observed:
            count += 1

    return observed, count / n_perm


print("\n=== PERMUTATION TEST (robust small-N inference) ===")

for name, x in features.items():
    r_obs, p_perm = permutation_test(x, fx_flag)
    print(f"{name:6s} | |r| = {r_obs:.3f} | p_perm = {p_perm:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(4,3))

plt.scatter(R21, fx_flag, s=80, edgecolor="black")
plt.xlabel("R21")
plt.ylabel("FX flag")
plt.title("FX vs R21")
plt.grid(alpha=0.3)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

# -------------------------
# INPUT DATA
# -------------------------
R21 = np.array([0.0678, 0.18489, 0.0967, 0.0705, 0.0733,
                0.1329, 0.2379, 0.1467, 0.180, 0.0704, 0.1081])

fx_flag = np.array([1, 1, 1, 1, 1,
                    1, 0, 1, 1, 1, 1])  # adjust if needed

# -------------------------
# SAFE CORRELATION FUNCTION
# -------------------------
def safe_corr(x, y):
    # avoid constant arrays
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

# -------------------------
# BOOTSTRAP
# -------------------------
n_boot = 5000
boot_corrs = []

n = len(R21)

for _ in range(n_boot):
    idx = np.random.randint(0, n, n)

    x = R21[idx]
    y = fx_flag[idx]

    r = safe_corr(x, y)

    if not np.isnan(r):
        boot_corrs.append(r)

boot_corrs = np.array(boot_corrs)

# -------------------------
# RESULTS
# -------------------------
mean_r = np.mean(boot_corrs)
ci_low = np.percentile(boot_corrs, 2.5)
ci_high = np.percentile(boot_corrs, 97.5)

p_negative = np.mean(boot_corrs < 0)

print("=== BOOTSTRAP RESULTS ===")
print(f"Mean r = {mean_r:.3f}")
print(f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]")
print(f"P(r < 0) = {p_negative:.3f}")
print(f"Valid samples used: {len(boot_corrs)} / {n_boot}")

# -------------------------
# PLOT
# -------------------------
plt.figure(figsize=(4,3))
plt.hist(boot_corrs, bins=25, color="steelblue", edgecolor="black", alpha=0.8)

plt.axvline(mean_r, color="red", label=f"mean r = {mean_r:.2f}")
plt.axvline(ci_low, linestyle="--", color="black")
plt.axvline(ci_high, linestyle="--", color="black")

plt.title("Bootstrap Distribution: R21 vs fx")
plt.xlabel("Correlation r")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------
# PLOT
# -------------------------
plt.figure(figsize=(6,5))

for i in range(len(stars)):

    plt.errorbar(
        phi21[i], R21[i],
        xerr=phi21_err[i],
        yerr=R21_err[i],
        fmt='o',
        capsize=2,
        markeredgecolor='black',
        alpha=0.9
    )

    plt.text(phi21[i], R21[i], stars[i], fontsize=7, alpha=0.75)

# -------------------------
# STYLE
# -------------------------
plt.xlabel(r'$\phi_{21}$')
plt.ylabel(r'$R_{21}$')
plt.title(r'Fourier Parameter Space: $R_{21}$ vs $\phi_{21}$')

plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("R21_vs_phi21_thesis.jpg", dpi=300, bbox_inches="tight")
plt.show()
