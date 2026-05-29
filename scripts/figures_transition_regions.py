# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import percusion
from percusion import utils
from doldrumsVerticalMotion import circleUtils
import matplotlib.colors as colors

from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%

ds_hamp = xr.open_dataset(
    "ipfs://bafybeicahqvp4lovuqpu63euo5kbc22sdq4jp5p6h6wib373x72ki34tiu", engine="zarr"
)

ds_ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)
ds_ds = ds_ds.swap_dims({"circle": "circle_id"})

# %%

hamp_orcestra = ds_hamp.sel(time=slice(utils.campaign_start, utils.campaign_end))
hamp_orcestra = hamp_orcestra.where(
    (hamp_orcestra["IWV"] > 0) & (hamp_orcestra["IWV"] < 100), drop=True
)
iwv_hamp_orcestra = hamp_orcestra["IWV"]

# %%

iwv_ds_orcestra = ds_ds["iwv_mean"].to_numpy()

iwv_circle_min = np.empty(len(ds_ds.circle_id))
iwv_circle_max = np.copy(iwv_circle_min)
iwv_circle_mean = np.copy(iwv_circle_min)

for i, circle_id in enumerate(ds_ds.circle_id.values):
    sonde_ids = circleUtils.get_sonde_serial_ids(ds_ds, circle_id)

    iwv_circle_i = ds_ds["iwv"].where(
        ds_ds.vaisala_serial_id.isin(sonde_ids), drop=True
    )

    iwv_circle_min[i] = iwv_circle_i.min().values
    iwv_circle_max[i] = iwv_circle_i.max().values
    iwv_circle_mean[i] = iwv_circle_i.mean().values

# %%
cwv_threshold = 48
transition_circles = np.where(
    (iwv_circle_min < cwv_threshold) & (iwv_circle_max > cwv_threshold)
)[0]

smaller_cwv_circles = np.where(iwv_circle_max < cwv_threshold)[0]
larger_cwv_circles = np.where(iwv_circle_min > cwv_threshold)[0]


# %%

x = np.arange(len(iwv_circle_min))

fig, ax = plt.subplots(
    1, 2, figsize=(10, 4), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
)

col_ds, col_hamp = "C0", "C1"

plt.sca(ax[0])

scatter_kwargs = {"color": col_ds, "clip_on": False}
hlines_kwargs = {"color": "k", "alpha": 0.5, "linewidth": 1, "linestyle": ":"}

for i_m, mask in enumerate([transition_circles, ~np.isin(x, transition_circles)]):

    alpha = 1.0 if i_m == 0 else 0.35
    marker = "o"  # if i_m == 0 else "x"

    plt.vlines(
        x[mask],
        iwv_circle_min[mask],
        iwv_circle_max[mask],
        alpha=alpha,
        linewidth=1,
        **scatter_kwargs,
    )

    plt.scatter(
        x[mask],
        iwv_circle_mean[mask],
        s=15,
        marker=marker,
        alpha=alpha,
        **scatter_kwargs,
    )


plt.axhline(48, **hlines_kwargs)

plt.xlabel("circle number")
plt.ylabel("CWV / mm")

plt.tight_layout()
plt.xlim(xmin=0)


plt.sca(ax[1])

bins = np.arange(30, 75, 1.0)

hist_kwargs = {
    "density": True,
    "orientation": "horizontal",
    "bins": bins,
}

for h_type in ["step"]:
    alpha = 1.0 if h_type == "step" else 0.25

    ax[1].hist(
        ds_ds.iwv.values,
        histtype=h_type,
        color=col_ds,
        alpha=alpha,
        label="dropsondes",
        **hist_kwargs,
    )

    ax[1].hist(
        iwv_hamp_orcestra.values,
        histtype=h_type,
        color=col_hamp,
        alpha=alpha,
        label="HAMP",
        **hist_kwargs,
    )

plt.axhline(48, **hlines_kwargs)
plt.ylim(bins[0], bins[-1])
plt.legend()

ax[1].set_xlabel("PDF")

for a in ax:
    a.spines["left"].set_position(("outward", 5))

sns.despine()

plt.savefig(
    f"{PROJECT_ROOT}/figures/transition_circles.pdf",
    bbox_inches="tight",
)

print(
    f"Out of {len(ds_ds.circle_id)}: \n {len(transition_circles)} circles are transition circles, \n"
    f" {len(smaller_cwv_circles)} circles have CWV entirely below {cwv_threshold} mm, \n"
    f" {len(larger_cwv_circles)} circles have CWV entirely above {cwv_threshold} mm."
)

# %%
# Cloud mask binned by IWV
ds_hamp_active = xr.open_dataset(
    "ipfs://bafybeigmd3dovwm45ylfqxnn2jphsrdjl2jt3dfytv7grkyhleaq42jthe", engine="zarr"
)

cloud_mask = xr.where(ds_hamp_active["Ze"] > 1e-3, 1, 0)
iwv_hamp_interpolated = iwv_hamp_orcestra.interp(time=ds_hamp_active.time)
cloud_mask_binned_iwv = cloud_mask.groupby_bins(iwv_hamp_interpolated, bins=bins).mean()

# %%
# Upwelling longwave radiation binned by IWV
ds_bacardi = xr.open_dataset(
    "ipfs://bafybeiaoalflfftmsfqakenwp5gpxnxfjhaxkrjq7gpa4ucpbwj5jdb6qi", engine="zarr"
)

ful_bacardi = ds_bacardi["FUL"]
iwv_bacardi_interpolated = iwv_hamp_orcestra.interp(time=ful_bacardi.TIME)
ful_binned_iwv = ful_bacardi.groupby_bins(iwv_bacardi_interpolated, bins=bins).mean()

# %%
fig, ax = plt.subplots(
    1,
    2,
    figsize=(12, 4),
    sharey=True,
    gridspec_kw={"width_ratios": [2, 1], "wspace": 0.3},
)

cwv_xmin, cwv_xmax = 40, 67.5

cbar = cloud_mask_binned_iwv.sel(height=slice(50, 14e3)).plot.contourf(
    levels=np.arange(0.02, 0.3, 0.02),
    y="height",
    ax=ax[0],
    label=f"IWV bin {bin}",
    cmap="pink_r",
    norm=colors.Normalize(vmin=0, vmax=0.3),
    add_colorbar=False,
)

pos = ax[0].get_position()

cax = fig.add_axes(
    [
        pos.x0 + 0.1 * pos.width,  # left
        pos.y1 + 0.05,  # bottom
        0.8 * pos.width,  # width
        0.02,  # height
    ]
)

cb = fig.colorbar(
    cbar,
    cax=cax,
    orientation="horizontal",
    label="frequency of Ze > 1e-3 / %",
    shrink=0.75,
)

cb.ax.xaxis.set_label_position("top")
cb.ax.xaxis.set_ticks_position("top")

# plt.colorbar(
#     cbar,
#     ax=ax[0],
#     #orientation="horizontal",
#     pad=0.1,
#     label="frequency of Ze > 1e-3 / %",
#     #location="top",
#     shrink=0.75,
# )

ax0_twin = ax[0].twinx()
color_ful = "teal"
ful_binned_iwv.plot(ax=ax0_twin, color=color_ful)
ax0_twin.set_ylabel("longwave radiation flux / W m$^{-2}$", color=color_ful)
ax0_twin.tick_params(axis="y", colors=color_ful)
ax0_twin.set_ylim(ymin=180)

cbar_cwv = (
    (ds_ds["rh"] * 100)
    .groupby_bins(ds_ds["iwv"], bins=bins)
    .mean()
    .plot(y="altitude", ax=ax[1], add_colorbar=False, cmap="Blues", vmin=20, vmax=100)
)


ax[0].set_xlim(cwv_xmin, cwv_xmax)
ax[0].set_ylim(ymin=0)
ax[0].set_ylabel("height / m")
ax[0].set_xlabel(" IWV / mm")

cols = ["#F2935C", "steelblue"]

for i_bounds, bounds in enumerate([(0, 55), (55, 100)]):

    if bounds == (0, 55):
        label = f"IWV < {bounds[1]} mm"
    else:
        label = f"IWV $\geq$ {bounds[0]} mm"

    ds_ds["wvel"].where(
        (ds_ds["iwv_mean"] >= bounds[0]) & (ds_ds["iwv_mean"] < bounds[1]), drop=True
    ).mean("circle_id").plot.line(
        y="altitude",
        ax=ax[1],
        label=label,
        color=cols[i_bounds],
    )

ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
ax[1].axvline(0, color="k", alpha=0.5, linestyle=":")
ax[1].set_xlim(-0.01, 0.03)
ax[1].set_ylim(0, 12.5e3)
ax[1].set_xlabel("vertical velocity / m s$^{-1}$")
ax[1].set_ylabel(" ")
sns.despine()

plt.savefig(
    f"{PROJECT_ROOT}/figures/itcz_cwv_space.pdf",
    bbox_inches="tight",
)
# %%
