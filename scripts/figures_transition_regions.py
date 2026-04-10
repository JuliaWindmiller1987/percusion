# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import percusion
from percusion import utils
from doldrumsVerticalMotion import circleUtils

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

iwv_hamp = ds_hamp["IWV"]
iwv_hamp = iwv_hamp.where((iwv_hamp > 0) & (iwv_hamp < 100))
iwv_hamp_orcestra = iwv_hamp.sel(time=slice(utils.campaign_start, utils.campaign_end))

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

print(
    f"Out of {len(ds_ds.circle_id)}, {len(transition_circles)} circles are transition circles."
)
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

bins = np.arange(30, 75, 0.8)

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
# %%
