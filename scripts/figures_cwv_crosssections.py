# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import percusion
from percusion import utils
import pandas as pd
import math

from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]
from orcestra import get_flight_segments
import xarray as xr

# %%

import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

G = 9.81
THETA_E0 = 340.0  # K, only for expressing B_L in K


ds_sondes = xr.open_dataset(
    "ipfs://bafybeidytxgujp3evqesvvuvc3vsllv6pbcu5gc2torqkz7yo6yuazwtmu", engine="zarr"
)


def layer_weights(dp_pbl, dp_lft, a=0.5, b=0.5):
    """Deep-inflow layer weights (Ahmed et al. 2020)."""
    w_pbl = (
        (a * dp_pbl) / (b * dp_lft) * np.log((a * dp_pbl + b * dp_lft) / (a * dp_pbl))
    )
    return w_pbl, 1.0 - w_pbl


def buoyancy_one_sonde(p_hpa, ta_k, q_kgkg, dp_pbl=150.0, p_top=500.0, dp=5.0):
    """B_L, CAPE_L and SUBSAT_L (in K) for a single profile.

    p_hpa, ta_k, q_kgkg: 1-D arrays on any vertical grid (e.g. BEACH altitude grid).
    """
    ok = np.isfinite(p_hpa) & np.isfinite(ta_k) & np.isfinite(q_kgkg)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan
    p, t, q = p_hpa[ok], ta_k[ok], q_kgkg[ok]
    order = np.argsort(p)  # np.interp needs increasing x
    p, t, q = p[order], t[order], q[order]

    p_s = p.max()  # lowest measured level as surface pressure
    p_pbl_top = p_s - dp_pbl
    if p.min() > p_top or p_s < 990:  # profile must reach 500 hPa and near the surface
        return np.nan, np.nan, np.nan

    p_grid = np.arange(p_top, p_s + 1e-6, dp)  # regular in p -> mass-weighted means
    t_g = np.interp(p_grid, p, t)
    q_g = np.interp(p_grid, p, q)

    P = p_grid * units.hPa
    T = t_g * units.K
    Td = mpcalc.dewpoint_from_specific_humidity(P, T, q_g * units("kg/kg"))
    theta_e = mpcalc.equivalent_potential_temperature(P, T, Td).m_as("K")
    theta_es = mpcalc.saturation_equivalent_potential_temperature(P, T).m_as("K")

    in_pbl = p_grid >= p_pbl_top
    in_lft = (p_grid < p_pbl_top) & (p_grid >= p_top)

    the_pbl = theta_e[in_pbl].mean()
    the_lft = theta_e[in_lft].mean()
    thes_lft = theta_es[in_lft].mean()

    w_pbl, w_lft = layer_weights(dp_pbl, p_pbl_top - p_top)

    # in m s^-2 ...
    cape_l = G * w_pbl * (the_pbl - thes_lft) / thes_lft
    subsat_l = (
        G * w_lft * (thes_lft - the_lft) / thes_lft
    )  # positive = more subsaturated
    b_l = cape_l - subsat_l
    # ... converted to K
    to_k = THETA_E0 / G
    return b_l * to_k, cape_l * to_k, subsat_l * to_k


def buoyancy_dataset(ds, p="p", ta="ta", q="q", sonde_dim="sonde"):
    """Apply to all sondes; returns a Dataset with B_L, CAPE_L, SUBSAT_L per sonde."""
    out = np.array(
        [
            buoyancy_one_sonde(
                ds[p].isel({sonde_dim: i}).values / 100.0,  # Pa -> hPa
                ds[ta].isel({sonde_dim: i}).values,
                ds[q].isel({sonde_dim: i}).values,
            )
            for i in range(ds.sizes[sonde_dim])
        ]
    )
    return xr.Dataset(
        {
            "B_L": (sonde_dim, out[:, 0]),
            "CAPE_L": (sonde_dim, out[:, 1]),
            "SUBSAT_L": (sonde_dim, out[:, 2]),
        },
        coords={sonde_dim: ds[sonde_dim]},
    ).assign(B_L=lambda x: x.B_L.assign_attrs(units="K"))


# %%


ds_hamp = xr.open_dataset(
    "ipfs://bafybeihvsq73hokuex44oueqvnotaeoikwcs5yyiydtc777td2ldcm6z4y", engine="zarr"
)

for var in ["TLWP", "LWP", "IWP"]:
    ds_hamp[var] = ds_hamp[var] / 1000.0  # g m-2 -> kg m-2
    ds_hamp[var].attrs["units"] = "kg m-2"

ds_hamp["TILWP"] = ds_hamp["TLWP"] + ds_hamp["IWP"]
ds_hamp["TILWP"].attrs["long_name"] = "IWP + LWP"
ds_hamp["TILWP"].attrs["units"] = "kg m-2"

# %%

flights = get_flight_segments()["HALO"]

# %%

# Dropsonde data
sondes_ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)

bl = buoyancy_dataset(sondes_ds)
sondes_ds["BL"] = bl.B_L
launch_time = sondes_ds.launch_time

# %%

tilwp_at_launch_time = ds_hamp.sel(time=launch_time, method="nearest")["TILWP"]
sondes_ds["TILWP"] = tilwp_at_launch_time

tilwp_at_launch_time = ds_hamp.sel(time=launch_time, method="nearest")["LWP"]
sondes_ds["LWP"] = tilwp_at_launch_time


# %%
# --- settings ---------------------------------------------------------------

east = True  # True for the eastern part (Sal), False for the western part (Barbados)

start_date = "2024-08-11" if east else "2024-09-07"
end_date = "2024-09-06" if east else "2024-09-28"

variables = ["IWV", "TILWP"]  # first one goes on the left axis, the others on the right

plot_settings = {
    "IWV": {
        "label": "CWV / mm",
        "color": "tab:blue",
        "light": "lightsteelblue",
        "ref": 48,
        "ref_label": "CWV = 48 mm",
    },
    "TILWP": {
        "label": "IWP + LWP / kg m$^{-2}$",
        "color": "tab:orange",
        "light": "navajowhite",
    },
}

lat_bins = np.arange(-2, 18, 0.1)
lat_bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
min_count_mean = 1000
min_count_flight = 100


def bin_by_lat(ds, variables, bins, labels, min_count):
    """Mean of each variable in latitude bins, masked per variable by its own count."""
    grouped = ds[variables].groupby_bins(ds.lat, bins=bins, labels=labels)
    return grouped.mean().where(grouped.count() >= min_count)


def make_axes(ax, variables):
    """Return one axis per variable: ax for the first, twin axes for the rest."""
    var_axes = {variables[0]: ax}
    for k, var in enumerate(variables[1:]):
        ax.set_ylim(40, 64)
        twin = ax.twinx()
        twin.set_ylim(0, 3)
        if k > 0:  # offset further right axes so they don't overlap
            twin.spines["right"].set_position(("axes", 1 + 0.15 * k))
        var_axes[var] = twin
    for var, a in var_axes.items():
        c = plot_settings[var]["color"]
        a.tick_params(axis="y", colors=c)
        a.yaxis.label.set_color(c)
    return var_axes


def add_reference_line(ax, var, with_label=True):
    s = plot_settings[var]
    if "ref" in s:
        ax.axhline(
            s["ref"],
            color=s["color"],
            ls="--",
            lw=0.8,
            label=s.get("ref_label") if with_label else None,
        )


# --- flight days and binning -------------------------------------------------
start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
flight_days = sorted(
    (pd.Timestamp(info["date"]), flight_id)
    for flight_id, info in flights.items()
    if start <= pd.Timestamp(info["date"]) <= end
)

binned_flights = {}
for day, flight_id in flight_days:
    hamp_day = ds_hamp.sel(time=day.strftime("%Y-%m-%d"))
    sondes_day = sondes_ds.sel(launch_time=day.strftime("%Y-%m-%d"))
    if hamp_day.time.size == 0:
        continue
    binned_flights[flight_id] = bin_by_lat(
        hamp_day, variables, lat_bins, lat_bin_centers, min_count_flight
    )
    binned_flights[flight_id]["sondes"] = sondes_day.launch_lat.values

hamp_period = ds_hamp.sel(time=slice(start_date, end_date))
binned_mean = bin_by_lat(
    hamp_period, variables, lat_bins, lat_bin_centers, min_count_mean
)

# %%

# --- figure 1: mean + all flights -------------------------------------------
fig, ax = plt.subplots(
    figsize=(4.5, 3), constrained_layout=True, sharex=True, sharey=True
)
var_axes = make_axes(ax, variables)

handles = []
for var, a in var_axes.items():
    s = plot_settings[var]
    # for binned_day in binned_flights.values():
    #     a.plot(binned_day.lat_bins, binned_day[var], color=s["light"], lw=1, zorder=1)
    (h,) = a.plot(
        binned_mean.lat_bins,
        binned_mean[var],
        color=s["color"],
        lw=1,
        zorder=2,
        label=f"{var} mean",
    )
    handles.append(h)
    a.set_ylabel(s["label"], color=s["color"])
    add_reference_line(a, var, with_label=True)

sns.despine()

var_axes[variables[0]].set_xlabel("latitude / °N")

# %%

# --- figure 2: one panel per flight -----------------------------------------
n_flights = len(binned_flights)
ncols = 4
nrows = math.ceil(n_flights / ncols)

fig2, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(3.2 * ncols, 2.4 * nrows),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    squeeze=False,
)

axes[0, 0].set_ylim(35, 68)


shared_twins = {}  # so the right-hand axes are also shared across panels
for idx, (ax, (flight_id, binned_day)) in enumerate(
    zip(axes.flat, binned_flights.items())
):
    var_axes = make_axes(ax, variables)
    col = idx % ncols
    last_in_row = col == ncols - 1 or idx == n_flights - 1

    for var, a in var_axes.items():
        s = plot_settings[var]
        if a is not ax:
            if var in shared_twins:
                a.sharey(shared_twins[var])
                a.set_ylim(0, 10)
            else:
                shared_twins[var] = a
            if not last_in_row:
                a.tick_params(labelright=False)
            else:
                a.set_ylabel(s["label"])
        # a.plot(binned_mean.lat_bins, binned_mean[var], color=s["light"], lw=2)
        a.plot(binned_day.lat_bins, binned_day[var], color=s["color"], lw=1.2)
        add_reference_line(a, var, with_label=False)

    ax.set_title(flight_id, fontsize=9)

for ax in axes.flat[n_flights:]:
    ax.set_axis_off()
for ax in axes[-1, :]:
    ax.set_xlabel("latitude / °N")
for ax in axes[:, 0]:
    ax.set_ylabel(
        plot_settings[variables[0]]["label"], color=plot_settings[variables[0]]["color"]
    )

sns.despine()

# %%

sondes_day = sondes_ds.sel(launch_time="2024-08-31")

lat_ranges = [(5, 13), (16, 25)]
cwv_threshold = 50  # mm

dic_var_mean = {}

for lat_min, lat_max in lat_ranges:

    sondes_sel = sondes_day.where(
        (sondes_day.launch_lat >= lat_min) & (sondes_day.launch_lat < lat_max)
    )

    sondes_sel = sondes_sel.where(sondes_sel.iwv > cwv_threshold, drop=True)

    print(len(sondes_sel.sonde.values))

    sondes_sel.rh.mean(dim="sonde").plot(y="altitude")  # %%

    print(sondes_sel.BL.mean(dim="sonde").values)

    dic_var_mean[f"{lat_min}"] = sondes_sel.rh.mean(dim="sonde").values


plt.figure(figsize=(12, 8))

plt.scatter(sondes_day.launch_lat, sondes_day.BL.values)
plt.ylim(-20, 5)
ax2 = plt.twinx(plt.gca())
ax2.scatter(sondes_day.launch_lat, sondes_day.iwv.values, color="tab:orange")
ax2.set_ylabel("IWV / mm", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(30, 70)


# %%

bl.B_L.plot.hist(bins=50, density=True)
# %%
plt.figure(figsize=(12, 8))
plt.scatter(
    sondes_ds.sel(launch_time=slice(start_date, end_date)).iwv,
    sondes_ds.sel(launch_time=slice(start_date, end_date)).BL,
    c=sondes_ds.sel(launch_time=slice(start_date, end_date)).launch_lat,
    s=20,
    vmin=5,
    vmax=15,
    # vmax = 100
)

plt.xlabel("IWV / mm")
plt.ylabel("B_L / K")
plt.colorbar(label="launch latitude / °N")
# %%

from xarray.groupers import BinGrouper, UniqueGrouper

dx_bins = 0.05
lat_bins = np.arange(-2, 18, dx_bins)
lat_bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
min_count = 10

iwv = (
    ds_hamp["IWV"]
    .assign_coords(lat=ds_hamp["lat"])
    .assign_coords(day=ds_hamp.time.dt.floor("D"))
)

# iwv = ds_hamp["IWV"].assign_coords(day=ds_hamp.time.dt.floor("D"))

grouped = iwv.groupby(
    lat=BinGrouper(bins=lat_bins, labels=lat_bin_centers),
    day=UniqueGrouper(),
)

iwv_mean = grouped.mean()
iwv_count = grouped.count()
iwv_daily = iwv_mean.where(iwv_count >= min_count)

from scipy.ndimage import gaussian_filter1d


def gaussian_smooth_nan(da, dim, sigma_deg, dx_deg):
    """Gaussian smoothing along `dim` that ignores NaNs (normalised convolution)."""
    sigma = sigma_deg / dx_deg  # in bins
    axis = da.get_axis_num(dim)
    values = da.values
    valid = np.isfinite(values)
    num = gaussian_filter1d(
        np.where(valid, values, 0.0), sigma, axis=axis, mode="constant"
    )
    den = gaussian_filter1d(valid.astype(float), sigma, axis=axis, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.3] = np.nan  # too little data within the kernel
    return da.copy(data=out)


def find_cwv_crit_crossings(iwv_da, threshold):
    """Find the latitudes where IWV crosses a threshold (e.g. 48 mm)."""
    iwv_da = iwv_da.dropna("lat_bins")
    iwv_values = iwv_da.values
    return iwv_da.isel(lat_bins=np.where(np.diff(np.sign(iwv_values - threshold)))[0])


def find_slopes(iwv_da, threshold=48, delta_lat=2, dx_bins=dx_bins):
    """Find the slopes of IWV at the crossings of a threshold (e.g. 48 mm)."""

    for day in iwv_da.day.values:
        iwv_day = iwv_da.sel(day=day)
        crossings = find_cwv_crit_crossings(iwv_day, threshold=threshold)
        for lat_bin in crossings.lat_bins.values:
            slope_max = 0
            for lat_edges in [
                (lat_bin - delta_lat, lat_bin),
                (lat_bin, lat_bin + delta_lat),
                (lat_bin - delta_lat / 2, lat_bin + delta_lat / 2),
            ]:
                iwv_slice = iwv_day.sel(lat_bins=slice(*lat_edges))
                if iwv_slice.count() < int(1 / dx_bins):
                    continue
                iwv_fit = iwv_slice.polyfit(dim="lat_bins", deg=1)
                slope = iwv_fit.polyfit_coefficients.sel(degree=1).values
                if np.abs(slope) > np.abs(slope_max):
                    slope_max = slope
            yield lat_bin, slope_max


iwv_smooth = gaussian_smooth_nan(iwv_daily, "lat_bins", sigma_deg=0.25, dx_deg=dx_bins)
lat_crossings, slopes = zip(*list(find_slopes(iwv_smooth, threshold=54)))

# %%
plt.figure()
for threshold in np.arange(45, 63, 1):
    lat_crossings, slopes = zip(
        *list(
            find_slopes(iwv_smooth, threshold=threshold, delta_lat=1, dx_bins=dx_bins)
        )
    )
    print(
        f"Threshold {threshold:.0f} mm: mean slope {np.nanmean(np.abs(slopes)):.2f} mm/°N"
    )
    print(
        f"Threshold {threshold:.0f} mm: median slope {np.nanmedian(np.abs(slopes)):.2f} mm/°N"
    )

    plt.scatter(threshold, np.nanmedian(np.abs(slopes)), color="tab:blue")

plt.xlabel("CWV threshold / mm")
plt.ylabel("Median absolute slope of IWV at crossings / mm/°N")

# %%
lat_crossings, slopes = zip(
    *list(find_slopes(iwv_smooth, threshold=48.0, delta_lat=1.5, dx_bins=dx_bins))
)

plt.scatter(lat_crossings, np.abs(slopes), color="tab:blue")
# %%

iwv_gradient = iwv_smooth.diff("lat_bins") / dx_bins

# %%
day = 2
iwv_smooth.isel(day=day).plot(x="lat_bins")
plt.axhline(48, color="k", lw=0.5, ls="--", zorder=0)

ax_twin = plt.twinx(plt.gca())
np.abs(iwv_gradient.isel(day=day)).plot(x="lat_bins", ax=ax_twin, color="tab:orange")
ax_twin.set_ylabel("Gradient of IWV / (mm/°N)", color="tab:orange")
ax_twin.tick_params(axis="y", labelcolor="tab:orange")

ax_twin.set_ylim(0, 10)
# %%

np.abs(iwv_gradient.isel(day=slice(1, 12))).groupby_bins(
    iwv_smooth.isel(day=slice(1, 12)).isel(lat_bins=slice(1, None)),
    bins=np.arange(44, 55, 0.5),
).mean().plot()

# %%
