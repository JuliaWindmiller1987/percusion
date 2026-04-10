# %%

from orcestra import get_flight_segments
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import percusion
from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%
flights = get_flight_segments()["HALO"]
ds_hamp = xr.open_dataset(
    "ipfs://bafybeigmd3dovwm45ylfqxnn2jphsrdjl2jt3dfytv7grkyhleaq42jthe", engine="zarr"
)

ds_meteor = xr.open_dataset(
    "ipfs://bafybeib5awa3le6nxi4rgepn2mwxj733aazpkmgtcpa3uc2744gxv7op44", engine="zarr"
)

# %%

flight_id = "HALO-20240903a"
flight = flights[flight_id]

flight_start, flight_end = flight["takeoff"], flight["landing"]
# %%

ds_hamp_flight = ds_hamp.sel(time=slice(flight_start, flight_end))
ds_hamp_flight["Ze"].plot(norm=colors.LogNorm(), x="time")
# %%

ec_underpass_events = [e for e in flight["events"] if "ec_underpass" in e["kinds"]]
metero_overpass_events = [
    e for e in flight["events"] if "meteor_overpass" in e["kinds"]
]
# %%

time_delta = np.timedelta64(20, "m")
ec_start, ec_end = ec_underpass_events[0]["time"] - np.timedelta64(
    20, "m"
), ec_underpass_events[0]["time"] + np.timedelta64(20, "m")

ds_hamp_ec_underpass = ds_hamp_flight.sel(time=slice(ec_start, ec_end))

fig, ax = plt.subplots(figsize=(10, 3))
ds_hamp_ec_underpass["Zg"].plot(
    norm=colors.LogNorm(),
    alpha=0.9,
    x="time",
    vmin=1e-3,
    vmax=1e3,
    ax=ax,
    cmap="YlGnBu",
)
plt.axvline(ec_underpass_events[0]["time"], color="C0", label="EC Underpass")

for event in metero_overpass_events:
    plt.axvline(event["time"], color="C1", label="METEOR Overpass")
    print(f"METEOR Overpass at {event['time']}")

plt.ylim(ymin=0)
sns.despine()

plt.savefig(f"{PROJECT_ROOT}/figures/mse_radiation.pdf", bbox_inches="tight")
# %%

ds_meteor_ec_underpass = ds_meteor.sel(time=slice(ec_start, ec_end))

num_dship_ts = len(ds_meteor_ec_underpass["time"])

sst = np.empty((num_dship_ts, 2))
sst[:, 0] = ds_meteor_ec_underpass["sst_extern_board"].values
sst[:, 1] = ds_meteor_ec_underpass["sst_extern_port"].values
sst_mean = np.mean(sst, axis=1)

t_air = np.empty((num_dship_ts, 2))
t_air[:, 0] = ds_meteor_ec_underpass["t_air_board"].values
t_air[:, 1] = ds_meteor_ec_underpass["t_air_port"].values
t_air_mean = np.mean(t_air, axis=1)

wspd = ds_meteor_ec_underpass["wspd"].values

# Need to check this!!!

rho = 1.2  # kg/m^3
cp = 1004  # J/kg/K
CH = 1.2e-3  # typical neutral value

sensible_heat_flux = rho * cp * CH * wspd * (sst_mean - t_air_mean)  # W/m^2
# %%

rh = np.empty((num_dship_ts, 2))
rh[:, 0] = ds_meteor_ec_underpass["rh_board"].values
rh[:, 1] = ds_meteor_ec_underpass["rh_port"].values
rh_mean = np.mean(rh, axis=1)


def saturation_vapor_pressure(T):
    # T in °C, returns hPa
    return 6.112 * np.exp(17.67 * T / (T + 243.5))


def specific_humidity(e, p):
    # e in hPa, p in hPa
    return 0.622 * e / p


def latent_heat_flux(Ts, Ta, RH, U, p=1013):
    # Ts, Ta in °C; RH in [0,1]; U in m/s

    es_Ts = saturation_vapor_pressure(Ts)
    es_Ta = saturation_vapor_pressure(Ta)

    qs = specific_humidity(es_Ts, p)
    qa = specific_humidity(RH * es_Ta, p)

    rho = 1.2
    Lv = 2.5e6
    CE = 1.2e-3

    QE = rho * Lv * CE * U * (qs - qa) / 1000

    print(qs - qa)

    return QE


latent_heat_flux_values = latent_heat_flux(sst_mean, t_air_mean, rh_mean / 100, wspd)
latent_heat_flux_values
# %%
