# %%
from orcestra import get_flight_segments
import importlib
import percusion.utils

importlib.reload(percusion.utils)
from percusion.utils import list_of_kinds, get_halo_position
import pandas as pd
import numpy as np
import xarray as xr
import copy

# %%
import percusion
from percusion.datasets import open_dataset
from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%
dropsonde = open_dataset("dropsondes")

# %%

flights = get_flight_segments()["HALO"]
flights.pop("HALO-20240809b", None)
flights.pop("HALO-20240929a", None)
flight_ids = list(flights.keys())

flights_ds_circles = copy.deepcopy(flights)

for flight in flights.keys():

    for i_seg, seg in enumerate(flights[flight]["segments"]):
        if "circle" in seg["kinds"]:
            if seg["segment_id"] in dropsonde.circle_id.values:

                num_nan_wvel = np.sum(
                    np.isnan(
                        dropsonde.where(
                            dropsonde.circle_id == seg["segment_id"], drop=True
                        ).wvel.values[0]
                    )
                )

                if num_nan_wvel / dropsonde.altitude.size == 1.0:
                    print(
                        f"Flight {flight} has circle segment {seg['segment_id']} with no vertical motion data in dropsonde dataset."
                    )

                else:
                    flights_ds_circles[flight]["segments"][i_seg]["kinds"] = seg[
                        "kinds"
                    ] + ["omega_c"]
            else:
                print(
                    f"Flight {flight} has circle segment {seg['segment_id']} not in dropsonde dataset."
                )
                print(f"Segment name: {seg['name']}, Remarks: {seg['remarks']}")

flights = flights_ds_circles
# %%

events = [
    {**e, "flight_id": flight_id}
    for flight_id, flight in flights.items()
    for e in flight["events"]
]

coordination_kinds = ["atr_coordination"] + list_of_kinds(events)

# %%

ds = get_halo_position()

campaign_start = flights[flight_ids[0]]["takeoff"]
campaign_end = flights[flight_ids[-1]]["landing"]
ds = ds.sel(time=slice(campaign_start, campaign_end))
# %%


def return_as_list(x):
    kinds_list = []
    for item in x:
        kinds_list += (
            item["kinds"] if isinstance(item["kinds"], (list)) else [item["kinds"]]
        )
    return kinds_list


abbreviations = {
    "bco_overpass": "BCO",
    "cvao_overpass": "CVAO",
    "ec_underpass": "EC",
    "meteor_overpass": "METEOR",
    "pace_underpass": "PACE",
    "atr_coordination": "ATR",
    "circle": " ",
}


def abbreviate_kinds(kinds):
    return [abbreviations[k] if k in abbreviations else k for k in kinds]


def count_events(x):
    list_counted = []
    for item in set(x):
        count = x.count(item)
        if count > 1:
            item_counted = f"{item} ({count})" if item != " " else f"{count}"
        else:
            item_counted = item
        list_counted.append(item_counted)
    return sorted(list_counted)


def specified_events(flight_info, specified_kinds):
    list_of_spec_kinds = []
    for e in return_as_list(flight_info["events"]):
        if e in specified_kinds:
            list_of_spec_kinds.append(e)

    for s in return_as_list(flight_info["segments"]):
        if s in specified_kinds:
            list_of_spec_kinds.append(s)
    return list_of_spec_kinds  # count_events(abbreviate_kinds(list_of_spec_kinds))


def get_takeoff_landing_location(
    flight_info, ds, flight_id=None, window=pd.Timedelta(hours=1)
):
    """Return the take-off and landing airport (SAL or BB) for one flight.

    Uses the mean HALO longitude in the first/last hour of the flight. If BAHAMAS
    has no valid data in that window, falls back to the first/last valid data point
    of the flight.
    """
    takeoff, landing = flight_info["takeoff"], flight_info["landing"]
    flight_lon = ds.lon.sel(time=slice(takeoff, landing)).dropna("time")
    error_msg = None
    dict_of_locations = {}
    for s in ["takeoff", "landing"]:
        time_slice = (
            slice(takeoff, takeoff + window)
            if s == "takeoff"
            else slice(landing - window, landing)
        )
        lon = ds.lon.sel(time=time_slice).dropna("time")

        if lon.size == 0:
            if flight_lon.size == 0:
                print(f"{flight_id}: no valid BAHAMAS data for the whole flight.")
                dict_of_locations[s] = None
                continue

            phase, which = ("beginning", "first") if s == "takeoff" else ("end", "last")
            lon = flight_lon.isel(time=[0 if s == "takeoff" else -1])
            t_fallback = pd.Timestamp(lon.time.values[0])
            error_msg = f"Problem with BAHAMAS at the {phase} of the flight ({which} available data point at {t_fallback:%H:%M} UTC)."

        mean_lon = float(lon.mean())
        dict_of_locations[s] = "SAL" if mean_lon > -30 else "BB"

    return dict_of_locations, error_msg


def count_all_occurrences(flights, specified_kinds):
    all_occurrences = []
    for flight_id, flight_info in flights.items():
        all_occurrences.extend(specified_events(flight_info, specified_kinds))

    return count_events(abbreviate_kinds(all_occurrences))


# %%

print_occurrences = [
    "circle",
    "omega_c",
    "ec_underpass",
    "bco_overpass",
    "cvao_overpass",
    "meteor_overpass",
    "pace_underpass",
    "atr_coordination",
]

for kind in print_occurrences:
    print(f"{kind}: {count_all_occurrences(flights, [kind])}")


# %%
# Extract flight dates from metadata
flight_data = []


def url_string(flight_id):
    return f"\\href{{https://orcestra-campaign.org/reports/{flight_id}.html}}{{{flight_id}}}"


duration_list = []

for flight_id, flight_info in flights.items():

    print(flight_id)

    date_obj = pd.to_datetime(flight_info.get("date"))

    month = date_obj.month
    day = date_obj.day

    takeoff_time = flight_info.get("takeoff")
    landing_time = flight_info.get("landing")
    flight_duration = landing_time.replace(
        second=0, microsecond=0
    ) - takeoff_time.replace(second=0, microsecond=0)
    duration_min = int(flight_duration.total_seconds() // 60)

    duration_list.append(duration_min)

    start_and_end_locations, error_msg = get_takeoff_landing_location(flight_info, ds)

    if error_msg not in [None, ""]:
        print(f"Warning for flight {flight_id}: {error_msg}")

    n_circles = len(specified_events(flight_info, ["circle"]))
    n_ds_circles = len(specified_events(flight_info, ["omega_c"]))

    flight_data.append(
        {
            "Flight ID": (
                url_string(flight_id)
                if flight_id not in ["HALO-20240822a"]
                else flight_id
            ),
            "Date": f"{month:02d}-{day:02d}",
            "Take-off": f"{takeoff_time.strftime("%H:%M")} ({start_and_end_locations['takeoff']})",
            "Landing": f"{landing_time.strftime("%H:%M")} ({start_and_end_locations['landing']})",
            "Duration": f"{duration_min // 60}:{duration_min % 60:02d}",
            "Coordination": ", ".join(
                count_events(
                    abbreviate_kinds(specified_events(flight_info, coordination_kinds))
                )
            ),
            "Circles": f"{n_circles} ({n_ds_circles})",
            # "Segments": ", ".join(s["name"] for s in flight_info["segments"]),
        }
    )

    # print(", ".join(specified_events(flight_info, coordination_kinds)), flight_id)

print(
    f"Median flight duration: {np.median(duration_list) // 60:.0f}:{int(np.median(duration_list) % 60):02d}"
)

# Create DataFrame
df = pd.DataFrame(flight_data)


caption = (
    "Overview of HALO flights during the campaign. "
    "Dates are given as month-day (MM-DD) in 2024. "
    "Takeoff and landing times are in UTC (HH:MM), with the airport in brackets "
    "(SAL: Sal, Cabo Verde; BB: Barbados), and durations are in hours and minutes (h:min). "
    "The \\textit{Coordination} column lists the platforms coordinated with during each flight. "
    "Numbers in brackets indicate repeated coordination events with the same platform. "
    "The \\textit{Circles} column shows how many circles were flown during the flight. "
    "The number of circles for which vertical motion was calculated is given in brackets."
)

# Convert to LaTeX table
latex_table = df.to_latex(index=False, caption=caption, label="tab:flights")
latex_table = latex_table.replace(r"\toprule", "")
latex_table = latex_table.replace(r"\bottomrule", "")
latex_table = latex_table.replace(r"\midrule", "")
print(latex_table)

with open(PROJECT_ROOT / "tables" / "flight_overview_table.tex", "w") as f:
    f.write(latex_table)
# %%
