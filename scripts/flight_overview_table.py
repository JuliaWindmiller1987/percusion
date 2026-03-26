# %%
from orcestra import get_flight_segments
import importlib
import percusion.utils

importlib.reload(percusion.utils)
from percusion.utils import list_of_kinds, get_halo_position
import pandas as pd
import numpy as np


# %%

flights = get_flight_segments()["HALO"]
flights.pop("HALO-20240809b", None)
flights.pop("HALO-20240929a", None)
flight_ids = list(flights.keys())

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
    return list_counted


def specified_events(flight_info, specified_kinds):
    list_of_spec_kinds = []
    for e in return_as_list(flight_info["events"]):
        if e in specified_kinds:
            list_of_spec_kinds.append(e)

    for s in return_as_list(flight_info["segments"]):
        if s in specified_kinds:
            list_of_spec_kinds.append(s)
    return sorted(count_events(abbreviate_kinds(list_of_spec_kinds)))


def get_takeoff_landing_location(flight_info, ds):
    dict_of_locations = {"takeoff": None, "landing": None}
    for s in dict_of_locations.keys():
        rel_time = flight_info.get(s)
        time_slice = (
            slice(rel_time - pd.Timedelta(hours=1), rel_time)
            if s == "landing"
            else slice(rel_time, rel_time + pd.Timedelta(hours=1))
        )
        position = ds.sel(time=time_slice)

        lon = int(position.lon.mean("time").values)
        loc = "SAL" if lon > -30 else "BB"
        dict_of_locations[s] = loc

    return dict_of_locations


# %%
# Extract flight dates from metadata
flight_data = []

for flight_id, flight_info in flights.items():

    date_obj = pd.to_datetime(flight_info.get("date"))

    month = date_obj.month
    day = date_obj.day

    takeoff_time = flight_info.get("takeoff")
    landing_time = flight_info.get("landing")
    flight_duration = landing_time - takeoff_time

    start_and_end_locations = get_takeoff_landing_location(flight_info, ds)

    flight_data.append(
        {
            "Flight ID": flight_id,
            "Date (MM-DD)": f"{month:02d}-{day:02d}",
            "Takeoff": f"{takeoff_time.strftime("%H:%M")} ({start_and_end_locations['takeoff']})",
            "Landing": f"{landing_time.strftime("%H:%M")} ({start_and_end_locations['landing']})",
            "Duration": (
                str(flight_duration).split(".")[0][:-3] if flight_duration else None
            ),
            "Coordination": ", ".join(
                specified_events(flight_info, coordination_kinds)
            ),
            "Circles": ", ".join(specified_events(flight_info, ["circle"])),
            # "Segments": ", ".join(s["name"] for s in flight_info["segments"]),
        }
    )

    # print(", ".join(specified_events(flight_info, coordination_kinds)), flight_id)


# Create DataFrame
df = pd.DataFrame(flight_data)

# Convert to LaTeX table
latex_table = df.to_latex(index=False, caption="HALO Flights", label="tab:flights")
print(latex_table)
# %%
