# %%
from orcestra import get_flight_segments
import importlib
import percusion.utils

importlib.reload(percusion.utils)
from percusion.utils import list_of_kinds
import pandas as pd

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

    flight_data.append(
        {
            "Flight ID": flight_id,
            "Month": month,
            "Day": day,
            "Takeoff": takeoff_time.strftime("%H:%M") if takeoff_time else None,
            "Landing": landing_time.strftime("%H:%M") if landing_time else None,
            "Duration": (
                str(flight_duration).split(".")[0] if flight_duration else None
            ),
            "Coordination": ", ".join(
                specified_events(flight_info, coordination_kinds)
            ),
            "# of circles": ", ".join(specified_events(flight_info, ["circle"])),
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
