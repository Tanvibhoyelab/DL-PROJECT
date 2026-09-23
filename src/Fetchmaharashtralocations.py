"""One-time fetch of every city/town in each Maharashtra district from
OpenStreetMap (via the Overpass API), saved as JSON.

Run this locally (needs internet):

    python fetch_maharashtra_locations.py

It writes data/maharashtra_locations.json, grouped by district. Put that
file in your project's data/ folder — src/india_locations.py will load it
automatically the next time the dashboard starts, and it will replace the
hand-typed Maharashtra locality lists with this fuller, auto-fetched set.

NOTE ON SCOPE: this fetches place=city and place=town only. Maharashtra
has roughly 40,000 villages (place=village / place=hamlet) — including
those would make the dropdown unusably long. Villages remain reachable
through the dashboard's existing free-text "Search any place in India"
mode, which works for any named place regardless of this list.

If a district returns zero results (uncommon, but Overpass district
boundary names occasionally differ slightly from the official name),
the script prints a warning so you can adjust the name and re-run just
that district if you want it filled in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

MAHARASHTRA_DISTRICTS = [
    "Mumbai City", "Mumbai Suburban", "Thane", "Palghar", "Raigad", "Ratnagiri",
    "Sindhudurg", "Pune", "Satara", "Sangli", "Solapur", "Kolhapur",
    "Nashik", "Dhule", "Nandurbar", "Jalgaon", "Ahmednagar",
    "Aurangabad", "Jalna", "Parbhani", "Hingoli", "Nanded", "Beed",
    "Latur", "Osmanabad", "Amravati", "Akola", "Washim", "Buldhana",
    "Yavatmal", "Nagpur", "Wardha", "Bhandara", "Gondia", "Chandrapur",
    "Gadchiroli",
]

QUERY_TEMPLATE = """
[out:json][timeout:180];
area["name"="Maharashtra"]["admin_level"="4"]->.mh;
area["name"="{district}"]["admin_level"="5"](area.mh)->.d;
(
  node["place"~"^(city|town)$"](area.d);
);
out tags;
"""


def fetch_district(district: str) -> list[str]:
    query = QUERY_TEMPLATE.format(district=district)
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=200)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])
    names = sorted({
        el["tags"]["name"]
        for el in elements
        if "tags" in el and "name" in el["tags"]
    })
    return names


def main() -> None:
    result: dict[str, list[str]] = {}
    for district in MAHARASHTRA_DISTRICTS:
        print(f"Fetching {district} ...")
        try:
            names = fetch_district(district)
        except requests.RequestException as exc:
            print(f"  ! Failed for {district}: {exc}")
            names = []

        if not names:
            print(f"  ! No results for {district} — check the district name spelling.")
        else:
            print(f"  {len(names)} places found.")

        result[district] = names

        # Be polite to the free public Overpass endpoint.
        time.sleep(2)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "maharashtra_locations.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path} ({sum(len(v) for v in result.values())} places total).")


if __name__ == "__main__":
    main()