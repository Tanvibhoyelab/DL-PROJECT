"""India location hierarchy, city lookup and geocoding.

Every state / union territory and every district of India is bundled offline in
``resources/india_districts.json``. Cities, towns and villages inside a district
are looked up live from OpenStreetMap (Overpass) the first time a district is
opened and then cached on disk, so the hierarchy covers the whole country
without shipping a multi-megabyte gazetteer.

Any place that is still missing can be reached through the free-text search or
by typing latitude / longitude directly.
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Optional

import requests
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from config import CITY_CACHE_DIR, INDIA_DISTRICTS_JSON, NOMINATIM_USER_AGENT

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 90


class GeocodingError(RuntimeError):
    pass


def _load_districts() -> dict[str, list[str]]:
    try:
        data = json.loads(Path(INDIA_DISTRICTS_JSON).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeocodingError(
            f"The bundled district list at {INDIA_DISTRICTS_JSON} is missing or unreadable: {exc}"
        ) from exc
    return {state: sorted(set(districts)) for state, districts in sorted(data.items())}


INDIA_DISTRICTS: dict[str, list[str]] = _load_districts()
STATES: list[str] = sorted(INDIA_DISTRICTS.keys())


def districts_of(state: str) -> list[str]:
    return INDIA_DISTRICTS.get(state, [])


# --------------------------------------------------------------------------
# Cities / towns inside a district (OpenStreetMap, cached)
# --------------------------------------------------------------------------
def _cache_path(state: str, district: str) -> Path:
    slug = f"{state}_{district}".lower().replace(" ", "-").replace("/", "-")
    return Path(CITY_CACHE_DIR) / f"{slug}.json"


def _overpass_places(state: str, district: str, include_villages: bool) -> list[dict]:
    place_filter = (
        "^(city|town|village)$" if include_villages else "^(city|town|suburb)$"
    )
    query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}];
    area["boundary"="administrative"]["name"="{state}"]["admin_level"="4"]->.state;
    (
      area["boundary"="administrative"]["name"~"^{district}",i]["admin_level"~"^(5|6)$"](area.state);
    )->.district;
    (
      node(area.district)["place"~"{place_filter}"]["name"];
    );
    out tags center 600;
    """
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=OVERPASS_TIMEOUT + 15,
        headers={"User-Agent": NOMINATIM_USER_AGENT},
    )
    response.raise_for_status()
    elements = response.json().get("elements", [])

    places: dict[str, dict] = {}
    for element in elements:
        tags = element.get("tags", {})
        name = tags.get("name:en") or tags.get("name")
        lat = element.get("lat", (element.get("center") or {}).get("lat"))
        lon = element.get("lon", (element.get("center") or {}).get("lon"))
        if not name or lat is None or lon is None:
            continue
        rank = {"city": 0, "town": 1, "suburb": 2, "village": 3}.get(tags.get("place", ""), 4)
        existing = places.get(name)
        if existing is None or rank < existing["rank"]:
            places[name] = {
                "name": name,
                "lat": float(lat),
                "lon": float(lon),
                "place": tags.get("place", ""),
                "rank": rank,
            }
    return sorted(places.values(), key=lambda p: (p["rank"], p["name"]))


def cities_of(state: str, district: str, include_villages: bool = False) -> list[dict]:
    """Return [{name, lat, lon, place}, ...] for a district.

    Results are cached on disk. An empty list means OpenStreetMap returned
    nothing for that district — the district itself can still be analysed.
    """
    cache = _cache_path(state, district)
    key = "with_villages" if include_villages else "cities_towns"
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if key in cached:
                return cached[key]
        except (OSError, json.JSONDecodeError):
            cached = {}
    else:
        cached = {}

    try:
        places = _overpass_places(state, district, include_villages)
    except requests.RequestException as exc:
        raise GeocodingError(
            "OpenStreetMap (Overpass) could not be reached for the city list of "
            f"{district}, {state}: {exc}. Use the search box or latitude/longitude instead."
        ) from exc

    cached[key] = places
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    return places


# --------------------------------------------------------------------------
# Geocoding
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1024)
def geocode(query: str) -> tuple[float, float, str]:
    """Return (lat, lon, display_name) for a free-text query inside India."""
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=15)
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            location = geolocator.geocode(
                query, country_codes="in", exactly_one=True, addressdetails=True
            )
            if location is None:
                raise GeocodingError(
                    f"No place in India matched '{query}'. Try a more specific name, "
                    "or enter latitude/longitude manually."
                )
            return float(location.latitude), float(location.longitude), location.address
        except (GeocoderTimedOut, GeocoderServiceError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise GeocodingError(
        "The geocoding service could not be reached. Check your internet connection "
        f"and try again. ({last_error})"
    )


@functools.lru_cache(maxsize=256)
def search(query: str, limit: int = 8) -> list[dict]:
    """Free-text search returning several candidate places inside India."""
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=15)
    try:
        matches = geolocator.geocode(
            query, country_codes="in", exactly_one=False, limit=limit, addressdetails=True
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        raise GeocodingError(f"The geocoding service could not be reached: {exc}") from exc
    if not matches:
        raise GeocodingError(
            f"No place in India matched '{query}'. Try a different spelling, add the "
            "district or state, or enter latitude/longitude manually."
        )
    return [
        {"lat": float(m.latitude), "lon": float(m.longitude), "display_name": m.address}
        for m in matches
    ]


def resolve_location(state: str, district: str, city: str = "") -> dict:
    """Geocode the most specific query first, then progressively fall back."""
    attempts = []
    if city:
        attempts.append(", ".join([city, district, state, "India"]))
        attempts.append(", ".join([city, state, "India"]))
    attempts.append(", ".join([district, state, "India"]))
    attempts.append(", ".join([state, "India"]))

    errors = []
    for query in attempts:
        try:
            lat, lon, display = geocode(query)
            return {"lat": lat, "lon": lon, "display_name": display, "query": query}
        except GeocodingError as exc:
            errors.append(str(exc))
    raise GeocodingError(" | ".join(errors))


def reverse(lat: float, lon: float) -> str:
    """Best-effort place name for a coordinate; falls back to the raw numbers."""
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=15)
    try:
        match = geolocator.reverse((lat, lon), exactly_one=True, language="en")
    except (GeocoderTimedOut, GeocoderServiceError):
        match = None
    return match.address if match else f"{lat:.5f}, {lon:.5f}"
