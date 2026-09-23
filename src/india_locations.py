"""India location hierarchy + geocoding.

The built-in dictionary is only a convenience starting point. Any place in
India can be reached through the free-text search or manual lat/lon entry.
"""

from __future__ import annotations

import functools
import time
from typing import Optional, Tuple

from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from config import NOMINATIM_USER_AGENT

# state -> district/city -> list of localities
INDIA_LOCATIONS: dict[str, dict[str, list[str]]] = {
    "Maharashtra": {
        "Mumbai": ["Bandra", "Andheri", "Dadar", "Powai", "Colaba", "Borivali", "Malad",
                   "Kurla", "Chembur", "Worli", "Juhu", "Ghatkopar", "Santacruz",
                   "Goregaon", "Mulund", "Byculla", "Mahim", "Vikhroli"],
        "Navi Mumbai": ["Vashi", "Nerul", "Belapur", "Kharghar", "Airoli", "Kopar Khairane",
                        "Panvel", "Ulwe", "Kamothe", "Turbhe", "Ghansoli", "Seawoods"],
        "Thane": ["Thane West", "Thane East", "Ghodbunder Road", "Kalwa", "Mumbra",
                  "Bhiwandi", "Kalyan", "Dombivli", "Vasai", "Virar", "Ambernath",
                  "Ulhasnagar", "Badlapur"],
        "Pune": ["Hinjewadi", "Kothrud", "Hadapsar", "Wakad", "Baner", "Kharadi",
                 "Viman Nagar", "Aundh", "Katraj", "Pimpri-Chinchwad", "Magarpatta",
                 "Camp", "Shivajinagar"],
        "Nagpur": ["Dharampeth", "Sitabuldi", "Wardhaman Nagar", "Civil Lines", "Sadar",
                   "Manish Nagar", "Hingna", "Kamptee"],
        "Nashik": ["Panchavati", "Gangapur Road", "Satpur", "Cidco", "Deolali",
                   "Indira Nagar", "College Road", "Nashik Road"],
        "Ahmednagar": ["Ahmednagar city", "Shirdi", "Sangamner", "Kopargaon", "Shrirampur"],
        "Akola": ["Akola city", "Akot", "Balapur", "Murtizapur"],
        "Amravati": ["Amravati city", "Achalpur", "Chandur Railway", "Daryapur"],
        "Beed": ["Beed city", "Georai", "Ambajogai", "Parli"],
        "Bhandara": ["Bhandara city", "Tumsar", "Pauni"],
        "Buldhana": ["Buldhana city", "Chikhli", "Khamgaon", "Malkapur"],
        "Chandrapur": ["Chandrapur city", "Ballarpur", "Warora", "Rajura"],
        "Chhatrapati Sambhajinagar (Aurangabad)": ["Aurangabad city", "Paithan", "Vaijapur", "Kannad"],
        "Dhule": ["Dhule city", "Shirpur", "Sindkheda"],
        "Gadchiroli": ["Gadchiroli town", "Aheri", "Desaiganj"],
        "Gondia": ["Gondia city", "Tirora", "Amgaon"],
        "Hingoli": ["Hingoli town", "Kalamnuri", "Basmath"],
        "Jalgaon": ["Jalgaon city", "Bhusawal", "Chalisgaon", "Amalner"],
        "Jalna": ["Jalna city", "Ambad", "Partur"],
        "Kolhapur": ["Kolhapur city", "Ichalkaranji", "Kagal", "Gadhinglaj"],
        "Latur": ["Latur city", "Udgir", "Ausa"],
        "Nanded": ["Nanded city", "Deglur", "Kinwat"],
        "Nandurbar": ["Nandurbar town", "Shahada", "Taloda"],
        "Osmanabad (Dharashiv)": ["Osmanabad town", "Tuljapur", "Umarga"],
        "Palghar": ["Palghar town", "Boisar", "Vasai-Virar", "Dahanu", "Jawhar"],
        "Parbhani": ["Parbhani city", "Gangakhed", "Jintur"],
        "Raigad": ["Alibaug", "Panvel", "Pen", "Mahad", "Khopoli"],
        "Ratnagiri": ["Ratnagiri city", "Chiplun", "Guhagar", "Dapoli"],
        "Sangli": ["Sangli city", "Miraj", "Tasgaon", "Islampur"],
        "Satara": ["Satara city", "Karad", "Mahabaleshwar", "Wai"],
        "Sindhudurg": ["Sindhudurg Nagari", "Kudal", "Malvan", "Vengurla"],
        "Solapur": ["Solapur city", "Pandharpur", "Barshi", "Akkalkot"],
        "Wardha": ["Wardha city", "Hinganghat", "Arvi"],
        "Washim": ["Washim city", "Risod", "Karanja"],
        "Yavatmal": ["Yavatmal city", "Pusad", "Wani", "Ghatanji"],
    },
    "Karnataka": {
        "Bengaluru Urban": ["Whitefield", "Electronic City", "Koramangala", "Hebbal", "Yelahanka"],
        "Mysuru": ["Vijayanagar", "Hebbal Mysuru", "Kuvempunagar"],
        "Mangaluru": ["Hampankatta", "Surathkal"],
    },
    "Gujarat": {
        "Ahmedabad": ["Bopal", "Maninagar", "Sarkhej", "Chandkheda"],
        "Surat": ["Adajan", "Vesu", "Katargam"],
        "Gandhinagar": ["Sector 21", "Kudasan", "GIFT City"],
    },
    "Delhi": {
        "New Delhi": ["Connaught Place", "Dwarka", "Rohini", "Saket", "Narela"],
    },
    "Tamil Nadu": {
        "Chennai": ["Adyar", "Velachery", "Sholinganallur", "Anna Nagar", "Tambaram"],
        "Coimbatore": ["Peelamedu", "Saravanampatti"],
        "Madurai": ["Anna Nagar Madurai", "Thiruparankundram"],
    },
    "West Bengal": {
        "Kolkata": ["Salt Lake", "New Town", "Howrah", "Behala"],
        "Siliguri": ["Matigara", "Bagdogra"],
    },
    "Telangana": {
        "Hyderabad": ["Gachibowli", "Hitec City", "Kukatpally", "Shamshabad"],
    },
    "Uttar Pradesh": {
        "Lucknow": ["Gomti Nagar", "Hazratganj"],
        "Noida": ["Sector 62", "Greater Noida", "Sector 137"],
        "Varanasi": ["Sarnath", "Lanka"],
    },
    "Rajasthan": {
        "Jaipur": ["Mansarovar", "Vaishali Nagar", "Sitapura"],
        "Jodhpur": ["Ratanada", "Shastri Nagar"],
    },
    "Kerala": {
        "Ernakulam": ["Kakkanad", "Fort Kochi"],
        "Thiruvananthapuram": ["Technopark", "Kazhakkoottam"],
    },
    "Punjab": {"Ludhiana": ["Model Town"], "Amritsar": ["Ranjit Avenue"]},
    "Madhya Pradesh": {"Indore": ["Vijay Nagar", "Rau"], "Bhopal": ["Arera Colony"]},
    "Bihar": {"Patna": ["Boring Road", "Danapur"]},
    "Odisha": {"Khordha": ["Bhubaneswar", "Patia"]},
    "Assam": {"Kamrup Metropolitan": ["Guwahati", "Beltola"]},
    "Haryana": {"Gurugram": ["Cyber City", "Sohna Road"], "Faridabad": ["Sector 21"]},
    "Andhra Pradesh": {"Visakhapatnam": ["Madhurawada"], "Guntur": ["Amaravati"]},
}

STATES = sorted(INDIA_LOCATIONS.keys())


def _load_fetched_maharashtra() -> None:
    """If data/maharashtra_locations.json exists (produced by
    fetch_maharashtra_locations.py), use it to replace the hand-typed
    Maharashtra district lists with the fuller, auto-fetched set.
    Falls back silently to the built-in list above if the file is
    missing, empty, or malformed — the dashboard always has *something*
    to show either way.
    """
    import json
    from pathlib import Path

    data_path = Path(__file__).resolve().parent.parent / "data" / "maharashtra_locations.json"
    if not data_path.is_file():
        return

    try:
        fetched = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    if not isinstance(fetched, dict) or not fetched:
        return

    # Only replace districts that actually got results — keep the
    # hand-typed fallback for any district Overpass returned nothing for.
    cleaned = {
        district: sorted(set(places))
        for district, places in fetched.items()
        if isinstance(places, list) and places
    }
    if cleaned:
        INDIA_LOCATIONS["Maharashtra"] = cleaned


_load_fetched_maharashtra()


def districts_of(state: str) -> list[str]:
    return sorted(INDIA_LOCATIONS.get(state, {}).keys())


def areas_of(state: str, district: str) -> list[str]:
    return sorted(INDIA_LOCATIONS.get(state, {}).get(district, []))


class GeocodingError(RuntimeError):
    pass


@functools.lru_cache(maxsize=512)
def geocode(query: str) -> Tuple[float, float, str]:
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


def build_query(state: str, district: str, area: str = "", extra: str = "") -> str:
    parts = [p.strip() for p in (extra, area, district, state, "India") if p and p.strip()]
    # de-duplicate while preserving order
    seen, out = set(), []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return ", ".join(out)


def resolve_location(state: str, district: str, area: str = "", extra: str = ""):
    """Geocode the most specific query first, then progressively fall back."""
    attempts = []
    if extra:
        attempts.append(build_query(state, district, area, extra))
    if area:
        attempts.append(build_query(state, district, area))
    attempts.append(build_query(state, district))
    attempts.append(build_query(state, ""))

    errors = []
    for q in attempts:
        try:
            lat, lon, display = geocode(q)
            return {"lat": lat, "lon": lon, "display_name": display, "query": q}
        except GeocodingError as exc:
            errors.append(str(exc))
    raise GeocodingError(" | ".join(errors))