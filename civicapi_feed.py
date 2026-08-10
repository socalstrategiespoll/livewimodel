"""
civicAPI live feed for the Wisconsin model.

Endpoint:  https://civicapi.org/api/v2/race/{race_id}
Race:      85787  (2026 Wisconsin Governor Democratic Primary)
Auth:      none. Attribution required for non-personal use, so credit civicapi.org
           anywhere this output is published.

WHAT'S DIFFERENT FROM THE MICHIGAN FEED

Wisconsin has no vote-mode inference layer at all (no theta, no mode gap, no
Detroit/Clarity sub-feeds) -- the model is deductive, not mode-based, per
Wilson's explicit instruction. So this client is simpler than MI's: it just
extracts Hong/Crowley/Other per county and hands them to
WisconsinPrimaryModel.update_county(). percent_reporting is still treated as a
PRECINCT metric, not a vote-completeness metric, on the same caution as MI's
feed -- unconfirmed for this specific payload, but civicAPI's schema has been
consistent across races so far.

Adopts MI's normalize_county() approach for name matching (strips "County",
"Saint"->"st", punctuation) since that's a proven pattern -- Wisconsin's
"St. Croix" is exactly the kind of name that breaks naive string matching.
"""

import re
import time
import unicodedata

try:
    import requests
except ImportError:
    requests = None

API_BASE = "https://civicapi.org/api/v2"
WISCONSIN_GOV_DEM_PRIMARY = 85787

# Substring match keys -- VERIFY against the actual payload once reachable.
HONG_KEYS = ("hong",)
CROWLEY_KEYS = ("crowley",)

REQUEST_TIMEOUT = 15
MAX_RETRIES = 4


def normalize_county(name: str) -> str:
    """Reduce a county name to a matching key. Handles 'St. Croix' against
    'st_croix' or 'Saint Croix', and a trailing 'County' if the feed adds one."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\bcounty\b", " ", text)
    text = re.sub(r"\bsaint\b", "st", text)
    text = text.replace(".", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def build_county_lookup(county_names) -> dict:
    return {normalize_county(c): c for c in county_names}


def fetch_race(race_id: int = WISCONSIN_GOV_DEM_PRIMARY,
               timeout: int = REQUEST_TIMEOUT,
               max_retries: int = MAX_RETRIES,
               session=None) -> dict:
    """GET a race payload, retrying on transient failure with backoff.
    Raises on exhaustion -- callers should catch and keep the last good snapshot."""
    if requests is None:
        raise RuntimeError("requests is not installed: pip install requests")

    url = "{}/race/{}".format(API_BASE, race_id)
    getter = session.get if session is not None else requests.get
    last_error = None

    for attempt in range(max_retries):
        try:
            response = getter(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError("civicAPI fetch failed after {} attempts: {}".format(
        max_retries, last_error))


def _match_candidate(name: str, keys: tuple) -> bool:
    lowered = str(name).lower()
    return any(k in lowered for k in keys)


def extract_three_bucket(candidate_list: list) -> tuple:
    """Pull Hong and Crowley votes out of a candidate array; everyone else
    (Rodriguez, Brennan, Roys, Hughes, write-ins) sums into 'other'.
    Returns (hong, crowley, other, matched_names)."""
    hong = crowley = other = 0
    matched = {"hong": None, "crowley": None}

    for entry in candidate_list or []:
        name = entry.get("name", "")
        votes = int(entry.get("votes") or 0)
        if _match_candidate(name, HONG_KEYS):
            hong += votes
            matched["hong"] = name
        elif _match_candidate(name, CROWLEY_KEYS):
            crowley += votes
            matched["crowley"] = name
        else:
            other += votes

    return hong, crowley, other, matched


def parse_payload(payload: dict, county_names) -> dict:
    """Turn a civicAPI race payload into county-level three-bucket vote counts.
    UNVERIFIED against the actual WI payload -- confirm before election night."""
    lookup = build_county_lookup(county_names)

    state_hong, state_crowley, state_other, matched_names = extract_three_bucket(
        payload.get("candidates"))

    records = {}
    unmatched = []

    for _slug, region in (payload.get("region_results") or {}).items():
        if str(region.get("type", "")).lower() not in ("county", ""):
            continue
        raw_name = region.get("name", _slug)
        key = normalize_county(raw_name)
        county = lookup.get(key)
        if county is None:
            unmatched.append(raw_name)
            continue

        hong, crowley, other, _ = extract_three_bucket(region.get("candidates"))
        total = hong + crowley + other
        if total <= 0:
            continue

        records[county] = {
            "hong": hong,
            "crowley": crowley,
            "other": other,
            "percent_precincts": region.get("percent_reporting"),
        }

    return {
        "election_name": payload.get("election_name"),
        "last_updated": payload.get("last_updated"),
        "percent_precincts_statewide": payload.get("percent_reporting"),
        "state_hong": state_hong,
        "state_crowley": state_crowley,
        "state_other": state_other,
        "candidate_names": matched_names,
        "counties": records,
        "unmatched": unmatched,
    }
