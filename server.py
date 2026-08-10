# Render web service: polls civicAPI on a background thread and serves the projection.
#
# Same reasoning as the Michigan build for why this is a web service, not a cron job:
# Render destroys a cron container after every run, which would wipe the county-level
# turnout/shift state this model accumulates over the night, and a cron job has no
# URL for a site to read. A persistent web service solves both.
#
# WHAT'S DIFFERENT FROM THE MICHIGAN SERVER
#
#     No mode_calibration, no vote_method_split, no Detroit/Clarity sub-feeds. Wisconsin
#     counties report in effectively random order with no consistent early/Election Day
#     pattern to exploit, so the model is deductive rather than mode-inferring: counted
#     votes held fixed, remainder projected at a credibility-blended margin, statewide +
#     regional + coalition-kernel shifts move counties that haven't reported. See
#     wisconsin_primary_model.py for the full mechanism.
#
#     Three tracked buckets instead of two candidates: Hong, Crowley, and Other (the
#     rest of the field -- Rodriguez, Brennan, Roys, Hughes -- summed together).
#
# DESIGN NOTES (unchanged from MI)
#
#     Stdlib only, single-process threading server -- gunicorn with multiple workers
#     would spawn multiple pollers fighting over the API and producing inconsistent
#     projections.
#
#     The poller never lets an exception escape. A civicAPI hiccup costs one update;
#     the previous projection stays served with its own timestamp.
#
#     State is in memory. Set STATE_DIR to a mounted Render disk if you want the
#     turnout-calibration and shift history to survive a restart.

import json
import os
import threading
import time
import traceback

import numpy as np

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wisconsin_primary_model import WisconsinPrimaryModel
from civicapi_feed import fetch_race, parse_payload, WISCONSIN_GOV_DEM_PRIMARY


PORT = int(os.environ.get("PORT", 10000))
RACE_ID = int(os.environ.get("RACE_ID", WISCONSIN_GOV_DEM_PRIMARY))
N_SIMS = int(os.environ.get("N_SIMS", 20000))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 60))
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", 2000))
STATE_DIR = os.environ.get("STATE_DIR", "")
BASELINE_PATH = os.environ.get("BASELINE_PATH", "wi_dem_primary_county_baselines.json")


class ModelState:
    """Everything the poller produces and the HTTP handler reads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.projection = None
        self.history = []
        self.error = None
        self.cycles = 0
        self.started_at = datetime.now(timezone.utc).isoformat()

    def publish(self, output: dict) -> None:
        with self.lock:
            self.projection = output
            self.history.append({
                "updated_at": output["updated_at"],
                "hong_pct": output["projection"]["hong_pct"],
                "crowley_pct": output["projection"]["crowley_pct"],
                "hong_win_probability": output["projection"]["hong_win_probability"],
                "interval_90": output["projection"]["interval_90"],
                "pct_counted": output["counted"]["pct_of_projected_turnout"],
                "counties_reporting": output["diagnostics"]["counties_reporting"],
                "statewide_shift": output["diagnostics"]["statewide_shift"],
            })
            if len(self.history) > HISTORY_LIMIT:
                self.history = self.history[-HISTORY_LIMIT:]
            self.error = None
            self.cycles += 1

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message

    def snapshot(self) -> tuple:
        with self.lock:
            return self.projection, list(self.history), self.error, self.cycles


STATE = ModelState()


def build_output(model: WisconsinPrimaryModel, sim: dict, proj: dict,
                 parsed: dict, race_id: int) -> dict:
    total_counted = proj["Hong_votes"] + proj["Crowley_votes"] + proj["Other_votes"]
    total_expected = sum(c.effective_turnout for c in model.counties.values())
    counted_actual = sum(c.counted_votes for c in model.counties.values())

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "civicapi.org",
        "attribution": "Election results from civicAPI (civicapi.org)",
        "race_id": race_id,
        "election_name": parsed.get("election_name"),
        "feed_last_updated": parsed.get("last_updated"),
        "counted": {
            "hong": parsed.get("state_hong"),
            "crowley": parsed.get("state_crowley"),
            "other": parsed.get("state_other"),
            "pct_of_projected_turnout": round(100 * counted_actual / max(total_expected, 1), 2),
            "pct_precincts_reporting": parsed.get("percent_precincts_statewide"),
        },
        "turnout": {
            "projected": round(total_expected),
        },
        "projection": {
            "hong_win_probability": round(sim["hong_win_prob"], 4),
            "crowley_win_probability": round(sim["crowley_win_prob"], 4),
            "median_margin": round(sim["p50"], 2),
            "interval_50": [round(sim["p10"], 2), round(sim["p90"], 2)],
            "interval_90": [round(sim["p05"], 2), round(sim["p95"], 2)],
            "hong_pct": round(proj["Hong_pct"], 2),
            "crowley_pct": round(proj["Crowley_pct"], 2),
            "other_pct": round(proj["Other_pct"], 2),
            "hong_votes": int(proj["Hong_votes"]),
            "crowley_votes": int(proj["Crowley_votes"]),
            "other_votes": int(proj["Other_votes"]),
            # Raw simulated distribution, thinned to ~60 percentiles -- the site draws
            # its density curve from this rather than assuming a normal shape, since
            # the posterior can be skewed while large counties are partly counted.
            "margin_percentiles": [
                round(float(v), 2) for v in
                np.percentile(sim["margins"], np.arange(1, 100, 1.65))
            ],
        },
        "counties": build_county_table(model),
        "diagnostics": {
            "counties_reporting": sum(1 for c in model.counties.values() if c.pct_reporting > 0),
            # Statewide shift is now tracked independently per candidate; this
            # summary number is Hong's shift minus Crowley's, matching the old
            # single-margin-shift meaning for anything still reading it as one number.
            "statewide_shift": round(model.statewide_shift["hong"] - model.statewide_shift["crowley"], 2),
            "statewide_shift_by_candidate": {k: round(v, 2) for k, v in model.statewide_shift.items()},
            "unmatched_counties": parsed.get("unmatched", []),
            "candidate_names": parsed.get("candidate_names"),
        },
        # Regional swing shown on the dashboard: Hong's regional shift minus
        # Crowley's, same summary convention as statewide_shift above.
        "regional_shift": {
            region: round(model.regional_shift["hong"][region] - model.regional_shift["crowley"][region], 2)
            for region in model.regional_shift["hong"]
        },
    }


def build_county_table(model: WisconsinPrimaryModel) -> list:
    """Per-county rows covering ALL 72 counties, not just the ones reporting --
    the maps need every county every cycle.

    Margins here use the same all-candidate-denominator convention as the
    statewide headline (Hong minus Crowley, as a share of the full electorate
    including Other) -- not a two-candidate-only normalization. The internal
    deductive math (project_county) still operates on two-way margins, since
    that's what's needed to split projected remainder ballots between Hong
    and Crowley specifically; this function converts that internal number to
    the all-candidate display convention before it's shown."""
    rows = []
    for name, c in model.counties.items():
        counted = c.hong_votes + c.crowley_votes
        total_counted = c.counted_votes
        margin = None
        if total_counted > 0:
            margin = round(100.0 * (c.hong_votes - c.crowley_votes) / total_counted, 1)

        two_way_proj_margin = model.project_county(c)
        # convert two-way projected margin to the all-candidate convention using
        # this county's projected Other rate (credibility-blended with the
        # statewide/regional/county Other-shift, not just this county's own data)
        other_rate = model.project_rate(c, "other")
        display_proj_margin = two_way_proj_margin * (1 - other_rate)

        remaining = max(0, c.effective_turnout - c.counted_votes)

        rows.append({
            "county": name,
            "region": c.region,
            "reporting": c.pct_reporting > 0,
            "hong": c.hong_votes,
            "crowley": c.crowley_votes,
            "other": c.other_votes,
            "votes": c.counted_votes,
            "margin": margin,
            "expected_baseline": round(c.baseline_margin * (1 - c.baseline_other_pct / 100.0), 1),
            "vs_expected": None if margin is None else round(
                margin - c.baseline_margin * (1 - c.baseline_other_pct / 100.0), 1),
            "county_shift": round(
                model.county_shift["hong"].get(name, 0.0) - model.county_shift["crowley"].get(name, 0.0), 1),
            "pct_precincts": c.pct_reporting * 100 if c.pct_reporting else None,
            "pct_of_projected": round(100 * c.counted_votes / max(c.effective_turnout, 1), 1),
            "projected_total": int(c.effective_turnout),
            "calibrated_turnout": int(c.calibrated_turnout) if c.calibrated_turnout else None,
            "remaining": int(round(remaining)),
            "remainder_margin": round(display_proj_margin, 1),
            "projected_final": round(display_proj_margin, 1),
        })

    rows.sort(key=lambda r: (-r["votes"], -r["projected_total"]))
    return rows


def save_state(model: WisconsinPrimaryModel) -> None:
    if not STATE_DIR:
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        snap = {
            name: {
                "hong": c.hong_votes, "crowley": c.crowley_votes, "other": c.other_votes,
                "pct_reporting": c.pct_reporting,
            }
            for name, c in model.counties.items() if c.pct_reporting > 0
        }
        with open(os.path.join(STATE_DIR, "feed_state.json"), "w") as handle:
            json.dump(snap, handle)
    except Exception:
        pass


def load_state(model: WisconsinPrimaryModel) -> None:
    if not STATE_DIR:
        return
    path = os.path.join(STATE_DIR, "feed_state.json")
    try:
        with open(path) as handle:
            stored = json.load(handle)
        for name, rec in stored.items():
            if name in model.counties:
                model.update_county(name, rec["hong"], rec["crowley"], rec["other"],
                                    rec["pct_reporting"])
        print("restored {} counties from {}".format(len(stored), path), flush=True)
    except Exception:
        pass


def poller() -> None:
    """Background loop. Never exits."""
    model = WisconsinPrimaryModel(BASELINE_PATH)
    load_state(model)
    county_names = list(model.counties.keys())

    print("poller started: race {} every {}s, {} sims".format(
        RACE_ID, POLL_INTERVAL, N_SIMS), flush=True)

    while True:
        started = time.time()
        try:
            payload = fetch_race(RACE_ID)
            parsed = parse_payload(payload, county_names)

            for county, record in parsed["counties"].items():
                pct = record.get("percent_precincts") or 0.0
                pct = pct / 100.0 if pct > 1 else pct
                model.update_county(county, record["hong"], record["crowley"],
                                    record["other"], pct)

            sim = model.run_simulation(n_sims=N_SIMS)
            proj = model.statewide_projection()
            output = build_output(model, sim, proj, parsed, RACE_ID)
            STATE.publish(output)
            save_state(model)

            names = output["diagnostics"].get("candidate_names") or {}
            if not names.get("hong") or not names.get("crowley"):
                print("!! CANDIDATE MATCH FAILED: hong={!r} crowley={!r} -- fix "
                      "HONG_KEYS / CROWLEY_KEYS in civicapi_feed.py".format(
                          names.get("hong"), names.get("crowley")), flush=True)
            else:
                print("   matched: {} vs {}".format(names["hong"], names["crowley"]), flush=True)
            if output["diagnostics"]["unmatched_counties"]:
                print("!! UNMATCHED COUNTIES: {} -- fix normalize_county() in "
                      "civicapi_feed.py".format(
                          output["diagnostics"]["unmatched_counties"]), flush=True)

            p = output["projection"]
            print("[{}] {:.1f}% counted | {} cty | Hong {:.1f}  Crowley {:.1f} | "
                  "margin {:+.1f} [{:+.1f}, {:+.1f}] | Hong win {:.1%}".format(
                      datetime.now().strftime("%H:%M:%S"),
                      output["counted"]["pct_of_projected_turnout"],
                      output["diagnostics"]["counties_reporting"],
                      p["hong_pct"], p["crowley_pct"], p["median_margin"],
                      p["interval_90"][0], p["interval_90"][1],
                      p["hong_win_probability"]), flush=True)

        except Exception as exc:
            STATE.fail(str(exc))
            print("[{}] cycle failed, serving last good projection: {}".format(
                datetime.now().strftime("%H:%M:%S"), exc), flush=True)
            traceback.print_exc()

        time.sleep(max(1.0, POLL_INTERVAL - (time.time() - started)))


class Handler(BaseHTTPRequestHandler):

    def _send(self, body, status=200, content_type="application/json"):
        encoded = (body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        projection, history, error, cycles = STATE.snapshot()

        if path in ("/", "/health"):
            return self._send({
                "ok": True, "cycles": cycles, "started_at": STATE.started_at,
                "last_error": error, "has_projection": projection is not None,
            })
        if path == "/api/projection":
            if projection is None:
                return self._send({"error": "no projection yet", "last_error": error}, status=503)
            return self._send(projection)
        if path == "/api/history":
            return self._send({"count": len(history), "cycles": history})
        return self._send({"error": "not found"}, status=404)

    def log_message(self, *args):
        return


def main():
    thread = threading.Thread(target=poller, daemon=True)
    thread.start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("serving on :{}".format(PORT), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
