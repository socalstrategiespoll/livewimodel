# Wisconsin Governor Democratic Primary — Live Model

County-level deductive live election-night model for the 2026 Wisconsin Governor
Democratic Primary (Hong vs. Crowley, with the rest of the field summed into
Other), fed by civicAPI race `85787`.

Results from [civicAPI](https://civicapi.org).

## How this differs from the Michigan build

Michigan's model infers vote MODE (absentee vs. Election Day) because Detroit and
most MI counties report those separately and in a somewhat predictable order.
Wisconsin counties report in effectively random order with no consistent
early/Election Day pattern, so this model is **deductive** instead:

- Counted votes per county are held fixed.
- The uncounted remainder is projected at that county's current credibility-blended
  margin (baseline shrunk toward observed as % reporting rises).
- Surprises feed a statewide DerSimonian-Laird shift, a regional shift (10 media-
  market-based regions), and a coalition-similarity kernel (2016 Clinton/Sanders
  index) that move counties that haven't reported yet.
- A global evidence floor prevents one early, lightly-weighted county from
  swinging the statewide number before other counties compete against it.
- Milwaukee and Dane carry a within-county heterogeneity term, same treatment as
  Wayne/Oakland in the MI build.
- Momentum constraint: once a county crosses 30% reporting, its projected margin
  can't drift more than 10 points from what's actually been observed there.
- Turnout is recalibrated live from feed-implied totals (counted / pct_reporting),
  credibility-ramped and clamped, same mechanism as MI's `turnout_calibration.py`.

There is no mode_calibration, no vote_method_split, no Detroit/Clarity sub-feeds.
No mode gap diagnostic on the dashboard, because there's no mode being inferred.

## How it fits together

```
civicAPI  ──►  Render web service  ──►  Cloudflare Pages
 (poll)         (model + JSON API)        (the site)
```

Same single-web-service architecture as MI, for the same reason: a cron container
gets destroyed after every run, wiping the turnout/shift state this model
accumulates, and has no URL for a site to read.

## Files

| File | Does |
|---|---|
| `server.py` | background poller + JSON API. The entrypoint |
| `civicapi_feed.py` | API client, payload parsing, county name matching |
| `wisconsin_primary_model.py` | 72-county baselines, deductive projection, Monte Carlo |
| `wi_dem_primary_county_baselines.json` | pre-election county baselines (Hong/Crowley/Other) |
| `wi-counties.geojson` | 72-county geometry, built from npm us-atlas (FIPS 55) |
| `index.html` / `app.js` / `style.css` | the static site |

## Endpoints

| Route | Returns |
|---|---|
| `/health` | uptime, cycle count, last error |
| `/api/projection` | the current projection, county table, diagnostics |
| `/api/history` | one compact record per cycle since start |

CORS is open, so the site can be hosted anywhere.

## Known limitations

- **civicAPI payload schema is UNVERIFIED for this race.** `civicapi_feed.py` was
  written on the same schema assumptions as the MI feed (which is confirmed
  working) but has not been tested against race 85787's actual response. Get a
  sample response before election night and fix the parsing if it doesn't match.
- **County name matching** needs verification against civicAPI's actual naming for
  "St. Croix" and any other punctuated names.
- **No sub-county vote-method split.** Wisconsin has a real structural analog (WEC
  Central Count Absentee municipalities, ~16 counties including Milwaukee city)
  but it isn't built into this model.
- **Coalition kernel is modest in practice.** WI's 2016 Clinton/Sanders split is
  compressed across most counties outside Milwaukee, so the regional layer does
  most of the differentiation.
- **State is in memory.** A restart costs the accumulated turnout/shift state. Set
  `STATE_DIR` to a mounted disk to avoid that.
