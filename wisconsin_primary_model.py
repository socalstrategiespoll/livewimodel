"""
Wisconsin Democratic Gubernatorial Primary - Live Election Night Model (v2)
Deductive remainder-projection architecture (SD/GA style), NOT vote-method
mode inference (WI counties report in effectively random order, so there's
no early/ED split to exploit).

v2 adds the four pieces flagged as missing vs. the MI/SD builds:
  1. Monte Carlo simulation (N_SIMS draws -> percentiles, win probability)
  2. Turnout recalibration (feed-implied turnout replaces static prior,
     credibility-ramped, clamped, propagated to not-yet-reported counties)
  3. Momentum constraint (once a county crosses a reporting threshold, its
     final margin can't drift further than a hard cap from observed)
  4. Hierarchical shift: universal (statewide) + regional + a coalition-
     similarity kernel (2016 Clinton/Sanders index) + idiosyncratic noise,
     replacing the flat regional-only shrinkage
"""

import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

# ------------------------------------------------------------------
# Tunable constants (matching MI/SD defaults per Wilson)
# ------------------------------------------------------------------
CREDIBILITY_EXPONENT = 2.0
OUTLIER_LAMBDA = 3.0
TAU_FLOOR = 0.08
N_SIMS = 20000

# Turnout recalibration
TURNOUT_FULL_TRUST_PCT = 0.25     # credibility ramp fully trusts feed-implied turnout by this %
TURNOUT_CLAMP = (0.40, 2.50)      # min/max ratio vs. prior turnout

# Momentum constraint
MOMENTUM_TRIGGER_PCT = 0.30       # county must be at least this reported to trigger the clamp
MOMENTUM_MAX_DRIFT = 10.0         # final margin can't be more than this many points from observed

# Global evidence floor: a weighted MEAN of a single data point equals that
# point regardless of its weight, so outlier dampening alone can't stop one
# lightly-reported county from swinging the statewide shift before any other
# county has competed against it. This shrinks the aggregate shift toward
# zero based on TOTAL accumulated weight, independent of per-county weighting.
GLOBAL_EVIDENCE_PRIOR = 500.0     # ~1.5 large fully-reported counties' worth of weight
REGIONAL_EVIDENCE_PRIOR = 50.0    # same idea, scaled down for a single-region pool
KERNEL_EVIDENCE_PRIOR = 50.0      # same idea, scaled down for the coalition-kernel pool

# Pre-election uncertainty about the TOPLINE itself (not the same thing as
# statewide_shift_var, which measures heterogeneity in observed county
# surprises and is near-zero pre-election by construction). The baselines
# come from limited polling plus a 2016 coalition proxy, not certainty --
# this reflects that genuine uncertainty and shrinks toward zero as real
# results accumulate evidence, using the same evidence-weighted shrink as
# the shift layers above.
PRE_ELECTION_MARGIN_SD = 9.0        # points; tuned (post per-candidate-independence rework) so Crowley's
                                    # pre-election win probability lands near 5%
OTHER_PRE_ELECTION_SD = 9.0         # points; SEPARATE knob from the above -- Other is an aggregate of
                                    # 4 different candidates with no direct polling on the bucket itself,
                                    # so this may need widening independently of the Hong/Crowley figure.
                                    # Starting at the same magnitude as PRE_ELECTION_MARGIN_SD; adjust
                                    # this one alone if Other's range looks too tight or too wide.

COUNTY_HETEROGENEITY = {
    "Milwaukee": 12.0,
    "Dane": 8.0,
    "DEFAULT": 2.5,
}

REGIONS = {
    "Milwaukee": ["Milwaukee"],
    "WOW": ["Waukesha", "Ozaukee", "Washington"],
    "Outer Milwaukee DMA": ["Kenosha", "Racine", "Walworth", "Jefferson", "Dodge", "Sheboygan"],
    "Dane": ["Dane"],
    "Outer Madison DMA": ["Columbia", "Grant", "Green", "Iowa", "Juneau", "Lafayette",
                           "Marquette", "Richland", "Rock", "Sauk"],
    "BOW": ["Brown", "Outagamie", "Winnebago"],
    "Outer Green Bay DMA": ["Calumet", "Door", "Fond du Lac", "Green Lake", "Kewaunee",
                             "Manitowoc", "Marinette", "Menominee", "Oconto", "Shawano",
                             "Waupaca", "Waushara"],
    "La Crosse-Eau Claire DMA": ["Buffalo", "Chippewa", "Clark", "Crawford", "Eau Claire",
                                  "Jackson", "La Crosse", "Monroe", "Rusk", "Trempealeau", "Vernon"],
    "Central & Northwoods": ["Adams", "Forest", "Langlade", "Lincoln", "Marathon", "Oneida",
                              "Portage", "Price", "Taylor", "Vilas", "Wood", "Florence"],
    "Duluth/MSP": ["Ashland", "Bayfield", "Douglas", "Iron", "Sawyer", "Washburn",
                   "Barron", "Burnett", "Dunn", "Pepin", "Pierce", "Polk", "St. Croix"],
}
COUNTY_REGION = {c: r for r, cs in REGIONS.items() for c in cs}

# 2016 Clinton/Sanders shares -- reused here as the "coalition kernel":
# counties with similar Sanders-overperformance are treated as more likely
# to move together than counties merely sharing a DMA region.
CLINTON_SANDERS = {
"Adams": (47.22,52.03), "Ashland": (36.00,63.57), "Barron": (46.11,53.16), "Bayfield": (36.09,63.52),
"Brown": (42.40,57.27), "Buffalo": (41.46,57.74), "Burnett": (49.04,50.00), "Calumet": (42.86,56.86),
"Chippewa": (43.77,55.79), "Clark": (42.67,57.04), "Columbia": (39.21,60.49), "Crawford": (41.61,57.81),
"Dane": (37.27,62.51), "Dodge": (41.75,57.91), "Door": (46.06,53.62), "Douglas": (43.82,55.27),
"Dunn": (35.95,63.54), "Eau Claire": (35.88,63.81), "Florence": (36.92,61.81), "Fond du Lac": (42.65,57.07),
"Forest": (47.25,51.67), "Grant": (40.37,59.10), "Green": (38.59,60.95), "Green Lake": (42.89,56.52),
"Iowa": (40.14,59.40), "Iron": (42.93,55.30), "Jackson": (40.64,58.79), "Jefferson": (38.60,61.07),
"Juneau": (42.21,57.29), "Kenosha": (42.49,57.13), "Kewaunee": (47.06,52.40), "La Crosse": (36.93,62.84),
"Lafayette": (46.37,52.75), "Langlade": (44.83,54.42), "Lincoln": (41.33,58.27), "Manitowoc": (43.37,56.03),
"Marathon": (40.66,58.87), "Marinette": (48.59,50.81), "Marquette": (42.62,56.56), "Menominee": (36.36,63.28),
"Milwaukee": (51.68,48.02), "Monroe": (38.86,60.61), "Oconto": (47.99,51.32), "Oneida": (39.40,60.09),
"Outagamie": (39.62,60.07), "Ozaukee": (48.75,51.04), "Pepin": (43.54,56.16), "Pierce": (41.82,57.27),
"Polk": (46.21,52.85), "Portage": (35.08,64.46), "Price": (37.56,61.79), "Racine": (48.84,50.82),
"Richland": (41.55,58.16), "Rock": (39.20,60.42), "Rusk": (42.52,56.90), "St. Croix": (45.90,53.25),
"Sauk": (38.48,61.22), "Sawyer": (36.86,62.46), "Shawano": (41.19,58.44), "Sheboygan": (44.13,55.30),
"Taylor": (39.68,59.57), "Trempealeau": (44.76,54.68), "Vernon": (35.60,64.01), "Vilas": (36.36,60.09),
"Walworth": (37.96,61.65), "Washburn": (42.32,56.76), "Washington": (45.24,54.46), "Waukesha": (48.28,51.40),
"Waupaca": (39.68,59.77), "Waushara": (43.45,56.02), "Winnebago": (38.44,61.22), "Wood": (39.46,60.20),
}
STATE_SANDERS = 56.59
SANDERS_INDEX = {c: sand / STATE_SANDERS for c, (clin, sand) in CLINTON_SANDERS.items()}


@dataclass
class CountyState:
    name: str
    region: str
    baseline_hong_pct: float
    baseline_crowley_pct: float
    baseline_other_pct: float
    expected_turnout: int              # ORIGINAL prior turnout, never mutated
    calibrated_turnout: Optional[float] = None   # feed-implied, set once counties start reporting
    pct_reporting: float = 0.0
    counted_votes: int = 0
    hong_votes: int = 0
    crowley_votes: int = 0
    other_votes: int = 0
    # Observed rate for each candidate = their votes / this county's TOTAL
    # counted votes (all-candidate denominator, not a two-way pool) -- these
    # are the three independent observed quantities the shift layers below
    # learn from. None until the county has any votes counted.
    observed_hong_rate: Optional[float] = None
    observed_crowley_rate: Optional[float] = None
    observed_other_rate: Optional[float] = None

    @property
    def effective_turnout(self) -> float:
        return self.calibrated_turnout if self.calibrated_turnout is not None else self.expected_turnout

    def baseline_rate(self, candidate: str) -> float:
        return {"hong": self.baseline_hong_pct, "crowley": self.baseline_crowley_pct,
                "other": self.baseline_other_pct}[candidate] / 100.0

    def observed_rate(self, candidate: str) -> Optional[float]:
        return {"hong": self.observed_hong_rate, "crowley": self.observed_crowley_rate,
                "other": self.observed_other_rate}[candidate]

    @property
    def observed_margin(self) -> Optional[float]:
        """Two-way Hong-vs-Crowley margin among counted votes -- kept as a
        convenience/display quantity; the projection engine itself no longer
        depends on this, it works from the three independent rates instead."""
        two_way = self.hong_votes + self.crowley_votes
        if two_way <= 0:
            return None
        return 100.0 * (self.hong_votes - self.crowley_votes) / two_way

    @property
    def baseline_margin(self) -> float:
        two_way = self.baseline_hong_pct + self.baseline_crowley_pct
        if two_way <= 0:
            return 0.0
        return 100.0 * (self.baseline_hong_pct - self.baseline_crowley_pct) / two_way

    @property
    def heterogeneity(self) -> float:
        return COUNTY_HETEROGENEITY.get(self.name, COUNTY_HETEROGENEITY["DEFAULT"])

    @property
    def credibility(self) -> float:
        if self.pct_reporting <= 0:
            return 0.0
        completeness_weight = self.pct_reporting ** (1 / CREDIBILITY_EXPONENT)
        design_var = (self.heterogeneity ** 2) * (1 - self.pct_reporting)
        noise_penalty = 1.0 / (1.0 + design_var / 50.0)
        return completeness_weight * noise_penalty

    @property
    def sanders_index(self) -> float:
        return SANDERS_INDEX.get(self.name, 1.0)


CANDIDATES = ("hong", "crowley", "other")


class WisconsinPrimaryModel:
    def __init__(self, baseline_path: str):
        with open(baseline_path) as f:
            baselines = json.load(f)
        self.counties: Dict[str, CountyState] = {}
        for name, b in baselines.items():
            self.counties[name] = CountyState(
                name=name, region=b["region"],
                baseline_hong_pct=b["Hong"], baseline_crowley_pct=b["Crowley"],
                baseline_other_pct=b["Other"], expected_turnout=b["turnout"],
            )
        self.total_evidence_weight = 0.0
        # Each candidate gets its own independent statewide / regional / county
        # shift, computed identically to the others -- no candidate's share is
        # derived as a residual of the other two. Final shares are normalized
        # to sum to 100% only at the projection step, not baked into the shift
        # math itself.
        self.statewide_shift: Dict[str, float] = {k: 0.0 for k in CANDIDATES}
        self.statewide_shift_var: Dict[str, float] = {k: TAU_FLOOR ** 2 for k in CANDIDATES}
        self.regional_shift: Dict[str, Dict[str, float]] = {k: {r: 0.0 for r in REGIONS} for k in CANDIDATES}
        self.county_shift: Dict[str, Dict[str, float]] = {k: {c: 0.0 for c in self.counties} for k in CANDIDATES}

    # ------------------------------------------------------------
    def update_county(self, name: str, hong: int, crowley: int, other: int, pct_reporting: float):
        c = self.counties[name]
        c.hong_votes, c.crowley_votes, c.other_votes = hong, crowley, other
        c.counted_votes = hong + crowley + other
        c.pct_reporting = pct_reporting
        if c.counted_votes > 0:
            c.observed_hong_rate = hong / c.counted_votes
            c.observed_crowley_rate = crowley / c.counted_votes
            c.observed_other_rate = other / c.counted_votes
        self._recalibrate_turnout()
        self._recompute_shifts()

    # ------------------------------------------------------------
    # (2) TURNOUT RECALIBRATION
    # ------------------------------------------------------------
    def _recalibrate_turnout(self):
        """Feed-implied turnout (counted/pct_reporting) replaces the static prior
        wherever a county has enough reporting to trust it, credibility-ramped
        and clamped. Counties still at 0% get the size-weighted median ratio
        from reporting counties, so a statewide turnout surprise (e.g. lower
        primary turnout than the 2024-primary-based prior assumed) propagates
        forward instead of being silently ignored."""
        ratios, sizes = [], []
        for c in self.counties.values():
            if c.pct_reporting > 0 and c.counted_votes > 0:
                implied = c.counted_votes / c.pct_reporting
                ratio = implied / c.expected_turnout
                ratio = min(max(ratio, TURNOUT_CLAMP[0]), TURNOUT_CLAMP[1])
                trust = min(c.pct_reporting / TURNOUT_FULL_TRUST_PCT, 1.0)
                c.calibrated_turnout = trust * (ratio * c.expected_turnout) + (1 - trust) * c.expected_turnout
                ratios.append(ratio)
                sizes.append(c.expected_turnout)

        if not ratios:
            return
        ratios = np.array(ratios)
        sizes = np.array(sizes)
        order = np.argsort(ratios)
        cum_size = np.cumsum(sizes[order])
        median_idx = np.searchsorted(cum_size, cum_size[-1] / 2.0)
        size_weighted_median_ratio = ratios[order][min(median_idx, len(ratios) - 1)]

        for c in self.counties.values():
            if c.pct_reporting == 0:
                c.calibrated_turnout = size_weighted_median_ratio * c.expected_turnout

    # ------------------------------------------------------------
    # (4) HIERARCHICAL SHIFT: universal + regional + coalition kernel,
    # computed independently for EACH of the three candidates
    # ------------------------------------------------------------
    KERNEL_BANDWIDTH = 0.12

    def _recompute_shifts(self):
        reporting = [c for c in self.counties.values() if c.pct_reporting > 0 and c.counted_votes > 0]
        if not reporting:
            self.total_evidence_weight = 0.0
            for k in CANDIDATES:
                self.statewide_shift[k] = 0.0
                self.regional_shift[k] = {r: 0.0 for r in REGIONS}
                self.county_shift[k] = {c: 0.0 for c in self.counties}
            return

        regions = np.array([c.region for c in reporting])
        sanders_idx = np.array([c.sanders_index for c in reporting])
        turnouts = np.array([c.effective_turnout for c in reporting])
        credibilities = np.array([c.credibility for c in reporting])
        base_w = credibilities * np.sqrt(turnouts)

        total_weight_for_evidence = 0.0

        for candidate in CANDIDATES:
            surprises = np.array([
                100.0 * (c.observed_rate(candidate) - c.baseline_rate(candidate)) for c in reporting
            ])
            outlier_factor = 1.0 / (1.0 + (np.abs(surprises) / OUTLIER_LAMBDA) ** 2)
            weights = base_w * outlier_factor
            total_weight_for_evidence = max(total_weight_for_evidence, float(weights.sum()))

            total_weight = weights.sum()
            if total_weight == 0:
                self.statewide_shift[candidate] = 0.0
            else:
                wmean = np.average(surprises, weights=weights)
                tau2 = max(TAU_FLOOR ** 2, np.average((surprises - wmean) ** 2, weights=weights))
                global_shrink = total_weight / (total_weight + GLOBAL_EVIDENCE_PRIOR)
                self.statewide_shift[candidate] = global_shrink * wmean
                self.statewide_shift_var[candidate] = tau2

            for region in REGIONS:
                idx = regions == region
                if not idx.any():
                    self.regional_shift[candidate][region] = self.statewide_shift[candidate]
                    continue
                r_wmean = (np.average(surprises[idx], weights=weights[idx])
                           if weights[idx].sum() > 0 else self.statewide_shift[candidate])
                shrink = weights[idx].sum() / (weights[idx].sum() + REGIONAL_EVIDENCE_PRIOR)
                self.regional_shift[candidate][region] = (
                    shrink * r_wmean + (1 - shrink) * self.statewide_shift[candidate])

            for name, county in self.counties.items():
                kernel_w = weights * np.exp(-((sanders_idx - county.sanders_index) ** 2) / (2 * self.KERNEL_BANDWIDTH ** 2))
                kernel_w = kernel_w * np.where(regions == county.region, 1.5, 1.0)
                if kernel_w.sum() <= 0:
                    local_est = self.regional_shift[candidate][county.region]
                else:
                    local_est = np.average(surprises, weights=kernel_w)
                shrink = kernel_w.sum() / (kernel_w.sum() + KERNEL_EVIDENCE_PRIOR)
                self.county_shift[candidate][name] = (
                    shrink * local_est + (1 - shrink) * self.regional_shift[candidate][county.region])

        self.total_evidence_weight = total_weight_for_evidence

    # ------------------------------------------------------------
    # (3) MOMENTUM CONSTRAINT applied inside the projection
    # ------------------------------------------------------------
    def project_rate(self, c: CountyState, candidate: str) -> float:
        """Projected share of the vote for one candidate in one county, as a
        fraction of the FULL electorate (0-1). Computed independently per
        candidate -- credibility-blended with that candidate's own
        statewide/regional/county shift, own momentum constraint. Three of
        these (hong/crowley/other) get normalized to sum to 1 wherever votes
        are actually allocated -- not before."""
        baseline_rate = c.baseline_rate(candidate)
        shift = self.county_shift[candidate].get(c.name, 0.0) / 100.0
        adjusted_baseline = min(max(baseline_rate + shift, 0.0), 0.97)

        observed = c.observed_rate(candidate)
        if c.pct_reporting >= 0.999:
            return observed if observed is not None else adjusted_baseline
        if observed is None:
            return adjusted_baseline

        w = c.credibility
        projected = w * observed + (1 - w) * adjusted_baseline

        if c.pct_reporting >= MOMENTUM_TRIGGER_PCT:
            lo = observed - MOMENTUM_MAX_DRIFT / 100.0
            hi = observed + MOMENTUM_MAX_DRIFT / 100.0
            projected = min(max(projected, lo), hi)

        return min(max(projected, 0.0), 0.97)

    def project_county(self, c: CountyState) -> float:
        """Two-way Hong-vs-Crowley margin, DERIVED from the two independent
        projected rates -- kept for callers that still want a margin number,
        not used internally by statewide_projection/run_simulation anymore."""
        hong = self.project_rate(c, "hong")
        crowley = self.project_rate(c, "crowley")
        if hong + crowley <= 0:
            return 0.0
        return 100.0 * (hong - crowley) / (hong + crowley)

        return projected

    # ------------------------------------------------------------
    def statewide_projection(self) -> Dict[str, float]:
        total_hong = total_crowley = total_other = 0.0
        for c in self.counties.values():
            remaining_votes = max(0, c.effective_turnout - c.counted_votes)

            raw_hong = self.project_rate(c, "hong")
            raw_crowley = self.project_rate(c, "crowley")
            raw_other = self.project_rate(c, "other")
            raw_total = raw_hong + raw_crowley + raw_other
            if raw_total <= 0:
                hong_share, crowley_share, other_share = 1/3, 1/3, 1/3
            else:
                hong_share = raw_hong / raw_total
                crowley_share = raw_crowley / raw_total
                other_share = raw_other / raw_total

            total_hong += c.hong_votes + remaining_votes * hong_share
            total_crowley += c.crowley_votes + remaining_votes * crowley_share
            total_other += c.other_votes + remaining_votes * other_share

        total = total_hong + total_crowley + total_other
        return {
            "Hong_pct": 100 * total_hong / total, "Crowley_pct": 100 * total_crowley / total,
            "Other_pct": 100 * total_other / total,
            "Hong_votes": total_hong, "Crowley_votes": total_crowley, "Other_votes": total_other,
            "statewide_shift": self.statewide_shift["hong"] - self.statewide_shift["crowley"],
        }

    # ------------------------------------------------------------
    # (1) MONTE CARLO SIMULATION
    # ------------------------------------------------------------
    def run_simulation(self, n_sims: int = N_SIMS, seed: Optional[int] = None) -> Dict:
        """Draws n_sims full-state simulations: each county's uncounted remainder
        margin is sampled from a Normal centered on its point projection, with
        SD shrinking as pct_reporting rises and inflated by county heterogeneity
        for still-thin counties. A shared statewide draw (from statewide_shift_var)
        is added to every county each simulation, so uncertainty is correlated
        across counties rather than independently averaging out -- this is what
        produces realistic (non-collapsed) win-probability and interval bands."""
        rng = np.random.default_rng(seed)
        counties = list(self.counties.values())
        n = len(counties)

        completeness = np.array([c.pct_reporting for c in counties])
        heterog = np.array([c.heterogeneity for c in counties])
        eff_turnout = np.array([c.effective_turnout for c in counties])
        counted = np.array([c.counted_votes for c in counties])
        remaining_votes = np.maximum(0, eff_turnout - counted)

        # per-county remainder SD: base uncertainty shrinking with completeness,
        # inflated for heterogeneous counties while thin -- same for every
        # candidate, since it's about how much a partially-reported county's
        # remainder can still move, not about any one candidate specifically
        base_sd = 8.0
        county_sd = base_sd * (1 - completeness) ** 0.5 + heterog * (1 - completeness) * 0.3
        county_sd = np.maximum(county_sd, 0.5)

        evidence_shrink = self.total_evidence_weight / (self.total_evidence_weight + GLOBAL_EVIDENCE_PRIOR)
        # Margin SD of ~PRE_ELECTION_MARGIN_SD requires each of Hong/Crowley's
        # independent noise sources to contribute ~SD/sqrt(2), since
        # Var(hong-crowley) = Var(hong) + Var(crowley) for independent noise.
        # Other gets its OWN separately-tunable prior (OTHER_PRE_ELECTION_SD)
        # rather than reusing the Hong/Crowley figure -- per Wilson, there's
        # real uncertainty in how big Other's bucket actually runs (it's an
        # aggregate of 4 different candidates, no direct polling on the
        # aggregate itself) and this may need independent widening later
        # without disturbing the Hong-vs-Crowley calibration.
        prior_sd = {
            "hong": (PRE_ELECTION_MARGIN_SD / 1.414) * (1 - evidence_shrink),
            "crowley": (PRE_ELECTION_MARGIN_SD / 1.414) * (1 - evidence_shrink),
            "other": OTHER_PRE_ELECTION_SD * (1 - evidence_shrink),
        }

        sim_votes = {}
        for candidate, actual_v in (("hong", "hong_votes"), ("crowley", "crowley_votes"), ("other", "other_votes")):
            point_rate = np.array([self.project_rate(c, candidate) for c in counties])
            statewide_sd = math.sqrt(self.statewide_shift_var[candidate]) * 15.0
            statewide_sd = math.sqrt(statewide_sd ** 2 + prior_sd[candidate] ** 2)

            momentum_active = np.array([
                c.pct_reporting >= MOMENTUM_TRIGGER_PCT and c.observed_rate(candidate) is not None
                for c in counties
            ])
            obs_arr = np.array([
                (c.observed_rate(candidate) if c.observed_rate(candidate) is not None else 0.0)
                for c in counties
            ])
            lo_bound = obs_arr - MOMENTUM_MAX_DRIFT / 100.0
            hi_bound = obs_arr + MOMENTUM_MAX_DRIFT / 100.0

            shared_shock = rng.normal(0, statewide_sd, size=(n_sims, 1)) / 100.0
            county_shock = rng.normal(0, 1, size=(n_sims, n)) * (county_sd[None, :] / 100.0)
            sim_rate = point_rate[None, :] + shared_shock + county_shock

            clipped = np.clip(sim_rate, lo_bound[None, :], hi_bound[None, :])
            sim_rate = np.where(momentum_active[None, :], clipped, sim_rate)
            sim_rate = np.clip(sim_rate, 0.0, 0.97)

            sim_votes[candidate] = sim_rate
            sim_votes[candidate + "_actual"] = np.array([getattr(c, actual_v) for c in counties], dtype=float)

        # normalize the three independently-noised rates to sum to 1 per
        # simulation per county, THEN allocate the remaining vote
        raw_total = sim_votes["hong"] + sim_votes["crowley"] + sim_votes["other"]
        raw_total = np.maximum(raw_total, 1e-9)
        hong_share = sim_votes["hong"] / raw_total
        crowley_share = sim_votes["crowley"] / raw_total
        other_share = sim_votes["other"] / raw_total

        hong_totals = (sim_votes["hong_actual"][None, :] + remaining_votes[None, :] * hong_share).sum(axis=1)
        crowley_totals = (sim_votes["crowley_actual"][None, :] + remaining_votes[None, :] * crowley_share).sum(axis=1)
        other_totals = (sim_votes["other_actual"][None, :] + remaining_votes[None, :] * other_share).sum(axis=1)

        grand_totals = hong_totals + crowley_totals + other_totals
        # Margin as share of the FULL electorate (all candidates in the
        # denominator), not normalized to just the Hong/Crowley two-way pool.
        results = 100 * hong_totals / grand_totals - 100 * crowley_totals / grand_totals

        # Per-candidate simulated statewide VOTE SHARE (not margin) -- each
        # candidate's own distribution across the n_sims draws, for the
        # 50%/90% range display Wilson asked for.
        hong_pct_sims = 100 * hong_totals / grand_totals
        crowley_pct_sims = 100 * crowley_totals / grand_totals
        other_pct_sims = 100 * other_totals / grand_totals

        def pct_range(arr):
            return {
                "p05": float(np.percentile(arr, 5)), "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)), "p95": float(np.percentile(arr, 95)),
            }

        return {
            "mean_margin": float(np.mean(results)),
            "p05": float(np.percentile(results, 5)),
            "p10": float(np.percentile(results, 10)),
            "p25": float(np.percentile(results, 25)),
            "p50": float(np.percentile(results, 50)),
            "p75": float(np.percentile(results, 75)),
            "p90": float(np.percentile(results, 90)),
            "p95": float(np.percentile(results, 95)),
            "hong_win_prob": float(np.mean(results > 0)),
            "crowley_win_prob": float(np.mean(results < 0)),
            "n_sims": n_sims,
            "margins": results,  # raw array -- server thins this into percentiles for the site's density curve
            "candidate_share_ranges": {
                "hong": pct_range(hong_pct_sims),
                "crowley": pct_range(crowley_pct_sims),
                "other": pct_range(other_pct_sims),
            },
        }
