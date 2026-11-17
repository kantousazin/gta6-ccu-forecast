"""Forecast a launch day's Twitch audience, hour by hour, from its first hour.

Written for make.fun's gta6-twitch-ccu-forecast-01. The program reads one case
(everything recorded up to the end of hour 1 of a game's release day, that
hour's viewer count included, plus the release-day curves of past big
launches) and writes five quantiles for each of hours 2..24.

Method, fitted by leave-one-out cross-validation over the comparable launches
themselves (never on any case's answer):

  * A launch's day curve, divided by its own hour-1 count, has a repeatable
    shape on the log scale: an overnight dip, a climb through the UTC day and
    an evening peak. But the shape only repeats among launches that were
    already live at hour 1. A single global unlock later in the UTC day starts
    near zero and explodes, which poisons the pool, so training launches are
    kept only when their hour-1 count reached at least FRAC_MIN of their own
    day peak (Grand Theft Auto VI unlocks at rolling local midnight, so its
    first hour is already a launched hour in most of the world).
  * Launches are weighted by how close their hour-1 count is to this case's,
    on the log scale (a Gaussian kernel of width KERNEL_BW): a launch that
    opened to 800k viewers is a better guide for another huge launch than one
    that opened to 6k.
  * Each hour's forecast is the weighted empirical quantiles of the training
    launches' log ratios for that hour, with the median shrunk slightly
    toward persistence (MEDIAN_SHRINK) and the spread widened slightly
    (SPREAD_WIDEN), both constants chosen on the same cross-validation.
  * With too few usable training launches the filters relax step by step,
    and with none at all the program falls back to persistence with a
    random-walk spread, so it always writes a valid forecast.

Standard library only, deterministic, no network. Input and output
directories come from MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR when set,
else /input and /output.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantiles at DEFAULT_LEVELS, for the fallback.
Z_LEVELS = [-1.645, -0.674, 0.0, 0.674, 1.645]
FALLBACK_SIGMA = 0.15

FRAC_MIN = 0.15        # hour 1 must reach this share of the launch's day peak
FRAC_RELAXED = 0.08    # first relaxation when too few launches qualify
MIN_TRAIN = 8          # launches wanted before a filter step is accepted
KERNEL_BW = 0.7        # Gaussian kernel width on log hour-1 counts
MEDIAN_SHRINK = 0.9    # median log ratio is multiplied by this
SPREAD_WIDEN = 1.1     # quantile offsets from the median are multiplied by this


def parse_instant(text):
    """An ISO-8601 UTC timestamp as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_key(instant):
    """The canonical string the data uses for an hour's first instant."""
    return instant.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_curve(comparable):
    """A comparable launch's release-day counts, {hour 1..24: viewers}.

    Hours the source never sampled are simply absent. Counts are floored
    at 1 so log ratios stay finite.
    """
    by_start = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        by_start[point["from"]] = max(1.0, float(viewers))
    start = parse_instant(comparable["day_start"])
    curve = {}
    for hour in range(1, 25):
        key = hour_key(start + timedelta(hours=hour - 1))
        if key in by_start:
            curve[hour] = by_start[key]
    return curve


def is_launch(comparable):
    """True for a real past launch, False for the target's own pre-history."""
    name = (comparable.get("name") or "").lower()
    unlock = (comparable.get("unlock") or "").lower()
    return "before release" not in name and not unlock.startswith("not a launch")


def training_pool(case, target_log_p1):
    """Weighted log-ratio samples per hour from the case's comparables.

    Returns {hour: [(log ratio, weight), ...]}. Each comparable contributes
    its release-day curve divided by its own hour-1 count; its weight says
    how similar its opening hour was to this case's.
    """
    launches = []
    for comparable in case.get("comparables") or []:
        if not is_launch(comparable):
            continue
        curve = day_curve(comparable)
        if 1 not in curve or len(curve) < 20:
            continue
        first = curve[1]
        peak = max(curve.values())
        distance = math.log(first) - target_log_p1
        weight = math.exp(-(distance * distance) / (2.0 * KERNEL_BW * KERNEL_BW))
        launches.append((curve, first / peak, weight))
    for floor in (FRAC_MIN, FRAC_RELAXED, 0.0):
        chosen = [(curve, w) for curve, frac, w in launches if frac >= floor]
        if len(chosen) >= MIN_TRAIN or floor == 0.0:
            pool = {}
            for curve, weight in chosen:
                first = curve[1]
                for hour in range(2, 25):
                    if hour in curve:
                        pool.setdefault(hour, []).append(
                            (math.log(curve[hour] / first), weight))
            return pool
    return {}


def weighted_quantile(samples, tau):
    """The tau quantile of (value, weight) samples, linearly interpolated."""
    ordered = sorted(samples)
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    acc = 0.0
    for index, (value, weight) in enumerate(ordered):
        acc += weight
        if acc >= target:
            if index + 1 < len(ordered) and acc - target < 1e-12:
                return (value + ordered[index + 1][0]) / 2.0
            return value
    return ordered[-1][0]


def round2(value):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 100
    return int(result) if float(result).is_integer() else result


def clamp_row(values):
    """Force a quantile row into the valid range, non-decreasing."""
    row = []
    previous = 0.0
    for value in values:
        value = min(max(value, 0.01), 1e9)
        value = max(value, previous)
        row.append(round2(value))
        previous = float(row[-1])
    return row


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    first = next((row for row in case["known"] if row["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    pool = training_pool(case, math.log(p1))
    rows = []
    for request in case["forecast"]:
        hour = request["hour"]
        samples = pool.get(hour) or []
        if len(samples) >= 4:
            median = weighted_quantile(samples, 0.5)
            quantiles = []
            for tau in levels:
                offset = weighted_quantile(samples, tau) - median
                log_q = MEDIAN_SHRINK * median + SPREAD_WIDEN * offset
                quantiles.append(p1 * math.exp(log_q))
        else:
            # Persistence fallback: hour 1's count, random-walk spread.
            spread = FALLBACK_SIGMA * math.sqrt(hour - 1)
            quantiles = []
            for tau in levels:
                if tau in DEFAULT_LEVELS:
                    z = Z_LEVELS[DEFAULT_LEVELS.index(tau)]
                else:
                    z = 0.0
                quantiles.append(p1 * math.exp(z * spread))
        rows.append({"hour": hour, "q": clamp_row(quantiles)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
