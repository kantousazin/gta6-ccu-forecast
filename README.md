# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kantousazin/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/kantousazin/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for the Twitch audience of a big game launch. Given everything
recorded up to the end of hour 1 of a game's release day — that hour's viewer
count included — plus the release-day curves of past big launches, `main.py`
writes five quantiles (5%, 25%, 50%, 75%, 95%) for each of hours 2 to 24.

It learns the day's shape from the comparable launches inside each case:
every past launch's curve is divided by its own first hour, launches that
were already live at hour 1 (a rolling local-midnight unlock, like GTA VI's)
are kept, launches are weighted by how close their opening hour was to this
case's, and each hour's forecast is the weighted empirical quantiles of the
training launches' log ratios, with the median shrunk slightly toward
persistence and the spread widened slightly. Both constants, the filter and
the kernel width were chosen by leave-one-out cross-validation over the
comparable launches themselves. Pure Python 3.13 standard library,
deterministic, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=et3tdj7w&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
