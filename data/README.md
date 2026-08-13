# Ulm district-heating data

`data_Ulm.csv` is a renamed copy of
[`data/data.csv`](https://github.com/finkenrm/deepDHC-user-guide/blob/main/data/data.csv)
from the public `finkenrm/deepDHC-user-guide` repository. It contains
anonymized, aggregate hourly district-heating load and weather observations for
Ulm, Germany and covers 70,928 consecutive UTC hours from 2014-09-02 00:00
through 2022-10-05 07:00.

The upstream repository is public and includes a permissive MIT-style
[`LICENSE`](https://github.com/finkenrm/deepDHC-user-guide/blob/main/LICENSE).
The license refers to software and associated documentation rather than stating
a separate data-specific license. Users should therefore retain its copyright
and permission notice and verify that its terms meet their intended data use.
Upstream copyright: © 2023 Fabian Behrens, Stefan Leiprecht, University of
Applied Sciences Kempten, Germany (deepDHC.de). This project's software license
does not replace those upstream terms.

The CSV includes an exported row index, timestamp (`MESS_DATUM`), load in MW,
observed temperature and dew point in °C, pressure in hPa, wind speed in m/s,
and wind direction in degrees. It also contains columns ending in `_Forecast`.
Those forecast columns are deliberately excluded from the public model input:
their issue times and forecast-lead structure cannot be verified, so using them
could introduce temporal leakage.

Run:

```bash
python src/scripts/prepare_ulm_data.py
```

This creates the ignored, reproducible `data_Ulm.pkl` with an hourly UTC
`DatetimeIndex` and the following columns:

| Column | Unit | Meaning |
|---|---:|---|
| `total_consumption` | kW | Aggregate thermal load |
| `temperature` | °C | Air temperature |
| `dewpoint` | °C | Dew-point temperature |
| `pressure` | hPa | Atmospheric pressure at station height |
| `wind_speed` | m/s | Wind speed |
| `wind_direction_sin` | 1 | Sine of wind direction |
| `wind_direction_cos` | 1 | Cosine of wind direction |

The source contains isolated values accidentally scaled by 1,000. The converter
repairs 9 load, 9 temperature, 10 dew-point, 11 wind-direction, and 11 wind-speed
values by dividing them by 1,000. It then converts load from MW to kW. The
result has no missing values; its load range is approximately 304--28,085 kW
and its temperature range is -20--34.6 °C.

The upstream project and forecasting context are described by Leiprecht et al.,
“Multi-Step Ahead Forecasting of Heat Load in District Heating Systems Using
Machine Learning Algorithms,” *Energy Reports* 7 (2021),
https://doi.org/10.1016/j.egyr.2021.08.140. This citation provides scientific
context; it is not a substitute for the data-permission statement above.

## Private network-expansion datasets

The `eurosun` and `malmoe` protocols expect their private pickle files at the
paths defined in `src/load_forecasting/protocols.py`. Both contain an hourly
`DatetimeIndex` and the same schema:

| Column | Unit | Meaning |
|---|---:|---|
| `demand` | kW | Aggregate thermal load |
| `temperature` | °C | Outdoor air temperature |
| `active_buildings` | count | Buildings active in the aggregate |

The Malmö file covers 2020-01-01 through 2025-12-31. The current EuroSun file
starts on 2020-04-25 and also contains observations after 2025; its protocol
deliberately ends the test set on 2025-12-31. These files are not included in
the public repository.

The runner treats these files as already resampled and quality-controlled. It
constructs a complete hourly index, interpolates only short temperature gaps,
and excludes forecast windows containing missing demand or long gaps. It does
not silently replace load outliers or long load gaps; such corrections require
a documented, dataset-specific preprocessing rule.
