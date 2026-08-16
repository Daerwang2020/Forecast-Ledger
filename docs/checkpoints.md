# Local P2 checkpoints

The following checkpoint directories are downloaded under `ts-repro/checkpoints`
from the public Hugging Face repositories named in the model catalog. They are
kept separate from the official source checkouts and are not enabled as
scientific runs until their model-specific bridge has passed a real inference
closure.

| Model | Repository | Local directory | Size (model file) |
| --- | --- | --- | ---: |
| Chronos-2 | `amazon/chronos-2` | `checkpoints/chronos-2` | 477,930,472 bytes |
| TimesFM | `google/timesfm-2.5-200m-pytorch` | `checkpoints/timesfm-2.5-200m-pytorch` | 925,181,104 bytes |
| Moirai-2 | `Salesforce/moirai-2.0-R-small` | `checkpoints/moirai-2.0-R-small` | 45,557,824 bytes |
| TTM | `ibm-granite/granite-timeseries-ttm-r2` | `checkpoints/granite-timeseries-ttm-r2` | 3,240,592 bytes |
| Time-MoE | `Maple728/TimeMoE-50M` | `checkpoints/TimeMoE-50M` | 226,760,264 bytes |
| Timer | `thuml/timer-base-84m` | `checkpoints/timer-base-84m` | downloaded |

Timer's upstream README identifies the public zero-shot checkpoint as
`thuml/timer-base-84m`; the previous catalog value `THUML/Timer-XL` was not a
valid Hugging Face repository and has been corrected.

The downloaded files are inputs for interface verification only. They do not
constitute benchmark results or reproduce the papers' training recipes.
