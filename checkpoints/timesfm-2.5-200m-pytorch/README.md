---
license: apache-2.0
library_name: timesfm
pipeline_tag: time-series-forecasting
---

# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google Research for time-series forecasting.

# Updates

* October 2, 2025: We changed the structure of the model to fuse QKV matrices into one for speed optimization.
  Please reinstall the latest version of the timesfm package to reflect these changes. Results should be unchanged.

**Resources and Technical Documentation**:
* Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688), ICML 2024.
* [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
* [GitHub repo](https://github.com/google-research/timesfm)

**Authors**: Google Research

This checkpoint is not an officially supported Google product. See [TimesFM in BigQuery](https://cloud.google.com/bigquery/docs/timesfm-model) for Google official support.

## Checkpoint `timesfm-2.5-200m`

`timesfm-2.5-200m` is the third open model checkpoint.


### Data

`timesfm-2.5-200m` is pretrained using

- [GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
- [Wikimedia Pageviews](https://meta.wikimedia.org/wiki/Pageviews_Analysis), cutoff Nov 2023 (see [paper](https://arxiv.org/abs/2310.10688) for details).
- [Google Trends](https://trends.google.com/trends/) top queries, cutoff EoY 2022 (see [paper](https://arxiv.org/abs/2310.10688) for details).
- Synthetic and augmented data.

### Install

`pip install` from PyPI coming soon. At this point, please run

```shell
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .
```

### Code Example

```python
import numpy as np
import timesfm
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=True)

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],  # Two dummy inputs
)
point_forecast.shape  # (2, 12)
quantile_forecast.shape  # (2, 12, 10): mean, then 10th to 90th quantiles.
```