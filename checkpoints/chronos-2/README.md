---
license: apache-2.0
model_id: chronos-2
tags:
  - time series
  - forecasting
  - foundation models
  - pretrained models
  - safetensors
paper:
  - https://arxiv.org/abs/2510.15821
datasets:
  - autogluon/chronos_datasets
  - Salesforce/GiftEvalPretrain
leaderboards:
  - Salesforce/GIFT-Eval
  - autogluon/fev-leaderboard
pipeline_tag: time-series-forecasting
library_name: chronos-forecasting

---

# Chronos-2

**Update Jun 5, 2026:** ☁️ Deploy Chronos-2 on AWS with AutoGluon-Cloud. Real-time, serverless, or batch inference in 3 lines of code — pandas DataFrames in, forecasts out. Check out the [new deployment guide](https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html).

**Chronos-2** is a 120M-parameter, encoder-only time series foundation model for zero-shot forecasting.
It supports **univariate**, **multivariate**, and **covariate-informed** tasks within a single architecture.
Inspired by the T5 encoder, Chronos-2 produces multi-step-ahead quantile forecasts and uses a group attention mechanism for efficient in-context learning across related series and covariates.
Trained on a combination of real-world and large-scale synthetic datasets, it achieves **state-of-the-art zero-shot accuracy** among public models on [**fev-bench**](https://huggingface.co/spaces/autogluon/fev-leaderboard), [**GIFT-Eval**](https://huggingface.co/spaces/Salesforce/GIFT-Eval), and [**Chronos Benchmark II**](https://arxiv.org/abs/2403.07815).
Chronos-2 is also **highly efficient**, delivering over 300 time series forecasts per second on a single A10G GPU and supporting both **GPU and CPU inference**.

## Links
- ☁️ [Deploy on SageMaker with AutoGluon-Cloud](https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html) (recommended)
- 🚀 [Deploy on SageMaker with JumpStart](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb)
- 📄 [Technical report](https://arxiv.org/abs/2510.15821v1)
- 💻 [GitHub](https://github.com/amazon-science/chronos-forecasting)
- 📘 [Example notebook](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb)
- 📰 [Amazon Science Blog](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)


## Overview

| Capability | Chronos-2 | Chronos-Bolt | Chronos |
|------------|-----------|--------------|----------|
| Univariate Forecasting | ✅ | ✅ | ✅ |
| Cross-learning across items | ✅ | ❌ | ❌ |
| Multivariate Forecasting | ✅ | ❌ | ❌ |
| Past-only (real/categorical) covariates | ✅ | ❌ | ❌ |
| Known future (real/categorical) covariates | ✅ | 🧩 | 🧩 |
| Max. Context Length | 8192 | 2048 | 512 |
| Max. Prediction Length | 1024 | 64 | 64 |

🧩 Chronos & Chronos-Bolt do not natively support future covariates, but they can be combined with external covariate regressors (see [AutoGluon tutorial](https://auto.gluon.ai/1.4.0/tutorials/timeseries/forecasting-chronos.html#incorporating-the-covariates)). This only models per-timestep effects, not effects across time. In contrast, Chronos-2 supports all covariate types natively.


## Running the model locally

For experimentation and local inference, you can use the [inference package](https://github.com/amazon-science/chronos-forecasting).

Install the package
```
pip install "chronos-forecasting>=2.0"
```

Make zero-shot predictions using the `pandas` API

```python
import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# Load historical target values and past values of covariates
context_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet")

# (Optional) Load future values of covariates
future_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet").drop(columns="target")

# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=24,  # Number of steps to forecast
    quantile_levels=[0.1, 0.5, 0.9],  # Quantiles for probabilistic forecast
    id_column="id",  # Column identifying different time series
    timestamp_column="timestamp",  # Column with datetime information
    target="target",  # Column(s) with time series values to predict
)
```

## Production use on Amazon SageMaker

For production use, we recommend deploying Chronos-2 to Amazon SageMaker. There are two options:

- **AutoGluon-Cloud** (recommended) — minimal setup with a high-level Python API: pass a pandas DataFrame in, get forecasts back. Supports real-time, serverless, and batch inference out of the box.
- **SageMaker JumpStart** — fine-grained control over the deployment configuration. JSON request/response payloads only; serverless inference and batch prediction require additional setup.

### ☁️ AutoGluon-Cloud

Install AutoGluon-Cloud:

```
pip install autogluon.cloud>=0.5.0
```

Make predictions from a pandas DataFrame

```python
from autogluon.cloud import TimeSeriesFoundationModel

model = TimeSeriesFoundationModel(model_name="chronos-2")

# Batch prediction
forecast_df = model.predict(df, prediction_length=24)

# Deploy & invoke a real-time endpoint
endpoint = model.deploy(instance_type="ml.g5.xlarge")
forecast_df = endpoint.predict(df, prediction_length=24)
```

For more details (e.g. serverless endpoints, covariate-aware forecasting), see the [full deployment guide](https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html).

### 🚀 SageMaker JumpStart

First, update the SageMaker SDK to make sure that all the latest models are available.

```
pip install -U 'sagemaker<3'
```

Deploy an inference endpoint to SageMaker.

```python
from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(
    model_id="pytorch-forecasting-chronos-2",
    instance_type="ml.g5.2xlarge",
)
predictor = model.deploy()
```

Now you can send time series data to the endpoint in JSON format.

```python
payload = {
    "inputs": [
        {"target": [1.0, 2.5, ..., 12.3]}
    ],
    "parameters": {
        "prediction_length": 24,
    }
}
forecast = predictor.predict(payload)["predictions"]
```

For more details about the endpoint API, check out the [example notebook](https://github.com/amazon-science/chronos-forecasting/blob/2.2.2/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb).


## Training data
More details about the training data are available in the [technical report](https://arxiv.org/abs/2510.15821).

- Subset of [Chronos Datasets](https://huggingface.co/datasets/autogluon/chronos_datasets) (excluding test portion of datasets that overlap with GIFT-Eval)
- Subset of [GIFT-Eval Pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
- Synthetic univariate and multivariate data


## Citation

If you find Chronos-2 useful for your research, please consider citing the associated paper:

```
@article{ansari2025chronos2,
  title        = {Chronos-2: From Univariate to Universal Forecasting},
  author       = {Abdul Fatir Ansari and Oleksandr Shchur and Jaris Küken and Andreas Auer and Boran Han and Pedro Mercado and Syama Sundar Rangapuram and Huibin Shen and Lorenzo Stella and Xiyuan Zhang and Mononito Goswami and Shubham Kapoor and Danielle C. Maddix and Pablo Guerron and Tony Hu and Junming Yin and Nick Erickson and Prateek Mutalik Desai and Hao Wang and Huzefa Rangwala and George Karypis and Yuyang Wang and Michael Bohlke-Schneider},
  year         = {2025},
  url          = {https://arxiv.org/abs/2510.15821}
}
```
