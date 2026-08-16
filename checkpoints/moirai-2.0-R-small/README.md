---
license: cc-by-nc-4.0
pipeline_tag: time-series-forecasting
tags:
  - time series
  - forecasting
  - pretrained models
  - foundation models
  - time series foundation models
  - time-series
---

# Moirai-2.0-R-Small

Moirai 2.0 is a decoder-only universal time series forecasting transformer model pre-trained on:
- Subset of [GIFT-Eval Pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain), and [Train](https://huggingface.co/datasets/Salesforce/GiftEval) datasets (Non-leaking historical context). 
- Mixup data generated from non-leaking subsets of [Chronos Dataset](https://arxiv.org/abs/2403.07815).
- Synthetic time series produced via KernelSynth introduced in [Chronos paper](https://arxiv.org/abs/2403.07815).
- Internal Salesforce operational data.

We make significant improvements over the first version of Moirai (please refer to the [paper](https://arxiv.org/abs/2402.02592) for previous version):
- Switched from a distributional loss to a quantile loss formulation.
- Moved from single-token to multi-token prediction, improving efficiency and stability.
- Added a data filtering mechanism to filter out non-forecastable, low quality, time series during pretraining.
- Added a new patch token embedding which includes missing value information.
- Added patch-level random mask to improve robustness of the model during inference.

## Usage
To perform inference with Moirai 2.0, install the uni2ts library from our [GitHub repo](https://github.com/SalesforceAIResearch/uni2ts).

1. Clone repository:
```shell
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
```

2) Create virtual environment:
```shell
virtualenv venv
. venv/bin/activate
```

3) Build from source:
```shell
pip install -e '.[notebook]'
```

4) Create a `.env` file:
```shell
touch .env
```

A simple notebook to get started: [github_notebook_link](https://github.com/SalesforceAIResearch/uni2ts/blob/main/example/moirai_forecast.ipynb)

## Citation

If you're using any Moirai model or Uni2TS in your research or applications, please cite it using this BibTeX:

```markdown
@article{liu2025moirai,
  title={Moirai 2.0: When less is more for time series forecasting},
  author={Liu, Chenghao and Aksu, Taha and Liu, Juncheng and Liu, Xu and Yan, Hanshu and Pham, Quang and Savarese, Silvio and Sahoo, Doyen and Xiong, Caiming and Li, Junnan},
  journal={arXiv preprint arXiv:2511.11698},
  year={2025}
}
```

## Ethical Considerations

This release is for research purposes only. The proprietary version of this model is used by Salesforce for business purposes. Our research models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people's lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.