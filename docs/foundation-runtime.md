# Foundation-model runtime pins

The Timer checkpoint README explicitly recommends Python 3.10 with
Transformers 4.40.1. Time-MoE's checkpoint metadata also declares
Transformers 4.40.1. With the current system Transformers 4.57.6, both remote
code models fail in `generate()` because the cache API removed legacy methods;
the failure is not a checkpoint problem.

Use a separate environment for these two models:

```text
python==3.10
transformers==4.40.1
tokenizers==0.19.1
```

In a controlled local test, both official calls then succeeded:

- Time-MoE: `generate(..., max_new_tokens=8)` → shape `(1, 72)`.
- Timer: `generate(..., max_new_tokens=8)` → shape `(1, 8)`.

Do not globally downgrade the shared environment: Chronos-2, TimesFM, TTM,
and the classic baselines have different dependency ranges. The catalog rows
carry this runtime pin as model metadata.
