# Trained weights

Not committed — they are large and regenerable.

## `plan_vectoriser.pt`

The learned floor-plan reader. Produced by
[`notebooks/train_plan_vectoriser_colab.ipynb`](../notebooks/train_plan_vectoriser_colab.ipynb)
on a free Colab T4 in about two hours.

Drop it here and stage 5 uses it. Leave it out and stage 5 uses the classical
engine. Both emit the same room masks and the harness scores them identically, so
you can measure whether the trained one is actually better:

```bash
python -m eval.harness --split dev --channel plan   # with, then without
```

The target from ROADMAP Sprint 3 is **≥80% room F1 on our own annotated plans** —
not on CubiCasa's validation split, which is Finnish and not what we receive.
