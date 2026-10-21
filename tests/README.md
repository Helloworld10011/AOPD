# CPU checks

In the external-teacher environment, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

The tests exercise the actual buffer grouping and loss. They verify all-wrong versus mixed groups, leave-one-out rewards across buffer slices, correct KL scaling, padding exclusion, and zero gradients on ungated zero-advantage responses. They load no models or datasets.

`VALIDATION.json` also records release-time comparisons of loss values and gradients against the original source in both environments. Those comparisons use the private source trees and are not a claim of new training or benchmark reproduction.
