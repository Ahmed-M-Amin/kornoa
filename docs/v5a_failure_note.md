# V5A Failure Note

V5A ConvNeXt-Tiny failed the controlled-improvement criteria:

```text
validation_f1 = 0.75855
threshold = 0.29
runtime_seconds = 31110.57
```

Conclusion:

```text
V5A is rejected.
Do not combine V5A with V2B or V3.
```

Likely reasons:

- It did not build on the accepted V2B EfficientNet-B1 baseline.
- It was too slow for the score it produced.
- The config was unstable.
- Focal loss plus weighted sampler likely overcorrected.
- The score/time tradeoff was poor.
