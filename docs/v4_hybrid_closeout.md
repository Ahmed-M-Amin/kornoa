# V4.2 Hybrid Inference Closeout

## Result Summary

V4.2 is the current best public submission for this project.

| Submission | Public score | Notes |
| --- | ---: | --- |
| V4.2 hybrid | 0.92181 | Current best public submission |
| V2B classifier | 0.92121 | Previous best classifier baseline |
| V3 detector-only | 0.74169 | Weak as a standalone submission |

V4.2 improved over V2B by `+0.00060` public F1.

## V2B to V4.2 Diff

The V4.2 submission changed 32 rows compared with V2B. All changed rows moved from `0` to `1`, and every changed row used the detector decision source.

This confirms that the hybrid gate made a small, targeted correction set rather than broadly replacing classifier behavior.

## Conclusion

The detector helps slightly when used as a hybrid fallback for selected uncertain classifier cases. However, the improvement is small, and the detector-only V3 score remains weak at `0.74169`.

The final `0.98` target requires V5 strong classifier work rather than further relying on detector-only behavior.
