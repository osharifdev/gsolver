### Worst cases
| Quantity | Worst value | Where |
|---|---|---|
| validity, upper: (‖e‖ − E)/E (tolerance 1e-9) | 2.81e-10 | KITTI 03 shipped, round 20, POSE_7.141701000 |
| validity, lower: (L − ‖e‖)/‖e‖ (tolerance 1e-9) | 2.83e-10 | KITTI 03 shipped, round 20, POSE_6.002373000 |
| patch lower bound vs exact marginal, Loewner margin (≥ 0 required) | -1.79e-08 | KITTI 06 prefix (161) shipped, round 300, POSE_0.207793000 |
| oracle ‖e_i‖ / ηS_lo at the last certifiable evaluation (largest) | 1.27e+05 | KITTI 03 shipped, round 130, POSE_11.488980000 |
| oracle ‖e_i‖ / ηS_lo at the last certifiable evaluation (smallest: the pose closest to passing) | 7.78 | KITTI 04 corrected, round 300, POSE_0.000000000 |
| certified E_i / ηS_lo at the last certifiable evaluation (smallest) | 7.78 | KITTI 04 corrected, round 300, POSE_0.000000000 |
| Jacobian vs central differences, corrected variant (requirement 1e-5) | 4.35e-06 | kitti10/corrected: prior slot 0 cols 3-5 |
| anchor 1e4 → 1e6: pose solution change / ηS_lo (with p_L) | 5.69 | KITTI 06 |
| anchor 1e4 → 1e6: relative pose solution change (with p_L) | 0.148 | KITTI 10, POSE_10.275010000 |
| anchor 1e4 → 1e6: relative pose solution change (without p_L) | 5.53e-06 | KITTI 04, POSE_7.394135000 |
| certificate model vs solver graph (p_L gap) / ηS_lo | 8.71 | KITTI 06 |
| certificate model vs solver graph (p_L gap), largest landmark difference (m) | 2.47e+06 | KITTI 10 |
