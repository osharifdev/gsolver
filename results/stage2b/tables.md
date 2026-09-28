### Newton
| Stop | Iterations | Converged | Final ‖F‖ | Pose move (D): median / p90 / p99 / max | Pose move / ηS_lo: median / max | Poses truly finished | Landmark move (D): median / p90 / p99 / max | Landmark move (m): median / p90 / p99 / max | Landmarks > c·0.1 m (c = 1, 2, 4, 8, 16) | Implemented equation's zero vs x̂ (poses, D): median / max |
|---|---|---|---|---|---|---|---|---|---|---|
| kitti06 r130 | 7 | yes | 2.84e-08 | 0.149 / 0.366 / 0.513 / 0.515 | 0.31 / 1.03 | 95.0 % | 0.025 / 511 / 1.59e+04 / 3.39e+05 | 0.00209 / 0.561 / 17.1 / 1.62e+03 | 961 / 726 / 551 / 389 / 266 | 3.97e-07 / 2.43e-06 |
| kitti03 r250 | 8 | yes | 5.53e-08 | 0.151 / 0.341 / 0.452 / 0.498 | 0.308 / 1.02 | 97.5 % | 0.0183 / 113 / 5.84e+03 / 3.19e+05 | 0.00108 / 0.17 / 9.3 / 1.05e+03 | 516 / 440 / 342 / 249 / 162 | 1.42e-06 / 5.27e-06 |
| kitti04 r170 | 8 | yes | 2.27e-08 | 0.417 / 0.837 / 0.963 / 0.978 | 0.871 / 1.96 | 57.5 % | 0.0308 / 0.278 / 8.57e+03 / 1.07e+05 | 0.00208 / 0.121 / 8.6 / 154 | 477 / 419 / 332 / 263 / 189 | 2.71e-06 / 6.42e-06 |
| kitti10 r50 | 7 | yes | 3.41e-08 | 0.341 / 0.439 / 0.465 / 0.465 | 0.691 / 0.931 | 100.0 % | 0.0922 / 116 / 4.26e+03 / 1.75e+05 | 0.00593 / 0.185 / 5 / 175 | 536 / 408 / 299 / 199 / 127 | 4.76e-06 / 1.21e-05 |

### T1 attempts
| Stop | Option | c | Verified | Landmarks with s_l > ū_l | Unbounded factor enclosures | Poses with infinite Ψ | Finite Ψ/ū, poses: max | s_i/ū_i, poses: median / max |
|---|---|---|---|---|---|---|---|---|
| kitti06 | A | 1 | no | 705 | 247 | 40 | inf | 0.842 / 1.67 |
| kitti06 | A | 2 | no | 604 | 681 | 40 | inf | 0.421 / 0.833 |
| kitti06 | A | 4 | no | 478 | 2142 | 40 | inf | 0.211 / 0.416 |
| kitti06 | A | 8 | no | 346 | 6385 | 40 | inf | 0.105 / 0.208 |
| kitti06 | A | 16 | no | 245 | 9192 | 40 | inf | 0.0527 / 0.104 |
| kitti06 | B | 1 | no | 0 | 69 | 40 | inf | 0.842 / 1.67 |
| kitti06 | B | 2 | no | 0 | 120 | 40 | inf | 0.421 / 0.833 |
| kitti06 | B | 4 | no | 0 | 230 | 40 | inf | 0.211 / 0.416 |
| kitti06 | B | 8 | no | 0 | 332 | 40 | inf | 0.105 / 0.208 |
| kitti06 | B | 16 | no | 0 | 553 | 40 | inf | 0.0527 / 0.104 |
| kitti03 | A | 1 | no | 418 | 445 | 40 | inf | 0.158 / 0.419 |
| kitti03 | A | 2 | no | 358 | 1376 | 40 | inf | 0.0789 / 0.21 |
| kitti03 | A | 4 | no | 294 | 3937 | 40 | inf | 0.0394 / 0.105 |
| kitti03 | A | 8 | no | 207 | 9775 | 40 | inf | 0.0197 / 0.0524 |
| kitti03 | A | 16 | no | 134 | 12613 | 40 | inf | 0.00986 / 0.0262 |
| kitti03 | B | 1 | no | 0 | 20 | 40 | inf | 0.158 / 0.419 |
| kitti03 | B | 2 | no | 0 | 52 | 40 | inf | 0.0789 / 0.21 |
| kitti03 | B | 4 | no | 0 | 123 | 40 | inf | 0.0394 / 0.105 |
| kitti03 | B | 8 | no | 0 | 255 | 40 | inf | 0.0197 / 0.0524 |
| kitti03 | B | 16 | no | 0 | 490 | 40 | inf | 0.00986 / 0.0262 |
| kitti04 | A | 1 | no | 390 | 505 | 40 | inf | 0.23 / 0.731 |
| kitti04 | A | 2 | no | 341 | 1373 | 40 | inf | 0.115 / 0.366 |
| kitti04 | A | 4 | no | 302 | 3512 | 40 | inf | 0.0575 / 0.183 |
| kitti04 | A | 8 | no | 244 | 9515 | 40 | inf | 0.0288 / 0.0914 |
| kitti04 | A | 16 | no | 180 | 13455 | 40 | inf | 0.0144 / 0.0457 |
| kitti04 | B | 1 | no | 0 | 49 | 40 | inf | 0.23 / 0.731 |
| kitti04 | B | 2 | no | 0 | 95 | 40 | inf | 0.115 / 0.366 |
| kitti04 | B | 4 | no | 0 | 164 | 40 | inf | 0.0575 / 0.183 |
| kitti04 | B | 8 | no | 0 | 248 | 40 | inf | 0.0288 / 0.0914 |
| kitti04 | B | 16 | no | 0 | 382 | 40 | inf | 0.0144 / 0.0457 |
| kitti10 | A | 1 | no | 355 | 700 | 40 | inf | 0.0818 / 0.306 |
| kitti10 | A | 2 | no | 306 | 2170 | 40 | inf | 0.0409 / 0.153 |
| kitti10 | A | 4 | no | 238 | 6474 | 40 | inf | 0.0205 / 0.0765 |
| kitti10 | A | 8 | no | 168 | 10650 | 40 | inf | 0.0102 / 0.0383 |
| kitti10 | A | 16 | no | 112 | 11826 | 40 | inf | 0.00512 / 0.0191 |
| kitti10 | B | 1 | no | 0 | 34 | 40 | inf | 0.0818 / 0.306 |
| kitti10 | B | 2 | no | 0 | 106 | 40 | inf | 0.0409 / 0.153 |
| kitti10 | B | 4 | no | 0 | 333 | 40 | inf | 0.0205 / 0.0765 |
| kitti10 | B | 8 | no | 0 | 753 | 40 | inf | 0.0102 / 0.0383 |
| kitti10 | B | 16 | no | 0 | 1508 | 40 | inf | 0.00512 / 0.0191 |

### Evaluator checks
| Stop | F(0) vs solver, stereo: median / max | F(0) vs solver, GP: median / max | F′ vs central differences | Point enclosure contains doubles | Random box points outside enclosure |
|---|---|---|---|---|---|
| kitti06 | 1.02e-09 / 1.45e-08 | 1.84e-07 / 4.13e-06 | 2.2e-08 | yes | 0 / 0 |
| kitti03 | 2.17e-09 / 2.1e-08 | 1.66e-07 / 3.96e-06 | 1.59e-08 | yes | 0 / 0 |
| kitti04 | 4.42e-10 / 5.23e-09 | 2.17e-07 / 5.3e-06 | 3.32e-08 | yes | 0 / 0 |
| kitti10 | 8.51e-13 / 4.98e-08 | 4.47e-08 / 4.77e-06 | 3.32e-09 | yes | 0 / 0 |
