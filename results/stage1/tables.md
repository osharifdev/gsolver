### stops
| Dataset | Variant | Stop | Round | Remaining GN move: median / p95 | True-J move median | ATE (m) | Total flops | vs fixed | Total flops, alt. baseline |
|---|---|---|---|---|---|---|---|---|---|
| KITTI 06 (40) | corrected | certified | 130 | 0.0279 / 0.334 | 0.0279 | 0.0820 | 2.44e+11 | 1.36× | 4.32e+10 |
| KITTI 06 (40) | corrected | fixed budget | 100 | 0.0319 / 0.651 | 0.0319 | 0.0823 | 1.8e+11 | 1.00× | 2.55e+10 |
| KITTI 06 (40) | corrected | change tau 1e-02 | 142 | 0.0265 / 0.261 | 0.0265 | 0.0819 | 2.56e+11 | 1.42× | 3.65e+10 |
| KITTI 06 (40) | corrected | change tau 1e-03 | 237 | 0.0222 / 0.0871 | 0.0222 | 0.0817 | 4.27e+11 | 2.37× | 6.09e+10 |
| KITTI 06 (40) | corrected | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| KITTI 06 (40) | shipped | certified | 150 | 0.173 / 0.633 | 0.336 | 0.0817 | 2.82e+11 | 1.57× | 5.07e+10 |
| KITTI 06 (40) | shipped | fixed budget | 100 | 0.195 / 1.09 | 0.448 | 0.0825 | 1.8e+11 | 1.00× | 2.55e+10 |
| KITTI 06 (40) | shipped | change tau 1e-02 | 166 | 0.17 / 0.633 | 0.335 | 0.0816 | 2.99e+11 | 1.66× | 4.26e+10 |
| KITTI 06 (40) | shipped | change tau 1e-03 | 263 | 0.113 / 0.633 | 0.311 | 0.0814 | 4.74e+11 | 2.63× | 6.76e+10 |
| KITTI 06 (40) | shipped | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| KITTI 03 (40) | corrected | certified | 250 | 0.357 / 0.591 | 0.357 | 0.2474 | 6.74e+11 | 2.62× | 1.07e+11 |
| KITTI 03 (40) | corrected | fixed budget | 100 | 8.4 / 16.4 | 8.4 | 0.2454 | 2.57e+11 | 1.00× | 3.06e+10 |
| KITTI 03 (40) | corrected | change tau 1e-02 | 265 | 0.26 / 0.43 | 0.26 | 0.2475 | 6.82e+11 | 2.65× | 8.16e+10 |
| KITTI 03 (40) | corrected | change tau 1e-03 | not by 300 | – / – | – | – | – | – | – |
| KITTI 03 (40) | corrected | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| KITTI 03 (40) | shipped | certified | not by 300 | – / – | – | – | – | – | – |
| KITTI 03 (40) | shipped | fixed budget | 100 | 28.2 / 196 | 34.6 | 0.2267 | 2.57e+11 | 1.00× | 3.06e+10 |
| KITTI 03 (40) | shipped | change tau 1e-02 | not by 300 | – / – | – | – | – | – | – |
| KITTI 03 (40) | shipped | change tau 1e-03 | not by 300 | – / – | – | – | – | – | – |
| KITTI 03 (40) | shipped | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| KITTI 04 (40) | corrected | certified | 170 | 0.359 / 0.582 | 0.359 | 0.1331 | 5.54e+11 | 1.78× | 8.27e+10 |
| KITTI 04 (40) | corrected | fixed budget | 100 | 1.17 / 1.8 | 1.17 | 0.1328 | 3.11e+11 | 1.00× | 3.38e+10 |
| KITTI 04 (40) | corrected | change tau 1e-02 | 168 | 0.37 / 0.599 | 0.37 | 0.1331 | 5.23e+11 | 1.68× | 5.71e+10 |
| KITTI 04 (40) | corrected | change tau 1e-03 | not by 300 | – / – | – | – | – | – | – |
| KITTI 04 (40) | corrected | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| KITTI 04 (40) | shipped | certified | not by 300 | – / – | – | – | – | – | – |
| KITTI 04 (40) | shipped | fixed budget | 100 | 4.84 / 15.1 | 4.84 | 0.1361 | 3.11e+11 | 1.00× | 3.38e+10 |
| KITTI 04 (40) | shipped | change tau 1e-02 | not by 300 | – / – | – | – | – | – | – |
| KITTI 04 (40) | shipped | change tau 1e-03 | not by 300 | – / – | – | – | – | – | – |
| KITTI 04 (40) | shipped | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| KITTI 10 (40) | corrected | certified | 50 | 0.454 / 0.786 | 0.454 | 0.0850 | 1.18e+11 | 0.53× | 2.04e+10 |
| KITTI 10 (40) | corrected | fixed budget | 100 | 0.0819 / 0.258 | 0.0819 | 0.0853 | 2.24e+11 | 1.00× | 2.81e+10 |
| KITTI 10 (40) | corrected | change tau 1e-02 | 209 | 0.0119 / 0.0287 | 0.0119 | 0.0853 | 4.68e+11 | 2.09× | 5.92e+10 |
| KITTI 10 (40) | corrected | change tau 1e-03 | 273 | 0.00525 / 0.0126 | 0.00525 | 0.0853 | 6.12e+11 | 2.73× | 7.73e+10 |
| KITTI 10 (40) | corrected | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| KITTI 10 (40) | shipped | certified | 240 | 0.351 / 0.898 | 0.545 | 0.0902 | 5.63e+11 | 2.51× | 9.29e+10 |
| KITTI 10 (40) | shipped | fixed budget | 100 | 4.89 / 19.5 | 4.91 | 0.1009 | 2.24e+11 | 1.00× | 2.81e+10 |
| KITTI 10 (40) | shipped | change tau 1e-02 | not by 300 | – / – | – | – | – | – | – |
| KITTI 10 (40) | shipped | change tau 1e-03 | not by 300 | – / – | – | – | – | – | – |
| KITTI 10 (40) | shipped | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| ChArUco islam | corrected | certified | not by 1030 | – / – | – | – | – | – | – |
| ChArUco islam | corrected | fixed budget | 103 | 1.42 / 19 | 1.42 | – | 2.1e+09 | 1.00× | 1.81e+09 |
| ChArUco islam | corrected | change tau 1e-02 | not by 1030 | – / – | – | – | – | – | – |
| ChArUco islam | corrected | change tau 1e-03 | not by 1030 | – / – | – | – | – | – | – |
| ChArUco islam | corrected | change tau 1e-04 | not by 1030 | – / – | – | – | – | – | – |
| ChArUco islam | shipped | certified | not by 1030 | – / – | – | – | – | – | – |
| ChArUco islam | shipped | fixed budget | 103 | 1.51 / 20.5 | 1.49 | – | 2.1e+09 | 1.00× | 1.81e+09 |
| ChArUco islam | shipped | change tau 1e-02 | not by 1030 | – / – | – | – | – | – | – |
| ChArUco islam | shipped | change tau 1e-03 | not by 1030 | – / – | – | – | – | – | – |
| ChArUco islam | shipped | change tau 1e-04 | not by 1030 | – / – | – | – | – | – | – |
| KITTI 06 prefix (161) | corrected | certified | 130 | 0.0072 / 0.267 | 0.0072 | 0.3824 | 1.32e+12 | 1.54× | 3.55e+11 |
| KITTI 06 prefix (161) | corrected | fixed budget | 100 | 0.0151 / 0.624 | 0.0151 | 0.3827 | 8.56e+11 | 1.00× | 1.12e+11 |
| KITTI 06 prefix (161) | corrected | change tau 1e-02 | 194 | 0.00417 / 0.132 | 0.00417 | 0.3823 | 1.66e+12 | 1.94× | 2.19e+11 |
| KITTI 06 prefix (161) | corrected | change tau 1e-03 | 252 | 0.00406 / 0.0973 | 0.00406 | 0.3823 | 2.16e+12 | 2.52× | 2.84e+11 |
| KITTI 06 prefix (161) | corrected | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |
| KITTI 06 prefix (161) | shipped | certified | not by 300 | – / – | – | – | – | – | – |
| KITTI 06 prefix (161) | shipped | fixed budget | 100 | 0.377 / 5.21 | 0.519 | 0.3842 | 8.56e+11 | 1.00× | 1.12e+11 |
| KITTI 06 prefix (161) | shipped | change tau 1e-02 | not by 300 | – / – | – | – | – | – | – |
| KITTI 06 prefix (161) | shipped | change tau 1e-03 | not by 300 | – / – | – | – | – | – | – |
| KITTI 06 prefix (161) | shipped | change tau 1e-04 | not by 300 | – / – | – | – | – | – | – |

### eta
| Dataset | Variant | η = 0.05 | η = 0.1 | η = 0.2 | η = 0.5 |
|---|---|---|---|---|---|
| KITTI 06 (40) | corrected | r160: 0.0245 / 3e+11 | r130: 0.0279 / 2.44e+11 | r100: 0.0319 / 1.88e+11 | r60: 0.0401 / 1.13e+11 |
| KITTI 06 (40) | shipped | r190: 0.154 / 3.57e+11 | r150: 0.173 / 2.82e+11 | r120: 0.181 / 2.26e+11 | r90: 0.22 / 1.7e+11 |
| KITTI 03 (40) | corrected | r290: 0.158 / 7.8e+11 | r250: 0.357 / 6.74e+11 | r220: 0.662 / 5.94e+11 | r180: 1.53 / 4.86e+11 |
| KITTI 03 (40) | shipped | not by end (0.00 pass) | not by end (0.23 pass) | not by end (0.40 pass) | not by end (0.55 pass) |
| KITTI 04 (40) | corrected | r230: 0.157 / 7.48e+11 | r170: 0.359 / 5.54e+11 | r140: 0.564 / 4.57e+11 | r100: 1.17 / 3.26e+11 |
| KITTI 04 (40) | shipped | not by end (0.12 pass) | not by end (0.72 pass) | not by end (0.88 pass) | r250: 0.764 / 8.1e+11 |
| KITTI 10 (40) | corrected | r90: 0.102 / 2.11e+11 | r50: 0.454 / 1.18e+11 | r40: 0.671 / 9.51e+10 | r20: 1.23 / 4.74e+10 |
| KITTI 10 (40) | shipped | r250: 0.292 / 5.86e+11 | r240: 0.351 / 5.63e+11 | r200: 0.738 / 4.7e+11 | r170: 0.974 / 4.01e+11 |
| ChArUco islam | corrected | not by end (0.65 pass) | not by end (0.78 pass) | not by end (0.87 pass) | not by end (0.93 pass) |
| ChArUco islam | shipped | not by end (0.65 pass) | not by end (0.78 pass) | not by end (0.86 pass) | not by end (0.91 pass) |
| KITTI 06 prefix (161) | corrected | r150: 0.00509 / 1.51e+12 | r130: 0.0072 / 1.32e+12 | r100: 0.0151 / 1.04e+12 | r60: 0.0387 / 6.55e+11 |
| KITTI 06 prefix (161) | shipped | not by end (0.86 pass) | not by end (0.91 pass) | not by end (0.92 pass) | not by end (0.98 pass) |

### validity
| Dataset | Variant | Pose-evaluations | Violations upper / lower | Worst upper margin | Worst lower margin | Loewner min (rounds) | E/‖e‖ median / p95 | Certificate = oracle decisions | Growth exits (stop / continue / cap / steps) |
|---|---|---|---|---|---|---|---|---|---|
| KITTI 06 (40) | corrected | 1200 | 0 / 0 | 7.88e-16 | 4.34e-16 | 2.28e-13 (10,150,300) | 1.47 / 8 | 0 disagreements | 1123 / 77 / 0 / 0 |
| KITTI 06 (40) | shipped | 1200 | 0 / 0 | -4.32e-16 | 4.32e-16 | 3.87e-13 (10,150,300) | 1.6 / 11 | 0 disagreements | 1076 / 124 / 0 / 0 |
| KITTI 03 (40) | corrected | 1200 | 0 / 0 | 1.62e-15 | 7.26e-16 | 2.81e-16 (10,150,300) | 1.23 / 2.42 | 0 disagreements | 449 / 751 / 0 / 0 |
| KITTI 03 (40) | shipped | 1200 | 0 / 0 | 6.32e-15 | 8.25e-15 | 5.52e-17 (10,150,300) | 1.07 / 1.86 | 0 disagreements | 97 / 1103 / 0 / 0 |
| KITTI 04 (40) | corrected | 1200 | 0 / 0 | 1.04e-15 | 1.08e-15 | -1.05e-13 (10,150,300) | 1.31 / 3.21 | 0 disagreements | 715 / 485 / 0 / 0 |
| KITTI 04 (40) | shipped | 1200 | 0 / 0 | 1.19e-15 | 1.58e-15 | -1.17e-13 (10,150,300) | 1.12 / 1.73 | 0 disagreements | 191 / 1009 / 0 / 0 |
| KITTI 10 (40) | corrected | 1200 | 0 / 0 | -1.12e-16 | 5.76e-16 | 1.06e-14 (10,150,300) | 1.33 / 2.44 | 0 disagreements | 1085 / 115 / 0 / 0 |
| KITTI 10 (40) | shipped | 1200 | 0 / 0 | 2.22e-14 | 4.98e-15 | -4.43e-16 (10,150,300) | 1.23 / 2.55 | 0 disagreements | 539 / 661 / 0 / 0 |
| ChArUco islam | corrected | 9295 | 0 / 0 | 2.22e-10 | 1.82e-10 | -5.39e-09 (10,510,1030) | 1 / 1 | 0 disagreements | 5630 / 3665 / 0 / 0 |
| ChArUco islam | shipped | 9295 | 0 / 0 | 6.6e-11 | 2.09e-10 | -3.48e-09 (10,510,1030) | 1 / 1 | 0 disagreements | 5402 / 3893 / 0 / 0 |
| KITTI 06 prefix (161) | corrected | 4830 | 0 / 0 | -0.000147 | -0.000146 | 3.22e-13 (10,150,300) | 1.9 / 9.4 | 0 disagreements | 4383 / 447 / 0 / 0 |
| KITTI 06 prefix (161) | shipped | 4830 | 0 / 0 | -0.000702 | -0.000703 | 3.66e-13 (10,150,300) | 2.41 / 26.9 | 0 disagreements | 3725 / 1105 / 0 / 0 |

### curve
| Dataset | Variant | Fraction of poses passing at rounds 10, 20, 30, 50, 100, 150, 200, 250, 300 (ChArUco: 100, 200, …, 1000) |
|---|---|---|
| KITTI 06 (40) | corrected | 0.42 · 0.80 · 0.82 · 0.85 · 0.95 · 1.00 · 1.00 · 1.00 · 1.00 |
| KITTI 06 (40) | shipped | 0.00 · 0.55 · 0.78 · 0.82 · 0.85 · 1.00 · 1.00 · 1.00 · 1.00 |
| KITTI 03 (40) | corrected | 0.00 · 0.00 · 0.00 · 0.00 · 0.03 · 0.23 · 0.38 · 1.00 · 1.00 |
| KITTI 03 (40) | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 · 0.00 · 0.00 · 0.20 · 0.23 |
| KITTI 04 (40) | corrected | 0.00 · 0.00 · 0.00 · 0.05 · 0.10 · 0.82 · 1.00 · 1.00 · 1.00 |
| KITTI 04 (40) | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 · 0.03 · 0.07 · 0.35 · 0.72 |
| KITTI 10 (40) | corrected | 0.00 · 0.15 · 0.33 · 1.00 · 1.00 · 1.00 · 1.00 · 1.00 · 1.00 |
| KITTI 10 (40) | shipped | 0.00 · 0.00 · 0.00 · 0.03 · 0.03 · 0.28 · 0.75 · 1.00 · 1.00 |
| ChArUco islam | corrected | 0.17 · 0.32 · 0.48 · 0.62 · 0.71 · 0.73 · 0.77 · 0.77 · 0.77 · 0.78 |
| ChArUco islam | shipped | 0.17 · 0.28 · 0.41 · 0.54 · 0.68 · 0.71 · 0.74 · 0.77 · 0.77 · 0.77 |
| KITTI 06 prefix (161) | corrected | 0.16 · 0.53 · 0.57 · 0.73 · 0.99 · 1.00 · 1.00 · 1.00 · 1.00 |
| KITTI 06 prefix (161) | shipped | 0.00 · 0.00 · 0.18 · 0.51 · 0.84 · 0.91 · 0.89 · 0.91 · 0.91 |

### cost
| Dataset | Variant | GBP flops per round (median) | Alt. baseline per round | Certificate flops per evaluation: median (max) | Criterion (τ) flops per round | Evaluator s per evaluation (median) |
|---|---|---|---|---|---|---|
| KITTI 06 (40) | corrected | 1.8e+09 | 2.55e+08 | 6.36e+08 (2.18e+09) | 1.88e+06 | 0.60 |
| KITTI 06 (40) | shipped | 1.8e+09 | 2.55e+08 | 6.57e+08 (1.66e+09) | 1.88e+06 | 0.62 |
| KITTI 03 (40) | corrected | 2.57e+09 | 3.06e+08 | 1.17e+09 (2.49e+09) | 2.23e+06 | 0.77 |
| KITTI 03 (40) | shipped | 2.57e+09 | 3.06e+08 | 1.31e+09 (1.95e+09) | 2.23e+06 | 0.74 |
| KITTI 04 (40) | corrected | 3.11e+09 | 3.38e+08 | 1.25e+09 (3.07e+09) | 2.46e+06 | 0.78 |
| KITTI 04 (40) | shipped | 3.11e+09 | 3.38e+08 | 1.21e+09 (2.89e+09) | 2.46e+06 | 0.80 |
| KITTI 10 (40) | corrected | 2.24e+09 | 2.81e+08 | 6.74e+08 (1.61e+09) | 2.05e+06 | 0.63 |
| KITTI 10 (40) | shipped | 2.24e+09 | 2.81e+08 | 9.62e+08 (2.32e+09) | 2.05e+06 | 0.66 |
| ChArUco islam | corrected | 3.48e+07 | 3e+07 | 6.56e+09 (5.89e+10) | 1.68e+05 | 0.44 |
| ChArUco islam | shipped | 3.48e+07 | 3e+07 | 6.56e+09 (6.28e+10) | 1.68e+05 | 0.44 |
| KITTI 06 prefix (161) | corrected | 8.56e+09 | 1.12e+09 | 9.45e+09 (6.96e+10) | 8.19e+06 | 3.80 |
| KITTI 06 prefix (161) | shipped | 8.56e+09 | 1.12e+09 | 1.02e+10 (9.7e+10) | 8.19e+06 | 3.89 |

