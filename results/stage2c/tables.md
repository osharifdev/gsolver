### Step 1: counts
| Dataset | Landmarks before | Removed (one observation) | Remaining | Stereo factors before | Removed | Remaining | Poses left without observations |
|---|---|---|---|---|---|---|---|
| KITTI 06 | 4679 | 1397 | 3282 | 11487 | 1397 | 10090 | 0 |
| KITTI 03 | 4628 | 533 | 4095 | 13807 | 533 | 13274 | 0 |
| KITTI 04 | 4652 | 463 | 4189 | 15267 | 463 | 14804 | 0 |
| KITTI 10 | 4228 | 545 | 3683 | 12680 | 545 | 12135 | 0 |

### Step 1.4: anchor check (corrected variant, round-100 state)
| Dataset | Pose solution change 1e4 → 1e6: max relative (worst pose) | max in D / ηS_lo | Landmark max relative | λ_min(H) 1e4 / 1e6 | λ_min(S) 1e4 / 1e6 | Without p_L: max relative; max D / ηS_lo | p_L gap at 1e4: max D / ηS_lo; poses above ηS_lo |
|---|---|---|---|---|---|---|---|
| KITTI 06 | 0.0516 (POSE_5.299899000) | 5.69 | 0.152 | 1e-06 / 1e-06 | 0.0856 / 7.52 | 1.11e-06; 0.000149 | 8.71; 39 of 40 |
| KITTI 03 | 0.000809 (POSE_14.692450000) | 0.201 | 0.0132 | 1.02e-06 / 1.02e-06 | 0.127 / 8.19 | 5.83e-08; 1.36e-05 | 0.278; 0 of 40 |
| KITTI 04 | 0.00167 (POSE_7.394135000) | 0.468 | 0.0107 | 1.55e-06 / 1.55e-06 | 0.0692 / 5.54 | 5.53e-06; 0.00156 | 0.574; 0 of 40 |
| KITTI 10 | 0.148 (POSE_10.275010000) | 2.85 | 0.531 | 1e-06 / 1e-06 | 0.204 / 2.96 | 2.7e-06; 7.55e-05 | 3.01; 37 of 40 |

### Step 2: stops
| Dataset | Variant | Certified | Fixed 100: move median / p95, ATE | τ = 1e-2 | τ = 1e-3 | τ = 1e-4 | Largest belief change, round 300 |
|---|---|---|---|---|---|---|---|
| KITTI 06 | corrected | not by 300 | 27.5 / 75.2, 0.080 m | not by 300 | not by 300 | not by 300 | 0.355 |
| KITTI 06 | shipped | not by 300 | 51.7 / 116, 0.072 m | not by 300 | not by 300 | not by 300 | 0.515 |
| KITTI 03 | corrected | not by 300 | 106 / 242, 0.284 m | not by 300 | not by 300 | not by 300 | 5.08 |
| KITTI 03 | shipped | not by 300 | 4.2e+03 / 7.93e+04, 11.616 m | fires at round 151 on a non-finite (diverged) state: not a stop | fires at round 151 on a non-finite (diverged) state: not a stop | fires at round 151 on a non-finite (diverged) state: not a stop | 0 |
| KITTI 04 | corrected | not by 300 | 37.2 / 104, 0.129 m | not by 300 | not by 300 | not by 300 | 0.025 |
| KITTI 04 | shipped | not by 300 | 66.6 / 185, 0.197 m | not by 300 | not by 300 | not by 300 | 0.112 |
| KITTI 10 | corrected | not by 300 | 12.4 / 20.8, 0.081 m | not by 300 | not by 300 | not by 300 | 6.12 |
| KITTI 10 | shipped | not by 300 | 102 / 193, 0.196 m | not by 300 | not by 300 | not by 300 | 0.807 |
| KITTI 06 prefix (161) | corrected | not by 300 | 250 / 576, 0.364 m | not by 300 | not by 300 | not by 300 | 6.93 |
| KITTI 06 prefix (161) | shipped | not by 300 | 760 / 1.57e+03, 0.670 m | not by 300 | not by 300 | not by 300 | 656 |

### Step 2: validity and decisions
| Dataset | Variant | Pose-evaluations | Violations upper / lower | Worst upper margin (round, pose) | Worst lower margin | Loewner min (rounds) | Certificate ≠ oracle | Rounds not certifiable | Growth exits stop / continue / cap / steps |
|---|---|---|---|---|---|---|---|---|---|
| KITTI 06 | corrected | 1200 | 0 / 0 | 2.06e-12 (r10, POSE_2.285652000) | 1.92e-12 | -1.17e-09 (10,150,300) | 0 | 0 | 0 / 1200 / 0 / 0 |
| KITTI 06 | shipped | 1200 | 0 / 0 | 9.68e-12 (r10, POSE_2.285652000) | 4.53e-12 | -6.11e-10 (10,150,300) | 0 | 0 | 0 / 1200 / 0 / 0 |
| KITTI 03 | corrected | 1200 | 0 / 0 | 7.69e-11 (r10, POSE_0.000000000) | 4.56e-11 | -2.11e-10 (10,150,300) | 0 | 0 | 0 / 1200 / 0 / 0 |
| KITTI 03 | shipped | 520 | 0 / 0 | 2.81e-10 (r20, POSE_7.141701000) | 2.83e-10 | -3.84e-10 (10) | 0 | 17 (140–300) | 0 / 520 / 0 / 0 |
| KITTI 04 | corrected | 1200 | 0 / 0 | 7.71e-13 (r10, POSE_0.000000000) | 6.2e-13 | -1.14e-09 (10,150,300) | 0 | 0 | 0 / 1200 / 0 / 0 |
| KITTI 04 | shipped | 1200 | 0 / 0 | 1.56e-11 (r10, POSE_3.854131000) | 7.26e-12 | -1.92e-09 (10,150,300) | 0 | 0 | 0 / 1200 / 0 / 0 |
| KITTI 10 | corrected | 1200 | 0 / 0 | 4.52e-12 (r20, POSE_9.652155000) | 5.11e-12 | -3.07e-10 (10,150,300) | 0 | 0 | 0 / 1200 / 0 / 0 |
| KITTI 10 | shipped | 1200 | 0 / 0 | 1.01e-11 (r50, POSE_10.586510000) | 1.61e-11 | -3.12e-10 (10,150,300) | 0 | 0 | 0 / 1200 / 0 / 0 |
| KITTI 06 prefix (161) | corrected | 4830 | 0 / 0 | 2.53e-11 (r20, POSE_14.449650000) | 4.32e-11 | -8.84e-09 (10,150,300) | 0 | 0 | 0 / 4654 / 0 / 15 |
| KITTI 06 prefix (161) | shipped | 4830 | 0 / 0 | 1.87e-10 (r60, POSE_7.899123000) | 2.01e-10 | -1.79e-08 (10,150,300) | 0 | 0 | 0 / 4661 / 0 / 8 |

### Step 2: η sensitivity (first round at which every pose passes E_i ≤ η S_lo,i)
| Dataset | Variant | η = 0.05 | η = 0.1 | η = 0.2 | η = 0.5 |
|---|---|---|---|---|---|
| KITTI 06 | corrected | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 06 | shipped | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 03 | corrected | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 03 | shipped | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 04 | corrected | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 04 | shipped | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 10 | corrected | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 10 | shipped | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 06 prefix (161) | corrected | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |
| KITTI 06 prefix (161) | shipped | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) | none (0.00 pass at the end) |

### Step 2: pass-fraction curve and oracle
| Dataset | Variant | Certified pass fraction at rounds 10, 50, 100, 200, 300 | Oracle pass fraction, same rounds | E/‖e‖ median |
|---|---|---|---|---|
| KITTI 06 | corrected | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |
| KITTI 06 | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |
| KITTI 03 | corrected | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |
| KITTI 03 | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · – · – | 1.000000 |
| KITTI 04 | corrected | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |
| KITTI 04 | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |
| KITTI 10 | corrected | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |
| KITTI 10 | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |
| KITTI 06 prefix (161) | corrected | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |
| KITTI 06 prefix (161) | shipped | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 · 0.00 · 0.00 | 1.000000 |

### Step 2: costs
| Dataset | Variant | GBP flops per round (median) | Belief-minus-incoming baseline per round | Certificate flops per evaluation: median (max) | Criterion flops per round | Evaluator s per evaluation (median) | GBP flops, 300 rounds | Certificate flops, 30 evaluations |
|---|---|---|---|---|---|---|---|---|
| KITTI 06 | corrected | 1.42e+09 | 2.25e+08 | 5.17e+08 (4.41e+09) | 1.64e+06 | 0.5 | 4.27e+11 | 1.94e+10 |
| KITTI 06 | shipped | 1.42e+09 | 2.25e+08 | 5.17e+08 (4.25e+09) | 1.64e+06 | 0.5 | 4.27e+11 | 1.92e+10 |
| KITTI 03 | corrected | 2.39e+09 | 2.94e+08 | 5.17e+08 (4.34e+09) | 2.14e+06 | 0.7 | 7.18e+11 | 1.93e+10 |
| KITTI 03 | shipped | 2.39e+09 | 2.94e+08 | 5.17e+08 (4.13e+09) | 2.14e+06 | 0.6 | 7.18e+11 | 1.03e+10 |
| KITTI 04 | corrected | 2.95e+09 | 3.28e+08 | 5.17e+08 (4.96e+09) | 2.38e+06 | 0.7 | 8.84e+11 | 1.99e+10 |
| KITTI 04 | shipped | 2.95e+09 | 3.28e+08 | 5.17e+08 (4.82e+09) | 2.38e+06 | 0.7 | 8.84e+11 | 1.98e+10 |
| KITTI 10 | corrected | 2.07e+09 | 2.69e+08 | 5.17e+08 (4.09e+09) | 1.96e+06 | 0.6 | 6.22e+11 | 1.91e+10 |
| KITTI 10 | shipped | 2.07e+09 | 2.69e+08 | 5.17e+08 (4.27e+09) | 1.96e+06 | 0.6 | 6.22e+11 | 1.92e+10 |
| KITTI 06 prefix (161) | corrected | 7.3e+09 | 1.03e+09 | 3.37e+10 (9.29e+11) | 7.46e+06 | 3.7 | 2.19e+12 | 2.59e+12 |
| KITTI 06 prefix (161) | shipped | 7.3e+09 | 1.03e+09 | 3.37e+10 (8.39e+11) | 7.46e+06 | 3.6 | 2.19e+12 | 2.48e+12 |

### Step 3.1: budgets (fraction of poses unfinished: current / relinearized; median and p95 remaining move; ATE)
| Dataset | Variant | 10 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|
| KITTI 06 | corrected | 1.00 / 1.00; 30.5, 77.6; 0.087 m | 1.00 / 1.00; 28.4, 76.9; 0.083 m | 1.00 / 1.00; 27.5, 75.2; 0.080 m | 1.00 / 1.00; 27.2, 71.9; 0.078 m | 1.00 / 1.00; 27.6, 69.7; 0.077 m |
| KITTI 06 | shipped | 1.00 / 1.00; 44.6, 111; 0.095 m | 1.00 / 1.00; 51.4, 117; 0.079 m | 1.00 / 1.00; 51.7, 116; 0.072 m | 1.00 / 1.00; 52.3, 116; 0.066 m | 1.00 / 1.00; 53.2, 116; 0.063 m |
| KITTI 03 | corrected | 1.00 / 1.00; 98.4, 258; 0.454 m | 1.00 / 1.00; 115, 244; 0.332 m | 1.00 / 1.00; 106, 242; 0.284 m | 1.00 / 1.00; 113, 229; 0.266 m | 1.00 / 1.00; 113, 220; 0.258 m |
| KITTI 03 | shipped | 1.00 / 1.00; 2.93e+03, 1.78e+04; 1.836 m | 1.00 / 1.00; 5.37e+04, 9.17e+04; 3.712 m | 1.00 / 1.00; 4.2e+03, 7.93e+04; 11.616 m | 1.00 / – (diverged) | 1.00 / – (diverged) |
| KITTI 04 | corrected | 1.00 / 1.00; 38.1, 104; 0.138 m | 1.00 / 1.00; 37.2, 106; 0.131 m | 1.00 / 1.00; 37.2, 104; 0.129 m | 1.00 / 1.00; 37.3, 102; 0.128 m | 1.00 / 1.00; 36.9, 99.7; 0.127 m |
| KITTI 04 | shipped | 1.00 / 1.00; 118, 300; 0.232 m | 1.00 / 1.00; 61.2, 188; 0.202 m | 1.00 / 1.00; 66.6, 185; 0.197 m | 1.00 / 1.00; 70.9, 183; 0.195 m | 1.00 / 1.00; 71.5, 180; 0.193 m |
| KITTI 10 | corrected | 1.00 / 1.00; 24.6, 41.1; 0.077 m | 1.00 / 1.00; 13.7, 27.9; 0.077 m | 1.00 / 1.00; 12.4, 20.8; 0.081 m | 1.00 / 1.00; 11.4, 15.7; 0.083 m | 1.00 / 1.00; 10.9, 15.5; 0.083 m |
| KITTI 10 | shipped | 1.00 / 1.00; 477, 689; 0.389 m | 1.00 / 1.00; 104, 228; 0.219 m | 1.00 / 1.00; 102, 193; 0.196 m | 1.00 / 1.00; 119, 171; 0.160 m | 1.00 / 1.00; 132, 162; 0.140 m |
| KITTI 06 prefix (161) | corrected | 1.00 / 1.00; 258, 582; 0.389 m | 1.00 / 1.00; 258, 578; 0.374 m | 1.00 / 1.00; 250, 576; 0.364 m | 1.00 / 1.00; 245, 573; 0.350 m | 1.00 / 1.00; 240, 571; 0.342 m |
| KITTI 06 prefix (161) | shipped | 1.00 / 1.00; 2.42e+03, 5.22e+03; 0.463 m | 1.00 / 1.00; 589, 1.19e+03; 0.491 m | 1.00 / 1.00; 760, 1.57e+03; 0.670 m | 1.00 / 1.00; 1.08e+03, 2.24e+03; 0.637 m | 1.00 / 1.00; 1.25e+03, 2.61e+03; 0.832 m |

### Step 4: Gate C
| Dataset | Certified stop | Poses within ηS_lo of x̂ | Passes |
|---|---|---|---|
| KITTI 06 | none by round 300 | – | no |
| KITTI 03 | none by round 300 | – | no |
| KITTI 04 | none by round 300 | – | no |
| KITTI 10 | none by round 300 | – | no |

### Old model against the clean model
| Quantity | Dataset | Variant | Old | Clean |
|---|---|---|---|---|
| landmarks in the model | KITTI 06 | both | 4.68e+03 | 3.28e+03 |
| stereo factors in the model | KITTI 06 | both | 1.15e+04 | 1.01e+04 |
| certified stop round (300-round runs) | KITTI 06 | corrected | 130 | none |
| unfinished fraction, current problem, at round 100 | KITTI 06 | corrected | 0.05 | 1 |
| median remaining GN move (D) at round 100 | KITTI 06 | corrected | 0.0319 | 27.5 |
| aligned keyframe ATE (m) at round 100 | KITTI 06 | corrected | 0.0823 | 0.0798 |
| unfinished fraction, current problem, at round 300 | KITTI 06 | corrected | 0 | 1 |
| median remaining GN move (D) at round 300 | KITTI 06 | corrected | 0.022 | 27.6 |
| aligned keyframe ATE (m) at round 300 | KITTI 06 | corrected | 0.0817 | 0.0775 |
| GBP flops per round (median) | KITTI 06 | corrected | 1.8e+09 | 1.42e+09 |
| certificate flops per evaluation (median) | KITTI 06 | corrected | 6.36e+08 | 5.17e+08 |
| certified stop round (300-round runs) | KITTI 06 | shipped | 150 | none |
| unfinished fraction, current problem, at round 100 | KITTI 06 | shipped | 0.15 | 1 |
| median remaining GN move (D) at round 100 | KITTI 06 | shipped | 0.195 | 51.7 |
| aligned keyframe ATE (m) at round 100 | KITTI 06 | shipped | 0.0825 | 0.0716 |
| unfinished fraction, current problem, at round 300 | KITTI 06 | shipped | 0 | 1 |
| median remaining GN move (D) at round 300 | KITTI 06 | shipped | 0.113 | 53.2 |
| aligned keyframe ATE (m) at round 300 | KITTI 06 | shipped | 0.0814 | 0.063 |
| GBP flops per round (median) | KITTI 06 | shipped | 1.8e+09 | 1.42e+09 |
| certificate flops per evaluation (median) | KITTI 06 | shipped | 6.57e+08 | 5.17e+08 |
| Newton check: stop state | KITTI 06 | corrected | round 130 | no certified stop |
| Newton check: poses within eta S_lo of x^ | KITTI 06 | corrected | 0.95 | no certified stop |
| Newton check: pose distance / E, median; max | KITTI 06 | corrected | 3.78; 6.76 | no certified stop |
| Newton check: largest landmark move (m) | KITTI 06 | corrected | 1.62e+03 | no certified stop |
| landmarks in the model | KITTI 03 | both | 4.63e+03 | 4.1e+03 |
| stereo factors in the model | KITTI 03 | both | 1.38e+04 | 1.33e+04 |
| certified stop round (300-round runs) | KITTI 03 | corrected | 250 | none |
| unfinished fraction, current problem, at round 100 | KITTI 03 | corrected | 0.975 | 1 |
| median remaining GN move (D) at round 100 | KITTI 03 | corrected | 8.4 | 106 |
| aligned keyframe ATE (m) at round 100 | KITTI 03 | corrected | 0.245 | 0.284 |
| unfinished fraction, current problem, at round 300 | KITTI 03 | corrected | 0 | 1 |
| median remaining GN move (D) at round 300 | KITTI 03 | corrected | 0.131 | 113 |
| aligned keyframe ATE (m) at round 300 | KITTI 03 | corrected | 0.248 | 0.258 |
| GBP flops per round (median) | KITTI 03 | corrected | 2.57e+09 | 2.39e+09 |
| certificate flops per evaluation (median) | KITTI 03 | corrected | 1.17e+09 | 5.17e+08 |
| certified stop round (300-round runs) | KITTI 03 | shipped | none | none |
| unfinished fraction, current problem, at round 100 | KITTI 03 | shipped | 1 | 1 |
| median remaining GN move (D) at round 100 | KITTI 03 | shipped | 28.2 | 4.2e+03 |
| aligned keyframe ATE (m) at round 100 | KITTI 03 | shipped | 0.227 | 11.6 |
| unfinished fraction, current problem, at round 300 | KITTI 03 | shipped | 0.775 | 1 |
| median remaining GN move (D) at round 300 | KITTI 03 | shipped | 1.45 | – |
| aligned keyframe ATE (m) at round 300 | KITTI 03 | shipped | 0.247 | – |
| GBP flops per round (median) | KITTI 03 | shipped | 2.57e+09 | 2.39e+09 |
| certificate flops per evaluation (median) | KITTI 03 | shipped | 1.31e+09 | 5.17e+08 |
| Newton check: stop state | KITTI 03 | corrected | round 250 | no certified stop |
| Newton check: poses within eta S_lo of x^ | KITTI 03 | corrected | 0.975 | no certified stop |
| Newton check: pose distance / E, median; max | KITTI 03 | corrected | 0.579; 2.23 | no certified stop |
| Newton check: largest landmark move (m) | KITTI 03 | corrected | 1.05e+03 | no certified stop |
| landmarks in the model | KITTI 04 | both | 4.65e+03 | 4.19e+03 |
| stereo factors in the model | KITTI 04 | both | 1.53e+04 | 1.48e+04 |
| certified stop round (300-round runs) | KITTI 04 | corrected | 170 | none |
| unfinished fraction, current problem, at round 100 | KITTI 04 | corrected | 0.9 | 1 |
| median remaining GN move (D) at round 100 | KITTI 04 | corrected | 1.17 | 37.2 |
| aligned keyframe ATE (m) at round 100 | KITTI 04 | corrected | 0.133 | 0.129 |
| unfinished fraction, current problem, at round 300 | KITTI 04 | corrected | 0 | 1 |
| median remaining GN move (D) at round 300 | KITTI 04 | corrected | 0.0644 | 36.9 |
| aligned keyframe ATE (m) at round 300 | KITTI 04 | corrected | 0.133 | 0.127 |
| GBP flops per round (median) | KITTI 04 | corrected | 3.11e+09 | 2.95e+09 |
| certificate flops per evaluation (median) | KITTI 04 | corrected | 1.25e+09 | 5.17e+08 |
| certified stop round (300-round runs) | KITTI 04 | shipped | none | none |
| unfinished fraction, current problem, at round 100 | KITTI 04 | shipped | 1 | 1 |
| median remaining GN move (D) at round 100 | KITTI 04 | shipped | 4.84 | 66.6 |
| aligned keyframe ATE (m) at round 100 | KITTI 04 | shipped | 0.136 | 0.197 |
| unfinished fraction, current problem, at round 300 | KITTI 04 | shipped | 0.275 | 1 |
| median remaining GN move (D) at round 300 | KITTI 04 | shipped | 0.415 | 71.5 |
| aligned keyframe ATE (m) at round 300 | KITTI 04 | shipped | 0.133 | 0.193 |
| GBP flops per round (median) | KITTI 04 | shipped | 3.11e+09 | 2.95e+09 |
| certificate flops per evaluation (median) | KITTI 04 | shipped | 1.21e+09 | 5.17e+08 |
| Newton check: stop state | KITTI 04 | corrected | round 170 | no certified stop |
| Newton check: poses within eta S_lo of x^ | KITTI 04 | corrected | 0.575 | no certified stop |
| Newton check: pose distance / E, median; max | KITTI 04 | corrected | 1.5; 2.9 | no certified stop |
| Newton check: largest landmark move (m) | KITTI 04 | corrected | 154 | no certified stop |
| landmarks in the model | KITTI 10 | both | 4.23e+03 | 3.68e+03 |
| stereo factors in the model | KITTI 10 | both | 1.27e+04 | 1.21e+04 |
| certified stop round (300-round runs) | KITTI 10 | corrected | 50 | none |
| unfinished fraction, current problem, at round 100 | KITTI 10 | corrected | 0 | 1 |
| median remaining GN move (D) at round 100 | KITTI 10 | corrected | 0.0819 | 12.4 |
| aligned keyframe ATE (m) at round 100 | KITTI 10 | corrected | 0.0853 | 0.0808 |
| unfinished fraction, current problem, at round 300 | KITTI 10 | corrected | 0 | 1 |
| median remaining GN move (D) at round 300 | KITTI 10 | corrected | 0.00474 | 10.9 |
| aligned keyframe ATE (m) at round 300 | KITTI 10 | corrected | 0.0853 | 0.0834 |
| GBP flops per round (median) | KITTI 10 | corrected | 2.24e+09 | 2.07e+09 |
| certificate flops per evaluation (median) | KITTI 10 | corrected | 6.74e+08 | 5.17e+08 |
| certified stop round (300-round runs) | KITTI 10 | shipped | 240 | none |
| unfinished fraction, current problem, at round 100 | KITTI 10 | shipped | 0.975 | 1 |
| median remaining GN move (D) at round 100 | KITTI 10 | shipped | 4.89 | 102 |
| aligned keyframe ATE (m) at round 100 | KITTI 10 | shipped | 0.101 | 0.196 |
| unfinished fraction, current problem, at round 300 | KITTI 10 | shipped | 0 | 1 |
| median remaining GN move (D) at round 300 | KITTI 10 | shipped | 0.27 | 132 |
| aligned keyframe ATE (m) at round 300 | KITTI 10 | shipped | 0.0901 | 0.14 |
| GBP flops per round (median) | KITTI 10 | shipped | 2.24e+09 | 2.07e+09 |
| certificate flops per evaluation (median) | KITTI 10 | shipped | 9.62e+08 | 5.17e+08 |
| Newton check: stop state | KITTI 10 | corrected | round 50 | no certified stop |
| Newton check: poses within eta S_lo of x^ | KITTI 10 | corrected | 1 | no certified stop |
| Newton check: pose distance / E, median; max | KITTI 10 | corrected | 0.891; 1.33 | no certified stop |
| Newton check: largest landmark move (m) | KITTI 10 | corrected | 175 | no certified stop |
| certified stop round (300-round runs) | KITTI 06 prefix (161) | corrected | 130 | none |
| unfinished fraction, current problem, at round 100 | KITTI 06 prefix (161) | corrected | 0.0124 | 1 |
| median remaining GN move (D) at round 100 | KITTI 06 prefix (161) | corrected | 0.0151 | 250 |
| aligned keyframe ATE (m) at round 100 | KITTI 06 prefix (161) | corrected | 0.383 | 0.364 |
| unfinished fraction, current problem, at round 300 | KITTI 06 prefix (161) | corrected | 0 | 1 |
| median remaining GN move (D) at round 300 | KITTI 06 prefix (161) | corrected | 0.00406 | 240 |
| aligned keyframe ATE (m) at round 300 | KITTI 06 prefix (161) | corrected | 0.382 | 0.342 |
| GBP flops per round (median) | KITTI 06 prefix (161) | corrected | 8.56e+09 | 7.3e+09 |
| certificate flops per evaluation (median) | KITTI 06 prefix (161) | corrected | 9.45e+09 | 3.37e+10 |
| certified stop round (300-round runs) | KITTI 06 prefix (161) | shipped | none | none |
| unfinished fraction, current problem, at round 100 | KITTI 06 prefix (161) | shipped | 0.161 | 1 |
| median remaining GN move (D) at round 100 | KITTI 06 prefix (161) | shipped | 0.377 | 760 |
| aligned keyframe ATE (m) at round 100 | KITTI 06 prefix (161) | shipped | 0.384 | 0.67 |
| unfinished fraction, current problem, at round 300 | KITTI 06 prefix (161) | shipped | 0.0932 | 1 |
| median remaining GN move (D) at round 300 | KITTI 06 prefix (161) | shipped | 0.313 | 1.25e+03 |
| aligned keyframe ATE (m) at round 300 | KITTI 06 prefix (161) | shipped | 0.383 | 0.832 |
| GBP flops per round (median) | KITTI 06 prefix (161) | shipped | 8.56e+09 | 7.3e+09 |
| certificate flops per evaluation (median) | KITTI 06 prefix (161) | shipped | 1.02e+10 | 3.37e+10 |
