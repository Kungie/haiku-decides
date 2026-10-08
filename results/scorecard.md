# Scorecard

## banking77

Items every system answered: 498

| System | Coverage | Refusals | Accuracy | ECE | Brier | Flip rate | TV shift | Repeat flip | p50 ms | p95 ms | $ / 1k | Cache read |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 1.000 | 0.000 | 0.799 [0.763, 0.833] | 0.088 [0.064, 0.120] | 0.299 [0.251, 0.349] | 0.040 | 0.051 | 0.000 | 267 | 426 | 0.0708 | 0.000 |
| openai | 0.996 | 0.004 | 0.793 [0.757, 0.827] | 0.085 [0.062, 0.118] | 0.319 [0.267, 0.373] | 0.300 | 0.225 | 0.000 | 156 | 372 | 0.0986 | 0.000 |
| haiku-single | 1.000 | 0.000 | 0.791 [0.755, 0.825] | none | none | 0.220 | none | 0.020 | 813 | 1386 | 0.0508 | 0.967 |
| haiku-sampled | 1.000 | 0.000 | 0.795 [0.759, 0.829] | 0.185 [0.153, 0.220] | 0.393 [0.327, 0.461] | 0.260 | 0.121 | 0.020 | 876 | 1600 | 0.4225 | 0.991 |
| haiku-shuffled | 1.000 | 0.000 | 0.803 [0.765, 0.837] | 0.114 [0.089, 0.149] | 0.327 [0.273, 0.386] | 0.080 | 0.044 | 0.000 | 1360 | 1571 | 2.1133 | 0.388 |

- Accuracy, jev vs haiku-single: not distinguishable
- Accuracy, jev vs haiku-sampled: not distinguishable
- Accuracy, jev vs haiku-shuffled: not distinguishable
- Accuracy, openai vs haiku-single: not distinguishable
- Accuracy, openai vs haiku-sampled: not distinguishable
- Accuracy, openai vs haiku-shuffled: not distinguishable

## ag_news

Items every system answered: 500

| System | Coverage | Refusals | Accuracy | ECE | Brier | Flip rate | TV shift | Repeat flip | p50 ms | p95 ms | $ / 1k | Cache read |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 1.000 | 0.000 | 0.892 [0.864, 0.918] | 0.074 [0.055, 0.103] | 0.173 [0.130, 0.220] | 0.020 | 0.010 | 0.000 | 278 | 411 | 0.0166 | 0.000 |
| openai | 1.000 | 0.000 | 0.882 [0.856, 0.910] | 0.092 [0.068, 0.117] | 0.201 [0.154, 0.247] | 0.040 | 0.018 | 0.000 | 252 | 465 | 0.0185 | 0.000 |
| haiku-single | 1.000 | 0.000 | 0.882 [0.852, 0.908] | none | none | 0.040 | none | 0.000 | 1121 | 1313 | 0.0538 | 0.000 |
| haiku-sampled | 1.000 | 0.000 | 0.884 [0.854, 0.910] | 0.112 [0.087, 0.142] | 0.224 [0.171, 0.282] | 0.040 | 0.016 | 0.000 | 1222 | 1530 | 0.5376 | 0.000 |
| haiku-shuffled | 1.000 | 0.000 | 0.880 [0.850, 0.908] | 0.105 [0.081, 0.135] | 0.214 [0.165, 0.267] | 0.000 | 0.007 | 0.000 | 1217 | 1573 | 0.5377 | 0.000 |

- Accuracy, jev vs haiku-single: not distinguishable
- Accuracy, jev vs haiku-sampled: not distinguishable
- Accuracy, jev vs haiku-shuffled: not distinguishable
- Accuracy, openai vs haiku-single: not distinguishable
- Accuracy, openai vs haiku-sampled: not distinguishable
- Accuracy, openai vs haiku-shuffled: not distinguishable

## boolq

Items every system answered: 500

| System | Coverage | Refusals | Accuracy | ECE | Brier | Repeat flip | p50 ms | p95 ms | $ / 1k | Cache read |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 1.000 | 0.000 | 0.910 [0.884, 0.936] | 0.021 [0.017, 0.053] | 0.142 [0.111, 0.175] | 0.040 | 265 | 304 | 0.0171 | 0.000 |
| openai | 1.000 | 0.000 | 0.838 [0.804, 0.868] | 0.109 [0.087, 0.142] | 0.254 [0.209, 0.309] | 0.000 | 139 | 319 | 0.0282 | 0.000 |
| haiku-single | 1.000 | 0.000 | 0.872 [0.842, 0.900] | none | none | 0.060 | 1227 | 1357 | 0.0525 | 0.000 |
| haiku-sampled | 1.000 | 0.000 | 0.878 [0.848, 0.904] | 0.116 [0.093, 0.147] | 0.236 [0.186, 0.293] | 0.020 | 1692 | 6594 | 0.5252 | 0.000 |

- Accuracy, jev vs haiku-single: jev higher
- Accuracy, jev vs haiku-sampled: jev higher
- Accuracy, openai vs haiku-single: not distinguishable
- Accuracy, openai vs haiku-sampled: haiku-sampled higher

## sms_spam

Items every system answered: 500

| System | Coverage | Refusals | Accuracy | ECE | Brier | Repeat flip | p50 ms | p95 ms | $ / 1k | Cache read |
|---|---|---|---|---|---|---|---|---|---|---|
| jev | 1.000 | 0.000 | 0.966 [0.950, 0.982] | 0.063 [0.053, 0.076] | 0.049 [0.035, 0.064] | 0.000 | 277 | 432 | 0.0139 | 0.000 |
| openai | 1.000 | 0.000 | 0.972 [0.958, 0.986] | 0.016 [0.008, 0.029] | 0.043 [0.025, 0.064] | 0.000 | 154 | 472 | 0.0202 | 0.000 |
| haiku-single | 1.000 | 0.000 | 0.962 [0.946, 0.978] | none | none | 0.000 | 1228 | 3255 | 0.0402 | 0.000 |
| haiku-sampled | 1.000 | 0.000 | 0.964 [0.948, 0.980] | 0.035 [0.020, 0.051] | 0.066 [0.035, 0.096] | 0.000 | 1203 | 1462 | 0.4021 | 0.000 |

- Accuracy, jev vs haiku-single: not distinguishable
- Accuracy, jev vs haiku-sampled: not distinguishable
- Accuracy, openai vs haiku-single: not distinguishable
- Accuracy, openai vs haiku-sampled: not distinguishable

## sst5

Items every system answered: 500

| System | Coverage | Refusals | Accuracy | MAE | ECE | Brier | Repeat flip | p50 ms | p95 ms | $ / 1k | Cache read |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | 1.000 | 0.000 | 0.558 [0.516, 0.602] | 0.516 [0.458, 0.572] | 0.209 [0.167, 0.248] | 0.622 [0.563, 0.681] | 0.020 | 261 | 345 | 0.0141 | 0.000 |
| openai | 1.000 | 0.000 | 0.502 [0.460, 0.544] | 0.580 [0.526, 0.636] | 0.269 [0.229, 0.311] | 0.718 [0.660, 0.779] | 0.000 | 217 | 441 | 0.0158 | 0.000 |
| haiku-single | 1.000 | 0.000 | 0.526 [0.482, 0.568] | 0.516 [0.466, 0.566] | none | none | 0.040 | 1172 | 3728 | 0.0487 | 0.000 |
| haiku-sampled | 1.000 | 0.000 | 0.532 [0.488, 0.574] | 0.508 [0.458, 0.558] | 0.441 [0.401, 0.488] | 0.902 [0.818, 0.988] | 0.000 | 2323 | 6734 | 0.4873 | 0.000 |

- Accuracy, jev vs haiku-single: not distinguishable
- Accuracy, jev vs haiku-sampled: not distinguishable
- Accuracy, openai vs haiku-single: not distinguishable
- Accuracy, openai vs haiku-sampled: not distinguishable

Brackets are 95% bootstrap intervals. `none`: the mode produces no probabilities. `n/a`: not applicable.

Accuracy verdicts test the paired difference on the items every system answered. Brier is the multiclass form (0 is perfect, 2 is worst). Flip rate is the share of items whose answer changed across five reorderings of the options; Repeat flip is the share whose answer changed between two runs with the same order, so it shows how much of the flip rate is sampling noise.

Measured from a single client machine on 2026-10-08. N=10 samples per decision in the sampled modes. Prices as of 2026-10-08.
