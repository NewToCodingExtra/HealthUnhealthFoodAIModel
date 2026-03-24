# NutriScan Feature Importance Report

Generated from current `trained_model.pkl` coefficients (calibrated logistic base estimator).

## Core Model (9 features)

| Feature | Coef (Healthy) | Coef (Unhealthy) | Direction | |coef| |
|---|---:|---:|---|---:|
| `saturated_fat` | -5.490941 | 5.490941 | Unhealthy (-) | 5.490941 |
| `sodium` | -4.920574 | 4.920574 | Unhealthy (-) | 4.920574 |
| `protein` | 3.781393 | -3.781393 | Healthy (+) | 3.781393 |
| `carbohydrates` | 3.500930 | -3.500930 | Healthy (+) | 3.500930 |
| `fried_index` | -3.327877 | 3.327877 | Unhealthy (-) | 3.327877 |
| `sugar` | -2.222710 | 2.222710 | Unhealthy (-) | 2.222710 |
| `fat` | 1.245112 | -1.245112 | Healthy (+) | 1.245112 |
| `fried_starchy` | -0.984143 | 0.984143 | Unhealthy (-) | 0.984143 |
| `calories` | 0.374130 | -0.374130 | Healthy (+) | 0.374130 |

## All-Features Model (14 features)

| Feature | Coef (Healthy) | Coef (Unhealthy) | Direction | |coef| |
|---|---:|---:|---|---:|
| `fiber` | 5.046468 | -5.046468 | Healthy (+) | 5.046468 |
| `saturated_fat` | -4.386863 | 4.386863 | Unhealthy (-) | 4.386863 |
| `added_sugar` | -3.980179 | 3.980179 | Unhealthy (-) | 3.980179 |
| `sodium` | -3.961950 | 3.961950 | Unhealthy (-) | 3.961950 |
| `fried_index` | -3.945929 | 3.945929 | Unhealthy (-) | 3.945929 |
| `carbohydrates` | 2.578054 | -2.578054 | Healthy (+) | 2.578054 |
| `protein` | 2.204067 | -2.204067 | Healthy (+) | 2.204067 |
| `fried_starchy` | -1.445564 | 1.445564 | Unhealthy (-) | 1.445564 |
| `fat` | 1.086619 | -1.086619 | Healthy (+) | 1.086619 |
| `sugar` | 1.055624 | -1.055624 | Healthy (+) | 1.055624 |
| `omega3` | -0.483595 | 0.483595 | Unhealthy (-) | 0.483595 |
| `vitamin_c` | -0.461448 | 0.461448 | Unhealthy (-) | 0.461448 |
| `calories` | -0.305010 | 0.305010 | Unhealthy (-) | 0.305010 |
| `cholesterol` | -0.070885 | 0.070885 | Unhealthy (-) | 0.070885 |

## Notes

- Positive coefficient means higher values push toward `Healthy`.
- Negative coefficient means higher values push toward `Unhealthy`.
- Magnitude shows relative influence after preprocessing.