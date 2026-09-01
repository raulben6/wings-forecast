# Restaurant Sales Forecasting

![Python](https://img.shields.io/badge/Python_3-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=flat-square)
![Tests](https://img.shields.io/badge/9_tests-6E9F18?style=flat-square&logo=pytest&logoColor=white)

**Forecasting daily sales for a small restaurant, using 227 days of its real operating data.**

A restaurant buys perishable stock against a guess. Guess high and it rots, guess low and it turns customers away. The question this project answers is narrow and practical: **how much better than "the same as last week" can we actually do, and is a machine learning model worth the complexity?**

The answer turned out to be interesting, and it is not the one a portfolio project usually reports.

---

## 🎯 The headline finding

> **A day-of-week historical average beat both machine learning models.**

| Method | MAE | RMSE | MAPE | vs. baseline |
|---|---:|---:|---:|---:|
| 🥇 **Day-of-week mean** | **25.83** | 33.44 | 34.8% | **+27.6%** |
| 🥈 Ridge + calendar | 26.44 | 34.85 | **33.3%** | +25.9% |
| 🥉 Gradient boosting | 27.33 | 36.77 | 34.9% | +23.4% |
| Moving average (6d) | 29.37 | 37.64 | 37.1% | +17.7% |
| *Seasonal naive (baseline)* | 35.69 | 45.69 | 43.9% | 0% |
| Naive (yesterday) | 37.74 | 47.50 | 47.2% | -5.7% |

![Forecast error by method](reports/figures/04_comparacion.png)

**And the honest caveat that matters more than the ranking:** the 95% bootstrap confidence interval for the winner's MAE is **[22.62, 29.28]**. Ridge and gradient boosting both sit inside it. **The top three are statistically tied.** Declaring a winner from these 152 forecasts would be reading noise.

So the defensible conclusion is not "day-of-week average is best". It is:

1. Every method **comfortably beats** the heuristic a manager would use by hand, by 18 to 28 percent.
2. **Nothing beats a weekday average by enough to justify its complexity.** Deploying a gradient boosting model here would add maintenance, a training pipeline and an inference dependency to buy an error difference the data cannot even resolve.

That is a real result, and shipping a model anyway would have been the wrong call.

---

## 🔍 Why the simple method holds up

The permutation importance explains it. Trained on held-out days, the gradient boosting model leans almost entirely on three things:

![Which variables carry the model](reports/figures/07_importancia.png)

| Variable | What it is |
|---|---|
| `media_24` | 4-week moving average, the recent **level** |
| `media_dia_semana` | historical mean for that **weekday** |
| `es_saturday` | the Saturday indicator |

The model spent 18 features to rediscover "*what day is it, and how has business been lately*". That is precisely what the day-of-week average already encodes, so it cannot do much better. Trend, month and most lags contribute nothing measurable.

---

## 📈 What the data looks like

![Daily sales](reports/figures/01_serie_temporal.png)

227 trading days, October 2025 to June 2026. The business is **closed on Sundays**, so the series runs on a six-day week and every lag in the code is measured in openings, not calendar days. "A week ago" is six rows back, which keeps weekday alignment intact.

The series is genuinely noisy: **coefficient of variation 41%**, values ranging from 23 to 304 around a mean of 100.

### The weekly pattern is the strongest signal

![Sales by day of week](reports/figures/02_dia_semana.png)

| Day | Mean | Std |
|---|---:|---:|
| Monday | 97.1 | 31.5 |
| Tuesday | 90.4 | 33.7 |
| Wednesday | 90.4 | 25.8 |
| Thursday | 114.1 | 50.2 |
| **Friday** | **137.2** | 41.2 |
| Saturday | 73.7 | 31.2 |

**Friday is the peak and Saturday is the trough.** That inverts the usual restaurant pattern and says something concrete about the business: it lives on weekday trade, not weekend leisure. Saturday sells barely half of Friday.

Thursday is the volatile one, with the widest spread of any day (std 50.2).

### Revenue mix

![Revenue mix by channel](reports/figures/03_canales.png)

Cash 47%, card 35%, delivery marketplace 18%.

---

## 🎯 Where the error actually lives

![Where the model misses](reports/figures/06_error_por_dia.png)

| Day | MAE |
|---|---:|
| Wednesday | 16.8 |
| Saturday | 22.7 |
| Tuesday | 25.0 |
| Monday | 28.4 |
| Friday | 30.4 |
| **Thursday** | **32.0** |

Error concentrates on **Thursday and Friday**, the two highest-volume days. That is not a modelling failure so much as a description of the business: the busy days are the variable ones, and they are also the days where a purchasing mistake costs the most. Wednesday, the most predictable day, is almost twice as accurate as Thursday.

![Actual vs predicted](reports/figures/05_real_vs_pred.png)

---

## 🧪 Method, and the mistake it avoids

**Validation is walk-forward (rolling origin), never a random split.**

This is the single most common error in time-series work. A random `train_test_split` scatters future days into training and past days into test, so the model forecasts Tuesday having already seen Thursday. The reported error looks superb and the model is worthless, because in production the future is not available.

Here, every one of the 152 forecasts uses **only data from days strictly before it**, and the models retrain at each step on the expanding window.

That claim is not left to trust. The test suite enforces it:

```bash
python -m pytest tests/ -q     # 9 passed
```

| Test | What it proves |
|---|---|
| `test_retardo_1_es_el_valor_anterior` | lag features point backwards, exactly one row |
| `test_retardo_semanal_apunta_a_la_semana_anterior` | the 6-row lag lands on the same weekday |
| `test_medias_moviles_no_incluyen_el_dia_actual` | rolling windows close at t-1 |
| `test_media_dia_semana_solo_mira_atras` | the expanding weekday mean excludes the current row |
| `test_ninguna_variable_correlaciona_perfecto_con_el_objetivo` | no feature is a copy of the target |
| `test_walk_forward_nunca_entrega_el_futuro` | the history handed to every method ends before the target date |

---

## ⚠️ Limitations, stated plainly

- **227 observations is a small sample.** Every metric here carries a wide interval, which is why the MAE is reported with a bootstrap CI rather than as a single number.
- **A 35% MAPE is high.** Useful for directional purchasing decisions, not for precise cash planning.
- **The biggest drivers are missing from the data**: weather, public holidays, local events, promotions, competitor activity, and menu changes. A large share of the residual variance is almost certainly explainable, just not by anything in this spreadsheet.
- **One establishment, nine months.** Nothing here generalises to restaurants in general.
- Sundays were dropped (2 exceptional openings out of 229), so the model says nothing about them.

## 🔭 What would actually improve it

In the order I would try them:

1. **A public holiday and local event calendar.** Cheap to add, and likely the single largest gain.
2. **Weather history**, joined by date. Rain probably moves both footfall and the delivery share.
3. **Item-level data from the POS**, which exists: [RestoPos](https://github.com/raulben6/restopos) is the point-of-sale system this business runs. Forecasting per product would turn this from a revenue estimate into an actual purchase order.
4. **Quantile forecasts instead of a point estimate.** For stock decisions, the 80th percentile of demand is a more useful number than the mean.

---

## 📁 Running it

```bash
git clone https://github.com/raulben6/wings-forecast.git
cd wings-forecast

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_analysis.py    # figures + reports/resultados.md
python -m pytest tests/ -q        # 9 tests
```

```
src/
  datos.py            loading, cleaning, the six-day week
  caracteristicas.py  calendar features, lags, rolling means (all closed at t-1)
  referencias.py      naive, seasonal naive, moving average, day-of-week mean
  modelos.py          Ridge and gradient boosting, wrapped for walk-forward
  evaluacion.py       rolling-origin validation, metrics, bootstrap CI
  graficos.py         report figures, colorblind-safe palette
scripts/
  preparar_datos.py   spreadsheet to indexed dataset (run once, locally)
  run_analysis.py     the full pipeline
reports/
  resultados.md       generated results
  metricas.json       machine-readable metrics
tests/
  test_sin_fuga.py    the leakage guarantees above
```

---

## 🔒 About the data

The sales figures belong to a real, identifiable business, so they are **not published in currency**. Every value is divided by the series mean and multiplied by 100, giving a mean of exactly 100 "sales index" units. The divisor is not published.

That transform is linear, which is the point: seasonality, relative variance, channel mix, correlations and every percentage metric (MAPE, improvement over baseline) are **completely unaffected**. The analysis is real. Only the absolute scale is withheld.

`scripts/preparar_datos.py` performs that step and is included so the process is auditable. The spreadsheet it reads is gitignored and never committed.

---

## License

[MIT](LICENSE) © Raúl Antonio Benítez Vásquez
