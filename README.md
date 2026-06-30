# FIFA World Cup 2026 Knockout Stage Predictor

A Python-based data analytics and Monte Carlo simulation model that predicts championship probabilities for all 32 teams in the FIFA World Cup 2026, plus an interactive tool to predict any head-to-head match.

---

## Features

- **Loads** historical international match data (2020–present) — 6,000+ matches
- **Computes** attack strength and defensive weakness for each team using weighted recent form
  - World Cup matches weighted 3× | Friendlies weighted 0.5×
- **Blends** a Poisson goals model (40%) with scaled FIFA rating quality (60%) to predict match outcomes
- **Runs 10,000** Monte Carlo simulations of the full knockout bracket
- **Locks in real results** as they happen — known outcomes are fixed, not re-simulated
- **Interactive match predictor** — enter any two teams and get a full prediction with chart
- **Outputs** championship probabilities, bracket path, results chart (JPG), and CSV

---

## Scripts

| Script | Description |
|--------|-------------|
| `src/main.py` | Full WC 2026 bracket simulation (10,000 Monte Carlo runs) |
| `src/predict_tonight.py` | Interactive: enter any 2 teams → get a prediction + chart |

---

## Results (Updated: June 30, 2026)

### Known R32 Results (locked in)
| Match | Result |
|-------|--------|
| M73 | Canada 1–0 South Africa |
| M75 | Paraguay 1–1 Germany (4–3 pens) |
| M81 | Brazil 2–1 Japan |
| M82 | Morocco 1–1 Netherlands (3–2 pens) |

### Championship Probabilities

| Rank | Team | Probability |
|------|------|-------------|
| 1 | Spain | ~17% |
| 2 | England | ~13% |
| 3 | France | ~11% |
| 4 | Argentina | ~8% |
| 5 | Brazil | ~7% |
| 6 | Portugal | ~7% |
| 7 | Morocco | ~6% |
| 8 | Canada | ~6% |

**Predicted Champion:** Spain
**Predicted Final:** Spain vs England

---

## Predicted Bracket Path

- **R16:** Spain, Portugal, Norway, France, Brazil, Mexico, England, USA
- **QF:** Spain, France, Brazil, England
- **SF:** Spain vs France → Spain | Brazil vs England → England
- **Final:** Spain vs England → **Spain wins**

---

## Interactive Match Predictor

```bash
python src/predict_tonight.py
```

```
Enter Team 1: Spain
Enter Team 2: Brazil

============================================================
  MATCH PREDICTION
  Spain  vs  Brazil
============================================================

  Win probability:
    Spain                      61.3%  ##############################
    Draw                       21.5%  ##########
    Brazil                     17.2%  ########

  Expected goals:  Spain 1.79  |  Brazil 0.89

  Team stats (last 20 matches):
    Spain     Attack: 1.981  Defense: 0.500  FIFA Elo: 1997
    Brazil    Attack: 1.251  Defense: 0.637  FIFA Elo: 1904

  >> PREDICTION: SPAIN wins  (61.3% knockout probability)
============================================================

  Save chart? (y/n): y
```

Supports 261 teams from the dataset. Type `list` to see all available teams.

---

## Model Details

| Component | Details |
|-----------|---------|
| Match history | 2020–present, 6,000+ matches |
| Tournament weighting | World Cup 3×, Euros/Copa 2.5×, Friendlies 0.5× |
| Goal model | Poisson distribution (expected goals per team) |
| Rating source | FIFA ranking points scaled to Elo range 1000–2000 |
| Blend ratio | 60% FIFA rating + 40% Poisson stats |
| Simulations | 10,000 Monte Carlo runs |

---

## Files

```
src/main.py                              — Full bracket simulation
src/predict_tonight.py                   — Interactive head-to-head predictor
data/raw/127/results.csv                 — International match history (Kaggle)
data/raw/fifa_ranking_2026-06-08.csv     — FIFA rankings (June 2026)
data/raw/wc.csv                          — WC 2026 group structure
output/wc2026_predictions.csv            — Championship probability results
output/wc2026_championship_probabilities.jpg — Results chart
```

---

## How to Run

```bash
pip install pandas numpy matplotlib

# Full tournament simulation
python src/main.py

# Interactive match predictor
python src/predict_tonight.py
```

---

## Data Sources

- Match history: [Kaggle — International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- FIFA Rankings: [FIFA Official Rankings](https://www.fifa.com/fifa-world-ranking)
