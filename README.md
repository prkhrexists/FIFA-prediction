# FIFA World Cup 2026 Knockout Stage Predictor

A Python-based Monte Carlo simulation model that predicts championship probabilities for all 32 teams in the FIFA World Cup 2026 knockout stage.

## What it does

- **Loads** historical international match data (2020–present)
- **Computes** attack strength and defensive weakness for each team using weighted recent form (World Cup matches weighted 6× more than friendlies)
- **Blends** Poisson goals model (40%) with FIFA rating quality (60%) to predict match outcomes
- **Runs 10,000** Monte Carlo simulations of the full knockout bracket
- **Outputs** championship probabilities, match-by-match bracket predictions, a results chart (JPG), and a CSV

## Results (as of June 29, 2026)

| Rank | Team | Championship Probability |
|------|------|--------------------------|
| 1 | Spain | ~15% |
| 2 | England | ~12% |
| 3 | France | ~11% |
| 4 | Argentina | ~8% |
| 5 | Germany | ~7% |

**Predicted Champion:** Spain  
**Predicted Final:** Spain vs England

## Bracket Path (Most Likely)

- **R32:** Spain, Germany, Portugal, France, Argentina, Brazil, Netherlands, Mexico, England, Belgium, USA, Switzerland advance
- **R16:** Spain, Germany, France, Argentina, Netherlands, Mexico, England, USA
- **QF:** Spain, France, Netherlands, England
- **SF:** Spain vs France → Spain | Netherlands vs England → England
- **Final:** Spain vs England → **Spain wins**

## Model Details

| Component | Details |
|-----------|---------|
| Match history | 2020–present, weighted by tournament importance |
| Goal model | Poisson distribution (expected goals per team) |
| Rating source | FIFA ranking points (scaled to Elo range 1000–2000) |
| Blend ratio | 60% FIFA rating + 40% Poisson stats |
| Simulations | 10,000 Monte Carlo runs |

## Files

```
src/main.py                          — Main prediction script
data/raw/127/results.csv             — International match history (Kaggle)
data/raw/fifa_ranking_2026-06-08.csv — FIFA rankings (June 2026)
data/raw/wc.csv                      — WC 2026 group structure
output/wc2026_predictions.csv        — Championship probability results
output/wc2026_championship_probabilities.jpg — Results chart
```

## How to run

```bash
pip install pandas numpy matplotlib
python src/main.py
```

## Data Sources

- Match history: [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- FIFA Rankings: [FIFA Official Rankings](https://www.fifa.com/fifa-world-ranking)
