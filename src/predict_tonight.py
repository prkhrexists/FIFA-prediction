"""
predict_match.py
Enter any two teams → get a full prediction using your dataset.
"""

import pandas as pd
import numpy as np
import math
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RESULTS_CSV  = os.path.join(PROJECT_ROOT, 'data', 'raw', '127', 'results.csv')
FIFA_CSV     = os.path.join(PROJECT_ROOT, 'data', 'raw', 'fifa_ranking_2026-06-08.csv')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Name normalisation ────────────────────────────────────────────────────────
NAME_MAP = {
    'United States':                'USA',
    'United States of America':     'USA',
    "Cote d'Ivoire":                'Ivory Coast',
    "Cote dIvoire":                 'Ivory Coast',
    "Côte d'Ivoire":                'Ivory Coast',
    'Korea Republic':               'South Korea',
    'IR Iran':                      'Iran',
    'Czech Republic':               'Czechia',
    'West Germany':                 'Germany',
    'Republic of Ireland':          'Ireland',
    'FYR Macedonia':                'North Macedonia',
    'Bosnia and Herzegovina':       'Bosnia-Herzegovina',
    'Bosnia & Herzegovina':         'Bosnia-Herzegovina',
    'Cabo Verde':                   'Cape Verde',
    'Cape Verde Islands':           'Cape Verde',
    'Congo DR':                     'DR Congo',
    'Congo, DR':                    'DR Congo',
    'Democratic Republic of Congo': 'DR Congo',
    'Congo':                        'DR Congo',
}

def normalize(name):
    if pd.isna(name): return name
    name = str(name).strip()
    return NAME_MAP.get(name, name)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(RESULTS_CSV)
df['home_team'] = df['home_team'].apply(normalize)
df['away_team'] = df['away_team'].apply(normalize)
df['date']      = pd.to_datetime(df['date'], errors='coerce')
df['year']      = df['date'].dt.year
df_modern       = df[df['year'] >= 2020].dropna(subset=['home_score','away_score'])
df_modern       = df_modern.sort_values('date').reset_index(drop=True)

WEIGHTS = {
    'FIFA World Cup': 3.0, 'FIFA World Cup qualification': 2.0,
    'UEFA Euro': 2.5, 'Copa America': 2.5,
    'AFC Asian Cup': 2.5, 'Africa Cup of Nations': 2.5,
    'CONCACAF Gold Cup': 2.0, 'UEFA Euro qualification': 1.5,
    'Copa America qualification': 1.5,
    'Africa Cup of Nations qualification': 1.5,
    'Friendly': 0.5, 'Friendlies': 0.5, 'International Friendly': 0.5,
}
df_modern['weight'] = df_modern['tournament'].apply(
    lambda t: WEIGHTS.get(str(t), 1.0) if not pd.isna(t) else 1.0
)

last500    = df_modern.tail(500)
GLOBAL_AVG = (last500['home_score'].mean() + last500['away_score'].mean()) / 2

# ── FIFA ratings ──────────────────────────────────────────────────────────────
df_r     = pd.read_csv(FIFA_CSV)
team_col = next((c for c in df_r.columns if c.lower() in ('team','country_full','name','country')), None)
pts_col  = next((c for c in df_r.columns if c.lower() in ('points','total_points','rating','elo')), None)
ratings  = {}
for _, row in df_r.iterrows():
    try:
        ratings[normalize(str(row[team_col]))] = float(row[pts_col])
    except Exception:
        pass

mn, mx = min(ratings.values()), max(ratings.values())
ELO    = {t: 1000 + (p - mn) / max(mx - mn, 1) * 1000 for t, p in ratings.items()}

def get_elo(team):
    return ELO.get(team, 1500)

# ── Build known-teams list ────────────────────────────────────────────────────
all_teams = sorted(set(
    list(df_modern['home_team'].unique()) +
    list(df_modern['away_team'].unique())
))

# ── Feature computation ───────────────────────────────────────────────────────
def compute_features(team, last_n=20):
    mask    = (df_modern['home_team'] == team) | (df_modern['away_team'] == team)
    matches = df_modern[mask].tail(last_n)
    if len(matches) == 0:
        return 1.0, 1.0, 0
    scored, conceded, weights = [], [], []
    for _, row in matches.iterrows():
        w = row.get('weight', 1.0)
        if row['home_team'] == team:
            scored.append(row['home_score']); conceded.append(row['away_score'])
        else:
            scored.append(row['away_score']); conceded.append(row['home_score'])
        weights.append(w)
    w  = np.array(weights, dtype=float)
    s  = np.array(scored,  dtype=float)
    c  = np.array(conceded, dtype=float)
    tw = w.sum()
    if tw == 0: return 1.0, 1.0, len(matches)
    avg_s = (s * w).sum() / tw
    avg_c = (c * w).sum() / tw
    att = max(0.5, min(avg_s / GLOBAL_AVG, 3.0)) if GLOBAL_AVG > 0 else 1.0
    dfw = max(0.5, min(avg_c / GLOBAL_AVG, 3.0)) if GLOBAL_AVG > 0 else 1.0
    return att, dfw, len(matches)

# ── Poisson predictor ─────────────────────────────────────────────────────────
def poisson_prob(k, lam):
    if lam <= 0: return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def predict(team1, team2):
    att1, dfw1, n1 = compute_features(team1)
    att2, dfw2, n2 = compute_features(team2)
    lam1 = max(GLOBAL_AVG * att1 * dfw2, 0.1)
    lam2 = max(GLOBAL_AVG * att2 * dfw1, 0.1)

    p1_wins = p_draw = p2_wins = 0.0
    for i in range(9):
        for j in range(9):
            p = poisson_prob(i, lam1) * poisson_prob(j, lam2)
            if   i > j: p1_wins += p
            elif i == j: p_draw += p
            else:        p2_wins += p

    total = p1_wins + p_draw + p2_wins
    if total > 0:
        p1_wins /= total; p_draw /= total; p2_wins /= total

    elo_p1 = 1 / (1 + 10 ** (-(get_elo(team1) - get_elo(team2)) / 400))
    b1  = 0.6 * elo_p1 + 0.4 * p1_wins
    rem = 1.0 - b1
    dt  = p_draw + p2_wins
    bd  = rem * (p_draw  / dt) if dt > 0 else 0.0
    b2  = rem * (p2_wins / dt) if dt > 0 else rem

    return {
        'team1': team1, 'team2': team2,
        'p1': round(b1 * 100, 1),
        'draw': round(bd * 100, 1),
        'p2': round(b2 * 100, 1),
        'lam1': round(lam1, 2), 'lam2': round(lam2, 2),
        'att1': round(att1, 3), 'dfw1': round(dfw1, 3),
        'att2': round(att2, 3), 'dfw2': round(dfw2, 3),
        'elo1': round(get_elo(team1)), 'elo2': round(get_elo(team2)),
        'matches1': n1, 'matches2': n2,
    }

def fuzzy_find(name, team_list):
    """Find closest team name (case-insensitive, partial match)."""
    name_l = name.lower()
    exact  = [t for t in team_list if t.lower() == name_l]
    if exact: return exact[0]
    partial = [t for t in team_list if name_l in t.lower()]
    return partial[0] if len(partial) == 1 else (partial if partial else None)

def print_result(r):
    t1, t2 = r['team1'], r['team2']
    winner = t1 if r['p1'] >= r['p2'] else t2
    win_p  = max(r['p1'], r['p2'])

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  MATCH PREDICTION")
    print(f"  {t1}  vs  {t2}")
    print(f"{sep}")
    print(f"\n  Win probability:")
    print(f"    {t1:<25} {r['p1']:>5.1f}%  {'#' * int(r['p1'] // 2)}")
    print(f"    {'Draw':<25} {r['draw']:>5.1f}%  {'#' * int(r['draw'] // 2)}")
    print(f"    {t2:<25} {r['p2']:>5.1f}%  {'#' * int(r['p2'] // 2)}")
    print(f"\n  Expected goals:  {t1} {r['lam1']}  |  {t2} {r['lam2']}")
    print(f"\n  Team stats (from dataset, last 20 matches):")
    print(f"    {t1:<25} Attack: {r['att1']}  Defense: {r['dfw1']}  FIFA Elo: {r['elo1']}  ({r['matches1']} matches)")
    print(f"    {t2:<25} Attack: {r['att2']}  Defense: {r['dfw2']}  FIFA Elo: {r['elo2']}  ({r['matches2']} matches)")
    print(f"\n  >> PREDICTION: {winner.upper()} wins  ({win_p:.1f}% knockout probability)")
    print(f"{sep}\n")

def generate_chart(r):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        BG    = '#0d0d1a'
        CARD  = '#131428'
        BLUE  = '#3b82f6'
        GREEN = '#22c55e'
        GREY  = '#64748b'
        WHITE = '#f1f5f9'
        AMBER = '#f59e0b'

        t1, t2 = r['team1'], r['team2']
        winner = t1 if r['p1'] >= r['p2'] else t2

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(BG)
        fig.suptitle(f"Match Prediction: {t1}  vs  {t2}",
                     color=WHITE, fontsize=15, fontweight='bold', y=1.02)

        # ── Left: Win probability pie ─────────────────────────────────────────
        ax1 = axes[0]
        ax1.set_facecolor(CARD)
        labels = [t1, 'Draw', t2]
        sizes  = [r['p1'], r['draw'], r['p2']]
        colors = [BLUE, GREY, GREEN]
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'color': WHITE, 'fontsize': 10},
            wedgeprops={'edgecolor': BG, 'linewidth': 2}
        )
        for at in autotexts:
            at.set_color(WHITE)
            at.set_fontweight('bold')
        ax1.set_title('Win Probability', color=WHITE, fontsize=12,
                      fontweight='bold', pad=12)

        # ── Right: Stats comparison bar chart ─────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor(CARD)
        ax2.tick_params(colors='#94a3b8', labelsize=9)
        for sp in ax2.spines.values():
            sp.set_edgecolor('#334155')

        metrics      = ['Attack\nStrength', 'Defensive\nWeakness', 'Expected\nGoals', 'FIFA Elo\n(scaled/100)']
        vals1        = [r['att1'], r['dfw1'], r['lam1'], r['elo1'] / 100]
        vals2        = [r['att2'], r['dfw2'], r['lam2'], r['elo2'] / 100]
        x            = np.arange(len(metrics))
        bar_w        = 0.35

        bars1 = ax2.bar(x - bar_w/2, vals1, bar_w, label=t1, color=BLUE,
                        edgecolor='#ffffff22', linewidth=0.5)
        bars2 = ax2.bar(x + bar_w/2, vals2, bar_w, label=t2, color=GREEN,
                        edgecolor='#ffffff22', linewidth=0.5)

        for bar in list(bars1) + list(bars2):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.03,
                     f'{bar.get_height():.2f}',
                     ha='center', va='bottom',
                     color=WHITE, fontsize=8, fontweight='bold')

        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, color='#94a3b8', fontsize=9)
        ax2.set_facecolor(CARD)
        ax2.set_title('Team Stats Comparison', color=WHITE, fontsize=12,
                      fontweight='bold', pad=12)
        ax2.legend(facecolor=CARD, edgecolor='#334155',
                   labelcolor=WHITE, fontsize=9)
        ax2.xaxis.grid(False)
        ax2.yaxis.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
        ax2.set_axisbelow(True)

        # Winner banner
        w_col = BLUE if winner == t1 else GREEN
        fig.text(0.5, -0.04,
                 f"Predicted Winner: {winner.upper()}  ({max(r['p1'], r['p2']):.1f}% knockout chance)",
                 ha='center', color=AMBER, fontsize=13, fontweight='bold')

        plt.tight_layout(pad=2.5)
        safe_name = f"{t1.replace(' ','_')}_vs_{t2.replace(' ','_')}.jpg"
        out_path  = os.path.join(OUTPUT_DIR, safe_name)
        plt.savefig(out_path, format='jpeg', dpi=150,
                    bbox_inches='tight', facecolor=BG)
        plt.close()
        print(f"  [Chart saved] -> {out_path}")

    except ImportError:
        print("  matplotlib not installed — skipping chart.")

# ── Main interactive loop ─────────────────────────────────────────────────────
print(f"  Dataset: {len(df_modern):,} matches loaded (2020-present)")
print(f"  Teams in dataset: {len(all_teams)}")
print(f"\nType team names to predict a match. Type 'list' to see all teams. Type 'quit' to exit.\n")

while True:
    print("-" * 40)
    t1_input = input("  Enter Team 1: ").strip()
    if t1_input.lower() in ('quit', 'exit', 'q'): break
    if t1_input.lower() == 'list':
        print("\n  " + "\n  ".join(all_teams) + "\n")
        continue

    t2_input = input("  Enter Team 2: ").strip()
    if t2_input.lower() in ('quit', 'exit', 'q'): break

    # Fuzzy match
    t1 = fuzzy_find(t1_input, all_teams)
    t2 = fuzzy_find(t2_input, all_teams)

    if isinstance(t1, list):
        print(f"  '{t1_input}' is ambiguous. Did you mean: {', '.join(t1)}?")
        continue
    if t1 is None:
        print(f"  '{t1_input}' not found. Try 'list' to see all available teams.")
        continue
    if isinstance(t2, list):
        print(f"  '{t2_input}' is ambiguous. Did you mean: {', '.join(t2)}?")
        continue
    if t2 is None:
        print(f"  '{t2_input}' not found. Try 'list' to see all available teams.")
        continue

    result = predict(t1, t2)
    print_result(result)

    save = input("  Save chart? (y/n): ").strip().lower()
    if save == 'y':
        generate_chart(result)

    again = input("  Predict another match? (y/n): ").strip().lower()
    if again != 'y':
        break

print("\nGoodbye!")
