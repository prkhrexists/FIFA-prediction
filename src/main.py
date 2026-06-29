
# ============================================================
# FIFA WORLD CUP 2026 — KNOCKOUT STAGE PREDICTOR
# ============================================================
# Predicts:
#   1. Championship probability for all 32 teams (Monte Carlo)
#   2. Most likely winner for every match until the Final
#   3. Saves results as a JPG chart + CSV
#
# Model: Poisson goals model (attack/defense stats) blended
#        with FIFA ratings for team quality.
# ============================================================

import pandas as pd
import numpy as np
import math
import random
import os
import csv
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: FILE PATHS
# ============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RESULTS_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw', '127', 'results.csv')
FIFA_CSV    = r'C:\Users\tusha\OneDrive\Desktop\Projects\fifa-predictor\data\raw\fifa_ranking_2026-06-08.csv'
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SECTION 2: TEAM NAME NORMALIZATION
# ============================================================
NAME_MAP = {
    'United States':                'USA',
    'United States of America':     'USA',
    'Korea Republic':               'South Korea',
    'Republic of Korea':            'South Korea',
    'IR Iran':                      'Iran',
    "Cote d'Ivoire":                'Ivory Coast',
    "Côte d'Ivoire":                'Ivory Coast',
    'Czech Republic':               'Czechia',
    'West Germany':                 'Germany',
    'Republic of Ireland':          'Ireland',
    'FYR Macedonia':                'North Macedonia',
    'Bosnia and Herzegovina':       'Bosnia-Herzegovina',
    'Bosnia & Herzegovina':         'Bosnia-Herzegovina',
    'Cabo Verde':                   'Cape Verde',
    'Cape Verde Islands':           'Cape Verde',
    'DR Congo':                     'DR Congo',
    'Congo DR':                     'DR Congo',
    'Congo, DR':                    'DR Congo',
    'Democratic Republic of Congo': 'DR Congo',
    'Congo':                        'DR Congo',
}

def normalize(name):
    """Converts any team name variant to our standard spelling."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return NAME_MAP.get(name, name)

# ============================================================
# SECTION 3: LOAD MATCH HISTORY
# ============================================================
print("=" * 55)
print("  FIFA WC 2026 KNOCKOUT STAGE PREDICTOR")
print("=" * 55)
print("\nStep 1: Loading match history...")

if not os.path.exists(RESULTS_CSV):
    print(f"  ERROR: results.csv not found at:\n  {RESULTS_CSV}")
    exit()

df = pd.read_csv(RESULTS_CSV)
df['home_team'] = df['home_team'].apply(normalize)
df['away_team'] = df['away_team'].apply(normalize)
df['date']      = pd.to_datetime(df['date'], errors='coerce')
df['year']      = df['date'].dt.year

# Use only matches from 2020+ with known scores (excludes future matches)
df_modern = df[(df['year'] >= 2020)].copy()
df_modern = df_modern.dropna(subset=['home_score', 'away_score'])
df_modern = df_modern.sort_values('date').reset_index(drop=True)
print(f"  Loaded {len(df_modern):,} completed matches (2020-present)")

# ============================================================
# SECTION 4: TOURNAMENT IMPORTANCE WEIGHTS
# ============================================================
# World Cup matches count more than friendlies when computing stats.
WEIGHTS = {
    'FIFA World Cup':                       3.0,
    'FIFA World Cup qualification':         2.0,
    'UEFA Euro':                            2.5,
    'Copa America':                         2.5,
    'AFC Asian Cup':                        2.5,
    'Africa Cup of Nations':                2.5,
    'CONCACAF Gold Cup':                    2.0,
    'UEFA Euro qualification':              1.5,
    'Copa America qualification':           1.5,
    'Africa Cup of Nations qualification':  1.5,
    'Friendly':                             0.5,
    'Friendlies':                           0.5,
    'International Friendly':               0.5,
}

def get_weight(t):
    if pd.isna(t): return 1.0
    return WEIGHTS.get(str(t), 1.0)

df_modern['weight'] = df_modern['tournament'].apply(get_weight)

# ============================================================
# SECTION 5: GLOBAL AVERAGE GOALS (the model baseline)
# ============================================================
last500 = df_modern.tail(500)
GLOBAL_AVG = (last500['home_score'].mean() + last500['away_score'].mean()) / 2
print(f"  Global avg goals per team per match: {GLOBAL_AVG:.3f}")

# ============================================================
# SECTION 6: COMPUTE TEAM STATS (attack & defense)
# ============================================================
def compute_features(team, df, global_avg, last_n=20):
    """
    Computes two key stats from a team's last N matches:

    attack_strength:
        How many goals they score vs the global average.
        1.0 = average scorer, 1.5 = 50% better than average.

    defensive_weakness:
        How many goals they concede vs the global average.
        1.0 = average defender, 0.7 = 30% harder to score against.

    Both use weighted averages (World Cup matches count 6x more than friendlies).
    """
    mask    = (df['home_team'] == team) | (df['away_team'] == team)
    matches = df[mask].tail(last_n)

    if len(matches) == 0:
        return 1.0, 1.0   # No data → assume average team

    scored, conceded, weights = [], [], []

    for _, row in matches.iterrows():
        w = row.get('weight', 1.0)
        if row['home_team'] == team:
            scored.append(row['home_score'])
            conceded.append(row['away_score'])
        else:
            scored.append(row['away_score'])
            conceded.append(row['home_score'])
        weights.append(w)

    w = np.array(weights, dtype=float)
    s = np.array(scored,  dtype=float)
    c = np.array(conceded, dtype=float)
    tw = w.sum()

    if tw == 0:
        return 1.0, 1.0

    avg_s = (s * w).sum() / tw
    avg_c = (c * w).sum() / tw

    # Normalize to global average. Clamp to avoid extreme outliers.
    att = max(0.5, min(avg_s / global_avg, 3.0)) if global_avg > 0 else 1.0
    dfw = max(0.5, min(avg_c / global_avg, 3.0)) if global_avg > 0 else 1.0

    return att, dfw

# ============================================================
# SECTION 7: LOAD FIFA RATINGS (team quality measure)
# ============================================================
print("\nStep 2: Loading FIFA ratings...")

def load_fifa_ratings(filepath):
    """
    Loads FIFA ranking points.
    Returns dict: {team_name: points}
    """
    if not os.path.exists(filepath):
        print("  FIFA ratings file not found. Defaulting all teams to 1500.")
        return {}

    df_r = pd.read_csv(filepath)

    # Auto-detect team and rating columns
    team_col = pts_col = None
    for col in df_r.columns:
        cl = col.lower()
        if cl in ('country_full', 'team', 'team_name', 'name', 'country'):
            team_col = col
        if cl in ('total_points', 'points', 'rating', 'elo', 'elo_rating'):
            pts_col = col

    if not team_col or not pts_col:
        print(f"  Could not detect columns. Found: {df_r.columns.tolist()}")
        return {}

    ratings = {}
    for _, row in df_r.iterrows():
        try:
            ratings[normalize(str(row[team_col]))] = float(row[pts_col])
        except (ValueError, TypeError):
            pass

    print(f"  Loaded FIFA ratings for {len(ratings)} teams")
    return ratings

raw_ratings = load_fifa_ratings(FIFA_CSV)

# Scale FIFA points to an Elo-like range (1000–2000) for consistent math
if raw_ratings:
    mn, mx = min(raw_ratings.values()), max(raw_ratings.values())
    ELO = {t: 1000 + (p - mn) / max(mx - mn, 1) * 1000 for t, p in raw_ratings.items()}
else:
    ELO = {}
    print("  Using default 1500 for all teams.")

def get_elo(team):
    """Return team's Elo-like rating (default 1500 if unknown)."""
    return ELO.get(team, 1500)

# ============================================================
# SECTION 8: THE OFFICIAL WC 2026 ROUND OF 32 BRACKET
# ============================================================
# Source: Official FIFA WC 2026 bracket (June 28 - July 3, 2026)
# Canada already beat South Africa 1-0 (Match 73) — result is locked in.

# Format: (team1, team2, match_number)
LEFT_BRACKET = [
    ('Canada',      'South Africa', 73),  # PLAYED: Canada won 1-0
    ('Spain',       'Austria',      74),
    ('Germany',     'Paraguay',     75),
    ('Portugal',    'Croatia',      76),
    ('Ivory Coast', 'Norway',       77),
    ('Australia',   'Egypt',        78),
    ('France',      'Sweden',       79),
    ('Argentina',   'Cape Verde',   80),
]

RIGHT_BRACKET = [
    ('Brazil',              'Japan',                81),
    ('Netherlands',         'Morocco',              82),
    ('Mexico',              'Ecuador',              83),
    ('Colombia',            'Ghana',                84),
    ('England',             'DR Congo',             85),
    ('Belgium',             'Senegal',              86),
    ('USA',                 'Bosnia-Herzegovina',   87),
    ('Switzerland',         'Algeria',              88),
]

# Known match results (already played)
KNOWN_RESULTS = {
    73: 'Canada',   # Canada beat South Africa 1–0
}

# All 32 teams
ALL_TEAMS = sorted(set(
    team
    for bracket in [LEFT_BRACKET, RIGHT_BRACKET]
    for t1, t2, _ in bracket
    for team in (t1, t2)
))

# ============================================================
# SECTION 9: BUILD FEATURE TABLE FOR ALL 32 TEAMS
# ============================================================
print("\nStep 3: Computing team features from match history...")
print(f"\n  {'Team':<28} {'Attack':>7} {'Defense':>9} {'FIFA Elo':>9}")
print("  " + "-" * 58)

FEATURES = {}
for team in sorted(ALL_TEAMS):
    att, dfw = compute_features(team, df_modern, GLOBAL_AVG)
    elo = get_elo(team)
    FEATURES[team] = (att, dfw)
    print(f"  {team:<28} {att:>7.3f} {dfw:>9.3f} {elo:>9.0f}")

# ============================================================
# SECTION 10: POISSON MATCH PREDICTION
# ============================================================
def poisson_prob(k, lam):
    """
    Probability of scoring exactly k goals when average = lam.
    Example: lam=1.5, k=2 → P(2 goals) ≈ 25.1%
    """
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def predict_match(team1, team2):
    """
    Predicts the outcome probabilities for team1 vs team2.

    How it works:
    1. Compute expected goals (lambda) for each team using attack/defense stats
    2. Build a 9×9 grid of every possible scoreline (0-0 up to 8-8)
    3. Sum probabilities to get P(team1 wins), P(draw), P(team2 wins)
    4. Blend with FIFA rating difference (Elo formula) for quality adjustment:
       - 70% weight on stats-based Poisson model
       - 30% weight on FIFA rating model

    Returns: (p1_wins, p_draw, p2_wins, lam1, lam2)
    """
    att1, dfw1 = FEATURES.get(team1, (1.0, 1.0))
    att2, dfw2 = FEATURES.get(team2, (1.0, 1.0))

    # Expected goals = global average × attacker strength × opponent defensive weakness
    lam1 = max(GLOBAL_AVG * att1 * dfw2, 0.1)
    lam2 = max(GLOBAL_AVG * att2 * dfw1, 0.1)

    # Build scoreline matrix and sum probabilities
    p1_wins = p_draw = p2_wins = 0.0
    for i in range(9):         # team1 goals: 0 to 8
        for j in range(9):     # team2 goals: 0 to 8
            p = poisson_prob(i, lam1) * poisson_prob(j, lam2)
            if   i > j:  p1_wins += p
            elif i == j: p_draw  += p
            else:        p2_wins += p

    # Normalize (small correction for capping at 8 goals)
    total = p1_wins + p_draw + p2_wins
    if total > 0:
        p1_wins /= total
        p_draw  /= total
        p2_wins /= total

    # Elo model: P(team1 wins) = 1 / (1 + 10^(-Δelo/400))
    elo_p1 = 1 / (1 + 10 ** (-(get_elo(team1) - get_elo(team2)) / 400))

    # Blend: 40% Poisson stats + 60% Elo quality rating
    b1  = 0.6 * elo_p1 + 0.4 * p1_wins
    rem = 1.0 - b1
    dt  = p_draw + p2_wins
    bd  = rem * (p_draw  / dt) if dt > 0 else 0.0
    b2  = rem * (p2_wins / dt) if dt > 0 else rem

    return b1, bd, b2, lam1, lam2

# ============================================================
# SECTION 11: SIMULATE ONE KNOCKOUT MATCH
# ============================================================
def simulate_match(team1, team2):
    """
    Randomly picks a winner for one knockout match.
    Draw probability is split 50/50 between both teams (must have a winner).
    """
    p1, pd_, p2, _, _ = predict_match(team1, team2)
    p1_ko = p1 + pd_ / 2    # team1's chance of winning (includes half the draw prob)
    return team1 if random.random() < p1_ko else team2

# ============================================================
# SECTION 12: SIMULATE ONE FULL TOURNAMENT
# ============================================================
def simulate_tournament():
    """
    Plays through the entire WC 2026 knockout bracket once.
    Returns the champion's name.
    """

    def play_round(matchups):
        """Given list of (t1, t2) pairs, returns list of winners."""
        return [simulate_match(t1, t2) for t1, t2 in matchups]

    # --- Round of 32 ---
    left_r32_winners  = []
    for t1, t2, mid in LEFT_BRACKET:
        if mid in KNOWN_RESULTS:
            left_r32_winners.append(KNOWN_RESULTS[mid])   # Use known result
        else:
            left_r32_winners.append(simulate_match(t1, t2))

    right_r32_winners = [simulate_match(t1, t2) for t1, t2, _ in RIGHT_BRACKET]

    # --- Round of 16 (pair up winners: 0v1, 2v3, etc.) ---
    def pair_winners(winners):
        return [(winners[i], winners[i+1]) for i in range(0, len(winners)-1, 2)]

    left_r16  = play_round(pair_winners(left_r32_winners))   # 4 winners
    right_r16 = play_round(pair_winners(right_r32_winners))  # 4 winners

    # --- Quarter-Finals ---
    left_qf  = play_round(pair_winners(left_r16))   # 2 winners
    right_qf = play_round(pair_winners(right_r16))  # 2 winners

    # --- Semi-Finals ---
    left_sf  = simulate_match(left_qf[0],  left_qf[1])
    right_sf = simulate_match(right_qf[0], right_qf[1])

    # --- Final ---
    champion = simulate_match(left_sf, right_sf)
    return champion

# ============================================================
# SECTION 13: RUN MONTE CARLO (10,000 tournaments)
# ============================================================
N_SIMS = 10000
print(f"\nStep 4: Running {N_SIMS:,} Monte Carlo simulations...")

win_counts = defaultdict(int)
for i in range(N_SIMS):
    champ = simulate_tournament()
    if champ:
        win_counts[champ] += 1
    if (i + 1) % 2000 == 0:
        print(f"  {i+1:,} / {N_SIMS:,} completed")

# Sort results by probability
results = sorted(
    [{'team': t, 'wins': win_counts.get(t, 0),
      'prob': round(win_counts.get(t, 0) / N_SIMS * 100, 2)}
     for t in ALL_TEAMS],
    key=lambda x: x['prob'], reverse=True
)

# ============================================================
# SECTION 14: DISPLAY CHAMPIONSHIP PROBABILITIES
# ============================================================
print("\n")
print("=" * 58)
print("  FIFA WC 2026 — CHAMPIONSHIP PROBABILITIES")
print(f"  Based on {N_SIMS:,} Monte Carlo simulations")
print("=" * 58)
print(f"  {'Rank':<5} {'Team':<26} {'Prob':>8}   Visual")
print("  " + "-" * 55)
for rank, r in enumerate(results, 1):
    bar = '#' * int(r['prob'])
    print(f"  {rank:<5} {r['team']:<26} {r['prob']:>6.2f}%  {bar}")
print("  " + "-" * 55)

# ============================================================
# SECTION 15: PREDICT MOST LIKELY WINNER FOR EVERY MATCH
# ============================================================
print("\n")
print("=" * 70)
print("  PREDICTED MATCH-BY-MATCH BRACKET PATH")
print("  (Showing most likely winner + win probability for each match)")
print("=" * 70)

def likely_winner(t1, t2, match_id=None):
    """Returns (winner, win_prob) for a match. Uses known result if available."""
    if match_id and match_id in KNOWN_RESULTS:
        return KNOWN_RESULTS[match_id], 1.0
    p1, pd_, p2, lam1, lam2 = predict_match(t1, t2)
    p1_ko = p1 + pd_ / 2
    p2_ko = p2 + pd_ / 2
    if p1_ko >= p2_ko:
        return t1, p1_ko
    return t2, p2_ko

def print_match_header():
    print(f"\n  {'M#':<5} {'Team 1':<25} {'Team 2':<25} {'Likely Winner':<25} {'Win Prob'}")
    print("  " + "-" * 88)

def print_match(mid, t1, t2, winner, prob, note=""):
    status = f"  [{note}]" if note else ""
    print(f"  {mid:<5} {t1:<25} {t2:<25} {winner:<25} {prob:.1%}{status}")

# --- Round of 32 ---
print("\n  ROUND OF 32")
print_match_header()

left_likely   = []
right_likely  = []

for t1, t2, mid in LEFT_BRACKET:
    w, p = likely_winner(t1, t2, mid)
    note = "PLAYED" if mid in KNOWN_RESULTS else ""
    print_match(f"M{mid}", t1, t2, w, p, note)
    left_likely.append(w)

for t1, t2, mid in RIGHT_BRACKET:
    w, p = likely_winner(t1, t2, mid)
    print_match(f"M{mid}", t1, t2, w, p)
    right_likely.append(w)

# --- Round of 16 ---
print("\n  ROUND OF 16")
print_match_header()

def pair_and_predict(winners, start_id):
    """Pairs up winners and predicts each match. Returns likely winners."""
    next_winners = []
    for i in range(0, len(winners) - 1, 2):
        t1, t2 = winners[i], winners[i+1]
        w, p = likely_winner(t1, t2)
        print_match(f"M{start_id + i//2}", t1, t2, w, p)
        next_winners.append(w)
    return next_winners

left_r16  = pair_and_predict(left_likely,  89)   # M89–M92
right_r16 = pair_and_predict(right_likely, 93)   # M93–M96

# --- Quarter-Finals ---
print("\n  QUARTER-FINALS")
print_match_header()

left_qf  = pair_and_predict(left_r16,  97)   # M97–M98
right_qf = pair_and_predict(right_r16, 99)   # M99–M100

# --- Semi-Finals ---
print("\n  SEMI-FINALS")
print_match_header()

sf1_w, sf1_p = likely_winner(left_qf[0],  left_qf[1])
sf2_w, sf2_p = likely_winner(right_qf[0], right_qf[1])
print_match("M101", left_qf[0],  left_qf[1],  sf1_w, sf1_p)
print_match("M102", right_qf[0], right_qf[1], sf2_w, sf2_p)

# --- Final ---
print("\n  WORLD CUP FINAL  (July 19, 2026 — MetLife Stadium, New York)")
print_match_header()

fin_w, fin_p = likely_winner(sf1_w, sf2_w)
print_match("M104", sf1_w, sf2_w, fin_w, fin_p)

print(f"\n  *** PREDICTED CHAMPION: {fin_w.upper()} ***")
print(f"  (Predicted win probability in the Final: {fin_p:.1%})")
print()

# ============================================================
# SECTION 16: GENERATE RESULTS CHART (JPG)
# ============================================================
print("Step 5: Generating results chart...")

try:
    import matplotlib
    matplotlib.use('Agg')       # Non-interactive: saves to file, no popup
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Filter to teams with > 0.1% chance, cap at 20 teams
    chart_data = [r for r in results if r['prob'] > 0.05][:20]
    teams = [r['team'] for r in chart_data]
    probs = [r['prob'] for r in chart_data]

    # Assign bar colors
    colors = []
    for i, r in enumerate(chart_data):
        if r['team'] == fin_w:
            colors.append('#FFD700')   # Gold for predicted champion
        elif i < 4:
            colors.append('#1565C0')   # Dark blue for top 4
        else:
            colors.append('#455A64')   # Dark grey for others

    # Build figure
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d1b2a')

    # Horizontal bar chart (teams on y-axis, probability on x-axis)
    bars = ax.barh(
        teams[::-1], probs[::-1],
        color=colors[::-1],
        edgecolor='#ffffff33',
        linewidth=0.5,
        height=0.65
    )

    # Add % label at the end of each bar
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f'{prob:.1f}%',
            va='center', ha='left',
            color='white', fontsize=9.5, fontweight='bold'
        )

    # Axis and title styling
    ax.set_xlabel('Championship Probability (%)', color='#cccccc', fontsize=11, labelpad=10)
    ax.set_title(
        f'FIFA World Cup 2026\nKnockout Stage — Championship Probabilities\n'
        f'({N_SIMS:,} Monte Carlo Simulations)  |  Predicted Champion: {fin_w}',
        color='white', fontsize=13, fontweight='bold', pad=14, linespacing=1.6
    )
    ax.tick_params(colors='#cccccc', labelsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#444')
    ax.set_xlim(0, max(probs) * 1.25)

    # Legend
    legend_items = [
        mpatches.Patch(color='#FFD700', label=f'Predicted Champion ({fin_w})'),
        mpatches.Patch(color='#1565C0', label='Top 4 contenders'),
        mpatches.Patch(color='#455A64', label='Other teams'),
    ]
    ax.legend(
        handles=legend_items, loc='lower right',
        facecolor='#0d0d1a', edgecolor='#555',
        labelcolor='white', fontsize=9
    )

    # Vertical gridlines for readability
    ax.xaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    img_path = os.path.join(OUTPUT_DIR, 'wc2026_championship_probabilities.jpg')
    plt.savefig(img_path, format='jpeg', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] Chart saved to: {img_path}")

except ImportError:
    print("  matplotlib not installed. Install with:  pip install matplotlib")
    print("  Skipping chart generation.")

# ============================================================
# SECTION 17: SAVE CSV
# ============================================================
csv_path = os.path.join(OUTPUT_DIR, 'wc2026_predictions.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['rank', 'team', 'wins', 'probability_pct'])
    writer.writeheader()
    for rank, r in enumerate(results, 1):
        writer.writerow({'rank': rank, 'team': r['team'],
                         'wins': r['wins'], 'probability_pct': r['prob']})

print(f"  [SAVED] CSV saved to:   {csv_path}")
print("\n[DONE]  All predictions complete!")