import pandas as pd
import numpy as np
import math
import os
import warnings

warnings.filterwarnings('ignore')

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
results_file = os.path.join(PROJECT_ROOT, 'data', 'raw', '127', 'results.csv')
fifa_file = os.path.join(PROJECT_ROOT, 'data', 'raw', 'fifa_ranking_2026-06-08.csv')
output_folder = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(output_folder, exist_ok=True)

# fix team names that appear differently across datasets
name_fixes = {
    'United States': 'USA',
    'United States of America': 'USA',
    "Cote d'Ivoire": 'Ivory Coast',
    "Cote dIvoire": 'Ivory Coast',
    "Côte d'Ivoire": 'Ivory Coast',
    'Korea Republic': 'South Korea',
    'IR Iran': 'Iran',
    'Czech Republic': 'Czechia',
    'West Germany': 'Germany',
    'Cabo Verde': 'Cape Verde',
    'Cape Verde Islands': 'Cape Verde',
    'Congo DR': 'DR Congo',
    'Democratic Republic of Congo': 'DR Congo',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Bosnia & Herzegovina': 'Bosnia-Herzegovina',
}

def fix_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    if name in name_fixes:
        return name_fixes[name]
    return name


print("Loading dataset...")

# load match history
all_matches = pd.read_csv(results_file)
all_matches['home_team'] = all_matches['home_team'].apply(fix_name)
all_matches['away_team'] = all_matches['away_team'].apply(fix_name)
all_matches['date'] = pd.to_datetime(all_matches['date'], errors='coerce')
all_matches['year'] = all_matches['date'].dt.year

# only use matches from 2020 onwards
recent_matches = all_matches[all_matches['year'] >= 2020].dropna(subset=['home_score', 'away_score'])
recent_matches = recent_matches.sort_values('date').reset_index(drop=True)

# weight each tournament by importance
tournament_weights = {
    'FIFA World Cup': 3.0,
    'FIFA World Cup qualification': 2.0,
    'UEFA Euro': 2.5,
    'Copa America': 2.5,
    'AFC Asian Cup': 2.5,
    'Africa Cup of Nations': 2.5,
    'CONCACAF Gold Cup': 2.0,
    'UEFA Euro qualification': 1.5,
    'Friendly': 0.5,
    'Friendlies': 0.5,
    'International Friendly': 0.5,
}

def get_weight(t):
    if pd.isna(t):
        return 1.0
    return tournament_weights.get(str(t), 1.0)

recent_matches['weight'] = recent_matches['tournament'].apply(get_weight)

# global average goals per team per match (baseline)
last_500 = recent_matches.tail(500)
avg_goals = (last_500['home_score'].mean() + last_500['away_score'].mean()) / 2


# load fifa ratings and scale to 1000-2000 range
df_ratings = pd.read_csv(fifa_file)

# auto detect columns
team_col = None
pts_col = None
for col in df_ratings.columns:
    if col.lower() in ('team', 'country_full', 'name', 'country'):
        team_col = col
    if col.lower() in ('points', 'total_points', 'rating', 'elo'):
        pts_col = col

ratings = {}
for i, row in df_ratings.iterrows():
    try:
        ratings[fix_name(str(row[team_col]))] = float(row[pts_col])
    except:
        pass

min_pts = min(ratings.values())
max_pts = max(ratings.values())
scaled_ratings = {}
for team, pts in ratings.items():
    scaled_ratings[team] = 1000 + (pts - min_pts) / max(max_pts - min_pts, 1) * 1000

def get_rating(team):
    return scaled_ratings.get(team, 1500)


# get all unique team names from the dataset
known_teams = sorted(set(
    list(recent_matches['home_team'].unique()) +
    list(recent_matches['away_team'].unique())
))

print(f"  Dataset: {len(recent_matches):,} matches loaded (2020-present)")
print(f"  Teams in dataset: {len(known_teams)}")
print(f"\nType team names to predict a match.")
print(f"Type 'list' to see all available teams.")
print(f"Type 'quit' to exit.\n")


def get_team_stats(team):
    # get this teams last 20 matches
    team_matches = recent_matches[
        (recent_matches['home_team'] == team) | (recent_matches['away_team'] == team)
    ].tail(20)

    if len(team_matches) == 0:
        return 1.0, 1.0, 0

    goals_scored = []
    goals_conceded = []
    weights = []

    for i, row in team_matches.iterrows():
        w = row.get('weight', 1.0)
        if row['home_team'] == team:
            goals_scored.append(row['home_score'])
            goals_conceded.append(row['away_score'])
        else:
            goals_scored.append(row['away_score'])
            goals_conceded.append(row['home_score'])
        weights.append(w)

    w_arr = np.array(weights, dtype=float)
    s_arr = np.array(goals_scored, dtype=float)
    c_arr = np.array(goals_conceded, dtype=float)
    total_w = w_arr.sum()

    if total_w == 0:
        return 1.0, 1.0, len(team_matches)

    avg_scored = (s_arr * w_arr).sum() / total_w
    avg_conceded = (c_arr * w_arr).sum() / total_w

    # attack = how many goals they score vs global average
    attack = max(0.5, min(avg_scored / avg_goals, 3.0)) if avg_goals > 0 else 1.0
    # defense = how many goals they concede vs global average (lower = better)
    defense = max(0.5, min(avg_conceded / avg_goals, 3.0)) if avg_goals > 0 else 1.0

    return attack, defense, len(team_matches)


def poisson(k, lam):
    # probability of scoring exactly k goals when average is lam
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict(team1, team2):
    att1, dfw1, n1 = get_team_stats(team1)
    att2, dfw2, n2 = get_team_stats(team2)

    # expected goals for each team
    exp1 = max(avg_goals * att1 * dfw2, 0.1)
    exp2 = max(avg_goals * att2 * dfw1, 0.1)

    # go through all possible scorelines 0-0 to 8-8
    team1_wins = 0.0
    draw = 0.0
    team2_wins = 0.0

    for g1 in range(9):
        for g2 in range(9):
            p = poisson(g1, exp1) * poisson(g2, exp2)
            if g1 > g2:
                team1_wins += p
            elif g1 == g2:
                draw += p
            else:
                team2_wins += p

    total = team1_wins + draw + team2_wins
    if total > 0:
        team1_wins = team1_wins / total
        draw = draw / total
        team2_wins = team2_wins / total

    # elo formula for rating-based probability
    elo_p1 = 1 / (1 + 10 ** (-(get_rating(team1) - get_rating(team2)) / 400))

    # blend 60% elo + 40% stats
    b1 = 0.6 * elo_p1 + 0.4 * team1_wins
    remaining = 1.0 - b1
    rest = draw + team2_wins

    if rest > 0:
        bd = remaining * (draw / rest)
        b2 = remaining * (team2_wins / rest)
    else:
        bd = 0.0
        b2 = remaining

    return {
        'team1': team1,
        'team2': team2,
        'p1': round(b1 * 100, 1),
        'draw': round(bd * 100, 1),
        'p2': round(b2 * 100, 1),
        'exp1': round(exp1, 2),
        'exp2': round(exp2, 2),
        'att1': round(att1, 3),
        'dfw1': round(dfw1, 3),
        'att2': round(att2, 3),
        'dfw2': round(dfw2, 3),
        'elo1': round(get_rating(team1)),
        'elo2': round(get_rating(team2)),
        'n1': n1,
        'n2': n2,
    }


def show_result(r):
    t1 = r['team1']
    t2 = r['team2']

    if r['p1'] >= r['p2']:
        winner = t1
        win_prob = r['p1']
    else:
        winner = t2
        win_prob = r['p2']

    print("\n" + "=" * 60)
    print(f"  MATCH PREDICTION")
    print(f"  {t1}  vs  {t2}")
    print("=" * 60)
    print(f"\n  Win probability:")
    print(f"    {t1:<25} {r['p1']:>5.1f}%  {'#' * int(r['p1'] // 2)}")
    print(f"    {'Draw':<25} {r['draw']:>5.1f}%  {'#' * int(r['draw'] // 2)}")
    print(f"    {t2:<25} {r['p2']:>5.1f}%  {'#' * int(r['p2'] // 2)}")
    print(f"\n  Expected goals:  {t1} {r['exp1']}  |  {t2} {r['exp2']}")
    print(f"\n  Team stats (last 20 matches from dataset):")
    print(f"    {t1:<25} Attack: {r['att1']}  Defense: {r['dfw1']}  FIFA Elo: {r['elo1']}  ({r['n1']} matches found)")
    print(f"    {t2:<25} Attack: {r['att2']}  Defense: {r['dfw2']}  FIFA Elo: {r['elo2']}  ({r['n2']} matches found)")
    print(f"\n  >> PREDICTION: {winner.upper()} wins  ({win_prob:.1f}% knockout probability)")
    print("=" * 60 + "\n")


def save_chart(r):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        t1 = r['team1']
        t2 = r['team2']

        if r['p1'] >= r['p2']:
            winner = t1
        else:
            winner = t2

        BG = '#0d0d1a'
        CARD = '#131428'
        BLUE = '#3b82f6'
        GREEN = '#22c55e'
        GREY = '#64748b'
        WHITE = '#f1f5f9'
        AMBER = '#f59e0b'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(BG)
        fig.suptitle(f"Match Prediction: {t1}  vs  {t2}", color=WHITE, fontsize=15, fontweight='bold', y=1.02)

        # pie chart for win probabilities
        ax1.set_facecolor(CARD)
        labels = [t1, 'Draw', t2]
        sizes = [r['p1'], r['draw'], r['p2']]
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
        ax1.set_title('Win Probability', color=WHITE, fontsize=12, fontweight='bold', pad=12)

        # bar chart comparing team stats
        ax2.set_facecolor(CARD)
        ax2.tick_params(colors='#94a3b8', labelsize=9)
        for sp in ax2.spines.values():
            sp.set_edgecolor('#334155')

        stat_names = ['Attack\nStrength', 'Defensive\nWeakness', 'Expected\nGoals', 'FIFA Elo\n(scaled/100)']
        t1_vals = [r['att1'], r['dfw1'], r['exp1'], r['elo1'] / 100]
        t2_vals = [r['att2'], r['dfw2'], r['exp2'], r['elo2'] / 100]
        x = np.arange(len(stat_names))
        bar_width = 0.35

        bars1 = ax2.bar(x - bar_width / 2, t1_vals, bar_width, label=t1, color=BLUE, edgecolor='#ffffff22', linewidth=0.5)
        bars2 = ax2.bar(x + bar_width / 2, t2_vals, bar_width, label=t2, color=GREEN, edgecolor='#ffffff22', linewidth=0.5)

        for bar in list(bars1) + list(bars2):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f'{bar.get_height():.2f}',
                ha='center', va='bottom',
                color=WHITE, fontsize=8, fontweight='bold'
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels(stat_names, color='#94a3b8', fontsize=9)
        ax2.set_facecolor(CARD)
        ax2.set_title('Team Stats Comparison', color=WHITE, fontsize=12, fontweight='bold', pad=12)
        ax2.legend(facecolor=CARD, edgecolor='#334155', labelcolor=WHITE, fontsize=9)
        ax2.yaxis.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
        ax2.set_axisbelow(True)

        fig.text(0.5, -0.04,
                 f"Predicted Winner: {winner.upper()}  ({max(r['p1'], r['p2']):.1f}% knockout chance)",
                 ha='center', color=AMBER, fontsize=13, fontweight='bold')

        plt.tight_layout(pad=2.5)

        # save with team names in filename
        filename = f"{t1.replace(' ', '_')}_vs_{t2.replace(' ', '_')}.jpg"
        out_path = os.path.join(output_folder, filename)
        plt.savefig(out_path, format='jpeg', dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        print(f"  [Chart saved] -> {out_path}")

    except ImportError:
        print("  matplotlib not installed. Run: pip install matplotlib")


def find_team(user_input, team_list):
    # try to find the team even with partial or lowercase input
    user_lower = user_input.lower()

    # exact match first
    for team in team_list:
        if team.lower() == user_lower:
            return team

    # partial match
    matches = []
    for team in team_list:
        if user_lower in team.lower():
            matches.append(team)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return matches  # ambiguous
    else:
        return None  # not found


# main loop - keep asking until user quits
while True:
    print("-" * 40)
    t1_input = input("  Enter Team 1: ").strip()

    if t1_input.lower() in ('quit', 'exit', 'q'):
        break

    if t1_input.lower() == 'list':
        print("\n  Available teams:")
        print("  " + "\n  ".join(known_teams) + "\n")
        continue

    t2_input = input("  Enter Team 2: ").strip()

    if t2_input.lower() in ('quit', 'exit', 'q'):
        break

    # find team 1
    t1 = find_team(t1_input, known_teams)
    if isinstance(t1, list):
        print(f"  Multiple matches for '{t1_input}': {', '.join(t1)}")
        print("  Please be more specific.")
        continue
    if t1 is None:
        print(f"  Could not find '{t1_input}'. Type 'list' to see all teams.")
        continue

    # find team 2
    t2 = find_team(t2_input, known_teams)
    if isinstance(t2, list):
        print(f"  Multiple matches for '{t2_input}': {', '.join(t2)}")
        print("  Please be more specific.")
        continue
    if t2 is None:
        print(f"  Could not find '{t2_input}'. Type 'list' to see all teams.")
        continue

    # run prediction and show result
    result = predict(t1, t2)
    show_result(result)

    # ask if they want to save a chart
    save = input("  Save chart? (y/n): ").strip().lower()
    if save == 'y':
        save_chart(result)

    # ask if they want another prediction
    again = input("  Predict another match? (y/n): ").strip().lower()
    if again != 'y':
        break

print("\nGoodbye!")
