import pandas as pd
import numpy as np
import math
import random
import os
import csv
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# figure out where the project folder is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# paths to the data files
results_file = os.path.join(PROJECT_ROOT, 'data', 'raw', '127', 'results.csv')
fifa_file = r'C:\Users\tusha\OneDrive\Desktop\Projects\fifa-predictor\data\raw\fifa_ranking_2026-06-08.csv'
output_folder = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(output_folder, exist_ok=True)

# some teams have different names in different datasets
# so i made a dictionary to fix them all to one standard name
name_fixes = {
    'United States': 'USA',
    'United States of America': 'USA',
    'Korea Republic': 'South Korea',
    'Republic of Korea': 'South Korea',
    'IR Iran': 'Iran',
    "Cote d'Ivoire": 'Ivory Coast',
    "Cote dIvoire": 'Ivory Coast',
    "Cote d Ivoire": 'Ivory Coast',
    "Cote dIvoire": 'Ivory Coast',
    "Côte d'Ivoire": 'Ivory Coast',
    'Czech Republic': 'Czechia',
    'West Germany': 'Germany',
    'Republic of Ireland': 'Ireland',
    'FYR Macedonia': 'North Macedonia',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Bosnia & Herzegovina': 'Bosnia-Herzegovina',
    'Cabo Verde': 'Cape Verde',
    'Cape Verde Islands': 'Cape Verde',
    'DR Congo': 'DR Congo',
    'Congo DR': 'DR Congo',
    'Congo, DR': 'DR Congo',
    'Democratic Republic of Congo': 'DR Congo',
    'Congo': 'DR Congo',
}

def fix_name(name):
    # if the name is missing just return it
    if pd.isna(name):
        return name
    name = str(name).strip()
    # check if we need to rename it, otherwise keep it
    if name in name_fixes:
        return name_fixes[name]
    return name


print("=" * 55)
print("  FIFA WC 2026 KNOCKOUT STAGE PREDICTOR")
print("=" * 55)
print("\nStep 1: Loading match history...")

# check the file actually exists before loading
if not os.path.exists(results_file):
    print("ERROR: cant find results.csv at:", results_file)
    exit()

# load all historical match data
all_matches = pd.read_csv(results_file)
all_matches['home_team'] = all_matches['home_team'].apply(fix_name)
all_matches['away_team'] = all_matches['away_team'].apply(fix_name)
all_matches['date'] = pd.to_datetime(all_matches['date'], errors='coerce')
all_matches['year'] = all_matches['date'].dt.year

# only keep matches from 2020 onwards (recent form matters more)
recent_matches = all_matches[all_matches['year'] >= 2020].copy()
recent_matches = recent_matches.dropna(subset=['home_score', 'away_score'])
recent_matches = recent_matches.sort_values('date').reset_index(drop=True)
print(f"  Loaded {len(recent_matches):,} completed matches (2020-present)")

# world cup matches should count more than random friendlies
# so i give each tournament type a weight/score
tournament_weights = {
    'FIFA World Cup': 3.0,
    'FIFA World Cup qualification': 2.0,
    'UEFA Euro': 2.5,
    'Copa America': 2.5,
    'AFC Asian Cup': 2.5,
    'Africa Cup of Nations': 2.5,
    'CONCACAF Gold Cup': 2.0,
    'UEFA Euro qualification': 1.5,
    'Copa America qualification': 1.5,
    'Africa Cup of Nations qualification': 1.5,
    'Friendly': 0.5,
    'Friendlies': 0.5,
    'International Friendly': 0.5,
}

def get_tournament_weight(tournament_name):
    if pd.isna(tournament_name):
        return 1.0
    # look it up, default to 1.0 if not in list
    return tournament_weights.get(str(tournament_name), 1.0)

recent_matches['weight'] = recent_matches['tournament'].apply(get_tournament_weight)

# calculate global average goals (baseline for the model)
# using last 500 matches to get a recent baseline
last_500 = recent_matches.tail(500)
avg_goals = (last_500['home_score'].mean() + last_500['away_score'].mean()) / 2
print(f"  Global avg goals per team per match: {avg_goals:.3f}")


print("\nStep 2: Loading FIFA ratings...")

def load_fifa_ratings(filepath):
    # check file exists
    if not os.path.exists(filepath):
        print("  FIFA file not found, using default 1500 for all teams")
        return {}

    df_ratings = pd.read_csv(filepath)

    # figure out which columns have team name and points
    team_col = None
    pts_col = None
    for col in df_ratings.columns:
        col_lower = col.lower()
        if col_lower in ('country_full', 'team', 'team_name', 'name', 'country'):
            team_col = col
        if col_lower in ('total_points', 'points', 'rating', 'elo', 'elo_rating'):
            pts_col = col

    if team_col is None or pts_col is None:
        print("  Could not find the right columns in fifa file:", df_ratings.columns.tolist())
        return {}

    ratings = {}
    for i, row in df_ratings.iterrows():
        try:
            team_name = fix_name(str(row[team_col]))
            rating = float(row[pts_col])
            ratings[team_name] = rating
        except:
            pass

    print(f"  Loaded FIFA ratings for {len(ratings)} teams")
    return ratings

raw_ratings = load_fifa_ratings(fifa_file)

# scale the FIFA points to a 1000-2000 range (like ELO ratings)
# this makes the math easier and more consistent
if raw_ratings:
    min_pts = min(raw_ratings.values())
    max_pts = max(raw_ratings.values())
    scaled_ratings = {}
    for team, pts in raw_ratings.items():
        scaled_ratings[team] = 1000 + (pts - min_pts) / max(max_pts - min_pts, 1) * 1000
else:
    scaled_ratings = {}
    print("  Using default 1500 for all teams")

def get_team_rating(team):
    # return rating or 1500 if we dont have data for them
    return scaled_ratings.get(team, 1500)


# WC 2026 Round of 32 bracket
# format is (team1, team2, match_number)
left_side = [
    ('Canada',      'South Africa', 73),
    ('Spain',       'Austria',      74),
    ('Germany',     'Paraguay',     75),
    ('Portugal',    'Croatia',      76),
    ('Ivory Coast', 'Norway',       77),
    ('Australia',   'Egypt',        78),
    ('France',      'Sweden',       79),
    ('Argentina',   'Cape Verde',   80),
]

right_side = [
    ('Brazil',           'Japan',              81),
    ('Netherlands',      'Morocco',            82),
    ('Mexico',           'Ecuador',            83),
    ('Colombia',         'Ghana',              84),
    ('England',          'DR Congo',           85),
    ('Belgium',          'Senegal',            86),
    ('USA',              'Bosnia-Herzegovina', 87),
    ('Switzerland',      'Algeria',            88),
]

# these matches have already been played - i lock in the real results
# so they dont get re-simulated
real_results = {
    73: 'Canada',    # Canada 1-0 South Africa
    75: 'Paraguay',  # Paraguay beat Germany on penalties (1-1, 4-3 pens)
    81: 'Brazil',    # Brazil 2-1 Japan
    82: 'Morocco',   # Morocco beat Netherlands on penalties (1-1, 3-2 pens)
}

# collect all 32 teams from the bracket
all_teams = []
for t1, t2, match_id in left_side + right_side:
    if t1 not in all_teams:
        all_teams.append(t1)
    if t2 not in all_teams:
        all_teams.append(t2)
all_teams = sorted(all_teams)


print("\nStep 3: Computing team features from match history...")
print(f"\n  {'Team':<28} {'Attack':>7} {'Defense':>9} {'FIFA Elo':>9}")
print("  " + "-" * 58)

def get_team_stats(team):
    # get all matches this team played
    team_matches = recent_matches[
        (recent_matches['home_team'] == team) | (recent_matches['away_team'] == team)
    ].tail(20)  # only use last 20 matches

    if len(team_matches) == 0:
        return 1.0, 1.0  # no data, assume average

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

    # calculate weighted averages
    w_array = np.array(weights, dtype=float)
    s_array = np.array(goals_scored, dtype=float)
    c_array = np.array(goals_conceded, dtype=float)
    total_weight = w_array.sum()

    if total_weight == 0:
        return 1.0, 1.0

    avg_scored = (s_array * w_array).sum() / total_weight
    avg_conceded = (c_array * w_array).sum() / total_weight

    # attack strength = how many goals they score vs global average
    # 1.0 means average, 1.5 means 50% better than average
    if avg_goals > 0:
        attack = avg_scored / avg_goals
    else:
        attack = 1.0

    # clamp to avoid crazy outliers
    attack = max(0.5, min(attack, 3.0))

    # defensive weakness = how many goals they concede vs average
    # lower is better (harder to score against)
    if avg_goals > 0:
        defense = avg_conceded / avg_goals
    else:
        defense = 1.0

    defense = max(0.5, min(defense, 3.0))

    return attack, defense

# compute and store stats for all 32 teams
team_stats = {}
for team in sorted(all_teams):
    att, dfw = get_team_stats(team)
    elo = get_team_rating(team)
    team_stats[team] = (att, dfw)
    print(f"  {team:<28} {att:>7.3f} {dfw:>9.3f} {elo:>9.0f}")


# poisson probability - chance of scoring exactly k goals when average is lam
# this is a standard stats formula
def poisson(k, lam):
    if lam <= 0:
        if k == 0:
            return 1.0
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict_match(team1, team2):
    # get each teams stats
    att1, dfw1 = team_stats.get(team1, (1.0, 1.0))
    att2, dfw2 = team_stats.get(team2, (1.0, 1.0))

    # expected goals = global avg * how good they attack * how weak the opponent defends
    expected1 = max(avg_goals * att1 * dfw2, 0.1)
    expected2 = max(avg_goals * att2 * dfw1, 0.1)

    # build a grid of all possible scorelines from 0-0 to 8-8
    # and add up the probabilities
    team1_wins = 0.0
    draw = 0.0
    team2_wins = 0.0

    for goals1 in range(9):
        for goals2 in range(9):
            prob = poisson(goals1, expected1) * poisson(goals2, expected2)
            if goals1 > goals2:
                team1_wins += prob
            elif goals1 == goals2:
                draw += prob
            else:
                team2_wins += prob

    # normalize (small rounding correction since we capped at 8 goals)
    total = team1_wins + draw + team2_wins
    if total > 0:
        team1_wins = team1_wins / total
        draw = draw / total
        team2_wins = team2_wins / total

    # also factor in FIFA rating difference using ELO formula
    rating_diff = get_team_rating(team1) - get_team_rating(team2)
    elo_prob_team1 = 1 / (1 + 10 ** (-rating_diff / 400))

    # blend: 60% elo rating + 40% poisson stats
    blended1 = 0.6 * elo_prob_team1 + 0.4 * team1_wins
    remaining = 1.0 - blended1
    draw_plus_loss = draw + team2_wins

    if draw_plus_loss > 0:
        blended_draw = remaining * (draw / draw_plus_loss)
        blended2 = remaining * (team2_wins / draw_plus_loss)
    else:
        blended_draw = 0.0
        blended2 = remaining

    return blended1, blended_draw, blended2, expected1, expected2


def pick_winner(team1, team2):
    # simulate one knockout match - must have a winner, no draws
    p1, pd, p2, e1, e2 = predict_match(team1, team2)

    # split the draw probability 50/50 between both teams
    p1_knockout = p1 + pd / 2
    p2_knockout = p2 + pd / 2

    rand = random.random()
    if rand < p1_knockout:
        return team1
    else:
        return team2


def run_tournament():
    # simulate one full World Cup tournament
    # returns the name of the champion

    # Round of 32 - left side of bracket
    left_r32 = []
    for t1, t2, match_id in left_side:
        if match_id in real_results:
            # use the actual result
            left_r32.append(real_results[match_id])
        else:
            left_r32.append(pick_winner(t1, t2))

    # Round of 32 - right side of bracket
    right_r32 = []
    for t1, t2, match_id in right_side:
        if match_id in real_results:
            right_r32.append(real_results[match_id])
        else:
            right_r32.append(pick_winner(t1, t2))

    # Round of 16 - pair up the winners (0 vs 1, 2 vs 3 etc)
    left_r16 = []
    for i in range(0, len(left_r32) - 1, 2):
        left_r16.append(pick_winner(left_r32[i], left_r32[i+1]))

    right_r16 = []
    for i in range(0, len(right_r32) - 1, 2):
        right_r16.append(pick_winner(right_r32[i], right_r32[i+1]))

    # Quarter Finals
    left_qf = []
    for i in range(0, len(left_r16) - 1, 2):
        left_qf.append(pick_winner(left_r16[i], left_r16[i+1]))

    right_qf = []
    for i in range(0, len(right_r16) - 1, 2):
        right_qf.append(pick_winner(right_r16[i], right_r16[i+1]))

    # Semi Finals
    left_sf_winner = pick_winner(left_qf[0], left_qf[1])
    right_sf_winner = pick_winner(right_qf[0], right_qf[1])

    # Final
    champion = pick_winner(left_sf_winner, right_sf_winner)
    return champion


# run 10000 simulations and count how many times each team wins
total_sims = 10000
print(f"\nStep 4: Running {total_sims:,} Monte Carlo simulations...")

win_counts = defaultdict(int)
for i in range(total_sims):
    winner = run_tournament()
    if winner:
        win_counts[winner] += 1
    if (i + 1) % 2000 == 0:
        print(f"  {i+1:,} / {total_sims:,} completed")

# sort by most wins
results = []
for team in all_teams:
    wins = win_counts.get(team, 0)
    prob = round(wins / total_sims * 100, 2)
    results.append({'team': team, 'wins': wins, 'prob': prob})

results = sorted(results, key=lambda x: x['prob'], reverse=True)


# print the final table
print("\n")
print("=" * 58)
print("  FIFA WC 2026 - CHAMPIONSHIP PROBABILITIES")
print(f"  Based on {total_sims:,} Monte Carlo simulations")
print("=" * 58)
print(f"  {'Rank':<5} {'Team':<26} {'Prob':>8}   Visual")
print("  " + "-" * 55)
for rank, r in enumerate(results, 1):
    bar = '#' * int(r['prob'])
    print(f"  {rank:<5} {r['team']:<26} {r['prob']:>6.2f}%  {bar}")
print("  " + "-" * 55)


# also predict the most likely winner for each match step by step
print("\n")
print("=" * 70)
print("  PREDICTED MATCH-BY-MATCH BRACKET PATH")
print("  (most likely winner + win probability for each match)")
print("=" * 70)

def get_likely_winner(t1, t2, match_id=None):
    # if we already know the result, return it
    if match_id is not None and match_id in real_results:
        return real_results[match_id], 1.0

    p1, pd, p2, lam1, lam2 = predict_match(t1, t2)
    p1_ko = p1 + pd / 2
    p2_ko = p2 + pd / 2

    if p1_ko >= p2_ko:
        return t1, p1_ko
    else:
        return t2, p2_ko

def print_header():
    print(f"\n  {'M#':<5} {'Team 1':<25} {'Team 2':<25} {'Likely Winner':<25} {'Win Prob'}")
    print("  " + "-" * 88)

def print_match_row(match_id, t1, t2, winner, prob, note=""):
    if note:
        extra = f"  [{note}]"
    else:
        extra = ""
    print(f"  {match_id:<5} {t1:<25} {t2:<25} {winner:<25} {prob:.1%}{extra}")


# Round of 32
print("\n  ROUND OF 32")
print_header()

left_predicted = []
right_predicted = []

for t1, t2, match_id in left_side:
    winner, prob = get_likely_winner(t1, t2, match_id)
    if match_id in real_results:
        note = "PLAYED"
    else:
        note = ""
    print_match_row(f"M{match_id}", t1, t2, winner, prob, note)
    left_predicted.append(winner)

for t1, t2, match_id in right_side:
    winner, prob = get_likely_winner(t1, t2, match_id)
    if match_id in real_results:
        note = "PLAYED"
    else:
        note = ""
    print_match_row(f"M{match_id}", t1, t2, winner, prob, note)
    right_predicted.append(winner)


# Round of 16
print("\n  ROUND OF 16")
print_header()

left_r16_predicted = []
match_num = 89
for i in range(0, len(left_predicted) - 1, 2):
    t1 = left_predicted[i]
    t2 = left_predicted[i + 1]
    winner, prob = get_likely_winner(t1, t2)
    print_match_row(f"M{match_num}", t1, t2, winner, prob)
    left_r16_predicted.append(winner)
    match_num += 1

right_r16_predicted = []
match_num = 93
for i in range(0, len(right_predicted) - 1, 2):
    t1 = right_predicted[i]
    t2 = right_predicted[i + 1]
    winner, prob = get_likely_winner(t1, t2)
    print_match_row(f"M{match_num}", t1, t2, winner, prob)
    right_r16_predicted.append(winner)
    match_num += 1


# Quarter Finals
print("\n  QUARTER-FINALS")
print_header()

left_qf_predicted = []
match_num = 97
for i in range(0, len(left_r16_predicted) - 1, 2):
    t1 = left_r16_predicted[i]
    t2 = left_r16_predicted[i + 1]
    winner, prob = get_likely_winner(t1, t2)
    print_match_row(f"M{match_num}", t1, t2, winner, prob)
    left_qf_predicted.append(winner)
    match_num += 1

right_qf_predicted = []
match_num = 99
for i in range(0, len(right_r16_predicted) - 1, 2):
    t1 = right_r16_predicted[i]
    t2 = right_r16_predicted[i + 1]
    winner, prob = get_likely_winner(t1, t2)
    print_match_row(f"M{match_num}", t1, t2, winner, prob)
    right_qf_predicted.append(winner)
    match_num += 1


# Semi Finals
print("\n  SEMI-FINALS")
print_header()

sf1_winner, sf1_prob = get_likely_winner(left_qf_predicted[0], left_qf_predicted[1])
sf2_winner, sf2_prob = get_likely_winner(right_qf_predicted[0], right_qf_predicted[1])
print_match_row("M101", left_qf_predicted[0], left_qf_predicted[1], sf1_winner, sf1_prob)
print_match_row("M102", right_qf_predicted[0], right_qf_predicted[1], sf2_winner, sf2_prob)


# Final
print("\n  WORLD CUP FINAL  (July 19, 2026 - MetLife Stadium, New York)")
print_header()

final_winner, final_prob = get_likely_winner(sf1_winner, sf2_winner)
print_match_row("M104", sf1_winner, sf2_winner, final_winner, final_prob)

print(f"\n  *** PREDICTED CHAMPION: {final_winner.upper()} ***")
print(f"  (Predicted win probability in the Final: {final_prob:.1%})")
print()


# generate a bar chart of the results
print("Step 5: Generating results chart...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # only show teams with more than 0.05% chance, max 20 teams
    chart_teams = [r for r in results if r['prob'] > 0.05][:20]
    team_names = [r['team'] for r in chart_teams]
    probabilities = [r['prob'] for r in chart_teams]

    # color the bars differently
    bar_colors = []
    for i, r in enumerate(chart_teams):
        if r['team'] == final_winner:
            bar_colors.append('#FFD700')  # gold for predicted champion
        elif i < 4:
            bar_colors.append('#1565C0')  # blue for top 4
        else:
            bar_colors.append('#455A64')  # grey for the rest

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d1b2a')

    # horizontal bar chart
    bars = ax.barh(
        team_names[::-1],
        probabilities[::-1],
        color=bar_colors[::-1],
        edgecolor='#ffffff33',
        linewidth=0.5,
        height=0.65
    )

    # add percentage label at end of each bar
    for bar, prob in zip(bars, probabilities[::-1]):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f'{prob:.1f}%',
            va='center',
            ha='left',
            color='white',
            fontsize=9.5,
            fontweight='bold'
        )

    ax.set_xlabel('Championship Probability (%)', color='#cccccc', fontsize=11, labelpad=10)
    ax.set_title(
        f'FIFA World Cup 2026\nKnockout Stage - Championship Probabilities\n'
        f'({total_sims:,} Monte Carlo Simulations)  |  Predicted Champion: {final_winner}',
        color='white', fontsize=13, fontweight='bold', pad=14, linespacing=1.6
    )
    ax.tick_params(colors='#cccccc', labelsize=9)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#444')

    ax.set_xlim(0, max(probabilities) * 1.25)

    legend_items = [
        mpatches.Patch(color='#FFD700', label=f'Predicted Champion ({final_winner})'),
        mpatches.Patch(color='#1565C0', label='Top 4 contenders'),
        mpatches.Patch(color='#455A64', label='Other teams'),
    ]
    ax.legend(
        handles=legend_items,
        loc='lower right',
        facecolor='#0d0d1a',
        edgecolor='#555',
        labelcolor='white',
        fontsize=9
    )

    ax.xaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    chart_path = os.path.join(output_folder, 'wc2026_championship_probabilities.jpg')
    plt.savefig(chart_path, format='jpeg', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] Chart saved to: {chart_path}")

except ImportError:
    print("  matplotlib not installed. Run: pip install matplotlib")
    print("  Skipping chart.")


# save results to csv
csv_path = os.path.join(output_folder, 'wc2026_predictions.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['rank', 'team', 'wins', 'probability_pct'])
    writer.writeheader()
    for rank, r in enumerate(results, 1):
        writer.writerow({
            'rank': rank,
            'team': r['team'],
            'wins': r['wins'],
            'probability_pct': r['prob']
        })

print(f"  [SAVED] CSV saved to:   {csv_path}")
print("\n[DONE]  All predictions complete!")