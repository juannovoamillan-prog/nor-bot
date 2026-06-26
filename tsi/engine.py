import math


def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def calculate_tsi(elo, form, attack, defense):
    """
    Compute the Team Strength Index (TSI) from raw team metrics.

    elo:     FIFA/Elo rating (expected range 1500–2200)
    form:    recent form score, 0–1
    attack:  attacking strength score, 0–1
    defense: defensive strength score, 0–1

    Returns a float in roughly [0, 1].
    """
    elo_score = normalize(elo, 1500, 2200)

    return round(
        0.45 * elo_score +
        0.25 * form +
        0.20 * attack +
        0.10 * defense,
        4,
    )


def predict_probabilities(tsi_a, tsi_b):
    """
    Convert two TSI scores into win/draw/loss probabilities for team A.

    Uses a logistic function on the TSI delta for the win probability and a
    closeness-based heuristic for the draw probability.
    """
    delta = tsi_a - tsi_b

    p_a_raw = 1 / (1 + math.exp(-5 * delta))
    p_b_raw = 1 - p_a_raw

    closeness = abs(delta)
    draw_prob = max(0.12, 0.28 - closeness * 0.35)

    scale = 1 - draw_prob
    p_a = p_a_raw * scale
    p_b = p_b_raw * scale

    total = p_a + p_b + draw_prob
    return {
        "team_a": round(p_a / total, 4),
        "draw":   round(draw_prob / total, 4),
        "team_b": round(p_b / total, 4),
    }


def calculate_ev(model_prob, odds):
    """Expected value of a bet: (p * decimal_odds) - 1."""
    return round((model_prob * odds) - 1, 4)


def run_match(team_a, team_b):
    """
    Run the full match prediction pipeline for two teams.

    Each team dict must contain:
        name, elo, form, attack, defense, odds
    team_a must also contain draw_odds.

    Returns a dict with TSI scores, probabilities, odds, and EV values.
    """
    tsi_a = calculate_tsi(team_a["elo"], team_a["form"], team_a["attack"], team_a["defense"])
    tsi_b = calculate_tsi(team_b["elo"], team_b["form"], team_b["attack"], team_b["defense"])

    probs = predict_probabilities(tsi_a, tsi_b)

    ev = {
        "team_a": calculate_ev(probs["team_a"], team_a["odds"]),
        "draw":   calculate_ev(probs["draw"],   team_a["draw_odds"]),
        "team_b": calculate_ev(probs["team_b"], team_b["odds"]),
    }

    return {
        "match": f"{team_a['name']} vs {team_b['name']}",
        "team_a": {
            "name":    team_a["name"],
            "elo":     team_a["elo"],
            "form":    team_a["form"],
            "attack":  team_a["attack"],
            "defense": team_a["defense"],
            "tsi":     tsi_a,
        },
        "team_b": {
            "name":    team_b["name"],
            "elo":     team_b["elo"],
            "form":    team_b["form"],
            "attack":  team_b["attack"],
            "defense": team_b["defense"],
            "tsi":     tsi_b,
        },
        "probabilities": probs,
        "odds": {
            "team_a": team_a["odds"],
            "draw":   team_a["draw_odds"],
            "team_b": team_b["odds"],
        },
        "ev": ev,
    }
