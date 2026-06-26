import pytest
from tsi import calculate_tsi, predict_probabilities, calculate_ev, run_match


def test_normalize_bounds():
    assert calculate_tsi(1500, 0, 0, 0) == pytest.approx(0.0, abs=1e-4)
    assert calculate_tsi(2200, 1, 1, 1) == pytest.approx(1.0, abs=1e-4)


def test_tsi_weights():
    # Only elo at midpoint (1850), rest zero → 0.45 * 0.5 = 0.225
    assert calculate_tsi(1850, 0, 0, 0) == pytest.approx(0.225, abs=1e-4)


def test_probabilities_sum_to_one():
    probs = predict_probabilities(0.6, 0.4)
    total = probs["team_a"] + probs["draw"] + probs["team_b"]
    assert total == pytest.approx(1.0, abs=1e-4)


def test_probabilities_symmetry():
    p = predict_probabilities(0.5, 0.5)
    assert p["team_a"] == pytest.approx(p["team_b"], abs=1e-4)


def test_ev_fair_odds():
    # If model prob == implied prob the EV should be 0
    # e.g. prob=0.5, odds=2.0 → EV = 0.5*2 - 1 = 0
    assert calculate_ev(0.5, 2.0) == pytest.approx(0.0, abs=1e-4)


def test_run_match_keys():
    team_a = {"name": "A", "elo": 2000, "form": 0.7, "attack": 0.7, "defense": 0.7, "odds": 2.0, "draw_odds": 3.5}
    team_b = {"name": "B", "elo": 1800, "form": 0.5, "attack": 0.5, "defense": 0.5, "odds": 3.5, "draw_odds": 3.5}
    result = run_match(team_a, team_b)
    for key in ("match", "team_a", "team_b", "probabilities", "odds", "ev"):
        assert key in result


def test_run_match_probabilities_sum():
    team_a = {"name": "A", "elo": 2000, "form": 0.7, "attack": 0.7, "defense": 0.7, "odds": 2.0, "draw_odds": 3.5}
    team_b = {"name": "B", "elo": 1800, "form": 0.5, "attack": 0.5, "defense": 0.5, "odds": 3.5, "draw_odds": 3.5}
    result = run_match(team_a, team_b)
    p = result["probabilities"]
    assert p["team_a"] + p["draw"] + p["team_b"] == pytest.approx(1.0, abs=1e-4)
