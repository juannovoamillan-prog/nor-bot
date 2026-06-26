#!/usr/bin/env python3
"""
CLI entry point — run a TSI match prediction and print JSON results.

Usage:
    python predict.py                  # runs the built-in World Cup example
"""

import json
from tsi import run_match

france = {
    "name": "France",
    "elo": 2050,
    "form": 0.82,
    "attack": 0.86,
    "defense": 0.80,
    "odds": 2.05,
    "draw_odds": 3.40,
}

senegal = {
    "name": "Senegal",
    "elo": 1880,
    "form": 0.68,
    "attack": 0.70,
    "defense": 0.72,
    "odds": 3.80,
    "draw_odds": 3.40,
}

if __name__ == "__main__":
    result = run_match(france, senegal)
    print(json.dumps(result, indent=2))
