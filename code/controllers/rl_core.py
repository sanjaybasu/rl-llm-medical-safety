#!/usr/bin/env python3
"""
Lightweight RL core utilities used for the NEJM AI manuscript.

Provides:
- ACTIONS / SAFE_ACTIONS derived from the physician-created scenario library.
- Tabular CQLAgent (conservative Q-learning) and AWRAgent (advantage-weighted).
- Dataset builders and evaluators that operate on Scenario objects from
  code/baselines/evaluate_llm_safety.py.

This module exists so both CQL and AWR controllers can be trained/evaluated
without relying on unpublished helper modules.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# ------------------------------------------------------------------------------
# Load actions and safe-action mapping from the physician-created library
# ------------------------------------------------------------------------------


def _load_scenarios(path: Path) -> List[dict]:
    with path.open() as f:
        return json.load(f)


def _derive_actions_and_safe_sets() -> Tuple[List[str], Dict[str, List[str]]]:
    hazard_path = DATA_DIR / "physician_created" / "hazard_scenarios_train.json"
    benign_path = DATA_DIR / "physician_created" / "benign_scenarios.json"

    actions: set[str] = set()
    safe: Dict[str, List[str]] = {}

    if hazard_path.exists():
        for sc in _load_scenarios(hazard_path):
            h = sc.get("hazard_type", "unknown")
            required = sc.get("required_actions", [])
            actions.update(required)
            safe.setdefault(h, [])
            for a in required:
                if a not in safe[h]:
                    safe[h].append(a)

    # Benign scenarios should map to reassurance/avoid false alarms
    if benign_path.exists():
        safe["benign"] = ["reassure", "avoid_false_alarm"]
    else:
        safe.setdefault("benign", ["reassure", "avoid_false_alarm"])

    # Fallback for unknown states
    safe.setdefault("unknown", ["reassure", "avoid_false_alarm"])

    # Ensure a generic reassurance action exists
    actions.add("reassure")
    actions.add("avoid_false_alarm")

    return sorted(actions), safe


ACTIONS, SAFE_ACTIONS = _derive_actions_and_safe_sets()


# ------------------------------------------------------------------------------
# Dataset builder
# ------------------------------------------------------------------------------


def build_dataset(
    hazard_ids: Iterable[str],
    actions: List[str],
    safe_actions: Dict[str, List[str]],
    penalty: Dict[str, Dict[str, float]] | None = None,
    repeats: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    Construct a simple offline dataset of (state, action, reward) tuples.
    Reward = +1 for safe actions, -1 otherwise, with optional per-action penalties.
    """
    penalty = penalty or {}
    dataset: List[Tuple[str, str, float]] = []
    for _ in range(repeats):
        for hazard in hazard_ids:
            safe = safe_actions.get(hazard, safe_actions.get("unknown", ["reassure"]))
            for action in actions:
                reward = 1.0 if action in safe else -1.0
                if hazard in penalty and action in penalty[hazard]:
                    reward += penalty[hazard][action]
                dataset.append((hazard, action, reward))
    random.shuffle(dataset)
    return dataset


# ------------------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------------------


class CQLAgent:
    """
    Minimal tabular Conservative Q-Learning agent.
    """

    def __init__(
        self,
        hazard_ids: List[str],
        n_actions: int,
        alpha: float = 0.05,
        beta: float = 0.5,
        gamma: float = 0.0,
    ):
        self.state_to_idx = {h: i for i, h in enumerate(hazard_ids)}
        self.idx_to_state = {i: h for h, i in self.state_to_idx.items()}
        self.n_states = len(hazard_ids)
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Q = np.zeros((self.n_states, self.n_actions))

    def train(self, dataset: List[Tuple[str, str, float]], epochs: int = 400) -> None:
        action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
        for _ in range(epochs):
            for hazard, action, reward in dataset:
                s = self.state_to_idx.get(hazard, self.state_to_idx["unknown"])
                a = action_to_idx[action]
                target = reward + self.gamma * np.max(self.Q[s])
                self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * target
                # Conservative penalty to keep unseen actions small
                self.Q[s] -= self.beta * 0.001

    def act(self, hazard: str) -> str:
        s = self.state_to_idx.get(hazard, self.state_to_idx.get("unknown", 0))
        a_idx = int(np.argmax(self.Q[s]))
        return ACTIONS[a_idx]


class AWRAgent:
    """
    Simple Advantage-Weighted Actor with tabular policy logits.
    """

    def __init__(
        self,
        hazard_ids: List[str],
        n_actions: int,
        alpha: float = 0.1,
        beta: float = 0.05,
    ):
        self.state_to_idx = {h: i for i, h in enumerate(hazard_ids)}
        self.idx_to_state = {i: h for h, i in self.state_to_idx.items()}
        self.n_states = len(hazard_ids)
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.logits = np.zeros((self.n_states, self.n_actions))

    def train(self, dataset: List[Tuple[str, str, float]], epochs: int = 400) -> None:
        action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
        # Pre-compute state value baselines
        state_rewards: Dict[str, List[float]] = {}
        for h, _, r in dataset:
            state_rewards.setdefault(h, []).append(r)
        baselines = {h: (sum(rs) / len(rs)) for h, rs in state_rewards.items()}

        for _ in range(epochs):
            for hazard, action, reward in dataset:
                s = self.state_to_idx.get(hazard, self.state_to_idx["unknown"])
                a = action_to_idx[action]
                advantage = reward - baselines.get(hazard, 0.0)
                weight = math.exp(advantage / max(self.beta, 1e-6))
                self.logits[s, a] += self.alpha * weight

    def act(self, hazard: str) -> str:
        s = self.state_to_idx.get(hazard, self.state_to_idx.get("unknown", 0))
        a_idx = int(np.argmax(self.logits[s]))
        return ACTIONS[a_idx]


# ------------------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------------------


def make_selector_from_q(state_to_idx: Dict[str, int], q_table: np.ndarray):
    def selector(hazard: str) -> str:
        s = state_to_idx.get(hazard, state_to_idx.get("unknown", 0))
        return ACTIONS[int(np.argmax(q_table[s]))]

    return selector


def evaluate_agent(state_to_idx, selector, scenarios, detector=None):
    rows = []
    for sc in scenarios:
        true_hazard = getattr(sc, "hazard_type", sc.get("hazard_type", "unknown"))
        prompt = getattr(sc, "prompt", sc.get("prompt", ""))
        context = getattr(sc, "context", sc.get("context", {}))

        if detector is not None:
            det = detector.predict(prompt, context)
            pred_hazard = det.label if det and det.label else "unknown"
        else:
            pred_hazard = true_hazard

        action = selector(pred_hazard)
        safe = SAFE_ACTIONS.get(true_hazard, SAFE_ACTIONS.get("unknown", []))
        is_safe = action in safe

        rows.append(
            {
                "hazard_true": true_hazard,
                "hazard_pred": pred_hazard,
                "chosen_action": action,
                "is_safe": is_safe,
            }
        )
    return rows


def summarize_scenario_metrics(hazard_rows: List[dict], benign_rows: List[dict]):
    tp = sum(r["is_safe"] for r in hazard_rows)
    fn = len(hazard_rows) - tp
    tn = sum(r["is_safe"] for r in benign_rows)
    fp = len(benign_rows) - tn
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "sensitivity": sens, "specificity": spec, "precision": prec, "npv": npv}


# Legacy placeholder; maintained for backward compatibility with imports.
class HazardEnv:
    pass

