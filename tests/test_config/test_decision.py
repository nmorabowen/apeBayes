"""Tests for DecisionConfig (equivalence-decision parameters)."""
from __future__ import annotations

import pytest

from apeBayes import DecisionConfig, ModelConfig


def test_decision_defaults_match_spec():
    cfg = DecisionConfig()
    assert cfg.alpha_eq == 0.4
    assert cfg.alpha_ladder == (0.4, 0.7, 1.1)
    assert cfg.p_star == 0.95
    assert cfg.denominators == ("src", "gm", "pred")


def test_alpha_eq_must_be_in_ladder():
    with pytest.raises(ValueError, match="alpha_eq"):
        DecisionConfig(alpha_eq=0.5, alpha_ladder=(0.4, 0.7, 1.1))


def test_p_star_range():
    with pytest.raises(ValueError, match="p_star"):
        DecisionConfig(p_star=0.0)
    with pytest.raises(ValueError, match="p_star"):
        DecisionConfig(p_star=1.0)
    with pytest.raises(ValueError, match="p_star"):
        DecisionConfig(p_star=-0.1)


def test_alpha_eq_positive():
    with pytest.raises(ValueError, match="alpha_eq"):
        DecisionConfig(alpha_eq=0.0, alpha_ladder=(0.0,))


def test_alpha_ladder_entries_positive():
    with pytest.raises(ValueError, match="alpha_ladder"):
        DecisionConfig(alpha_eq=0.4, alpha_ladder=(0.4, 0.0, 1.1))


def test_alpha_ladder_nonempty():
    with pytest.raises(ValueError, match="alpha_ladder"):
        DecisionConfig(alpha_eq=0.4, alpha_ladder=())


def test_unknown_denominator_rejected():
    with pytest.raises(ValueError, match="denominator"):
        DecisionConfig(denominators=("src", "bogus"))


def test_duplicate_denominators_rejected():
    with pytest.raises(ValueError, match="unique"):
        DecisionConfig(denominators=("src", "src"))


def test_denominators_nonempty():
    with pytest.raises(ValueError, match="denominators"):
        DecisionConfig(denominators=())


def test_model_config_carries_decision():
    cfg = ModelConfig()
    assert isinstance(cfg.decision, DecisionConfig)
    assert cfg.decision.alpha_eq == 0.4


def test_model_config_accepts_custom_decision():
    dec = DecisionConfig(alpha_eq=0.7, alpha_ladder=(0.4, 0.7, 1.1), p_star=0.90)
    cfg = ModelConfig(decision=dec)
    assert cfg.decision.alpha_eq == 0.7
    assert cfg.decision.p_star == 0.90


def test_custom_denominator_subset():
    cfg = DecisionConfig(denominators=("gm",))
    assert cfg.denominators == ("gm",)
