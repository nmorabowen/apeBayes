"""Tests for per-variant sigma_GM dispatch per uncertanty_measures.md \u00a77."""
from __future__ import annotations

import numpy as np
import pytest

from apeBayes.model import (
    FlatConfigModel,
    HierarchicalConfigModel,
    RandomSlopesInteractionModel,
    RandomSlopesModel,
)


def test_flat_sigma_GM_is_sigma_src(stub_posterior_v3):
    """v1-v3: \u03c3_GM = \u03c3_src (no interaction layer)."""
    model = FlatConfigModel()
    gm = model.sigma_GM(stub_posterior_v3)
    np.testing.assert_array_equal(gm, stub_posterior_v3.sigma_src())


def test_hierarchical_sigma_GM_is_sigma_src(stub_posterior_v3):
    """Hierarchical: \u03c3_GM = \u03c3_src (\u03c4_config is prior, not aleatory)."""
    model = HierarchicalConfigModel()
    gm = model.sigma_GM(stub_posterior_v3)
    np.testing.assert_array_equal(gm, stub_posterior_v3.sigma_src())


def test_random_slopes_sigma_GM_is_sigma_src(stub_posterior_v8):
    """v4-v7: \u03c3_GM = \u03c3_src (\u03bb_case absorbed into effective slope)."""
    model = RandomSlopesModel()
    gm = model.sigma_GM(stub_posterior_v8)
    np.testing.assert_array_equal(gm, stub_posterior_v8.sigma_src())


def test_random_slopes_interaction_v8_sigma_GM(stub_posterior_v8):
    """v8: \u03c3_GM = \u221a(\u03c3_src\u00b2 + \u03c3_inter\u00b2)."""
    model = RandomSlopesInteractionModel()
    gm = model.sigma_GM(stub_posterior_v8)
    p = stub_posterior_v8
    expected = np.sqrt(p.sigma_src() ** 2 + p.sigma_inter() ** 2)
    np.testing.assert_allclose(gm, expected)


def test_v8_sigma_GM_magnitude_plausible(stub_posterior_v8):
    """Sanity: v8 \u03c3_GM should land in the seismic-application plausible range.

    Spec note (uncertanty_measures.md \u00a75): realistic posteriors land
    \u03c3_GM \u2208 [0.5, 0.8]. Our stub uses \u03c3_src ~ N(0.55, 0.05) and
    \u03c3_inter ~ N(0.30, 0.04), so \u221a(\u03c3_src\u00b2 + \u03c3_inter\u00b2) \u2248 0.63.
    """
    model = RandomSlopesInteractionModel()
    gm = model.sigma_GM(stub_posterior_v8)
    med = float(np.median(gm))
    assert 0.5 <= med <= 0.8, f"\u03c3_GM median {med:.3f} outside plausible band [0.5, 0.8]"


def test_v9_sigma_GM_raises(stub_posterior_v9):
    """v9 (xi_case present): NotImplementedError, not a silent v8 reuse."""
    model = RandomSlopesInteractionModel(interaction_loading=True)
    with pytest.raises(NotImplementedError, match=r"v9"):
        model.sigma_GM(stub_posterior_v9)


def test_v9_error_references_spec_section(stub_posterior_v9):
    """Error message should point at the spec so future-you knows where to look."""
    model = RandomSlopesInteractionModel(interaction_loading=True)
    with pytest.raises(NotImplementedError, match=r"uncertanty_measures\.md"):
        model.sigma_GM(stub_posterior_v9)


def test_interaction_model_missing_sigma_inter_raises(stub_posterior_v3):
    """v8 formula on a posterior without \u03c3_inter surfaces a clear error."""
    model = RandomSlopesInteractionModel()
    with pytest.raises(ValueError, match=r"sigma_inter"):
        model.sigma_GM(stub_posterior_v3)
