"""Tests for the 'name: <self.name>' watermark stamped on every plot."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt
import numpy as np

from apeBayes import BayesEpistemicModel
from apeBayes.model import RandomSlopesInteractionModel


def _texts_on(fig) -> list[str]:
    """Collect every string rendered as a matplotlib Text on the figure."""
    return [t.get_text() for t in fig.findobj(plt.Text)]


def _attach(model: BayesEpistemicModel, post, builder=None):
    """Inject a stub posterior + builder so we can call plot methods without fit."""
    model._posterior = post
    model._builder = builder if builder is not None else RandomSlopesInteractionModel()


def test_stamp_present_when_name_set(
    synthetic_long_df, default_config, stub_posterior_v8,
):
    model = BayesEpistemicModel(synthetic_long_df, cfg=default_config, name="roof_v8")
    _attach(model, stub_posterior_v8)

    fig, _ = model.plot_variance_ratio()
    try:
        assert any(
            t == "name: roof_v8" for t in _texts_on(fig)
        ), f"Expected 'name: roof_v8' on figure, got texts: {_texts_on(fig)}"
    finally:
        plt.close(fig)


def test_stamp_absent_when_name_is_none(
    synthetic_long_df, default_config, stub_posterior_v8,
):
    model = BayesEpistemicModel(synthetic_long_df, cfg=default_config)  # name=None
    _attach(model, stub_posterior_v8)

    fig, _ = model.plot_variance_ratio()
    try:
        assert not any(t.startswith("name: ") for t in _texts_on(fig))
    finally:
        plt.close(fig)


def test_stamp_absent_when_toggle_off(
    synthetic_long_df, default_config, stub_posterior_v8,
):
    model = BayesEpistemicModel(synthetic_long_df, cfg=default_config, name="hidden")
    _attach(model, stub_posterior_v8)
    model.show_model_name_on_plots = False

    fig, _ = model.plot_variance_ratio()
    try:
        assert not any("hidden" in t for t in _texts_on(fig))
    finally:
        plt.close(fig)


def test_stamp_handles_single_figure_return(
    synthetic_long_df, default_config, stub_posterior_v8,
):
    """plot_trace returns a bare Figure (not a tuple); stamp must still apply."""
    model = BayesEpistemicModel(synthetic_long_df, cfg=default_config, name="trace_model")
    _attach(model, stub_posterior_v8)

    # plot_trace hits arviz; to avoid needing a fully-specced posterior, just
    # test that _stamp itself handles the bare-Figure path correctly.
    fig = plt.figure()
    try:
        returned = model._stamp(fig)
        assert returned is fig  # _stamp returns the input unchanged
        assert any(t == "name: trace_model" for t in _texts_on(fig))
    finally:
        plt.close(fig)


def test_stamp_tuple_passthrough(
    synthetic_long_df, default_config, stub_posterior_v8,
):
    """_stamp must preserve tuple return shape (fig, axes, ...) unchanged."""
    model = BayesEpistemicModel(synthetic_long_df, cfg=default_config, name="tuple_model")
    _attach(model, stub_posterior_v8)

    fig, axes = plt.subplots(1, 2)
    axes_arr = np.array(axes)
    try:
        result = model._stamp((fig, axes_arr))
        assert isinstance(result, tuple)
        assert result[0] is fig
        assert result[1] is axes_arr
    finally:
        plt.close(fig)


def test_stamp_is_at_bottom_right(
    synthetic_long_df, default_config, stub_posterior_v8,
):
    """Verify the watermark lives in the bottom-right of the figure."""
    model = BayesEpistemicModel(synthetic_long_df, cfg=default_config, name="corner")
    _attach(model, stub_posterior_v8)

    fig = plt.figure()
    try:
        model._stamp(fig)
        matches = [t for t in fig.findobj(plt.Text) if t.get_text() == "name: corner"]
        assert len(matches) == 1
        txt = matches[0]
        x, y = txt.get_position()
        # Bottom-right ⇒ x close to 1, y close to 0 (figure fraction).
        assert x > 0.9, f"expected x near 1.0, got {x}"
        assert y < 0.1, f"expected y near 0.0, got {y}"
        assert txt.get_ha() == "right"
        assert txt.get_va() == "bottom"
    finally:
        plt.close(fig)
