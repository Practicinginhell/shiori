"""Community forecast for the water office: how many homes will need urgent water each day.

This is where a local AI model fits: planning, not safety. The kitchen light never uses it.
The history is the town's daily count of homes in the top two delivery tiers (`Community.daily`);
blizzard days are passed as a known future input, the way a weather forecast would be.

    baseline_model   the mean of the last 7 days (what a planner would guess)
    timesfm_model    Google's TimesFM 3.0 on Apple silicon (MLX), if installed; loaded once

`backtest` scores both on the same past days, so the demo shows whether the model really
beats the simple guess. Everything here is SIMULATED data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

# A model takes (history, horizon, storms) where `storms` covers history + horizon days.
Model = Callable[[Sequence[float], int, Sequence[bool]], list[float]]

TIMESFM_CHECKPOINT = "google/timesfm-3.0-pytorch"  # the MLX backend loads these same weights
_timesfm = None


def baseline(history: Sequence[float], horizon: int) -> list[float]:
    """The mean of the last 7 days, repeated."""
    week = list(history[-7:])
    return [sum(week) / len(week) if week else 0.0] * horizon


def baseline_model(history: Sequence[float], horizon: int, storms: Sequence[bool]) -> list[float]:
    return baseline(history, horizon)


def timesfm_available() -> bool:
    try:
        import timesfm3.mlx  # noqa: F401
    except ImportError:
        return False
    return True


def timesfm_model(history: Sequence[float], horizon: int, storms: Sequence[bool]) -> list[float]:
    """TimesFM 3.0 (MLX) with the storm schedule as a past-and-future covariate.

    The first call loads the model (about 1.3 GB, downloaded once from Hugging Face).
    The weights are under the TimesFM non-commercial licence: fine for a demo, not a product.
    """
    global _timesfm
    import numpy as np
    from timesfm3.mlx import TimesFM3Forecaster

    if _timesfm is None:
        _timesfm = TimesFM3Forecaster.from_pretrained(TIMESFM_CHECKPOINT)
    covariate = np.asarray([[1.0 if s else 0.0 for s in storms[: len(history) + horizon]]], dtype=np.float32)
    out = _timesfm.predict(np.asarray(history, dtype=np.float32), horizon=horizon,
                           past_future_covariates=covariate, make_positive=True)
    return [max(0.0, float(v)) for v in out.forecast]


def backtest(series: Sequence[float], storms: Sequence[bool], models: dict[str, Model], horizon: int = 7,
             origins: int = 4) -> dict[str, float]:
    """Mean absolute error of each model over the last `origins` forecast weeks of `series`."""
    errors: dict[str, list[float]] = {name: [] for name in models}
    for k in range(origins, 0, -1):
        cut = len(series) - k * horizon
        history, actual = series[:cut], series[cut:cut + horizon]
        for name, model in models.items():
            predicted = model(history, horizon, storms[:cut + horizon])
            errors[name].extend(abs(p - a) for p, a in zip(predicted, actual, strict=True))
    return {name: round(sum(e) / len(e), 2) for name, e in errors.items()}


def town_inputs(town, horizon: int = 7) -> tuple[list[float], list[bool]]:
    """The town's daily urgent-water counts, and its storm schedule over history + horizon."""
    series = [float(r["urgent_water"]) for r in town.daily]
    storms = [r["storm"] for r in town.daily] + [town.is_storm_day(d) for d in range(len(series), len(series) + horizon)]
    return series, storms


def forecast_week(series: Sequence[float], storms: Sequence[bool], n_homes: int, use_ai: bool = True,
                  horizon: int = 7) -> dict:
    """Score the models on the last 4 weeks, then forecast with the better one (capped at the town size)."""
    models: dict[str, Model] = {"baseline (last 7 days)": baseline_model}
    if use_ai and timesfm_available():
        models["TimesFM 3.0 (local, MLX)"] = timesfm_model
    # Score only the weeks there is history for (a model needs at least a week before each).
    origins = min(4, max(0, (len(series) - 7) // horizon))
    model_error = None
    try:
        scores = backtest(series, storms, models, horizon=horizon, origins=origins) if origins else {}
        best = min(scores, key=lambda name: scores[name]) if scores else "baseline (last 7 days)"
        predicted = models[best](series, horizon, storms)
    except Exception as exc:  # e.g. the weights can't be downloaded offline: plan with the baseline
        model_error = f"{type(exc).__name__}: {exc}"
        scores, best = {}, "baseline (last 7 days)"
        predicted = baseline(series, horizon)
    return {"scores": scores, "model": best, "predicted": [round(min(float(n_homes), v), 1) for v in predicted],
            "timesfm_installed": timesfm_available(), "model_error": model_error}
