"""Foundation-model live track (ADR 0010, protocol v2): TTM r2.1 zero-shot and its mean with the headline.

For the latest origin t and each horizon h, the model reads the last 512 values of the annualised log
realised volatility over h returns (`features.ttm_input`, prices through t, no OVX) and its output at
step h is the log of the forecast volatility. `ttm_headline_mean` is the equal-weight mean in log
volatility with the headline forecast of the primary track for the same asset and origin. Records have
the same write-once format as the primary track (`oilfc.live.records`), in a separate folder.

Only the standard library, numpy and pandas are needed here. torch and granite-tsfm are imported when a
`TTMPredictor` is built, because their pinned versions cannot live in the main requirements.
"""

from importlib import metadata

import numpy as np
import pandas as pd

from oilfc.features.ttm_input import ttm_input

SPEC_VERSION = "ttm-1"
MODELS = ("ttm", "ttm_headline_mean")
HORIZONS = (5, 22, 66)
HEADLINE = "lhar_ovx_bc_lag1"
REPO = "ibm-granite/granite-timeseries-ttm-r2"
BRANCH = "512-96-ft-r2.1"
REVISION = "3442406f106122f414a11f72b4ebfe4878bec89d"  # commit of BRANCH, committed 2025-02-26
CONTEXT = 512
PREDICTION_LENGTH = 96
DAILY_FREQ_TOKEN = 8  # tsfm_public DEFAULT_FREQUENCY_MAPPING["d"]
FIRST_ORIGIN_AFTER = pd.Timestamp("2026-11-27")  # date of the protocol-v2-frozen tag


class NotScoredOrigin(ValueError):
    """The origin cannot be part of the claim (too early, or no valid input)."""


class TTMPredictor:
    """TTM r2.1 `512-96-ft-r2.1`, pinned to a commit, run zero-shot on the CPU."""

    def __init__(self, repo: str = REPO, revision: str = REVISION):
        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        self._torch = torch
        self.model = TinyTimeMixerForPrediction.from_pretrained(repo, revision=revision).eval()
        cfg = self.model.config
        assert (cfg.context_length, cfg.prediction_length) == (CONTEXT, PREDICTION_LENGTH)

    def versions(self) -> dict:
        names = ("torch", "transformers", "granite-tsfm", "numpy")
        return {n: metadata.version(n) for n in names}

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Next PREDICTION_LENGTH values of a series, given its last CONTEXT values."""
        torch = self._torch
        past = torch.tensor(np.asarray(context, dtype=np.float32)).reshape(1, CONTEXT, 1)
        with torch.no_grad():
            out = self.model(
                past_values=past,
                freq_token=torch.tensor([DAILY_FREQ_TOKEN]),
                return_loss=False,
            )
        return out.prediction_outputs[0, :, 0].numpy().astype(float)


def returns_of(panel: pd.DataFrame, asset: str, origin: pd.Timestamp | None = None) -> pd.Series:
    """Daily log returns of the asset's spot price, up to and including `origin` when given."""
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = np.log(price).diff().dropna()
    return r if origin is None else r.loc[:origin]


def ttm_forecasts(
    panel: pd.DataFrame,
    asset: str,
    headline_file: dict,
    predictor,
    horizons: tuple[int, ...] = HORIZONS,
    origin: pd.Timestamp | None = None,
    code_version: str | None = None,
) -> dict:
    """Record of annualised volatility forecasts (decimal) for the latest origin, per horizon.

    headline_file: the stored primary record (`{"content": ..., "sha256": ...}`) for the same asset and
    origin. predictor: anything with `predict(context)` (and `versions()` for the record). Information
    timing: prices through the origin only. Raises NotScoredOrigin if the origin is not after the
    protocol-v2 tag or any input value is not finite.
    """
    r = returns_of(panel, asset, origin)
    origin_date = str(r.index[-1].date())
    head = headline_file["content"]
    if (head["asset"], head["origin_date"]) != (asset, origin_date):
        raise ValueError("the headline record is for another asset or origin")
    if r.index[-1] <= FIRST_ORIGIN_AFTER:
        raise NotScoredOrigin(f"origin {origin_date} is not after {FIRST_ORIGIN_AFTER.date()}")
    out = {}
    for h in horizons:
        x = ttm_input(r, h).iloc[-CONTEXT:]
        if len(x) < CONTEXT or not np.isfinite(x.to_numpy()).all():
            raise NotScoredOrigin(f"{asset} {h}d: the input series has no 512 finite values")
        log_vol = float(predictor.predict(x.to_numpy())[h - 1])
        headline = float(head["horizons"][str(h)][HEADLINE])
        row = {
            "ttm": float(np.exp(log_vol)),
            "ttm_headline_mean": float(np.exp(0.5 * log_vol + 0.5 * np.log(headline))),
            "headline": headline,
        }
        bad = [m for m, v in row.items() if not np.isfinite(v) or v <= 0]
        if bad:
            raise NotScoredOrigin(f"{asset} {h}d: no valid forecast for {bad}")
        out[str(h)] = {m: round(v, 8) for m, v in row.items()}
    return {
        "asset": asset,
        "origin_date": origin_date,
        "n_returns": len(r),
        "horizons": out,
        "model": {
            "repo": REPO,
            "branch": BRANCH,
            "revision": REVISION,
            "context": CONTEXT,
            "mode": "zero_shot",
        },
        "versions": predictor.versions() if hasattr(predictor, "versions") else {},
        "headline_record": {
            "file": f"{asset}_{origin_date}.json",
            "sha256": headline_file["sha256"],
        },
        "code_version": code_version,
        "spec_version": SPEC_VERSION,
    }
