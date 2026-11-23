# E0-2. Validated settings from the YAML configs

**Module you create:** `src/oilfc/settings.py`  **Tests:** `tests/test_settings.py`

## Why this matters

Today `configs/data.yaml` and `configs/protocol.yaml` are read as untyped dictionaries. A typo (`"weeks"` instead of `"days"`, horizons out of order, a target that does not exist) would only blow up in the middle of an experiment, or worse, silently change results. You will turn the configs into typed models that fail immediately with a clear message. The same models later describe API request and response bodies.

## Read first

- Pydantic docs: *Concepts* (models, validators, `Literal`, `Field`). Read it fully.
- `pydantic-settings` docs: *Settings management* (`BaseSettings`, `env_prefix`, `SecretStr`).

## Interface

```python
from oilfc.settings import load_settings, Settings

s = load_settings(config_dir)  # config_dir: pathlib.Path containing data.yaml and protocol.yaml
s.protocol.daily.horizons  # [1, 5, 10, 22, 66]
s.data.target  # "brent_spot"
s.data.fred["cpi"].lag.kind  # "days"
s.eia_api_key  # SecretStr | None, read from env var OILFC_EIA_API_KEY
```

Rules the tests check:
- `lag.kind` must be exactly `"bdays"` or `"days"`.
- `horizons` must be positive and strictly increasing.
- `data.target` must be one of the series defined under `eia`.
- A missing config file raises `FileNotFoundError`; invalid content raises `pydantic.ValidationError`.
- The API key never appears in `repr(settings)`.

## Decision you must make (write it as ADR 0003 if you like)

`configs/protocol.yaml` has keys the tests do not look at (`design`, `metrics`, `tests`, ...). Do you forbid unknown keys (`extra="forbid"`), ignore them, or model all of them? What does each choice do when someone mistypes a key?

## Hints (not answers)

- A validator that needs to compare two fields of the same model is a *model* validator, not a field validator.
- `BaseSettings` reads environment variables itself; you do not call `os.environ`.
- YAML gives you plain dicts. The point of the exercise is the step *between* the dict and the model.

## Done when

- All tests pass and you removed the `xfail` marker.
- Self-check: (1) what is the difference between validation and parsing? (2) why is failing at startup better than failing in hour three of a backtest? (3) why `SecretStr` for a key?

## Stretch

Make `build.py` use `load_settings` instead of `yaml.safe_load`, so the pipeline fails fast on a bad config.
