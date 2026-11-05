"""E0-2: validated settings. See docs/tasks/E0-2_settings.md.

Marked xfail until you implement src/oilfc/settings.py. When all tests pass, delete the pytestmark line.
"""

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

pytestmark = pytest.mark.xfail(reason="E0-2: implement oilfc.settings", strict=False)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _edited_configs(tmp_path, edit_data=None, edit_protocol=None):
    """Copy the real configs to tmp_path and apply an edit function to the parsed YAML."""
    for name, edit in (("data.yaml", edit_data), ("protocol.yaml", edit_protocol)):
        shutil.copy(CONFIGS / name, tmp_path / name)
        if edit:
            doc = yaml.safe_load((tmp_path / name).read_text())
            edit(doc)
            (tmp_path / name).write_text(yaml.safe_dump(doc))
    return tmp_path


def test_loads_the_real_configs():
    from oilfc.settings import load_settings

    s = load_settings(CONFIGS)
    assert s.protocol.daily.horizons == [1, 5, 10, 22, 66]
    assert s.protocol.monthly.horizons == [1, 3, 6, 12, 24]
    assert s.data.target == "brent_spot"
    assert s.data.eia["brent_spot"].key == "RBRTE"
    assert s.data.fred["cpi"].lag.kind == "days"
    assert s.data.fred["cpi"].lag.n == 45


def test_rejects_an_unknown_lag_kind(tmp_path):
    from oilfc.settings import load_settings

    cfg = _edited_configs(
        tmp_path, edit_data=lambda d: d["fred"]["ovx"]["lag"].update(kind="weeks")
    )
    with pytest.raises(ValidationError):
        load_settings(cfg)


@pytest.mark.parametrize("bad", [[5, 1], [0, 5], [1, 1, 5]])
def test_rejects_horizons_that_are_not_positive_and_strictly_increasing(tmp_path, bad):
    from oilfc.settings import load_settings

    cfg = _edited_configs(tmp_path, edit_protocol=lambda d: d["daily"].update(horizons=bad))
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_target_must_be_a_series_defined_under_eia(tmp_path):
    from oilfc.settings import load_settings

    cfg = _edited_configs(tmp_path, edit_data=lambda d: d.update(target="not_a_series"))
    with pytest.raises(ValidationError):
        load_settings(cfg)


def test_missing_config_directory_raises_file_not_found(tmp_path):
    from oilfc.settings import load_settings

    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "does_not_exist")


def test_api_key_comes_from_the_environment_and_is_not_printed(monkeypatch):
    from oilfc.settings import load_settings

    monkeypatch.setenv("OILFC_EIA_API_KEY", "super-secret-value")
    s = load_settings(CONFIGS)
    assert s.eia_api_key.get_secret_value() == "super-secret-value"
    assert "super-secret-value" not in repr(s)


def test_api_key_defaults_to_none(monkeypatch):
    from oilfc.settings import load_settings

    monkeypatch.delenv("OILFC_EIA_API_KEY", raising=False)
    assert load_settings(CONFIGS).eia_api_key is None
