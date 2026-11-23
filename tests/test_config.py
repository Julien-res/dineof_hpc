import pytest

from dineof_hpc.config import load_dine_opts


def test_default_profile_nev_automatic():
    opts = load_dine_opts(profile="conservative")
    assert opts["nev"] is None
    assert opts["alpha"] == 0.01 and opts["cv_patience"] == 3


def test_fast_profile_and_override():
    assert load_dine_opts(profile="fast")["nev"] == 15
    assert load_dine_opts(profile="fast", nev=30)["nev"] == 30


def test_custom_file_and_unknown_profile(tmp_path, monkeypatch):
    f = tmp_path / "p.toml"
    f.write_text("[a]\nnev = 12\n")
    monkeypatch.setenv("DINEOF_CONFIG", str(f))
    assert load_dine_opts(profile="a")["nev"] == 12
    with pytest.raises(ValueError):
        load_dine_opts(profile="missing")
