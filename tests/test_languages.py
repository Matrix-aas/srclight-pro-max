import importlib
from pathlib import Path

from srclight import languages


def test_get_language_falls_back_to_language_pack_for_dart_and_swift(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name in {"tree_sitter_dart", "tree_sitter_swift"}:
            raise ImportError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    languages._LANGUAGES.clear()

    assert languages.get_language("dart") is not None
    languages._LANGUAGES.pop("swift", None)
    assert languages.get_language("swift") is not None


def test_detect_language_maps_vue_special_case(tmp_path):
    path = tmp_path / "Component.vue"
    path.write_text("<template><div /></template>\n")

    assert languages.detect_language(Path(path)) == "vue"
