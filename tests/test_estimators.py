import importlib


def test_estimators_module_import():
    importlib.import_module("causalx.analysis.estimators")

