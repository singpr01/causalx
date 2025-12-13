import importlib


def test_assignment_modules_import():
    importlib.import_module("causalx.assignment.random_assign")
    importlib.import_module("causalx.assignment.stratified_assign")
    importlib.import_module("causalx.assignment.covariate_balance")
