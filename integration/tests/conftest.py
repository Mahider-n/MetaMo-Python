import pytest


def pytest_addoption(parser):
    parser.addoption("--run-pipeline", action="store_true",
                     help="also run the slow tests that use the real simulator and the real pilot policy")


def pytest_configure(config):
    config.addinivalue_line("markers", "pipeline: slow end-to-end test on the real simulator (needs --run-pipeline)")
    # Always print skip reasons in the summary (same as passing -rs), so a skipped
    # module is never silent.
    chars = config.option.reportchars or ""
    if "s" not in chars and "A" not in chars:
        config.option.reportchars = chars + "s"


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-pipeline"):
        return
    skip = pytest.mark.skip(reason="slow end-to-end test; pass --run-pipeline to run it")
    for item in items:
        if "pipeline" in item.keywords:
            item.add_marker(skip)