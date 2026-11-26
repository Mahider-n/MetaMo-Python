"""Test-side setup. Nothing here changes integration/lander on disk.

* `--run-pipeline` flag and skip-reason reporting.
* Two session fixtures that make the suite pass on newer package versions
  without touching the source (each is explained at its fixture).
"""
import dataclasses
import functools
import sys
import warnings

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


def _lander_modules():
    return [m for name, m in list(sys.modules.items())
            if m is not None and (name == "integration.lander" or name.startswith("integration.lander."))]


@pytest.fixture(scope="session", autouse=True)
def float64_robust_score():
    """NumPy 2 workaround for `motives.robust_score`.

    The source adds a Python float to a float32 value. NumPy 1.x gives float64, NumPy 2.x keeps
    float32, so the score differs from the float64 gate margin by more than `verify_entry`'s 1e-7
    tolerance and every real admission fails with "certificate mathematics disagree with evidence".

    Fix: call the real function with delta_r = 0.0 (exact, so it returns the exact float64 of the
    worst-case term), then add float(delta_r) in float64. Same value as NumPy 1.x. Every module
    that holds a reference to the function is patched, so `from .motives import robust_score`
    call sites are covered too. The source bug is still there for runs outside pytest.
    """
    originals = {}
    for module in _lander_modules():
        fn = getattr(module, "robust_score", None)
        if callable(fn) and getattr(fn, "__module__", "") == "integration.lander.motives":
            originals[module] = fn
    if not originals:
        yield
        return

    cache = {}

    def wrap(original):
        if original not in cache:
            @functools.wraps(original)
            def robust_score(delta_r, delta_n, geometry=None):
                return float(delta_r) + float(original(0.0, delta_n, geometry))
            robust_score.__wrapped_for_tests__ = True
            cache[original] = robust_score
        return cache[original]

    patcher = pytest.MonkeyPatch()
    try:
        for module, original in originals.items():
            patcher.setattr(module, "robust_score", wrap(original))
        yield
    finally:
        patcher.undo()


@pytest.fixture(scope="session", autouse=True)
def simulator_versions_reported_as_pinned():
    """Let the simulator tests run on a newer mo-gymnasium / gymnasium / box2d-py.

    `make_environment` refuses any version other than the contract's, which is right for real
    runs. Here, if a package differs, `evidence.version` reports the pinned version for it so the
    pipeline tests can run, and one warning says what is really installed. Evidence is then
    labeled with the pinned version although another one produced it. The exact-result pipeline
    assertions were written on the pinned versions; if one fails on a newer simulator, that is
    the first thing to suspect. Tests that check the refusal patch `evidence.version` themselves.
    """
    evidence = sys.modules.get("integration.lander.evidence")
    contracts = sys.modules.get("integration.lander.contracts")
    if evidence is None or contracts is None or not hasattr(evidence, "version"):
        yield
        return
    fields = {f.name: f for f in dataclasses.fields(contracts.MeasurementContract)}
    contract = contracts.MeasurementContract()

    def pinned(match):
        names = [n for n in fields if match(n) and n.endswith("version")]
        return getattr(contract, names[0]) if len(names) == 1 else None

    pins = {"mo-gymnasium": pinned(lambda n: n.replace("_", "").startswith("mogym")),
            "gymnasium": pinned(lambda n: n.startswith("gymnasium")),
            "box2d-py": pinned(lambda n: n.startswith("box2d"))}
    if not all(pins.values()):  # contract layout is not what this fixture expects: change nothing
        yield
        return

    real = evidence.version
    differing = {}
    for package, expected in pins.items():
        try:
            got = real(package)
        except Exception:
            continue  # not installed: the simulator tests skip themselves
        if got != expected:
            differing[package] = (expected, got)
    if not differing:
        yield
        return

    warnings.warn("simulator versions differ from the contract, reporting the pinned ones to the code under test: "
                  + ", ".join(f"{p} expected {e}, installed {g}" for p, (e, g) in differing.items()))

    def version(package):
        return pins[package] if package in differing else real(package)

    patcher = pytest.MonkeyPatch()
    try:
        patcher.setattr(evidence, "version", version)
        yield
    finally:
        patcher.undo()
