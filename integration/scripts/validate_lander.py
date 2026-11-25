"""Run the full targeted suite, measured experiment, and delivery generation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/validated-delivery"))
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "report.json").exists():
        parser.error("choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    test_xml = output / "tests.xml"
    tests = ["integration/tests", *["external/subrep/tests/" + name + ".py" for name in (
        "test_certification_gates", "test_certificate_storage", "test_env", "test_executor",
        "test_skill_library", "test_support_geometry", "test_mdn_support_pipeline",
        "test_certificate_mdn_audit_storage", "test_zero_shot_reuse")]]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "external/subrep")}
    subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", f"--junitxml={test_xml}"], cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, "-m", "integration.lander.complete", "--output", str(output)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "integration/scripts/build_delivery.py"), "--run", str(output), "--tests", str(test_xml)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
