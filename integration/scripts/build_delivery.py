"""Archive a measured run and regenerate the Word report and audit closure."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED
from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FINDINGS = [
    ("A1", "Exact factory identity", "registry.py", "Entrypoint module/name/code, defaults, closure/global configuration, source and declared artifacts are fingerprinted; registry bindings are read-only.", "test_exact_factory_entrypoint_and_defaults_are_bound"),
    ("A2", "Suspension across execution paths", "recovery.py", "Backup verification consults durable suspension state before decision and execution; existing approvals cannot bypass suspension.", "test_suspended_backup_is_blocked_after_anomaly_and_restart"),
    ("A3", "Bounded decision waiting", "deadlines.py", "Default Linux process watchdog isolates decision callbacks. Timeout returns stop without waiting for the callback or doing synchronous trace I/O. OS scheduling/IPC overhead is reported separately; this is not hard real time.", "test_blocked_decision_callback_cannot_block_host"),
    ("A4", "Retry budget epochs", "scheduler.py", "Epoch IDs are durable; reopening invalidates older queued requests and begin() checks epoch identity.", "test_old_queued_request_cannot_enter_reopened_epoch"),
    ("L1", "Priority dispatch", "scheduler.py", "next_due()/dispatch() order due requests using current decayed penalty, cooldown and budget, with deterministic ties. The full experiment uses dispatch.", "test_dispatch_priority_is_recomputed_after_decay"),
    ("L2", "Measured backup domain", "recovery.py", "Removed the expanded box. Approval and execution are restricted to evaluated seed/start pairs, with numerical tolerance 1e-7. No midflight recovery approval is implied.", "test_backup_has_no_unmeasured_margin_and_checks_seed_on_execution"),
    ("L3", "Feedback-to-recertification", "reviews.py", "Durable repair requests enter the bounded scheduler. A separate comparison protocol includes the anomalous seed; a replacement identity is certified and linked to resolution. Old evidence and old suspension remain unchanged.", "test_feedback_queues_repair_and_certifies_new_version_without_rewriting_old"),
    ("L4", "Validation scope", "contextual.py / experiments.py", "Held-out CDS/PDS-5 and both selection owners are checked. Both subsystems are measured at every atomic boundary, including full offline bimonad diagnostics. Motivations are fixed inside each rollout. Target-rule agreement and sampled checks are not safety proofs.", "test_contextual_checkpoint_has_disjoint_heldout_acceptance"),
    ("L5", "Restored delivery", "integration/scripts/build_delivery.py", "Word report, machine-readable metrics, raw evidence/model archive, dependencies, test evidence, source hashes, provenance and README commands are regenerated.", "archive integrity and source-hash verification"),
    ("L6", "Explicit contracts", "motives.py / coordinator.py", "Geometry includes versioned objective mode/order/units/schema; every eligible action has certificate references. Selections include behavior versions, recovery includes backup digest and NoSelection reference, and outcomes link selection/recovery and geometry.", "test_eligible_contract_contains_all_certificate_references"),
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/audit-fixed-final")
    parser.add_argument("--tests", type=Path, default=ROOT / "artifacts/audit-fixed-tests-final.xml")
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text())
    validation = ROOT / "integration/docs/validation"
    validation.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "model_validation.json"):
        shutil.copy2(run / name, validation / name)
    shutil.copy2(args.tests, validation / "tests.xml")
    suites = list(ET.parse(args.tests).getroot().iter("testsuite"))
    count = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    assert failed == 0, "Do not publish closure with a failed validation suite"
    names = {case.get("name") for case in ET.parse(args.tests).getroot().iter("testcase")}
    for finding in FINDINGS:
        if finding[4].startswith("test_"):
            assert finding[4] in names, finding[4]
    with ZipFile(validation / "full-evidence.zip", "w", ZIP_DEFLATED) as archive:
        for p in sorted(run.rglob("*")):
            if p.is_file() and not p.is_symlink():
                archive.write(p, p.relative_to(run))
    with ZipFile(validation / "full-evidence.zip") as archive:
        assert archive.testzip() is None
    (validation / "dependencies.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(report["dependencies"].items())) + "\n")
    original = Path('/home/mahider_n/Documents/METAMO_SUBREP_TECHNICAL_DOCUMENTATION (1).docx')
    write_json(validation / "source_document.json", {"filename": original.name, "sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
    sources = [ROOT / "README.md", ROOT / ".gitmodules", ROOT / "requirements-lander.txt",
               ROOT / "category/bimonad.py", Path(__file__), *sorted((ROOT / "integration").rglob("*.py")),
               ]
    write_json(validation / "delivery_sources.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    closure = {"date": "2026-11-24", "tests": count, "failures": failed,
               "findings": [{"id": f[0], "status": "addressed", "title": f[1], "source": f[2], "resolution": f[3], "evidence": f[4]} for f in FINDINGS],
               "excluded": ["optional symbolic integration", "eight-objective goal measurements"],
               "acceptance_scope": "Linux single-writer atomic-controller host; process-watchdog wall-clock waiting budget with OS scheduling/IPC overhead, not hard real time"}
    (ROOT / "integration/docs/audit").mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "integration/docs/audit/resolution.json", closure)
    resolution = ["AUDIT RESOLUTION — 24 November 2026", "", "Baseline: findings A1–A4 and L1–L6 from the implementation audit.",
                  "This resolution supersedes its open findings for the current sources.", f"Validation: {count} passed, {failed} failures/errors.", ""]
    for identifier, title, source, description, test in FINDINGS:
        location = source if source.startswith("integration/") else "integration/lander/" + source
        resolution.extend([identifier + " — ADDRESSED: " + title, "Source: " + location,
                           description, "Evidence: " + test, ""])
    resolution.extend(["REMAINING SCIENTIFIC/PLATFORM LIMITS (not missing required features)",
                       "Linux is not a hard-real-time OS. The watchdog bounds waiting for decision callbacks; scheduling/IPC and final response overhead cannot be guaranteed to take zero time.",
                       "Model targets are engineered motive/support rules; agreement is not general environmental safety evidence.",
                       "Backup authority is deliberately limited to exact evaluated reset seeds/observations. Other states stop.",
                       "Goal/modulator dynamics are sampled at each atomic boundary because they remain fixed during the controller rollout.",
                       "Feedback-repair integration has deterministic synthetic regression coverage; real runtime evaluation does not fabricate an anomaly merely to report a repair.",
                       "Symbolic integration and eight measured goal objectives are excluded by the user.",
                       "See integration/docs/IMPLEMENTATION_REPORT.docx and integration/docs/validation/full-evidence.zip for the current delivery."])
    (ROOT / "integration/docs/audit/RESOLUTION_REPORT.txt").write_text("\n".join(resolution) + "\n")
    build_word(report, count)
    print(f"Delivery generated: {count} tests; {len(FINDINGS)} findings addressed")


def build_word(report, count):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Inches(.65)
    section.left_margin = section.right_margin = Inches(.7)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    section.header.paragraphs[0].text = "MetaMo–SubRep | Validated implementation and audit resolution"
    section.footer.paragraphs[0].text = "24 November 2026 • MO-LunarLander • two environment objectives"
    def p(text): doc.add_paragraph(text)
    def h(text): doc.add_heading(text, 1)
    def page(): doc.add_page_break()
    def table(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers)); t.style = "Light Shading Accent 1"
        for cell, v in zip(t.rows[0].cells, headers): cell.text = str(v)
        for row in rows:
            for cell, v in zip(t.add_row().cells, row): cell.text = str(v)
    def code(text):
        run = doc.add_paragraph().add_run(text); run.font.name = "Consolas"; run.font.size = Pt(8)

    doc.add_heading("MetaMo–SubRep integration", 0)
    h("Implementation, validation and audit resolution")
    p("This document records the implementation against METAMO_SUBREP_TECHNICAL_DOCUMENTATION (1).docx, "
      "dated 17 October 2026, and the corrections to the 24 November audit. It describes the current code "
      "and a fresh complete experiment, rather than repeating historical metrics as new measurements.")
    table(["Design stage", "Delivered behavior"], [
        ["0 — Reproduce SubRep", "Separate native pipeline reproduction, model-source labels, actual parameters, synchronized store/library IDs."],
        ["1 — MetaMo boundary", "Objective/reward contracts, motive mapping, controller registry, native gates, persistence, screened selection and explicit recovery."],
        ["2 — Contextual geometry", "Trained 18-input SASP MDN, held-out CDS/PDS and selector checks, support feasibility/coverage, explicit activation and fallback."],
        ["3 — Providers and reevaluation", "Verified executable factories, rejection history, decaying priority dispatcher, durable retry epochs, cooldowns and provider repair requests."],
        ["4 — Reuse, recovery and feedback", "Both selection owners, motive-shift reuse, constrained backup approval, cancellation, replacement certification and measured diagnostic reports."],
    ])
    p(f"Validation: {count} targeted tests passed. The full experiment includes global and contextual "
      "admission, both selectors, measured controller outcomes, backup-or-stop probes and scheduler "
      "restart/decay checks. Symbolic integration and eight measured goal objectives are excluded.")
    p("Scope: synchronous atomic controllers on a Linux host, one writer per evidence repository and one "
      "controller per simulator. The deliverable is functional finite-sample validation, not a proof "
      "of universal safety, optimality or hard-real-time operation.")

    page(); h("1. Architecture, reward semantics and motive conversion")
    table(["Module", "Responsibility"], [
        ["contracts.py / motives.py", "Strict versioned objectives, finite vectors, 18-value context, normalized weights and epsilon policy."],
        ["registry.py / evidence.py", "Executable identities, original Action descriptors, actual matched resets and native executor outcomes."],
        ["persistence.py / storage.py / contextual.py", "Shared durable JSON writes, native certificates, MeTTa/library generations, trained geometry and held-out activation."],
        ["coordinator.py / deadlines.py", "Shared screening, one final selector, watchdog budget, normal/backup outcome attribution."],
        ["scheduler.py / reviews.py / recovery.py", "Decayed priority queues, retry epochs, repair-to-replacement workflow and measured backup authority."],
        ["experiments.py / complete.py", "Reproducible data collection, full evaluation and metrics with explicit denominators."],
    ])
    p("The environment is mo-lunar-lander-v3. Canonical measured objectives are [safety, fuel], "
      "both in environment reward units and both larger-is-better. Safety is terminal reward plus "
      "dense shaping; fuel is the unchanged sum of negative engine costs. Landing and crash diagnostics "
      "remain separate from shaping. Scalar payoff is the discounted sum of the two objectives.")
    code("Safety priority = Individuation + Ethics + Securing + stimulus.risk\n"
         "Fuel priority = Self + Resolution + stimulus.effort + (1 - Transcendence)\n"
         "weights = [Safety, Fuel] / (Safety + Fuel); zero-total default [0.5, 0.5]\n"
         "epsilon = explicitly configured nonnegative deficit budget")
    p("The mapping is an explicit engineering choice, not a formula mandated by the source document. "
      "Context records the eight goals, six modulators and four stimulus values in the specified order "
      "after subsystem appraisal and conservative consensus merging. Normalization occurs only inside "
      "the adapter; downstream malformed weights are rejected. Geometry now carries objective "
      "mode/order/units/schema and its own contract version.")
    p("Baseline and candidates use matching reset seeds, horizon, gamma and aggregation. Every rollout "
      "resets the actual evaluation simulator and creates a fresh seeded policy. Raw rewards, actual "
      "initial observations, returns, seeds and stop reasons remain auditable. initial_obs skips a "
      "second reset; it never reconstructs hidden simulator state.")

    page(); h("2. Certification, persistence and exact executable identity")
    code("delta_r = candidate_mean_payoff - baseline_mean_payoff\n"
         "delta_n = candidate_mean_objectives - baseline_mean_objectives\n"
         "robust = delta_r + min_w_in_region(w.dot(delta_n))\n"
         "CDS: robust >= 0; PDS: robust >= -epsilon\n"
         "FULL_SIMPLEX: min_w(w.dot(delta_n)) = min(delta_n)")
    p("Native SubRep improvement and gate classes determine admission. A failed controller produces "
      "persistent rejection evidence without an operational certificate. Successful admission inserts "
      "a native frozen certificate into real Hyperon MeTTa, then explicitly supplies matching region "
      "audit fields to SkillLibrary. False returns and exceptions roll back only the new insertion.")
    p("Certificates, library entries, manifests and evidence are committed as a generation selected by "
      "an atomic CURRENT pointer. Rejection, scheduler and feedback state are persisted separately. "
      "Reload checks hashes, complete identity sets, reward meanings, evidence mathematics, exact "
      "certificate contents and executable/model resolution before reattaching policies.")
    p("Provider digests now bind exact factory module/qualified name and code, defaults, closure/global "
      "configuration, source and declared artifacts. Distinct functions in one source file no longer "
      "share a behavior digest. Registry bindings are read-only. Changed behavior requires compatible "
      "new identity and measured evidence; version labels alone cannot reset retry history.")
    p("Nine controlled policy types are provided: deterministic/stochastic PPO, perturbed PPO, a PPO/side "
      "tradeoff, fixed left/main/right engines, idle and seeded random. PPO uses the pinned SubRep "
      "checkpoint. Original MAGUS action objects are preserved; illustrative correlations are not "
      "treated as measured outcomes. Built-in predicted goal deltas remain zero.")
    p("Eligibility decisions now include a certificate-reference map for every eligible action. "
      "Selections name behavior versions, recovery records carry backup digests and NoSelection "
      "references, and outcomes identify their selection or recovery, geometry and baseline contract.")

    page(); h("3. Contextual MDN training and held-out validation")
    m = report["model_validation"]; v = m["validation"]
    p("The native SASP MDN has 18 inputs, two objectives and two hidden layers of width 32. Training "
      "uses seeds 100–107; held-out contexts use seeds 108–111. With 16 contexts per seed there are "
      "128 training and 64 held-out contexts. Context includes motivation and observed stimulus; "
      "the MDN is not a planner or candidate-conditioned outcome predictor.")
    p("Targets distill the declared motive rule and an explicit support rule: slack = 0.10 + "
      "0.10*dispersion/(1+dispersion), using measured candidate payoff dispersion; each support cap "
      "is min(1, mapped_weight + slack). Training uses 600 Adam steps and seed 17. The SASP head "
      "constructs feasible simplex-cap regions. Coverage of the actual current weight is checked independently.")
    table(["Held-out measure", "Result"], [[k, str(v[k])] for k in
          ("weight_mse", "support_mse", "weight_coverage", "invalid_support_events", "gate_agreement", "selection_agreement")])
    table(["Separate acceptance cases", "Agreement"], list(v["per_gate_agreement"].items()) + list(v["per_selector_agreement"].items()))
    p("Activation checks context, objective and measurement semantics, source architecture digest, "
      "checkpoint identity, seed separation and held-out thresholds. The updated model version "
      "requires both CDS and PDS at the activated budget of 5, plus MAGUS and SubRep selector "
      "agreement. MetaMo is the default weight owner; explicit MDN ownership uses alpha/sum(alpha). "
      "Uncovered weights fail closed. Explicit global fallback excludes contextual certificates.")
    p("These are held-out agreement metrics against declared engineered targets, not independent "
      "evidence that the model predicts all environmental outcomes. Larger unseen distributions "
      "and other operating budgets require additional validation. No learned safety guarantee is claimed.")

    page(); h("4. Selection ownership and bounded decision waiting")
    p("Both subsystem recommendations and final MAGUS consensus see exactly the same eligible actions. "
      "MAGUS consensus uses mean subsystem score minus 0.25 times disagreement, retaining original "
      "ordering for ties. In subrep_selects mode only SubRep ranks by delta_r + w.dot(delta_n), "
      "with lexical skill-ID ties; MAGUS cannot replace the choice.")
    p("Before execution the host verifies snapshot freshness, current budget, certificate/model identity "
      "and eligibility again. Selected actions execute once. A motive-shift probe uses exact weights "
      "[0.8,0.2] and [0.2,0.8] with unchanged observation, policies and evidence. Global eligibility "
      "can survive; contextual coverage failure produces a structured stop without retraining.")
    p("The default Linux process watchdog isolates appraisal, registry, evidence verification and model "
      "calls from the waiting host. At timeout the host stops waiting, kills the worker and returns "
      "DECISION_BUDGET_EXHAUSTED without invoking a backup or another slow callback. Decision traces "
      "are buffered; execute flushes them, and stop-only callers use flush_traces outside the deadline.")
    p("Deadline acceptance is now explicitly a wall-clock waiting budget for isolated decision work, "
      "plus OS scheduling/IPC and response overhead. This is an enforcement improvement over the old "
      "cooperative check, but ordinary Linux cannot promise a zero-overhead hard-real-time response. "
      "The regression blocks registry work for 800 ms under a 50 ms budget and verifies that the host "
      "returns a timeout stop within a 150 ms scheduling tolerance, without executing an action.")
    p("deadline_mode='cooperative' remains an explicit diagnostic option for tests inspecting in-process "
      "callbacks. It is not the default bounded-wait mode. One host/writer and at most 128 candidates "
      "per decision are supported. The operational environment must not be stepped or reset between "
      "selection and execution. Policy rollout cancellation is checked before each simulator step.")

    page(); h("5. Rejection dispatch and feedback-to-replacement workflow")
    code("deficit = max(0, -(robust_score + epsilon))\n"
         "P_after = min(P_max, P_decayed + a + b*deficit)\n"
         "P(t) = P_after * 2**(-(t-last_failure)/H)\n"
         "dispatch_priority = base_priority - current_P(t)")
    p("Defaults remain H=3600 seconds, cooldown=60 seconds, a=1, b=0.1, penalty cap=100 and three "
      "retries per useful-change epoch. next_due/dispatch recompute current decayed priority and "
      "order eligible queued requests, respecting cooldown, due time, consumed triggers and budget. "
      "Retries and queues survive restart; interrupted attempts remain counted.")
    p("Epoch IDs are durable. A relevant change may reopen an exhausted budget, but invalidates older "
      "queued requests. begin rejects an old epoch even when its attempt_count happens to match. "
      "Actual artifact revision, changed contextual region or verified nonduplicate evidence is "
      "required; new UUIDs and version labels alone cannot bypass history. Penalty decay never "
      "changes admission evidence or thresholds.")
    p("Unexpected crashes create durable FeedbackReviews events with observed evidence and optional "
      "suspension. Providers inspect repair_requests and enqueue a new immutable replacement identity. "
      "A separate review repository must use a declared baseline protocol including the anomalous "
      "reset seed. The bounded scheduler evaluates the replacement and certifies it through the same "
      "native gates. Resolution links the replacement certificate and review contract. A failed "
      "replacement returns to repair-required status.")
    p("Old certificates are never rewritten, and a passing replacement does not silently unsuspend "
      "the old version. That old suspension also blocks backup execution. The integrated workflow "
      "has deterministic synthetic regression coverage; the full runtime experiment does not invent "
      "a crash in order to report a real repair event. External providers still own actual behavior revision.")

    page(); h("6. Backup authority, feedback and runtime results")
    p("The old expanded min/max observation box has been removed. A backup approval requires at least "
      "five measured episodes and a global CDS certificate, with default success ≥0.2 and crash ≤0.2. "
      "It binds the exact evaluated seed/start pairs, controller artifact, objective contract, certificate "
      "and evidence. Selection checks the observation, and execution checks its matching seed. The "
      "comparison idle baseline is not implicitly promoted to backup.")
    p("A suspended controller, unmeasured start, mismatched seed, changed evidence or expired budget "
      "produces stop. There is deliberately no midflight or extrapolated-envelope approval. Backup "
      "execution is recorded separately with its approval/recovery reference, without inventing "
      "an ordinary selection certificate reference. Rejected predictions are never applied.")
    p("Normal successful execution applies only the selected predicted delta once through existing "
      "damping/projection/blending. Observations feed the next appraisal: novelty=0, conduciveness=1 "
      "only for a landing, risk=1 for crash / 0 for landing / 0.5 otherwise, and effort is clipped "
      "discounted fuel cost per step. Backup observations also affect the next appraisal only.")
    table(["Runtime mode", "Success", "Negative transfer", "Mean payoff lift vs PPO"], [
        [e["geometry"] + " / " + e["selector"],
         f"{e['success_rate']['numerator']}/{e['success_rate']['denominator']}",
         f"{e['negative_transfer_rate']['numerator']}/{e['negative_transfer_rate']['denominator']}",
         e["selection_lift"]["mean_payoff_difference"]] for e in report["evaluations"]])
    p("Runtime configurations reuse the same three seeds; they are not independent performance samples. "
      "Negative transfer and lift use actual same-seed comparison rollouts in a separate evaluation "
      "environment. Both motivational subsystems are now measured at every atomic boundary, with "
      "separate goal/modulator variation and offline full bimonad diagnostics. Motivations remain "
      "fixed inside a controller rollout, so no within-rollout motivational dynamics are claimed. "
      "Sampled checks are not a mathematical contractivity proof.")

    page(); h("7. Audit resolution, reproduction and artifacts")
    table(["Audit finding", "Current status"], [[f[0] + " — " + f[1], "Addressed; see integration/docs/audit/RESOLUTION_REPORT.txt and regression evidence"] for f in FINDINGS])
    code("git submodule update --init --recursive\n"
         "python3.12 -m venv .venv\nsource .venv/bin/activate\n"
         "python -m pip install swig\nexport PATH=\"$VIRTUAL_ENV/bin:$PATH\"\n"
         "python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lander.txt\n"
         "python -m integration.lander.complete --output artifacts/my-validated-run\n"
         "python -m pytest integration/tests -q --junitxml=artifacts/audit-fixed-tests-final.xml\n"
         "python integration/scripts/build_delivery.py --run artifacts/my-validated-run --tests artifacts/audit-fixed-tests-final.xml")
    p("Use Python 3.12 on Linux with a C++ compiler. The run command needs a fresh output directory; "
      "--horizon, --epochs and --skip-reproduction configure the experiment. Old model/provider identities "
      "fail closed; they are not silently migrated into new certification. Exact dependencies and "
      "source hashes are archived under integration/docs/validation.")
    p("For the complete targeted suite, measured experiment and regenerated delivery in one command, "
      "run python integration/scripts/validate_lander.py --output artifacts/validated-delivery. This produces the "
      "JUnit file consumed by the document builder, rather than relying on a previous test archive.")
    p("The fresh run includes a separate native demo reproduction. Historical document values are not "
      "substituted for current measurements. The native demo's baseline runs to environment termination; "
      "the main integration uses matched baseline/candidate horizons and seeds. The trained integration "
      "MDN and the native demo's stub fallback are labeled separately.")
    p("integration/docs/validation/full-evidence.zip retains the model, dataset, raw training/controller evidence, "
      "both certificate libraries, scheduler state, execution traces, recovery approval and native reports. "
      "The local reproduction PPO symlink is excluded; the pinned submodule supplies that artifact. "
      "Runtime model paths are bound to their digests; regenerate runtime artifacts in a new checkout "
      "rather than silently attaching a different model. The old audit remains as a historical baseline.")
    p("Excluded as requested: optional symbolic integration and eight-objective goal measurements. "
      "The remaining caveats describe measurement and platform limits, not unimplemented claims of "
      "general safety or hard-real-time operation.")
    doc.save(ROOT / "integration/docs/IMPLEMENTATION_REPORT.docx")


if __name__ == "__main__":
    main()
