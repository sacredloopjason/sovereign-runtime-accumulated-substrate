import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE16_RUNTIME = PROJECT_DIR / "slice16_runtime.py"
SLICE16_EVIDENCE = PROJECT_DIR / "evidence" / "slice16" / "standing_wave_boundary.json"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice17"
SEED_LITERAL = "SLICE6_SEED"
PARENT_CYCLE_BOUND = 8
EXTENDED_CYCLE_BOUND = 15
MAX_CANDIDATE_PERIOD = 7
REPLICATION_COUNT = 2
OUTCOME_CONFIRMED = "PERIODIC_ORBIT_CONFIRMED"
OUTCOME_NOT_CONFIRMED = "PERIODIC_ORBIT_NOT_CONFIRMED"
OUTCOME_FAIL = "FAIL"


def load_slice16_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice16_accumulated_runtime_for_slice17", SLICE16_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 16 runtime: {SLICE16_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def pair_ids(cycle):
    return [
        [int(value) for value in cycle["T_A_produced"]["token_ids"]],
        [int(value) for value in cycle["T_B_produced"]["token_ids"]],
    ]


def pair_comparison(cycles, left, right):
    left_ids = pair_ids(cycles[left])
    right_ids = pair_ids(cycles[right])
    return {
        "left_cycle": left,
        "right_cycle": right,
        "left_R_sha256": cycles[left]["R_k_sha256"],
        "right_R_sha256": cycles[right]["R_k_sha256"],
        "exact_ordered_token_ID_identity": left_ids == right_ids,
    }


def candidate_period_report(cycles, period):
    comparisons = [
        pair_comparison(cycles, left, left + period)
        for left in range(len(cycles) - period)
    ]
    return {
        "period": period,
        "supported": (
            len(cycles) >= (2 * period + 1)
            and all(item["exact_ordered_token_ID_identity"] for item in comparisons)
        ),
        "comparisons": comparisons,
    }


def kv_history_forward_only(cycles):
    if len(cycles) != EXTENDED_CYCLE_BOUND:
        return False
    for index, cycle in enumerate(cycles):
        if not (
            cycle["A_pre"]["sequence_length"]
            < cycle["A_post"]["sequence_length"]
            < cycle["A_after_T_B"]["sequence_length"]
            and cycle["B_pre"]["sequence_length"]
            < cycle["B_post"]["sequence_length"]
        ):
            return False
        if index > 0:
            previous = cycles[index - 1]
            if not (
                cycle["A_pre"] == previous["A_post"]
                and cycle["A_post"] == previous["A_after_T_B"]
                and cycle["B_pre"] == previous["B_post"]
            ):
                return False
    return True


def enrich_replicate(replicate, inherited_replicate):
    cycles = replicate["cycles"]
    reports = [
        candidate_period_report(cycles, period)
        for period in range(1, MAX_CANDIDATE_PERIOD + 1)
    ]
    minimal_period = next(
        (report["period"] for report in reports if report["supported"]), None
    )
    initial_repeat = pair_comparison(cycles, 0, 7)
    closure = pair_comparison(cycles, 0, 14)
    prefix_reproduced = cycles[:8] == inherited_replicate["cycles"]
    forward_only = kv_history_forward_only(cycles)
    replicate["slice16_prefix_reproduced"] = prefix_reproduced
    replicate["initial_repeat"] = initial_repeat
    replicate["candidate_period_reports"] = reports
    replicate["minimal_period"] = minimal_period
    replicate["closure"] = closure
    replicate["periodic_orbit_result"] = (
        OUTCOME_CONFIRMED if minimal_period is not None else OUTCOME_NOT_CONFIRMED
    )
    replicate["checks"].update(
        {
            "SLICE16_PREFIX_REPRODUCED": prefix_reproduced,
            "EXTENDED_TRAJECTORY_R0_THROUGH_R14": len(cycles)
            == EXTENDED_CYCLE_BOUND,
            "INITIAL_REPEAT_R0_EQUALS_R7": initial_repeat[
                "exact_ordered_token_ID_identity"
            ],
            "KV_HISTORY_FORWARD_ONLY": forward_only,
        }
    )
    return replicate


def deterministic_projection(replicate):
    return {key: value for key, value in replicate.items() if key != "replicate"}


def comparison_line(comparison):
    status = "PASS" if comparison["exact_ordered_token_ID_identity"] else "FAIL"
    return (
        f"R_{comparison['left_cycle']} = R_{comparison['right_cycle']} "
        f"{status} left_sha256={comparison['left_R_sha256']} "
        f"right_sha256={comparison['right_R_sha256']}"
    )


def make_receipt(result):
    if result["outcome"] == OUTCOME_FAIL:
        failed = result["failed_requirements"][0]
        return "\n".join(
            [
                "OUTCOME",
                OUTCOME_FAIL,
                "",
                "EVIDENCE",
                "",
                "failed_requirement:",
                failed,
                "",
                "causal_boundary:",
                result["failed_boundaries"][failed],
                "",
                "substrate_evidence:",
                json.dumps(result["checks"], sort_keys=True),
                "",
            ]
        )

    evidence = result["replicates"][0]
    initial = evidence["initial_repeat"]
    if result["outcome"] == OUTCOME_NOT_CONFIRMED:
        period_seven = evidence["candidate_period_reports"][6]
        first_failed = next(
            item
            for item in period_seven["comparisons"]
            if not item["exact_ordered_token_ID_identity"]
        )
        return "\n".join(
            [
                "OUTCOME",
                OUTCOME_NOT_CONFIRMED,
                "",
                "EVIDENCE",
                "",
                "initial_repeat:",
                comparison_line(initial),
                "",
                "first_failed_periodic_comparison:",
                comparison_line(first_failed),
                "",
                "DETERMINISTIC_REPLICATION:",
                "PASS",
                "",
            ]
        )

    minimal_period = evidence["minimal_period"]
    period_report = evidence["candidate_period_reports"][minimal_period - 1]
    lines = [
        "OUTCOME",
        OUTCOME_CONFIRMED,
        "",
        "EVIDENCE",
        "",
        "minimal_period:",
        str(minimal_period),
        "",
        "initial_repeat:",
        comparison_line(initial),
        "",
        "period_comparisons:",
    ]
    lines.extend(comparison_line(item) for item in period_report["comparisons"])
    lines.extend(
        [
            "",
            "closure:",
            comparison_line(evidence["closure"]),
            "",
            "KV_HISTORY_FORWARD_ONLY:",
            "PASS",
            "",
            "A_KV_CAUSAL_CONTINUITY:",
            "PASS",
            "",
            "B_KV_CAUSAL_CONTINUITY:",
            "PASS",
            "",
            "EXACT_TOKEN_TRANSPOSITION:",
            "PASS",
            "",
            "PARENT_UNCHANGED:",
            "PASS",
            "",
            "PARENT_CONTINUATION_CONTROL:",
            "PASS",
            "",
            "DETERMINISTIC_REPLICATION:",
            "PASS",
            "",
        ]
    )
    return "\n".join(lines)


def execute_acceptance():
    slice16 = load_slice16_runtime()
    inherited = json.loads(SLICE16_EVIDENCE.read_text(encoding="utf-8"))
    if inherited["outcome"] != slice16.OUTCOME_NO_RECURRENCE:
        raise RuntimeError("The inherited Slice 16 evidence did not pass")
    if inherited["local_cycle_bound"] != PARENT_CYCLE_BOUND:
        raise RuntimeError("The inherited Slice 16 cycle bound changed")

    original_lower_wave = slice16.run_boundary_preserving_wave

    def extended_lower_wave(slice11, context, child, replicate):
        previous_bound = slice16.LOCAL_CYCLE_BOUND
        slice16.LOCAL_CYCLE_BOUND = EXTENDED_CYCLE_BOUND
        try:
            return original_lower_wave(slice11, context, child, replicate)
        finally:
            slice16.LOCAL_CYCLE_BOUND = previous_bound

    slice16.run_boundary_preserving_wave = extended_lower_wave
    slice16.LOCAL_CYCLE_BOUND = PARENT_CYCLE_BOUND
    slice15 = slice16.load_slice15_runtime()
    slice14 = slice15.load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()
    replicates = [
        enrich_replicate(
            slice16.execute_replicate(slice15, slice14, slice11, context, number),
            inherited["replicates"][number - 1],
        )
        for number in range(1, REPLICATION_COUNT + 1)
    ]

    checks = {
        f"REPLICATE_{rep['replicate']}_{name}": passed
        for rep in replicates
        for name, passed in rep["checks"].items()
    }
    checks["DETERMINISTIC_REPLICATION"] = (
        deterministic_projection(replicates[0])
        == deterministic_projection(replicates[1])
    )
    checks["CANDIDATE_PERIOD_RESULT_IDENTITY"] = (
        replicates[0]["candidate_period_reports"]
        == replicates[1]["candidate_period_reports"]
    )
    checks["MINIMAL_PERIOD_RESULT_IDENTITY"] = (
        replicates[0]["minimal_period"] == replicates[1]["minimal_period"]
    )
    checks["PARENT_PRESERVATION_RESULT_IDENTITY"] = (
        replicates[0]["parent_preservation"]
        == replicates[1]["parent_preservation"]
    )
    checks["PERIODIC_OUTCOME_IDENTITY"] = (
        replicates[0]["periodic_orbit_result"]
        == replicates[1]["periodic_orbit_result"]
    )
    failed = [name for name, passed in checks.items() if not passed]

    boundary_by_check = {
        "SLICE16_PREFIX_REPRODUCED": "inherited R_0..R_7 -> clean Slice 17 R_0..R_7",
        "EXTENDED_TRAJECTORY_R0_THROUGH_R14": "live descended KV trajectory -> R_14",
        "INITIAL_REPEAT_R0_EQUALS_R7": "exact ordered token IDs R_0 -> R_7",
        "KV_HISTORY_FORWARD_ONLY": "immediately preceding pole KV post-state -> next pole pre-state",
        "DETERMINISTIC_REPLICATION": "replicate 1 complete projection -> replicate 2 complete projection",
        "CANDIDATE_PERIOD_RESULT_IDENTITY": "replicate 1 candidate periods -> replicate 2 candidate periods",
        "MINIMAL_PERIOD_RESULT_IDENTITY": "replicate 1 minimal period -> replicate 2 minimal period",
        "PARENT_PRESERVATION_RESULT_IDENTITY": "replicate 1 parent control -> replicate 2 parent control",
        "PERIODIC_OUTCOME_IDENTITY": "replicate 1 empirical outcome -> replicate 2 empirical outcome",
    }
    failed_boundaries = {}
    for failed_name in failed:
        base_name = failed_name
        if failed_name.startswith("REPLICATE_"):
            base_name = failed_name.split("_", 2)[2]
        failed_boundaries[failed_name] = boundary_by_check.get(
            base_name, "inherited Slice 16 acceptance boundary"
        )

    empirical_outcome = replicates[0]["periodic_orbit_result"]
    result = {
        "schema": "SLICE17_PERIODIC_ORBIT_TEST_V1",
        "seed": SEED_LITERAL,
        "parent_cycle_bound": PARENT_CYCLE_BOUND,
        "extended_cycle_bound": EXTENDED_CYCLE_BOUND,
        "candidate_period_maximum": MAX_CANDIDATE_PERIOD,
        "replication_count": REPLICATION_COUNT,
        "inherited_slice16_projection_sha256": inherited[
            "deterministic_projection_sha256"
        ],
        "replicates": replicates,
        "checks": checks,
        "failed_requirements": failed,
        "failed_boundaries": failed_boundaries,
        "deterministic_projection_sha256": stable_sha256(
            b"SLICE17_PERIODIC_ORBIT_PROJECTION_V1",
            deterministic_projection(replicates[0]),
        ),
        "outcome": OUTCOME_FAIL if failed else empirical_outcome,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "periodic_orbit_test.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(
        receipt, encoding="utf-8"
    )
    gc.collect()
    return result, receipt


def run_verification(seed):
    if seed != SEED_LITERAL:
        raise SystemExit(f"The originating seed must be exactly {SEED_LITERAL}")
    result, receipt = execute_acceptance()
    print(receipt, end="", flush=True)
    return 0 if result["outcome"] != OUTCOME_FAIL else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seed")
    args = parser.parse_args()
    if not args.verify:
        parser.error("Use --verify")
    return run_verification(args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
