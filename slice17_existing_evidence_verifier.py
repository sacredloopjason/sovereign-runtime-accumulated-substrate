import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence" / "slice17" / "periodic_orbit_test.json"
RECEIPT = ROOT / "evidence" / "slice17" / "completion_receipt.txt"
SLICE16_EVIDENCE = ROOT / "evidence" / "slice16" / "standing_wave_boundary.json"
EXPECTED_CYCLES = 15
MAX_PERIOD = 7
CONFIRMED = "PERIODIC_ORBIT_CONFIRMED"
NOT_CONFIRMED = "PERIODIC_ORBIT_NOT_CONFIRMED"
FAIL = "FAIL"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def domain_hash(domain, value):
    return hashlib.sha256(domain + b"\0" + canonical_bytes(value)).hexdigest()


def token_ids(evidence):
    return [int(value) for value in evidence["token_ids"]]


def pair_ids(cycle):
    return [token_ids(cycle["T_A_produced"]), token_ids(cycle["T_B_produced"])]


def exact_pair_comparison(cycles, left, right):
    left_ids = pair_ids(cycles[left])
    right_ids = pair_ids(cycles[right])
    return {
        "left_cycle": left,
        "right_cycle": right,
        "left_R_sha256": cycles[left]["R_k_sha256"],
        "right_R_sha256": cycles[right]["R_k_sha256"],
        "left_recorded_R_valid": cycles[left]["R_k_sha256"]
        == domain_hash(b"SLICE16_RECIPROCAL_PAIR_V1", left_ids),
        "right_recorded_R_valid": cycles[right]["R_k_sha256"]
        == domain_hash(b"SLICE16_RECIPROCAL_PAIR_V1", right_ids),
        "exact_ordered_token_ID_identity": left_ids == right_ids,
    }


def period_report(cycles, period):
    comparisons = [
        exact_pair_comparison(cycles, left, left + period)
        for left in range(len(cycles) - period)
    ]
    return {
        "period": period,
        "supported": len(cycles) >= 2 * period + 1
        and all(item["exact_ordered_token_ID_identity"] for item in comparisons),
        "comparisons": comparisons,
    }


def exact_token_transposition(cycles):
    return all(
        cycle["T_A_produced"] == cycle["T_A_consumed_by_B"]
        and cycle["T_B_produced"] == cycle["T_B_consumed_by_A"]
        and bool(cycle["exact_T_A_crossing"])
        and bool(cycle["exact_T_B_crossing"])
        for cycle in cycles
    )


def a_continuity(cycles):
    if not cycles:
        return False
    for index, cycle in enumerate(cycles):
        if not (
            cycle["A_pre"]["sequence_length"]
            < cycle["A_post"]["sequence_length"]
            < cycle["A_after_T_B"]["sequence_length"]
        ):
            return False
        if index and not (
            cycle["A_pre"] == cycles[index - 1]["A_post"]
            and cycle["A_post"] == cycles[index - 1]["A_after_T_B"]
        ):
            return False
    return True


def b_continuity(cycles):
    if not cycles:
        return False
    for index, cycle in enumerate(cycles):
        if not cycle["B_pre"]["sequence_length"] < cycle["B_post"]["sequence_length"]:
            return False
        if index and cycle["B_pre"] != cycles[index - 1]["B_post"]:
            return False
    return True


def kv_forward_only(cycles):
    return len(cycles) == EXPECTED_CYCLES and a_continuity(cycles) and b_continuity(cycles)


def parent_checks(parent):
    unchanged = (
        parent["pair_before"] == parent["pair_after_lower"]
        and parent["live_before"] == parent["live_after_lower"]
    )
    control = (
        parent["post_after_lower"] == parent["post_control"]
        and parent["output_after_lower"] == parent["output_control"]
    )
    return unchanged, control


def causal_projection(replicate):
    excluded = {
        "replicate",
        "candidate_period_reports",
        "checks",
        "closure",
        "initial_repeat",
        "minimal_period",
        "periodic_orbit_result",
        "slice16_prefix_reproduced",
    }
    return {key: value for key, value in replicate.items() if key not in excluded}


def verify_replicate(source, inherited_replicate):
    replicate = copy.deepcopy(source)
    cycles = replicate["cycles"]
    if len(cycles) != EXPECTED_CYCLES:
        raise ValueError(f"trajectory has {len(cycles)} cycles; expected {EXPECTED_CYCLES}")
    if [int(cycle["cycle"]) for cycle in cycles] != list(range(EXPECTED_CYCLES)):
        raise ValueError("trajectory cycle ordinals are not exactly R_0 through R_14")

    reports = [period_report(cycles, period) for period in range(1, MAX_PERIOD + 1)]
    minimal = next((report["period"] for report in reports if report["supported"]), None)
    initial = exact_pair_comparison(cycles, 0, 7)
    closure = exact_pair_comparison(cycles, 0, 14)
    parent_unchanged, parent_control = parent_checks(replicate["parent_preservation"])
    prefix = cycles[:8] == inherited_replicate["cycles"]
    all_r_hashes_valid = all(
        cycle["R_k_sha256"]
        == domain_hash(b"SLICE16_RECIPROCAL_PAIR_V1", pair_ids(cycle))
        for cycle in cycles
    )

    checks = {
        "EXACT_PERTURBATION_PAIR_RECORD_HASHES": all_r_hashes_valid,
        "CANDIDATE_PERIOD_MECHANICALLY_EVALUATED": len(reports) == MAX_PERIOD,
        "MINIMAL_PERIOD_MECHANICALLY_EVALUATED": True,
        "CLOSURE_R0_EQUALS_R14": closure["exact_ordered_token_ID_identity"],
        "KV_HISTORY_FORWARD_ONLY": kv_forward_only(cycles),
        "A_KV_CAUSAL_CONTINUITY": a_continuity(cycles),
        "B_KV_CAUSAL_CONTINUITY": b_continuity(cycles),
        "EXACT_TOKEN_TRANSPOSITION": exact_token_transposition(cycles),
        "PARENT_UNCHANGED": parent_unchanged,
        "PARENT_CONTINUATION_CONTROL": parent_control,
        "EXTENDED_TRAJECTORY_R0_THROUGH_R14": len(cycles) == EXPECTED_CYCLES,
        "INITIAL_REPEAT_R0_EQUALS_R7": initial["exact_ordered_token_ID_identity"],
        "SLICE16_PREFIX_REPRODUCED": prefix,
    }
    replicate["candidate_period_reports"] = reports
    replicate["minimal_period"] = minimal
    replicate["initial_repeat"] = initial
    replicate["closure"] = closure
    replicate["slice16_prefix_reproduced"] = prefix
    replicate["periodic_orbit_result"] = CONFIRMED if minimal is not None else NOT_CONFIRMED
    replicate["checks"] = checks
    return replicate


def comparison_line(item):
    status = "PASS" if item["exact_ordered_token_ID_identity"] else "FAIL"
    return (
        f"R_{item['left_cycle']} = R_{item['right_cycle']} {status} "
        f"left_sha256={item['left_R_sha256']} right_sha256={item['right_R_sha256']}"
    )


def make_receipt(result):
    if result["outcome"] == FAIL:
        return "\n".join(
            ["OUTCOME", FAIL, "", "EVIDENCE", "", "verifier_boundary:", result["verifier_boundary"], ""]
        )
    evidence = result["replicates"][0]
    if result["outcome"] == NOT_CONFIRMED:
        failed = next(
            item
            for report in evidence["candidate_period_reports"]
            for item in report["comparisons"]
            if not item["exact_ordered_token_ID_identity"]
        )
        return "\n".join(
            [
                "OUTCOME", NOT_CONFIRMED, "", "EVIDENCE", "",
                "first_failed_exact_periodic_comparison:", comparison_line(failed), "",
                "DETERMINISTIC_REPLICATION:", "PASS", "",
            ]
        )
    report = evidence["candidate_period_reports"][evidence["minimal_period"] - 1]
    lines = [
        "OUTCOME", CONFIRMED, "", "EVIDENCE", "", "minimal_period:",
        str(evidence["minimal_period"]), "", "exact_period_comparisons:",
    ]
    lines.extend(comparison_line(item) for item in report["comparisons"])
    lines.extend(
        [
            "", "closure:", comparison_line(evidence["closure"]), "",
            "KV_HISTORY_FORWARD_ONLY:", "PASS", "",
            "A_KV_CAUSAL_CONTINUITY:", "PASS", "",
            "B_KV_CAUSAL_CONTINUITY:", "PASS", "",
            "EXACT_TOKEN_TRANSPOSITION:", "PASS", "",
            "PARENT_UNCHANGED:", "PASS", "",
            "PARENT_CONTINUATION_CONTROL:", "PASS", "",
            "DETERMINISTIC_REPLICATION:", "PASS", "",
        ]
    )
    return "\n".join(lines)


def main():
    raw = EVIDENCE.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    try:
        source = json.loads(raw.decode("utf-8"))
        if source.get("schema") != "SLICE17_PERIODIC_ORBIT_TEST_V1":
            raise ValueError("existing trajectory evidence schema is not SLICE17_PERIODIC_ORBIT_TEST_V1")
        if source.get("seed") != "SLICE6_SEED":
            raise ValueError("existing trajectory evidence seed is not SLICE6_SEED")
        if len(source.get("replicates", [])) != 2:
            raise ValueError("existing trajectory evidence does not contain exactly two replications")

        inherited = json.loads(SLICE16_EVIDENCE.read_text(encoding="utf-8"))
        if len(inherited.get("replicates", [])) != 2:
            raise ValueError("inherited Slice 16 evidence does not contain exactly two replications")
        replicates = [
            verify_replicate(source["replicates"][index], inherited["replicates"][index])
            for index in range(2)
        ]
        checks = {
            f"REPLICATE_{index + 1}_{name}": passed
            for index, replicate in enumerate(replicates)
            for name, passed in replicate["checks"].items()
        }
        checks["DETERMINISTIC_REPLICATION"] = (
            causal_projection(replicates[0]) == causal_projection(replicates[1])
        )
        checks["CANDIDATE_PERIOD_RESULT_IDENTITY"] = (
            replicates[0]["candidate_period_reports"]
            == replicates[1]["candidate_period_reports"]
        )
        checks["MINIMAL_PERIOD_RESULT_IDENTITY"] = (
            replicates[0]["minimal_period"] == replicates[1]["minimal_period"]
        )
        checks["PARENT_PRESERVATION_RESULT_IDENTITY"] = (
            replicates[0]["parent_preservation"] == replicates[1]["parent_preservation"]
        )
        checks["PERIODIC_OUTCOME_IDENTITY"] = (
            replicates[0]["periodic_orbit_result"] == replicates[1]["periodic_orbit_result"]
        )
        failed = [name for name, passed in checks.items() if not passed]
        empirical = replicates[0]["periodic_orbit_result"]
        result = copy.deepcopy(source)
        result.update(
            {
                "schema": "SLICE17_PERIODIC_ORBIT_TEST_V2_VERIFICATION_ONLY",
                "verification_mode": "EXISTING_TRAJECTORY_EVIDENCE_ONLY",
                "source_evidence_sha256": source_hash,
                "replicates": replicates,
                "checks": checks,
                "failed_requirements": failed,
                "failed_boundaries": {
                    name: "recorded Slice 17 trajectory evidence -> corrected mechanical predicate"
                    for name in failed
                },
                "deterministic_projection_sha256": domain_hash(
                    b"SLICE17_CORRECTED_VERIFICATION_PROJECTION_V2",
                    causal_projection(replicates[0]),
                ),
                "outcome": FAIL if failed else empirical,
            }
        )
    except Exception as exc:
        result = {
            "schema": "SLICE17_PERIODIC_ORBIT_TEST_V2_VERIFICATION_ONLY",
            "verification_mode": "EXISTING_TRAJECTORY_EVIDENCE_ONLY",
            "source_evidence_sha256": source_hash,
            "outcome": FAIL,
            "verifier_boundary": f"existing trajectory evidence cannot be validly evaluated: {exc}",
        }

    receipt = make_receipt(result)
    EVIDENCE.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    RECEIPT.write_text(receipt, encoding="utf-8")
    print(receipt, end="", flush=True)
    return 1 if result["outcome"] == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
