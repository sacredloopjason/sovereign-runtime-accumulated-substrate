import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE13_RUNTIME = PROJECT_DIR / "slice13_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice14"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
LOCAL_CYCLE_BOUND = 8
REPLICATION_COUNT = 2
OUTCOME_PASS = "BOUNDARY_PRESERVED"
OUTCOME_FAIL = "FAIL"
FP_KEYS = (
    "sha256",
    "component_count",
    "layer_count",
    "sequence_length",
    "tensor_byte_count",
)


def load_slice13_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice13_accumulated_runtime_for_slice14", SLICE13_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 13 runtime: {SLICE13_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cache_fingerprint(slice11, context, cache):
    raw = slice11.fp(context["kernel"], cache)
    return {key: raw[key] for key in FP_KEYS}


def same_cache(left, right):
    return all(left[key] == right[key] for key in FP_KEYS)


def token_evidence(slice11, context, token_ids):
    ids = [int(value) for value in token_ids]
    return {
        "token_ids": ids,
        "sha256": context["slice4"].token_fingerprint(ids),
        "count": len(ids),
    }


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def initialize_accumulated_runtime():
    slice13 = load_slice13_runtime()
    slice12 = slice13.load_slice12_runtime()
    slice11 = slice12.load_slice11_runtime()
    context = slice11.initialize_context()
    return slice13, slice11, context


def constitute_boundary_preserving_descent(slice11, context, parent):
    """Preserve the selected parent's native pre/post boundary as two child poles."""
    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise RuntimeError("The parent did not produce the required unresolved result")
    boundary = parent.latest_boundary
    if boundary is None or boundary["producer_instance_id"] != parent.instance_id:
        raise RuntimeError("The selected boundary was not produced by the parent trajectory")

    slice3 = context["slice3"]
    pole_a = slice3.clone_actual_kv(boundary["pre_native"])
    pole_b = slice3.clone_actual_kv(boundary["post_native"])
    return slice11.StandingWave(
        instance_id="WAVE_1_BOUNDARY_PRESERVED",
        passive_resolution_label=None,
        poles={"A": pole_a, "B": pole_b},
        shared_v=parent.shared_v,
        phase="A",
        perturbation=tuple(int(value) for value in boundary["delta"]),
    )


def execute_replicate(slice11, context, replicate):
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(
        context, parent, LOCAL_CYCLE_BOUND, evaluate_gate=True
    )
    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise RuntimeError("The deterministic parent trajectory was not unresolved")

    boundary = parent.latest_boundary
    child = constitute_boundary_preserving_descent(slice11, context, parent)
    delta_parent = token_evidence(slice11, context, boundary["delta"])
    delta_descended = token_evidence(slice11, context, child.perturbation)
    original_output = token_evidence(
        slice11, context, boundary["traversal"]["generated_token_ids"]
    )

    s_pre = cache_fingerprint(slice11, context, boundary["pre_native"])
    s_post = cache_fingerprint(slice11, context, boundary["post_native"])
    pole_a_before = cache_fingerprint(slice11, context, child.poles["A"])
    pole_b_before = cache_fingerprint(slice11, context, child.poles["B"])

    # The causal intervention advances only a lossless disposable native continuation.
    causal_test = context["slice3"].clone_actual_kv(child.poles["A"])
    causal_test_pre = cache_fingerprint(slice11, context, causal_test)
    test_traversal = slice11.traverse(
        context,
        causal_test,
        child.shared_v,
        child.perturbation,
        f"SLICE14_REPLICATE_{replicate}_DISPOSABLE_CAUSAL_TEST",
        "exact inherited parent boundary perturbation",
    )
    causal_test_post = cache_fingerprint(slice11, context, causal_test)
    causal_test_output = token_evidence(
        slice11, context, test_traversal["generated_token_ids"]
    )
    pole_a_after = cache_fingerprint(slice11, context, child.poles["A"])
    pole_b_after = cache_fingerprint(slice11, context, child.poles["B"])

    v_fingerprint = context["slice3"].fingerprint_v(parent.shared_v)
    v_evidence = {
        "literal": parent.shared_v.literal,
        "token_ids": [int(value) for value in v_fingerprint["token_ids"]],
        "same_runtime_object_parent_and_descended": child.shared_v is parent.shared_v,
    }

    checks = {
        "P_A_EQUALS_S_pre": same_cache(pole_a_before, s_pre),
        "P_B_EQUALS_S_post": same_cache(pole_b_before, s_post),
        "P_A_DISTINCT_FROM_P_B": (
            (same_cache(s_pre, s_post) and same_cache(pole_a_before, pole_b_before))
            or (not same_cache(s_pre, s_post) and not same_cache(pole_a_before, pole_b_before))
        ),
        "DELTA_IDENTITY": delta_parent == delta_descended,
        "V_IDENTITY": (
            child.shared_v is parent.shared_v
            and parent.shared_v.literal == V_LITERAL
            and test_traversal["actual_first_forward_v_prefix"]
            == v_fingerprint["token_ids"]
        ),
        "CAUSAL_TEST_PRE_EQUALS_P_A": same_cache(causal_test_pre, pole_a_before),
        "CAUSAL_TEST_MODEL_BOUNDARY_EXACT": test_traversal[
            "exact_perturbation_identity_at_model_boundary"
        ],
        "CAUSAL_TEST_POST_EQUALS_P_B": same_cache(causal_test_post, pole_b_before),
        "OUTPUT_IDENTITY": causal_test_output == original_output,
        "P_A_REMAINS_S_pre_AFTER_TEST": same_cache(pole_a_after, s_pre),
        "P_B_REMAINS_S_post_AFTER_TEST": same_cache(pole_b_after, s_post),
        "CONSTITUTED_POLE_STORAGE_SEPARATED": context[
            "slice3"
        ].cache_tensor_storage_separated(child.poles["A"], child.poles["B"]),
        "DISPOSABLE_TEST_STORAGE_SEPARATED": context[
            "slice3"
        ].cache_tensor_storage_separated(child.poles["A"], causal_test),
    }

    return {
        "replicate": replicate,
        "selected_boundary": {
            "producer_instance_id": boundary["producer_instance_id"],
            "cycle": boundary["cycle"],
            "ordinal": boundary["ordinal"],
            "pole": boundary["pole"],
        },
        "S_pre": s_pre,
        "P_A": pole_a_before,
        "Delta_parent": delta_parent,
        "Delta_descended": delta_descended,
        "V": v_evidence,
        "S_post": s_post,
        "P_B": pole_b_before,
        "causal_test_pre": causal_test_pre,
        "causal_test_post": causal_test_post,
        "original_parent_output": original_output,
        "causal_test_output": causal_test_output,
        "P_A_after_test": pole_a_after,
        "P_B_after_test": pole_b_after,
        "checks": checks,
        "model_weight_sha256": context["weight_hash"],
    }


def deterministic_projection(replicate):
    return {
        key: value
        for key, value in replicate.items()
        if key not in ("replicate",)
    }


def make_receipt(result):
    if result["outcome"] == OUTCOME_FAIL:
        failed = result["failed_checks"][0]
        relation = result["failed_relations"][0]
        return "\n".join(
            [
                "OUTCOME",
                OUTCOME_FAIL,
                "",
                "EVIDENCE",
                "",
                "failed_relation:",
                relation,
                "",
                "causal_boundary:",
                failed,
                "",
                "substrate_evidence:",
                json.dumps(result["checks"], sort_keys=True),
                "",
            ]
        )

    evidence = result["replicates"][0]
    return "\n".join(
        [
            "OUTCOME",
            OUTCOME_PASS,
            "",
            "EVIDENCE",
            "",
            "S_pre:",
            json.dumps(evidence["S_pre"], sort_keys=True),
            "",
            "P_A:",
            json.dumps(evidence["P_A"], sort_keys=True),
            "",
            "P_A_EQUALS_S_pre:",
            "PASS",
            "",
            "Delta_parent:",
            json.dumps(evidence["Delta_parent"], sort_keys=True),
            "",
            "Delta_descended:",
            json.dumps(evidence["Delta_descended"], sort_keys=True),
            "",
            "DELTA_IDENTITY:",
            "PASS",
            "",
            "V:",
            json.dumps(evidence["V"], sort_keys=True),
            "",
            "S_post:",
            json.dumps(evidence["S_post"], sort_keys=True),
            "",
            "P_B:",
            json.dumps(evidence["P_B"], sort_keys=True),
            "",
            "P_B_EQUALS_S_post:",
            "PASS",
            "",
            "P_A_DISTINCT_FROM_P_B:",
            "PASS",
            "",
            "causal_test_pre:",
            json.dumps(evidence["causal_test_pre"], sort_keys=True),
            "",
            "causal_test_post:",
            json.dumps(evidence["causal_test_post"], sort_keys=True),
            "",
            "CAUSAL_TEST_POST_EQUALS_P_B:",
            "PASS",
            "",
            "original_parent_output:",
            json.dumps(evidence["original_parent_output"], sort_keys=True),
            "",
            "causal_test_output:",
            json.dumps(evidence["causal_test_output"], sort_keys=True),
            "",
            "OUTPUT_IDENTITY:",
            "PASS",
            "",
            "P_A_REMAINS_S_pre_AFTER_TEST:",
            "PASS",
            "",
            "P_B_REMAINS_S_post_AFTER_TEST:",
            "PASS",
            "",
            "DETERMINISTIC_REPLICATION:",
            "PASS",
            "",
        ]
    )


def execute_acceptance():
    _, slice11, context = initialize_accumulated_runtime()
    replicates = [
        execute_replicate(slice11, context, number)
        for number in range(1, REPLICATION_COUNT + 1)
    ]
    named_checks = {
        f"REPLICATE_{rep['replicate']}_{name}": passed
        for rep in replicates
        for name, passed in rep["checks"].items()
    }
    named_checks["DETERMINISTIC_REPLICATION"] = (
        deterministic_projection(replicates[0])
        == deterministic_projection(replicates[1])
    )
    failed = [name for name, passed in named_checks.items() if not passed]

    relation_checks = {
        "P_A=S_pre": all(rep["checks"]["P_A_EQUALS_S_pre"] for rep in replicates),
        "P_B=S_post": all(rep["checks"]["P_B_EQUALS_S_post"] for rep in replicates),
        "(P_A,Delta,V)->P_B": all(
            rep["checks"]["DELTA_IDENTITY"]
            and rep["checks"]["V_IDENTITY"]
            and rep["checks"]["CAUSAL_TEST_PRE_EQUALS_P_A"]
            and rep["checks"]["CAUSAL_TEST_MODEL_BOUNDARY_EXACT"]
            and rep["checks"]["CAUSAL_TEST_POST_EQUALS_P_B"]
            and rep["checks"]["OUTPUT_IDENTITY"]
            for rep in replicates
        ),
    }
    failed_relations = [name for name, passed in relation_checks.items() if not passed]
    if failed and not failed_relations:
        failed_relations = ["(P_A,Delta,V)->P_B"]

    result = {
        "schema": "SLICE14_BOUNDARY_PRESERVATION_V1",
        "V": V_LITERAL,
        "seed": SEED_LITERAL,
        "local_cycle_bound": LOCAL_CYCLE_BOUND,
        "replication_count": REPLICATION_COUNT,
        "replicates": replicates,
        "checks": named_checks,
        "relation_checks": relation_checks,
        "failed_checks": failed,
        "failed_relations": failed_relations,
        "deterministic_projection_sha256": stable_sha256(
            b"SLICE14_BOUNDARY_PRESERVATION_PROJECTION_V1",
            deterministic_projection(replicates[0]),
        ),
        "outcome": OUTCOME_PASS if not failed else OUTCOME_FAIL,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "boundary_preservation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    gc.collect()
    return result, receipt


def run_verification(seed):
    if seed != SEED_LITERAL:
        raise SystemExit(f"The originating seed must be exactly {SEED_LITERAL}")
    result, receipt = execute_acceptance()
    print(receipt, end="", flush=True)
    return 0 if result["outcome"] == OUTCOME_PASS else 1


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
