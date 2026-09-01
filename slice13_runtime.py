import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE12_RUNTIME = PROJECT_DIR / "slice12_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice13"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
LOCAL_CYCLE_BOUND = 8
MAXIMUM_OBSERVED_RESOLUTIONS = 6
REPLICATION_COUNT = 2
OUTCOME_NO_NEW = "NO_NEW_CAUSAL_DEGREE_OF_FREEDOM"
OUTCOME_FAIL = "FAIL"


def load_slice12_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice12_accumulated_runtime_for_slice13", SLICE12_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 12 runtime: {SLICE12_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def projection_fingerprint(value):
    return stable_sha256(b"SLICE13_CAUSAL_PROJECTION_V1", value)


def same_cache(left, right):
    keys = ("sha256", "component_count", "layer_count", "sequence_length", "tensor_byte_count")
    return all(left[key] == right[key] for key in keys)


def cache_projection(fingerprint):
    return {
        key: fingerprint[key]
        for key in (
            "sha256",
            "component_count",
            "layer_count",
            "sequence_length",
            "tensor_byte_count",
        )
    }


def causal_interface(slice11):
    fields = [item.name for item in dataclasses.fields(slice11.StandingWave)]
    model_inputs = ["poles", "shared_v", "perturbation"]
    local_control = ["phase", "tokens", "previous_pair"]
    gate_and_descent = ["result", "latest_boundary"]
    observational = [
        "instance_id",
        "passive_resolution_label",
        "cycles",
        "bootstrap",
        "mechanism_calls",
        "descent_calls",
    ]
    classified = model_inputs + local_control + gate_and_descent + observational
    return {
        "standing_wave_fields": fields,
        "model_inputs": model_inputs,
        "local_control": local_control,
        "gate_and_descent": gate_and_descent,
        "observational_or_history": observational,
        "all_fields_classified": sorted(fields) == sorted(classified),
        "wave_callable": slice11.callable_evidence(slice11.run_standing_wave),
        "descent_callable": slice11.callable_evidence(slice11.apply_whole_boundary_descent),
    }


def constitution_record(slice11, context, parent, child, depth):
    kernel = context["kernel"]
    boundary = parent.latest_boundary
    parent_pre = cache_projection(boundary["pre"])
    parent_post = cache_projection(boundary["post"])
    child_a = cache_projection(slice11.fp(kernel, child.poles["A"]))
    child_b = cache_projection(slice11.fp(kernel, child.poles["B"]))
    parent_fields = [item.name for item in dataclasses.fields(parent)]
    child_fields = [item.name for item in dataclasses.fields(child)]
    bootstrap = child.bootstrap
    checks = {
        "SAME_STANDING_WAVE_TYPE": type(parent) is type(child),
        "NO_CHILD_ONLY_FIELD": parent_fields == child_fields,
        "S_T_REPLAYED_TO_S_T_PLUS_1_EXACTLY": same_cache(bootstrap["A_post"], bootstrap["S_t_plus_1"]),
        "CHILD_A_EQUALS_PARENT_S_T_PLUS_1": same_cache(child_a, parent_post),
        "CHILD_B_EQUALS_PARENT_S_T_PLUS_1": same_cache(child_b, parent_post),
        "NO_NEW_CACHE_COMPONENT": child_a["component_count"] == parent_post["component_count"]
        and child_b["component_count"] == parent_post["component_count"],
        "NO_NEW_LAYER": child_a["layer_count"] == parent_post["layer_count"]
        and child_b["layer_count"] == parent_post["layer_count"],
        "NO_NEW_SEQUENCE_COORDINATE_AT_DESCENT": child_a["sequence_length"] == parent_post["sequence_length"]
        and child_b["sequence_length"] == parent_post["sequence_length"],
        "NO_NEW_TENSOR_BYTES_AT_DESCENT": child_a["tensor_byte_count"] == parent_post["tensor_byte_count"]
        and child_b["tensor_byte_count"] == parent_post["tensor_byte_count"],
        "EXISTING_PARENT_DELTA_REPLAYED_EXACTLY": bootstrap["Delta_t"] == bootstrap["delta_consumed"],
        "EXISTING_PARENT_OUTPUT_BECOMES_CHILD_PERTURBATION": list(child.perturbation)
        == bootstrap["original_generated_tokens"],
        "V_UNCHANGED_AND_SHARED": child.shared_v is parent.shared_v
        and child.shared_v.literal == V_LITERAL,
        "DUPLICATES_STORAGE_SEPARATED": context["slice3"].cache_tensor_storage_separated(
            child.poles["A"], child.poles["B"]
        ),
        "PARENT_ALREADY_EXPOSES_PRE_POST_AND_DELTA": all(
            key in boundary for key in ("pre_native", "post_native", "delta")
        ),
    }
    return {
        "descent": f"W{depth}_TO_W{depth + 1}",
        "parent_boundary": {
            "S_t": parent_pre,
            "Delta_t": list(boundary["delta"]),
            "S_t_plus_1": parent_post,
        },
        "child_at_constitution": {
            "pole_A": child_a,
            "pole_B": child_b,
            "phase": child.phase,
            "perturbation": list(child.perturbation),
        },
        "checks": checks,
    }


def execute_replicate(slice11, context, replicate):
    labels = (
        list(range(MAXIMUM_OBSERVED_RESOLUTIONS))
        if replicate == 1
        else ["altered-W0", 991, -44, None, {"passive": True}, "altered-W5"]
    )
    wave = slice11.initialize_origin(context, labels[0])
    waves = []
    descents = []
    trajectories = []
    for depth in range(MAXIMUM_OBSERVED_RESOLUTIONS):
        slice11.run_standing_wave(context, wave, LOCAL_CYCLE_BOUND, evaluate_gate=True)
        waves.append(wave)
        trajectories.append(
            {
                "resolution": depth,
                "result": wave.result,
                "cycle_count": len(wave.cycles),
                "cycle_pair_hashes": [item["W_sha256"] for item in wave.cycles],
                "boundary": {
                    "S_t": cache_projection(wave.latest_boundary["pre"]),
                    "Delta_t": list(wave.latest_boundary["delta"]),
                    "S_t_plus_1": cache_projection(wave.latest_boundary["post"]),
                },
            }
        )
        if wave.result != "NO_RECURRENCE_WITHIN_BOUND":
            raise RuntimeError(f"W{depth} did not reproduce Slice 12 unresolved state")
        if depth + 1 == MAXIMUM_OBSERVED_RESOLUTIONS:
            break
        child = slice11.apply_whole_boundary_descent(
            context, wave, f"WAVE_{depth + 1}", labels[depth + 1]
        )
        descents.append(constitution_record(slice11, context, wave, child, depth))
        wave = child
        gc.collect()
    return {
        "replicate": replicate,
        "trajectory": trajectories,
        "descents": descents,
    }


def deterministic_projection(rep):
    return {
        "trajectory": rep["trajectory"],
        "descents": rep["descents"],
    }


def evaluate(interface, replicates):
    all_descent_checks = [
        passed
        for rep in replicates
        for descent in rep["descents"]
        for passed in descent["checks"].values()
    ]
    parent_fields = set(interface["standing_wave_fields"])
    causally_operative = set(
        interface["model_inputs"]
        + interface["local_control"]
        + interface["gate_and_descent"]
    )
    checks = {
        "SLICE12_TRAJECTORY_REPRODUCED": all(
            len(rep["trajectory"]) == MAXIMUM_OBSERVED_RESOLUTIONS
            and all(
                wave["result"] == "NO_RECURRENCE_WITHIN_BOUND"
                and wave["cycle_count"] == LOCAL_CYCLE_BOUND
                for wave in rep["trajectory"]
            )
            for rep in replicates
        ),
        "ALL_FIVE_DESCENTS_EXAMINED": all(len(rep["descents"]) == 5 for rep in replicates),
        "ALL_DESCENT_CONSTITUTIONS_EXACT": bool(all_descent_checks) and all(all_descent_checks),
        "ALL_RUNTIME_FIELDS_CLASSIFIED": interface["all_fields_classified"],
        "NO_CAUSAL_FIELD_UNAVAILABLE_AT_PARENT": causally_operative.issubset(parent_fields),
        "SAME_CAUSAL_INTERFACE_PARENT_AND_CHILD": all(
            descent["checks"]["SAME_STANDING_WAVE_TYPE"]
            and descent["checks"]["NO_CHILD_ONLY_FIELD"]
            for rep in replicates
            for descent in rep["descents"]
        ),
        "NO_NEW_INTERNAL_COORDINATE_AT_CONSTITUTION": all(
            descent["checks"]["NO_NEW_CACHE_COMPONENT"]
            and descent["checks"]["NO_NEW_LAYER"]
            and descent["checks"]["NO_NEW_SEQUENCE_COORDINATE_AT_DESCENT"]
            and descent["checks"]["NO_NEW_TENSOR_BYTES_AT_DESCENT"]
            for rep in replicates
            for descent in rep["descents"]
        ),
        "ONLY_EXISTING_PARENT_CAUSAL_VALUES_INHERITED": all(
            descent["checks"]["S_T_REPLAYED_TO_S_T_PLUS_1_EXACTLY"]
            and descent["checks"]["CHILD_A_EQUALS_PARENT_S_T_PLUS_1"]
            and descent["checks"]["CHILD_B_EQUALS_PARENT_S_T_PLUS_1"]
            and descent["checks"]["EXISTING_PARENT_DELTA_REPLAYED_EXACTLY"]
            and descent["checks"]["EXISTING_PARENT_OUTPUT_BECOMES_CHILD_PERTURBATION"]
            and descent["checks"]["V_UNCHANGED_AND_SHARED"]
            for rep in replicates
            for descent in rep["descents"]
        ),
        "DUPLICATION_IS_ADDRESS_SEPARATION_ONLY": all(
            descent["checks"]["DUPLICATES_STORAGE_SEPARATED"]
            and descent["checks"]["PARENT_ALREADY_EXPOSES_PRE_POST_AND_DELTA"]
            for rep in replicates
            for descent in rep["descents"]
        ),
        "PASSIVE_DEPTH_LABEL_INTERVENTION_HAS_NO_CAUSAL_EFFECT": deterministic_projection(
            replicates[0]
        )
        == deterministic_projection(replicates[1]),
    }
    return checks


def make_receipt(result):
    if result["outcome"] == OUTCOME_FAIL:
        return "\n".join(
            [
                "BUILD_SLICE_RECEIPT",
                "slice: 13",
                "result: FAIL",
                "",
                "failed_requirement:",
                result["failed_checks"][0],
                "",
                "final_result:",
                OUTCOME_FAIL,
                "END_BUILD_SLICE_RECEIPT",
                "",
            ]
        )
    first = result["replicates"][0]
    descents = []
    for item in first["descents"]:
        boundary = item["parent_boundary"]
        child = item["child_at_constitution"]
        descents.append(
            {
                "descent": item["descent"],
                "parent_S_t_plus_1_sha256": boundary["S_t_plus_1"]["sha256"],
                "child_A_sha256": child["pole_A"]["sha256"],
                "child_B_sha256": child["pole_B"]["sha256"],
                "parent_post_sequence_length": boundary["S_t_plus_1"]["sequence_length"],
                "child_A_sequence_length": child["pole_A"]["sequence_length"],
                "child_B_sequence_length": child["pole_B"]["sequence_length"],
            }
        )
    return "\n".join(
        [
            "BUILD_SLICE_RECEIPT",
            "slice: 13",
            "result: PASS",
            "",
            "OUTCOME",
            OUTCOME_NO_NEW,
            "",
            "EVIDENCE",
            f"replications: {REPLICATION_COUNT}",
            "descents_examined_per_replication: 5",
            f"causal_interface_fields: {json.dumps(result['interface']['standing_wave_fields'])}",
            f"descent_constitutions: {json.dumps(descents, sort_keys=True)}",
            f"deterministic_projection_sha256: {result['projection_sha256']}",
            "all_checks: PASS",
            "",
            "final_result:",
            OUTCOME_NO_NEW,
            "END_BUILD_SLICE_RECEIPT",
            "",
        ]
    )


def execute_acceptance():
    slice12 = load_slice12_runtime()
    slice11 = slice12.load_slice11_runtime()
    context = slice11.initialize_context()
    interface = causal_interface(slice11)
    replicates = [
        execute_replicate(slice11, context, number)
        for number in range(1, REPLICATION_COUNT + 1)
    ]
    checks = evaluate(interface, replicates)
    failed = [name for name, passed in checks.items() if not passed]
    projections = [deterministic_projection(rep) for rep in replicates]
    result = {
        "schema": "SLICE13_CAUSAL_RESOLUTION_TEST_V1",
        "V": V_LITERAL,
        "seed": SEED_LITERAL,
        "local_cycle_bound": LOCAL_CYCLE_BOUND,
        "maximum_observed_resolutions": MAXIMUM_OBSERVED_RESOLUTIONS,
        "replication_count": REPLICATION_COUNT,
        "interface": interface,
        "replicates": replicates,
        "checks": checks,
        "failed_checks": failed,
        "projection_sha256": projection_fingerprint(projections[0]),
        "outcome": OUTCOME_NO_NEW if not failed else OUTCOME_FAIL,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "causal_resolution_test.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, context["weight_hash"], receipt


def run_verification(seed):
    if seed != SEED_LITERAL:
        raise SystemExit(f"The originating seed must be exactly {SEED_LITERAL}")
    print("SLICE13_RUNTIME_READY", flush=True)
    print(f"V={V_LITERAL}", flush=True)
    print(f"SEED={seed}", flush=True)
    print(f"LOCAL_CYCLE_BOUND={LOCAL_CYCLE_BOUND}", flush=True)
    print(f"MAXIMUM_OBSERVED_RESOLUTIONS={MAXIMUM_OBSERVED_RESOLUTIONS}", flush=True)
    print(f"REPLICATION_COUNT={REPLICATION_COUNT}", flush=True)
    result, weight_hash, receipt = execute_acceptance()
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    for name, passed in result["checks"].items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    print(f"OUTCOME={result['outcome']}", flush=True)
    print(f"FINAL_RESULT={'PASS' if result['outcome'] != OUTCOME_FAIL else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={EVIDENCE_DIR / 'completion_receipt.txt'}", flush=True)
    print("RECEIPT_BEGIN", flush=True)
    print(receipt, end="", flush=True)
    print("RECEIPT_END", flush=True)
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
