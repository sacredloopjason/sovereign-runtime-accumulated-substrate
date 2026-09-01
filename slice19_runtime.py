import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SLICE18_RUNTIME = ROOT / "slice18_runtime.py"
EVIDENCE_DIR = ROOT / "evidence" / "slice19"
SEED_LITERAL = "SLICE6_SEED"
EXPECTED_F_IDENTITY = "38c9de28173df67de170a7f36f4d895ad2dfea251b8460734a0cb4cb4a3f024f"
REPLICATION_COUNT = 2
OUTCOME_OBSERVED = "UPWARD_CAUSAL_PROPAGATION_OBSERVED"
OUTCOME_NOT_OBSERVED = "UPWARD_CAUSAL_PROPAGATION_NOT_OBSERVED"
OUTCOME_FAIL = "FAIL"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def stable_hash(domain, value):
    return hashlib.sha256(domain + b"\0" + canonical_bytes(value)).hexdigest()


def same_cache(slice16, left, right):
    return slice16.same_cache(left, right)


def compact_traversal(slice16, context, traversal, post):
    return {
        "input": slice16.token_evidence(context, traversal["perturbation_token_ids_supplied"]),
        "actual_first_forward_input_ids": traversal["actual_first_forward_input_ids"],
        "actual_first_forward_v_prefix": traversal["actual_first_forward_v_prefix"],
        "actual_first_forward_perturbation_ids": traversal["actual_first_forward_perturbation_ids"],
        "exact_perturbation_identity_at_model_boundary": traversal[
            "exact_perturbation_identity_at_model_boundary"
        ],
        "first_forward_pre_sequence_length": traversal["forward_events"][0][
            "pre_sequence_length"
        ],
        "first_forward_post_sequence_length": traversal["forward_events"][0][
            "post_sequence_length"
        ],
        "post": post,
        "generated": slice16.token_evidence(context, traversal["generated_token_ids"]),
    }


def execute_replicate(slice18, slice16, slice15, slice14, slice11, context, replicate):
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(context, parent, slice16.LOCAL_CYCLE_BOUND, evaluate_gate=True)
    if parent.result != slice16.OUTCOME_NO_RECURRENCE:
        raise RuntimeError("The deterministic parent trajectory was not unresolved")

    boundary = parent.latest_boundary
    parent_live_name = boundary["pole"]
    parent_live = parent.poles[parent_live_name]
    child = slice14.constitute_boundary_preserving_descent(slice11, context, parent)

    parent_pair_initial = slice16.pair_fingerprint(slice11, context, parent.poles)
    parent_live_initial = slice16.cache_fingerprint(slice11, context, parent_live)
    parent_branch_control = context["slice3"].clone_actual_kv(parent_live)
    parent_branch_fact = context["slice3"].clone_actual_kv(parent_live)
    parent_clean_reference = context["slice3"].clone_actual_kv(parent_live)
    parent_pre_control = slice16.cache_fingerprint(slice11, context, parent_branch_control)
    parent_pre_fact = slice16.cache_fingerprint(slice11, context, parent_branch_fact)

    slice18.CAPTURES.pop(replicate, None)
    lower = slice18.run_slice18_wave(slice16, slice11, context, child, replicate)
    fact = slice18.CAPTURES[replicate]["fact"]
    if fact is None:
        raise RuntimeError("Slice 18 did not produce the stabilized fact")

    lower_poles_before_upward = slice16.pair_fingerprint(slice11, context, child.poles)
    lower_orbit_before = stable_hash(b"SLICE19_LOWER_ORBIT_EVIDENCE_V1", lower["cycles"])
    fact_bytes_before = fact["bytes"]
    fact_sha_before = fact["sha256"]

    fact_text = fact_bytes_before.decode("ascii")
    fact_up_ids = tuple(
        int(value)
        for value in context["tokenizer"](
            fact_text, add_special_tokens=False
        ).input_ids
    )
    recovered_text = context["tokenizer"].decode(
        fact_up_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    recovered_bytes = recovered_text.encode("ascii")
    lossless = recovered_bytes == fact_bytes_before

    ordinary_ids = tuple(
        int(value)
        for value in context["slice7"].token_ids(
            context["tokenizer"], slice16.PARENT_CONTINUE_LITERAL
        )
    )
    control = slice11.traverse(
        context,
        parent_branch_control,
        parent.shared_v,
        ordinary_ids,
        f"SLICE19_REPLICATE_{replicate}_MATCHED_PARENT_CONTROL",
        "inherited narrow ordinary parent continuation condition",
    )
    control_post = slice16.cache_fingerprint(slice11, context, parent_branch_control)

    fact_condition = slice11.traverse(
        context,
        parent_branch_fact,
        parent.shared_v,
        fact_up_ids,
        f"SLICE19_REPLICATE_{replicate}_MATCHED_PARENT_F",
        "lossless tokenizer representation of exact Slice 18 F bytes",
    )
    fact_post = slice16.cache_fingerprint(slice11, context, parent_branch_fact)

    control_compact = compact_traversal(slice16, context, control, control_post)
    fact_compact = compact_traversal(slice16, context, fact_condition, fact_post)
    post_differs = not same_cache(slice16, control_post, fact_post)
    output_differs = control_compact["generated"] != fact_compact["generated"]
    divergence = post_differs or output_differs
    earliest_divergence = None
    if (
        control_compact["first_forward_post_sequence_length"]
        != fact_compact["first_forward_post_sequence_length"]
    ):
        earliest_divergence = {
            "boundary": "first model.forward return",
            "field": "native DynamicCache post_sequence_length",
            "control": control_compact["first_forward_post_sequence_length"],
            "F": fact_compact["first_forward_post_sequence_length"],
        }
    elif post_differs:
        earliest_divergence = {
            "boundary": "completed parent traversal",
            "field": "complete native DynamicCache fingerprint",
            "control": control_post["sha256"],
            "F": fact_post["sha256"],
        }
    elif output_differs:
        earliest_divergence = {
            "boundary": "generated token sequence",
            "field": "exact ordered token IDs",
            "control": control_compact["generated"]["token_ids"],
            "F": fact_compact["generated"]["token_ids"],
        }

    lower_poles_after_upward = slice16.pair_fingerprint(slice11, context, child.poles)
    lower_orbit_after = stable_hash(b"SLICE19_LOWER_ORBIT_EVIDENCE_V1", lower["cycles"])
    parent_pair_after_test = slice16.pair_fingerprint(slice11, context, parent.poles)
    parent_live_after_test = slice16.cache_fingerprint(slice11, context, parent_live)

    parent_live_probe = context["slice3"].clone_actual_kv(parent_live)
    live_probe_pre = slice16.cache_fingerprint(slice11, context, parent_live_probe)
    clean_reference_pre = slice16.cache_fingerprint(slice11, context, parent_clean_reference)
    live_probe = slice11.traverse(
        context,
        parent_live_probe,
        parent.shared_v,
        ordinary_ids,
        f"SLICE19_REPLICATE_{replicate}_LIVE_PARENT_PROBE",
        "ordinary continuation on disposable clone of unchanged live parent",
    )
    live_probe_post = slice16.cache_fingerprint(slice11, context, parent_live_probe)
    clean_reference = slice11.traverse(
        context,
        parent_clean_reference,
        parent.shared_v,
        ordinary_ids,
        f"SLICE19_REPLICATE_{replicate}_CLEAN_PARENT_REFERENCE",
        "ordinary continuation on preserved clean parent reference",
    )
    clean_reference_post = slice16.cache_fingerprint(slice11, context, parent_clean_reference)

    fact_participated = (
        fact_condition["exact_perturbation_identity_at_model_boundary"]
        and fact_condition["actual_first_forward_perturbation_ids"] == list(fact_up_ids)
        and fact_condition["actual_first_forward_v_prefix"]
        == [int(value) for value in parent.shared_v.token_ids]
    )
    checks = {
        "F_IDENTITY": fact_sha_before == EXPECTED_F_IDENTITY,
        "LOSSLESS_F_TO_UPWARD_REPRESENTATION": lossless,
        "PARENT_PRE_STATES_EQUAL": same_cache(slice16, parent_pre_control, parent_pre_fact),
        "PARENT_BRANCH_STORAGE_SEPARATED": context[
            "slice3"
        ].cache_tensor_storage_separated(parent_branch_control, parent_branch_fact),
        "F_PARTICIPATED_IN_PARENT_INFERENCE": fact_participated,
        "CONTROL_USED_INHERITED_ORDINARY_CONTINUATION": control[
            "actual_first_forward_perturbation_ids"
        ] == list(ordinary_ids),
        "F_UNCHANGED": fact["bytes"] == fact_bytes_before and fact["sha256"] == fact_sha_before,
        "LOWER_POLES_UNCHANGED": (
            slice16.same_pair(lower_poles_before_upward, lower_poles_after_upward)
            and lower_orbit_before == lower_orbit_after
        ),
        "LIVE_PARENT_UNCHANGED_BY_TEST": (
            slice16.same_pair(parent_pair_initial, parent_pair_after_test)
            and same_cache(slice16, parent_live_initial, parent_live_after_test)
        ),
        "LIVE_PARENT_CONTINUATION_CONTROL": (
            same_cache(slice16, live_probe_pre, clean_reference_pre)
            and same_cache(slice16, live_probe_post, clean_reference_post)
            and live_probe["generated_token_ids"] == clean_reference["generated_token_ids"]
            and live_probe["exact_perturbation_identity_at_model_boundary"]
            and clean_reference["exact_perturbation_identity_at_model_boundary"]
        ),
        "LOWER_PERIODIC_ORBIT_REPRODUCED": (
            fact["record"]["period"] == 7
            and fact["record"]["closure_R7"]
            == fact["record"]["orbit_R0_through_R6"][0]
            and not any(
                all(
                    fact["record"]["orbit_R0_through_R6"][index]
                    == fact["record"]["orbit_R0_through_R6"][(index + candidate) % 7]
                    for index in range(7)
                )
                for candidate in range(1, 7)
            )
        ),
    }

    return {
        "replicate": replicate,
        "F": {
            "identity": fact_sha_before,
            "substrate": fact["substrate"],
            "bytes_length": len(fact_bytes_before),
        },
        "F_upward_representation": {
            "operation": "canonical ASCII F bytes -> UTF-8-identical ASCII text -> tokenizer token IDs",
            "token_ids": list(fact_up_ids),
            "token_count": len(fact_up_ids),
            "recovered_F_bytes_sha256": hashlib.sha256(recovered_bytes).hexdigest(),
            "exact_recovery": lossless,
        },
        "parent_boundary": {
            "producer_instance_id": boundary["producer_instance_id"],
            "cycle": boundary["cycle"],
            "ordinal": boundary["ordinal"],
            "live_pole": parent_live_name,
        },
        "parent_pre_control": parent_pre_control,
        "parent_pre_F": parent_pre_fact,
        "control": control_compact,
        "F_condition": fact_compact,
        "causal_comparison": {
            "complete_post_KV_differs": post_differs,
            "generated_token_sequence_differs": output_differs,
            "downstream_divergence": divergence,
            "earliest_divergent_boundary": earliest_divergence,
        },
        "preservation": {
            "lower_poles_before_upward": lower_poles_before_upward,
            "lower_poles_after_upward": lower_poles_after_upward,
            "lower_orbit_evidence_before": lower_orbit_before,
            "lower_orbit_evidence_after": lower_orbit_after,
            "parent_pair_initial": parent_pair_initial,
            "parent_pair_after_test": parent_pair_after_test,
            "parent_live_initial": parent_live_initial,
            "parent_live_after_test": parent_live_after_test,
            "live_parent_control_post": live_probe_post,
            "clean_reference_post": clean_reference_post,
        },
        "checks": checks,
    }


def deterministic_projection(replicate):
    return {key: value for key, value in replicate.items() if key != "replicate"}


def execute_acceptance():
    slice18 = load_module(SLICE18_RUNTIME, "slice18_accumulated_runtime_for_slice19")
    slice16 = slice18.load_module(slice18.SLICE16_RUNTIME, "slice16_accumulated_runtime_for_slice19")
    slice15 = slice16.load_slice15_runtime()
    slice14 = slice15.load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()

    replicates = [
        execute_replicate(slice18, slice16, slice15, slice14, slice11, context, number)
        for number in range(1, REPLICATION_COUNT + 1)
    ]
    checks = {
        f"REPLICATE_{rep['replicate']}_{name}": passed
        for rep in replicates
        for name, passed in rep["checks"].items()
    }
    checks["DETERMINISTIC_REPLICATION"] = (
        deterministic_projection(replicates[0]) == deterministic_projection(replicates[1])
    )
    failed = [name for name, passed in checks.items() if not passed]
    valid_experiment = not failed
    divergence = all(rep["causal_comparison"]["downstream_divergence"] for rep in replicates)
    if not valid_experiment:
        outcome = OUTCOME_FAIL
    elif divergence:
        outcome = OUTCOME_OBSERVED
    else:
        outcome = OUTCOME_NOT_OBSERVED
    result = {
        "schema": "SLICE19_UPWARD_CAUSAL_PROPAGATION_V1",
        "seed": SEED_LITERAL,
        "replication_count": REPLICATION_COUNT,
        "replicates": replicates,
        "checks": checks,
        "failed_requirements": failed,
        "outcome": outcome,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "upward_causal_propagation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, receipt


def make_receipt(result):
    if result["outcome"] == OUTCOME_FAIL:
        failed = result["failed_requirements"][0] if result["failed_requirements"] else "controlled experiment"
        return "\n".join([
            "OUTCOME", "FAIL", "", "EVIDENCE", "", "failed_requirement:", failed, "",
            "causal_boundary:", "exact Slice 18 F -> matched disposable parent inference", "",
            "substrate_evidence:", json.dumps(result["checks"], sort_keys=True), "",
        ])

    rep = result["replicates"][0]
    if result["outcome"] == OUTCOME_NOT_OBSERVED:
        return "\n".join([
            "OUTCOME", OUTCOME_NOT_OBSERVED, "", "EVIDENCE", "",
            "F_PARTICIPATED_IN_PARENT_INFERENCE:", "PASS", "",
            "CONTROL_AND_F_TRAJECTORIES_EQUAL:", "PASS", "",
            "F_UNCHANGED:", "PASS", "", "LOWER_POLES_UNCHANGED:", "PASS", "",
            "LIVE_PARENT_UNCHANGED_BY_TEST:", "PASS", "",
            "DETERMINISTIC_REPLICATION:", "PASS", "",
        ])

    return "\n".join([
        "OUTCOME", OUTCOME_OBSERVED, "", "EVIDENCE", "",
        "F_IDENTITY:", rep["F"]["identity"], "",
        "F_SUBSTRATE:", rep["F"]["substrate"], "",
        "F_UPWARD_REPRESENTATION:", json.dumps(rep["F_upward_representation"], separators=(",", ":")), "",
        "LOSSLESS_F_TO_UPWARD_REPRESENTATION:", "PASS", "",
        "PARENT_PRE_CONTROL:", rep["parent_pre_control"]["sha256"], "",
        "PARENT_PRE_F:", rep["parent_pre_F"]["sha256"], "",
        "PARENT_PRE_STATES_EQUAL:", "PASS", "",
        "CONTROL_INPUT:", json.dumps(rep["control"]["input"], separators=(",", ":")), "",
        "CONTROL_POST:", rep["control"]["post"]["sha256"], "",
        "CONTROL_OUTPUT:", json.dumps(rep["control"]["generated"], separators=(",", ":")), "",
        "F_CONDITION_INPUT:", json.dumps(rep["F_condition"]["input"], separators=(",", ":")), "",
        "F_CONDITION_POST:", rep["F_condition"]["post"]["sha256"], "",
        "F_CONDITION_OUTPUT:", json.dumps(rep["F_condition"]["generated"], separators=(",", ":")), "",
        "F_PARTICIPATED_IN_PARENT_INFERENCE:", "PASS", "",
        "DOWNSTREAM_DIVERGENCE:", "PASS", "",
        "earliest_divergent_boundary:", json.dumps(rep["causal_comparison"]["earliest_divergent_boundary"], separators=(",", ":")), "",
        "F_UNCHANGED:", "PASS", "", "LOWER_POLES_UNCHANGED:", "PASS", "",
        "LIVE_PARENT_UNCHANGED_BY_TEST:", "PASS", "",
        "LIVE_PARENT_CONTINUATION_CONTROL:", "PASS", "",
        "DETERMINISTIC_REPLICATION:", "PASS", "",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seed")
    args = parser.parse_args()
    if not args.verify:
        parser.error("Use --verify")
    if args.seed != SEED_LITERAL:
        raise SystemExit(f"The originating seed must be exactly {SEED_LITERAL}")
    result, receipt = execute_acceptance()
    print(receipt, end="", flush=True)
    return 0 if result["outcome"] != OUTCOME_FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
