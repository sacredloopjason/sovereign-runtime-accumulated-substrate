import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE14_RUNTIME = PROJECT_DIR / "slice14_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice15"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
INTERVENTION_LITERAL = "SLICE15_INTERVENTION"
LOCAL_CYCLE_BOUND = 8
REPLICATION_COUNT = 2
OUTCOME_NEW = "NEW_CAUSAL_DEGREE_OF_FREEDOM"
OUTCOME_NO_NEW = "NO_NEW_CAUSAL_DEGREE_OF_FREEDOM"
OUTCOME_FAIL = "FAIL"
FP_KEYS = (
    "sha256",
    "component_count",
    "layer_count",
    "sequence_length",
    "tensor_byte_count",
)


def load_slice14_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice14_accumulated_runtime_for_slice15", SLICE14_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 14 runtime: {SLICE14_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def cache_fingerprint(slice11, context, cache):
    complete = slice11.fp(context["kernel"], cache)
    return {key: complete[key] for key in FP_KEYS}


def same_cache(left, right):
    return all(left[key] == right[key] for key in FP_KEYS)


def token_evidence(context, token_ids):
    ids = [int(value) for value in token_ids]
    return {
        "token_ids": ids,
        "sha256": context["slice4"].token_fingerprint(ids),
        "count": len(ids),
    }


def first_token_divergence(a_ids, b_ids):
    limit = min(len(a_ids), len(b_ids))
    for index in range(limit):
        if a_ids[index] != b_ids[index]:
            return {
                "kind": "generated_token",
                "index": index,
                "A_token_id": int(a_ids[index]),
                "B_token_id": int(b_ids[index]),
            }
    if len(a_ids) != len(b_ids):
        return {
            "kind": "generated_token_sequence_length",
            "index": limit,
            "A_count": len(a_ids),
            "B_count": len(b_ids),
        }
    return None


def parent_collapse_evidence(slice11, context, parent, s_pre, s_post):
    operative = {
        name: cache_fingerprint(slice11, context, cache)
        for name, cache in sorted(parent.poles.items())
    }
    pre_matches = [name for name, value in operative.items() if same_cache(value, s_pre)]
    post_matches = [name for name, value in operative.items() if same_cache(value, s_post)]
    has_same_pair = bool(pre_matches) and bool(post_matches) and (
        set(pre_matches) != set(post_matches) or not same_cache(s_pre, s_post)
    )
    return {
        "operative_parent_structure": "StandingWave.poles",
        "parent_operative_poles": operative,
        "S_pre_operative_pole_matches": pre_matches,
        "S_post_operative_pole_matches": post_matches,
        "boundary_evidence_fields": ["pre_native", "post_native"],
        "boundary_evidence_counted_as_operative_poles": False,
        "parent_has_both_as_simultaneous_operative_poles": has_same_pair,
    }


def execute_intervention(slice11, context, child, selected_pole, replicate):
    other_pole = "B" if selected_pole == "A" else "A"
    intervention_ids = tuple(
        int(value)
        for value in context["slice7"].token_ids(
            context["tokenizer"], INTERVENTION_LITERAL
        )
    )
    selected_actual_before = cache_fingerprint(
        slice11, context, child.poles[selected_pole]
    )
    other_actual_before = cache_fingerprint(slice11, context, child.poles[other_pole])
    continuation = context["slice3"].clone_actual_kv(child.poles[selected_pole])
    continuation_pre = cache_fingerprint(slice11, context, continuation)
    traversal = slice11.traverse(
        context,
        continuation,
        child.shared_v,
        intervention_ids,
        f"SLICE15_REPLICATE_{replicate}_INTERVENTION_{selected_pole}",
        "fixed Slice 15 acceptance perturbation",
    )
    continuation_post = cache_fingerprint(slice11, context, continuation)
    selected_actual_after = cache_fingerprint(
        slice11, context, child.poles[selected_pole]
    )
    other_actual_after = cache_fingerprint(slice11, context, child.poles[other_pole])
    output = token_evidence(context, traversal["generated_token_ids"])
    return {
        "selected_pole": selected_pole,
        "other_pole": other_pole,
        "complete_pre_KV": continuation_pre,
        "complete_post_KV": continuation_post,
        "output": output,
        "selected_actual_before": selected_actual_before,
        "selected_actual_after": selected_actual_after,
        "other_actual_before": other_actual_before,
        "other_actual_after": other_actual_after,
        "intervention": token_evidence(context, intervention_ids),
        "actual_first_forward_perturbation_ids": [
            int(value)
            for value in traversal["actual_first_forward_perturbation_ids"]
        ],
        "actual_first_forward_v_prefix": [
            int(value) for value in traversal["actual_first_forward_v_prefix"]
        ],
        "model_boundary_exact": traversal[
            "exact_perturbation_identity_at_model_boundary"
        ],
        "continuation_storage_separated_from_selected_actual": context[
            "slice3"
        ].cache_tensor_storage_separated(
            continuation, child.poles[selected_pole]
        ),
        "selected_actual_unchanged": same_cache(
            selected_actual_before, selected_actual_after
        ),
        "other_actual_unchanged": same_cache(other_actual_before, other_actual_after),
    }


def execute_replicate(slice14, slice11, context, replicate):
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(
        context, parent, LOCAL_CYCLE_BOUND, evaluate_gate=True
    )
    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise RuntimeError("The deterministic parent trajectory was not unresolved")

    boundary = parent.latest_boundary
    # Two clean constitutions are created from the same native parent boundary.
    child_for_a = slice14.constitute_boundary_preserving_descent(
        slice11, context, parent
    )
    child_for_b = slice14.constitute_boundary_preserving_descent(
        slice11, context, parent
    )
    s_pre = cache_fingerprint(slice11, context, boundary["pre_native"])
    s_post = cache_fingerprint(slice11, context, boundary["post_native"])
    p_a = cache_fingerprint(slice11, context, child_for_a.poles["A"])
    p_b = cache_fingerprint(slice11, context, child_for_a.poles["B"])
    clean_b_p_a = cache_fingerprint(slice11, context, child_for_b.poles["A"])
    clean_b_p_b = cache_fingerprint(slice11, context, child_for_b.poles["B"])

    intervention_a = execute_intervention(
        slice11, context, child_for_a, "A", replicate
    )
    intervention_b = execute_intervention(
        slice11, context, child_for_b, "B", replicate
    )
    a_ids = intervention_a["output"]["token_ids"]
    b_ids = intervention_b["output"]["token_ids"]
    token_divergence = first_token_divergence(a_ids, b_ids)
    post_kv_divergence = not same_cache(
        intervention_a["complete_post_KV"], intervention_b["complete_post_KV"]
    )
    earliest = token_divergence
    if earliest is None and post_kv_divergence:
        earliest = {
            "kind": "complete_post_KV",
            "A_sha256": intervention_a["complete_post_KV"]["sha256"],
            "B_sha256": intervention_b["complete_post_KV"]["sha256"],
        }

    v_fingerprint = context["slice3"].fingerprint_v(parent.shared_v)
    v_evidence = {
        "literal": parent.shared_v.literal,
        "token_ids": [int(value) for value in v_fingerprint["token_ids"]],
        "same_runtime_object_parent_A_and_B": (
            parent.shared_v is child_for_a.shared_v
            and parent.shared_v is child_for_b.shared_v
        ),
    }
    parent_collapse = parent_collapse_evidence(
        slice11, context, parent, s_pre, s_post
    )
    checks = {
        "P_A_EQUALS_S_pre": same_cache(p_a, s_pre),
        "P_B_EQUALS_S_post": same_cache(p_b, s_post),
        "MATCHED_CLEAN_CONSTITUTIONS": (
            same_cache(p_a, clean_b_p_a) and same_cache(p_b, clean_b_p_b)
        ),
        "POLES_SIMULTANEOUS_AND_STORAGE_SEPARATED": context[
            "slice3"
        ].cache_tensor_storage_separated(
            child_for_a.poles["A"], child_for_a.poles["B"]
        ),
        "IDENTICAL_INTERVENTION": (
            intervention_a["intervention"] == intervention_b["intervention"]
            and intervention_a["intervention"]["token_ids"]
            == list(intervention_a["actual_first_forward_perturbation_ids"])
            and intervention_b["intervention"]["token_ids"]
            == list(intervention_b["actual_first_forward_perturbation_ids"])
        ),
        "MODEL_BOUNDARY_EXACT_A": intervention_a["model_boundary_exact"],
        "MODEL_BOUNDARY_EXACT_B": intervention_b["model_boundary_exact"],
        "V_IDENTITY": (
            v_evidence["literal"] == V_LITERAL
            and v_evidence["same_runtime_object_parent_A_and_B"]
            and intervention_a["actual_first_forward_v_prefix"]
            == v_evidence["token_ids"]
            and intervention_b["actual_first_forward_v_prefix"]
            == v_evidence["token_ids"]
        ),
        "P_A_CAUSALLY_CONTINUABLE": (
            intervention_a["complete_pre_KV"] == p_a
            and intervention_a[
                "continuation_storage_separated_from_selected_actual"
            ]
        ),
        "P_B_CAUSALLY_CONTINUABLE": (
            intervention_b["complete_pre_KV"] == p_b
            and intervention_b[
                "continuation_storage_separated_from_selected_actual"
            ]
        ),
        "P_A_DID_NOT_MUTATE_P_B": intervention_a["other_actual_unchanged"],
        "P_B_DID_NOT_MUTATE_P_A": intervention_b["other_actual_unchanged"],
        "ACTUAL_P_A_PRESERVED": (
            intervention_a["selected_actual_unchanged"]
            and intervention_b["other_actual_unchanged"]
        ),
        "ACTUAL_P_B_PRESERVED": (
            intervention_a["other_actual_unchanged"]
            and intervention_b["selected_actual_unchanged"]
        ),
        "DOWNSTREAM_DIVERGENCE": (
            token_divergence is not None or post_kv_divergence
        ),
        "PARENT_LACKS_SAME_INDEPENDENT_DEGREE": not parent_collapse[
            "parent_has_both_as_simultaneous_operative_poles"
        ],
    }
    return {
        "replicate": replicate,
        "selected_parent_boundary": {
            "producer_instance_id": boundary["producer_instance_id"],
            "cycle": boundary["cycle"],
            "ordinal": boundary["ordinal"],
            "pole": boundary["pole"],
        },
        "S_pre": s_pre,
        "P_A": p_a,
        "S_post": s_post,
        "P_B": p_b,
        "V": v_evidence,
        "intervention": token_evidence(
            context, intervention_a["intervention"]["token_ids"]
        ),
        "A": intervention_a,
        "B": intervention_b,
        "token_divergence": token_divergence,
        "post_KV_divergence": post_kv_divergence,
        "earliest_divergent_boundary": earliest,
        "parent_collapse_test": parent_collapse,
        "checks": checks,
        "model_weight_sha256": context["weight_hash"],
    }


def deterministic_projection(replicate):
    return {key: value for key, value in replicate.items() if key != "replicate"}


def reason_not_new(evidence):
    checks = evidence["checks"]
    if not checks["PARENT_LACKS_SAME_INDEPENDENT_DEGREE"]:
        return "parent already exposes S_pre and S_post as simultaneous operative poles"
    if not checks["POLES_SIMULTANEOUS_AND_STORAGE_SEPARATED"]:
        return "descended pole separation is not sovereign storage separation"
    if not (
        checks["P_A_CAUSALLY_CONTINUABLE"]
        and checks["P_B_CAUSALLY_CONTINUABLE"]
        and checks["P_A_DID_NOT_MUTATE_P_B"]
        and checks["P_B_DID_NOT_MUTATE_P_A"]
    ):
        return "descended poles do not expose independent causal intervention"
    if not checks["DOWNSTREAM_DIVERGENCE"]:
        return "identical intervention does not change downstream causal possibilities"
    return "parent and descended realizations expose the same causal structure"


def make_receipt(result):
    if result["outcome"] == OUTCOME_FAIL:
        return "\n".join(
            [
                "OUTCOME",
                OUTCOME_FAIL,
                "",
                "EVIDENCE",
                "",
                "failed_requirement:",
                result["failed_requirements"][0],
                "",
                "causal_boundary:",
                result["failed_requirements"][0],
                "",
                "substrate_evidence:",
                json.dumps(result["checks"], sort_keys=True),
                "",
            ]
        )
    evidence = result["replicates"][0]
    if result["outcome"] == OUTCOME_NO_NEW:
        return "\n".join(
            [
                "OUTCOME",
                OUTCOME_NO_NEW,
                "",
                "EVIDENCE",
                "",
                "candidate_d:",
                "independent causal accessibility of both sides of one unresolved boundary",
                "",
                "reason_not_new:",
                reason_not_new(evidence),
                "",
                "DETERMINISTIC_REPLICATION:",
                "PASS",
                "",
            ]
        )
    return "\n".join(
        [
            "OUTCOME",
            OUTCOME_NEW,
            "",
            "EVIDENCE",
            "",
            "candidate_d:",
            "independent causal accessibility of both sides of one unresolved boundary",
            "",
            "S_pre:",
            json.dumps(evidence["S_pre"], sort_keys=True),
            "",
            "P_A:",
            json.dumps(evidence["P_A"], sort_keys=True),
            "",
            "S_post:",
            json.dumps(evidence["S_post"], sort_keys=True),
            "",
            "P_B:",
            json.dumps(evidence["P_B"], sort_keys=True),
            "",
            "P_A_EQUALS_S_pre:",
            "PASS",
            "",
            "P_B_EQUALS_S_post:",
            "PASS",
            "",
            "INTERVENTION:",
            INTERVENTION_LITERAL,
            "",
            "INTERVENTION_TOKEN_EVIDENCE:",
            json.dumps(evidence["intervention"], sort_keys=True),
            "",
            "V:",
            json.dumps(evidence["V"], sort_keys=True),
            "",
            "A_pre:",
            json.dumps(evidence["A"]["complete_pre_KV"], sort_keys=True),
            "",
            "A_post:",
            json.dumps(evidence["A"]["complete_post_KV"], sort_keys=True),
            "",
            "A_output:",
            json.dumps(evidence["A"]["output"], sort_keys=True),
            "",
            "B_pre:",
            json.dumps(evidence["B"]["complete_pre_KV"], sort_keys=True),
            "",
            "B_post:",
            json.dumps(evidence["B"]["complete_post_KV"], sort_keys=True),
            "",
            "B_output:",
            json.dumps(evidence["B"]["output"], sort_keys=True),
            "",
            "DOWNSTREAM_DIVERGENCE:",
            "PASS",
            "",
            "earliest_divergent_boundary:",
            json.dumps(evidence["earliest_divergent_boundary"], sort_keys=True),
            "",
            "P_A_DID_NOT_MUTATE_P_B:",
            "PASS",
            "",
            "P_B_DID_NOT_MUTATE_P_A:",
            "PASS",
            "",
            "parent_operative_poles:",
            json.dumps(
                evidence["parent_collapse_test"]["parent_operative_poles"],
                sort_keys=True,
            ),
            "",
            "parent_S_pre_operative_matches:",
            json.dumps(
                evidence["parent_collapse_test"]["S_pre_operative_pole_matches"]
            ),
            "",
            "parent_S_post_operative_matches:",
            json.dumps(
                evidence["parent_collapse_test"]["S_post_operative_pole_matches"]
            ),
            "",
            "PARENT_HAS_SAME_INDEPENDENT_DEGREE_OF_FREEDOM:",
            "NO",
            "",
            "DETERMINISTIC_REPLICATION:",
            "PASS",
            "",
        ]
    )


def execute_acceptance():
    slice14 = load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()
    replicates = [
        execute_replicate(slice14, slice11, context, number)
        for number in range(1, REPLICATION_COUNT + 1)
    ]
    checks = {
        f"REPLICATE_{rep['replicate']}_{name}": passed
        for rep in replicates
        for name, passed in rep["checks"].items()
    }
    deterministic = (
        deterministic_projection(replicates[0])
        == deterministic_projection(replicates[1])
    )
    checks["DETERMINISTIC_REPLICATION"] = deterministic

    experiment_requirements = (
        "P_A_EQUALS_S_pre",
        "P_B_EQUALS_S_post",
        "MATCHED_CLEAN_CONSTITUTIONS",
        "IDENTICAL_INTERVENTION",
        "MODEL_BOUNDARY_EXACT_A",
        "MODEL_BOUNDARY_EXACT_B",
        "V_IDENTITY",
    )
    failed_requirements = [
        f"REPLICATE_{rep['replicate']}_{name}"
        for rep in replicates
        for name in experiment_requirements
        if not rep["checks"][name]
    ]
    if not deterministic:
        failed_requirements.append("DETERMINISTIC_REPLICATION")

    positive_criteria = (
        "P_A_EQUALS_S_pre",
        "P_B_EQUALS_S_post",
        "POLES_SIMULTANEOUS_AND_STORAGE_SEPARATED",
        "P_A_CAUSALLY_CONTINUABLE",
        "P_B_CAUSALLY_CONTINUABLE",
        "P_A_DID_NOT_MUTATE_P_B",
        "P_B_DID_NOT_MUTATE_P_A",
        "DOWNSTREAM_DIVERGENCE",
        "PARENT_LACKS_SAME_INDEPENDENT_DEGREE",
    )
    all_positive = all(
        rep["checks"][name]
        for rep in replicates
        for name in positive_criteria
    )
    if failed_requirements:
        outcome = OUTCOME_FAIL
    elif all_positive:
        outcome = OUTCOME_NEW
    else:
        outcome = OUTCOME_NO_NEW

    result = {
        "schema": "SLICE15_CAUSAL_RESOLUTION_V1",
        "V": V_LITERAL,
        "seed": SEED_LITERAL,
        "intervention": INTERVENTION_LITERAL,
        "local_cycle_bound": LOCAL_CYCLE_BOUND,
        "replication_count": REPLICATION_COUNT,
        "replicates": replicates,
        "checks": checks,
        "failed_requirements": failed_requirements,
        "deterministic_projection_sha256": stable_sha256(
            b"SLICE15_CAUSAL_RESOLUTION_PROJECTION_V1",
            deterministic_projection(replicates[0]),
        ),
        "outcome": outcome,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "causal_resolution.json").write_text(
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
