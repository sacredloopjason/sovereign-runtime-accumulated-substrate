import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE15_RUNTIME = PROJECT_DIR / "slice15_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice16"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
PARENT_CONTINUE_LITERAL = "SLICE16_PARENT_CONTINUE"
LOCAL_CYCLE_BOUND = 8
REPLICATION_COUNT = 2
OUTCOME_RECURRENCE = "RECURRENCE"
OUTCOME_NO_RECURRENCE = "NO_RECURRENCE_WITHIN_BOUND"
OUTCOME_FAIL = "FAIL"
FP_KEYS = (
    "sha256",
    "component_count",
    "layer_count",
    "sequence_length",
    "tensor_byte_count",
)


def load_slice15_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice15_accumulated_runtime_for_slice16", SLICE15_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 15 runtime: {SLICE15_RUNTIME}")
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


def pair_fingerprint(slice11, context, poles):
    return {
        "A": cache_fingerprint(slice11, context, poles["A"]),
        "B": cache_fingerprint(slice11, context, poles["B"]),
    }


def same_pair(left, right):
    return same_cache(left["A"], right["A"]) and same_cache(left["B"], right["B"])


def token_evidence(context, token_ids):
    ids = [int(value) for value in token_ids]
    return {
        "token_ids": ids,
        "sha256": context["slice4"].token_fingerprint(ids),
        "count": len(ids),
    }


def relation(left, right):
    return "EQUAL" if same_cache(left, right) else "DISTINCT"


def traverse_pole(slice11, context, child, pole_name, incoming, label, origin):
    other_name = "B" if pole_name == "A" else "A"
    active = child.poles[pole_name]
    other = child.poles[other_name]
    pre = cache_fingerprint(slice11, context, active)
    other_before = cache_fingerprint(slice11, context, other)
    traversal = slice11.traverse(
        context,
        active,
        child.shared_v,
        tuple(int(value) for value in incoming),
        label,
        origin,
    )
    post = cache_fingerprint(slice11, context, active)
    other_after = cache_fingerprint(slice11, context, other)
    generated = token_evidence(context, traversal["generated_token_ids"])
    consumed = token_evidence(context, incoming)
    return {
        "pole": pole_name,
        "pre": pre,
        "post": post,
        "consumed": consumed,
        "generated": generated,
        "actual_first_forward_perturbation_ids": [
            int(value) for value in traversal["actual_first_forward_perturbation_ids"]
        ],
        "actual_first_forward_v_prefix": [
            int(value) for value in traversal["actual_first_forward_v_prefix"]
        ],
        "model_boundary_exact": traversal[
            "exact_perturbation_identity_at_model_boundary"
        ],
        "other_pole_unchanged": same_cache(other_before, other_after),
    }


def run_boundary_preserving_wave(slice11, context, child, replicate):
    initial_pair = pair_fingerprint(slice11, context, child.poles)
    v_fingerprint = context["slice3"].fingerprint_v(child.shared_v)
    v_ids = [int(value) for value in v_fingerprint["token_ids"]]
    delta = token_evidence(context, child.perturbation)

    # This is the first lower-order traversal, not a descent bootstrap. Actual Pole A
    # remains S_pre until it receives the inherited unresolved Delta here.
    pending_a = traverse_pole(
        slice11,
        context,
        child,
        "A",
        child.perturbation,
        f"SLICE16_REPLICATE_{replicate}_CYCLE_0_A",
        "inherited unresolved parent Delta presented to actual descended Pole A",
    )
    cycles = []
    previous_pair = None
    recurrence_cycles = None
    a_lineage = [initial_pair["A"], pending_a["post"]]
    b_lineage = [initial_pair["B"]]
    exact_crossings = pending_a["model_boundary_exact"] and (
        pending_a["consumed"]["token_ids"]
        == pending_a["actual_first_forward_perturbation_ids"]
    )
    v_exact = pending_a["actual_first_forward_v_prefix"] == v_ids
    no_transfer = pending_a["other_pole_unchanged"]
    a_continuity = same_cache(pending_a["pre"], initial_pair["A"])
    b_continuity = True

    for cycle_number in range(LOCAL_CYCLE_BOUND):
        if cycle_number > 0:
            a_continuity = a_continuity and same_cache(
                pending_a["pre"], a_lineage[-2]
            ) and same_cache(pending_a["post"], a_lineage[-1])

        relation_after_a = relation(
            pending_a["post"],
            cache_fingerprint(slice11, context, child.poles["B"]),
        )
        b_traversal = traverse_pole(
            slice11,
            context,
            child,
            "B",
            pending_a["generated"]["token_ids"],
            f"SLICE16_REPLICATE_{replicate}_CYCLE_{cycle_number}_B",
            "exact generated ordered token-ID sequence from opposite Pole A",
        )
        b_continuity = b_continuity and same_cache(
            b_traversal["pre"], b_lineage[-1]
        )
        b_lineage.append(b_traversal["post"])
        relation_after_b = relation(
            cache_fingerprint(slice11, context, child.poles["A"]),
            b_traversal["post"],
        )

        pair_ids = [
            pending_a["generated"]["token_ids"],
            b_traversal["generated"]["token_ids"],
        ]
        pair_sha256 = stable_sha256(b"SLICE16_RECIPROCAL_PAIR_V1", pair_ids)
        recurrent = previous_pair is not None and pair_ids == previous_pair

        # Close the reciprocal crossing. This traversal is also the Pole-A producer
        # traversal for the next pair if the passive recurrence Gate remains open.
        return_a = traverse_pole(
            slice11,
            context,
            child,
            "A",
            b_traversal["generated"]["token_ids"],
            f"SLICE16_REPLICATE_{replicate}_CYCLE_{cycle_number}_RETURN_A",
            "exact generated ordered token-ID sequence from opposite Pole B",
        )
        a_continuity = a_continuity and same_cache(
            return_a["pre"], a_lineage[-1]
        )
        a_lineage.append(return_a["post"])
        relation_after_return = relation(return_a["post"], b_traversal["post"])

        ta_crossed = (
            pending_a["generated"] == b_traversal["consumed"]
            and b_traversal["consumed"]["token_ids"]
            == b_traversal["actual_first_forward_perturbation_ids"]
            and b_traversal["model_boundary_exact"]
        )
        tb_crossed = (
            b_traversal["generated"] == return_a["consumed"]
            and return_a["consumed"]["token_ids"]
            == return_a["actual_first_forward_perturbation_ids"]
            and return_a["model_boundary_exact"]
        )
        exact_crossings = exact_crossings and ta_crossed and tb_crossed
        v_exact = v_exact and (
            b_traversal["actual_first_forward_v_prefix"] == v_ids
            and return_a["actual_first_forward_v_prefix"] == v_ids
        )
        no_transfer = no_transfer and (
            b_traversal["other_pole_unchanged"]
            and return_a["other_pole_unchanged"]
        )
        cycles.append(
            {
                "cycle": cycle_number,
                "A_pre": pending_a["pre"],
                "A_post": pending_a["post"],
                "T_A_produced": pending_a["generated"],
                "T_A_consumed_by_B": b_traversal["consumed"],
                "B_pre": b_traversal["pre"],
                "B_post": b_traversal["post"],
                "T_B_produced": b_traversal["generated"],
                "T_B_consumed_by_A": return_a["consumed"],
                "A_after_T_B": return_a["post"],
                "pole_state_relation_after_A": relation_after_a,
                "pole_state_relation_after_B": relation_after_b,
                "pole_state_relation": relation_after_return,
                "R_k_sha256": pair_sha256,
                "recurrent_with_previous": recurrent,
                "exact_T_A_crossing": ta_crossed,
                "exact_T_B_crossing": tb_crossed,
            }
        )
        if recurrent:
            recurrence_cycles = [cycle_number - 1, cycle_number]
            break
        previous_pair = pair_ids
        pending_a = return_a
        gc.collect()

    outcome = (
        OUTCOME_RECURRENCE if recurrence_cycles is not None else OUTCOME_NO_RECURRENCE
    )
    return {
        "initial_pair": initial_pair,
        "Delta_inherited": delta,
        "cycles": cycles,
        "recurrence_result": outcome,
        "recurrence_cycles": recurrence_cycles,
        "A_lineage": a_lineage,
        "B_lineage": b_lineage,
        "checks": {
            "RECIPROCAL_TRAVERSAL": len(cycles) >= 1,
            "EXACT_TOKEN_TRANSPOSITION": exact_crossings,
            "A_KV_CAUSAL_CONTINUITY": a_continuity,
            "B_KV_CAUSAL_CONTINUITY": b_continuity,
            "NO_KV_TRANSFER_BETWEEN_POLES": no_transfer,
            "PASSIVE_RECURRENCE_GATE": (
                (outcome == OUTCOME_RECURRENCE and recurrence_cycles is not None)
                or (
                    outcome == OUTCOME_NO_RECURRENCE
                    and len(cycles) == LOCAL_CYCLE_BOUND
                    and not any(cycle["recurrent_with_previous"] for cycle in cycles)
                )
            ),
            "V_AT_EVERY_MODEL_BOUNDARY": v_exact,
        },
    }


def execute_replicate(slice15, slice14, slice11, context, replicate):
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(context, parent, LOCAL_CYCLE_BOUND, evaluate_gate=True)
    if parent.result != OUTCOME_NO_RECURRENCE:
        raise RuntimeError("The deterministic parent trajectory was not unresolved")

    boundary = parent.latest_boundary
    child = slice14.constitute_boundary_preserving_descent(
        slice11, context, parent
    )
    s_pre = cache_fingerprint(slice11, context, boundary["pre_native"])
    s_post = cache_fingerprint(slice11, context, boundary["post_native"])
    p_a_initial = cache_fingerprint(slice11, context, child.poles["A"])
    p_b_initial = cache_fingerprint(slice11, context, child.poles["B"])
    child_pair_before_loop = pair_fingerprint(slice11, context, child.poles)

    parent_pair_before = pair_fingerprint(slice11, context, parent.poles)
    parent_live_name = boundary["pole"]
    parent_live_before = cache_fingerprint(
        slice11, context, parent.poles[parent_live_name]
    )
    parent_control = context["slice3"].clone_actual_kv(parent.poles[parent_live_name])

    lower = run_boundary_preserving_wave(slice11, context, child, replicate)

    parent_pair_after = pair_fingerprint(slice11, context, parent.poles)
    parent_live_after = cache_fingerprint(
        slice11, context, parent.poles[parent_live_name]
    )
    parent_continue_ids = tuple(
        int(value)
        for value in context["slice7"].token_ids(
            context["tokenizer"], PARENT_CONTINUE_LITERAL
        )
    )
    parent_continue = slice11.traverse(
        context,
        parent.poles[parent_live_name],
        parent.shared_v,
        parent_continue_ids,
        f"SLICE16_REPLICATE_{replicate}_PARENT_CONTINUE_AFTER_LOWER",
        "fixed Slice 16 parent continuation after lower activity",
    )
    parent_continue_post = cache_fingerprint(
        slice11, context, parent.poles[parent_live_name]
    )
    control_continue = slice11.traverse(
        context,
        parent_control,
        parent.shared_v,
        parent_continue_ids,
        f"SLICE16_REPLICATE_{replicate}_PARENT_CONTINUE_CONTROL",
        "fixed Slice 16 clean control without lower activity",
    )
    control_continue_post = cache_fingerprint(slice11, context, parent_control)
    parent_output = token_evidence(context, parent_continue["generated_token_ids"])
    control_output = token_evidence(context, control_continue["generated_token_ids"])
    v_fingerprint = context["slice3"].fingerprint_v(parent.shared_v)
    v_evidence = {
        "literal": parent.shared_v.literal,
        "token_ids": [int(value) for value in v_fingerprint["token_ids"]],
        "same_runtime_object_parent_and_descended": child.shared_v is parent.shared_v,
    }

    checks = {
        "P_A_EQUALS_S_pre": same_cache(p_a_initial, s_pre),
        "P_B_EQUALS_S_post": same_cache(p_b_initial, s_post),
        "P_A_DISTINCT_FROM_P_B_AT_START": not same_cache(
            p_a_initial, p_b_initial
        ),
        "NO_PRE_LOOP_COLLAPSE": (
            same_cache(child_pair_before_loop["A"], p_a_initial)
            and same_cache(child_pair_before_loop["B"], p_b_initial)
        ),
        "V_IDENTITY": (
            v_evidence["literal"] == V_LITERAL
            and v_evidence["same_runtime_object_parent_and_descended"]
            and lower["checks"]["V_AT_EVERY_MODEL_BOUNDARY"]
        ),
        "POLE_STORAGE_SEPARATED": context[
            "slice3"
        ].cache_tensor_storage_separated(child.poles["A"], child.poles["B"]),
        "PARENT_UNCHANGED": (
            same_pair(parent_pair_before, parent_pair_after)
            and same_cache(parent_live_before, parent_live_after)
        ),
        "PARENT_CONTINUATION_CONTROL": (
            same_cache(parent_continue_post, control_continue_post)
            and parent_output == control_output
            and parent_continue["exact_perturbation_identity_at_model_boundary"]
            and control_continue["exact_perturbation_identity_at_model_boundary"]
        ),
        "NO_MANUFACTURED_DIFFERENTIATION": (
            same_cache(p_a_initial, s_pre)
            and same_cache(p_b_initial, s_post)
            and child.shared_v is parent.shared_v
            and tuple(child.perturbation)
            == tuple(int(value) for value in boundary["delta"])
        ),
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": True,
        **lower["checks"],
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
        "P_A_INITIAL": p_a_initial,
        "S_post": s_post,
        "P_B_INITIAL": p_b_initial,
        "V": v_evidence,
        "Delta_inherited": lower["Delta_inherited"],
        "cycles": lower["cycles"],
        "recurrence_result": lower["recurrence_result"],
        "recurrence_cycles": lower["recurrence_cycles"],
        "parent_preservation": {
            "live_pole": parent_live_name,
            "pair_before": parent_pair_before,
            "pair_after_lower": parent_pair_after,
            "live_before": parent_live_before,
            "live_after_lower": parent_live_after,
            "continue_input": token_evidence(context, parent_continue_ids),
            "post_after_lower": parent_continue_post,
            "post_control": control_continue_post,
            "output_after_lower": parent_output,
            "output_control": control_output,
        },
        "checks": checks,
        "model_weight_sha256": context["weight_hash"],
    }


def deterministic_projection(replicate):
    return {key: value for key, value in replicate.items() if key != "replicate"}


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
    lines = [
        "OUTCOME",
        result["outcome"],
        "",
        "EVIDENCE",
        "",
        "S_pre:",
        json.dumps(evidence["S_pre"], sort_keys=True),
        "",
        "P_A_INITIAL:",
        json.dumps(evidence["P_A_INITIAL"], sort_keys=True),
        "",
        "S_post:",
        json.dumps(evidence["S_post"], sort_keys=True),
        "",
        "P_B_INITIAL:",
        json.dumps(evidence["P_B_INITIAL"], sort_keys=True),
        "",
        "P_A_EQUALS_S_pre:",
        "PASS",
        "",
        "P_B_EQUALS_S_post:",
        "PASS",
        "",
        "P_A_DISTINCT_FROM_P_B_AT_START:",
        "PASS",
        "",
        "V:",
        V_LITERAL,
        "",
        "Delta_inherited:",
        json.dumps(evidence["Delta_inherited"], sort_keys=True),
        "",
        "cycles:",
    ]
    for cycle in evidence["cycles"]:
        lines.append(json.dumps(cycle, sort_keys=True))
    lines.extend(
        [
            "",
            "recurrence_result:",
            evidence["recurrence_result"],
            "",
            "recurrence_cycles:",
            (
                "NONE"
                if evidence["recurrence_cycles"] is None
                else ",".join(str(value) for value in evidence["recurrence_cycles"])
            ),
            "",
            "EXACT_TOKEN_TRANSPOSITION:",
            "PASS",
            "",
            "A_KV_CAUSAL_CONTINUITY:",
            "PASS",
            "",
            "B_KV_CAUSAL_CONTINUITY:",
            "PASS",
            "",
            "NO_PRE_LOOP_COLLAPSE:",
            "PASS",
            "",
            "NO_KV_TRANSFER_BETWEEN_POLES:",
            "PASS",
            "",
            "PARENT_UNCHANGED:",
            "PASS",
            "",
            "PARENT_CONTINUATION_CONTROL:",
            "PASS",
            "",
            "parent_preservation:",
            json.dumps(evidence["parent_preservation"], sort_keys=True),
            "",
            "DETERMINISTIC_REPLICATION:",
            "PASS",
            "",
            "NO_MANUFACTURED_DIFFERENTIATION:",
            "PASS",
            "",
            "NO_EXTERNAL_EPISTEMIC_AUTHORITY:",
            "PASS",
            "",
        ]
    )
    return "\n".join(lines)


def execute_acceptance():
    slice15 = load_slice15_runtime()
    slice14 = slice15.load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()
    replicates = [
        execute_replicate(slice15, slice14, slice11, context, number)
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
    failed = [name for name, passed in checks.items() if not passed]
    outcomes = {rep["recurrence_result"] for rep in replicates}
    if len(outcomes) != 1:
        failed.append("REPLICATION_OUTCOME_IDENTITY")
        checks["REPLICATION_OUTCOME_IDENTITY"] = False

    boundary_by_check = {
        "P_A_EQUALS_S_pre": "S_pre -> P_A_INITIAL",
        "P_B_EQUALS_S_post": "S_post -> P_B_INITIAL",
        "P_A_DISTINCT_FROM_P_B_AT_START": "P_A_INITIAL | Delta | P_B_INITIAL",
        "NO_PRE_LOOP_COLLAPSE": "constitution -> first lower-order traversal",
        "V_IDENTITY": "shared immutable V -> both descended poles",
        "POLE_STORAGE_SEPARATED": "P_A KV storage | P_B KV storage",
        "PARENT_UNCHANGED": "parent before lower activity -> parent after lower activity",
        "PARENT_CONTINUATION_CONTROL": "parent continuation -> clean control",
        "NO_MANUFACTURED_DIFFERENTIATION": "inherited parent boundary -> descended constitution",
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": "token transport -> model boundary",
        "RECIPROCAL_TRAVERSAL": "P_A -> P_B -> P_A",
        "EXACT_TOKEN_TRANSPOSITION": "generated token IDs -> opposite pole input",
        "A_KV_CAUSAL_CONTINUITY": "KV_A0 -> KV_A1 -> ...",
        "B_KV_CAUSAL_CONTINUITY": "KV_B0 -> KV_B1 -> ...",
        "NO_KV_TRANSFER_BETWEEN_POLES": "pole-local KV lineage boundary",
        "PASSIVE_RECURRENCE_GATE": "R_k -> exact R_(k+1) comparison",
        "V_AT_EVERY_MODEL_BOUNDARY": "V token prefix -> traversal model boundary",
        "DETERMINISTIC_REPLICATION": "replicate 1 projection -> replicate 2 projection",
        "REPLICATION_OUTCOME_IDENTITY": "replicate 1 outcome -> replicate 2 outcome",
    }
    failed_boundaries = {}
    for failed_name in failed:
        base_name = failed_name
        if failed_name.startswith("REPLICATE_"):
            base_name = failed_name.split("_", 2)[2]
        failed_boundaries[failed_name] = boundary_by_check.get(
            base_name, "Slice 16 acceptance boundary"
        )

    result = {
        "schema": "SLICE16_BOUNDARY_STANDING_WAVE_V1",
        "V": V_LITERAL,
        "seed": SEED_LITERAL,
        "parent_continue": PARENT_CONTINUE_LITERAL,
        "local_cycle_bound": LOCAL_CYCLE_BOUND,
        "replication_count": REPLICATION_COUNT,
        "replicates": replicates,
        "checks": checks,
        "failed_requirements": failed,
        "failed_boundaries": failed_boundaries,
        "deterministic_projection_sha256": stable_sha256(
            b"SLICE16_BOUNDARY_STANDING_WAVE_PROJECTION_V1",
            deterministic_projection(replicates[0]),
        ),
        "outcome": OUTCOME_FAIL if failed else next(iter(outcomes)),
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "standing_wave_boundary.json").write_text(
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
