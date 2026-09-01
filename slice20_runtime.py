import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
SLICE18_RUNTIME = ROOT / "slice18_runtime.py"
EVIDENCE_DIR = ROOT / "evidence" / "slice20"
SEED_LITERAL = "SLICE6_SEED"
EXPECTED_F_IDENTITY = "38c9de28173df67de170a7f36f4d895ad2dfea251b8460734a0cb4cb4a3f024f"
PARENT_CYCLE_BOUND = 16
REPLICATION_COUNT = 2
OUTCOME_CLEARED = "PARENT_GATE_CLEARED_BY_UPWARD_FACT"
OUTCOME_NONATTRIBUTABLE = "PARENT_STABILIZATION_OBSERVED_BUT_NOT_ATTRIBUTABLE_TO_F"
OUTCOME_CHANGED_UNRESOLVED = "UPWARD_FACT_CHANGED_PARENT_BUT_GATE_REMAINS_UNRESOLVED"
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


def pair_identity(pair):
    return stable_hash(b"SLICE20_PARENT_PAIR_FINGERPRINT_V1", pair)


def clone_parent_branch(context, parent):
    return SimpleNamespace(
        poles={
            "A": context["slice3"].clone_actual_kv(parent.poles["A"]),
            "B": context["slice3"].clone_actual_kv(parent.poles["B"]),
        },
        shared_v=parent.shared_v,
    )


def token_record(slice16, context, ids):
    return slice16.token_evidence(context, tuple(int(value) for value in ids))


def apply_parent_input(slice16, slice11, context, branch, pole_name, ids, replicate, condition):
    traversal = slice16.traverse_pole(
        slice11,
        context,
        branch,
        pole_name,
        ids,
        f"SLICE20_REPLICATE_{replicate}_{condition}_UPWARD_BOUNDARY",
        "matched immediate parent input boundary",
    )
    return {
        "pole": pole_name,
        "input": traversal["consumed"],
        "pre": traversal["pre"],
        "post": traversal["post"],
        "generated": traversal["generated"],
        "actual_first_forward_perturbation_ids": traversal[
            "actual_first_forward_perturbation_ids"
        ],
        "actual_first_forward_v_prefix": traversal["actual_first_forward_v_prefix"],
        "model_boundary_exact": traversal["model_boundary_exact"],
        "other_pole_unchanged": traversal["other_pole_unchanged"],
    }


def detect_closures(relations):
    closures = []
    for later in range(len(relations)):
        for earlier in range(later):
            if relations[later]["token_ids"] == relations[earlier]["token_ids"]:
                closures.append(
                    {
                        "earlier_cycle": earlier,
                        "later_cycle": later,
                        "period": later - earlier,
                        "R_earlier_sha256": relations[earlier]["sha256"],
                        "R_later_sha256": relations[later]["sha256"],
                        "exact_ordered_token_ID_identity": True,
                    }
                )
    if not closures:
        return None, []
    minimal = min(item["period"] for item in closures)
    return minimal, [item for item in closures if item["period"] == minimal]


def continue_parent(slice16, slice11, context, branch, initial_incoming, initial_phase, replicate, condition):
    incoming = tuple(int(value) for value in initial_incoming)
    phase = initial_phase
    previous_post = {
        "A": slice16.cache_fingerprint(slice11, context, branch.poles["A"]),
        "B": slice16.cache_fingerprint(slice11, context, branch.poles["B"]),
    }
    relations = []
    cycles = []
    a_continuity = True
    b_continuity = True
    exact_transposition = True
    forward_only = True
    v_ids = [int(value) for value in context["slice3"].fingerprint_v(branch.shared_v)["token_ids"]]
    for cycle_number in range(PARENT_CYCLE_BOUND):
        traversals = []
        produced = {}
        order = (phase, "B" if phase == "A" else "A")
        for ordinal, pole_name in enumerate(order):
            supplied = tuple(incoming)
            traversal = slice16.traverse_pole(
                slice11,
                context,
                branch,
                pole_name,
                supplied,
                f"SLICE20_REPLICATE_{replicate}_{condition}_CYCLE_{cycle_number}_{ordinal}_{pole_name}",
                "exact generated ordered token-ID sequence from opposite parent pole",
            )
            continuity = slice16.same_cache(traversal["pre"], previous_post[pole_name])
            if pole_name == "A":
                a_continuity = a_continuity and continuity
            else:
                b_continuity = b_continuity and continuity
            exact_transposition = exact_transposition and (
                traversal["consumed"]["token_ids"]
                == traversal["actual_first_forward_perturbation_ids"]
                and traversal["model_boundary_exact"]
                and traversal["other_pole_unchanged"]
                and traversal["actual_first_forward_v_prefix"] == v_ids
            )
            forward_only = forward_only and (
                traversal["post"]["sequence_length"] > traversal["pre"]["sequence_length"]
            )
            previous_post[pole_name] = traversal["post"]
            produced[pole_name] = traversal["generated"]["token_ids"]
            traversals.append(
                {
                    "pole": pole_name,
                    "pre": traversal["pre"],
                    "consumed": traversal["consumed"],
                    "post": traversal["post"],
                    "generated": traversal["generated"],
                    "model_boundary_exact": traversal["model_boundary_exact"],
                    "other_pole_unchanged": traversal["other_pole_unchanged"],
                }
            )
            incoming = tuple(traversal["generated"]["token_ids"])
        relation_ids = [produced["A"], produced["B"]]
        relation = {
            "cycle": cycle_number,
            "token_ids": relation_ids,
            "sha256": stable_hash(b"SLICE20_PARENT_RELATION_V1", relation_ids),
        }
        relations.append(relation)
        cycles.append(
            {
                "cycle": cycle_number,
                "R_parent": relation,
                "traversals": traversals,
                "parent_poles_after_cycle": slice16.pair_fingerprint(
                    slice11, context, branch.poles
                ),
            }
        )
        gc.collect()
    minimal_period, closures = detect_closures(relations)
    return {
        "result": "STABILIZATION" if minimal_period is not None else "NO_STABILIZATION_WITHIN_BOUND",
        "minimal_period": minimal_period,
        "closure_evidence": closures,
        "relations": relations,
        "cycles": cycles,
        "final_poles": slice16.pair_fingerprint(slice11, context, branch.poles),
        "checks": {
            "KV_HISTORY_FORWARD_ONLY": forward_only,
            "A_KV_CAUSAL_CONTINUITY": a_continuity,
            "B_KV_CAUSAL_CONTINUITY": b_continuity,
            "EXACT_TOKEN_TRANSPOSITION": exact_transposition,
        },
    }


def execute_replicate(slice18, slice16, slice14, slice11, context, replicate):
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(context, parent, slice16.LOCAL_CYCLE_BOUND, evaluate_gate=True)
    if parent.result != slice16.OUTCOME_NO_RECURRENCE:
        raise RuntimeError("The deterministic parent trajectory was not unresolved")
    archived_parent_before = slice16.pair_fingerprint(slice11, context, parent.poles)
    boundary = parent.latest_boundary
    live_pole = boundary["pole"]

    child = slice14.constitute_boundary_preserving_descent(slice11, context, parent)
    slice18.CAPTURES.pop(replicate, None)
    lower = slice18.run_slice18_wave(slice16, slice11, context, child, replicate)
    fact = slice18.CAPTURES[replicate]["fact"]
    if fact is None:
        raise RuntimeError("Slice 18 did not produce the stabilized fact")
    fact_bytes_before = fact["bytes"]
    fact_identity_before = fact["sha256"]
    lower_poles_before = slice16.pair_fingerprint(slice11, context, child.poles)
    lower_orbit_before = stable_hash(b"SLICE20_LOWER_ORBIT_V1", lower["cycles"])

    fact_text = fact_bytes_before.decode("ascii")
    fact_ids = tuple(
        int(value)
        for value in context["tokenizer"](fact_text, add_special_tokens=False).input_ids
    )
    recovered = context["tokenizer"].decode(
        fact_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ).encode("ascii")
    ordinary_ids = tuple(
        int(value)
        for value in context["slice7"].token_ids(
            context["tokenizer"], slice16.PARENT_CONTINUE_LITERAL
        )
    )

    control_branch = clone_parent_branch(context, parent)
    fact_branch = clone_parent_branch(context, parent)
    parent_pre_control = slice16.pair_fingerprint(slice11, context, control_branch.poles)
    parent_pre_fact = slice16.pair_fingerprint(slice11, context, fact_branch.poles)
    pre_states_equal = slice16.same_pair(parent_pre_control, parent_pre_fact)
    branch_storage_separated = all(
        context["slice3"].cache_tensor_storage_separated(
            control_branch.poles[pole], fact_branch.poles[pole]
        )
        for pole in ("A", "B")
    )

    control_input = apply_parent_input(
        slice16, slice11, context, control_branch, live_pole, ordinary_ids, replicate, "CONTROL"
    )
    fact_input = apply_parent_input(
        slice16, slice11, context, fact_branch, live_pole, fact_ids, replicate, "FACT"
    )
    immediate_divergence = (
        not slice16.same_cache(control_input["post"], fact_input["post"])
        or control_input["generated"] != fact_input["generated"]
    )
    next_phase = "B" if live_pole == "A" else "A"
    control = continue_parent(
        slice16,
        slice11,
        context,
        control_branch,
        control_input["generated"]["token_ids"],
        next_phase,
        replicate,
        "CONTROL",
    )
    fact_condition = continue_parent(
        slice16,
        slice11,
        context,
        fact_branch,
        fact_input["generated"]["token_ids"],
        next_phase,
        replicate,
        "FACT",
    )

    archived_parent_after = slice16.pair_fingerprint(slice11, context, parent.poles)
    lower_poles_after = slice16.pair_fingerprint(slice11, context, child.poles)
    lower_orbit_after = stable_hash(b"SLICE20_LOWER_ORBIT_V1", lower["cycles"])
    v_after = context["slice3"].fingerprint_v(parent.shared_v)
    lossless = recovered == fact_bytes_before
    fact_participated = (
        fact_input["input"]["token_ids"] == list(fact_ids)
        and fact_input["actual_first_forward_perturbation_ids"] == list(fact_ids)
        and fact_input["model_boundary_exact"]
    )
    checks = {
        "F_IDENTITY": fact_identity_before == EXPECTED_F_IDENTITY,
        "LOSSLESS_F_PROPAGATION": lossless,
        "F_PARTICIPATED_IN_PARENT_INFERENCE": fact_participated,
        "PARENT_PRE_STATES_EQUAL": pre_states_equal,
        "PARENT_BRANCH_STORAGE_SEPARATED": branch_storage_separated,
        "F_CAUSAL_EFFECT": immediate_divergence,
        "CONTROL_INPUT_BOUNDARY_EXACT": (
            control_input["input"]["token_ids"] == list(ordinary_ids)
            and control_input["actual_first_forward_perturbation_ids"] == list(ordinary_ids)
            and control_input["model_boundary_exact"]
        ),
        "KV_HISTORY_FORWARD_ONLY": (
            control["checks"]["KV_HISTORY_FORWARD_ONLY"]
            and fact_condition["checks"]["KV_HISTORY_FORWARD_ONLY"]
        ),
        "A_KV_CAUSAL_CONTINUITY": (
            control["checks"]["A_KV_CAUSAL_CONTINUITY"]
            and fact_condition["checks"]["A_KV_CAUSAL_CONTINUITY"]
        ),
        "B_KV_CAUSAL_CONTINUITY": (
            control["checks"]["B_KV_CAUSAL_CONTINUITY"]
            and fact_condition["checks"]["B_KV_CAUSAL_CONTINUITY"]
        ),
        "EXACT_TOKEN_TRANSPOSITION": (
            control["checks"]["EXACT_TOKEN_TRANSPOSITION"]
            and fact_condition["checks"]["EXACT_TOKEN_TRANSPOSITION"]
        ),
        "LOWER_FACT_UNCHANGED": (
            fact["bytes"] == fact_bytes_before and fact["sha256"] == fact_identity_before
        ),
        "LOWER_POLES_UNCHANGED": (
            slice16.same_pair(lower_poles_before, lower_poles_after)
            and lower_orbit_before == lower_orbit_after
        ),
        "ARCHIVED_PARENT_UNCHANGED": slice16.same_pair(
            archived_parent_before, archived_parent_after
        ),
        "SHARED_V_IMMUTABLE": (
            parent.shared_v.literal == slice16.V_LITERAL
            and v_after == context["slice3"].fingerprint_v(child.shared_v)
            and control_branch.shared_v is parent.shared_v
            and fact_branch.shared_v is parent.shared_v
        ),
        "ONLY_F_DIFFERED_INTENTIONALLY": (
            pre_states_equal and branch_storage_separated and control_branch.shared_v is fact_branch.shared_v
        ),
    }
    return {
        "replicate": replicate,
        "F": {
            "identity": fact_identity_before,
            "bytes_length": len(fact_bytes_before),
            "token_representation": token_record(slice16, context, fact_ids),
            "recovered_bytes_sha256": hashlib.sha256(recovered).hexdigest(),
        },
        "parent_boundary": {
            "producer_instance_id": boundary["producer_instance_id"],
            "cycle": boundary["cycle"],
            "ordinal": boundary["ordinal"],
            "live_pole": live_pole,
        },
        "parent_pre_control": parent_pre_control,
        "parent_pre_F": parent_pre_fact,
        "parent_pre_control_identity": pair_identity(parent_pre_control),
        "parent_pre_F_identity": pair_identity(parent_pre_fact),
        "control_input": control_input,
        "fact_input": fact_input,
        "immediate_causal_divergence": immediate_divergence,
        "control": control,
        "fact_condition": fact_condition,
        "preservation": {
            "archived_parent_before": archived_parent_before,
            "archived_parent_after": archived_parent_after,
            "lower_poles_before": lower_poles_before,
            "lower_poles_after": lower_poles_after,
            "lower_orbit_before": lower_orbit_before,
            "lower_orbit_after": lower_orbit_after,
        },
        "checks": checks,
    }


def deterministic_projection(replicate):
    return {key: value for key, value in replicate.items() if key != "replicate"}


def execute_acceptance():
    slice18 = load_module(SLICE18_RUNTIME, "slice18_accumulated_runtime_for_slice20")
    slice16 = slice18.load_module(slice18.SLICE16_RUNTIME, "slice16_accumulated_runtime_for_slice20")
    slice15 = slice16.load_slice15_runtime()
    slice14 = slice15.load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()
    replicates = [
        execute_replicate(slice18, slice16, slice14, slice11, context, number)
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
    checks["CONTROL_OUTCOME_REPLICATED"] = (
        replicates[0]["control"]["result"] == replicates[1]["control"]["result"]
        and replicates[0]["control"]["minimal_period"]
        == replicates[1]["control"]["minimal_period"]
    )
    checks["FACT_OUTCOME_REPLICATED"] = (
        replicates[0]["fact_condition"]["result"]
        == replicates[1]["fact_condition"]["result"]
        and replicates[0]["fact_condition"]["minimal_period"]
        == replicates[1]["fact_condition"]["minimal_period"]
    )
    failed = [name for name, passed in checks.items() if not passed]
    control_stabilized = all(rep["control"]["minimal_period"] is not None for rep in replicates)
    fact_stabilized = all(
        rep["fact_condition"]["minimal_period"] is not None for rep in replicates
    )
    if failed:
        outcome = OUTCOME_FAIL
    elif fact_stabilized and not control_stabilized:
        outcome = OUTCOME_CLEARED
    elif fact_stabilized and control_stabilized:
        outcome = OUTCOME_NONATTRIBUTABLE
    else:
        outcome = OUTCOME_CHANGED_UNRESOLVED
    result = {
        "schema": "SLICE20_PARENT_GATE_CLEARANCE_V1",
        "seed": SEED_LITERAL,
        "parent_cycle_bound": PARENT_CYCLE_BOUND,
        "replication_count": REPLICATION_COUNT,
        "replicates": replicates,
        "checks": checks,
        "failed_requirements": failed,
        "outcome": outcome,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "parent_gate_clearance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, receipt


def result_evidence(trajectory):
    if trajectory["minimal_period"] is None:
        return "NO_STABILIZATION_WITHIN_BOUND"
    return json.dumps(
        {
            "result": "STABILIZATION",
            "minimal_period": trajectory["minimal_period"],
            "closure": trajectory["closure_evidence"],
        },
        separators=(",", ":"),
    )


def make_receipt(result):
    if result["outcome"] == OUTCOME_FAIL:
        failed = result["failed_requirements"][0] if result["failed_requirements"] else "controlled parent-Gate experiment"
        return "\n".join(
            [
                "OUTCOME",
                "FAIL",
                "",
                "EVIDENCE",
                "",
                "failed_requirement:",
                failed,
                "",
                "causal_boundary:",
                "matched parent pre-state -> upward F -> 16-cycle parent Gate",
                "",
                "substrate_evidence:",
                json.dumps(result["checks"], sort_keys=True),
                "",
            ]
        )
    rep = result["replicates"][0]
    if result["outcome"] == OUTCOME_CHANGED_UNRESOLVED:
        return "\n".join(
            [
                "OUTCOME",
                OUTCOME_CHANGED_UNRESOLVED,
                "",
                "EVIDENCE",
                "",
                "F_CAUSAL_EFFECT:",
                "PASS",
                "",
                "CONTROL_RESULT:",
                result_evidence(rep["control"]),
                "",
                "FACT_CONDITION_RESULT:",
                "NO_STABILIZATION_WITHIN_BOUND",
                "",
                "DETERMINISTIC_REPLICATION:",
                "PASS",
                "",
            ]
        )
    if result["outcome"] == OUTCOME_NONATTRIBUTABLE:
        return "\n".join(
            [
                "OUTCOME",
                OUTCOME_NONATTRIBUTABLE,
                "",
                "EVIDENCE",
                "",
                "CONTROL_RESULT:",
                result_evidence(rep["control"]),
                "",
                "FACT_CONDITION_RESULT:",
                result_evidence(rep["fact_condition"]),
                "",
                "reason_not_attributable:",
                "The matched no-F control also entered exact parent stabilization within the same bound.",
                "",
                "DETERMINISTIC_REPLICATION:",
                "PASS",
                "",
            ]
        )
    return "\n".join(
        [
            "OUTCOME",
            OUTCOME_CLEARED,
            "",
            "EVIDENCE",
            "",
            "F_IDENTITY:",
            rep["F"]["identity"],
            "",
            "PARENT_PRE_CONTROL:",
            rep["parent_pre_control_identity"],
            "",
            "PARENT_PRE_F:",
            rep["parent_pre_F_identity"],
            "",
            "PARENT_PRE_STATES_EQUAL:",
            "PASS",
            "",
            "CONTROL_RESULT:",
            "NO_STABILIZATION_WITHIN_BOUND",
            "",
            "FACT_CONDITION_RESULT:",
            "STABILIZATION",
            "",
            "FACT_CONDITION_MINIMAL_PERIOD:",
            str(rep["fact_condition"]["minimal_period"]),
            "",
            "FACT_CONDITION_CLOSURE:",
            json.dumps(rep["fact_condition"]["closure_evidence"], separators=(",", ":")),
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
            "LOWER_FACT_UNCHANGED:",
            "PASS",
            "",
            "LOWER_POLES_UNCHANGED:",
            "PASS",
            "",
            "ONLY_F_DIFFERED_INTENTIONALLY:",
            "PASS",
            "",
            "DETERMINISTIC_REPLICATION:",
            "PASS",
            "",
        ]
    )


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
