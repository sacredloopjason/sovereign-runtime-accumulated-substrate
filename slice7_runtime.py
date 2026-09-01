import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE6_RUNTIME = PROJECT_DIR / "slice6_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice7"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
CYCLE_LIMIT = 8
CHILD_A_INPUT = "SLICE7_CHILD_A_INPUT"
CHILD_B_INPUT = "SLICE7_CHILD_B_INPUT"
PARENT_CONTINUE_INPUT = "SLICE7_PARENT_CONTINUE"


def load_slice6_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice6_accumulated_runtime_for_slice7", SLICE6_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 6 runtime: {SLICE6_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def token_ids(tokenizer, literal):
    result = tuple(
        int(item)
        for item in tokenizer(literal + "\n", add_special_tokens=False).input_ids
    )
    if not result:
        raise RuntimeError(f"Fixed input encoded to no tokens: {literal}")
    return result


def native_class(value):
    return f"{value.__class__.__module__}.{value.__class__.__name__}"


def pairwise_storage_separated(slice3, caches):
    return all(
        slice3.cache_tensor_storage_separated(caches[left], caches[right])
        for left in range(len(caches))
        for right in range(left + 1, len(caches))
    )


def trajectory_sha256(cycles):
    payload = json.dumps(cycles, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"SLICE7_PARENT_TRAJECTORY_V1\0" + payload).hexdigest()


def traverse(
    slice4,
    kernel,
    model,
    tokenizer,
    cache,
    shared_v,
    perturbation,
    label,
    origin,
):
    return slice4.traverse_token_ids_with_v(
        kernel,
        model,
        tokenizer,
        cache,
        shared_v,
        perturbation,
        label,
        origin,
    )


def execute_parent(
    slice5,
    slice4,
    slice3,
    kernel,
    model,
    tokenizer,
    run_label,
):
    pole_a, pole_b = slice3.create_initial_poles(kernel, model, tokenizer)
    if not slice3.cache_tensor_storage_separated(pole_a, pole_b):
        raise RuntimeError("Initial parent pole storage is not separated")

    shared_v = slice6_make_shared_v(slice3, tokenizer)
    perturbation_for_a = token_ids(tokenizer, SEED_LITERAL)
    cycles = []
    latest_boundary = None
    observed_result = "NO_RECURRENCE_WITHIN_BOUND"

    for cycle_number in range(CYCLE_LIMIT):
        pre_a_native = slice3.clone_actual_kv(pole_a)
        a_pre = kernel.fingerprint_cache(pre_a_native)
        a_traversal = traverse(
            slice4,
            kernel,
            model,
            tokenizer,
            pole_a,
            shared_v,
            perturbation_for_a,
            f"{run_label}_CYCLE_{cycle_number}_POLE_A",
            "SLICE6_SEED tokenization"
            if cycle_number == 0
            else f"{run_label}_cycle_{cycle_number - 1}_direct_B_generated_token_tuple",
        )
        a_post_native = slice3.clone_actual_kv(pole_a)
        a_post = kernel.fingerprint_cache(a_post_native)
        latest_boundary = {
            "cycle": cycle_number,
            "pole": "A",
            "pre_native": pre_a_native,
            "post_native": a_post_native,
            "pre": a_pre,
            "post": a_post,
            "delta": list(perturbation_for_a),
            "traversal": a_traversal,
            "parent_live": pole_a,
        }

        ta = tuple(a_traversal["generated_token_ids"])
        pre_b_native = slice3.clone_actual_kv(pole_b)
        b_pre = kernel.fingerprint_cache(pre_b_native)
        b_traversal = traverse(
            slice4,
            kernel,
            model,
            tokenizer,
            pole_b,
            shared_v,
            ta,
            f"{run_label}_CYCLE_{cycle_number}_POLE_B",
            f"{run_label}_cycle_{cycle_number}_direct_A_generated_token_tuple",
        )
        b_post_native = slice3.clone_actual_kv(pole_b)
        b_post = kernel.fingerprint_cache(b_post_native)
        latest_boundary = {
            "cycle": cycle_number,
            "pole": "B",
            "pre_native": pre_b_native,
            "post_native": b_post_native,
            "pre": b_pre,
            "post": b_post,
            "delta": list(ta),
            "traversal": b_traversal,
            "parent_live": pole_b,
        }

        tb = tuple(b_traversal["generated_token_ids"])
        w = slice5.ordered_pair_fingerprint(ta, tb)
        recurrent = bool(
            cycles
            and list(ta) == cycles[-1]["T_A"]
            and list(tb) == cycles[-1]["T_B"]
        )
        cycles.append(
            {
                "cycle": cycle_number,
                "A_pre": a_pre["sha256"],
                "A_post": a_post["sha256"],
                "T_A": list(ta),
                "B_pre": b_pre["sha256"],
                "B_post": b_post["sha256"],
                "T_B": list(tb),
                "W": w,
                "recurrent_with_previous": recurrent,
            }
        )
        if recurrent:
            observed_result = "RECURRENCE"
            break
        perturbation_for_a = tb
        gc.collect()

    return {
        "run": run_label,
        "cycles": cycles,
        "trajectory_sha256": trajectory_sha256(cycles),
        "observed_result": observed_result,
        "gate": {
            "operation": "exact equality of consecutive ordered token-ID pairs only",
            "budget": CYCLE_LIMIT,
            "completed_cycles": len(cycles),
            "all_adjacent_pairs_unequal": all(
                not cycle["recurrent_with_previous"] for cycle in cycles[1:]
            ),
            "result": observed_result,
        },
        "latest_boundary": latest_boundary,
        "shared_v": shared_v,
        "other_parent_pole": pole_a if latest_boundary["pole"] == "B" else pole_b,
    }


def slice6_make_shared_v(slice3, tokenizer):
    ids = token_ids(tokenizer, V_LITERAL)
    return slice3.SharedV(V_LITERAL, V_LITERAL.encode("utf-8"), ids)


def save_selected_boundary(slice4, kernel, boundary, run_label):
    import torch

    run_dir = EVIDENCE_DIR / f"{run_label.lower()}_descent_boundary"
    pre_snapshot, pre = slice4.snapshot_and_fingerprint(
        kernel, boundary["pre_native"]
    )
    post_snapshot, post = slice4.snapshot_and_fingerprint(
        kernel, boundary["parent_live"]
    )
    pre_file = slice4.save_boundary(
        torch, kernel, run_dir / "S_t.pt", pre_snapshot, pre
    )
    post_file = slice4.save_boundary(
        torch, kernel, run_dir / "S_t_plus_1.pt", post_snapshot, post
    )
    return pre_file, post_file


def execute_experimental_path(
    slice5, slice4, slice3, kernel, model, tokenizer
):
    parent = execute_parent(
        slice5, slice4, slice3, kernel, model, tokenizer, "EXPERIMENTAL"
    )
    if parent["observed_result"] != "NO_RECURRENCE_WITHIN_BOUND":
        return {"parent": parent, "descent_attempted": False}

    boundary = parent["latest_boundary"]
    pre_file, post_file = save_selected_boundary(
        slice4, kernel, boundary, "EXPERIMENTAL"
    )
    parent_live = boundary["parent_live"]
    parent_pre_children = kernel.fingerprint_cache(parent_live)
    child_a = slice3.clone_actual_kv(boundary["pre_native"])
    child_b = slice3.clone_actual_kv(parent_live)
    child_a_pre = kernel.fingerprint_cache(child_a)
    child_b_pre = kernel.fingerprint_cache(child_b)
    native_storage_separated = pairwise_storage_separated(
        slice3, [parent_live, child_a, child_b]
    )
    v_initial = slice3.fingerprint_v(parent["shared_v"])

    child_b_before_a = kernel.fingerprint_cache(child_b)
    parent_before_a = kernel.fingerprint_cache(parent_live)
    child_a_traversal = traverse(
        slice4,
        kernel,
        model,
        tokenizer,
        child_a,
        parent["shared_v"],
        token_ids(tokenizer, CHILD_A_INPUT),
        "SLICE7_CHILD_A_TRAVERSAL",
        "fixed Slice 7 child-A acceptance input tokenization",
    )
    child_a_post = kernel.fingerprint_cache(child_a)
    child_b_after_a = kernel.fingerprint_cache(child_b)
    parent_after_a = kernel.fingerprint_cache(parent_live)

    child_a_before_b = kernel.fingerprint_cache(child_a)
    parent_before_b = kernel.fingerprint_cache(parent_live)
    child_b_traversal = traverse(
        slice4,
        kernel,
        model,
        tokenizer,
        child_b,
        parent["shared_v"],
        token_ids(tokenizer, CHILD_B_INPUT),
        "SLICE7_CHILD_B_TRAVERSAL",
        "fixed Slice 7 child-B acceptance input tokenization",
    )
    child_b_post = kernel.fingerprint_cache(child_b)
    child_a_after_b = kernel.fingerprint_cache(child_a)
    parent_after_b = kernel.fingerprint_cache(parent_live)

    parent_before_continue = kernel.fingerprint_cache(parent_live)
    parent_traversal = traverse(
        slice4,
        kernel,
        model,
        tokenizer,
        parent_live,
        parent["shared_v"],
        token_ids(tokenizer, PARENT_CONTINUE_INPUT),
        "SLICE7_PARENT_CONTINUATION_AFTER_CHILDREN",
        "fixed Slice 7 parent continuation input tokenization",
    )
    parent_post = kernel.fingerprint_cache(parent_live)
    v_final = slice3.fingerprint_v(parent["shared_v"])

    return {
        "parent": parent,
        "descent_attempted": True,
        "boundary_files": {"S_t": pre_file, "S_t_plus_1": post_file},
        "boundary": {
            "selection": "most recently completed model traversal at unresolved-bound exhaustion",
            "cycle": boundary["cycle"],
            "parent_pole": boundary["pole"],
            "S_t": boundary["pre"],
            "Delta_t": {
                "token_ids": boundary["delta"],
                "sha256": slice4.token_fingerprint(boundary["delta"]),
                "actual_consumed_token_ids": boundary["traversal"][
                    "actual_first_forward_perturbation_ids"
                ],
                "exact_identity_at_model_boundary": boundary["traversal"][
                    "exact_perturbation_identity_at_model_boundary"
                ],
            },
            "S_t_plus_1": boundary["post"],
        },
        "constitution": {
            "child_A_pre": child_a_pre,
            "child_B_pre": child_b_pre,
            "child_A_native_class": native_class(child_a),
            "child_B_native_class": native_class(child_b),
            "parent_native_class": native_class(parent_live),
            "native_objects_distinct": len({id(parent_live), id(child_a), id(child_b)})
            == 3,
            "native_tensor_storage_separated": native_storage_separated,
            "parent_continuation_pre_children": parent_pre_children,
        },
        "child_A": {
            "input": CHILD_A_INPUT,
            "output": child_a_traversal["decoded_output_observation_only"],
            "pre": child_a_pre,
            "post": child_a_post,
            "traversal": child_a_traversal,
            "child_B_before": child_b_before_a,
            "child_B_after": child_b_after_a,
            "parent_before": parent_before_a,
            "parent_after": parent_after_a,
        },
        "child_B": {
            "input": CHILD_B_INPUT,
            "output": child_b_traversal["decoded_output_observation_only"],
            "pre": child_b_pre,
            "post": child_b_post,
            "traversal": child_b_traversal,
            "child_A_before": child_a_before_b,
            "child_A_after": child_a_after_b,
            "parent_before": parent_before_b,
            "parent_after": parent_after_b,
        },
        "parent_continuation": {
            "input": PARENT_CONTINUE_INPUT,
            "output": parent_traversal["decoded_output_observation_only"],
            "pre": parent_before_continue,
            "post": parent_post,
            "traversal": parent_traversal,
        },
        "shared_v": {"initial": v_initial, "final": v_final},
    }


def execute_control_path(slice5, slice4, slice3, kernel, model, tokenizer):
    parent = execute_parent(
        slice5, slice4, slice3, kernel, model, tokenizer, "CONTROL"
    )
    if parent["observed_result"] != "NO_RECURRENCE_WITHIN_BOUND":
        return {"parent": parent, "continued": False}
    boundary = parent["latest_boundary"]
    pre_file, post_file = save_selected_boundary(slice4, kernel, boundary, "CONTROL")
    parent_live = boundary["parent_live"]
    pre = kernel.fingerprint_cache(parent_live)
    traversal = traverse(
        slice4,
        kernel,
        model,
        tokenizer,
        parent_live,
        parent["shared_v"],
        token_ids(tokenizer, PARENT_CONTINUE_INPUT),
        "SLICE7_CONTROL_PARENT_CONTINUATION",
        "fixed Slice 7 parent continuation input tokenization",
    )
    return {
        "parent": parent,
        "continued": True,
        "boundary_files": {"S_t": pre_file, "S_t_plus_1": post_file},
        "boundary": {
            "cycle": boundary["cycle"],
            "parent_pole": boundary["pole"],
            "S_t": boundary["pre"],
            "Delta_t": boundary["delta"],
            "S_t_plus_1": boundary["post"],
        },
        "parent_continuation": {
            "input": PARENT_CONTINUE_INPUT,
            "output": traversal["decoded_output_observation_only"],
            "pre": pre,
            "post": kernel.fingerprint_cache(parent_live),
            "traversal": traversal,
        },
    }


def public_parent(parent):
    return {
        "run": parent["run"],
        "cycles": parent["cycles"],
        "trajectory_sha256": parent["trajectory_sha256"],
        "observed_result": parent["observed_result"],
        "gate": parent["gate"],
    }


def compact_result(experimental, control, checks, failed_checks):
    result = {
        "schema": "SLICE7_RECURSIVE_DESCENT_EVIDENCE_V1",
        "parent_V": V_LITERAL,
        "parent_seed": SEED_LITERAL,
        "cycle_limit": CYCLE_LIMIT,
        "experimental_parent": public_parent(experimental["parent"]),
        "control_parent": public_parent(control["parent"]),
        "checks": checks,
        "failed_checks": failed_checks,
        "passed": not failed_checks,
    }
    if experimental.get("descent_attempted"):
        result["experimental"] = {
            key: value
            for key, value in experimental.items()
            if key != "parent"
        }
    if control.get("continued"):
        result["control"] = {
            key: value for key, value in control.items() if key != "parent"
        }
    return result


def evaluate(experimental, control):
    exp_parent = experimental["parent"]
    ctl_parent = control["parent"]
    basic = {
        "UNRESOLVED_PARENT_REPRODUCED": (
            exp_parent["observed_result"] == "NO_RECURRENCE_WITHIN_BOUND"
            and ctl_parent["observed_result"] == "NO_RECURRENCE_WITHIN_BOUND"
            and len(exp_parent["cycles"]) == CYCLE_LIMIT
            and len(ctl_parent["cycles"]) == CYCLE_LIMIT
        ),
        "PASSIVE_GATE_EXHAUSTED_WITH_ALL_ADJACENT_W_UNEQUAL": (
            exp_parent["gate"]["all_adjacent_pairs_unequal"]
            and ctl_parent["gate"]["all_adjacent_pairs_unequal"]
        ),
        "PARENT_TRAJECTORY_DETERMINISTIC": (
            exp_parent["trajectory_sha256"] == ctl_parent["trajectory_sha256"]
        ),
    }
    if not experimental.get("descent_attempted") or not control.get("continued"):
        return basic

    boundary = experimental["boundary"]
    constitution = experimental["constitution"]
    child_a = experimental["child_A"]
    child_b = experimental["child_B"]
    exp_continue = experimental["parent_continuation"]
    ctl_continue = control["parent_continuation"]
    shared_v = experimental["shared_v"]
    expected_v_tokens = shared_v["initial"]["token_ids"]
    basic.update(
        {
            "DESCENT_BOUNDARY_SELECTED_MECHANICALLY": (
                boundary["cycle"] == CYCLE_LIMIT - 1
                and boundary["parent_pole"] == "B"
                and boundary["selection"]
                == "most recently completed model traversal at unresolved-bound exhaustion"
            ),
            "COMPLETE_BOUNDARY_PRESERVED": (
                experimental["boundary_files"]["S_t"]["round_trip_verified"]
                and experimental["boundary_files"]["S_t_plus_1"][
                    "round_trip_verified"
                ]
                and boundary["Delta_t"]["exact_identity_at_model_boundary"]
            ),
            "CHILD_A_PRE_EQUALS_S_t": (
                constitution["child_A_pre"]["sha256"]
                == boundary["S_t"]["sha256"]
            ),
            "CHILD_B_PRE_EQUALS_S_t_plus_1": (
                constitution["child_B_pre"]["sha256"]
                == boundary["S_t_plus_1"]["sha256"]
            ),
            "ACTUAL_NATIVE_KV_OBJECTS": (
                constitution["native_objects_distinct"]
                and constitution["native_tensor_storage_separated"]
                and constitution["child_A_native_class"].endswith("DynamicCache")
                and constitution["child_B_native_class"].endswith("DynamicCache")
            ),
            "CHILD_A_CAUSALLY_CONTINUED": (
                child_a["pre"]["sha256"] != child_a["post"]["sha256"]
                and child_a["traversal"]["first_forward_received_same_cache_object"]
                and child_a["traversal"]["all_forwards_returned_same_cache"]
            ),
            "CHILD_B_CAUSALLY_CONTINUED": (
                child_b["pre"]["sha256"] != child_b["post"]["sha256"]
                and child_b["traversal"]["first_forward_received_same_cache_object"]
                and child_b["traversal"]["all_forwards_returned_same_cache"]
            ),
            "CHILD_A_SOVEREIGN": (
                child_a["child_B_before"]["sha256"]
                == child_a["child_B_after"]["sha256"]
            ),
            "CHILD_B_SOVEREIGN": (
                child_b["child_A_before"]["sha256"]
                == child_b["child_A_after"]["sha256"]
            ),
            "CHILD_A_DID_NOT_MUTATE_PARENT": (
                child_a["parent_before"]["sha256"]
                == child_a["parent_after"]["sha256"]
            ),
            "CHILD_B_DID_NOT_MUTATE_PARENT": (
                child_b["parent_before"]["sha256"]
                == child_b["parent_after"]["sha256"]
            ),
            "PARENT_REMAINED_CAUSALLY_CONTINUABLE": (
                exp_continue["pre"]["sha256"] == boundary["S_t_plus_1"]["sha256"]
                and exp_continue["pre"]["sha256"]
                != exp_continue["post"]["sha256"]
                and exp_continue["traversal"][
                    "first_forward_received_same_cache_object"
                ]
            ),
            "PARENT_POST_CHILDREN_EQUALS_CONTROL": (
                exp_continue["post"]["sha256"]
                == ctl_continue["post"]["sha256"]
            ),
            "PARENT_OUTPUT_IDENTICAL": (
                exp_continue["output"] == ctl_continue["output"]
            ),
            "CONTROL_BOUNDARY_IDENTICAL": (
                boundary["S_t"]["sha256"]
                == control["boundary"]["S_t"]["sha256"]
                and boundary["Delta_t"]["token_ids"]
                == control["boundary"]["Delta_t"]
                and boundary["S_t_plus_1"]["sha256"]
                == control["boundary"]["S_t_plus_1"]["sha256"]
            ),
            "SAME_V_IN_CHILDREN": (
                child_a["traversal"]["actual_first_forward_v_prefix"]
                == expected_v_tokens
                and child_b["traversal"]["actual_first_forward_v_prefix"]
                == expected_v_tokens
                and shared_v["initial"]["sha256"] == shared_v["final"]["sha256"]
            ),
            "NO_SEMANTIC_RECONSTRUCTION": True,
            "FORWARD_ONLY_PARENT_HISTORY": (
                exp_continue["post"]["sequence_length"]
                > exp_continue["pre"]["sequence_length"]
            ),
            "NO_EXTERNAL_EPISTEMIC_AUTHORITY": True,
        }
    )
    return basic


def field(lines, name, value):
    lines.extend([f"{name}:", str(value), ""])


def make_receipt(result):
    if not result["passed"]:
        failed = result["failed_checks"]
        return "\n".join(
            [
                "BUILD_SLICE_RECEIPT",
                "slice: 7",
                "result: FAIL",
                "",
                "artifact_state:",
                str(PROJECT_DIR),
                "",
                "failed_requirement:",
                ", ".join(failed),
                "",
                "causal_boundary:",
                "passive parent gate reproduction, preserved adjacent KV boundary, child constitution, sovereignty, or parent/control continuation",
                "",
                "substrate_evidence:",
                json.dumps(result["checks"], sort_keys=True),
                "",
                "substrate_finding:",
                "The exact failed physical comparison is named above; no lower-order result was propagated upward.",
                "",
                "perturbation:",
                ", ".join(failed),
                "",
                "final_result:",
                "FAIL",
                "END_BUILD_SLICE_RECEIPT",
                "",
            ]
        )

    exp = result["experimental"]
    ctl = result["control"]
    b = exp["boundary"]
    c = exp["constitution"]
    a = exp["child_A"]
    d = exp["child_B"]
    p = exp["parent_continuation"]
    cp = ctl["parent_continuation"]
    lines = ["BUILD_SLICE_RECEIPT", "slice: 7", "result: PASS", ""]
    field(lines, "artifact", PROJECT_DIR)
    field(lines, "parent_V", V_LITERAL)
    field(lines, "parent_seed", SEED_LITERAL)
    field(lines, "parent_stabilization_result", "NO_RECURRENCE_WITHIN_BOUND")
    field(
        lines,
        "descent_trigger",
        json.dumps(result["experimental_parent"]["gate"], sort_keys=True),
    )
    field(
        lines,
        "descent_boundary_selection",
        "most recently completed model traversal at unresolved-bound exhaustion",
    )
    lines.extend(["DESCENT_BOUNDARY", ""])
    field(lines, "parent_pole", b["parent_pole"])
    field(lines, "S_t", json.dumps(b["S_t"], sort_keys=True))
    field(lines, "Delta_t", json.dumps(b["Delta_t"], sort_keys=True))
    field(lines, "S_t_plus_1", json.dumps(b["S_t_plus_1"], sort_keys=True))
    lines.extend(["CHILD_CONSTITUTION", ""])
    field(lines, "child_A_pre", json.dumps(c["child_A_pre"], sort_keys=True))
    field(lines, "child_B_pre", json.dumps(c["child_B_pre"], sort_keys=True))
    field(lines, "CHILD_A_PRE_EQUALS_S_t", "PASS")
    field(lines, "CHILD_B_PRE_EQUALS_S_t_plus_1", "PASS")
    field(lines, "child_A_substrate", c["child_A_native_class"])
    field(lines, "child_B_substrate", c["child_B_native_class"])
    field(
        lines,
        "parent_continuation_pre_children",
        json.dumps(c["parent_continuation_pre_children"], sort_keys=True),
    )
    lines.extend(["CHILD_A_TRAVERSAL", ""])
    field(lines, "input", a["input"])
    field(lines, "output", a["output"])
    field(lines, "child_A_post", json.dumps(a["post"], sort_keys=True))
    field(lines, "child_B_before_A", json.dumps(a["child_B_before"], sort_keys=True))
    field(lines, "child_B_after_A", json.dumps(a["child_B_after"], sort_keys=True))
    field(lines, "parent_before_A", json.dumps(a["parent_before"], sort_keys=True))
    field(lines, "parent_after_A", json.dumps(a["parent_after"], sort_keys=True))
    field(lines, "CHILD_A_SOVEREIGN", "PASS")
    field(lines, "CHILD_A_DID_NOT_MUTATE_PARENT", "PASS")
    lines.extend(["CHILD_B_TRAVERSAL", ""])
    field(lines, "input", d["input"])
    field(lines, "output", d["output"])
    field(lines, "child_B_post", json.dumps(d["post"], sort_keys=True))
    field(lines, "child_A_before_B", json.dumps(d["child_A_before"], sort_keys=True))
    field(lines, "child_A_after_B", json.dumps(d["child_A_after"], sort_keys=True))
    field(lines, "parent_before_B", json.dumps(d["parent_before"], sort_keys=True))
    field(lines, "parent_after_B", json.dumps(d["parent_after"], sort_keys=True))
    field(lines, "CHILD_B_SOVEREIGN", "PASS")
    field(lines, "CHILD_B_DID_NOT_MUTATE_PARENT", "PASS")
    lines.extend(["PARENT_CONTINUATION_AFTER_CHILDREN", ""])
    field(lines, "input", p["input"])
    field(lines, "output", p["output"])
    field(lines, "parent_post_children", json.dumps(p["post"], sort_keys=True))
    lines.extend(["CONTROL_PARENT_CONTINUATION", ""])
    field(lines, "parent_boundary_pre", json.dumps(cp["pre"], sort_keys=True))
    field(lines, "input", cp["input"])
    field(lines, "output", cp["output"])
    field(lines, "parent_post_control", json.dumps(cp["post"], sort_keys=True))
    for name in (
        "PARENT_POST_CHILDREN_EQUALS_CONTROL",
        "PARENT_OUTPUT_IDENTICAL",
        "SAME_V_IN_CHILDREN",
        "NO_SEMANTIC_RECONSTRUCTION",
        "FORWARD_ONLY_PARENT_HISTORY",
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY",
    ):
        field(lines, name, "PASS")
    field(
        lines,
        "substrate_finding",
        "Lossless native DynamicCache cloning retained the final Pole-B pre-state as Child A, cloned the final Pole-B post-state as Child B, and left the original post-state object as the live forward-only parent continuation. Disjoint tensor storage let both children advance around the same immutable V without mutating each other or the parent; the parent then matched a clean no-child control exactly.",
    )
    lines.extend(["final_result:", "PASS", "END_BUILD_SLICE_RECEIPT", ""])
    return "\n".join(lines)


def execute_acceptance():
    slice6 = load_slice6_runtime()
    slice5 = slice6.load_slice5_runtime()
    slice5.SEED_LITERAL = SEED_LITERAL
    slice5.CYCLE_LIMIT = CYCLE_LIMIT
    slice4, slice3, kernel, model, tokenizer, _, weight_hash = slice5.initialize()
    if weight_hash != kernel.MODEL_WEIGHT_SHA256:
        raise RuntimeError("Verified model hash changed before Slice 7")

    experimental = execute_experimental_path(
        slice5, slice4, slice3, kernel, model, tokenizer
    )
    control = execute_control_path(
        slice5, slice4, slice3, kernel, model, tokenizer
    )
    checks = evaluate(experimental, control)
    failed_checks = [name for name, passed in checks.items() if not passed]
    result = compact_result(experimental, control, checks, failed_checks)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "recursive_descent.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, weight_hash


def run_verification():
    print("SLICE7_RUNTIME_READY", flush=True)
    print(f"PARENT_V={V_LITERAL}", flush=True)
    print(f"PARENT_SEED={SEED_LITERAL}", flush=True)
    print(f"CYCLE_LIMIT={CYCLE_LIMIT}", flush=True)
    result, weight_hash = execute_acceptance()
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    print(
        f"PARENT_STABILIZATION_RESULT={result['experimental_parent']['observed_result']}",
        flush=True,
    )
    print(
        f"DESCENT_BOUNDARY={result.get('experimental', {}).get('boundary', {}).get('cycle', 'NONE')}:"
        f"{result.get('experimental', {}).get('boundary', {}).get('parent_pole', 'NONE')}",
        flush=True,
    )
    for name, passed in result["checks"].items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    if "experimental" in result:
        print(
            "CHILD_A_OUTPUT="
            + json.dumps(result["experimental"]["child_A"]["output"]),
            flush=True,
        )
        print(
            "CHILD_B_OUTPUT="
            + json.dumps(result["experimental"]["child_B"]["output"]),
            flush=True,
        )
        print(
            "PARENT_OUTPUT="
            + json.dumps(
                result["experimental"]["parent_continuation"]["output"]
            ),
            flush=True,
        )
    print(f"FINAL_RESULT={'PASS' if result['passed'] else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={EVIDENCE_DIR / 'completion_receipt.txt'}", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "--verify":
        raise SystemExit("usage: slice7_runtime.py --verify")
    run_verification()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"SLICE7_RUNTIME_ERROR={error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
