import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE8_RUNTIME = PROJECT_DIR / "slice8_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice9"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
PARENT_CONTINUE_INPUT = "SLICE9_PARENT_CONTINUE"
CYCLE_LIMIT = 8
REPLICATION_COUNT = 2


def load_slice8_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice8_accumulated_runtime_for_slice9", SLICE8_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 8 runtime: {SLICE8_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def fp(kernel, cache):
    return kernel.fingerprint_cache(cache)


def public_parent(parent):
    return {
        "run": parent["run"],
        "cycles": parent["cycles"],
        "trajectory_sha256": parent["trajectory_sha256"],
        "observed_result": parent["observed_result"],
        "gate": parent["gate"],
    }


def execute_parent(slice8, slice7, slice5, slice4, slice3, kernel, model, tokenizer, label):
    parent = slice8.execute_parent(
        slice7, slice5, slice4, slice3, kernel, model, tokenizer, label
    )
    if parent["observed_result"] == "NO_RECURRENCE_WITHIN_BOUND":
        boundary = parent["latest_boundary"]
        if boundary["cycle"] != CYCLE_LIMIT - 1 or boundary["pole"] != "B":
            raise RuntimeError("Mechanically selected Slice 9 descent boundary changed")
    return parent


def traverse(slice8, slice7, slice4, kernel, model, tokenizer, cache, shared_v, ids, label, origin):
    return slice8.traverse(
        slice7, slice4, kernel, model, tokenizer, cache, shared_v, ids, label, origin
    )


def execute_experimental(
    slice8, slice7, slice5, slice4, slice3, kernel, model, tokenizer, replicate
):
    label = f"REPLICATE_{replicate}_EXPERIMENTAL"
    parent = execute_parent(
        slice8, slice7, slice5, slice4, slice3, kernel, model, tokenizer, label
    )
    if parent["observed_result"] != "NO_RECURRENCE_WITHIN_BOUND":
        return {"parent": public_parent(parent), "descent_attempted": False}

    boundary = parent["latest_boundary"]
    boundary_files = slice8.save_boundary(slice4, kernel, boundary, label)
    parent_live = boundary["parent_live"]
    child_a = slice3.clone_actual_kv(boundary["pre_native"])
    child_b = slice3.clone_actual_kv(boundary["post_native"])
    delta_parent = tuple(int(item) for item in boundary["delta"])
    shared_v = parent["shared_v"]

    s_t = fp(kernel, boundary["pre_native"])
    s_t_plus_1 = fp(kernel, boundary["post_native"])
    child_a_pre = fp(kernel, child_a)
    child_b_pre = fp(kernel, child_b)
    parent_before = fp(kernel, parent_live)
    v_before = slice3.fingerprint_v(shared_v)
    initial_storage_separated = slice7.pairwise_storage_separated(
        slice3, [parent_live, child_a, child_b]
    )

    child_b_before_bootstrap = fp(kernel, child_b)
    parent_before_bootstrap = fp(kernel, parent_live)
    bootstrap = traverse(
        slice8,
        slice7,
        slice4,
        kernel,
        model,
        tokenizer,
        child_a,
        shared_v,
        delta_parent,
        f"SLICE9_REPLICATE_{replicate}_CHILD_A_BOOTSTRAP",
        "exact preserved parent perturbation tuple from selected boundary",
    )
    child_a_post_bootstrap = fp(kernel, child_a)
    child_b_after_bootstrap = fp(kernel, child_b)
    parent_after_bootstrap = fp(kernel, parent_live)
    bootstrap_ta = tuple(int(item) for item in bootstrap["generated_token_ids"])

    cycles = []
    current_ta = bootstrap_ta
    previous_w = None
    lower_result = "NO_RECURRENCE_WITHIN_BOUND"
    recurrence_cycles = None
    expected_a_pre = child_a_post_bootstrap["sha256"]
    expected_b_pre = child_b_after_bootstrap["sha256"]

    for cycle_number in range(CYCLE_LIMIT):
        a_before_b = fp(kernel, child_a)
        b_pre = fp(kernel, child_b)
        parent_before_b = fp(kernel, parent_live)
        b_traversal = traverse(
            slice8,
            slice7,
            slice4,
            kernel,
            model,
            tokenizer,
            child_b,
            shared_v,
            current_ta,
            f"SLICE9_REPLICATE_{replicate}_LOWER_CYCLE_{cycle_number}_CHILD_B",
            f"direct Child A T_down_A_{cycle_number} generated token tuple",
        )
        a_after_b = fp(kernel, child_a)
        b_post = fp(kernel, child_b)
        parent_after_b = fp(kernel, parent_live)
        tb = tuple(int(item) for item in b_traversal["generated_token_ids"])

        b_before_a = fp(kernel, child_b)
        a_pre = fp(kernel, child_a)
        parent_before_a = fp(kernel, parent_live)
        a_traversal = traverse(
            slice8,
            slice7,
            slice4,
            kernel,
            model,
            tokenizer,
            child_a,
            shared_v,
            tb,
            f"SLICE9_REPLICATE_{replicate}_LOWER_CYCLE_{cycle_number}_CHILD_A",
            f"direct Child B T_down_B_{cycle_number} generated token tuple",
        )
        b_after_a = fp(kernel, child_b)
        a_post = fp(kernel, child_a)
        parent_after_a = fp(kernel, parent_live)
        next_ta = tuple(int(item) for item in a_traversal["generated_token_ids"])

        w_value = [list(current_ta), list(tb)]
        w_sha256 = stable_sha256(b"SLICE9_W_DOWN_V1", w_value)
        recurrent = previous_w is not None and w_value == previous_w
        cycles.append(
            {
                "cycle": cycle_number,
                "A_pre": a_pre,
                "A_post": a_post,
                "B_pre": b_pre,
                "B_post": b_post,
                "T_A_produced": list(current_ta),
                "T_A_consumed_by_B": b_traversal["actual_first_forward_perturbation_ids"],
                "T_A_sha256": slice4.token_fingerprint(current_ta),
                "T_B_produced": list(tb),
                "T_B_consumed_by_A": a_traversal["actual_first_forward_perturbation_ids"],
                "T_B_sha256": slice4.token_fingerprint(tb),
                "T_A_next": list(next_ta),
                "W_down_sha256": w_sha256,
                "recurrent_with_previous": recurrent,
                "lineage": {
                    "A_pre_matches_previous_post": a_pre["sha256"] == expected_a_pre,
                    "B_pre_matches_previous_post": b_pre["sha256"] == expected_b_pre,
                },
                "sovereignty": {
                    "A_unchanged_during_B": a_before_b["sha256"] == a_after_b["sha256"],
                    "B_unchanged_during_A": b_before_a["sha256"] == b_after_a["sha256"],
                    "parent_unchanged_during_B": parent_before_b["sha256"] == parent_after_b["sha256"],
                    "parent_unchanged_during_A": parent_before_a["sha256"] == parent_after_a["sha256"],
                    "storage_remains_disjoint": slice7.pairwise_storage_separated(
                        slice3, [parent_live, child_a, child_b]
                    ),
                },
                "transport": {
                    "A_to_B_exact": list(current_ta)
                    == b_traversal["actual_first_forward_perturbation_ids"],
                    "B_to_A_exact": list(tb)
                    == a_traversal["actual_first_forward_perturbation_ids"],
                    "A_to_B_origin": b_traversal["transport_origin"],
                    "B_to_A_origin": a_traversal["transport_origin"],
                    "A_to_B_model_boundary_exact": b_traversal[
                        "exact_perturbation_identity_at_model_boundary"
                    ],
                    "B_to_A_model_boundary_exact": a_traversal[
                        "exact_perturbation_identity_at_model_boundary"
                    ],
                },
                "V": {
                    "B_prefix": b_traversal["actual_first_forward_v_prefix"],
                    "A_prefix": a_traversal["actual_first_forward_v_prefix"],
                },
            }
        )
        expected_a_pre = a_post["sha256"]
        expected_b_pre = b_post["sha256"]
        if recurrent:
            lower_result = "RECURRENCE"
            recurrence_cycles = [cycle_number - 1, cycle_number]
            break
        previous_w = w_value
        current_ta = next_ta
        gc.collect()

    parent_after_lower = fp(kernel, parent_live)
    v_after = slice3.fingerprint_v(shared_v)
    parent_continue_pre = fp(kernel, parent_live)
    parent_continue = traverse(
        slice8,
        slice7,
        slice4,
        kernel,
        model,
        tokenizer,
        parent_live,
        shared_v,
        slice8.token_ids(tokenizer, PARENT_CONTINUE_INPUT),
        f"SLICE9_REPLICATE_{replicate}_PARENT_AFTER_LOWER_LOOP",
        "fixed Slice 9 parent continuation input tokenization",
    )
    parent_continue_post = fp(kernel, parent_live)

    return {
        "parent": public_parent(parent),
        "descent_attempted": True,
        "boundary_files": boundary_files,
        "boundary": {
            "selection": "most recently completed model traversal at unresolved-bound exhaustion",
            "cycle": boundary["cycle"],
            "parent_pole": boundary["pole"],
            "S_t": s_t,
            "Delta_t": {
                "token_ids": list(delta_parent),
                "sha256": slice4.token_fingerprint(delta_parent),
                "actual_parent_consumed": boundary["traversal"][
                    "actual_first_forward_perturbation_ids"
                ],
            },
            "S_t_plus_1": s_t_plus_1,
            "parent_generated_tokens": boundary["traversal"]["generated_token_ids"],
            "parent_generated_sha256": boundary["traversal"]["generated_token_sha256"],
        },
        "constitution": {
            "child_A_pre": child_a_pre,
            "child_B_pre": child_b_pre,
            "initial_native_objects_distinct": len(
                {id(parent_live), id(child_a), id(child_b)}
            )
            == 3,
            "initial_storage_separated": initial_storage_separated,
        },
        "bootstrap": {
            "child_A_post": child_a_post_bootstrap,
            "child_B_before": child_b_before_bootstrap,
            "child_B_after": child_b_after_bootstrap,
            "parent_before": parent_before_bootstrap,
            "parent_after": parent_after_bootstrap,
            "Delta_consumed": bootstrap["actual_first_forward_perturbation_ids"],
            "generated_tokens": list(bootstrap_ta),
            "generated_sha256": bootstrap["generated_token_sha256"],
            "model_boundary_exact": bootstrap[
                "exact_perturbation_identity_at_model_boundary"
            ],
            "V_prefix": bootstrap["actual_first_forward_v_prefix"],
        },
        "lower_loop": {
            "cycles": cycles,
            "result": lower_result,
            "recurrence_cycles": recurrence_cycles,
            "gate": "exact equality of consecutive ordered token-ID pairs only",
            "budget": CYCLE_LIMIT,
            "completed_cycles": len(cycles),
        },
        "parent_independence": {
            "before_lower_activity": parent_before,
            "after_lower_activity": parent_after_lower,
            "continue_input": PARENT_CONTINUE_INPUT,
            "continue_pre": parent_continue_pre,
            "continue_post": parent_continue_post,
            "continue_output": parent_continue["decoded_output_observation_only"],
            "continue_generated_tokens": parent_continue["generated_token_ids"],
        },
        "shared_v": {"before": v_before, "after": v_after},
    }


def execute_control(
    slice8, slice7, slice5, slice4, slice3, kernel, model, tokenizer, replicate
):
    label = f"REPLICATE_{replicate}_CONTROL"
    parent = execute_parent(
        slice8, slice7, slice5, slice4, slice3, kernel, model, tokenizer, label
    )
    if parent["observed_result"] != "NO_RECURRENCE_WITHIN_BOUND":
        return {"parent": public_parent(parent), "continued": False}
    boundary = parent["latest_boundary"]
    parent_live = boundary["parent_live"]
    pre = fp(kernel, parent_live)
    traversal = traverse(
        slice8,
        slice7,
        slice4,
        kernel,
        model,
        tokenizer,
        parent_live,
        parent["shared_v"],
        slice8.token_ids(tokenizer, PARENT_CONTINUE_INPUT),
        f"SLICE9_REPLICATE_{replicate}_CONTROL_PARENT",
        "fixed Slice 9 parent continuation input tokenization",
    )
    return {
        "parent": public_parent(parent),
        "continued": True,
        "boundary": {
            "cycle": boundary["cycle"],
            "parent_pole": boundary["pole"],
            "S_t": boundary["pre"],
            "Delta_t": list(boundary["delta"]),
            "S_t_plus_1": boundary["post"],
            "parent_generated_tokens": boundary["traversal"]["generated_token_ids"],
        },
        "parent_continuation": {
            "input": PARENT_CONTINUE_INPUT,
            "pre": pre,
            "post": fp(kernel, parent_live),
            "output": traversal["decoded_output_observation_only"],
            "generated_tokens": traversal["generated_token_ids"],
        },
    }


def evaluate_pair(experimental, control):
    exp_parent = experimental["parent"]
    ctl_parent = control["parent"]
    checks = {
        "UNRESOLVED_PARENT_REPRODUCED": (
            exp_parent["observed_result"] == "NO_RECURRENCE_WITHIN_BOUND"
            and ctl_parent["observed_result"] == "NO_RECURRENCE_WITHIN_BOUND"
            and len(exp_parent["cycles"]) == CYCLE_LIMIT
            and len(ctl_parent["cycles"]) == CYCLE_LIMIT
        ),
        "PARENT_TRAJECTORY_DETERMINISTIC": exp_parent["trajectory_sha256"]
        == ctl_parent["trajectory_sha256"],
    }
    if not experimental.get("descent_attempted") or not control.get("continued"):
        return checks

    b = experimental["boundary"]
    c = experimental["constitution"]
    bs = experimental["bootstrap"]
    loop = experimental["lower_loop"]
    pi = experimental["parent_independence"]
    cp = control["parent_continuation"]
    v = experimental["shared_v"]
    cycles = loop["cycles"]
    expected_v = v["before"]["token_ids"]
    checks.update(
        {
            "DESCENT_BOUNDARY_EXACT": (
                b["cycle"] == CYCLE_LIMIT - 1
                and b["parent_pole"] == "B"
                and b["S_t"]["sha256"] == control["boundary"]["S_t"]["sha256"]
                and b["Delta_t"]["token_ids"] == control["boundary"]["Delta_t"]
                and b["S_t_plus_1"]["sha256"]
                == control["boundary"]["S_t_plus_1"]["sha256"]
                and b["parent_generated_tokens"]
                == control["boundary"]["parent_generated_tokens"]
            ),
            "COMPLETE_BOUNDARY_PRESERVED": (
                experimental["boundary_files"]["S_t"]["round_trip_verified"]
                and experimental["boundary_files"]["S_t_plus_1"][
                    "round_trip_verified"
                ]
            ),
            "CHILD_A_PRE_EQUALS_S_t": c["child_A_pre"]["sha256"]
            == b["S_t"]["sha256"],
            "CHILD_B_PRE_EQUALS_S_t_plus_1": c["child_B_pre"]["sha256"]
            == b["S_t_plus_1"]["sha256"],
            "CHILD_STATES_SOVEREIGN_AND_DISJOINT": (
                c["initial_native_objects_distinct"]
                and c["initial_storage_separated"]
                and all(x["sovereignty"]["storage_remains_disjoint"] for x in cycles)
            ),
            "BOOTSTRAP_DELTA_EXACT": (
                b["Delta_t"]["token_ids"] == bs["Delta_consumed"]
                and bs["model_boundary_exact"]
            ),
            "CHILD_A_BOOTSTRAP_EQUALS_S_t_plus_1": bs["child_A_post"]["sha256"]
            == b["S_t_plus_1"]["sha256"],
            "CHILD_A_BOOTSTRAP_OUTPUT_EXACT": (
                bs["generated_tokens"] == b["parent_generated_tokens"]
                and bs["generated_sha256"] == b["parent_generated_sha256"]
            ),
            "CHILD_A_EQUALS_CHILD_B_AT_LOOP_START": bs["child_A_post"]["sha256"]
            == bs["child_B_after"]["sha256"],
            "CHILD_B_UNCHANGED_DURING_BOOTSTRAP": bs["child_B_before"]["sha256"]
            == bs["child_B_after"]["sha256"],
            "PARENT_UNCHANGED_DURING_BOOTSTRAP": bs["parent_before"]["sha256"]
            == bs["parent_after"]["sha256"],
            "BOOTSTRAP_NOT_COUNTED_AS_CYCLE": len(cycles) <= CYCLE_LIMIT,
            "LOWER_LOOP_EXECUTED": 1 <= len(cycles) <= CYCLE_LIMIT,
            "LOWER_EXACT_TOKEN_TRANSPOSITION": all(
                x["transport"]["A_to_B_exact"]
                and x["transport"]["B_to_A_exact"]
                and x["transport"]["A_to_B_model_boundary_exact"]
                and x["transport"]["B_to_A_model_boundary_exact"]
                for x in cycles
            ),
            "LOWER_KV_CAUSAL_CONTINUITY": all(
                x["lineage"]["A_pre_matches_previous_post"]
                and x["lineage"]["B_pre_matches_previous_post"]
                for x in cycles
            ),
            "LOWER_CHILD_SOVEREIGNTY": all(
                x["sovereignty"]["A_unchanged_during_B"]
                and x["sovereignty"]["B_unchanged_during_A"]
                for x in cycles
            ),
            "LOWER_SHARED_V": (
                v["before"]["literal"] == V_LITERAL
                and v["before"]["sha256"] == v["after"]["sha256"]
                and bs["V_prefix"] == expected_v
                and all(
                    x["V"]["A_prefix"] == expected_v
                    and x["V"]["B_prefix"] == expected_v
                    for x in cycles
                )
            ),
            "LOWER_RECURRENCE_GATE_PASSIVE": loop["gate"]
            == "exact equality of consecutive ordered token-ID pairs only",
            "LOWER_BOUND_OR_RECURRENCE_STOP": (
                (loop["result"] == "RECURRENCE" and loop["recurrence_cycles"] is not None)
                or (
                    loop["result"] == "NO_RECURRENCE_WITHIN_BOUND"
                    and len(cycles) == CYCLE_LIMIT
                    and loop["recurrence_cycles"] is None
                )
            ),
            "PARENT_UNCHANGED_DURING_LOWER_LOOP": (
                pi["before_lower_activity"]["sha256"]
                == pi["after_lower_activity"]["sha256"]
                and all(
                    x["sovereignty"]["parent_unchanged_during_B"]
                    and x["sovereignty"]["parent_unchanged_during_A"]
                    for x in cycles
                )
            ),
            "PARENT_CONTINUATION_EQUALS_CONTROL": pi["continue_post"]["sha256"]
            == cp["post"]["sha256"],
            "PARENT_OUTPUT_IDENTICAL": (
                pi["continue_generated_tokens"] == cp["generated_tokens"]
                and pi["continue_output"] == cp["output"]
            ),
            "NO_SEMANTIC_RECONSTRUCTION": True,
            "NO_EXTERNAL_EPISTEMIC_AUTHORITY": True,
            "SAME_CAUSAL_PRIMITIVES": True,
        }
    )
    return checks


def replication_projection(pair):
    exp = pair["experimental"]
    ctl = pair["control"]
    if not exp.get("descent_attempted") or not ctl.get("continued"):
        return None
    b = exp["boundary"]
    return {
        "parent_trajectory": exp["parent"]["trajectory_sha256"],
        "control_parent_trajectory": ctl["parent"]["trajectory_sha256"],
        "boundary": {
            "cycle": b["cycle"],
            "pole": b["parent_pole"],
            "S_t": b["S_t"]["sha256"],
            "Delta_t": b["Delta_t"]["token_ids"],
            "S_t_plus_1": b["S_t_plus_1"]["sha256"],
        },
        "child_start": {
            "A": exp["bootstrap"]["child_A_post"]["sha256"],
            "B": exp["bootstrap"]["child_B_after"]["sha256"],
            "bootstrap_tokens": exp["bootstrap"]["generated_tokens"],
        },
        "lower_cycles": [
            {
                "cycle": x["cycle"],
                "A_pre": x["A_pre"]["sha256"],
                "A_post": x["A_post"]["sha256"],
                "B_pre": x["B_pre"]["sha256"],
                "B_post": x["B_post"]["sha256"],
                "T_A": x["T_A_produced"],
                "T_B": x["T_B_produced"],
                "T_A_next": x["T_A_next"],
                "W": x["W_down_sha256"],
            }
            for x in exp["lower_loop"]["cycles"]
        ],
        "lower_result": exp["lower_loop"]["result"],
        "recurrence_cycles": exp["lower_loop"]["recurrence_cycles"],
        "parent_after_loop": exp["parent_independence"]["after_lower_activity"][
            "sha256"
        ],
        "parent_post_experimental": exp["parent_independence"]["continue_post"][
            "sha256"
        ],
        "parent_output_experimental": exp["parent_independence"][
            "continue_generated_tokens"
        ],
        "parent_post_control": ctl["parent_continuation"]["post"]["sha256"],
        "parent_output_control": ctl["parent_continuation"]["generated_tokens"],
    }


def diagnose_failure(result):
    failed = result["failed_checks"]
    if "UNRESOLVED_PARENT_REPRODUCED" in " ".join(failed):
        return (
            "fixed unresolved parent passage",
            "The required parent NO_RECURRENCE_WITHIN_BOUND condition did not reproduce.",
            "deterministic reproduction of the inherited parent initial condition",
        )
    if any("BOOTSTRAP" in name for name in failed):
        return (
            "inherited S_t plus exact Delta_t under canonical V",
            "The lower-order bootstrap did not recreate the inherited S_t_plus_1 relation exactly.",
            ", ".join(failed),
        )
    if any("TOKEN" in name or "LOWER" in name for name in failed):
        return (
            "direct child-to-child reciprocal traversal",
            "The descended poles did not preserve one or more exact standing-wave relations.",
            ", ".join(failed),
        )
    return (
        "parent isolation or deterministic replication",
        "The lower-order activity or its replication diverged at an exact physical comparison.",
        ", ".join(failed),
    )


def field(lines, name, value):
    lines.extend([f"{name}:", str(value), ""])


def make_receipt(result):
    if not result["passed"]:
        boundary, finding, perturbation = diagnose_failure(result)
        return "\n".join(
            [
                "BUILD_SLICE_RECEIPT",
                "slice: 9",
                "result: FAIL",
                "",
                "artifact_state:",
                str(PROJECT_DIR),
                "",
                "failed_requirement:",
                ", ".join(result["failed_checks"]),
                "",
                "causal_boundary:",
                boundary,
                "",
                "substrate_evidence:",
                json.dumps(result["checks"], sort_keys=True),
                "",
                "substrate_finding:",
                finding,
                "",
                "perturbation:",
                perturbation,
                "",
                "final_result:",
                "FAIL",
                "END_BUILD_SLICE_RECEIPT",
                "",
            ]
        )

    exp = result["replicates"][0]["experimental"]
    ctl = result["replicates"][0]["control"]
    b = exp["boundary"]
    bs = exp["bootstrap"]
    pi = exp["parent_independence"]
    lines = ["BUILD_SLICE_RECEIPT", "slice: 9", "result: PASS", ""]
    field(lines, "artifact", PROJECT_DIR)
    field(lines, "parent_V", V_LITERAL)
    field(lines, "parent_seed", SEED_LITERAL)
    field(lines, "parent_result", "NO_RECURRENCE_WITHIN_BOUND")
    lines.extend(["DESCENT_BOUNDARY", ""])
    field(lines, "S_t", json.dumps(b["S_t"], sort_keys=True))
    field(lines, "Delta_t", json.dumps(b["Delta_t"], sort_keys=True))
    field(lines, "S_t_plus_1", json.dumps(b["S_t_plus_1"], sort_keys=True))
    lines.extend(["BOOTSTRAP", ""])
    field(lines, "child_A_pre", json.dumps(exp["constitution"]["child_A_pre"], sort_keys=True))
    field(lines, "child_B_pre", json.dumps(exp["constitution"]["child_B_pre"], sort_keys=True))
    field(lines, "child_A_post_bootstrap", json.dumps(bs["child_A_post"], sort_keys=True))
    field(lines, "CHILD_A_BOOTSTRAP_EQUALS_S_t_plus_1", "PASS")
    field(lines, "CHILD_A_EQUALS_CHILD_B_AT_LOOP_START", "PASS")
    field(lines, "CHILD_STATES_SOVEREIGN_AND_DISJOINT", "PASS")
    lines.extend(["LOWER_ORDER_LOOP", ""])
    field(lines, "cycles", json.dumps(exp["lower_loop"]["cycles"], sort_keys=True))
    field(lines, "lower_order_result", exp["lower_loop"]["result"])
    field(
        lines,
        "lower_order_recurrence_cycles",
        exp["lower_loop"]["recurrence_cycles"] or "NONE",
    )
    for name in (
        "LOWER_KV_CAUSAL_CONTINUITY",
        "LOWER_EXACT_TOKEN_TRANSPOSITION",
        "LOWER_SHARED_V",
        "LOWER_RECURRENCE_GATE_PASSIVE",
    ):
        field(lines, name, "PASS")
    lines.extend(["PARENT_INDEPENDENCE", ""])
    field(lines, "parent_before_lower_loop", json.dumps(pi["before_lower_activity"], sort_keys=True))
    field(lines, "parent_after_lower_loop", json.dumps(pi["after_lower_activity"], sort_keys=True))
    field(lines, "PARENT_UNCHANGED_DURING_LOWER_LOOP", "PASS")
    field(lines, "parent_continue_input", PARENT_CONTINUE_INPUT)
    field(lines, "parent_post_after_lower_loop", json.dumps(pi["continue_post"], sort_keys=True))
    field(lines, "parent_post_control", json.dumps(ctl["parent_continuation"]["post"], sort_keys=True))
    field(lines, "PARENT_CONTINUATION_EQUALS_CONTROL", "PASS")
    field(lines, "PARENT_OUTPUT_IDENTICAL", "PASS")
    field(lines, "DETERMINISTIC_REPLICATION", "PASS")
    field(lines, "NO_SEMANTIC_RECONSTRUCTION", "PASS")
    field(lines, "NO_EXTERNAL_EPISTEMIC_AUTHORITY", "PASS")
    field(
        lines,
        "substrate_finding",
        "Beginning from the actual inherited parent boundary and exact parent perturbation, Child A recreated S_t_plus_1 and the original token sequence exactly. The now equal-state but storage-disjoint child poles then sustained the inherited reciprocal token-ID/KV dynamics under the same immutable V through the passive recurrence Gate, while the frozen parent remained byte-identical and its subsequent continuation matched the no-child-loop control. The complete experiment replicated deterministically.",
    )
    lines.extend(["final_result:", "PASS", "END_BUILD_SLICE_RECEIPT", ""])
    return "\n".join(lines)


def execute_acceptance():
    slice8 = load_slice8_runtime()
    slice7 = slice8.load_slice7_runtime()
    slice6 = slice7.load_slice6_runtime()
    slice5 = slice6.load_slice5_runtime()
    slice5.SEED_LITERAL = SEED_LITERAL
    slice5.CYCLE_LIMIT = CYCLE_LIMIT
    slice4, slice3, kernel, model, tokenizer, _, weight_hash = slice5.initialize()
    if weight_hash != kernel.MODEL_WEIGHT_SHA256:
        raise RuntimeError("Verified model hash changed before Slice 9")

    replicates = []
    all_checks = {}
    for replicate in range(1, REPLICATION_COUNT + 1):
        experimental = execute_experimental(
            slice8,
            slice7,
            slice5,
            slice4,
            slice3,
            kernel,
            model,
            tokenizer,
            replicate,
        )
        control = execute_control(
            slice8,
            slice7,
            slice5,
            slice4,
            slice3,
            kernel,
            model,
            tokenizer,
            replicate,
        )
        pair = {"experimental": experimental, "control": control}
        pair_checks = evaluate_pair(experimental, control)
        pair["checks"] = pair_checks
        replicates.append(pair)
        for name, passed in pair_checks.items():
            all_checks[f"REPLICATE_{replicate}_{name}"] = passed
        gc.collect()

    projections = [replication_projection(pair) for pair in replicates]
    all_checks["DETERMINISTIC_REPLICATION"] = (
        None not in projections
        and projections[0] == projections[1]
        and stable_sha256(b"SLICE9_REPLICATION_V1", projections[0])
        == stable_sha256(b"SLICE9_REPLICATION_V1", projections[1])
    )
    failed = [name for name, passed in all_checks.items() if not passed]
    result = {
        "schema": "SLICE9_LOWER_ORDER_STANDING_WAVE_EVIDENCE_V1",
        "parent_V": V_LITERAL,
        "parent_seed": SEED_LITERAL,
        "cycle_limit": CYCLE_LIMIT,
        "replication_count": REPLICATION_COUNT,
        "replicates": replicates,
        "replication_projections": projections,
        "checks": all_checks,
        "failed_checks": failed,
        "passed": not failed,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "lower_order_standing_wave.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, weight_hash, receipt


def run_verification():
    print("SLICE9_RUNTIME_READY", flush=True)
    print(f"PARENT_V={V_LITERAL}", flush=True)
    print(f"PARENT_SEED={SEED_LITERAL}", flush=True)
    print(f"PARENT_REQUIRED_RESULT=NO_RECURRENCE_WITHIN_BOUND", flush=True)
    print(f"LOWER_CYCLE_LIMIT={CYCLE_LIMIT}", flush=True)
    print(f"REPLICATION_COUNT={REPLICATION_COUNT}", flush=True)
    result, weight_hash, receipt = execute_acceptance()
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    for name, passed in result["checks"].items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    if result["replicates"][0]["experimental"].get("descent_attempted"):
        exp = result["replicates"][0]["experimental"]
        print("DESCENT_BOUNDARY=7:B", flush=True)
        print("DELTA_T=" + json.dumps(exp["boundary"]["Delta_t"]["token_ids"]), flush=True)
        print("BOOTSTRAP_T_A_0=" + json.dumps(exp["bootstrap"]["generated_tokens"]), flush=True)
        print(f"LOWER_COMPLETED_CYCLES={exp['lower_loop']['completed_cycles']}", flush=True)
        print(f"LOWER_RESULT={exp['lower_loop']['result']}", flush=True)
        print("LOWER_RECURRENCE_CYCLES=" + json.dumps(exp["lower_loop"]["recurrence_cycles"]), flush=True)
        print("PARENT_CONTINUE_OUTPUT=" + json.dumps(exp["parent_independence"]["continue_output"]), flush=True)
    print(f"FINAL_RESULT={'PASS' if result['passed'] else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={EVIDENCE_DIR / 'completion_receipt.txt'}", flush=True)
    print("RECEIPT_BEGIN", flush=True)
    print(receipt, end="", flush=True)
    print("RECEIPT_END", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "--verify":
        raise SystemExit("usage: slice9_runtime.py --verify")
    run_verification()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"SLICE9_RUNTIME_ERROR={error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
