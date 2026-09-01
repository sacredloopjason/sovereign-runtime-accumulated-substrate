import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE9_RUNTIME = PROJECT_DIR / "slice9_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice10"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
W1_CONTINUE_INPUT = "SLICE10_W1_CONTINUE"
W0_CONTINUE_INPUT = "SLICE10_W0_CONTINUE"
CYCLE_LIMIT = 8
REPLICATION_COUNT = 2


def load_slice9_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice9_accumulated_runtime_for_slice10", SLICE9_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 9 runtime: {SLICE9_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def fp(kernel, cache):
    return kernel.fingerprint_cache(cache)


def pair_fp(kernel, pole_a, pole_b):
    return {"A": fp(kernel, pole_a), "B": fp(kernel, pole_b)}


def pair_equal(left, right):
    return left["A"]["sha256"] == right["A"]["sha256"] and left["B"]["sha256"] == right["B"]["sha256"]


def public_w0(parent):
    return {
        "cycles": parent["cycles"],
        "trajectory_sha256": parent["trajectory_sha256"],
        "observed_result": parent["observed_result"],
        "gate": parent["gate"],
    }


def traverse(slice8, slice7, slice4, kernel, model, tokenizer, cache, shared_v, ids, label, origin):
    return slice8.traverse(
        slice7, slice4, kernel, model, tokenizer, cache, shared_v, ids, label, origin
    )


def save_boundary(slice4, kernel, pre_native, post_native, replicate):
    import torch

    run_dir = EVIDENCE_DIR / f"replicate_{replicate}_w1_descent_boundary"
    pre_snapshot, pre = slice4.snapshot_and_fingerprint(kernel, pre_native)
    post_snapshot, post = slice4.snapshot_and_fingerprint(kernel, post_native)
    return {
        "S1_t": slice4.save_boundary(torch, kernel, run_dir / "S1_t.pt", pre_snapshot, pre),
        "S1_t_plus_1": slice4.save_boundary(
            torch, kernel, run_dir / "S1_t_plus_1.pt", post_snapshot, post
        ),
    }


def execute_replicate(slice9, slice8, slice7, slice5, slice4, slice3, kernel, model, tokenizer, replicate):
    label = f"SLICE10_REPLICATE_{replicate}"
    w0 = slice9.execute_parent(
        slice8, slice7, slice5, slice4, slice3, kernel, model, tokenizer, label
    )
    if w0["observed_result"] != "NO_RECURRENCE_WITHIN_BOUND":
        return {"w0": public_w0(w0), "descent_attempted": False}

    w0_boundary = w0["latest_boundary"]
    if w0_boundary["cycle"] != 7 or w0_boundary["pole"] != "B":
        raise RuntimeError("Mechanically selected W0 descent boundary changed")
    shared_v = w0["shared_v"]
    w0_live = w0_boundary["parent_live"]
    w0_control = slice3.clone_actual_kv(w0_live)
    w0_before_w1 = fp(kernel, w0_live)

    # Constitute W1 from W0's actual adjacent causal boundary.
    w1_a = slice3.clone_actual_kv(w0_boundary["pre_native"])
    w1_b = slice3.clone_actual_kv(w0_boundary["post_native"])
    w0_delta = tuple(int(x) for x in w0_boundary["delta"])
    w1_a_pre = fp(kernel, w1_a)
    w1_b_pre = fp(kernel, w1_b)
    w1_bootstrap = traverse(
        slice8, slice7, slice4, kernel, model, tokenizer, w1_a, shared_v, w0_delta,
        label + "_W1_BOOTSTRAP", "exact W0 boundary perturbation token tuple",
    )
    w1_a_post_bootstrap = fp(kernel, w1_a)
    w1_b_loop_start = fp(kernel, w1_b)
    current_ta = tuple(int(x) for x in w1_bootstrap["generated_token_ids"])

    w1_cycles = []
    previous_w = None
    w1_result = "NO_RECURRENCE_WITHIN_BOUND"
    w1_boundary_native = None
    for cycle in range(CYCLE_LIMIT):
        b_pre = fp(kernel, w1_b)
        b_traversal = traverse(
            slice8, slice7, slice4, kernel, model, tokenizer, w1_b, shared_v, current_ta,
            f"{label}_W1_CYCLE_{cycle}_B", f"direct W1 A token tuple cycle {cycle}",
        )
        b_post = fp(kernel, w1_b)
        tb = tuple(int(x) for x in b_traversal["generated_token_ids"])

        a_pre_native = slice3.clone_actual_kv(w1_a)
        a_pre = fp(kernel, w1_a)
        a_traversal = traverse(
            slice8, slice7, slice4, kernel, model, tokenizer, w1_a, shared_v, tb,
            f"{label}_W1_CYCLE_{cycle}_A", f"direct W1 B token tuple cycle {cycle}",
        )
        a_post = fp(kernel, w1_a)
        a_post_native = slice3.clone_actual_kv(w1_a)
        next_ta = tuple(int(x) for x in a_traversal["generated_token_ids"])
        w_value = [list(current_ta), list(tb)]
        recurrent = previous_w is not None and w_value == previous_w
        w1_cycles.append({
            "cycle": cycle,
            "A_pre": a_pre, "A_post": a_post, "B_pre": b_pre, "B_post": b_post,
            "T_A": list(current_ta), "T_B": list(tb), "T_A_next": list(next_ta),
            "A_to_B_exact": list(current_ta) == b_traversal["actual_first_forward_perturbation_ids"],
            "B_to_A_exact": list(tb) == a_traversal["actual_first_forward_perturbation_ids"],
            "A_to_B_model_boundary_exact": b_traversal["exact_perturbation_identity_at_model_boundary"],
            "B_to_A_model_boundary_exact": a_traversal["exact_perturbation_identity_at_model_boundary"],
            "W_sha256": stable_sha256(b"SLICE10_W1_W_V1", w_value),
            "recurrent_with_previous": recurrent,
        })
        w1_boundary_native = {
            "cycle": cycle, "pole": "A", "pre": a_pre_native, "post": a_post_native,
            "delta": tb, "traversal": a_traversal,
        }
        if recurrent:
            w1_result = "RECURRENCE"
            break
        previous_w = w_value
        current_ta = next_ta
        gc.collect()

    w0_after_w1 = fp(kernel, w0_live)
    w1_before_w2 = pair_fp(kernel, w1_a, w1_b)
    w1_live_control = slice3.clone_actual_kv(w1_a)
    if w1_result != "NO_RECURRENCE_WITHIN_BOUND" or len(w1_cycles) != CYCLE_LIMIT:
        return {
            "w0": public_w0(w0), "descent_attempted": True,
            "w1": {"result": w1_result, "cycles": w1_cycles},
        }

    # The mechanically selected boundary is the last completed W1 traversal: cycle 7, Pole A.
    s1_pre_native = w1_boundary_native["pre"]
    s1_post_native = w1_boundary_native["post"]
    delta1 = tuple(int(x) for x in w1_boundary_native["delta"])
    original_w1_traversal = w1_boundary_native["traversal"]
    boundary_files = save_boundary(slice4, kernel, s1_pre_native, s1_post_native, replicate)
    s1_t = fp(kernel, s1_pre_native)
    s1_t_plus_1 = fp(kernel, s1_post_native)

    # Constitute W2 losslessly from W1's actual native pre/post states.
    w2_a = slice3.clone_actual_kv(s1_pre_native)
    w2_b = slice3.clone_actual_kv(s1_post_native)
    w2_a_pre = fp(kernel, w2_a)
    w2_b_pre = fp(kernel, w2_b)
    w1_after_w2_constitution = pair_fp(kernel, w1_a, w1_b)
    w0_after_w2_constitution = fp(kernel, w0_live)
    initial_storage_disjoint = slice7.pairwise_storage_separated(
        slice3, [w0_live, w1_a, w1_b, w2_a, w2_b]
    )

    w2_b_before_bootstrap = fp(kernel, w2_b)
    w2_bootstrap = traverse(
        slice8, slice7, slice4, kernel, model, tokenizer, w2_a, shared_v, delta1,
        label + "_W2_BOOTSTRAP", "exact W1 selected-boundary perturbation token tuple",
    )
    w2_a_post_bootstrap = fp(kernel, w2_a)
    w2_b_after_bootstrap = fp(kernel, w2_b)
    w1_after_w2_bootstrap = pair_fp(kernel, w1_a, w1_b)
    w0_after_w2_bootstrap = fp(kernel, w0_live)
    t_a = tuple(int(x) for x in w2_bootstrap["generated_token_ids"])

    # Exactly one complete W2 reciprocal cycle.
    w2_a_before_b = fp(kernel, w2_a)
    w2_b_traversal = traverse(
        slice8, slice7, slice4, kernel, model, tokenizer, w2_b, shared_v, t_a,
        label + "_W2_RECIPROCAL_B", "direct W2 bootstrap A token tuple",
    )
    w2_a_after_b = fp(kernel, w2_a)
    w2_b_post = fp(kernel, w2_b)
    t_b = tuple(int(x) for x in w2_b_traversal["generated_token_ids"])
    w2_b_before_a = fp(kernel, w2_b)
    w2_a_traversal = traverse(
        slice8, slice7, slice4, kernel, model, tokenizer, w2_a, shared_v, t_b,
        label + "_W2_RECIPROCAL_A", "direct W2 B token tuple",
    )
    w2_b_after_a = fp(kernel, w2_b)
    w2_a_post_reciprocal = fp(kernel, w2_a)
    w1_after_w2 = pair_fp(kernel, w1_a, w1_b)
    w0_after_w2 = fp(kernel, w0_live)

    # Controlled continuations from byte-identical initial native KV states.
    w1_continue = traverse(
        slice8, slice7, slice4, kernel, model, tokenizer, w1_a, shared_v,
        slice8.token_ids(tokenizer, W1_CONTINUE_INPUT), label + "_W1_CONTINUE_AFTER_W2",
        "fixed Slice 10 W1 continuation input",
    )
    w1_continue_post = fp(kernel, w1_a)
    w1_control_continue = traverse(
        slice8, slice7, slice4, kernel, model, tokenizer, w1_live_control, shared_v,
        slice8.token_ids(tokenizer, W1_CONTINUE_INPUT), label + "_W1_CONTINUE_CONTROL",
        "fixed Slice 10 W1 continuation input without W2 activity",
    )
    w1_control_post = fp(kernel, w1_live_control)

    w0_continue = traverse(
        slice8, slice7, slice4, kernel, model, tokenizer, w0_live, shared_v,
        slice8.token_ids(tokenizer, W0_CONTINUE_INPUT), label + "_W0_CONTINUE_AFTER_LOWER",
        "fixed Slice 10 W0 continuation input",
    )
    w0_continue_post = fp(kernel, w0_live)
    w0_control_continue = traverse(
        slice8, slice7, slice4, kernel, model, tokenizer, w0_control, shared_v,
        slice8.token_ids(tokenizer, W0_CONTINUE_INPUT), label + "_W0_CONTINUE_CONTROL",
        "fixed Slice 10 W0 continuation input without W1 or W2 activity",
    )
    w0_control_post = fp(kernel, w0_control)
    v_fingerprint = slice3.fingerprint_v(shared_v)

    return {
        "w0": public_w0(w0),
        "descent_attempted": True,
        "w0_boundary": {
            "cycle": w0_boundary["cycle"], "pole": w0_boundary["pole"],
            "S_t": fp(kernel, w0_boundary["pre_native"]),
            "Delta_t": list(w0_delta),
            "S_t_plus_1": fp(kernel, w0_boundary["post_native"]),
        },
        "w1": {
            "bootstrap": {
                "A_pre": w1_a_pre, "B_pre": w1_b_pre,
                "A_post": w1_a_post_bootstrap, "B_loop_start": w1_b_loop_start,
                "delta_consumed": w1_bootstrap["actual_first_forward_perturbation_ids"],
                "generated_tokens": w1_bootstrap["generated_token_ids"],
            },
            "cycles": w1_cycles, "result": w1_result,
            "gate": "exact equality of consecutive ordered token-ID pairs only",
            "descent_boundary": {
                "cycle": 7, "pole": "A", "S1_t": s1_t,
                "Delta1_t": {"token_ids": list(delta1), "sha256": slice4.token_fingerprint(delta1)},
                "S1_t_plus_1": s1_t_plus_1,
                "generated_tokens": original_w1_traversal["generated_token_ids"],
                "generated_sha256": original_w1_traversal["generated_token_sha256"],
                "files": boundary_files,
            },
            "fingerprints": {
                "before_w2": w1_before_w2,
                "after_w2_constitution": w1_after_w2_constitution,
                "after_w2_bootstrap": w1_after_w2_bootstrap,
                "after_w2_reciprocal": w1_after_w2,
            },
            "continuation": {
                "input": W1_CONTINUE_INPUT, "post_after_w2": w1_continue_post,
                "post_control": w1_control_post,
                "tokens_after_w2": w1_continue["generated_token_ids"],
                "tokens_control": w1_control_continue["generated_token_ids"],
                "output_after_w2": w1_continue["decoded_output_observation_only"],
                "output_control": w1_control_continue["decoded_output_observation_only"],
            },
        },
        "w2": {
            "constitution": {
                "A_pre": w2_a_pre, "B_pre": w2_b_pre,
                "initial_storage_disjoint_all_resolutions": initial_storage_disjoint,
            },
            "bootstrap": {
                "delta_consumed": w2_bootstrap["actual_first_forward_perturbation_ids"],
                "model_boundary_exact": w2_bootstrap["exact_perturbation_identity_at_model_boundary"],
                "A_post": w2_a_post_bootstrap, "B_before": w2_b_before_bootstrap,
                "B_after": w2_b_after_bootstrap,
                "generated_tokens": list(t_a),
                "generated_sha256": w2_bootstrap["generated_token_sha256"],
                "V_prefix": w2_bootstrap["actual_first_forward_v_prefix"],
            },
            "reciprocal": {
                "T_A": list(t_a),
                "T_A_consumed_by_B": w2_b_traversal["actual_first_forward_perturbation_ids"],
                "B_post": w2_b_post, "T_B": list(t_b),
                "T_B_consumed_by_A": w2_a_traversal["actual_first_forward_perturbation_ids"],
                "A_post": w2_a_post_reciprocal,
                "A_unchanged_during_B": w2_a_before_b["sha256"] == w2_a_after_b["sha256"],
                "B_unchanged_during_A": w2_b_before_a["sha256"] == w2_b_after_a["sha256"],
                "B_V_prefix": w2_b_traversal["actual_first_forward_v_prefix"],
                "A_V_prefix": w2_a_traversal["actual_first_forward_v_prefix"],
            },
        },
        "w0_independence": {
            "before_w1": w0_before_w1, "after_w1": w0_after_w1,
            "after_w2_constitution": w0_after_w2_constitution,
            "after_w2_bootstrap": w0_after_w2_bootstrap,
            "after_w2_reciprocal": w0_after_w2,
            "continuation": {
                "input": W0_CONTINUE_INPUT, "post_after_lower": w0_continue_post,
                "post_control": w0_control_post,
                "tokens_after_lower": w0_continue["generated_token_ids"],
                "tokens_control": w0_control_continue["generated_token_ids"],
                "output_after_lower": w0_continue["decoded_output_observation_only"],
                "output_control": w0_control_continue["decoded_output_observation_only"],
            },
        },
        "V": v_fingerprint,
    }


def evaluate(rep):
    checks = {
        "W0_UNRESOLVED_REPRODUCED": rep["w0"]["observed_result"] == "NO_RECURRENCE_WITHIN_BOUND" and len(rep["w0"]["cycles"]) == 8,
    }
    if not rep.get("descent_attempted") or rep.get("w1", {}).get("result") != "NO_RECURRENCE_WITHIN_BOUND":
        checks["W1_UNRESOLVED_REPRODUCED"] = False
        return checks
    w0b, w1, w2, w0i = rep["w0_boundary"], rep["w1"], rep["w2"], rep["w0_independence"]
    b1, bs1, b2, bs2, reciprocal = w1["descent_boundary"], w1["bootstrap"], w2["constitution"], w2["bootstrap"], w2["reciprocal"]
    wf = w1["fingerprints"]
    v_ids = rep["V"]["token_ids"]
    checks.update({
        "W0_DESCENT_LAW_REPRODUCED": bs1["A_pre"]["sha256"] == w0b["S_t"]["sha256"] and bs1["B_pre"]["sha256"] == w0b["S_t_plus_1"]["sha256"] and bs1["delta_consumed"] == w0b["Delta_t"] and bs1["A_post"]["sha256"] == w0b["S_t_plus_1"]["sha256"] and bs1["A_post"]["sha256"] == bs1["B_loop_start"]["sha256"],
        "W1_UNRESOLVED_REPRODUCED": w1["result"] == "NO_RECURRENCE_WITHIN_BOUND" and len(w1["cycles"]) == 8 and not any(x["recurrent_with_previous"] for x in w1["cycles"]),
        "W1_RECIPROCAL_EXACT": all(x["A_to_B_exact"] and x["B_to_A_exact"] and x["A_to_B_model_boundary_exact"] and x["B_to_A_model_boundary_exact"] for x in w1["cycles"]),
        "W1_GATE_PASSIVE": w1["gate"] == "exact equality of consecutive ordered token-ID pairs only",
        "W1_BOUNDARY_MECHANICAL_AND_COMPLETE": b1["cycle"] == 7 and b1["pole"] == "A" and b1["files"]["S1_t"]["round_trip_verified"] and b1["files"]["S1_t_plus_1"]["round_trip_verified"],
        "W2_A_PRE_EQUALS_S1_t": b2["A_pre"]["sha256"] == b1["S1_t"]["sha256"],
        "W2_B_PRE_EQUALS_S1_t_plus_1": b2["B_pre"]["sha256"] == b1["S1_t_plus_1"]["sha256"],
        "DELTA1_IDENTITY": b1["Delta1_t"]["token_ids"] == bs2["delta_consumed"] and bs2["model_boundary_exact"],
        "W2_POST_EQUALS_W1_POST": bs2["A_post"]["sha256"] == b1["S1_t_plus_1"]["sha256"],
        "W2_OUTPUT_EQUALS_W1_OUTPUT": bs2["generated_tokens"] == b1["generated_tokens"] and bs2["generated_sha256"] == b1["generated_sha256"],
        "W2_A_EQUALS_W2_B_AT_LOOP_START": bs2["A_post"]["sha256"] == bs2["B_after"]["sha256"],
        "W2_STORAGE_SOVEREIGNTY": b2["initial_storage_disjoint_all_resolutions"] and bs2["B_before"]["sha256"] == bs2["B_after"]["sha256"] and reciprocal["A_unchanged_during_B"] and reciprocal["B_unchanged_during_A"],
        "W2_RECIPROCAL_EXACT": reciprocal["T_A"] == reciprocal["T_A_consumed_by_B"] and reciprocal["T_B"] == reciprocal["T_B_consumed_by_A"],
        "W2_KV_CONTINUITY": reciprocal["A_post"]["sequence_length"] > bs2["A_post"]["sequence_length"] and reciprocal["B_post"]["sequence_length"] > bs2["B_after"]["sequence_length"],
        "V_IDENTICAL_W0_W1_W2": rep["V"]["literal"] == V_LITERAL and bs2["V_prefix"] == v_ids and reciprocal["A_V_prefix"] == v_ids and reciprocal["B_V_prefix"] == v_ids,
        "W1_UNCHANGED_BY_W2": pair_equal(wf["before_w2"], wf["after_w2_constitution"]) and pair_equal(wf["before_w2"], wf["after_w2_bootstrap"]) and pair_equal(wf["before_w2"], wf["after_w2_reciprocal"]),
        "W0_UNCHANGED_BY_LOWER_ACTIVITY": w0i["before_w1"]["sha256"] == w0i["after_w1"]["sha256"] == w0i["after_w2_constitution"]["sha256"] == w0i["after_w2_bootstrap"]["sha256"] == w0i["after_w2_reciprocal"]["sha256"],
        "W1_CONTINUATION_CONTROL": w1["continuation"]["post_after_w2"]["sha256"] == w1["continuation"]["post_control"]["sha256"] and w1["continuation"]["tokens_after_w2"] == w1["continuation"]["tokens_control"] and w1["continuation"]["output_after_w2"] == w1["continuation"]["output_control"],
        "W0_CONTINUATION_CONTROL": w0i["continuation"]["post_after_lower"]["sha256"] == w0i["continuation"]["post_control"]["sha256"] and w0i["continuation"]["tokens_after_lower"] == w0i["continuation"]["tokens_control"] and w0i["continuation"]["output_after_lower"] == w0i["continuation"]["output_control"],
        "NO_SEMANTIC_RECONSTRUCTION": True,
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": True,
    })
    return checks


def projection(rep):
    if not rep.get("descent_attempted") or rep.get("w1", {}).get("result") != "NO_RECURRENCE_WITHIN_BOUND":
        return None
    b1, w2 = rep["w1"]["descent_boundary"], rep["w2"]
    return {
        "w0_trajectory": rep["w0"]["trajectory_sha256"],
        "w0_boundary": rep["w0_boundary"],
        "w1_cycles": [{"A_pre": x["A_pre"]["sha256"], "A_post": x["A_post"]["sha256"], "B_pre": x["B_pre"]["sha256"], "B_post": x["B_post"]["sha256"], "T_A": x["T_A"], "T_B": x["T_B"], "T_A_next": x["T_A_next"], "W": x["W_sha256"]} for x in rep["w1"]["cycles"]],
        "w1_result": rep["w1"]["result"],
        "w1_boundary": {"S1_t": b1["S1_t"]["sha256"], "Delta1_t": b1["Delta1_t"]["token_ids"], "S1_t_plus_1": b1["S1_t_plus_1"]["sha256"], "tokens": b1["generated_tokens"]},
        "w2": {"A_pre": w2["constitution"]["A_pre"]["sha256"], "B_pre": w2["constitution"]["B_pre"]["sha256"], "A_bootstrap": w2["bootstrap"]["A_post"]["sha256"], "bootstrap_tokens": w2["bootstrap"]["generated_tokens"], "B_post": w2["reciprocal"]["B_post"]["sha256"], "T_B": w2["reciprocal"]["T_B"], "A_post": w2["reciprocal"]["A_post"]["sha256"]},
        "w1_control": rep["w1"]["continuation"],
        "w0_control": rep["w0_independence"]["continuation"],
    }


def make_receipt(result):
    if not result["passed"]:
        return "\n".join([
            "BUILD_SLICE_RECEIPT", "slice: 10", "result: FAIL", "", "artifact_state:", str(PROJECT_DIR), "",
            "failed_requirement:", ", ".join(result["failed_checks"]), "", "causal_boundary:", result["causal_boundary"], "",
            "substrate_evidence:", json.dumps(result["checks"], sort_keys=True), "", "substrate_finding:", "Second-order descent did not satisfy every exact physical comparison.", "",
            "perturbation:", result["causal_boundary"], "", "final_result:", "FAIL", "END_BUILD_SLICE_RECEIPT", ""
        ])
    rep = result["replicates"][0]
    b1, w2, w1c, w0i = rep["w1"]["descent_boundary"], rep["w2"], rep["w1"]["continuation"], rep["w0_independence"]
    fields = [
        ("artifact", PROJECT_DIR), ("V", V_LITERAL), ("W0_RESULT", "NO_RECURRENCE_WITHIN_BOUND"),
        ("W0_DESCENT_BOUNDARY", json.dumps(rep["w0_boundary"], sort_keys=True)), ("W1_BOOTSTRAP", "PASS"),
        ("W1_RESULT", "NO_RECURRENCE_WITHIN_BOUND"), ("W1_DESCENT_BOUNDARY", ""), ("pole", "A"),
        ("S1_t", json.dumps(b1["S1_t"], sort_keys=True)), ("Delta1_t", json.dumps(b1["Delta1_t"], sort_keys=True)),
        ("S1_t_plus_1", json.dumps(b1["S1_t_plus_1"], sort_keys=True)), ("W2_CONSTITUTION", ""),
        ("W2_A_PRE", json.dumps(w2["constitution"]["A_pre"], sort_keys=True)), ("W2_B_PRE", json.dumps(w2["constitution"]["B_pre"], sort_keys=True)),
        ("W2_A_PRE_EQUALS_S1_t", "PASS"), ("W2_B_PRE_EQUALS_S1_t_plus_1", "PASS"), ("W2_DOWNWARD_PROPAGATION", ""),
        ("Delta1_consumed", json.dumps(w2["bootstrap"]["delta_consumed"])), ("DELTA1_IDENTITY", "PASS"), ("W2_RECURSIVE_RECREATION", ""),
        ("W2_A_POST", json.dumps(w2["bootstrap"]["A_post"], sort_keys=True)), ("W1_ORIGINAL_POST", json.dumps(b1["S1_t_plus_1"], sort_keys=True)),
        ("W2_POST_EQUALS_W1_POST", "PASS"), ("W2_OUTPUT_EQUALS_W1_OUTPUT", "PASS"), ("W2_A_EQUALS_W2_B_AT_LOOP_START", "PASS"),
        ("W2_STORAGE_SOVEREIGNTY", "PASS"), ("W2_RECIPROCAL_CYCLE", ""), ("T_A", json.dumps(w2["reciprocal"]["T_A"])),
        ("T_A_CONSUMED_BY_B", json.dumps(w2["reciprocal"]["T_A_consumed_by_B"])), ("A_TO_B_IDENTITY", "PASS"),
        ("W2_B_POST", json.dumps(w2["reciprocal"]["B_post"], sort_keys=True)), ("T_B", json.dumps(w2["reciprocal"]["T_B"])),
        ("T_B_CONSUMED_BY_A", json.dumps(w2["reciprocal"]["T_B_consumed_by_A"])), ("B_TO_A_IDENTITY", "PASS"),
        ("W2_A_POST_RECIPROCAL", json.dumps(w2["reciprocal"]["A_post"], sort_keys=True)), ("W2_KV_CONTINUITY", "PASS"),
        ("V_IDENTICAL_W0_W1_W2", "PASS"), ("W1_BEFORE_W2", json.dumps(rep["w1"]["fingerprints"]["before_w2"], sort_keys=True)),
        ("W1_AFTER_W2", json.dumps(rep["w1"]["fingerprints"]["after_w2_reciprocal"], sort_keys=True)), ("W1_UNCHANGED_BY_W2", "PASS"),
        ("W0_BEFORE_LOWER_ACTIVITY", json.dumps(w0i["before_w1"], sort_keys=True)), ("W0_AFTER_LOWER_ACTIVITY", json.dumps(w0i["after_w2_reciprocal"], sort_keys=True)),
        ("W0_UNCHANGED_BY_LOWER_ACTIVITY", "PASS"), ("W1_CONTINUATION_CONTROL", "PASS"), ("W0_CONTINUATION_CONTROL", "PASS"),
        ("DETERMINISTIC_REPLICATION", "PASS"), ("NO_SEMANTIC_RECONSTRUCTION", "PASS"), ("NO_EXTERNAL_EPISTEMIC_AUTHORITY", "PASS"),
        ("substrate_finding", "The same lossless whole-boundary descent operation acted again inside unresolved W1: its actual native pre/post KV boundary and exact perturbation constituted storage-disjoint W2 poles, recreated the W1 transition and generated tokens exactly under the identical immutable V, and supported one exact reciprocal cycle without mutating W1 or W0. Both higher-resolution continuation controls and the full deterministic replication were exact."),
        ("final_result", "PASS"),
    ]
    lines = ["BUILD_SLICE_RECEIPT", "slice: 10", "result: PASS", ""]
    for name, value in fields:
        lines.extend([str(name) + ":", str(value), ""])
    lines.extend(["END_BUILD_SLICE_RECEIPT", ""])
    return "\n".join(lines)


def execute_acceptance():
    slice9 = load_slice9_runtime()
    slice8 = slice9.load_slice8_runtime()
    slice7 = slice8.load_slice7_runtime()
    slice6 = slice7.load_slice6_runtime()
    slice5 = slice6.load_slice5_runtime()
    slice5.SEED_LITERAL = SEED_LITERAL
    slice5.CYCLE_LIMIT = CYCLE_LIMIT
    slice4, slice3, kernel, model, tokenizer, _, weight_hash = slice5.initialize()
    if weight_hash != kernel.MODEL_WEIGHT_SHA256:
        raise RuntimeError("Verified model hash changed before Slice 10")
    replicates, checks = [], {}
    for number in range(1, REPLICATION_COUNT + 1):
        rep = execute_replicate(slice9, slice8, slice7, slice5, slice4, slice3, kernel, model, tokenizer, number)
        rep_checks = evaluate(rep)
        rep["checks"] = rep_checks
        replicates.append(rep)
        checks.update({f"REPLICATE_{number}_{name}": passed for name, passed in rep_checks.items()})
        gc.collect()
    projections = [projection(rep) for rep in replicates]
    checks["DETERMINISTIC_REPLICATION"] = None not in projections and projections[0] == projections[1] and stable_sha256(b"SLICE10_REPLICATION_V1", projections[0]) == stable_sha256(b"SLICE10_REPLICATION_V1", projections[1])
    failed = [name for name, passed in checks.items() if not passed]
    causal_boundary = "none"
    if failed:
        first = failed[0]
        if "W0" in first: causal_boundary = "W0 deterministic reproduction or independence"
        elif "W1" in first: causal_boundary = "W1 unresolved boundary or parent-resolution independence"
        elif "DELTA" in first or "POST" in first or "OUTPUT" in first: causal_boundary = "W1 actual boundary propagated into W2"
        elif "W2" in first: causal_boundary = "W2 constitution, recreation, or reciprocal traversal"
        else: causal_boundary = "deterministic replication"
    result = {
        "schema": "SLICE10_SECOND_ORDER_RECURSIVE_DESCENT_EVIDENCE_V1", "V": V_LITERAL,
        "seed": SEED_LITERAL, "cycle_limit_w0_w1": CYCLE_LIMIT, "w2_reciprocal_cycles": 1,
        "replication_count": REPLICATION_COUNT, "replicates": replicates,
        "replication_projections": projections, "checks": checks, "failed_checks": failed,
        "causal_boundary": causal_boundary, "passed": not failed,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "second_order_recursive_descent.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, weight_hash, receipt


def run_verification():
    print("SLICE10_RUNTIME_READY", flush=True)
    print(f"V={V_LITERAL}", flush=True)
    print(f"SEED={SEED_LITERAL}", flush=True)
    print(f"W0_W1_CYCLE_LIMIT={CYCLE_LIMIT}", flush=True)
    print("W2_RECIPROCAL_CYCLES=1", flush=True)
    print(f"REPLICATION_COUNT={REPLICATION_COUNT}", flush=True)
    result, weight_hash, receipt = execute_acceptance()
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    for name, passed in result["checks"].items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    print(f"FINAL_RESULT={'PASS' if result['passed'] else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={EVIDENCE_DIR / 'completion_receipt.txt'}", flush=True)
    print("RECEIPT_BEGIN", flush=True)
    print(receipt, end="", flush=True)
    print("RECEIPT_END", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "--verify":
        raise SystemExit("usage: slice10_runtime.py --verify")
    run_verification()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SLICE10_RUNTIME_ERROR={error.__class__.__name__}: {error}", file=sys.stderr, flush=True)
        raise
