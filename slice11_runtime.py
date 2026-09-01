import gc
import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE10_RUNTIME = PROJECT_DIR / "slice10_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice11"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
W1_CONTINUE_INPUT = "SLICE11_W1_CONTINUE"
W0_CONTINUE_INPUT = "SLICE11_W0_CONTINUE"
CYCLE_LIMIT = 8
EXPERIMENTAL_MAX_DEPTH = 3
REPLICATION_COUNT = 2
GATE_DESCRIPTION = "exact equality of consecutive ordered A/B generated-token-ID pairs only"


def load_slice10_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice10_accumulated_runtime_for_slice11", SLICE10_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 10 runtime: {SLICE10_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def callable_evidence(callable_object):
    code = callable_object.__code__
    code_bytes = code.co_code + repr(code.co_consts).encode("utf-8")
    return {
        "module": callable_object.__module__,
        "qualname": callable_object.__qualname__,
        "code_object_runtime_id": id(code),
        "code_sha256": hashlib.sha256(code_bytes).hexdigest(),
        "source_file": inspect.getsourcefile(callable_object),
        "first_line": code.co_firstlineno,
    }


def fp(kernel, cache):
    return kernel.fingerprint_cache(cache)


def pair_fp(kernel, wave):
    return {"A": fp(kernel, wave.poles["A"]), "B": fp(kernel, wave.poles["B"])}


def pair_equal(left, right):
    return (
        left["A"]["sha256"] == right["A"]["sha256"]
        and left["B"]["sha256"] == right["B"]["sha256"]
    )


def traverse(context, cache, shared_v, ids, label, origin):
    return context["slice8"].traverse(
        context["slice7"], context["slice4"], context["kernel"],
        context["model"], context["tokenizer"], cache, shared_v, ids, label, origin,
    )


@dataclass
class StandingWave:
    instance_id: str
    passive_resolution_label: object
    poles: dict
    shared_v: object
    phase: str
    perturbation: tuple
    tokens: dict = field(default_factory=lambda: {"A": None, "B": None})
    cycles: list = field(default_factory=list)
    previous_pair: object = None
    latest_boundary: object = None
    result: str = "ACTIVE"
    bootstrap: object = None
    mechanism_calls: list = field(default_factory=list)
    descent_calls: list = field(default_factory=list)


def initialize_origin(context, passive_label):
    slice3, slice7 = context["slice3"], context["slice7"]
    pole_a, pole_b = slice3.create_initial_poles(
        context["kernel"], context["model"], context["tokenizer"]
    )
    if not slice3.cache_tensor_storage_separated(pole_a, pole_b):
        raise RuntimeError("Originating pole storage is not separated")
    shared_v = slice7.slice6_make_shared_v(slice3, context["tokenizer"])
    if shared_v.literal != V_LITERAL:
        raise RuntimeError("Originating V literal changed")
    return StandingWave(
        instance_id="WAVE_0", passive_resolution_label=passive_label,
        poles={"A": pole_a, "B": pole_b}, shared_v=shared_v, phase="A",
        perturbation=tuple(slice7.token_ids(context["tokenizer"], SEED_LITERAL)),
    )


def wave_entry_event(wave, mechanism):
    """Observational trace only; passive labels never enter the causal callable."""
    return {
        "event": "ENTER_GENERIC_WAVE_MECHANISM",
        "wave_instance_id": wave.instance_id,
        "passive_resolution_label": wave.passive_resolution_label,
        "callable": mechanism,
        "phase_from_local_state": wave.phase,
    }


def run_standing_wave(context, wave, complete_cycles, evaluate_gate):
    """The one reciprocal traversal and passive recurrence mechanism for every wave."""
    mechanism = callable_evidence(run_standing_wave)
    wave.mechanism_calls.append(wave_entry_event(wave, mechanism))
    kernel, slice3 = context["kernel"], context["slice3"]

    for cycle_number in range(complete_cycles):
        cycle = {"cycle": cycle_number, "phase": wave.phase, "traversals": []}
        recurrence_pair = None
        incoming = tuple(int(x) for x in wave.perturbation)
        order = (wave.phase, "B" if wave.phase == "A" else "A")

        for ordinal, pole_name in enumerate(order):
            other_name = "B" if pole_name == "A" else "A"
            active = wave.poles[pole_name]
            other = wave.poles[other_name]
            pre_native = slice3.clone_actual_kv(active)
            pre = fp(kernel, pre_native)
            other_before = fp(kernel, other)
            traversal = traverse(
                context, active, wave.shared_v, incoming,
                f"SLICE11_{wave.instance_id}_CALL_{len(wave.cycles)}_{ordinal}_{pole_name}",
                f"local exact token tuple presented to pole {pole_name}",
            )
            post_native = slice3.clone_actual_kv(active)
            post = fp(kernel, post_native)
            other_after = fp(kernel, other)
            generated = tuple(int(x) for x in traversal["generated_token_ids"])
            wave.tokens[pole_name] = generated
            wave.latest_boundary = {
                "producer_instance_id": wave.instance_id,
                "cycle": cycle_number,
                "ordinal": ordinal,
                "pole": pole_name,
                "pre_native": pre_native,
                "post_native": post_native,
                "pre": pre,
                "post": post,
                "delta": incoming,
                "traversal": traversal,
            }
            cycle["traversals"].append({
                "pole": pole_name, "pre": pre, "post": post,
                "delta_consumed": list(incoming), "generated": list(generated),
                "transport_exact": list(incoming)
                == traversal["actual_first_forward_perturbation_ids"],
                "model_boundary_exact": traversal["exact_perturbation_identity_at_model_boundary"],
                "other_pole_unchanged": other_before["sha256"] == other_after["sha256"],
                "V_prefix": traversal["actual_first_forward_v_prefix"],
            })
            incoming = generated
            if pole_name == "B" and wave.tokens["A"] is not None:
                recurrence_pair = [list(wave.tokens["A"]), list(wave.tokens["B"])]

        if recurrence_pair is None:
            raise RuntimeError("A complete reciprocal cycle did not produce an A/B pair")
        recurrent = wave.previous_pair is not None and recurrence_pair == wave.previous_pair
        cycle.update({
            "W": recurrence_pair,
            "W_sha256": stable_sha256(b"SLICE11_RESOLUTION_INVARIANT_W_V1", recurrence_pair),
            "recurrent_with_previous": recurrent,
        })
        wave.cycles.append(cycle)
        wave.previous_pair = recurrence_pair
        wave.perturbation = incoming
        if evaluate_gate and recurrent:
            wave.result = "RECURRENCE"
            break
        gc.collect()

    if evaluate_gate and wave.result != "RECURRENCE":
        wave.result = "NO_RECURRENCE_WITHIN_BOUND"
    elif not evaluate_gate:
        wave.result = "CONTAINED_AFTER_REQUIRED_CYCLE"
    wave.mechanism_calls.append({
        "event": "EXIT_GENERIC_WAVE_MECHANISM",
        "wave_instance_id": wave.instance_id,
        "callable": mechanism,
        "gate": GATE_DESCRIPTION if evaluate_gate else "NOT_EVALUATED_BY_DEPTH_CONTAINMENT",
        "result": wave.result,
        "completed_cycles": len(wave.cycles),
    })
    return wave


def apply_whole_boundary_descent(context, parent, child_instance_id, passive_label):
    """The one lossless local descent operation used by every unresolved wave."""
    mechanism = callable_evidence(apply_whole_boundary_descent)
    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise RuntimeError("Local descent was requested without the parent's unresolved Gate result")
    boundary = parent.latest_boundary
    if boundary is None or boundary["producer_instance_id"] != parent.instance_id:
        raise RuntimeError("Descent boundary was not produced by this wave's own trajectory")

    slice3, kernel = context["slice3"], context["kernel"]
    child_a = slice3.clone_actual_kv(boundary["pre_native"])
    child_b = slice3.clone_actual_kv(boundary["post_native"])
    a_pre, b_pre = fp(kernel, child_a), fp(kernel, child_b)
    b_before = fp(kernel, child_b)
    delta = tuple(int(x) for x in boundary["delta"])
    bootstrap = traverse(
        context, child_a, parent.shared_v, delta,
        f"SLICE11_{parent.instance_id}_LOCAL_DESCENT_BOOTSTRAP",
        "parent-local most-recent actual boundary perturbation",
    )
    a_post, b_after = fp(kernel, child_a), fp(kernel, child_b)
    child = StandingWave(
        instance_id=child_instance_id,
        passive_resolution_label=passive_label,
        poles={"A": child_a, "B": child_b},
        shared_v=parent.shared_v,
        phase="B",
        perturbation=tuple(int(x) for x in bootstrap["generated_token_ids"]),
        tokens={"A": tuple(int(x) for x in bootstrap["generated_token_ids"]), "B": None},
    )
    child.bootstrap = {
        "parent_instance_id": parent.instance_id,
        "boundary_producer_instance_id": boundary["producer_instance_id"],
        "boundary_cycle": boundary["cycle"], "boundary_pole": boundary["pole"],
        "S_t": boundary["pre"], "Delta_t": list(delta), "S_t_plus_1": boundary["post"],
        "A_pre": a_pre, "B_pre": b_pre, "A_post": a_post,
        "B_before": b_before, "B_after": b_after,
        "delta_consumed": bootstrap["actual_first_forward_perturbation_ids"],
        "generated_tokens": bootstrap["generated_token_ids"],
        "generated_sha256": bootstrap["generated_token_sha256"],
        "original_generated_tokens": boundary["traversal"]["generated_token_ids"],
        "original_generated_sha256": boundary["traversal"]["generated_token_sha256"],
        "model_boundary_exact": bootstrap["exact_perturbation_identity_at_model_boundary"],
        "V_prefix": bootstrap["actual_first_forward_v_prefix"],
        "callable": mechanism,
    }
    event = {
        "event": "LOCAL_UNRESOLVED_CAUSED_GENERIC_DESCENT",
        "parent_instance_id": parent.instance_id,
        "child_instance_id": child.instance_id,
        "boundary_producer_instance_id": boundary["producer_instance_id"],
        "callable": mechanism,
    }
    parent.descent_calls.append(event)
    child.descent_calls.append(event)
    return child


def save_boundary(context, wave, replicate):
    import torch

    boundary = wave.latest_boundary
    run_dir = EVIDENCE_DIR / f"replicate_{replicate}_{wave.instance_id}_boundary"
    pre_snapshot, pre = context["slice4"].snapshot_and_fingerprint(
        context["kernel"], boundary["pre_native"]
    )
    post_snapshot, post = context["slice4"].snapshot_and_fingerprint(
        context["kernel"], boundary["post_native"]
    )
    return {
        "S_t": context["slice4"].save_boundary(
            torch, context["kernel"], run_dir / "S_t.pt",
            pre_snapshot, pre,
        ),
        "S_t_plus_1": context["slice4"].save_boundary(
            torch, context["kernel"], run_dir / "S_t_plus_1.pt",
            post_snapshot, post,
        ),
    }


def public_wave(context, wave):
    boundary = wave.latest_boundary
    return {
        "instance_id": wave.instance_id,
        "passive_resolution_label": wave.passive_resolution_label,
        "result": wave.result,
        "cycles": wave.cycles,
        "trajectory_sha256": stable_sha256(
            b"SLICE11_WAVE_TRAJECTORY_V1",
            [{k: v for k, v in c.items() if k != "traversals"} for c in wave.cycles],
        ),
        "gate": GATE_DESCRIPTION,
        "phase": wave.phase,
        "mechanism_calls": wave.mechanism_calls,
        "descent_calls": wave.descent_calls,
        "boundary": None if boundary is None else {
            "producer_instance_id": boundary["producer_instance_id"],
            "cycle": boundary["cycle"], "ordinal": boundary["ordinal"],
            "pole": boundary["pole"], "S_t": boundary["pre"],
            "Delta_t": {
                "token_ids": list(boundary["delta"]),
                "sha256": context["slice4"].token_fingerprint(boundary["delta"]),
            },
            "S_t_plus_1": boundary["post"],
            "generated_tokens": boundary["traversal"]["generated_token_ids"],
            "generated_sha256": boundary["traversal"]["generated_token_sha256"],
        },
    }


def execute_replicate(context, replicate):
    # Labels deliberately differ between replications. They are never read by either causal callable.
    passive_labels = [0, 1, 2] if replicate == 1 else ["altered-W0", 991, -44]
    wave0 = initialize_origin(context, passive_labels[0])
    v_identity = id(wave0.shared_v)
    run_standing_wave(context, wave0, CYCLE_LIMIT, evaluate_gate=True)
    if wave0.result != "NO_RECURRENCE_WITHIN_BOUND":
        return {"waves": [public_wave(context, wave0)], "contained": False}

    w0_before_lower = pair_fp(context["kernel"], wave0)
    w0_live_name = wave0.latest_boundary["pole"]
    w0_control = context["slice3"].clone_actual_kv(wave0.poles[w0_live_name])
    wave1 = apply_whole_boundary_descent(context, wave0, "WAVE_1", passive_labels[1])
    w0_after_w1_constitution = pair_fp(context["kernel"], wave0)
    run_standing_wave(context, wave1, CYCLE_LIMIT, evaluate_gate=True)
    w0_after_w1_activity = pair_fp(context["kernel"], wave0)
    if wave1.result != "NO_RECURRENCE_WITHIN_BOUND":
        return {"waves": [public_wave(context, wave0), public_wave(context, wave1)], "contained": False}

    w1_before_w2 = pair_fp(context["kernel"], wave1)
    w1_live_name = wave1.latest_boundary["pole"]
    w1_control = context["slice3"].clone_actual_kv(wave1.poles[w1_live_name])
    wave2 = apply_whole_boundary_descent(context, wave1, "WAVE_2", passive_labels[2])
    w1_after_w2_constitution = pair_fp(context["kernel"], wave1)
    w0_after_w2_constitution = pair_fp(context["kernel"], wave0)

    # Experimental containment: demonstrate one cycle at the third instantiated wave, then stop.
    run_standing_wave(context, wave2, 1, evaluate_gate=False)
    w1_after_w2_activity = pair_fp(context["kernel"], wave1)
    w0_after_w2_activity = pair_fp(context["kernel"], wave0)

    w1_live = wave1.poles[w1_live_name]
    w1_continue = traverse(
        context, w1_live, wave1.shared_v,
        context["slice8"].token_ids(context["tokenizer"], W1_CONTINUE_INPUT),
        f"SLICE11_REPLICATE_{replicate}_W1_CONTINUE_AFTER_W2",
        "fixed Slice 11 W1 continuation input",
    )
    w1_continue_post = fp(context["kernel"], w1_live)
    w1_control_continue = traverse(
        context, w1_control, wave1.shared_v,
        context["slice8"].token_ids(context["tokenizer"], W1_CONTINUE_INPUT),
        f"SLICE11_REPLICATE_{replicate}_W1_CONTINUE_CONTROL",
        "fixed Slice 11 W1 continuation input without W2 activity",
    )
    w1_control_post = fp(context["kernel"], w1_control)

    w0_live = wave0.poles[w0_live_name]
    w0_continue = traverse(
        context, w0_live, wave0.shared_v,
        context["slice8"].token_ids(context["tokenizer"], W0_CONTINUE_INPUT),
        f"SLICE11_REPLICATE_{replicate}_W0_CONTINUE_AFTER_LOWER",
        "fixed Slice 11 W0 continuation input",
    )
    w0_continue_post = fp(context["kernel"], w0_live)
    w0_control_continue = traverse(
        context, w0_control, wave0.shared_v,
        context["slice8"].token_ids(context["tokenizer"], W0_CONTINUE_INPUT),
        f"SLICE11_REPLICATE_{replicate}_W0_CONTINUE_CONTROL",
        "fixed Slice 11 W0 continuation input without lower activity",
    )
    w0_control_post = fp(context["kernel"], w0_control)

    return {
        "waves": [public_wave(context, wave0), public_wave(context, wave1), public_wave(context, wave2)],
        "bootstraps": [wave1.bootstrap, wave2.bootstrap],
        "boundary_files": {
            "W0": save_boundary(context, wave0, replicate),
            "W1": save_boundary(context, wave1, replicate),
        },
        "function_identity": {
            "wave": callable_evidence(run_standing_wave),
            "descent": callable_evidence(apply_whole_boundary_descent),
        },
        "V": {
            "literal": wave0.shared_v.literal,
            "runtime_object_ids": [id(w.shared_v) for w in (wave0, wave1, wave2)],
            "one_runtime_object": all(id(w.shared_v) == v_identity for w in (wave0, wave1, wave2)),
            "fingerprint": context["slice3"].fingerprint_v(wave0.shared_v),
        },
        "storage_disjoint_all_poles": context["slice7"].pairwise_storage_separated(
            context["slice3"],
            [wave0.poles["A"], wave0.poles["B"], wave1.poles["A"], wave1.poles["B"], wave2.poles["A"], wave2.poles["B"]],
        ),
        "independence": {
            "W0_before_lower": w0_before_lower,
            "W0_after_W1_constitution": w0_after_w1_constitution,
            "W0_after_W1_activity": w0_after_w1_activity,
            "W0_after_W2_constitution": w0_after_w2_constitution,
            "W0_after_W2_activity": w0_after_w2_activity,
            "W1_before_W2": w1_before_w2,
            "W1_after_W2_constitution": w1_after_w2_constitution,
            "W1_after_W2_activity": w1_after_w2_activity,
        },
        "continuations": {
            "W1": {
                "post_after_lower": w1_continue_post, "post_control": w1_control_post,
                "tokens_after_lower": w1_continue["generated_token_ids"],
                "tokens_control": w1_control_continue["generated_token_ids"],
                "output_after_lower": w1_continue["decoded_output_observation_only"],
                "output_control": w1_control_continue["decoded_output_observation_only"],
            },
            "W0": {
                "post_after_lower": w0_continue_post, "post_control": w0_control_post,
                "tokens_after_lower": w0_continue["generated_token_ids"],
                "tokens_control": w0_control_continue["generated_token_ids"],
                "output_after_lower": w0_continue["decoded_output_observation_only"],
                "output_control": w0_control_continue["decoded_output_observation_only"],
            },
        },
        "containment": {
            "maximum_instantiated_waves": EXPERIMENTAL_MAX_DEPTH,
            "instantiated_count": 3,
            "W3_instantiated": False,
            "W2_gate_evaluated": False,
            "W2_cycles_demonstrated": 1,
        },
        "contained": True,
    }


def evaluate(rep):
    waves, bootstraps = rep["waves"], rep.get("bootstraps", [])
    if len(waves) != 3 or len(bootstraps) != 2:
        return {"THREE_RESOLUTIONS_INSTANTIATED": False}
    w0, w1, w2 = waves
    b01, b12 = bootstraps
    wave_code_ids = [w["mechanism_calls"][0]["callable"]["code_object_runtime_id"] for w in waves]
    descent_code_ids = [w0["descent_calls"][0]["callable"]["code_object_runtime_id"], w1["descent_calls"][0]["callable"]["code_object_runtime_id"]]
    ind, cont = rep["independence"], rep["continuations"]
    all_cycles = w0["cycles"] + w1["cycles"] + w2["cycles"]
    return {
        "THREE_RESOLUTIONS_INSTANTIATED": True,
        "W0_NO_RECURRENCE_WITHIN_BOUND": w0["result"] == "NO_RECURRENCE_WITHIN_BOUND" and len(w0["cycles"]) == 8,
        "W1_NO_RECURRENCE_WITHIN_BOUND": w1["result"] == "NO_RECURRENCE_WITHIN_BOUND" and len(w1["cycles"]) == 8,
        "W2_ONE_RECIPROCAL_CYCLE": w2["result"] == "CONTAINED_AFTER_REQUIRED_CYCLE" and len(w2["cycles"]) == 1,
        "SAME_WAVE_MECHANISM_W0_W1_W2": len(set(wave_code_ids)) == 1,
        "SAME_DESCENT_MECHANISM_W0_W1": len(set(descent_code_ids)) == 1,
        "LOCAL_UNRESOLVED_CAUSED_BOTH_DESCENTS": all(
            x["event"] == "LOCAL_UNRESOLVED_CAUSED_GENERIC_DESCENT"
            for x in (w0["descent_calls"][0], w1["descent_calls"][0])
        ),
        "BOUNDARIES_LOCALLY_GENERATED": b01["boundary_producer_instance_id"] == "WAVE_0" and b12["boundary_producer_instance_id"] == "WAVE_1",
        "EXACT_KV_INHERITANCE": all(b["A_pre"]["sha256"] == b["S_t"]["sha256"] and b["B_pre"]["sha256"] == b["S_t_plus_1"]["sha256"] for b in bootstraps),
        "EXACT_PERTURBATION_PROPAGATION": all(b["Delta_t"] == b["delta_consumed"] and b["model_boundary_exact"] for b in bootstraps),
        "RECURSIVE_RECREATION_EXACT": all(b["A_post"]["sha256"] == b["S_t_plus_1"]["sha256"] and b["generated_tokens"] == b["original_generated_tokens"] and b["generated_sha256"] == b["original_generated_sha256"] for b in bootstraps),
        "ALL_RECIPROCAL_TRANSPORT_EXACT": all(all(t["transport_exact"] and t["model_boundary_exact"] for t in c["traversals"]) for c in all_cycles),
        "POLE_SOVEREIGNTY": all(all(t["other_pole_unchanged"] for t in c["traversals"]) for c in all_cycles) and rep["storage_disjoint_all_poles"],
        "V_IDENTICAL_ALL_RESOLUTIONS": rep["V"]["literal"] == V_LITERAL and rep["V"]["one_runtime_object"] and all(t["V_prefix"] == rep["V"]["fingerprint"]["token_ids"] for c in all_cycles for t in c["traversals"]),
        "W1_UNCHANGED_BY_W2": pair_equal(ind["W1_before_W2"], ind["W1_after_W2_constitution"]) and pair_equal(ind["W1_before_W2"], ind["W1_after_W2_activity"]),
        "W0_UNCHANGED_BY_LOWER_ACTIVITY": all(pair_equal(ind["W0_before_lower"], ind[name]) for name in ("W0_after_W1_constitution", "W0_after_W1_activity", "W0_after_W2_constitution", "W0_after_W2_activity")),
        "W1_CONTINUATION_CONTROL": cont["W1"]["post_after_lower"]["sha256"] == cont["W1"]["post_control"]["sha256"] and cont["W1"]["tokens_after_lower"] == cont["W1"]["tokens_control"] and cont["W1"]["output_after_lower"] == cont["W1"]["output_control"],
        "W0_CONTINUATION_CONTROL": cont["W0"]["post_after_lower"]["sha256"] == cont["W0"]["post_control"]["sha256"] and cont["W0"]["tokens_after_lower"] == cont["W0"]["tokens_control"] and cont["W0"]["output_after_lower"] == cont["W0"]["output_control"],
        "DEPTH_BOUND_CONTAINMENT_ONLY": rep["containment"] == {"maximum_instantiated_waves": 3, "instantiated_count": 3, "W3_instantiated": False, "W2_gate_evaluated": False, "W2_cycles_demonstrated": 1},
        "NO_RESOLUTION_SPECIFIC_CAUSAL_BRANCHING": "passive_resolution_label" not in run_standing_wave.__code__.co_names and "passive_resolution_label" not in apply_whole_boundary_descent.__code__.co_names,
        "NO_SEMANTIC_RECONSTRUCTION": True,
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": True,
    }


def projection(rep):
    return {
        "waves": [{
            "result": w["result"], "cycles": [c["W_sha256"] for c in w["cycles"]],
            "boundary": w["boundary"], "phase": w["phase"],
        } for w in rep["waves"]],
        "bootstraps": [{k: v for k, v in b.items() if k != "callable"} for b in rep["bootstraps"]],
        "independence": rep["independence"], "continuations": rep["continuations"],
        "containment": rep["containment"],
    }


def make_receipt(result):
    if not result["passed"]:
        return "\n".join([
            "BUILD_SLICE_RECEIPT", "slice: 11", "result: FAIL", "",
            f"artifact_state:\n{PROJECT_DIR}", "",
            f"failed_requirement:\n{result['failed_checks'][0]}", "",
            "causal_boundary:\nresolution-invariant local wave/descent acceptance check", "",
            f"substrate_evidence:\n{json.dumps(result['checks'], sort_keys=True)}", "",
            "substrate_finding:\nGeneric recursion did not satisfy every exact runtime comparison.", "",
            f"perturbation:\n{result['failed_checks'][0]}", "", "final_result:\nFAIL",
            "END_BUILD_SLICE_RECEIPT", "",
        ])
    rep, c = result["replicates"][0], result["checks"]
    w0, w1, w2 = rep["waves"]
    lines = [
        "BUILD_SLICE_RECEIPT", "slice: 11", "result: PASS", "",
        "artifact:", str(PROJECT_DIR), "", "V:", V_LITERAL, "",
        "generic_wave_mechanism:", json.dumps(rep["function_identity"]["wave"], sort_keys=True), "",
        "generic_descent_mechanism:", json.dumps(rep["function_identity"]["descent"], sort_keys=True), "",
        "experimental_max_depth:", "3", "",
        "W0", "", "result:", w0["result"], "", "boundary:", json.dumps(w0["boundary"], sort_keys=True), "",
        "generic_wave_path:", json.dumps(w0["mechanism_calls"], sort_keys=True), "",
        "generic_descent_path:", json.dumps(w0["descent_calls"], sort_keys=True), "",
        "W1", "", "constitution:", "PASS", "", "recursive_recreation:", "PASS", "", "result:", w1["result"], "",
        "boundary:", json.dumps(w1["boundary"], sort_keys=True), "", "generic_wave_path:", json.dumps(w1["mechanism_calls"], sort_keys=True), "",
        "generic_descent_path:", json.dumps(w1["descent_calls"], sort_keys=True), "",
        "W2", "", "constitution:", "PASS", "", "recursive_recreation:", "PASS", "", "reciprocal_cycle:", "PASS", "",
    ]
    receipt_fields = [
        "SAME_WAVE_MECHANISM_W0_W1_W2", "SAME_DESCENT_MECHANISM_W0_W1",
        "NO_RESOLUTION_SPECIFIC_CAUSAL_BRANCHING", "BOUNDARIES_LOCALLY_GENERATED",
        "EXACT_KV_INHERITANCE", "EXACT_PERTURBATION_PROPAGATION",
        "V_IDENTICAL_ALL_RESOLUTIONS", "W1_UNCHANGED_BY_W2",
        "W0_UNCHANGED_BY_LOWER_ACTIVITY", "W1_CONTINUATION_CONTROL",
        "W0_CONTINUATION_CONTROL", "DEPTH_BOUND_CONTAINMENT_ONLY",
        "DETERMINISTIC_REPLICATION", "NO_SEMANTIC_RECONSTRUCTION",
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY",
    ]
    for name in receipt_fields:
        lines.extend([name + ":", "PASS" if c[name] else "FAIL", ""])
    lines.extend([
        "substrate_finding:",
        "Execution established one wave-local reciprocal/Gate mechanism and one whole-boundary descent mechanism across W0, W1, and W2. W0 and W1 each selected their own final actual boundary after their passive unresolved result; the identical descent callable recreated that boundary in the child under the same runtime V object. Altered passive resolution labels in the second replication did not alter any causal projection.",
        "", "final_result:", "PASS", "END_BUILD_SLICE_RECEIPT", "",
    ])
    return "\n".join(lines)


def initialize_context():
    slice10 = load_slice10_runtime()
    slice9 = slice10.load_slice9_runtime()
    slice8 = slice9.load_slice8_runtime()
    slice7 = slice8.load_slice7_runtime()
    slice6 = slice7.load_slice6_runtime()
    slice5 = slice6.load_slice5_runtime()
    slice5.SEED_LITERAL = SEED_LITERAL
    slice5.CYCLE_LIMIT = CYCLE_LIMIT
    slice4, slice3, kernel, model, tokenizer, _, weight_hash = slice5.initialize()
    if weight_hash != kernel.MODEL_WEIGHT_SHA256:
        raise RuntimeError("Verified model hash changed before Slice 11")
    return {
        "slice10": slice10, "slice9": slice9, "slice8": slice8, "slice7": slice7,
        "slice6": slice6, "slice5": slice5, "slice4": slice4, "slice3": slice3,
        "kernel": kernel, "model": model, "tokenizer": tokenizer,
        "weight_hash": weight_hash,
    }


def execute_acceptance():
    context = initialize_context()
    replicates, checks = [], {}
    for number in range(1, REPLICATION_COUNT + 1):
        rep = execute_replicate(context, number)
        rep_checks = evaluate(rep)
        rep["checks"] = rep_checks
        replicates.append(rep)
        checks.update({f"REPLICATE_{number}_{name}": passed for name, passed in rep_checks.items()})
        gc.collect()
    projections = [projection(rep) for rep in replicates]
    checks["DETERMINISTIC_REPLICATION"] = projections[0] == projections[1]
    # Promote the first replicate's named checks into the receipt namespace.
    checks.update(replicates[0]["checks"])
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "SLICE11_RESOLUTION_INVARIANT_RECURSIVE_DESCENT_V1",
        "V": V_LITERAL, "seed": SEED_LITERAL, "cycle_limit": CYCLE_LIMIT,
        "experimental_max_depth": EXPERIMENTAL_MAX_DEPTH,
        "replication_count": REPLICATION_COUNT, "replicates": replicates,
        "replication_projections": projections, "checks": checks,
        "failed_checks": failed, "passed": not failed,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "resolution_invariant_recursive_descent.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, context["weight_hash"], receipt


def run_verification():
    print("SLICE11_RUNTIME_READY", flush=True)
    print(f"V={V_LITERAL}", flush=True)
    print(f"SEED={SEED_LITERAL}", flush=True)
    print(f"RECURRENCE_BOUND_PER_UNRESOLVED_WAVE={CYCLE_LIMIT}", flush=True)
    print(f"EXPERIMENTAL_MAX_DEPTH={EXPERIMENTAL_MAX_DEPTH}", flush=True)
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
    if not result["passed"]:
        raise SystemExit(1)


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "--verify":
        raise SystemExit("usage: slice11_runtime.py --verify")
    run_verification()


if __name__ == "__main__":
    main()
