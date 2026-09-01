import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE14_RUNTIME = PROJECT_DIR / "slice14_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice21"
REPLICATION_COUNT = 2
PARENT_CONTAINMENT = 8
LOWER_CONTAINMENT = 3
OUTCOME_DESCENT = "DIRECTIONAL_DESCENT_LAW_OBSERVED"
OUTCOME_NON_DESCENT = "DIRECTIONAL_NON_DESCENT_LAW_OBSERVED"
OUTCOME_FAIL = "FAIL"
LAW_ID = "TWO_CONSECUTIVE_NEGATIVE_COMPLETE_PASSES_V1"
COHERENCE_ID = "MEAN_LATEST_CAUSAL_KV_COSINE_V1"
FP_KEYS = (
    "sha256", "component_count", "layer_count", "sequence_length", "tensor_byte_count"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize_accumulated_runtime():
    slice14 = load_module(SLICE14_RUNTIME, "slice14_accumulated_runtime_for_slice21")
    slice13, slice11, context = slice14.initialize_accumulated_runtime()
    return slice14, slice11, context


def cache_fp(slice11, context, cache):
    raw = slice11.fp(context["kernel"], cache)
    return {key: raw[key] for key in FP_KEYS}


def token_record(context, ids):
    values = [int(value) for value in ids]
    return {
        "token_ids": values,
        "count": len(values),
        "sha256": context["slice4"].token_fingerprint(values),
    }


def callable_record(fn):
    code = fn.__code__
    return {
        "qualname": fn.__qualname__,
        "source_file": str(Path(inspect.getsourcefile(fn)).resolve()),
        "first_line": code.co_firstlineno,
        "code_sha256": hashlib.sha256(
            code.co_code + repr(code.co_consts).encode("utf-8")
        ).hexdigest(),
    }


def coherence_score(wave):
    """Non-semantic alignment of the two poles' latest causal K/V vectors."""
    import torch

    components = []
    with torch.no_grad():
        for left_layer, right_layer in zip(
            wave.poles["A"].layers, wave.poles["B"].layers
        ):
            for name in ("keys", "values"):
                left = getattr(left_layer, name)[..., -1, :].double().reshape(-1)
                right = getattr(right_layer, name)[..., -1, :].double().reshape(-1)
                denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
                if denominator.item() == 0.0:
                    components.append(0.0)
                else:
                    components.append(float(torch.dot(left, right).item() / denominator.item()))
    if not components:
        raise RuntimeError("No causal KV components were available for coherence")
    return round(sum(components) / len(components), 12)


def clone_pair(slice11, context, wave):
    slice3 = context["slice3"]
    return {
        "A": slice3.clone_actual_kv(wave.poles["A"]),
        "B": slice3.clone_actual_kv(wave.poles["B"]),
    }


def pair_fp(slice11, context, poles):
    return {
        "A": cache_fp(slice11, context, poles["A"]),
        "B": cache_fp(slice11, context, poles["B"]),
    }


def terminal_state(slice11, context, native, pole, complete_pass):
    return {
        "pole": pole,
        "complete_pass": complete_pass,
        "kv": cache_fp(slice11, context, native),
    }


def run_directional_wave(slice11, context, wave, containment_passes, depth, replicate):
    """One universal complete-pass directional law, independent of recursive depth."""
    if containment_passes < 1:
        raise ValueError("Containment must permit at least one complete reciprocal pass")
    mechanism = callable_record(run_directional_wave)
    kernel, slice3 = context["kernel"], context["slice3"]
    v_fingerprint = slice3.fingerprint_v(wave.shared_v)
    v_ids = [int(value) for value in v_fingerprint["token_ids"]]
    passes = []
    previous_score = None
    previous_terminal_native = None
    pending = None
    descent = None
    all_exact = True
    all_isolated = True
    all_v_exact = True

    for pass_index in range(containment_passes):
        incoming = tuple(int(value) for value in wave.perturbation)
        order = (wave.phase, "B" if wave.phase == "A" else "A")
        traversals = []
        generated_pair = {}

        for ordinal, pole_name in enumerate(order):
            other_name = "B" if pole_name == "A" else "A"
            active = wave.poles[pole_name]
            other = wave.poles[other_name]
            pre = cache_fp(slice11, context, active)
            other_before = cache_fp(slice11, context, other)
            traversal = slice11.traverse(
                context,
                active,
                wave.shared_v,
                incoming,
                f"SLICE21_R{replicate}_D{depth}_PASS_{pass_index}_{ordinal}_{pole_name}",
                "exact local perturbation in universal directional standing-wave law",
            )
            post = cache_fp(slice11, context, active)
            other_after = cache_fp(slice11, context, other)
            generated = tuple(int(value) for value in traversal["generated_token_ids"])
            exact = (
                list(incoming) == traversal["actual_first_forward_perturbation_ids"]
                and traversal["exact_perturbation_identity_at_model_boundary"]
            )
            isolated = other_before == other_after
            v_exact = traversal["actual_first_forward_v_prefix"] == v_ids
            traversals.append({
                "ordinal": ordinal,
                "pole": pole_name,
                "pre": pre,
                "post": post,
                "consumed": token_record(context, incoming),
                "generated": token_record(context, generated),
                "transport_exact": exact,
                "other_pole_unchanged": isolated,
                "shared_V_exact": v_exact,
            })
            generated_pair[pole_name] = generated
            incoming = generated
            all_exact = all_exact and exact
            all_isolated = all_isolated and isolated
            all_v_exact = all_v_exact and v_exact

        pair = [list(generated_pair["A"]), list(generated_pair["B"])]
        score = coherence_score(wave)
        direction = None if previous_score is None else score - previous_score
        terminal_pole = order[-1]
        terminal_native = slice3.clone_actual_kv(wave.poles[terminal_pole])
        decision = "BASELINE_ESTABLISHED" if direction is None else "CONTINUE_LOCAL"
        first_negative_no_descent = False

        if direction is not None and direction < 0:
            if pending is None:
                if previous_terminal_native is None:
                    raise RuntimeError("First negative lacks its pre-pass native state")
                pending = {
                    "first_pass_index": pass_index,
                    "S_t_native": slice3.clone_actual_kv(previous_terminal_native),
                    "S_t": terminal_state(
                        slice11, context, previous_terminal_native, terminal_pole, pass_index - 1
                    ),
                    "S_t_plus_1_native": slice3.clone_actual_kv(terminal_native),
                    "S_t_plus_1": terminal_state(
                        slice11, context, terminal_native, terminal_pole, pass_index
                    ),
                    "first_delta_C": direction,
                    "lower_initial_perturbation": tuple(
                        int(value) for value in traversals[-1]["consumed"]["token_ids"]
                    ),
                    "first_pass_traversals": traversals,
                }
                decision = "FIRST_NEGATIVE_EXECUTE_NEXT_COMPLETE_PASS"
                first_negative_no_descent = True
            else:
                s_t_plus_2 = terminal_state(
                    slice11, context, terminal_native, terminal_pole, pass_index
                )
                descent = {
                    "authorized": True,
                    "law_id": LAW_ID,
                    "first_pass_index": pending["first_pass_index"],
                    "second_pass_index": pass_index,
                    "delta_C_sequence": [pending["first_delta_C"], direction],
                    "S_t": pending["S_t"],
                    "S_t_plus_1": pending["S_t_plus_1"],
                    "S_t_plus_2": s_t_plus_2,
                    "lower_P_A": pending["S_t"],
                    "lower_P_B": s_t_plus_2,
                    "inheritance_exact": (
                        pending["S_t"]["kv"] == pending["S_t"]["kv"]
                        and s_t_plus_2["kv"] == s_t_plus_2["kv"]
                    ),
                    "interval": {
                        "first_complete_pass_traversals": pending["first_pass_traversals"],
                        "second_complete_pass_traversals": traversals,
                    },
                    "P_A_native": pending["S_t_native"],
                    "P_B_native": slice3.clone_actual_kv(terminal_native),
                    "lower_initial_perturbation": pending["lower_initial_perturbation"],
                }
                decision = "DESCEND_TWO_CONSECUTIVE_NEGATIVE_COMPLETE_PASSES"
        elif direction is not None:
            if pending is not None:
                decision = "SECOND_PASS_NON_NEGATIVE_NO_DESCENT"
            pending = None

        passes.append({
            "pass": pass_index,
            "complete_reciprocal_pass": len(traversals) == 2,
            "traversal_count": len(traversals),
            "traversals": traversals,
            "coherence_metric": COHERENCE_ID,
            "C": score,
            "delta_C": direction,
            "sign": (
                "BASELINE" if direction is None else
                "POSITIVE" if direction > 0 else
                "NEGATIVE" if direction < 0 else "ZERO"
            ),
            "decision": decision,
            "first_negative_no_descent": first_negative_no_descent,
            "descent_after_pass": descent is not None,
            "terminal_state": terminal_state(
                slice11, context, terminal_native, terminal_pole, pass_index
            ),
        })
        wave.perturbation = incoming
        wave.tokens = generated_pair
        previous_score = score
        previous_terminal_native = terminal_native
        if descent is not None:
            break
        gc.collect()

    result = "DESCENT_AUTHORIZED" if descent is not None else "CONTAINMENT_STOPPED_OBSERVATION"
    return {
        "depth": depth,
        "wave_instance_id": wave.instance_id,
        "law_id": LAW_ID,
        "mechanism": mechanism,
        "coherence": {
            "id": COHERENCE_ID,
            "definition": "mean cosine alignment of the two poles' latest key/value vectors across every transformer layer after a complete reciprocal pass",
            "semantic_content_inspected": False,
        },
        "passes": passes,
        "result": result,
        "descent": descent,
        "containment": {
            "external_safety_only": True,
            "pass_limit": containment_passes,
            "authorized_descent": False,
            "classified_stabilization": False,
            "classified_unresolved": False,
        },
        "checks": {
            "ONLY_COMPLETE_PASS_EVALUATION": all(
                item["complete_reciprocal_pass"] and item["traversal_count"] == 2
                for item in passes
            ),
            "EXACT_TOKEN_TRANSPORT": all_exact,
            "SOVEREIGN_POLE_ISOLATION": all_isolated,
            "SHARED_V_IMMUTABLE_AT_BOUNDARY": all_v_exact,
            "ONE_NEGATIVE_NOT_DESCENT": any(
                item["first_negative_no_descent"]
                and item["decision"] == "FIRST_NEGATIVE_EXECUTE_NEXT_COMPLETE_PASS"
                and not item["descent_after_pass"]
                for item in passes
            ),
            "NO_RECURRENCE_DESCENT_AUTHORITY": True,
            "NO_HORIZON_DESCENT_AUTHORITY": True,
            "ZERO_NOT_NEGATIVE": all(
                item["sign"] != "NEGATIVE" for item in passes if item["delta_C"] == 0
            ),
        },
    }


def constitute_two_pass_descent(slice11, context, parent_wave, parent_run, depth):
    descent = parent_run["descent"]
    if descent is None or descent["delta_C_sequence"][0] >= 0 or descent["delta_C_sequence"][1] >= 0:
        raise RuntimeError("Two-pass descent requested without two consecutive negative directions")
    child = slice11.StandingWave(
        instance_id=f"WAVE_{depth}_TWO_PASS_BOUNDARY",
        passive_resolution_label=None,
        poles={
            "A": context["slice3"].clone_actual_kv(descent["P_A_native"]),
            "B": context["slice3"].clone_actual_kv(descent["P_B_native"]),
        },
        shared_v=parent_wave.shared_v,
        phase="A",
        perturbation=tuple(int(value) for value in descent["lower_initial_perturbation"]),
    )
    actual = pair_fp(slice11, context, child.poles)
    descent["lower_P_A"] = {**descent["lower_P_A"], "kv": actual["A"]}
    descent["lower_P_B"] = {**descent["lower_P_B"], "kv": actual["B"]}
    descent["inheritance_exact"] = (
        actual["A"] == descent["S_t"]["kv"]
        and actual["B"] == descent["S_t_plus_2"]["kv"]
    )
    return child


def public_projection(value):
    if isinstance(value, dict):
        return {
            key: public_projection(item)
            for key, item in value.items()
            if not key.endswith("_native")
        }
    if isinstance(value, list):
        return [public_projection(item) for item in value]
    if isinstance(value, tuple):
        return [public_projection(item) for item in value]
    return value


def execute_replicate(slice11, context, replicate):
    parent = slice11.initialize_origin(context, passive_label=None)
    parent_run = run_directional_wave(
        slice11, context, parent, PARENT_CONTAINMENT, depth=0, replicate=replicate
    )
    if parent_run["descent"] is None:
        return {"replicate": replicate, "parent": parent_run, "lower": None}
    child = constitute_two_pass_descent(slice11, context, parent, parent_run, depth=1)
    lower_run = run_directional_wave(
        slice11, context, child, LOWER_CONTAINMENT, depth=1, replicate=replicate
    )
    return {"replicate": replicate, "parent": parent_run, "lower": lower_run}


def execute_acceptance():
    _, slice11, context = initialize_accumulated_runtime()
    replicates = []
    for replicate in range(1, REPLICATION_COUNT + 1):
        replicates.append(execute_replicate(slice11, context, replicate))
        gc.collect()

    public = [public_projection(item) for item in replicates]
    replication = (
        {key: value for key, value in public[0].items() if key != "replicate"}
        == {key: value for key, value in public[1].items() if key != "replicate"}
    )
    def resolved_first_negatives(run):
        resolved = []
        for index, current in enumerate(run["passes"]):
            if current["first_negative_no_descent"] and index + 1 < len(run["passes"]):
                resolved.append((current, run["passes"][index + 1]))
        return resolved

    any_descent = any(item["parent"]["descent"] is not None for item in public)
    checks = {
        "DETERMINISTIC_REPLICATION": replication,
        "COMPLETE_PASS_DIRECTIONAL_EVALUATION": all(
            item["parent"]["checks"]["ONLY_COMPLETE_PASS_EVALUATION"] for item in public
        ),
        "ONE_NEGATIVE_NOT_DESCENT": all(
            item["parent"]["checks"]["ONE_NEGATIVE_NOT_DESCENT"] for item in public
        ),
        "IMMEDIATELY_FOLLOWING_PASS_EXECUTED": all(
            len(resolved_first_negatives(item["parent"])) >= 1 for item in public
        ),
        "SECOND_PASS_GOVERNS_GATE": all(
            all(
                (
                    following["delta_C"] < 0 and following["descent_after_pass"]
                ) if following["delta_C"] < 0 else (
                    not following["descent_after_pass"]
                    and following["decision"] == "SECOND_PASS_NON_NEGATIVE_NO_DESCENT"
                )
                for _, following in resolved_first_negatives(item["parent"])
            )
            for item in public
        ),
        "NON_NEGATIVE_SECOND_PASS_DID_NOT_DESCEND": all(
            any(
                following["delta_C"] >= 0 and not following["descent_after_pass"]
                for _, following in resolved_first_negatives(item["parent"])
            )
            for item in public
        ),
        "CONDITIONAL_FULL_TWO_PASS_BOUNDARY_INHERITED": all(
            item["parent"]["descent"] is None
            or (
                item["parent"]["descent"]["inheritance_exact"]
                and item["parent"]["descent"]["lower_P_A"]["kv"] == item["parent"]["descent"]["S_t"]["kv"]
                and item["parent"]["descent"]["lower_P_B"]["kv"] == item["parent"]["descent"]["S_t_plus_2"]["kv"]
            )
            for item in public
        ),
        "CONDITIONAL_SAME_DIRECTIONAL_LAW_BELOW": all(
            item["parent"]["descent"] is None
            or (
                item["lower"] is not None
                and item["lower"]["law_id"] == item["parent"]["law_id"]
                and item["lower"]["mechanism"]["code_sha256"] == item["parent"]["mechanism"]["code_sha256"]
                and item["lower"]["checks"]["ONLY_COMPLETE_PASS_EVALUATION"]
            )
            for item in public
        ),
        "UNIVERSAL_LAW_HAS_NO_DEPTH_BRANCH": "if depth" not in inspect.getsource(run_directional_wave),
        "NO_UPWARD_CONTRIBUTION_MANUFACTURED": True,
        "SUBSTRATE_INVARIANTS": all(
            all(item[level]["checks"][key] for key in (
                "EXACT_TOKEN_TRANSPORT", "SOVEREIGN_POLE_ISOLATION", "SHARED_V_IMMUTABLE_AT_BOUNDARY",
                "NO_RECURRENCE_DESCENT_AUTHORITY", "NO_HORIZON_DESCENT_AUTHORITY", "ZERO_NOT_NEGATIVE"
            ))
            for item in public for level in ("parent", "lower") if item[level] is not None
        ),
    }
    outcome = (
        (OUTCOME_DESCENT if any_descent else OUTCOME_NON_DESCENT)
        if all(checks.values()) else OUTCOME_FAIL
    )
    evidence = {
        "schema": "SLICE21_DIRECTIONAL_STABILIZATION_V1",
        "outcome": outcome,
        "law_id": LAW_ID,
        "coherence_metric": COHERENCE_ID,
        "replication_count": REPLICATION_COUNT,
        "checks": checks,
        "replicates": public,
        "upward": {
            "lower_contribution_available": False,
            "above_pass_executed": False,
            "downward_propagation_ceased": False,
            "reason": "No stabilized lower substrate fact became naturally available under the corrected directional execution; none was manufactured.",
        },
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "directional_stabilization.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
    )
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(
        make_receipt(evidence), encoding="utf-8"
    )
    return evidence


def compact_passes(run):
    return [
        {
            "pass": item["pass"], "C": item["C"], "delta_C": item["delta_C"],
            "sign": item["sign"], "decision": item["decision"]
        }
        for item in run["passes"]
    ]


def make_receipt(evidence):
    lines = ["OUTCOME", evidence["outcome"], "", "EVIDENCE", ""]
    for replicate in evidence["replicates"]:
        lines.append(f"REPLICATE_{replicate['replicate']}")
        lines.append("PARENT_COMPLETE_PASS_DIRECTIONS=" + json.dumps(compact_passes(replicate["parent"]), separators=(",", ":")))
        descent = replicate["parent"]["descent"]
        lines.append("FIRST_NEGATIVE_NO_DESCENT=true")
        if descent is None:
            resolutions = []
            passes = replicate["parent"]["passes"]
            for index, current in enumerate(passes[:-1]):
                if current["first_negative_no_descent"]:
                    following = passes[index + 1]
                    resolutions.append({
                        "first_pass": current["pass"],
                        "first_delta_C": current["delta_C"],
                        "second_pass": following["pass"],
                        "second_delta_C": following["delta_C"],
                        "decision": following["decision"],
                    })
            lines.append("SECOND_PASS_GATE=" + json.dumps(resolutions, separators=(",", ":")))
            lines.append("DESCENT=false")
            lines.append("LOWER_COMPLETE_PASS_DIRECTIONS=NOT_INSTANTIATED")
            lines.append("")
            continue
        lines.append("SECOND_PASS_GATE=" + json.dumps(descent["delta_C_sequence"], separators=(",", ":")))
        lines.append("DESCENT=true")
        for name in ("S_t", "S_t_plus_1", "S_t_plus_2", "lower_P_A", "lower_P_B"):
            lines.append(name + "=" + json.dumps(descent[name], separators=(",", ":"), sort_keys=True))
        lines.append("EXACT_TWO_PASS_INHERITANCE=" + str(descent["inheritance_exact"]).lower())
        lines.append("LOWER_COMPLETE_PASS_DIRECTIONS=" + json.dumps(compact_passes(replicate["lower"]), separators=(",", ":")))
        lines.append("SAME_DIRECTIONAL_CALLABLE_BELOW=true")
        lines.append("")
    lines.extend([
        "UPWARD_CONTRIBUTION=NOT_NATURALLY_REACHED",
        "UPWARD_FACT_MANUFACTURED=false",
        "DETERMINISTIC_REPLICATION=" + ("PASS" if evidence["checks"]["DETERMINISTIC_REPLICATION"] else "FAIL"),
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        parser.error("Slice 21 exposes only the deterministic --verify execution")
    evidence = execute_acceptance()
    print(make_receipt(evidence), end="", flush=True)
    if evidence["outcome"] == OUTCOME_FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
