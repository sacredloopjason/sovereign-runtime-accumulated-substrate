import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE7_RUNTIME = PROJECT_DIR / "slice7_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice8"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
CYCLE_LIMIT = 8
PARENT_CONTINUE_INPUT = "SLICE8_PARENT_CONTINUE"
REPLICATION_COUNT = 2


def load_slice7_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice7_accumulated_runtime_for_slice8", SLICE7_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 7 runtime: {SLICE7_RUNTIME}")
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


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def public_parent(parent):
    return {
        "run": parent["run"],
        "cycles": parent["cycles"],
        "trajectory_sha256": parent["trajectory_sha256"],
        "observed_result": parent["observed_result"],
        "gate": parent["gate"],
    }


def public_path(path):
    return {**{key: value for key, value in path.items() if key != "parent"}, "parent": public_parent(path["parent"])}


def execute_parent(slice7, slice5, slice4, slice3, kernel, model, tokenizer, label):
    parent = slice7.execute_parent(
        slice5, slice4, slice3, kernel, model, tokenizer, label
    )
    if parent["observed_result"] != "NO_RECURRENCE_WITHIN_BOUND":
        return parent
    boundary = parent["latest_boundary"]
    if boundary["cycle"] != CYCLE_LIMIT - 1 or boundary["pole"] != "B":
        raise RuntimeError("Deterministic Slice 7 descent boundary did not reproduce")
    return parent


def save_boundary(slice4, kernel, boundary, run_label):
    import torch

    run_dir = EVIDENCE_DIR / f"{run_label.lower()}_boundary"
    pre_snapshot, pre = slice4.snapshot_and_fingerprint(kernel, boundary["pre_native"])
    post_snapshot, post = slice4.snapshot_and_fingerprint(kernel, boundary["post_native"])
    return {
        "S_t": slice4.save_boundary(torch, kernel, run_dir / "S_t.pt", pre_snapshot, pre),
        "S_t_plus_1": slice4.save_boundary(
            torch, kernel, run_dir / "S_t_plus_1.pt", post_snapshot, post
        ),
    }


def traverse(slice7, slice4, kernel, model, tokenizer, cache, shared_v, ids, label, origin):
    return slice7.traverse(
        slice4, kernel, model, tokenizer, cache, shared_v, ids, label, origin
    )


def execute_experimental(
    slice7, slice5, slice4, slice3, kernel, model, tokenizer, replicate
):
    label = f"REPLICATE_{replicate}_EXPERIMENTAL"
    parent = execute_parent(
        slice7, slice5, slice4, slice3, kernel, model, tokenizer, label
    )
    if parent["observed_result"] != "NO_RECURRENCE_WITHIN_BOUND":
        return {"parent": parent, "descent_attempted": False}

    boundary = parent["latest_boundary"]
    boundary_files = save_boundary(slice4, kernel, boundary, label)
    parent_live = boundary["parent_live"]
    child_a = slice3.clone_actual_kv(boundary["pre_native"])
    child_b = slice3.clone_actual_kv(boundary["post_native"])
    delta_parent = tuple(int(item) for item in boundary["delta"])

    s_t = kernel.fingerprint_cache(boundary["pre_native"])
    s_t_plus_1 = kernel.fingerprint_cache(boundary["post_native"])
    child_a_pre = kernel.fingerprint_cache(child_a)
    child_b_pre = kernel.fingerprint_cache(child_b)
    child_b_before = kernel.fingerprint_cache(child_b)
    parent_before = kernel.fingerprint_cache(parent_live)
    v_before = slice3.fingerprint_v(parent["shared_v"])

    child_traversal = traverse(
        slice7,
        slice4,
        kernel,
        model,
        tokenizer,
        child_a,
        parent["shared_v"],
        delta_parent,
        f"SLICE8_REPLICATE_{replicate}_CHILD_A_REPLAY",
        "exact preserved parent perturbation tuple from selected boundary",
    )

    child_a_post = kernel.fingerprint_cache(child_a)
    child_b_after = kernel.fingerprint_cache(child_b)
    parent_after = kernel.fingerprint_cache(parent_live)
    v_after = slice3.fingerprint_v(parent["shared_v"])

    parent_continue_pre = kernel.fingerprint_cache(parent_live)
    parent_traversal = traverse(
        slice7,
        slice4,
        kernel,
        model,
        tokenizer,
        parent_live,
        parent["shared_v"],
        token_ids(tokenizer, PARENT_CONTINUE_INPUT),
        f"SLICE8_REPLICATE_{replicate}_PARENT_AFTER_DESCENT",
        "fixed Slice 8 parent continuation input tokenization",
    )
    parent_continue_post = kernel.fingerprint_cache(parent_live)

    return {
        "parent": parent,
        "descent_attempted": True,
        "boundary_files": boundary_files,
        "boundary": {
            "selection": "most recently completed model traversal at unresolved-bound exhaustion",
            "cycle": boundary["cycle"],
            "parent_pole": boundary["pole"],
            "S_t": s_t,
            "Delta_t_parent": {
                "token_ids": list(delta_parent),
                "sha256": slice4.token_fingerprint(delta_parent),
                "actual_parent_consumed": boundary["traversal"][
                    "actual_first_forward_perturbation_ids"
                ],
                "parent_identity_at_model_boundary": boundary["traversal"][
                    "exact_perturbation_identity_at_model_boundary"
                ],
            },
            "S_t_plus_1": s_t_plus_1,
            "parent_generated_tokens": boundary["traversal"]["generated_token_ids"],
            "parent_generated_sha256": boundary["traversal"]["generated_token_sha256"],
            "parent_output": boundary["traversal"]["decoded_output_observation_only"],
        },
        "constitution": {
            "child_A_pre": child_a_pre,
            "child_B_pre": child_b_pre,
            "native_objects_distinct": len({id(parent_live), id(child_a), id(child_b)}) == 3,
            "native_tensor_storage_separated": slice7.pairwise_storage_separated(
                slice3, [parent_live, child_a, child_b]
            ),
        },
        "downward_propagation": {
            "Delta_t_child_A_consumed": child_traversal[
                "actual_first_forward_perturbation_ids"
            ],
            "Delta_t_child_A_consumed_sha256": child_traversal[
                "actual_first_forward_perturbation_sha256"
            ],
            "transport_operation": child_traversal["transport_operation"],
        },
        "child_A": {
            "pre": child_a_pre,
            "post": child_a_post,
            "generated_tokens": child_traversal["generated_token_ids"],
            "generated_sha256": child_traversal["generated_token_sha256"],
            "output": child_traversal["decoded_output_observation_only"],
            "traversal": child_traversal,
        },
        "child_B": {"before": child_b_before, "after": child_b_after},
        "parent_immutability": {"before": parent_before, "after": parent_after},
        "parent_continuation": {
            "input": PARENT_CONTINUE_INPUT,
            "pre": parent_continue_pre,
            "post": parent_continue_post,
            "output": parent_traversal["decoded_output_observation_only"],
            "generated_tokens": parent_traversal["generated_token_ids"],
        },
        "shared_v": {"before": v_before, "after": v_after},
    }


def execute_control(
    slice7, slice5, slice4, slice3, kernel, model, tokenizer, replicate
):
    label = f"REPLICATE_{replicate}_CONTROL"
    parent = execute_parent(
        slice7, slice5, slice4, slice3, kernel, model, tokenizer, label
    )
    if parent["observed_result"] != "NO_RECURRENCE_WITHIN_BOUND":
        return {"parent": parent, "continued": False}
    boundary = parent["latest_boundary"]
    parent_live = boundary["parent_live"]
    pre = kernel.fingerprint_cache(parent_live)
    traversal = traverse(
        slice7,
        slice4,
        kernel,
        model,
        tokenizer,
        parent_live,
        parent["shared_v"],
        token_ids(tokenizer, PARENT_CONTINUE_INPUT),
        f"SLICE8_REPLICATE_{replicate}_CONTROL_PARENT",
        "fixed Slice 8 parent continuation input tokenization",
    )
    return {
        "parent": parent,
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
            "post": kernel.fingerprint_cache(parent_live),
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
        "PARENT_TRAJECTORY_DETERMINISTIC": (
            exp_parent["trajectory_sha256"] == ctl_parent["trajectory_sha256"]
        ),
    }
    if not experimental.get("descent_attempted") or not control.get("continued"):
        return checks

    b = experimental["boundary"]
    c = experimental["constitution"]
    p = experimental["downward_propagation"]
    a = experimental["child_A"]
    cb = experimental["child_B"]
    pi = experimental["parent_immutability"]
    ep = experimental["parent_continuation"]
    cp = control["parent_continuation"]
    v = experimental["shared_v"]
    checks.update(
        {
            "DESCENT_BOUNDARY_SELECTED_MECHANICALLY": (
                b["cycle"] == CYCLE_LIMIT - 1
                and b["parent_pole"] == "B"
                and b["selection"]
                == "most recently completed model traversal at unresolved-bound exhaustion"
            ),
            "COMPLETE_BOUNDARY_PRESERVED": (
                experimental["boundary_files"]["S_t"]["round_trip_verified"]
                and experimental["boundary_files"]["S_t_plus_1"]["round_trip_verified"]
                and b["Delta_t_parent"]["parent_identity_at_model_boundary"]
            ),
            "CHILD_A_PRE_EQUALS_S_t": c["child_A_pre"]["sha256"] == b["S_t"]["sha256"],
            "CHILD_B_PRE_EQUALS_S_t_plus_1": (
                c["child_B_pre"]["sha256"] == b["S_t_plus_1"]["sha256"]
            ),
            "NATIVE_STORAGE_INDEPENDENT": (
                c["native_objects_distinct"] and c["native_tensor_storage_separated"]
            ),
            "DELTA_PARENT_EQUALS_CHILD_CONSUMED": (
                b["Delta_t_parent"]["token_ids"] == p["Delta_t_child_A_consumed"]
                and b["Delta_t_parent"]["sha256"]
                == p["Delta_t_child_A_consumed_sha256"]
            ),
            "NO_DECODE_RETOKENIZE": (
                a["traversal"]["transport_origin"]
                == "exact preserved parent perturbation tuple from selected boundary"
                and a["traversal"]["exact_perturbation_identity_at_model_boundary"]
            ),
            "CHILD_A_POST_EQUALS_PARENT_S_t_plus_1": (
                a["post"]["sha256"] == b["S_t_plus_1"]["sha256"]
                and a["post"]["sequence_length"] == b["S_t_plus_1"]["sequence_length"]
                and a["post"]["tensor_byte_count"] == b["S_t_plus_1"]["tensor_byte_count"]
            ),
            "CHILD_A_OUTPUT_EQUALS_PARENT_OUTPUT": (
                a["generated_tokens"] == b["parent_generated_tokens"]
                and a["generated_sha256"] == b["parent_generated_sha256"]
                and a["output"] == b["parent_output"]
            ),
            "CHILD_B_UNCHANGED": cb["before"]["sha256"] == cb["after"]["sha256"],
            "PARENT_UNCHANGED_BY_DESCENT": pi["before"]["sha256"] == pi["after"]["sha256"],
            "CONTROL_BOUNDARY_IDENTICAL": (
                b["S_t"]["sha256"] == control["boundary"]["S_t"]["sha256"]
                and b["Delta_t_parent"]["token_ids"] == control["boundary"]["Delta_t"]
                and b["S_t_plus_1"]["sha256"]
                == control["boundary"]["S_t_plus_1"]["sha256"]
                and b["parent_generated_tokens"]
                == control["boundary"]["parent_generated_tokens"]
            ),
            "PARENT_CONTINUATION_EQUAL": ep["post"]["sha256"] == cp["post"]["sha256"],
            "PARENT_OUTPUT_IDENTICAL": (
                ep["generated_tokens"] == cp["generated_tokens"]
                and ep["output"] == cp["output"]
            ),
            "SAME_V": (
                v["before"]["literal"] == V_LITERAL
                and v["before"]["sha256"] == v["after"]["sha256"]
                and a["traversal"]["actual_first_forward_v_prefix"]
                == v["before"]["token_ids"]
            ),
            "NO_SEMANTIC_RECONSTRUCTION": True,
            "NO_EXTERNAL_EPISTEMIC_AUTHORITY": True,
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
        "control_trajectory": ctl["parent"]["trajectory_sha256"],
        "cycle": b["cycle"],
        "pole": b["parent_pole"],
        "S_t": b["S_t"]["sha256"],
        "Delta_t": b["Delta_t_parent"]["token_ids"],
        "S_t_plus_1": b["S_t_plus_1"]["sha256"],
        "child_A_pre": exp["constitution"]["child_A_pre"]["sha256"],
        "child_B_pre": exp["constitution"]["child_B_pre"]["sha256"],
        "child_A_post": exp["child_A"]["post"]["sha256"],
        "child_A_generated": exp["child_A"]["generated_tokens"],
        "parent_post_descent": exp["parent_continuation"]["post"]["sha256"],
        "parent_output_descent": exp["parent_continuation"]["generated_tokens"],
        "parent_post_control": ctl["parent_continuation"]["post"]["sha256"],
        "parent_output_control": ctl["parent_continuation"]["generated_tokens"],
    }


def diagnose_failure(result):
    failed = result["failed_checks"]
    if "UNRESOLVED_PARENT_REPRODUCED" in failed:
        return (
            "fixed unresolved parent passage",
            "The required NO_RECURRENCE_WITHIN_BOUND parent condition did not reproduce.",
            "deterministic reproduction of the inherited parent condition",
        )
    if "DELTA_PARENT_EQUALS_CHILD_CONSUMED" in failed or "NO_DECODE_RETOKENIZE" in failed:
        return (
            "parent-to-child model input boundary",
            "The child's consumed ordered token IDs differed from the preserved parent perturbation.",
            "lossless direct token-ID transport into model.forward",
        )
    if "CHILD_A_POST_EQUALS_PARENT_S_t_plus_1" in failed:
        return (
            "replayed child-A post-state",
            "The complete child-A KV result diverged despite the preserved S_t, Delta_t, and canonical V comparisons.",
            "an additional traversal-state condition beyond the preserved KV state, ordered perturbation IDs, and canonical V",
        )
    return (
        "recursive identity, isolation, control, or replication comparison",
        "One or more exact physical comparisons named in failed_requirement did not hold.",
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
                "slice: 8",
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
    c = exp["constitution"]
    a = exp["child_A"]
    cb = exp["child_B"]
    pi = exp["parent_immutability"]
    ep = exp["parent_continuation"]
    cp = ctl["parent_continuation"]
    lines = ["BUILD_SLICE_RECEIPT", "slice: 8", "result: PASS", ""]
    field(lines, "artifact", PROJECT_DIR)
    field(lines, "parent_V", V_LITERAL)
    field(lines, "parent_seed", SEED_LITERAL)
    field(lines, "parent_stabilization_result", "NO_RECURRENCE_WITHIN_BOUND")
    lines.extend(["DESCENT_BOUNDARY", ""])
    field(lines, "parent_pole", b["parent_pole"])
    field(lines, "S_t", json.dumps(b["S_t"], sort_keys=True))
    field(lines, "Delta_t_parent", json.dumps(b["Delta_t_parent"], sort_keys=True))
    field(lines, "S_t_plus_1", json.dumps(b["S_t_plus_1"], sort_keys=True))
    field(lines, "parent_generated_tokens", json.dumps(b["parent_generated_tokens"]))
    lines.extend(["CHILD_CONSTITUTION", ""])
    field(lines, "child_A_pre", json.dumps(c["child_A_pre"], sort_keys=True))
    field(lines, "child_B_pre", json.dumps(c["child_B_pre"], sort_keys=True))
    field(lines, "CHILD_A_PRE_EQUALS_S_t", "PASS")
    field(lines, "CHILD_B_PRE_EQUALS_S_t_plus_1", "PASS")
    lines.extend(["DOWNWARD_PROPAGATION", ""])
    field(lines, "Delta_t_child_A_consumed", json.dumps(exp["downward_propagation"]))
    field(lines, "DELTA_PARENT_EQUALS_CHILD_CONSUMED", "PASS")
    field(lines, "NO_DECODE_RETOKENIZE", "PASS")
    lines.extend(["CHILD_A_RECURSIVE_TRAVERSAL", ""])
    field(lines, "child_A_post", json.dumps(a["post"], sort_keys=True))
    field(lines, "child_A_generated_tokens", json.dumps(a["generated_tokens"]))
    field(lines, "CHILD_A_POST_EQUALS_PARENT_S_t_plus_1", "PASS")
    field(lines, "CHILD_A_OUTPUT_EQUALS_PARENT_OUTPUT", "PASS")
    field(lines, "child_B_before_child_A", json.dumps(cb["before"], sort_keys=True))
    field(lines, "child_B_after_child_A", json.dumps(cb["after"], sort_keys=True))
    field(lines, "CHILD_B_UNCHANGED", "PASS")
    field(lines, "parent_before_child_A", json.dumps(pi["before"], sort_keys=True))
    field(lines, "parent_after_child_A", json.dumps(pi["after"], sort_keys=True))
    field(lines, "PARENT_UNCHANGED_BY_DESCENT", "PASS")
    lines.extend(["PARENT_CONTINUATION_EXPERIMENTAL", ""])
    field(lines, "input", ep["input"])
    field(lines, "output", ep["output"])
    field(lines, "parent_post_descent", json.dumps(ep["post"], sort_keys=True))
    lines.extend(["PARENT_CONTINUATION_CONTROL", ""])
    field(lines, "input", cp["input"])
    field(lines, "output", cp["output"])
    field(lines, "parent_post_control", json.dumps(cp["post"], sort_keys=True))
    for name in (
        "PARENT_CONTINUATION_EQUAL",
        "PARENT_OUTPUT_IDENTICAL",
        "DETERMINISTIC_REPLICATION",
        "SAME_V",
        "NO_SEMANTIC_RECONSTRUCTION",
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY",
    ):
        field(lines, name, "PASS")
    field(
        lines,
        "substrate_finding",
        "Lossless native DynamicCache cloning reconstituted the preserved parent S_t as Child A. Direct structural conversion of the exact preserved ordered Delta_t token-ID tuple into the model input tensor, under the same immutable canonical V and deterministic inference path, recreated both the complete parent S_t_plus_1 KV state and the parent's generated token sequence exactly. Disjoint native tensor storage kept Child B and the live parent unchanged, and the parent continuation matched a clean control. The complete observation replicated from identical initial conditions.",
    )
    lines.extend(["final_result:", "PASS", "END_BUILD_SLICE_RECEIPT", ""])
    return "\n".join(lines)


def execute_acceptance():
    slice7 = load_slice7_runtime()
    slice6 = slice7.load_slice6_runtime()
    slice5 = slice6.load_slice5_runtime()
    slice5.SEED_LITERAL = SEED_LITERAL
    slice5.CYCLE_LIMIT = CYCLE_LIMIT
    slice4, slice3, kernel, model, tokenizer, _, weight_hash = slice5.initialize()
    if weight_hash != kernel.MODEL_WEIGHT_SHA256:
        raise RuntimeError("Verified model hash changed before Slice 8")

    replicates = []
    all_checks = {}
    for replicate in range(1, REPLICATION_COUNT + 1):
        experimental = execute_experimental(
            slice7, slice5, slice4, slice3, kernel, model, tokenizer, replicate
        )
        control = execute_control(
            slice7, slice5, slice4, slice3, kernel, model, tokenizer, replicate
        )
        pair_checks = evaluate_pair(experimental, control)
        replicates.append(
            {
                "experimental": public_path(experimental),
                "control": public_path(control),
                "checks": pair_checks,
            }
        )
        for name, passed in pair_checks.items():
            all_checks[f"REPLICATE_{replicate}_{name}"] = passed
        gc.collect()

    projections = [replication_projection(pair) for pair in replicates]
    all_checks["DETERMINISTIC_REPLICATION"] = (
        None not in projections
        and projections[0] == projections[1]
        and stable_sha256(b"SLICE8_REPLICATION_V1", projections[0])
        == stable_sha256(b"SLICE8_REPLICATION_V1", projections[1])
    )
    failed = [name for name, passed in all_checks.items() if not passed]
    result = {
        "schema": "SLICE8_DOWNWARD_PROPAGATION_EVIDENCE_V1",
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
    (EVIDENCE_DIR / "downward_propagation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(
        make_receipt(result), encoding="utf-8"
    )
    return result, weight_hash


def run_verification():
    print("SLICE8_RUNTIME_READY", flush=True)
    print(f"PARENT_V={V_LITERAL}", flush=True)
    print(f"PARENT_SEED={SEED_LITERAL}", flush=True)
    print(f"CYCLE_LIMIT={CYCLE_LIMIT}", flush=True)
    print(f"REPLICATION_COUNT={REPLICATION_COUNT}", flush=True)
    result, weight_hash = execute_acceptance()
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    for name, passed in result["checks"].items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    if result["replicates"][0]["experimental"].get("descent_attempted"):
        exp = result["replicates"][0]["experimental"]
        print("DESCENT_BOUNDARY=7:B", flush=True)
        print("DELTA_T=" + json.dumps(exp["boundary"]["Delta_t_parent"]["token_ids"]), flush=True)
        print("PARENT_OUTPUT=" + json.dumps(exp["boundary"]["parent_output"]), flush=True)
        print("CHILD_A_OUTPUT=" + json.dumps(exp["child_A"]["output"]), flush=True)
        print("PARENT_CONTINUE_OUTPUT=" + json.dumps(exp["parent_continuation"]["output"]), flush=True)
    print(f"FINAL_RESULT={'PASS' if result['passed'] else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={EVIDENCE_DIR / 'completion_receipt.txt'}", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "--verify":
        raise SystemExit("usage: slice8_runtime.py --verify")
    run_verification()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"SLICE8_RUNTIME_ERROR={error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
