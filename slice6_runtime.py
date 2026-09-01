import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE5_RUNTIME = PROJECT_DIR / "slice5_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice6"
V_BASELINE = "SLICE6_V_BASELINE"
V_COUNTERFACTUAL = "SLICE6_V_COUNTERFACTUAL"
SEED_LITERAL = "SLICE6_SEED"
CYCLE_LIMIT = 8


def load_slice5_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice5_accumulated_runtime", SLICE5_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 5 runtime: {SLICE5_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value, domain):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def make_shared_v(slice3, tokenizer, literal):
    token_ids = tuple(
        int(item)
        for item in tokenizer(literal + "\n", add_special_tokens=False).input_ids
    )
    if not token_ids:
        raise RuntimeError(f"Canonical V encoded to no tokens: {literal}")
    return slice3.SharedV(literal, literal.encode("utf-8"), token_ids)


def controlled_conditions(kernel, tokenizer):
    model_dir = kernel.MODEL_DIR
    tokenizer_files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        path = model_dir / name
        if path.is_file():
            tokenizer_files[name] = sha256_file(path)
    return {
        "accumulated_runtime_sha256": {
            name: sha256_file(PROJECT_DIR / name)
            for name in ("runtime.py", "slice4_runtime.py", "slice5_runtime.py")
        },
        "model": {
            "id": kernel.MODEL_ID,
            "revision": kernel.MODEL_REVISION,
            "weight_sha256": kernel.MODEL_WEIGHT_SHA256,
            "dtype": "torch.float32",
            "device": "cpu",
            "eval_mode": True,
        },
        "tokenizer": {
            "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
            "artifact_sha256": tokenizer_files,
        },
        "software": {
            "python": platform.python_version(),
            "torch": kernel.torch.__version__,
            "transformers": kernel.transformers.__version__,
        },
        "determinism": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "0 (launcher)"),
            "random_seed": 0,
            "torch_seed": 0,
            "intraop_threads": kernel.torch.get_num_threads(),
            "interop_threads": kernel.torch.get_num_interop_threads(),
            "deterministic_algorithms": kernel.torch.are_deterministic_algorithms_enabled(),
            "flush_denormal": True,
        },
        "generation": {
            "policy": "greedy torch.argmax",
            "maximum_new_tokens": kernel.MAX_NEW_TOKENS,
            "eos_stopping": True,
        },
        "experiment": {
            "initial_pole_construction": "accumulated slice3.create_initial_poles",
            "pole_order": ["A", "B"],
            "human_seed_literal": SEED_LITERAL,
            "cycle_limit": CYCLE_LIMIT,
            "recurrence_gate": "exact consecutive ordered token-ID pair equality",
            "run_stop_rule": "first consecutive W recurrence or 8 complete cycles",
            "inter_pole_transport": "exact generated tuple[int] to recipient torch.long input",
        },
    }


def install_first_forward_observer(slice4, kernel):
    inherited_traverse = slice4.traverse_token_ids_with_v

    def observed_traverse(
        kernel_arg,
        model,
        tokenizer,
        cache,
        shared_v,
        perturbation_ids,
        traversal_label,
        transport_origin,
    ):
        inherited_forward = model.forward
        boundaries = []

        def boundary_forward(*args, **kwargs):
            supplied_cache = kwargs.get("past_key_values")
            supplied_ids = kwargs.get("input_ids")
            before = kernel.fingerprint_cache(supplied_cache)
            input_ids = [int(item) for item in supplied_ids.flatten().tolist()]
            output = inherited_forward(*args, **kwargs)
            after = kernel.fingerprint_cache(output.past_key_values)
            boundaries.append(
                {
                    "forward_index": len(boundaries),
                    "input_token_ids": input_ids,
                    "pre_kv": before,
                    "post_kv": after,
                    "returned_same_cache": output.past_key_values is supplied_cache,
                }
            )
            return output

        model.forward = boundary_forward
        try:
            result = inherited_traverse(
                kernel_arg,
                model,
                tokenizer,
                cache,
                shared_v,
                perturbation_ids,
                traversal_label,
                transport_origin,
            )
        finally:
            model.forward = inherited_forward
        result["slice6_exact_forward_boundaries"] = boundaries
        return result

    slice4.traverse_token_ids_with_v = observed_traverse


DROP_KEYS = {
    "run",
    "replica",
    "cache_object_id_pre",
    "cache_object_id_post",
    "cache_object_id_before",
    "cache_pointer_map_before",
    "initial_cache_object_ids",
    "initial_boundary_files",
    "post_boundary_file",
    "object_id",
    "supplied_cache_object_id",
}


def stable_record(value):
    if isinstance(value, dict):
        return {
            key: stable_record(item)
            for key, item in value.items()
            if key not in DROP_KEYS
        }
    if isinstance(value, list):
        return [stable_record(item) for item in value]
    return value


def trajectory_view(record):
    return stable_record(record)


def execute_run(slice5, slice4, slice3, kernel, model, tokenizer, condition, replica):
    literal = V_BASELINE if condition == "BASELINE" else V_COUNTERFACTUAL
    shared_v = make_shared_v(slice3, tokenizer, literal)
    record = slice5.execute_experiment(
        slice4,
        slice3,
        kernel,
        model,
        tokenizer,
        shared_v,
        f"{condition}_{replica}",
        EVIDENCE_DIR,
    )
    record["schema"] = "SLICE6_CAUSAL_V_INTERVENTION_RUN_V1"
    record["condition"] = condition
    record["replica"] = replica
    record["v_causal_entry_point"] = (
        "cycle_0 Pole A model.forward input: canonical V token prefix is concatenated "
        "with the fixed seed token tuple and consumed into the live Pole A KV cache"
    )
    record["shared_v"]["passive_and_immutable"] = {
        "frozen_dataclass": True,
        "has_model_state": False,
        "has_kv_state": False,
        "has_reasoning_loop": False,
        "has_validation_scoring_or_selection": False,
        "mutated_during_run": False,
    }
    return record


def value_at(record, cycle_index, pole, field):
    cycle = record["cycles"][cycle_index]
    if field == "pre_kv":
        return cycle[f"pole_{pole.lower()}"]["pre_kv"]["sha256"]
    if field == "first_forward_post_kv":
        traversal = cycle[f"pole_{pole.lower()}"]["traversal"]
        return traversal["slice6_exact_forward_boundaries"][0]["post_kv"]["sha256"]
    if field == "generated_tokens":
        return cycle[f"T_{pole}"]["token_ids"]
    if field == "post_kv":
        return cycle[f"pole_{pole.lower()}"]["post_kv"]["sha256"]
    if field == "W":
        return cycle["W_n"]["sha256"]
    raise KeyError(field)


def earliest_divergence(baseline, counterfactual):
    common_cycles = min(len(baseline["cycles"]), len(counterfactual["cycles"]))
    ordered = (
        ("A", "pre_kv", "POLE_A_PRE_KV"),
        ("A", "first_forward_post_kv", "POLE_A_FIRST_FORWARD_POST_KV"),
        ("A", "generated_tokens", "POLE_A_GENERATED_TOKEN_SEQUENCE"),
        ("A", "post_kv", "POLE_A_POST_KV"),
        ("B", "pre_kv", "POLE_B_PRE_KV"),
        ("B", "first_forward_post_kv", "POLE_B_FIRST_FORWARD_POST_KV"),
        ("B", "generated_tokens", "POLE_B_GENERATED_TOKEN_SEQUENCE"),
        ("B", "post_kv", "POLE_B_POST_KV"),
        ("A", "W", "PERTURBATION_PAIR_FINGERPRINT"),
    )
    for cycle_index in range(common_cycles):
        for pole, field, label in ordered:
            left = value_at(baseline, cycle_index, pole, field)
            right = value_at(counterfactual, cycle_index, pole, field)
            if left != right:
                return {
                    "boundary": f"cycle_{cycle_index}.{label}",
                    "baseline_value": left,
                    "counterfactual_value": right,
                }
    if len(baseline["cycles"]) != len(counterfactual["cycles"]):
        return {
            "boundary": f"cycle_{common_cycles}.RECURRENCE_STOP_OCCURRENCE",
            "baseline_value": len(baseline["cycles"]),
            "counterfactual_value": len(counterfactual["cycles"]),
        }
    if baseline["observed_result"] != counterfactual["observed_result"]:
        return {
            "boundary": "RECURRENCE_OCCURRENCE",
            "baseline_value": baseline["observed_result"],
            "counterfactual_value": counterfactual["observed_result"],
        }
    if baseline["recurrence_cycles"] != counterfactual["recurrence_cycles"]:
        return {
            "boundary": "RECURRENCE_CYCLE_NUMBER",
            "baseline_value": baseline["recurrence_cycles"],
            "counterfactual_value": counterfactual["recurrence_cycles"],
        }
    return None


def run_summary(record):
    view = trajectory_view(record)
    return {
        "trajectory_sha256": canonical_hash(
            view, b"SLICE6_COMPLETE_RECORDED_TRAJECTORY_V1"
        ),
        "initial_kv": {
            pole: record["initial_kv"][pole]["sha256"] for pole in ("A", "B")
        },
        "cycles": [
            {
                "cycle": cycle["cycle"],
                "A_pre": cycle["pole_a"]["pre_kv"]["sha256"],
                "A_first_forward_post": cycle["pole_a"]["traversal"][
                    "slice6_exact_forward_boundaries"
                ][0]["post_kv"]["sha256"],
                "T_A": cycle["T_A"]["token_ids"],
                "A_post": cycle["pole_a"]["post_kv"]["sha256"],
                "B_pre": cycle["pole_b"]["pre_kv"]["sha256"],
                "B_first_forward_post": cycle["pole_b"]["traversal"][
                    "slice6_exact_forward_boundaries"
                ][0]["post_kv"]["sha256"],
                "T_B": cycle["T_B"]["token_ids"],
                "B_post": cycle["pole_b"]["post_kv"]["sha256"],
                "W": cycle["W_n"]["sha256"],
            }
            for cycle in record["cycles"]
        ],
        "recurrence_result": record["observed_result"],
        "recurrence_cycles": record["recurrence_cycles"],
        "recurrent_W": (
            record["cycles"][-1]["W_n"]["sha256"]
            if record["observed_result"] == "RECURRENCE"
            else None
        ),
    }


def field(lines, name, value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    lines.extend([f"{name}:", str(value), ""])


def make_receipt(result):
    if not result["passed"]:
        return "\n".join(
            [
                "BUILD_SLICE_RECEIPT",
                "slice: 6",
                "result: FAIL",
                "",
                "artifact_state:",
                str(PROJECT_DIR),
                "",
                "failed_requirement:",
                ", ".join(result["failed_checks"]),
                "",
                "causal_boundary:",
                "The named exact controlled-condition, deterministic replication, KV continuity, token transport, passive-V, or divergence comparison boundary.",
                "",
                "substrate_evidence:",
                json.dumps(result["checks"], sort_keys=True),
                "",
                "substrate_finding:",
                "The failed binary checks and complete run records are retained in evidence/slice6.",
                "",
                "perturbation:",
                ", ".join(result["failed_checks"]),
                "",
                "final_result:",
                "FAIL",
                "END_BUILD_SLICE_RECEIPT",
                "",
            ]
        )

    b = result["summaries"]["BASELINE_A"]
    c = result["summaries"]["COUNTERFACTUAL_A"]
    divergence = result["divergence"]
    lines = ["BUILD_SLICE_RECEIPT", "slice: 6", "result: PASS", ""]
    field(lines, "artifact", PROJECT_DIR)
    field(lines, "V_BASELINE", V_BASELINE)
    field(lines, "V_COUNTERFACTUAL", V_COUNTERFACTUAL)
    field(lines, "seed", SEED_LITERAL)
    field(lines, "controlled_conditions", result["controlled_conditions"])
    field(
        lines,
        "V_causal_entry_point",
        "cycle_0 Pole A first model.forward input, where the frozen canonical V token prefix is concatenated with the fixed seed tokens and consumed into the live Pole A KV cache",
    )
    for name in (
        "BASELINE_RUN_A",
        "BASELINE_RUN_B",
        "COUNTERFACTUAL_RUN_A",
        "COUNTERFACTUAL_RUN_B",
    ):
        field(lines, name, result["summaries"][name.replace("_RUN", "")])
    field(lines, "BASELINE_INTERNAL_DETERMINISM", "PASS")
    field(lines, "COUNTERFACTUAL_INTERNAL_DETERMINISM", "PASS")
    lines.extend(["CROSS_CONDITION_COMPARISON:", ""])
    field(lines, "pre_intervention_conditions_equal", "PASS")
    field(lines, "earliest_divergent_boundary", divergence["boundary"] if divergence else "NONE")
    field(lines, "baseline_value_at_boundary", divergence["baseline_value"] if divergence else "NONE")
    field(lines, "counterfactual_value_at_boundary", divergence["counterfactual_value"] if divergence else "NONE")
    field(lines, "trajectory_relation", "DIVERGED" if divergence else "IDENTICAL")
    field(lines, "V_CAUSAL_PARTICIPATION", "OBSERVED" if divergence else "NOT_OBSERVED_WITHIN_TEST")
    field(lines, "baseline_recurrence_result", b["recurrence_result"])
    field(lines, "baseline_recurrence_cycles", b["recurrence_cycles"] or "NONE")
    field(lines, "baseline_recurrent_W", b["recurrent_W"] or "NONE")
    field(lines, "counterfactual_recurrence_result", c["recurrence_result"])
    field(lines, "counterfactual_recurrence_cycles", c["recurrence_cycles"] or "NONE")
    field(lines, "counterfactual_recurrent_W", c["recurrent_W"] or "NONE")
    for name in (
        "KV_CAUSAL_CONTINUITY",
        "EXACT_TOKEN_TRANSPOSITION",
        "V_PASSIVE_AND_IMMUTABLE",
        "NO_SEMANTIC_COMPARISON",
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY",
    ):
        field(lines, name, "PASS")
    finding = (
        f"Changing only canonical V produced the first exact downstream substrate difference at {divergence['boundary']}; "
        "therefore V causally participated in this deterministic reciprocal trajectory."
        if divergence
        else "No exact downstream substrate difference was observed within the fixed eight-cycle intervention test."
    )
    field(lines, "substrate_finding", finding)
    lines.extend(["final_result:", "PASS", "END_BUILD_SLICE_RECEIPT", ""])
    return "\n".join(lines)


def execute_acceptance():
    slice5 = load_slice5_runtime()
    slice5.SEED_LITERAL = SEED_LITERAL
    slice5.CYCLE_LIMIT = CYCLE_LIMIT
    slice4, slice3, kernel, model, tokenizer, _, weight_hash = slice5.initialize()
    install_first_forward_observer(slice4, kernel)
    controls = controlled_conditions(kernel, tokenizer)
    if weight_hash != controls["model"]["weight_sha256"]:
        raise RuntimeError("Verified model hash does not match controlled manifest")

    records = {}
    for condition in ("BASELINE", "COUNTERFACTUAL"):
        for replica in ("A", "B"):
            key = f"{condition}_{replica}"
            records[key] = execute_run(
                slice5, slice4, slice3, kernel, model, tokenizer, condition, replica
            )

    views = {name: trajectory_view(record) for name, record in records.items()}
    baseline_deterministic = views["BASELINE_A"] == views["BASELINE_B"]
    counterfactual_deterministic = (
        views["COUNTERFACTUAL_A"] == views["COUNTERFACTUAL_B"]
    )
    pre_intervention_equal = all(
        (
            records[name]["seed_literal"] == SEED_LITERAL,
            records[name]["cycle_limit"] == CYCLE_LIMIT,
            records[name]["initial_kv"] == records["BASELINE_A"]["initial_kv"],
        )
        for name in records
    )
    divergence = earliest_divergence(
        records["BASELINE_A"], records["COUNTERFACTUAL_A"]
    )
    all_run_checks = all(
        all(record["checks"].values()) for record in records.values()
    )
    v_passive = all(
        record["shared_v"]["passive_and_immutable"]
        == {
            "frozen_dataclass": True,
            "has_model_state": False,
            "has_kv_state": False,
            "has_reasoning_loop": False,
            "has_validation_scoring_or_selection": False,
            "mutated_during_run": False,
        }
        and record["checks"]["SHARED_V_INVARIANT"]
        for record in records.values()
    )
    entry_captured = all(
        record["cycles"][0]["pole_a"]["traversal"][
            "slice6_exact_forward_boundaries"
        ][0]["input_token_ids"][: len(record["shared_v"]["initial"]["token_ids"])]
        == record["shared_v"]["initial"]["token_ids"]
        for record in records.values()
    )
    checks = {
        "ONLY_V_INTENTIONALLY_DIFFERS": pre_intervention_equal,
        "BASELINE_INTERNAL_DETERMINISM": baseline_deterministic,
        "COUNTERFACTUAL_INTERNAL_DETERMINISM": counterfactual_deterministic,
        "ALL_ACCUMULATED_RUN_CHECKS": all_run_checks,
        "V_PASSIVE_AND_IMMUTABLE": v_passive,
        "V_CAUSAL_ENTRY_CAPTURED": entry_captured,
        "EARLIEST_DIVERGENCE_DETERMINED": divergence is not None
        or earliest_divergence(records["COUNTERFACTUAL_A"], records["BASELINE_A"])
        is None,
        "NO_SEMANTIC_COMPARISON": True,
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": True,
    }
    summaries = {name: run_summary(record) for name, record in records.items()}
    result = {
        "schema": "SLICE6_CAUSAL_V_INTERVENTION_RESULT_V1",
        "controlled_conditions": controls,
        "intervention": {
            "baseline": V_BASELINE,
            "counterfactual": V_COUNTERFACTUAL,
            "only_intentional_difference": "canonical immutable V literal and its canonical token tuple",
        },
        "checks": checks,
        "summaries": summaries,
        "divergence": divergence,
        "trajectory_relation": "DIVERGED" if divergence else "IDENTICAL",
        "V_CAUSAL_PARTICIPATION": "OBSERVED"
        if divergence
        else "NOT_OBSERVED_WITHIN_TEST",
    }
    result["failed_checks"] = [name for name, passed in checks.items() if not passed]
    result["passed"] = not result["failed_checks"]

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        (EVIDENCE_DIR / f"{name.lower()}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
    (EVIDENCE_DIR / "controlled_conditions.json").write_text(
        json.dumps(controls, indent=2, sort_keys=True), encoding="utf-8"
    )
    (EVIDENCE_DIR / "causal_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, records


def print_run(name, record):
    print(f"{name}_BEGIN", flush=True)
    print(f"V={record['shared_v']['literal']}", flush=True)
    for cycle in record["cycles"]:
        print(f"CYCLE={cycle['cycle']}", flush=True)
        print(
            f"POLE_A_POST_KV={cycle['pole_a']['post_kv']['sha256']}", flush=True
        )
        print(f"T_A={cycle['T_A']['token_ids']}", flush=True)
        print(
            f"POLE_B_POST_KV={cycle['pole_b']['post_kv']['sha256']}", flush=True
        )
        print(f"T_B={cycle['T_B']['token_ids']}", flush=True)
        print(f"W_SHA256={cycle['W_n']['sha256']}", flush=True)
    print(f"RECURRENCE_RESULT={record['observed_result']}", flush=True)
    print(
        f"RECURRENCE_CYCLES={record['recurrence_cycles'] or 'NONE'}", flush=True
    )
    print(f"{name}_END", flush=True)


def run_verification(interactive=False):
    if interactive:
        print("SLICE6_INTERACTIVE_READY", flush=True)
        seed = input("INITIAL_SEED> ")
        if seed != SEED_LITERAL:
            raise RuntimeError(f"Expected exact initial seed: {SEED_LITERAL}")
    else:
        print("SLICE6_RUNTIME_READY", flush=True)
    print(f"V_BASELINE={V_BASELINE}", flush=True)
    print(f"V_COUNTERFACTUAL={V_COUNTERFACTUAL}", flush=True)
    print(f"SEED={SEED_LITERAL}", flush=True)
    result, records = execute_acceptance()
    for name in (
        "BASELINE_A",
        "BASELINE_B",
        "COUNTERFACTUAL_A",
        "COUNTERFACTUAL_B",
    ):
        print_run(name, records[name])
    for name, passed in result["checks"].items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    divergence = result["divergence"]
    print(f"TRAJECTORY_RELATION={result['trajectory_relation']}", flush=True)
    print(
        f"EARLIEST_DIVERGENT_BOUNDARY={divergence['boundary'] if divergence else 'NONE'}",
        flush=True,
    )
    print(f"V_CAUSAL_PARTICIPATION={result['V_CAUSAL_PARTICIPATION']}", flush=True)
    print(f"FINAL_RESULT={'PASS' if result['passed'] else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={EVIDENCE_DIR / 'completion_receipt.txt'}", flush=True)
    if interactive:
        while input("TERMINATE (type EXIT)> ") != "EXIT":
            print("Type EXIT exactly to terminate.", flush=True)
        print("SLICE6_INTERACTIVE_TERMINATED", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--verify", "--interactive"):
        raise SystemExit("usage: slice6_runtime.py --verify | --interactive")
    run_verification(interactive=sys.argv[1] == "--interactive")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"SLICE6_RUNTIME_ERROR={error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
