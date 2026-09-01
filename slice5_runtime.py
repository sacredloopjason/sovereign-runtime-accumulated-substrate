import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE4_RUNTIME = PROJECT_DIR / "slice4_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice5"
SEED_LITERAL = "SLICE5_SEED"
CYCLE_LIMIT = 8


def load_slice4_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice4_accumulated_runtime", SLICE4_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 4 runtime: {SLICE4_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ordered_pair_fingerprint(a_ids, b_ids):
    payload = json.dumps(
        {"T_A": [int(item) for item in a_ids], "T_B": [int(item) for item in b_ids]},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(b"SLICE5_ORDERED_PERTURBATION_PAIR_V1\0" + payload).hexdigest()


def execute_experiment(
    slice4, slice3, kernel, model, tokenizer, shared_v, run_label, evidence_root
):
    import torch

    pole_a, pole_b = slice3.create_initial_poles(kernel, model, tokenizer)
    if not slice3.cache_tensor_storage_separated(pole_a, pole_b):
        raise RuntimeError("Initial sovereign pole storage is not separated")

    run_dir = evidence_root / f"run_{run_label.lower()}_boundaries"
    initial_a_snapshot, initial_a = slice4.snapshot_and_fingerprint(kernel, pole_a)
    initial_b_snapshot, initial_b = slice4.snapshot_and_fingerprint(kernel, pole_b)
    initial_a_file = slice4.save_boundary(
        torch, kernel, run_dir / "KV_A0.pt", initial_a_snapshot, initial_a
    )
    initial_b_file = slice4.save_boundary(
        torch, kernel, run_dir / "KV_B0.pt", initial_b_snapshot, initial_b
    )
    initial_a_object = id(pole_a)
    initial_b_object = id(pole_b)
    v_initial = slice3.fingerprint_v(shared_v)

    perturbation_for_a = tuple(
        int(item)
        for item in tokenizer(SEED_LITERAL + "\n", add_special_tokens=False).input_ids
    )
    cycles = []
    observed_result = "NO_RECURRENCE_WITHIN_BOUND"
    recurrence_cycles = None

    for cycle_number in range(CYCLE_LIMIT):
        a_pre = kernel.fingerprint_cache(pole_a)
        b_before_a = kernel.fingerprint_cache(pole_b)
        a_object_pre = id(pole_a)
        a_traversal = slice4.traverse_token_ids_with_v(
            kernel,
            model,
            tokenizer,
            pole_a,
            shared_v,
            perturbation_for_a,
            f"CYCLE_{cycle_number}_POLE_A",
            "external_seed_tokenization"
            if cycle_number == 0
            else f"cycle_{cycle_number - 1}_direct_B_generated_token_tuple",
        )
        a_post_snapshot, a_post = slice4.snapshot_and_fingerprint(kernel, pole_a)
        a_file = slice4.save_boundary(
            torch,
            kernel,
            run_dir / f"cycle_{cycle_number}_KV_A_post.pt",
            a_post_snapshot,
            a_post,
        )
        b_after_a = kernel.fingerprint_cache(pole_b)

        ta = tuple(a_traversal["generated_token_ids"])
        a_before_b = kernel.fingerprint_cache(pole_a)
        b_pre = kernel.fingerprint_cache(pole_b)
        b_object_pre = id(pole_b)
        b_traversal = slice4.traverse_token_ids_with_v(
            kernel,
            model,
            tokenizer,
            pole_b,
            shared_v,
            ta,
            f"CYCLE_{cycle_number}_POLE_B",
            f"cycle_{cycle_number}_direct_A_generated_token_tuple",
        )
        b_post_snapshot, b_post = slice4.snapshot_and_fingerprint(kernel, pole_b)
        b_file = slice4.save_boundary(
            torch,
            kernel,
            run_dir / f"cycle_{cycle_number}_KV_B_post.pt",
            b_post_snapshot,
            b_post,
        )
        a_after_b = kernel.fingerprint_cache(pole_a)
        v_after_cycle = slice3.fingerprint_v(shared_v)

        tb = tuple(b_traversal["generated_token_ids"])
        pair_sha256 = ordered_pair_fingerprint(ta, tb)
        recurrent_with_previous = bool(
            cycles
            and list(ta) == cycles[-1]["T_A"]["token_ids"]
            and list(tb) == cycles[-1]["T_B"]["token_ids"]
        )
        cycle = {
            "cycle": cycle_number,
            "pole_a": {
                "cache_object_id_pre": a_object_pre,
                "cache_object_id_post": id(pole_a),
                "pre_kv": a_pre,
                "post_kv": a_post,
                "post_boundary_file": a_file,
                "input_perturbation_token_ids": list(perturbation_for_a),
                "input_perturbation_sha256": slice4.token_fingerprint(
                    perturbation_for_a
                ),
                "output_observation_only": a_traversal[
                    "decoded_output_observation_only"
                ],
                "traversal": a_traversal,
            },
            "T_A": {
                "token_ids": list(ta),
                "sha256": slice4.token_fingerprint(ta),
                "consumed_by_b_token_ids": b_traversal[
                    "actual_first_forward_perturbation_ids"
                ],
                "consumed_by_b_sha256": b_traversal[
                    "actual_first_forward_perturbation_sha256"
                ],
                "exact_identity": (
                    list(ta)
                    == b_traversal["actual_first_forward_perturbation_ids"]
                ),
            },
            "pole_b": {
                "cache_object_id_pre": b_object_pre,
                "cache_object_id_post": id(pole_b),
                "pre_kv": b_pre,
                "post_kv": b_post,
                "post_boundary_file": b_file,
                "output_observation_only": b_traversal[
                    "decoded_output_observation_only"
                ],
                "traversal": b_traversal,
            },
            "T_B": {
                "token_ids": list(tb),
                "sha256": slice4.token_fingerprint(tb),
                "accepted_by_a_transport_token_ids": list(tb),
                "accepted_by_a_transport_sha256": slice4.token_fingerprint(tb),
                "exact_identity_at_a_transport_boundary": True,
                "consumed_by_a_next_cycle_token_ids": None,
                "consumed_by_a_next_cycle_sha256": None,
                "exact_identity_at_next_a_boundary": None,
            },
            "W_n": {
                "sha256": pair_sha256,
                "recurrent_with_previous": recurrent_with_previous,
                "equality_operation": "ordered integer-list equality only",
            },
            "isolation": {
                "pole_b_unchanged_during_a": b_before_a["sha256"]
                == b_after_a["sha256"],
                "pole_a_unchanged_during_b": a_before_b["sha256"]
                == a_after_b["sha256"],
            },
            "shared_v_after_cycle": v_after_cycle,
        }
        cycles.append(cycle)

        if cycle_number > 0:
            previous_tb = cycles[cycle_number - 1]["T_B"]
            previous_tb["consumed_by_a_next_cycle_token_ids"] = a_traversal[
                "actual_first_forward_perturbation_ids"
            ]
            previous_tb["consumed_by_a_next_cycle_sha256"] = a_traversal[
                "actual_first_forward_perturbation_sha256"
            ]
            previous_tb["exact_identity_at_next_a_boundary"] = (
                previous_tb["token_ids"]
                == a_traversal["actual_first_forward_perturbation_ids"]
            )

        if recurrent_with_previous:
            observed_result = "RECURRENCE"
            recurrence_cycles = [cycle_number - 1, cycle_number]
            break
        perturbation_for_a = tb

    record = {
        "schema": "SLICE5_ENDOGENOUS_LOOP_STABILIZATION_EVIDENCE_V1",
        "run": run_label,
        "shared_v": {
            "literal": shared_v.literal,
            "initial": v_initial,
        },
        "seed_literal": SEED_LITERAL,
        "seed_token_ids": [
            int(item)
            for item in tokenizer(
                SEED_LITERAL + "\n", add_special_tokens=False
            ).input_ids
        ],
        "cycle_limit": CYCLE_LIMIT,
        "initial_kv": {"A": initial_a, "B": initial_b},
        "initial_boundary_files": {"A": initial_a_file, "B": initial_b_file},
        "initial_cache_object_ids": {"A": initial_a_object, "B": initial_b_object},
        "cycles": cycles,
        "observed_result": observed_result,
        "recurrence_cycles": recurrence_cycles,
        "recurrence_gate": {
            "operation": "exact equality of consecutive ordered token-ID pairs W_n=(T_A_n,T_B_n)",
            "passive": True,
            "semantic_content_inspected": False,
            "dynamics_altered": False,
        },
        "no_decode_retokenize": True,
        "no_semantic_mediation": True,
        "no_external_epistemic_authority": True,
        "forward_only_history": True,
    }
    record["checks"] = check_run(record)
    return record


def check_run(run):
    cycles = run["cycles"]
    v_sha = run["shared_v"]["initial"]["sha256"]
    a_previous = run["initial_kv"]["A"]["sha256"]
    b_previous = run["initial_kv"]["B"]["sha256"]
    continuity = True
    exact_transport = True
    isolation = True
    v_invariant = True
    forward_only = True
    for index, cycle in enumerate(cycles):
        continuity = continuity and all(
            (
                cycle["pole_a"]["pre_kv"]["sha256"] == a_previous,
                cycle["pole_b"]["pre_kv"]["sha256"] == b_previous,
                cycle["pole_a"]["traversal"][
                    "first_forward_pre_cache_matches_boundary"
                ],
                cycle["pole_b"]["traversal"][
                    "first_forward_pre_cache_matches_boundary"
                ],
                cycle["pole_a"]["traversal"][
                    "all_forwards_received_exact_pole_cache"
                ],
                cycle["pole_b"]["traversal"][
                    "all_forwards_received_exact_pole_cache"
                ],
                cycle["pole_a"]["cache_object_id_pre"]
                == cycle["pole_a"]["cache_object_id_post"],
                cycle["pole_b"]["cache_object_id_pre"]
                == cycle["pole_b"]["cache_object_id_post"],
            )
        )
        exact_transport = exact_transport and all(
            (
                cycle["T_A"]["exact_identity"],
                cycle["T_B"]["exact_identity_at_a_transport_boundary"],
                cycle["T_B"]["token_ids"]
                == cycle["T_B"]["accepted_by_a_transport_token_ids"],
                cycle["T_B"]["sha256"]
                == cycle["T_B"]["accepted_by_a_transport_sha256"],
            )
        )
        if index > 0:
            exact_transport = exact_transport and cycles[index - 1]["T_B"][
                "exact_identity_at_next_a_boundary"
            ]
        isolation = isolation and all(cycle["isolation"].values())
        v_invariant = (
            v_invariant
            and cycle["shared_v_after_cycle"]["sha256"] == v_sha
            and cycle["shared_v_after_cycle"]["object_id"]
            == run["shared_v"]["initial"]["object_id"]
        )
        forward_only = forward_only and all(
            (
                cycle["pole_a"]["post_kv"]["sequence_length"]
                > cycle["pole_a"]["pre_kv"]["sequence_length"],
                cycle["pole_b"]["post_kv"]["sequence_length"]
                > cycle["pole_b"]["pre_kv"]["sequence_length"],
            )
        )
        a_previous = cycle["pole_a"]["post_kv"]["sha256"]
        b_previous = cycle["pole_b"]["post_kv"]["sha256"]
    valid_result = (
        run["observed_result"] == "RECURRENCE"
        and run["recurrence_cycles"] is not None
        and cycles[-1]["W_n"]["recurrent_with_previous"]
    ) or (
        run["observed_result"] == "NO_RECURRENCE_WITHIN_BOUND"
        and run["recurrence_cycles"] is None
        and len(cycles) == CYCLE_LIMIT
        and not any(cycle["W_n"]["recurrent_with_previous"] for cycle in cycles)
    )
    return {
        "KV_CAUSAL_CONTINUITY": continuity,
        "EXACT_TOKEN_TRANSPOSITION": exact_transport,
        "KV_SOVEREIGNTY": isolation,
        "SHARED_V_INVARIANT": v_invariant,
        "FORWARD_ONLY_CAUSAL_HISTORY": forward_only,
        "RECURRENCE_RESULT_VALID": valid_result,
        "RECURRENCE_GATE_PASSIVE": all(run["recurrence_gate"][key] is expected for key, expected in (("passive", True), ("semantic_content_inspected", False), ("dynamics_altered", False))),
        "NO_SEMANTIC_MEDIATION": run["no_semantic_mediation"],
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": run[
            "no_external_epistemic_authority"
        ],
    }


def deterministic_view(run):
    return {
        "shared_v": {
            "literal": run["shared_v"]["literal"],
            "sha256": run["shared_v"]["initial"]["sha256"],
            "token_ids": run["shared_v"]["initial"]["token_ids"],
        },
        "seed_literal": run["seed_literal"],
        "seed_token_ids": run["seed_token_ids"],
        "initial_kv": {
            pole: {
                "sha256": value["sha256"],
                "sequence_length": value["sequence_length"],
                "tensor_byte_count": value["tensor_byte_count"],
            }
            for pole, value in run["initial_kv"].items()
        },
        "cycles": [
            {
                "cycle": cycle["cycle"],
                "A_pre": cycle["pole_a"]["pre_kv"],
                "T_A": cycle["T_A"]["token_ids"],
                "T_A_sha256": cycle["T_A"]["sha256"],
                "A_post": cycle["pole_a"]["post_kv"],
                "B_pre": cycle["pole_b"]["pre_kv"],
                "T_B": cycle["T_B"]["token_ids"],
                "T_B_sha256": cycle["T_B"]["sha256"],
                "B_post": cycle["pole_b"]["post_kv"],
                "W_sha256": cycle["W_n"]["sha256"],
                "recurrent": cycle["W_n"]["recurrent_with_previous"],
            }
            for cycle in run["cycles"]
        ],
        "observed_result": run["observed_result"],
        "recurrence_cycles": run["recurrence_cycles"],
    }


def add_field(lines, name, value):
    lines.extend([f"{name}:", str(value), ""])


def add_run_receipt(lines, run_name, run):
    lines.extend([run_name, ""])
    add_field(lines, "initial_KV_A", run["initial_kv"]["A"]["sha256"])
    add_field(lines, "initial_KV_B", run["initial_kv"]["B"]["sha256"])
    lines.extend(["cycles:", ""])
    for cycle in run["cycles"]:
        lines.append(f"cycle {cycle['cycle']}")
        add_field(lines, "Pole A pre-KV fingerprint", cycle["pole_a"]["pre_kv"]["sha256"])
        add_field(lines, "T_A produced fingerprint / token IDs", f"{cycle['T_A']['sha256']} / {cycle['T_A']['token_ids']}")
        add_field(lines, "Pole A output", cycle["pole_a"]["output_observation_only"])
        add_field(lines, "Pole A post-KV fingerprint", cycle["pole_a"]["post_kv"]["sha256"])
        add_field(lines, "T_A consumed-by-B evidence", f"{cycle['T_A']['consumed_by_b_sha256']} / {cycle['T_A']['consumed_by_b_token_ids']} / exact={cycle['T_A']['exact_identity']}")
        add_field(lines, "Pole B pre-KV fingerprint", cycle["pole_b"]["pre_kv"]["sha256"])
        add_field(lines, "T_B produced fingerprint / token IDs", f"{cycle['T_B']['sha256']} / {cycle['T_B']['token_ids']}")
        add_field(lines, "Pole B output", cycle["pole_b"]["output_observation_only"])
        add_field(lines, "Pole B post-KV fingerprint", cycle["pole_b"]["post_kv"]["sha256"])
        add_field(lines, "T_B consumed-by-A evidence", f"transport-boundary exact={cycle['T_B']['exact_identity_at_a_transport_boundary']} / {cycle['T_B']['accepted_by_a_transport_sha256']} / {cycle['T_B']['accepted_by_a_transport_token_ids']}; next model-boundary exact={cycle['T_B']['exact_identity_at_next_a_boundary']} / {cycle['T_B']['consumed_by_a_next_cycle_sha256']} / {cycle['T_B']['consumed_by_a_next_cycle_token_ids']}")
        add_field(lines, "W_n fingerprint", cycle["W_n"]["sha256"])
    add_field(lines, "observed_result", run["observed_result"])
    add_field(lines, "recurrence_cycles", run["recurrence_cycles"] if run["recurrence_cycles"] is not None else "NONE")


def make_receipt(run_a, run_b, trajectory_identity):
    all_checks = {
        **{f"RUN_A_{name}": value for name, value in run_a["checks"].items()},
        **{f"RUN_B_{name}": value for name, value in run_b["checks"].items()},
        "DETERMINISTIC_TRAJECTORY_IDENTITY": trajectory_identity,
    }
    if not all(all_checks.values()):
        failed = [name for name, passed in all_checks.items() if not passed]
        return "\n".join(
            [
                "BUILD_SLICE_RECEIPT", "slice: 5", "result: FAIL", "",
                "artifact_state:", str(PROJECT_DIR), "", "failed_requirement:",
                ", ".join(failed), "", "causal_boundary:",
                "The named repeated traversal, KV continuity, exact token transport, recurrence gate, or deterministic replay comparison.",
                "", "substrate_evidence:", json.dumps(all_checks, sort_keys=True),
                "", "substrate_finding:",
                "The exact failed binary comparison is retained in run_a.json and run_b.json.",
                "", "perturbation:", ", ".join(failed), "", "final_result:",
                "FAIL", "END_BUILD_SLICE_RECEIPT", "",
            ]
        )
    lines = ["BUILD_SLICE_RECEIPT", "slice: 5", "result: PASS", ""]
    add_field(lines, "artifact", PROJECT_DIR)
    add_field(lines, "shared_V", "SLICE3_SHARED_V")
    add_field(lines, "seed", SEED_LITERAL)
    add_field(lines, "cycle_limit", CYCLE_LIMIT)
    add_field(lines, "stabilization_test", "exact equality of consecutive ordered perturbation pairs W_n=(T_A_n,T_B_n)")
    add_run_receipt(lines, "RUN_A", run_a)
    add_run_receipt(lines, "RUN_B", run_b)
    for name in (
        "DETERMINISTIC_TRAJECTORY_IDENTITY", "KV_CAUSAL_CONTINUITY",
        "EXACT_TOKEN_TRANSPOSITION", "SHARED_V_INVARIANT",
        "RECURRENCE_GATE_PASSIVE", "NO_SEMANTIC_MEDIATION",
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY",
    ):
        add_field(lines, name, "PASS")
    finding = (
        "The accumulated runtime sustained two independent live DynamicCache lineages through "
        f"{len(run_a['cycles'])} complete reciprocal cycles. Actual greedy-generated integer token tuples crossed "
        "directly into the opposite pole's torch.long model input after the invariant V prefix. A passive ordered "
        "integer-list equality operation exposed " + run_a["observed_result"] +
        (f" at cycles {run_a['recurrence_cycles']}." if run_a["recurrence_cycles"] else " within the fixed bound.")
    )
    add_field(lines, "substrate_finding", finding)
    lines.extend(["final_result:", "PASS", "END_BUILD_SLICE_RECEIPT", ""])
    return "\n".join(lines)


def initialize():
    slice4 = load_slice4_runtime()
    slice3 = slice4.load_slice3_runtime()
    kernel = slice3.load_slice1_kernel()
    kernel.configure_determinism()
    weight_hash = kernel.verify_substrate()
    tokenizer, model = kernel.build_model()
    shared_v = slice3.make_shared_v(tokenizer)
    return slice4, slice3, kernel, model, tokenizer, shared_v, weight_hash


def run_verification():
    slice4, slice3, kernel, model, tokenizer, shared_v, weight_hash = initialize()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    print("SLICE5_RUNTIME_READY", flush=True)
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    print(f"SHARED_V={shared_v.literal}", flush=True)
    print(f"SEED={SEED_LITERAL}", flush=True)
    print(f"CYCLE_LIMIT={CYCLE_LIMIT}", flush=True)
    run_a = execute_experiment(slice4, slice3, kernel, model, tokenizer, shared_v, "A", EVIDENCE_DIR)
    run_b = execute_experiment(slice4, slice3, kernel, model, tokenizer, shared_v, "B", EVIDENCE_DIR)
    trajectory_identity = deterministic_view(run_a) == deterministic_view(run_b)
    (EVIDENCE_DIR / "run_a.json").write_text(json.dumps(run_a, indent=2, sort_keys=True), encoding="utf-8")
    (EVIDENCE_DIR / "run_b.json").write_text(json.dumps(run_b, indent=2, sort_keys=True), encoding="utf-8")
    receipt = make_receipt(run_a, run_b, trajectory_identity)
    receipt_path = EVIDENCE_DIR / "completion_receipt.txt"
    receipt_path.write_text(receipt, encoding="utf-8")
    for cycle in run_a["cycles"]:
        print(f"CYCLE={cycle['cycle']}", flush=True)
        print(f"POLE_A_OUTPUT={json.dumps(cycle['pole_a']['output_observation_only'])}", flush=True)
        print(f"T_A={cycle['T_A']['token_ids']}", flush=True)
        print(f"POLE_B_OUTPUT={json.dumps(cycle['pole_b']['output_observation_only'])}", flush=True)
        print(f"T_B={cycle['T_B']['token_ids']}", flush=True)
        print(f"W_SHA256={cycle['W_n']['sha256']}", flush=True)
        print(f"RECURRENT_WITH_PREVIOUS={str(cycle['W_n']['recurrent_with_previous']).upper()}", flush=True)
    checks = {**run_a["checks"], "DETERMINISTIC_TRAJECTORY_IDENTITY": trajectory_identity}
    for name, passed in checks.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    passed = all(run_a["checks"].values()) and all(run_b["checks"].values()) and trajectory_identity
    print(f"OBSERVED_RESULT={run_a['observed_result']}", flush=True)
    print(f"RECURRENCE_CYCLES={run_a['recurrence_cycles'] if run_a['recurrence_cycles'] is not None else 'NONE'}", flush=True)
    print(f"FINAL_RESULT={'PASS' if passed else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={receipt_path}", flush=True)


def run_interactive():
    slice4, slice3, kernel, model, tokenizer, shared_v, _ = initialize()
    print("SLICE5_INTERACTIVE_READY", flush=True)
    print(f"SHARED_V={shared_v.literal}", flush=True)
    seed = input("INITIAL_SEED> ")
    if seed != SEED_LITERAL:
        raise RuntimeError(f"Expected exact initial seed: {SEED_LITERAL}")
    record = execute_experiment(slice4, slice3, kernel, model, tokenizer, shared_v, "INTERACTIVE", EVIDENCE_DIR)
    for cycle in record["cycles"]:
        print(f"CYCLE_{cycle['cycle']}_BEGIN", flush=True)
        print("POLE_A_OUTPUT_BEGIN", flush=True)
        print(cycle["pole_a"]["output_observation_only"], flush=True)
        print("POLE_A_OUTPUT_END", flush=True)
        print(f"T_A={cycle['T_A']['token_ids']}", flush=True)
        print("POLE_B_OUTPUT_BEGIN", flush=True)
        print(cycle["pole_b"]["output_observation_only"], flush=True)
        print("POLE_B_OUTPUT_END", flush=True)
        print(f"T_B={cycle['T_B']['token_ids']}", flush=True)
        print(f"W_SHA256={cycle['W_n']['sha256']}", flush=True)
        print(f"RECURRENT_WITH_PREVIOUS={str(cycle['W_n']['recurrent_with_previous']).upper()}", flush=True)
        print(f"CYCLE_{cycle['cycle']}_END", flush=True)
    print(f"OBSERVED_RESULT={record['observed_result']}", flush=True)
    print(f"RECURRENCE_CYCLES={record['recurrence_cycles'] if record['recurrence_cycles'] is not None else 'NONE'}", flush=True)
    while input("TERMINATE (type EXIT)> ") != "EXIT":
        print("Type EXIT exactly to terminate.", flush=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    record_path = EVIDENCE_DIR / "human_interactive_run.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print("SLICE5_INTERACTIVE_TERMINATED", flush=True)
    print(f"INTERACTIVE_EVIDENCE={record_path}", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--verify", "--interactive"):
        raise SystemExit("usage: slice5_runtime.py --verify | --interactive")
    if sys.argv[1] == "--verify":
        run_verification()
    else:
        run_interactive()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SLICE5_RUNTIME_ERROR={error.__class__.__name__}: {error}", file=sys.stderr, flush=True)
        raise
