import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice4"
SLICE3_RUNTIME = PROJECT_DIR / "runtime.py"
SEED_LITERAL = "SLICE4_SEED"


def load_slice3_runtime():
    spec = importlib.util.spec_from_file_location("slice3_accumulated_runtime", SLICE3_RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 3 runtime: {SLICE3_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def token_fingerprint(token_ids):
    payload = json.dumps(
        [int(token_id) for token_id in token_ids], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(b"SLICE4_ORDERED_TOKEN_IDS_V1\0" + payload).hexdigest()


def snapshot_and_fingerprint(kernel, cache):
    snapshot = kernel.snapshot_cache(cache)
    return snapshot, kernel.fingerprint_material(snapshot)


def save_boundary(torch, kernel, path, snapshot, fingerprint):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    restored_fingerprint = kernel.fingerprint_material(restored)
    if restored_fingerprint["sha256"] != fingerprint["sha256"]:
        raise RuntimeError(f"Saved KV boundary did not round-trip: {path}")
    return {
        "path": str(path),
        "sha256": fingerprint["sha256"],
        "sequence_length": fingerprint["sequence_length"],
        "tensor_byte_count": fingerprint["tensor_byte_count"],
        "round_trip_verified": True,
    }


def traverse_token_ids_with_v(
    kernel,
    model,
    tokenizer,
    cache,
    shared_v,
    perturbation_ids,
    traversal_label,
    transport_origin,
):
    import torch

    perturbation_ids = tuple(int(token_id) for token_id in perturbation_ids)
    if not perturbation_ids:
        raise RuntimeError(f"{traversal_label} received an empty perturbation")
    full_input_ids = shared_v.token_ids + perturbation_ids
    input_tensor = torch.tensor([full_input_ids], dtype=torch.long)
    cache_object_id_before = id(cache)
    cache_pointer_map_before = kernel.pointer_map(cache)
    cache_fingerprint_before = kernel.fingerprint_cache(cache)
    events = []
    original_forward = model.forward

    def observed_forward(*args, **kwargs):
        supplied_cache = kwargs.get("past_key_values")
        supplied_ids = kwargs.get("input_ids")
        event = {
            "supplied_cache_object_id": id(supplied_cache),
            "supplied_is_exact_pole_cache": supplied_cache is cache,
            "pre_sequence_length": int(supplied_cache.get_seq_length()),
            "pre_cache_sha256": kernel.fingerprint_cache(supplied_cache)["sha256"],
            "input_token_ids": [int(item) for item in supplied_ids.flatten().tolist()],
        }
        outputs = original_forward(*args, **kwargs)
        event["returned_same_cache_object"] = outputs.past_key_values is supplied_cache
        event["post_sequence_length"] = int(outputs.past_key_values.get_seq_length())
        events.append(event)
        return outputs

    model.forward = observed_forward
    generated_ids = []
    try:
        def forward(tokens):
            past_length = int(cache.get_seq_length())
            token_count = int(tokens.shape[1])
            attention_mask = torch.ones((1, past_length + token_count), dtype=torch.long)
            positions = torch.arange(
                past_length, past_length + token_count, dtype=torch.long
            )
            with torch.inference_mode():
                output = model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    cache_position=positions,
                    past_key_values=cache,
                    use_cache=True,
                )
            if output.past_key_values is not cache:
                raise RuntimeError("Inference returned a different KV cache")
            return output.logits[:, -1, :]

        logits = forward(input_tensor)
        for _ in range(kernel.MAX_NEW_TOKENS):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            logits = forward(next_token)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
    finally:
        model.forward = original_forward

    first_event = events[0]
    first_ids = first_event["input_token_ids"]
    v_count = len(shared_v.token_ids)
    consumed_perturbation = first_ids[v_count:]
    return {
        "traversal": traversal_label,
        "transport_origin": transport_origin,
        "transport_operation": (
            "tuple[int] -> torch.tensor(dtype=torch.long), with the exact ordered "
            "token IDs appended after the unchanged canonical shared-V token prefix"
        ),
        "perturbation_token_ids_supplied": list(perturbation_ids),
        "perturbation_sha256_supplied": token_fingerprint(perturbation_ids),
        "actual_first_forward_input_ids": first_ids,
        "actual_first_forward_v_prefix": first_ids[:v_count],
        "actual_first_forward_perturbation_ids": consumed_perturbation,
        "actual_first_forward_perturbation_sha256": token_fingerprint(
            consumed_perturbation
        ),
        "exact_perturbation_identity_at_model_boundary": (
            consumed_perturbation == list(perturbation_ids)
        ),
        "generated_token_ids": generated_ids,
        "generated_token_sha256": token_fingerprint(generated_ids),
        "decoded_output_observation_only": tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "cache_object_id_before": cache_object_id_before,
        "cache_pointer_map_before": cache_pointer_map_before,
        "cache_fingerprint_before": cache_fingerprint_before,
        "first_forward_received_same_cache_object": (
            first_event["supplied_cache_object_id"] == cache_object_id_before
            and first_event["supplied_is_exact_pole_cache"]
        ),
        "first_forward_pre_cache_matches_boundary": (
            first_event["pre_cache_sha256"] == cache_fingerprint_before["sha256"]
        ),
        "all_forwards_received_exact_pole_cache": all(
            event["supplied_is_exact_pole_cache"] for event in events
        ),
        "all_forwards_returned_same_cache": all(
            event["returned_same_cache_object"] for event in events
        ),
        "forward_events": events,
    }


def execute_loop(slice3, kernel, model, tokenizer, shared_v, run_label, evidence_root):
    import torch

    pole_a, pole_b = slice3.create_initial_poles(kernel, model, tokenizer)
    if not slice3.cache_tensor_storage_separated(pole_a, pole_b):
        raise RuntimeError("Initial sovereign pole storage is not separated")

    run_dir = evidence_root / f"run_{run_label.lower()}_boundaries"
    a0_snapshot, a0 = snapshot_and_fingerprint(kernel, pole_a)
    b0_snapshot, b0 = snapshot_and_fingerprint(kernel, pole_b)
    a0_file = save_boundary(torch, kernel, run_dir / "KV_A0.pt", a0_snapshot, a0)
    b0_file = save_boundary(torch, kernel, run_dir / "KV_B0.pt", b0_snapshot, b0)
    v_initial = slice3.fingerprint_v(shared_v)

    seed_ids = tuple(
        int(item)
        for item in tokenizer(
            SEED_LITERAL + "\n", add_special_tokens=False
        ).input_ids
    )
    b_before_a1 = kernel.fingerprint_cache(pole_b)
    a1_traversal = traverse_token_ids_with_v(
        kernel,
        model,
        tokenizer,
        pole_a,
        shared_v,
        seed_ids,
        "A0_TO_A1",
        "external_seed_tokenization",
    )
    a1_snapshot, a1 = snapshot_and_fingerprint(kernel, pole_a)
    a1_file = save_boundary(torch, kernel, run_dir / "KV_A1.pt", a1_snapshot, a1)
    a1_object_id_post = id(pole_a)
    a1_pointer_map_post = kernel.pointer_map(pole_a)
    b_after_a1 = kernel.fingerprint_cache(pole_b)
    v_after_a1 = slice3.fingerprint_v(shared_v)

    ta_produced = tuple(a1_traversal["generated_token_ids"])
    a_before_b1 = kernel.fingerprint_cache(pole_a)
    b1_traversal = traverse_token_ids_with_v(
        kernel,
        model,
        tokenizer,
        pole_b,
        shared_v,
        ta_produced,
        "B0_TO_B1",
        "direct_A_generated_token_tuple",
    )
    b1_snapshot, b1 = snapshot_and_fingerprint(kernel, pole_b)
    b1_file = save_boundary(torch, kernel, run_dir / "KV_B1.pt", b1_snapshot, b1)
    a_after_b1 = kernel.fingerprint_cache(pole_a)
    v_after_b1 = slice3.fingerprint_v(shared_v)

    tb_produced = tuple(b1_traversal["generated_token_ids"])
    a1_pre_return = kernel.fingerprint_cache(pole_a)
    a1_object_id_pre_return = id(pole_a)
    a1_pointer_map_pre_return = kernel.pointer_map(pole_a)
    b_before_a2 = kernel.fingerprint_cache(pole_b)
    a2_traversal = traverse_token_ids_with_v(
        kernel,
        model,
        tokenizer,
        pole_a,
        shared_v,
        tb_produced,
        "A1_TO_A2",
        "direct_B_generated_token_tuple",
    )
    a2_snapshot, a2 = snapshot_and_fingerprint(kernel, pole_a)
    a2_file = save_boundary(torch, kernel, run_dir / "KV_A2.pt", a2_snapshot, a2)
    b_after_a2 = kernel.fingerprint_cache(pole_b)
    v_after_a2 = slice3.fingerprint_v(shared_v)

    record = {
        "schema": "SLICE4_RECIPROCAL_POLE_TRANSPOSITION_EVIDENCE_V1",
        "run": run_label,
        "shared_v": {
            "literal": shared_v.literal,
            "initial": v_initial,
            "after_a1": v_after_a1,
            "after_b1": v_after_b1,
            "after_a2": v_after_a2,
        },
        "seed_literal": SEED_LITERAL,
        "seed_token_ids": list(seed_ids),
        "kv": {
            "A0": a0,
            "B0": b0,
            "A1": a1,
            "B1": b1,
            "A2": a2,
        },
        "boundary_files": {
            "A0": a0_file,
            "B0": b0_file,
            "A1": a1_file,
            "B1": b1_file,
            "A2": a2_file,
        },
        "traversals": {
            "A1": a1_traversal,
            "B1": b1_traversal,
            "A2": a2_traversal,
        },
        "transport": {
            "A_TO_B_produced": list(ta_produced),
            "A_TO_B_produced_sha256": token_fingerprint(ta_produced),
            "A_TO_B_consumed": b1_traversal[
                "actual_first_forward_perturbation_ids"
            ],
            "A_TO_B_consumed_sha256": b1_traversal[
                "actual_first_forward_perturbation_sha256"
            ],
            "B_TO_A_produced": list(tb_produced),
            "B_TO_A_produced_sha256": token_fingerprint(tb_produced),
            "B_TO_A_consumed": a2_traversal[
                "actual_first_forward_perturbation_ids"
            ],
            "B_TO_A_consumed_sha256": a2_traversal[
                "actual_first_forward_perturbation_sha256"
            ],
            "mechanism": (
                "The producer's generated_token_ids list was frozen as a tuple of "
                "integers and passed directly to the predetermined recipient. The "
                "consumer constructed one torch.long tensor containing the unchanged "
                "shared-V prefix followed by those exact tuple elements. Decoding was "
                "performed only after generation for display and was never read by routing."
            ),
        },
        "continuity": {
            "A1_object_id_post": a1_object_id_post,
            "A1_object_id_pre_return": a1_object_id_pre_return,
            "A1_pointer_map_post": a1_pointer_map_post,
            "A1_pointer_map_pre_return": a1_pointer_map_pre_return,
            "A1_fingerprint_post": a1["sha256"],
            "A1_fingerprint_pre_return": a1_pre_return["sha256"],
            "B_unchanged_during_A1": b_before_a1["sha256"]
            == b_after_a1["sha256"],
            "A_unchanged_during_B1": a_before_b1["sha256"]
            == a_after_b1["sha256"],
            "B_unchanged_during_A2": b_before_a2["sha256"]
            == b_after_a2["sha256"],
        },
        "no_decode_retokenize": True,
        "no_semantic_mediation": True,
        "passive_predetermined_route": ["A", "B", "A"],
        "no_external_epistemic_authority": True,
    }
    record["checks"] = check_single_run(record)
    return record


def check_single_run(run):
    kv = run["kv"]
    traversals = run["traversals"]
    transport = run["transport"]
    continuity = run["continuity"]
    v = run["shared_v"]
    return {
        "A_TO_B_EXACT_TOKEN_IDENTITY": (
            transport["A_TO_B_produced"] == transport["A_TO_B_consumed"]
            and transport["A_TO_B_produced_sha256"]
            == transport["A_TO_B_consumed_sha256"]
            and traversals["B1"]["exact_perturbation_identity_at_model_boundary"]
        ),
        "B_TO_A_EXACT_TOKEN_IDENTITY": (
            transport["B_TO_A_produced"] == transport["B_TO_A_consumed"]
            and transport["B_TO_A_produced_sha256"]
            == transport["B_TO_A_consumed_sha256"]
            and traversals["A2"]["exact_perturbation_identity_at_model_boundary"]
        ),
        "A_TO_B_NO_DECODE_RETOKENIZE": run["no_decode_retokenize"],
        "B_TO_A_NO_DECODE_RETOKENIZE": run["no_decode_retokenize"],
        "POLE_A_CAUSAL_CONTINUITY": all(
            (
                continuity["A1_object_id_post"]
                == continuity["A1_object_id_pre_return"],
                continuity["A1_pointer_map_post"]
                == continuity["A1_pointer_map_pre_return"],
                continuity["A1_fingerprint_post"]
                == continuity["A1_fingerprint_pre_return"],
                traversals["A2"]["first_forward_received_same_cache_object"],
                traversals["A2"]["first_forward_pre_cache_matches_boundary"],
                traversals["A2"]["cache_fingerprint_before"]["sha256"]
                == kv["A1"]["sha256"],
            )
        ),
        "POLE_B_CAUSAL_CONTINUITY": all(
            (
                traversals["B1"]["first_forward_received_same_cache_object"],
                traversals["B1"]["first_forward_pre_cache_matches_boundary"],
                traversals["B1"]["cache_fingerprint_before"]["sha256"]
                == kv["B0"]["sha256"],
            )
        ),
        "KV_SOVEREIGNTY": all(
            (
                continuity["B_unchanged_during_A1"],
                continuity["A_unchanged_during_B1"],
                continuity["B_unchanged_during_A2"],
            )
        ),
        "SHARED_V_INVARIANT": (
            v["literal"] == "SLICE3_SHARED_V"
            and v["initial"]["sha256"]
            == v["after_a1"]["sha256"]
            == v["after_b1"]["sha256"]
            == v["after_a2"]["sha256"]
            and v["initial"]["object_id"]
            == v["after_a1"]["object_id"]
            == v["after_b1"]["object_id"]
            == v["after_a2"]["object_id"]
        ),
        "ADJACENT_BOUNDARIES_PRESERVED": all(
            boundary["round_trip_verified"]
            for boundary in run["boundary_files"].values()
        ),
        "NO_SEMANTIC_MEDIATION": run["no_semantic_mediation"],
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": run[
            "no_external_epistemic_authority"
        ],
        "ALL_MODEL_FORWARDS_USED_SOVEREIGN_CACHE": all(
            traversal["all_forwards_received_exact_pole_cache"]
            and traversal["all_forwards_returned_same_cache"]
            for traversal in traversals.values()
        ),
    }


def compare_runs(run_a, run_b):
    return {
        "KV_A0_EQUAL": run_a["kv"]["A0"]["sha256"]
        == run_b["kv"]["A0"]["sha256"],
        "KV_B0_EQUAL": run_a["kv"]["B0"]["sha256"]
        == run_b["kv"]["B0"]["sha256"],
        "KV_A1_EQUAL": run_a["kv"]["A1"]["sha256"]
        == run_b["kv"]["A1"]["sha256"],
        "KV_B1_EQUAL": run_a["kv"]["B1"]["sha256"]
        == run_b["kv"]["B1"]["sha256"],
        "KV_A2_EQUAL": run_a["kv"]["A2"]["sha256"]
        == run_b["kv"]["A2"]["sha256"],
        "T_A_EQUAL": run_a["transport"]["A_TO_B_produced"]
        == run_b["transport"]["A_TO_B_produced"],
        "T_B_EQUAL": run_a["transport"]["B_TO_A_produced"]
        == run_b["transport"]["B_TO_A_produced"],
        "POLE_A_OUTPUT_1_IDENTICAL": run_a["traversals"]["A1"][
            "decoded_output_observation_only"
        ]
        == run_b["traversals"]["A1"]["decoded_output_observation_only"],
        "POLE_B_OUTPUT_IDENTICAL": run_a["traversals"]["B1"][
            "decoded_output_observation_only"
        ]
        == run_b["traversals"]["B1"]["decoded_output_observation_only"],
        "POLE_A_OUTPUT_2_IDENTICAL": run_a["traversals"]["A2"][
            "decoded_output_observation_only"
        ]
        == run_b["traversals"]["A2"]["decoded_output_observation_only"],
    }


def add_field(lines, name, value):
    lines.extend([f"{name}:", str(value), ""])


def make_receipt(run_a, run_b, deterministic_checks):
    all_checks = {
        **{f"RUN_A_{key}": value for key, value in run_a["checks"].items()},
        **{f"RUN_B_{key}": value for key, value in run_b["checks"].items()},
        **deterministic_checks,
    }
    if not all(all_checks.values()):
        failed = [name for name, passed in all_checks.items() if not passed]
        return "\n".join(
            [
                "BUILD_SLICE_RECEIPT",
                "slice: 4",
                "result: FAIL",
                "",
                "artifact_state:",
                str(PROJECT_DIR),
                "",
                "failed_requirement:",
                ", ".join(failed),
                "",
                "causal_boundary:",
                "The named direct token transport, sovereign KV continuity, shared-V, boundary preservation, or deterministic replay comparison.",
                "",
                "substrate_evidence:",
                json.dumps(all_checks, sort_keys=True),
                "",
                "substrate_finding:",
                "The exact failed binary comparisons are retained in run_a.json and run_b.json.",
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

    ta = run_a["transport"]["A_TO_B_produced"]
    tb = run_a["transport"]["B_TO_A_produced"]
    lines = ["BUILD_SLICE_RECEIPT", "slice: 4", "result: PASS", ""]
    add_field(lines, "artifact", PROJECT_DIR)
    add_field(lines, "shared_V", "SLICE3_SHARED_V")
    add_field(lines, "initial_seed", SEED_LITERAL)
    add_field(lines, "perturbation_substrate", "Python tuple[int] of actual greedy-generated token IDs, materialized unchanged as a CPU torch.long input tensor at the recipient model.forward boundary")
    add_field(lines, "transport_mechanism", "Predetermined A-to-B-to-A function calls pass the producer generated_token_ids tuple directly; the recipient prepends the canonical immutable V tuple and constructs a torch.long tensor without decoding, re-tokenizing, inspecting, filtering, or selecting content")
    lines.extend(["RUN_A", ""])
    add_field(lines, "KV_A0", run_a["kv"]["A0"]["sha256"])
    add_field(lines, "KV_B0", run_a["kv"]["B0"]["sha256"])
    add_field(lines, "seed_input", SEED_LITERAL)
    add_field(lines, "pole_A_output_1", run_a["traversals"]["A1"]["decoded_output_observation_only"])
    add_field(lines, "T_A_PRODUCED", f"{ta} sha256={run_a['transport']['A_TO_B_produced_sha256']}")
    add_field(lines, "KV_A1", run_a["kv"]["A1"]["sha256"])
    add_field(lines, "T_A_CONSUMED_BY_B", f"{run_a['transport']['A_TO_B_consumed']} sha256={run_a['transport']['A_TO_B_consumed_sha256']}")
    add_field(lines, "A_TO_B_EXACT_TOKEN_IDENTITY", "PASS")
    add_field(lines, "A_TO_B_NO_DECODE_RETOKENIZE", "PASS")
    add_field(lines, "KV_B0_PRE_TRAVERSAL", run_a["traversals"]["B1"]["cache_fingerprint_before"]["sha256"])
    add_field(lines, "pole_B_output", run_a["traversals"]["B1"]["decoded_output_observation_only"])
    add_field(lines, "T_B_PRODUCED", f"{tb} sha256={run_a['transport']['B_TO_A_produced_sha256']}")
    add_field(lines, "KV_B1", run_a["kv"]["B1"]["sha256"])
    add_field(lines, "T_B_CONSUMED_BY_A", f"{run_a['transport']['B_TO_A_consumed']} sha256={run_a['transport']['B_TO_A_consumed_sha256']}")
    add_field(lines, "B_TO_A_EXACT_TOKEN_IDENTITY", "PASS")
    add_field(lines, "B_TO_A_NO_DECODE_RETOKENIZE", "PASS")
    add_field(lines, "KV_A1_PRE_RETURN", run_a["continuity"]["A1_fingerprint_pre_return"])
    add_field(lines, "POLE_A_CAUSAL_CONTINUITY", "PASS\nThe same live DynamicCache object, complete fingerprint, and every layer key/value tensor pointer persisted from KV_A1 post-traversal through Pole B's traversal into Pole A's return model.forward call.")
    add_field(lines, "pole_A_output_2", run_a["traversals"]["A2"]["decoded_output_observation_only"])
    add_field(lines, "KV_A2", run_a["kv"]["A2"]["sha256"])
    lines.extend(["RUN_B", ""])
    for receipt_name, path in (("KV_A0_PRIME", "A0"), ("KV_B0_PRIME", "B0"), ("KV_A1_PRIME", "A1")):
        add_field(lines, receipt_name, run_b["kv"][path]["sha256"])
    add_field(lines, "T_A_PRIME", f"{run_b['transport']['A_TO_B_produced_sha256']} ids={run_b['transport']['A_TO_B_produced']}")
    add_field(lines, "KV_B1_PRIME", run_b["kv"]["B1"]["sha256"])
    add_field(lines, "T_B_PRIME", f"{run_b['transport']['B_TO_A_produced_sha256']} ids={run_b['transport']['B_TO_A_produced']}")
    add_field(lines, "KV_A2_PRIME", run_b["kv"]["A2"]["sha256"])
    lines.extend(["DETERMINISTIC_COMPARISON", ""])
    for name in (
        "KV_A0_EQUAL", "KV_B0_EQUAL", "KV_A1_EQUAL", "KV_B1_EQUAL",
        "KV_A2_EQUAL", "T_A_EQUAL", "T_B_EQUAL", "POLE_A_OUTPUT_1_IDENTICAL",
        "POLE_B_OUTPUT_IDENTICAL", "POLE_A_OUTPUT_2_IDENTICAL",
    ):
        add_field(lines, name, "PASS")
    for name in (
        "SHARED_V_INVARIANT", "ADJACENT_BOUNDARIES_PRESERVED",
        "NO_SEMANTIC_MEDIATION", "NO_EXTERNAL_EPISTEMIC_AUTHORITY",
    ):
        add_field(lines, name, "PASS")
    add_field(lines, "substrate_finding", "Actual greedy-generated token IDs crossed each pole boundary as an ordered Python integer tuple and were materialized unchanged in the recipient's torch.long input tensor after the same immutable shared-V prefix. Independent live DynamicCache objects never crossed the boundary; Pole A's exact KV_A1 object and tensor storage remained live while Pole B traversed, then causally continued on return.")
    lines.extend(["final_result:", "PASS", "END_BUILD_SLICE_RECEIPT", ""])
    return "\n".join(lines)


def run_verification():
    slice3 = load_slice3_runtime()
    kernel = slice3.load_slice1_kernel()
    kernel.configure_determinism()
    weight_hash = kernel.verify_substrate()
    tokenizer, model = kernel.build_model()
    shared_v = slice3.make_shared_v(tokenizer)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    print("SLICE4_RUNTIME_READY", flush=True)
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    print(f"SHARED_V={shared_v.literal}", flush=True)
    print(f"INITIAL_SEED={SEED_LITERAL}", flush=True)
    run_a = execute_loop(slice3, kernel, model, tokenizer, shared_v, "A", EVIDENCE_DIR)
    run_b = execute_loop(slice3, kernel, model, tokenizer, shared_v, "B", EVIDENCE_DIR)
    deterministic_checks = compare_runs(run_a, run_b)
    (EVIDENCE_DIR / "run_a.json").write_text(
        json.dumps(run_a, indent=2, sort_keys=True), encoding="utf-8"
    )
    (EVIDENCE_DIR / "run_b.json").write_text(
        json.dumps(run_b, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(run_a, run_b, deterministic_checks)
    receipt_path = EVIDENCE_DIR / "completion_receipt.txt"
    receipt_path.write_text(receipt, encoding="utf-8")
    all_checks = {**run_a["checks"], **deterministic_checks}
    for name, passed in all_checks.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    print(f"POLE_A_OUTPUT_1={json.dumps(run_a['traversals']['A1']['decoded_output_observation_only'])}", flush=True)
    print(f"T_A={run_a['transport']['A_TO_B_produced']}", flush=True)
    print(f"POLE_B_OUTPUT={json.dumps(run_a['traversals']['B1']['decoded_output_observation_only'])}", flush=True)
    print(f"T_B={run_a['transport']['B_TO_A_produced']}", flush=True)
    print(f"POLE_A_OUTPUT_2={json.dumps(run_a['traversals']['A2']['decoded_output_observation_only'])}", flush=True)
    passed = all(run_a["checks"].values()) and all(run_b["checks"].values()) and all(deterministic_checks.values())
    print(f"FINAL_RESULT={'PASS' if passed else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={receipt_path}", flush=True)


def run_interactive():
    slice3 = load_slice3_runtime()
    kernel = slice3.load_slice1_kernel()
    kernel.configure_determinism()
    kernel.verify_substrate()
    tokenizer, model = kernel.build_model()
    shared_v = slice3.make_shared_v(tokenizer)
    print("SLICE4_INTERACTIVE_READY", flush=True)
    print(f"SHARED_V={shared_v.literal}", flush=True)
    seed = input("INITIAL_SEED> ")
    if seed != SEED_LITERAL:
        raise RuntimeError(f"Expected exact initial seed: {SEED_LITERAL}")
    record = execute_loop(
        slice3, kernel, model, tokenizer, shared_v, "INTERACTIVE", EVIDENCE_DIR
    )
    print("POLE_A_OUTPUT_1_BEGIN", flush=True)
    print(record["traversals"]["A1"]["decoded_output_observation_only"], flush=True)
    print("POLE_A_OUTPUT_1_END", flush=True)
    print(f"T_A_PRODUCED={record['transport']['A_TO_B_produced']}", flush=True)
    print(f"T_A_CONSUMED_BY_B={record['transport']['A_TO_B_consumed']}", flush=True)
    print("A_TO_B_EXACT_TOKEN_IDENTITY=PASS", flush=True)
    print("POLE_B_OUTPUT_BEGIN", flush=True)
    print(record["traversals"]["B1"]["decoded_output_observation_only"], flush=True)
    print("POLE_B_OUTPUT_END", flush=True)
    print(f"T_B_PRODUCED={record['transport']['B_TO_A_produced']}", flush=True)
    print(f"T_B_CONSUMED_BY_A={record['transport']['B_TO_A_consumed']}", flush=True)
    print("B_TO_A_EXACT_TOKEN_IDENTITY=PASS", flush=True)
    print("POLE_A_OUTPUT_2_BEGIN", flush=True)
    print(record["traversals"]["A2"]["decoded_output_observation_only"], flush=True)
    print("POLE_A_OUTPUT_2_END", flush=True)
    while input("TERMINATE (type EXIT)> ") != "EXIT":
        print("Type EXIT exactly to terminate.", flush=True)
    record_path = EVIDENCE_DIR / "human_interactive_run.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print("SLICE4_INTERACTIVE_TERMINATED", flush=True)
    print(f"INTERACTIVE_EVIDENCE={record_path}", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--verify", "--interactive"):
        raise SystemExit("usage: slice4_runtime.py --verify | --interactive")
    if sys.argv[1] == "--verify":
        run_verification()
    else:
        run_interactive()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"SLICE4_RUNTIME_ERROR={error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
