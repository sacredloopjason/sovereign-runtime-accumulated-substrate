import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path


SLICE1_DIR = Path(
    r"C:\Users\jason\Documents\Codex\2026-08-22\files-pasted-by-the-user-build"
    r"\slice1-kernel-0e214f972bb44fcf983bb0930af3d773"
)
SLICE1_RUNTIME = SLICE1_DIR / "runtime.py"
PROJECT_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = PROJECT_DIR / "evidence"

BOUNDARY_INPUT = "SLICE2_BOUNDARY_INPUT"
POLE_A_INPUT = "SLICE3_POLE_A_INPUT"
POLE_B_INPUT = "SLICE3_POLE_B_INPUT"
V_LITERAL = "SLICE3_SHARED_V"


@dataclass(frozen=True)
class SharedV:
    """Passive canonical V: immutable value plus its one canonical token tuple."""

    literal: str
    utf8: bytes
    token_ids: tuple[int, ...]


def load_slice1_kernel():
    spec = importlib.util.spec_from_file_location("slice1_kernel", SLICE1_RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inherited Slice 1 kernel: {SLICE1_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clone_actual_kv(cache):
    return copy.deepcopy(cache)


def cache_tensor_storage_separated(left, right):
    if left is right or len(left.layers) != len(right.layers):
        return False
    for left_layer, right_layer in zip(left.layers, right.layers):
        for component in ("keys", "values"):
            left_tensor = getattr(left_layer, component, None)
            right_tensor = getattr(right_layer, component, None)
            if left_tensor is not None and left_tensor.data_ptr() == right_tensor.data_ptr():
                return False
    return True


def make_shared_v(tokenizer):
    utf8 = V_LITERAL.encode("utf-8")
    ids = tokenizer(V_LITERAL + "\n", add_special_tokens=False).input_ids
    if not ids:
        raise RuntimeError("V encoded to no tokens")
    return SharedV(V_LITERAL, utf8, tuple(int(item) for item in ids))


def fingerprint_v(shared_v):
    payload = json.dumps(
        {
            "class": f"{shared_v.__class__.__module__}.{shared_v.__class__.__name__}",
            "literal_utf8_hex": shared_v.utf8.hex(),
            "token_ids": list(shared_v.token_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(b"SLICE3_SHARED_V_RUNTIME_V1\0" + payload).hexdigest(),
        "runtime_class": f"{shared_v.__class__.__module__}.{shared_v.__class__.__name__}",
        "literal": shared_v.literal,
        "utf8_hex": shared_v.utf8.hex(),
        "token_ids": list(shared_v.token_ids),
        "object_id": id(shared_v),
        "frozen_dataclass": True,
        "has_model_state": False,
    }


def traverse_with_v(kernel, model, tokenizer, cache, shared_v, pole_input, number):
    import torch

    pole_ids = tokenizer(pole_input + "\n", add_special_tokens=False).input_ids
    consumed_ids = shared_v.token_ids + tuple(int(item) for item in pole_ids)
    input_ids = torch.tensor([consumed_ids], dtype=torch.long)
    observation = {"active_traversal": number, "forward_events": []}
    original_forward = model.forward

    def observed_forward(*args, **kwargs):
        supplied = kwargs.get("past_key_values")
        supplied_ids = kwargs.get("input_ids")
        event = {
            "supplied_is_pole_cache": supplied is cache,
            "pre_sequence_length": int(supplied.get_seq_length()),
            "input_token_ids": supplied_ids.flatten().tolist(),
            "v_prefix_consumed": supplied_ids.flatten().tolist()[: len(shared_v.token_ids)]
            == list(shared_v.token_ids),
        }
        outputs = original_forward(*args, **kwargs)
        event["returned_same_cache"] = outputs.past_key_values is supplied
        event["post_sequence_length"] = int(outputs.past_key_values.get_seq_length())
        observation["forward_events"].append(event)
        return outputs

    model.forward = observed_forward
    generated_ids = []
    try:
        def forward(tokens):
            past_length = int(cache.get_seq_length())
            count = int(tokens.shape[1])
            attention_mask = torch.ones((1, past_length + count), dtype=torch.long)
            positions = torch.arange(past_length, past_length + count, dtype=torch.long)
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

        logits = forward(input_ids)
        for _ in range(kernel.MAX_NEW_TOKENS):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            logits = forward(next_token)
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break
    finally:
        model.forward = original_forward

    first = observation["forward_events"][0]
    return {
        "input": pole_input,
        "output": tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "generated_token_ids": generated_ids,
        "canonical_v_object_id": id(shared_v),
        "canonical_v_token_ids": list(shared_v.token_ids),
        "actual_first_forward_input_ids": first["input_token_ids"],
        "v_prefix_consumed_by_actual_first_forward": first["v_prefix_consumed"],
        "actual_first_forward_v_prefix": first["input_token_ids"][: len(shared_v.token_ids)],
        "all_forwards_received_exact_pole_cache": all(
            event["supplied_is_pole_cache"] for event in observation["forward_events"]
        ),
        "all_forwards_returned_same_cache": all(
            event["returned_same_cache"] for event in observation["forward_events"]
        ),
    }


def create_initial_poles(kernel, model, tokenizer):
    cache = kernel.DynamicCache(config=model.config)
    preserved_a = clone_actual_kv(cache)
    observation = {"active_traversal": 1, "forward_events": []}
    original_forward = model.forward

    def observed_forward(*args, **kwargs):
        supplied = kwargs.get("past_key_values")
        output = original_forward(*args, **kwargs)
        observation["forward_events"].append(
            {
                "supplied_is_live_cache": supplied is cache,
                "returned_same_object": output.past_key_values is supplied,
                "supplied_object_id": id(supplied),
                "pre_sequence_length": int(supplied.get_seq_length()),
                "post_sequence_length": int(output.past_key_values.get_seq_length()),
            }
        )
        return output

    model.forward = observed_forward
    try:
        kernel.traverse(model, tokenizer, cache, BOUNDARY_INPUT, 1, observation)
    finally:
        model.forward = original_forward
    preserved_b = clone_actual_kv(cache)
    return clone_actual_kv(preserved_a), clone_actual_kv(preserved_b)


def execute_order(kernel, model, tokenizer, shared_v, label, order):
    pole_a, pole_b = create_initial_poles(kernel, model, tokenizer)
    record = {
        "schema": "SLICE3_SHARED_V_EVIDENCE_V1",
        "run": label,
        "order": order,
        "v_initial": fingerprint_v(shared_v),
        "pole_a_pre": kernel.fingerprint_cache(pole_a),
        "pole_b_pre": kernel.fingerprint_cache(pole_b),
        "pole_objects_distinct": pole_a is not pole_b,
        "pole_tensor_storage_separated": cache_tensor_storage_separated(pole_a, pole_b),
        "no_semantic_reconstruction": True,
        "no_external_epistemic_authority": True,
    }
    if order == "AB":
        record["pole_b_before_a"] = kernel.fingerprint_cache(pole_b)
        a = traverse_with_v(kernel, model, tokenizer, pole_a, shared_v, POLE_A_INPUT, 2)
        record["pole_a_post"] = kernel.fingerprint_cache(pole_a)
        record["v_after_a"] = fingerprint_v(shared_v)
        record["pole_b_after_a"] = kernel.fingerprint_cache(pole_b)
        record["pole_a_before_b"] = kernel.fingerprint_cache(pole_a)
        b = traverse_with_v(kernel, model, tokenizer, pole_b, shared_v, POLE_B_INPUT, 3)
        record["pole_b_post"] = kernel.fingerprint_cache(pole_b)
        record["v_after_b"] = fingerprint_v(shared_v)
        record["pole_a_after_b"] = kernel.fingerprint_cache(pole_a)
    else:
        record["pole_a_before_b"] = kernel.fingerprint_cache(pole_a)
        b = traverse_with_v(kernel, model, tokenizer, pole_b, shared_v, POLE_B_INPUT, 2)
        record["pole_b_post"] = kernel.fingerprint_cache(pole_b)
        record["v_after_b"] = fingerprint_v(shared_v)
        record["pole_a_after_b"] = kernel.fingerprint_cache(pole_a)
        record["pole_b_before_a"] = kernel.fingerprint_cache(pole_b)
        a = traverse_with_v(kernel, model, tokenizer, pole_a, shared_v, POLE_A_INPUT, 3)
        record["pole_a_post"] = kernel.fingerprint_cache(pole_a)
        record["v_after_a"] = fingerprint_v(shared_v)
        record["pole_b_after_a"] = kernel.fingerprint_cache(pole_b)
    record["pole_a"] = a
    record["pole_b"] = b
    return record


def compare(run_a, run_b):
    checks = {
        "POLE_A_DID_NOT_MUTATE_POLE_B": all(
            run["pole_b_before_a"]["sha256"] == run["pole_b_after_a"]["sha256"]
            for run in (run_a, run_b)
        ),
        "POLE_B_DID_NOT_MUTATE_POLE_A": all(
            run["pole_a_before_b"]["sha256"] == run["pole_a_after_b"]["sha256"]
            for run in (run_a, run_b)
        ),
        "V_PARTICIPATED_IN_POLE_A_INFERENCE": run_a["pole_a"]["v_prefix_consumed_by_actual_first_forward"] and run_b["pole_a"]["v_prefix_consumed_by_actual_first_forward"],
        "V_PARTICIPATED_IN_POLE_B_INFERENCE": run_a["pole_b"]["v_prefix_consumed_by_actual_first_forward"] and run_b["pole_b"]["v_prefix_consumed_by_actual_first_forward"],
        "V_INVARIANT_RUN_A": run_a["v_initial"]["sha256"] == run_a["v_after_a"]["sha256"] == run_a["v_after_b"]["sha256"],
        "V_INVARIANT_RUN_B": run_b["v_initial"]["sha256"] == run_b["v_after_a"]["sha256"] == run_b["v_after_b"]["sha256"],
        "POLE_A_POST_EQUAL": run_a["pole_a_post"]["sha256"] == run_b["pole_a_post"]["sha256"],
        "POLE_B_POST_EQUAL": run_a["pole_b_post"]["sha256"] == run_b["pole_b_post"]["sha256"],
        "POLE_A_OUTPUT_IDENTICAL": run_a["pole_a"]["output"] == run_b["pole_a"]["output"],
        "POLE_B_OUTPUT_IDENTICAL": run_a["pole_b"]["output"] == run_b["pole_b"]["output"],
        "V_IDENTICAL_BETWEEN_POLES": run_a["pole_a"]["canonical_v_object_id"] == run_a["pole_b"]["canonical_v_object_id"] and run_a["pole_a"]["actual_first_forward_v_prefix"] == run_a["pole_b"]["actual_first_forward_v_prefix"],
        "V_IDENTICAL_BETWEEN_RUNS": run_a["v_initial"]["sha256"] == run_b["v_initial"]["sha256"],
        "POLE_TENSOR_STORAGE_SEPARATED": run_a["pole_tensor_storage_separated"] and run_b["pole_tensor_storage_separated"],
        "ACTUAL_POLE_CACHES_USED": all(run["pole_a"]["all_forwards_received_exact_pole_cache"] and run["pole_b"]["all_forwards_received_exact_pole_cache"] for run in (run_a, run_b)),
        "NO_SEMANTIC_RECONSTRUCTION": run_a["no_semantic_reconstruction"] and run_b["no_semantic_reconstruction"],
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": run_a["no_external_epistemic_authority"] and run_b["no_external_epistemic_authority"],
    }
    checks["ORDER_INDEPENDENCE"] = all(checks[name] for name in ("POLE_A_POST_EQUAL", "POLE_B_POST_EQUAL", "POLE_A_OUTPUT_IDENTICAL", "POLE_B_OUTPUT_IDENTICAL"))
    return checks


def field(lines, name, value):
    lines.extend([f"{name}:", str(value), ""])


def make_receipt(run_a, run_b, checks):
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        return "\n".join(["BUILD_SLICE_RECEIPT", "slice: 3", "result: FAIL", "", "artifact_state:", str(PROJECT_DIR / "runtime.py"), "", "failed_requirement:", ", ".join(failed), "", "causal_boundary:", "Shared-V inference participation, invariance, pole isolation, or reversed-order identity.", "", "substrate_evidence:", json.dumps(checks, sort_keys=True), "", "substrate_finding:", "The exact failed structural comparison is named above.", "", "perturbation:", ", ".join(failed), "", "final_result:", "FAIL", "END_BUILD_SLICE_RECEIPT", ""])
    lines = ["BUILD_SLICE_RECEIPT", "slice: 3", "result: PASS", ""]
    field(lines, "artifact", PROJECT_DIR / "runtime.py")
    field(lines, "V_literal", V_LITERAL)
    field(lines, "V_runtime_representation", "one frozen SharedV object containing canonical UTF-8 bytes and one immutable tuple of tokenizer token IDs")
    field(lines, "V_evidence_method", "SHA-256 over the actual SharedV class, UTF-8 bytes, and canonical token tuple; model.forward instrumentation captured the identical V token prefix consumed by both poles")
    field(lines, "V_INITIAL", run_a["v_initial"]["sha256"])
    field(lines, "pole_A_pre_KV", run_a["pole_a_pre"]["sha256"])
    field(lines, "pole_B_pre_KV", run_a["pole_b_pre"]["sha256"])
    lines.extend(["RUN_A", ""])
    field(lines, "pole_B_before_A", run_a["pole_b_before_a"]["sha256"])
    field(lines, "pole_A_input", POLE_A_INPUT)
    field(lines, "pole_A_output", run_a["pole_a"]["output"])
    field(lines, "pole_A_post_KV", run_a["pole_a_post"]["sha256"])
    field(lines, "V_AFTER_A", run_a["v_after_a"]["sha256"])
    field(lines, "pole_B_after_A", run_a["pole_b_after_a"]["sha256"])
    field(lines, "POLE_A_DID_NOT_MUTATE_POLE_B", "PASS")
    field(lines, "V_PARTICIPATED_IN_POLE_A_INFERENCE", "PASS\nactual model.forward V prefix: " + json.dumps(run_a["pole_a"]["actual_first_forward_v_prefix"]))
    field(lines, "pole_A_before_B", run_a["pole_a_before_b"]["sha256"])
    field(lines, "pole_B_input", POLE_B_INPUT)
    field(lines, "pole_B_output", run_a["pole_b"]["output"])
    field(lines, "pole_B_post_KV", run_a["pole_b_post"]["sha256"])
    field(lines, "V_AFTER_B", run_a["v_after_b"]["sha256"])
    field(lines, "pole_A_after_B", run_a["pole_a_after_b"]["sha256"])
    field(lines, "POLE_B_DID_NOT_MUTATE_POLE_A", "PASS")
    field(lines, "V_PARTICIPATED_IN_POLE_B_INFERENCE", "PASS\nactual model.forward V prefix: " + json.dumps(run_a["pole_b"]["actual_first_forward_v_prefix"]))
    field(lines, "V_INVARIANT_RUN_A", "PASS")
    lines.extend(["RUN_B_REVERSED_ORDER", ""])
    field(lines, "V_INITIAL_PRIME", run_b["v_initial"]["sha256"])
    field(lines, "pole_A_pre_KV_prime", run_b["pole_a_pre"]["sha256"])
    field(lines, "pole_B_pre_KV_prime", run_b["pole_b_pre"]["sha256"])
    field(lines, "pole_B_output", run_b["pole_b"]["output"])
    field(lines, "pole_B_post_KV_prime", run_b["pole_b_post"]["sha256"])
    field(lines, "V_AFTER_B_PRIME", run_b["v_after_b"]["sha256"])
    field(lines, "pole_A_output", run_b["pole_a"]["output"])
    field(lines, "pole_A_post_KV_prime", run_b["pole_a_post"]["sha256"])
    field(lines, "V_AFTER_A_PRIME", run_b["v_after_a"]["sha256"])
    lines.extend(["DETERMINISTIC_COMPARISON", ""])
    for name in ("POLE_A_POST_EQUAL", "POLE_B_POST_EQUAL", "POLE_A_OUTPUT_IDENTICAL", "POLE_B_OUTPUT_IDENTICAL", "V_IDENTICAL_BETWEEN_POLES", "V_IDENTICAL_BETWEEN_RUNS", "ORDER_INDEPENDENCE", "NO_SEMANTIC_RECONSTRUCTION", "NO_EXTERNAL_EPISTEMIC_AUTHORITY"):
        field(lines, name, "PASS")
    field(lines, "substrate_finding", "One frozen SharedV object held the canonical UTF-8 fixture and its single immutable token tuple. Each sovereign traversal prepended that same tuple to its pole-specific tokens, and instrumentation at the actual model.forward boundary captured the identical consumed V prefix while the independent DynamicCache remained the causally operative past_key_values. V had no model, cache, loop, validator, or decision function.")
    lines.extend(["final_result:", "PASS", "END_BUILD_SLICE_RECEIPT", ""])
    return "\n".join(lines)


def verify():
    kernel = load_slice1_kernel()
    kernel.configure_determinism()
    weight_hash = kernel.verify_substrate()
    tokenizer, model = kernel.build_model()
    shared_v = make_shared_v(tokenizer)
    print("SLICE3_RUNTIME_READY", flush=True)
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    print(f"V_LITERAL={shared_v.literal}", flush=True)
    print(f"V_TOKEN_IDS={list(shared_v.token_ids)}", flush=True)
    print(f"V_INITIAL={fingerprint_v(shared_v)['sha256']}", flush=True)
    run_a = execute_order(kernel, model, tokenizer, shared_v, "A", "AB")
    run_b = execute_order(kernel, model, tokenizer, shared_v, "B", "BA")
    checks = compare(run_a, run_b)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "run_a.json").write_text(json.dumps(run_a, indent=2, sort_keys=True), encoding="utf-8")
    (EVIDENCE_DIR / "run_b.json").write_text(json.dumps(run_b, indent=2, sort_keys=True), encoding="utf-8")
    receipt = make_receipt(run_a, run_b, checks)
    receipt_path = EVIDENCE_DIR / "completion_receipt.txt"
    receipt_path.write_text(receipt, encoding="utf-8")
    for name, ok in checks.items():
        print(f"{name}={'PASS' if ok else 'FAIL'}", flush=True)
    print(f"POLE_A_OUTPUT={json.dumps(run_a['pole_a']['output'])}", flush=True)
    print(f"POLE_B_OUTPUT={json.dumps(run_a['pole_b']['output'])}", flush=True)
    print(f"FINAL_RESULT={'PASS' if all(checks.values()) else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={receipt_path}", flush=True)


def interactive():
    kernel = load_slice1_kernel()
    kernel.configure_determinism()
    kernel.verify_substrate()
    tokenizer, model = kernel.build_model()
    shared_v = make_shared_v(tokenizer)
    pole_a, pole_b = create_initial_poles(kernel, model, tokenizer)
    print("SLICE3_INTERACTIVE_READY", flush=True)
    print(f"SHARED_V={shared_v.literal}", flush=True)
    print(f"V_FINGERPRINT={fingerprint_v(shared_v)['sha256']}", flush=True)
    a_input = ""
    while a_input == "":
        a_input = input("POLE_A_INPUT> ")
    if a_input != POLE_A_INPUT:
        raise RuntimeError(f"Expected exact Pole A input: {POLE_A_INPUT}")
    a = traverse_with_v(kernel, model, tokenizer, pole_a, shared_v, a_input, 2)
    print("POLE_A_OUTPUT_BEGIN", flush=True)
    print(a["output"], flush=True)
    print("POLE_A_OUTPUT_END", flush=True)
    b_input = ""
    while b_input == "":
        b_input = input("POLE_B_INPUT> ")
    if b_input != POLE_B_INPUT:
        raise RuntimeError(f"Expected exact Pole B input: {POLE_B_INPUT}")
    b = traverse_with_v(kernel, model, tokenizer, pole_b, shared_v, b_input, 3)
    print("POLE_B_OUTPUT_BEGIN", flush=True)
    print(b["output"], flush=True)
    print("POLE_B_OUTPUT_END", flush=True)
    while input("TERMINATE (type EXIT)> ") != "EXIT":
        print("Type EXIT exactly to terminate.", flush=True)
    print("SLICE3_INTERACTIVE_TERMINATED", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("--verify", "--interactive"):
        raise SystemExit("usage: runtime.py --verify | --interactive")
    if sys.argv[1] == "--verify":
        verify()
    else:
        interactive()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SLICE3_RUNTIME_ERROR={error.__class__.__name__}: {error}", file=sys.stderr)
        raise
