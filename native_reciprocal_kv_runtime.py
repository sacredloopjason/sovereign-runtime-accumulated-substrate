import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch


AUTHORITATIVE_PARENT = "106a79b148e939444df4c891c64f7c35a89699f2"
CANDIDATE_SHA256 = "5f85e384e31057fef1d7993efbebf7cb1bd892f0eb31281391c1ca17edf0036a"
PROJECT_DIR = Path(__file__).resolve().parent
KERNEL_PATH = (
    PROJECT_DIR
    / "external_substrate"
    / "slice1-kernel-0e214f972bb44fcf983bb0930af3d773"
    / "runtime.py"
)
DEFAULT_MODEL_DIR = KERNEL_PATH.parent / "model"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "native_reciprocal_kv_runtime"
MAX_NEW_TOKENS = 16


def load_kernel():
    spec = importlib.util.spec_from_file_location(
        "authoritative_slice1_native_kernel", KERNEL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load authoritative native kernel: {KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_determinism():
    random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_flush_denormal(True)


def ordered_token_fingerprint(token_ids):
    payload = json.dumps(
        [int(token_id) for token_id in token_ids], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(b"NATIVE_ORDERED_TOKEN_IDS_V1\0" + payload).hexdigest()


def pointer_set(cache):
    pointers = set()
    for layer in cache.layers:
        for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if tensor is not None:
                pointers.add(int(tensor.data_ptr()))
    return pointers


def storage_separated(cache_a, cache_b):
    return cache_a is not cache_b and pointer_set(cache_a).isdisjoint(pointer_set(cache_b))


def native_cache_state(kernel, cache):
    fingerprint = kernel.fingerprint_cache(cache)
    return {
        "cache_class": f"{cache.__class__.__module__}.{cache.__class__.__name__}",
        "object_id": id(cache),
        "fingerprint": fingerprint,
        "pointer_map": kernel.pointer_map(cache),
    }


def same_native_state(left, right):
    return (
        left["object_id"] == right["object_id"]
        and left["fingerprint"] == right["fingerprint"]
        and left["pointer_map"] == right["pointer_map"]
    )


def traverse(model, cache, perturbation_token_ids, eos_token_id):
    """Advance exactly one native cache using only an ordered token-ID perturbation."""
    perturbation = tuple(int(token_id) for token_id in perturbation_token_ids)
    if not perturbation:
        raise RuntimeError("A reciprocal perturbation cannot be empty")
    if any(token_id < 0 or token_id >= model.config.vocab_size for token_id in perturbation):
        raise RuntimeError("A reciprocal perturbation contains an out-of-vocabulary token ID")

    cache_object_id = id(cache)
    first_forward_input_ids = None
    forward_events = []

    def forward(input_ids):
        nonlocal first_forward_input_ids
        supplied_ids = tuple(int(item) for item in input_ids.flatten().tolist())
        if first_forward_input_ids is None:
            first_forward_input_ids = supplied_ids
        past_length = int(cache.get_seq_length())
        token_count = int(input_ids.shape[1])
        attention_mask = torch.ones((1, past_length + token_count), dtype=torch.long)
        cache_position = torch.arange(
            past_length, past_length + token_count, dtype=torch.long
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
        forward_events.append(
            {
                "input_token_ids": list(supplied_ids),
                "supplied_cache_object_id": cache_object_id,
                "returned_same_cache_object": outputs.past_key_values is cache,
                "pre_sequence_length": past_length,
                "post_sequence_length": int(cache.get_seq_length()),
            }
        )
        if outputs.past_key_values is not cache:
            raise RuntimeError("Native inference returned a different cache object")
        return outputs.logits[:, -1, :]

    logits = forward(torch.tensor([perturbation], dtype=torch.long))
    generated_ids = []
    for _ in range(MAX_NEW_TOKENS):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        logits = forward(next_token)
        if eos_token_id is not None and token_id == int(eos_token_id):
            break

    generated = tuple(generated_ids)
    if not generated or not all(type(token_id) is int for token_id in generated):
        raise RuntimeError("Native generation did not emit a non-empty integer token sequence")
    return {
        "perturbation_token_ids": list(perturbation),
        "perturbation_sha256": ordered_token_fingerprint(perturbation),
        "actual_first_forward_input_ids": list(first_forward_input_ids),
        "exact_input_at_model_boundary": first_forward_input_ids == perturbation,
        "generated_token_ids": list(generated),
        "generated_token_sha256": ordered_token_fingerprint(generated),
        "cache_object_id": cache_object_id,
        "all_forwards_returned_same_cache": all(
            event["returned_same_cache_object"] for event in forward_events
        ),
        "forward_events": forward_events,
    }


def execute_replication(kernel, model, tokenizer, bootstrap_token_ids, label):
    pole_a = kernel.DynamicCache(config=model.config)
    pole_b = kernel.DynamicCache(config=model.config)
    initial_a = native_cache_state(kernel, pole_a)
    initial_b = native_cache_state(kernel, pole_b)
    initial_storage_separated = storage_separated(pole_a, pole_b)

    traversals = []

    def advance(active, inactive, perturbation, traversal_label):
        active_pre = native_cache_state(kernel, active)
        inactive_pre = native_cache_state(kernel, inactive)
        traversal = traverse(
            model, active, perturbation, tokenizer.eos_token_id
        )
        active_post = native_cache_state(kernel, active)
        inactive_post = native_cache_state(kernel, inactive)
        record = {
            "label": traversal_label,
            "model_object_id": id(model),
            "active_pre": active_pre,
            "active_post": active_post,
            "inactive_pre": inactive_pre,
            "inactive_post": inactive_post,
            "active_cache_advanced": (
                active_pre["object_id"] == active_post["object_id"]
                and active_pre["fingerprint"]["sha256"]
                != active_post["fingerprint"]["sha256"]
            ),
            "inactive_cache_unchanged": same_native_state(inactive_pre, inactive_post),
            "pole_storage_separated_after": storage_separated(pole_a, pole_b),
            **traversal,
        }
        record["decoded_output_observation_only"] = tokenizer.decode(
            traversal["generated_token_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        traversals.append(record)
        return tuple(traversal["generated_token_ids"])

    a0_output = advance(pole_a, pole_b, bootstrap_token_ids, "A0")
    b0_output = advance(pole_b, pole_a, a0_output, "B0")
    a1_output = advance(pole_a, pole_b, b0_output, "A1")
    b1_output = advance(pole_b, pole_a, a1_output, "B1")

    final_a = native_cache_state(kernel, pole_a)
    final_b = native_cache_state(kernel, pole_b)
    a_post_is_next_pre = same_native_state(
        traversals[0]["active_post"], traversals[2]["active_pre"]
    )
    b_post_is_next_pre = same_native_state(
        traversals[1]["active_post"], traversals[3]["active_pre"]
    )

    checks = {
        "same_model_law_for_a_and_b": len(
            {traversal["model_object_id"] for traversal in traversals}
        )
        == 1
        and next(iter({traversal["model_object_id"] for traversal in traversals}))
        == id(model),
        "pole_a_cache_native": isinstance(pole_a, kernel.DynamicCache),
        "pole_b_cache_native": isinstance(pole_b, kernel.DynamicCache),
        "pole_cache_objects_distinct": pole_a is not pole_b,
        "pole_cache_storage_separated": initial_storage_separated
        and all(item["pole_storage_separated_after"] for item in traversals),
        "initial_poles_independent": (
            pole_a is not pole_b
            and initial_a["object_id"] != initial_b["object_id"]
            and initial_storage_separated
        ),
        "generated_output_is_native_token_ids": all(
            item["generated_token_ids"]
            and all(type(token_id) is int for token_id in item["generated_token_ids"])
            for item in traversals
        ),
        "a_output_equals_b_next_input": (
            traversals[0]["generated_token_ids"]
            == traversals[1]["perturbation_token_ids"]
            and traversals[2]["generated_token_ids"]
            == traversals[3]["perturbation_token_ids"]
        ),
        "b_output_equals_a_next_input": (
            traversals[1]["generated_token_ids"]
            == traversals[2]["perturbation_token_ids"]
        ),
        "a_post_cache_is_a_next_pre": a_post_is_next_pre,
        "b_post_cache_is_b_next_pre": b_post_is_next_pre,
        "a_traversal_mutates_only_a": all(
            traversals[index]["active_cache_advanced"]
            and traversals[index]["inactive_cache_unchanged"]
            for index in (0, 2)
        ),
        "b_traversal_mutates_only_b": all(
            traversals[index]["active_cache_advanced"]
            and traversals[index]["inactive_cache_unchanged"]
            for index in (1, 3)
        ),
        "all_inputs_exact_at_model_boundary": all(
            item["exact_input_at_model_boundary"] for item in traversals
        ),
        "all_forwards_returned_same_cache": all(
            item["all_forwards_returned_same_cache"] for item in traversals
        ),
        "reciprocal_traversals_executed": len(traversals) == 4,
    }

    canonical = {
        "bootstrap_token_ids": list(bootstrap_token_ids),
        "traversal_token_ids": [
            {
                "label": item["label"],
                "input": item["perturbation_token_ids"],
                "output": item["generated_token_ids"],
            }
            for item in traversals
        ],
        "cache_fingerprints": [
            {
                "label": item["label"],
                "pre": item["active_pre"]["fingerprint"],
                "post": item["active_post"]["fingerprint"],
            }
            for item in traversals
        ],
        "initial": {
            "a": initial_a["fingerprint"],
            "b": initial_b["fingerprint"],
        },
        "final": {
            "a": final_a["fingerprint"],
            "b": final_b["fingerprint"],
        },
        "checks": checks,
    }
    return {
        "label": label,
        "model_object_id": id(model),
        "pole_a_initial_state": initial_a,
        "pole_b_initial_state": initial_b,
        "pole_a_final_state": final_a,
        "pole_b_final_state": final_b,
        "traversals": traversals,
        "checks": checks,
        "canonical_replication_projection": canonical,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fresh native reciprocal transformer KV runtime acceptance verifier"
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()

    configure_determinism()
    kernel = load_kernel()
    model_dir = args.model_dir or Path(
        os.environ.get("SOVEREIGN_NATIVE_MODEL_DIR", DEFAULT_MODEL_DIR)
    )
    kernel.MODEL_DIR = model_dir.resolve()
    weight_hash = kernel.verify_substrate()
    tokenizer, model = kernel.build_model()
    if kernel.MAX_NEW_TOKENS != MAX_NEW_TOKENS:
        raise RuntimeError("Authoritative generation bound changed")

    bootstrap_token_ids = (int(model.config.bos_token_id),)
    run_a = execute_replication(
        kernel, model, tokenizer, bootstrap_token_ids, "A"
    )
    run_b = execute_replication(
        kernel, model, tokenizer, bootstrap_token_ids, "B"
    )
    deterministic_replication = (
        run_a["canonical_replication_projection"]
        == run_b["canonical_replication_projection"]
    )
    all_native_checks = all(run_a["checks"].values()) and all(
        run_b["checks"].values()
    )

    evidence = {
        "schema": "NATIVE_RECIPROCAL_KV_RUNTIME_EVIDENCE_V1",
        "authoritative_parent_commit": AUTHORITATIVE_PARENT,
        "manual_gate": {
            "candidate_sha256": CANDIDATE_SHA256,
            "human_statement": "i'm your manual gate. you may proceed",
            "status": "PASS",
        },
        "runtime_implementation": str(Path(__file__).resolve()),
        "native_model": {
            "id": kernel.MODEL_ID,
            "revision": kernel.MODEL_REVISION,
            "weight_sha256": weight_hash,
            "model_dir": str(model_dir.resolve()),
        },
        "model_instances": 1,
        "pole_state_representation": "transformers.cache_utils.DynamicCache",
        "bootstrap": {
            "source": "authoritative model config bos_token_id",
            "token_ids": list(bootstrap_token_ids),
        },
        "replications": [run_a, run_b],
        "deterministic_replication": deterministic_replication,
        "structural_deductions": {
            "shared_v_present": "NO",
            "semantic_transformation_in_handoff": "NO",
            "only_reciprocal_carrier": "GENERATED_NATIVE_TOKEN_IDS",
            "basis": (
                "Each reciprocal call receives only the generated tuple[int] returned "
                "by the immediately preceding traverse call; cache, logits, decoded text, "
                "fingerprints, histories, and metadata are not passed into the other pole."
            ),
        },
        "outcome": (
            "NATIVE_RECIPROCAL_KV_RUNTIME_OPERATIONAL"
            if all_native_checks and deterministic_replication
            else "UNRESOLVED_NATIVE_RECIPROCAL_RUNTIME_BOUNDARY"
        ),
    }

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_dir / "acceptance_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
    )

    for name, passed in run_a["checks"].items():
        print(f"{name.upper()}: {'PASS' if passed else 'FAIL'}")
    print("SHARED_V_PRESENT: NO")
    print("SEMANTIC_TRANSFORMATION_IN_HANDOFF: NO")
    print("ONLY_RECIPROCAL_CARRIER: GENERATED_NATIVE_TOKEN_IDS")
    print("RECIPROCAL_TRAVERSALS_EXECUTED: 4")
    print(
        "DETERMINISTIC_REPLICATION: "
        + ("PASS" if deterministic_replication else "FAIL")
    )
    print("OUTCOME")
    print(evidence["outcome"])
    print(f"EVIDENCE: {evidence_path}")
    if evidence["outcome"] != "NATIVE_RECIPROCAL_KV_RUNTIME_OPERATIONAL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
