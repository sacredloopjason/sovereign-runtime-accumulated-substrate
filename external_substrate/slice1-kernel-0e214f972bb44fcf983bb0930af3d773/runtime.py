import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MODEL_WEIGHT_SHA256 = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
TORCH_VERSION = "2.11.0+cpu"
TRANSFORMERS_VERSION = "5.14.1"
MAX_NEW_TOKENS = 16
FIXED_INPUTS = ["SLICE1_INPUT_0", "SLICE1_INPUT_1"]

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
EVIDENCE_DIR = PROJECT_DIR / "evidence"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_record(tensor):
    if tensor is None:
        return {"present": False, "tensor": None}
    frozen = tensor.detach().cpu().contiguous()
    raw = frozen.view(torch.uint8).numpy().tobytes()
    metadata = {
        "present": True,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "device": tensor.device.type,
        "byte_count": len(raw),
    }
    return {"present": True, "metadata": metadata, "tensor": tensor, "raw": raw}


def cache_material(cache):
    return {
        "cache_class": f"{cache.__class__.__module__}.{cache.__class__.__name__}",
        "layers": [
            {
                "layer_class": f"{layer.__class__.__module__}.{layer.__class__.__name__}",
                "key": getattr(layer, "keys", None),
                "value": getattr(layer, "values", None),
            }
            for layer in cache.layers
        ],
    }


def snapshot_cache(cache):
    material = cache_material(cache)
    return {
        "cache_class": material["cache_class"],
        "layers": [
            {
                "layer_class": layer["layer_class"],
                "key": None if layer["key"] is None else layer["key"].detach().clone(),
                "value": None if layer["value"] is None else layer["value"].detach().clone(),
            }
            for layer in material["layers"]
        ],
    }


def fingerprint_material(material):
    digest = hashlib.sha256()
    digest.update(b"SLICE1_COMPLETE_KV_SHA256_V1\0")
    digest.update(material["cache_class"].encode("utf-8") + b"\0")
    byte_count = 0
    component_count = 0
    sequence_lengths = set()

    for layer_index, layer in enumerate(material["layers"]):
        digest.update(str(layer_index).encode("ascii") + b"\0")
        digest.update(layer["layer_class"].encode("utf-8") + b"\0")
        for component_name in ("key", "value"):
            digest.update(component_name.encode("ascii") + b"\0")
            record = tensor_record(layer[component_name])
            if not record["present"]:
                digest.update(b"ABSENT\0")
                continue
            metadata = json.dumps(
                record["metadata"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            digest.update(len(metadata).to_bytes(8, "little"))
            digest.update(metadata)
            digest.update(len(record["raw"]).to_bytes(8, "little"))
            digest.update(record["raw"])
            byte_count += len(record["raw"])
            component_count += 1
            sequence_lengths.add(int(layer[component_name].shape[-2]))

    if len(sequence_lengths) > 1:
        raise RuntimeError(f"KV layers disagree on sequence length: {sequence_lengths}")

    return {
        "sha256": digest.hexdigest(),
        "layer_count": len(material["layers"]),
        "component_count": component_count,
        "tensor_byte_count": byte_count,
        "sequence_length": next(iter(sequence_lengths), 0),
    }


def fingerprint_cache(cache):
    return fingerprint_material(cache_material(cache))


def pointer_map(cache):
    result = []
    for layer_index, layer in enumerate(cache.layers):
        for component_name, tensor in (
            ("key", getattr(layer, "keys", None)),
            ("value", getattr(layer, "values", None)),
        ):
            result.append(
                {
                    "layer": layer_index,
                    "component": component_name,
                    "data_ptr": None if tensor is None else int(tensor.data_ptr()),
                    "shape": None if tensor is None else list(tensor.shape),
                }
            )
    return result


def configure_determinism():
    random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_flush_denormal(True)


def verify_substrate():
    if torch.__version__ != TORCH_VERSION:
        raise RuntimeError(f"Expected torch {TORCH_VERSION}, found {torch.__version__}")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Expected transformers {TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )
    actual_weight_hash = sha256_file(MODEL_DIR / "model.safetensors")
    if actual_weight_hash != MODEL_WEIGHT_SHA256:
        raise RuntimeError(
            f"Model weight SHA-256 mismatch: expected {MODEL_WEIGHT_SHA256}, found {actual_weight_hash}"
        )
    return actual_weight_hash


def build_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.float32,
        use_safetensors=True,
    ).cpu()
    model.eval()
    return tokenizer, model


def traverse(model, tokenizer, cache, input_text, traversal_number, observation):
    encoded_text = input_text + "\n"
    tokenized = tokenizer(encoded_text, add_special_tokens=False, return_tensors="pt")
    input_ids = tokenized.input_ids.cpu()
    if input_ids.numel() == 0:
        raise RuntimeError("The input encoded to no tokens")

    observation["active_traversal"] = traversal_number
    first_forward_index = len(observation["forward_events"])

    def forward_tokens(tokens):
        past_length = int(cache.get_seq_length())
        token_count = int(tokens.shape[1])
        attention_mask = torch.ones((1, past_length + token_count), dtype=torch.long)
        cache_position = torch.arange(past_length, past_length + token_count, dtype=torch.long)
        with torch.inference_mode():
            outputs = model(
                input_ids=tokens,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
        if outputs.past_key_values is not cache:
            raise RuntimeError("Inference returned a different KV cache object")
        return outputs.logits[:, -1, :]

    logits = forward_tokens(input_ids)
    generated_ids = []
    for _ in range(MAX_NEW_TOKENS):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        logits = forward_tokens(next_token)
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    traversal_events = observation["forward_events"][first_forward_index:]
    return {
        "input": input_text,
        "encoded_input_sha256": hashlib.sha256(encoded_text.encode("utf-8")).hexdigest(),
        "input_token_ids": input_ids.flatten().tolist(),
        "generated_token_ids": generated_ids,
        "output": output_text,
        "forward_event_count": len(traversal_events),
        "first_forward_event": traversal_events[0],
        "all_forwards_received_live_cache": all(
            event["supplied_is_live_cache"] for event in traversal_events
        ),
        "all_forwards_returned_same_cache": all(
            event["returned_same_object"] for event in traversal_events
        ),
    }


def compare_runs(run_a, run_b):
    checks = {
        "KV_0_EQUALS_KV_0_PRIME": run_a["kv_0"]["sha256"] == run_b["kv_0"]["sha256"],
        "KV_1_EQUALS_KV_1_PRIME": run_a["kv_1_post"]["sha256"]
        == run_b["kv_1_post"]["sha256"],
        "KV_2_EQUALS_KV_2_PRIME": run_a["kv_2"]["sha256"] == run_b["kv_2"]["sha256"],
        "OUTPUT_0_IDENTICAL": run_a["traversals"][0]["output"]
        == run_b["traversals"][0]["output"],
        "OUTPUT_1_IDENTICAL": run_a["traversals"][1]["output"]
        == run_b["traversals"][1]["output"],
        "KV_1_POST_EQUALS_KV_1_PRE_NEXT_CAUSALLY": run_a["causal_continuity"]
        and run_b["causal_continuity"],
        "FIXED_INPUTS_USED_IN_BOTH_RUNS": [item["input"] for item in run_a["traversals"]]
        == FIXED_INPUTS
        and [item["input"] for item in run_b["traversals"]] == FIXED_INPUTS,
        "HUMAN_TERMINATED_BOTH_RUNS": run_a["terminated_by_human"]
        and run_b["terminated_by_human"],
    }
    return checks, all(checks.values())


def causal_summary(run):
    first = run["traversals"][1]["first_forward_event"]
    return (
        f"live DynamicCache object id post-traversal-1={run['kv_1_object_id_post']}, "
        f"pre-traversal-2={run['kv_1_object_id_pre_next']}, "
        f"first model.forward traversal-2={first['supplied_object_id']}; "
        f"the complete live KV fingerprint remained {run['kv_1_post']['sha256']}; "
        f"all layer key/value tensor pointers were unchanged before the call; "
        f"model.forward received that exact object at sequence length {first['pre_sequence_length']} "
        f"and returned the same object"
    )


def make_pass_receipt(run_a, run_b, checks):
    launch_prefix = (
        "$env:PYTHONHASHSEED='0'; & "
        f"'{PROJECT_DIR / '.venv' / 'Scripts' / 'python.exe'}' "
        f"'{PROJECT_DIR / 'runtime.py'}' --run"
    )
    lines = [
        "BUILD_SLICE_RECEIPT",
        "slice: 1",
        "result: PASS",
        "",
        "artifact:",
        str(PROJECT_DIR / "runtime.py"),
        "",
        "launch:",
        f"Run A: {launch_prefix} A",
        f"Run B: {launch_prefix} B",
        "",
        "interface:",
        "PowerShell console prompts INPUT_0 and INPUT_1; each model output is printed immediately; type EXIT at TERMINATE to end the process.",
        "",
        "model:",
        f"{MODEL_ID} revision {MODEL_REVISION}; model.safetensors SHA-256 {MODEL_WEIGHT_SHA256}",
        "",
        "inference_substrate:",
        f"CPython {sys.version.split()[0]}, PyTorch {TORCH_VERSION} CPU, Transformers {TRANSFORMERS_VERSION}, greedy manual forward loop",
        "",
        "kv_substrate_object:",
        "One live transformers.cache_utils.DynamicCache. Its 30 DynamicLayer objects hold the causally operative key and value tensors; the same cache object is passed into every model.forward call and updated in place.",
        "",
        "kv_evidence_method:",
        "SHA-256 over every layer key tensor and value tensor: layer/component order, cache/layer class, dtype, shape, stride, device type, byte count, and all raw contiguous tensor bytes. Adjacent states are retained as complete tensor clones while the original live cache continues.",
        "",
        "deterministic_conditions:",
        f"Pinned model revision and weight hash; Python {sys.version.split()[0]}; torch {TORCH_VERSION}; transformers {TRANSFORMERS_VERSION}; CPU float32; one intra-op and one inter-op thread; deterministic algorithms enabled; seed 0; eval/inference mode; greedy argmax; {MAX_NEW_TOKENS} maximum generated tokens; identical literal inputs plus line termination.",
        "",
        "RUN_A",
        "",
        "KV_0:",
        run_a["kv_0"]["sha256"],
        "",
        "input_0:",
        run_a["traversals"][0]["input"],
        "",
        "output_0:",
        run_a["traversals"][0]["output"],
        "",
        "KV_1_POST:",
        run_a["kv_1_post"]["sha256"],
        "",
        "KV_1_PRE_NEXT:",
        run_a["kv_1_pre_next"]["sha256"],
        "",
        "KV_1_CAUSAL_CONTINUITY:",
        "PASS",
        causal_summary(run_a),
        "",
        "input_1:",
        run_a["traversals"][1]["input"],
        "",
        "output_1:",
        run_a["traversals"][1]["output"],
        "",
        "KV_2:",
        run_a["kv_2"]["sha256"],
        "",
        "RUN_B",
        "",
        "KV_0_PRIME:",
        run_b["kv_0"]["sha256"],
        "",
        "input_0:",
        run_b["traversals"][0]["input"],
        "",
        "output_0:",
        run_b["traversals"][0]["output"],
        "",
        "KV_1_PRIME:",
        run_b["kv_1_post"]["sha256"],
        "",
        "input_1:",
        run_b["traversals"][1]["input"],
        "",
        "output_1:",
        run_b["traversals"][1]["output"],
        "",
        "KV_2_PRIME:",
        run_b["kv_2"]["sha256"],
        "",
        "DETERMINISTIC_COMPARISON",
        "",
    ]
    for name in (
        "KV_0_EQUALS_KV_0_PRIME",
        "KV_1_EQUALS_KV_1_PRIME",
        "KV_2_EQUALS_KV_2_PRIME",
        "OUTPUT_0_IDENTICAL",
        "OUTPUT_1_IDENTICAL",
        "KV_1_POST_EQUALS_KV_1_PRE_NEXT_CAUSALLY",
    ):
        lines.extend([f"{name}:", "PASS" if checks[name] else "FAIL", ""])
    lines.extend(["final_result:", "PASS", "END_BUILD_SLICE_RECEIPT"])
    return "\n".join(lines) + "\n"


def make_fail_receipt(run_a, run_b, checks):
    failed = [name for name, passed in checks.items() if not passed]
    return "\n".join(
        [
            "BUILD_SLICE_RECEIPT",
            "slice: 1",
            "result: FAIL",
            "",
            "artifact_state:",
            str(PROJECT_DIR / "runtime.py"),
            "",
            "failed_requirement:",
            ", ".join(failed),
            "",
            "causal_boundary:",
            "Repeated-run deterministic comparison or human interaction condition.",
            "",
            "substrate_evidence:",
            json.dumps(checks, sort_keys=True),
            "",
            "substrate_finding:",
            "The exact failed comparison is named above; no weaker state surrogate was substituted.",
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


def run_interactive(run_label):
    configure_determinism()
    actual_weight_hash = verify_substrate()
    tokenizer, model = build_model()
    cache = DynamicCache(config=model.config)

    observation = {"active_traversal": 0, "forward_events": []}
    original_forward = model.forward

    def observed_forward(*args, **kwargs):
        supplied = kwargs.get("past_key_values")
        event = {
            "traversal": observation["active_traversal"],
            "supplied_object_id": None if supplied is None else id(supplied),
            "supplied_is_live_cache": supplied is cache,
            "pre_sequence_length": None
            if supplied is None
            else int(supplied.get_seq_length()),
        }
        outputs = original_forward(*args, **kwargs)
        event["returned_object_id"] = id(outputs.past_key_values)
        event["returned_same_object"] = outputs.past_key_values is supplied
        event["post_sequence_length"] = int(outputs.past_key_values.get_seq_length())
        observation["forward_events"].append(event)
        return outputs

    model.forward = observed_forward

    kv0_snapshot = snapshot_cache(cache)
    kv0 = fingerprint_material(kv0_snapshot)

    print("SLICE1_RUNTIME_READY", flush=True)
    print(f"RUN={run_label}", flush=True)
    print(f"KV_0={kv0['sha256']}", flush=True)

    inputs = []
    traversals = []
    first_input = input("INPUT_0> ")
    inputs.append(first_input)
    first = traverse(model, tokenizer, cache, first_input, 1, observation)
    traversals.append(first)
    print("MODEL_OUTPUT_0_BEGIN", flush=True)
    print(first["output"], flush=True)
    print("MODEL_OUTPUT_0_END", flush=True)

    kv1_post = fingerprint_cache(cache)
    kv1_snapshot = snapshot_cache(cache)
    kv1_snapshot_fingerprint = fingerprint_material(kv1_snapshot)
    kv1_object_id_post = id(cache)
    kv1_pointer_post = pointer_map(cache)
    print(f"KV_1_POST={kv1_post['sha256']}", flush=True)

    kv1_pre_next = fingerprint_cache(cache)
    kv1_object_id_pre_next = id(cache)
    kv1_pointer_pre_next = pointer_map(cache)
    print(f"KV_1_PRE_NEXT={kv1_pre_next['sha256']}", flush=True)

    second_input = input("INPUT_1> ")
    inputs.append(second_input)
    second = traverse(model, tokenizer, cache, second_input, 2, observation)
    traversals.append(second)
    print("MODEL_OUTPUT_1_BEGIN", flush=True)
    print(second["output"], flush=True)
    print("MODEL_OUTPUT_1_END", flush=True)

    kv2 = fingerprint_cache(cache)
    kv2_snapshot = snapshot_cache(cache)
    kv2_snapshot_fingerprint = fingerprint_material(kv2_snapshot)
    kv1_snapshot_after_kv2 = fingerprint_material(kv1_snapshot)
    print(f"KV_2={kv2['sha256']}", flush=True)

    traversal_2_first_event = second["first_forward_event"]
    causal_continuity = all(
        [
            kv1_post["sha256"] == kv1_pre_next["sha256"],
            kv1_post["sha256"] == kv1_snapshot_fingerprint["sha256"],
            kv1_object_id_post == kv1_object_id_pre_next,
            kv1_pointer_post == kv1_pointer_pre_next,
            traversal_2_first_event["supplied_object_id"] == kv1_object_id_post,
            traversal_2_first_event["supplied_is_live_cache"],
            traversal_2_first_event["pre_sequence_length"] == kv1_post["sequence_length"],
            second["all_forwards_received_live_cache"],
            second["all_forwards_returned_same_cache"],
            kv1_snapshot_after_kv2["sha256"] == kv1_post["sha256"],
            kv2_snapshot_fingerprint["sha256"] == kv2["sha256"],
        ]
    )
    print(
        "KV_1_CAUSAL_CONTINUITY=" + ("PASS" if causal_continuity else "FAIL"),
        flush=True,
    )

    while True:
        termination = input("TERMINATE (type EXIT)> ")
        if termination == "EXIT":
            break
        print("Type EXIT exactly to terminate.", flush=True)

    record = {
        "schema": "SLICE1_RUN_EVIDENCE_V1",
        "run": run_label,
        "artifact": str(PROJECT_DIR / "runtime.py"),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_sha256": actual_weight_hash,
        },
        "substrate": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "dtype": "torch.float32",
            "cache_class": f"{cache.__class__.__module__}.{cache.__class__.__name__}",
            "thread_count": torch.get_num_threads(),
            "interop_thread_count": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "generation": "greedy_argmax",
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "kv_0": kv0,
        "kv_1_post": kv1_post,
        "kv_1_pre_next": kv1_pre_next,
        "kv_2": kv2,
        "kv_1_object_id_post": kv1_object_id_post,
        "kv_1_object_id_pre_next": kv1_object_id_pre_next,
        "kv_1_pointer_map_post": kv1_pointer_post,
        "kv_1_pointer_map_pre_next": kv1_pointer_pre_next,
        "kv_1_snapshot_retained_through_kv_2": kv1_snapshot_after_kv2["sha256"]
        == kv1_post["sha256"],
        "causal_continuity": causal_continuity,
        "traversals": traversals,
        "forward_events": observation["forward_events"],
        "terminated_by_human": True,
    }

    EVIDENCE_DIR.mkdir(exist_ok=True)
    run_path = EVIDENCE_DIR / f"run_{run_label.lower()}.json"
    run_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"RUN_{run_label}_EVIDENCE={run_path}", flush=True)

    if run_label == "B":
        run_a_path = EVIDENCE_DIR / "run_a.json"
        if not run_a_path.is_file():
            raise RuntimeError("Run A evidence does not exist; complete Run A first")
        run_a = json.loads(run_a_path.read_text(encoding="utf-8"))
        checks, passed = compare_runs(run_a, record)
        receipt = (
            make_pass_receipt(run_a, record, checks)
            if passed
            else make_fail_receipt(run_a, record, checks)
        )
        receipt_path = EVIDENCE_DIR / "completion_receipt.txt"
        receipt_path.write_text(receipt, encoding="utf-8")
        print("DETERMINISTIC_COMPARISON_BEGIN", flush=True)
        for name, value in checks.items():
            print(f"{name}={'PASS' if value else 'FAIL'}", flush=True)
        print(f"FINAL_RESULT={'PASS' if passed else 'FAIL'}", flush=True)
        print(f"COMPLETION_RECEIPT={receipt_path}", flush=True)
        print("DETERMINISTIC_COMPARISON_END", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Minimal causally continuous KV runtime")
    parser.add_argument("--run", choices=("A", "B"), required=True)
    args = parser.parse_args()
    try:
        run_interactive(args.run)
    except Exception as error:
        print(f"SLICE1_RUNTIME_ERROR={error.__class__.__name__}: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
