import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "kv_causal_intervention_lab"
SNAPSHOT_PATH = EVIDENCE_DIR / "K_t_snapshot.pt"
EVIDENCE_PATH = EVIDENCE_DIR / "kv_causal_intervention_lab.json"
RECEIPT_PATH = EVIDENCE_DIR / "completion_receipt.txt"
REPLICATION_COUNT = 2
FP_KEYS = (
    "sha256",
    "layer_count",
    "component_count",
    "tensor_byte_count",
    "sequence_length",
)
INTERVENTION_LAYER = 0
INTERVENTION_COMPONENT = "key"
INTERVENTION_INDEX = (0, 0, 0, 0)


def load_slice14_runtime():
    path = PROJECT_DIR / "slice14_runtime.py"
    spec = importlib.util.spec_from_file_location("kv_lab_inherited_slice14_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inherited Slice 14 runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(kernel, cache):
    raw = kernel.fingerprint_cache(cache)
    return {key: raw[key] for key in FP_KEYS}


def same_fingerprint(left, right):
    return all(left[key] == right[key] for key in FP_KEYS)


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(-1).view(sys.modules["torch"].uint8).numpy().tobytes()


def snapshot_content_equal(left, right):
    if left["cache_class"] != right["cache_class"] or len(left["layers"]) != len(right["layers"]):
        return False
    for left_layer, right_layer in zip(left["layers"], right["layers"]):
        if left_layer["layer_class"] != right_layer["layer_class"]:
            return False
        for component in ("key", "value"):
            a, b = left_layer[component], right_layer[component]
            if (a is None) != (b is None):
                return False
            if a is not None and (
                a.dtype != b.dtype
                or tuple(a.shape) != tuple(b.shape)
                or tuple(a.stride()) != tuple(b.stride())
                or tensor_bytes(a) != tensor_bytes(b)
            ):
                return False
    return True


def restore_dynamic_cache(kernel, model, snapshot):
    torch = sys.modules["torch"]
    restored = kernel.DynamicCache(config=model.config)
    actual_cache_class = f"{restored.__class__.__module__}.{restored.__class__.__name__}"
    if actual_cache_class != snapshot["cache_class"]:
        raise RuntimeError(f"cache class mismatch: {actual_cache_class} != {snapshot['cache_class']}")
    if len(restored.layers) != len(snapshot["layers"]):
        raise RuntimeError("restored cache layer count differs from snapshot")
    for layer_number, (native_layer, saved_layer) in enumerate(zip(restored.layers, snapshot["layers"])):
        actual_layer_class = f"{native_layer.__class__.__module__}.{native_layer.__class__.__name__}"
        if actual_layer_class != saved_layer["layer_class"]:
            raise RuntimeError(f"layer {layer_number} class mismatch")
        key, value = saved_layer["key"], saved_layer["value"]
        if (key is None) != (value is None):
            raise RuntimeError(f"layer {layer_number} has incomplete KV material")
        if key is None:
            continue
        if key.device.type != "cpu" or value.device.type != "cpu":
            raise RuntimeError("persisted snapshot is not on the pinned CPU device")
        native_layer.keys = key.detach().clone(memory_format=torch.preserve_format)
        native_layer.values = value.detach().clone(memory_format=torch.preserve_format)
        native_layer.dtype = native_layer.keys.dtype
        native_layer.device = native_layer.keys.device
        native_layer.is_initialized = True
    restored_snapshot = kernel.snapshot_cache(restored)
    if not snapshot_content_equal(snapshot, restored_snapshot):
        raise RuntimeError("native restored cache content differs from persisted snapshot")
    return restored


def intervention_value(tensor, coordinate):
    torch = sys.modules["torch"]
    before_tensor = tensor[coordinate].detach().cpu().clone()
    zero = torch.zeros((), dtype=before_tensor.dtype)
    after_tensor = torch.ones((), dtype=before_tensor.dtype) if torch.equal(before_tensor, zero) else zero
    return before_tensor, after_tensor


def exact_intervention_support(kernel, baseline_cache, intervened_cache, before_tensor, after_tensor):
    baseline = kernel.snapshot_cache(baseline_cache)
    intervened = kernel.snapshot_cache(intervened_cache)
    changed_components = []
    non_support_changed_byte_count = 0
    support_changed_byte_offsets = []
    support_element_size = before_tensor.element_size()

    for layer_number, (left_layer, right_layer) in enumerate(zip(baseline["layers"], intervened["layers"])):
        for component in ("key", "value"):
            left, right = left_layer[component], right_layer[component]
            left_raw, right_raw = tensor_bytes(left), tensor_bytes(right)
            differing = [index for index, pair in enumerate(zip(left_raw, right_raw)) if pair[0] != pair[1]]
            if differing:
                changed_components.append({"layer": layer_number, "component": component, "changed_byte_offsets": differing})
            if layer_number == INTERVENTION_LAYER and component == INTERVENTION_COMPONENT:
                support_start = 0
                support_end = support_element_size
                support_changed_byte_offsets = [index for index in differing if support_start <= index < support_end]
                non_support_changed_byte_count += sum(index < support_start or index >= support_end for index in differing)
            else:
                non_support_changed_byte_count += len(differing)

    exact = (
        len(changed_components) == 1
        and changed_components[0]["layer"] == INTERVENTION_LAYER
        and changed_components[0]["component"] == INTERVENTION_COMPONENT
        and bool(support_changed_byte_offsets)
        and non_support_changed_byte_count == 0
        and tensor_bytes(before_tensor) != tensor_bytes(after_tensor)
    )
    return {
        "exact": exact,
        "changed_components": changed_components,
        "declared_support_byte_range_in_contiguous_tensor": [0, support_element_size],
        "support_changed_byte_offsets": support_changed_byte_offsets,
        "non_support_changed_byte_count": non_support_changed_byte_count,
        "before_bytes_hex": tensor_bytes(before_tensor).hex(),
        "after_bytes_hex": tensor_bytes(after_tensor).hex(),
    }


def apply_intervention(kernel, baseline_cache, intervened_cache):
    target = getattr(intervened_cache.layers[INTERVENTION_LAYER], "keys")
    before_tensor, after_tensor = intervention_value(target, INTERVENTION_INDEX)
    target[INTERVENTION_INDEX] = after_tensor.to(device=target.device)
    observed_after = target[INTERVENTION_INDEX].detach().cpu().clone()
    if tensor_bytes(observed_after) != tensor_bytes(after_tensor):
        raise RuntimeError("declared scalar replacement did not survive assignment")
    support = exact_intervention_support(kernel, baseline_cache, intervened_cache, before_tensor, observed_after)
    if not support["exact"]:
        raise RuntimeError("intervention changed content outside its declared scalar support")
    return {
        "layer": INTERVENTION_LAYER,
        "component": INTERVENTION_COMPONENT,
        "index": list(INTERVENTION_INDEX),
        "operation": "replace with float32 0.0 unless already exactly 0.0, then replace with float32 1.0",
        "before": float(before_tensor.item()),
        "after": float(observed_after.item()),
        "before_bytes_hex": tensor_bytes(before_tensor).hex(),
        "after_bytes_hex": tensor_bytes(observed_after).hex(),
        "support_evidence": support,
    }


def traverse_observed(slice11, context, cache, shared_v, delta, label):
    model = context["model"]
    kernel = context["kernel"]
    inherited_forward = model.forward
    first_forward_post = None

    def boundary_observer(*args, **kwargs):
        nonlocal first_forward_post
        outputs = inherited_forward(*args, **kwargs)
        if first_forward_post is None:
            first_forward_post = fingerprint(kernel, outputs.past_key_values)
        return outputs

    model.forward = boundary_observer
    try:
        traversal = slice11.traverse(
            context,
            cache,
            shared_v,
            delta,
            label,
            "exact selected Slice 14 source-boundary perturbation",
        )
    finally:
        model.forward = inherited_forward
    if first_forward_post is None:
        raise RuntimeError("no model.forward boundary was observed")
    return traversal, first_forward_post


def token_evidence(context, token_ids):
    ids = [int(value) for value in token_ids]
    return {
        "token_ids": ids,
        "count": len(ids),
        "sha256": context["slice4"].token_fingerprint(ids),
    }


def physics_evidence(slice11, context, shared_v):
    torch = sys.modules["torch"]
    kernel = context["kernel"]
    return {
        "model_object_id": id(context["model"]),
        "model_weight_sha256": context["weight_hash"],
        "model_revision": kernel.MODEL_REVISION,
        "tokenizer_object_id": id(context["tokenizer"]),
        "shared_v_object_id": id(shared_v),
        "shared_v_literal": shared_v.literal,
        "shared_v_token_ids": list(shared_v.token_ids),
        "traversal_callable_module": slice11.traverse.__module__,
        "traversal_callable_qualname": slice11.traverse.__qualname__,
        "generation_max_new_tokens": kernel.MAX_NEW_TOKENS,
        "generation_rule": "greedy torch.argmax in inherited traversal",
        "dtype": "torch.float32",
        "device": "cpu",
        "torch_version": torch.__version__,
        "transformers_version": sys.modules["transformers"].__version__,
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cache_class": "transformers.cache_utils.DynamicCache",
    }


def execute_replicate(number, slice11, context, persisted_path, original_pre, original_post, original_output, shared_v, delta):
    torch = sys.modules["torch"]
    kernel = context["kernel"]
    loaded_baseline = torch.load(persisted_path, map_location="cpu", weights_only=False)
    baseline_cache = restore_dynamic_cache(kernel, context["model"], loaded_baseline)
    restored_fp = fingerprint(kernel, baseline_cache)
    baseline_pre_snapshot = kernel.snapshot_cache(baseline_cache)

    loaded_intervention = torch.load(persisted_path, map_location="cpu", weights_only=False)
    intervention_cache = restore_dynamic_cache(kernel, context["model"], loaded_intervention)
    intervention_initial_fp = fingerprint(kernel, intervention_cache)
    intervention = apply_intervention(kernel, baseline_cache, intervention_cache)
    intervention_pre_fp = fingerprint(kernel, intervention_cache)

    baseline_traversal, baseline_first_post = traverse_observed(
        slice11, context, baseline_cache, shared_v, delta, f"KV_CAUSAL_LAB_R{number}_BASELINE"
    )
    baseline_post_fp = fingerprint(kernel, baseline_cache)
    baseline_post_snapshot = kernel.snapshot_cache(baseline_cache)

    intervention_traversal, intervention_first_post = traverse_observed(
        slice11, context, intervention_cache, shared_v, delta, f"KV_CAUSAL_LAB_R{number}_INTERVENTION"
    )
    intervention_post_fp = fingerprint(kernel, intervention_cache)

    baseline_supplied = token_evidence(context, baseline_traversal["perturbation_token_ids_supplied"])
    baseline_observed = token_evidence(context, baseline_traversal["actual_first_forward_perturbation_ids"])
    intervention_supplied = token_evidence(context, intervention_traversal["perturbation_token_ids_supplied"])
    intervention_observed = token_evidence(context, intervention_traversal["actual_first_forward_perturbation_ids"])
    baseline_output = token_evidence(context, baseline_traversal["generated_token_ids"])
    intervention_output = token_evidence(context, intervention_traversal["generated_token_ids"])

    if not same_fingerprint(restored_fp, original_pre):
        raise RuntimeError("native restore does not equal original K_t")
    if not snapshot_content_equal(loaded_baseline, baseline_pre_snapshot):
        raise RuntimeError("native restore is not content-identical to persisted K_t")
    if not same_fingerprint(intervention_initial_fp, restored_fp):
        raise RuntimeError("intervention condition did not start from the same restored K_t")
    if not same_fingerprint(baseline_post_fp, original_post):
        raise RuntimeError("baseline replay did not reproduce original K_t+1")
    if not snapshot_content_equal(baseline_post_snapshot, original_post["snapshot"]):
        raise RuntimeError("baseline replay content did not reproduce original K_t+1")
    if baseline_output != original_output:
        raise RuntimeError("baseline generated output differs from source transition")
    if not (baseline_supplied == baseline_observed == intervention_supplied == intervention_observed):
        raise RuntimeError("baseline/intervention delta identity failed")

    causal_effect = (
        not same_fingerprint(intervention_post_fp, original_post)
        or intervention_output != original_output
    )
    if not same_fingerprint(baseline_first_post, intervention_first_post):
        earliest = {
            "boundary": "first model.forward return",
            "baseline_complete_kv": baseline_first_post,
            "intervention_complete_kv": intervention_first_post,
        }
    elif baseline_output != intervention_output:
        first_difference = next(
            (index for index, pair in enumerate(zip(baseline_output["token_ids"], intervention_output["token_ids"])) if pair[0] != pair[1]),
            min(baseline_output["count"], intervention_output["count"]),
        )
        earliest = {"boundary": "generated-token trajectory", "first_differing_token_index": first_difference}
    elif not same_fingerprint(baseline_post_fp, intervention_post_fp):
        earliest = {"boundary": "final complete KV state", "baseline": baseline_post_fp, "intervention": intervention_post_fp}
    else:
        earliest = "NONE"

    return {
        "replicate": number,
        "restored_K_t": restored_fp,
        "intervention_initial_K_t": intervention_initial_fp,
        "intervention_spec": intervention,
        "intervened_pre_state": intervention_pre_fp,
        "baseline_delta_supplied": baseline_supplied,
        "baseline_delta_observed": baseline_observed,
        "intervention_delta_supplied": intervention_supplied,
        "intervention_delta_observed": intervention_observed,
        "baseline_first_forward_post": baseline_first_post,
        "intervention_first_forward_post": intervention_first_post,
        "baseline_post_state": baseline_post_fp,
        "intervention_post_state": intervention_post_fp,
        "original_output": original_output,
        "baseline_output": baseline_output,
        "intervention_output": intervention_output,
        "causal_result": "CAUSAL_EFFECT_OBSERVED" if causal_effect else "NO_CAUSAL_EFFECT_OBSERVED",
        "earliest_observed_downstream_divergence": earliest,
        "checks": {
            "RESTORE_EXACT": True,
            "BASELINE_REPLAY_DELTA_EXACT": True,
            "BASELINE_REPLAY_EQUALS_ORIGINAL": True,
            "BASELINE_OUTPUT_EQUALS_ORIGINAL": True,
            "INTERVENTION_SUPPORT_EXACTLY_BOUNDED": intervention["support_evidence"]["exact"],
            "NON_INTERVENED_KV_CONTENT_UNCHANGED": intervention["support_evidence"]["non_support_changed_byte_count"] == 0,
            "INTERVENTION_DELTA_EXACTLY_EQUALS_BASELINE_DELTA": True,
        },
    }


def replication_projection(rep):
    return {key: value for key, value in rep.items() if key != "replicate"}


def make_receipt(result):
    rep = result["replicates"][0]
    spec = rep["intervention_spec"]
    fields = [
        ("SOURCE_BOUNDARY", json.dumps(result["source_boundary"], sort_keys=True)),
        ("K_t_ORIGINAL", json.dumps(result["K_t_original"], sort_keys=True)),
        ("DELTA_t", json.dumps(result["delta_t"], sort_keys=True)),
        ("K_t_PLUS_1_ORIGINAL", json.dumps(result["K_t_plus_1_original"], sort_keys=True)),
        ("PERSISTED_K_t", json.dumps(result["persisted_K_t"], sort_keys=True)),
        ("RESTORED_K_t", json.dumps(rep["restored_K_t"], sort_keys=True)),
        ("RESTORE_EXACT", "PASS"),
        ("BASELINE_REPLAY_DELTA_EXACT", "PASS"),
        ("BASELINE_REPLAY_K_t_PLUS_1", json.dumps(rep["baseline_post_state"], sort_keys=True)),
        ("BASELINE_REPLAY_EQUALS_ORIGINAL", "PASS"),
        ("BASELINE_OUTPUT_EQUALS_ORIGINAL", "PASS"),
        ("INTERVENTION_SPEC", "\n".join([
            f"layer={spec['layer']}", f"component={spec['component']}",
            f"index={json.dumps(spec['index'])}", f"operation={spec['operation']}",
            f"before={spec['before']} bytes={spec['before_bytes_hex']}",
            f"after={spec['after']} bytes={spec['after_bytes_hex']}",
        ])),
        ("INTERVENTION_SUPPORT_EXACTLY_BOUNDED", "PASS"),
        ("NON_INTERVENED_KV_CONTENT_UNCHANGED", "PASS"),
        ("INTERVENED_PRE_STATE", json.dumps(rep["intervened_pre_state"], sort_keys=True)),
        ("INTERVENTION_DELTA_EXACTLY_EQUALS_BASELINE_DELTA", "PASS"),
        ("INTERVENTION_POST_STATE", json.dumps(rep["intervention_post_state"], sort_keys=True)),
        ("CAUSAL_RESULT", rep["causal_result"]),
        ("EARLIEST_OBSERVED_DOWNSTREAM_DIVERGENCE", json.dumps(rep["earliest_observed_downstream_divergence"], sort_keys=True)),
        ("ORIGINAL_SOURCE_BOUNDARY_UNCHANGED", "PASS"),
        ("RUNTIME_PHYSICS_OTHER_THAN_I_K_t_IDENTICAL", "PASS"),
        ("DETERMINISTIC_REPLICATION", "PASS"),
    ]
    lines = ["OUTCOME", "KV_CAUSAL_INTERVENTION_LAB_OPERATIONAL", "", "EVIDENCE", ""]
    for name, value in fields:
        lines.extend([f"{name}:", value, ""])
    return "\n".join(lines)


def run():
    import torch
    import transformers
    slice14 = load_slice14_runtime()

    slice13, slice11, context = slice14.initialize_accumulated_runtime()
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(context, parent, slice14.LOCAL_CYCLE_BOUND, evaluate_gate=True)
    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise RuntimeError("selected Slice 14 parent did not produce its required source boundary")
    boundary = parent.latest_boundary
    kernel = context["kernel"]
    original_pre_snapshot = kernel.snapshot_cache(boundary["pre_native"])
    original_post_snapshot = kernel.snapshot_cache(boundary["post_native"])
    original_pre = fingerprint(kernel, boundary["pre_native"])
    original_post_fp = fingerprint(kernel, boundary["post_native"])
    original_post = dict(original_post_fp)
    original_post["snapshot"] = original_post_snapshot
    original_output = token_evidence(context, boundary["traversal"]["generated_token_ids"])
    delta = tuple(int(value) for value in boundary["delta"])

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(original_pre_snapshot, SNAPSHOT_PATH)
    loaded = torch.load(SNAPSHOT_PATH, map_location="cpu", weights_only=False)
    if not snapshot_content_equal(original_pre_snapshot, loaded):
        raise RuntimeError("persisted K_t did not round-trip exactly")
    persisted_material_fp = kernel.fingerprint_material(loaded)
    if persisted_material_fp["sha256"] != original_pre["sha256"]:
        raise RuntimeError("persisted K_t fingerprint differs from original K_t")

    original_pre_guard = kernel.fingerprint_material(original_pre_snapshot)
    original_post_guard = kernel.fingerprint_material(original_post_snapshot)
    physics = physics_evidence(slice11, context, parent.shared_v)
    replicates = [
        execute_replicate(
            number, slice11, context, SNAPSHOT_PATH, original_pre, original_post,
            original_output, parent.shared_v, delta,
        )
        for number in range(1, REPLICATION_COUNT + 1)
    ]
    deterministic = replication_projection(replicates[0]) == replication_projection(replicates[1])
    source_unchanged = (
        same_fingerprint(fingerprint(kernel, boundary["pre_native"]), original_pre)
        and same_fingerprint(fingerprint(kernel, boundary["post_native"]), original_post_fp)
        and kernel.fingerprint_material(original_pre_snapshot) == original_pre_guard
        and kernel.fingerprint_material(original_post_snapshot) == original_post_guard
    )
    physics_valid = (
        physics["model_weight_sha256"] == kernel.MODEL_WEIGHT_SHA256
        and physics["torch_version"] == kernel.TORCH_VERSION
        and physics["transformers_version"] == kernel.TRANSFORMERS_VERSION
        and physics["intraop_threads"] == 1
        and physics["interop_threads"] == 1
        and physics["deterministic_algorithms"]
    )
    if not deterministic:
        raise RuntimeError("complete experiment did not replicate deterministically")
    if not source_unchanged:
        raise RuntimeError("original source boundary changed during disposable replay")
    if not physics_valid:
        raise RuntimeError("inherited deterministic runtime physics changed")

    result = {
        "schema": "KV_CAUSAL_INTERVENTION_LAB_V1",
        "outcome": "KV_CAUSAL_INTERVENTION_LAB_OPERATIONAL",
        "source_boundary": {
            "fixture_status": "experimental fixture, not runtime law",
            "producer_instance_id": boundary["producer_instance_id"],
            "cycle": boundary["cycle"],
            "ordinal": boundary["ordinal"],
            "pole": boundary["pole"],
        },
        "K_t_original": original_pre,
        "delta_t": token_evidence(context, delta),
        "K_t_plus_1_original": original_post_fp,
        "persisted_K_t": {
            "path": str(SNAPSHOT_PATH),
            "file_sha256": sha256_file(SNAPSHOT_PATH),
            "content_fingerprint": {key: persisted_material_fp[key] for key in FP_KEYS},
            "round_trip_content_exact": True,
        },
        "physics": physics,
        "replicates": replicates,
        "checks": {
            "ORIGINAL_SOURCE_BOUNDARY_UNCHANGED": source_unchanged,
            "RUNTIME_PHYSICS_OTHER_THAN_I_K_t_IDENTICAL": physics_valid,
            "DETERMINISTIC_REPLICATION": deterministic,
        },
    }
    EVIDENCE_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    receipt = make_receipt(result)
    RECEIPT_PATH.write_text(receipt, encoding="utf-8")
    print(receipt, flush=True)
    print(f"EVIDENCE_JSON={EVIDENCE_PATH}", flush=True)
    print(f"COMPLETION_RECEIPT={RECEIPT_PATH}", flush=True)
    gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as error:
        print(f"KV_CAUSAL_INTERVENTION_LAB_ERROR={error.__class__.__name__}: {error}", file=sys.stderr, flush=True)
        raise
