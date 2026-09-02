from __future__ import annotations

import gc
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import transformers
import kv_causal_intervention_lab as lab
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import transformers.integrations.sdpa_attention as sdpa_mod

PARENT_COMMIT = "0115e21b280141967b98bab4ec5b64d81c528ac4"
OUT_DIR = ROOT / "evidence" / "realized_attention_state"
RAW_A_DIR = OUT_DIR / "raw_a"
RAW_B_DIR = OUT_DIR / "raw_b"
ARTIFACT_DIR = OUT_DIR / "artifacts"
RESULT_PATH = OUT_DIR / "realized_attention_state_exposure.json"
RECEIPT_PATH = OUT_DIR / "completion_receipt.txt"
MANIFEST_PATH = OUT_DIR / "attention_manifest.json"
DESCRIPTOR_PATH = OUT_DIR / "descriptor_rows.json"

RECONSTRUCTED_OUTPUT_ABS_TOLERANCE = 1.0e-5
ARTIFACT_FILE_SIZE_GUARD_BYTES = 90_000_000
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
TENSOR_FIELDS = (
    "query",
    "key",
    "value",
    "attention_mask",
    "native_output",
    "logits",
    "attention",
)


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(tensor):
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def update_tensor_hash(digest, name, tensor):
    if tensor is None:
        digest.update(
            canonical_json({"name": name, "none": True}).encode("ascii")
        )
        return
    metadata = {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
    }
    digest.update(canonical_json(metadata).encode("ascii"))
    digest.update(tensor_bytes(tensor))


def semantic_event_hash(event, include_reconstruction):
    digest = hashlib.sha256()
    digest.update(canonical_json(event["metadata"]).encode("ascii"))
    for layer in event["layers"]:
        metadata = {
            key: value
            for key, value in layer.items()
            if key not in TENSOR_FIELDS
        }
        digest.update(canonical_json(metadata).encode("ascii"))
        names = [
            "query",
            "key",
            "value",
            "attention_mask",
            "native_output",
        ]
        if include_reconstruction:
            names.extend(["logits", "attention"])
        for name in names:
            update_tensor_hash(digest, name, layer.get(name))
    return digest.hexdigest()


def fingerprint(kernel, cache):
    return lab.fingerprint(kernel, cache)


def token_evidence(context, token_ids):
    return lab.token_evidence(context, token_ids)


def source_hashes(paths):
    return {
        str(path): sha256_file(path)
        for path in paths
    }


class NativeAttentionCapture:
    def __init__(self, model, raw_directory):
        self.model = model
        self.raw_directory = Path(raw_directory)
        self.raw_directory.mkdir(parents=True, exist_ok=True)
        attention = model.model.layers[0].self_attn
        self.registry = attention.forward.__globals__["ALL_ATTENTION_FUNCTIONS"]
        self.original_callable = self.registry.get_interface("sdpa", None)
        self.previous_local = self.registry._local_mapping.get("sdpa")
        self.layers = []
        self.events = []
        self.forward_event_index = 0

    def flush_event(self):
        if not self.layers:
            return
        expected_layers = list(range(len(self.model.model.layers)))
        actual_layers = [item["layer_index"] for item in self.layers]
        if actual_layers != expected_layers:
            raise RuntimeError(
                f"native attention layer order mismatch: {actual_layers}"
            )
        event = {
            "metadata": {
                "forward_event_index": self.forward_event_index,
                "layer_count": len(self.layers),
                "attention_backend": "sdpa",
                "native_callable": (
                    self.original_callable.__module__
                    + "."
                    + self.original_callable.__name__
                ),
            },
            "layers": self.layers,
        }
        raw_hash = semantic_event_hash(event, include_reconstruction=False)
        event["metadata"]["raw_semantic_sha256"] = raw_hash
        path = self.raw_directory / (
            f"event_{self.forward_event_index:03d}.pt"
        )
        torch.save(event, path)
        self.events.append(
            {
                "forward_event_index": self.forward_event_index,
                "path": str(path),
                "raw_semantic_sha256": raw_hash,
                "file_sha256": sha256_file(path),
                "file_bytes": path.stat().st_size,
            }
        )
        self.forward_event_index += 1
        self.layers = []

    def wrapper(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        is_causal=None,
        position_bias=None,
        **kwargs,
    ):
        layer_index = int(module.layer_idx)
        if layer_index == 0 and self.layers:
            self.flush_event()

        original_result = self.original_callable(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            position_bias=position_bias,
            **kwargs,
        )

        query_length = int(query.shape[2])
        key_length = int(key.shape[2])
        resolved_is_causal = (
            is_causal
            if is_causal is not None
            else getattr(module, "is_causal", True)
        )
        resolved_is_causal = bool(
            query_length > 1
            and attention_mask is None
            and resolved_is_causal
        )
        native_crop = bool(
            resolved_is_causal
            and attention_mask is None
            and query_length > 1
            and key_length > query_length
        )
        effective_key_length = (
            query_length if native_crop else key_length
        )
        use_native_gqa = bool(
            module.num_key_value_groups > 1
            and sdpa_mod.use_gqa_in_sdpa(attention_mask, key, value)
        )

        cache_position = kwargs.get("cache_position")
        if (
            isinstance(cache_position, torch.Tensor)
            and cache_position.numel() == query_length
        ):
            query_positions = [
                int(item)
                for item in cache_position.detach().cpu().reshape(-1).tolist()
            ]
            position_basis = "captured cache_position"
        else:
            query_positions = list(
                range(key_length - query_length, key_length)
            )
            position_basis = (
                "DynamicCache contiguous positions derived from "
                "post-update key length"
            )

        if attention_mask is not None:
            mask_recipe = {
                "kind": "materialized_operand",
                "dtype": str(attention_mask.dtype),
                "shape": list(attention_mask.shape),
            }
        elif resolved_is_causal:
            mask_recipe = {"kind": "implicit_upper_left_causal"}
        else:
            mask_recipe = {
                "kind": "all_key_positions_admissible"
            }

        self.layers.append(
            {
                "forward_event_index": self.forward_event_index,
                "layer_index": layer_index,
                "attention_backend": "sdpa",
                "query_shape": list(query.shape),
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "query_dtype": str(query.dtype),
                "key_dtype": str(key.dtype),
                "value_dtype": str(value.dtype),
                "query_device": str(query.device),
                "key_device": str(key.device),
                "value_device": str(value.device),
                "scaling_factor": float(scaling),
                "query_head_count": int(query.shape[1]),
                "key_value_head_count": int(key.shape[1]),
                "head_dimension": int(query.shape[-1]),
                "group_count": int(module.num_key_value_groups),
                "query_to_kv_head": [
                    index // int(module.num_key_value_groups)
                    for index in range(int(query.shape[1]))
                ],
                "gqa_native_mode": (
                    "enable_gqa_compact_kv"
                    if use_native_gqa
                    else "repeat_kv_inside_native_callable"
                ),
                "native_is_causal": resolved_is_causal,
                "native_kv_crop_to_query_length": native_crop,
                "effective_key_length": effective_key_length,
                "mask_recipe": mask_recipe,
                "query_positions": query_positions,
                "key_positions": list(range(effective_key_length)),
                "position_basis": position_basis,
                "native_kwargs_keys": sorted(kwargs.keys()),
                "position_bias_present": position_bias is not None,
                "query": query.detach().cpu().clone(
                    memory_format=torch.preserve_format
                ),
                "key": key.detach().cpu().clone(
                    memory_format=torch.preserve_format
                ),
                "value": value.detach().cpu().clone(
                    memory_format=torch.preserve_format
                ),
                "attention_mask": (
                    None
                    if attention_mask is None
                    else attention_mask.detach().cpu().clone(
                        memory_format=torch.preserve_format
                    )
                ),
                "native_output": original_result[0]
                .detach()
                .cpu()
                .clone(memory_format=torch.preserve_format),
            }
        )
        return original_result

    def __enter__(self):
        self.registry._local_mapping["sdpa"] = self.wrapper
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            if exception_type is None:
                self.flush_event()
        finally:
            if self.previous_local is None:
                self.registry._local_mapping.pop("sdpa", None)
            else:
                self.registry._local_mapping["sdpa"] = self.previous_local


def execute_transition(
    slice11,
    context,
    shared_v,
    delta,
    label,
    raw_directory=None,
):
    kernel = context["kernel"]
    persisted = torch.load(
        lab.SNAPSHOT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    cache = lab.restore_dynamic_cache(
        kernel,
        context["model"],
        persisted,
    )
    start_snapshot = kernel.snapshot_cache(cache)
    start_fingerprint = fingerprint(kernel, cache)

    if raw_directory is None:
        traversal = slice11.traverse(
            context,
            cache,
            shared_v,
            delta,
            label,
            "exact selected Slice 14 source-boundary perturbation",
        )
        capture_manifest = None
    else:
        with NativeAttentionCapture(
            context["model"],
            raw_directory,
        ) as capture:
            traversal = slice11.traverse(
                context,
                cache,
                shared_v,
                delta,
                label,
                "exact selected Slice 14 source-boundary perturbation",
            )
        capture_manifest = capture.events

    post_snapshot = kernel.snapshot_cache(cache)
    return {
        "start_fingerprint": start_fingerprint,
        "start_snapshot": start_snapshot,
        "delta": token_evidence(
            context,
            traversal["actual_first_forward_perturbation_ids"],
        ),
        "generated": token_evidence(
            context,
            traversal["generated_token_ids"],
        ),
        "post_fingerprint": fingerprint(kernel, cache),
        "post_snapshot": post_snapshot,
        "capture_manifest": capture_manifest,
    }


def transitions_identical(left, right):
    return (
        left["start_fingerprint"] == right["start_fingerprint"]
        and lab.snapshot_content_equal(
            left["start_snapshot"],
            right["start_snapshot"],
        )
        and left["delta"] == right["delta"]
        and left["generated"] == right["generated"]
        and left["post_fingerprint"] == right["post_fingerprint"]
        and lab.snapshot_content_equal(
            left["post_snapshot"],
            right["post_snapshot"],
        )
    )



def reconstruct_layer(layer):
    query = layer["query"].float()
    key = layer["key"].float()
    value = layer["value"].float()

    if layer["position_bias_present"]:
        raise RuntimeError(
            "unexpected position bias requires an additional exact path"
        )

    if layer["native_kv_crop_to_query_length"]:
        length = layer["effective_key_length"]
        key = key[:, :, :length, :]
        value = value[:, :, :length, :]

    if key.shape[1] != query.shape[1]:
        key = sdpa_mod.repeat_kv(
            key,
            layer["group_count"],
        )
        value = sdpa_mod.repeat_kv(
            value,
            layer["group_count"],
        )

    logits = (
        torch.matmul(query, key.transpose(-2, -1))
        * layer["scaling_factor"]
    )
    mask = layer["attention_mask"]

    if mask is not None:
        mask = mask[
            ...,
            : logits.shape[-2],
            : logits.shape[-1],
        ]
        if mask.dtype == torch.bool:
            admissible = mask.expand(logits.shape)
            logits = logits.masked_fill(
                ~admissible,
                torch.finfo(logits.dtype).min,
            )
        else:
            expanded_mask = mask.float().expand(logits.shape)
            logits = logits + expanded_mask
            admissible = (
                expanded_mask
                != torch.finfo(expanded_mask.dtype).min
            )
    elif layer["native_is_causal"]:
        query_length, key_length = logits.shape[-2:]
        causal = torch.ones(
            (query_length, key_length),
            dtype=torch.bool,
        ).tril()
        admissible = causal.reshape(
            1,
            1,
            query_length,
            key_length,
        ).expand(logits.shape)
        logits = logits.masked_fill(
            ~admissible,
            torch.finfo(logits.dtype).min,
        )
    else:
        admissible = torch.ones_like(
            logits,
            dtype=torch.bool,
        )

    attention = torch.softmax(
        logits,
        dim=-1,
        dtype=torch.float32,
    )
    reconstructed_output = torch.matmul(
        attention,
        value,
    ).transpose(1, 2).contiguous()
    maximum_difference = float(
        (
            reconstructed_output
            - layer["native_output"].float()
        )
        .abs()
        .max()
        .item()
    )
    if (
        maximum_difference
        > RECONSTRUCTED_OUTPUT_ABS_TOLERANCE
    ):
        raise RuntimeError(
            "offline reconstruction output difference "
            f"{maximum_difference} exceeds the predeclared tolerance"
        )

    descriptor_rows = []
    for head in range(attention.shape[1]):
        for query_index in range(attention.shape[2]):
            allowed = admissible[0, head, query_index]
            weights = attention[
                0,
                head,
                query_index,
            ][allowed].double()
            count = int(weights.numel())
            maximum_mass = float(weights.max().item())
            if count == 1:
                normalized_entropy = 0.0
            else:
                positive = weights[weights > 0]
                normalized_entropy = float(
                    (
                        -(
                            positive
                            * positive.log()
                        ).sum()
                        / torch.tensor(
                            float(count),
                            dtype=torch.float64,
                        ).log()
                    ).item()
                )
            normalized_effective_support = float(
                (
                    (1.0 / weights.square().sum())
                    / count
                ).item()
            )
            descriptor_rows.append(
                {
                    "forward_event": layer[
                        "forward_event_index"
                    ],
                    "layer": layer["layer_index"],
                    "head": head,
                    "query_position": layer[
                        "query_positions"
                    ][query_index],
                    "admissible_key_count": count,
                    "maximum_attention_mass": maximum_mass,
                    "normalized_entropy": normalized_entropy,
                    "normalized_effective_support": (
                        normalized_effective_support
                    ),
                }
            )

    layer["logits"] = logits.cpu()
    layer["attention"] = attention.cpu()
    layer[
        "reconstruction_native_output_max_abs_difference"
    ] = maximum_difference
    return descriptor_rows, maximum_difference


def process_raw_events(
    raw_manifest,
    save_final,
    reference_hashes=None,
):
    descriptor_rows = []
    final_manifest = []
    overall_maximum_difference = 0.0

    for raw_record in raw_manifest:
        event = torch.load(
            raw_record["path"],
            map_location="cpu",
            weights_only=False,
        )
        for layer in event["layers"]:
            rows, difference = reconstruct_layer(layer)
            descriptor_rows.extend(rows)
            overall_maximum_difference = max(
                overall_maximum_difference,
                difference,
            )

        estimated_tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for layer in event["layers"]
            for field in TENSOR_FIELDS
            for tensor in [layer.get(field)]
            if isinstance(tensor, torch.Tensor)
        )
        if (
            estimated_tensor_bytes
            >= ARTIFACT_FILE_SIZE_GUARD_BYTES
        ):
            raise RuntimeError(
                "event artifact estimate reaches the "
                "predeclared individual-file guard"
            )

        final_hash = semantic_event_hash(
            event,
            include_reconstruction=True,
        )
        record = {
            "forward_event_index": event[
                "metadata"
            ]["forward_event_index"],
            "raw_semantic_sha256": raw_record[
                "raw_semantic_sha256"
            ],
            "final_semantic_sha256": final_hash,
            "estimated_tensor_bytes_before_write": (
                estimated_tensor_bytes
            ),
        }

        if save_final:
            artifact_path = ARTIFACT_DIR / (
                "attention_event_"
                f"{record['forward_event_index']:03d}.pt"
            )
            torch.save(event, artifact_path)
            record.update(
                {
                    "path": str(
                        artifact_path.relative_to(ROOT)
                    ),
                    "file_sha256": sha256_file(
                        artifact_path
                    ),
                    "file_bytes": (
                        artifact_path.stat().st_size
                    ),
                }
            )

        final_manifest.append(record)

    final_hashes = [
        item["final_semantic_sha256"]
        for item in final_manifest
    ]
    if (
        reference_hashes is not None
        and final_hashes != reference_hashes
    ):
        raise RuntimeError(
            "replicated reconstructed event hashes differ"
        )

    return (
        descriptor_rows,
        final_manifest,
        overall_maximum_difference,
    )


def distribution_summary(rows, field):
    values = torch.tensor(
        [row[field] for row in rows],
        dtype=torch.float64,
    )
    return {
        "count": int(values.numel()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "standard_deviation": float(
            values.std(unbiased=False).item()
        ),
        "quantiles": {
            f"{quantile:.2f}": float(
                torch.quantile(
                    values,
                    quantile,
                ).item()
            )
            for quantile in QUANTILES
        },
    }



def reconstruct_layer(layer):
    query = layer["query"].float()
    key = layer["key"].float()
    value = layer["value"].float()

    if layer["position_bias_present"]:
        raise RuntimeError(
            "unexpected position bias requires an additional exact path"
        )

    if layer["native_kv_crop_to_query_length"]:
        length = layer["effective_key_length"]
        key = key[:, :, :length, :]
        value = value[:, :, :length, :]

    if key.shape[1] != query.shape[1]:
        key = sdpa_mod.repeat_kv(
            key,
            layer["group_count"],
        )
        value = sdpa_mod.repeat_kv(
            value,
            layer["group_count"],
        )

    logits = (
        torch.matmul(query, key.transpose(-2, -1))
        * layer["scaling_factor"]
    )
    mask = layer["attention_mask"]

    if mask is not None:
        mask = mask[
            ...,
            : logits.shape[-2],
            : logits.shape[-1],
        ]
        if mask.dtype == torch.bool:
            admissible = mask.expand(logits.shape)
            logits = logits.masked_fill(
                ~admissible,
                torch.finfo(logits.dtype).min,
            )
        else:
            expanded_mask = mask.float().expand(logits.shape)
            logits = logits + expanded_mask
            admissible = (
                expanded_mask
                != torch.finfo(expanded_mask.dtype).min
            )
    elif layer["native_is_causal"]:
        query_length, key_length = logits.shape[-2:]
        causal = torch.ones(
            (query_length, key_length),
            dtype=torch.bool,
        ).tril()
        admissible = causal.reshape(
            1,
            1,
            query_length,
            key_length,
        ).expand(logits.shape)
        logits = logits.masked_fill(
            ~admissible,
            torch.finfo(logits.dtype).min,
        )
    else:
        admissible = torch.ones_like(
            logits,
            dtype=torch.bool,
        )

    attention = torch.softmax(
        logits,
        dim=-1,
        dtype=torch.float32,
    )
    reconstructed_output = torch.matmul(
        attention,
        value,
    ).transpose(1, 2).contiguous()
    maximum_difference = float(
        (
            reconstructed_output
            - layer["native_output"].float()
        )
        .abs()
        .max()
        .item()
    )
    if (
        maximum_difference
        > RECONSTRUCTED_OUTPUT_ABS_TOLERANCE
    ):
        raise RuntimeError(
            "offline reconstruction output difference "
            f"{maximum_difference} exceeds the predeclared tolerance"
        )

    descriptor_rows = []
    for head in range(attention.shape[1]):
        for query_index in range(attention.shape[2]):
            allowed = admissible[0, head, query_index]
            weights = attention[
                0,
                head,
                query_index,
            ][allowed].double()
            count = int(weights.numel())
            maximum_mass = float(weights.max().item())
            if count == 1:
                normalized_entropy = 0.0
            else:
                positive = weights[weights > 0]
                normalized_entropy = float(
                    (
                        -(
                            positive
                            * positive.log()
                        ).sum()
                        / torch.tensor(
                            float(count),
                            dtype=torch.float64,
                        ).log()
                    ).item()
                )
            normalized_effective_support = float(
                (
                    (1.0 / weights.square().sum())
                    / count
                ).item()
            )
            descriptor_rows.append(
                {
                    "forward_event": layer[
                        "forward_event_index"
                    ],
                    "layer": layer["layer_index"],
                    "head": head,
                    "query_position": layer[
                        "query_positions"
                    ][query_index],
                    "admissible_key_count": count,
                    "maximum_attention_mass": maximum_mass,
                    "normalized_entropy": normalized_entropy,
                    "normalized_effective_support": (
                        normalized_effective_support
                    ),
                }
            )

    layer["logits"] = logits.cpu()
    layer["attention"] = attention.cpu()
    layer[
        "reconstruction_native_output_max_abs_difference"
    ] = maximum_difference
    return descriptor_rows, maximum_difference


def process_raw_events(
    raw_manifest,
    save_final,
    reference_hashes=None,
):
    descriptor_rows = []
    final_manifest = []
    overall_maximum_difference = 0.0

    for raw_record in raw_manifest:
        event = torch.load(
            raw_record["path"],
            map_location="cpu",
            weights_only=False,
        )
        for layer in event["layers"]:
            rows, difference = reconstruct_layer(layer)
            descriptor_rows.extend(rows)
            overall_maximum_difference = max(
                overall_maximum_difference,
                difference,
            )

        estimated_tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for layer in event["layers"]
            for field in TENSOR_FIELDS
            for tensor in [layer.get(field)]
            if isinstance(tensor, torch.Tensor)
        )
        if (
            estimated_tensor_bytes
            >= ARTIFACT_FILE_SIZE_GUARD_BYTES
        ):
            raise RuntimeError(
                "event artifact estimate reaches the "
                "predeclared individual-file guard"
            )

        final_hash = semantic_event_hash(
            event,
            include_reconstruction=True,
        )
        record = {
            "forward_event_index": event[
                "metadata"
            ]["forward_event_index"],
            "raw_semantic_sha256": raw_record[
                "raw_semantic_sha256"
            ],
            "final_semantic_sha256": final_hash,
            "estimated_tensor_bytes_before_write": (
                estimated_tensor_bytes
            ),
        }

        if save_final:
            artifact_path = ARTIFACT_DIR / (
                "attention_event_"
                f"{record['forward_event_index']:03d}.pt"
            )
            torch.save(event, artifact_path)
            record.update(
                {
                    "path": str(
                        artifact_path.relative_to(ROOT)
                    ),
                    "file_sha256": sha256_file(
                        artifact_path
                    ),
                    "file_bytes": (
                        artifact_path.stat().st_size
                    ),
                }
            )

        final_manifest.append(record)

    final_hashes = [
        item["final_semantic_sha256"]
        for item in final_manifest
    ]
    if (
        reference_hashes is not None
        and final_hashes != reference_hashes
    ):
        raise RuntimeError(
            "replicated reconstructed event hashes differ"
        )

    return (
        descriptor_rows,
        final_manifest,
        overall_maximum_difference,
    )


def distribution_summary(rows, field):
    values = torch.tensor(
        [row[field] for row in rows],
        dtype=torch.float64,
    )
    return {
        "count": int(values.numel()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "standard_deviation": float(
            values.std(unbiased=False).item()
        ),
        "quantiles": {
            f"{quantile:.2f}": float(
                torch.quantile(
                    values,
                    quantile,
                ).item()
            )
            for quantile in QUANTILES
        },
    }



def reconstruct_layer(layer):
    query = layer["query"].float()
    key = layer["key"].float()
    value = layer["value"].float()

    if layer["position_bias_present"]:
        raise RuntimeError(
            "unexpected position bias requires an additional exact path"
        )

    if layer["native_kv_crop_to_query_length"]:
        length = layer["effective_key_length"]
        key = key[:, :, :length, :]
        value = value[:, :, :length, :]

    if key.shape[1] != query.shape[1]:
        key = sdpa_mod.repeat_kv(
            key,
            layer["group_count"],
        )
        value = sdpa_mod.repeat_kv(
            value,
            layer["group_count"],
        )

    logits = (
        torch.matmul(query, key.transpose(-2, -1))
        * layer["scaling_factor"]
    )
    mask = layer["attention_mask"]

    if mask is not None:
        mask = mask[
            ...,
            : logits.shape[-2],
            : logits.shape[-1],
        ]
        if mask.dtype == torch.bool:
            admissible = mask.expand(logits.shape)
            logits = logits.masked_fill(
                ~admissible,
                torch.finfo(logits.dtype).min,
            )
        else:
            expanded_mask = mask.float().expand(logits.shape)
            logits = logits + expanded_mask
            admissible = (
                expanded_mask
                != torch.finfo(expanded_mask.dtype).min
            )
    elif layer["native_is_causal"]:
        query_length, key_length = logits.shape[-2:]
        causal = torch.ones(
            (query_length, key_length),
            dtype=torch.bool,
        ).tril()
        admissible = causal.reshape(
            1,
            1,
            query_length,
            key_length,
        ).expand(logits.shape)
        logits = logits.masked_fill(
            ~admissible,
            torch.finfo(logits.dtype).min,
        )
    else:
        admissible = torch.ones_like(
            logits,
            dtype=torch.bool,
        )

    attention = torch.softmax(
        logits,
        dim=-1,
        dtype=torch.float32,
    )
    reconstructed_output = torch.matmul(
        attention,
        value,
    ).transpose(1, 2).contiguous()
    maximum_difference = float(
        (
            reconstructed_output
            - layer["native_output"].float()
        )
        .abs()
        .max()
        .item()
    )
    if (
        maximum_difference
        > RECONSTRUCTED_OUTPUT_ABS_TOLERANCE
    ):
        raise RuntimeError(
            "offline reconstruction output difference "
            f"{maximum_difference} exceeds the predeclared tolerance"
        )

    descriptor_rows = []
    for head in range(attention.shape[1]):
        for query_index in range(attention.shape[2]):
            allowed = admissible[0, head, query_index]
            weights = attention[
                0,
                head,
                query_index,
            ][allowed].double()
            count = int(weights.numel())
            maximum_mass = float(weights.max().item())
            if count == 1:
                normalized_entropy = 0.0
            else:
                positive = weights[weights > 0]
                normalized_entropy = float(
                    (
                        -(
                            positive
                            * positive.log()
                        ).sum()
                        / torch.tensor(
                            float(count),
                            dtype=torch.float64,
                        ).log()
                    ).item()
                )
            normalized_effective_support = float(
                (
                    (1.0 / weights.square().sum())
                    / count
                ).item()
            )
            descriptor_rows.append(
                {
                    "forward_event": layer[
                        "forward_event_index"
                    ],
                    "layer": layer["layer_index"],
                    "head": head,
                    "query_position": layer[
                        "query_positions"
                    ][query_index],
                    "admissible_key_count": count,
                    "maximum_attention_mass": maximum_mass,
                    "normalized_entropy": normalized_entropy,
                    "normalized_effective_support": (
                        normalized_effective_support
                    ),
                }
            )

    layer["logits"] = logits.cpu()
    layer["attention"] = attention.cpu()
    layer[
        "reconstruction_native_output_max_abs_difference"
    ] = maximum_difference
    return descriptor_rows, maximum_difference


def process_raw_events(
    raw_manifest,
    save_final,
    reference_hashes=None,
):
    descriptor_rows = []
    final_manifest = []
    overall_maximum_difference = 0.0

    for raw_record in raw_manifest:
        event = torch.load(
            raw_record["path"],
            map_location="cpu",
            weights_only=False,
        )
        for layer in event["layers"]:
            rows, difference = reconstruct_layer(layer)
            descriptor_rows.extend(rows)
            overall_maximum_difference = max(
                overall_maximum_difference,
                difference,
            )

        estimated_tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for layer in event["layers"]
            for field in TENSOR_FIELDS
            for tensor in [layer.get(field)]
            if isinstance(tensor, torch.Tensor)
        )
        if (
            estimated_tensor_bytes
            >= ARTIFACT_FILE_SIZE_GUARD_BYTES
        ):
            raise RuntimeError(
                "event artifact estimate reaches the "
                "predeclared individual-file guard"
            )

        final_hash = semantic_event_hash(
            event,
            include_reconstruction=True,
        )
        record = {
            "forward_event_index": event[
                "metadata"
            ]["forward_event_index"],
            "raw_semantic_sha256": raw_record[
                "raw_semantic_sha256"
            ],
            "final_semantic_sha256": final_hash,
            "estimated_tensor_bytes_before_write": (
                estimated_tensor_bytes
            ),
        }

        if save_final:
            artifact_path = ARTIFACT_DIR / (
                "attention_event_"
                f"{record['forward_event_index']:03d}.pt"
            )
            torch.save(event, artifact_path)
            record.update(
                {
                    "path": str(
                        artifact_path.relative_to(ROOT)
                    ),
                    "file_sha256": sha256_file(
                        artifact_path
                    ),
                    "file_bytes": (
                        artifact_path.stat().st_size
                    ),
                }
            )

        final_manifest.append(record)

    final_hashes = [
        item["final_semantic_sha256"]
        for item in final_manifest
    ]
    if (
        reference_hashes is not None
        and final_hashes != reference_hashes
    ):
        raise RuntimeError(
            "replicated reconstructed event hashes differ"
        )

    return (
        descriptor_rows,
        final_manifest,
        overall_maximum_difference,
    )


def distribution_summary(rows, field):
    values = torch.tensor(
        [row[field] for row in rows],
        dtype=torch.float64,
    )
    return {
        "count": int(values.numel()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "standard_deviation": float(
            values.std(unbiased=False).item()
        ),
        "quantiles": {
            f"{quantile:.2f}": float(
                torch.quantile(
                    values,
                    quantile,
                ).item()
            )
            for quantile in QUANTILES
        },
    }



def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    RAW_A_DIR.mkdir(parents=True)
    RAW_B_DIR.mkdir(parents=True)
    ARTIFACT_DIR.mkdir(parents=True)

    slice14 = lab.load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(
        context,
        parent,
        slice14.LOCAL_CYCLE_BOUND,
        evaluate_gate=True,
    )
    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise RuntimeError(
            "authoritative shared-V source boundary did not reproduce"
        )

    inherited = json.loads(
        lab.EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    delta = tuple(inherited["delta_t"]["token_ids"])
    expected_start = inherited["K_t_original"]
    expected_post = inherited["K_t_plus_1_original"]
    expected_output = inherited["replicates"][0]["original_output"]
    model = context["model"]
    attention_module = model.model.layers[0].self_attn
    native_callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        "sdpa",
        None,
    )
    guarded_paths = [
        ROOT / "kv_causal_intervention_lab.py",
        ROOT / "slice11_runtime.py",
        ROOT / "slice14_runtime.py",
        lab.SNAPSHOT_PATH,
        Path(inspect.getsourcefile(attention_module.__class__)),
        Path(inspect.getsourcefile(native_callable)),
    ]
    guards_before = source_hashes(guarded_paths)

    control = execute_transition(
        slice11,
        context,
        parent.shared_v,
        delta,
        "REALIZED_ATTENTION_CONTROL",
    )
    if (
        control["start_fingerprint"] != expected_start
        or control["post_fingerprint"] != expected_post
        or control["generated"] != expected_output
    ):
        raise RuntimeError(
            "authoritative uninstrumented baseline did not reproduce"
        )

    observed_a = execute_transition(
        slice11,
        context,
        parent.shared_v,
        delta,
        "REALIZED_ATTENTION_OBSERVED_A",
        RAW_A_DIR,
    )
    if not transitions_identical(control, observed_a):
        mutation = {
            "outcome": "ATTENTION_OBSERVATION_MUTATED_RUNTIME",
            "control_output": control["generated"],
            "observed_output": observed_a["generated"],
            "control_K_t_plus_1": control["post_fingerprint"],
            "observed_K_t_plus_1": observed_a["post_fingerprint"],
            "source_files_modified": False,
        }
        RESULT_PATH.write_text(
            json.dumps(mutation, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("OUTCOME")
        print("ATTENTION_OBSERVATION_MUTATED_RUNTIME")
        return 2

    rows_a, manifest_a, difference_a = process_raw_events(
        observed_a["capture_manifest"],
        save_final=True,
    )



    observed_b = execute_transition(
        slice11,
        context,
        parent.shared_v,
        delta,
        "REALIZED_ATTENTION_OBSERVED_B",
        RAW_B_DIR,
    )
    if not transitions_identical(control, observed_b):
        raise RuntimeError(
            "second attention observation mutated runtime"
        )

    raw_hashes_a = [
        item["raw_semantic_sha256"]
        for item in observed_a["capture_manifest"]
    ]
    raw_hashes_b = [
        item["raw_semantic_sha256"]
        for item in observed_b["capture_manifest"]
    ]
    if raw_hashes_a != raw_hashes_b:
        raise RuntimeError(
            "replicated raw attention operands differ"
        )

    reference_hashes = [
        item["final_semantic_sha256"]
        for item in manifest_a
    ]
    rows_b, manifest_b, difference_b = process_raw_events(
        observed_b["capture_manifest"],
        save_final=False,
        reference_hashes=reference_hashes,
    )
    if rows_a != rows_b:
        raise RuntimeError(
            "replicated descriptor rows differ"
        )

    summaries = {
        "maximum_attention_mass": distribution_summary(
            rows_a,
            "maximum_attention_mass",
        ),
        "normalized_entropy": distribution_summary(
            rows_a,
            "normalized_entropy",
        ),
        "normalized_effective_support": distribution_summary(
            rows_a,
            "normalized_effective_support",
        ),
    }
    DESCRIPTOR_PATH.write_text(
        json.dumps(rows_a, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "schema": "REALIZED_ATTENTION_EVENT_MANIFEST_V1",
                "event_order": [
                    item["forward_event_index"]
                    for item in manifest_a
                ],
                "events": manifest_a,
                "replicate_b_final_semantic_hashes": [
                    item["final_semantic_sha256"]
                    for item in manifest_b
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    shutil.rmtree(RAW_A_DIR)
    shutil.rmtree(RAW_B_DIR)
    if guards_before != source_hashes(guarded_paths):
        raise RuntimeError(
            "authoritative artifact or installed source changed"
        )

    forward_event_count = len(manifest_a)
    attention_event_count = (
        forward_event_count * len(model.model.layers)
    )
    maximum_difference = max(difference_a, difference_b)



    result = {
        "schema": "REALIZED_ATTENTION_STATE_EXPOSURE_V1",
        "outcome": "REALIZED_ATTENTION_STATE_EXPOSED",
        "authoritative_parent_commit": PARENT_COMMIT,
        "native_attention_path": {
            "model_class": (
                model.__class__.__module__
                + "."
                + model.__class__.__name__
            ),
            "attention_class": (
                attention_module.__class__.__module__
                + "."
                + attention_module.__class__.__name__
            ),
            "backend": "sdpa",
            "callable": (
                native_callable.__module__
                + "."
                + native_callable.__name__
            ),
            "attention_source": inspect.getsourcefile(
                attention_module.__class__
            ),
            "native_callable_source": inspect.getsourcefile(
                native_callable
            ),
            "rotary_class": (
                model.model.rotary_emb.__class__.__module__
                + "."
                + model.model.rotary_emb.__class__.__name__
            ),
            "cache_update_order": (
                "post-RoPE key/value update before native callable"
            ),
            "native_A_directly_materialized": "NO",
        },
        "model_attention_configuration": {
            "query_heads": model.config.num_attention_heads,
            "kv_heads": model.config.num_key_value_heads,
            "head_dimension": attention_module.head_dim,
            "scale": attention_module.scaling,
            "group_count": attention_module.num_key_value_groups,
        },
        "authoritative_source_K_t": control["start_fingerprint"],
        "authoritative_delta_t": control["delta"],
        "control_K_t_plus_1": control["post_fingerprint"],
        "observed_K_t_plus_1": observed_a["post_fingerprint"],
        "generated_tokens": control["generated"],
        "forward_event_count": forward_event_count,
        "attention_event_count": attention_event_count,
        "descriptor_row_count": len(rows_a),
        "offline_reconstruction_output_max_abs_difference": (
            maximum_difference
        ),
        "offline_reconstruction_output_abs_tolerance_predeclared": (
            RECONSTRUCTED_OUTPUT_ABS_TOLERANCE
        ),
        "native_reconstruction_validation": "NOT_AVAILABLE",
        "summaries": summaries,
        "artifacts": {
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "descriptor_rows": str(
                DESCRIPTOR_PATH.relative_to(ROOT)
            ),
            "descriptor_rows_sha256": sha256_file(
                DESCRIPTOR_PATH
            ),
        },
        "checks": {
            "AUTHORITATIVE_BASELINE_REPRODUCED": "PASS",
            "ATTENTION_NATIVE_PATH_RESOLVED": "PASS",
            "Q_NATIVE_OPERAND_CAPTURE": "PASS",
            "K_NATIVE_OPERAND_CAPTURE": "PASS",
            "V_NATIVE_OPERAND_CAPTURE": "PASS",
            "MASK_OR_EXACT_MASK_RECIPE_CAPTURE": "PASS",
            "SCALING_CAPTURE": "PASS",
            "HEAD_MAPPING_CAPTURE": "PASS",
            "ATTENTION_RECONSTRUCTION": "PASS",
            "COMPLETE_ATTENTION_ARTIFACT_PRESERVED": "PASS",
            "DESCRIPTOR_DISTRIBUTIONS_PRESERVED": "PASS",
            "INSTRUMENTED_TRAVERSAL_EQUALS_CONTROL": "PASS",
            "DETERMINISTIC_ATTENTION_REPLICATION": "PASS",
            "SOURCE_AUTHORITATIVE_FACT_MUTATED": "NO",
            "INSTALLED_TRANSFORMERS_SOURCE_MODIFIED": "NO",
            "PREFERRED_NUMERICAL_TARGET_USED": "NO",
        },
    }
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )



    receipt = [
        "OUTCOME",
        "REALIZED_ATTENTION_STATE_EXPOSED",
        "",
        "AUTHORITATIVE_PARENT_COMMIT: " + PARENT_COMMIT,
        "NEW_COMMIT: PENDING_PUBLICATION",
        (
            "NATIVE_ATTENTION_PATH: "
            + result["native_attention_path"]["model_class"]
            + " / "
            + result["native_attention_path"]["attention_class"]
            + " / sdpa / "
            + result["native_attention_path"]["callable"]
        ),
        (
            "MODEL_ATTENTION_CONFIGURATION: "
            "query_heads=9 kv_heads=3 "
            "head_dimension=64 scale=0.125 grouping=3"
        ),
        (
            "AUTHORITATIVE_SOURCE_K_t: "
            + canonical_json(control["start_fingerprint"])
        ),
        (
            "AUTHORITATIVE_DELTA_t: "
            + canonical_json(control["delta"])
        ),
        (
            "CONTROL_K_t_PLUS_1: "
            + canonical_json(control["post_fingerprint"])
        ),
        (
            "OBSERVED_K_t_PLUS_1: "
            + canonical_json(observed_a["post_fingerprint"])
        ),
        "INSTRUMENTED_TRAVERSAL_EQUALS_CONTROL: PASS",
        "ATTENTION_EVENT_COUNT: " + str(attention_event_count),
        (
            "CAPTURED_OPERANDS: Q=PASS K=PASS V=PASS "
            "MASK=PASS SCALE=PASS HEAD_MAPPING=PASS"
        ),
        "NATIVE_A_DIRECTLY_MATERIALIZED: NO",
        "OFFLINE_RECONSTRUCTION: PASS",
        "NATIVE_RECONSTRUCTION_VALIDATION: NOT_AVAILABLE",
        (
            "COMPLETE_ATTENTION_ARTIFACTS: "
            + str(MANIFEST_PATH.relative_to(ROOT))
        ),
        "DESCRIPTOR_ROW_COUNT: " + str(len(rows_a)),
        (
            "MAXIMUM_ATTENTION_MASS_DISTRIBUTION: "
            + canonical_json(
                summaries["maximum_attention_mass"]
            )
        ),
        (
            "NORMALIZED_ENTROPY_DISTRIBUTION: "
            + canonical_json(summaries["normalized_entropy"])
        ),
        (
            "NORMALIZED_EFFECTIVE_SUPPORT_DISTRIBUTION: "
            + canonical_json(
                summaries["normalized_effective_support"]
            )
        ),
        "DETERMINISTIC_ATTENTION_REPLICATION: PASS",
        "SOURCE_AUTHORITATIVE_FACT_MUTATED: NO",
        "INSTALLED_TRANSFORMERS_SOURCE_MODIFIED: NO",
        "PREFERRED_NUMERICAL_TARGET_USED: NO",
    ]
    RECEIPT_PATH.write_text(
        "\n".join(receipt) + "\n",
        encoding="utf-8",
    )
    print(
        RECEIPT_PATH.read_text(encoding="utf-8"),
        flush=True,
    )
    print("RESULT_JSON=" + str(RESULT_PATH), flush=True)
    print("MANIFEST=" + str(MANIFEST_PATH), flush=True)
    gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            "REALIZED_ATTENTION_STATE_EXPOSURE_ERROR="
            f"{error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
