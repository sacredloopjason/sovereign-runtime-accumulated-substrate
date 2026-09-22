"""Concrete sealed CORE for the authoritative local transformer substrate."""

import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from universal_membrane import (
    BytesValue,
    Membrane,
    TEXT_UTF8_FORM,
    canonical_membrane_record,
    read_text,
    text_membrane,
)


PROJECT_DIR = Path(__file__).resolve().parent
KERNEL_PATH = (
    PROJECT_DIR
    / "external_substrate"
    / "slice1-kernel-0e214f972bb44fcf983bb0930af3d773"
    / "runtime.py"
)
MAX_NEW_TOKENS = 16


def load_native_kernel():
    name = "core_private_authoritative_slice1_kernel"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, KERNEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load authoritative kernel: {KERNEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_determinism() -> None:
    random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_flush_denormal(True)


def reset_deterministic_seed() -> None:
    random.seed(0)
    torch.manual_seed(0)


def _sha256_json(value: object, domain: bytes) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + encoded).hexdigest()


@dataclass(slots=True)
class _IngressBoundaryPair:
    membrane: Membrane
    native_input: object


@dataclass(slots=True)
class _EgressBoundaryPair:
    native_output: object
    membrane: Membrane


class TransformerCore:
    """A state-owning module whose public operational surface is invoke(M)->M."""

    accepted_form = TEXT_UTF8_FORM

    def __init__(self, model_directory: Path):
        self._kernel = load_native_kernel()
        self._kernel.MODEL_DIR = model_directory.resolve()
        self._weight_sha256 = self._kernel.verify_substrate()
        self._tokenizer, self._model = self._kernel.build_model()
        self._cache = self._kernel.DynamicCache(config=self._model.config)
        self._ingress_boundary: list[_IngressBoundaryPair] = []
        self._egress_boundary: list[_EgressBoundaryPair] = []
        self._invocation_states: list[dict] = []

    def _state_fingerprint(self) -> dict:
        return self._kernel.fingerprint_cache(self._cache)

    def invoke(self, membrane: Membrane) -> Membrane:
        text = read_text(membrane)
        encoded_text = text + "\n"
        tokenized = self._tokenizer(
            encoded_text, add_special_tokens=False, return_tensors="pt"
        )
        input_ids = tokenized.input_ids.cpu()
        if input_ids.numel() == 0:
            raise RuntimeError("CORE ingress encoded to no native input")

        # Passive writer: retain the realized pair without judging or changing it.
        self._ingress_boundary.append(_IngressBoundaryPair(membrane, input_ids))
        state_pre = self._state_fingerprint()

        def forward(tokens):
            past_length = int(self._cache.get_seq_length())
            token_count = int(tokens.shape[1])
            attention_mask = torch.ones(
                (1, past_length + token_count), dtype=torch.long
            )
            cache_position = torch.arange(
                past_length, past_length + token_count, dtype=torch.long
            )
            with torch.inference_mode():
                outputs = self._model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=self._cache,
                    use_cache=True,
                )
            if outputs.past_key_values is not self._cache:
                raise RuntimeError("CORE traversal replaced its private native state")
            return outputs.logits[:, -1, :]

        logits = forward(input_ids)
        generated_ids: list[int] = []
        for _ in range(MAX_NEW_TOKENS):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            logits = forward(next_token)
            if (
                self._tokenizer.eos_token_id is not None
                and token_id == int(self._tokenizer.eos_token_id)
            ):
                break
        if not generated_ids:
            raise RuntimeError("CORE egress produced no native output")

        output_text = self._tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        result = text_membrane(output_text)
        native_output = tuple(generated_ids)
        # Passive writer: retain the realized pair without judging or changing it.
        self._egress_boundary.append(_EgressBoundaryPair(native_output, result))
        state_post = self._state_fingerprint()
        self._invocation_states.append({"pre": state_pre, "post": state_post})
        return result

    # The methods below are verifier-only evidence projections. They are never
    # supplied to the outer runtime and do not export native values.
    def verifier_state_fingerprint(self) -> dict:
        return dict(self._state_fingerprint())

    def verifier_boundary_evidence(self) -> dict:
        ingress = [
            {
                "membrane_sha256": _sha256_json(
                    canonical_membrane_record(pair.membrane), b"CORE_INGRESS_M_V1"
                ),
                "native_input_present": pair.native_input is not None,
            }
            for pair in self._ingress_boundary
        ]
        egress = [
            {
                "native_output_present": pair.native_output is not None,
                "membrane_sha256": _sha256_json(
                    canonical_membrane_record(pair.membrane), b"CORE_EGRESS_M_V1"
                ),
            }
            for pair in self._egress_boundary
        ]
        return {"ingress": ingress, "egress": egress}

    def verifier_invocation_states(self) -> list[dict]:
        return [
            {"pre": dict(item["pre"]), "post": dict(item["post"])}
            for item in self._invocation_states
        ]

    def verifier_last_ingress_is(self, membrane: Membrane) -> bool:
        return bool(self._ingress_boundary) and self._ingress_boundary[-1].membrane is membrane

    def verifier_independent_from(self, other: "TransformerCore") -> dict:
        self_parameter = next(self._model.parameters())
        other_parameter = next(other._model.parameters())
        return {
            "core_objects_distinct": self is not other,
            "model_objects_distinct": self._model is not other._model,
            "model_parameter_storage_distinct": (
                int(self_parameter.data_ptr()) != int(other_parameter.data_ptr())
            ),
            "tokenizer_objects_distinct": self._tokenizer is not other._tokenizer,
            "state_objects_distinct": self._cache is not other._cache,
            "state_storage_disjoint": self._pointer_set().isdisjoint(other._pointer_set()),
            "boundary_records_distinct": (
                self._ingress_boundary is not other._ingress_boundary
                and self._egress_boundary is not other._egress_boundary
            ),
        }

    def _pointer_set(self) -> set[int]:
        pointers: set[int] = set()
        for layer in self._cache.layers:
            for value in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if value is not None:
                    pointers.add(int(value.data_ptr()))
        return pointers
