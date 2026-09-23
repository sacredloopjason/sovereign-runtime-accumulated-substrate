"""Bounded heterogeneous text/exact-bytes FILES exposure."""

import hashlib
import json
from dataclasses import dataclass

from files_module import EXACT_BYTES_FORM
from universal_membrane import (
    BytesValue,
    Form,
    Membrane,
    OrderedSequenceValue,
    TEXT_UTF8_FORM,
    canonical_membrane_record,
)


ORDERED_TEXT_BINARY_SOURCES_FORM = Form(
    canonical_name="SOVEREIGN_ORDERED_TEXT_BINARY_SOURCES_V1",
    interpretation_law=(
        "VALUE is ORDERED_SEQUENCE of exactly two independently complete "
        "SOVEREIGN_EXACT_BYTES_V1 membrane representations; child 1 is the "
        "source intended for strict UTF-8 exposure, child 2 is the source "
        "subjected to the same strict UTF-8 exposure attempt; child boundaries "
        "and order are significant."
    ),
)

ORDERED_TEXT_AND_EXACT_BYTES_FORM = Form(
    canonical_name="SOVEREIGN_ORDERED_TEXT_AND_EXACT_BYTES_V1",
    interpretation_law=(
        "VALUE is ORDERED_SEQUENCE of exactly two independently complete "
        "membrane representations; child 1 conforms exactly to "
        "SOVEREIGN_SHARED_TEXT_UTF8_V1; child 2 conforms exactly to "
        "SOVEREIGN_EXACT_BYTES_V1; child boundaries and order are significant."
    ),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_membrane(membrane: Membrane) -> str:
    encoded = json.dumps(
        canonical_membrane_record(membrane), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


@dataclass(frozen=True, slots=True)
class _NativeArtifact:
    content: bytes


@dataclass(frozen=True, slots=True)
class _NativeText:
    content: str


@dataclass(frozen=True, slots=True)
class _IngressBoundaryPair:
    membrane: Membrane
    native_artifacts: tuple[_NativeArtifact, _NativeArtifact]


@dataclass(frozen=True, slots=True)
class _EgressBoundaryPair:
    native_text: _NativeText
    native_artifact: _NativeArtifact
    binary_text_translation_succeeded: bool
    membrane: Membrane


class HeterogeneousFilesModule:
    """Exactly one strict-text source and one strict-UTF-8 rejection control."""

    accepted_form = ORDERED_TEXT_BINARY_SOURCES_FORM

    def __init__(self) -> None:
        self._artifacts: tuple[_NativeArtifact, _NativeArtifact] | None = None
        self._ingress_boundary: list[_IngressBoundaryPair] = []
        self._egress_boundary: list[_EgressBoundaryPair] = []

    def invoke(self, membrane: Membrane) -> Membrane:
        if membrane.form != ORDERED_TEXT_BINARY_SOURCES_FORM:
            raise ValueError(
                "exact SOVEREIGN_ORDERED_TEXT_BINARY_SOURCES_V1 FORM equality "
                "is required"
            )
        if not isinstance(membrane.value, OrderedSequenceValue):
            raise TypeError("the heterogeneous source FORM requires ORDERED_SEQUENCE")
        if len(membrane.value.items) != 2:
            raise ValueError("this bounded forcing case requires exactly two children")

        children = membrane.value.items
        for child in children:
            if child.form != EXACT_BYTES_FORM:
                raise ValueError("every source child requires exact-byte FORM equality")
            if not isinstance(child.value, BytesValue):
                raise TypeError("every exact-byte child requires BYTES")
            if not child.value.payload:
                raise ValueError("each bounded source artifact must be non-empty")

        artifacts = (
            _NativeArtifact(bytes(bytearray(children[0].value.payload))),
            _NativeArtifact(bytes(bytearray(children[1].value.payload))),
        )
        self._artifacts = artifacts
        self._ingress_boundary.append(_IngressBoundaryPair(membrane, artifacts))

        native_text = _NativeText(
            artifacts[0].content.decode("utf-8", errors="strict")
        )
        try:
            artifacts[1].content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            binary_text_translation_succeeded = False
        else:
            raise ValueError(
                "the second source must deterministically reject strict UTF-8"
            )

        text_child = Membrane(
            TEXT_UTF8_FORM,
            BytesValue(native_text.content.encode("utf-8", errors="strict")),
        )
        exact_bytes_child = Membrane(
            EXACT_BYTES_FORM,
            BytesValue(bytes(bytearray(artifacts[1].content))),
        )
        emitted = Membrane(
            ORDERED_TEXT_AND_EXACT_BYTES_FORM,
            OrderedSequenceValue((text_child, exact_bytes_child)),
        )
        self._egress_boundary.append(
            _EgressBoundaryPair(
                native_text=native_text,
                native_artifact=artifacts[1],
                binary_text_translation_succeeded=binary_text_translation_succeeded,
                membrane=emitted,
            )
        )
        return emitted

    def verifier_reacquire_source_bytes(self, position: int) -> bytes:
        if self._artifacts is None:
            raise RuntimeError("FILES has no heterogeneous native custody")
        if position not in (1, 2):
            raise IndexError("bounded source position must be 1 or 2")
        return bytes(bytearray(self._artifacts[position - 1].content))

    def verifier_custody_evidence(self) -> dict:
        if self._artifacts is None:
            raise RuntimeError("FILES has no heterogeneous native custody")
        return {
            "ordered_positions": [1, 2],
            "child_count": 2,
            "children": [
                {
                    "position": position,
                    "byte_count": len(artifact.content),
                    "sha256": _sha256_bytes(artifact.content),
                }
                for position, artifact in enumerate(self._artifacts, start=1)
            ],
            "native_custody_objects_distinct": self._artifacts[0]
            is not self._artifacts[1],
            "native_byte_objects_distinct": self._artifacts[0].content
            is not self._artifacts[1].content,
            "custody_is_tuple": type(self._artifacts) is tuple,
        }

    def verifier_boundary_evidence(self) -> dict:
        ingress = [
            {
                "membrane_sha256": _sha256_membrane(pair.membrane),
                "child_count": len(pair.native_artifacts),
                "ordered_native_artifacts": [
                    {
                        "position": position,
                        "byte_count": len(artifact.content),
                        "sha256": _sha256_bytes(artifact.content),
                    }
                    for position, artifact in enumerate(
                        pair.native_artifacts, start=1
                    )
                ],
            }
            for pair in self._ingress_boundary
        ]
        egress = [
            {
                "native_text_present": pair.native_text is not None,
                "native_text_utf8_sha256": _sha256_bytes(
                    pair.native_text.content.encode("utf-8", errors="strict")
                ),
                "binary_text_translation_succeeded": (
                    pair.binary_text_translation_succeeded
                ),
                "binary_native_artifact_present": pair.native_artifact is not None,
                "binary_native_byte_count": len(pair.native_artifact.content),
                "binary_native_sha256": _sha256_bytes(pair.native_artifact.content),
                "membrane_sha256": _sha256_membrane(pair.membrane),
                "text_native_equals_emitted_child_bytes": (
                    isinstance(pair.membrane.value, OrderedSequenceValue)
                    and isinstance(pair.membrane.value.items[0].value, BytesValue)
                    and pair.native_text.content.encode("utf-8", errors="strict")
                    == pair.membrane.value.items[0].value.payload
                ),
                "binary_native_equals_emitted_child_bytes": (
                    isinstance(pair.membrane.value, OrderedSequenceValue)
                    and isinstance(pair.membrane.value.items[1].value, BytesValue)
                    and pair.native_artifact.content
                    == pair.membrane.value.items[1].value.payload
                ),
            }
            for pair in self._egress_boundary
        ]
        return {"ingress": ingress, "egress": egress}

    def verifier_last_output_is(self, membrane: Membrane) -> bool:
        return bool(self._egress_boundary) and self._egress_boundary[-1].membrane is membrane
