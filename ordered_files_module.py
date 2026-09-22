"""Bounded ordered two-artifact FILES module over the existing membrane grammar."""

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


ORDERED_EXACT_ARTIFACTS_FORM = Form(
    canonical_name="SOVEREIGN_ORDERED_EXACT_ARTIFACTS_V1",
    interpretation_law=(
        "VALUE is ORDERED_SEQUENCE of independently complete "
        "SOVEREIGN_EXACT_BYTES_V1 membrane representations; child boundaries "
        "are significant and child order is significant."
    ),
)

ORDERED_SHARED_TEXTS_FORM = Form(
    canonical_name="SOVEREIGN_ORDERED_SHARED_TEXTS_V1",
    interpretation_law=(
        "VALUE is ORDERED_SEQUENCE of independently complete "
        "SOVEREIGN_SHARED_TEXT_UTF8_V1 membrane representations; child "
        "boundaries are significant and child order is significant."
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
    native_texts: tuple[_NativeText, _NativeText]
    membrane: Membrane


class OrderedFilesModule:
    """Exactly two independently bounded exact-byte artifacts in structural order."""

    accepted_form = ORDERED_EXACT_ARTIFACTS_FORM

    def __init__(self) -> None:
        self._artifacts: tuple[_NativeArtifact, _NativeArtifact] | None = None
        self._ingress_boundary: list[_IngressBoundaryPair] = []
        self._egress_boundary: list[_EgressBoundaryPair] = []

    def invoke(self, membrane: Membrane) -> Membrane:
        if membrane.form != ORDERED_EXACT_ARTIFACTS_FORM:
            raise ValueError(
                "exact SOVEREIGN_ORDERED_EXACT_ARTIFACTS_V1 FORM equality is required"
            )
        if not isinstance(membrane.value, OrderedSequenceValue):
            raise TypeError("the ordered-artifacts FORM requires ORDERED_SEQUENCE")
        if len(membrane.value.items) != 2:
            raise ValueError("this bounded slice requires exactly two children")

        source_children = membrane.value.items
        for child in source_children:
            if child.form != EXACT_BYTES_FORM:
                raise ValueError("every child requires exact SOVEREIGN_EXACT_BYTES_V1 FORM")
            if not isinstance(child.value, BytesValue):
                raise TypeError("every exact-byte child requires BYTES")
            if not child.value.payload:
                raise ValueError("each bounded source artifact must be non-empty")

        # Each round trip allocates a separate byte-owning native custody unit.
        artifacts = (
            _NativeArtifact(bytes(bytearray(source_children[0].value.payload))),
            _NativeArtifact(bytes(bytearray(source_children[1].value.payload))),
        )
        self._artifacts = artifacts
        self._ingress_boundary.append(_IngressBoundaryPair(membrane, artifacts))

        native_texts = (
            _NativeText(artifacts[0].content.decode("utf-8", errors="strict")),
            _NativeText(artifacts[1].content.decode("utf-8", errors="strict")),
        )
        text_children = (
            Membrane(
                TEXT_UTF8_FORM,
                BytesValue(native_texts[0].content.encode("utf-8", errors="strict")),
            ),
            Membrane(
                TEXT_UTF8_FORM,
                BytesValue(native_texts[1].content.encode("utf-8", errors="strict")),
            ),
        )
        emitted = Membrane(
            ORDERED_SHARED_TEXTS_FORM,
            OrderedSequenceValue(text_children),
        )
        self._egress_boundary.append(_EgressBoundaryPair(native_texts, emitted))
        return emitted

    def verifier_reacquire_source_bytes(self, position: int) -> bytes:
        if self._artifacts is None:
            raise RuntimeError("FILES has no ordered native artifacts in custody")
        if position not in (1, 2):
            raise IndexError("bounded source position must be 1 or 2")
        return bytes(bytearray(self._artifacts[position - 1].content))

    def verifier_custody_evidence(self) -> dict:
        if self._artifacts is None:
            raise RuntimeError("FILES has no ordered native artifacts in custody")
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
            "native_custody_objects_distinct": self._artifacts[0] is not self._artifacts[1],
            "native_byte_objects_distinct": (
                self._artifacts[0].content is not self._artifacts[1].content
            ),
            "custody_is_tuple": type(self._artifacts) is tuple,
        }

    def verifier_boundary_evidence(self) -> dict:
        ingress = []
        for pair in self._ingress_boundary:
            ingress.append(
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
            )
        egress = []
        for pair in self._egress_boundary:
            children = pair.membrane.value.items
            egress.append(
                {
                    "native_text_child_count": len(pair.native_texts),
                    "ordered_native_texts": [
                        {
                            "position": position,
                            "utf8_byte_count": len(
                                native_text.content.encode("utf-8", errors="strict")
                            ),
                            "utf8_sha256": _sha256_bytes(
                                native_text.content.encode("utf-8", errors="strict")
                            ),
                            "equals_emitted_child_bytes": (
                                isinstance(children[position - 1].value, BytesValue)
                                and native_text.content.encode("utf-8", errors="strict")
                                == children[position - 1].value.payload
                            ),
                        }
                        for position, native_text in enumerate(
                            pair.native_texts, start=1
                        )
                    ],
                    "membrane_sha256": _sha256_membrane(pair.membrane),
                }
            )
        return {"ingress": ingress, "egress": egress}

    def verifier_last_output_is(self, membrane: Membrane) -> bool:
        return bool(self._egress_boundary) and self._egress_boundary[-1].membrane is membrane
