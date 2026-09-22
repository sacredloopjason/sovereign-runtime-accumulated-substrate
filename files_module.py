"""First bounded FILES module over the universal membrane."""

import hashlib
import json
from dataclasses import dataclass

from universal_membrane import (
    BytesValue,
    Form,
    Membrane,
    TEXT_UTF8_FORM,
    canonical_membrane_record,
)


EXACT_BYTES_FORM = Form(
    canonical_name="SOVEREIGN_EXACT_BYTES_V1",
    interpretation_law=(
        "VALUE is BYTES containing the exact ordered byte sequence supplied; the "
        "membrane assigns no content type, encoding, file type, modality, semantic "
        "interpretation, or native object structure to those bytes."
    ),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_membrane(membrane: Membrane) -> str:
    record = canonical_membrane_record(membrane)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _NativeArtifact:
    content: bytes


@dataclass(frozen=True, slots=True)
class _NativeText:
    content: str


@dataclass(frozen=True, slots=True)
class _IngressBoundaryPair:
    membrane: Membrane
    native_artifact: _NativeArtifact


@dataclass(frozen=True, slots=True)
class _EgressBoundaryPair:
    native_text: _NativeText
    membrane: Membrane


class FilesModule:
    """One-fixture FILES realization with private source custody."""

    accepted_form = EXACT_BYTES_FORM

    def __init__(self) -> None:
        self._artifacts: list[_NativeArtifact] = []
        self._ingress_boundary: list[_IngressBoundaryPair] = []
        self._egress_boundary: list[_EgressBoundaryPair] = []

    def invoke(self, membrane: Membrane) -> Membrane:
        if membrane.form != EXACT_BYTES_FORM:
            raise ValueError("exact SOVEREIGN_EXACT_BYTES_V1 FORM equality is required")
        if not isinstance(membrane.value, BytesValue):
            raise TypeError("the exact-bytes FORM requires BYTES")
        if not membrane.value.payload:
            raise ValueError("the bounded source artifact must be non-empty")

        # The round trip through bytearray guarantees distinct byte-owning custody.
        artifact = _NativeArtifact(bytes(bytearray(membrane.value.payload)))
        self._artifacts.append(artifact)
        self._ingress_boundary.append(_IngressBoundaryPair(membrane, artifact))

        decoded = _NativeText(artifact.content.decode("utf-8", errors="strict"))
        emitted = Membrane(
            TEXT_UTF8_FORM,
            BytesValue(decoded.content.encode("utf-8", errors="strict")),
        )
        self._egress_boundary.append(_EgressBoundaryPair(decoded, emitted))
        return emitted

    # These projections are verifier-only and never enter runtime invocation.
    def verifier_reacquire_source_bytes(self) -> bytes:
        if not self._artifacts:
            raise RuntimeError("FILES has no native artifact in custody")
        return bytes(bytearray(self._artifacts[-1].content))

    def verifier_custody_evidence(self) -> dict:
        if not self._artifacts:
            raise RuntimeError("FILES has no native artifact in custody")
        artifact = self._artifacts[-1]
        return {
            "native_artifact_present": True,
            "native_artifact_byte_count": len(artifact.content),
            "native_artifact_sha256": _sha256_bytes(artifact.content),
        }

    def verifier_boundary_evidence(self) -> dict:
        ingress = [
            {
                "membrane_sha256": _sha256_membrane(pair.membrane),
                "native_artifact_present": pair.native_artifact is not None,
                "native_artifact_byte_count": len(pair.native_artifact.content),
                "native_artifact_sha256": _sha256_bytes(pair.native_artifact.content),
            }
            for pair in self._ingress_boundary
        ]
        egress = [
            {
                "native_decoded_text_present": pair.native_text is not None,
                "native_text_utf8_sha256": _sha256_bytes(
                    pair.native_text.content.encode("utf-8", errors="strict")
                ),
                "membrane_sha256": _sha256_membrane(pair.membrane),
                "native_text_utf8_equals_emitted_bytes": (
                    isinstance(pair.membrane.value, BytesValue)
                    and pair.native_text.content.encode("utf-8", errors="strict")
                    == pair.membrane.value.payload
                ),
            }
            for pair in self._egress_boundary
        ]
        return {"ingress": ingress, "egress": egress}

    def verifier_last_output_is(self, membrane: Membrane) -> bool:
        return bool(self._egress_boundary) and self._egress_boundary[-1].membrane is membrane
