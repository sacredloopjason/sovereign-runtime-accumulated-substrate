"""Bounded state-relative positional recontact inside one FILES instance."""

import hashlib
import json
from dataclasses import dataclass

from files_module import EXACT_BYTES_FORM
from ordered_files_module import ORDERED_EXACT_ARTIFACTS_FORM
from universal_membrane import (
    BytesValue,
    Form,
    Membrane,
    OrderedSequenceValue,
    canonical_membrane_record,
)


STATE_RELATIVE_POSITION_FORM = Form(
    canonical_name="SOVEREIGN_STATE_RELATIVE_POSITION_U8_V1",
    interpretation_law=(
        "VALUE is BYTES containing exactly one unsigned one-byte zero-based "
        "structural position selecting one member from the receiving module's "
        "currently retained ordered-artifact state; the VALUE carries no "
        "artifact content or artifact identity."
    ),
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _membrane_record_sha256(membrane: Membrane) -> str:
    encoded = json.dumps(
        canonical_membrane_record(membrane), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


@dataclass(frozen=True, slots=True)
class _NativeArtifact:
    content: bytes


@dataclass(frozen=True, slots=True)
class _OrderedAdmissionBoundaryPair:
    membrane: Membrane
    native_artifacts: tuple[_NativeArtifact, _NativeArtifact]


@dataclass(frozen=True, slots=True)
class _SelectorAdmissionBoundaryPair:
    membrane: Membrane
    selector_position: int
    ordered_state_present: bool
    resolved_native_position: int | None


@dataclass(frozen=True, slots=True)
class _RecontactOutputBoundaryPair:
    native_artifact: _NativeArtifact
    membrane: Membrane


class StateRelativeFilesModule:
    """Exactly two private ordered artifacts plus bounded position-zero recontact."""

    def __init__(self) -> None:
        self._ordered_artifacts: tuple[_NativeArtifact, _NativeArtifact] | None = None
        self._ordered_admission_boundary: list[_OrderedAdmissionBoundaryPair] = []
        self._selector_admission_boundary: list[_SelectorAdmissionBoundaryPair] = []
        self._recontact_output_boundary: list[_RecontactOutputBoundaryPair] = []

    def prime_ordered_state(self, membrane: Membrane) -> Membrane:
        """Admit exactly two ordered exact artifacts into this instance's custody."""
        if membrane.form != ORDERED_EXACT_ARTIFACTS_FORM:
            raise ValueError(
                "exact SOVEREIGN_ORDERED_EXACT_ARTIFACTS_V1 FORM equality is required"
            )
        if not isinstance(membrane.value, OrderedSequenceValue):
            raise TypeError("the ordered-artifacts FORM requires ORDERED_SEQUENCE")
        if len(membrane.value.items) != 2:
            raise ValueError("this bounded state requires exactly two children")

        children = membrane.value.items
        for child in children:
            if child.form != EXACT_BYTES_FORM:
                raise ValueError("every child requires exact SOVEREIGN_EXACT_BYTES_V1 FORM")
            if not isinstance(child.value, BytesValue):
                raise TypeError("every exact-byte child requires BYTES")
            if not child.value.payload:
                raise ValueError("each bounded source artifact must be non-empty")

        artifacts = (
            _NativeArtifact(bytes(bytearray(children[0].value.payload))),
            _NativeArtifact(bytes(bytearray(children[1].value.payload))),
        )
        self._ordered_artifacts = artifacts
        self._ordered_admission_boundary.append(
            _OrderedAdmissionBoundaryPair(membrane, artifacts)
        )
        return membrane

    def recontact_position(self, membrane: Membrane) -> Membrane:
        """Resolve only position zero against this instance's retained ordered state."""
        if membrane.form != STATE_RELATIVE_POSITION_FORM:
            raise ValueError(
                "exact SOVEREIGN_STATE_RELATIVE_POSITION_U8_V1 FORM equality is required"
            )
        if not isinstance(membrane.value, BytesValue):
            raise TypeError("the state-relative position FORM requires BYTES")
        if len(membrane.value.payload) != 1:
            raise ValueError("the selector must contain exactly one unsigned byte")

        position = membrane.value.payload[0]
        if position != 0:
            raise IndexError("this bounded slice permits only structural position zero")

        state_present = self._ordered_artifacts is not None
        self._selector_admission_boundary.append(
            _SelectorAdmissionBoundaryPair(
                membrane=membrane,
                selector_position=position,
                ordered_state_present=state_present,
                resolved_native_position=position if state_present else None,
            )
        )
        if self._ordered_artifacts is None:
            raise LookupError("FILES has no ordered native artifact state")

        selected = self._ordered_artifacts[position]
        emitted = Membrane(
            EXACT_BYTES_FORM,
            BytesValue(bytes(bytearray(selected.content))),
        )
        self._recontact_output_boundary.append(
            _RecontactOutputBoundaryPair(selected, emitted)
        )
        return emitted

    # These projections are verifier-only and never enter runtime invocation.
    def verifier_ordered_custody_evidence(self) -> dict:
        if self._ordered_artifacts is None:
            return {"present": False, "ordered_artifacts": []}
        return {
            "present": True,
            "ordered_artifacts": [
                {
                    "position": position,
                    "byte_count": len(artifact.content),
                    "sha256": _sha256_bytes(artifact.content),
                }
                for position, artifact in enumerate(self._ordered_artifacts)
            ],
            "custody_objects_distinct": (
                self._ordered_artifacts[0] is not self._ordered_artifacts[1]
            ),
            "byte_objects_distinct": (
                self._ordered_artifacts[0].content
                is not self._ordered_artifacts[1].content
            ),
        }

    def verifier_boundary_evidence(self) -> dict:
        ordered_admissions = [
            {
                "membrane_sha256": _membrane_record_sha256(pair.membrane),
                "ordered_native_artifacts": [
                    {
                        "position": position,
                        "byte_count": len(artifact.content),
                        "sha256": _sha256_bytes(artifact.content),
                    }
                    for position, artifact in enumerate(pair.native_artifacts)
                ],
            }
            for pair in self._ordered_admission_boundary
        ]
        selector_admissions = [
            {
                "membrane": canonical_membrane_record(pair.membrane),
                "selector_position": pair.selector_position,
                "ordered_state_present": pair.ordered_state_present,
                "resolved_native_position": pair.resolved_native_position,
            }
            for pair in self._selector_admission_boundary
        ]
        recontact_outputs = [
            {
                "native_artifact": {
                    "byte_count": len(pair.native_artifact.content),
                    "sha256": _sha256_bytes(pair.native_artifact.content),
                },
                "membrane": canonical_membrane_record(pair.membrane),
                "native_bytes_equal_membrane_bytes": (
                    isinstance(pair.membrane.value, BytesValue)
                    and pair.native_artifact.content == pair.membrane.value.payload
                ),
            }
            for pair in self._recontact_output_boundary
        ]
        return {
            "ordered_state_admissions": ordered_admissions,
            "selector_admissions": selector_admissions,
            "recontact_outputs": recontact_outputs,
        }

    def verifier_shares_native_state_with(
        self, other: "StateRelativeFilesModule"
    ) -> bool:
        if not isinstance(other, StateRelativeFilesModule):
            raise TypeError("comparison requires another StateRelativeFilesModule")
        if self._ordered_artifacts is None or other._ordered_artifacts is None:
            return self._ordered_artifacts is other._ordered_artifacts and (
                self._ordered_artifacts is not None
            )
        return self._ordered_artifacts is other._ordered_artifacts or any(
            left is right or left.content is right.content
            for left in self._ordered_artifacts
            for right in other._ordered_artifacts
        )
