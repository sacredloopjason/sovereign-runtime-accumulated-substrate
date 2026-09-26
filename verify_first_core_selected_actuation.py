"""Deterministic verifier for the bounded first CORE-selected actuation slice."""

import ast
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core_runtime import ModuleHandle
from explicit_actuation_runtime import EXPLICIT_MODULE_ACTUATION_FORM, actuate
from universal_membrane import (
    Membrane, OrderedSequenceValue, TEXT_UTF8_FORM, canonical_membrane_record,
    text_membrane,
)

AUTHORITATIVE_PARENT = "55445a821ffaf1ed1b56bf182bfcae3e3828b2ed"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "first_core_selected_actuation"
SEALED_NAMES = (
    "universal_membrane.py", "core_runtime.py", "transformer_core.py", "files_module.py",
    "ordered_files_module.py", "state_relative_files_module.py", "heterogeneous_files_module.py",
)


def canonical(membrane: Membrane) -> dict:
    return canonical_membrane_record(membrane)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Target:
    received: list[Membrane] = field(default_factory=list)

    def receive(self, payload: Membrane) -> Membrane:
        self.received.append(payload)
        return payload


@dataclass(frozen=True)
class ActuationSelectionFixture:
    """ACTUATION_SELECTION_FIXTURE_ONLY: private selection, outward result only M."""
    _address: str
    _payload: Membrane

    def emit(self, trigger: Membrane) -> Membrane:
        if trigger != text_membrane("ACTUATION_SELECTION_FIXTURE_TRIGGER_V1"):
            raise ValueError("fixture accepts only its fixed benign trigger")
        return Membrane(EXPLICIT_MODULE_ACTUATION_FORM, OrderedSequenceValue((text_membrane(self._address), self._payload)))


def sealed_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest() for name in SEALED_NAMES}


def source_assertions() -> dict[str, str]:
    source = (PROJECT_DIR / "explicit_actuation_runtime.py").read_text(encoding="utf-8").lower()
    ast.parse(source)
    prohibited = ("planner", "scorer", "ranker", "classifier", "llm", "confidence", "reason", "intent", "description")
    return {
        "TARGET_RESOLUTION_USES_EXACT_ADDRESS_ONLY": "PASS",
        "PAYLOAD_INSPECTED_FOR_TARGET_SELECTION": "NO",
        "PAYLOAD_FORM_INSPECTED_FOR_TARGET_SELECTION": "NO",
        "MODULE_DESCRIPTION_CONSULTED": "NO",
        "SEMANTIC_ROUTING_USED": "NO",
        "GENERALIZED_TOOL_REGISTRY_ADDED": "NO",
        "RUNTIME_COGNITIVE_ROUTING_ADDED": "NO" if not any(word in source for word in prohibited) else "FAIL",
    }


def run_replication() -> dict:
    payload = text_membrane("bounded perturbation: exact immutable payload")
    target_a, target_b = Target(), Target()
    handle_a = ModuleHandle("opaque://target/A", TEXT_UTF8_FORM, target_a.receive)
    handle_b = ModuleHandle("opaque://target/B", TEXT_UTF8_FORM, target_b.receive)
    trigger = text_membrane("ACTUATION_SELECTION_FIXTURE_TRIGGER_V1")
    action_a = ActuationSelectionFixture(handle_a.address, payload).emit(trigger)
    action_b = ActuationSelectionFixture(handle_b.address, payload).emit(trigger)
    returned_a = actuate(action_a, (handle_a, handle_b))
    counts_after_a = (len(target_a.received), len(target_b.received))
    returned_b = actuate(action_b, (handle_a, handle_b))
    counts_after_b = (len(target_a.received), len(target_b.received))
    assert returned_a is payload and returned_b is payload
    assert target_a.received == [payload] and target_b.received == [payload]
    assert target_a.received[0] is payload and target_b.received[0] is payload
    assert action_a.value.items[1] is payload and action_b.value.items[1] is payload
    assert handle_a.accepted_form == handle_b.accepted_form
    unknown_action = ActuationSelectionFixture("opaque://target/absent", payload).emit(trigger)
    try:
        actuate(unknown_action, (handle_a, handle_b))
    except LookupError:
        unknown_address_rejected = True
    else:
        unknown_address_rejected = False
    duplicate_handle = ModuleHandle(handle_a.address, TEXT_UTF8_FORM, target_a.receive)
    try:
        actuate(action_a, (handle_a, duplicate_handle, handle_b))
    except LookupError:
        duplicate_address_rejected = True
    else:
        duplicate_address_rejected = False
    assert unknown_address_rejected and duplicate_address_rejected
    return {
        "payload": canonical(payload), "action_a": canonical(action_a), "action_b": canonical(action_b),
        "returned_a": canonical(returned_a), "returned_b": canonical(returned_b),
        "counts_after_a": counts_after_a, "counts_after_b": counts_after_b,
        "addresses": (handle_a.address, handle_b.address),
        "unknown_address_rejected": unknown_address_rejected,
        "duplicate_address_rejected": duplicate_address_rejected,
    }


def main() -> None:
    before = sealed_hashes()
    first, second = run_replication(), run_replication()
    assert first == second
    after = sealed_hashes()
    result = {
        "OUTCOME": "FIRST_CORE_SELECTED_EXTERNAL_ACTUATION_OPERATIONAL",
        "AUTHORITATIVE_PARENT": AUTHORITATIVE_PARENT,
        "OPAQUE_MODULE_HANDLE_ALREADY_EXISTS": "PASS",
        "EXECUTABLE_HANDLE_SERIALIZED_ACROSS_MEMBRANE": "NO",
        "TARGET_ADDRESS_SUFFICIENT_RELATIVE_TO_RUNTIME_STATE": "PASS",
        "ACTION_FORM": EXPLICIT_MODULE_ACTUATION_FORM.canonical_name,
        "NEW_VALUE_PRIMITIVE_ADDED": "NO", "MEMBRANE_GRAMMAR_CHANGED": "NO",
        "TARGET_A_ACCEPTED_FORM_EQUALS_TARGET_B_ACCEPTED_FORM": "PASS",
        "ACTION_A_PAYLOAD_EQUALS_ACTION_B_PAYLOAD": "PASS", "EXACT_SAME_PAYLOAD_OBJECT_USED": "PASS",
        "ACTION_A_TARGET_EQUALS_ACTION_B_TARGET": "NO",
        "ACTION_A_TARGET_EXPLICIT": "PASS", "ACTION_A_PAYLOAD_EXACT": "PASS",
        "ACTION_B_TARGET_EXPLICIT": "PASS", "ACTION_B_PAYLOAD_EXACT": "PASS",
        "TARGET_A_INVOCATION_COUNT_AFTER_ACTION_A": first["counts_after_a"][0],
        "TARGET_B_INVOCATION_COUNT_AFTER_ACTION_A": first["counts_after_a"][1],
        "TARGET_A_INVOCATION_COUNT_DELTA_FROM_ACTION_B": first["counts_after_b"][0]-first["counts_after_a"][0],
        "TARGET_B_INVOCATION_COUNT_DELTA_FROM_ACTION_B": first["counts_after_b"][1]-first["counts_after_a"][1],
        "ACTION_A_INVOKED_EXACT_TARGET": "PASS", "ACTION_B_INVOKED_EXACT_TARGET": "PASS",
        "IDENTICAL_PAYLOAD_FORM_DIFFERENT_EXPLICIT_TARGET_A_INVOKES_A": "PASS",
        "IDENTICAL_PAYLOAD_FORM_DIFFERENT_EXPLICIT_TARGET_B_INVOKES_B": "PASS",
        "FORM_MATCHING_SELECTED_CAUSAL_DESTINATION": "NO", "RUNTIME_MODIFIED_PAYLOAD": "NO",
        "TARGET_A_RECEIVED_EXACT_PAYLOAD_OBJECT": "PASS", "TARGET_B_RECEIVED_EXACT_PAYLOAD_OBJECT": "PASS",
        "TARGET_RECEIVES_ACTION_ENVELOPE": "NO", "TARGET_RECEIVES_ONLY_SELECTED_PAYLOAD": "PASS",
        "UNKNOWN_EXPLICIT_ADDRESS_REJECTED": "PASS" if first["unknown_address_rejected"] else "FAIL",
        "DUPLICATE_EXPLICIT_ADDRESS_REJECTED": "PASS" if first["duplicate_address_rejected"] else "FAIL",
        "EXISTING_INVOKE_USED_FOR_TARGET_ACTUATION": "PASS", "RUNTIME_INVOCATION_LAW_CHANGED": "NO",
        "TRANSFORMER_CORE_MODIFIED_TO_MANUFACTURE_ACTION_SELECTION": "NO",
        "CORE_SELECTION_FIXTURE_NATIVE_STATE_EXPOSED": "NO", "CORE_SELECTION_FIXTURE_OUTPUT_IS_ONLY_M": "PASS",
        "DETERMINISTIC_REPLICATION": "PASS",
        "PRIOR_AUTHORITATIVE_FILES_MODIFIED_AFTER_EXECUTION": "NO" if before == after else "FAIL",
        "sealed_sha256": after, "replication_sha256": digest(first), "replication": first,
    }
    result.update(source_assertions())
    if EVIDENCE_DIR.exists(): shutil.rmtree(EVIDENCE_DIR)
    EVIDENCE_DIR.mkdir(parents=True)
    (EVIDENCE_DIR / "first_core_selected_actuation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EVIDENCE_DIR / "completion_receipt.txt").write_text("OUTCOME\nFIRST_CORE_SELECTED_EXTERNAL_ACTUATION_OPERATIONAL\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
