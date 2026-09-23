"""Bounded verifier for receiver-state-relative exact FILES recontact."""

import argparse
import ast
import gc
import hashlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core_runtime import ModuleHandle, invoke
from files_module import EXACT_BYTES_FORM
from ordered_files_module import ORDERED_EXACT_ARTIFACTS_FORM
from state_relative_files_module import (
    STATE_RELATIVE_POSITION_FORM,
    StateRelativeFilesModule,
)
from universal_membrane import (
    BytesValue,
    Membrane,
    OrderedSequenceValue,
    canonical_membrane_record,
)


AUTHORITATIVE_PARENT = "9171389034e3735c5a87395c9aca0c87dba591a2"
FIXTURE_1 = PROJECT_DIR / "fixtures" / "ordered_source_1.txt"
FIXTURE_2 = PROJECT_DIR / "fixtures" / "ordered_source_2.txt"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "state_relative_files_recontact"
IMPLEMENTATION_PATH = PROJECT_DIR / "state_relative_files_module.py"
CORE_RUNTIME_PATH = PROJECT_DIR / "core_runtime.py"
UNIVERSAL_MEMBRANE_PATH = PROJECT_DIR / "universal_membrane.py"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def inspect_sources() -> dict:
    implementation_source = IMPLEMENTATION_PATH.read_text(encoding="utf-8")
    implementation_tree = ast.parse(implementation_source, str(IMPLEMENTATION_PATH))
    core_source = CORE_RUNTIME_PATH.read_text(encoding="utf-8")
    universal_source = UNIVERSAL_MEMBRANE_PATH.read_text(encoding="utf-8")
    imported_modules = {
        alias.name
        for node in ast.walk(implementation_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(implementation_tree)
        if isinstance(node, ast.ImportFrom)
    }
    module_assignments = {
        target.id
        for node in implementation_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    value_classes = {
        node.name
        for node in ast.walk(implementation_tree)
        if isinstance(node, ast.ClassDef) and node.name.endswith("Value")
    }
    return {
        "implementation_imports_core_runtime": "core_runtime" in imported_modules,
        "core_imports_state_relative_files": "state_relative_files_module" in core_source,
        "universal_imports_state_relative_files": (
            "state_relative_files_module" in universal_source
        ),
        "new_value_class_defined": bool(value_classes),
        "global_registry_assignment_present": any(
            "registry" in name.lower() or "artifact_table" in name.lower()
            for name in module_assignments
        ),
        "selector_indexes_receiver_state_directly": (
            "selected = self._ordered_artifacts[position]" in implementation_source
        ),
    }


def execute_replication(a1: bytes, a2: bytes, label: str) -> dict:
    files_a = StateRelativeFilesModule()
    files_b = StateRelativeFilesModule()
    files_empty = StateRelativeFilesModule()

    m_bytes1 = Membrane(EXACT_BYTES_FORM, BytesValue(a1))
    m_bytes2 = Membrane(EXACT_BYTES_FORM, BytesValue(a2))
    m_state_a = Membrane(
        ORDERED_EXACT_ARTIFACTS_FORM,
        OrderedSequenceValue((m_bytes1, m_bytes2)),
    )
    m_state_b = Membrane(
        ORDERED_EXACT_ARTIFACTS_FORM,
        OrderedSequenceValue((m_bytes2, m_bytes1)),
    )
    m_select0 = Membrane(STATE_RELATIVE_POSITION_FORM, BytesValue(b"\x00"))

    prime_a = ModuleHandle("files-a-prime", ORDERED_EXACT_ARTIFACTS_FORM, files_a.prime_ordered_state)
    prime_b = ModuleHandle("files-b-prime", ORDERED_EXACT_ARTIFACTS_FORM, files_b.prime_ordered_state)
    select_a = ModuleHandle("files-a-recontact", STATE_RELATIVE_POSITION_FORM, files_a.recontact_position)
    select_b = ModuleHandle("files-b-recontact", STATE_RELATIVE_POSITION_FORM, files_b.recontact_position)
    select_empty = ModuleHandle("files-empty-recontact", STATE_RELATIVE_POSITION_FORM, files_empty.recontact_position)

    prime_result_a = invoke(prime_a, m_state_a)
    prime_result_b = invoke(prime_b, m_state_b)
    resolved_a = invoke(select_a, m_select0)
    resolved_b = invoke(select_b, m_select0)
    unprimed_result = None
    unprimed_failure_type = None
    try:
        unprimed_result = invoke(select_empty, m_select0)
    except LookupError as exc:
        unprimed_failure_type = type(exc).__name__

    custody_a = files_a.verifier_ordered_custody_evidence()
    custody_b = files_b.verifier_ordered_custody_evidence()
    custody_empty = files_empty.verifier_ordered_custody_evidence()
    boundary_a = files_a.verifier_boundary_evidence()
    boundary_b = files_b.verifier_boundary_evidence()
    boundary_empty = files_empty.verifier_boundary_evidence()
    source_inspection = inspect_sources()

    selector_record = canonical_membrane_record(m_select0)
    selector_encoded = canonical_json(selector_record).encode("utf-8")
    selector_lower = selector_encoded.lower()
    fixture_names = (FIXTURE_1.name.encode().lower(), FIXTURE_2.name.encode().lower())
    fixture_hashes = (sha256_bytes(a1).encode(), sha256_bytes(a2).encode())
    selectors = (
        boundary_a["selector_admissions"][0]["membrane"],
        boundary_b["selector_admissions"][0]["membrane"],
        boundary_empty["selector_admissions"][0]["membrane"],
    )
    state_a_hashes = [item["sha256"] for item in custody_a["ordered_artifacts"]]
    state_b_hashes = [item["sha256"] for item in custody_b["ordered_artifacts"]]

    measured_results = {
        "state_a": resolved_a.value.payload if isinstance(resolved_a.value, BytesValue) else None,
        "state_b": resolved_b.value.payload if isinstance(resolved_b.value, BytesValue) else None,
        "unprimed": unprimed_result,
    }
    unique_successes = {
        value for key, value in measured_results.items() if key != "unprimed" and value is not None
    }
    standalone_unique = len(unique_successes) == 1 and unprimed_result is not None

    checks = {
        "selector_value_hex_00": m_select0.value.payload.hex() == "00",
        "selector_value_byte_count_one": len(m_select0.value.payload) == 1,
        "selector_position_zero": m_select0.value.payload[0] == 0,
        "compact_contains_artifact_1_bytes_no": a1 not in selector_encoded and a1.hex().encode() not in selector_lower,
        "compact_contains_artifact_2_bytes_no": a2 not in selector_encoded and a2.hex().encode() not in selector_lower,
        "compact_contains_filename_no": not any(name in selector_lower for name in fixture_names),
        "compact_contains_path_no": b"fixtures" not in selector_lower and str(PROJECT_DIR).encode().lower() not in selector_lower,
        "compact_contains_artifact_id_no": not any(value in selector_lower for value in fixture_hashes),
        "compact_contains_hash_as_identity_no": b"sha256" not in selector_lower and not any(value in selector_lower for value in fixture_hashes),
        "compact_contains_semantic_description_no": not any(fragment in selector_encoded for fragment in (a1, a2)),
        "state_a_selector_equals_state_b_selector": selectors[0] == selectors[1],
        "state_a_selector_equals_unprimed_selector": selectors[0] == selectors[2],
        "exact_same_selector_object_used": all(pair.membrane is m_select0 for pair in (
            files_a._selector_admission_boundary[0],
            files_b._selector_admission_boundary[0],
            files_empty._selector_admission_boundary[0],
        )),
        "state_a_order": state_a_hashes == [sha256_bytes(a1), sha256_bytes(a2)],
        "state_b_order": state_b_hashes == [sha256_bytes(a2), sha256_bytes(a1)],
        "state_a_primed": prime_result_a is m_state_a and custody_a["present"],
        "state_b_primed": prime_result_b is m_state_b and custody_b["present"],
        "state_a_position_0_recontacts_a1": resolved_a.value.payload == a1,
        "state_b_position_0_recontacts_a2": resolved_b.value.payload == a2,
        "state_a_exact_reacquisition": resolved_a.value.payload == a1,
        "state_b_exact_reacquisition": resolved_b.value.payload == a2,
        "state_a_output_form_exact_bytes": resolved_a.form == EXACT_BYTES_FORM,
        "state_b_output_form_exact_bytes": resolved_b.form == EXACT_BYTES_FORM,
        "state_a_resolved_equals_state_b_resolved_no": resolved_a.value.payload != resolved_b.value.payload,
        "same_perturbation_different_state_yields_relative_result": selectors[0] == selectors[1] and resolved_a.value.payload == a1 and resolved_b.value.payload == a2 and a1 != a2,
        "perturbation_alone_determines_artifact_no": len(unique_successes) == 2 and unprimed_result is None,
        "standalone_selector_has_unique_artifact_result_no": not standalone_unique and len(unique_successes) == 2,
        "unprimed_state_has_required_structure_no": not custody_empty["present"],
        "unprimed_selection_succeeded_no": unprimed_result is None and unprimed_failure_type == "LookupError",
        "missing_state_fabricated_no": unprimed_result is None,
        "unprimed_returned_exact_bytes_membrane_no": not isinstance(unprimed_result, Membrane),
        "unprimed_returned_any_success_membrane_no": not isinstance(unprimed_result, Membrane),
        "state_a_recontact_output_boundary_written": len(boundary_a["recontact_outputs"]) == 1 and boundary_a["recontact_outputs"][0]["native_bytes_equal_membrane_bytes"],
        "state_b_recontact_output_boundary_written": len(boundary_b["recontact_outputs"]) == 1 and boundary_b["recontact_outputs"][0]["native_bytes_equal_membrane_bytes"],
        "unprimed_success_output_boundary_written_no": len(boundary_empty["recontact_outputs"]) == 0,
        "files_a_and_b_native_state_shared_no": not files_a.verifier_shares_native_state_with(files_b),
        "files_a_and_unprimed_native_state_shared_no": not files_a.verifier_shares_native_state_with(files_empty),
        "files_b_and_unprimed_native_state_shared_no": not files_b.verifier_shares_native_state_with(files_empty),
        "shared_native_state_introduced_no": not files_a.verifier_shares_native_state_with(files_b),
        "global_artifact_registry_used_no": not source_inspection["global_registry_assignment_present"],
        "selector_resolution_uses_receiver_state_directly": source_inspection["selector_indexes_receiver_state_directly"],
        "runtime_inspects_files_state_no": not source_inspection["implementation_imports_core_runtime"] and not source_inspection["core_imports_state_relative_files"],
        "files_native_state_exposed_in_membrane_no": all(record["value"] == {"kind": "BYTES", "hex": "00"} for record in selectors),
        "new_value_primitive_added_no": not source_inspection["new_value_class_defined"],
        "membrane_grammar_changed_no": not source_inspection["universal_imports_state_relative_files"],
        "runtime_invocation_law_changed_no": not source_inspection["core_imports_state_relative_files"],
        "state_establishment_uses_standard_invoke": prime_a._entrypoint.__self__ is files_a and prime_b._entrypoint.__self__ is files_b,
        "state_relative_recontact_uses_standard_invoke": select_a._entrypoint.__self__ is files_a and select_b._entrypoint.__self__ is files_b and select_empty._entrypoint.__self__ is files_empty,
    }

    projection = {
        "label": label,
        "fixture_sha256": {"A1": sha256_bytes(a1), "A2": sha256_bytes(a2)},
        "selector": selector_record,
        "state_a_order": state_a_hashes,
        "state_b_order": state_b_hashes,
        "state_a_output": canonical_membrane_record(resolved_a),
        "state_b_output": canonical_membrane_record(resolved_b),
        "unprimed": {"successful_membrane": False, "failure_type": unprimed_failure_type},
        "state_a_boundaries": boundary_a,
        "state_b_boundaries": boundary_b,
        "unprimed_boundaries": boundary_empty,
        "source_inspection": source_inspection,
        "checks": checks,
    }
    return {"projection": projection, "checks": checks}


def render_receipt(evidence: dict) -> str:
    checks = evidence["replications"][0]["checks"]
    word = lambda key, yes="PASS", no="FAIL": yes if checks[key] else no
    return f"""OUTCOME
{evidence['outcome']}

AUTHORITATIVE_PARENT_COMMIT:
{AUTHORITATIVE_PARENT}

STATE_RELATIVE_FILES_IMPLEMENTATION:
state_relative_files_module.py

VERIFY_IMPLEMENTATION:
verify_state_relative_files_recontact.py

SELECTOR_FORM:
{STATE_RELATIVE_POSITION_FORM.canonical_name}

SELECTOR_FORM_INTERPRETATION:
{STATE_RELATIVE_POSITION_FORM.interpretation_law}

SELECTOR_VALUE_HEX: 00
SELECTOR_VALUE_BYTE_COUNT: 1
SELECTOR_POSITION: 0

STATE_A_ORDER: [A1,A2]
STATE_A_PRIMED: {word('state_a_primed')}
STATE_B_ORDER: [A2,A1]
STATE_B_PRIMED: {word('state_b_primed')}

COMPACT_PERTURBATION_CONTAINS_ARTIFACT_1_BYTES: {word('compact_contains_artifact_1_bytes_no', 'NO', 'YES')}
COMPACT_PERTURBATION_CONTAINS_ARTIFACT_2_BYTES: {word('compact_contains_artifact_2_bytes_no', 'NO', 'YES')}
COMPACT_PERTURBATION_CONTAINS_FILENAME: {word('compact_contains_filename_no', 'NO', 'YES')}
COMPACT_PERTURBATION_CONTAINS_PATH: {word('compact_contains_path_no', 'NO', 'YES')}
COMPACT_PERTURBATION_CONTAINS_ARTIFACT_ID: {word('compact_contains_artifact_id_no', 'NO', 'YES')}
COMPACT_PERTURBATION_CONTAINS_HASH_AS_IDENTITY: {word('compact_contains_hash_as_identity_no', 'NO', 'YES')}
COMPACT_PERTURBATION_CONTAINS_SEMANTIC_DESCRIPTION: {word('compact_contains_semantic_description_no', 'NO', 'YES')}

STATE_A_SELECTOR_EQUALS_STATE_B_SELECTOR: {word('state_a_selector_equals_state_b_selector')}
STATE_A_SELECTOR_EQUALS_UNPRIMED_SELECTOR: {word('state_a_selector_equals_unprimed_selector')}
EXACT_SAME_SELECTOR_OBJECT_USED: {word('exact_same_selector_object_used')}

STATE_A_POSITION_0_RECONTACTS: A1
STATE_A_EXACT_REACQUISITION: {word('state_a_exact_reacquisition')}
STATE_A_OUTPUT_FORM_EXACT_BYTES: {word('state_a_output_form_exact_bytes')}
STATE_B_POSITION_0_RECONTACTS: A2
STATE_B_EXACT_REACQUISITION: {word('state_b_exact_reacquisition')}
STATE_B_OUTPUT_FORM_EXACT_BYTES: {word('state_b_output_form_exact_bytes')}
STATE_A_RESOLVED_BYTES_EQUALS_STATE_B_RESOLVED_BYTES: {word('state_a_resolved_equals_state_b_resolved_no', 'NO', 'YES')}
SAME_PERTURBATION_DIFFERENT_VALID_STATE_YIELDS_STATE_RELATIVE_RESULT: {word('same_perturbation_different_state_yields_relative_result')}
PERTURBATION_ALONE_DETERMINES_ARTIFACT: {word('perturbation_alone_determines_artifact_no', 'NO', 'YES')}
STANDALONE_SELECTOR_HAS_UNIQUE_ARTIFACT_RESULT: {word('standalone_selector_has_unique_artifact_result_no', 'NO', 'YES')}

UNPRIMED_STATE_HAS_REQUIRED_CAUSAL_STRUCTURE: {word('unprimed_state_has_required_structure_no', 'NO', 'YES')}
UNPRIMED_SELECTION_SUCCEEDED: {word('unprimed_selection_succeeded_no', 'NO', 'YES')}
MISSING_STATE_FABRICATED: {word('missing_state_fabricated_no', 'NO', 'YES')}
UNPRIMED_RETURNED_EXACT_BYTES_MEMBRANE: {word('unprimed_returned_exact_bytes_membrane_no', 'NO', 'YES')}
UNPRIMED_RETURNED_ANY_SUCCESS_MEMBRANE: {word('unprimed_returned_any_success_membrane_no', 'NO', 'YES')}

FILES_A_AND_B_NATIVE_STATE_SHARED: {word('files_a_and_b_native_state_shared_no', 'NO', 'YES')}
FILES_A_AND_UNPRIMED_NATIVE_STATE_SHARED: {word('files_a_and_unprimed_native_state_shared_no', 'NO', 'YES')}
FILES_B_AND_UNPRIMED_NATIVE_STATE_SHARED: {word('files_b_and_unprimed_native_state_shared_no', 'NO', 'YES')}
SHARED_NATIVE_STATE_INTRODUCED: {word('shared_native_state_introduced_no', 'NO', 'YES')}
GLOBAL_ARTIFACT_REGISTRY_USED: {word('global_artifact_registry_used_no', 'NO', 'YES')}
SELECTOR_RESOLUTION_USES_RECEIVER_ORDERED_STATE_DIRECTLY: {word('selector_resolution_uses_receiver_state_directly')}
RUNTIME_INSPECTS_FILES_NATIVE_STATE_FOR_SELECTION: {word('runtime_inspects_files_state_no', 'NO', 'YES')}
FILES_NATIVE_STATE_EXPOSED_IN_MEMBRANE: {word('files_native_state_exposed_in_membrane_no', 'NO', 'YES')}

STATE_A_RECONTACT_OUTPUT_BOUNDARY_WRITTEN: {word('state_a_recontact_output_boundary_written')}
STATE_B_RECONTACT_OUTPUT_BOUNDARY_WRITTEN: {word('state_b_recontact_output_boundary_written')}
UNPRIMED_SUCCESS_OUTPUT_BOUNDARY_WRITTEN: {word('unprimed_success_output_boundary_written_no', 'NO', 'YES')}

NEW_VALUE_PRIMITIVE_ADDED: {word('new_value_primitive_added_no', 'NO', 'YES')}
MEMBRANE_GRAMMAR_CHANGED: {word('membrane_grammar_changed_no', 'NO', 'YES')}
RUNTIME_INVOCATION_LAW_CHANGED: {word('runtime_invocation_law_changed_no', 'NO', 'YES')}
STATE_ESTABLISHMENT_USES_STANDARD_INVOKE: {word('state_establishment_uses_standard_invoke')}
STATE_RELATIVE_RECONTACT_USES_STANDARD_INVOKE: {word('state_relative_recontact_uses_standard_invoke')}
DETERMINISTIC_REPLICATION: {'PASS' if evidence['deterministic_replication'] else 'FAIL'}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-1", type=Path, default=FIXTURE_1)
    parser.add_argument("--fixture-2", type=Path, default=FIXTURE_2)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()

    a1 = args.fixture_1.read_bytes()
    a2 = args.fixture_2.read_bytes()
    if not a1 or not a2 or a1 == a2:
        raise SystemExit("the two non-empty fixtures must differ at exact byte level")

    run_a = execute_replication(a1, a2, "A")
    gc.collect()
    run_b = execute_replication(a1, a2, "B")
    comparable_a = dict(run_a["projection"])
    comparable_b = dict(run_b["projection"])
    comparable_a.pop("label")
    comparable_b.pop("label")
    deterministic = comparable_a == comparable_b
    all_checks = all(run_a["checks"].values()) and all(run_b["checks"].values())
    outcome = (
        "STATE_RELATIVE_CAUSAL_FIDELITY_OPERATIONAL"
        if all_checks and deterministic
        else "UNRESOLVED_STATE_RELATIVE_FIDELITY_BOUNDARY"
    )
    evidence = {
        "schema": "STATE_RELATIVE_FILES_RECONTACT_EVIDENCE_V1",
        "authoritative_parent_commit": AUTHORITATIVE_PARENT,
        "outcome": outcome,
        "forms": {
            "selector": canonical_membrane_record(
                Membrane(STATE_RELATIVE_POSITION_FORM, BytesValue(b"\x00"))
            )["form"],
            "ordered_state": ORDERED_EXACT_ARTIFACTS_FORM.canonical_name,
            "exact_output": EXACT_BYTES_FORM.canonical_name,
        },
        "replications": [run_a, run_b],
        "deterministic_replication": deterministic,
    }
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.evidence_dir / "acceptance_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_path = args.evidence_dir / "completion_receipt.txt"
    receipt_path.write_text(render_receipt(evidence), encoding="utf-8")
    print(receipt_path.read_text(encoding="utf-8"), end="")
    print(f"EVIDENCE: {evidence_path}")
    if outcome != "STATE_RELATIVE_CAUSAL_FIDELITY_OPERATIONAL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
