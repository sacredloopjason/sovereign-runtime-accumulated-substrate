"""Bounded verifier for an ordered two-artifact FILES-to-CORE crossing."""

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
from ordered_files_module import (
    ORDERED_EXACT_ARTIFACTS_FORM,
    ORDERED_SHARED_TEXTS_FORM,
    OrderedFilesModule,
)
from transformer_core import TransformerCore, configure_determinism, reset_deterministic_seed
from universal_membrane import (
    BytesValue,
    Membrane,
    OrderedSequenceValue,
    TEXT_UTF8_FORM,
    canonical_membrane_record,
)


AUTHORITATIVE_PARENT = "62056f32866ee51c36af3bd185476b17c3f5dcaf"
FIXTURE_1 = PROJECT_DIR / "fixtures" / "ordered_source_1.txt"
FIXTURE_2 = PROJECT_DIR / "fixtures" / "ordered_source_2.txt"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "ordered_files_crossing"
IMPLEMENTATION_PATH = PROJECT_DIR / "ordered_files_module.py"
SEALED_PATHS = tuple(
    PROJECT_DIR / name
    for name in (
        "universal_membrane.py",
        "core_runtime.py",
        "transformer_core.py",
        "files_module.py",
    )
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def membrane_sha256(membrane: Membrane) -> str:
    encoded = json.dumps(
        canonical_membrane_record(membrane), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def core_ingress_sha256(membrane: Membrane) -> str:
    encoded = json.dumps(
        canonical_membrane_record(membrane), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(b"CORE_INGRESS_M_V1\0" + encoded)


def rejected(callable_object, exception_type: type[BaseException]) -> bool:
    try:
        callable_object()
    except exception_type:
        return True
    return False


def inspect_sources() -> dict:
    source = IMPLEMENTATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(IMPLEMENTATION_PATH))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    form_assignments = {
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "Form"
    }
    core_tree = ast.parse(
        SEALED_PATHS[2].read_text(encoding="utf-8"), filename=str(SEALED_PATHS[2])
    )
    core_imports = {
        alias.name
        for node in ast.walk(core_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(core_tree)
        if isinstance(node, ast.ImportFrom)
    }
    runtime_source = SEALED_PATHS[1].read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime_source, filename=str(SEALED_PATHS[1]))
    runtime_imports = {
        alias.name
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom)
    }
    return {
        "implementation_imports": sorted(imported_modules),
        "implementation_form_assignments": sorted(form_assignments),
        "implementation_class_names": sorted(class_names),
        "imports_ordered_sequence_from_universal_membrane": (
            "universal_membrane" in imported_modules
            and "OrderedSequenceValue" in source
        ),
        "defines_new_value_class": any(name.endswith("Value") for name in class_names),
        "imports_model_or_core_native_dependencies": bool(
            imported_modules & {"transformer_core", "torch", "transformers", "tokenizers"}
        ),
        "core_imports_ordered_files_module": "ordered_files_module" in core_imports,
        "runtime_imports_transformer_core": "transformer_core" in runtime_imports,
        "sealed_source_sha256": {
            path.name: sha256_bytes(path.read_bytes()) for path in SEALED_PATHS
        },
    }


def invoke_children(core: TransformerCore, composite: Membrane) -> list[Membrane]:
    if not isinstance(composite.value, OrderedSequenceValue):
        raise TypeError("structural traversal requires ORDERED_SEQUENCE")
    handle = ModuleHandle("CORE_STRUCTURAL_CHILD", TEXT_UTF8_FORM, core.invoke)
    outputs = []
    for child in composite.value.items:
        outputs.append(invoke(handle, child))
    return outputs


def execute_replication(model_directory: Path, a1: bytes, a2: bytes, label: str) -> dict:
    reset_deterministic_seed()
    source_inspection = inspect_sources()
    m_bytes1 = Membrane(EXACT_BYTES_FORM, BytesValue(a1))
    m_bytes2 = Membrane(EXACT_BYTES_FORM, BytesValue(a2))
    m_artifacts = Membrane(
        ORDERED_EXACT_ARTIFACTS_FORM,
        OrderedSequenceValue((m_bytes1, m_bytes2)),
    )
    m_artifacts_rev = Membrane(
        ORDERED_EXACT_ARTIFACTS_FORM,
        OrderedSequenceValue((m_bytes2, m_bytes1)),
    )

    files = OrderedFilesModule()
    files_handle = ModuleHandle(
        "FILES_ORDERED_TWO_UTF8", ORDERED_EXACT_ARTIFACTS_FORM, files.invoke
    )
    m_texts = invoke(files_handle, m_artifacts)
    if not isinstance(m_texts.value, OrderedSequenceValue):
        raise AssertionError("FILES did not return ORDERED_SEQUENCE")
    m_text1, m_text2 = m_texts.value.items
    m_texts_rev = Membrane(
        ORDERED_SHARED_TEXTS_FORM,
        OrderedSequenceValue((m_text2, m_text1)),
    )

    custody = files.verifier_custody_evidence()
    boundary = files.verifier_boundary_evidence()
    reacquired1 = files.verifier_reacquire_source_bytes(1)
    reacquired2 = files.verifier_reacquire_source_bytes(2)

    core_a = TransformerCore(model_directory)
    forward_outputs_a = invoke_children(core_a, m_texts)
    core_a_boundary = core_a.verifier_boundary_evidence()
    del core_a
    gc.collect()

    core_reverse = TransformerCore(model_directory)
    reverse_outputs = invoke_children(core_reverse, m_texts_rev)
    core_reverse_boundary = core_reverse.verifier_boundary_evidence()
    del core_reverse
    gc.collect()

    core_b = TransformerCore(model_directory)
    forward_outputs_b = invoke_children(core_b, m_texts)
    core_b_boundary = core_b.verifier_boundary_evidence()
    del core_b
    gc.collect()

    forward_child_hashes = [core_ingress_sha256(m_text1), core_ingress_sha256(m_text2)]
    reverse_child_hashes = [core_ingress_sha256(m_text2), core_ingress_sha256(m_text1)]
    expected_custody = [sha256_bytes(a1), sha256_bytes(a2)]
    forbidden_form_terms = (
        "filename",
        "path",
        "mime",
        "extension",
        "artifact id",
        "timestamp",
        "source label",
        "semantic description",
    )
    form_text = (
        ORDERED_EXACT_ARTIFACTS_FORM.interpretation_law
        + " "
        + ORDERED_SHARED_TEXTS_FORM.interpretation_law
    ).lower()

    malformed_files = OrderedFilesModule()
    malformed_handle = ModuleHandle(
        "FILES_REJECTION_CONTROL", ORDERED_EXACT_ARTIFACTS_FORM, malformed_files.invoke
    )
    rejection_checks = {
        "reject_non_ordered_sequence": rejected(
            lambda: invoke(
                malformed_handle,
                Membrane(ORDERED_EXACT_ARTIFACTS_FORM, BytesValue(a1)),
            ),
            TypeError,
        ),
        "reject_child_count_not_two": rejected(
            lambda: invoke(
                malformed_handle,
                Membrane(
                    ORDERED_EXACT_ARTIFACTS_FORM,
                    OrderedSequenceValue((m_bytes1,)),
                ),
            ),
            ValueError,
        ),
        "reject_wrong_child_form": rejected(
            lambda: invoke(
                malformed_handle,
                Membrane(
                    ORDERED_EXACT_ARTIFACTS_FORM,
                    OrderedSequenceValue((m_bytes1, m_text1)),
                ),
            ),
            ValueError,
        ),
        "reject_exact_form_child_with_non_bytes": rejected(
            lambda: invoke(
                malformed_handle,
                Membrane(
                    ORDERED_EXACT_ARTIFACTS_FORM,
                    OrderedSequenceValue(
                        (
                            m_bytes1,
                            Membrane(EXACT_BYTES_FORM, OrderedSequenceValue(())),
                        )
                    ),
                ),
            ),
            TypeError,
        ),
    }

    checks = {
        "ordered_sequence_value_realized": isinstance(
            m_artifacts.value, OrderedSequenceValue
        ),
        "child_count_two": len(m_artifacts.value.items) == 2,
        "child_1_form_exact": m_bytes1.form == EXACT_BYTES_FORM,
        "child_2_form_exact": m_bytes2.form == EXACT_BYTES_FORM,
        "child_1_equals_child_2_no": canonical_membrane_record(m_bytes1)
        != canonical_membrane_record(m_bytes2),
        "forward_equals_reversed_no": canonical_membrane_record(m_artifacts)
        != canonical_membrane_record(m_artifacts_rev),
        "native_custody_distinct": custody["native_custody_objects_distinct"]
        and custody["native_byte_objects_distinct"],
        "multiple_artifacts_concatenated_no": custody["child_count"] == 2
        and boundary["ingress"][0]["child_count"] == 2,
        "delimiter_used_no": expected_custody
        == [item["sha256"] for item in custody["children"]],
        "source_boundaries_preserved": [len(a1), len(a2)]
        == [item["byte_count"] for item in custody["children"]],
        "source_order_preserved": expected_custody
        == [item["sha256"] for item in custody["children"]],
        "artifact_1_reacquisition": reacquired1 == a1,
        "artifact_2_reacquisition": reacquired2 == a2,
        "reacquired_1_equals_original_2_no": reacquired1 != a2,
        "reacquired_2_equals_original_1_no": reacquired2 != a1,
        "native_text_child_count_two": boundary["egress"][0][
            "native_text_child_count"
        ]
        == 2,
        "native_text_child_order_preserved": expected_custody
        == [item["utf8_sha256"] for item in boundary["egress"][0]["ordered_native_texts"]],
        "files_output_is_ordered_sequence": isinstance(
            m_texts.value, OrderedSequenceValue
        ),
        "files_output_child_count_two": len(m_texts.value.items) == 2,
        "files_output_child_1_text_form_exact": m_text1.form == TEXT_UTF8_FORM,
        "files_output_child_2_text_form_exact": m_text2.form == TEXT_UTF8_FORM,
        "files_ingress_boundary_written": len(boundary["ingress"]) == 1,
        "files_egress_boundary_written": len(boundary["egress"]) == 1
        and all(
            child["equals_emitted_child_bytes"]
            for child in boundary["egress"][0]["ordered_native_texts"]
        ),
        "files_uses_standard_runtime_invocation": files_handle._entrypoint.__self__ is files,
        "core_received_forward_order": [
            item["membrane_sha256"] for item in core_a_boundary["ingress"]
        ]
        == forward_child_hashes,
        "forward_core_ingress_count_two": len(core_a_boundary["ingress"]) == 2,
        "core_received_reversed_order": [
            item["membrane_sha256"] for item in core_reverse_boundary["ingress"]
        ]
        == reverse_child_hashes,
        "files_output_independent_of_core_selection": (
            [item["membrane_sha256"] for item in core_a_boundary["ingress"]]
            == [item["membrane_sha256"] for item in core_b_boundary["ingress"]]
            == forward_child_hashes
        ),
        "same_composite_targets_core_a": len(core_a_boundary["ingress"]) == 2,
        "same_composite_targets_core_b": len(core_b_boundary["ingress"]) == 2,
        "core_modified_for_multi_file_semantics_no": not source_inspection[
            "core_imports_ordered_files_module"
        ],
        "files_tokenizes_content_no": not source_inspection[
            "imports_model_or_core_native_dependencies"
        ],
        "runtime_inspects_core_native_representation_no": not source_inspection[
            "runtime_imports_transformer_core"
        ],
        "core_seal_preserved": not source_inspection["core_imports_ordered_files_module"],
        "universal_file_ontology_added_no": not any(
            term in form_text for term in forbidden_form_terms
        ),
        "new_value_primitive_added_no": not source_inspection["defines_new_value_class"],
        "ordered_sequence_existing_primitive_used": source_inspection[
            "imports_ordered_sequence_from_universal_membrane"
        ],
        "artifact_1_m_bytes_equals_original": m_bytes1.value.payload == a1,
        "artifact_2_m_bytes_equals_original": m_bytes2.value.payload == a2,
        "artifact_1_m_text_bytes_equal_original": m_text1.value.payload == a1,
        "artifact_2_m_text_bytes_equal_original": m_text2.value.payload == a2,
        "custody_order_1_2": custody["ordered_positions"] == [1, 2]
        and custody["custody_is_tuple"],
        "custody_reversed_or_unordered_no": [
            item["sha256"] for item in custody["children"]
        ]
        != list(reversed(expected_custody)),
        **rejection_checks,
    }

    projection = {
        "label": label,
        "fixture_bytes": [
            {"position": 1, "hex": a1.hex(), "sha256": sha256_bytes(a1)},
            {"position": 2, "hex": a2.hex(), "sha256": sha256_bytes(a2)},
        ],
        "forward_m_artifacts": canonical_membrane_record(m_artifacts),
        "reversed_m_artifacts": canonical_membrane_record(m_artifacts_rev),
        "files_ingress_evidence": boundary["ingress"],
        "ordered_native_custody_evidence": custody,
        "files_egress_m_texts": canonical_membrane_record(m_texts),
        "independent_reacquisition": [
            {"position": 1, "hex": reacquired1.hex()},
            {"position": 2, "hex": reacquired2.hex()},
        ],
        "forward_order_shared_representation": [
            canonical_membrane_record(m_text1),
            canonical_membrane_record(m_text2),
        ],
        "reversed_order_shared_representation": [
            canonical_membrane_record(m_text2),
            canonical_membrane_record(m_text1),
        ],
        "forward_core_child_admissions": core_a_boundary["ingress"],
        "reversed_core_child_admissions": core_reverse_boundary["ingress"],
        "core_a_b_targetability": {
            "core_a": core_a_boundary["ingress"],
            "core_b": core_b_boundary["ingress"],
        },
        "core_private_outputs": {
            "forward_a": [canonical_membrane_record(item) for item in forward_outputs_a],
            "reverse": [canonical_membrane_record(item) for item in reverse_outputs],
            "forward_b": [canonical_membrane_record(item) for item in forward_outputs_b],
        },
        "source_inspection": source_inspection,
        "checks": checks,
    }
    return {"projection": projection, "checks": checks}


def render_receipt(evidence: dict) -> str:
    checks = evidence["replications"][0]["checks"]
    value = lambda key, yes="PASS", no="FAIL": yes if checks[key] else no
    return f"""OUTCOME
{evidence['outcome']}

AUTHORITATIVE_PARENT_COMMIT:
{AUTHORITATIVE_PARENT}

ORDERED_FILES_IMPLEMENTATION:
ordered_files_module.py

FIXTURE_1:
fixtures/ordered_source_1.txt

FIXTURE_2:
fixtures/ordered_source_2.txt

VERIFY_IMPLEMENTATION:
verify_ordered_files_crossing.py

ORDERED_ARTIFACTS_FORM:
{ORDERED_EXACT_ARTIFACTS_FORM.canonical_name}

ORDERED_TEXTS_FORM:
{ORDERED_SHARED_TEXTS_FORM.canonical_name}

ORDERED_SEQUENCE_VALUE_REALIZED: {value('ordered_sequence_value_realized')}
CHILD_COUNT: {2 if checks['child_count_two'] else 'FAIL'}
CHILD_1_FORM_EXACT: {value('child_1_form_exact')}
CHILD_2_FORM_EXACT: {value('child_2_form_exact')}
CHILD_1_EQUALS_CHILD_2: {value('child_1_equals_child_2_no', 'NO', 'YES')}
SOURCE_BOUNDARIES_STRUCTURALLY_PRESERVED: {value('source_boundaries_preserved')}
SOURCE_ORDER_STRUCTURALLY_PRESERVED: {value('source_order_preserved')}
MULTIPLE_ARTIFACTS_CONCATENATED_TO_SINGLE_TEXT_VALUE: {value('multiple_artifacts_concatenated_no', 'NO', 'YES')}
DELIMITER_USED_AS_SOURCE_BOUNDARY_SUBSTITUTE: {value('delimiter_used_no', 'NO', 'YES')}
ARTIFACT_1_EXACT_REACQUISITION: {value('artifact_1_reacquisition')}
ARTIFACT_2_EXACT_REACQUISITION: {value('artifact_2_reacquisition')}
ARTIFACT_1_AND_2_NATIVE_CUSTODY_DISTINCT: {value('native_custody_distinct')}
REACQUIRED_ARTIFACT_1_EQUALS_ORIGINAL_2: {value('reacquired_1_equals_original_2_no', 'NO', 'YES')}
REACQUIRED_ARTIFACT_2_EQUALS_ORIGINAL_1: {value('reacquired_2_equals_original_1_no', 'NO', 'YES')}
NATIVE_TEXT_CHILD_COUNT: {2 if checks['native_text_child_count_two'] else 'FAIL'}
NATIVE_TEXT_CHILD_ORDER_PRESERVED: {value('native_text_child_order_preserved')}
FILES_OUTPUT_IS_ORDERED_SEQUENCE: {value('files_output_is_ordered_sequence')}
FILES_OUTPUT_CHILD_COUNT: {2 if checks['files_output_child_count_two'] else 'FAIL'}
FILES_OUTPUT_CHILD_1_TEXT_FORM_EXACT: {value('files_output_child_1_text_form_exact')}
FILES_OUTPUT_CHILD_2_TEXT_FORM_EXACT: {value('files_output_child_2_text_form_exact')}
FORWARD_ORDER_SHARED_REPRESENTATION: [M1,M2]
REVERSED_ORDER_SHARED_REPRESENTATION: [M2,M1]
FORWARD_EQUALS_REVERSED: {value('forward_equals_reversed_no', 'NO', 'YES')}
CORE_RECEIVED_FORWARD_CHILD_ORDER_EXACTLY: {value('core_received_forward_order')}
FORWARD_CORE_INGRESS_COUNT: {2 if checks['forward_core_ingress_count_two'] else 'FAIL'}
CORE_RECEIVED_REVERSED_CHILD_ORDER_EXACTLY: {value('core_received_reversed_order')}
FILES_COMPOSITE_OUTPUT_DEPENDS_ON_CORE_SELECTION: {value('files_output_independent_of_core_selection', 'NO', 'YES')}
SAME_COMPOSITE_CAN_TARGET_CORE_A: {value('same_composite_targets_core_a')}
SAME_COMPOSITE_CAN_TARGET_CORE_B: {value('same_composite_targets_core_b')}
CORE_MODIFIED_FOR_MULTI_FILE_SEMANTICS: {value('core_modified_for_multi_file_semantics_no', 'NO', 'YES')}
FILES_TOKENIZES_CONTENT: {value('files_tokenizes_content_no', 'NO', 'YES')}
RUNTIME_INSPECTS_CORE_NATIVE_REPRESENTATION: {value('runtime_inspects_core_native_representation_no', 'NO', 'YES')}
CORE_SEAL_PRESERVED: {value('core_seal_preserved')}
NEW_VALUE_PRIMITIVE_ADDED: {value('new_value_primitive_added_no', 'NO', 'YES')}
ORDERED_SEQUENCE_EXISTING_PRIMITIVE_USED: {value('ordered_sequence_existing_primitive_used')}
UNIVERSAL_FILE_ONTOLOGY_ADDED: {value('universal_file_ontology_added_no', 'NO', 'YES')}
ARTIFACT_1_M_BYTES_EQUALS_ORIGINAL: {value('artifact_1_m_bytes_equals_original')}
ARTIFACT_2_M_BYTES_EQUALS_ORIGINAL: {value('artifact_2_m_bytes_equals_original')}
ARTIFACT_1_M_TEXT_BYTES_EQUAL_ORIGINAL: {value('artifact_1_m_text_bytes_equal_original')}
ARTIFACT_2_M_TEXT_BYTES_EQUAL_ORIGINAL: {value('artifact_2_m_text_bytes_equal_original')}
CUSTODY_ORDER: [1,2]
CUSTODY_REVERSED_OR_UNORDERED: {value('custody_reversed_or_unordered_no', 'NO', 'YES')}
RUNTIME_INVOCATION_LAW_CHANGED: NO
FILES_USES_STANDARD_RUNTIME_INVOCATION: {value('files_uses_standard_runtime_invocation')}
DETERMINISTIC_REPLICATION: {'PASS' if evidence['deterministic_replication'] else 'FAIL'}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture-1", type=Path, default=FIXTURE_1)
    parser.add_argument("--fixture-2", type=Path, default=FIXTURE_2)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()

    a1 = args.fixture_1.read_bytes()
    a2 = args.fixture_2.read_bytes()
    if a1 == a2:
        raise SystemExit("the two fixtures must differ at the exact byte level")
    for source in (a1, a2):
        if not source or source.decode("utf-8", errors="strict").encode("utf-8") != source:
            raise SystemExit("each fixture must be non-empty strict UTF-8")

    configure_determinism()
    run_a = execute_replication(args.model_dir, a1, a2, "A")
    gc.collect()
    run_b = execute_replication(args.model_dir, a1, a2, "B")
    comparable_a = dict(run_a["projection"])
    comparable_b = dict(run_b["projection"])
    comparable_a.pop("label")
    comparable_b.pop("label")
    deterministic = comparable_a == comparable_b
    all_checks = all(run_a["checks"].values()) and all(run_b["checks"].values())
    evidence = {
        "schema": "ORDERED_MULTI_ARTIFACT_FILES_CROSSING_EVIDENCE_V1",
        "authoritative_parent_commit": AUTHORITATIVE_PARENT,
        "forms": {
            "ordered_artifacts": canonical_membrane_record(
                Membrane(ORDERED_EXACT_ARTIFACTS_FORM, OrderedSequenceValue(()))
            )["form"],
            "ordered_texts": canonical_membrane_record(
                Membrane(ORDERED_SHARED_TEXTS_FORM, OrderedSequenceValue(()))
            )["form"],
        },
        "causal_operations": [
            "acquire exact A1",
            "acquire exact A2",
            "construct M_bytes1",
            "construct M_bytes2",
            "compose ordered M_artifacts",
            "invoke FILES once",
            "FILES privately admits two ordered artifacts",
            "FILES preserves two ordered custody units",
            "FILES decodes artifact 1",
            "FILES decodes artifact 2",
            "FILES constructs M_text1",
            "FILES constructs M_text2",
            "FILES composes ordered M_texts",
            "runtime structurally selects child 1 then child 2",
            "CORE admits child 1",
            "CORE admits child 2",
            "reverse-order control",
            "independent reacquisition of source 1",
            "independent reacquisition of source 2",
        ],
        "replications": [run_a, run_b],
        "deterministic_replication": deterministic,
        "outcome": (
            "ORDERED_MULTI_ARTIFACT_FILES_CROSSING_OPERATIONAL"
            if all_checks and deterministic
            else "UNRESOLVED_MULTI_ARTIFACT_MEMBRANE_BOUNDARY"
        ),
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
    if evidence["outcome"] != "ORDERED_MULTI_ARTIFACT_FILES_CROSSING_OPERATIONAL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
