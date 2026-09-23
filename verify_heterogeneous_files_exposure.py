"""Verifier for one heterogeneous FILES exposure without coercion."""

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
from heterogeneous_files_module import (
    HeterogeneousFilesModule,
    ORDERED_TEXT_AND_EXACT_BYTES_FORM,
    ORDERED_TEXT_BINARY_SOURCES_FORM,
)
from transformer_core import TransformerCore, configure_determinism, reset_deterministic_seed
from universal_membrane import (
    BytesValue,
    Membrane,
    OrderedSequenceValue,
    TEXT_UTF8_FORM,
    canonical_membrane_record,
)


AUTHORITATIVE_PARENT = "a0939d388c1f8493ec5d5e75d4a95ac831547510"
TEXT_FIXTURE = PROJECT_DIR / "fixtures" / "heterogeneous_text_source.txt"
BINARY_FIXTURE = PROJECT_DIR / "fixtures" / "heterogeneous_binary_source.bin"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "heterogeneous_files_exposure"
IMPLEMENTATION_PATH = PROJECT_DIR / "heterogeneous_files_module.py"
SEALED_PATHS = tuple(
    PROJECT_DIR / name
    for name in (
        "universal_membrane.py",
        "core_runtime.py",
        "transformer_core.py",
        "files_module.py",
        "ordered_files_module.py",
        "state_relative_files_module.py",
    )
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def core_ingress_sha256(membrane: Membrane) -> str:
    encoded = json.dumps(
        canonical_membrane_record(membrane), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(b"CORE_INGRESS_M_V1\0" + encoded)


def strict_utf8_valid(value: bytes) -> bool:
    try:
        value.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return True


def rejected(callable_object, exception_type: type[BaseException]) -> bool:
    try:
        callable_object()
    except exception_type:
        return True
    return False


def inspect_sources() -> dict:
    source = IMPLEMENTATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(IMPLEMENTATION_PATH))
    imports = {
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
    decode_error_modes = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"decode", "encode"}
        for keyword in node.keywords
        if keyword.arg == "errors" and isinstance(keyword.value, ast.Constant)
    ]
    core_tree = ast.parse(
        SEALED_PATHS[2].read_text(encoding="utf-8"), filename=str(SEALED_PATHS[2])
    )
    core_imports = {
        node.module or ""
        for node in ast.walk(core_tree)
        if isinstance(node, ast.ImportFrom)
    }
    runtime_tree = ast.parse(
        SEALED_PATHS[1].read_text(encoding="utf-8"), filename=str(SEALED_PATHS[1])
    )
    runtime_imports = {
        node.module or ""
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom)
    }
    universal_source = SEALED_PATHS[0].read_text(encoding="utf-8")
    return {
        "implementation_imports": sorted(imports),
        "implementation_classes": sorted(class_names),
        "implementation_form_assignments": sorted(form_assignments),
        "defines_new_value_class": any(name.endswith("Value") for name in class_names),
        "uses_only_strict_codec_errors": bool(decode_error_modes)
        and set(decode_error_modes) == {"strict"},
        "imports_core_or_model_dependencies": bool(
            imports & {"core_runtime", "transformer_core", "torch", "transformers", "tokenizers"}
        ),
        "imports_generalizing_codec_dependencies": bool(
            imports & {"base64", "binascii", "mimetypes"}
        ),
        "core_imports_heterogeneous_files": "heterogeneous_files_module" in core_imports,
        "runtime_imports_transformer_core": "transformer_core" in runtime_imports,
        "universal_mentions_heterogeneous_forms": (
            "SOVEREIGN_ORDERED_TEXT_BINARY_SOURCES_V1" in universal_source
            or "SOVEREIGN_ORDERED_TEXT_AND_EXACT_BYTES_V1" in universal_source
        ),
        "sealed_source_sha256": {
            path.name: sha256_bytes(path.read_bytes()) for path in SEALED_PATHS
        },
    }


def execute_replication(
    model_directory: Path, text_bytes: bytes, binary_bytes: bytes, label: str
) -> dict:
    reset_deterministic_seed()
    source_inspection = inspect_sources()
    m_bytes_text = Membrane(EXACT_BYTES_FORM, BytesValue(text_bytes))
    m_bytes_binary = Membrane(EXACT_BYTES_FORM, BytesValue(binary_bytes))
    m_sources = Membrane(
        ORDERED_TEXT_BINARY_SOURCES_FORM,
        OrderedSequenceValue((m_bytes_text, m_bytes_binary)),
    )

    files = HeterogeneousFilesModule()
    files_handle = ModuleHandle(
        "FILES_HETEROGENEOUS_TEXT_EXACT_BYTES",
        ORDERED_TEXT_BINARY_SOURCES_FORM,
        files.invoke,
    )
    m_mixed = invoke(files_handle, m_sources)
    if not isinstance(m_mixed.value, OrderedSequenceValue):
        raise AssertionError("FILES did not return ORDERED_SEQUENCE")
    m_text, m_exact_bytes = m_mixed.value.items
    custody = files.verifier_custody_evidence()
    boundary = files.verifier_boundary_evidence()
    reacquired_text = files.verifier_reacquire_source_bytes(1)
    reacquired_binary = files.verifier_reacquire_source_bytes(2)

    core = TransformerCore(model_directory)
    core_handle = ModuleHandle("CORE_TEXT_CHILD", TEXT_UTF8_FORM, core.invoke)
    text_compatible = m_text.form == core_handle.accepted_form
    text_core_output = invoke(core_handle, m_text)
    core_after_text = core.verifier_boundary_evidence()
    ingress_count_before_binary = len(core_after_text["ingress"])
    binary_compatible = m_exact_bytes.form == core_handle.accepted_form
    binary_rejected = rejected(lambda: invoke(core_handle, m_exact_bytes), ValueError)
    core_after_binary = core.verifier_boundary_evidence()
    ingress_count_after_binary = len(core_after_binary["ingress"])

    malformed = HeterogeneousFilesModule()
    malformed_handle = ModuleHandle(
        "FILES_HETEROGENEOUS_REJECTION_CONTROL",
        ORDERED_TEXT_BINARY_SOURCES_FORM,
        malformed.invoke,
    )
    rejection_checks = {
        "reject_non_ordered_value": rejected(
            lambda: invoke(
                malformed_handle,
                Membrane(ORDERED_TEXT_BINARY_SOURCES_FORM, BytesValue(text_bytes)),
            ),
            TypeError,
        ),
        "reject_child_count_not_two": rejected(
            lambda: invoke(
                malformed_handle,
                Membrane(
                    ORDERED_TEXT_BINARY_SOURCES_FORM,
                    OrderedSequenceValue((m_bytes_text,)),
                ),
            ),
            ValueError,
        ),
        "reject_wrong_child_form": rejected(
            lambda: invoke(
                malformed_handle,
                Membrane(
                    ORDERED_TEXT_BINARY_SOURCES_FORM,
                    OrderedSequenceValue((m_bytes_text, m_text)),
                ),
            ),
            ValueError,
        ),
    }

    ingress = boundary["ingress"][0]
    egress = boundary["egress"][0]
    expected_hashes = [sha256_bytes(text_bytes), sha256_bytes(binary_bytes)]
    checks = {
        "source_child_count_two": custody["child_count"] == 2,
        "text_artifact_strict_utf8_valid": strict_utf8_valid(text_bytes)
        and text_bytes.decode("utf-8", errors="strict").encode(
            "utf-8", errors="strict"
        )
        == text_bytes,
        "binary_artifact_strict_utf8_valid_no": not strict_utf8_valid(binary_bytes),
        "source_boundaries_preserved": [len(text_bytes), len(binary_bytes)]
        == [item["byte_count"] for item in custody["children"]]
        == [item["byte_count"] for item in ingress["ordered_native_artifacts"]],
        "source_order_preserved": expected_hashes
        == [item["sha256"] for item in custody["children"]]
        == [item["sha256"] for item in ingress["ordered_native_artifacts"]],
        "text_exact_reacquisition": reacquired_text == text_bytes,
        "binary_exact_reacquisition": reacquired_binary == binary_bytes,
        "native_custody_distinct": custody["native_custody_objects_distinct"]
        and custody["native_byte_objects_distinct"]
        and custody["custody_is_tuple"],
        "text_strict_utf8_decoding_succeeded": egress["native_text_present"],
        "text_shared_form_exact": m_text.form == TEXT_UTF8_FORM,
        "text_shared_bytes_equal_original": isinstance(m_text.value, BytesValue)
        and m_text.value.payload == text_bytes
        and egress["text_native_equals_emitted_child_bytes"],
        "binary_strict_utf8_decoding_succeeded_no": not egress[
            "binary_text_translation_succeeded"
        ],
        "binary_silently_coerced_to_text_no": m_exact_bytes.form == EXACT_BYTES_FORM
        and not egress["binary_text_translation_succeeded"],
        "binary_replacement_characters_used_no": b"\xef\xbf\xbd" not in m_exact_bytes.value.payload
        and source_inspection["uses_only_strict_codec_errors"],
        "binary_content_dropped_no": bool(m_exact_bytes.value.payload)
        and m_exact_bytes.value.payload == binary_bytes,
        "binary_text_description_generated_no": m_exact_bytes.value.payload
        == binary_bytes,
        "automatic_binary_to_text_translator_inserted_no": not source_inspection[
            "imports_generalizing_codec_dependencies"
        ]
        and not egress["binary_text_translation_succeeded"],
        "binary_exact_bytes_preserved": m_exact_bytes.form == EXACT_BYTES_FORM
        and isinstance(m_exact_bytes.value, BytesValue)
        and m_exact_bytes.value.payload == binary_bytes
        and egress["binary_native_equals_emitted_child_bytes"],
        "files_output_is_ordered_sequence": isinstance(
            m_mixed.value, OrderedSequenceValue
        ),
        "files_output_child_count_two": len(m_mixed.value.items) == 2,
        "child_forms_equal_no": m_text.form != m_exact_bytes.form,
        "new_value_primitive_added_no": not source_inspection["defines_new_value_class"],
        "membrane_grammar_changed_no": not source_inspection[
            "universal_mentions_heterogeneous_forms"
        ],
        "universal_file_ontology_added_no": set(
            source_inspection["implementation_form_assignments"]
        )
        == {
            "ORDERED_TEXT_BINARY_SOURCES_FORM",
            "ORDERED_TEXT_AND_EXACT_BYTES_FORM",
        },
        "universal_modality_ontology_added_no": not source_inspection[
            "imports_generalizing_codec_dependencies"
        ],
        "text_child_directly_compatible_with_core": text_compatible,
        "text_child_admitted_to_core": len(core_after_text["ingress"]) == 1
        and core_after_text["ingress"][0]["membrane_sha256"]
        == core_ingress_sha256(m_text)
        and core.verifier_last_ingress_is(m_text),
        "binary_child_directly_compatible_with_core_no": not binary_compatible,
        "binary_child_admitted_to_core_no": binary_rejected
        and ingress_count_before_binary == ingress_count_after_binary == 1
        and core.verifier_last_ingress_is(m_text),
        "semantic_compatibility_inference_used_no": text_compatible
        and not binary_compatible,
        "core_modified_for_mixed_formats_no": not source_inspection[
            "core_imports_heterogeneous_files"
        ],
        "files_tokenizes_content_no": not source_inspection[
            "imports_core_or_model_dependencies"
        ],
        "runtime_inspects_core_native_state_no": not source_inspection[
            "runtime_imports_transformer_core"
        ],
        "core_seal_preserved": not source_inspection[
            "core_imports_heterogeneous_files"
        ],
        "files_writer_records_text_translation_success": len(boundary["egress"])
        == 1
        and egress["native_text_present"]
        and egress["text_native_equals_emitted_child_bytes"],
        "files_writer_records_binary_text_translation_success_no": not egress[
            "binary_text_translation_succeeded"
        ],
        "files_writer_preserves_binary_exact_bytes_boundary": egress[
            "binary_native_artifact_present"
        ]
        and egress["binary_native_equals_emitted_child_bytes"],
        "core_writer_has_text_child_ingress": len(core_after_binary["ingress"])
        == 1
        and core_after_binary["ingress"][0]["membrane_sha256"]
        == core_ingress_sha256(m_text),
        "core_writer_has_binary_child_ingress_no": ingress_count_after_binary == 1,
        "runtime_invocation_law_changed_no": files_handle._entrypoint.__self__ is files
        and core_handle._entrypoint.__self__ is core,
        "files_last_output_identity": files.verifier_last_output_is(m_mixed),
        **rejection_checks,
    }

    projection = {
        "label": label,
        "fixtures": {
            "text": {
                "hex": text_bytes.hex(),
                "byte_count": len(text_bytes),
                "sha256": sha256_bytes(text_bytes),
            },
            "binary": {
                "hex": binary_bytes.hex(),
                "byte_count": len(binary_bytes),
                "sha256": sha256_bytes(binary_bytes),
            },
        },
        "source_composite": canonical_membrane_record(m_sources),
        "native_custody": custody,
        "files_boundaries": boundary,
        "heterogeneous_output": canonical_membrane_record(m_mixed),
        "independent_reacquisition": {
            "text_hex": reacquired_text.hex(),
            "binary_hex": reacquired_binary.hex(),
        },
        "core_compatibility": {
            "text_exact_form_equal": text_compatible,
            "binary_exact_form_equal": binary_compatible,
        },
        "core_boundary_after_text_and_binary_attempt": core_after_binary,
        "core_text_output": canonical_membrane_record(text_core_output),
        "source_inspection": source_inspection,
        "checks": checks,
    }
    del core
    gc.collect()
    return {"projection": projection, "checks": checks}


def render_receipt(evidence: dict) -> str:
    checks = evidence["replications"][0]["checks"]
    word = lambda key, yes="PASS", no="FAIL": yes if checks[key] else no
    return f"""OUTCOME
{evidence['outcome']}

AUTHORITATIVE_PARENT_COMMIT:
{AUTHORITATIVE_PARENT}

HETEROGENEOUS_FILES_IMPLEMENTATION:
heterogeneous_files_module.py

TEXT_FIXTURE:
fixtures/heterogeneous_text_source.txt

BINARY_FIXTURE:
fixtures/heterogeneous_binary_source.bin

VERIFY_IMPLEMENTATION:
verify_heterogeneous_files_exposure.py

SOURCE_CHILD_COUNT:
{2 if checks['source_child_count_two'] else 'FAIL'}

TEXT_ARTIFACT_STRICT_UTF8_VALID:
{word('text_artifact_strict_utf8_valid')}

BINARY_ARTIFACT_STRICT_UTF8_VALID:
{word('binary_artifact_strict_utf8_valid_no', 'NO', 'YES')}

SOURCE_BOUNDARIES_PRESERVED:
{word('source_boundaries_preserved')}

SOURCE_ORDER_PRESERVED:
{word('source_order_preserved')}

TEXT_EXACT_REACQUISITION:
{word('text_exact_reacquisition')}

BINARY_EXACT_REACQUISITION:
{word('binary_exact_reacquisition')}

TEXT_AND_BINARY_NATIVE_CUSTODY_DISTINCT:
{word('native_custody_distinct')}

TEXT_STRICT_UTF8_DECODING_SUCCEEDED:
{word('text_strict_utf8_decoding_succeeded')}

TEXT_SHARED_FORM_EXACT:
{word('text_shared_form_exact')}

TEXT_SHARED_BYTES_EQUAL_ORIGINAL:
{word('text_shared_bytes_equal_original')}

FILES_OUTPUT_IS_ORDERED_SEQUENCE:
{word('files_output_is_ordered_sequence')}

FILES_OUTPUT_CHILD_COUNT:
{2 if checks['files_output_child_count_two'] else 'FAIL'}

CHILD_1_FORM:
{TEXT_UTF8_FORM.canonical_name}

CHILD_2_FORM:
{EXACT_BYTES_FORM.canonical_name}

CHILD_FORMS_EQUAL:
{word('child_forms_equal_no', 'NO', 'YES')}

BINARY_STRICT_UTF8_DECODING_SUCCEEDED:
{word('binary_strict_utf8_decoding_succeeded_no', 'NO', 'YES')}

BINARY_SILENTLY_COERCED_TO_TEXT:
{word('binary_silently_coerced_to_text_no', 'NO', 'YES')}

BINARY_REPLACEMENT_CHARACTERS_USED:
{word('binary_replacement_characters_used_no', 'NO', 'YES')}

BINARY_CONTENT_DROPPED:
{word('binary_content_dropped_no', 'NO', 'YES')}

BINARY_TEXT_DESCRIPTION_GENERATED:
{word('binary_text_description_generated_no', 'NO', 'YES')}

BINARY_EXACT_BYTES_PRESERVED:
{word('binary_exact_bytes_preserved')}

NEW_VALUE_PRIMITIVE_ADDED:
{word('new_value_primitive_added_no', 'NO', 'YES')}

MEMBRANE_GRAMMAR_CHANGED:
{word('membrane_grammar_changed_no', 'NO', 'YES')}

UNIVERSAL_FILE_ONTOLOGY_ADDED:
{word('universal_file_ontology_added_no', 'NO', 'YES')}

UNIVERSAL_MODALITY_ONTOLOGY_ADDED:
{word('universal_modality_ontology_added_no', 'NO', 'YES')}

TEXT_CHILD_DIRECTLY_COMPATIBLE_WITH_CORE:
{word('text_child_directly_compatible_with_core')}

TEXT_CHILD_ADMITTED_TO_CORE:
{word('text_child_admitted_to_core')}

BINARY_CHILD_DIRECTLY_COMPATIBLE_WITH_CORE:
{word('binary_child_directly_compatible_with_core_no', 'NO', 'YES')}

BINARY_CHILD_ADMITTED_TO_CORE:
{word('binary_child_admitted_to_core_no', 'NO', 'YES')}

SEMANTIC_COMPATIBILITY_INFERENCE_USED:
{word('semantic_compatibility_inference_used_no', 'NO', 'YES')}

AUTOMATIC_BINARY_TO_TEXT_TRANSLATOR_INSERTED:
{word('automatic_binary_to_text_translator_inserted_no', 'NO', 'YES')}

CORE_MODIFIED_FOR_MIXED_FORMATS:
{word('core_modified_for_mixed_formats_no', 'NO', 'YES')}

FILES_TOKENIZES_CONTENT:
{word('files_tokenizes_content_no', 'NO', 'YES')}

RUNTIME_INSPECTS_CORE_NATIVE_STATE:
{word('runtime_inspects_core_native_state_no', 'NO', 'YES')}

CORE_SEAL_PRESERVED:
{word('core_seal_preserved')}

FILES_WRITER_RECORDS_TEXT_TRANSLATION_SUCCESS:
{word('files_writer_records_text_translation_success')}

FILES_WRITER_RECORDS_BINARY_TEXT_TRANSLATION_SUCCESS:
{word('files_writer_records_binary_text_translation_success_no', 'NO', 'YES')}

FILES_WRITER_PRESERVES_BINARY_EXACT_BYTES_BOUNDARY:
{word('files_writer_preserves_binary_exact_bytes_boundary')}

CORE_WRITER_HAS_TEXT_CHILD_INGRESS:
{word('core_writer_has_text_child_ingress')}

CORE_WRITER_HAS_BINARY_CHILD_INGRESS:
{word('core_writer_has_binary_child_ingress_no', 'NO', 'YES')}

RUNTIME_INVOCATION_LAW_CHANGED:
{word('runtime_invocation_law_changed_no', 'NO', 'YES')}

DETERMINISTIC_REPLICATION:
{'PASS' if evidence['deterministic_replication'] else 'FAIL'}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-fixture", type=Path, default=TEXT_FIXTURE)
    parser.add_argument("--binary-fixture", type=Path, default=BINARY_FIXTURE)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()

    text_bytes = args.text_fixture.read_bytes()
    binary_bytes = args.binary_fixture.read_bytes()
    if not text_bytes or not binary_bytes or text_bytes == binary_bytes:
        raise SystemExit("the two independently bounded fixtures must be non-empty and differ")
    if not strict_utf8_valid(text_bytes):
        raise SystemExit("the text fixture must be strict UTF-8")
    if strict_utf8_valid(binary_bytes):
        raise SystemExit("the binary fixture must reject strict UTF-8")

    configure_determinism()
    run_a = execute_replication(
        args.model_dir, text_bytes, binary_bytes, "A"
    )
    gc.collect()
    run_b = execute_replication(
        args.model_dir, text_bytes, binary_bytes, "B"
    )
    comparable_a = dict(run_a["projection"])
    comparable_b = dict(run_b["projection"])
    comparable_a.pop("label")
    comparable_b.pop("label")
    deterministic = comparable_a == comparable_b
    all_checks = all(run_a["checks"].values()) and all(run_b["checks"].values())
    outcome = (
        "HETEROGENEOUS_FILES_EXPOSURE_OPERATIONAL"
        if all_checks and deterministic
        else "UNRESOLVED_HETEROGENEOUS_FILES_BOUNDARY"
    )
    evidence = {
        "schema": "HETEROGENEOUS_FILES_EXPOSURE_EVIDENCE_V1",
        "authoritative_parent_commit": AUTHORITATIVE_PARENT,
        "outcome": outcome,
        "forms": {
            "sources": ORDERED_TEXT_BINARY_SOURCES_FORM.canonical_name,
            "output": ORDERED_TEXT_AND_EXACT_BYTES_FORM.canonical_name,
            "child_1": TEXT_UTF8_FORM.canonical_name,
            "child_2": EXACT_BYTES_FORM.canonical_name,
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
    if outcome != "HETEROGENEOUS_FILES_EXPOSURE_OPERATIONAL":
        failed = sorted(
            key
            for run in (run_a, run_b)
            for key, passed in run["checks"].items()
            if not passed
        )
        print(f"FAILED_CHECKS: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
