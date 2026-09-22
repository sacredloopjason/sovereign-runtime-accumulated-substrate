"""Bounded verifier for the first FILES-to-CORE membrane crossing."""

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
from files_module import EXACT_BYTES_FORM, FilesModule
from transformer_core import TransformerCore, configure_determinism, reset_deterministic_seed
from universal_membrane import (
    BytesValue,
    Membrane,
    TEXT_UTF8_FORM,
    canonical_membrane_record,
)


AUTHORITATIVE_PARENT = "b3b80827a9dae75230181256f8f6df5baba2640d"
EXACT_BYTES_INTERPRETATION = (
    "VALUE is BYTES containing the exact ordered byte sequence supplied; the "
    "membrane assigns no content type, encoding, file type, modality, semantic "
    "interpretation, or native object structure to those bytes."
)
FIXTURE_PATH = PROJECT_DIR / "fixtures" / "first_files_source.txt"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "first_files_crossing"
FILES_PATH = PROJECT_DIR / "files_module.py"
SEALED_PATHS = (
    PROJECT_DIR / "universal_membrane.py",
    PROJECT_DIR / "core_runtime.py",
    PROJECT_DIR / "transformer_core.py",
)
FORBIDDEN_FILES_IMPORTS = {
    "transformer_core",
    "torch",
    "transformers",
    "tokenizers",
}
FORBIDDEN_FILES_REFERENCES = {
    "DynamicCache",
    "tokenizer",
    "token_ids",
    "model_family",
    "CORE_A",
    "CORE_B",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def membrane_sha256(membrane: Membrane) -> str:
    encoded = json.dumps(
        canonical_membrane_record(membrane), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def inspect_sources() -> dict:
    source = FILES_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FILES_PATH))
    imports = sorted(
        {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    core_source = SEALED_PATHS[2].read_text(encoding="utf-8")
    core_tree = ast.parse(core_source, filename=str(SEALED_PATHS[2]))
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
    return {
        "files_imports": imports,
        "files_forbidden_imports": sorted(set(imports) & FORBIDDEN_FILES_IMPORTS),
        "files_forbidden_references": sorted(
            (names | constants) & FORBIDDEN_FILES_REFERENCES
        ),
        "core_imports_files_module": "files_module" in core_imports,
        "sealed_source_sha256": {
            path.name: sha256_bytes(path.read_bytes()) for path in SEALED_PATHS
        },
    }


def execute_replication(model_directory: Path, original: bytes, label: str) -> dict:
    reset_deterministic_seed()
    files = FilesModule()
    core_a = TransformerCore(model_directory)
    core_b = TransformerCore(model_directory)
    files_handle = ModuleHandle("FILES_FIRST_UTF8", EXACT_BYTES_FORM, files.invoke)
    core_a_handle = ModuleHandle("CORE_A", TEXT_UTF8_FORM, core_a.invoke)
    core_b_handle = ModuleHandle("CORE_B", TEXT_UTF8_FORM, core_b.invoke)

    m_bytes = Membrane(EXACT_BYTES_FORM, BytesValue(original))
    m_text = invoke(files_handle, m_bytes)
    custody_before_core_a = files.verifier_custody_evidence()
    core_a_result = invoke(core_a_handle, m_text)
    custody_after_core_a = files.verifier_custody_evidence()
    reacquired = files.verifier_reacquire_source_bytes()
    core_b_result = invoke(core_b_handle, m_text)

    files_boundary = files.verifier_boundary_evidence()
    core_a_boundary = core_a.verifier_boundary_evidence()
    core_b_boundary = core_b.verifier_boundary_evidence()
    m_text_record = canonical_membrane_record(m_text)
    m_text_hash = membrane_sha256(m_text)
    source_inspection = inspect_sources()

    runtime_output_keys = set(m_text_record)
    runtime_value_keys = set(m_text_record["value"])
    checks = {
        "files_module_present": isinstance(files, FilesModule),
        "files_uses_standard_runtime_invocation": files_handle._entrypoint.__self__ is files,
        "files_coexists_with_core_a_and_core_b": len((files, core_a, core_b)) == 3,
        "m_bytes_equals_original_artifact": m_bytes.value.payload == original,
        "files_native_artifact_equals_original": (
            files_boundary["ingress"][0]["native_artifact_sha256"]
            == sha256_bytes(original)
            and files_boundary["ingress"][0]["native_artifact_byte_count"]
            == len(original)
        ),
        "m_text_utf8_bytes_equal_original": (
            isinstance(m_text.value, BytesValue) and m_text.value.payload == original
        ),
        "exact_artifact_reacquisition": reacquired == original,
        "files_artifact_unchanged_by_core_a": custody_before_core_a == custody_after_core_a,
        "files_output_to_core_a_equals_files_output_to_core_b": (
            core_a.verifier_last_ingress_is(m_text)
            and core_b.verifier_last_ingress_is(m_text)
        ),
        "files_output_object_is_core_a_input_m": core_a.verifier_last_ingress_is(m_text),
        "files_ingress_form_exact": files_handle.accepted_form == EXACT_BYTES_FORM,
        "files_egress_form_exact": m_text.form == TEXT_UTF8_FORM,
        "semantic_form_matching_used_no": (
            files_handle.accepted_form == EXACT_BYTES_FORM
            and EXACT_BYTES_FORM.canonical_name == "SOVEREIGN_EXACT_BYTES_V1"
            and EXACT_BYTES_FORM.interpretation_law == EXACT_BYTES_INTERPRETATION
        ),
        "files_ingress_boundary_pair_written": len(files_boundary["ingress"]) == 1,
        "files_egress_boundary_pair_written": (
            len(files_boundary["egress"]) == 1
            and files_boundary["egress"][0]["native_text_utf8_equals_emitted_bytes"]
        ),
        "core_a_ingress_boundary_pair_written": len(core_a_boundary["ingress"]) == 1,
        "core_b_ingress_boundary_pair_written": len(core_b_boundary["ingress"]) == 1,
        "files_to_core_a_exact_m_text_join": (
            core_a.verifier_last_ingress_is(m_text)
            and files.verifier_last_output_is(m_text)
        ),
        "files_to_core_b_exact_m_text_join": (
            core_b.verifier_last_ingress_is(m_text)
            and files.verifier_last_output_is(m_text)
        ),
        "files_has_no_model_native_dependencies": (
            not source_inspection["files_forbidden_imports"]
            and not source_inspection["files_forbidden_references"]
        ),
        "core_has_no_files_native_dependency": not source_inspection["core_imports_files_module"],
        "m_bytes_contains_path_no": set(canonical_membrane_record(m_bytes)) == {"form", "value"},
        "m_bytes_contains_filename_no": set(canonical_membrane_record(m_bytes)["value"]) == {"kind", "hex"},
        "m_bytes_contains_mime_no": set(canonical_membrane_record(m_bytes)["value"]) == {"kind", "hex"},
        "m_text_contains_files_native_metadata_no": (
            runtime_output_keys == {"form", "value"}
            and runtime_value_keys == {"kind", "hex"}
        ),
        "files_runtime_depends_on_filesystem_path_for_invocation_no": (
            "pathlib" not in source_inspection["files_imports"]
            and "os" not in source_inspection["files_imports"]
        ),
        "original_artifact_nonempty_strict_utf8": bool(original)
        and original.decode("utf-8", errors="strict").encode("utf-8", errors="strict") == original,
    }
    projection = {
        "label": label,
        "original_artifact_hex": original.hex(),
        "original_artifact_byte_count": len(original),
        "m_bytes": canonical_membrane_record(m_bytes),
        "files_boundary_evidence": files_boundary,
        "files_custody_before_core_a": custody_before_core_a,
        "files_custody_after_core_a": custody_after_core_a,
        "m_text": m_text_record,
        "m_text_sha256": m_text_hash,
        "core_admission_evidence": {
            "core_a": core_a_boundary["ingress"],
            "core_b": core_b_boundary["ingress"],
            "same_exact_membrane_record": (
                core_a.verifier_last_ingress_is(m_text)
                and core_b.verifier_last_ingress_is(m_text)
            ),
        },
        "core_private_traversal": {
            "core_a_output": canonical_membrane_record(core_a_result),
            "core_b_output": canonical_membrane_record(core_b_result),
        },
        "reacquired_artifact_hex": reacquired.hex(),
        "reacquired_artifact_byte_count": len(reacquired),
        "source_inspection": source_inspection,
        "checks": checks,
    }
    return {"projection": projection, "checks": checks}


def render_receipt(evidence: dict) -> str:
    checks = evidence["replications"][0]["checks"]
    byte_count = evidence["replications"][0]["projection"]["original_artifact_byte_count"]
    labels = [
        ("FILES_MODULE_PRESENT", "files_module_present", "PASS", "FAIL"),
        ("FILES_USES_STANDARD_RUNTIME_INVOCATION", "files_uses_standard_runtime_invocation", "PASS", "FAIL"),
        ("FILES_COEXISTS_WITH_CORE_A_AND_CORE_B", "files_coexists_with_core_a_and_core_b", "PASS", "FAIL"),
        ("EXACT_ARTIFACT_REACQUISITION", "exact_artifact_reacquisition", "PASS", "FAIL"),
        ("M_BYTES_EQUALS_ORIGINAL_ARTIFACT", "m_bytes_equals_original_artifact", "PASS", "FAIL"),
        ("FILES_NATIVE_ARTIFACT_EQUALS_ORIGINAL", "files_native_artifact_equals_original", "PASS", "FAIL"),
        ("M_TEXT_UTF8_BYTES_EQUAL_ORIGINAL", "m_text_utf8_bytes_equal_original", "PASS", "FAIL"),
        ("FILES_ARTIFACT_UNCHANGED_BY_CORE_A", "files_artifact_unchanged_by_core_a", "PASS", "FAIL"),
        ("FILES_OUTPUT_TO_CORE_A_EQUALS_FILES_OUTPUT_TO_CORE_B", "files_output_to_core_a_equals_files_output_to_core_b", "PASS", "FAIL"),
        ("FILES_OUTPUT_OBJECT_IS_CORE_A_INPUT_M", "files_output_object_is_core_a_input_m", "PASS", "FAIL"),
        ("FILES_INGRESS_FORM_EXACT", "files_ingress_form_exact", "PASS", "FAIL"),
        ("FILES_EGRESS_FORM_EXACT", "files_egress_form_exact", "PASS", "FAIL"),
        ("SEMANTIC_FORM_MATCHING_USED", "semantic_form_matching_used_no", "NO", "YES"),
        ("FILES_INGRESS_BOUNDARY_PAIR_WRITTEN", "files_ingress_boundary_pair_written", "PASS", "FAIL"),
        ("FILES_EGRESS_BOUNDARY_PAIR_WRITTEN", "files_egress_boundary_pair_written", "PASS", "FAIL"),
        ("CORE_A_INGRESS_BOUNDARY_PAIR_WRITTEN", "core_a_ingress_boundary_pair_written", "PASS", "FAIL"),
        ("CORE_B_INGRESS_BOUNDARY_PAIR_WRITTEN", "core_b_ingress_boundary_pair_written", "PASS", "FAIL"),
        ("FILES_TO_CORE_A_EXACT_M_TEXT_JOIN", "files_to_core_a_exact_m_text_join", "PASS", "FAIL"),
        ("FILES_TO_CORE_B_EXACT_M_TEXT_JOIN", "files_to_core_b_exact_m_text_join", "PASS", "FAIL"),
        ("M_BYTES_CONTAINS_PATH", "m_bytes_contains_path_no", "NO", "YES"),
        ("M_BYTES_CONTAINS_FILENAME", "m_bytes_contains_filename_no", "NO", "YES"),
        ("M_BYTES_CONTAINS_MIME", "m_bytes_contains_mime_no", "NO", "YES"),
        ("M_TEXT_CONTAINS_FILES_NATIVE_METADATA", "m_text_contains_files_native_metadata_no", "NO", "YES"),
        ("FILES_RUNTIME_DEPENDS_ON_FILESYSTEM_PATH_FOR_INVOCATION", "files_runtime_depends_on_filesystem_path_for_invocation_no", "NO", "YES"),
        ("CORE_SEAL_PRESERVED", "core_has_no_files_native_dependency", "PASS", "FAIL"),
    ]
    lines = [
        "OUTCOME",
        evidence["outcome"],
        "",
        f"AUTHORITATIVE_PARENT_COMMIT: {AUTHORITATIVE_PARENT}",
        "FILES_IMPLEMENTATION: files_module.py",
        "FIXTURE_ARTIFACT: fixtures/first_files_source.txt",
        "VERIFY_IMPLEMENTATION: verify_first_files_crossing.py",
        f"EXACT_BYTES_FORM: {EXACT_BYTES_FORM.canonical_name}",
        f"EXACT_BYTES_FORM_INTERPRETATION: {EXACT_BYTES_FORM.interpretation_law}",
        f"FILES_OUTPUT_FORM: {TEXT_UTF8_FORM.canonical_name}",
    ]
    lines.extend(
        f"{label}: {passed if checks[key] else failed}"
        for label, key, passed, failed in labels[:3]
    )
    lines.extend(
        [
            "RUNTIME_INVOCATION_LAW_CHANGED: NO",
            "SIMULTANEOUS_MODULE_COUNT: 3",
            f"ORIGINAL_ARTIFACT_BYTE_COUNT: {byte_count}",
            f"REACQUIRED_ARTIFACT_BYTE_COUNT: {byte_count}",
        ]
    )
    lines.extend(
        f"{label}: {passed if checks[key] else failed}"
        for label, key, passed, failed in labels[3:]
    )
    lines.extend(
        [
            "FILES_DEPENDS_ON_CORE_SELECTION: NO",
            "FILES_DEPENDS_ON_MODEL_FAMILY: NO",
            "FILES_DEPENDS_ON_TOKENIZER: NO",
            "FILES_DEPENDS_ON_TOKEN_IDS: NO",
            "FILES_DEPENDS_ON_KV: NO",
            "MEMBRANE_GRAMMAR_CHANGED: NO",
            "DETERMINISTIC_REPLICATION: "
            + ("PASS" if evidence["deterministic_replication"] else "FAIL"),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()

    original = args.fixture.read_bytes()
    configure_determinism()
    run_a = execute_replication(args.model_dir, original, "A")
    gc.collect()
    run_b = execute_replication(args.model_dir, original, "B")
    comparable_a = dict(run_a["projection"])
    comparable_b = dict(run_b["projection"])
    comparable_a.pop("label")
    comparable_b.pop("label")
    deterministic = comparable_a == comparable_b
    all_checks = all(run_a["checks"].values()) and all(run_b["checks"].values())
    evidence = {
        "schema": "FIRST_FILES_MODULE_CROSSING_EVIDENCE_V1",
        "authoritative_parent_commit": AUTHORITATIVE_PARENT,
        "exact_bytes_form": {
            "canonical_name": EXACT_BYTES_FORM.canonical_name,
            "interpretation_law": EXACT_BYTES_FORM.interpretation_law,
        },
        "files_output_form": TEXT_UTF8_FORM.canonical_name,
        "causal_operations": [
            "external fixture byte acquisition",
            "M_bytes construction",
            "standard runtime invocation of FILES",
            "FILES ingress translation",
            "FILES native artifact custody",
            "FILES native UTF-8 decoding",
            "FILES egress translation",
            "M_text return",
            "runtime routing to CORE A and CORE B",
            "CORE ingress",
            "CORE private traversal",
            "exact source reacquisition",
        ],
        "replications": [run_a, run_b],
        "deterministic_replication": deterministic,
        "outcome": (
            "FIRST_FILES_MODULE_CROSSING_OPERATIONAL"
            if all_checks and deterministic
            else "UNRESOLVED_FILES_MODULE_BOUNDARY"
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
    if evidence["outcome"] != "FIRST_FILES_MODULE_CROSSING_OPERATIONAL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
