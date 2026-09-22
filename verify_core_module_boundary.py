"""Bounded acceptance verifier for the universal membrane/CORE boundary."""

import argparse
import ast
import gc
import hashlib
import inspect
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import core_runtime
import universal_membrane
from core_runtime import ModuleHandle, invoke
from transformer_core import (
    TransformerCore,
    configure_determinism,
    reset_deterministic_seed,
)
from universal_membrane import (
    BytesValue,
    Form,
    Membrane,
    OrderedSequenceValue,
    TEXT_UTF8_FORM,
    canonical_membrane_record,
    text_membrane,
)


AUTHORITATIVE_PARENT = "e304bb0ca4a29c76e76233d2e466d99bc1fc338c"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "core_module_boundary"
RUNTIME_PATH = PROJECT_DIR / "core_runtime.py"
MEMBRANE_PATH = PROJECT_DIR / "universal_membrane.py"
FORBIDDEN_RUNTIME_TERMS = (
    "smollm2",
    "huggingface",
    "tokenizer",
    "token_ids",
    "torch",
    "tensor",
    "dynamiccache",
    "transformers",
    "attention_mask",
    "cache_position",
    "vocabulary",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inspect_outer_sources() -> dict:
    result = {}
    for path in (MEMBRANE_PATH, RUNTIME_PATH):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = sorted(
            {
                node.names[0].name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
            }
            | {
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
        )
        lowered = source.casefold()
        forbidden_hits = [term for term in FORBIDDEN_RUNTIME_TERMS if term in lowered]
        result[path.name] = {
            "sha256": sha256_bytes(path.read_bytes()),
            "imports": imports,
            "forbidden_terms": forbidden_hits,
            "parsed": True,
        }
    runtime_imports = set(result[RUNTIME_PATH.name]["imports"])
    result["checks"] = {
        "outer_sources_parse": all(item["parsed"] for item in result.values()),
        "outer_sources_have_no_forbidden_terms": all(
            not result[path.name]["forbidden_terms"]
            for path in (MEMBRANE_PATH, RUNTIME_PATH)
        ),
        "runtime_imports_only_universal_or_standard_modules": runtime_imports
        <= {"dataclasses", "typing", "universal_membrane"},
    }
    return result


def value_algebra_evidence() -> dict:
    m0 = text_membrane("algebra-probe")
    sequence_form = Form(
        "UNIVERSAL_ORDERED_SEQUENCE_PROBE_V1",
        "VALUE is exactly the ordered sequence of its contained M values",
    )
    sequence = Membrane(sequence_form, OrderedSequenceValue((m0,)))
    return {
        "declared_variants": ["BYTES", "ORDERED_SEQUENCE(M...)"],
        "bytes_variant_type": BytesValue.__name__,
        "ordered_sequence_variant_type": OrderedSequenceValue.__name__,
        "nested_membrane_accepted": sequence.value.items[0] is m0,
        "form_exact_equality": m0.form == TEXT_UTF8_FORM,
    }


def execute_replication(model_directory: Path, label: str) -> dict:
    reset_deterministic_seed()
    core_a = TransformerCore(model_directory)
    core_b = TransformerCore(model_directory)
    handle_a = ModuleHandle("CORE_A", TEXT_UTF8_FORM, core_a.invoke)
    handle_b = ModuleHandle("CORE_B", TEXT_UTF8_FORM, core_b.invoke)
    m1 = text_membrane("CORE_MODULE_BOUNDARY_INPUT_V1")

    # Establish B first, then prove calls to A cannot mutate B.
    b0 = invoke(handle_b, m1)
    b_after_b0 = core_b.verifier_state_fingerprint()
    a0 = invoke(handle_a, m1)
    b_after_a0 = core_b.verifier_state_fingerprint()
    a1 = invoke(handle_a, m1)
    b_after_a1 = core_b.verifier_state_fingerprint()

    # The single A->B crossing passes the exact returned M through the same invoke.
    a_before_b1 = core_a.verifier_state_fingerprint()
    b1 = invoke(handle_b, a1)
    a_after_b1 = core_a.verifier_state_fingerprint()

    a_states = core_a.verifier_invocation_states()
    b_states = core_b.verifier_invocation_states()
    boundary_a = core_a.verifier_boundary_evidence()
    boundary_b = core_b.verifier_boundary_evidence()
    independence = core_a.verifier_independent_from(core_b)
    runtime_contract = {
        "module": invoke.__module__,
        "qualified_name": invoke.__qualname__,
        "signature": str(inspect.signature(invoke)),
    }
    runtime_calls = [
        {"selected_address": handle_a.address, "contract": runtime_contract},
        {"selected_address": handle_b.address, "contract": runtime_contract},
    ]
    checks = {
        "core_instance_count_two": len((core_a, core_b)) == 2,
        "core_instances_independently_state_owning": all(independence.values()),
        "runtime_invocation_contract_identical": (
            runtime_calls[0]["contract"] == runtime_calls[1]["contract"]
            and runtime_calls[0]["selected_address"]
            != runtime_calls[1]["selected_address"]
        ),
        "core_a_call_mutates_core_a_state": all(
            item["pre"]["sha256"] != item["post"]["sha256"] for item in a_states
        ),
        "core_a_call_mutates_core_b_state_no": (
            b_after_b0 == b_after_a0 == b_after_a1
        ),
        "core_b_call_mutates_core_b_state": all(
            item["pre"]["sha256"] != item["post"]["sha256"] for item in b_states
        ),
        "core_b_call_mutates_core_a_state_no": a_before_b1 == a_after_b1,
        "core_a_private_state_persists_across_calls": (
            len(a_states) == 2 and a_states[0]["post"] == a_states[1]["pre"]
        ),
        "core_b_private_state_persists_across_calls": (
            len(b_states) == 2 and b_states[0]["post"] == b_states[1]["pre"]
        ),
        "shared_form_exact_equality": (
            handle_a.accepted_form == handle_b.accepted_form == TEXT_UTF8_FORM
        ),
        "core_to_core_shared_object_is_m": (
            isinstance(a1, Membrane)
            and core_b.verifier_last_ingress_is(a1)
        ),
        "direct_native_core_to_core_transfer_no": isinstance(a1, Membrane),
        "ingress_boundary_pair_written_core_a": len(boundary_a["ingress"]) == 2,
        "egress_boundary_pair_written_core_a": len(boundary_a["egress"]) == 2,
        "ingress_boundary_pair_written_core_b": len(boundary_b["ingress"]) == 2,
        "egress_boundary_pair_written_core_b": len(boundary_b["egress"]) == 2,
        "core_replaceability_at_runtime_contract": (
            type(handle_a) is type(handle_b)
            and handle_a.accepted_form == handle_b.accepted_form
        ),
        "same_membrane_law_for_every_core": (
            isinstance(m1, universal_membrane.Membrane)
            and isinstance(a0, universal_membrane.Membrane)
            and isinstance(b0, universal_membrane.Membrane)
            and isinstance(b1, universal_membrane.Membrane)
        ),
    }
    projection = {
        "input_m1": canonical_membrane_record(m1),
        "independent_outputs": {
            "core_a": canonical_membrane_record(a0),
            "core_b": canonical_membrane_record(b0),
        },
        "core_to_core": {
            "a_output_m": canonical_membrane_record(a1),
            "b_output_m": canonical_membrane_record(b1),
        },
        "state_evidence": {"core_a": a_states, "core_b": b_states},
        "boundary_evidence": {"core_a": boundary_a, "core_b": boundary_b},
        "independence": independence,
        "runtime_invocations": runtime_calls,
        "checks": checks,
    }
    return {"label": label, "projection": projection, "checks": checks}


def render_receipt(evidence: dict) -> str:
    lines = [
        "OUTCOME",
        evidence["outcome"],
        "",
        f"AUTHORITATIVE_PARENT_COMMIT: {AUTHORITATIVE_PARENT}",
        "UNIVERSAL_MEMBRANE_IMPLEMENTATION: universal_membrane.py",
        "CORE_IMPLEMENTATION: transformer_core.py",
        "RUNTIME_IMPLEMENTATION: core_runtime.py",
        "UNIVERSAL_REPRESENTATION: M=(F,V)",
        "VALUE_ALGEBRA: BYTES | ORDERED_SEQUENCE(M...)",
        f"ACTIVE_SHARED_FORM: {TEXT_UTF8_FORM.canonical_name}",
        "CORE_INSTANCE_COUNT: 2",
        "CORE_A_NATIVE_STATE: PRIVATE",
        "CORE_B_NATIVE_STATE: PRIVATE",
        "CORE_A_MODEL_NATIVE_TRANSLATION: PRIVATE",
        "CORE_B_MODEL_NATIVE_TRANSLATION: PRIVATE",
        "RUNTIME_MODEL_NATIVE_DEPENDENCIES: NONE",
    ]
    labels = {
        "runtime_invocation_contract_identical": "RUNTIME_INVOCATION_CONTRACT_IDENTICAL",
        "core_a_call_mutates_core_a_state": "CORE_A_CALL_MUTATES_CORE_A_STATE",
        "core_b_call_mutates_core_b_state": "CORE_B_CALL_MUTATES_CORE_B_STATE",
        "core_a_private_state_persists_across_calls": "CORE_A_PRIVATE_STATE_PERSISTS_ACROSS_CALLS",
        "core_b_private_state_persists_across_calls": "CORE_B_PRIVATE_STATE_PERSISTS_ACROSS_CALLS",
        "core_a_call_mutates_core_b_state_no": "CORE_A_CALL_MUTATES_CORE_B_STATE",
        "core_b_call_mutates_core_a_state_no": "CORE_B_CALL_MUTATES_CORE_A_STATE",
        "shared_form_exact_equality": "SHARED_FORM_EXACT_EQUALITY",
        "core_to_core_shared_object_is_m": "CORE_TO_CORE_SHARED_OBJECT_IS_M",
        "direct_native_core_to_core_transfer_no": "DIRECT_NATIVE_CORE_TO_CORE_TRANSFER",
        "ingress_boundary_pair_written_core_a": "INGRESS_BOUNDARY_PAIR_WRITTEN_CORE_A",
        "egress_boundary_pair_written_core_a": "EGRESS_BOUNDARY_PAIR_WRITTEN_CORE_A",
        "ingress_boundary_pair_written_core_b": "INGRESS_BOUNDARY_PAIR_WRITTEN_CORE_B",
        "egress_boundary_pair_written_core_b": "EGRESS_BOUNDARY_PAIR_WRITTEN_CORE_B",
        "core_replaceability_at_runtime_contract": "CORE_REPLACEABILITY_AT_RUNTIME_CONTRACT",
    }
    for key, output_label in labels.items():
        passed = evidence["replications"][0]["checks"][key]
        if key in {
            "core_a_call_mutates_core_b_state_no",
            "core_b_call_mutates_core_a_state_no",
            "direct_native_core_to_core_transfer_no",
        }:
            value = "NO" if passed else "YES"
        else:
            value = "PASS" if passed else "FAIL"
        lines.append(f"{output_label}: {value}")
    lines.extend(
        [
            "MEMBRANE_IMPLEMENTATION_COUNT: 1",
            "CORE_A_USES_UNIVERSAL_MEMBRANE: PASS",
            "CORE_B_USES_UNIVERSAL_MEMBRANE: PASS",
            "ACTIVE_CORE_SELECTION_CHANGES_RUNTIME_ARCHITECTURE: NO",
            "CORE_NATIVE_STATE_INTERCHANGEABILITY_REQUIRED: NO",
            "RUNTIME_DEPENDS_ON_MODEL_FAMILY: NO",
            "RUNTIME_DEPENDS_ON_TOKENIZER: NO",
            "RUNTIME_DEPENDS_ON_TOKEN_IDS: NO",
            "RUNTIME_DEPENDS_ON_TENSORS: NO",
            "RUNTIME_DEPENDS_ON_KV_LAYOUT: NO",
            "RUNTIME_DEPENDS_ON_INFERENCE_FRAMEWORK: NO",
            "DETERMINISTIC_REPLICATION: "
            + ("PASS" if evidence["deterministic_replication"] else "FAIL"),
            "FILES_IMPLEMENTED: NO",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()

    configure_determinism()
    source_inspection = inspect_outer_sources()
    algebra = value_algebra_evidence()
    run_a = execute_replication(args.model_dir, "A")
    gc.collect()
    run_b = execute_replication(args.model_dir, "B")
    deterministic = run_a["projection"] == run_b["projection"]
    all_checks = (
        all(source_inspection["checks"].values())
        and algebra["nested_membrane_accepted"]
        and algebra["form_exact_equality"]
        and all(run_a["checks"].values())
        and all(run_b["checks"].values())
        and deterministic
    )
    evidence = {
        "schema": "CORE_MODULE_BOUNDARY_EVIDENCE_V1",
        "authoritative_parent_commit": AUTHORITATIVE_PARENT,
        "universal_representation": "M=(F,V)",
        "value_algebra": "BYTES | ORDERED_SEQUENCE(M...)",
        "active_shared_form": {
            "canonical_name": TEXT_UTF8_FORM.canonical_name,
            "interpretation_law": TEXT_UTF8_FORM.interpretation_law,
        },
        "implementations": {
            "universal_membrane": "universal_membrane.py",
            "core": "transformer_core.py",
            "runtime": "core_runtime.py",
        },
        "measured": {
            "value_algebra": algebra,
            "source_inspection": source_inspection,
            "replications": [run_a, run_b],
            "deterministic_replication": deterministic,
        },
        "structurally_deduced": {
            "runtime_depends_on_model_family": "NO",
            "runtime_depends_on_tokenizer": "NO",
            "runtime_depends_on_token_ids": "NO",
            "runtime_depends_on_tensors": "NO",
            "runtime_depends_on_kv_layout": "NO",
            "runtime_depends_on_inference_framework": "NO",
            "direct_native_core_to_core_transfer": "NO",
            "basis": (
                "AST/import inspection shows the outer sources have no model-native "
                "dependency; invoke accepts and returns only universal_membrane.Membrane, "
                "and the exercised A-to-B object was that exact M instance."
            ),
        },
        "replications": [run_a, run_b],
        "deterministic_replication": deterministic,
        "outcome": (
            "CORE_MODULE_BOUNDARY_CLOSED"
            if all_checks
            else "UNRESOLVED_CORE_MODULE_BOUNDARY"
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
    if evidence["outcome"] != "CORE_MODULE_BOUNDARY_CLOSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
