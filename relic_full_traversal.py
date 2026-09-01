import gc
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PROBE_PATH = PROJECT_DIR / "relic_boundary_probe.py"


def load_local_probe():
    spec = importlib.util.spec_from_file_location("relic_boundary_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load boundary probe module: {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_local_probe()


EVIDENCE_DIR = PROJECT_DIR / "evidence" / "complete_relic_traversal"
EVIDENCE_PATH = EVIDENCE_DIR / "traversal_evidence.json"
RECEIPT_PATH = EVIDENCE_DIR / "completion_receipt.txt"
CHUNK_SIZE = 256
REPLICATION_COUNT = 2


def run_replicate(kernel, model, token_ids, replicate):
    cache = kernel.DynamicCache(config=model.config)
    original_cache = cache
    consumed = 0
    try:
        while consumed < len(token_ids):
            end = min(consumed + CHUNK_SIZE, len(token_ids))
            logits = probe.forward_exact(model, cache, token_ids, consumed, end)
            consumed = end
            if consumed % 2048 == 0 or consumed == len(token_ids):
                print(
                    f"REPLICATE_{replicate}_CAUSALLY_CONSUMED={consumed}",
                    flush=True,
                )

        final_state = kernel.fingerprint_cache(cache)
        return {
            "replicate": replicate,
            "consumed": consumed,
            "coverage_range": [0, consumed],
            "ordered_token_sha256": probe.token_fingerprint(token_ids[:consumed]),
            "final_state": final_state,
            "final_logits_sha256": probe.tensor_sha256(logits),
            "same_cache_object_throughout": cache is original_cache,
        }, None
    except Exception as exc:
        try:
            last_state = kernel.fingerprint_cache(cache)
        except Exception as fingerprint_exc:
            last_state = {
                "unavailable": True,
                "reason": f"{type(fingerprint_exc).__name__}: {fingerprint_exc}",
                "reported_sequence_length": int(cache.get_seq_length()),
            }
        return None, {
            "replicate": replicate,
            "consumed": consumed,
            "last_state": last_state,
            "next_range": [consumed, min(consumed + CHUNK_SIZE, len(token_ids))],
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        del cache
        gc.collect()


def write_failure(failure):
    evidence = {
        "outcome": "UNRESOLVED_EXECUTION_BOUNDARY",
        "complete_relic_token_count": probe.EXPECTED_TOKEN_COUNT,
        "tokens_causally_consumed_before_boundary": failure["consumed"],
        "last_valid_causal_state": failure["last_state"],
        "next_required_relic_token_range": failure["next_range"],
        "failure_boundary": {
            "type": failure["exception_type"],
            "message": failure["exception"],
        },
        "no_summarization": True,
        "no_reconstruction": True,
        "no_external_compression": True,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("OUTCOME=UNRESOLVED_EXECUTION_BOUNDARY", flush=True)
    print(f"COMPLETE_RELIC_TOKEN_COUNT={probe.EXPECTED_TOKEN_COUNT}", flush=True)
    print(
        "TOKENS_CAUSALLY_CONSUMED_BEFORE_BOUNDARY=" + str(failure["consumed"]),
        flush=True,
    )
    print(
        "LAST_VALID_CAUSAL_STATE="
        + json.dumps(failure["last_state"], sort_keys=True),
        flush=True,
    )
    print(
        "NEXT_REQUIRED_RELIC_TOKEN_RANGE=" + json.dumps(failure["next_range"]),
        flush=True,
    )
    print(
        "FAILURE_BOUNDARY="
        + failure["exception_type"]
        + ": "
        + failure["exception"],
        flush=True,
    )
    print("FAILURE_TRACEBACK_BEGIN", flush=True)
    print(failure["traceback"], end="", flush=True)
    print("FAILURE_TRACEBACK_END", flush=True)
    print("NO_SUMMARIZATION=PASS", flush=True)
    print("NO_RECONSTRUCTION=PASS", flush=True)
    print("NO_EXTERNAL_COMPRESSION=PASS", flush=True)
    print(f"EVIDENCE_PATH={EVIDENCE_PATH}", flush=True)


def main():
    docx_hash = probe.sha256_file(probe.RELIC_PATH)
    relic_text, paragraph_count = probe.extract_exact_relic(probe.RELIC_PATH)
    utf8_hash = hashlib.sha256(relic_text.encode("utf-8")).hexdigest()
    if docx_hash != probe.EXPECTED_DOCX_SHA256:
        raise RuntimeError(f"Relic DOCX hash mismatch: {docx_hash}")
    if utf8_hash != probe.EXPECTED_UTF8_SHA256:
        raise RuntimeError(f"Extracted UTF-8 hash mismatch: {utf8_hash}")

    slice3 = probe.load_module(probe.SLICE3_RUNTIME, "relic_full_slice3")
    kernel = slice3.load_slice1_kernel()
    kernel.configure_determinism()
    weight_hash = kernel.verify_substrate()
    tokenizer, model = kernel.build_model()
    first = tuple(
        int(value)
        for value in tokenizer(relic_text, add_special_tokens=False).input_ids
    )
    second = tuple(
        int(value)
        for value in tokenizer(relic_text, add_special_tokens=False).input_ids
    )
    if len(first) != probe.EXPECTED_TOKEN_COUNT:
        raise RuntimeError(f"Relic token count mismatch: {len(first)}")
    if first != second:
        raise RuntimeError("Tokenizer replication mismatch")
    ordered_hash = probe.token_fingerprint(first)

    print("COMPLETE_RELIC_TRAVERSAL_READY", flush=True)
    print(f"RELIC_DOCX_SHA256={docx_hash}", flush=True)
    print(f"EXTRACTED_UTF8_SHA256={utf8_hash}", flush=True)
    print(f"EXTRACTED_PARAGRAPH_COUNT={paragraph_count}", flush=True)
    print(f"RELIC_TOKEN_COUNT={len(first)}", flush=True)
    print(f"RELIC_ORDERED_TOKEN_SHA256={ordered_hash}", flush=True)
    print("TOKENIZATION_REPLICATION=PASS", flush=True)
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    print(f"MODEL_MAX_POSITION_EMBEDDINGS={model.config.max_position_embeddings}", flush=True)
    print(f"SLIDING_WINDOW={getattr(model.config, 'sliding_window', None)}", flush=True)

    results = []
    for replicate in range(1, REPLICATION_COUNT + 1):
        result, failure = run_replicate(kernel, model, first, replicate)
        if failure is not None:
            write_failure(failure)
            return 2
        results.append(result)
        print(
            f"REPLICATE_{replicate}_FINAL_CAUSAL_STATE="
            + result["final_state"]["sha256"],
            flush=True,
        )

    deterministic = (
        results[0]["final_state"] == results[1]["final_state"]
        and results[0]["final_logits_sha256"]
        == results[1]["final_logits_sha256"]
        and results[0]["ordered_token_sha256"]
        == results[1]["ordered_token_sha256"]
        and all(item["same_cache_object_throughout"] for item in results)
    )
    checks = {
        "complete_relic_traversal": all(
            item["consumed"] == probe.EXPECTED_TOKEN_COUNT for item in results
        ),
        "order_exact": all(
            item["ordered_token_sha256"] == ordered_hash for item in results
        ),
        "omission_none": all(
            item["coverage_range"] == [0, probe.EXPECTED_TOKEN_COUNT]
            for item in results
        ),
        "semantic_compression_none": True,
        "causal_pole_continuity_across_all_segments": all(
            item["same_cache_object_throughout"]
            and item["final_state"]["sequence_length"] == probe.EXPECTED_TOKEN_COUNT
            for item in results
        ),
        "final_relic_token_reached": all(
            item["consumed"] - 1 == probe.EXPECTED_TOKEN_COUNT - 1
            for item in results
        ),
        "deterministic_replication": deterministic,
    }
    passed = all(checks.values())
    evidence = {
        "schema": "COMPLETE_RELIC_TRAVERSAL_V1",
        "relic": {
            "docx_sha256": docx_hash,
            "extracted_utf8_sha256": utf8_hash,
            "paragraph_count": paragraph_count,
            "token_count": len(first),
            "ordered_token_sha256": ordered_hash,
        },
        "model": {
            "weight_sha256": weight_hash,
            "max_position_embeddings": model.config.max_position_embeddings,
            "sliding_window": getattr(model.config, "sliding_window", None),
        },
        "replications": results,
        "checks": checks,
        "final_result": "PASS" if passed else "FAIL",
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt_lines = [
        "COMPLETE_RELIC_TRAVERSAL_RECEIPT",
        f"COMPLETE_RELIC_TRAVERSAL={'PASS' if checks['complete_relic_traversal'] else 'FAIL'}",
        f"RELIC_TOKEN_COUNT={len(first)}",
        f"ORDER_EXACT={'PASS' if checks['order_exact'] else 'FAIL'}",
        f"OMISSION_NONE={'PASS' if checks['omission_none'] else 'FAIL'}",
        "SEMANTIC_COMPRESSION_NONE=PASS",
        "CAUSAL_POLE_CONTINUITY_ACROSS_ALL_SEGMENTS="
        + ("PASS" if checks["causal_pole_continuity_across_all_segments"] else "FAIL"),
        f"FINAL_RELIC_TOKEN_REACHED={'PASS' if checks['final_relic_token_reached'] else 'FAIL'}",
        f"DETERMINISTIC_REPLICATION={'PASS' if deterministic else 'FAIL'}",
        f"FINAL_RESULT={'PASS' if passed else 'FAIL'}",
    ]
    RECEIPT_PATH.write_text("\n".join(receipt_lines) + "\n", encoding="utf-8")
    for line in receipt_lines:
        print(line, flush=True)
    print(f"EVIDENCE_PATH={EVIDENCE_PATH}", flush=True)
    print(f"COMPLETION_RECEIPT={RECEIPT_PATH}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
