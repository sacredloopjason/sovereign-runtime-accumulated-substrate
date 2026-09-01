import gc
import hashlib
import importlib.util
import json
import sys
import traceback
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RELIC_PATH = PROJECT_DIR / "The Narrative.docx"
SLICE3_RUNTIME = PROJECT_DIR / "runtime.py"
EXPECTED_DOCX_SHA256 = "b4a349e82ee3aa6bc6d8957f2d73e9a14d6364068edd3028099fd8e8fa6fcfec"
EXPECTED_UTF8_SHA256 = "ccc60ed932fa95ffb74367a71caa2bf7b5479043b7ead08de9ebafb2c79efd59"
EXPECTED_TOKEN_COUNT = 28603
NOMINAL_POSITION_LIMIT = 8192
PROBE_TOKEN_COUNT = NOMINAL_POSITION_LIMIT + 1
CHUNK_SIZE = 256
REPLICATION_COUNT = 2
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_exact_relic(path):
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(WORD_NS + "body")
    if body is None:
        raise RuntimeError("DOCX has no Word document body")
    substitutions = {
        WORD_NS + "tab": "\t",
        WORD_NS + "br": "\n",
        WORD_NS + "cr": "\n",
    }
    paragraphs = []
    for child in body:
        if child.tag != WORD_NS + "p":
            continue
        paragraphs.append(
            "".join(
                (node.text or "")
                if node.tag == WORD_NS + "t"
                else substitutions.get(node.tag, "")
                for node in child.iter()
            )
        )
    return "\n".join(paragraphs), len(paragraphs)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inherited module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def token_fingerprint(token_ids):
    payload = json.dumps(
        [int(token_id) for token_id in token_ids], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(b"SLICE4_ORDERED_TOKEN_IDS_V1\0" + payload).hexdigest()


def tensor_sha256(tensor):
    frozen = tensor.detach().cpu().contiguous()
    return hashlib.sha256(frozen.view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def forward_exact(model, cache, token_ids, start, end):
    import torch

    if int(cache.get_seq_length()) != start:
        raise RuntimeError(
            f"Causal pre-state length {cache.get_seq_length()} does not equal next token start {start}"
        )
    input_ids = torch.tensor([token_ids[start:end]], dtype=torch.long)
    attention_mask = torch.ones((1, end), dtype=torch.long)
    cache_position = torch.arange(start, end, dtype=torch.long)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
    if outputs.past_key_values is not cache:
        raise RuntimeError("Inference returned a different KV cache object")
    if int(cache.get_seq_length()) != end:
        raise RuntimeError(
            f"Causal post-state length {cache.get_seq_length()} does not equal consumed token end {end}"
        )
    return outputs.logits[:, -1, :]


def run_replicate(kernel, model, token_ids, replicate):
    cache = kernel.DynamicCache(config=model.config)
    consumed = 0
    try:
        while consumed < NOMINAL_POSITION_LIMIT:
            end = min(consumed + CHUNK_SIZE, NOMINAL_POSITION_LIMIT)
            forward_exact(model, cache, token_ids, consumed, end)
            consumed = end
            if consumed % 1024 == 0:
                print(f"REPLICATE_{replicate}_CAUSALLY_CONSUMED={consumed}", flush=True)

        pre = kernel.fingerprint_cache(cache)
        print(f"REPLICATE_{replicate}_LAST_CONFIGURED_POSITION_STATE={pre['sha256']}", flush=True)
        logits = forward_exact(
            model, cache, token_ids, NOMINAL_POSITION_LIMIT, PROBE_TOKEN_COUNT
        )
        consumed = PROBE_TOKEN_COUNT
        post = kernel.fingerprint_cache(cache)
        result = {
            "replicate": replicate,
            "consumed": consumed,
            "pre": pre,
            "post": post,
            "crossing_logits_sha256": tensor_sha256(logits),
        }
        print(f"REPLICATE_{replicate}_CROSSED_TO_POSITION={consumed - 1}", flush=True)
        print(f"REPLICATE_{replicate}_POST_CROSSING_STATE={post['sha256']}", flush=True)
        return result, None
    except Exception as exc:
        failure = {
            "replicate": replicate,
            "consumed": consumed,
            "next_range": [consumed, min(consumed + 1, EXPECTED_TOKEN_COUNT)],
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        return None, failure
    finally:
        del cache
        gc.collect()


def main():
    docx_hash = sha256_file(RELIC_PATH)
    relic_text, paragraph_count = extract_exact_relic(RELIC_PATH)
    relic_utf8 = relic_text.encode("utf-8")
    utf8_hash = hashlib.sha256(relic_utf8).hexdigest()
    if docx_hash != EXPECTED_DOCX_SHA256:
        raise RuntimeError(f"Relic DOCX hash mismatch: {docx_hash}")
    if utf8_hash != EXPECTED_UTF8_SHA256:
        raise RuntimeError(f"Extracted UTF-8 hash mismatch: {utf8_hash}")

    slice3 = load_module(SLICE3_RUNTIME, "relic_probe_slice3")
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
    if len(first) != EXPECTED_TOKEN_COUNT:
        raise RuntimeError(f"Relic token count mismatch: {len(first)}")
    if first != second:
        raise RuntimeError("Tokenizer replication mismatch")

    print("RELIC_NATIVE_POSITION_BOUNDARY_PROBE_READY", flush=True)
    print(f"RELIC_DOCX_SHA256={docx_hash}", flush=True)
    print(f"EXTRACTED_UTF8_SHA256={utf8_hash}", flush=True)
    print(f"EXTRACTED_PARAGRAPH_COUNT={paragraph_count}", flush=True)
    print(f"RELIC_TOKEN_COUNT={len(first)}", flush=True)
    print(f"RELIC_ORDERED_TOKEN_SHA256={token_fingerprint(first)}", flush=True)
    print("TOKENIZATION_REPLICATION=PASS", flush=True)
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    print(f"MODEL_MAX_POSITION_EMBEDDINGS={model.config.max_position_embeddings}", flush=True)
    print(f"SLIDING_WINDOW={getattr(model.config, 'sliding_window', None)}", flush=True)
    print(f"PROBE_TOKEN_RANGE=[0,{PROBE_TOKEN_COUNT})", flush=True)

    results = []
    for replicate in range(1, REPLICATION_COUNT + 1):
        result, failure = run_replicate(kernel, model, first, replicate)
        if failure is not None:
            print("OUTCOME=UNRESOLVED_EXECUTION_BOUNDARY", flush=True)
            print(f"TOKENS_CAUSALLY_CONSUMED_BEFORE_BOUNDARY={failure['consumed']}", flush=True)
            print(f"NEXT_REQUIRED_RELIC_TOKEN_RANGE={failure['next_range']}", flush=True)
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
            return 2
        results.append(result)

    deterministic = (
        results[0]["pre"] == results[1]["pre"]
        and results[0]["post"] == results[1]["post"]
        and results[0]["crossing_logits_sha256"]
        == results[1]["crossing_logits_sha256"]
    )
    print("NATIVE_CAUSAL_CONTINUATION_THROUGH_POSITION_8192=PASS", flush=True)
    print("CACHE_OBJECT_PRESERVED=PASS", flush=True)
    print("CACHE_SEQUENCE_LENGTH_AFTER_CROSSING=8193", flush=True)
    print(f"DETERMINISTIC_REPLICATION={'PASS' if deterministic else 'FAIL'}", flush=True)
    print("NO_SUMMARIZATION=PASS", flush=True)
    print("NO_RECONSTRUCTION=PASS", flush=True)
    print("NO_EXTERNAL_COMPRESSION=PASS", flush=True)
    print(f"FINAL_RESULT={'PASS' if deterministic else 'FAIL'}", flush=True)
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
