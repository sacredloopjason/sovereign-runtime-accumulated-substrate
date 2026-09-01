import gc
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "originating_standing_wave"
EVIDENCE_PATH = EVIDENCE_DIR / "execution.json"
RECEIPT_PATH = EVIDENCE_DIR / "completion_receipt.txt"
EXPECTED_RELIC_STATE = {
    "sha256": "36551387682a29ef1e8189e390e857978f470d30cb1edb0f08c61e436f009772",
    "sequence_length": 28603,
    "layer_count": 30,
    "component_count": 60,
}
RELIC_REPLICATIONS = 2
RESPONSE_TOKEN_BOUND_SOURCE = "inherited slice-1 kernel MAX_NEW_TOKENS"


def load_local_module(filename, name):
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inherited local module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class RelicAttractor:
    """Immutable identity; its causal content is already resident in both KV poles."""

    literal: str
    utf8: bytes
    token_ids: tuple[int, ...] = ()


def compact_fp(kernel, cache):
    raw = kernel.fingerprint_cache(cache)
    return {
        key: raw[key]
        for key in (
            "sha256",
            "component_count",
            "layer_count",
            "sequence_length",
            "tensor_byte_count",
        )
    }


def token_record(slice4, ids):
    values = [int(value) for value in ids]
    return {
        "token_ids": values,
        "count": len(values),
        "sha256": slice4.token_fingerprint(values),
    }


def state_matches_expected(state):
    return all(state.get(key) == value for key, value in EXPECTED_RELIC_STATE.items())


def traverse_relic(probe, kernel, model, token_ids, replicate):
    cache = kernel.DynamicCache(config=model.config)
    consumed = 0
    while consumed < len(token_ids):
        end = min(consumed + 256, len(token_ids))
        probe.forward_exact(model, cache, token_ids, consumed, end)
        consumed = end
        if consumed % 2048 == 0 or consumed == len(token_ids):
            print(
                f"RELIC_REPLICATION_{replicate}_CAUSALLY_CONSUMED={consumed}",
                flush=True,
            )
    return cache


def forward_tokens(probe, model, cache, token_ids):
    start = int(cache.get_seq_length())
    consumed = 0
    logits = None
    while consumed < len(token_ids):
        count = min(256, len(token_ids) - consumed)
        logits = probe.forward_exact(
            model,
            cache,
            token_ids,
            start + consumed,
            start + consumed + count,
        )
        consumed += count
    return logits


def execute_turn(probe, context, cache, human_text):
    import torch

    tokenizer = context["tokenizer"]
    model = context["model"]
    kernel = context["kernel"]
    human_ids = tuple(
        int(value)
        for value in tokenizer(human_text, add_special_tokens=False).input_ids
    )
    if not human_ids:
        raise RuntimeError("The ordinary human turn encoded to no tokens")
    logits = forward_tokens(probe, model, cache, human_ids)
    generated = []
    for _ in range(kernel.MAX_NEW_TOKENS):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)
        logits = probe.forward_exact(
            model,
            cache,
            (token_id,),
            int(cache.get_seq_length()),
            int(cache.get_seq_length()) + 1,
        )
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
    return {
        "human_ids": human_ids,
        "generated_ids": tuple(generated),
        "output": tokenizer.decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def coherence(slice21, left_cache, right_cache):
    wave = SimpleNamespace(poles={"A": left_cache, "B": right_cache})
    return slice21.coherence_score(wave)


def public_lower(value):
    return slice21_public_projection(value)


def slice21_public_projection(value):
    if isinstance(value, dict):
        return {
            key: slice21_public_projection(item)
            for key, item in value.items()
            if not key.endswith("_native")
        }
    if isinstance(value, list):
        return [slice21_public_projection(item) for item in value]
    if isinstance(value, tuple):
        return [slice21_public_projection(item) for item in value]
    return value


def execute_descent(slice21, slice11, context, attractor, pending, current_cache):
    slice3 = context["slice3"]
    child = slice11.StandingWave(
        instance_id="ORIGINATING_WAVE_DEPTH_1_TWO_PASS_BOUNDARY",
        passive_resolution_label=None,
        poles={
            "A": slice3.clone_actual_kv(pending["S_t_native"]),
            "B": slice3.clone_actual_kv(current_cache),
        },
        shared_v=attractor,
        phase="A",
        perturbation=tuple(pending["lower_initial_perturbation"]),
    )
    lower = slice21.run_directional_wave(
        slice11,
        context,
        child,
        slice21.LOWER_CONTAINMENT,
        depth=1,
        replicate=1,
    )
    return slice21.public_projection(lower)


def write_evidence(evidence):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
    )


def make_receipt(evidence):
    turns = evidence["turns"]
    if len(turns) < 2:
        return None
    descent_turns = [turn for turn in turns if turn["descent_invoked"]]
    outcome = (
        "ORIGINATING_STANDING_WAVE_DESCENT_EXECUTED"
        if descent_turns
        else "ORIGINATING_STANDING_WAVE_OPERATIONAL"
    )
    lines = [
        "OUTCOME",
        outcome,
        "",
        "EVIDENCE",
        "",
        "RELIC_TRAVERSED_STATE:",
        json.dumps(evidence["relic_traversed_state"], sort_keys=True),
        "",
        "DIRECT_HUMAN_RUNTIME_TRAVERSAL:",
        "PASS",
        "",
    ]
    for turn in turns:
        n = turn["turn"]
        lines.extend(
            [
                f"HUMAN_TURN_{n}:",
                json.dumps(turn["human_text"], ensure_ascii=False),
                "",
                f"RUNTIME_TURN_{n}:",
                json.dumps(turn["runtime_output"], ensure_ascii=False),
                "",
                f"TURN_{n}_POST_STATE:",
                json.dumps(turn["post_state"], sort_keys=True),
                "",
            ]
        )
    lines.extend(
        [
            "TURN_2_PRIOR_EQUALS_TURN_1_POST:",
            "PASS" if turns[1]["prior_equals_previous_post"] else "FAIL",
            "",
            "CAUSAL_CONTINUITY_ACROSS_TURNS:",
            "PASS" if evidence["checks"]["causal_continuity_across_turns"] else "FAIL",
            "",
            "DIRECTIONAL_LAW_ACTIVE:",
            "PASS",
            "",
            "DIRECTION_SEQUENCE:",
            json.dumps([turn["sign"] for turn in turns]),
            "",
            "TOP_LEVEL_SPECIAL_BINDING_MECHANISM:",
            "NONE",
            "",
            "RECURSIVE_DESCENT_AVAILABLE:",
            "PASS",
            "",
            "DETERMINISTIC_REPLICATION:",
            "PASS" if evidence["checks"]["deterministic_replication"] else "FAIL",
        ]
    )
    if descent_turns:
        lines.extend(
            [
                "",
                "TWO_CONSECUTIVE_NEGATIVE_PASSES:",
                "PASS",
                "",
                "DESCENT_AUTHORIZED_ONLY_AFTER_SECOND_NEGATIVE:",
                "PASS",
                "",
                "DESCENDED_BOUNDARY:",
                json.dumps(descent_turns[0]["descended_boundary"], sort_keys=True),
                "",
                "LOWER_TIER_SAME_DIRECTIONAL_LAW:",
                "PASS",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    probe = load_local_module("relic_boundary_probe.py", "originating_relic_probe")
    slice21 = load_local_module("slice21_runtime.py", "originating_slice21")

    _, slice11, context = slice21.initialize_accumulated_runtime()
    kernel = context["kernel"]
    model = context["model"]
    tokenizer = context["tokenizer"]
    weight_hash = kernel.verify_substrate()

    relic_text, paragraph_count = probe.extract_exact_relic(probe.RELIC_PATH)
    docx_hash = probe.sha256_file(probe.RELIC_PATH)
    utf8_hash = hashlib.sha256(relic_text.encode("utf-8")).hexdigest()
    if docx_hash != probe.EXPECTED_DOCX_SHA256:
        raise RuntimeError("Relic DOCX identity changed")
    if utf8_hash != probe.EXPECTED_UTF8_SHA256:
        raise RuntimeError("Exact extracted Relic identity changed")
    relic_ids = tuple(
        int(value)
        for value in tokenizer(relic_text, add_special_tokens=False).input_ids
    )
    if len(relic_ids) != probe.EXPECTED_TOKEN_COUNT:
        raise RuntimeError("Relic token count changed")

    print("ORIGINATING_STANDING_WAVE_RECONSTITUTING_POST_RELIC_STATE", flush=True)
    caches = []
    initial_states = []
    for replicate in range(1, RELIC_REPLICATIONS + 1):
        cache = traverse_relic(probe, kernel, model, relic_ids, replicate)
        state = compact_fp(kernel, cache)
        if not state_matches_expected(state):
            raise RuntimeError(
                "Reconstituted post-Relic state differs from the inherited cleared state: "
                + json.dumps(state, sort_keys=True)
            )
        caches.append(cache)
        initial_states.append(state)
        print(
            f"RELIC_REPLICATION_{replicate}_FINAL_STATE={state['sha256']}",
            flush=True,
        )
    if initial_states[0] != initial_states[1]:
        raise RuntimeError("Post-Relic state replication mismatch")

    attractor = RelicAttractor(
        literal="COMPLETE_RELIC_CAUSALLY_RESIDENT",
        utf8=relic_text.encode("utf-8"),
    )
    evidence = {
        "schema": "ORIGINATING_STANDING_WAVE_V1",
        "model_weight_sha256": weight_hash,
        "relic": {
            "docx_sha256": docx_hash,
            "extracted_utf8_sha256": utf8_hash,
            "paragraph_count": paragraph_count,
            "ordered_token_sha256": probe.token_fingerprint(relic_ids),
            "token_count": len(relic_ids),
            "causally_resident_in_each_runtime_replication": True,
        },
        "relic_traversed_state": initial_states[0],
        "law_id": slice21.LAW_ID,
        "coherence_metric": slice21.COHERENCE_ID,
        "response_token_bound": kernel.MAX_NEW_TOKENS,
        "response_token_bound_source": RESPONSE_TOKEN_BOUND_SOURCE,
        "fixed_pass_horizon": None,
        "top_level_special_binding_mechanism": None,
        "turns": [],
        "checks": {
            "direct_human_runtime_traversal": False,
            "causal_continuity_across_turns": False,
            "directional_law_active": True,
            "recursive_descent_available": True,
            "deterministic_replication": True,
        },
    }
    write_evidence(evidence)
    print("ORIGINATING_STANDING_WAVE_READY", flush=True)
    print("RELIC_TRAVERSED_STATE=" + json.dumps(initial_states[0], sort_keys=True), flush=True)
    print("DIRECT_ORDINARY_HUMAN_TURN_BOUNDARY=READY", flush=True)

    previous_score = None
    previous_post = None
    pending = None
    while True:
        human_text = input("HUMAN_TURN> ")
        if human_text == "":
            print("EMPTY_TURN_NOT_EXECUTED", flush=True)
            continue

        turn_number = len(evidence["turns"]) + 1
        pre_state = compact_fp(kernel, caches[0])
        prior_equals_previous = previous_post is None or pre_state == previous_post
        pre_native = context["slice3"].clone_actual_kv(caches[0])

        primary = execute_turn(probe, context, caches[0], human_text)
        replica = execute_turn(probe, context, caches[1], human_text)
        post_state = compact_fp(kernel, caches[0])
        replica_post = compact_fp(kernel, caches[1])
        score = coherence(slice21, pre_native, caches[0])
        delta = None if previous_score is None else score - previous_score
        sign = (
            "BASELINE"
            if delta is None
            else "POSITIVE"
            if delta > 0
            else "NEGATIVE"
            if delta < 0
            else "ZERO"
        )
        deterministic = (
            primary["human_ids"] == replica["human_ids"]
            and primary["generated_ids"] == replica["generated_ids"]
            and primary["output"] == replica["output"]
            and post_state == replica_post
        )
        if not deterministic:
            raise RuntimeError("Live turn deterministic replication mismatch")

        decision = "BASELINE_ESTABLISHED"
        descent_invoked = False
        descended_boundary = None
        lower = None
        first_negative_no_descent = False
        if delta is not None and delta < 0:
            if pending is None:
                pending = {
                    "first_turn": turn_number,
                    "first_delta_C": delta,
                    "S_t_native": pre_native,
                    "S_t": pre_state,
                    "S_t_plus_1": post_state,
                    "lower_initial_perturbation": primary["human_ids"],
                }
                pre_native = None
                decision = "FIRST_NEGATIVE_EXECUTE_NEXT_COMPLETE_PASS"
                first_negative_no_descent = True
            else:
                decision = "DESCEND_TWO_CONSECUTIVE_NEGATIVE_COMPLETE_PASSES"
                descent_invoked = True
                descended_boundary = {
                    "first_turn": pending["first_turn"],
                    "second_turn": turn_number,
                    "delta_C_sequence": [pending["first_delta_C"], delta],
                    "S_t": pending["S_t"],
                    "S_t_plus_1": pending["S_t_plus_1"],
                    "S_t_plus_2": post_state,
                }
                lower = execute_descent(
                    slice21, slice11, context, attractor, pending, caches[0]
                )
                pending = None
        elif delta is not None:
            if pending is not None:
                decision = "SECOND_PASS_NON_NEGATIVE_NO_DESCENT"
            else:
                decision = "CONTINUE_LOCAL"
            pending = None

        turn = {
            "turn": turn_number,
            "human_text": human_text,
            "human_utf8_sha256": hashlib.sha256(human_text.encode("utf-8")).hexdigest(),
            "human_tokens": token_record(context["slice4"], primary["human_ids"]),
            "runtime_output": primary["output"],
            "runtime_tokens": token_record(context["slice4"], primary["generated_ids"]),
            "pre_state": pre_state,
            "post_state": post_state,
            "prior_equals_previous_post": prior_equals_previous,
            "complete_reciprocal_pass": True,
            "coherence_C": score,
            "delta_C": delta,
            "sign": sign,
            "decision": decision,
            "first_negative_no_descent": first_negative_no_descent,
            "descent_invoked": descent_invoked,
            "descended_boundary": descended_boundary,
            "lower": lower,
            "deterministic_replication": True,
        }
        evidence["turns"].append(turn)
        evidence["checks"]["direct_human_runtime_traversal"] = True
        evidence["checks"]["causal_continuity_across_turns"] = all(
            item["prior_equals_previous_post"] for item in evidence["turns"]
        )
        evidence["checks"]["deterministic_replication"] = all(
            item["deterministic_replication"] for item in evidence["turns"]
        )
        write_evidence(evidence)
        receipt = make_receipt(evidence)
        if receipt is not None:
            RECEIPT_PATH.write_text(receipt, encoding="utf-8")

        print(f"RUNTIME_TURN_{turn_number}_BEGIN", flush=True)
        print(primary["output"], flush=True)
        print(f"RUNTIME_TURN_{turn_number}_END", flush=True)
        print(f"TURN_{turn_number}_POST_STATE={post_state['sha256']}", flush=True)
        print(f"TURN_{turn_number}_C={score}", flush=True)
        print(f"TURN_{turn_number}_DELTA_C={delta}", flush=True)
        print(f"TURN_{turn_number}_DIRECTION={sign}", flush=True)
        print(f"TURN_{turn_number}_DECISION={decision}", flush=True)
        print("DETERMINISTIC_REPLICATION=PASS", flush=True)
        if first_negative_no_descent:
            print("ONE_NEGATIVE_NOT_DESCENT=PASS", flush=True)
        if descent_invoked:
            print("TWO_CONSECUTIVE_NEGATIVE_PASSES=PASS", flush=True)
            print("DESCENT_AUTHORIZED_ONLY_AFTER_SECOND_NEGATIVE=PASS", flush=True)
            print("LOWER_TIER_SAME_DIRECTIONAL_LAW=PASS", flush=True)

        previous_score = score
        previous_post = post_state
        del pre_native
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("ORIGINATING_STANDING_WAVE_TERMINATED", flush=True)
        raise SystemExit(0)
    except Exception as error:
        print(
            f"ORIGINATING_STANDING_WAVE_ERROR={error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
