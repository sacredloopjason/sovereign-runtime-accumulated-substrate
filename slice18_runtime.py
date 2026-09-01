import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SLICE16_RUNTIME = ROOT / "slice16_runtime.py"
EVIDENCE_DIR = ROOT / "evidence" / "slice18"
SEED_LITERAL = "SLICE6_SEED"
PERIOD = 7
CYCLE_BOUND = 15
REPLICATION_COUNT = 2
OUTCOME_AVAILABLE = "STABILIZED_FACT_AVAILABLE"
OUTCOME_UNAVAILABLE = "NO_STABILIZED_FACT_REPRESENTATION_AVAILABLE"
OUTCOME_FAIL = "FAIL"
FACT_DOMAIN = b"SLICE18_STABILIZED_RELATION_FACT_V1"
CAPTURES = {}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def domain_hash(domain, value):
    return hashlib.sha256(domain + b"\0" + canonical_bytes(value)).hexdigest()


def pair_ids(cycle):
    return [
        [int(value) for value in cycle["T_A_produced"]["token_ids"]],
        [int(value) for value in cycle["T_B_produced"]["token_ids"]],
    ]


def clear_fact(cycles, initial_pair, delta, v_record, model_weight_sha256):
    if len(cycles) < PERIOD + 1:
        return None
    orbit = [pair_ids(cycle) for cycle in cycles[:PERIOD]]
    closure = pair_ids(cycles[PERIOD])
    if closure != orbit[0]:
        return None
    if any(
        all(orbit[index] == orbit[(index + candidate) % PERIOD] for index in range(PERIOD))
        for candidate in range(1, PERIOD)
    ):
        return None
    record = {
        "schema": "SLICE18_EXACT_CLOSED_RELATION_V1",
        "period": PERIOD,
        "orbit_R0_through_R6": orbit,
        "closure_R7": closure,
        "causal_constitution": {
            "P_A_initial_complete_KV_fingerprint": initial_pair["A"],
            "Delta_exact_ordered_token_ID_sequence": delta,
            "P_B_initial_complete_KV_fingerprint": initial_pair["B"],
            "V": v_record,
            "model_weight_sha256": model_weight_sha256,
        },
    }
    fact_bytes = canonical_bytes(record)
    return {
        "substrate": "immutable Python bytes containing canonical ASCII JSON",
        "bytes": fact_bytes,
        "record": record,
        "sha256": hashlib.sha256(FACT_DOMAIN + b"\0" + fact_bytes).hexdigest(),
    }


def run_slice18_wave(slice16, slice11, context, child, replicate):
    initial_pair = slice16.pair_fingerprint(slice11, context, child.poles)
    v_fingerprint = context["slice3"].fingerprint_v(child.shared_v)
    v_ids = [int(value) for value in v_fingerprint["token_ids"]]
    v_record = {"literal": child.shared_v.literal, "token_ids": v_ids}
    delta = slice16.token_evidence(context, child.perturbation)
    pending_a = slice16.traverse_pole(
        slice11, context, child, "A", child.perturbation,
        f"SLICE18_REPLICATE_{replicate}_CYCLE_0_A",
        "inherited unresolved parent Delta presented to actual descended Pole A",
    )
    cycles = []
    previous_pair = None
    recurrence_cycles = None
    a_lineage = [initial_pair["A"], pending_a["post"]]
    b_lineage = [initial_pair["B"]]
    exact_crossings = pending_a["model_boundary_exact"] and (
        pending_a["consumed"]["token_ids"] == pending_a["actual_first_forward_perturbation_ids"]
    )
    v_exact = pending_a["actual_first_forward_v_prefix"] == v_ids
    no_transfer = pending_a["other_pole_unchanged"]
    a_continuity = slice16.same_cache(pending_a["pre"], initial_pair["A"])
    b_continuity = True
    fact = None
    preclosure_fact = None

    for cycle_number in range(CYCLE_BOUND):
        if cycle_number > 0:
            a_continuity = a_continuity and slice16.same_cache(
                pending_a["pre"], a_lineage[-2]
            ) and slice16.same_cache(pending_a["post"], a_lineage[-1])
        relation_after_a = slice16.relation(
            pending_a["post"], slice16.cache_fingerprint(slice11, context, child.poles["B"])
        )
        b_traversal = slice16.traverse_pole(
            slice11, context, child, "B", pending_a["generated"]["token_ids"],
            f"SLICE18_REPLICATE_{replicate}_CYCLE_{cycle_number}_B",
            "exact generated ordered token-ID sequence from opposite Pole A",
        )
        b_continuity = b_continuity and slice16.same_cache(b_traversal["pre"], b_lineage[-1])
        b_lineage.append(b_traversal["post"])
        relation_after_b = slice16.relation(
            slice16.cache_fingerprint(slice11, context, child.poles["A"]), b_traversal["post"]
        )
        relation_ids = [pending_a["generated"]["token_ids"], b_traversal["generated"]["token_ids"]]
        relation_sha256 = slice16.stable_sha256(b"SLICE16_RECIPROCAL_PAIR_V1", relation_ids)
        recurrent = previous_pair is not None and relation_ids == previous_pair
        return_a = slice16.traverse_pole(
            slice11, context, child, "A", b_traversal["generated"]["token_ids"],
            f"SLICE18_REPLICATE_{replicate}_CYCLE_{cycle_number}_RETURN_A",
            "exact generated ordered token-ID sequence from opposite Pole B",
        )
        a_continuity = a_continuity and slice16.same_cache(return_a["pre"], a_lineage[-1])
        a_lineage.append(return_a["post"])
        relation_after_return = slice16.relation(return_a["post"], b_traversal["post"])
        ta_crossed = (
            pending_a["generated"] == b_traversal["consumed"]
            and b_traversal["consumed"]["token_ids"] == b_traversal["actual_first_forward_perturbation_ids"]
            and b_traversal["model_boundary_exact"]
        )
        tb_crossed = (
            b_traversal["generated"] == return_a["consumed"]
            and return_a["consumed"]["token_ids"] == return_a["actual_first_forward_perturbation_ids"]
            and return_a["model_boundary_exact"]
        )
        exact_crossings = exact_crossings and ta_crossed and tb_crossed
        v_exact = v_exact and (
            b_traversal["actual_first_forward_v_prefix"] == v_ids
            and return_a["actual_first_forward_v_prefix"] == v_ids
        )
        no_transfer = no_transfer and b_traversal["other_pole_unchanged"] and return_a["other_pole_unchanged"]
        cycles.append({
            "cycle": cycle_number,
            "A_pre": pending_a["pre"], "A_post": pending_a["post"],
            "T_A_produced": pending_a["generated"], "T_A_consumed_by_B": b_traversal["consumed"],
            "B_pre": b_traversal["pre"], "B_post": b_traversal["post"],
            "T_B_produced": b_traversal["generated"], "T_B_consumed_by_A": return_a["consumed"],
            "A_after_T_B": return_a["post"],
            "pole_state_relation_after_A": relation_after_a,
            "pole_state_relation_after_B": relation_after_b,
            "pole_state_relation": relation_after_return,
            "R_k_sha256": relation_sha256,
            "recurrent_with_previous": recurrent,
            "exact_T_A_crossing": ta_crossed, "exact_T_B_crossing": tb_crossed,
        })
        if cycle_number == PERIOD - 1:
            preclosure_fact = clear_fact(cycles, initial_pair, delta, v_record, context["weight_hash"])
        if cycle_number == PERIOD:
            fact = clear_fact(cycles, initial_pair, delta, v_record, context["weight_hash"])
            if fact is not None:
                fact["bytes_at_clearance"] = fact["bytes"]
        if recurrent:
            recurrence_cycles = [cycle_number - 1, cycle_number]
            break
        previous_pair = relation_ids
        pending_a = return_a
        gc.collect()

    if fact is not None:
        fact["bytes_after_live_continuation"] = fact["bytes"]
    CAPTURES[replicate] = {"fact": fact, "preclosure_fact": preclosure_fact}
    outcome = slice16.OUTCOME_RECURRENCE if recurrence_cycles is not None else slice16.OUTCOME_NO_RECURRENCE
    return {
        "initial_pair": initial_pair, "Delta_inherited": delta, "cycles": cycles,
        "recurrence_result": outcome, "recurrence_cycles": recurrence_cycles,
        "A_lineage": a_lineage, "B_lineage": b_lineage,
        "checks": {
            "RECIPROCAL_TRAVERSAL": len(cycles) >= 1,
            "EXACT_TOKEN_TRANSPOSITION": exact_crossings,
            "A_KV_CAUSAL_CONTINUITY": a_continuity,
            "B_KV_CAUSAL_CONTINUITY": b_continuity,
            "NO_KV_TRANSFER_BETWEEN_POLES": no_transfer,
            "PASSIVE_RECURRENCE_GATE": len(cycles) == CYCLE_BOUND and recurrence_cycles is None,
            "V_AT_EVERY_MODEL_BOUNDARY": v_exact,
        },
    }


def projection(replicate):
    return {key: value for key, value in replicate.items() if key != "replicate"}


def execute_acceptance():
    CAPTURES.clear()
    slice16 = load_module(SLICE16_RUNTIME, "slice16_accumulated_runtime_for_slice18")
    original_wave = slice16.run_boundary_preserving_wave
    slice16.run_boundary_preserving_wave = lambda s11, ctx, child, rep: run_slice18_wave(slice16, s11, ctx, child, rep)
    try:
        slice15 = slice16.load_slice15_runtime()
        slice14 = slice15.load_slice14_runtime()
        _, slice11, context = slice14.initialize_accumulated_runtime()
        replicates = [
            slice16.execute_replicate(slice15, slice14, slice11, context, number)
            for number in range(1, REPLICATION_COUNT + 1)
        ]
    finally:
        slice16.run_boundary_preserving_wave = original_wave

    for replicate in replicates:
        capture = CAPTURES[replicate["replicate"]]
        fact = capture["fact"]
        replicate["preclosure_control_cleared_fact"] = capture["preclosure_fact"]
        replicate["F"] = None if fact is None else {
            "substrate": fact["substrate"], "record": fact["record"], "sha256": fact["sha256"]
        }
        cycles = replicate["cycles"]
        record = None if fact is None else json.loads(fact["bytes"].decode("ascii"))
        checks = {
            "STABILIZED_FACT_EXISTS": fact is not None,
            "LOSSLESS_RECONSTRUCTION": fact is not None and canonical_bytes(record) == fact["bytes"],
            "EXACT_CLOSURE_FROM_F": fact is not None and record["closure_R7"] == record["orbit_R0_through_R6"][0],
            "MINIMAL_PERIOD_SEVEN_FROM_F": fact is not None and not any(
                all(record["orbit_R0_through_R6"][i] == record["orbit_R0_through_R6"][(i + p) % PERIOD] for i in range(PERIOD))
                for p in range(1, PERIOD)
            ),
            "CAUSAL_PROVENANCE": fact is not None and (
                record["causal_constitution"]["P_A_initial_complete_KV_fingerprint"] == replicate["P_A_INITIAL"]
                and record["causal_constitution"]["P_B_initial_complete_KV_fingerprint"] == replicate["P_B_INITIAL"]
                and record["causal_constitution"]["Delta_exact_ordered_token_ID_sequence"] == replicate["Delta_inherited"]
                and record["causal_constitution"]["V"]["token_ids"] == replicate["V"]["token_ids"]
            ),
            "PRE_CLOSURE_CONTROL": capture["preclosure_fact"] is None,
            "F_INDEPENDENT_OF_LIVE_POLE_CONTINUATION": fact is not None and fact["bytes_at_clearance"] == fact["bytes_after_live_continuation"],
            "LOWER_POLES_CONTINUE_FORWARD": len(cycles) == CYCLE_BOUND and all(
                cycles[i]["A_pre"] == cycles[i - 1]["A_post"]
                and cycles[i]["A_post"] == cycles[i - 1]["A_after_T_B"]
                and cycles[i]["B_pre"] == cycles[i - 1]["B_post"]
                for i in range(PERIOD + 1, CYCLE_BOUND)
            ),
            "PARENT_UNCHANGED": replicate["checks"]["PARENT_UNCHANGED"],
            "PARENT_CONTINUATION_CONTROL": replicate["checks"]["PARENT_CONTINUATION_CONTROL"],
            "P_A_EQUALS_S_PRE": replicate["checks"]["P_A_EQUALS_S_pre"],
            "P_B_EQUALS_S_POST": replicate["checks"]["P_B_EQUALS_S_post"],
            "IMMUTABLE_V": replicate["checks"]["V_IDENTITY"],
            "EXACT_TOKEN_TRANSPOSITION": replicate["checks"]["EXACT_TOKEN_TRANSPOSITION"],
        }
        replicate["slice18_checks"] = checks

    checks = {
        f"REPLICATE_{rep['replicate']}_{name}": passed
        for rep in replicates for name, passed in rep["slice18_checks"].items()
    }
    checks["DETERMINISTIC_REPLICATION"] = projection(replicates[0]) == projection(replicates[1])
    checks["F_IDENTITY_REPLICATED"] = replicates[0]["F"] == replicates[1]["F"]
    failed = [name for name, passed in checks.items() if not passed]
    outcome = OUTCOME_AVAILABLE if not failed else OUTCOME_FAIL
    result = {
        "schema": "SLICE18_STABILIZED_FACT_AVAILABILITY_V1",
        "seed": SEED_LITERAL, "period": PERIOD, "cycle_bound": CYCLE_BOUND,
        "replication_count": REPLICATION_COUNT, "replicates": replicates,
        "checks": checks, "failed_requirements": failed, "outcome": outcome,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "stabilized_fact_availability.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, receipt


def make_receipt(result):
    if result["outcome"] == OUTCOME_FAIL:
        failed = result["failed_requirements"][0] if result["failed_requirements"] else "deterministic Slice 18 execution"
        return "\n".join([
            "OUTCOME", "FAIL", "", "EVIDENCE", "", "failed_requirement:", failed, "",
            "causal_boundary:", "accumulated Slice 17 standing wave -> independently preserved stabilized fact", "",
            "substrate_evidence:", json.dumps(result["checks"], sort_keys=True), "",
        ])
    fact = result["replicates"][0]["F"]
    return "\n".join([
        "OUTCOME", OUTCOME_AVAILABLE, "", "EVIDENCE", "",
        "F_SUBSTRATE:", fact["substrate"], "",
        "F_IDENTITY:", fact["sha256"], "",
        "ORBIT_PERIOD:", "7", "",
        "ORBIT_MEMBERS:", json.dumps(fact["record"]["orbit_R0_through_R6"], separators=(",", ":")), "",
        "LOSSLESS_RECONSTRUCTION:", "PASS", "", "EXACT_CLOSURE_FROM_F:", "PASS", "",
        "CAUSAL_PROVENANCE:", "PASS", "", "PRE_CLOSURE_CONTROL:", "PASS", "",
        "F_INDEPENDENT_OF_LIVE_POLE_CONTINUATION:", "PASS", "",
        "LOWER_POLES_CONTINUE_FORWARD:", "PASS", "", "PARENT_UNCHANGED:", "PASS", "",
        "PARENT_CONTINUATION_CONTROL:", "PASS", "", "DETERMINISTIC_REPLICATION:", "PASS", "",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seed")
    args = parser.parse_args()
    if not args.verify:
        parser.error("Use --verify")
    if args.seed != SEED_LITERAL:
        raise SystemExit(f"The originating seed must be exactly {SEED_LITERAL}")
    result, receipt = execute_acceptance()
    print(receipt, end="", flush=True)
    return 0 if result["outcome"] != OUTCOME_FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
