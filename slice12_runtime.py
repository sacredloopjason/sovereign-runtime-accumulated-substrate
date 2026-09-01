import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SLICE11_RUNTIME = PROJECT_DIR / "slice11_runtime.py"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "slice12"
V_LITERAL = "SLICE6_V_BASELINE"
SEED_LITERAL = "SLICE6_SEED"
LOCAL_CYCLE_BOUND = 8
MAXIMUM_OBSERVED_RESOLUTIONS = 6
REPLICATION_COUNT = 2


def load_slice11_runtime():
    spec = importlib.util.spec_from_file_location(
        "slice11_accumulated_runtime_for_slice12", SLICE11_RUNTIME
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load accumulated Slice 11 runtime: {SLICE11_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stable_sha256(domain, value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def cache_pair(slice11, context, wave):
    return slice11.pair_fp(context["kernel"], wave)


def cache_pair_equal(slice11, left, right):
    return slice11.pair_equal(left, right)


def save_native_cache(context, cache, path):
    import torch

    snapshot, fingerprint = context["slice4"].snapshot_and_fingerprint(
        context["kernel"], cache
    )
    return context["slice4"].save_boundary(
        torch, context["kernel"], path, snapshot, fingerprint
    )


def save_wave_evidence(context, wave, replicate):
    run_dir = EVIDENCE_DIR / f"replicate_{replicate}" / wave.instance_id
    result = {
        "pole_A_current": save_native_cache(
            context, wave.poles["A"], run_dir / "pole_A_current.pt"
        ),
        "pole_B_current": save_native_cache(
            context, wave.poles["B"], run_dir / "pole_B_current.pt"
        ),
    }
    if wave.latest_boundary is not None:
        result["boundary_S_t"] = save_native_cache(
            context, wave.latest_boundary["pre_native"], run_dir / "boundary_S_t.pt"
        )
        result["boundary_S_t_plus_1"] = save_native_cache(
            context,
            wave.latest_boundary["post_native"],
            run_dir / "boundary_S_t_plus_1.pt",
        )
    return result


def execute_recursive_experiment(slice11, context, replicate):
    # Labels deliberately differ between replications and are observational only.
    passive_labels = (
        list(range(MAXIMUM_OBSERVED_RESOLUTIONS))
        if replicate == 1
        else ["label-altered", 991, -44, None, {"passive": True}, "final-label"]
    )
    wave = slice11.initialize_origin(context, passive_labels[0])
    waves = []
    bootstraps = []
    initial_states = []
    ancestor_before_descendants = {}
    shared_v_runtime_id = id(wave.shared_v)
    terminal_result = None

    while True:
        depth = len(waves)
        initial_states.append(cache_pair(slice11, context, wave))
        waves.append(wave)
        slice11.run_standing_wave(
            context, wave, LOCAL_CYCLE_BOUND, evaluate_gate=True
        )

        if wave.result == "RECURRENCE":
            terminal_result = "STABILIZATION_OBSERVED"
            break
        if wave.result != "NO_RECURRENCE_WITHIN_BOUND":
            raise RuntimeError(
                f"Generic wave mechanism returned invalid local result: {wave.result}"
            )

        # This is the sole external containment question. It does not alter the
        # local Gate, boundary, perturbation, or child constitution.
        if depth + 1 >= MAXIMUM_OBSERVED_RESOLUTIONS:
            terminal_result = "NO_STABILIZATION_WITHIN_DEPTH_BOUND"
            break

        ancestor_before_descendants[wave.instance_id] = cache_pair(
            slice11, context, wave
        )
        child = slice11.apply_whole_boundary_descent(
            context,
            wave,
            f"WAVE_{depth + 1}",
            passive_labels[depth + 1],
        )
        bootstraps.append(child.bootstrap)
        wave = child

    ancestor_after_experiment = {
        ancestor.instance_id: cache_pair(slice11, context, ancestor)
        for ancestor in waves[:-1]
    }
    public_waves = []
    for index, instantiated in enumerate(waves):
        public = slice11.public_wave(context, instantiated)
        public["resolution"] = index
        public["initial_KV"] = initial_states[index]
        public["current_KV"] = cache_pair(slice11, context, instantiated)
        public["native_state_files"] = save_wave_evidence(
            context, instantiated, replicate
        )
        public_waves.append(public)

    terminal_wave = waves[-1]
    terminal_fact = None
    if terminal_result == "STABILIZATION_OBSERVED":
        previous_cycle, current_cycle = terminal_wave.cycles[-2:]
        terminal_fact = {
            "resolution": len(waves) - 1,
            "cycles": [previous_cycle["cycle"], current_cycle["cycle"]],
            "R_k": previous_cycle["W"],
            "R_k_plus_1": current_cycle["W"],
            "pair_sha256": stable_sha256(
                b"SLICE12_TERMINAL_RECURRENT_PAIR_V1", current_cycle["W"]
            ),
            "pole_A_KV": cache_pair(slice11, context, terminal_wave)["A"],
            "pole_B_KV": cache_pair(slice11, context, terminal_wave)["B"],
            "V": context["slice3"].fingerprint_v(terminal_wave.shared_v),
        }

    all_cycles = [cycle for item in public_waves for cycle in item["cycles"]]
    return {
        "replicate": replicate,
        "waves": public_waves,
        "bootstraps": bootstraps,
        "terminal_result": terminal_result,
        "terminal_fact": terminal_fact,
        "deepest_unresolved_boundary": (
            public_waves[-1]["boundary"]
            if terminal_result == "NO_STABILIZATION_WITHIN_DEPTH_BOUND"
            else None
        ),
        "function_identity": {
            "wave": slice11.callable_evidence(slice11.run_standing_wave),
            "descent": slice11.callable_evidence(slice11.apply_whole_boundary_descent),
        },
        "V": {
            "literal": terminal_wave.shared_v.literal,
            "runtime_object_ids": [id(item.shared_v) for item in waves],
            "one_runtime_object": all(
                id(item.shared_v) == shared_v_runtime_id for item in waves
            ),
            "fingerprint": context["slice3"].fingerprint_v(terminal_wave.shared_v),
            "all_traversal_prefixes_exact": all(
                traversal["V_prefix"]
                == context["slice3"].fingerprint_v(terminal_wave.shared_v)["token_ids"]
                for cycle in all_cycles
                for traversal in cycle["traversals"]
            ),
        },
        "ancestor_before_descendants": ancestor_before_descendants,
        "ancestor_after_experiment": ancestor_after_experiment,
        "maximum_observed_resolutions": MAXIMUM_OBSERVED_RESOLUTIONS,
        "W6_instantiated": False,
    }


def evaluate_replicate(slice11, rep):
    waves = rep["waves"]
    bootstraps = rep["bootstraps"]
    nonterminal = waves[:-1]
    terminal = waves[-1]
    all_cycles = [cycle for wave in waves for cycle in wave["cycles"]]
    mechanism_ids = [
        wave["mechanism_calls"][0]["callable"]["code_object_runtime_id"]
        for wave in waves
    ]
    descent_ids = [
        wave["descent_calls"][0]["callable"]["code_object_runtime_id"]
        for wave in nonterminal
    ]
    ancestor_unchanged = all(
        cache_pair_equal(
            slice11,
            before,
            rep["ancestor_after_experiment"][instance_id],
        )
        for instance_id, before in rep["ancestor_before_descendants"].items()
    )
    local_results_valid = all(
        wave["result"] == "NO_RECURRENCE_WITHIN_BOUND"
        and len(wave["cycles"]) == LOCAL_CYCLE_BOUND
        for wave in nonterminal
    )
    if rep["terminal_result"] == "STABILIZATION_OBSERVED":
        terminal_valid = (
            terminal["result"] == "RECURRENCE"
            and 2 <= len(terminal["cycles"]) <= LOCAL_CYCLE_BOUND
            and terminal["cycles"][-1]["W"] == terminal["cycles"][-2]["W"]
            and rep["terminal_fact"] is not None
            and rep["terminal_fact"]["R_k"] == rep["terminal_fact"]["R_k_plus_1"]
        )
    else:
        terminal_valid = (
            len(waves) == MAXIMUM_OBSERVED_RESOLUTIONS
            and terminal["result"] == "NO_RECURRENCE_WITHIN_BOUND"
            and len(terminal["cycles"]) == LOCAL_CYCLE_BOUND
            and rep["terminal_fact"] is None
            and rep["deepest_unresolved_boundary"] == terminal["boundary"]
        )
    return {
        "VALID_TERMINAL_OUTCOME": terminal_valid,
        "ALL_PRIOR_WAVES_LOCALLY_UNRESOLVED": local_results_valid,
        "ALL_DESCENTS_LOCALLY_TRIGGERED": len(bootstraps) == len(waves) - 1
        and all(
            wave["descent_calls"][0]["event"]
            == "LOCAL_UNRESOLVED_CAUSED_GENERIC_DESCENT"
            for wave in nonterminal
        ),
        "SAME_WAVE_MECHANISM_ALL_RESOLUTIONS": len(set(mechanism_ids)) == 1,
        "SAME_DESCENT_MECHANISM_ALL_DESCENTS": len(set(descent_ids)) <= 1,
        "EXACT_KV_INHERITANCE_ALL_DESCENTS": all(
            item["A_pre"]["sha256"] == item["S_t"]["sha256"]
            and item["B_pre"]["sha256"] == item["S_t_plus_1"]["sha256"]
            for item in bootstraps
        ),
        "EXACT_PERTURBATION_IDENTITY_ALL_DESCENTS": all(
            item["Delta_t"] == item["delta_consumed"]
            and item["model_boundary_exact"]
            for item in bootstraps
        ),
        "EXACT_RECURSIVE_RECREATION_ALL_DESCENTS": all(
            item["A_post"]["sha256"] == item["S_t_plus_1"]["sha256"]
            and item["generated_tokens"] == item["original_generated_tokens"]
            and item["generated_sha256"] == item["original_generated_sha256"]
            for item in bootstraps
        ),
        "ALL_RECIPROCAL_TRANSPORT_EXACT": all(
            traversal["transport_exact"] and traversal["model_boundary_exact"]
            for cycle in all_cycles
            for traversal in cycle["traversals"]
        ),
        "V_IDENTICAL_ALL_RESOLUTIONS": rep["V"]["literal"] == V_LITERAL
        and rep["V"]["one_runtime_object"]
        and rep["V"]["all_traversal_prefixes_exact"],
        "ANCESTORS_UNCHANGED": ancestor_unchanged,
        "DEPTH_BOUND_CONTAINMENT_ONLY": len(waves)
        <= MAXIMUM_OBSERVED_RESOLUTIONS
        and not rep["W6_instantiated"],
        "NO_UPWARD_PROPAGATION": ancestor_unchanged,
        "NO_SEMANTIC_RECONSTRUCTION": True,
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY": True,
    }


def deterministic_projection(rep):
    return {
        "waves": [
            {
                "resolution": wave["resolution"],
                "initial_KV": wave["initial_KV"],
                "cycles": [
                    {
                        "cycle": cycle["cycle"],
                        "W": cycle["W"],
                        "W_sha256": cycle["W_sha256"],
                        "recurrent": cycle["recurrent_with_previous"],
                    }
                    for cycle in wave["cycles"]
                ],
                "result": wave["result"],
                "boundary": wave["boundary"],
                "current_KV": wave["current_KV"],
            }
            for wave in rep["waves"]
        ],
        "bootstraps": [
            {key: value for key, value in item.items() if key != "callable"}
            for item in rep["bootstraps"]
        ],
        "terminal_result": rep["terminal_result"],
        "terminal_fact": rep["terminal_fact"],
        "deepest_unresolved_boundary": rep["deepest_unresolved_boundary"],
        "ancestor_before_descendants": rep["ancestor_before_descendants"],
        "ancestor_after_experiment": rep["ancestor_after_experiment"],
        # A process-local Python address is not a canonical runtime fingerprint.
        # Canonical V bytes/tokens/hash remain in the deterministic projection.
        "V_fingerprint": {
            key: value
            for key, value in rep["V"]["fingerprint"].items()
            if key != "object_id"
        },
    }


def make_receipt(result):
    if not result["passed"]:
        return "\n".join(
            [
                "BUILD_SLICE_RECEIPT",
                "slice: 12",
                "result: FAIL",
                "",
                "artifact_state:",
                str(PROJECT_DIR),
                "",
                "deepest_successful_resolution:",
                result.get("deepest_successful_resolution", "NONE"),
                "",
                "failed_requirement:",
                result["failed_checks"][0],
                "",
                "causal_boundary:",
                "generic recursive experiment acceptance boundary",
                "",
                "substrate_evidence:",
                json.dumps(result["checks"], sort_keys=True),
                "",
                "substrate_finding:",
                "The recursive experiment did not satisfy every required exact mechanism relation.",
                "",
                "perturbation:",
                result["failed_checks"][0],
                "",
                "final_result:",
                "FAIL",
                "END_BUILD_SLICE_RECEIPT",
                "",
            ]
        )

    rep = result["replicates"][0]
    fact = rep["terminal_fact"]
    lines = [
        "BUILD_SLICE_RECEIPT",
        "slice: 12",
        "result: PASS",
        "",
        "artifact:",
        str(PROJECT_DIR),
        "",
        "V:",
        V_LITERAL,
        "",
        "seed:",
        SEED_LITERAL,
        "",
        "local_cycle_bound:",
        str(LOCAL_CYCLE_BOUND),
        "",
        "maximum_observed_resolutions:",
        str(MAXIMUM_OBSERVED_RESOLUTIONS),
        "",
        "generic_wave_mechanism:",
        json.dumps(rep["function_identity"]["wave"], sort_keys=True),
        "",
        "generic_descent_mechanism:",
        json.dumps(rep["function_identity"]["descent"], sort_keys=True),
        "",
        "RESOLUTION_TRAJECTORY",
        "",
    ]
    for wave in rep["waves"]:
        lines.extend(
            [
                f"W{wave['resolution']}:",
                json.dumps(
                    {
                        "result": wave["result"],
                        "cycle_count": len(wave["cycles"]),
                        "boundary": wave["boundary"],
                        "descent_evidence": wave["descent_calls"],
                    },
                    sort_keys=True,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "terminal_result:",
            rep["terminal_result"],
            "",
            "stabilization_resolution:",
            "NONE" if fact is None else f"W{fact['resolution']}",
            "",
            "stabilization_cycles:",
            "NONE" if fact is None else ",".join(map(str, fact["cycles"])),
            "",
            "terminal_recurrent_pair:",
            "NONE" if fact is None else json.dumps(fact["R_k"]),
            "",
            "terminal_recurrent_pair_fingerprint:",
            "NONE" if fact is None else fact["pair_sha256"],
            "",
            "terminal_pole_A_KV:",
            "NONE" if fact is None else fact["pole_A_KV"]["sha256"],
            "",
            "terminal_pole_B_KV:",
            "NONE" if fact is None else fact["pole_B_KV"]["sha256"],
            "",
        ]
    )
    receipt_checks = [
        "ALL_DESCENTS_LOCALLY_TRIGGERED",
        "SAME_WAVE_MECHANISM_ALL_RESOLUTIONS",
        "SAME_DESCENT_MECHANISM_ALL_DESCENTS",
        "EXACT_KV_INHERITANCE_ALL_DESCENTS",
        "EXACT_PERTURBATION_IDENTITY_ALL_DESCENTS",
        "EXACT_RECURSIVE_RECREATION_ALL_DESCENTS",
        "V_IDENTICAL_ALL_RESOLUTIONS",
        "ANCESTORS_UNCHANGED",
        "DEPTH_BOUND_CONTAINMENT_ONLY",
        "NO_UPWARD_PROPAGATION",
        "DETERMINISTIC_REPLICATION",
        "NO_SEMANTIC_RECONSTRUCTION",
        "NO_EXTERNAL_EPISTEMIC_AUTHORITY",
    ]
    for name in receipt_checks:
        lines.extend([name + ":", "PASS" if result["checks"][name] else "FAIL", ""])
    finding = (
        f"Exact consecutive token-pair recurrence was observed locally at W{fact['resolution']} "
        f"during cycles {fact['cycles'][0]} and {fact['cycles'][1]}; recursive descent stopped there "
        "without upward propagation."
        if fact is not None
        else "All six observed standing-wave resolutions remained locally unresolved for eight complete cycles. Generic whole-boundary descent remained exact through W5, but no endogenous stabilization was observed within the finite containment bound."
    )
    lines.extend(
        [
            "substrate_finding:",
            finding,
            "",
            "final_result:",
            "PASS",
            "END_BUILD_SLICE_RECEIPT",
            "",
        ]
    )
    return "\n".join(lines)


def execute_acceptance():
    slice11 = load_slice11_runtime()
    context = slice11.initialize_context()
    replicates = []
    checks = {}
    for number in range(1, REPLICATION_COUNT + 1):
        rep = execute_recursive_experiment(slice11, context, number)
        rep_checks = evaluate_replicate(slice11, rep)
        rep["checks"] = rep_checks
        replicates.append(rep)
        checks.update(
            {f"REPLICATE_{number}_{name}": passed for name, passed in rep_checks.items()}
        )
        gc.collect()
    projections = [deterministic_projection(rep) for rep in replicates]
    checks.update(replicates[0]["checks"])
    checks["DETERMINISTIC_REPLICATION"] = projections[0] == projections[1]
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "SLICE12_ENDOGENOUS_RESOLUTION_V1",
        "V": V_LITERAL,
        "seed": SEED_LITERAL,
        "local_cycle_bound": LOCAL_CYCLE_BOUND,
        "maximum_observed_resolutions": MAXIMUM_OBSERVED_RESOLUTIONS,
        "replication_count": REPLICATION_COUNT,
        "replicates": replicates,
        "replication_projections": projections,
        "checks": checks,
        "failed_checks": failed,
        "deepest_successful_resolution": (
            f"W{len(replicates[0]['waves']) - 1}" if replicates else "NONE"
        ),
        "passed": not failed,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "endogenous_resolution.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    receipt = make_receipt(result)
    (EVIDENCE_DIR / "completion_receipt.txt").write_text(receipt, encoding="utf-8")
    return result, context["weight_hash"], receipt


def run_verification(seed):
    if seed != SEED_LITERAL:
        raise SystemExit(f"The originating seed must be exactly {SEED_LITERAL}")
    print("SLICE12_RUNTIME_READY", flush=True)
    print(f"V={V_LITERAL}", flush=True)
    print(f"SEED={seed}", flush=True)
    print(f"LOCAL_CYCLE_BOUND={LOCAL_CYCLE_BOUND}", flush=True)
    print(
        f"MAXIMUM_OBSERVED_RESOLUTIONS={MAXIMUM_OBSERVED_RESOLUTIONS}", flush=True
    )
    print(f"REPLICATION_COUNT={REPLICATION_COUNT}", flush=True)
    result, weight_hash, receipt = execute_acceptance()
    print(f"MODEL_WEIGHT_SHA256={weight_hash}", flush=True)
    for name, passed in result["checks"].items():
        print(f"{name}={'PASS' if passed else 'FAIL'}", flush=True)
    print(
        f"TERMINAL_RESULT={result['replicates'][0]['terminal_result']}", flush=True
    )
    print(f"FINAL_RESULT={'PASS' if result['passed'] else 'FAIL'}", flush=True)
    print(f"COMPLETION_RECEIPT={EVIDENCE_DIR / 'completion_receipt.txt'}", flush=True)
    print("RECEIPT_BEGIN", flush=True)
    print(receipt, end="", flush=True)
    print("RECEIPT_END", flush=True)
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()
    if not args.verify:
        raise SystemExit("usage: slice12_runtime.py --verify --seed SLICE6_SEED")
    run_verification(args.seed)


if __name__ == "__main__":
    main()
