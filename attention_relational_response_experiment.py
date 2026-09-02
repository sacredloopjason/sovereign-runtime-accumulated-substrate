from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import kv_causal_intervention_lab as lab
import kv_causal_response_structure as response
import realized_attention_state_exposure as attention

PARENT_COMMIT = "2600b5a687af1062668e0f365a2df3d96c5993f4"
OUT_DIR = ROOT / "evidence" / "attention_relational_response"
RAW_DIR = OUT_DIR / "temporary_raw"
RESULT_PATH = OUT_DIR / "attention_relational_response.json"
RELATION_PATH = OUT_DIR / "event_relation_matrices.json"
RECEIPT_PATH = OUT_DIR / "completion_receipt.txt"
BASELINE_DIR = ROOT / "evidence" / "realized_attention_state"
BASELINE_RESULT_PATH = BASELINE_DIR / "realized_attention_state_exposure.json"
BASELINE_MANIFEST_PATH = BASELINE_DIR / "attention_manifest.json"
SOURCE_SNAPSHOT = ROOT / "evidence" / "kv_causal_intervention_lab" / "K_t_snapshot.pt"
PERMUTATION_COUNT = 10000
ALPHA = 0.003125
EXPECTED_SOURCE_SHA256 = "8d5f495ed7c83b37201804e7bfea91f5b9481aa133fea015aa4ea15415aa0eaf"
EXPECTED_DELTA_SHA256 = "7b5154e56e0e904f4820b0dfad049b8e40b9706a4a36143b01983b4ed6e7b78c"
EXPECTED_DELTA_IDS = [
    10199, 15799, 38, 79, 6493, 3010, 198, 10199,
    15799, 38, 79, 6493, 3010, 198, 10199, 15799,
]
EXPECTED_SITES = [
    (0, 1, "key", [0, 1, 160, 0]),
    (1, 3, "value", [0, 1, 160, 0]),
    (2, 6, "key", [0, 1, 160, 0]),
    (3, 8, "value", [0, 1, 160, 0]),
    (4, 11, "key", [0, 1, 160, 0]),
    (5, 13, "value", [0, 1, 160, 0]),
    (6, 16, "key", [0, 1, 160, 0]),
    (7, 18, "value", [0, 1, 160, 0]),
    (8, 21, "key", [0, 1, 160, 0]),
    (9, 23, "value", [0, 1, 160, 0]),
    (10, 26, "key", [0, 1, 160, 0]),
    (11, 28, "value", [0, 1, 160, 0]),
]


class ExecutionBoundary(RuntimeError):
    def __init__(self, capability, details):
        super().__init__(capability)
        self.capability = capability
        self.details = details


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_response_hash(event_index, layers, token_status):
    digest = hashlib.sha256()
    digest.update(b"DERIVED_ATTENTION_CAUSAL_RESPONSE_EVENT_V1")
    digest.update(canonical_json({
        "forward_event_index": event_index,
        "token_history_alignment_status": token_status,
    }).encode("ascii"))
    for layer in layers:
        digest.update(canonical_json({
            "layer": layer["layer"],
            "query_heads": layer["query_heads"],
            "query_positions": layer["query_positions"],
            "key_positions": layer["key_positions"],
            "shape": list(layer["response_tensor"].shape),
            "dtype": str(layer["response_tensor"].dtype),
        }).encode("ascii"))
        digest.update(attention.tensor_bytes(layer["response_tensor"]))
    return digest.hexdigest()


def semantic_artifact_hash(events):
    digest = hashlib.sha256()
    digest.update(b"DERIVED_ATTENTION_CAUSAL_RESPONSE_TRAJECTORY_V1")
    for event in events:
        digest.update(event["semantic_sha256"].encode("ascii"))
    return digest.hexdigest()


def guard_authoritative_files():
    paths = [
        SOURCE_SNAPSHOT,
        ROOT / "kv_causal_intervention_lab.py",
        ROOT / "kv_causal_response_structure.py",
        ROOT / "realized_attention_state_exposure.py",
        BASELINE_RESULT_PATH,
        BASELINE_MANIFEST_PATH,
        BASELINE_DIR / "completion_receipt.txt",
    ]
    paths.extend(sorted((BASELINE_DIR / "artifacts").glob("attention_event_*.pt")))
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
    }


def load_and_verify_baseline():
    result = json.loads(BASELINE_RESULT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(BASELINE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if result["authoritative_source_K_t"]["sha256"] != EXPECTED_SOURCE_SHA256:
        raise ExecutionBoundary("authoritative baseline K_t identity failed", result["authoritative_source_K_t"])
    if result["authoritative_delta_t"]["token_ids"] != EXPECTED_DELTA_IDS:
        raise ExecutionBoundary("authoritative baseline delta_t token identity failed", result["authoritative_delta_t"])
    if result["authoritative_delta_t"]["sha256"] != EXPECTED_DELTA_SHA256:
        raise ExecutionBoundary("authoritative baseline delta_t fingerprint failed", result["authoritative_delta_t"])
    if result["forward_event_count"] != 17 or manifest["event_order"] != list(range(17)):
        raise ExecutionBoundary("authoritative baseline event order failed", manifest["event_order"])
    if sha256_file(BASELINE_MANIFEST_PATH) != result["artifacts"]["manifest_sha256"]:
        raise ExecutionBoundary("authoritative attention manifest file identity failed", {})
    if [
        item["final_semantic_sha256"] for item in manifest["events"]
    ] != manifest["replicate_b_final_semantic_hashes"]:
        raise ExecutionBoundary("authoritative baseline attention replication identity failed", {})
    for item in manifest["events"]:
        path = ROOT / item["path"]
        if sha256_file(path) != item["file_sha256"]:
            raise ExecutionBoundary("authoritative baseline attention artifact identity failed", {
                "forward_event_index": item["forward_event_index"],
                "path": item["path"],
            })
    return result, manifest


def restore_and_intervene(kernel, context, source_snapshot, source_fp, site):
    loaded = torch.load(SOURCE_SNAPSHOT, map_location="cpu", weights_only=False)
    cache = lab.restore_dynamic_cache(kernel, context["model"], loaded)
    restored = kernel.snapshot_cache(cache)
    if not lab.snapshot_content_equal(source_snapshot, restored):
        raise ExecutionBoundary("condition did not begin from exact K_t", {"intervention": site["j"]})
    if not lab.same_fingerprint(lab.fingerprint(kernel, cache), source_fp):
        raise ExecutionBoundary("condition K_t fingerprint failed", {"intervention": site["j"]})
    specification = response.apply_intervention(kernel, cache, source_snapshot, site)
    start_snapshot = kernel.snapshot_cache(cache)
    start_fingerprint = lab.fingerprint(kernel, cache)
    return cache, specification, start_snapshot, start_fingerprint


def execute_condition(name, observed, slice11, context, shared_v, source_snapshot, source_fp, site, raw_path=None):
    kernel = context["kernel"]
    cache, specification, start_snapshot, start_fingerprint = restore_and_intervene(
        kernel, context, source_snapshot, source_fp, site
    )
    label = f"ATTENTION_RELATIONAL_{name}_I{site['j']:02d}_{'OBSERVED' if observed else 'CONTROL'}"
    if observed:
        with attention.NativeAttentionCapture(context["model"], raw_path) as capture:
            traversal = slice11.traverse(
                context, cache, shared_v, EXPECTED_DELTA_IDS, label,
                "exact established KV causal-intervention perturbation",
            )
        capture_manifest = capture.events
    else:
        traversal = slice11.traverse(
            context, cache, shared_v, EXPECTED_DELTA_IDS, label,
            "exact established KV causal-intervention perturbation",
        )
        capture_manifest = None
    delta = response.exact_delta(context, traversal)
    generated = lab.token_evidence(context, traversal["generated_token_ids"])
    post_snapshot = kernel.snapshot_cache(cache)
    post_fingerprint = lab.fingerprint(kernel, cache)
    return {
        "specification": specification,
        "start_snapshot": start_snapshot,
        "start_fingerprint": start_fingerprint,
        "delta": delta,
        "generated": generated,
        "post_snapshot": post_snapshot,
        "post_fingerprint": post_fingerprint,
        "capture_manifest": capture_manifest,
    }


def first_transition_divergence(control, observed):
    if control["specification"] != observed["specification"]:
        return {"field": "intervention_specification"}
    if control["start_fingerprint"] != observed["start_fingerprint"]:
        return {"field": "starting_intervened_K_t_fingerprint"}
    if not lab.snapshot_content_equal(control["start_snapshot"], observed["start_snapshot"]):
        return {"field": "starting_intervened_K_t"}
    if control["delta"] != observed["delta"]:
        return {"field": "supplied_delta_t", "control": control["delta"], "observed": observed["delta"]}
    if control["generated"] != observed["generated"]:
        left = control["generated"]["token_ids"]
        right = observed["generated"]["token_ids"]
        index = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), min(len(left), len(right)))
        return {"field": "generated_token_ids", "first_divergent_token_index": index, "control": left, "observed": right}
    if control["post_fingerprint"] != observed["post_fingerprint"]:
        return {"field": "final_complete_KV_fingerprint", "control": control["post_fingerprint"], "observed": observed["post_fingerprint"]}
    if not lab.snapshot_content_equal(control["post_snapshot"], observed["post_snapshot"]):
        return {"field": "final_complete_KV_state"}
    return None


def event_token_status(event_index, generated_ids, baseline_generated_ids):
    prefix_length = min(event_index, len(generated_ids), len(baseline_generated_ids))
    aligned = (
        event_index <= len(generated_ids)
        and event_index <= len(baseline_generated_ids)
        and generated_ids[:prefix_length] == baseline_generated_ids[:prefix_length]
    )
    return "TOKEN_HISTORY_ALIGNED" if aligned else "TOKEN_HISTORY_DIVERGED"


def alignable(intervened, baseline):
    if len(intervened["layers"]) != len(baseline["layers"]):
        return False
    for left, right in zip(intervened["layers"], baseline["layers"]):
        if left["layer_index"] != right["layer_index"]:
            return False
        if left["query_positions"] != right["query_positions"]:
            return False
        if left["key_positions"] != right["key_positions"]:
            return False
        if tuple(left["attention"].shape) != tuple(right["attention"].shape):
            return False
    return True


def process_capture(replicate, intervention, capture_manifest, baseline_manifest, generated_ids, baseline_generated_ids, save_artifact):
    events = []
    raw_hashes = []
    maximum_reconstruction_difference = 0.0
    first_unalignable_event = None
    for raw_record, baseline_record in zip(capture_manifest, baseline_manifest["events"]):
        event_index = raw_record["forward_event_index"]
        if event_index != baseline_record["forward_event_index"]:
            first_unalignable_event = event_index
            break
        raw_path = Path(raw_record["path"])
        if sha256_file(raw_path) != raw_record["file_sha256"]:
            raise ExecutionBoundary("captured native operand file identity failed", {
                "replicate": replicate, "intervention": intervention, "event": event_index,
            })
        raw_hashes.append(raw_record["raw_semantic_sha256"])
        intervened = torch.load(raw_path, map_location="cpu", weights_only=False)
        for layer in intervened["layers"]:
            _, difference = attention.reconstruct_layer(layer)
            maximum_reconstruction_difference = max(maximum_reconstruction_difference, difference)
        baseline = torch.load(ROOT / baseline_record["path"], map_location="cpu", weights_only=False)
        if not alignable(intervened, baseline):
            first_unalignable_event = event_index
            del intervened, baseline
            gc.collect()
            break
        token_status = event_token_status(event_index, generated_ids, baseline_generated_ids)
        response_layers = []
        profile = []
        nonzero = False
        for left, right in zip(intervened["layers"], baseline["layers"]):
            delta = (left["attention"] - right["attention"]).detach().cpu().contiguous()
            nonzero = nonzero or bool(torch.count_nonzero(delta).item())
            for head in range(delta.shape[1]):
                value = math.sqrt(float(torch.sum(delta[0, head].to(torch.float64) ** 2).item()))
                profile.append(value)
            response_layers.append({
                "layer": left["layer_index"],
                "query_heads": int(delta.shape[1]),
                "query_positions": left["query_positions"],
                "key_positions": left["key_positions"],
                "response_tensor": delta,
            })
        if len(profile) != 270:
            raise ExecutionBoundary("event response profile did not contain 270 canonical coordinates", {
                "replicate": replicate, "intervention": intervention, "event": event_index, "count": len(profile),
            })
        magnitude = math.sqrt(sum(value * value for value in profile))
        normalized = None if magnitude == 0.0 else [value / magnitude for value in profile]
        semantic_hash = semantic_response_hash(event_index, response_layers, token_status)
        event_input = None if event_index == 0 else generated_ids[event_index - 1]
        events.append({
            "forward_event_index": event_index,
            "layer_count": len(response_layers),
            "query_head_count": 9,
            "token_history_alignment_status": token_status,
            "generated_token_id_entering_event": event_input,
            "generated_token_history_before_or_at_event": generated_ids[:event_index],
            "nonzero": nonzero,
            "response_l2_norm": magnitude,
            "energy_profile": profile,
            "normalized_energy_profile": normalized,
            "semantic_sha256": semantic_hash,
            "layers": response_layers,
        })
        del intervened, baseline
        gc.collect()
    artifact_semantic_hash = semantic_artifact_hash(events)
    artifact_record = {
        "intervention": intervention,
        "represented_replicates": ["A", "B"],
        "event_count": len(events),
        "semantic_sha256": artifact_semantic_hash,
    }
    if save_artifact:
        artifact_path = OUT_DIR / f"response_{intervention:02d}.pt"
        torch.save({
            "schema": "DERIVED_ATTENTION_CAUSAL_RESPONSE_TRAJECTORY_V1",
            "intervention": intervention,
            "represented_replicates": ["A", "B"],
            "derived_attention_status": "OFFLINE_RECONSTRUCTED_FROM_DIRECTLY_OBSERVED_NATIVE_OPERANDS",
            "native_A_directly_materialized": "NO",
            "events": events,
        }, artifact_path)
        if artifact_path.stat().st_size >= 100_000_000:
            raise ExecutionBoundary("complete response artifact exceeds GitHub individual-file limit", {
                "path": artifact_path.relative_to(ROOT).as_posix(),
                "bytes": artifact_path.stat().st_size,
            })
        artifact_record.update({
            "path": artifact_path.relative_to(ROOT).as_posix(),
            "file_sha256": sha256_file(artifact_path),
            "byte_count": artifact_path.stat().st_size,
        })
    summary_events = [{
        key: value for key, value in event.items() if key != "layers"
    } for event in events]
    return {
        "events": summary_events,
        "artifact": artifact_record,
        "raw_operand_semantic_hashes": raw_hashes,
        "first_unalignable_event": first_unalignable_event,
        "maximum_reconstruction_output_difference": maximum_reconstruction_difference,
    }


def relation_matrix(event_profiles):
    matrix = []
    eligible = all(profile is not None for profile in event_profiles)
    for left in event_profiles:
        row = []
        for right in event_profiles:
            if left is None or right is None:
                row.append("UNDEFINED_ZERO_RESPONSE")
            else:
                row.append(sum(a * b for a, b in zip(left, right)))
        matrix.append(row)
    return matrix, eligible


def upper_triangle(matrix):
    return [matrix[i][j] for i in range(12) for j in range(i + 1, 12)]


def pearson(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    a = [value - left_mean for value in left]
    b = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in a) * sum(value * value for value in b))
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(a, b)) / denominator


def adjacent_test(event_index, left_matrix, right_matrix):
    observed = pearson(upper_triangle(left_matrix), upper_triangle(right_matrix))
    if observed is None:
        return {
            "from_event": event_index,
            "to_event": event_index + 1,
            "eligible_all_twelve_nonzero": True,
            "r": "UNDEFINED_DEGENERATE_RELATION",
            "p": "UNDEFINED_DEGENERATE_RELATION",
            "threshold": ALPHA,
            "passes": False,
        }
    generator = random.Random(0)
    exceed = 0
    left = upper_triangle(left_matrix)
    labels = list(range(12))
    for _ in range(PERMUTATION_COUNT):
        permutation = labels[:]
        generator.shuffle(permutation)
        permuted = [
            right_matrix[permutation[i]][permutation[j]]
            for i in range(12) for j in range(i + 1, 12)
        ]
        candidate = pearson(left, permuted)
        if candidate is not None and candidate >= observed:
            exceed += 1
    p_value = (1 + exceed) / 10001
    return {
        "from_event": event_index,
        "to_event": event_index + 1,
        "eligible_all_twelve_nonzero": True,
        "r": observed,
        "p": p_value,
        "permutation_count": PERMUTATION_COUNT,
        "prng": "Python random.Random(0)",
        "threshold": ALPHA,
        "passes": p_value <= ALPHA,
    }


def qualifying_runs(tests):
    by_start = {item["from_event"]: item for item in tests}
    runs = []
    for event in sorted(by_start):
        if (
            by_start[event]["passes"]
            and event + 1 in by_start
            and by_start[event + 1]["passes"]
        ):
            runs.append([event, event + 1, event + 2])
    return runs


def analyze_scope(replicate, event_count):
    nonzero_counts = [
        sum(item["nonzero"] for item in condition["response"]["events"][:event_count])
        for condition in replicate["conditions"]
    ]
    multi_intervention = sum(count > 1 for count in nonzero_counts) >= 3
    nondegenerate = False
    for event in range(event_count):
        hashes = {
            condition["response"]["events"][event]["semantic_sha256"]
            for condition in replicate["conditions"]
            if condition["response"]["events"][event]["nonzero"]
        }
        if len(hashes) >= 3:
            nondegenerate = True
            break
    matrices = []
    eligible = []
    for event in range(event_count):
        profiles = [
            condition["response"]["events"][event]["normalized_energy_profile"]
            for condition in replicate["conditions"]
        ]
        matrix, is_eligible = relation_matrix(profiles)
        matrices.append(matrix)
        eligible.append(is_eligible)
    tests = []
    for event in range(max(0, event_count - 1)):
        if eligible[event] and eligible[event + 1]:
            tests.append(adjacent_test(event, matrices[event], matrices[event + 1]))
    runs = qualifying_runs(tests)
    return {
        "event_count": event_count,
        "nonzero_attention_response_counts_by_intervention": nonzero_counts,
        "nontrivial_multi_intervention_response": multi_intervention,
        "non_degenerate_relation": nondegenerate,
        "event_relation_matrices": matrices,
        "eligible_events": eligible,
        "adjacent_event_relational_tests": tests,
        "qualifying_temporal_runs": runs,
        "positive_gate": multi_intervention and nondegenerate and bool(runs),
    }


def execute_replicate(name, slice11, context, shared_v, source_snapshot, source_fp, sites, baseline_result, baseline_manifest, reference=None):
    baseline_generated = baseline_result["generated_tokens"]["token_ids"]
    conditions = []
    for site in sites:
        raw_path = RAW_DIR / name / f"I{site['j']:02d}"
        control = execute_condition(
            name, False, slice11, context, shared_v, source_snapshot, source_fp, site
        )
        observed = execute_condition(
            name, True, slice11, context, shared_v, source_snapshot, source_fp, site, raw_path
        )
        divergence = first_transition_divergence(control, observed)
        if divergence is not None:
            raise ExecutionBoundary(
                "existing attention observation is not causally invariant under the bounded intervention conditions",
                {"replicate": name, "intervention": site["j"], "first_divergence": divergence},
            )
        processed = process_capture(
            name, site["j"], observed["capture_manifest"], baseline_manifest,
            observed["generated"]["token_ids"], baseline_generated, save_artifact=(name == "A"),
        )
        if raw_path.exists():
            shutil.rmtree(raw_path)
        condition = {
            "intervention": site["j"],
            "specification": observed["specification"],
            "generated": observed["generated"],
            "start_fingerprint": observed["start_fingerprint"],
            "final_fingerprint": observed["post_fingerprint"],
            "observer_control": "PASS",
            "response": processed,
        }
        if reference is not None:
            prior = reference["conditions"][site["j"]]
            checks = {
                "specification": condition["specification"] == prior["specification"],
                "generated": condition["generated"] == prior["generated"],
                "start_fingerprint": condition["start_fingerprint"] == prior["start_fingerprint"],
                "final_fingerprint": condition["final_fingerprint"] == prior["final_fingerprint"],
                "raw_operand_semantic_hashes": processed["raw_operand_semantic_hashes"] == prior["response"]["raw_operand_semantic_hashes"],
                "response_semantic_hash": processed["artifact"]["semantic_sha256"] == prior["response"]["artifact"]["semantic_sha256"],
                "events": processed["events"] == prior["response"]["events"],
            }
            if not all(checks.values()):
                raise ExecutionBoundary("complete derived attention response failed deterministic replication", {
                    "intervention": site["j"], "checks": checks,
                })
            processed["artifact"] = prior["response"]["artifact"]
        conditions.append(condition)
        del control, observed
        gc.collect()
    common_count = min(len(item["response"]["events"]) for item in conditions)
    token_aligned_count = 0
    for event in range(common_count):
        if all(
            item["response"]["events"][event]["token_history_alignment_status"] == "TOKEN_HISTORY_ALIGNED"
            for item in conditions
        ):
            token_aligned_count += 1
        else:
            break
    replicate = {
        "replicate": name,
        "conditions": conditions,
        "common_shape_alignable_forward_event_count": common_count,
        "token_history_aligned_forward_event_count": token_aligned_count,
    }
    replicate["complete_scope"] = analyze_scope(replicate, common_count)
    replicate["token_aligned_scope"] = analyze_scope(replicate, token_aligned_count)
    return replicate


def strip_large_replication(replicate):
    return {
        "replicate": replicate["replicate"],
        "conditions": [{
            "intervention": item["intervention"],
            "specification": item["specification"],
            "generated": item["generated"],
            "start_fingerprint": item["start_fingerprint"],
            "final_fingerprint": item["final_fingerprint"],
            "observer_control": item["observer_control"],
            "response": {
                "artifact": item["response"]["artifact"],
                "raw_operand_semantic_hashes": item["response"]["raw_operand_semantic_hashes"],
                "first_unalignable_event": item["response"]["first_unalignable_event"],
                "maximum_reconstruction_output_difference": item["response"]["maximum_reconstruction_output_difference"],
                "events": [{
                    key: value for key, value in event.items()
                    if key not in ("energy_profile", "normalized_energy_profile")
                } for event in item["response"]["events"]],
            },
        } for item in replicate["conditions"]],
        "common_shape_alignable_forward_event_count": replicate["common_shape_alignable_forward_event_count"],
        "token_history_aligned_forward_event_count": replicate["token_history_aligned_forward_event_count"],
    }


def write_boundary(error):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "ATTENTION_RELATIONAL_RESPONSE_EXPERIMENT_V1",
        "outcome": "UNRESOLVED_EXECUTION_BOUNDARY",
        "failed_capability": error.capability,
        "details": error.details,
        "authoritative_parent_commit": PARENT_COMMIT,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    RECEIPT_PATH.write_text(
        "OUTCOME\nUNRESOLVED_EXECUTION_BOUNDARY\n\nFAILED_CAPABILITY:\n"
        + error.capability + "\n\nDETAILS:\n"
        + json.dumps(error.details, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(RECEIPT_PATH.read_text(encoding="utf-8"), flush=True)


def run():
    guards_before = guard_authoritative_files()
    baseline_result, baseline_manifest = load_and_verify_baseline()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    RAW_DIR.mkdir(parents=True)

    slice14 = lab.load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(
        context, parent, slice14.LOCAL_CYCLE_BOUND, evaluate_gate=True
    )
    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise ExecutionBoundary("authoritative shared-V source boundary did not reproduce", {
            "result": parent.result,
        })

    kernel = context["kernel"]
    source_snapshot = torch.load(SOURCE_SNAPSHOT, map_location="cpu", weights_only=False)
    source_material = kernel.fingerprint_material(source_snapshot)
    source_fp = {key: source_material[key] for key in lab.FP_KEYS}
    if source_fp["sha256"] != EXPECTED_SOURCE_SHA256:
        raise ExecutionBoundary("persisted K_t content fingerprint failed", source_fp)
    _, sites = response.derive_sites(source_snapshot)
    actual_sites = [
        (site["j"], site["layer"], site["component"], site["native_coordinate"])
        for site in sites
    ]
    if actual_sites != EXPECTED_SITES:
        raise ExecutionBoundary("authoritative 12 intervention sites failed identity", {
            "actual": actual_sites, "expected": EXPECTED_SITES,
        })

    replicate_a = execute_replicate(
        "A", slice11, context, parent.shared_v, source_snapshot, source_fp,
        sites, baseline_result, baseline_manifest
    )
    replicate_b = execute_replicate(
        "B", slice11, context, parent.shared_v, source_snapshot, source_fp,
        sites, baseline_result, baseline_manifest, reference=replicate_a
    )

    complete_relation_reproduces = replicate_a["complete_scope"] == replicate_b["complete_scope"]
    token_relation_reproduces = replicate_a["token_aligned_scope"] == replicate_b["token_aligned_scope"]
    counts_reproduce = (
        replicate_a["common_shape_alignable_forward_event_count"]
        == replicate_b["common_shape_alignable_forward_event_count"]
        and replicate_a["token_history_aligned_forward_event_count"]
        == replicate_b["token_history_aligned_forward_event_count"]
    )
    if not (complete_relation_reproduces and token_relation_reproduces and counts_reproduce):
        raise ExecutionBoundary("event relation matrices or relational tests failed deterministic replication", {
            "complete_relation_reproduces": complete_relation_reproduces,
            "token_relation_reproduces": token_relation_reproduces,
            "event_counts_reproduce": counts_reproduce,
        })

    complete_pass = replicate_a["complete_scope"]["positive_gate"]
    token_pass = replicate_a["token_aligned_scope"]["positive_gate"]
    outcome = (
        "REPRODUCIBLE_ATTENTION_RELATIONAL_STRUCTURE_OBSERVED"
        if complete_pass
        else "NO_REPRODUCIBLE_ATTENTION_RELATIONAL_STRUCTURE_OBSERVED"
    )
    relation_payload = {
        "schema": "ATTENTION_EVENT_RELATION_MATRICES_V1",
        "intervention_order": list(range(12)),
        "profile_order": "layer-major then query-head-major",
        "complete_shape_alignable_common_prefix": replicate_a["complete_scope"],
        "token_history_aligned_prefix": replicate_a["token_aligned_scope"],
        "replication": "A and B exact identity",
    }
    RELATION_PATH.write_text(
        json.dumps(relation_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    if guard_authoritative_files() != guards_before:
        raise ExecutionBoundary("a prior authoritative source or artifact was mutated", {})

    result = {
        "schema": "ATTENTION_RELATIONAL_RESPONSE_EXPERIMENT_V1",
        "outcome": outcome,
        "authoritative_parent_commit": PARENT_COMMIT,
        "source_K_t": source_fp,
        "source_delta_t": {
            "token_ids": EXPECTED_DELTA_IDS,
            "count": len(EXPECTED_DELTA_IDS),
            "sha256": EXPECTED_DELTA_SHA256,
        },
        "intervention_count": 12,
        "intervention_sites": sites,
        "derived_attention_status": "OFFLINE_RECONSTRUCTED_FROM_DIRECTLY_OBSERVED_NATIVE_OPERANDS",
        "native_A_directly_materialized": "NO",
        "baseline_forward_event_count": 17,
        "common_shape_alignable_forward_event_count": replicate_a["common_shape_alignable_forward_event_count"],
        "token_history_aligned_forward_event_count": replicate_a["token_history_aligned_forward_event_count"],
        "nonzero_attention_response_counts_by_intervention": replicate_a["complete_scope"]["nonzero_attention_response_counts_by_intervention"],
        "complete_response_artifacts": [
            item["response"]["artifact"] for item in replicate_a["conditions"]
        ],
        "event_relation_matrices": {
            "path": RELATION_PATH.relative_to(ROOT).as_posix(),
            "file_sha256": sha256_file(RELATION_PATH),
        },
        "adjacent_event_relational_tests": replicate_a["complete_scope"]["adjacent_event_relational_tests"],
        "qualifying_temporal_runs": replicate_a["complete_scope"]["qualifying_temporal_runs"],
        "relational_persistence_complete_common_prefix": "PASS" if complete_pass else "NO",
        "relational_persistence_token_history_aligned_prefix": "PASS" if token_pass else "NO",
        "relational_persistence_before_endogenous_token_divergence": "PASS" if token_pass else "NO",
        "deterministic_replication": "PASS",
        "replications": {
            "A": strip_large_replication(replicate_a),
            "B": strip_large_replication(replicate_b),
        },
        "checks": {
            "ALL_INTERVENTION_SUPPORTS_EXACTLY_BOUNDED": "PASS",
            "ALL_DELTAS_IDENTICAL": "PASS",
            "ALL_INSTRUMENTED_TRAJECTORIES_EQUAL_UNINSTRUMENTED_CONTROLS": "PASS",
            "RUNTIME_PHYSICS_INVARIANT": "PASS",
            "COMPLETE_RESPONSE_ARTIFACTS_REPRODUCE": "PASS",
            "EVENT_RELATION_MATRICES_REPRODUCE": "PASS",
            "PERMUTATION_TEST_RESULTS_REPRODUCE": "PASS",
            "SOURCE_AUTHORITATIVE_FACT_MUTATED": "NO",
            "PREFERRED_HEAD_COUNT_TARGETED": "NO",
            "PREFERRED_NUMERICAL_TARGET_USED": "NO",
        },
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    receipt = [
        "OUTCOME", outcome, "",
        "AUTHORITATIVE_PARENT_COMMIT:", PARENT_COMMIT, "",
        "NEW_COMMIT:", "PENDING_PUBLICATION", "",
        "SOURCE_K_t:", canonical_json(source_fp), "",
        "SOURCE_DELTA_t:", canonical_json(result["source_delta_t"]), "",
        "INTERVENTION_COUNT:", "12", "",
        "INTERVENTION_SITES:", canonical_json(sites), "",
        "ALL_INTERVENTION_SUPPORTS_EXACTLY_BOUNDED:", "PASS", "",
        "ALL_DELTAS_IDENTICAL:", "PASS", "",
        "ALL_INSTRUMENTED_TRAJECTORIES_EQUAL_UNINSTRUMENTED_CONTROLS:", "PASS", "",
        "DERIVED_ATTENTION_STATUS:", result["derived_attention_status"], "",
        "NATIVE_A_DIRECTLY_MATERIALIZED:", "NO", "",
        "BASELINE_FORWARD_EVENT_COUNT:", "17", "",
        "COMMON_SHAPE_ALIGNABLE_FORWARD_EVENT_COUNT:", str(result["common_shape_alignable_forward_event_count"]), "",
        "TOKEN_HISTORY_ALIGNED_FORWARD_EVENT_COUNT:", str(result["token_history_aligned_forward_event_count"]), "",
        "NONZERO_ATTENTION_RESPONSE_COUNTS_BY_INTERVENTION:", canonical_json(result["nonzero_attention_response_counts_by_intervention"]), "",
        "COMPLETE_RESPONSE_ARTIFACTS:", canonical_json(result["complete_response_artifacts"]), "",
        "EVENT_RELATION_MATRICES:", canonical_json(result["event_relation_matrices"]), "",
        "ADJACENT_EVENT_RELATIONAL_TESTS:", canonical_json(result["adjacent_event_relational_tests"]), "",
        "QUALIFYING_TEMPORAL_RUNS:", canonical_json(result["qualifying_temporal_runs"]), "",
        "RELATIONAL_PERSISTENCE_COMPLETE_COMMON_PREFIX:", result["relational_persistence_complete_common_prefix"], "",
        "RELATIONAL_PERSISTENCE_TOKEN_HISTORY_ALIGNED_PREFIX:", result["relational_persistence_token_history_aligned_prefix"], "",
        "RELATIONAL_PERSISTENCE_BEFORE_ENDOGENOUS_TOKEN_DIVERGENCE:", result["relational_persistence_before_endogenous_token_divergence"], "",
        "DETERMINISTIC_REPLICATION:", "PASS", "",
        "SOURCE_AUTHORITATIVE_FACT_MUTATED:", "NO", "",
        "PREFERRED_HEAD_COUNT_TARGETED:", "NO", "",
        "PREFERRED_NUMERICAL_TARGET_USED:", "NO", "",
    ]
    RECEIPT_PATH.write_text("\n".join(receipt), encoding="utf-8")
    print(RECEIPT_PATH.read_text(encoding="utf-8"), flush=True)
    print(f"RESULT_JSON={RESULT_PATH}", flush=True)
    print(f"RELATION_MATRICES={RELATION_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except ExecutionBoundary as error:
        write_boundary(error)
        raise SystemExit(2)
    except Exception as error:
        boundary = ExecutionBoundary(
            "required attention-relational comparison could not be completed with the established mechanisms",
            {"error_type": error.__class__.__name__, "error": str(error)},
        )
        write_boundary(boundary)
        raise
