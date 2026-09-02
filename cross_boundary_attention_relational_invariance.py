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
import attention_relational_response_experiment as local
import kv_causal_intervention_lab as lab
import kv_causal_response_structure as response
import realized_attention_state_exposure as attention

PARENT_COMMIT = "e7bde59edf648b759ef0479627db29677245ea31"
CYCLES = (1, 4, 8)
BOUNDARY_NAMES = {1: "B1", 4: "B4", 8: "B8"}
PAIR_NAMES = (("B1", "B4"), ("B1", "B8"), ("B4", "B8"))
PERMUTATION_COUNT = 10000
LOCAL_ALPHA = 0.003125
CROSS_ALPHA = 0.0009803921568627451
EXPECTED_B8 = {
    "K_t": "8d5f495ed7c83b37201804e7bfea91f5b9481aa133fea015aa4ea15415aa0eaf",
    "delta_t": "7b5154e56e0e904f4820b0dfad049b8e40b9706a4a36143b01983b4ed6e7b78c",
    "K_t_plus_1": "04c99a9d3d5d3500a1eaf93e1b7ccade9cad2950c918ae691c85fc02a8c7713b",
}
OUT_DIR = ROOT / "evidence" / "cross_boundary_attention_relational_invariance"
TEMP_DIR = OUT_DIR / "temporary_raw"
RESULT_PATH = OUT_DIR / "cross_boundary_attention_relational_invariance.json"
RELATION_PATH = OUT_DIR / "cross_boundary_relation_matrices.json"
RECEIPT_PATH = OUT_DIR / "completion_receipt.txt"
AUTHORITATIVE_LOCAL_RESULT = (
    ROOT / "evidence" / "attention_relational_response" / "attention_relational_response.json"
)
AUTHORITATIVE_LOCAL_RELATIONS = (
    ROOT / "evidence" / "attention_relational_response" / "event_relation_matrices.json"
)
AUTHORITATIVE_BASELINE_MANIFEST = (
    ROOT / "evidence" / "realized_attention_state" / "attention_manifest.json"
)


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


def semantic_json_hash(domain, value):
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(value).encode("ascii"))
    return digest.hexdigest()


def fingerprint_snapshot(kernel, snapshot):
    material = kernel.fingerprint_material(snapshot)
    return {key: material[key] for key in lab.FP_KEYS}


def token_evidence(context, token_ids):
    return lab.token_evidence(context, [int(value) for value in token_ids])


def guarded_files():
    excluded_names = {
        "cross_boundary_attention_relational_invariance.py",
        "Verify Cross Boundary Attention Relational Invariance.ps1",
    }
    result = {}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if relative.parts[:2] == ("evidence", "cross_boundary_attention_relational_invariance"):
            continue
        if relative.name in excluded_names:
            continue
        result[relative.as_posix()] = sha256_file(path)
    return result


def capture_eight_cycle_history(slice11, context):
    kernel = context["kernel"]
    original = slice11.traverse
    captured = {}
    call_index = 0

    def observed_traverse(ctx, cache, shared_v, ids, label, origin):
        nonlocal call_index
        pre_snapshot = kernel.snapshot_cache(cache)
        traversal = original(ctx, cache, shared_v, ids, label, origin)
        post_snapshot = kernel.snapshot_cache(cache)
        cycle = call_index // 2 + 1
        ordinal = call_index % 2
        if ordinal == 1 and cycle in CYCLES:
            captured[cycle] = {
                "cycle": cycle,
                "ordinal": ordinal,
                "K_t_snapshot": pre_snapshot,
                "K_t_plus_1_snapshot": post_snapshot,
                "K_t": fingerprint_snapshot(kernel, pre_snapshot),
                "delta_t": token_evidence(context, ids),
                "K_t_plus_1": fingerprint_snapshot(kernel, post_snapshot),
                "generated_tokens": token_evidence(
                    context, traversal["generated_token_ids"]
                ),
            }
        call_index += 1
        return traversal

    slice11.traverse = observed_traverse
    try:
        parent = slice11.initialize_origin(context, passive_label=None)
        slice11.run_standing_wave(context, parent, 8, evaluate_gate=True)
    finally:
        slice11.traverse = original

    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise ExecutionBoundary(
            "independent 8-cycle parent history did not complete",
            {"result": parent.result},
        )
    if sorted(captured) != list(CYCLES):
        raise ExecutionBoundary(
            "observational history wrapper did not capture fixed cycle boundaries",
            {"captured_cycles": sorted(captured)},
        )
    return captured


def realize_prefix(slice11, context, cycle_count):
    parent = slice11.initialize_origin(context, passive_label=None)
    slice11.run_standing_wave(
        context, parent, cycle_count, evaluate_gate=True
    )
    if parent.result != "NO_RECURRENCE_WITHIN_BOUND":
        raise ExecutionBoundary(
            "independent parent prefix did not complete without recurrence",
            {"cycle_count": cycle_count, "result": parent.result},
        )
    boundary = parent.latest_boundary
    if boundary is None:
        raise ExecutionBoundary(
            "independent parent prefix exposed no latest native boundary",
            {"cycle_count": cycle_count},
        )
    if boundary["cycle"] != cycle_count - 1 or boundary["ordinal"] != 1:
        raise ExecutionBoundary(
            "latest native boundary did not match fixed completed-cycle position",
            {
                "cycle_count": cycle_count,
                "boundary_cycle": boundary["cycle"],
                "boundary_ordinal": boundary["ordinal"],
            },
        )
    kernel = context["kernel"]
    pre_snapshot = kernel.snapshot_cache(boundary["pre_native"])
    post_snapshot = kernel.snapshot_cache(boundary["post_native"])
    return {
        "name": BOUNDARY_NAMES[cycle_count],
        "cycle_count": cycle_count,
        "producer_instance_id": boundary["producer_instance_id"],
        "cycle_zero_based": boundary["cycle"],
        "ordinal": boundary["ordinal"],
        "pole": boundary["pole"],
        "K_t_snapshot": pre_snapshot,
        "K_t_plus_1_snapshot": post_snapshot,
        "K_t": fingerprint_snapshot(kernel, pre_snapshot),
        "delta_t": token_evidence(context, boundary["delta"]),
        "K_t_plus_1": fingerprint_snapshot(kernel, post_snapshot),
        "generated_tokens": token_evidence(
            context, boundary["traversal"]["generated_token_ids"]
        ),
        "shared_v": parent.shared_v,
    }


def boundary_equal(left, right):
    return (
        lab.snapshot_content_equal(
            left["K_t_snapshot"], right["K_t_snapshot"]
        )
        and left["K_t"] == right["K_t"]
        and left["delta_t"] == right["delta_t"]
        and lab.snapshot_content_equal(
            left["K_t_plus_1_snapshot"], right["K_t_plus_1_snapshot"]
        )
        and left["K_t_plus_1"] == right["K_t_plus_1"]
        and left["generated_tokens"] == right["generated_tokens"]
    )


def pairwise_identity(boundaries):
    result = {}
    for left_name, right_name in PAIR_NAMES:
        left = boundaries[left_name]
        right = boundaries[right_name]
        fields = {
            "K_t": left["K_t"]["sha256"] == right["K_t"]["sha256"],
            "delta_t": (
                left["delta_t"]["sha256"] == right["delta_t"]["sha256"]
                and left["delta_t"]["token_ids"] == right["delta_t"]["token_ids"]
            ),
            "K_t_plus_1": (
                left["K_t_plus_1"]["sha256"]
                == right["K_t_plus_1"]["sha256"]
            ),
        }
        fields["complete_ordered_triple"] = all(fields.values())
        result[f"{left_name}-{right_name}"] = fields
    return result


def persist_boundary_snapshots(boundaries):
    boundary_dir = OUT_DIR / "boundaries"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    for name, boundary in boundaries.items():
        pre_path = boundary_dir / f"{name}_K_t_snapshot.pt"
        post_path = boundary_dir / f"{name}_K_t_plus_1_snapshot.pt"
        torch.save(boundary["K_t_snapshot"], pre_path)
        torch.save(boundary["K_t_plus_1_snapshot"], post_path)
        pre_loaded = torch.load(pre_path, map_location="cpu", weights_only=False)
        post_loaded = torch.load(post_path, map_location="cpu", weights_only=False)
        if not lab.snapshot_content_equal(boundary["K_t_snapshot"], pre_loaded):
            raise ExecutionBoundary(
                "persisted native pre-state snapshot did not round-trip exactly",
                {"boundary": name},
            )
        if not lab.snapshot_content_equal(
            boundary["K_t_plus_1_snapshot"], post_loaded
        ):
            raise ExecutionBoundary(
                "persisted native post-state snapshot did not round-trip exactly",
                {"boundary": name},
            )
        boundary["snapshot_artifacts"] = {
            "K_t": {
                "path": pre_path.relative_to(ROOT).as_posix(),
                "file_sha256": sha256_file(pre_path),
                "byte_count": pre_path.stat().st_size,
            },
            "K_t_plus_1": {
                "path": post_path.relative_to(ROOT).as_posix(),
                "file_sha256": sha256_file(post_path),
                "byte_count": post_path.stat().st_size,
            },
        }


def realize_boundaries(replicate, slice11, context, reference=None):
    history = capture_eight_cycle_history(slice11, context)
    direct = {
        BOUNDARY_NAMES[cycle]: realize_prefix(slice11, context, cycle)
        for cycle in CYCLES
    }
    prefix_checks = {}
    for cycle in CYCLES:
        name = BOUNDARY_NAMES[cycle]
        prefix_checks[name] = boundary_equal(direct[name], history[cycle])
        if not prefix_checks[name]:
            raise ExecutionBoundary(
                "independent parent prefixes failed exact prefix reproduction",
                {"replicate": replicate, "boundary": name},
            )

    b8 = direct["B8"]
    anchor = {
        "K_t": b8["K_t"]["sha256"] == EXPECTED_B8["K_t"],
        "delta_t": b8["delta_t"]["sha256"] == EXPECTED_B8["delta_t"],
        "K_t_plus_1": (
            b8["K_t_plus_1"]["sha256"] == EXPECTED_B8["K_t_plus_1"]
        ),
    }
    anchor["PASS"] = all(anchor.values())
    if not anchor["PASS"]:
        raise ExecutionBoundary(
            "independent parent-boundary realization does not reproduce authoritative anchor boundary",
            {"replicate": replicate, "actual": {
                "K_t": b8["K_t"]["sha256"],
                "delta_t": b8["delta_t"]["sha256"],
                "K_t_plus_1": b8["K_t_plus_1"]["sha256"],
            }, "expected": EXPECTED_B8},
        )

    identities = pairwise_identity(direct)
    distinct_count = len({
        (
            item["K_t"]["sha256"],
            item["delta_t"]["sha256"],
            item["K_t_plus_1"]["sha256"],
        )
        for item in direct.values()
    })
    if distinct_count < 2:
        raise ExecutionBoundary(
            "mechanically selected parent trajectory does not furnish multiple distinct realized causal boundaries",
            {
                "replicate": replicate,
                "distinct_realized_boundary_count": distinct_count,
                "pairwise_identity": identities,
            },
        )

    if replicate == "A":
        persist_boundary_snapshots(direct)
    elif reference is not None:
        for name in BOUNDARY_NAMES.values():
            if not boundary_equal(direct[name], reference[name]):
                raise ExecutionBoundary(
                    "boundary realization failed deterministic replication",
                    {"boundary": name},
                )
            direct[name]["snapshot_artifacts"] = reference[name][
                "snapshot_artifacts"
            ]

    return {
        "boundaries": direct,
        "prefix_reproduction": prefix_checks,
        "B8_authoritative_anchor": anchor,
        "pairwise_identity": identities,
        "distinct_realized_boundary_count": distinct_count,
    }


def verify_delta(context, traversal, expected):
    supplied = token_evidence(
        context, traversal["perturbation_token_ids_supplied"]
    )
    observed = token_evidence(
        context, traversal["actual_first_forward_perturbation_ids"]
    )
    if supplied != expected or observed != expected:
        raise ExecutionBoundary(
            "within-boundary traversal did not preserve exact delta_t",
            {"expected": expected, "supplied": supplied, "observed": observed},
        )
    return expected


def execute_transition(
    replicate,
    boundary,
    site,
    observed,
    slice11,
    context,
    raw_path=None,
):
    kernel = context["kernel"]
    cache = lab.restore_dynamic_cache(
        kernel, context["model"], boundary["K_t_snapshot"]
    )
    restored = kernel.snapshot_cache(cache)
    if not lab.snapshot_content_equal(boundary["K_t_snapshot"], restored):
        raise ExecutionBoundary(
            "condition did not begin from exact native pre-state",
            {
                "replicate": replicate,
                "boundary": boundary["name"],
                "intervention": None if site is None else site["j"],
            },
        )
    if lab.fingerprint(kernel, cache) != boundary["K_t"]:
        raise ExecutionBoundary(
            "condition native pre-state fingerprint mismatch",
            {
                "replicate": replicate,
                "boundary": boundary["name"],
                "intervention": None if site is None else site["j"],
            },
        )

    if site is None:
        specification = {"condition": "baseline"}
    else:
        specification = response.apply_intervention(
            kernel, cache, boundary["K_t_snapshot"], site
        )
    start_snapshot = kernel.snapshot_cache(cache)
    start_fingerprint = lab.fingerprint(kernel, cache)
    intervention_label = (
        "BASELINE" if site is None else f"I{site['j']:02d}"
    )
    label = (
        f"CROSS_BOUNDARY_{replicate}_{boundary['name']}_"
        f"{intervention_label}_{'OBSERVED' if observed else 'CONTROL'}"
    )

    if observed:
        with attention.NativeAttentionCapture(
            context["model"], raw_path
        ) as capture:
            traversal = slice11.traverse(
                context,
                cache,
                boundary["shared_v"],
                boundary["delta_t"]["token_ids"],
                label,
                "exact boundary-specific native perturbation",
            )
        capture_manifest = capture.events
    else:
        traversal = slice11.traverse(
            context,
            cache,
            boundary["shared_v"],
            boundary["delta_t"]["token_ids"],
            label,
            "exact boundary-specific native perturbation",
        )
        capture_manifest = None

    delta = verify_delta(context, traversal, boundary["delta_t"])
    generated = token_evidence(context, traversal["generated_token_ids"])
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


def transition_divergence(control, observed):
    if control["specification"] != observed["specification"]:
        return "intervention specification"
    if control["start_fingerprint"] != observed["start_fingerprint"]:
        return "starting K_t fingerprint"
    if not lab.snapshot_content_equal(
        control["start_snapshot"], observed["start_snapshot"]
    ):
        return "starting K_t content"
    if control["delta"] != observed["delta"]:
        return "delta_t"
    if control["generated"] != observed["generated"]:
        return "generated token IDs"
    if control["post_fingerprint"] != observed["post_fingerprint"]:
        return "final complete KV fingerprint"
    if not lab.snapshot_content_equal(
        control["post_snapshot"], observed["post_snapshot"]
    ):
        return "final complete KV state"
    return None


def prepare_baseline(capture_manifest, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    events = []
    raw_hashes = []
    final_hashes = []
    maximum_difference = 0.0
    for record in capture_manifest:
        path = Path(record["path"])
        if sha256_file(path) != record["file_sha256"]:
            raise ExecutionBoundary(
                "baseline captured native operand file identity failed",
                {"event": record["forward_event_index"]},
            )
        event = torch.load(path, map_location="cpu", weights_only=False)
        for layer in event["layers"]:
            _, difference = attention.reconstruct_layer(layer)
            maximum_difference = max(maximum_difference, difference)
        final_hash = attention.semantic_event_hash(
            event, include_reconstruction=True
        )
        derived_path = target_dir / (
            f"baseline_event_{record['forward_event_index']:03d}.pt"
        )
        torch.save(event, derived_path)
        events.append({
            "forward_event_index": record["forward_event_index"],
            "path": str(derived_path),
            "file_sha256": sha256_file(derived_path),
            "raw_semantic_sha256": record["raw_semantic_sha256"],
            "final_semantic_sha256": final_hash,
        })
        raw_hashes.append(record["raw_semantic_sha256"])
        final_hashes.append(final_hash)
        del event
        gc.collect()
    return {
        "events": events,
        "raw_operand_semantic_hashes": raw_hashes,
        "final_semantic_hashes": final_hashes,
        "maximum_reconstruction_output_difference": maximum_difference,
    }


def process_intervention_capture(
    replicate,
    boundary_name,
    intervention,
    capture_manifest,
    baseline_manifest,
    generated_ids,
    baseline_generated_ids,
    save_artifact,
    reference_artifact=None,
):
    events = []
    raw_hashes = []
    maximum_difference = 0.0
    first_unalignable_event = None

    for raw_record, baseline_record in zip(
        capture_manifest, baseline_manifest["events"]
    ):
        event_index = raw_record["forward_event_index"]
        if event_index != baseline_record["forward_event_index"]:
            first_unalignable_event = event_index
            break
        raw_path = Path(raw_record["path"])
        if sha256_file(raw_path) != raw_record["file_sha256"]:
            raise ExecutionBoundary(
                "captured native operand file identity failed",
                {
                    "replicate": replicate,
                    "boundary": boundary_name,
                    "intervention": intervention,
                    "event": event_index,
                },
            )
        raw_hashes.append(raw_record["raw_semantic_sha256"])
        intervened = torch.load(
            raw_path, map_location="cpu", weights_only=False
        )
        for layer in intervened["layers"]:
            _, difference = attention.reconstruct_layer(layer)
            maximum_difference = max(maximum_difference, difference)
        baseline = torch.load(
            Path(baseline_record["path"]),
            map_location="cpu",
            weights_only=False,
        )
        if not local.alignable(intervened, baseline):
            first_unalignable_event = event_index
            del intervened, baseline
            gc.collect()
            break

        token_status = local.event_token_status(
            event_index, generated_ids, baseline_generated_ids
        )
        response_layers = []
        profile = []
        nonzero = False
        for left, right in zip(
            intervened["layers"], baseline["layers"]
        ):
            delta = (
                left["attention"] - right["attention"]
            ).detach().cpu().contiguous()
            nonzero = nonzero or bool(torch.count_nonzero(delta).item())
            for head in range(delta.shape[1]):
                profile.append(math.sqrt(float(torch.sum(
                    delta[0, head].to(torch.float64) ** 2
                ).item())))
            response_layers.append({
                "layer": left["layer_index"],
                "query_heads": int(delta.shape[1]),
                "query_positions": left["query_positions"],
                "key_positions": left["key_positions"],
                "response_tensor": delta,
            })
        if len(profile) != 270:
            raise ExecutionBoundary(
                "event response profile did not contain 270 canonical coordinates",
                {
                    "replicate": replicate,
                    "boundary": boundary_name,
                    "intervention": intervention,
                    "event": event_index,
                    "count": len(profile),
                },
            )
        magnitude = math.sqrt(sum(value * value for value in profile))
        normalized = (
            None
            if magnitude == 0.0
            else [value / magnitude for value in profile]
        )
        semantic_hash = local.semantic_response_hash(
            event_index, response_layers, token_status
        )
        event_input = (
            None if event_index == 0 else generated_ids[event_index - 1]
        )
        events.append({
            "forward_event_index": event_index,
            "layer_count": len(response_layers),
            "query_head_count": 9,
            "token_history_alignment_status": token_status,
            "generated_token_id_entering_event": event_input,
            "generated_token_history_before_or_at_event": (
                generated_ids[:event_index]
            ),
            "nonzero": nonzero,
            "response_l2_norm": magnitude,
            "energy_profile": profile,
            "normalized_energy_profile": normalized,
            "semantic_sha256": semantic_hash,
            "layers": response_layers,
        })
        del intervened, baseline
        gc.collect()

    trajectory_hash = local.semantic_artifact_hash(events)
    artifact_record = {
        "intervention": intervention,
        "represented_replicates": ["A", "B"],
        "event_count": len(events),
        "semantic_sha256": trajectory_hash,
    }
    if save_artifact:
        boundary_dir = OUT_DIR / "responses" / boundary_name
        boundary_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = boundary_dir / f"response_{intervention:02d}.pt"
        torch.save({
            "schema": "CROSS_BOUNDARY_DERIVED_ATTENTION_CAUSAL_RESPONSE_TRAJECTORY_V1",
            "boundary": boundary_name,
            "intervention": intervention,
            "represented_replicates": ["A", "B"],
            "derived_attention_status": (
                "OFFLINE_RECONSTRUCTED_FROM_DIRECTLY_OBSERVED_NATIVE_OPERANDS"
            ),
            "native_A_directly_materialized": "NO",
            "events": events,
        }, artifact_path)
        if artifact_path.stat().st_size >= 100_000_000:
            raise ExecutionBoundary(
                "complete response artifact exceeds GitHub individual-file limit",
                {
                    "path": artifact_path.relative_to(ROOT).as_posix(),
                    "bytes": artifact_path.stat().st_size,
                },
            )
        artifact_record.update({
            "path": artifact_path.relative_to(ROOT).as_posix(),
            "file_sha256": sha256_file(artifact_path),
            "byte_count": artifact_path.stat().st_size,
        })
    else:
        artifact_record = reference_artifact
        if artifact_record["semantic_sha256"] != trajectory_hash:
            raise ExecutionBoundary(
                "complete attention response failed deterministic replication",
                {
                    "boundary": boundary_name,
                    "intervention": intervention,
                },
            )

    summaries = [
        {key: value for key, value in event.items() if key != "layers"}
        for event in events
    ]
    return {
        "events": summaries,
        "artifact": artifact_record,
        "raw_operand_semantic_hashes": raw_hashes,
        "first_unalignable_event": first_unalignable_event,
        "maximum_reconstruction_output_difference": maximum_difference,
    }


def execute_boundary_experiment(
    replicate,
    boundary,
    slice11,
    context,
    authoritative_baseline_manifest,
    authoritative_relations,
    reference=None,
):
    name = boundary["name"]
    _, sites = response.derive_sites(boundary["K_t_snapshot"])
    baseline_raw = TEMP_DIR / replicate / name / "baseline_raw"
    baseline_control = execute_transition(
        replicate, boundary, None, False, slice11, context
    )
    baseline_observed = execute_transition(
        replicate,
        boundary,
        None,
        True,
        slice11,
        context,
        baseline_raw,
    )
    divergence = transition_divergence(
        baseline_control, baseline_observed
    )
    if divergence is not None:
        raise ExecutionBoundary(
            "attention observer is not causally invariant for boundary baseline",
            {"replicate": replicate, "boundary": name, "field": divergence},
        )
    for candidate in (baseline_control, baseline_observed):
        if (
            candidate["generated"] != boundary["generated_tokens"]
            or candidate["post_fingerprint"] != boundary["K_t_plus_1"]
            or not lab.snapshot_content_equal(
                candidate["post_snapshot"],
                boundary["K_t_plus_1_snapshot"],
            )
        ):
            raise ExecutionBoundary(
                "selected native boundary cannot be replayed exactly",
                {"replicate": replicate, "boundary": name},
            )

    baseline_derived_dir = (
        TEMP_DIR / replicate / name / "baseline_derived"
    )
    baseline_manifest = prepare_baseline(
        baseline_observed["capture_manifest"], baseline_derived_dir
    )
    if name == "B8":
        authoritative_hashes = [
            item["final_semantic_sha256"]
            for item in authoritative_baseline_manifest["events"]
        ]
        if baseline_manifest["final_semantic_hashes"] != authoritative_hashes:
            raise ExecutionBoundary(
                "B8 native attention baseline did not reproduce authoritative semantic identities",
                {
                    "replicate": replicate,
                    "actual": baseline_manifest["final_semantic_hashes"],
                    "expected": authoritative_hashes,
                },
            )

    conditions = []
    for site in sites:
        raw_path = (
            TEMP_DIR / replicate / name / f"I{site['j']:02d}_raw"
        )
        control = execute_transition(
            replicate, boundary, site, False, slice11, context
        )
        observed = execute_transition(
            replicate,
            boundary,
            site,
            True,
            slice11,
            context,
            raw_path,
        )
        divergence = transition_divergence(control, observed)
        if divergence is not None:
            raise ExecutionBoundary(
                "attention observer is not causally invariant for all cross-boundary intervention conditions",
                {
                    "replicate": replicate,
                    "boundary": name,
                    "intervention": site["j"],
                    "field": divergence,
                },
            )
        processed = process_intervention_capture(
            replicate,
            name,
            site["j"],
            observed["capture_manifest"],
            baseline_manifest,
            observed["generated"]["token_ids"],
            baseline_observed["generated"]["token_ids"],
            save_artifact=(replicate == "A"),
            reference_artifact=(
                None
                if reference is None
                else reference["conditions"][site["j"]]["response"][
                    "artifact"
                ]
            ),
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
                "specification": (
                    condition["specification"] == prior["specification"]
                ),
                "generated": condition["generated"] == prior["generated"],
                "start_fingerprint": (
                    condition["start_fingerprint"]
                    == prior["start_fingerprint"]
                ),
                "final_fingerprint": (
                    condition["final_fingerprint"]
                    == prior["final_fingerprint"]
                ),
                "raw_operand_semantic_hashes": (
                    processed["raw_operand_semantic_hashes"]
                    == prior["response"]["raw_operand_semantic_hashes"]
                ),
                "events": (
                    processed["events"] == prior["response"]["events"]
                ),
            }
            if not all(checks.values()):
                raise ExecutionBoundary(
                    "boundary-local attention response failed deterministic replication",
                    {
                        "boundary": name,
                        "intervention": site["j"],
                        "checks": checks,
                    },
                )
        conditions.append(condition)
        del control, observed
        gc.collect()

    common_count = min(
        len(item["response"]["events"]) for item in conditions
    )
    token_aligned_count = 0
    for event in range(common_count):
        if all(
            item["response"]["events"][event][
                "token_history_alignment_status"
            ] == "TOKEN_HISTORY_ALIGNED"
            for item in conditions
        ):
            token_aligned_count += 1
        else:
            break

    local_replicate = {
        "replicate": replicate,
        "conditions": conditions,
    }
    complete_scope = local.analyze_scope(
        local_replicate, common_count
    )
    token_scope = local.analyze_scope(
        local_replicate, token_aligned_count
    )
    if name == "B8":
        authoritative_scope = authoritative_relations[
            "complete_shape_alignable_common_prefix"
        ]
        if complete_scope != authoritative_scope:
            raise ExecutionBoundary(
                "B8 event relation matrices or adjacent-event tests did not reproduce authoritative local result",
                {"replicate": replicate},
            )

    if baseline_raw.exists():
        shutil.rmtree(baseline_raw)
    if baseline_derived_dir.exists():
        shutil.rmtree(baseline_derived_dir)

    return {
        "name": name,
        "canonical_scalar_count_N": response.derive_sites(
            boundary["K_t_snapshot"]
        )[0],
        "intervention_sites": sites,
        "baseline": {
            "generated": baseline_observed["generated"],
            "final_fingerprint": baseline_observed["post_fingerprint"],
            "raw_operand_semantic_hashes": baseline_manifest[
                "raw_operand_semantic_hashes"
            ],
            "final_semantic_hashes": baseline_manifest[
                "final_semantic_hashes"
            ],
            "maximum_reconstruction_output_difference": baseline_manifest[
                "maximum_reconstruction_output_difference"
            ],
            "observer_control": "PASS",
            "replay_exact": "PASS",
        },
        "conditions": conditions,
        "common_shape_alignable_forward_event_count": common_count,
        "token_history_aligned_forward_event_count": token_aligned_count,
        "complete_scope": complete_scope,
        "token_aligned_scope": token_scope,
        "local_relational_persistence": (
            "PASS" if complete_scope["positive_gate"] else "NO"
        ),
    }


def cross_event_test(left_matrix, right_matrix, event):
    left = local.upper_triangle(left_matrix)
    right = local.upper_triangle(right_matrix)
    observed = local.pearson(left, right)
    if observed is None:
        return {
            "event": event,
            "eligible_all_twelve_nonzero_both_boundaries": True,
            "r": "UNDEFINED_DEGENERATE_RELATION",
            "p": "UNDEFINED_DEGENERATE_RELATION",
            "threshold": CROSS_ALPHA,
            "passes": False,
        }
    generator = random.Random(0)
    labels = list(range(12))
    exceed = 0
    for _ in range(PERMUTATION_COUNT):
        permutation = labels[:]
        generator.shuffle(permutation)
        permuted = [
            right_matrix[permutation[i]][permutation[j]]
            for i in range(12)
            for j in range(i + 1, 12)
        ]
        candidate = local.pearson(left, permuted)
        if candidate is not None and candidate >= observed:
            exceed += 1
    p_value = (1 + exceed) / 10001
    return {
        "event": event,
        "eligible_all_twelve_nonzero_both_boundaries": True,
        "r": observed,
        "p": p_value,
        "permutation_count": PERMUTATION_COUNT,
        "prng": "Python random.Random(0)",
        "threshold": CROSS_ALPHA,
        "passes": p_value <= CROSS_ALPHA,
    }


def three_event_runs(tests):
    by_event = {item["event"]: item for item in tests}
    runs = []
    for event in sorted(by_event):
        if (
            by_event[event]["passes"]
            and event + 1 in by_event
            and by_event[event + 1]["passes"]
            and event + 2 in by_event
            and by_event[event + 2]["passes"]
        ):
            runs.append([event, event + 1, event + 2])
    return runs


def cross_analysis(boundary_results):
    pair_results = {}
    runs_by_pair = {}
    for left_name, right_name in PAIR_NAMES:
        left = boundary_results[left_name]["complete_scope"]
        right = boundary_results[right_name]["complete_scope"]
        event_count = min(left["event_count"], right["event_count"])
        tests = []
        eligible_events = []
        for event in range(event_count):
            if left["eligible_events"][event] and right["eligible_events"][event]:
                eligible_events.append(event)
                tests.append(cross_event_test(
                    left["event_relation_matrices"][event],
                    right["event_relation_matrices"][event],
                    event,
                ))
        pair_name = f"{left_name}-{right_name}"
        runs = three_event_runs(tests)
        pair_results[pair_name] = {
            "eligible_shared_forward_event_ordinals": eligible_events,
            "event_tests": tests,
            "qualifying_runs": runs,
        }
        runs_by_pair[pair_name] = runs

    shared = None
    for pair_name in (
        "B1-B4",
        "B1-B8",
        "B4-B8",
    ):
        current = {tuple(run) for run in runs_by_pair[pair_name]}
        shared = current if shared is None else shared & current
    shared_runs = [list(run) for run in sorted(shared or set())]

    multi = {
        name: result["complete_scope"][
            "nontrivial_multi_intervention_response"
        ]
        for name, result in boundary_results.items()
    }
    positive = (
        all(multi.values())
        and all(bool(runs_by_pair[name]) for name in runs_by_pair)
        and bool(shared_runs)
    )
    return {
        "threshold": CROSS_ALPHA,
        "permutation_count": PERMUTATION_COUNT,
        "prng": "Python random.Random(0)",
        "pairs": pair_results,
        "qualifying_runs_by_boundary_pair": runs_by_pair,
        "shared_qualifying_temporal_runs": shared_runs,
        "multi_intervention_response_by_boundary": multi,
        "positive_gate": positive,
    }


def public_boundary(boundary):
    return {
        "name": boundary["name"],
        "cycle_count": boundary["cycle_count"],
        "producer_instance_id": boundary["producer_instance_id"],
        "cycle_zero_based": boundary["cycle_zero_based"],
        "ordinal": boundary["ordinal"],
        "pole": boundary["pole"],
        "K_t": boundary["K_t"],
        "delta_t": boundary["delta_t"],
        "K_t_plus_1": boundary["K_t_plus_1"],
        "generated_tokens": boundary["generated_tokens"],
        "snapshot_artifacts": boundary["snapshot_artifacts"],
    }


def projection(realization, boundary_results, cross):
    return {
        "boundaries": {
            name: public_boundary(boundary)
            for name, boundary in realization["boundaries"].items()
        },
        "prefix_reproduction": realization["prefix_reproduction"],
        "B8_authoritative_anchor": realization[
            "B8_authoritative_anchor"
        ],
        "pairwise_identity": realization["pairwise_identity"],
        "distinct_realized_boundary_count": realization[
            "distinct_realized_boundary_count"
        ],
        "boundary_results": boundary_results,
        "cross_boundary": cross,
    }


def strip_layers_and_profiles(value):
    if isinstance(value, list):
        return [strip_layers_and_profiles(item) for item in value]
    if isinstance(value, dict):
        return {
            key: strip_layers_and_profiles(item)
            for key, item in value.items()
            if key not in ("energy_profile", "normalized_energy_profile")
        }
    return value


def write_boundary(error):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "CROSS_BOUNDARY_ATTENTION_RELATIONAL_INVARIANCE_V1",
        "outcome": "UNRESOLVED_EXECUTION_BOUNDARY",
        "failed_capability": error.capability,
        "details": error.details,
        "authoritative_parent_commit": PARENT_COMMIT,
    }
    RESULT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    receipt = (
        "OUTCOME\nUNRESOLVED_EXECUTION_BOUNDARY\n\n"
        "FAILED_CAPABILITY:\n"
        + error.capability
        + "\n\nDETAILS:\n"
        + json.dumps(error.details, indent=2, sort_keys=True)
        + "\n"
    )
    RECEIPT_PATH.write_text(receipt, encoding="utf-8")
    print(receipt, flush=True)


def run():
    guards_before = guarded_files()
    authoritative_local_result = json.loads(
        AUTHORITATIVE_LOCAL_RESULT.read_text(encoding="utf-8")
    )
    authoritative_relations = json.loads(
        AUTHORITATIVE_LOCAL_RELATIONS.read_text(encoding="utf-8")
    )
    authoritative_baseline_manifest = json.loads(
        AUTHORITATIVE_BASELINE_MANIFEST.read_text(encoding="utf-8")
    )
    if (
        authoritative_local_result["outcome"]
        != "REPRODUCIBLE_ATTENTION_RELATIONAL_STRUCTURE_OBSERVED"
    ):
        raise ExecutionBoundary(
            "authoritative B8 local relational result is unavailable",
            {"outcome": authoritative_local_result.get("outcome")},
        )

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    TEMP_DIR.mkdir(parents=True)

    slice14 = lab.load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()

    realization_a = realize_boundaries(
        "A", slice11, context
    )
    results_a = {}
    for name in BOUNDARY_NAMES.values():
        results_a[name] = execute_boundary_experiment(
            "A",
            realization_a["boundaries"][name],
            slice11,
            context,
            authoritative_baseline_manifest,
            authoritative_relations,
        )
    cross_a = cross_analysis(results_a)

    realization_b = realize_boundaries(
        "B",
        slice11,
        context,
        reference=realization_a["boundaries"],
    )
    results_b = {}
    for name in BOUNDARY_NAMES.values():
        results_b[name] = execute_boundary_experiment(
            "B",
            realization_b["boundaries"][name],
            slice11,
            context,
            authoritative_baseline_manifest,
            authoritative_relations,
            reference=results_a[name],
        )
    cross_b = cross_analysis(results_b)

    projection_a = projection(realization_a, results_a, cross_a)
    projection_b = projection(realization_b, results_b, cross_b)
    deterministic = projection_a == projection_b
    if not deterministic:
        raise ExecutionBoundary(
            "complete cross-boundary experiment failed deterministic replication",
            {
                "replicate_A_sha256": semantic_json_hash(
                    "CROSS_BOUNDARY_REPLICATE_V1", projection_a
                ),
                "replicate_B_sha256": semantic_json_hash(
                    "CROSS_BOUNDARY_REPLICATE_V1", projection_b
                ),
            },
        )

    outcome = (
        "CROSS_BOUNDARY_ATTENTION_RELATIONAL_INVARIANT_OBSERVED"
        if cross_a["positive_gate"]
        else "NO_CROSS_BOUNDARY_ATTENTION_RELATIONAL_INVARIANT_OBSERVED"
    )

    relation_payload = {
        "schema": "CROSS_BOUNDARY_ATTENTION_RELATION_MATRICES_V1",
        "intervention_order": list(range(12)),
        "profile_order": "layer-major then query-head-major",
        "within_boundary": {
            name: {
                "event_relation_matrices": result["complete_scope"][
                    "event_relation_matrices"
                ],
                "eligible_events": result["complete_scope"][
                    "eligible_events"
                ],
                "adjacent_event_relational_tests": result[
                    "complete_scope"
                ]["adjacent_event_relational_tests"],
                "qualifying_temporal_runs": result["complete_scope"][
                    "qualifying_temporal_runs"
                ],
            }
            for name, result in results_a.items()
        },
        "cross_boundary": cross_a,
        "replication": "A and B exact identity",
    }
    RELATION_PATH.write_text(
        json.dumps(relation_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    guards_after = guarded_files()
    if guards_before != guards_after:
        changed = sorted(
            set(guards_before) | set(guards_after)
        )
        changed = [
            path
            for path in changed
            if guards_before.get(path) != guards_after.get(path)
        ]
        raise ExecutionBoundary(
            "a prior authoritative source or artifact was mutated",
            {"changed_paths": changed},
        )

    result = {
        "schema": "CROSS_BOUNDARY_ATTENTION_RELATIONAL_INVARIANCE_V1",
        "outcome": outcome,
        "authoritative_parent_commit": PARENT_COMMIT,
        "new_commit": "PENDING_PUBLICATION",
        "boundary_selection_rule": (
            "fresh deterministic parent prefixes at cycles [1,4,8]"
        ),
        "boundaries": {
            name: public_boundary(boundary)
            for name, boundary in realization_a["boundaries"].items()
        },
        "B8_authoritative_anchor": "PASS",
        "prefix_reproduction": "PASS",
        "prefix_reproduction_detail": realization_a[
            "prefix_reproduction"
        ],
        "pairwise_boundary_identity": realization_a[
            "pairwise_identity"
        ],
        "distinct_realized_boundary_count": realization_a[
            "distinct_realized_boundary_count"
        ],
        "intervention_count_per_boundary": 12,
        "intervention_selection_rule": (
            "q_j=floor(((2j+1)N_c)/24), j=0..11"
        ),
        "all_intervention_supports_exactly_bounded": "PASS",
        "all_within_boundary_deltas_exact": "PASS",
        "all_instrumented_trajectories_equal_controls": "PASS",
        "derived_attention_status": (
            "OFFLINE_RECONSTRUCTED_FROM_DIRECTLY_OBSERVED_NATIVE_OPERANDS"
        ),
        "local_relational_persistence": {
            name: item["local_relational_persistence"]
            for name, item in results_a.items()
        },
        "cross_boundary_test_threshold": CROSS_ALPHA,
        "cross_boundary_event_tests": cross_a["pairs"],
        "qualifying_runs_by_boundary_pair": cross_a[
            "qualifying_runs_by_boundary_pair"
        ],
        "shared_qualifying_temporal_runs": cross_a[
            "shared_qualifying_temporal_runs"
        ],
        "deterministic_replication": "PASS",
        "replicate_semantic_sha256": semantic_json_hash(
            "CROSS_BOUNDARY_REPLICATE_V1", projection_a
        ),
        "replications": {
            "A": strip_layers_and_profiles(projection_a),
            "B": {
                "semantic_sha256": semantic_json_hash(
                    "CROSS_BOUNDARY_REPLICATE_V1", projection_b
                ),
                "exact_identity_with_A": True,
            },
        },
        "complete_relation_artifact": {
            "path": RELATION_PATH.relative_to(ROOT).as_posix(),
            "file_sha256": sha256_file(RELATION_PATH),
            "byte_count": RELATION_PATH.stat().st_size,
        },
        "source_authoritative_fact_mutated": "NO",
        "preferred_head_count_targeted": "NO",
        "preferred_numerical_target_used": "NO",
        "semantic_interpretation_performed": "NO",
    }
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    receipt_lines = [
        "OUTCOME",
        outcome,
        "",
        "AUTHORITATIVE_PARENT_COMMIT:",
        PARENT_COMMIT,
        "",
        "NEW_COMMIT:",
        "PENDING_PUBLICATION",
        "",
        "BOUNDARY_SELECTION_RULE:",
        result["boundary_selection_rule"],
        "",
        "BOUNDARIES:",
        canonical_json(result["boundaries"]),
        "",
        "B8_AUTHORITATIVE_ANCHOR:",
        "PASS",
        "",
        "PREFIX_REPRODUCTION:",
        "PASS",
        "",
        "DISTINCT_REALIZED_BOUNDARY_COUNT:",
        str(result["distinct_realized_boundary_count"]),
        "",
        "INTERVENTION_COUNT_PER_BOUNDARY:",
        "12",
        "",
        "INTERVENTION_SELECTION_RULE:",
        result["intervention_selection_rule"],
        "",
        "ALL_INTERVENTION_SUPPORTS_EXACTLY_BOUNDED:",
        "PASS",
        "",
        "ALL_WITHIN_BOUNDARY_DELTAS_EXACT:",
        "PASS",
        "",
        "ALL_INSTRUMENTED_TRAJECTORIES_EQUAL_CONTROLS:",
        "PASS",
        "",
        "DERIVED_ATTENTION_STATUS:",
        result["derived_attention_status"],
        "",
        "LOCAL_RELATIONAL_PERSISTENCE:",
        canonical_json(result["local_relational_persistence"]),
        "",
        "CROSS_BOUNDARY_TEST_THRESHOLD:",
        str(CROSS_ALPHA),
        "",
        "CROSS_BOUNDARY_EVENT_TESTS:",
        canonical_json(result["cross_boundary_event_tests"]),
        "",
        "QUALIFYING_RUNS_BY_BOUNDARY_PAIR:",
        canonical_json(result["qualifying_runs_by_boundary_pair"]),
        "",
        "SHARED_QUALIFYING_TEMPORAL_RUNS:",
        canonical_json(result["shared_qualifying_temporal_runs"]),
        "",
        "DETERMINISTIC_REPLICATION:",
        "PASS",
        "",
        "SOURCE_AUTHORITATIVE_FACT_MUTATED:",
        "NO",
        "",
        "PREFERRED_HEAD_COUNT_TARGETED:",
        "NO",
        "",
        "PREFERRED_NUMERICAL_TARGET_USED:",
        "NO",
        "",
        "SEMANTIC_INTERPRETATION_PERFORMED:",
        "NO",
        "",
    ]
    RECEIPT_PATH.write_text(
        "\n".join(receipt_lines), encoding="utf-8"
    )
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
            "required cross-boundary attention-relational experiment could not be completed with established mechanisms",
            {
                "error_type": error.__class__.__name__,
                "error": str(error),
            },
        )
        write_boundary(boundary)
        raise
