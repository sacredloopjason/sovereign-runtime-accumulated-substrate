import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import transformers

LAB_PATH = Path(__file__).resolve().parent / "kv_causal_intervention_lab.py"
LAB_SPEC = importlib.util.spec_from_file_location("kv_causal_response_structure_inherited_lab", LAB_PATH)
if LAB_SPEC is None or LAB_SPEC.loader is None:
    raise RuntimeError(f"Cannot load inherited KV causal-intervention lab: {LAB_PATH}")
lab = importlib.util.module_from_spec(LAB_SPEC)
sys.modules[LAB_SPEC.name] = lab
LAB_SPEC.loader.exec_module(lab)


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_EVIDENCE_DIR = PROJECT_DIR / "evidence" / "kv_causal_intervention_lab"
SOURCE_SNAPSHOT = SOURCE_EVIDENCE_DIR / "K_t_snapshot.pt"
SOURCE_JSON = SOURCE_EVIDENCE_DIR / "kv_causal_intervention_lab.json"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "kv_causal_response_structure"
EVIDENCE_JSON = EVIDENCE_DIR / "kv_causal_response_structure.json"
RECEIPT_PATH = EVIDENCE_DIR / "completion_receipt.txt"
INTERVENTION_COUNT = 12
COMPONENTS = ("key", "value")
EXPECTED_SOURCE_SHA256 = "8d5f495ed7c83b37201804e7bfea91f5b9481aa133fea015aa4ea15415aa0eaf"
EXPECTED_DELTA_IDS = [
    10199, 15799, 38, 79, 6493, 3010, 198, 10199,
    15799, 38, 79, 6493, 3010, 198, 10199, 15799,
]
EXPECTED_DELTA_SHA256 = "7b5154e56e0e904f4820b0dfad049b8e40b9706a4a36143b01983b4ed6e7b78c"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_tensor_digest_update(digest, layer, component, tensor):
    header = {
        "layer": layer,
        "component": component,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    raw = lab.tensor_bytes(tensor)
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def response_identity(response):
    digest = hashlib.sha256()
    digest.update(b"KV_CAUSAL_RESPONSE_TENSOR_STRUCTURE_V1")
    for layer, material in enumerate(response["layers"]):
        for component in COMPONENTS:
            canonical_tensor_digest_update(digest, layer, component, material[component])
    return digest.hexdigest()


def response_equal(left, right):
    if len(left["layers"]) != len(right["layers"]):
        return False
    for left_layer, right_layer in zip(left["layers"], right["layers"]):
        for component in COMPONENTS:
            a = left_layer[component]
            b = right_layer[component]
            if (
                a.dtype != b.dtype
                or tuple(a.shape) != tuple(b.shape)
                or lab.tensor_bytes(a) != lab.tensor_bytes(b)
            ):
                return False
    return True


def subtract_snapshots(intervened, baseline):
    if len(intervened["layers"]) != len(baseline["layers"]):
        raise RuntimeError("response snapshots have different layer counts")
    layers = []
    for left_layer, right_layer in zip(intervened["layers"], baseline["layers"]):
        output = {}
        for component in COMPONENTS:
            left = left_layer[component]
            right = right_layer[component]
            if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
                raise RuntimeError("response tensor shape or dtype mismatch")
            output[component] = (left - right).detach().cpu().contiguous()
        layers.append(output)
    return {"layers": layers}


def response_descriptors(response):
    component_energy = []
    nonzero_scalar_count = 0
    layers_with_nonzero = 0
    components_with_nonzero = 0
    total_squared = 0.0
    for material in response["layers"]:
        layer_nonzero = False
        for component in COMPONENTS:
            tensor = material[component]
            count = int(torch.count_nonzero(tensor).item())
            energy_squared = float(torch.sum(tensor.to(torch.float64) ** 2).item())
            energy = math.sqrt(energy_squared)
            component_energy.append(energy)
            nonzero_scalar_count += count
            total_squared += energy_squared
            if count:
                layer_nonzero = True
                components_with_nonzero += 1
        if layer_nonzero:
            layers_with_nonzero += 1
    return {
        "complete_response_sha256": response_identity(response),
        "nonzero_scalar_count": nonzero_scalar_count,
        "layers_with_nonzero_response": layers_with_nonzero,
        "components_with_nonzero_response": components_with_nonzero,
        "l2_norm": math.sqrt(total_squared),
        "component_energy_profile": component_energy,
    }


def persist_response(response, replicate, intervention, boundary):
    path = EVIDENCE_DIR / f"response_{intervention:02d}_{boundary}.pt"
    payload = {
        "schema": "KV_CAUSAL_RESPONSE_TENSOR_ARTIFACT_V1",
        "represented_replicates": ["A", "B"],
        "created_from_replicate": replicate,
        "intervention": intervention,
        "boundary": boundary,
        "representation": "complete dense native-dtype tensors in canonical layer/key/value order",
        "layers": response["layers"],
    }
    torch.save(payload, path)
    if path.stat().st_size >= 100_000_000:
        raise RuntimeError(f"response artifact exceeds GitHub individual-file limit: {path}")
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    restored = {"layers": reloaded["layers"]}
    if not response_equal(response, restored):
        raise RuntimeError(f"response artifact did not round-trip exactly: {path}")
    return {
        "path": path.relative_to(PROJECT_DIR).as_posix(),
        "file_sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def load_response(artifact):
    payload = torch.load(PROJECT_DIR / artifact["path"], map_location="cpu", weights_only=False)
    return {"layers": payload["layers"]}


def flat_coordinate(flat_index, shape):
    coordinate = []
    remaining = flat_index
    for size in reversed(shape):
        coordinate.append(remaining % size)
        remaining //= size
    if remaining:
        raise RuntimeError("flat coordinate exceeds tensor shape")
    return tuple(reversed(coordinate))


def derive_sites(snapshot):
    universe = []
    total = 0
    for layer, material in enumerate(snapshot["layers"]):
        for component in COMPONENTS:
            tensor = material[component]
            if tensor is None:
                continue
            start = total
            total += tensor.numel()
            universe.append((start, total, layer, component, tuple(tensor.shape)))
    sites = []
    for j in range(INTERVENTION_COUNT):
        q = ((2 * j + 1) * total) // 24
        match = next((entry for entry in universe if entry[0] <= q < entry[1]), None)
        if match is None:
            raise RuntimeError(f"canonical scalar index did not map to a tensor: {q}")
        start, _, layer, component, shape = match
        local_flat_index = q - start
        sites.append({
            "j": j,
            "q_j": q,
            "layer": layer,
            "component": component,
            "native_coordinate": list(flat_coordinate(local_flat_index, shape)),
            "component_flat_index": local_flat_index,
        })
    return total, sites


def exact_support(source_snapshot, intervened_snapshot, site, before, after):
    changed_components = []
    non_support_changed_byte_count = 0
    support_changed_byte_offsets = []
    element_size = before.element_size()
    support_start = site["component_flat_index"] * element_size
    support_end = support_start + element_size

    for layer, (left_layer, right_layer) in enumerate(
        zip(source_snapshot["layers"], intervened_snapshot["layers"])
    ):
        for component in COMPONENTS:
            left = left_layer[component]
            right = right_layer[component]
            left_raw = lab.tensor_bytes(left)
            right_raw = lab.tensor_bytes(right)
            differing = [
                offset
                for offset, pair in enumerate(zip(left_raw, right_raw))
                if pair[0] != pair[1]
            ]
            if differing:
                changed_components.append({
                    "layer": layer,
                    "component": component,
                    "changed_byte_offsets": differing,
                })
            if layer == site["layer"] and component == site["component"]:
                support_changed_byte_offsets = [
                    offset for offset in differing if support_start <= offset < support_end
                ]
                non_support_changed_byte_count += sum(
                    offset < support_start or offset >= support_end for offset in differing
                )
            else:
                non_support_changed_byte_count += len(differing)

    exact = (
        len(changed_components) == 1
        and changed_components[0]["layer"] == site["layer"]
        and changed_components[0]["component"] == site["component"]
        and bool(support_changed_byte_offsets)
        and non_support_changed_byte_count == 0
        and lab.tensor_bytes(before) != lab.tensor_bytes(after)
    )
    return {
        "INTERVENTION_SUPPORT_EXACTLY_BOUNDED": exact,
        "CHANGED_LAYER": site["layer"],
        "CHANGED_COMPONENT": site["component"],
        "CHANGED_COORDINATE": site["native_coordinate"],
        "declared_support_byte_range_in_contiguous_tensor": [support_start, support_end],
        "support_changed_byte_offsets": support_changed_byte_offsets,
        "NON_INTERVENED_KV_CHANGED_BYTE_COUNT": non_support_changed_byte_count,
        "before_bytes_hex": lab.tensor_bytes(before).hex(),
        "after_bytes_hex": lab.tensor_bytes(after).hex(),
    }


def apply_intervention(kernel, cache, source_snapshot, site):
    native_layer = cache.layers[site["layer"]]
    target = getattr(native_layer, "keys" if site["component"] == "key" else "values")
    coordinate = tuple(site["native_coordinate"])
    before = target[coordinate].detach().cpu().clone()
    one = torch.ones((), dtype=before.dtype)
    after = before + one
    operation = "x -> x + 1.0"
    fallback = False
    if not bool(torch.isfinite(after).item()) or lab.tensor_bytes(before) == lab.tensor_bytes(after):
        after = before - one
        operation = "x -> x - 1.0"
        fallback = True
    if not bool(torch.isfinite(after).item()) or lab.tensor_bytes(before) == lab.tensor_bytes(after):
        raise RuntimeError(f"neither permitted scalar operation produced a finite byte-distinct value at site {site['j']}")
    target[coordinate] = after.to(device=target.device)
    observed = target[coordinate].detach().cpu().clone()
    if lab.tensor_bytes(observed) != lab.tensor_bytes(after):
        raise RuntimeError("scalar intervention did not survive native assignment")
    intervened_snapshot = kernel.snapshot_cache(cache)
    support = exact_support(source_snapshot, intervened_snapshot, site, before, observed)
    if not support["INTERVENTION_SUPPORT_EXACTLY_BOUNDED"]:
        raise RuntimeError(f"intervention support was not exactly bounded at site {site['j']}")
    specification = dict(site)
    specification.update({
        "operation": operation,
        "fallback_used": fallback,
        "before": float(before.item()),
        "after": float(observed.item()),
        "before_bytes_hex": lab.tensor_bytes(before).hex(),
        "after_bytes_hex": lab.tensor_bytes(observed).hex(),
        "support": support,
    })
    return specification


def traverse_complete(slice11, context, cache, shared_v, delta, label):
    model = context["model"]
    kernel = context["kernel"]
    inherited_forward = model.forward
    first_forward_snapshot = None

    def boundary_observer(*args, **kwargs):
        nonlocal first_forward_snapshot
        outputs = inherited_forward(*args, **kwargs)
        if first_forward_snapshot is None:
            first_forward_snapshot = kernel.snapshot_cache(outputs.past_key_values)
        return outputs

    model.forward = boundary_observer
    try:
        traversal = slice11.traverse(
            context,
            cache,
            shared_v,
            delta,
            label,
            "exact established KV causal-intervention perturbation",
        )
    finally:
        model.forward = inherited_forward
    if first_forward_snapshot is None:
        raise RuntimeError("no first model.forward return was observed")
    return traversal, first_forward_snapshot, kernel.snapshot_cache(cache)


def exact_delta(context, traversal):
    supplied = lab.token_evidence(context, traversal["perturbation_token_ids_supplied"])
    observed = lab.token_evidence(context, traversal["actual_first_forward_perturbation_ids"])
    expected = {
        "token_ids": EXPECTED_DELTA_IDS,
        "count": len(EXPECTED_DELTA_IDS),
        "sha256": EXPECTED_DELTA_SHA256,
    }
    if supplied != expected or observed != expected:
        raise RuntimeError("traversal did not preserve the exact established delta_t")
    return expected


def execute_replicate(name, slice11, context, shared_v, source_snapshot, source_fp, expected_post_fp, sites, artifacts=None):
    kernel = context["kernel"]
    loaded_baseline = torch.load(SOURCE_SNAPSHOT, map_location="cpu", weights_only=False)
    baseline_cache = lab.restore_dynamic_cache(kernel, context["model"], loaded_baseline)
    baseline_pre = kernel.snapshot_cache(baseline_cache)
    if not lab.snapshot_content_equal(source_snapshot, baseline_pre):
        raise RuntimeError(f"replicate {name} baseline did not begin at exact K_t")

    baseline_traversal, baseline_first, baseline_final = traverse_complete(
        slice11, context, baseline_cache, shared_v, EXPECTED_DELTA_IDS,
        f"KV_RESPONSE_STRUCTURE_{name}_BASELINE",
    )
    delta_evidence = exact_delta(context, baseline_traversal)
    baseline_fp = lab.fingerprint(kernel, baseline_cache)
    if not lab.same_fingerprint(baseline_fp, expected_post_fp):
        raise RuntimeError(f"replicate {name} baseline replay did not reproduce K_t+1")
    baseline_output = lab.token_evidence(context, baseline_traversal["generated_token_ids"])

    specifications = []
    response_records = []
    first_records = []
    complete_identity = []
    all_pre_equal = True
    all_support_exact = True
    all_non_support_zero = True
    all_deltas_equal = True

    for site in sites:
        loaded = torch.load(SOURCE_SNAPSHOT, map_location="cpu", weights_only=False)
        cache = lab.restore_dynamic_cache(kernel, context["model"], loaded)
        restored_pre = kernel.snapshot_cache(cache)
        pre_equal = lab.snapshot_content_equal(source_snapshot, restored_pre)
        all_pre_equal = all_pre_equal and pre_equal
        if not pre_equal or not lab.same_fingerprint(lab.fingerprint(kernel, cache), source_fp):
            raise RuntimeError(f"replicate {name} intervention {site['j']} did not begin at exact K_t")

        specification = apply_intervention(kernel, cache, source_snapshot, site)
        specifications.append(specification)
        all_support_exact = all_support_exact and specification["support"]["INTERVENTION_SUPPORT_EXACTLY_BOUNDED"]
        all_non_support_zero = all_non_support_zero and specification["support"]["NON_INTERVENED_KV_CHANGED_BYTE_COUNT"] == 0

        traversal, first_post, final_post = traverse_complete(
            slice11, context, cache, shared_v, EXPECTED_DELTA_IDS,
            f"KV_RESPONSE_STRUCTURE_{name}_I{site['j']:02d}",
        )
        try:
            exact_delta(context, traversal)
        except Exception:
            all_deltas_equal = False
            raise

        final_response = subtract_snapshots(final_post, baseline_final)
        first_response = subtract_snapshots(first_post, baseline_first)
        final_descriptor = response_descriptors(final_response)
        first_descriptor = response_descriptors(first_response)

        if name == "A":
            final_artifact = persist_response(final_response, name, site["j"], "final")
            first_artifact = persist_response(first_response, name, site["j"], "first_forward")
        else:
            final_artifact = artifacts["final"][site["j"]]
            first_artifact = artifacts["first"][site["j"]]
            if not response_equal(final_response, load_response(final_artifact)):
                raise RuntimeError(f"complete final response A/B mismatch at intervention {site['j']}")
            if not response_equal(first_response, load_response(first_artifact)):
                raise RuntimeError(f"complete first-forward response A/B mismatch at intervention {site['j']}")

        final_descriptor["artifact"] = final_artifact
        first_descriptor["artifact"] = first_artifact
        response_records.append(final_descriptor)
        first_records.append(first_descriptor)
        complete_identity.append(
            final_descriptor["complete_response_sha256"]
            == load_response(final_artifact) and False
        ) if False else None
        del final_response, first_response, first_post, final_post, cache
        gc.collect()

    return {
        "replicate": name,
        "baseline_K_t_plus_1": baseline_fp,
        "baseline_output": baseline_output,
        "delta_t": delta_evidence,
        "intervention_specifications": specifications,
        "responses": response_records,
        "first_forward_responses": first_records,
        "checks": {
            "ALL_PRE_STATES_EQUAL_K_t": all_pre_equal,
            "ALL_INTERVENTION_SUPPORTS_EXACTLY_BOUNDED": all_support_exact,
            "ALL_NON_INTERVENED_PRE_STATE_BYTES_UNCHANGED": all_non_support_zero,
            "ALL_DELTAS_IDENTICAL": all_deltas_equal,
            "BASELINE_REPLAY_EXACT": True,
        },
        "_baseline_first": baseline_first,
        "_baseline_final": baseline_final,
    }


def component_dot(left, right):
    total = 0.0
    for left_layer, right_layer in zip(left["layers"], right["layers"]):
        for component in COMPONENTS:
            a = left_layer[component].to(torch.float64)
            b = right_layer[component].to(torch.float64)
            total += float(torch.sum(a * b).item())
    return total


def response_separation(left, right):
    total = 0.0
    for left_layer, right_layer in zip(left["layers"], right["layers"]):
        for component in COMPONENTS:
            delta = left_layer[component].to(torch.float64) - right_layer[component].to(torch.float64)
            total += float(torch.sum(delta * delta).item())
    return math.sqrt(total)


def cosine_from_vectors(left, right):
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return "UNDEFINED_ZERO_RESPONSE"
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def relation_matrices(records):
    responses = [load_response(record["artifact"]) for record in records]
    size = len(responses)
    exact = [[False for _ in range(size)] for _ in range(size)]
    cosine = [[None for _ in range(size)] for _ in range(size)]
    separation = [[None for _ in range(size)] for _ in range(size)]
    profile_cosine = [[None for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            exact[i][j] = response_equal(responses[i], responses[j])
            left_norm = records[i]["l2_norm"]
            right_norm = records[j]["l2_norm"]
            if left_norm == 0.0 or right_norm == 0.0:
                cosine[i][j] = "UNDEFINED_ZERO_RESPONSE"
            else:
                cosine[i][j] = component_dot(responses[i], responses[j]) / (left_norm * right_norm)
            separation[i][j] = response_separation(responses[i], responses[j])
            profile_cosine[i][j] = cosine_from_vectors(
                records[i]["component_energy_profile"],
                records[j]["component_energy_profile"],
            )
    del responses
    gc.collect()
    return {
        "exact_equality": exact,
        "full_response_cosine_similarity": cosine,
        "full_response_euclidean_separation": separation,
        "component_profile_cosine_similarity": profile_cosine,
    }


def strip_internal(replicate):
    return {key: value for key, value in replicate.items() if not key.startswith("_")}


def guard_original_facts():
    paths = [
        SOURCE_SNAPSHOT,
        SOURCE_JSON,
        SOURCE_EVIDENCE_DIR / "completion_receipt.txt",
        PROJECT_DIR / "kv_causal_intervention_lab.py",
    ]
    paths.extend(sorted(PROJECT_DIR.glob("slice*_runtime.py")))
    return {path.relative_to(PROJECT_DIR).as_posix(): sha256_file(path) for path in paths}


def make_receipt(result):
    fields = [
        ("OUTCOME", result["outcome"]),
        ("SOURCE_K_t", json.dumps(result["source_K_t"], sort_keys=True)),
        ("SOURCE_DELTA_t", json.dumps(result["source_delta_t"], sort_keys=True)),
        ("BASELINE_K_t_PLUS_1", json.dumps(result["baseline_K_t_plus_1"], sort_keys=True)),
        ("INTERVENTION_SELECTION_RULE", "q_j=floor(((2j+1)N)/24), j=0..11"),
        ("INTERVENTION_COUNT", "12"),
        ("INTERVENTION_SITES", json.dumps(result["intervention_sites"], sort_keys=True)),
        ("ALL_PRE_STATES_EQUAL_K_t", "PASS"),
        ("ALL_INTERVENTION_SUPPORTS_EXACTLY_BOUNDED", "PASS"),
        ("ALL_NON_INTERVENED_PRE_STATE_BYTES_UNCHANGED", "PASS"),
        ("ALL_DELTAS_IDENTICAL", "PASS"),
        ("RUNTIME_PHYSICS_INVARIANT", "PASS"),
        ("BASELINE_REPLAY_EXACT", "PASS"),
        ("RESPONSES", json.dumps(result["replications"]["A"]["responses"], sort_keys=True)),
        ("FIRST_FORWARD_RESPONSES", json.dumps(result["replications"]["A"]["first_forward_responses"], sort_keys=True)),
        ("PAIRWISE_FULL_RESPONSE_RELATION_MATRIX", json.dumps({
            key: result["pairwise_response_relation"][key]
            for key in ("exact_equality", "full_response_cosine_similarity", "full_response_euclidean_separation")
        }, sort_keys=True)),
        ("PAIRWISE_COMPONENT_PROFILE_MATRIX", json.dumps(
            result["pairwise_response_relation"]["component_profile_cosine_similarity"],
            sort_keys=True,
        )),
        ("NONZERO_RESPONSE_COUNT", str(result["nonzero_response_count"])),
        ("DISTINCT_COMPLETE_RESPONSE_COUNT", str(result["distinct_complete_response_count"])),
        ("DISTINCT_NONZERO_COMPLETE_RESPONSE_COUNT", str(result["distinct_nonzero_complete_response_count"])),
        ("REPLICATION_A_B_COMPLETE_RESPONSE_IDENTITY", "PASS"),
        ("PAIRWISE_RELATION_REPLICATION", "PASS"),
        ("ORIGINAL_AUTHORITATIVE_FACT_MUTATED", "NO"),
        ("NEW_ARTIFACTS", json.dumps(result["new_artifacts"], sort_keys=True)),
    ]
    lines = []
    for name, value in fields:
        lines.extend([f"{name}:", value, ""])
    return "\n".join(lines)


def run():
    if sha256_file(SOURCE_SNAPSHOT) != "e01fdf5b2fdc3a1ea5bc46d2664372874b6735adfb198d4663813772f46926ab":
        raise RuntimeError("persisted K_t file fingerprint differs from the authoritative fingerprint")
    source_lab = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    if source_lab["delta_t"]["token_ids"] != EXPECTED_DELTA_IDS:
        raise RuntimeError("source lab delta token IDs differ from the required exact delta")
    if source_lab["delta_t"]["sha256"] != EXPECTED_DELTA_SHA256:
        raise RuntimeError("source lab delta fingerprint differs from the required exact delta")

    original_guard_before = guard_original_facts()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    slice14 = lab.load_slice14_runtime()
    _, slice11, context = slice14.initialize_accumulated_runtime()
    parent = slice11.initialize_origin(context, passive_label=None)
    kernel = context["kernel"]
    source_snapshot = torch.load(SOURCE_SNAPSHOT, map_location="cpu", weights_only=False)
    source_material = kernel.fingerprint_material(source_snapshot)
    source_fp = {key: source_material[key] for key in lab.FP_KEYS}
    if source_fp["sha256"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("persisted K_t content fingerprint differs from the authoritative fingerprint")

    scalar_count, sites = derive_sites(source_snapshot)
    physics = lab.physics_evidence(slice11, context, parent.shared_v)
    physics_valid = (
        physics["model_weight_sha256"] == kernel.MODEL_WEIGHT_SHA256
        and physics["torch_version"] == kernel.TORCH_VERSION
        and physics["transformers_version"] == kernel.TRANSFORMERS_VERSION
        and physics["intraop_threads"] == 1
        and physics["interop_threads"] == 1
        and physics["deterministic_algorithms"]
        and physics["dtype"] == "torch.float32"
        and physics["device"] == "cpu"
    )
    if not physics_valid:
        raise RuntimeError("inherited deterministic runtime physics changed")

    expected_post_fp = source_lab["K_t_plus_1_original"]
    replicate_a = execute_replicate(
        "A", slice11, context, parent.shared_v, source_snapshot,
        source_fp, expected_post_fp, sites,
    )
    artifacts = {
        "final": [record["artifact"] for record in replicate_a["responses"]],
        "first": [record["artifact"] for record in replicate_a["first_forward_responses"]],
    }
    replicate_b = execute_replicate(
        "B", slice11, context, parent.shared_v, source_snapshot,
        source_fp, expected_post_fp, sites, artifacts,
    )

    if not lab.snapshot_content_equal(replicate_a["_baseline_first"], replicate_b["_baseline_first"]):
        raise RuntimeError("baseline first-forward post-state did not reproduce across A/B")
    if not lab.snapshot_content_equal(replicate_a["_baseline_final"], replicate_b["_baseline_final"]):
        raise RuntimeError("baseline final post-state did not reproduce across A/B")

    clean_a = strip_internal(replicate_a)
    clean_b = strip_internal(replicate_b)
    specifications_reproduce = (
        clean_a["intervention_specifications"] == clean_b["intervention_specifications"]
    )
    complete_responses_reproduce = all(
        left["complete_response_sha256"] == right["complete_response_sha256"]
        for left, right in zip(clean_a["responses"], clean_b["responses"])
    )
    first_responses_reproduce = all(
        left["complete_response_sha256"] == right["complete_response_sha256"]
        for left, right in zip(clean_a["first_forward_responses"], clean_b["first_forward_responses"])
    )
    if not specifications_reproduce or not complete_responses_reproduce or not first_responses_reproduce:
        raise RuntimeError("intervention specifications or complete response tensors failed A/B identity")

    relation_a = relation_matrices(clean_a["responses"])
    relation_b = relation_matrices(clean_b["responses"])
    pairwise_reproduces = relation_a == relation_b
    if not pairwise_reproduces:
        raise RuntimeError("pairwise response relation failed deterministic A/B replication")

    nonzero = [
        record for record in clean_a["responses"]
        if record["nonzero_scalar_count"] > 0
    ]
    nonzero_count = len(nonzero)
    distinct_all = len({
        record["complete_response_sha256"] for record in clean_a["responses"]
    })
    distinct_nonzero = len({
        record["complete_response_sha256"] for record in nonzero
    })
    positive = (
        specifications_reproduce
        and complete_responses_reproduce
        and nonzero_count >= 3
        and distinct_nonzero >= 3
        and pairwise_reproduces
    )
    outcome = (
        "REPRODUCIBLE_CAUSAL_STRUCTURE_OBSERVED"
        if positive
        else "NO_REPRODUCIBLE_CAUSAL_STRUCTURE_OBSERVED"
    )

    original_guard_after = guard_original_facts()
    if original_guard_before != original_guard_after:
        raise RuntimeError("an original authoritative source or evidence artifact was mutated")

    artifacts_created = sorted(
        path.relative_to(PROJECT_DIR).as_posix()
        for path in EVIDENCE_DIR.iterdir()
        if path.is_file()
    )
    artifacts_created.extend([
        "kv_causal_response_structure.py",
        "Verify KV Causal Response Structure.ps1",
        "evidence/kv_causal_response_structure/completion_receipt.txt",
        "evidence/kv_causal_response_structure/kv_causal_response_structure.json",
    ])

    result = {
        "schema": "KV_CAUSAL_RESPONSE_STRUCTURE_V1",
        "outcome": outcome,
        "interpretive_limit": (
            "A positive result establishes only a reproducible non-degenerate causal response "
            "relation for this bounded intervention set; it does not establish dimensionality or meaning."
        ),
        "source_K_t": source_fp,
        "source_delta_t": {
            "token_ids": EXPECTED_DELTA_IDS,
            "count": len(EXPECTED_DELTA_IDS),
            "sha256": EXPECTED_DELTA_SHA256,
        },
        "baseline_K_t_plus_1": clean_a["baseline_K_t_plus_1"],
        "canonical_scalar_count_N": scalar_count,
        "intervention_selection_rule": "q_j=floor(((2j+1)N)/24), j=0..11",
        "intervention_count": INTERVENTION_COUNT,
        "intervention_sites": sites,
        "physics": physics,
        "replications": {"A": clean_a, "B": clean_b},
        "pairwise_response_relation": relation_a,
        "nonzero_response_count": nonzero_count,
        "distinct_complete_response_count": distinct_all,
        "distinct_nonzero_complete_response_count": distinct_nonzero,
        "checks": {
            "ALL_PRE_STATES_EQUAL_K_t": True,
            "ALL_INTERVENTION_SUPPORTS_EXACTLY_BOUNDED": True,
            "ALL_NON_INTERVENED_PRE_STATE_BYTES_UNCHANGED": True,
            "ALL_DELTAS_IDENTICAL": True,
            "RUNTIME_PHYSICS_INVARIANT": True,
            "BASELINE_REPLAY_EXACT": True,
            "INTERVENTION_SPECIFICATIONS_REPRODUCE": specifications_reproduce,
            "REPLICATION_A_B_COMPLETE_RESPONSE_IDENTITY": complete_responses_reproduce,
            "REPLICATION_A_B_FIRST_FORWARD_RESPONSE_IDENTITY": first_responses_reproduce,
            "PAIRWISE_RELATION_REPLICATION": pairwise_reproduces,
            "ORIGINAL_AUTHORITATIVE_FACT_MUTATED": False,
        },
        "new_artifacts": artifacts_created,
    }
    EVIDENCE_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    receipt = make_receipt(result)
    RECEIPT_PATH.write_text(receipt, encoding="utf-8")
    print(receipt, flush=True)
    print(f"EVIDENCE_JSON={EVIDENCE_JSON}", flush=True)
    print(f"COMPLETION_RECEIPT={RECEIPT_PATH}", flush=True)
    gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as error:
        print("UNRESOLVED_EXECUTION_BOUNDARY", flush=True)
        print(
            f"KV_CAUSAL_RESPONSE_STRUCTURE_ERROR={error.__class__.__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
