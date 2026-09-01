import gc
import hashlib
import io
import json
import math
import re
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "evidence" / "kv_causal_response_structure"
SOURCE_MANIFEST = SOURCE_DIR / "kv_causal_response_structure.json"
EVIDENCE_DIR = PROJECT_DIR / "evidence" / "kv_causal_response_dimensionality"
EVIDENCE_JSON = EVIDENCE_DIR / "kv_causal_response_dimensionality.json"
RECEIPT_PATH = EVIDENCE_DIR / "completion_receipt.txt"
AUTHORITATIVE_PARENT_COMMIT = "d62f1af9834722c0a59253947edf5ff6639f4f29"
RELATIVE_L2_RECONSTRUCTION_TOLERANCE = 1e-6
NEGATIVE_EIGENVALUE_RELATIVE_SAFEGUARD = 1e-12
INTERVENTION_IDS = tuple(range(12))
COMPONENTS = ("key", "value")
FAMILY_SPECS = {
    "final": {
        "manifest_key": "responses",
        "boundary": "final",
        "label": "FINAL_RESPONSE_FAMILY",
    },
    "first_forward": {
        "manifest_key": "first_forward_responses",
        "boundary": "first_forward",
        "label": "FIRST_FORWARD_RESPONSE_FAMILY",
    },
}


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(tensor):
    material = tensor.detach().cpu().contiguous().view(torch.uint8)
    return material.numpy().tobytes(order="C")


def exact_payload_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and tensor_bytes(left) == tensor_bytes(right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if list(left.keys()) != list(right.keys()):
            return False
        return all(exact_payload_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            exact_payload_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def artifact_intervention_id(path_text, boundary):
    pattern = rf"^evidence/kv_causal_response_structure/response_(\d{{2}})_{re.escape(boundary)}\.pt$"
    match = re.fullmatch(pattern, path_text)
    if match is None:
        raise RuntimeError(f"unexpected authoritative artifact path: {path_text}")
    return int(match.group(1))


def authoritative_records(manifest, family_name):
    spec = FAMILY_SPECS[family_name]
    replicate_a = manifest["replications"]["A"][spec["manifest_key"]]
    replicate_b = manifest["replications"]["B"][spec["manifest_key"]]
    if len(replicate_a) != 12 or len(replicate_b) != 12:
        raise RuntimeError(f"{family_name}: authoritative manifest does not contain exactly 12 records")
    records = {}
    for record_a, record_b in zip(replicate_a, replicate_b):
        artifact_a = record_a["artifact"]
        artifact_b = record_b["artifact"]
        if artifact_a != artifact_b:
            raise RuntimeError(f"{family_name}: replicate A/B artifact identity disagreement")
        intervention = artifact_intervention_id(artifact_a["path"], spec["boundary"])
        if intervention in records:
            raise RuntimeError(f"{family_name}: duplicate intervention {intervention}")
        records[intervention] = artifact_a
    if tuple(sorted(records)) != INTERVENTION_IDS:
        raise RuntimeError(f"{family_name}: intervention IDs are not exactly 0 through 11")
    return records


def validate_layers(layers, family_name, intervention):
    if not isinstance(layers, list) or not layers:
        raise RuntimeError(f"{family_name} intervention {intervention}: missing layer list")
    for layer_index, material in enumerate(layers):
        if not isinstance(material, dict):
            raise RuntimeError(
                f"{family_name} intervention {intervention} layer {layer_index}: invalid material"
            )
        if tuple(material.keys()) != COMPONENTS:
            raise RuntimeError(
                f"{family_name} intervention {intervention} layer {layer_index}: "
                "canonical key/value order not preserved"
            )
        for component in COMPONENTS:
            tensor = material[component]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"{family_name} intervention {intervention} layer {layer_index} "
                    f"{component}: not a tensor"
                )
            if tensor.device.type != "cpu":
                raise RuntimeError("authoritative tensor did not load on CPU")


def load_validate_roundtrip(family_name, intervention, artifact):
    spec = FAMILY_SPECS[family_name]
    path = PROJECT_DIR / artifact["path"]
    if not path.is_file():
        raise RuntimeError(f"missing authoritative response artifact: {path}")
    current_sha = sha256_file(path)
    if current_sha != artifact["file_sha256"]:
        raise RuntimeError(f"authoritative SHA-256 mismatch: {artifact['path']}")
    if path.stat().st_size != artifact["byte_count"]:
        raise RuntimeError(f"authoritative byte-count mismatch: {artifact['path']}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "KV_CAUSAL_RESPONSE_TENSOR_ARTIFACT_V1":
        raise RuntimeError(f"unexpected response schema: {artifact['path']}")
    if payload.get("intervention") != intervention:
        raise RuntimeError(f"intervention metadata mismatch: {artifact['path']}")
    if payload.get("boundary") != spec["boundary"]:
        raise RuntimeError(f"boundary metadata mismatch: {artifact['path']}")
    validate_layers(payload.get("layers"), family_name, intervention)

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    replay = torch.load(buffer, map_location="cpu", weights_only=False)
    if not exact_payload_equal(payload, replay):
        raise RuntimeError(f"artifact failed exact in-memory round-trip: {artifact['path']}")
    del replay, buffer

    after_sha = sha256_file(path)
    if after_sha != current_sha:
        raise RuntimeError(f"source artifact changed during read-only loading: {artifact['path']}")

    identity = {
        "intervention": intervention,
        "path": artifact["path"],
        "file_sha256": current_sha,
        "byte_count": path.stat().st_size,
        "round_trip_exact": True,
        "read_only_load": True,
    }
    return {"layers": payload["layers"]}, identity


def validate_family_shapes(response_map, family_name):
    reference = response_map[0]["layers"]
    for intervention in INTERVENTION_IDS:
        layers = response_map[intervention]["layers"]
        if len(layers) != len(reference):
            raise RuntimeError(f"{family_name}: inconsistent layer count")
        for layer_index, (expected, observed) in enumerate(zip(reference, layers)):
            for component in COMPONENTS:
                a = expected[component]
                b = observed[component]
                if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
                    raise RuntimeError(
                        f"{family_name}: inconsistent tensor at intervention "
                        f"{intervention}, layer {layer_index}, {component}"
                    )


def full_tensor_gram(response_map):
    gram = torch.zeros((12, 12), dtype=torch.float64, device="cpu")
    layer_count = len(response_map[0]["layers"])
    for layer_index in range(layer_count):
        for component in COMPONENTS:
            block = torch.stack(
                [
                    response_map[intervention]["layers"][layer_index][component]
                    .detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .to(torch.float64)
                    for intervention in INTERVENTION_IDS
                ],
                dim=0,
            )
            gram += block @ block.transpose(0, 1)
            del block
    if not torch.equal(gram, gram.transpose(0, 1)):
        raise RuntimeError("full-tensor Gram matrix is not exactly symmetric")
    return gram


def checked_eigh(gram, context):
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    lambda_max = float(eigenvalues[-1].item())
    if not math.isfinite(lambda_max) or lambda_max <= 0.0:
        raise RuntimeError(f"{context}: non-positive or non-finite lambda_max")
    lower_bound = -NEGATIVE_EIGENVALUE_RELATIVE_SAFEGUARD * lambda_max
    minimum = float(eigenvalues[0].item())
    if minimum < lower_bound:
        raise RuntimeError(
            f"{context}: eigenvalue {minimum:.17g} is below fixed safeguard "
            f"{lower_bound:.17g}"
        )
    eigenvalues = torch.where(
        eigenvalues < 0.0,
        torch.zeros_like(eigenvalues),
        eigenvalues,
    )
    eigenvalues = torch.flip(eigenvalues, dims=(0,))
    eigenvectors = torch.flip(eigenvectors, dims=(1,))
    return eigenvalues, eigenvectors


def relative_residual(norm_squared, captured_squared, context):
    remainder = norm_squared - captured_squared
    safeguard = NEGATIVE_EIGENVALUE_RELATIVE_SAFEGUARD * max(norm_squared, 1.0)
    if remainder < -safeguard:
        raise RuntimeError(
            f"{context}: negative residual square {remainder:.17g} "
            f"below safeguard {-safeguard:.17g}"
        )
    remainder = max(remainder, 0.0)
    if norm_squared <= 0.0:
        raise RuntimeError(f"{context}: zero response norm")
    return math.sqrt(remainder / norm_squared)


def analyze_gram(gram, family_name):
    eigenvalues, eigenvectors = checked_eigh(gram, f"{family_name} in-sample")
    in_sample_rows = []
    for dimension in range(1, 13):
        residuals = []
        for intervention in INTERVENTION_IDS:
            captured = 0.0
            for basis_index in range(dimension):
                captured += (
                    float(eigenvalues[basis_index].item())
                    * float(eigenvectors[intervention, basis_index].item()) ** 2
                )
            residuals.append(
                relative_residual(
                    float(gram[intervention, intervention].item()),
                    captured,
                    f"{family_name} in-sample d={dimension} intervention={intervention}",
                )
            )
        in_sample_rows.append(
            {
                "dimension": dimension,
                "relative_residuals": residuals,
                "maximum_relative_residual": max(residuals),
                "mean_relative_residual": sum(residuals) / len(residuals),
            }
        )

    in_sample_minimum = next(
        (
            row["dimension"]
            for row in in_sample_rows
            if row["maximum_relative_residual"]
            <= RELATIVE_L2_RECONSTRUCTION_TOLERANCE
        ),
        None,
    )
    if in_sample_minimum is None:
        raise RuntimeError(
            f"{family_name}: even the complete 12-response span failed the fixed criterion"
        )

    held_out_by_dimension = {
        dimension: [None for _ in INTERVENTION_IDS]
        for dimension in range(1, 12)
    }
    fold_eigenvalues = {}
    for held_out in INTERVENTION_IDS:
        training = [j for j in INTERVENTION_IDS if j != held_out]
        index = torch.tensor(training, dtype=torch.long)
        training_gram = gram.index_select(0, index).index_select(1, index)
        fold_values, fold_vectors = checked_eigh(
            training_gram,
            f"{family_name} held-out intervention {held_out}",
        )
        fold_eigenvalues[str(held_out)] = [
            float(value) for value in fold_values.tolist()
        ]
        cross = gram[held_out, index]
        coefficients = cross @ fold_vectors
        captured = 0.0
        for dimension in range(1, 12):
            eigenvalue = float(fold_values[dimension - 1].item())
            coefficient = float(coefficients[dimension - 1].item())
            if eigenvalue > 0.0:
                captured += (coefficient * coefficient) / eigenvalue
            elif coefficient != 0.0:
                raise RuntimeError(
                    f"{family_name} held-out {held_out}: nonzero coupling to exact null direction"
                )
            held_out_by_dimension[dimension][held_out] = relative_residual(
                float(gram[held_out, held_out].item()),
                captured,
                f"{family_name} LOIO d={dimension} held-out={held_out}",
            )

    loio_rows = []
    for dimension in range(1, 12):
        residuals = held_out_by_dimension[dimension]
        loio_rows.append(
            {
                "dimension": dimension,
                "held_out_relative_residuals": residuals,
                "maximum_relative_residual": max(residuals),
                "mean_relative_residual": sum(residuals) / len(residuals),
            }
        )
    loio_minimum = next(
        (
            row["dimension"]
            for row in loio_rows
            if row["maximum_relative_residual"]
            <= RELATIVE_L2_RECONSTRUCTION_TOLERANCE
        ),
        None,
    )

    if in_sample_minimum == 12:
        bounded_minimum = 12
        dimensionality_status = "RESOLVED_OBSERVED_DIMENSION_12"
        loio_result = (
            "NO_DIMENSION_1_THROUGH_11_RECONSTRUCTS_ALL_HELD_OUT_RESPONSES_"
            "WITHIN_TOLERANCE"
            if loio_minimum is None
            else f"DIMENSION_{loio_minimum}_SUPPORTED"
        )
        loio_dimension_12_testable = False
        loio_dimension_12_reason = "Each fold contains only 11 training responses."
    elif loio_minimum is None:
        bounded_minimum = "DIMENSIONALITY_NOT_RESOLVED"
        dimensionality_status = "NOT_RESOLVED_NO_LOIO_SUPPORTED_DIMENSION_THROUGH_11"
        loio_result = "NONE_THROUGH_11"
        loio_dimension_12_testable = False
        loio_dimension_12_reason = "Each fold contains only 11 training responses."
    elif loio_minimum < in_sample_minimum:
        raise RuntimeError(
            f"{family_name}: LOIO minimum {loio_minimum} is below "
            f"in-sample minimum {in_sample_minimum}"
        )
    else:
        bounded_minimum = loio_minimum
        dimensionality_status = "RESOLVED"
        loio_result = f"DIMENSION_{loio_minimum}_SUPPORTED"
        loio_dimension_12_testable = False
        loio_dimension_12_reason = "Each fold contains only 11 training responses."

    return {
        "fixed_reconstruction_tolerance": RELATIVE_L2_RECONSTRUCTION_TOLERANCE,
        "centered": False,
        "individual_response_normalization_before_basis": False,
        "numerical_dtype": "torch.float64",
        "device": "cpu",
        "in_sample_minimum_dimension": in_sample_minimum,
        "loio_minimum_supported_dimension": (
            loio_minimum if loio_minimum is not None else "NONE_THROUGH_11"
        ),
        "bounded_minimum_response_dimension": bounded_minimum,
        "dimensionality_status": dimensionality_status,
        "loio_result": loio_result,
        "loio_dimension_12_testable": loio_dimension_12_testable,
        "loio_dimension_12_reason": loio_dimension_12_reason,
        "gram_matrix": gram.tolist(),
        "gram_eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "in_sample_residuals_by_dimension": in_sample_rows,
        "loio_residuals_by_dimension_and_held_out_intervention": loio_rows,
        "loio_fold_eigenvalues": fold_eigenvalues,
    }


def compute_family_in_order(family_name, records, requested_order):
    response_map = {}
    identities = []
    for intervention in requested_order:
        response, identity = load_validate_roundtrip(
            family_name,
            intervention,
            records[intervention],
        )
        response_map[intervention] = response
        identities.append(identity)
    validate_family_shapes(response_map, family_name)
    gram = full_tensor_gram(response_map)
    analysis = analyze_gram(gram, family_name)
    identities.sort(key=lambda item: item["intervention"])
    del response_map, gram
    gc.collect()
    return {
        "artifact_identities": identities,
        "analysis": analysis,
    }


def analyze_family(family_name, records):
    canonical = compute_family_in_order(
        family_name,
        records,
        list(INTERVENTION_IDS),
    )
    reverse = compute_family_in_order(
        family_name,
        records,
        list(reversed(INTERVENTION_IDS)),
    )
    if canonical != reverse:
        raise RuntimeError(f"{family_name}: canonical/reverse order results differ exactly")
    result = dict(canonical["analysis"])
    result["artifact_identities"] = canonical["artifact_identities"]
    result["order_independence"] = "PASS"
    result["orders_executed"] = [
        list(INTERVENTION_IDS),
        list(reversed(INTERVENTION_IDS)),
    ]
    return result


def source_artifact_guard(records_by_family):
    paths = []
    for records in records_by_family.values():
        paths.extend(PROJECT_DIR / records[j]["path"] for j in INTERVENTION_IDS)
    paths = sorted(set(paths))
    return {
        path.relative_to(PROJECT_DIR).as_posix(): sha256_file(path)
        for path in paths
    }


def one_complete_analysis():
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8-sig"))
    records_by_family = {
        family_name: authoritative_records(manifest, family_name)
        for family_name in FAMILY_SPECS
    }
    return {
        "schema": "KV_CAUSAL_RESPONSE_DIMENSIONALITY_ANALYSIS_RUN_V1",
        "authoritative_parent_commit": AUTHORITATIVE_PARENT_COMMIT,
        "relative_l2_reconstruction_tolerance": RELATIVE_L2_RECONSTRUCTION_TOLERANCE,
        "negative_eigenvalue_relative_safeguard": (
            NEGATIVE_EIGENVALUE_RELATIVE_SAFEGUARD
        ),
        "final_response_family": analyze_family(
            "final",
            records_by_family["final"],
        ),
        "first_forward_response_family": analyze_family(
            "first_forward",
            records_by_family["first_forward"],
        ),
    }, records_by_family


def canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def family_receipt_block(label, family):
    return [
        label,
        "",
        "FIXED_RECONSTRUCTION_TOLERANCE:",
        "1e-6",
        "",
        "IN_SAMPLE_MINIMUM_DIMENSION:",
        str(family["in_sample_minimum_dimension"]),
        "",
        "LOIO_MINIMUM_SUPPORTED_DIMENSION:",
        str(family["loio_minimum_supported_dimension"]),
        "",
        "BOUNDED_MINIMUM_RESPONSE_DIMENSION:",
        str(family["bounded_minimum_response_dimension"]),
        "",
        "DIMENSIONALITY_STATUS:",
        family["dimensionality_status"],
        "",
        "LOIO_RESULT:",
        family["loio_result"],
        "",
        "LOIO_DIMENSION_12_TESTABLE:",
        "YES" if family["loio_dimension_12_testable"] else "NO",
        "",
        "REASON:",
        family["loio_dimension_12_reason"],
        "",
        "IN_SAMPLE_RESIDUALS_BY_DIMENSION:",
        json.dumps(
            family["in_sample_residuals_by_dimension"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "",
        "LOIO_RESIDUALS_BY_DIMENSION_AND_HELD_OUT_INTERVENTION:",
        json.dumps(
            family["loio_residuals_by_dimension_and_held_out_intervention"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "",
        "GRAM_EIGENVALUES:",
        json.dumps(family["gram_eigenvalues"], separators=(",", ":")),
        "",
        "DETERMINISTIC_REPLICATION:",
        "PASS",
        "",
        "ORDER_INDEPENDENCE:",
        family["order_independence"],
        "",
    ]


def make_receipt(result):
    final_family = result["final_response_family"]
    first_family = result["first_forward_response_family"]
    lines = [
        "OUTCOME",
        result["outcome"],
        "",
        "AUTHORITATIVE_PARENT_COMMIT:",
        AUTHORITATIVE_PARENT_COMMIT,
        "",
    ]
    lines.extend(family_receipt_block("FINAL_RESPONSE_FAMILY", final_family))
    lines.extend(
        family_receipt_block("FIRST_FORWARD_RESPONSE_FAMILY", first_family)
    )
    lines.extend(
        [
            "RESPONSE_FAMILIES_AGREE:",
            result["response_families_agree"],
            "",
        ]
    )
    if result["response_families_agree"] == "YES":
        lines.extend(
            [
                "COMMON_BOUNDED_MINIMUM_DIMENSION:",
                str(result["common_bounded_minimum_dimension"]),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "FINAL_BOUNDED_MINIMUM_DIMENSION:",
                str(final_family["bounded_minimum_response_dimension"]),
                "",
                "FIRST_FORWARD_BOUNDED_MINIMUM_DIMENSION:",
                str(first_family["bounded_minimum_response_dimension"]),
                "",
            ]
        )
    lines.extend(
        [
            "SOURCE_RESPONSE_ARTIFACTS_MUTATED:",
            "NO",
            "",
            "TRANSFORMER_RUNTIME_EXECUTED:",
            "NO",
            "",
            "DETERMINISTIC_ANALYSIS_REPLICATION:",
            "PASS",
            "",
            "ORDER_INDEPENDENCE:",
            "PASS",
            "",
        ]
    )
    return "\n".join(lines)


def run():
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8-sig"))
    initial_records = {
        family_name: authoritative_records(manifest, family_name)
        for family_name in FAMILY_SPECS
    }
    guard_before = source_artifact_guard(initial_records)

    run_a, records_a = one_complete_analysis()
    serialized_a = canonical_json_bytes(run_a)
    guard_between = source_artifact_guard(records_a)
    if guard_between != guard_before:
        raise RuntimeError("source response artifact changed during analysis replication A")

    run_b, records_b = one_complete_analysis()
    serialized_b = canonical_json_bytes(run_b)
    if serialized_a != serialized_b:
        raise RuntimeError("complete deterministic analysis serialization differs across A/B")
    guard_after = source_artifact_guard(records_b)
    if guard_after != guard_before:
        raise RuntimeError("source response artifact changed during analysis replication B")

    final_family = run_a["final_response_family"]
    first_family = run_a["first_forward_response_family"]
    final_bounded = final_family["bounded_minimum_response_dimension"]
    first_bounded = first_family["bounded_minimum_response_dimension"]
    both_resolved = (
        isinstance(final_bounded, int)
        and isinstance(first_bounded, int)
    )
    outcome = (
        "MINIMUM_RESPONSE_DIMENSION_RESOLVED"
        if both_resolved
        else "DIMENSIONALITY_NOT_RESOLVED"
    )
    families_agree = "YES" if final_bounded == first_bounded else "NO"

    result = dict(run_a)
    result.update(
        {
            "outcome": outcome,
            "response_families_agree": families_agree,
            "common_bounded_minimum_dimension": (
                final_bounded if families_agree == "YES" else None
            ),
            "source_response_artifacts_mutated": False,
            "transformer_runtime_executed": False,
            "deterministic_analysis_replication": "PASS",
            "order_independence": "PASS",
            "replication_serialized_sha256": hashlib.sha256(
                serialized_a
            ).hexdigest(),
        }
    )

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "analysis_run_a.json").write_bytes(serialized_a)
    (EVIDENCE_DIR / "analysis_run_b.json").write_bytes(serialized_b)
    if (
        (EVIDENCE_DIR / "analysis_run_a.json").read_bytes()
        != (EVIDENCE_DIR / "analysis_run_b.json").read_bytes()
    ):
        raise RuntimeError("persisted replication serializations differ")

    torch.save(
        torch.tensor(final_family["gram_matrix"], dtype=torch.float64),
        EVIDENCE_DIR / "final_gram_matrix_float64.pt",
    )
    torch.save(
        torch.tensor(first_family["gram_matrix"], dtype=torch.float64),
        EVIDENCE_DIR / "first_forward_gram_matrix_float64.pt",
    )
    EVIDENCE_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    receipt = make_receipt(result)
    RECEIPT_PATH.write_text(receipt, encoding="utf-8")

    print(receipt, flush=True)
    print(f"EVIDENCE_JSON={EVIDENCE_JSON}", flush=True)
    print(f"COMPLETION_RECEIPT={RECEIPT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as error:
        print("OUTCOME", flush=True)
        print("UNRESOLVED_DIMENSIONALITY_EXECUTION", flush=True)
        print(
            f"DIMENSIONALITY_ERROR={error.__class__.__name__}: {error}",
            flush=True,
        )
        raise

