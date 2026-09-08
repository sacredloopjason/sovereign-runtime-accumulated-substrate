$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$kernelPath = Join-Path $repo 'functorial_membrane.cs'
$evidencePath = Join-Path $repo 'evidence\functorial_triadic_relation'
$receiptPath = Join-Path $evidencePath 'completion_receipt.txt'
$runAPath = Join-Path $evidencePath 'verification_run_a.json'
$runBPath = Join-Path $evidencePath 'verification_run_b.json'

New-Item -ItemType Directory -Path $evidencePath -Force | Out-Null
Add-Type -Path $kernelPath

$baseType = [FunctorialMembrane.PrimitiveCarrier]
$membraneType = [FunctorialMembrane.Membrane]
$publicInstance = [System.Reflection.BindingFlags]::Public -bor [System.Reflection.BindingFlags]::Instance
$allInstance = $publicInstance -bor [System.Reflection.BindingFlags]::NonPublic
$publicStatic = [System.Reflection.BindingFlags]::Public -bor [System.Reflection.BindingFlags]::Static
$declaredPublicStatic = $publicStatic -bor [System.Reflection.BindingFlags]::DeclaredOnly

$nestedTypes = @(
    $baseType.GetNestedTypes([System.Reflection.BindingFlags]::Public) |
        Where-Object { $_.BaseType -eq $baseType }
)
$primitiveNames = @('T_P', 'T_C', 'T_H')
$actualNames = @($nestedTypes | ForEach-Object Name)
$tpType = $nestedTypes | Where-Object Name -eq 'T_P'
$tcType = $nestedTypes | Where-Object Name -eq 'T_C'
$thType = $nestedTypes | Where-Object Name -eq 'T_H'
$unaryPayloadOperationType = [System.Func[object, object]]
$binaryPayloadOperationType = [System.Func[object, object, object]]

$primitiveSetClosed =
    $actualNames.Count -eq 3 -and
    @($actualNames | Where-Object { $_ -notin $primitiveNames }).Count -eq 0 -and
    @($primitiveNames | Where-Object { $_ -notin $actualNames }).Count -eq 0

$baseConstructors = @($baseType.GetConstructors($allInstance))
$baseClosed =
    $baseType.IsAbstract -and
    @($baseType.GetConstructors($publicInstance)).Count -eq 0 -and
    $baseConstructors.Count -eq 1 -and
    $baseConstructors[0].IsPrivate

$variantsSealed =
    @($nestedTypes | Where-Object { -not $_.IsSealed }).Count -eq 0

$membraneMethods = @($membraneType.GetMethods($declaredPublicStatic))
$passageMethods = @($membraneMethods | Where-Object Name -eq 'Passage')
$passageClosed =
    $passageMethods.Count -eq 3 -and
    @(
        $passageMethods |
            Where-Object {
                $parameters = $_.GetParameters()
                $parameters.Count -ne 2 -or
                $_.ReturnType -ne $parameters[0].ParameterType -or
                $parameters[0].ParameterType.Name -notin $primitiveNames -or
                $parameters[1].ParameterType -ne $unaryPayloadOperationType
            }
    ).Count -eq 0

$primitivePairMethods = @(
    $membraneMethods |
        Where-Object {
            $parameters = $_.GetParameters()
            $parameters.Count -ge 2 -and
            ($nestedTypes -contains $parameters[0].ParameterType) -and
            ($nestedTypes -contains $parameters[1].ParameterType)
        }
)

$authorizedTriadicMethods = @(
    $primitivePairMethods |
        Where-Object {
            $parameters = $_.GetParameters()
            $parameters.Count -eq 3 -and
            $parameters[0].ParameterType -eq $tpType -and
            $parameters[1].ParameterType -eq $tcType -and
            $parameters[2].ParameterType -eq $binaryPayloadOperationType -and
            $_.ReturnType -eq $thType
        }
)

$alternatePrimitivePairMethods = @(
    $primitivePairMethods |
        Where-Object {
            $parameters = $_.GetParameters()
            -not (
                $parameters.Count -eq 3 -and
                $parameters[0].ParameterType -eq $tpType -and
                $parameters[1].ParameterType -eq $tcType -and
                $parameters[2].ParameterType -eq $binaryPayloadOperationType -and
                $_.ReturnType -eq $thType
            )
        }
)

$genericPrimitivePairEntrypoints = @(
    $membraneMethods |
        Where-Object {
            $parameters = $_.GetParameters()
            $parameters.Count -ge 2 -and
            (
                $parameters[0].ParameterType -eq [object] -or
                $parameters[0].ParameterType -eq $baseType -or
                $parameters[1].ParameterType -eq [object] -or
                $parameters[1].ParameterType -eq $baseType
            )
        }
)

$triadicRelationPresent =
    $authorizedTriadicMethods.Count -eq 1 -and
    $primitivePairMethods.Count -eq 1

$triadicInput1 = if ($authorizedTriadicMethods.Count -eq 1) {
    $authorizedTriadicMethods[0].GetParameters()[0].ParameterType.Name
}
else { $null }

$triadicInput2 = if ($authorizedTriadicMethods.Count -eq 1) {
    $authorizedTriadicMethods[0].GetParameters()[1].ParameterType.Name
}
else { $null }

$triadicOutput = if ($authorizedTriadicMethods.Count -eq 1) {
    $authorizedTriadicMethods[0].ReturnType.Name
}
else { $null }

$kernelSource = Get-Content -LiteralPath $kernelPath -Raw
$payloadSemanticInspectionPresent =
    $kernelSource -match '\b(if|else|switch|case|is|as)\b|\.GetType\(|\.ToString\(|Metadata|Provenance|Token|Label|Score|Classif'

$validatorOrJudgePresent =
    @(
        $membraneMethods |
            Where-Object { $_.Name -match 'Valid|Judge|Classif|Approve|Promot' }
    ).Count -gt 0

$rejectDecisionPresent =
    @(
        $membraneMethods |
            Where-Object { $_.Name -match 'Reject' }
    ).Count -gt 0 -or
    $kernelSource -match '\b(PENDING|PASS|FAIL|UNKNOWN|REJECT|PROMOTED|VALID|INVALID)\b'

function Invoke-TriadicFixture {
    $transform = [System.Func[object, object]] {
        param($payload)
        ([string]$payload) + '|F'
    }

    $attemptedPrimitiveSelection = [System.Func[object, object]] {
        param($payload)
        [FunctorialMembrane.PrimitiveCarrier+T_C]::new($payload)
    }

    $payloadRelation = [System.Func[object, object, object]] {
        param($potentialityPayload, $constraintPayload)
        ([string]$potentialityPayload) + '|' +
            ([string]$constraintPayload) + '|G'
    }

    $attemptedTriadicSelection = [System.Func[object, object, object]] {
        param($potentialityPayload, $constraintPayload)
        [FunctorialMembrane.PrimitiveCarrier+T_C]::new(
            ([string]$potentialityPayload) + '|' +
            ([string]$constraintPayload) + '|INNER_T_C'
        )
    }

    $tp = [FunctorialMembrane.PrimitiveCarrier+T_P]::new('p0')
    $tc = [FunctorialMembrane.PrimitiveCarrier+T_C]::new('c0')
    $th = [FunctorialMembrane.PrimitiveCarrier+T_H]::new('h0')

    $tpAfter = [FunctorialMembrane.Membrane]::Passage($tp, $transform)
    $tcAfter = [FunctorialMembrane.Membrane]::Passage($tc, $transform)
    $thAfter = [FunctorialMembrane.Membrane]::Passage($th, $transform)
    $tpSelectionAttempt = [FunctorialMembrane.Membrane]::Passage(
        $tp,
        $attemptedPrimitiveSelection
    )

    $triadicResult = [FunctorialMembrane.Membrane]::Realize(
        $tp,
        $tc,
        $payloadRelation
    )
    $triadicSelectionAttempt = [FunctorialMembrane.Membrane]::Realize(
        $tp,
        $tc,
        $attemptedTriadicSelection
    )

    $fourthRoleRepresentable = $false
    try {
        $null = New-Object `
            -TypeName 'FunctorialMembrane.PrimitiveCarrier+T_X' `
            -ArgumentList 'x0' `
            -ErrorAction Stop
        $fourthRoleRepresentable = $true
    }
    catch {
        $fourthRoleRepresentable = $false
    }

    $runPassed =
        $primitiveSetClosed -and
        $baseClosed -and
        $variantsSealed -and
        $passageClosed -and
        $triadicRelationPresent -and
        $triadicInput1 -eq 'T_P' -and
        $triadicInput2 -eq 'T_C' -and
        $triadicOutput -eq 'T_H' -and
        $authorizedTriadicMethods.Count -eq 1 -and
        $alternatePrimitivePairMethods.Count -eq 0 -and
        $genericPrimitivePairEntrypoints.Count -eq 0 -and
        (-not $payloadSemanticInspectionPresent) -and
        (-not $validatorOrJudgePresent) -and
        (-not $rejectDecisionPresent) -and
        (-not $fourthRoleRepresentable) -and
        $tpAfter.GetType() -eq $tpType -and
        $tpAfter.Payload -eq 'p0|F' -and
        $tcAfter.GetType() -eq $tcType -and
        $tcAfter.Payload -eq 'c0|F' -and
        $thAfter.GetType() -eq $thType -and
        $thAfter.Payload -eq 'h0|F' -and
        $tpSelectionAttempt.GetType() -eq $tpType -and
        $triadicResult.GetType() -eq $thType -and
        $triadicResult.Payload -eq 'p0|c0|G' -and
        $triadicSelectionAttempt.GetType() -eq $thType -and
        $triadicSelectionAttempt.Payload.GetType() -eq $tcType

    [ordered]@{
        primitive_count = $nestedTypes.Count
        primitives = $primitiveNames
        structurally_closed = ($primitiveSetClosed -and $baseClosed -and $variantsSealed)
        individual_passage_present = $passageClosed
        triadic_relation_present = $triadicRelationPresent
        triadic_input_1 = $triadicInput1
        triadic_input_2 = $triadicInput2
        triadic_output = $triadicOutput
        triadic_output_payload = $triadicResult.Payload
        authorized_triadic_signature_count = $authorizedTriadicMethods.Count
        authorized_triadic_signature = '(T_P,T_C)->T_H'
        alternate_primitive_pair_signature_present = ($alternatePrimitivePairMethods.Count -gt 0)
        generic_primitive_pair_entrypoint_present = ($genericPrimitivePairEntrypoints.Count -gt 0)
        g_can_select_output_primitive = ($triadicSelectionAttempt.GetType() -ne $thType)
        g_challenge_outer_type = $triadicSelectionAttempt.GetType().Name
        g_challenge_inner_payload_type = $triadicSelectionAttempt.Payload.GetType().Name
        payload_semantics_inspected = $payloadSemanticInspectionPresent
        fourth_causal_role_representable = $fourthRoleRepresentable
        validator_or_judge_present = $validatorOrJudgePresent
        reject_decision_present = $rejectDecisionPresent
        t_p_form_preserved_under_f = (
            $tpAfter.GetType() -eq $tpType -and
            $tpAfter.Payload -eq 'p0|F'
        )
        t_c_form_preserved_under_f = (
            $tcAfter.GetType() -eq $tcType -and
            $tcAfter.Payload -eq 'c0|F'
        )
        t_h_form_preserved_under_f = (
            $thAfter.GetType() -eq $thType -and
            $thAfter.Payload -eq 'h0|F'
        )
        f_can_select_output_primitive = (
            $tpSelectionAttempt.GetType() -ne $tpType
        )
        final_outcome = if ($runPassed) {
            'FUNCTORIAL_TRIADIC_RELATION_OPERATIONAL'
        }
        else {
            'UNRESOLVED_TRIADIC_RELATION_IMPLEMENTATION_BOUNDARY'
        }
    }
}

$runA = Invoke-TriadicFixture
$runB = Invoke-TriadicFixture
$jsonA = $runA | ConvertTo-Json -Depth 8
$jsonB = $runB | ConvertTo-Json -Depth 8
$deterministic = $jsonA -ceq $jsonB

Set-Content -LiteralPath $runAPath -Value $jsonA -Encoding utf8
Set-Content -LiteralPath $runBPath -Value $jsonB -Encoding utf8

$statusPaths = @(
    git -C $repo status --porcelain=v1 |
        ForEach-Object { $_.Substring(3).Trim('"') }
)
$unexpectedPaths = @(
    $statusPaths |
        Where-Object {
            $_ -ne 'functorial_membrane.cs' -and
            $_ -ne 'Verify Functorial Membrane.ps1' -and
            $_ -ne 'Verify Functorial Triadic Relation.ps1' -and
            $_ -notlike 'evidence/functorial_triadic_relation/*'
        }
)
$priorNonMembraneAuthoritativeFilesModified = $unexpectedPaths.Count -eq 0

$allPassed =
    $runA.final_outcome -eq 'FUNCTORIAL_TRIADIC_RELATION_OPERATIONAL' -and
    $deterministic -and
    $priorNonMembraneAuthoritativeFilesModified

$outcome = if ($allPassed) {
    'FUNCTORIAL_TRIADIC_RELATION_OPERATIONAL'
}
else {
    'UNRESOLVED_TRIADIC_RELATION_IMPLEMENTATION_BOUNDARY'
}

$receipt = @(
    'OUTCOME'
    $outcome
    ''
    'AUTHORITATIVE_PARENT_COMMIT:'
    'cf170044e1c99b0f8b8f27a64c781f12c80dbfd9'
    ''
    'MEMBRANE_IMPLEMENTATION:'
    'functorial_membrane.cs'
    ''
    'PRIMITIVE_COUNT:'
    $runA.primitive_count
    ''
    'PRIMITIVES:'
    '[T_P,T_C,T_H]'
    ''
    'STRUCTURALLY_CLOSED:'
    $(if ($runA.structurally_closed) { 'PASS' } else { 'FAIL' })
    ''
    'INDIVIDUAL_PASSAGE_PRESENT:'
    $(if ($runA.individual_passage_present) { 'PASS' } else { 'FAIL' })
    ''
    'INDIVIDUAL_PASSAGE_OPERATION:'
    'Membrane.Passage(T_P,Func<object,object>)->T_P; Membrane.Passage(T_C,Func<object,object>)->T_C; Membrane.Passage(T_H,Func<object,object>)->T_H'
    ''
    'TRIADIC_RELATION_PRESENT:'
    $(if ($runA.triadic_relation_present) { 'PASS' } else { 'FAIL' })
    ''
    'TRIADIC_OPERATION:'
    'FunctorialMembrane.Membrane.Realize'
    ''
    'TRIADIC_SIGNATURE:'
    $runA.authorized_triadic_signature
    ''
    'TRIADIC_INPUT_1:'
    $runA.triadic_input_1
    ''
    'TRIADIC_INPUT_2:'
    $runA.triadic_input_2
    ''
    'TRIADIC_OUTPUT:'
    $runA.triadic_output
    ''
    'TRIADIC_OUTPUT_PAYLOAD:'
    $runA.triadic_output_payload
    ''
    'TRIADIC_PAYLOAD_RELATION:'
    'caller-supplied opaque payload operation G'
    ''
    'G_CAN_SELECT_OUTPUT_PRIMITIVE:'
    $(if ($runA.g_can_select_output_primitive) { 'YES' } else { 'NO' })
    ''
    'AUTHORIZED_TRIADIC_SIGNATURE_COUNT:'
    $runA.authorized_triadic_signature_count
    ''
    'AUTHORIZED_TRIADIC_SIGNATURE:'
    $runA.authorized_triadic_signature
    ''
    'ALTERNATE_PRIMITIVE_PAIR_SIGNATURE_PRESENT:'
    $(if ($runA.alternate_primitive_pair_signature_present) { 'YES' } else { 'NO' })
    ''
    'GENERIC_PRIMITIVE_PAIR_ENTRYPOINT:'
    $(if ($runA.generic_primitive_pair_entrypoint_present) { 'PRESENT' } else { 'ABSENT' })
    ''
    'PAYLOAD_SEMANTIC_INSPECTION:'
    $(if ($runA.payload_semantics_inspected) { 'PRESENT' } else { 'ABSENT' })
    ''
    'FOURTH_CAUSAL_ROLE_REPRESENTABLE:'
    $(if ($runA.fourth_causal_role_representable) { 'YES' } else { 'NO' })
    ''
    'VALIDATOR_OR_JUDGE:'
    $(if ($runA.validator_or_judge_present) { 'PRESENT' } else { 'ABSENT' })
    ''
    'REJECT_DECISION:'
    $(if ($runA.reject_decision_present) { 'PRESENT' } else { 'ABSENT' })
    ''
    'T_P_FUNCTORIAL_PRESERVATION:'
    $(if ($runA.t_p_form_preserved_under_f) { 'PASS' } else { 'FAIL' })
    ''
    'T_C_FUNCTORIAL_PRESERVATION:'
    $(if ($runA.t_c_form_preserved_under_f) { 'PASS' } else { 'FAIL' })
    ''
    'T_H_FUNCTORIAL_PRESERVATION:'
    $(if ($runA.t_h_form_preserved_under_f) { 'PASS' } else { 'FAIL' })
    ''
    'F_CAN_SELECT_OUTPUT_PRIMITIVE:'
    $(if ($runA.f_can_select_output_primitive) { 'YES' } else { 'NO' })
    ''
    'TRIADIC_CAUSAL_FORM:'
    'P+C->H'
    ''
    'DETERMINISTIC_REPLICATION:'
    $(if ($deterministic) { 'PASS' } else { 'FAIL' })
    ''
    'PRIOR_NON_MEMBRANE_AUTHORITATIVE_FILES_MODIFIED:'
    $(if ($priorNonMembraneAuthoritativeFilesModified) { 'NO' } else { 'YES' })
)

Set-Content -LiteralPath $receiptPath -Value $receipt -Encoding utf8
$receipt | ForEach-Object { Write-Output $_ }

if (-not $allPassed) {
    exit 1
}
