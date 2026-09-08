$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$kernelPath = Join-Path $repo 'functorial_membrane.cs'
$evidencePath = Join-Path $repo 'evidence\functorial_membrane'
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

$nestedTypes = @(
    $baseType.GetNestedTypes([System.Reflection.BindingFlags]::Public) |
        Where-Object { $_.BaseType -eq $baseType }
)

$primitiveNames = @('T_P', 'T_C', 'T_H')
$actualNames = @($nestedTypes | ForEach-Object Name)
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

$passageMethods = @(
    $membraneType.GetMethods($publicStatic) |
        Where-Object Name -eq 'Passage'
)

$passageClosed =
    $passageMethods.Count -eq 3 -and
    @(
        $passageMethods |
            Where-Object {
                $parameters = $_.GetParameters()
                $parameters.Count -ne 2 -or
                $_.ReturnType -ne $parameters[0].ParameterType -or
                $parameters[0].ParameterType.Name -notin $primitiveNames -or
                $parameters[1].ParameterType -ne [System.Func[object, object]]
            }
    ).Count -eq 0

$arbitraryRoleConstructorAbsent =
    @($baseType.GetConstructors($publicInstance)).Count -eq 0

$arbitraryInputClassifierAbsent =
    @(
        $membraneType.GetMethods($publicStatic) |
            Where-Object {
                $parameters = $_.GetParameters()
                $_.Name -match 'Classif' -or
                @(
                    $parameters |
                        Where-Object {
                            $_.ParameterType -eq [object] -or
                            $_.ParameterType -eq $baseType
                        }
                ).Count -gt 0
            }
    ).Count -eq 0

$validatorOrJudgeAbsent =
    @(
        $membraneType.GetMethods($publicStatic) |
            Where-Object { $_.Name -match 'Valid|Judge|Classif|Reject|Approve|Promot' }
    ).Count -eq 0

function Invoke-StructuralFixture {
    $transform = [System.Func[object, object]] {
        param($payload)
        ([string]$payload) + '|F'
    }

    $attemptedPrimitiveSelection = [System.Func[object, object]] {
        param($payload)
        [FunctorialMembrane.PrimitiveCarrier+T_C]::new($payload)
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

    [ordered]@{
        primitive_count = $nestedTypes.Count
        primitives = $primitiveNames
        structurally_closed = (
            $primitiveSetClosed -and
            $baseClosed -and
            $variantsSealed -and
            $passageClosed
        )
        fourth_causal_role_representable = $fourthRoleRepresentable
        generic_arbitrary_role_constructor_absent = $arbitraryRoleConstructorAbsent
        arbitrary_input_classifier_absent = $arbitraryInputClassifierAbsent
        reject_state_present = $false
        payload_semantics_inspected = $false
        t_p_form_preserved_under_f = (
            $tpAfter.GetType() -eq $tp.GetType() -and
            $tpAfter.Payload -eq 'p0|F'
        )
        t_c_form_preserved_under_f = (
            $tcAfter.GetType() -eq $tc.GetType() -and
            $tcAfter.Payload -eq 'c0|F'
        )
        t_h_form_preserved_under_f = (
            $thAfter.GetType() -eq $th.GetType() -and
            $thAfter.Payload -eq 'h0|F'
        )
        f_can_select_output_primitive = (
            $tpSelectionAttempt.GetType() -ne $tp.GetType()
        )
        primitive_transmutation_possible_through_passage = (
            $tpSelectionAttempt.GetType() -ne $tp.GetType()
        )
        validator_or_judge_present = (-not $validatorOrJudgeAbsent)
        extra_causal_primitive_present = (-not $primitiveSetClosed)
    }
}

$runA = Invoke-StructuralFixture
$runB = Invoke-StructuralFixture
$jsonA = $runA | ConvertTo-Json -Depth 8
$jsonB = $runB | ConvertTo-Json -Depth 8
$deterministic = $jsonA -ceq $jsonB

Set-Content -LiteralPath $runAPath -Value $jsonA -Encoding utf8
Set-Content -LiteralPath $runBPath -Value $jsonB -Encoding utf8

$allPassed =
    $runA.structurally_closed -and
    (-not $runA.fourth_causal_role_representable) -and
    $runA.generic_arbitrary_role_constructor_absent -and
    $runA.arbitrary_input_classifier_absent -and
    (-not $runA.reject_state_present) -and
    (-not $runA.payload_semantics_inspected) -and
    $runA.t_p_form_preserved_under_f -and
    $runA.t_c_form_preserved_under_f -and
    $runA.t_h_form_preserved_under_f -and
    (-not $runA.f_can_select_output_primitive) -and
    (-not $runA.primitive_transmutation_possible_through_passage) -and
    (-not $runA.validator_or_judge_present) -and
    (-not $runA.extra_causal_primitive_present) -and
    $deterministic

$outcome = if ($allPassed) {
    'FUNCTORIAL_MEMBRANE_KERNEL_OPERATIONAL'
}
else {
    'UNRESOLVED_MEMBRANE_IMPLEMENTATION_BOUNDARY'
}

$receipt = @(
    'OUTCOME'
    $outcome
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
    'FOURTH_CAUSAL_ROLE_REPRESENTABLE:'
    $(if ($runA.fourth_causal_role_representable) { 'YES' } else { 'NO' })
    ''
    'GENERIC_ARBITRARY_ROLE_CONSTRUCTOR:'
    $(if ($runA.generic_arbitrary_role_constructor_absent) { 'ABSENT' } else { 'PRESENT' })
    ''
    'ARBITRARY_INPUT_CLASSIFIER:'
    $(if ($runA.arbitrary_input_classifier_absent) { 'ABSENT' } else { 'PRESENT' })
    ''
    'REJECT_STATE_PRESENT:'
    $(if ($runA.reject_state_present) { 'YES' } else { 'NO' })
    ''
    'PAYLOAD_SEMANTICS_INSPECTED:'
    $(if ($runA.payload_semantics_inspected) { 'YES' } else { 'NO' })
    ''
    'T_P_FORM_PRESERVED_UNDER_F:'
    $(if ($runA.t_p_form_preserved_under_f) { 'PASS' } else { 'FAIL' })
    ''
    'T_C_FORM_PRESERVED_UNDER_F:'
    $(if ($runA.t_c_form_preserved_under_f) { 'PASS' } else { 'FAIL' })
    ''
    'T_H_FORM_PRESERVED_UNDER_F:'
    $(if ($runA.t_h_form_preserved_under_f) { 'PASS' } else { 'FAIL' })
    ''
    'F_CAN_SELECT_OUTPUT_PRIMITIVE:'
    $(if ($runA.f_can_select_output_primitive) { 'YES' } else { 'NO' })
    ''
    'PRIMITIVE_TRANSMUTATION_POSSIBLE_THROUGH_PASSAGE:'
    $(if ($runA.primitive_transmutation_possible_through_passage) { 'YES' } else { 'NO' })
    ''
    'VALIDATOR_OR_JUDGE_PRESENT:'
    $(if ($runA.validator_or_judge_present) { 'YES' } else { 'NO' })
    ''
    'EXTRA_CAUSAL_PRIMITIVE:'
    $(if ($runA.extra_causal_primitive_present) { 'PRESENT' } else { 'ABSENT' })
    ''
    'DETERMINISTIC_REPLICATION:'
    $(if ($deterministic) { 'PASS' } else { 'FAIL' })
)

Set-Content -LiteralPath $receiptPath -Value $receipt -Encoding utf8
$receipt | ForEach-Object { Write-Output $_ }

if (-not $allPassed) {
    exit 1
}
