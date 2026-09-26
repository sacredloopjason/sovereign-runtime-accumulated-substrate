"""Mechanical realization of an explicitly selected runtime-owned module."""

from core_runtime import ModuleHandle, invoke
from universal_membrane import (
    Form,
    Membrane,
    OrderedSequenceValue,
    TEXT_UTF8_FORM,
    read_text,
)


EXPLICIT_MODULE_ACTUATION_FORM = Form(
    canonical_name="SOVEREIGN_EXPLICIT_MODULE_ACTUATION_V1",
    interpretation_law=(
        "VALUE is ORDERED_SEQUENCE of exactly two independently complete membrane "
        "representations; child 1 is SOVEREIGN_SHARED_TEXT_UTF8_V1 containing the "
        "exact opaque address of the runtime-owned ModuleHandle explicitly selected "
        "by the emitting module; child 2 is the exact membrane perturbation selected "
        "for that target; child order is significant; target selection is not inferred "
        "from payload content or FORM."
    ),
)


def actuate(action: Membrane, loaded_handles: tuple[ModuleHandle, ...]) -> Membrane:
    """Resolve only an explicit opaque address, then use the inherited invoke law."""
    if not isinstance(action, Membrane) or action.form != EXPLICIT_MODULE_ACTUATION_FORM:
        raise ValueError("actuation requires the exact explicit-actuation FORM")
    if not isinstance(action.value, OrderedSequenceValue) or len(action.value.items) != 2:
        raise ValueError("an explicit action has exactly target and payload children")
    target, payload = action.value.items
    if target.form != TEXT_UTF8_FORM:
        raise ValueError("the explicit target address requires the exact shared text FORM")
    address = read_text(target)
    matches = tuple(handle for handle in loaded_handles if handle.address == address)
    if len(matches) != 1:
        raise LookupError("the explicit opaque address must resolve to exactly one loaded handle")
    return invoke(matches[0], payload)
