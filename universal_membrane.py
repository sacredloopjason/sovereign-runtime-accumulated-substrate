"""The universal, module-independent membrane representation M=(F,V)."""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Form:
    """An exact interpretation contract; compatibility is ordinary equality."""

    canonical_name: str
    interpretation_law: str


@dataclass(frozen=True, slots=True)
class BytesValue:
    payload: bytes

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise TypeError("BYTES payload must be exactly bytes")


@dataclass(frozen=True, slots=True)
class OrderedSequenceValue:
    items: tuple["Membrane", ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or not all(
            isinstance(item, Membrane) for item in self.items
        ):
            raise TypeError("ORDERED_SEQUENCE must contain only membrane values")


Value: TypeAlias = BytesValue | OrderedSequenceValue


@dataclass(frozen=True, slots=True)
class Membrane:
    form: Form
    value: Value

    def __post_init__(self) -> None:
        if not isinstance(self.form, Form):
            raise TypeError("M.F must be a Form")
        if not isinstance(self.value, (BytesValue, OrderedSequenceValue)):
            raise TypeError("M.V must be BYTES or ORDERED_SEQUENCE(M...)")


TEXT_UTF8_FORM = Form(
    canonical_name="SOVEREIGN_SHARED_TEXT_UTF8_V1",
    interpretation_law=(
        "VALUE is BYTES containing exactly one strict UTF-8 encoding of text; "
        "no normalization, coercion, guessing, or semantic comparison is permitted"
    ),
)


def text_membrane(text: str) -> Membrane:
    if type(text) is not str:
        raise TypeError("shared text construction requires exactly str")
    return Membrane(TEXT_UTF8_FORM, BytesValue(text.encode("utf-8", errors="strict")))


def read_text(membrane: Membrane) -> str:
    if membrane.form != TEXT_UTF8_FORM:
        raise ValueError("exact shared text FORM equality is required")
    if not isinstance(membrane.value, BytesValue):
        raise TypeError("the shared text FORM requires BYTES")
    return membrane.value.payload.decode("utf-8", errors="strict")


def canonical_membrane_record(membrane: Membrane) -> dict:
    if isinstance(membrane.value, BytesValue):
        value = {"kind": "BYTES", "hex": membrane.value.payload.hex()}
    else:
        value = {
            "kind": "ORDERED_SEQUENCE",
            "items": [canonical_membrane_record(item) for item in membrane.value.items],
        }
    return {
        "form": {
            "canonical_name": membrane.form.canonical_name,
            "interpretation_law": membrane.form.interpretation_law,
        },
        "value": value,
    }
