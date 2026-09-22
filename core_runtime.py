"""Minimal selection and invocation through the universal membrane."""

from dataclasses import dataclass, field
from typing import Callable

from universal_membrane import Form, Membrane


@dataclass(frozen=True, slots=True)
class ModuleHandle:
    address: str
    accepted_form: Form
    _entrypoint: Callable[[Membrane], Membrane] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.address) is not str or not self.address:
            raise ValueError("an opaque non-empty address is required")
        if not isinstance(self.accepted_form, Form):
            raise TypeError("the declared FORM must be universal")
        if not callable(self._entrypoint):
            raise TypeError("the selected address must resolve to a callable")


def invoke(handle: ModuleHandle, membrane: Membrane) -> Membrane:
    """The one runtime-facing invocation operation for every module."""
    if not isinstance(handle, ModuleHandle):
        raise TypeError("selection requires an opaque module handle")
    if not isinstance(membrane, Membrane):
        raise TypeError("invocation accepts only M")
    if membrane.form != handle.accepted_form:
        raise ValueError("direct crossing requires exact FORM equality")
    result = handle._entrypoint(membrane)
    if not isinstance(result, Membrane):
        raise TypeError("invocation must return only M")
    return result
