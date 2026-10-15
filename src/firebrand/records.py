"""Fixed-width integer record layouts serialized as consecutive tokens.

Bounds are checked when packing/unpacking; symbolic reads lower to read_at.
Schemas do not tag tokens or prove that a runtime pointer names this record type.
"""

from dataclasses import dataclass

from .ir import constant, integer, read_at, select


@dataclass(frozen=True)
class Field:
    name: str
    lower: int
    upper: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or integer(self.lower) > integer(self.upper):
            raise ValueError("invalid field name or range")


@dataclass(frozen=True)
class Record:
    name: str
    fields: tuple[Field, ...]

    def __post_init__(self):
        names = [f.name for f in self.fields]
        if not self.fields or len(set(names)) != len(names):
            raise ValueError("record fields must be nonempty and unique")

    @property
    def width(self):
        return len(self.fields)

    def offset(self, name):
        for offset, field in enumerate(self.fields):
            if field.name == name:
                return offset
        raise KeyError(name)

    def read(self, base, field):
        return read_at(constant(base) + self.offset(field))

    def pack(self, **values):
        if set(values) != {f.name for f in self.fields}:
            raise ValueError("record values must match its fields")
        result = []
        for field in self.fields:
            value = integer(values[field.name])
            if not field.lower <= value <= field.upper:
                raise ValueError(f"{field.name} outside declared range")
            result.append(value)
        return result

    def emit(self, phase, **values):
        """Expression selecting one field to emit; phase must be 0..width-1.

        The caller supplies a phase expression and allocates zero-based record
        addresses. This constructs a circuit, not a host-side generation loop.
        Values are eager, so unused reads must still name valid addresses.
        """
        if set(values) != {field.name for field in self.fields}:
            raise ValueError("emitted values must match the record fields")
        phase = constant(phase)
        result = constant(values[self.fields[-1].name])
        for offset in reversed(range(self.width - 1)):
            result = select(phase.equals(offset), values[self.fields[offset].name], result)
        return result

    def unpack(self, tokens, base=0):
        if not 0 <= integer(base) <= len(tokens) - self.width:
            raise ValueError("incomplete record")
        result = {field.name: tokens[base + offset] for offset, field in enumerate(self.fields)}
        self.pack(**result)
        return result
