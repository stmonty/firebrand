"""Compile programs into exact transformer weights."""

from .compiler import compile_machine
from .ir import (
    Machine,
    constant,
    history,
    latest,
    match,
    position,
    position_digit,
    position_mod,
    position_quotient,
    read_at,
    select,
)
from .machine import RecordMachine, Step, Transition
from .memory import IndexedMemory, MemoryUpdate, WriteLog
from .optimization import fuse_assignments
from .program import Array, Int, Memory, emit, program
from .records import Field, Record
from .runtime import Transformer
from .sequence import Assign, Branch, Halt, Jump, Label, Read, SequenceMachine, Write

__all__ = [
    "Array",
    "Assign",
    "Branch",
    "Field",
    "Halt",
    "IndexedMemory",
    "Int",
    "Jump",
    "Label",
    "Machine",
    "Memory",
    "MemoryUpdate",
    "Read",
    "Record",
    "RecordMachine",
    "SequenceMachine",
    "Step",
    "Transformer",
    "Transition",
    "Write",
    "WriteLog",
    "compile_machine",
    "constant",
    "emit",
    "fuse_assignments",
    "history",
    "latest",
    "match",
    "position",
    "position_digit",
    "position_mod",
    "position_quotient",
    "program",
    "read_at",
    "select",
]
