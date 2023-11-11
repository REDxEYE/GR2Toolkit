from dataclasses import dataclass, field
from enum import IntEnum, Enum
from functools import cached_property
from typing import Optional

import numpy as np

from file_utils import MemoryBuffer, Readable, Buffer, WritableMemoryBuffer
from granny2_wrapper import Granny2Native

LE64MAGIC = b'\xe5\x9bI^oc\x1f\x14\x1e\x13\xeb\xa9\x90\xbe\xed\xc4'
LE64MAGIC2 = b'\xe5/J\xe1o\xc2\x8a\xee\x1e\xd2\xb4L\x90\xd7U\xaf'
LE32MAGIC6 = b'\xb8g\xb0\xca\xf8m\xb1\x0f\x84r\x8c~^\x19\x00\x1e'

supported_idents = [LE64MAGIC, LE64MAGIC2, LE32MAGIC6]


@dataclass(slots=True)
class Magic(Readable):
    sig: bytes
    header_size: int
    header_format: int
    reserved: int

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Magic':
        sig = buffer.read(16)
        if sig not in supported_idents:
            raise ValueError("Invalid signature")
        return cls(sig, *buffer.read_fmt('2IQ'))

    @cached_property
    def compressed(self):
        return self.header_format != 0

    @cached_property
    def little_endian(self):
        return self.sig == LE64MAGIC or self.sig == LE64MAGIC2


@dataclass(slots=True)
class SectionReference(Readable):
    section: int
    offset: int

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'SectionReference':
        return cls(*buffer.read_fmt("2I"))


@dataclass(slots=True)
class SectionHeader(Readable):
    compression: int
    offset: int
    comp_size: int
    decomp_size: int
    alignment: int
    first16bits: int
    first8bits: int
    relocations_offset: int
    relocations_count: int
    mixed_marshaling_data_offset: int
    mixed_marshaling_data_count: int

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'SectionHeader':
        return cls(*buffer.read_fmt("11I"))


@dataclass(slots=True)
class Header(Readable):
    version: int
    size: int
    crc: int
    sections_offset: int
    sections_count: int
    root_type: SectionReference
    root_node: SectionReference
    tag: int
    extra_tags: tuple[int, int, int, int]

    string_table_crc: int = 0
    reserved0: int = 0
    reserved1: int = 0
    reserved2: int = 0

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Header':
        header = cls(buffer.read_uint32(),
                     buffer.read_uint32(),
                     buffer.read_uint32(),
                     buffer.read_uint32(),
                     buffer.read_uint32(),
                     SectionReference.from_buffer(buffer),
                     SectionReference.from_buffer(buffer),
                     buffer.read_uint32(), buffer.read_fmt("4I"))

        if header.version >= 7:
            header.string_table_crc = buffer.read_uint32()
            header.reserved0 = buffer.read_uint32()
            header.reserved1 = buffer.read_uint32()
            header.reserved2 = buffer.read_uint32()

        return header


_REF_CACHE: dict[int, object] = {}


@dataclass(slots=True)
class RelocatableReference(Readable):
    _buffer: Buffer
    offset: int

    @property
    def is_valid(self):
        return self.offset != 0

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'RelocatableReference':
        return cls(buffer, buffer.read_ptr())

    def __hash__(self) -> int:
        return hash(self.offset)


class MemberType(IntEnum):
    None_ = 0
    Inline = 1
    Reference = 2
    ReferenceToArray = 3
    ArrayOfReferences = 4
    VariantReference = 5
    ReferenceToVariantArray = 7
    String = 8
    Transform = 9
    Real32 = 10
    Int8 = 11
    UInt8 = 12
    BinormalInt8 = 13
    NormalUInt8 = 14
    Int16 = 15
    UInt16 = 16
    BinormalInt16 = 17
    NormalUInt16 = 18
    Int32 = 19
    UInt32 = 20
    Real16 = 21
    EmptyReference = 22
    Max = EmptyReference
    Invalid = 0xffffffff


class Int8(int):
    pass


class UInt8(int):
    pass


class BinormalInt8(int):
    pass


class NormalUInt8(int):
    pass


class Int16(int):
    pass


class UInt16(int):
    pass


class BinormalInt16(int):
    pass


class NormalUInt16(int):
    pass


class Int32(int):
    pass


class UInt32(int):
    pass


class Real16(float):
    pass


@dataclass(slots=True)
class MemberDefinition(Readable):
    type_id: MemberType
    name: str
    definition: 'StructReference'
    array_size: int
    extra: tuple[int, int, int]
    unk: int

    @property
    def is_valid(self):
        return self.type_id != MemberType.None_ and self.type_id != MemberType.Invalid

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'MemberDefinition':
        type_id = MemberType(buffer.read_uint32())
        name_ref = StringReference.from_buffer(buffer)
        if name_ref.is_valid:
            name = name_ref.resolve(buffer)
        else:
            name = "<INVALID>"
        definition = StructReference.from_buffer(buffer)
        array_size = buffer.read_uint32()
        extra = buffer.read_fmt('3I')
        unk = buffer.read_ptr()
        return cls(type_id, name, definition, array_size, extra, unk)

    def __hash__(self) -> int:
        return hash((self.type_id.value, self.name, self.array_size, self.extra, self.unk, hash(self.definition)))


@dataclass(slots=True)
class StructDefinition(Readable):
    members: list[MemberDefinition] = field(default_factory=list)

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'StructDefinition':
        struct = cls()
        while True:
            member = MemberDefinition.from_buffer(buffer)
            if not member.is_valid:
                break
            struct.members.append(member)
        return struct

    def __hash__(self) -> int:
        return hash(tuple(self.members))


TYPES = {}


class StructReference(RelocatableReference):

    def resolve(self, buffer: Buffer) -> StructDefinition:
        stype = TYPES.get(self, None)
        if stype is None:
            with buffer.save_current_offset():
                buffer.seek(self.offset)
                stype = StructDefinition.from_buffer(buffer)
                TYPES[self] = stype
        return stype


@dataclass(slots=True)
class ArrayReference(RelocatableReference):
    size: int

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'ArrayReference':
        size = buffer.read_uint32()
        offset = buffer.read_ptr()
        return cls(buffer, offset, size)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.size))


@dataclass(slots=True)
class ArrayIndicesReference(ArrayReference):
    size: int
    items: Optional[list[RelocatableReference]] = None

    def resolve(self, buffer: Buffer) -> list[RelocatableReference]:
        if self.items is None:
            with buffer.save_current_offset():
                buffer.seek(self.offset)
                if buffer.is_64:
                    self.items = [RelocatableReference(buffer, offset) for offset in buffer.read_fmt(f"{self.size}Q")]
                else:
                    self.items = [RelocatableReference(buffer, offset) for offset in buffer.read_fmt(f"{self.size}I")]
        return self.items

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.size, hash(tuple(self.items))))


@dataclass(slots=True)
class StringReference(RelocatableReference):
    _cache: Optional[str] = field(init=False, default=None)

    def resolve(self, buffer: Buffer) -> str:
        if self._cache is None:
            with buffer.save_current_offset():
                buffer.seek(self.offset)
                self._cache = buffer.read_ascii_string()
        return self._cache

    def __str__(self):
        return self.resolve(self._buffer)

    __repr__ = __str__


@dataclass(slots=True)
class Transform:
    flags: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    shear_scale: tuple[tuple[float, ...], ...]


def read_not_implemented(buffer: Buffer, member: MemberDefinition):
    raise NotImplementedError()


def read_transform(member: MemberDefinition, buffer: Buffer):
    flags, *data = buffer.read_fmt("I16f")
    pos, rot, mat = data[:3], data[3:7], data[7:]
    return Transform(Int32(flags), pos, rot, np.asarray(mat, np.float32).reshape((3, 3)))


def read_string_reference(member: MemberDefinition, buffer: Buffer):
    str_ref = StringReference.from_buffer(buffer)
    cache_data = _REF_CACHE.get(str_ref.offset, None)
    if str_ref.offset == 0:
        return None
    if cache_data is None:
        cache_data = _REF_CACHE[str_ref.offset] = str_ref.resolve(buffer)
    return cache_data


def read_reference_to_variant_array(member: MemberDefinition, buffer: Buffer):
    struct_ref = StructReference.from_buffer(buffer)
    items_ref = ArrayReference.from_buffer(buffer)
    if items_ref.size == 0 or items_ref.offset == 0:
        return []
    cache_data = _REF_CACHE.get(items_ref.offset, None)
    if cache_data is None:
        struct = struct_ref.resolve(buffer)
        with buffer.read_from_offset(items_ref.offset):
            cache_data = _REF_CACHE[items_ref.offset] = ArrayNode(
                read_struct(struct, buffer) for _ in range(items_ref.size))
    return cache_data


def read_reference_to_array(member: MemberDefinition, buffer: Buffer):
    items_ref = ArrayReference.from_buffer(buffer)
    if items_ref.size == 0 or items_ref.offset == 0:
        return ArrayNode()
    cache_data = _REF_CACHE.get(items_ref.offset, None)
    if cache_data is None:
        struct = member.definition.resolve(buffer)
        with buffer.read_from_offset(items_ref.offset):
            cache_data = _REF_CACHE[items_ref.offset] = ArrayNode(read_struct(struct, buffer) for _ in
                                                                  range(items_ref.size))
    return cache_data


def read_array_of_references(member: MemberDefinition, buffer: Buffer):
    array_ref = ArrayIndicesReference.from_buffer(buffer)
    cache_data = _REF_CACHE.get(array_ref.offset, None)
    if array_ref.offset == 0 or array_ref.size == 0:
        return ArrayNode()
    if cache_data is None:
        array_of_refs = array_ref.resolve(buffer)
        struct = member.definition.resolve(buffer)
        cache_data = _REF_CACHE[array_ref.offset] = ArrayNode()
        with buffer.save_current_offset():
            for ref in array_of_refs:
                buffer.seek(ref.offset)
                cache_data.append(read_struct(struct, buffer))
    return cache_data


def read_variant_member_reference(member: MemberDefinition, buffer: Buffer):
    struct_ref = StructReference.from_buffer(buffer)
    ref = RelocatableReference.from_buffer(buffer)
    if ref.offset == 0:
        return None
    cache_data = _REF_CACHE.get(ref.offset, None)
    if cache_data is None:
        with buffer.save_current_offset():
            buffer.seek(ref.offset)
            struct = struct_ref.resolve(buffer)
            cache_data = _REF_CACHE[ref.offset] = read_struct(struct, buffer)
    return cache_data


def read_reference_member(member: MemberDefinition, buffer: Buffer):
    ref = RelocatableReference.from_buffer(buffer)
    if ref.offset == 0:
        return None
    cache_data = _REF_CACHE.get(ref.offset, None)
    if cache_data is None:
        with buffer.save_current_offset():
            buffer.seek(ref.offset)
            struct = member.definition.resolve(buffer)
            cache_data = _REF_CACHE[ref.offset] = read_struct(struct, buffer)
    return cache_data


_MEMBER_READERS = {
    MemberType.Inline: read_not_implemented,
    MemberType.Reference: read_reference_member,
    MemberType.ReferenceToArray: read_reference_to_array,
    MemberType.ArrayOfReferences: read_array_of_references,
    MemberType.VariantReference: read_variant_member_reference,
    MemberType.ReferenceToVariantArray: read_reference_to_variant_array,
    MemberType.String: read_string_reference,
    MemberType.Transform: read_transform,
    MemberType.Real32: lambda m, b: b.read_float(),
    MemberType.Int8: lambda m, b: Int8(b.read_int8()),
    MemberType.UInt8: lambda m, b: UInt8(b.read_uint8()),
    MemberType.BinormalInt8: lambda m, b: BinormalInt8(b.read_int8()),
    MemberType.NormalUInt8: lambda m, b: NormalUInt8(b.read_uint8()),
    MemberType.Int16: lambda m, b: Int16(b.read_int16()),
    MemberType.UInt16: lambda m, b: UInt16(b.read_uint16()),
    MemberType.BinormalInt16: lambda m, b: BinormalInt16(b.read_int16()),
    MemberType.NormalUInt16: lambda m, b: NormalUInt16(b.read_uint16()),
    MemberType.Int32: lambda m, b: Int32(b.read_int32()),
    MemberType.UInt32: lambda m, b: UInt32(b.read_uint32()),
    MemberType.Real16: lambda m, b: Real16(b.read_hfloat()),
    MemberType.EmptyReference: read_not_implemented,
}


def default_array_read(member, buffer):
    return ArrayNode(_MEMBER_READERS[member.type_id](member, buffer) for _ in range(member.array_size))


_ARRAY_MEMBER_READERS = {
    MemberType.Inline: read_not_implemented,
    MemberType.Reference: default_array_read,
    MemberType.ReferenceToArray: default_array_read,
    MemberType.ArrayOfReferences: default_array_read,
    MemberType.VariantReference: default_array_read,
    MemberType.ReferenceToVariantArray: default_array_read,
    MemberType.String: default_array_read,
    MemberType.Transform: default_array_read,
    MemberType.Real32: lambda m, b: b.read_fmt(f"{m.array_size}f"),
    MemberType.Int8: lambda m, b: ArrayNode(map(Int8, b.read_fmt(f"{m.array_size}b"))),
    MemberType.UInt8: lambda m, b: ArrayNode(map(UInt8, b.read_fmt(f"{m.array_size}B"))),
    MemberType.BinormalInt8: lambda m, b: ArrayNode(map(BinormalInt8, b.read_fmt(f"{m.array_size}b"))),
    MemberType.NormalUInt8: lambda m, b: ArrayNode(map(NormalUInt8, b.read_fmt(f"{m.array_size}B"))),
    MemberType.Int16: lambda m, b: ArrayNode(map(Int16, b.read_fmt(f"{m.array_size}h"))),
    MemberType.UInt16: lambda m, b: ArrayNode(map(UInt16, b.read_fmt(f"{m.array_size}H"))),
    MemberType.BinormalInt16: lambda m, b: ArrayNode(map(BinormalInt16, b.read_fmt(f"{m.array_size}h"))),
    MemberType.NormalUInt16: lambda m, b: ArrayNode(map(NormalUInt16, b.read_fmt(f"{m.array_size}H"))),
    MemberType.Int32: lambda m, b: ArrayNode(map(Int32, b.read_fmt(f"{m.array_size}i"))),
    MemberType.UInt32: lambda m, b: ArrayNode(map(UInt32, b.read_fmt(f"{m.array_size}I"))),
    MemberType.Real16: lambda m, b: ArrayNode(map(Real16, b.read_fmt(f"{m.array_size}e"))),
    MemberType.EmptyReference: read_not_implemented,
}


class Node:
    pass


class DictNode(dict, Node):
    pass


class ArrayNode(list, Node):
    pass


def read_struct(struct: StructDefinition, buffer: Buffer):
    data = DictNode()
    for member in struct.members:
        if member.array_size == 0:
            result = _MEMBER_READERS[member.type_id](member, buffer)
        else:
            result = _ARRAY_MEMBER_READERS[member.type_id](member, buffer)
        data[member.name] = result
    return data


def deserialize_granny2_file(file_data: bytes):
    _REF_CACHE.clear()
    TYPES.clear()

    buffer = MemoryBuffer(file_data)
    granny2_native = Granny2Native
    magic = Magic.from_buffer(buffer)
    if magic.sig in (LE32MAGIC6, LE64MAGIC, LE64MAGIC2):
        buffer.set_endian(True)
    else:
        buffer.set_endian(False)
    if magic.sig == LE32MAGIC6:
        buffer.set_ptr_type("I")
    else:
        buffer.set_ptr_type("Q")

    if magic.compressed:
        raise NotImplementedError("Compressed GR2 files are not supported")
    header = Header.from_buffer(buffer)
    section_headers = []
    for _ in range(header.sections_count):
        section_headers.append(SectionHeader.from_buffer(buffer))

    decompressed_stream_size = sum(section.decomp_size for section in section_headers)
    decompressed_stream = bytes()
    decomp_offset = 0
    for section in section_headers:
        buffer.seek(section.offset)
        compressed_data = buffer.read(section.comp_size)
        section.offset = decomp_offset
        decomp_offset += section.decomp_size

        if section.compression == 0:
            decompressed_stream += compressed_data
        elif section.decomp_size > 0:
            if section.compression == 4:
                decompressed_stream += granny2_native.decompress4(compressed_data, section.decomp_size)
            else:
                decompressed_stream += granny2_native.decompress(
                    section.compression,
                    compressed_data,
                    section.decomp_size,
                    section.first16bits,
                    section.first8bits,
                    section.decomp_size
                )
    assert len(decompressed_stream) == decompressed_stream_size
    decompressed_stream = WritableMemoryBuffer(decompressed_stream)
    decompressed_stream.copy_settings(buffer)

    def apply_relocations(section: SectionHeader, relocation_buffer: Buffer):
        for _ in range(section.relocations_count):
            offset_in_section = relocation_buffer.read_uint32()
            ref = SectionReference.from_buffer(relocation_buffer)
            decompressed_stream.seek(section.offset + offset_in_section)
            fixup_address = section_headers[ref.section].offset + ref.offset
            decompressed_stream.write_uint32(fixup_address)

    for section in section_headers:
        if not section.relocations_count:
            continue
        buffer.seek(section.relocations_offset)
        if section.compression == 4:
            compressed_size = buffer.read_uint32()
            compressed_data = buffer.read(compressed_size)
            decompressed_relocation = granny2_native.decompress4(compressed_data, section.relocations_count * 12)
            memory_buffer = MemoryBuffer(decompressed_relocation)
            memory_buffer.copy_settings(buffer)
            apply_relocations(section, memory_buffer)
        else:
            apply_relocations(section, buffer)

    root_struct_ref = StructReference(decompressed_stream,
                                      section_headers[header.root_type.section].offset + header.root_type.offset)
    decompressed_stream.seek(header.root_node.offset)
    root_struct = root_struct_ref.resolve(decompressed_stream)

    root_data = read_struct(root_struct, decompressed_stream)
    return root_data
