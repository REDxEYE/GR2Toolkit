import ctypes
from pathlib import Path
from typing import TypeVar, get_args, get_origin, Generic, Sequence, Iterator, \
    Any, Type, Union
import numpy as np

_path = Path(__file__).absolute().parent

T = TypeVar("T")
V = TypeVar("V")


class CType(Generic[T]):
    pass


# class Array(CType[T]):
#     pass


class Ptr(CType[T]):
    pass


class CStr:
    pass


class ArrayOfPtrCType(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("_size", ctypes.c_uint32),
        ("_ptr", ctypes.c_void_p)
    ]

    def __class_getitem__(cls, item: Type):
        copy = {"_t": item, "_pack_": 1}
        return type(cls.__name__ + item.__name__, (cls,), copy)

    _size: int
    _ptr: ctypes.c_void_p
    _t: Any

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, item):
        if item > self._size:
            raise IndexError("ArrayOfPtr index is out of bounds")
        array_ptr = ctypes.cast(self._ptr, ctypes.POINTER(ctypes.POINTER(self._t) * self._size))
        return array_ptr.contents[item].contents

    def get_ptr(self):
        return self._ptr


class ArrayPtrCType(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("_size", ctypes.c_uint32),
        ("_ptr", ctypes.c_void_p)
    ]

    def __class_getitem__(cls, item: Type):
        copy = {"_t": item, "_pack_": 1}
        return type(cls.__name__ + item.__name__, (cls,), copy)

    _size: int
    _ptr: ctypes.c_void_p
    _t: Any

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, item):
        if item > self._size:
            raise IndexError("ArrayPtr index is out of bounds")
        array_ptr = ctypes.cast(self._ptr, ctypes.POINTER(self._t * self._size))
        return array_ptr.contents[item]

    def get_ptr(self):
        return self._ptr


class ArrayWrapper(Sequence[T]):
    def __init__(self, array):
        self._array = array

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, index: Union[int, slice]) -> Sequence[T]:
        if isinstance(index, slice):
            items = []
            for i in range(index.start, index.stop, index.step):
                items.append(self._array[i])
            return items
        return self._array[index]

    def __iter__(self) -> Iterator:
        return iter(self._array[i] for i in range(len(self._array)))


class ArrayOfPtr(ArrayWrapper[T]):
    def __init__(self, array: ArrayOfPtrCType):
        super().__init__(array)


class ArrayPtr(ArrayWrapper[T]):

    def __init__(self, array: ArrayPtrCType):
        super().__init__(array)

    def as_np(self, ctype):
        return np.ctypeslib.as_array(ctypes.cast(self._array.get_ptr(), ctypes.POINTER(ctype)),
                                     shape=(len(self._array),))


class Array(Generic[T, V]):
    pass


class CStruct(type(ctypes.Structure)):
    def __new__(cls, class_name, bases, classdict):
        if bases == (ctypes.Structure,) and "_fields_" not in classdict:
            fields = classdict["_fields_"] = []
            classdict["_pack_"] = 1

            annotations = classdict.get("__annotations__", {})
            for field_name, field_type in annotations.copy().items():
                if get_origin(field_type) is Ptr:
                    def _create_getter(name: str):
                        def _getter(self):
                            ptr = getattr(self, "__" + name)
                            if ctypes.cast(ptr, ctypes.c_void_p).value is None:
                                return None
                            return ptr.contents

                        return _getter

                    classdict[field_name] = property(_create_getter(field_name))
                    fields.append(("__" + field_name, ctypes.POINTER(get_args(field_type)[0])))
                elif get_origin(field_type) is ArrayOfPtr:
                    def _create_getter(name: str):
                        def _getter(self):
                            return ArrayOfPtr(getattr(self, "__" + name))

                        return _getter

                    classdict[field_name] = property(_create_getter(field_name))
                    fields.append(("__" + field_name, ArrayOfPtrCType[get_args(field_type)[0]]))
                elif get_origin(field_type) is ArrayPtr:
                    def _create_getter(name: str):
                        def _getter(self):
                            return ArrayPtr(getattr(self, "__" + name))

                        return _getter

                    classdict[field_name] = property(_create_getter(field_name))
                    fields.append(("__" + field_name, ArrayPtrCType[get_args(field_type)[0]]))
                elif field_type is CStr:
                    def _create_getter(name: str):
                        def _getter(self):
                            return getattr(self, name).decode("utf8")

                        return _getter

                    classdict[field_name] = property(_create_getter("__" + field_name))
                    fields.append(("__" + field_name, ctypes.c_char_p))
                elif get_origin(field_type) is Array:
                    def _create_getter(name: str):
                        def _getter(self):
                            return tuple(getattr(self, name))

                        return _getter

                    inner_type, count = get_args(field_type)
                    classdict[field_name] = property(_create_getter("__" + field_name))
                    fields.append(("__" + field_name, inner_type * count))
                else:
                    fields.append((field_name, field_type))
            classdict["__annotations__"] = annotations
        return super().__new__(cls, class_name, bases, classdict)


class ArtToolInfo(ctypes.Structure, metaclass=CStruct):
    from_art_tool_name: CStr
    art_tool_major_revision: ctypes.c_int32
    art_tool_minor_revision: ctypes.c_int32
    pad: ctypes.c_int32
    units_per_meter: ctypes.c_float
    origin: Array[ctypes.c_float, 3]
    right_vector: Array[ctypes.c_float, 3]
    up_vector: Array[ctypes.c_float, 3]
    back_vector: Array[ctypes.c_float, 3]


class ExporterInfo(ctypes.Structure, metaclass=CStruct):
    exporter_name: CStr
    exporter_major_revision: ctypes.c_int32
    exporter_minor_revision: ctypes.c_int32
    exporter_customization: ctypes.c_int32
    exporter_build_number: ctypes.c_int32


class Textures(ctypes.Structure):
    pass


class Material(ctypes.Structure):
    pass


class Skeleton(ctypes.Structure):
    pass


class VertexData(ctypes.Structure):
    pass


class TriTopology(ctypes.Structure):
    pass


class VertexAnnotationSets(ctypes.Structure):
    pass


class StringMember(ctypes.Structure, metaclass=CStruct):
    type: ctypes.c_uint32
    name: CStr
    struct_offset: ctypes.c_void_p
    count: ctypes.c_uint32
    x: Array[ctypes.c_uint32, 4]


class PrimaryVertexData(ctypes.Structure, metaclass=CStruct):
    vertices_strings: Ptr[StringMember]
    vertices_count: ctypes.c_int32
    vertices: ctypes.c_void_p
    vertex_annotation_sets: ArrayPtr[VertexAnnotationSets]


class MorphTargets(ctypes.Structure):
    pass


class Groups(ctypes.Structure):
    pass


class VertexToVertexMap(ctypes.Structure):
    pass


class VertexToTriangleMap(ctypes.Structure):
    pass


class SideToNeighborMap(ctypes.Structure):
    pass


class BonesForTriangle(ctypes.Structure):
    pass


class TriangleToBoneIndices(ctypes.Structure):
    pass


class TriAnnotationSets(ctypes.Structure):
    pass


class PrimaryTopology(ctypes.Structure, metaclass=CStruct):
    groups: ArrayPtr[Groups]
    indices: ArrayPtr[ctypes.c_uint32]
    indices16: ArrayPtr[ctypes.c_uint16]
    vertex_to_vertex_map: ArrayPtr[VertexToVertexMap]
    vertex_to_triangle_map: ArrayPtr[VertexToTriangleMap]
    side_to_neighbor_map: ArrayPtr[SideToNeighborMap]
    bones_for_triangle: ArrayPtr[BonesForTriangle]
    triangle_to_bone_indices: ArrayPtr[TriangleToBoneIndices]
    tri_annotation_sets: ArrayPtr[TriAnnotationSets]


class MaterialBinding(ctypes.Structure):
    pass


class BoneBinding(ctypes.Structure, metaclass=CStruct):
    bone_name: CStr
    obb_min: Array[ctypes.c_float, 3]
    obb_max: Array[ctypes.c_float, 3]
    triangle_indices: ArrayPtr[ctypes.c_int32]


class Mesh(ctypes.Structure, metaclass=CStruct):
    name: CStr
    primary_vertex_data: Ptr[PrimaryVertexData]
    morph_targets: ArrayPtr[MorphTargets]
    primary_topology: Ptr[PrimaryTopology]
    material_bindings: ArrayPtr[MaterialBinding]
    bone_bindings: ArrayPtr[BoneBinding]
    extended_data: ctypes.c_void_p

    def get_indices(self):
        return Granny2Native.get_mesh_indices(self)

    # def get_vertex_buffer(self):
    #     return Granny2Native.get_mesh_vertices(self)

    def get_vertex_type(self) -> StringMember:
        return Granny2Native.get_mesh_vertex_type(self)


class Model(ctypes.Structure):
    pass


class TrackGroup(ctypes.Structure):
    pass


class Animation(ctypes.Structure):
    pass


class GrannyFileInfo(ctypes.Structure, metaclass=CStruct):
    art_tool_info: Ptr[ArtToolInfo]
    exporter_info: Ptr[ExporterInfo]
    from_file_name: CStr
    textures: ArrayOfPtr[Textures]
    materials: ArrayOfPtr[Material]
    skeletons: ArrayOfPtr[Skeleton]
    vertex_datas: ArrayOfPtr[VertexData]
    tri_topologies: ArrayOfPtr[TriTopology]
    meshes: ArrayOfPtr[Mesh]
    models: ArrayOfPtr[Model]
    track_groups: ArrayOfPtr[TrackGroup]
    animations: ArrayOfPtr[Animation]


FUNCTYPE_A = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_char_p)
FUNCTYPE_B = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_char_p)


class GrannyLogger(ctypes.Structure):
    _fields_ = [
        ('a', FUNCTYPE_A),
        # ('b', FUNCTYPE_B)
        ('b', ctypes.c_int32),
        ('c', ctypes.c_int32),
    ]


class Granny2Native:
    lib = ctypes.CDLL(str(_path / "granny2.dll"))

    _granny_decompress_data = lib.GrannyDecompressData
    _granny_decompress_data.argtypes = [ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32,
                                        ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p]
    _granny_decompress_data.restype = ctypes.c_bool

    _granny_begin_file_decompression = lib.GrannyBeginFileDecompression
    _granny_begin_file_decompression.argtypes = [ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_char_p,
                                                 ctypes.c_int32, ctypes.c_char_p]
    _granny_begin_file_decompression.restype = ctypes.c_void_p

    _granny_decompress_incremental = lib.GrannyDecompressIncremental
    _granny_decompress_incremental.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p]
    _granny_decompress_incremental.restype = ctypes.c_bool

    _granny_end_file_decompression = lib.GrannyEndFileDecompression
    _granny_end_file_decompression.argtypes = [ctypes.c_void_p]
    _granny_end_file_decompression.restype = ctypes.c_bool

    # __int64 __fastcall GrannyReadEntireFileFromMemory(int a1, __int64 a2)
    _granny_read_entire_file_from_memory = lib.GrannyReadEntireFileFromMemory
    _granny_read_entire_file_from_memory.argtypes = [ctypes.c_int32, ctypes.c_char_p]
    _granny_read_entire_file_from_memory.restype = ctypes.c_void_p

    # __int64 __fastcall GrannyReadEntireFile(LPCSTR lpFileName)
    _granny_read_entire_file = lib.GrannyReadEntireFile
    _granny_read_entire_file.argtypes = [ctypes.c_char_p]
    _granny_read_entire_file.restype = ctypes.c_void_p

    # void __fastcall GrannyFreeFile(GrannyFile* file)
    _granny_free_file = lib.GrannyReadEntireFile
    _granny_free_file.argtypes = [ctypes.c_void_p]
    _granny_free_file.restype = None

    # __int64 __fastcall GrannyFilterAllMessages(int a1, __int64 a2)
    _granny_filter_all_messages = lib.GrannyFilterAllMessages
    _granny_filter_all_messages.argtypes = [ctypes.c_char]
    _granny_filter_all_messages.restype = ctypes.c_uint8

    # __int64 __fastcall GrannyGetFileInfo(__int64 a1)
    _granny_get_file_info = lib.GrannyGetFileInfo
    _granny_get_file_info.argtypes = [ctypes.c_void_p]
    _granny_get_file_info.restype = ctypes.POINTER(GrannyFileInfo)

    # void __fastcall GrannyGetCameraBack(__int64 a1, Vector3f *buffer)
    _granny_get_camera_back = lib.GrannyGetCameraBack
    _granny_get_camera_back.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    _granny_get_camera_back.restype = None

    # __int64 __fastcall GrannyGetMeshVertexType(__int64 a1)
    _granny_get_mesh_vertex_type = lib.GrannyGetMeshVertexType
    _granny_get_mesh_vertex_type.argtypes = [ctypes.POINTER(Mesh)]
    _granny_get_mesh_vertex_type.restype = ctypes.POINTER(StringMember)

    # __int64 __fastcall GrannyGetMeshVertices(__int64 a1)
    _granny_get_mesh_vertices = lib.GrannyGetMeshVertices
    _granny_get_mesh_vertices.argtypes = [ctypes.POINTER(Mesh)]
    _granny_get_mesh_vertices.restype = ctypes.c_void_p

    # __int64 __fastcall GrannyGetMeshIndices(__int64 a1)
    _granny_get_mesh_indices = lib.GrannyGetMeshIndices
    _granny_get_mesh_indices.argtypes = [ctypes.POINTER(Mesh)]
    _granny_get_mesh_indices.restype = ctypes.c_void_p

    # __int64 __fastcall GrannyGetMeshIndexCount(__int64 a1)
    _granny_get_mesh_index_count = lib.GrannyGetMeshIndexCount
    _granny_get_mesh_index_count.argtypes = [ctypes.POINTER(Mesh)]
    _granny_get_mesh_index_count.restype = ctypes.c_void_p

    # __int64 __fastcall GrannyGetMeshBytesPerIndex(__int64 a1)
    _granny_get_mesh_bytes_per_index = lib.GrannyGetMeshBytesPerIndex
    _granny_get_mesh_bytes_per_index.argtypes = [ctypes.POINTER(Mesh)]
    _granny_get_mesh_bytes_per_index.restype = ctypes.c_void_p

    # __int64 __fastcall GrannyGetFileInfo(int a1, __int64 a2)
    _granny_file_crc_is_valid_from_memory = lib.GrannyFileCRCIsValidFromMemory
    _granny_file_crc_is_valid_from_memory.argtypes = [ctypes.c_int32, ctypes.c_char_p]
    _granny_file_crc_is_valid_from_memory.restype = ctypes.c_bool

    # __int64 __fastcall GrannySetLogCallback(int a1)
    _granny_set_log_callback = lib.GrannySetLogCallback
    _granny_set_log_callback.argtypes = [ctypes.POINTER(GrannyLogger)]
    _granny_set_log_callback.restype = None

    @classmethod
    def decompress(cls, data_format: int, compressed: bytes, decompressed_size: int, stop0: int, stop1: int,
                   stop2: int):
        decompressed = bytes(decompressed_size)
        ok = cls._granny_decompress_data(data_format, False, len(compressed), compressed, stop0, stop1, stop2,
                                         decompressed)
        if not ok:
            raise RuntimeError("Failed to decompress Oodle compressed section.")
        return decompressed

    @classmethod
    def decompress4(cls, compressed: bytes, decompressed_size: int):
        work_mem = bytes(0x4000)
        decompressed = bytes(decompressed_size)
        state = cls._granny_begin_file_decompression(4, False, decompressed_size, decompressed, 0x4000, work_mem)
        pos = 0
        comp_size = len(compressed)
        while pos < comp_size:
            chunk_size = min(comp_size - pos, 0x2000)
            increment_ok = cls._granny_decompress_incremental(state, chunk_size, compressed[pos:])
            if not increment_ok:
                raise RuntimeError("Failed to decompress GR2 section increment.")
            pos += chunk_size
        ok = cls._granny_end_file_decompression(state)
        if not ok:
            raise RuntimeError("Failed to finish GR2 section decompression.")
        return decompressed

    @classmethod
    def read_entire_file_from_memory(cls, data: bytes):
        return cls._granny_read_entire_file_from_memory(len(data), data)

    @classmethod
    def read_entire_file(cls, path: str):
        return cls._granny_read_entire_file(path.encode("ascii"))

    @classmethod
    def file_crc_is_valid_from_memory(cls, data: bytes) -> bool:
        return cls._granny_file_crc_is_valid_from_memory(len(data), data)

    @classmethod
    def filter_all_messages(cls, flag):
        return cls._granny_filter_all_messages(flag)

    @classmethod
    def get_file_info(cls, granny_file):
        return cls._granny_get_file_info(granny_file)

    @classmethod
    def get_camera_back(cls, granny_file) -> np.ndarray:
        vec = np.zeros((3,), np.float32)
        ptr = vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cls._granny_get_camera_back(granny_file, ptr)
        return vec

    @classmethod
    def get_mesh_bytes_per_index(cls, mesh: Mesh) -> int:
        return cls._granny_get_mesh_bytes_per_index(mesh)

    @classmethod
    def get_mesh_index_count(cls, mesh: Mesh) -> int:
        return cls._granny_get_mesh_index_count(mesh)

    @classmethod
    def get_mesh_indices(cls, mesh: Mesh) -> np.ndarray:
        dtype = ctypes.c_uint32 if cls.get_mesh_bytes_per_index(mesh) == 4 else ctypes.c_uint16
        count = cls.get_mesh_index_count(mesh)
        ptr = cls._granny_get_mesh_indices(mesh)
        return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(dtype)),
                                     shape=(count,))

    # @classmethod
    # def get_mesh_vertices(cls, mesh: Mesh) -> np.ndarray:
    #     ptr = cls._granny_get_mesh_vertices(mesh)
    #     return None

    @classmethod
    def get_mesh_vertex_type(cls, mesh: Mesh):
        vertex_type = cls._granny_get_mesh_vertex_type(mesh)
        return vertex_type.contents

    @classmethod
    def free_file(cls, granny_file):
        cls._granny_free_file(granny_file)

    @classmethod
    def set_logging_callback(cls, log_func0):
        logger = GrannyLogger(FUNCTYPE_A(log_func0), 0, 0)
        cls._granny_set_log_callback(logger)


class GrannyFile:
    def __init__(self, ptr: ctypes.c_void_p):
        self._ptr = ptr

    @classmethod
    def from_path(cls, path: Path):
        g = Granny2Native.read_entire_file(str(path))
        if g is None:
            raise RuntimeError("Failed to read file")
        return cls(g)

    @classmethod
    def from_memory(cls, data: bytes):
        g = Granny2Native.read_entire_file_from_memory(data)
        if g is None:
            raise RuntimeError("Failed to read file")
        return cls(g)

    def file_info(self) -> GrannyFileInfo:
        return Granny2Native.get_file_info(self._ptr).contents

    def free(self):
        Granny2Native.free_file(self._ptr)
