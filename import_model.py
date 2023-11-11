import math
from typing import Optional

import bpy
from mathutils import Vector, Matrix, Quaternion
from pathlib import Path
import numpy as np

from .granny2 import deserialize_granny2_file, Transform


def import_model(path: Path, mega_arm: Optional[bpy.types.Object] = None):
    granny_data = deserialize_granny2_file(path.read_bytes())
    models_defs = granny_data.get("Models", [])
    if mega_arm is None:
        root = bpy.data.objects.new(path.stem, None)
        root.matrix_local = Matrix.Rotation(math.radians(90), 4, (1, 0, 0))
        bpy.context.scene.collection.objects.link(root)
    else:
        root = mega_arm
        parent = mega_arm
    for model_def in models_defs:
        name = model_def["Name"]
        print("Loading", name)
        skeleton = model_def["Skeleton"]
        if mega_arm is None:
            if skeleton["Bones"]:
                parent = create_skeleton(name, skeleton)
            else:
                parent = bpy.data.objects.new(name, None)
                bpy.context.scene.collection.objects.link(parent)
            parent.parent = root

        for mesh_def in model_def["MeshBindings"]:
            mesh = mesh_def["Mesh"]
            extended_data = mesh["ExtendedData"]
            if extended_data.get("LOD", 0) > 0:
                continue
            assert not mesh["MorphTargets"]
            mesh_name = mesh["Name"]
            print("\tLoading", mesh_name)
            vertex_data = mesh["PrimaryVertexData"]["Vertices"]
            topology = mesh["PrimaryTopology"]
            sub_mesh_infos = topology["Groups"]
            total_indices = 0
            for sub_mesh_info in sub_mesh_infos:
                total_indices += sub_mesh_info["TriCount"]
            material_indices = np.zeros(total_indices, np.int32)
            if topology.get("Indices16", []):
                indices_data = topology["Indices16"]
                index_key = "Int16"
                np_type = np.int16
            else:
                indices_data = topology["Indices"]
                index_key = "Int32"
                np_type = np.uint32
            indices = np.zeros((total_indices, 3), np_type)
            for sub_mesh_info in sub_mesh_infos:
                tri_offset = sub_mesh_info["TriFirst"]
                material_indices[tri_offset:tri_offset + sub_mesh_info["TriCount"]] = sub_mesh_info["MaterialIndex"]
                for i in range(sub_mesh_info["TriCount"]):
                    indices[i + tri_offset][0] = indices_data[i * 3 + tri_offset * 3 + 0][index_key]
                    indices[i + tri_offset][1] = indices_data[i * 3 + tri_offset * 3 + 1][index_key]
                    indices[i + tri_offset][2] = indices_data[i * 3 + tri_offset * 3 + 2][index_key]
            if np_type == np.int16:
                indices = indices.view(np.uint16).astype(np.uint32)

            template = vertex_data[0]
            dtype = []
            for key, value in template.items():
                type1 = np.int32 if isinstance(value[0], int) else np.float32
                dtype.append((key, type1, len(value)))
            vertices = np.zeros(len(vertex_data), dtype)
            for i, vertex in enumerate(vertex_data):
                for attr, value in vertex.items():
                    vertices[i][attr] = value

            mesh_data = bpy.data.meshes.new(mesh_name + f"_MESH")
            mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
            mesh_obj.parent = parent

            mesh_data.from_pydata(vertices["Position"], [], indices)
            mesh_data.update()
            bpy.context.scene.collection.objects.link(mesh_obj)

            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)

            for name, _, _ in dtype:
                if name.startswith("TextureCoordinates"):
                    uv_data = mesh_data.uv_layers.new(name=name)
                    uvs = vertices[name].copy()
                    uvs[:, 1] = 1 - uvs[:, 1]
                    uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())
                    continue
                if name.startswith("DiffuseColor"):
                    vertex_colors = mesh_data.vertex_colors.new(name=name)
                    vertex_colors_data = vertex_colors.data
                    vertex_colors_data.foreach_set("color", vertices[name][vertex_indices].ravel().astype(np.float32)/255)
                    continue
                if name == "QTangent":
                    qtangent_data = vertices["QTangent"].astype(np.float32) / 32767
                    normals = np.zeros((len(qtangent_data), 3), np.float32)
                    for i, q in enumerate(qtangent_data):
                        m = Matrix.Identity(3)
                        m[0][0] = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
                        m[0][1] = 2.0 * (q[0] * q[1] + q[3] * q[2])
                        m[0][2] = 2.0 * (q[0] * q[2] - q[3] * q[1])

                        m[1][0] = 2.0 * (q[0] * q[1] - q[3] * q[2])
                        m[1][1] = 1.0 - 2.0 * (q[0] * q[0] + q[2] * q[2])
                        m[1][2] = 2.0 * (q[1] * q[2] + q[3] * q[0])

                        m[2] = m[0].cross(m[1]) * (-1.0 if q[3] < 0.0 else 1.0)
                        normals[i] = m[2]

                    mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
                    mesh_data.normals_split_custom_set_from_vertices(normals)
                    mesh_data.use_auto_smooth = True
                    continue
                if name == "BoneIndices":
                    if "BoneWeights" not in vertices.dtype.names:
                        continue
                    bone_indices = vertices[name]
                    bone_weights = vertices["BoneWeights"]
                    bone_names = [bone["BoneName"] for bone in mesh["BoneBindings"]]
                    weight_groups = {bone: mesh_obj.vertex_groups.new(name=bone) for bone in bone_names}
                    for n, bone_indices in enumerate(bone_indices):
                        weights = bone_weights[n] / 255
                        for bone_index, weight in zip(bone_indices, weights):
                            if weight > 0:
                                weight_groups[bone_names[bone_index]].add([n], weight, 'REPLACE')
                    modifier = mesh_obj.modifiers.new(type="ARMATURE", name="Armature")
                    modifier.object = parent
                    continue


def create_skeleton(name, skeleton):
    arm_data = bpy.data.armatures.new(name + "_ARMDATA")
    parent = bpy.data.objects.new(name + "_ARM", arm_data)
    bpy.context.scene.collection.objects.link(parent)
    parent.show_in_front = True
    parent.select_set(True)
    bpy.context.view_layer.objects.active = parent
    bpy.ops.object.mode_set(mode='EDIT')
    g_scale = Vector((1, 1, 1))

    def create_bone(bone_data):
        bl_bone = arm_data.edit_bones.new(bone_data["Name"])
        bl_bone.tail = Vector([0, 0, 0.1]) + bl_bone.head
        transform: Transform = bone_data["Transform"]
        for i, row in enumerate(transform.shear_scale):
            g_scale[i] *= np.linalg.norm(row)
        matrix = Matrix.LocRotScale(Vector(transform.translation) * g_scale,
                                    Quaternion((transform.rotation[3], *transform.rotation[:3])),
                                    Vector((1, 1, 1))
                                    )
        parent_id = bone_data["ParentIndex"]
        if parent_id != -1:
            bl_bone.parent = arm_data.edit_bones[parent_id]
            bl_bone.matrix = bl_bone.parent.matrix @ matrix
        else:
            bl_bone.matrix = matrix

    for bone in skeleton["Bones"]:
        create_bone(bone)
    bpy.ops.object.mode_set(mode='OBJECT')
    return parent


def collect_mega_skeleton(path: Path, bones: dict[str, dict]):
    print(f"Collecting bones from {path.stem}")
    granny_data = deserialize_granny2_file(path.read_bytes())
    for skeleton in granny_data["Skeletons"]:
        bone_defs = skeleton["Bones"]
        for bone_def in bone_defs:

            name = bone_def["Name"]
            parent_id = bone_def["ParentIndex"]
            bone_data = bones.get(name, None)

            if bone_data is None:
                bone_data = {}
                transform: Transform = bone_def["Transform"]
                scale = [0, 0, 0]
                for i, row in enumerate(transform.shear_scale):
                    scale[i] = np.linalg.norm(row)
                matrix = Matrix.LocRotScale(Vector(transform.translation),
                                            Quaternion((transform.rotation[3], *transform.rotation[:3])),
                                            Vector(scale)
                                            )

                if parent_id != -1:
                    parent_bone_name = bone_defs[parent_id]["Name"]
                    parent_bone = bones[parent_bone_name]
                    matrix = parent_bone["matrix"] @ matrix
                    bone_data["parent_name"] = parent_bone_name
                bone_data["matrix"] = matrix
            elif "parent_name" not in bone_data and parent_id != -1:
                parent_bone_name = bone_defs[parent_id]["Name"]
                bone_data["parent_name"] = parent_bone_name

            bones[name] = bone_data


def create_mega_skeleton(bones: dict[str, dict]):
    arm_data = bpy.data.armatures.new("MEGA_ARMDATA")
    mega_arm = bpy.data.objects.new("MEGA_ARM", arm_data)
    bpy.context.scene.collection.objects.link(mega_arm)
    mega_arm.show_in_front = True
    mega_arm.select_set(True)
    bpy.context.view_layer.objects.active = mega_arm
    bpy.ops.object.mode_set(mode='EDIT')
    for name, data in bones.items():
        bl_bone = arm_data.edit_bones.new(name)
        bl_bone.tail = Vector([0, 0, 0.1]) + bl_bone.head
        if "parent_name" in data:
            bl_bone.parent = arm_data.edit_bones[data["parent_name"]]
        bl_bone.matrix = data["matrix"]
    mega_arm.matrix_local = Matrix.Rotation(math.radians(90), 4, (1, 0, 0))
    return mega_arm


if __name__ == '__main__':
    # import_model(Path(
    #     r"E:\BG3Stuff\bg3-modders-multitool\UnpackedData\Models\Generated\Public\SharedDev\Assets\Characters\_Anims\Dragonborn\_Male\Resources\DGB_M_NKD_Head_A.GR2"))

    # import_model(Path(
    #     r"D:\SteamLibrary\steamapps\common\Baldurs Gate 3\bg3-modders-multitool\UnpackedData\Models\Generated\Public\SharedDev\Assets\Characters\_Models\Dragonborn\_Male\Resources\DGB_M_NKD_Head_G.GR2"))
    #
    # import_model(Path(
    #     r"D:\SteamLibrary\steamapps\common\Baldurs Gate 3\bg3-modders-multitool\UnpackedData\Models\Generated\Public\SharedDev\Assets\Characters\_Models\Dragonborn\_Male\Resources\DGB_M_NKD_Body_Leaf_A.GR2"))
    # import_model(Path(
    #     r"D:\SteamLibrary\steamapps\common\Baldurs Gate 3\bg3-modders-multitool\UnpackedData\Models\Generated\Public\SharedDev\Assets\Characters\_Models\Dragonborn\_Male\Resources\DGB_M_NKD_Body_Genital_B.GR2"))
    # import_model(Path(
    #     r"D:\SteamLibrary\steamapps\common\Baldurs Gate 3\bg3-modders-multitool\UnpackedData\Models\Generated\Public\SharedDev\Assets\Characters\_Models\Dragonborn\_Male\Resources\DGB_M_NKD_Body_Genital_A.GR2"))
    # import_model(Path(
    #     r"D:\SteamLibrary\steamapps\common\Baldurs Gate 3\bg3-modders-multitool\UnpackedData\Models\Generated\Public\SharedDev\Assets\Characters\_Models\Dragonborn\_Male\Resources\DGB_M_NKD_Tail_B.GR2"))
    bones = {}
    # prefixes = ["NKD", "ARM", "CLT"]
    prefixes = ["_"]
    folder = Path(
        r"E:\BG3Stuff\bg3-modders-multitool\UnpackedData\Models\Generated\Public\Shared\Assets\Characters\_Models\_Creatures\Ogre\Resources")
    # for prefix in prefixes:
    #     for file in folder.rglob(f"*{prefix}*.GR2"):
    #         collect_mega_skeleton(file, bones)
    # mega_arm = create_mega_skeleton(bones)
    # for prefix in prefixes:
    #     for file in folder.rglob(f"*{prefix}*.GR2"):
    #         import_model(file, mega_arm)
    import_model(Path(r"C:\Users\AORUS\Downloads\HAIR_HUM_F_GnomeCut_Long_C_Spring.GR2"), None)
    bpy.context.scene.tool_settings.lock_object_mode = False
    bpy.ops.wm.save_as_mainfile(filepath=r"E:\PY_PROJECTS\GR2Toolkit\test.blend")
