import argparse
import json
import os
import sys
import tempfile

import numpy as np

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import util.blender_utils as blender_utils
from util.blender_utils import bpy as bpy

class HiddenPrints:
    def __init__(self, enable=True, suppress_err=False):
        self.enabled = enable
        self.suppress_err = suppress_err
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def __enter__(self):
        if not self.enabled:
            return
        sys.stdout = open(os.devnull, "w")
        if self.suppress_err:
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        sys.stdout.close()
        sys.stdout = self._original_stdout
        if self.suppress_err:
            sys.stderr.close()
            sys.stderr = self._original_stderr

def clear_scene() -> None:
    """Remove all objects from the Blender scene before importing new ones."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def load_fbx(filepath: str) -> bpy.types.Object | None:
    """Import an FBX file and return the armature."""
    bpy.ops.import_scene.fbx(filepath=filepath)
    for obj in bpy.context.selected_objects:
        if obj.type == "ARMATURE":
            return obj
    return None  # No armature found


def get_skeleton_data(fbx_path: str) -> dict:
    """Extract skeleton data from an FBX file and return it in a serializable format."""
    clear_scene()
    armature_obj = load_fbx(fbx_path)
    if not armature_obj:
        msg = f"No armature found in {fbx_path}"
        raise ValueError(msg)

    bones_data = {}
    for bone in armature_obj.data.bones:
        bones_data[bone.name] = {
            'name': bone.name,
            'parent': bone.parent.name if bone.parent else None,
            'children': [child.name for child in bone.children],
            'head': [float(x) for x in bone.head_local],
            'tail': [float(x) for x in bone.tail_local],
            'matrix_local': [[float(x) for x in row] for row in bone.matrix_local]
        }
    clear_scene()
    return bones_data


def copy_pose(source_armature: bpy.types.Object, target_armature: bpy.types.Object) -> None:
    """Copy the pose from the source armature to the target armature."""
    for bone_name, source_bone in source_armature.pose.bones.items():
        if bone_name in target_armature.pose.bones:
            target_bone = target_armature.pose.bones[bone_name]
            source_bone.matrix = target_bone.matrix  # Apply transformation
            bpy.context.view_layer.update()

    # Reset and apply rest pose
    blender_utils.set_rest_bones(source_armature, reset_as_rest=True)


def delete_object_and_mesh(obj: bpy.types.Object) -> None:
    """Delete both the armature and any linked meshes."""
    if obj and obj.name in bpy.data.objects:
        for child in obj.children:
            if child.type == "MESH":
                bpy.data.objects.remove(child, do_unlink=True)
        bpy.data.objects.remove(obj, do_unlink=True)


def apply_transformation_and_export(target_armature: bpy.types.Object, target_fbx_path: str) -> None:
    """Apply the pose and export the transformed model."""
    bpy.context.view_layer.objects.active = target_armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.export_scene.fbx(
        filepath=target_fbx_path,
        apply_unit_scale=True,
        global_scale=1.0,
    )


def apply_pose(source_fbx: str, target_fbx: str, output_fbx: str) -> None:
    """Transfer a pose from a source FBX to a target FBX and export the result."""
    # First clear scene and load source armature
    clear_scene()
    source_armature = load_fbx(source_fbx)
    if not source_armature:
        msg = f"No armature found in source FBX: {source_fbx}"
        raise ValueError(msg)

    # Then load target armature
    target_armature = load_fbx(target_fbx)
    if not target_armature:
        msg = f"No armature found in target FBX: {target_fbx}"
        raise ValueError(msg)

    # Copy pose from target to source
    copy_pose(source_armature, target_armature)

    # Delete target since we don't need it anymore
    delete_object_and_mesh(target_armature)

    # Apply transformation and export
    apply_transformation_and_export(source_armature, output_fbx)

    # Clean up
    clear_scene()

    print(f"Pose transfer completed! Saved to {output_fbx}")


def is_finger(bone_name: str):
    return any(f in bone_name for f in {"Thumb", "Index", "Middle", "Ring", "Pinky"})


def remove_fingers_from_data(data: np.ndarray, bones_idx_dict: dict[str, int], is_bw=False):
    assert data.shape[0] == len(bones_idx_dict)
    if is_bw:
        for k, v in bones_idx_dict.items():
            if is_finger(k):
                hand = "Left" if "Left" in k else "Right"
                data[bones_idx_dict[f"mixamorig:{hand}Hand"]] += data[v]
    data_new = [None] * len(bones_idx_dict)
    for k, v in bones_idx_dict.items():
        if is_finger(k):
            continue
        data_new[v] = data[v]
    data_new = np.stack([x for x in data_new if x is not None], axis=0)
    return data_new


def main(args: argparse.Namespace):
    if isinstance(args.input_path, str):
        data = np.load(args.input_path, allow_pickle=True)
    else:
        assert isinstance(args.input_path, dict)
        data = args.input_path
    # mesh = data["mesh"]
    # if isinstance(mesh, np.ndarray):
    #     mesh: trimesh.Trimesh = mesh.item()
    gs = data["gs"]
    joints = data["joints"]
    joints_tail = data["joints_tail"]
    bw = data["bw"]
    pose = data["pose"]
    bones_idx_dict = data["bones_idx_dict"]
    if isinstance(bones_idx_dict, np.ndarray):
        bones_idx_dict: dict[str, int] = bones_idx_dict.item()
    pose_ignore_list = list(data.get("pose_ignore_list", []))

    if args.remove_fingers:
        joints = remove_fingers_from_data(joints, bones_idx_dict)
        joints_tail = remove_fingers_from_data(joints_tail, bones_idx_dict)
        bw = remove_fingers_from_data(bw.T, bones_idx_dict, is_bw=True).T
        if pose is not None:
            pose = remove_fingers_from_data(pose, bones_idx_dict)
        joints_list = [None] * len(bones_idx_dict)
        for k, v in bones_idx_dict.items():
            joints_list[v] = k
        joints_list = [x for x in joints_list if not is_finger(x)]
        bones_idx_dict = {name: i for i, name in enumerate(joints_list)}
        assert len(bones_idx_dict) == joints.shape[0]

    with HiddenPrints(suppress_err=True):
        blender_utils.reset()

        template = blender_utils.load_file(args.template_path)
        for mesh_obj in blender_utils.get_all_mesh_obj(template):
            bpy.data.objects.remove(mesh_obj, do_unlink=True)
        armature_obj = blender_utils.get_armature_obj(template)
        armature_obj.animation_data_clear()
        with blender_utils.Mode("POSE", armature_obj):
            bpy.ops.pose.select_all(action="SELECT")
            bpy.ops.pose.transforms_clear()
        if args.keep_raw:
            scaling = 1.0
            bpy.context.view_layer.objects.active = armature_obj
            armature_obj.select_set(state=True)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        else:
            matrix_world = armature_obj.matrix_world.copy()
            scaling = matrix_world.to_scale()[0]
            armature_obj.matrix_world.identity()
        blender_utils.update()

        with tempfile.NamedTemporaryFile(suffix=".glb") as f:
            # if not args.keep_raw:
            #     verts = mesh.vertices
            #     verts[:, 1], verts[:, 2] = verts[:, 2].copy(), -verts[:, 1].copy()
            #     mesh.vertices = verts / scaling
            # mesh.export(f.name)
            mesh_obj = blender_utils.load_file(args.mesh_path)
            mesh_obj = blender_utils.get_all_mesh_obj(mesh_obj)[0]
            mesh_data: bpy.types.Mesh = mesh_obj.data
            mesh_obj.name = mesh_data.name = "mesh"
            
            # Apply scale transformation to vertices
            if not args.keep_raw:
                # Swap Y and Z coordinates and negate Y (same as in make_it_animatable.py)
                for vertex in mesh_data.vertices:
                    y, z = vertex.co[1], vertex.co[2]
                    vertex.co[1] = z
                    vertex.co[2] = -y
                    # Apply scaling
                    vertex.co /= scaling
                mesh_data.update()
            for mat in mesh_data.materials:
                for link in mat.node_tree.links:
                    if link.from_node.bl_idname == "ShaderNodeNormalMap":
                        mat.node_tree.links.remove(link)

        blender_utils.set_rest_bones(armature_obj, joints / scaling, joints_tail / scaling, bones_idx_dict)
        if args.keep_raw:
            bpy.context.view_layer.objects.active = armature_obj
            armature_obj.select_set(state=True)
            armature_obj.rotation_mode = "XYZ"
            armature_obj.rotation_euler[0] = np.pi / 2
            bpy.ops.object.transform_apply(rotation=True)
            blender_utils.update()
        blender_utils.set_armature_parent([mesh_obj], armature_obj)
        blender_utils.set_weights([mesh_obj], bw, bones_idx_dict)
        if not args.keep_raw:
            armature_obj.matrix_world = matrix_world
        blender_utils.remove_empty()
        blender_utils.update()
        if pose is not None and args.apply_pose:
            pose_inv = pose
            if not args.pose_local:
                pose_inv[:, :3, 3] /= scaling
            blender_utils.set_bone_pose(armature_obj, pose_inv, bones_idx_dict, local=args.pose_local)
            for bone in armature_obj.pose.bones:
                bone.location = (0, 0, 0)
                if pose_ignore_list and any(x in bone.name for x in pose_ignore_list):
                    bone.matrix_basis = blender_utils.mathutils.Quaternion().to_matrix().to_4x4()
                blender_utils.update()
            if args.reset_to_rest:
                blender_utils.set_rest_bones(armature_obj, reset_as_rest=True)
            if args.rest_path:
                bpy.ops.export_scene.gltf(
                    filepath=args.rest_path,
                    check_existing=False,
                    use_selection=False,
                    export_animations=False,
                    export_rest_position_armature=False,
                    # export_yup=False,
                )
        if args.animation_path:
            blender_utils.load_mixamo_anim(
                [armature_obj, mesh_obj], args.animation_path, do_retarget=args.retarget, inplace=args.inplace
            )

        blender_utils.update()
        if args.output_path.endswith(".fbx"):
            bpy.ops.export_scene.fbx(
                filepath=args.output_path,
                check_existing=False,
                use_selection=False,
                add_leaf_bones=False,
                bake_anim=bool(args.animation_path),
                path_mode="COPY",
                embed_textures=True,
            )
        elif args.output_path.endswith(".blend"):
            scn = bpy.context.scene
            cam_obj = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
            cam_obj.location = (2, -3.5, 1)
            cam_obj.rotation_euler = (75 / 180 * np.pi, 0, 30 / 180 * np.pi)
            scn.collection.objects.link(cam_obj)
            scn.camera = cam_obj

            keyframes = blender_utils.get_keyframes([armature_obj])
            if keyframes:
                scn.frame_start, scn.frame_end = min(keyframes), max(keyframes)
                scn.frame_set(scn.frame_start)
                scn.render.image_settings.file_format = "FFMPEG"
                scn.render.ffmpeg.format = "MPEG4"
            scn.render.resolution_x = 1024
            scn.render.resolution_y = 1024
            scn.render.resolution_percentage = 50

            # if gs is not None:
            #     mesh_obj.hide_set(True)
            #     with tempfile.NamedTemporaryFile(suffix=".ply") as f:
            #         gs = torch.from_numpy(gs)
            #         gs = transform_gs(gs, transform=(Scale(1 / scaling)))
            #         save_gs(gs, f.name)
            #         gs_obj = blender_utils.load_3dgs(f.name)
            #         gs_obj = blender_utils.get_all_mesh_obj(gs_obj)[0]
            #         gs_obj.name = gs_obj.data.name = "gs"
            #     blender_utils.set_armature_parent([gs_obj], armature_obj, type="ARMATURE_NAME", no_inv=True)
            #     blender_utils.set_weights([gs_obj], bw.repeat(4, axis=0), bones_idx_dict)
            #     bpy.ops.sna.dgs__set_render_engine_to_eevee_7516e()
            #     # bpy.ops.sna.dgs__start_camera_update_9eaff()

            if os.path.isfile(args.output_path):
                os.remove(args.output_path)
            bpy.ops.wm.save_as_mainfile(filepath=args.output_path)
        else:
            raise ValueError(f"Unsupported output format: {args.output_path}")


if __name__ == "__main__":    
    # Get all args after -- as those are for our script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["skeleton", "pose", "animation"], required=True)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--mesh_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--get_skeleton", action="store_true", help="Extract skeleton data from input_path")
    parser.add_argument("--template_path", type=str, default=None)
    parser.add_argument("--keep_raw", default=False, action="store_true")
    parser.add_argument("--rest_path", type=str, default=None)
    parser.add_argument("--pose_local", default=False, action="store_true")
    parser.add_argument("--reset_to_rest", default=False, action="store_true")
    parser.add_argument("--remove_fingers", default=False, action="store_true")
    parser.add_argument("--animation_path", type=str, default=None)
    parser.add_argument("--retarget", default=False, action="store_true")
    parser.add_argument("--inplace", default=False, action="store_true")
    parser.add_argument("--apply_pose", default=False, action="store_true")
    parser.add_argument("--source_fbx", type=str, help="Source FBX file for pose transfer")
    parser.add_argument("--target_fbx", type=str, help="Target FBX file for pose transfer")
    
    # Parse only our arguments after --
    args = parser.parse_args(argv)
    
    if args.task == "skeleton":
        if not args.input_path:
            msg = "--input_path is required with --get_skeleton"
            raise ValueError(msg)
        skeleton_data = get_skeleton_data(args.input_path)
        print("Skeleton data:")
        print(json.dumps(skeleton_data))
        print("Skeleton data end")
    elif args.task == "pose":
        if not args.output_path:
            msg = "--output_path is required for pose transfer"
            raise ValueError(msg)
        apply_pose(args.source_fbx, args.target_fbx, args.output_path)
    elif args.task == "animation":
        if not args.output_path:
            msg = "--output_path is required for normal processing"
            raise ValueError(msg)
        main(args)
    else:
        msg = "Must provide either --get_skeleton with --input_path, or (--source_fbx, --target_fbx) for pose transfer, or (--input_path, --mesh_path) for normal processing"
        raise ValueError(msg)
