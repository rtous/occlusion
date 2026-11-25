"""Run inside Blender:  blender -b --python scripts/blender_mixamo.py -- job.json

Poses a Mixamo-rigged character (.fbx from models/) so that its bones follow a set of 3D body
joints, and renders it with the image camera (one RGBA PNG per person). Only joint positions are
needed, so any body model works (SAM 3D Body keypoints, SMPL-X joints, ...).

Retargeting: every bone keeps the character's own length and is rotated so that it points from
its joint to the next one (minimal swing from the parent's orientation). The hips, the chest and
the head get a full frame (up direction + left->right direction) to fix the twist. The character
is scaled so that its leg + torso length matches the person's, and placed at the person's hips.

job.json: {"width", "height", "K": 3x3, "persons": [{"model": path.fbx, "out": path.png,
           "joints": {name: [x, y, z], ...}}]}   (OpenCV camera coordinates, metres)
Joint names: pelvis, neck, head, l_eye, r_eye, and l_/r_ + hip, knee, ankle, toe, shoulder,
elbow, wrist, hand (hand = base of the middle finger). Missing joints leave the bone unposed.
"""
import json
import math
import sys

import bpy
from mathutils import Matrix, Vector

CV2BL = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))  # OpenCV -> Blender cam

# Mixamo bone -> (from joint, to joint) in the person, and the bone whose rest head marks the
# end of the segment in the character (bone tails are arbitrary in Mixamo FBX files, so the rest
# direction is measured between bone heads); joints may be "mid:a,b" (midpoint)
SWING = {
    "Spine": ("mid:l_hip,r_hip", "neck", "Neck"), "Spine1": ("mid:l_hip,r_hip", "neck", "Neck"),
    "Neck": ("neck", "head", "Head"),
    "LeftArm": ("l_shoulder", "l_elbow", "LeftForeArm"), "LeftForeArm": ("l_elbow", "l_wrist", "LeftHand"),
    "LeftHand": ("l_wrist", "l_hand", "LeftHandMiddle1"),
    "RightArm": ("r_shoulder", "r_elbow", "RightForeArm"), "RightForeArm": ("r_elbow", "r_wrist", "RightHand"),
    "RightHand": ("r_wrist", "r_hand", "RightHandMiddle1"),
    "LeftUpLeg": ("l_hip", "l_knee", "LeftLeg"), "LeftLeg": ("l_knee", "l_ankle", "LeftFoot"),
    "LeftFoot": ("l_ankle", "l_toe", "LeftToeBase"),
    "RightUpLeg": ("r_hip", "r_knee", "RightLeg"), "RightLeg": ("r_knee", "r_ankle", "RightFoot"),
    "RightFoot": ("r_ankle", "r_toe", "RightToeBase"),
}
# bone -> (up: from, to), (side: from, to) in target joints, and the same in rest bones (heads)
FRAME = {
    "Hips": ((("mid:l_hip,r_hip", "neck"), ("l_hip", "r_hip")),
             (("Hips", "Neck"), ("LeftUpLeg", "RightUpLeg"))),
    "Spine2": ((("mid:l_hip,r_hip", "neck"), ("l_shoulder", "r_shoulder")),
               (("Hips", "Neck"), ("LeftArm", "RightArm"))),
    "Head": ((("neck", "head"), ("l_eye", "r_eye")),
             (("Neck", "Head"), ("LeftEye", "RightEye"))),
}


def frame(up, side):
    """Orthonormal 3x3 from an up vector and a (roughly perpendicular) side vector."""
    u = up.normalized()
    s = (side - u * side.dot(u)).normalized()
    return Matrix((s, u, s.cross(u))).transposed()


def pose(arm, joints):
    pre = next(b.name for b in arm.data.bones if b.name.endswith("Hips"))[:-4]  # "mixamorigN:"
    bones = arm.data.bones
    rest = {b.name[len(pre):]: b.matrix_local.copy() for b in bones}
    head = lambda n: rest[n].translation if n in rest else None
    has = lambda n: all(x in joints for x in (n[4:].split(",") if n.startswith("mid:") else [n]))

    # scale: character leg + torso length -> person
    def L(pts):
        return sum((pts[i + 1] - pts[i]).length for i in range(len(pts) - 1))
    s = 1.0
    if all(has(n) for n in ("l_hip", "l_knee", "l_ankle", "neck")):
        J = lambda n: Vector(joints[n])
        tgt = L([J("l_hip"), J("l_knee"), J("l_ankle")]) + ((J("l_hip") + J("r_hip")) / 2 - J("neck")).length
        rst = L([head("LeftUpLeg"), head("LeftLeg"), head("LeftFoot")]) + \
            ((head("LeftUpLeg") + head("RightUpLeg")) / 2 - head("Neck")).length
        s = tgt / rst
    arm.matrix_world = CV2BL @ Matrix.Scale(s, 4)          # armature space = OpenCV camera / s

    def P(n):                                              # target joint in armature space
        if n.startswith("mid:"):
            a, b = n[4:].split(",")
            return (Vector(joints[a]) + Vector(joints[b])) / (2 * s)
        return Vector(joints[n]) / s

    G = {}                                                 # posed armature-space matrices
    for b in bones:                                        # bones are listed parents first
        n = b.name[len(pre):]
        R = rest[n]
        if b.parent is None:
            G0 = R.copy()
        else:
            p = b.parent.name[len(pre):]
            G0 = G[p] @ rest[p].inverted() @ R
        Gb = G0
        if n in FRAME and all(has(x) for pair in FRAME[n][0] for x in pair) and \
                all(x in rest for pair in FRAME[n][1] for x in pair):
            (tu, ts), (ru, rs) = FRAME[n]
            Ft = frame(P(tu[1]) - P(tu[0]), P(ts[1]) - P(ts[0]))
            Fr = frame(head(ru[1]) - head(ru[0]), head(rs[1]) - head(rs[0]))
            rot = (Ft @ Fr.inverted()).to_4x4()
            Gb = rot @ R
            if b.parent is None:                           # hips: put the thigh midpoint on the person's
                mid_rest = (head("LeftUpLeg") + head("RightUpLeg")) / 2
                Gb.translation = P("mid:l_hip,r_hip") - rot.to_3x3() @ (mid_rest - R.translation)
            else:
                Gb.translation = G0.translation
        elif n in SWING and all(has(x) for x in SWING[n][:2]) and SWING[n][2] in rest:
            a, c, e = SWING[n]
            d = P(c) - P(a)
            y = (G0 @ R.inverted()).to_3x3() @ (head(e) - head(n))    # current segment direction
            q = y.rotation_difference(d)
            h = G0.translation
            Gb = Matrix.Translation(h) @ q.to_matrix().to_4x4() @ Matrix.Translation(-h) @ G0
        G[n] = Gb
        L_ = (G0.inverted() @ Gb)                          # local pose relative to the inherited one
        pb = arm.pose.bones[b.name]
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis = L_
    return s


def setup_scene(job):
    sc = bpy.context.scene
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.matrix_world = Matrix.Identity(4)                  # at the origin, looking down -Z
    W, H, K = job["width"], job["height"], job["K"]
    c = cam.data
    c.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
    c.sensor_width = c.sensor_height = 36.0
    c.lens = K[0][0] / max(W, H) * 36.0
    c.shift_x = -(K[0][2] - W / 2) / max(W, H)
    c.shift_y = (K[1][2] - H / 2) / max(W, H)
    c.clip_start, c.clip_end = 0.05, 1000
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sc.collection.objects.link(sun)
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(20), math.radians(-25), 0)    # from the camera, above/left
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bgn = world.node_tree.nodes["Background"]
    bgn.inputs[0].default_value = (1, 1, 1, 1)
    bgn.inputs[1].default_value = 0.6
    sc.world = world
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            pass
    sc.view_settings.view_transform = "Standard"
    r = sc.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"


def render_person(job, p):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=p["model"])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    arm.animation_data_clear()                             # Mixamo files carry an animation
    for o in bpy.data.objects:
        if o.type == "MESH" and o.parent is None:          # keep loose meshes with the character
            mw = o.matrix_world.copy()
            o.parent = arm
            o.matrix_parent_inverse = arm.matrix_world.inverted()
            o.matrix_world = mw
    for m in bpy.data.materials:                           # matte look: some Mixamo materials are glossy
        for nd in (m.node_tree.nodes if m.use_nodes else []):
            if nd.type == "BSDF_PRINCIPLED":
                for k, v in (("Roughness", 0.8), ("Specular IOR Level", 0.1), ("Specular", 0.1)):
                    if k in nd.inputs and not nd.inputs[k].is_linked:
                        nd.inputs[k].default_value = v
                if "Roughness" in nd.inputs and nd.inputs["Roughness"].is_linked:
                    for l in list(nd.inputs["Roughness"].links):
                        m.node_tree.links.remove(l)
                    nd.inputs["Roughness"].default_value = 0.8
    pose(arm, p["joints"])
    setup_scene(job)
    bpy.context.scene.render.filepath = p["out"]
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
    for person in job["persons"]:
        render_person(job, person)
        print("rendered", person["out"])
