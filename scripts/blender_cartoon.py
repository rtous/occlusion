"""Run inside Blender:  blender -b --python scripts/blender_cartoon.py -- job.json

Poses a cartoon character (a .blend with an SMPL-X armature named "SMPL", rest pose =
SMPL-X T-pose in the canonical frame, see models/) with SMPL-X joint rotations from
Multi-HMR and renders it with the image camera. One RGBA PNG per person.

job.json: {"width", "height", "K": 3x3, "persons": [{"model": path.blend, "rotvec": 53x3,
           "pelvis": [x,y,z] (OpenCV camera coords), "out": path.png}, ...]}
"""
import json
import math
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

# SMPL-X joint order of Multi-HMR's rotvec: global, 21 body joints, 15 + 15 hand joints, jaw
BODY = ["left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
        "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar",
        "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
FINGERS = ["index", "middle", "pinky", "ring", "thumb"]
HAND = lambda side: [f"{side}_{f}{i}" for f in FINGERS for i in (1, 2, 3)]
JOINTS = ["pelvis"] + BODY + HAND("left") + HAND("right") + ["jaw"]
CV2BL = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))   # OpenCV camera coords -> Blender camera coords


def rotvec_to_matrix(v):
    v = Vector(v)
    return Quaternion(v.normalized(), v.length).to_matrix() if v.length > 1e-8 else Matrix.Identity(3)


def render_person(job, p):
    bpy.ops.wm.open_mainfile(filepath=p["model"])
    sc = bpy.context.scene
    arm = bpy.data.objects["SMPL"]
    # the armature space is the SMPL-X canonical frame; its world matrix maps it to camera coords
    # (OpenCV) and then to Blender; the camera sits at the origin looking down -Z
    for name, rv in zip(JOINTS, p["rotvec"]):
        if name not in arm.pose.bones:
            continue
        pb = arm.pose.bones[name]
        B = pb.bone.matrix_local.to_3x3()
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (B.inverted() @ rotvec_to_matrix(rv) @ B).to_quaternion()
    pelvis_rest = arm.data.bones["pelvis"].head_local
    t = Vector(p["pelvis"]) - pelvis_rest           # pelvis rotates about its head, which stays put
    arm.matrix_world = (CV2BL.to_4x4() @ Matrix.Translation(t))

    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        sc.collection.objects.link(cam)
    sc.camera = cam
    cam.matrix_world = Matrix.Identity(4)
    W, H, K = job["width"], job["height"], job["K"]
    c = cam.data
    c.type = "PERSP"
    c.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
    c.sensor_width = c.sensor_height = 36.0
    c.lens = K[0][0] / (W if W >= H else H) * 36.0
    c.shift_x = -(K[0][2] - W / 2) / max(W, H)
    c.shift_y = (K[1][2] - H / 2) / max(W, H)
    c.clip_start, c.clip_end = 0.05, 1000
    for o in bpy.data.objects:                     # the file's lights were placed for another scene
        if o.type == "LIGHT" or (o.type == "MESH" and o.parent is None):
            o.hide_render = True
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sc.collection.objects.link(sun)
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(25), math.radians(-15), 0)   # from the camera, slightly above/left
    r = sc.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.filepath = p["out"]
    bpy.ops.render.render(write_still=True)


job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
for person in job["persons"]:
    render_person(job, person)
    print("rendered", person["out"])
