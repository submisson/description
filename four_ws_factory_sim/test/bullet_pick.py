"""Optional independent contact regression. Run with blender -b -P path/to/bullet_pick.py.

Reads the SDF pallet solids. Only the two kinematic blades are animated. The
pallet has 25 kg mass and a compound collision shape, and remains unconstrained.
This reduced test omits the Gazebo chassis, its joints, the LiDARs and ROS.
"""
from pathlib import Path
import sys
import math
import xml.etree.ElementTree as ET

import bpy
from mathutils import Vector

PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG))
from four_ws_factory_sim.dimensions import HEIGHT_IN, HEIGHT_CARRY, INSERTION, TINE_X

bpy.ops.wm.read_factory_settings(use_empty=True)
sc=bpy.context.scene
sc.render.fps=50
sc.frame_end=525


def cuboid(name,size,xyz):
    bpy.ops.mesh.primitive_cube_add(size=1,location=xyz)
    ob=bpy.context.object
    ob.name=name
    ob.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    return ob


def rigid(ob,mode,shape,friction):
    bpy.context.view_layer.objects.active=ob
    ob.select_set(True)
    bpy.ops.rigidbody.object_add()
    rb=ob.rigid_body
    rb.type=mode
    rb.collision_shape=shape
    rb.friction=friction
    rb.use_margin=True
    rb.collision_margin=.0004
    return rb


root=ET.parse(PKG/'models/epal_1/model.sdf')
centre=float(root.findtext('.//inertial/pose').split()[2])
body=cuboid('free_pallet',(.001,.001,.001),(0,0,centre+.001))
rb=rigid(body,'ACTIVE','COMPOUND',.85)
rb.mass=float(root.findtext('.//inertial/mass'))
rb.use_deactivation=False
for node in root.findall('.//collision'):
    p=list(map(float,node.findtext('pose').split()[:3]))
    size=list(map(float,node.findtext('geometry/box/size').split()))
    ob=cuboid(node.get('name'),size,(0,0,0))
    ob.parent=body
    ob.location=(p[0],p[1],p[2]-centre)
    rigid(ob,'ACTIVE','BOX',.85)
floor=cuboid('floor',(10,10,.1),(0,0,-.05))
rigid(floor,'PASSIVE','BOX',.95)
# Tip from an 180 mm gap to 690 mm penetration. Translate while raised.
poses=[(1,0,.58,HEIGHT_IN),(40,0,.58,HEIGHT_IN),(160,0,.4-INSERTION,HEIGHT_IN),
       (223,0,.4-INSERTION,HEIGHT_CARRY),(240,0,.4-INSERTION,HEIGHT_CARRY),
       (375,.62,.4-INSERTION,HEIGHT_CARRY),(388,.62,.4-INSERTION,HEIGHT_CARRY),
       (451,.62,.4-INSERTION,HEIGHT_IN),(470,.62,.4-INSERTION,HEIGHT_IN),
       (520,.62,.58,HEIGHT_IN)]
for i,x in enumerate(TINE_X):
    blade=cuboid('tine_'+str(i),(.080,.810,.026),(0,0,0))
    b=rigid(blade,'ACTIVE','BOX',.95)
    b.kinematic=True
    for frame,dx,tip,z in poses:
        blade.location=(x+dx,tip+.405,z)
        blade.keyframe_insert(data_path='location',frame=frame)
    for fcurve in blade.animation_data.action.fcurves:
        for key in fcurve.keyframe_points:
            key.interpolation='LINEAR'
world=sc.rigidbody_world
world.substeps_per_frame=10
world.solver_iterations=35
world.point_cache.frame_end=sc.frame_end
data={}
for frame in range(1,sc.frame_end+1):
    sc.frame_set(frame)
    if frame in (35,156,234,385,523):
        evaluated=body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        data[frame]=evaluated.matrix_world.translation.copy()
p=data[35]
entry_motion=(data[156]-p).length
rise=data[234].z-p.z
air_distance=math.hypot(data[385].x-p.x,data[385].y-p.y)
rest=data[523].z-p.z
print(f'Contact regression: before lift {entry_motion:.4f} m, rise {rise:.4f} m, '
      f'loaded travel {air_distance:.4f} m, final dz {rest:.4f} m')
assert entry_motion<.015 and rise>.12 and air_distance>.42 and abs(rest)<.025, data
