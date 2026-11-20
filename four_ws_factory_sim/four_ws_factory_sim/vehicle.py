"""Build Gazebo SDF and matching TF URDF from one joint/shape description."""
from dataclasses import dataclass, field
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from .dimensions import BLADE_Z, TINE_X, TOOL_X, SHIFT_LIMIT, CLEAR_REACH

ZERO = (0., 0., 0.)
STEEL = (.30, .38, .40, 1.)
SLIDER = (.12, .24, .25, 1.)


def tag(parent, key, text_value=None, **attributes):
    el = ET.SubElement(parent, key, {k: str(v) for k, v in attributes.items()})
    if text_value is not None:
        el.text = str(text_value)
    return el


def numbers(values):
    return ' '.join(f'{n:.10g}' for n in values)


@dataclass
class Shape:
    name: str
    kind: str
    size: tuple
    xyz: tuple = ZERO
    rpy: tuple = ZERO
    show: bool = True
    solid: bool = True
    color: tuple = STEEL
    mu: float = .85


@dataclass
class Body:
    name: str
    parent: str = ''
    joint: str = ''
    kind: str = 'fixed'
    xyz: tuple = ZERO
    rpy: tuple = ZERO
    axis: tuple = ZERO
    limits: tuple = ()             # lower, upper, force, speed
    mass: float = .1
    com: tuple = ZERO
    extent: tuple = (.07,.07,.07)  # cuboid inertia approximation
    shapes: list = field(default_factory=list)


def mesh(name, file, xyz=ZERO, rpy=ZERO):
    return Shape(name, 'mesh', (file,), xyz, rpy, solid=False)


def box(name, size, xyz, **options):
    return Shape(name, 'box', size, xyz, **options)


def bodies(kit_dir):
    cad = 'package://four_ws_vehicle_description/mesh/'
    parts = 'package://four_ws_factory_sim/meshes/'
    root = Body('base_link', mass=480, com=(TOOL_X,.055,.48), extent=(1.76,.97,.82))
    root.shapes = [mesh('body',parts+'body.dae',(TOOL_X,0,0)),
                   mesh('cabinet_glazing',parts+'cabinet_glazing.dae',(TOOL_X,0,0)),
                   mesh('sensor_beam',cad+'sensor_rail.dae',(TOOL_X,0,0)),
                   box('bay_floor',(.76,.83,.044),(TOOL_X,.112,.402),show=False),
                   box('rear_crossbar',(1.76,.055,.055),(TOOL_X,.498,.445),show=False),
                   box('top_bar',(1.99,.066,.038),(TOOL_X,.44,1.705),show=False)]
    for sign in (-1,1):
        x = TOOL_X + sign*.643
        root.shapes += [box('deck_'+str(sign),(.518,1.066,.077),(x,-.005,.458),show=False),
                        box('cabinet_'+str(sign),(.46,.20,1.14),(x,.44,1.058),show=False),
                        box('rail_'+str(sign),(.048,.99,.036),(TOOL_X+sign*.345,-.02,.422),
                            color=SLIDER)]
    out = [root]
    for edge, x, y in [('FL',1.2165,.340148044),('FR',1.2165,-.339851956),
                       ('RL',0.,.339851956),('RR',0.,-.340148044)]:
        susp, steer, wheel = [stem+'_'+edge for stem in ('susp','steer','wheel')]
        out.append(Body(susp,'base_link',susp+'_joint',xyz=(x,y,.4055),
                        mass=4.5,extent=(.20,.16,.18),shapes=[mesh('cad',cad+susp+'.dae')]))
        out.append(Body(steer,susp,steer+'_joint','revolute',axis=(0,0,1),
                        limits=(-math.pi/2,math.pi/2,520.,1.25),mass=8,
                        com=(0,0,-.13),extent=(.20,.18,.24),shapes=[mesh('cad',cad+steer+'.dae')]))
        out.append(Body(wheel,steer,wheel+'_joint','continuous',xyz=(0,math.copysign(.003,y),-.2305),
                        axis=(0,1,0),limits=(-1e16,1e16,110.,9.),mass=9,extent=(.30,.16,.30),
                        shapes=[mesh('cad',cad+wheel+'.dae'),
                                Shape('tyre','cylinder',(.175,.099),rpy=(math.pi/2,0,0),show=False,mu=1.15)]))
    inner = Body('reach_inner','base_link','reach_inner_joint','prismatic',(TOOL_X,0,0),
                 axis=(0,-1,0),limits=(0.,.59,5200.,.14),mass=14,com=(0,.02,.416),extent=(.68,.93,.06))
    for i,x in enumerate((-.34,.34)):
        inner.shapes.append(box('runner_'+str(i),(.031,.98,.035),(x,-.012,.426),color=SLIDER))
    mast = Body('reach_mast','reach_inner','reach_outer_joint','prismatic',axis=(0,-1,0),
                limits=(0.,.59,5200.,.14),mass=47,com=(0,.433,.89),extent=(.75,.18,1.12),
                shapes=[mesh('mast',parts+'mast.dae')])
    for i,x in enumerate((-.34,.34)):
        mast.shapes.append(box('upright_'+str(i),(.052,.12,1.02),(x,.441,.991),show=False))
    mast.shapes.append(box('foot',(.745,.16,.085),(0,.425,.515),show=False))
    shift = Body('shift_carriage','reach_mast','side_shift_joint','prismatic',axis=(1,0,0),
                 limits=(-SHIFT_LIMIT,SHIFT_LIMIT,2300.,.11),mass=6,com=(0,.376,.743),extent=(.64,.05,.08),
                 shapes=[box('guide',(.66,.042,.045),(0,.373,.743),color=SLIDER)])
    fork = Body('fork_carrier','shift_carriage','fork_lift_joint','prismatic',axis=(0,0,1),
                limits=(-.420,.62,6800.,.16),mass=29,com=(0,.084,.625),extent=(.60,.84,.62),
                shapes=[mesh('original_tines_and_backrest',parts+'forks.dae'),
                        box('backrest',(.60,.045,.52),(0,.350,.776),show=False)])
    for name,x in zip(('a','b'),TINE_X):
        # One contact solid per flat working length, aligned with the original tapered CAD visual.
        fork.shapes.append(box('blade_'+name,(.080,.81,.026),(x,-.134,BLADE_Z),show=False,mu=.95))
        fork.shapes.append(box('heel_'+name,(.08,.05,.41),(x,.296,.665),show=False))
    out.extend([inner,mast,shift,fork])
    def load(file):
        return yaml.safe_load((Path(kit_dir)/'config'/file).read_text())
    def pose(entry):
        return tuple(float(entry[k]) for k in ('x','y','z')), tuple(float(entry[k]) for k in ('roll','pitch','yaw'))
    xyz,rpy = pose(load('sensors_calibration.yaml')['base_link']['sensor_kit_base_link'])
    out.append(Body('sensor_kit_base_link','base_link','sensor_kit_base_joint',xyz=xyz,rpy=rpy))
    calibration = load('sensor_kit_calibration.yaml')['sensor_kit_base_link']
    for k in range(1,6):
        frame = 'mid360_l'+str(k)
        xyz,rpy = pose(calibration[frame+'_base_link'])
        out.append(Body(frame+'_base_link','sensor_kit_base_link',frame+'_mount_joint',xyz=xyz,rpy=rpy,
                        mass=.265,shapes=[mesh('housing',cad+'sensor_mid360.dae'),
                                          box('case',(.065,.065,.060),ZERO,show=False)]))
        out.append(Body(frame,frame+'_base_link',frame+'_joint',mass=.01))
    # Preserve the other sensor housings. Their drivers are outside this simulation.
    for frame,p in calibration.items():
        if frame.startswith('mid360_'):
            continue
        xyz,rpy = pose(p)
        device = frame.split('/')[0]
        file = ('sensor_'+device if device.startswith('CAM_') else
                'sensor_imu' if device=='tamagawa' else 'sensor_gnss') + '.dae'
        undo_yaw = -rpy[2] if device.startswith('CAM_') else 0.
        out.append(Body(frame,'sensor_kit_base_link',device+'_mount_joint',xyz=xyz,rpy=rpy,
                        shapes=[mesh('cad',cad+file,rpy=(0,0,undo_yaw))]))
    return out


def geometry(parent, shape, sdf, paths):
    geo = tag(parent,'geometry')
    if shape.kind == 'mesh':
        uri = shape.size[0]
        if sdf:
            pkg,rel = uri[len('package://'):].split('/',1)
            tag(tag(geo,'mesh'),'uri',(Path(paths[pkg])/rel).resolve().as_uri())
        else:
            tag(geo,'mesh',filename=uri)
    elif shape.kind == 'box':
        if sdf:
            tag(tag(geo,'box'),'size',numbers(shape.size))
        else:
            tag(geo,'box',size=numbers(shape.size))
    else:
        radius,length = shape.size
        if sdf:
            cyl=tag(geo,'cylinder')
            tag(cyl,'radius',radius)
            tag(cyl,'length',length)
        else:
            tag(geo,'cylinder',radius=radius,length=length)


def add_body(model, b, sdf, paths):
    link=tag(model,'link',name=b.name)
    def origin(el, xyz, rpy=ZERO):
        if sdf:
            tag(el,'pose',numbers((*xyz,*rpy)))
        else:
            tag(el,'origin',xyz=numbers(xyz),rpy=numbers(rpy))
    if sdf and b.parent:
        tag(link,'pose',numbers((*b.xyz,*b.rpy)),relative_to=b.parent)
    inertia=tag(link,'inertial')
    origin(inertia,b.com)
    a,c,d=b.extent
    vals=dict(ixx=b.mass*(c*c+d*d)/12,iyy=b.mass*(a*a+d*d)/12,izz=b.mass*(a*a+c*c)/12,
              ixy=0.,ixz=0.,iyz=0.)
    if sdf:
        tag(inertia,'mass',b.mass)
        moment=tag(inertia,'inertia')
        for key,num in vals.items():
            tag(moment,key,num)
    else:
        tag(inertia,'mass',value=b.mass)
        tag(inertia,'inertia',**vals)
    for shape in b.shapes:
        for kind,enabled in [('visual',shape.show),('collision',shape.solid)]:
            if not enabled:
                continue
            el=tag(link,kind,name=shape.name if kind=='collision' else 'v_'+shape.name)
            origin(el,shape.xyz,shape.rpy)
            geometry(el,shape,sdf,paths)
            if sdf and kind=='visual' and shape.name=='cabinet_glazing':
                tag(el,'transparency',.68)
            if kind=='visual' and shape.kind!='mesh':
                mat=tag(el,'material',**({} if sdf else {'name':shape.name+'_finish'}))
                if sdf:
                    tag(mat,'ambient',numbers(shape.color))
                    tag(mat,'diffuse',numbers(shape.color))
                else:
                    tag(mat,'color',rgba=numbers(shape.color))
            if sdf and kind=='collision':
                friction=tag(tag(tag(el,'surface'),'friction'),'ode')
                tag(friction,'mu',shape.mu)
                tag(friction,'mu2',shape.mu)
    if b.parent:
        j=tag(model,'joint',name=b.joint,type='revolute' if sdf and b.kind=='continuous' else b.kind)
        if sdf:
            tag(j,'parent',b.parent)
            tag(j,'child',b.name)
        else:
            tag(j,'parent',link=b.parent)
            tag(j,'child',link=b.name)
            origin(j,b.xyz,b.rpy)
        if b.limits:
            lower,upper,effort,speed=b.limits
            if sdf:
                axis=tag(j,'axis')
                tag(axis,'xyz',numbers(b.axis))
                lim=tag(axis,'limit')
                for key,val in zip(('lower','upper','effort','velocity'),b.limits):
                    tag(lim,key,val)
            else:
                tag(j,'axis',xyz=numbers(b.axis))
                options={} if b.kind=='continuous' else dict(lower=lower,upper=upper)
                tag(j,'limit',effort=effort,velocity=speed,**options)
    return link


PROFILES={'light':(360,32), 'standard':(500,40), 'dense':(720,56)}


def add_lidar(link, frame, profile, backend):
    h,v=PROFILES[profile]
    sensor=tag(link,'sensor',name=frame+'_laser',type=backend)
    tag(sensor,'always_on','true')
    tag(sensor,'update_rate',10)
    tag(sensor,'visualize','false')
    ray=tag(sensor,'ray')
    scan=tag(ray,'scan')
    # Avoid sampling the 0/360-degree seam twice.
    for name,n,lo,hi in [('horizontal',h,-math.pi,math.pi-2*math.pi/h),
                         ('vertical',v,math.radians(-7),math.radians(52))]:
        part=tag(scan,name)
        for key,val in [('samples',n),('resolution',1),('min_angle',lo),('max_angle',hi)]:
            tag(part,key,val)
    rng=tag(ray,'range')
    for key,val in [('min',.1),('max',40.),('resolution',.01)]:
        tag(rng,key,val)
    noise=tag(ray,'noise')
    tag(noise,'type','gaussian')
    tag(noise,'mean',0)
    tag(noise,'stddev',.003)
    plugin=tag(sensor,'plugin',name=frame+'_points',filename='libgazebo_ros_ray_sensor.so')
    tag(tag(plugin,'ros'),'remapping','~/out:=/sensing/lidar/'+frame.split('_')[1]+'/pointcloud_before_sync')
    tag(plugin,'output_type','sensor_msgs/PointCloud2')
    tag(plugin,'frame_name',frame)


def descriptions(paths, profile='standard', backend='ray'):
    if backend not in ('ray','gpu_ray','pattern') or profile not in PROFILES:
        raise ValueError('Invalid LiDAR configuration')
    sdf=ET.Element('sdf',version='1.7')
    model=tag(sdf,'model',name='four_ws')
    tag(model,'allow_auto_disable','false')
    urdf=ET.Element('robot',name='four_ws')
    for b in bodies(paths['four_ws_sensor_kit_description']):
        add_body(urdf,b,False,paths)
        link=add_body(model,b,True,paths)
        if backend!='pattern' and b.name in {f'mid360_l{i}' for i in range(1,6)}:
            add_lidar(link,b.name,profile,backend)
        if b.name=='fork_carrier':
            for blade in ('a','b'):
                sensor=tag(link,'sensor',name='contact_'+blade,type='contact')
                tag(sensor,'always_on','true')
                tag(sensor,'update_rate',30)
                tag(tag(sensor,'contact'),'collision','blade_'+blade)
                plugin=tag(sensor,'plugin',name='fork_contact_'+blade,filename='libgazebo_ros_bumper.so')
                tag(tag(plugin,'ros'),'remapping','bumper_states:=/fork/contact_'+blade)
                tag(plugin,'frame_name','world')
    if backend=='pattern':
        plugin=tag(model,'plugin',name='mid360_five_heads',filename='libfactory_mid360_pattern.so')
        h,v=PROFILES[profile]
        tag(plugin,'points_per_frame',h*v)
    plugin=tag(model,'plugin',name='factory_drive',filename='libfactory_vehicle_control.so')
    tag(plugin,'clear_reach',CLEAR_REACH)
    return ET.tostring(urdf,encoding='unicode'),ET.tostring(sdf,encoding='unicode')
