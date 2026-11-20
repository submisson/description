#!/usr/bin/env python3
"""Regenerate the installed indoor world, local OGRE materials and EPAL geometry."""
import math
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PACKAGE=Path(__file__).resolve().parents[1]
MODELS=PACKAGE/'models'
MATERIAL=MODELS/'factory_finishes'/'materials'
WOOD='Factory/Wood'
FLOOR='Factory/Concrete'
DARK=(.13,.17,.19)
RACK=(.10,.28,.39)
BEAM=(.90,.39,.10)
YELLOW=(.98,.69,.12)
GREY=(.53,.59,.60)
TEAL=(.15,.42,.42)
WHITE=(.81,.84,.81)
RED=(.78,.19,.13)


def add(parent, key, value=None, **attrs):
    e=E.SubElement(parent,key,{k:str(v) for k,v in attrs.items()})
    if value is not None:
        e.text=str(value)
    return e


def nums(v):
    return ' '.join(f'{a:.7g}' for a in v)


def write(path, root):
    path.parent.mkdir(parents=True,exist_ok=True)
    E.indent(root,space='  ')
    E.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def model_config(name, caption):
    root=E.Element('model')
    add(root,'name',caption)
    add(root,'version','1.0')
    add(root,'sdf','model.sdf',version='1.7')
    write(MODELS/name/'model.config',root)


def finishes():
    scripts=MATERIAL/'scripts'
    tex=MATERIAL/'textures'
    for d in (scripts,tex):
        d.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(721)
    n=512
    x,y=np.meshgrid(np.arange(n),np.arange(n))
    streak=(np.sin(x*.016+np.sin(y*.03)*.8)+np.sin(x*.023+y*.24)*.32)
    grain=4*np.sin(y*.53+2*streak)+6*np.sin(y*.11+.6*np.sin(x*.012))+rng.normal(0,2,(n,n))
    for cx,cy in ((120,101),(395,386)):
        r=np.sqrt(((x-cx)/2.7)**2+((y-cy)*1.4)**2)
        grain+=13*np.cos(r*1.8)*np.exp(-r/11)
    resampling=getattr(Image,'Resampling',Image)
    sizes=[(8,6.),(32,3.)]
    concrete=rng.normal(0,2.8,(n,n))
    for k,weight in sizes:
        coarse=Image.fromarray(rng.integers(0,255,(k,k),dtype=np.uint8)).resize((n,n),resampling.BICUBIC)
        concrete+=weight*(np.asarray(coarse)/64.-2.)
    for name,base,noise in [('wood',(176,139,91),grain),('concrete',(148,151,145),concrete)]:
        pixels=np.clip(np.array(base)[None,None,:]+noise[:,:,None],0,255).astype(np.uint8)
        Image.fromarray(pixels).save(tex/(name+'.png'))
    def ogre(name,filename,scale):
        return f"""material {name}
{{
  technique
  {{
    pass
    {{
      ambient 0.84 0.84 0.84
      diffuse 0.95 0.95 0.95
      specular 0.08 0.08 0.08 1 8
      texture_unit
      {{
        texture {filename}
        scale {scale}
        filtering trilinear
      }}
    }}
  }}
}}
"""
    materials=ogre(WOOD,'wood.png','1 1')+ogre(FLOOR,'concrete.png','.080 .125')
    font=ImageFont.truetype('DejaVuSans-Bold.ttf',156)
    for letter in 'ABCD':
        tile=Image.new('RGB',(256,256),(45,60,59))
        draw=ImageDraw.Draw(tile)
        draw.rounded_rectangle((9,9,246,246),radius=8,outline=(229,190,92),width=9)
        draw.text((128,122),letter,font=font,anchor='mm',fill=(240,230,204))
        file='bay_'+letter+'.png'
        tile.save(tex/file)
        materials+=ogre('Factory/Bay'+letter,file,'1 1')
    (scripts/'factory.material').write_text(materials)
    bay_sign_mesh()
    model_config('factory_finishes','Factory surface materials')
    root=E.Element('sdf',version='1.7')
    m=add(root,'model',name='factory_finishes')
    add(m,'static','true')
    add(m,'link',name='origin')
    write(MODELS/'factory_finishes'/'model.sdf',root)


def appearance(element, color, emissive=False):
    mat=add(element,'material')
    if isinstance(color,str):
        script=add(mat,'script')
        for sub in ('scripts','textures'):
            add(script,'uri','model://factory_finishes/materials/'+sub)
        add(script,'name',color)
    else:
        add(mat,'ambient',nums([*(u*.85 for u in color),1]))
        add(mat,'diffuse',nums([*color,1]))
        if emissive:
            add(mat,'emissive',nums([*(u*.8 for u in color),1]))


def shape(link,name,xyz,size,color,*,rpy=(0,0,0),solid=True,kind='box',visual_mesh=None,emissive=False,mu=.85):
    for mode in ('collision','visual'):
        if mode=='collision' and not solid:
            continue
        component=add(link,mode,name=name if mode=='collision' else 'v_'+name)
        add(component,'pose',nums((*xyz,*rpy)))
        geo=add(component,'geometry')
        if visual_mesh and mode=='visual':
            add(add(geo,'mesh'),'uri',visual_mesh)
        elif kind=='box':
            add(add(geo,'box'),'size',nums(size))
        else:
            cyl=add(geo,'cylinder')
            add(cyl,'radius',size[0])
            add(cyl,'length',size[1])
        if mode=='visual':
            appearance(component,color,emissive)
        elif color==WOOD or name=='floor':
            ode=add(add(add(component,'surface'),'friction'),'ode')
            add(ode,'mu',mu)
            add(ode,'mu2',mu)


def box_faces(size, center=(0, 0, 0)):
    a, b, c = np.asarray(size) / 2
    corners = np.array([(x,y,z) for x in (-a,a) for y in (-b,b) for z in (-c,c)]) + center
    return [[corners[i] for i in order] for order in
            [(0,1,3,2), (4,6,7,5), (0,4,5,1), (2,3,7,6), (0,2,6,4), (1,5,7,3)]]


def bay_sign_mesh():
    width, depth = .45, .002
    # The lettering stays at local z=0. The back of the plate rests on the floor.
    faces = box_faces((width, width, depth), center=(0, 0, -depth/2))

    def sign_uv(p, n):
        if abs(n[2]) > .5:
            return p[0]/width + .5, p[1]/width + .5
        along = p[1] if abs(n[0]) > .5 else p[0]
        return along/width + .5, 1 + p[2]/depth

    write_textured_mesh(MODELS/'factory_finishes'/'meshes'/'bay_sign.dae', faces, sign_uv)


def plank_mesh(name,center,size,runner):
    # Clip the outside pallet corners at 45 degrees and chamfer the runner edges.
    # All clipping works in local coordinates. Per-face UVs follow the long grain.
    path=MODELS/'epal_1'/'meshes'/(name+'.dae')
    faces=box_faces(size)
    planes=[]
    for sx in (-1,1):
        for sy in (-1,1):
            planes.append((np.array([sx,sy,0.]),.6+.4-.015-sx*center[0]-sy*center[1]))
    if runner:
        # Top edges facing a fork wheel are relieved on the bottom boards.
        for sy in (-1,1):
            if center[1]*sy<=0:
                planes.append((np.array([0.,sy,1.]),(size[1]+size[2])/2-.010))
    for normal,offset in planes:
        new=[]
        cap=[]
        for face in faces:
            poly=[]
            for i,p in enumerate(face):
                q=face[(i+1)%len(face)]
                dp,dq=np.dot(p,normal)-offset,np.dot(q,normal)-offset
                if dp<=1e-9:
                    poly.append(p)
                if (dp>1e-9)!=(dq>1e-9):
                    cut=p+(q-p)*dp/(dp-dq)
                    poly.append(cut)
                    cap.append(cut)
            if len(poly)>=3:
                new.append(poly)
        if len(cap)>=3:
            cap=np.unique(np.round(cap,9),axis=0)
            if len(cap)>=3:
                mid=cap.mean(axis=0)
                u=cap[0]-mid
                v=np.cross(normal,u)
                phi=np.arctan2((cap-mid)@v/np.linalg.norm(v),(cap-mid)@u/np.linalg.norm(u))
                new.append(list(cap[np.argsort(phi)]))
        faces=new

    def grain_uv(p, n):
        if abs(n[2]) > .6:
            u,v = (p[0],p[1]) if size[0]>=size[1] else (p[1],p[0])
        else:
            u,v = (p[0] if abs(n[1])>abs(n[0]) else p[1]),p[2]
        return u*1.9+center[0], v*3.1+center[1]

    write_textured_mesh(path, faces, grain_uv)
    return 'model://epal_1/meshes/'+path.name


def write_textured_mesh(path, faces, texcoord):
    verts=[]
    norms=[]
    uvs=[]
    indices=[]
    for poly in faces:
        n=np.cross(poly[1]-poly[0],poly[2]-poly[0])
        length=np.linalg.norm(n)
        if length<1e-10:
            continue
        n/=length
        start=len(verts)
        for p in poly:
            verts.append(p)
            norms.append(n)
            uvs.append(texcoord(p,n))
        for j in range(1,len(poly)-1):
            indices.extend([start,start+j,start+j+1])
    root=E.Element('COLLADA',xmlns='http://www.collada.org/2005/11/COLLADASchema',version='1.4.1')
    asset=add(root,'asset')
    add(asset,'unit',name='meter',meter=1)
    add(asset,'up_axis','Z_UP')
    mesh=add(add(add(root,'library_geometries'),'geometry',id='geometry'),'mesh')
    for key,data,params in [('pos',verts,'XYZ'),('norm',norms,'XYZ'),('uv',uvs,'ST')]:
        source=add(mesh,'source',id=key)
        add(source,'float_array',nums(np.asarray(data).ravel()),id=key+'_a',count=len(data)*len(params))
        acc=add(add(source,'technique_common'),'accessor',source='#'+key+'_a',count=len(data),stride=len(params))
        for s in params:
            add(acc,'param',name=s,type='float')
    add(add(mesh,'vertices',id='vertices'),'input',semantic='POSITION',source='#pos')
    tri=add(mesh,'triangles',count=len(indices)//3)
    # Gazebo Classic expects a separate index slot per input in each corner.
    add(tri,'input',semantic='VERTEX',source='#vertices',offset=0)
    add(tri,'input',semantic='NORMAL',source='#norm',offset=1)
    add(tri,'input',semantic='TEXCOORD',source='#uv',offset=2,set=0)
    add(tri,'p',' '.join(f'{i} {i} {i}' for i in indices))
    vs=add(add(root,'library_visual_scenes'),'visual_scene',id='scene')
    add(add(vs,'node',id='piece'),'instance_geometry',url='#geometry')
    add(add(root,'scene'),'instance_visual_scene',url='#scene')
    write(path,root)


def pallet():
    root=E.Element('sdf',version='1.7')
    model=add(root,'model',name='epal_1')
    add(model,'static','false')
    add(model,'allow_auto_disable','false')
    link=add(model,'link',name='wood')
    pieces=[]
    def wood(name,xyz,size):
        pieces.append((np.array(xyz),np.array(size)))
        uri=plank_mesh(name,xyz,size,name.startswith('runner'))
        shape(link,name,xyz,size,WOOD,visual_mesh=uri)
    for j,(y,width) in enumerate([(-.35,.10),(0,.145),(.35,.10)]):
        wood('runner_'+str(j),(0,y,.011),(1.2,width,.022))
        for i,x in enumerate((-.5275,0,.5275)):
            wood(f'block_{i}_{j}',(x,y,.061),(.145,width,.078))
    for i,x in enumerate((-.5275,0,.5275)):
        wood('cross_'+str(i),(x,0,.111),(.145,.8,.022))
    edge=-.4
    for i,width in enumerate((.145,.100,.145,.100,.145)):
        wood('deck_'+str(i),(0,edge+width/2,.133),(1.2,width,.022))
        edge+=width+.04125
    # Volume-weighted centres, parallel-axis theorem, 25 kg empty pallet.
    vol=sum(np.prod(size) for _,size in pieces)
    com=sum(p*np.prod(size) for p,size in pieces)/vol
    inertia=np.zeros((3,3))
    for p,size in pieces:
        mass=25*np.prod(size)/vol
        r=p-com
        inertia+=np.diag(mass*(np.sum(size**2)-size**2)/12)+mass*(np.dot(r,r)*np.eye(3)-np.outer(r,r))
    el=add(link,'inertial')
    add(el,'pose',nums([*com,0,0,0]))
    add(el,'mass',25)
    moment=add(el,'inertia')
    for key,x,y in [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]:
        add(moment,key,float(inertia[x,y]))
    model_config('epal_1','EPAL 1 empty wooden pallet')
    write(MODELS/'epal_1'/'model.sdf',root)


def static(world,name):
    m=add(world,'model',name=name)
    add(m,'static','true')
    return add(m,'link',name='structure')


def diagonal(link,name,p,q,width,color,solid=False):
    p,q=np.array(p),np.array(q)
    d=q-p
    shape(link,name,(p+q)/2,(width,width,np.linalg.norm(d)),color,
          rpy=(0,math.atan2(math.hypot(d[0],d[1]),d[2]),math.atan2(d[1],d[0])),solid=solid)


def rack(world,x,y,name):
    link=static(world,name)
    for i,dx in enumerate((-2.65,0,2.65)):
        for j,dy in enumerate((-.57,.57)):
            shape(link,f'post_{i}_{j}',(x+dx,y+dy,1.7),(.105,.105,3.4),RACK)
            shape(link,f'guard_{i}_{j}',(x+dx,y+dy,.19),(.15,.15,.38),YELLOW)
        for z in (.28,1.80):
            diagonal(link,f'brace_{i}_{z}',(x+dx,y-.56,z),(x+dx,y+.56,z+1.35),.045,GREY)
    for i,z in enumerate((.32,1.7,3.02)):
        for j,dy in enumerate((-.56,.56)):
            shape(link,f'beam_{i}_{j}',(x,y+dy,z),(5.35,.10,.15),BEAM)
        shape(link,f'shelf_{i}',(x,y,z+.069),(5.30,1.12,.054),GREY)
        for j,dx in enumerate((-2.1,-1.12,.65,1.68)):
            if (i+j)%5==0:
                continue
            h=.54+.12*((i+2*j)%4)
            shape(link,f'carton_{i}_{j}',(x+dx,y,z+.097+h/2),(.81,.76,h),
                  [( .64,.48,.32),(.71,.57,.39),(.55,.45,.33)][(i+j)%3])
            shape(link,f'band_{i}_{j}',(x+dx+.16,y-.382,z+.098+h/2),(.033,.006,h),
                  (.27,.30,.27),solid=False)


def plant():
    root=E.Element('sdf',version='1.7')
    w=add(root,'world',name='factory')
    add(w,'gravity','0 0 -9.81')
    physics=add(w,'physics',type='ode',name='factory_ode')
    add(physics,'max_step_size','.002')
    add(physics,'real_time_update_rate',500)
    ode=add(physics,'ode')
    solve=add(ode,'solver')
    add(solve,'type','quick')
    add(solve,'iters',90)
    add(solve,'sor',1.15)
    constraints=add(ode,'constraints')
    add(constraints,'contact_surface_layer','.0003')
    add(constraints,'contact_max_correcting_vel','.25')
    scene=add(w,'scene')
    add(scene,'ambient','.56 .58 .60 1')
    add(scene,'background','.77 .81 .83 1')
    add(scene,'shadows','false')
    gui=add(w,'gui')
    cam=add(gui,'camera',name='operator_view')
    add(cam,'pose','3.75 -5.90 2.70 0 0.32 2.15')
    add(cam,'view_controller','orbit')
    light=add(w,'light',name='clerestory',type='directional')
    add(light,'direction','-.44 .43 -.88')
    add(light,'diffuse','.83 .83 .78 1')
    add(light,'specular','.24 .25 .25 1')
    add(light,'cast_shadows','true')
    shell=static(w,'factory_shell')
    shape(shell,'floor',(0,0,-.09),(28,18,.18),FLOOR,mu=1.1)
    for i,y in enumerate((-9.,9.)):
        shape(shell,'wall_'+str(i),(0,y,2.75),(28,.16,5.5),(.68,.72,.72))
        shape(shell,'wall_base_'+str(i),(0,y-math.copysign(.091,y),.56),(27.95,.026,1.12),(.31,.39,.40),solid=False)
        for j,x in enumerate(range(-12,14,4)):
            shape(shell,f'glass_{i}_{j}',(x,y-math.copysign(.09,y),3.75),(3.57,.045,1.34),(.45,.64,.68),solid=False)
            shape(shell,f'mullion_{i}_{j}',(x,y-math.copysign(.12,y),3.75),(.045,.04,1.34),GREY,solid=False)
    for x in (-14.,14.):
        shape(shell,'end_'+str(x),(x,0,2.75),(.16,18,5.5),(.65,.69,.70))
    for j,x in enumerate((-12,-8,-4,0,4,8,12)):
        for i,y in enumerate((-8.78,8.78)):
            shape(shell,f'column_{j}_{i}',(x,y,2.70),(.20,.22,5.4),RACK)
            shape(shell,f'foot_{j}_{i}',(x,y,.4),(.24,.26,.8),YELLOW)
        shape(shell,'rafter_'+str(j),(x,0,5.29),(.13,17.62,.24),GREY)
        for k,y in enumerate((-4.,4.)):
            shape(shell,f'luminaire_{j}_{k}',(x,y,5.12),(.32,1.8,.08),(.95,.95,.85),solid=False,emissive=True)
    for k,y in enumerate((-4.,3.)):
        light=add(w,'light',name='high_bay_'+str(k),type='point')
        add(light,'pose',nums((0,y,4.98,0,0,0)))
        add(light,'diffuse','.55 .54 .49 1')
        add(light,'specular','.08 .08 .08 1')
        add(light,'cast_shadows','false')
        att=add(light,'attenuation')
        for key,n in [('range',16),('constant',.7),('linear',.075),('quadratic',.014)]:
            add(att,key,n)
    for j,y in enumerate((-4.8,0,4.8)):
        shape(shell,'door_'+str(j),(-13.90,y,1.95),(.05,3.4,3.9),(.43,.49,.50))
        for i in range(12):
            shape(shell,f'slat_{j}_{i}',(-13.866,y,.19+i*.304),(.02,3.37,.017),GREY,solid=False)
        for s in (-1,1):
            shape(shell,f'bollard_{j}_{s}',(-13.25,y+s*1.92,.43),(.095,.86),YELLOW,kind='cylinder')
    for i,x in enumerate((-8.1,-1.6,5.)):
        rack(w,x,6.60,'rack_'+str(i+1))
    for i,x in enumerate((-8.4,5.6)):
        l=static(w,'machining_cell_'+str(i))
        y=-6.65
        shape(l,'pedestal',(x,y,.19),(2.95,1.55,.38),DARK)
        shape(l,'enclosure',(x,y,1.12),(2.8,1.48,1.50),TEAL)
        shape(l,'window',(x+.28,y+.75,1.24),(1.58,.035,.71),(.20,.34,.38))
        shape(l,'console',(x-.98,y+.79,1.05),(.45,.15,.57),WHITE)
        shape(l,'display',(x-.98,y+.875,1.13),(.31,.021,.22),DARK,solid=False)
        for k,c in enumerate((.33,.5,.66)):
            shape(l,'lamp_'+str(k),(x-1.12,y,1.84+c/2),(.053,.084),(RED,YELLOW,(.18,.68,.38))[k],kind='cylinder',solid=False)
    conveyor=static(w,'roller_conveyor')
    for y in (-4.8,-3.82):
        shape(conveyor,'side_'+str(y),(10.2,y,.73),(4.9,.09,.15),RACK)
        for x in (8.10,10.25,12.3):
            shape(conveyor,f'leg_{x}_{y}',(x,y,.34),(.09,.09,.68),DARK)
    for i in range(28):
        shape(conveyor,'roller_'+str(i),(7.87+i*.174,-4.31,.80),(.055,.95),GREY,kind='cylinder',rpy=(math.pi/2,0,0))
    paint=static(w,'floor_markings')
    for i,y in enumerate((1.8,4.2)):
        shape(paint,'aisle_'+str(i),(0,y,.001),(25,.07,.002),YELLOW,solid=False)
    for i in range(9):
        shape(paint,'crosswalk_'+str(i),(-10.85+i*.27,3,.002),(.14,2.26,.003),WHITE,solid=False)
    positions=[('A',.665,-2.32,0.),('B',4.10,-2.52,.09),('C',-3.60,-2.42,-.105),('D',8.4,2.8,math.pi/2)]
    for name,x,y,yaw in positions:
        inc=add(w,'include')
        add(inc,'uri','model://epal_1')
        add(inc,'name','pallet_'+name)
        add(inc,'pose',nums((x,y,.001,0,0,yaw)))
        shape(paint,'station_'+name,(x+.96*math.cos(yaw),y+.96*math.sin(yaw),.002),
              (.45,.45,.003),'Factory/Bay'+name,rpy=(0,0,yaw),solid=False,
              visual_mesh='model://factory_finishes/meshes/bay_sign.dae')
        for s in (-1,1):
            # Corner bay markings outside the pallet footprint.
            for k,t in enumerate((-1,1)):
                dx,dy=s*.75,t*.56
                xx,yy=x+dx*math.cos(yaw)-dy*math.sin(yaw),y+dx*math.sin(yaw)+dy*math.cos(yaw)
                shape(paint,f'bay_{name}_{s}_{k}',(xx,yy,.0015),(.24,.05,.002),WHITE,rpy=(0,0,yaw),solid=False)
    state=add(w,'plugin',name='gazebo_ros_state',filename='libgazebo_ros_state.so')
    add(add(state,'ros'),'namespace','/gazebo')
    add(state,'update_rate',20)
    write(PACKAGE/'worlds'/'factory.world',root)


if __name__=='__main__':
    finishes()
    pallet()
    plant()
