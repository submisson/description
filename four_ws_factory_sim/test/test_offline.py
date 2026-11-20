"""CPU ray/box intersections, independent of ROS. Run from the workspace root."""
import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np

PKG=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PKG))
from four_ws_factory_sim.vehicle import descriptions, PROFILES
from four_ws_factory_sim.dimensions import TOOL_X, TINE_X, TIP_Y, BLADE_Z, HEIGHT_IN, INSERTION
from four_ws_factory_sim.perception import front_faces

PATHS={p.name:p for p in PKG.parent.iterdir() if p.is_dir()}
PALLET=ET.parse(PKG/'models/epal_1/model.sdf')
WORLD=ET.parse(PKG/'worlds/factory.world')
def vec(e,path):
    return np.fromstring(e.findtext(path),sep=' ')


def ryz(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def ray_scan(profile,yaw=0.,origin=(.665,-2.32,0),hide_block=False,pattern_frame=None):
    # Analytic first return from all 20 actual pallet collision solids and the floor.
    # Use both right-side inverted CAD extrinsics. L5 is the primary view at station A.
    import yaml
    cfg=yaml.safe_load((PATHS['four_ws_sensor_kit_description']/'config/sensor_kit_calibration.yaml').read_text())['sensor_kit_base_link']
    n,m=PROFILES[profile]
    a,b=np.meshgrid(np.linspace(-math.pi,math.pi,n,endpoint=False),np.deg2rad(np.linspace(-7,52,m)))
    rays=np.stack((np.cos(b)*np.cos(a),np.cos(b)*np.sin(a),np.sin(b)),axis=-1).reshape(-1,3)
    rot=ryz(yaw)
    cloud=[]
    rng=np.random.default_rng(139)
    for k in (4,5):
        mount=cfg[f'mid360_l{k}_base_link']
        o=np.array([mount[a] for a in 'xyz'])
        if pattern_frame is not None:
            sample=np.arange(n*m)*(1.)+pattern_frame*n*m+1.
            az=2*math.pi*np.mod(.7548776662466927*sample+.137*(k-1),1.)
            vz=-.12186934340514748+(.788010753606722+.12186934340514748)*np.mod(.5698402909980532*sample+.193*(k-1),1.)
            xy=np.sqrt(1.-vz*vz)
            rays=np.column_stack((xy*np.cos(az),xy*np.sin(az),vz))
        directions=rays*np.array([1,-1,-1])  # supplied CAD roll=pi
        local_o=(o-origin)@rot
        local_d=directions@rot
        inv=1/np.where(abs(local_d)>1e-10,local_d,1e-10)
        best=np.where(directions[:,2]<-1e-8,-o[2]/np.minimum(directions[:,2],-1e-8),41.)
        for link in PALLET.findall('.//collision'):
            if hide_block and link.get('name','').startswith('block_1_'):
                continue
            centre=vec(link,'pose')[:3]
            half=vec(link,'geometry/box/size')/2
            t1=(centre-half-local_o)*inv
            t2=(centre+half-local_o)*inv
            entry=np.minimum(t1,t2).max(axis=1)
            leave=np.maximum(t1,t2).min(axis=1)
            good=(entry>0.10)&(leave>=entry)&(entry<best)
            best[good]=entry[good]
        mask=(best<8)
        cloud.append(o+directions[mask]*(best[mask]+rng.normal(0,.003,mask.sum()))[:,None])
    return np.concatenate(cloud)


class GeometryTests(unittest.TestCase):
    def test_factory_render_meshes(self):
        # Classic reads a triangle corner as one index per input semantic.
        # Validate the actual files referenced by the world and its pallet model.
        uris = {e.text for tree in (WORLD, PALLET) for e in tree.findall('.//mesh/uri')}
        self.assertTrue(uris)
        for uri in sorted(uris):
            with self.subTest(mesh=uri):
                mesh = ET.parse(PKG/'models'/uri.removeprefix('model://')).find('.//{*}mesh')
                sources = {el.get('id'): el for el in mesh}
                groups = mesh.findall('{*}triangles')
                self.assertTrue(groups)
                for group in groups:
                    inputs = group.findall('{*}input')
                    offsets = [int(item.get('offset')) for item in inputs]
                    self.assertEqual(sorted(offsets), list(range(len(inputs))))
                    count = int(group.get('count'))
                    self.assertGreater(count, 0)
                    indices = np.fromstring(group.findtext('{*}p'), sep=' ', dtype=int)
                    self.assertEqual(indices.size, 3 * count * len(inputs))
                    corners = indices.reshape(-1, len(inputs))
                    for item, offset in zip(inputs, offsets):
                        semantic = item.get('semantic')
                        source = sources[item.get('source')[1:]]
                        if semantic == 'VERTEX':
                            pos = source.find("{*}input[@semantic='POSITION']")
                            source = sources[pos.get('source')[1:]]
                        accessor = source.find('{*}technique_common/{*}accessor')
                        values = np.fromstring(source.findtext('{*}float_array'), sep=' ')
                        values = values.reshape(int(accessor.get('count')), int(accessor.get('stride')))
                        self.assertTrue(np.isfinite(values).all())
                        column = corners[:, offset]
                        self.assertGreaterEqual(column.min(), 0)
                        self.assertLess(column.max(), len(values))
                        if semantic == 'VERTEX':
                            positions = values[column]
                            a, b, c = positions.reshape(-1, 3, 3).transpose(1, 0, 2)
                            areas = np.linalg.norm(np.cross(b - a, c - a), axis=1)
                            self.assertTrue((areas > 0).all())

    def test_all_profiles_and_frames(self):
        for quality,(h,v) in PROFILES.items():
            u,s=descriptions(PATHS,quality)
            urdf,sdf=ET.fromstring(u),ET.fromstring(s)
            self.assertEqual({e.get('name') for e in urdf.findall('link')},
                             {e.get('name') for e in sdf.findall('model/link')})
            self.assertEqual({e.get('name') for e in urdf.findall('joint')},
                             {e.get('name') for e in sdf.findall('model/joint')})
            sensors=sdf.findall(".//sensor[@type='ray']")
            self.assertEqual(len(sensors),5)
            self.assertEqual({q.findtext('plugin/frame_name') for q in sensors},{f'mid360_l{k}' for k in range(1,6)})
            for q in sensors:
                self.assertEqual((int(q.findtext('ray/scan/horizontal/samples')),int(q.findtext('ray/scan/vertical/samples'))),(h,v))
            for e in sdf.findall('.//mesh/uri'):
                from urllib.parse import urlparse, unquote
                self.assertTrue(Path(unquote(urlparse(e.text).path)).is_file(),e.text)
            for uri in WORLD.findall('.//uri')+PALLET.findall('.//uri'):
                if uri.text.startswith('model://'):
                    self.assertTrue((PKG/'models'/uri.text[8:]).exists(),uri.text)
            self.assertEqual(len(sdf.findall(".//sensor[@type='contact']")),2)

    def test_pallet_and_entry(self):
        parts=PALLET.findall('.//collision')
        self.assertEqual(len(parts),20)
        bounds=np.array([(vec(e,'pose')[:3]-vec(e,'geometry/box/size')/2,
                          vec(e,'pose')[:3]+vec(e,'geometry/box/size')/2) for e in parts])
        np.testing.assert_allclose(bounds[:,1,:].max(0)-bounds[:,0,:].min(0),(1.2,.8,.144),atol=1e-8)
        self.assertEqual(len(WORLD.findall('world/include')),4)
        for x in TINE_X:
            # Sweep each physical blade from first entry to full working depth.
            # It must clear all twenty components, then meet wood on the lift.
            contact=False
            for depth in np.linspace(0,INSERTION,61):
                blade=np.array([[x-.04,.4-depth,HEIGHT_IN-.013],
                                [x+.04,.4-depth+.81,HEIGHT_IN+.013]])
                overlap=(np.minimum(blade[1],bounds[:,1])-np.maximum(blade[0],bounds[:,0])>1e-6).all(1)
                self.assertFalse(overlap.any())
            blade[:,2]+=.064
            contact=(np.minimum(blade[1],bounds[:,1])-np.maximum(blade[0],bounds[:,0])>1e-6).all(1)
            self.assertTrue(contact.any())

    def test_measured_face(self):
        for quality,yaw in [('light',0),('standard',0),('standard',.09),('dense',-.105)]:
            with self.subTest(quality=quality,yaw=yaw):
                faces=front_faces(ray_scan(quality,yaw))
                self.assertTrue(faces)
                face=faces[0]
                np.testing.assert_allclose([face.x,face.y],
                    [.665-.4*math.sin(yaw),-2.32+.4*math.cos(yaw)],atol=.045)
                self.assertAlmostEqual(face.yaw,yaw,delta=.045)

    def test_pattern_sweep(self):
        for frame in (0,1,19):
            faces=front_faces(ray_scan('standard',.07,pattern_frame=frame))
            self.assertTrue(faces)
            expected=(.665-.4*math.sin(.07),-2.32+.4*math.cos(.07))
            np.testing.assert_allclose((faces[0].x,faces[0].y),expected,atol=.045)
        m=ET.fromstring(descriptions(PATHS,'standard','pattern')[1])
        self.assertEqual(m.findtext(".//plugin[@name='mid360_five_heads']/points_per_frame"),'20000')
        self.assertEqual(len(m.findall(".//sensor[@type='ray']")),0)
        self.assertEqual(len([e for e in m.findall('model/link') if e.get('name').startswith('mid360_l')]),10)

    def test_entry_plan_in_any_world_heading(self):
        from four_ws_factory_sim.pick_geometry import PlanarPose, make_plan
        from four_ws_factory_sim.dimensions import REACH
        for base in (PlanarPose(0,0,0),PlanarPose(1.3,-2.2,1.26)):
            obs=PlanarPose(.665,-1.92,.08)
            plan=make_plan(base,obs)
            centre=base.offset(obs.x,obs.y)
            target=PlanarPose(centre.x,centre.y,base.yaw+obs.yaw)
            for vehicle,depth in [(plan.approach,.18),(plan.inserted,-INSERTION)]:
                tip=vehicle.offset(TOOL_X+plan.shift,TIP_Y-REACH)
                np.testing.assert_allclose(target.local(tip.x,tip.y),(0,depth),atol=1e-8)
            np.testing.assert_allclose(plan.inserted.local(plan.drop.x,plan.drop.y),(.62,0),atol=1e-8)
            np.testing.assert_allclose(plan.drop.local(plan.clear.x,plan.clear.y),(0,INSERTION+.18),atol=1e-8)

    def test_missing_centre_rejected(self):
        self.assertFalse(front_faces(ray_scan('standard',hide_block=True)))


if __name__=='__main__':
    unittest.main(verbosity=2)
