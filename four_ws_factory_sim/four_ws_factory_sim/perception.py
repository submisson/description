"""Fit the three block faces of a floor-level EPAL 1 long side, in base_link."""
from dataclasses import dataclass
import math
import numpy as np

from .dimensions import TOOL_X

BLOCK_PITCH = .5275


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    yaw: float                 # local +x follows the long edge, +y faces the vehicle
    score: float


def front_faces(points, max_range=4.3):
    """Return independent detections, ordered by proximity to the fork working axis."""
    xyz = np.asarray(points)[:,:3]
    z = xyz[:,2]
    roi = ((xyz[:,1]<-.88) & (xyz[:,1]>-max_range) &
           (np.abs(xyz[:,0]-TOOL_X)<4.5) & (z>.028) & (z<.153))
    patch=xyz[roi]
    feet=patch[patch[:,2]<.096,:2]
    deck=patch[patch[:,2]>.110,:2]
    if len(feet)<10 or len(deck)<3:
        return []
    candidates=[]
    for angle in range(-14,15,2):
        a=math.radians(angle)
        c,s=math.cos(a),math.sin(a)
        # Coordinates of points in the tested face orientation.
        axes=np.array([[c,-s],[s,c]])
        uv=feet @ axes
        top=deck @ axes
        for phase in (0.,.0225):
            bins=np.floor((uv[:,1]+phase)/.045).astype(int)
            order=np.argsort(bins)
            edges=np.flatnonzero(np.diff(bins[order]))+1
            for indices in np.split(order,edges):
                if len(indices)<10:
                    continue
                row=uv[indices]
                row=row[row[:,0].argsort()]
                runs=np.split(row,np.flatnonzero(np.diff(row[:,0])>.065)+1)
                for left,mid,right in zip(runs,runs[1:],runs[2:]):
                    groups=(left,mid,right)
                    if min(map(len,groups))<2:
                        continue
                    spans=np.array([g[-1,0]-g[0,0] for g in groups])
                    if spans.min()<.032 or spans.max()>.205:
                        continue
                    centres=np.array([(g[-1,0]+g[0,0])/2 for g in groups])
                    err=np.max(np.abs(np.diff(centres)-BLOCK_PITCH))
                    if err>.054:
                        continue
                    vs=np.array([np.median(g[:,1]) for g in groups])
                    # Refine coarse angle and translation using the measured front plane.
                    slope=(vs[2]-vs[0])/(centres[2]-centres[0])
                    yaw=a+math.atan(slope)
                    if abs(yaw)>math.radians(15.) or abs(vs[1]-(vs[0]+vs[2])/2)>.023:
                        continue
                    u=float(centres.mean())
                    v=float(np.median(vs-slope*(centres-u)))
                    upper=top[(np.abs(top[:,1]-v-slope*(top[:,0]-u))<.047)&
                              (np.abs(top[:,0]-u)<.62),0]
                    if len(upper)<3 or np.ptp(upper)<.66:
                        continue
                    # The deck edge bridges at least one opening between the feet.
                    # This rejects three unrelated isolated low posts.
                    across=np.abs(upper-u)
                    if not np.any((across>.12)&(across<.42)):
                        continue
                    score=float(2*err + abs(vs[1]-(vs[0]+vs[2])/2) +
                                .25*np.mean(np.abs(spans-.135)) -
                                .00008*min(sum(map(len,groups)),100))
                    candidates.append(Face(c*u-s*v,s*u+c*v,yaw,score))
    result=[]
    for f in sorted(candidates,key=lambda p:p.score):
        if all(math.hypot(f.x-g.x,f.y-g.y)>.28 for g in result):
            result.append(f)
    return sorted(result,key=lambda p:abs(p.x-TOOL_X)+.18*abs(p.y))


def packed_merged_xyz(msg):
    """The workspace merger explicitly publishes a packed little-endian XYZI float32 layout."""
    if (msg.height!=1 or msg.is_bigendian or msg.point_step!=16 or msg.row_step!=16*msg.width or
        [(f.name,f.offset,f.datatype,f.count) for f in msg.fields] !=
        [(n,4*i,7,1) for i,n in enumerate(('x','y','z','intensity'))]):
        raise ValueError('Expected merged float32 XYZI cloud')
    return np.frombuffer(msg.data,dtype='<f4',count=msg.width*4).reshape(-1,4)[:,:3]
