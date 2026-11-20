"""Ground-plane poses for side entry, relative to the detected near face."""
from dataclasses import dataclass
import math

from .dimensions import TOOL_X, REACH, TIP_Y, INSERTION, SHIFT_LIMIT


def wrap(a):
    return math.atan2(math.sin(a),math.cos(a))


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float

    def offset(self, dx, dy):
        c,s=math.cos(self.yaw),math.sin(self.yaw)
        return PlanarPose(self.x+c*dx-s*dy,self.y+s*dx+c*dy,self.yaw)

    def local(self, x, y):
        dx,dy=x-self.x,y-self.y
        c,s=math.cos(self.yaw),math.sin(self.yaw)
        return c*dx+s*dy,-s*dx+c*dy


@dataclass(frozen=True)
class PickPlan:
    shift: float
    approach: PlanarPose
    inserted: PlanarPose
    drop: PlanarPose
    clear: PlanarPose


def make_plan(base, face_in_base):
    observed=base.offset(face_in_base.x,face_in_base.y)
    face=PlanarPose(observed.x,observed.y,wrap(base.yaw+face_in_base.yaw))
    ux=PlanarPose(base.x,base.y,face.yaw).local(face.x,face.y)[0]
    shift=max(-SHIFT_LIMIT+.01,min(SHIFT_LIMIT-.01,ux-TOOL_X))
    gap=.18
    approach=face.offset(-(TOOL_X+shift),REACH-TIP_Y+gap)
    inserted=approach.offset(0,-(gap+INSERTION))
    drop=inserted.offset(.62,0)
    return PickPlan(shift,approach,inserted,drop,drop.offset(0,gap+INSERTION))
