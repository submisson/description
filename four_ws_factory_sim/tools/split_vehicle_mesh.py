#!/usr/bin/env python3
"""Split the supplied single chassis CAD mesh without changing its home-pose geometry."""
from copy import deepcopy
from pathlib import Path

from lxml import etree as ET
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PACKAGE=Path(__file__).resolve().parents[1]
SOURCE=PACKAGE.parent/'four_ws_vehicle_description/mesh/chassis.dae'
FORK_PARTS={168,169,171,172,187,188,241,242,254,256,257,258,259,260,264,266,268,269}


def split():
    doc=ET.parse(str(SOURCE))
    ns={'d':doc.getroot().nsmap[None]}
    coords=np.fromstring(doc.find('.//d:float_array',ns).text,sep=' ').reshape(-1,3)
    unique,inverse=np.unique(coords.round(5),axis=0,return_inverse=True)
    buckets={k:{} for k in ('body','mast','forks','cabinet_glazing')}
    for material,element in enumerate(doc.xpath('//d:triangles',namespaces=ns)):
        triangles=np.fromstring(element.find('d:p',ns).text,sep=' ',dtype=int).reshape(-1,3)
        shared=inverse[triangles]
        graph=coo_matrix((np.ones(shared.size),(shared.ravel(),np.roll(shared,-1,axis=1).ravel())),
                         shape=(len(unique),len(unique)))
        _,labels=connected_components(graph,directed=False)
        parts=labels[shared[:,0]]
        for i,label in enumerate(np.unique(parts)):
            # Component 59 is a pallet baked into the supplied chassis triangle soup.
            # The empty truck is spawned without that duplicate load.
            if material==0 and i==59:
                continue
            faces=triangles[parts==label]
            vertices=coords[np.unique(faces)]
            lo,hi=vertices.min(0),vertices.max(0)
            in_mast=lo[0]>-.382 and hi[0]<.384 and lo[1]>.29 and hi[2]>.469
            name='mast' if in_mast else 'body'
            if material==0 and i in FORK_PARTS:
                name='forks'
            if material==1 and i in (1,34):
                name='cabinet_glazing'
            # Reuse the original amber finish on the two tines and working platforms.
            finish=2 if material==0 and i in (0,27,302,303,171,264) else material
            buckets[name].setdefault(finish,[]).append(faces)
    for name,groups in buckets.items():
        result=deepcopy(doc)
        ids=np.unique(np.concatenate([a.ravel() for group in groups.values() for a in group]))
        remap=np.empty(len(coords),dtype=int)
        remap[ids]=np.arange(len(ids))
        array=result.find('.//d:float_array',ns)
        array.text=' '.join('%.8g'%x for x in coords[ids].ravel())
        array.set('count',str(len(ids)*3))
        result.find('.//d:accessor',ns).set('count',str(len(ids)))
        for mat,el in enumerate(result.xpath('//d:triangles',namespaces=ns)):
            if mat in groups:
                tri=remap[np.vstack(groups[mat])]
                el.set('count',str(len(tri)))
                el.find('d:p',ns).text=' '.join(map(str,tri.ravel()))
            else:
                el.getparent().remove(el)
        result.write(str(PACKAGE/'meshes'/(name+'.dae')),encoding='UTF-8',xml_declaration=True)


if __name__=='__main__':
    split()
