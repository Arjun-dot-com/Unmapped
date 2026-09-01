"""Small standards-compliant GLB and single-tile 3D Tiles writer."""
from __future__ import annotations
import json, struct
from pathlib import Path
import numpy as np

def write_glb(path, vertices, faces, colors=None):
    v = np.asarray(vertices, np.float32); f = np.asarray(faces, np.uint32).reshape(-1,3)
    c = np.asarray(colors if colors is not None else np.ones((len(v),3)), np.float32)
    blob = bytearray(); views=[]; access=[]
    def add(a, target):
        while len(blob)%4: blob.append(0)
        off=len(blob); blob.extend(np.asarray(a).tobytes()); views.append({"buffer":0,"byteOffset":off,"byteLength":len(blob)-off,"target":target}); return len(views)-1
    pv=add(v,34962); pc=add(c,34962); pi=add(f,34963)
    mn=v.min(0).tolist() if len(v) else [0,0,0]; mx=v.max(0).tolist() if len(v) else [0,0,0]
    acc=[{"bufferView":pv,"componentType":5126,"count":len(v),"type":"VEC3","min":mn,"max":mx},{"bufferView":pc,"componentType":5126,"count":len(c),"type":"VEC3"},{"bufferView":pi,"componentType":5125,"count":f.size,"type":"SCALAR"}]
    # Use an unlit material so the reconstructed surface remains readable even
    # without a Cesium globe/terrain light source.
    gltf={"asset":{"version":"2.0","generator":"Unmapped Phase 4"},"extensionsUsed":["KHR_materials_unlit"],"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":2,"mode":4,"material":0}]}],"materials":[{"name":"reconstruction-surface","extensions":{"KHR_materials_unlit":{}},"pbrMetallicRoughness":{"baseColorFactor":[0.12,0.58,0.72,1.0],"metallicFactor":0,"roughnessFactor":0.9}}],"buffers":[{"byteLength":len(blob)}],"bufferViews":views,"accessors":acc}
    j=json.dumps(gltf,separators=(",",":"),ensure_ascii=False).encode(); j += b" "*((4-len(j)%4)%4); b=bytes(blob)+b"\0"*((4-len(blob)%4)%4)
    out=b"glTF"+struct.pack("<II",2,12+8+len(j)+8+len(b))+struct.pack("<II",len(j),0x4E4F534A)+j+struct.pack("<II",len(b),0x004E4942)+b
    Path(path).write_bytes(out); return str(path)

def write_tileset(path, glb_name, origin):
    lat,lon,h=np.deg2rad(origin[0]),np.deg2rad(origin[1]),origin[2]
    a=6378137.0; e2=6.69437999014e-3; n=a/np.sqrt(1-e2*np.sin(lat)**2)
    t=[(n+h)*np.cos(lat)*np.cos(lon),(n+h)*np.cos(lat)*np.sin(lon),(n*(1-e2)+h)*np.sin(lat)]
    # Cesium tiles use column-major matrices. The model axes are ENU.
    east=[-np.sin(lon),np.cos(lon),0]; north=[-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)]; up=[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]
    transform=[east[0],east[1],east[2],0,north[0],north[1],north[2],0,up[0],up[1],up[2],0,t[0],t[1],t[2],1]
    tileset={"asset":{"version":"1.1"},"geometricError":0,"root":{"transform":transform,"boundingVolume":{"region":[np.deg2rad(origin[1])-1e-5,np.deg2rad(origin[0])-1e-5,np.deg2rad(origin[1])+1e-5,np.deg2rad(origin[0])+1e-5,origin[2]-100,origin[2]+100]},"geometricError":0,"refine":"ADD","content":{"uri":glb_name}}}
    Path(path).write_text(json.dumps(tileset, indent=2), encoding="utf-8"); return str(path)
