"""Minimal binary FBX 7.4 writer for a single vertex-coloured triangle mesh (no external SDK)."""
import struct
import zlib

import numpy as np

_HEADER = b"Kaydara FBX Binary  \x00\x1a\x00"
_FOOTER_ID = bytes([0xFA, 0xBC, 0xAB, 0x09, 0xD0, 0xC8, 0xD4, 0x66, 0xB1, 0x76, 0xFB, 0x83, 0x1C, 0xF7, 0x26, 0x7E])
_FOOTER_MAGIC = bytes([0xF8, 0x5A, 0x8C, 0x6A, 0xDE, 0xF5, 0xD9, 0x7E, 0xEC, 0xE9, 0x0C, 0xE3, 0x75, 0x8F, 0x29, 0x0B])


def _prop(v):
    if isinstance(v, bool):
        return b"C" + struct.pack("<B", int(v))
    if isinstance(v, np.integer):
        return b"L" + struct.pack("<q", int(v))
    if isinstance(v, int):
        return (b"I" + struct.pack("<i", v)) if -2**31 <= v < 2**31 else (b"L" + struct.pack("<q", v))
    if isinstance(v, float):
        return b"D" + struct.pack("<d", v)
    if isinstance(v, bytes):
        return b"S" + struct.pack("<I", len(v)) + v
    if isinstance(v, str):
        b = v.encode()
        return b"S" + struct.pack("<I", len(b)) + b
    if isinstance(v, np.ndarray):
        code = {np.dtype("float64"): b"d", np.dtype("int32"): b"i", np.dtype("float32"): b"f", np.dtype("int64"): b"l"}[v.dtype]
        raw = np.ascontiguousarray(v).tobytes()
        comp = zlib.compress(raw, 6)
        return code + struct.pack("<III", v.size, 1, len(comp)) + comp
    raise TypeError(type(v))


class Node:
    def __init__(self, name, *props, children=None):
        self.name, self.props, self.children = name, props, list(children or [])

    def add(self, *a, **k):
        n = Node(*a, **k); self.children.append(n); return n

    def encode(self, offset):
        props = b"".join(_prop(p) for p in self.props)
        name = self.name.encode()
        head_len = 13 + len(name) + len(props)
        body = b""
        if self.children:
            pos = offset + head_len
            for c in self.children:
                enc = c.encode(pos); body += enc; pos += len(enc)
            body += b"\x00" * 13
        end = offset + head_len + len(body)
        return struct.pack("<IIIB", end, len(self.props), len(props), len(name)) + name + props + body


L = np.int64


def _p70(rows):
    n = Node("Properties70")
    for r in rows:
        n.add("P", *r)
    return n


def write_fbx(path, vertices, faces, colors=None, name="VoxelFlightMesh", unit_scale_cm=100.0, up_axis_z=True):
    """vertices (N,3) metres, faces (M,3) int, colors (N,3) uint8/float[0..1]."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64).copy()
    idx = f.astype(np.int32)
    idx[:, 2] = -idx[:, 2] - 1  # last index of each polygon is bitwise-negated
    top = []
    hdr = Node("FBXHeaderExtension")
    hdr.add("FBXHeaderVersion", 1003); hdr.add("FBXVersion", 7400)
    ts = hdr.add("CreationTimeStamp")
    for k, val in [("Version", 1000), ("Year", 2026), ("Month", 10), ("Day", 3), ("Hour", 0), ("Minute", 0), ("Second", 0), ("Millisecond", 0)]:
        ts.add(k, val)
    hdr.add("Creator", "VoxelFlight FBX writer")
    top.append(hdr)
    top.append(Node("FileId", b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"))
    top.append(Node("CreationTime", "2026-10-03 00:00:00:000"))
    top.append(Node("Creator", "VoxelFlight"))
    gs = Node("GlobalSettings"); gs.add("Version", 1000)
    up = 2 if up_axis_z else 1
    gs.children.append(_p70([
        ("UpAxis", "int", "Integer", "", up), ("UpAxisSign", "int", "Integer", "", 1),
        ("FrontAxis", "int", "Integer", "", 1 if up_axis_z else 2), ("FrontAxisSign", "int", "Integer", "", -1 if up_axis_z else 1),
        ("CoordAxis", "int", "Integer", "", 0), ("CoordAxisSign", "int", "Integer", "", 1),
        ("OriginalUpAxis", "int", "Integer", "", up), ("OriginalUpAxisSign", "int", "Integer", "", 1),
        ("UnitScaleFactor", "double", "Number", "", float(unit_scale_cm)),
        ("OriginalUnitScaleFactor", "double", "Number", "", float(unit_scale_cm)),
    ]))
    top.append(gs)
    docs = Node("Documents"); docs.add("Count", 1)
    d = docs.add("Document", L(1000000), "Scene", "Scene")
    d.children.append(_p70([("SourceObject", "object", "", "")])); d.add("RootNode", L(0))
    top.append(docs)
    top.append(Node("References"))
    defs = Node("Definitions"); defs.add("Version", 100); defs.add("Count", 3)
    defs.add("ObjectType", "GlobalSettings").add("Count", 1)
    defs.add("ObjectType", "Model").add("Count", 1)
    defs.add("ObjectType", "Geometry").add("Count", 1)
    top.append(defs)
    objs = Node("Objects")
    gid, mid = L(2000000), L(3000000)
    g = objs.add("Geometry", gid, f"{name}\x00\x01Geometry", "Mesh")
    g.add("Vertices", v.ravel())
    g.add("PolygonVertexIndex", idx.ravel())
    g.add("GeometryVersion", 124)
    layer = Node("Layer", 0); layer.add("Version", 100)
    if colors is not None:
        c = np.asarray(colors, np.float64)
        if c.max() > 1.0:
            c = c / 255.0
        c = np.column_stack([c[:, :3], np.ones(len(c))])
        le = g.add("LayerElementColor", 0)
        le.add("Version", 101); le.add("Name", "Col")
        le.add("MappingInformationType", "ByVertice"); le.add("ReferenceInformationType", "Direct")
        le.add("Colors", c.ravel())
        e = layer.add("LayerElement"); e.add("Type", "LayerElementColor"); e.add("TypedIndex", 0)
    g.children.append(layer)
    mdl = objs.add("Model", mid, f"{name}\x00\x01Model", "Mesh")
    mdl.add("Version", 232)
    mdl.children.append(_p70([("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)]))
    mdl.add("Shading", True); mdl.add("Culling", "CullingOff")
    top.append(objs)
    con = Node("Connections")
    con.add("C", "OO", gid, mid)
    con.add("C", "OO", mid, L(0))
    top.append(con)
    out = bytearray(_HEADER + struct.pack("<I", 7400))
    for n in top:
        out += n.encode(len(out))
    out += b"\x00" * 13
    # footer
    out += bytes([0xFA, 0xBC, 0xAB, 0x09, 0xD0, 0xC8, 0xD4, 0x66, 0xB1, 0x76, 0xFB, 0x83, 0x1C, 0xF7, 0x26, 0x7E])
    out += b"\x00" * ((16 - len(out) % 16) % 16 or 16)
    out += b"\x00" * 4 + struct.pack("<I", 7400) + b"\x00" * 120 + _FOOTER_MAGIC
    with open(path, "wb") as fh:
        fh.write(out)
    return len(out)
