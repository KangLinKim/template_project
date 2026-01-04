import os
import io
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from pygltflib import GLTF2

WINDOW_SIZE = (1280, 720)


def draw_text_gl(text, x, y, size=20, color=(255, 255, 255)):
    font = pygame.font.SysFont("Arial", size, bold=True)
    surf = font.render(text, True, color)
    w, h = surf.get_width(), surf.get_height()
    raw = pygame.image.tostring(surf, "RGBA", True)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_SIZE[0], WINDOW_SIZE[1], 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x,     y)
    glTexCoord2f(1, 1); glVertex2f(x + w, y)
    glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
    glTexCoord2f(0, 0); glVertex2f(x,     y + h)
    glEnd()

    glDeleteTextures([tex])
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def read_accessor_array(gltf, accessor_index, blob):
    if accessor_index is None:
        return None
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    comp_map = {
        5120: np.int8, 5121: np.uint8, 5122: np.int16,
        5123: np.uint16, 5125: np.uint32, 5126: np.float32
    }

    dtype = comp_map[accessor.componentType]
    count_map = {"SCALAR":1, "VEC2":2, "VEC3":3, "VEC4":4}
    comps = count_map[accessor.type]
    comp_size = np.dtype(dtype).itemsize
    elem_size = comp_size * comps
    stride = getattr(view, "byteStride", None) or elem_size
    base = (getattr(view, "byteOffset", 0) or 0) + (getattr(accessor, "byteOffset", 0) or 0)
    arr = np.zeros((accessor.count, comps), dtype=dtype)
    for i in range(accessor.count):
        start = base + i * stride
        chunk = blob[start:start + elem_size]
        if len(chunk) < elem_size:
            chunk = chunk + b'\x00' * (elem_size - len(chunk))

        arr[i, :] = np.frombuffer(chunk, dtype=dtype, count=comps)

    return arr

def read_indices(gltf, accessor_index, blob):
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    ct = accessor.componentType
    if ct == 5121:
        dtype = np.uint8; size = 1

    elif ct == 5123:
        dtype = np.uint16; size = 2

    elif ct == 5125:
        dtype = np.uint32; size = 4

    else:
        raise RuntimeError("Unsupported index type")
    
    start = (getattr(view, "byteOffset", 0) or 0) + (getattr(accessor, "byteOffset", 0) or 0)
    buf = blob[start:start + accessor.count * size]

    return np.frombuffer(buf, dtype=dtype).copy()

def load_texture_from_surface(surface):
    surface = pygame.transform.flip(surface, False, True)
    w, h = surface.get_size()
    raw = pygame.image.tostring(surface, "RGBA", True)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
    glBindTexture(GL_TEXTURE_2D, 0)

    return tex

def load_texture_from_gltf(gltf, image_index, blob, folder):
    if image_index is None:
        return None
    img = gltf.images[image_index]
    uri = getattr(img, "uri", None)
    if uri:
        path = os.path.join(folder, os.path.basename(uri))
        if os.path.exists(path):
            surf = pygame.image.load(path).convert_alpha()
            return load_texture_from_surface(surf)
        
    if getattr(img, "bufferView", None) is not None:
        bv = gltf.bufferViews[img.bufferView]
        start = getattr(bv, "byteOffset", 0)
        length = getattr(bv, "byteLength", 0)
        data = blob[start:start + length]
        try:
            surf = pygame.image.load(io.BytesIO(data)).convert_alpha()

        except Exception:
            tmp = f"__tmp_img_{image_index}.png"
            with open(tmp, "wb") as f:
                f.write(data)

            surf = pygame.image.load(tmp).convert_alpha()
            os.remove(tmp)

        return load_texture_from_surface(surf)
    
    return None

def load_glb_model(path):
    gltf = GLTF2().load(path)
    folder = os.path.dirname(path)
    blob = gltf.binary_blob()

    all_v = []
    all_uv = []
    all_idx = []
    offset = 0
    texid = None

    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            pos = read_accessor_array(gltf, prim.attributes.POSITION, blob)
            uv = read_accessor_array(gltf, getattr(prim.attributes, "TEXCOORD_0", None), blob)
            if uv is None:
                uv = np.zeros((pos.shape[0], 2), dtype=np.float32)

            idx_local = read_indices(gltf, prim.indices, blob)
            idx_global = idx_local + offset
            offset += pos.shape[0]
            all_v.append(pos.astype(np.float32))
            all_uv.append(uv.astype(np.float32))
            all_idx.append(idx_global)
            
            if texid is None and prim.material is not None:
                mat = gltf.materials[prim.material]
                pbr = getattr(mat, "pbrMetallicRoughness", None)
                if pbr and getattr(pbr, "baseColorTexture", None):
                    tex_idx = pbr.baseColorTexture.index
                    if tex_idx < len(gltf.textures):
                        src = gltf.textures[tex_idx].source
                        if src is not None:
                            texid = load_texture_from_gltf(gltf, src, blob, folder)

    V = np.concatenate(all_v, axis=0)
    UV = np.concatenate(all_uv, axis=0)
    IDX = np.concatenate(all_idx, axis=0)

    mn = V.min(axis=0)
    mx = V.max(axis=0)
    center = (mn + mx) * 0.5
    V = V - center
    size = mx - mn
    maxdim = float(np.max(size)) if V.size else 1.0

    return V, UV, IDX, texid, maxdim

def load_background_layers(file_list):
    layers = []
    n = len(file_list)
    for i, path in enumerate(file_list):
        if not os.path.exists(path):
            print(f"[warn] background file not found: {path}")
            texid = None
            tw, th = 1, 1
            
        else:
            surf = pygame.image.load(path).convert_alpha()
            surf = pygame.transform.rotate(surf, -90)
            tw, th = surf.get_size()
            texid = load_texture_from_surface(surf)
            
        base_speed = 0.012
        speed_multiplier = base_speed * (1.0 + (i / max(1, n - 1)) * 3.0)
        
        layers.append({
            "path": path,
            "tex": texid,
            "width": float(tw),
            "height": float(th),
            "speed": float(speed_multiplier),
            "index": i
        })
        
    return layers