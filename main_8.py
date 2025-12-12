import os
import io
import math
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from pygltflib import GLTF2
import random

MODEL_PATH = r"files/source/police_car.glb"
WINDOW_SIZE = (1280, 720)
MAX_SPEED = 80.0

## obstacle_dict라는 이름으로 파일 명과 
OBSTACLE_DICT = {
    "cone": r'files/source/obstacle_cone.glb',
    "broken_glass": r'files/source/obstacle_broken_glass.glb',
    "cylinder": r'files/source/obstacle_cylinder.glb',
}

RENDER_DISTANCE = 60.0
LOD_PROXY_DISTANCE = 30.0
TRACK_VISIBLE_AHEAD = 80.0
TRACK_VISIBLE_BEHIND = 20.0

SCORE = 0

BACKGROUND_LAYER_FILES = [
    r"files/background/Layer_0.png",
    r"files/background/Layer_1.png",
    r"files/background/Layer_2.png",
    r"files/background/Layer_3.png",
    r"files/background/Layer_4.png",
]


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

def draw_cube_wire(size=0.5):
    hs = size / 2.0
    v = [
        [-hs,-hs,-hs],[ hs,-hs,-hs],[ hs, hs,-hs],[-hs, hs,-hs],
        [-hs,-hs, hs],[ hs,-hs, hs],[ hs, hs, hs],[-hs, hs, hs]
    ]

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    glBegin(GL_LINES)
    for a,b in edges:
        glVertex3fv(v[a]); glVertex3fv(v[b])

    glEnd()


class ObstacleModel:
    def __init__(self, path):
        self.path = path
        self.vertices, self.uvs, self.indices, self.texture, md = load_glb_model(path)
        self.scale = (1.0 / md) * 1.6 if md > 0 else 1.0
        bbox = (self.vertices.max(axis=0) - self.vertices.min(axis=0))
        self.bounding_radius = np.linalg.norm(bbox) * 0.5 * self.scale
        self.full_list = glGenLists(1)
        glNewList(self.full_list, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for idx0 in self.indices:
            i = int(idx0)
            if self.uvs is not None and i < len(self.uvs):
                glTexCoord2f(float(self.uvs[i,0]), float(self.uvs[i,1]))

            vx, vy, vz = self.vertices[i]
            glVertex3f(float(vx), float(vy), float(vz))

        glEnd()
        glEndList()

        self.proxy_list = glGenLists(1)
        glNewList(self.proxy_list, GL_COMPILE)
        draw_cube_wire(0.6)
        glEndList()

    def draw_at(self, x, y, z, use_proxy=False):
        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(self.scale, self.scale, self.scale)
        if use_proxy:
            glCallList(self.proxy_list)
        else:
            glCallList(self.full_list)
        glPopMatrix()

class InfiniteTrack:
    def __init__(self, lane_width=2.5, seg_len=5.0):
        self.lane_width = lane_width
        self.seg_len = seg_len
        self.visible_len = TRACK_VISIBLE_AHEAD
        self.segments = []
        self.init_segments()

    def init_segments(self):
        for i in range(int(self.visible_len / self.seg_len)):
            self.add_segment(i * self.seg_len)

    def add_segment(self, z):
        obstacles = []
        lanes = [-1, 0, 1]
        random.shuffle(lanes)
        for lane in lanes:
            if len(obstacles) >= 1:
                break

            if random.random() < 0.35:
                otype = random.choice(list(OBSTACLE_DICT.keys()))
                x = lane * self.lane_width
                obstacles.append([x, 0.25, z + self.seg_len * 0.5, otype])

        self.segments.append([z, obstacles])

    def update(self, car_pos):
        while self.segments and (self.segments[0][0] + self.seg_len) < car_pos[2] - TRACK_VISIBLE_BEHIND:
            self.segments.pop(0)
            
        while self.segments and (self.segments[-1][0] < car_pos[2] + self.visible_len):
            last = self.segments[-1][0]
            self.add_segment(last + self.seg_len)
            
        if not self.segments:
            self.add_segment(car_pos[2])

    def draw(self, camera_z):
        glDisable(GL_CULL_FACE)

        ground_y = -50
        ground_width = self.lane_width * 20.0
        extra_range = 1000

        gz0 = camera_z - TRACK_VISIBLE_BEHIND - extra_range
        gz1 = camera_z + TRACK_VISIBLE_AHEAD  + extra_range

        glColor3f(0.45, 0.30, 0.15)

        glBegin(GL_QUADS)
        glVertex3f(-ground_width, ground_y, gz0)
        glVertex3f( ground_width, ground_y, gz0)
        glVertex3f( ground_width, ground_y, gz1)
        glVertex3f(-ground_width, ground_y, gz1)
        glEnd()

        lane_start = camera_z - TRACK_VISIBLE_BEHIND
        lane_end   = camera_z + self.visible_len
        step = self.seg_len
        glColor3f(0.28, 0.75, 0.28)

        z = lane_start
        while z < lane_end:
            for lane in [-1, 0, 1]:
                x = lane * self.lane_width
                glBegin(GL_QUADS)
                glVertex3f(x - self.lane_width * 0.5, 0.0, z)
                glVertex3f(x + self.lane_width * 0.5, 0.0, z)
                glVertex3f(x + self.lane_width * 0.5, 0.0, z + step)
                glVertex3f(x - self.lane_width * 0.5, 0.0, z + step)
                glEnd()
            z += step

        groups = {}
        for seg_z, obslist in self.segments:
            for o in obslist:
                ox, oy, oz, otype = o
                dist = abs(oz - camera_z)
                if dist > RENDER_DISTANCE:
                    continue

                groups.setdefault(otype, []).append((ox, oy, oz, dist))

        for otype, items in groups.items():
            model = obstacle_models.get(otype)
            if model and model.texture:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, model.texture)
                glColor3f(1.0, 1.0, 1.0)
                for ox, oy, oz, dist in items:
                    if dist > LOD_PROXY_DISTANCE:
                        model.draw_at(ox, oy, oz, use_proxy=True)

                    else:
                        model.draw_at(ox, oy, oz, use_proxy=False)

                glBindTexture(GL_TEXTURE_2D, 0)
                glDisable(GL_TEXTURE_2D)

            elif model:
                glColor3f(0.85, 0.15, 0.15)
                for ox, oy, oz, dist in items:
                    if dist > LOD_PROXY_DISTANCE:
                        model.draw_at(ox, oy, oz, use_proxy=True)

                    else:
                        model.draw_at(ox, oy, oz, use_proxy=False)
            else:
                glColor3f(0.85, 0.15, 0.15)
                for ox, oy, oz, dist in items:
                    glPushMatrix()
                    glTranslatef(ox, oy, oz)
                    draw_cube_wire(0.6)
                    glPopMatrix()


class CarGLB:
    def __init__(self, track, path):
        self.track = track
        self.pos = np.array([0.0, 0.25, 0.0], dtype=np.float32)
        self.speed = 8.0
        self.target_lane = 0
        self.pushback_vel = 0.0
        self.pushback_timer = 0.0
        self.invuln = 0.0
        self.collision_frames = 0

        v, uv, idx, tex, md = load_glb_model(path)
        self.vertices = v
        self.uvs = uv
        self.indices = idx
        self.texture = tex
        self.scale = (0.5 / md) if md > 0 else 1.0

        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for i0 in self.indices:
            i = int(i0)
            if self.uvs is not None and i < len(self.uvs):
                glTexCoord2f(float(self.uvs[i,0]), float(self.uvs[i,1]))

            vx, vy, vz = self.vertices[i]

            glVertex3f(float(vx), float(vy), float(vz))

        glEnd()
        glEndList()

    def handle_input(self, events):
        for e in events:
            if e.type == KEYDOWN:
                if e.key == K_d:
                    self.target_lane = min(1, self.target_lane + 1)

                elif e.key == K_a:
                    self.target_lane = max(-1, self.target_lane - 1)

    def check_collision(self):
        if not hasattr(self, "prev_z"):
            self.prev_z = self.pos[2]

        prev_z = self.prev_z
        curr_z = self.pos[2]
        zmin = min(prev_z, curr_z) - 0.5
        zmax = max(prev_z, curr_z) + 0.5
        for seg_z, obslist in self.track.segments:
            for o in obslist:
                ox, oy, oz, _otype = o

                if oz >= zmin and oz <= zmax:
                    if abs(self.pos[0] - ox) < 0.6:
                        self.prev_z = curr_z

                        return True, o
                    
        self.prev_z = curr_z

        return False, None

    def on_collision(self, o):
        global SCORE

        ox, oy, oz, _otype = o
        self.pushback_vel = -6.0
        self.pushback_timer = 0.35
        dx = self.pos[0] - ox
        if dx == 0:
            dx = 0.01

        self.pos[0] += math.copysign(0.9, dx)
        self.speed *= 0.25
        self.invuln = 0.9
        self.collision_frames = 0

        SCORE -= 10

    def update(self, dt, keys, events):
        self.handle_input(events)

        if self.invuln > 0.0:
            self.invuln -= dt

        tx = self.target_lane * self.track.lane_width
        self.pos[0] += (tx - self.pos[0]) * min(1.0, 8.0 * dt)

        if keys[K_w]:
            self.speed += 6.0 * dt

        else:
            self.speed *= (1.0 - min(0.5 * dt, 0.5))

        if keys[K_s]:
            self.speed = max(0.0, self.speed - 10.0 * dt)

        self.speed = min(self.speed, MAX_SPEED)

        if self.pushback_timer > 0.0:
            self.pos[2] += self.pushback_vel * dt
            self.pushback_vel *= max(0.0, 1.0 - 3.0 * dt)
            self.pushback_timer -= dt

        else:
            self.pos[2] += self.speed * dt

        if self.invuln <= 0.0 and self.pushback_timer <= 0.0:
            collided, o = self.check_collision()
            if collided:
                self.collision_frames += 1

            else:
                self.collision_frames = 0

            if self.collision_frames >= 2:
                self.on_collision(o)

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(90.0, 1.0, 0.0, 0.0)
        glRotatef(-90.0, 0.0, 1.0, 0.0)
        glRotatef(-90.0, 0.0, 1.0, 0.0)
        glScalef(self.scale * 3.0, self.scale * 3.0, self.scale * 3.0)

        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glColor3f(1.0, 1.0, 1.0)

        else:
            glDisable(GL_TEXTURE_2D)
            glColor3f(0.9, 0.9, 0.9)

        glCallList(self.display_list)

        if self.texture:
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)

        glPopMatrix()


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

def draw_background_panels(layers, car_pos):
    glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_CURRENT_BIT)
    glDisable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glDepthMask(GL_FALSE)

    base_lane_half_width = 4.0
    layer_gap = 2.5
    panel_height = 30.0
    panel_length = 40.0
    repeat_count = 6
    panel_y_offset = -8.0

    cam_z = float(car_pos[2])
    cam_x = float(car_pos[0])

    total_layers = max(1, len(layers) - 1)

    for L in layers:
        tex = L["tex"]
        if tex is None:
            continue

        base_speed = L["speed"]
        layer_index = L["index"]

        depth_factor = (layer_index / total_layers)
        speed = base_speed * (0.3 + depth_factor * 2.0)

        v_off = cam_z * speed
        u_off = 0.0

        lane_distance = base_lane_half_width + layer_index * layer_gap

        parallax_strength = 1.0 - 0.8 * (layer_index / total_layers)
        x_parallax = cam_x * parallax_strength

        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        for side in (-1, 1):
            side_x = (lane_distance * side) + (x_parallax * side)

            for i in range(-repeat_count, repeat_count + 1):
                z0 = cam_z + i * panel_length
                z1 = z0 + panel_length

                glBegin(GL_QUADS)
                glTexCoord2f(0.0 + u_off, 0.0 + v_off); glVertex3f(side_x, panel_y_offset, z0)
                glTexCoord2f(1.0 + u_off, 0.0 + v_off); glVertex3f(side_x, panel_y_offset + panel_height, z0)
                glTexCoord2f(1.0 + u_off, 1.0 + v_off); glVertex3f(side_x, panel_y_offset + panel_height, z1)
                glTexCoord2f(0.0 + u_off, 1.0 + v_off); glVertex3f(side_x, panel_y_offset, z1)
                glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)

    glDepthMask(True)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_CULL_FACE)
    glPopAttrib()


def main():
    global SCORE

    pygame.init()
    pygame.font.init()
    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Debug Viewer - Side Panels Corrected")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.2, 0.4, 1.0, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, WINDOW_SIZE[0] / WINDOW_SIZE[1], 0.01, 2000.0)
    glMatrixMode(GL_MODELVIEW)

    global obstacle_models
    obstacle_models = {}
    for name, path in OBSTACLE_DICT.items():
        try:
            obstacle_models[name] = ObstacleModel(path)
        except Exception as e:
            print(f"[warn] failed to load obstacle '{name}' from {path}: {e}")
            obstacle_models[name] = None

    bg_layers = load_background_layers(BACKGROUND_LAYER_FILES)

    track = InfiniteTrack()
    try:
        car = CarGLB(track, MODEL_PATH)

    except Exception as e:
        print("[error] failed to load car model:", e)
        return

    last_z_for_score = car.pos[2]
    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(60) / 1000.0
        events = pygame.event.get()
        keys = pygame.key.get_pressed()

        for ev in events:
            if ev.type == QUIT:
                pygame.quit(); return
            if ev.type == KEYDOWN and ev.key == K_ESCAPE:
                pygame.quit(); return

        if car.pos[2] > last_z_for_score + 5.0:
            SCORE += 1
            last_z_for_score = car.pos[2]

        car.update(dt, keys, events)
        track.update(car.pos)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        cam_distance = 8.0
        cam_height = 4.0
        cam_pos = car.pos + np.array([0.0, cam_height, -cam_distance], dtype=np.float32)
        target_pos = car.pos + np.array([0.0, 0.5, 3.0], dtype=np.float32)
        gluLookAt(cam_pos[0], cam_pos[1], cam_pos[2],
                  target_pos[0], target_pos[1], target_pos[2],
                  0.0, 1.0, 0.0)

        draw_background_panels(bg_layers, car.pos)

        track.draw(car.pos[2])
        car.draw()

        draw_text_gl(f"Speed: {car.speed:.1f}", 20, 24, size=24, color=(255, 255, 255))
        draw_text_gl(f"Score: {SCORE}", 20, 56, size=24, color=(255, 230, 0))

        pygame.display.flip()


if __name__ == "__main__":
    main()