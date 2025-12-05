# pip install pygame PyOpenGL PyOpenGL_accelerate pygltflib numpy


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
MAX_SPEED = 80


def draw_text_gl(text, x, y, size = 28, color = (255,255,255)):
    font = pygame.font.SysFont("Arial", size, bold=True)
    surf = font.render(text, True, color)
    w, h = surf.get_width(), surf.get_height()
    text_raw = pygame.image.tostring(surf, "RGBA", True)

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

    texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, text_raw)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glColor3f(1, 1, 1)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x,     y    )
    glTexCoord2f(1, 1); glVertex2f(x + w, y    )
    glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
    glTexCoord2f(0, 0); glVertex2f(x,     y + h)
    glEnd()

    glDeleteTextures([texid])

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

    comp_type_map = {
        5120: np.int8,    5121: np.uint8,   5122: np.int16,
        5123: np.uint16,  5125: np.uint32,  5126: np.float32
    }

    dtype = comp_type_map[accessor.componentType]
    type_map = {"SCALAR":1, "VEC2":2, "VEC3":3, "VEC4":4}
    comps = type_map[accessor.type]

    comp_size = np.dtype(dtype).itemsize
    element_size = comp_size * comps
    stride = getattr(view, "byteStride", None) or element_size

    base = (getattr(view,"byteOffset",0) or 0) + (getattr(accessor,"byteOffset",0) or 0)
    count = accessor.count

    arr = np.zeros((count, comps), dtype=dtype)
    for i in range(count):
        start = base + i*stride
        chunk = blob[start:start+element_size]

        if len(chunk) < element_size:
            chunk = chunk + b"\x00"*(element_size-len(chunk))

        arr[i,:] = np.frombuffer(chunk, dtype=dtype, count=comps)

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

    start = (getattr(view,"byteOffset",0) or 0) + (getattr(accessor,"byteOffset",0) or 0)
    buf = blob[start:start + accessor.count*size]

    return np.frombuffer(buf, dtype=dtype).copy()

def load_texture_from_surface_path(path):
    surf = pygame.image.load(path).convert_alpha()

    return create_gl_texture_from_surface(surf)

def create_gl_texture_from_surface(surf):
    surf = pygame.transform.flip(surf, False, True)
    w,h = surf.get_size()
    raw = pygame.image.tostring(surf,"RGBA",True)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,raw)
    glBindTexture(GL_TEXTURE_2D,0)

    return tex

def load_texture_from_gltf(gltf, image_index, blob, folder):
    img = gltf.images[image_index]
    if getattr(img,"uri",None):
        u = os.path.join(folder, os.path.basename(img.uri))
        if os.path.exists(u):
            return load_texture_from_surface_path(u)

    if getattr(img,"bufferView",None) is not None:
        bv = gltf.bufferViews[img.bufferView]
        start = getattr(bv,"byteOffset",0)
        data = blob[start:start+bv.byteLength]
        try:
            surf = pygame.image.load(io.BytesIO(data)).convert_alpha()

        except:
            tmp = f"tmp_img_{image_index}.png"
            with open(tmp,"wb") as f:
                f.write(data)
            
            surf = pygame.image.load(tmp).convert_alpha()
            os.remove(tmp)

        return create_gl_texture_from_surface(surf)
    
    return None


def load_glb_model(path):
    gltf = GLTF2().load(path)
    folder = os.path.dirname(path)
    blob = gltf.binary_blob()

    V, UV, IDX = [], [], []
    offset = 0
    texid = None

    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            pos = read_accessor_array(gltf, prim.attributes.POSITION, blob)
            uv = read_accessor_array(gltf, getattr(prim.attributes,"TEXCOORD_0",None), blob)
            if uv is None:
                uv = np.zeros((pos.shape[0],2), np.float32)

            idx_local = read_indices(gltf, prim.indices, blob)
            idx_global = idx_local + offset
            offset += pos.shape[0]

            V.append(pos)
            UV.append(uv)
            IDX.append(idx_global)

            if texid is None and prim.material is not None:
                mat = gltf.materials[prim.material]
                pbr = getattr(mat,"pbrMetallicRoughness",None)
                if pbr and pbr.baseColorTexture:
                    tindex = pbr.baseColorTexture.index
                    src = gltf.textures[tindex].source
                    texid = load_texture_from_gltf(gltf, src, blob, folder)

    V = np.concatenate(V, axis=0)
    UV = np.concatenate(UV, axis=0)
    IDX = np.concatenate(IDX, axis=0)

    mn = V.min(axis=0)
    mx = V.max(axis=0)
    center = (mn+mx)/2
    V = V - center
    size = mx - mn
    maxdim = float(np.max(size))

    return V, UV, IDX, texid, maxdim


def draw_cube(size=0.5):
    half = size/2
    v = [
        [-half,-half,-half],
        [ half,-half,-half],
        [ half, half,-half],
        [-half, half,-half],
        [-half,-half, half],
        [ half,-half, half],
        [ half, half, half],
        [-half, half, half]
    ]

    edges = [
        (0,1),(1,2),(2,3),
        (3,0),(4,5),(5,6),
        (6,7),(7,4),(0,4),
        (1,5),(2,6),(3,7)
    ]
    
    glBegin(GL_LINES)

    for a,b in edges:
        glVertex3fv(v[a]); glVertex3fv(v[b])

    glEnd()


class InfiniteTrack:
    def __init__(self, lane_width=2.5, seg_len=5.0):
        self.lane_width = lane_width
        self.seg_len = seg_len
        self.visible_len = 80
        self.segments = []
        self.init()

    def init(self):
        for i in range(int(self.visible_len/self.seg_len)):
            self.add_segment(i * self.seg_len)

    def add_segment(self, z):
        # return
        obstacles = []
        lanes = [-1, 0, 1]
        random.shuffle(lanes)

        for lane in lanes:
            if len(obstacles)>=1:
                break

            if random.random() < 0.35:
                x = lane*self.lane_width
                obstacles.append([x, 0.25, z + self.seg_len/2])

        self.segments.append([z, obstacles])

        print(f"Z is {z}")
        print(f"obstacles created : {obstacles}")

    def update(self, car_pos):
        while self.segments and (self.segments[0][0] + self.seg_len) < car_pos[2]-20:
            self.segments.pop(0)

        if len(self.segments) > 0:
            while self.segments[-1][0] < car_pos[2] + self.visible_len:
                last = self.segments[-1][0]
                self.add_segment(last + self.seg_len)

    def draw(self, car_pos_z):
        lane_start = car_pos_z - 20
        lane_end   = car_pos_z + 80
        step = self.seg_len

        glColor3f(0.3, 0.9, 0.3)
        z = lane_start
        while z < lane_end:
            for lane in [-1, 0, 1]:
                x = lane * self.lane_width
                glBegin(GL_QUADS)
                glVertex3f(x - self.lane_width/2, 0, z)
                glVertex3f(x + self.lane_width/2, 0, z)
                glVertex3f(x + self.lane_width/2, 0, z + step)
                glVertex3f(x - self.lane_width/2, 0, z + step)
                glEnd()

            z += step

        glColor3f(0.8, 0.1, 0.1)
        for seg_z, obslist in self.segments:
            for o in obslist:
                glPushMatrix()
                glTranslatef(o[0], o[1], o[2])
                draw_cube(0.5)
                glPopMatrix()


class CarGLB:
    def __init__(self, track, path):
        self.track=track
        self.pos = np.array([0.0, 0.25, 0.0],dtype=np.float32)
        self.speed = 8.0
        self.lane = 0
        self.target_lane = 0

        self.pushback_vel = 0
        self.pushback_timer = 0
        self.invuln = 0
        self.collision_frames = 0

        v,uv,idx,tex,md = load_glb_model(path)
        self.vertices = v
        self.uvs = uv
        self.indices = idx
        self.texture = tex
        self.scale = 0.5/md if md > 0 else 1.0

    def handle_input(self, events):
        for e in events:
            if e.type==KEYDOWN:
                if e.key == K_d:
                    self.target_lane = max(-1, self.target_lane-1)
                
                elif e.key == K_a:
                    self.target_lane = min(1, self.target_lane+1)

    def check_collision(self):
        if not hasattr(self, "prev_z"):
            self.prev_z = self.pos[2]

        prev_z = self.prev_z
        curr_z = self.pos[2]

        z_min = min(prev_z, curr_z)
        z_max = max(prev_z, curr_z)

        for z, obs in self.track.segments:
            for o in obs:
                ox, oy, oz = o

                if z_min - 0.5 <= oz <= z_max + 0.5:
                    if abs(self.pos[0] - ox) < 0.6:
                        self.prev_z = curr_z

                        return True, o

        self.prev_z = curr_z

        return False, None

    def on_collision(self, o):
        self.pushback_vel = -6.0
        self.pushback_timer = 0.35
        dx = self.pos[0] - o[0]
        if dx == 0:
            dx = 0.01

        self.pos[0] += math.copysign(0.9, dx)
        self.speed *= 0.25
        self.invuln = 0.9
        self.collision_frames = 0

    def update(self,dt,keys,events):
        self.handle_input(events)

        if self.invuln > 0:
            self.invuln -= dt

        tx = self.target_lane*self.track.lane_width
        self.pos[0] += (tx-self.pos[0])*min(1.0,8*dt)

        if keys[K_w]:
            self.speed +=6*dt

        else:
            self.speed *= (1 - min(0.5*dt,0.5))
        
        if keys[K_s]:
            self.speed = max(0,self.speed-10*dt)
        
        self.speed = min(self.speed, MAX_SPEED)

        if self.pushback_timer > 0:
            self.pos[2] += self.pushback_vel*dt
            self.pushback_vel *= max(0, 1-3*dt)
            self.pushback_timer -= dt

        else:
            self.pos[2] += self.speed*dt

        if self.invuln <= 0 and self.pushback_timer <= 0:
            col, o = self.check_collision()
            if col:
                self.collision_frames+=1

            else:
                self.collision_frames=0

            if self.collision_frames>=2:
                self.on_collision(o)

    def draw(self):
        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef( 90, 1, 0, 0)
        glRotatef(-90, 0, 1, 0)
        glRotatef(-90, 0, 1, 0)

        glScalef(self.scale*3, self.scale*3, self.scale*3)

        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glColor3f(1,1,1)

        else:
            glDisable(GL_TEXTURE_2D)
            glColor3f(1,1,1)

        glBegin(GL_TRIANGLES)
        for i0 in self.indices:
            i = int(i0)
            if self.texture and i < len(self.uvs):
                glTexCoord2f(self.uvs[i][0], self.uvs[i][1])

            vx,vy,vz = self.vertices[i]
            glVertex3f(vx,vy,vz)

        glEnd()

        if self.texture:
            glBindTexture(GL_TEXTURE_2D,0)
            glDisable(GL_TEXTURE_2D)

        glPopMatrix()


def main():
    pygame.init()
    pygame.font.init()

    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF|OPENGL)
    pygame.display.set_caption("휴몬랩코딩 프로젝트")

    glEnable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.2, 0.4, 1.0, 1.0)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, WINDOW_SIZE[0]/WINDOW_SIZE[1], 0.01, 2000)

    track = InfiniteTrack()
    car = CarGLB(track, MODEL_PATH)

    score=0
    last_z=0

    clock = pygame.time.Clock()

    while True:
        dt=clock.tick(60)/1000
        events=pygame.event.get()
        keys=pygame.key.get_pressed()

        for e in events:
            if e.type==QUIT: pygame.quit(); return
            if e.type==KEYDOWN and e.key==K_ESCAPE:
                pygame.quit(); return

        # score update
        if car.pos[2] > last_z + 5:
            score += 1
            last_z = car.pos[2]

        car.update(dt, keys, events)
        track.update(car.pos)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cam = car.pos + np.array([0,4,-8])
        target = car.pos + np.array([0,0.5,3])
        gluLookAt(cam[0],cam[1],cam[2],
                  target[0],target[1],target[2],
                  0,1,0)

        track.draw(car.pos[2])
        car.draw()

        draw_text_gl(f"Speed: {car.speed:.1f}", 20, 25, 28, (255,255,255))
        draw_text_gl(f"Score: {score}", 20, 65, 28, (255,255,0))

        pygame.display.flip()


if __name__=="__main__":
    main()
