# pip install pygame PyOpenGL PyOpenGL_accelerate pygltflib numpy
# python -m pip install pygame PyOpenGL PyOpenGL_accelerate pygltflib numpy


import math
import random

from codes.loader import *
from codes.constants import *



class WorldObject:
    def __init__(self, x, y, z):
        self.pos = np.array([x, y, z], dtype=np.float32)
        self.active = True

    def draw(self, camera_z):
        pass

    def on_collision(self, car):
        pass

    def is_harmful(self):
        return False

class ObstacleObject(WorldObject):
    def __init__(self, x, y, z, model):
        super().__init__(x, y, z)
        self.model = model
        self.radius = model.bounding_radius if model else 0.6
        self.type = OBJ_OBSTACLE

    def draw(self, camera_z):
        if not self.active:
            return

        dist = abs(self.pos[2] - camera_z)
        use_proxy = dist > LOD_PROXY_DISTANCE
        self.model.draw_at(
            self.pos[0], self.pos[1], self.pos[2],
            use_proxy=use_proxy
        )

    def on_collision(self, car):
        if not self.active:
            return
        self.active = False
        car.on_collision_obstacle(self)

    def is_harmful(self):
        return True

global item_models
item_models = {}
class ItemObject(WorldObject):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.active = True
        self.type = None

    def on_collision(self, car):
        self.active = False

    def draw(self, camera_z):
        if not self.active:
            return
        
        model = item_models.get(self.type)

        if not model:
            return

        dist = abs(self.pos[2] - camera_z)

        if self.type == OBJ_BULLET:
            glColor3f(0.2, 0.4, 1.0)
        elif self.type == OBJ_BOOSTER:
            glColor3f(1.0, 0.9, 0.2)
        else:
            glColor3f(1.0, 1.0, 1.0)

        if model.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, model.texture)

        model.draw_at(
            self.pos[0],
            self.pos[1],
            self.pos[2],
            use_proxy=False
        )

        if model.texture:
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)

class BoosterItem(ItemObject):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.type = OBJ_BOOSTER

    def on_collision(self, car):
        super().on_collision(car)
        car.speed = min(car.speed + 25.0, MAX_SPEED)

class AmmoItem(ItemObject):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.type = OBJ_BULLET

    def on_collision(self, car):
        self.active = False
        if not hasattr(car, "ammo"):
            car.ammo = 0

        car.ammo = min(car.ammo + 1, 3)

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
        self.type = OBJ_OBSTACLE

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
        objects = []
        lanes = [-1, 0, 1]
        random.shuffle(lanes)

        for lane in random.choices(lanes, k=random.randint(0, 2), weights=[0.4, 0.2, 0.2]):
            x = lane * self.lane_width

            if random.random() < 0.8:
                otype = random.choice(list(obstacle_models.keys()))
                model = obstacle_models.get(otype)
                if model:
                    objects.append(ObstacleObject(x, 0.25, z + self.seg_len * 0.5, model))

            else:
                if random.random() < 0.5:
                    objects.append(
                        BoosterItem(x, 0.25, z + self.seg_len * 0.5)
                    )

                else:
                    objects.append(
                        AmmoItem(x, 0.25, z + self.seg_len * 0.5)
                    )

        self.segments.append({
            "z": z,
            "objects": objects
        })

    def update(self, car_pos):
        while self.segments and \
              self.segments[0]["z"] + self.seg_len < car_pos[2] - TRACK_VISIBLE_BEHIND:
            self.segments.pop(0)

        while self.segments and \
              self.segments[-1]["z"] < car_pos[2] + self.visible_len:
            last_z = self.segments[-1]["z"]
            self.add_segment(last_z + self.seg_len)

    def draw(self, camera_z):
        glDisable(GL_CULL_FACE)

        ground_y = -100
        ground_width = self.lane_width * 60.0
        extra_range = 2000

        gz0 = camera_z - TRACK_VISIBLE_BEHIND - extra_range
        gz1 = camera_z + TRACK_VISIBLE_AHEAD  + extra_range

        r, g, b = 171/255.0, 148/255.0, 122/255.0
        glColor3f(r, g, b)

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
        for seg in self.segments:
            for obj in seg["objects"]:
                if not obj.active:
                    continue

                glColor3f(0.85, 0.15, 0.15)
                obj.draw(camera_z)

        for otype, items in groups.items():
            model = obstacle_models.get(otype)
            if model and model.texture:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, model.texture)
                glColor3f(1.0, 1.0, 0.0)
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

    # def draw(self, camera_z):
    #     for seg in self.segments:
    #         for obj in seg["objects"]:
    #             obj.draw(camera_z)


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
        self.ammo = 0
        self.shoot_cooldown = 0.0

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
                if e.key == K_a:
                    self.target_lane = min(1, self.target_lane + 1)

                elif e.key == K_d:
                    self.target_lane = max(-1, self.target_lane - 1)

    def check_collision(self):
        if self.invuln > 0.0:
            return None

        for seg in self.track.segments:
            for obj in seg["objects"]:
                if not obj.active:
                    continue

                dz = abs(self.pos[2] - obj.pos[2])
                dx = abs(self.pos[0] - obj.pos[0])

                if dz < 0.6 and dx < 0.6:
                    return obj

        return None

    def on_collision(self, obj):
        global SCORE

        ox, oy, oz = obj.pos
        obj.active = False
        self.pushback_vel = -12.0
        self.pushback_timer = 0.4

        dx = self.pos[0] - ox
        if dx == 0:
            dx = random.choice([-1.0, 1.0])

        self.pos[0] += math.copysign(1.2, dx)
        self.speed = max(2.0, self.speed * 0.3)

        self.invuln = 1.0
        self.collision_frames = 0

        SCORE -= 10

    def handle_collision(self, obj):
        if obj.type == OBJ_OBSTACLE:
            self.on_collision_obstacle(obj)

        elif obj.type == OBJ_BOOSTER:
            self.on_collision_booster(obj)

        elif obj.type == OBJ_BULLET:
            self.on_collision_bullet(obj)

    def update(self, dt, keys, events):
        self.handle_input(events)

        if self.invuln > 0.0:
            self.invuln -= dt

        self.shoot_cooldown = max(0, self.shoot_cooldown - dt)

        for e in events:
            if e.type == MOUSEBUTTONDOWN and e.button == 1:
                self.try_shoot()

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
            obj = self.check_collision()
            if obj:
                obj.on_collision(self)

    def try_shoot(self):
        if self.ammo <= 0:
            return
        
        if self.shoot_cooldown > 0.0:
            return

        self.ammo -= 1
        self.shoot_cooldown = 0.25

        hit = self.find_front_obstacle()
        if hit:
            hit.active = False
            
            global DESTROY_SCORE
            DESTROY_SCORE += 50

    def find_front_obstacle(self):
        max_dist = 50.0
        lane_half = self.track.lane_width * 0.45

        for seg in self.track.segments:
            for obj in seg["objects"]:
                if not obj.active:
                    continue

                dz = obj.pos[2] - self.pos[2]
                if dz < 0.0 or dz > max_dist:
                    continue

                dx = abs(obj.pos[0] - self.pos[0])
                if dx <= lane_half:
                    return obj

        return None

    def on_collision_obstacle(self, obj):
        global SCORE

        obj.active = False

        self.pushback_vel = -12.0
        self.pushback_timer = 0.4

        dx = self.pos[0] - obj.pos[0]
        if dx == 0:
            dx = random.choice([-1.0, 1.0])
        self.pos[0] += math.copysign(1.0, dx)

        self.speed = max(2.0, self.speed * 0.3)

        self.invuln = 1.0
        SCORE -= 10

    def on_collision_booster(self, obj):
        global SCORE

        obj.active = False

        self.speed = min(MAX_SPEED * 1.3, self.speed + 15.0)
        self.pos[2] += 1.2
        self.invuln = 0.2

        SCORE += 5

    def on_collision_bullet(self, obj):
        # global SCORE

        # obj.active = False

        # self.speed *= 0.6

        # self.pushback_vel = -4.0
        # self.pushback_timer = 0.15

        # self.invuln = 0.6
        # SCORE -= 5
        return

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

def draw_background_panels(layers, car_pos):
    glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_CURRENT_BIT)
    glDisable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glDepthMask(GL_FALSE)

    base_lane_half_width = 5.0
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

def draw_bullet_ui(car):
    if not bullet_ui_tex or car.ammo <= 0:
        return

    icon_size = 32
    padding = 8
    total_w = car.ammo * (icon_size + padding)
    start_x = WINDOW_SIZE[0] - total_w - 20
    y = 20

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_SIZE[0], WINDOW_SIZE[1], 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glColor3f(1, 1, 1)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    for i in range(car.ammo):
        draw_image_2d(
            bullet_ui_tex,
            start_x + i * (icon_size + padding),
            y,
            icon_size,
            icon_size
        )
    glDisable(GL_BLEND)

    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_sky_box_front_top(car_pos):
    sky_size = 200.0
    sky_height = 120.0

    cx, cy, cz = car_pos

    r, g, b = 144/255.0, 211/255.0, 1.0

    glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_CURRENT_BIT)

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)
    glDepthMask(GL_FALSE)
    glDisable(GL_CULL_FACE)

    glColor3f(r, g, b)

    glPushMatrix()
    glTranslatef(cx, 0.0, cz)

    glBegin(GL_QUADS)

    glVertex3f(-sky_size, 0.0,  sky_size)
    glVertex3f( sky_size, 0.0,  sky_size)
    glVertex3f( sky_size, sky_height, sky_size)
    glVertex3f(-sky_size, sky_height, sky_size)

    glVertex3f(-sky_size, sky_height, -sky_size)
    glVertex3f( sky_size, sky_height, -sky_size)
    glVertex3f( sky_size, sky_height,  sky_size)
    glVertex3f(-sky_size, sky_height,  sky_size)

    glEnd()

    glPopMatrix()

    glDepthMask(GL_TRUE)
    glEnable(GL_CULL_FACE)

    glPopAttrib()

def countdown(car, track, bg_layers):
    countdown_max_size = 120
    countdown_min_size = 40
    countdown_clock = pygame.time.Clock()
    countdown_duration = 1000

    for count in range(5, 0, -1):
        start_time = pygame.time.get_ticks()

        while True:
            now = pygame.time.get_ticks()
            elapsed = now - start_time
            if elapsed >= countdown_duration:
                break

            countdown_clock.tick(60)

            for ev in pygame.event.get():
                if ev.type == QUIT:
                    pygame.quit()
                    return
                if ev.type == KEYDOWN and ev.key == K_ESCAPE:
                    pygame.quit()
                    return

            t = elapsed / countdown_duration

            font_size = int(
                countdown_max_size * (1.0 - t) +
                countdown_min_size * t
            )

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            cam_distance = 8.0
            cam_height = 4.0
            cam_pos = car.pos + np.array([0.0, cam_height, -cam_distance], dtype=np.float32)
            target_pos = car.pos + np.array([0.0, 0.5, 3.0], dtype=np.float32)

            gluLookAt(
                cam_pos[0], cam_pos[1], cam_pos[2],
                target_pos[0], target_pos[1], target_pos[2],
                0.0, 1.0, 0.0
            )

            draw_sky_box_front_top(car.pos)
            draw_background_panels(bg_layers, car.pos)
            
            track.draw(car.pos[2])
            car.draw()

            text = str(count)
            font = pygame.font.SysFont("Arial", font_size, bold=True)
            surf = font.render(text, True, (255, 255, 255))
            w, h = surf.get_size()

            draw_text_gl(
                text,
                (WINDOW_SIZE[0] - w) // 2,
                (WINDOW_SIZE[1] - h) // 2,
                size=font_size,
                color=(255, 255, 255)
            )

            pygame.display.flip()


def draw_image_2d(tex, x, y, w, h):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x,     y)
    glTexCoord2f(1, 1); glVertex2f(x + w, y)
    glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
    glTexCoord2f(0, 0); glVertex2f(x,     y + h)
    glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)


def title_scene():
    clock = pygame.time.Clock()
    pygame.display.set_caption(PROJECT_NAME)

    def load_ui_texture(path):
        surf = pygame.image.load(path).convert_alpha()
        surf = pygame.transform.flip(surf, False, True)
        return load_texture_from_surface(surf)

    bg_tex = load_ui_texture(TITLE_BG_PATH)

    start_on  = load_ui_texture(UI_IMAGES['start_button_on'])
    start_off = load_ui_texture(UI_IMAGES['start_button_off'])
    quit_on   = load_ui_texture(UI_IMAGES['quit_button_on'])
    quit_off  = load_ui_texture(UI_IMAGES['quit_button_off'])

    btn_w, btn_h = 200, 70

    start_rect = pygame.Rect(
        (WINDOW_SIZE[0] - btn_w) // 2,
        WINDOW_SIZE[1] // 2 - 50,
        btn_w, btn_h
    )

    quit_rect = pygame.Rect(
        (WINDOW_SIZE[0] - btn_w) // 2,
        WINDOW_SIZE[1] // 2 + 40,
        btn_w, btn_h
    )

    while True:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = False

        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit()
                return False
            if ev.type == KEYDOWN and ev.key == K_ESCAPE:
                pygame.quit()
                return False
            if ev.type == MOUSEBUTTONDOWN and ev.button == 1:
                mouse_click = True

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, WINDOW_SIZE[0], WINDOW_SIZE[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

        draw_image_2d(bg_tex, 0, 0, WINDOW_SIZE[0], WINDOW_SIZE[1])

        if start_rect.collidepoint(mouse_pos):
            draw_image_2d(start_off, *start_rect)
            if mouse_click:
                glEnable(GL_DEPTH_TEST)
                return True
            
        else:
            draw_image_2d(start_on, *start_rect)

        if quit_rect.collidepoint(mouse_pos):
            draw_image_2d(quit_off, *quit_rect)
            if mouse_click:
                pygame.quit()
                return False
            
        else:
            draw_image_2d(quit_on, *quit_rect)

        pygame.display.flip()


def main_scene():
    global SCORE, DESTROY_SCORE

    pygame.init()
    pygame.font.init()
    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(PROJECT_NAME)

    global item_models
    # print(ITEM_MODELS)
    item_models = {}
    item_models = {
        OBJ_BOOSTER: ObstacleModel(ITEM_MODELS['booster']),
        OBJ_BULLET:  ObstacleModel(ITEM_MODELS['bullet']),
    }

    global bullet_ui_tex
    bullet_ui_tex = None

    if os.path.exists(ITEM_IMAGES['bullet']):
        surf = pygame.image.load(ITEM_IMAGES['bullet']).convert_alpha()
        surf = pygame.transform.flip(surf, False, True)
        bullet_ui_tex = load_texture_from_surface(surf)

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
    for name, path in OBSTACLE_MODELS.items():
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

    countdown(car, track, bg_layers)


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
        
        draw_sky_box_front_top(car.pos)
        draw_background_panels(bg_layers, car.pos)

        track.draw(car.pos[2])
        car.draw()

        draw_bullet_ui(car)
        draw_text_gl(f"Speed: {car.speed:.1f}", 20, 24, size=24, color=(255, 255, 255))
        draw_text_gl(f"Score: {SCORE}", 20, 56, size=24, color=(255, 230, 0))
        draw_text_gl(f"+ {DESTROY_SCORE}", 100 + len(str(SCORE+1)) * 20, 56, size=24, color=(255, 140, 0))

        pygame.display.flip()


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)

    if title_scene():
        main_scene()