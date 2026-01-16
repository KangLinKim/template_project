import os


PROJECT_NAME = "휴몬랩코딩 7주차"

MAX_SPEED = 80.0
MAX_AMMO = 3

RENDER_DISTANCE = 60.0
LOD_PROXY_DISTANCE = 30.0
TRACK_VISIBLE_AHEAD = 80.0
TRACK_VISIBLE_BEHIND = 20.0

OBJ_OBSTACLE = "obstacle"
OBJ_BOOSTER  = "booster"
OBJ_BULLET   = "bullet"

SCORE = 0
DESTROY_SCORE = 0


CURRENT_FILE = os.path.abspath(__file__)
CURRENT_FILE = os.path.dirname(CURRENT_FILE)
CURRENT_FILE = os.path.dirname(CURRENT_FILE)

MODEL_PATH = os.path.join(CURRENT_FILE, r"files/source/police_car.glb")

TITLE_BG_PATH = os.path.join(CURRENT_FILE, r"files/images/title.jpg")

UI_IMAGES = {
    "quit_button_on"  : os.path.join(CURRENT_FILE, r'files/UI/quit_button_on.png'),
    "quit_button_off" : os.path.join(CURRENT_FILE, r'files/UI/quit_button_off.png'),
    "start_button_on" : os.path.join(CURRENT_FILE, r'files/UI/start_button_on.png'),
    "start_button_off": os.path.join(CURRENT_FILE, r'files/UI/start_button_off.png'),
}

ITEM_IMAGES = {
    "bullet": os.path.join(CURRENT_FILE, r'files/images/item_bullet.png'),
}

OBSTACLE_MODELS = {
    "cone": os.path.join(CURRENT_FILE, r'files/source/obstacle_cone.glb'),
    "broken_glass": os.path.join(CURRENT_FILE, r'files/source/obstacle_broken_glass.glb'),
    "cylinder": os.path.join(CURRENT_FILE, r'files/source/obstacle_cylinder.glb'),
}

ITEM_MODELS = {
    "booster": os.path.join(CURRENT_FILE, r"files/source/item_oil_barrel.glb"),
    "bullet": os.path.join(CURRENT_FILE, r"files/source/item_bullet.glb"),
}

BACKGROUND_LAYER_FILES = [
    os.path.join(CURRENT_FILE, r"files/background/Layer_0.png"),
    os.path.join(CURRENT_FILE, r"files/background/Layer_1.png"),
    os.path.join(CURRENT_FILE, r"files/background/Layer_2.png"),
    os.path.join(CURRENT_FILE, r"files/background/Layer_3.png"),
    os.path.join(CURRENT_FILE, r"files/background/Layer_4.png"),
]
