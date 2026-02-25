import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import pygame
import numpy as np
import random
import math
import sys
import os

# ============================================================
# CONFIGURATION - TUNE THESE IF EXPRESSIONS DON'T TRIGGER
# ============================================================
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
FPS = 60
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Expression sensitivity thresholds
SMILE_THRESHOLD       = 0.44
MOUTH_OPEN_THRESHOLD  = 0.07
BLINK_THRESHOLD       = 0.18   # Lowered - requires a hard deliberate squeeze
EYEBROW_THRESHOLD     = 0.22
HEAD_TURN_THRESHOLD   = 0.12

# ============================================================
# COLORS
# ============================================================
BLACK     = (0,   0,   0  )
WHITE     = (255, 255, 255)
RED       = (255, 50,  50 )
GREEN     = (50,  220, 50 )
BLUE      = (50,  150, 255)
YELLOW    = (255, 230, 50 )
CYAN      = (50,  255, 230)
ORANGE    = (255, 160, 30 )
PURPLE    = (180, 80,  255)
DARK_BLUE = (8,   8,   32 )
GREY      = (80,  80,  80 )

# ============================================================
# MEDIAPIPE LANDMARK INDICES (same for Tasks API)
# ============================================================
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]

LEFT_EYEBROW_TOP  = 336
RIGHT_EYEBROW_TOP = 107
LEFT_EYE_TOP      = 386
RIGHT_EYE_TOP     = 159

MOUTH_LEFT  = 61
MOUTH_RIGHT = 291
UPPER_LIP   = 13
LOWER_LIP   = 14

NOSE_TIP    = 1
FACE_LEFT   = 234
FACE_RIGHT  = 454

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_point(landmarks, index, w, h):
    """Get pixel coordinates of a landmark."""
    lm = landmarks[index]
    return (lm.x * w, lm.y * h)

def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [get_point(landmarks, i, w, h) for i in eye_indices]
    A = distance(pts[1], pts[5])
    B = distance(pts[2], pts[4])
    C = distance(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.3

def detect_expressions(landmarks, w, h):
    expr = {
        'smile':        False,
        'mouth_open':   False,
        'eyebrow_raise': False,
        'head_left':    False,
        'head_right':   False,
        'blink':        False,
    }

    face_left_pt  = get_point(landmarks, FACE_LEFT,  w, h)
    face_right_pt = get_point(landmarks, FACE_RIGHT, w, h)
    face_width    = distance(face_left_pt, face_right_pt)

    if face_width < 1:
        return expr, {}

    # --- SMILE ---
    mouth_left_pt  = get_point(landmarks, MOUTH_LEFT,  w, h)
    mouth_right_pt = get_point(landmarks, MOUTH_RIGHT, w, h)
    mouth_width    = distance(mouth_left_pt, mouth_right_pt)
    smile_ratio    = mouth_width / face_width
    expr['smile']  = smile_ratio > SMILE_THRESHOLD

    # --- MOUTH OPEN ---
    upper_lip_pt      = get_point(landmarks, UPPER_LIP, w, h)
    lower_lip_pt      = get_point(landmarks, LOWER_LIP, w, h)
    lip_gap           = distance(upper_lip_pt, lower_lip_pt)
    mouth_open_ratio  = lip_gap / face_width
    expr['mouth_open'] = mouth_open_ratio > MOUTH_OPEN_THRESHOLD

    # --- EYEBROW RAISE ---
    left_brow_pt  = get_point(landmarks, LEFT_EYEBROW_TOP,  w, h)
    right_brow_pt = get_point(landmarks, RIGHT_EYEBROW_TOP, w, h)
    left_eye_pt   = get_point(landmarks, LEFT_EYE_TOP,  w, h)
    right_eye_pt  = get_point(landmarks, RIGHT_EYE_TOP, w, h)

    left_brow_gap  = abs(left_brow_pt[1]  - left_eye_pt[1])  / face_width
    right_brow_gap = abs(right_brow_pt[1] - right_eye_pt[1]) / face_width
    avg_brow_gap   = (left_brow_gap + right_brow_gap) / 2
    expr['eyebrow_raise'] = avg_brow_gap > EYEBROW_THRESHOLD

    # --- HEAD TURN ---
    nose_pt       = get_point(landmarks, NOSE_TIP, w, h)
    face_center_x = (face_left_pt[0] + face_right_pt[0]) / 2
    nose_offset   = (nose_pt[0] - face_center_x) / face_width
    expr['head_left']  = nose_offset < -HEAD_TURN_THRESHOLD
    expr['head_right'] = nose_offset >  HEAD_TURN_THRESHOLD

    # --- BLINK ---
    left_ear   = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    right_ear  = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    avg_ear    = (left_ear + right_ear) / 2
    expr['blink'] = avg_ear < BLINK_THRESHOLD

    metrics = {
        'smile':  round(smile_ratio, 3),
        'mouth':  round(mouth_open_ratio, 3),
        'brow':   round(avg_brow_gap, 3),
        'ear':    round(avg_ear, 3),
        'head':   round(nose_offset, 3),
    }

    return expr, metrics

# ============================================================
# GAME OBJECT CLASSES
# ============================================================

class Star:
    def __init__(self):
        self.reset(spawn=False)

    def reset(self, spawn=True):
        self.x     = random.randint(0, SCREEN_WIDTH) if not spawn else SCREEN_WIDTH
        self.y     = random.randint(0, SCREEN_HEIGHT)
        self.speed = random.uniform(0.3, 2.5)
        self.size  = random.randint(1, 3)
        self.brightness = random.randint(100, 255)

    def update(self):
        self.x -= self.speed
        if self.x < 0:
            self.reset()

    def draw(self, screen):
        c = self.brightness
        pygame.draw.circle(screen, (c, c, c),
                           (int(self.x), int(self.y)), self.size)


class Particle:
    def __init__(self, x, y, color):
        self.x  = x
        self.y  = y
        angle   = random.uniform(0, math.pi * 2)
        speed   = random.uniform(1, 6)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life     = random.randint(15, 35)
        self.max_life = self.life
        self.color    = color
        self.size     = random.randint(2, 5)

    def update(self):
        self.x    += self.vx
        self.y    += self.vy
        self.vy   += 0.15
        self.life -= 1

    def draw(self, screen):
        alpha  = self.life / self.max_life
        r = int(self.color[0] * alpha)
        g = int(self.color[1] * alpha)
        b = int(self.color[2] * alpha)
        size = max(1, int(self.size * alpha))
        pygame.draw.circle(screen, (r, g, b),
                           (int(self.x), int(self.y)), size)

    def is_dead(self):
        return self.life <= 0


class Bullet:
    def __init__(self, x, y):
        self.x      = float(x)
        self.y      = float(y)
        self.speed  = 14
        self.width  = 22
        self.height = 6
        self.alive  = True

    def update(self):
        self.x += self.speed
        if self.x > SCREEN_WIDTH + 50:
            self.alive = False

    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW,
                         (int(self.x), int(self.y) - 2,
                          self.width, self.height),
                         border_radius=3)
        pygame.draw.circle(screen, WHITE,
                           (int(self.x + self.width), int(self.y) + 1), 4)

    def get_rect(self):
        return pygame.Rect(int(self.x), int(self.y) - 3,
                           self.width, self.height)


class Player:
    def __init__(self):
        self.x       = 160
        self.y       = SCREEN_HEIGHT // 2
        self.w       = 60
        self.h       = 36
        self.speed   = 5
        self.health  = 100
        self.score   = 0
        self.bullets = []
        self.shoot_cd  = 0
        self.invincible = 0
        self.trail   = []

    def move(self, dx, dy):
        self.x = max(self.w // 2 + 10,
                     min(SCREEN_WIDTH // 2,
                         self.x + dx * self.speed))
        self.y = max(self.h // 2 + 10,
                     min(SCREEN_HEIGHT - self.h // 2 - 10,
                         self.y + dy * self.speed))

    def shoot(self):
        if self.shoot_cd <= 0:
            self.bullets.append(
                Bullet(self.x + self.w // 2, self.y)
            )
            self.shoot_cd = 12

    def update(self):
        if self.shoot_cd  > 0: self.shoot_cd  -= 1
        if self.invincible > 0: self.invincible -= 1

        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 12:
            self.trail.pop(0)

        for b in self.bullets[:]:
            b.update()
            if not b.alive:
                self.bullets.remove(b)

    def draw(self, screen):
        for i, (tx, ty) in enumerate(self.trail):
            ratio = i / max(len(self.trail), 1)
            r = int(30  * ratio)
            g = int(100 * ratio)
            b = int(255 * ratio)
            pygame.draw.circle(screen, (r, g, b), (tx, ty),
                               max(1, int(5 * ratio)))

        if self.invincible % 4 < 2:
            body = [
                (self.x + self.w // 2,  self.y),
                (self.x - self.w // 2,  self.y - self.h // 2),
                (self.x - self.w // 4,  self.y),
                (self.x - self.w // 2,  self.y + self.h // 2),
            ]
            pygame.draw.polygon(screen, CYAN, body)
            pygame.draw.circle(screen, (150, 230, 255),
                               (int(self.x + self.w // 6), int(self.y)), 8)
            pygame.draw.circle(screen, ORANGE,
                               (int(self.x - self.w // 2), int(self.y)), 10)
            pygame.draw.circle(screen, YELLOW,
                               (int(self.x - self.w // 2), int(self.y)), 5)

        for b in self.bullets:
            b.draw(screen)

    def get_rect(self):
        return pygame.Rect(
            int(self.x - self.w // 2),
            int(self.y - self.h // 2),
            self.w, self.h
        )


class Enemy:
    COLORS = [RED, ORANGE, PURPLE, (200, 50, 200)]

    def __init__(self, level=1):
        self.x       = float(SCREEN_WIDTH + 60)
        self.y       = float(random.randint(60, SCREEN_HEIGHT - 60))
        self.w       = 52
        self.h       = 38
        self.speed   = random.uniform(2.0, 2.8 + level * 0.3)
        self.max_hp  = 1 + level // 4
        self.health  = self.max_hp
        self.points  = 10 + level * 5
        self.color   = random.choice(self.COLORS)
        self.alive   = True

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        body = [
            (self.x - self.w // 2,  self.y),
            (self.x + self.w // 2,  self.y - self.h // 2),
            (self.x + self.w // 4,  self.y),
            (self.x + self.w // 2,  self.y + self.h // 2),
        ]
        pygame.draw.polygon(screen, self.color, body)
        pygame.draw.circle(screen, (255, 180, 50),
                           (int(self.x + self.w // 2), int(self.y)), 7)

        if self.max_hp > 1:
            bar_w = 44
            filled = int(bar_w * self.health / self.max_hp)
            pygame.draw.rect(screen, RED,
                             (int(self.x - bar_w // 2),
                              int(self.y - self.h // 2 - 8), bar_w, 5))
            pygame.draw.rect(screen, GREEN,
                             (int(self.x - bar_w // 2),
                              int(self.y - self.h // 2 - 8), filled, 5))

    def get_rect(self):
        return pygame.Rect(
            int(self.x - self.w // 2),
            int(self.y - self.h // 2),
            self.w, self.h
        )

    def is_off_screen(self):
        return self.x < -80

# ============================================================
# MAIN GAME CLASS
# ============================================================

class FaceGame:

    def __init__(self):
        # --- Pygame ---
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(
            "Face Space Shooter - Controlled by YOUR EXPRESSIONS!"
        )
        self.clock = pygame.time.Clock()

        # --- Fonts ---
        self.f_xl  = pygame.font.SysFont("Arial", 52, bold=True)
        self.f_lg  = pygame.font.SysFont("Arial", 34, bold=True)
        self.f_md  = pygame.font.SysFont("Arial", 22)
        self.f_sm  = pygame.font.SysFont("Arial", 16)

        # --- MediaPipe Face Landmarker (new Tasks API) ---
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "face_landmarker.task")
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.65,
            min_face_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
        print("[INFO] MediaPipe FaceLandmarker loaded successfully (Tasks API).")

        # --- YOLOv8 (optional) ---
        try:
            from ultralytics import YOLO
            self.yolo = YOLO("yolov8n.pt")
            print("[INFO] YOLOv8 loaded successfully.")
            self.use_yolo = True
        except Exception as e:
            print(f"[INFO] YOLOv8 not loaded ({e}). Using MediaPipe only.")
            self.use_yolo = False

        # --- Camera ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        # --- Game State ---
        self.state   = "MENU"
        self.expressions = {}
        self.metrics     = {}
        self.blink_cd    = 0
        self._init_game()

    def _init_game(self):
        self.player    = Player()
        self.enemies   = []
        self.particles = []
        self.stars     = [Star() for _ in range(120)]
        self.level          = 1
        self.spawn_timer    = 0
        self.spawn_rate     = 90
        self.score_to_level = 150

    # ----------------------------------------------------------
    # CAMERA + EXPRESSION PROCESSING
    # ----------------------------------------------------------

    def process_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_timestamp_ms += 33  # ~30 fps timestamps

        result = self.face_landmarker.detect_for_video(
            mp_image, self.frame_timestamp_ms
        )

        if result.face_landmarks and len(result.face_landmarks) > 0:
            landmarks = result.face_landmarks[0]
            self.expressions, self.metrics = detect_expressions(
                landmarks, w, h
            )

            # Draw face dots on camera preview
            for lm_idx in [MOUTH_LEFT, MOUTH_RIGHT, UPPER_LIP,
                           LOWER_LIP, NOSE_TIP]:
                lm = landmarks[lm_idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 100), -1)
        else:
            self.expressions = {}

        return frame

    # ----------------------------------------------------------
    # INPUT: TRANSLATE EXPRESSIONS → ACTIONS
    # ----------------------------------------------------------

    def handle_expressions(self):
        if not self.expressions:
            return

        dx, dy = 0, 0

        if self.expressions.get('head_left'):
            dx = -1
        if self.expressions.get('head_right'):
            dx = 1
        if self.expressions.get('eyebrow_raise'):
            dy = -1
        if self.expressions.get('smile') and \
           not self.expressions.get('mouth_open'):
            dy = 1

        self.player.move(dx, dy)

        if self.expressions.get('mouth_open'):
            self.player.shoot()

        # Blink no longer pauses - use P key instead to avoid accidental pauses
        if self.blink_cd > 0:
            self.blink_cd -= 1

    # ----------------------------------------------------------
    # GAME LOGIC
    # ----------------------------------------------------------

    def spawn_enemies(self):
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.enemies.append(Enemy(self.level))
            self.spawn_timer = 0
            self.spawn_rate = max(25, 90 - self.level * 6)

    def check_collisions(self):
        for enemy in self.enemies[:]:
            for bullet in self.player.bullets[:]:
                if bullet.get_rect().colliderect(enemy.get_rect()):
                    enemy.health -= 1
                    if bullet in self.player.bullets:
                        self.player.bullets.remove(bullet)
                    if enemy.health <= 0:
                        self.player.score += enemy.points
                        self._explode(enemy.x, enemy.y, enemy.color)
                        if enemy in self.enemies:
                            self.enemies.remove(enemy)
                        new_level = 1 + self.player.score // self.score_to_level
                        if new_level > self.level:
                            self.level = new_level
                    break

            if enemy in self.enemies:
                if enemy.get_rect().colliderect(self.player.get_rect()):
                    if self.player.invincible <= 0:
                        self.player.health -= 20
                        self.player.invincible = 60
                        self._explode(enemy.x, enemy.y, ORANGE)
                        if enemy in self.enemies:
                            self.enemies.remove(enemy)

            if enemy in self.enemies and enemy.is_off_screen():
                self.enemies.remove(enemy)
                self.player.health -= 5

        if self.player.health <= 0:
            self.player.health = 0
            self.state = "GAME_OVER"

    def _explode(self, x, y, color):
        for _ in range(20):
            self.particles.append(Particle(x, y, color))

    # ----------------------------------------------------------
    # DRAWING
    # ----------------------------------------------------------

    def draw_stars(self):
        for star in self.stars:
            star.update()
            star.draw(self.screen)

    def draw_hud(self):
        pygame.draw.rect(self.screen, (80, 0, 0),   (15, 15, 200, 22), border_radius=5)
        pygame.draw.rect(self.screen, GREEN,
                         (15, 15,
                          int(200 * max(0, self.player.health) / 100), 22),
                         border_radius=5)
        pygame.draw.rect(self.screen, WHITE, (15, 15, 200, 22), 2, border_radius=5)
        hp_text = self.f_sm.render(f"HP: {self.player.health}", True, WHITE)
        self.screen.blit(hp_text, (225, 18))

        score_surf = self.f_md.render(
            f"Score: {self.player.score}", True, YELLOW
        )
        self.screen.blit(score_surf,
                         (SCREEN_WIDTH // 2 - score_surf.get_width() // 2, 12))

        level_surf = self.f_md.render(f"Lvl {self.level}", True, CYAN)
        self.screen.blit(level_surf, (SCREEN_WIDTH - 80, 12))

        indicators = [
            ("SMILE -> DOWN",        'smile',         GREEN ),
            ("BROWS -> UP",          'eyebrow_raise', CYAN  ),
            ("MOUTH -> SHOOT",       'mouth_open',    YELLOW),
            ("HEAD -> LEFT/RIGHT",   'head_left',     ORANGE),
            ("P KEY -> PAUSE",       '_pause_key',    PURPLE),
        ]

        y0 = SCREEN_HEIGHT - 160
        for i, (label, key, color) in enumerate(indicators):
            active = self.expressions.get(key, False) or (
                key == 'head_left' and
                self.expressions.get('head_right', False)
            )
            bg = color if active else (40, 40, 55)
            pygame.draw.rect(self.screen, bg,
                             (10, y0 + i * 30, 210, 24),
                             border_radius=6)
            tc = BLACK if active else (160, 160, 160)
            surf = self.f_sm.render(label, True, tc)
            self.screen.blit(surf, (16, y0 + i * 30 + 4))

        debug_info = [
            f"smile : {self.metrics.get('smile', 0):.3f}  (>{SMILE_THRESHOLD})",
            f"mouth : {self.metrics.get('mouth', 0):.3f}  (>{MOUTH_OPEN_THRESHOLD})",
            f"brow  : {self.metrics.get('brow',  0):.3f}  (>{EYEBROW_THRESHOLD})",
            f"ear   : {self.metrics.get('ear',   0):.3f}  (<{BLINK_THRESHOLD})",
            f"head  : {self.metrics.get('head',  0):.3f}  (+/-{HEAD_TURN_THRESHOLD})",
        ]
        for i, txt in enumerate(debug_info):
            surf = self.f_sm.render(txt, True, GREY)
            self.screen.blit(surf, (SCREEN_WIDTH - 200, 44 + i * 20))

    def draw_camera_preview(self, frame):
        if frame is None:
            return
        small  = cv2.resize(frame, (192, 144))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        surf   = pygame.surfarray.make_surface(np.rot90(rgb))
        px     = SCREEN_WIDTH  - 200
        py     = SCREEN_HEIGHT - 152
        self.screen.blit(surf, (px, py))
        pygame.draw.rect(self.screen, CYAN, (px, py, 192, 144), 2)
        label  = self.f_sm.render("YOUR FACE", True, CYAN)
        self.screen.blit(label, (px, py - 18))

    def draw_menu(self):
        self.screen.fill(DARK_BLUE)
        self.draw_stars()

        title = self.f_xl.render("FACE SPACE SHOOTER", True, CYAN)
        self.screen.blit(title,
                         (SCREEN_WIDTH // 2 - title.get_width() // 2, 80))

        sub = self.f_lg.render(
            "Control your ship with facial expressions!", True, YELLOW
        )
        self.screen.blit(sub,
                         (SCREEN_WIDTH // 2 - sub.get_width() // 2, 150))

        controls = [
            ("SMILE",            "Move ship DOWN"),
            ("RAISE EYEBROWS",   "Move ship UP"),
            ("OPEN MOUTH",       "SHOOT laser"),
            ("TURN HEAD LEFT",   "Move ship LEFT"),
            ("TURN HEAD RIGHT",  "Move ship RIGHT"),
            ("BLINK BOTH EYES",  "PAUSE / RESUME"),
        ]

        for i, (expr, action) in enumerate(controls):
            ex_surf  = self.f_md.render(expr,   True, ORANGE)
            act_surf = self.f_md.render(action, True, WHITE)
            x0 = SCREEN_WIDTH // 2 - 250
            y0 = 230 + i * 38
            self.screen.blit(ex_surf,  (x0,        y0))
            self.screen.blit(act_surf, (x0 + 240,  y0))

        blink_surf = self.f_lg.render(
            ">> BLINK to START <<", True, GREEN
        )
        self.screen.blit(blink_surf,
                         (SCREEN_WIDTH // 2 - blink_surf.get_width() // 2,
                          SCREEN_HEIGHT - 70))

    def draw_paused(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(140)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))

        pause_surf = self.f_xl.render("PAUSED", True, YELLOW)
        self.screen.blit(pause_surf,
                         (SCREEN_WIDTH // 2 - pause_surf.get_width() // 2,
                          SCREEN_HEIGHT // 2 - 60))

        resume_surf = self.f_md.render("Press P to resume", True, WHITE)
        self.screen.blit(resume_surf,
                         (SCREEN_WIDTH // 2 - resume_surf.get_width() // 2,
                          SCREEN_HEIGHT // 2 + 20))

    def draw_game_over(self):
        self.screen.fill(DARK_BLUE)
        self.draw_stars()

        go_surf = self.f_xl.render("GAME OVER", True, RED)
        self.screen.blit(go_surf,
                         (SCREEN_WIDTH // 2 - go_surf.get_width() // 2, 160))

        sc_surf = self.f_lg.render(
            f"Final Score : {self.player.score}", True, YELLOW
        )
        self.screen.blit(sc_surf,
                         (SCREEN_WIDTH // 2 - sc_surf.get_width() // 2, 260))

        lv_surf = self.f_lg.render(
            f"Level Reached : {self.level}", True, CYAN
        )
        self.screen.blit(lv_surf,
                         (SCREEN_WIDTH // 2 - lv_surf.get_width() // 2, 320))

        tip_surf = self.f_md.render(
            "BLINK to Play Again  |  Press Q to Quit", True, WHITE
        )
        self.screen.blit(tip_surf,
                         (SCREEN_WIDTH // 2 - tip_surf.get_width() // 2, 430))

    # ----------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------

    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_SPACE and \
                       self.state == "MENU":
                        self.state = "PLAYING"
                    if event.key == pygame.K_p:
                        if self.state == "PLAYING":
                            self.state = "PAUSED"
                        elif self.state == "PAUSED":
                            self.state = "PLAYING"
                    if event.key == pygame.K_r and \
                       self.state == "GAME_OVER":
                        self._init_game()
                        self.state = "PLAYING"

            frame = self.process_camera()

            if self.state == "MENU":
                if self.expressions.get('blink') and self.blink_cd <= 0:
                    self.state = "PLAYING"
                    self.blink_cd = 90
                if self.blink_cd > 0:
                    self.blink_cd -= 1
                self.draw_menu()

            elif self.state == "PLAYING":
                self.handle_expressions()
                self.player.update()
                self.spawn_enemies()

                for e in self.enemies:
                    e.update()

                for p in self.particles[:]:
                    p.update()
                    if p.is_dead():
                        self.particles.remove(p)

                self.check_collisions()

                self.screen.fill(DARK_BLUE)
                self.draw_stars()
                self.player.draw(self.screen)
                for e in self.enemies:
                    e.draw(self.screen)
                for p in self.particles:
                    p.draw(self.screen)
                self.draw_hud()

            elif self.state == "PAUSED":
                self.handle_expressions()
                self.draw_paused()

            elif self.state == "GAME_OVER":
                if self.expressions.get('blink') and self.blink_cd <= 0:
                    self._init_game()
                    self.state = "PLAYING"
                    self.blink_cd = 90
                if self.blink_cd > 0:
                    self.blink_cd -= 1
                self.draw_game_over()

            self.draw_camera_preview(frame)

            pygame.display.flip()
            self.clock.tick(FPS)

        # --- Cleanup ---
        self.cap.release()
        self.face_landmarker.close()
        pygame.quit()
        cv2.destroyAllWindows()
        sys.exit()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  FACE SPACE SHOOTER")
    print("  Make sure your webcam is ON and your face")
    print("  is clearly visible in good lighting!")
    print("=" * 50)
    game = FaceGame()
    game.run()
