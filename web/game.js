// ============================================================
// FACE SPACE SHOOTER – Web Version
// HTML5 Canvas + MediaPipe FaceLandmarker JS
// ============================================================

import { FilesetResolver, FaceLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

// ============================================================
// CONFIGURATION
// ============================================================
let W = window.innerWidth, H = window.innerHeight;
const FPS = 60;

const THRESHOLDS = {
  smile: 0.44,
  mouthOpen: 0.07,
  blink: 0.18,
  eyebrow: 0.22,
  headTurn: 0.12,
};

// Landmark indices (same as Python version)
const LM = {
  LEFT_EYE: [362, 385, 387, 263, 373, 380],
  RIGHT_EYE: [33, 160, 158, 133, 153, 144],
  L_BROW: 336, R_BROW: 107,
  L_EYE_TOP: 386, R_EYE_TOP: 159,
  MOUTH_L: 61, MOUTH_R: 291,
  UPPER_LIP: 13, LOWER_LIP: 14,
  NOSE: 1, FACE_L: 234, FACE_R: 454,
};

// Colors
const COL = {
  black: '#000000',
  white: '#ffffff',
  red: '#ff3232',
  green: '#32dc32',
  blue: '#3296ff',
  yellow: '#ffe632',
  cyan: '#32ffe6',
  orange: '#ffa01e',
  purple: '#b450ff',
  darkBlue: '#080820',
  grey: '#505050',
};

// ============================================================
// UTILITIES
// ============================================================
function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
function pt(lms, i, w, h) {
  const l = lms[i];
  return [l.x * w, l.y * h];
}
function ear(lms, idx, w, h) {
  const p = idx.map(i => pt(lms, i, w, h));
  const A = dist(p[1], p[5]);
  const B = dist(p[2], p[4]);
  const C = dist(p[0], p[3]);
  return C > 0 ? (A + B) / (2 * C) : 0.3;
}
function rand(a, b) { return a + Math.random() * (b - a); }
function randInt(a, b) { return Math.floor(rand(a, b + 1)); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function rectOverlap(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x &&
    a.y < b.y + b.h && a.y + a.h > b.y;
}

// ============================================================
// EXPRESSION DETECTION
// ============================================================
function detectExpressions(lms, w, h) {
  const expr = {
    smile: false, mouthOpen: false, eyebrowRaise: false,
    headLeft: false, headRight: false, blink: false
  };
  const faceL = pt(lms, LM.FACE_L, w, h);
  const faceR = pt(lms, LM.FACE_R, w, h);
  const fw = dist(faceL, faceR);
  if (fw < 1) return [expr, {}];

  // Smile
  const mw = dist(pt(lms, LM.MOUTH_L, w, h), pt(lms, LM.MOUTH_R, w, h));
  const smileR = mw / fw;
  expr.smile = smileR > THRESHOLDS.smile;

  // Mouth open
  const lipGap = dist(pt(lms, LM.UPPER_LIP, w, h), pt(lms, LM.LOWER_LIP, w, h));
  const moR = lipGap / fw;
  expr.mouthOpen = moR > THRESHOLDS.mouthOpen;

  // Eyebrow
  const lbg = Math.abs(pt(lms, LM.L_BROW, w, h)[1] - pt(lms, LM.L_EYE_TOP, w, h)[1]) / fw;
  const rbg = Math.abs(pt(lms, LM.R_BROW, w, h)[1] - pt(lms, LM.R_EYE_TOP, w, h)[1]) / fw;
  const browR = (lbg + rbg) / 2;
  expr.eyebrowRaise = browR > THRESHOLDS.eyebrow;

  // Head turn
  const nose = pt(lms, LM.NOSE, w, h);
  const cx = (faceL[0] + faceR[0]) / 2;
  const headR = (nose[0] - cx) / fw;
  // Webcam is mirrored: swap so user's right = ship right
  expr.headLeft = headR > THRESHOLDS.headTurn;
  expr.headRight = headR < -THRESHOLDS.headTurn;

  // Blink
  const earAvg = (ear(lms, LM.LEFT_EYE, w, h) + ear(lms, LM.RIGHT_EYE, w, h)) / 2;
  expr.blink = earAvg < THRESHOLDS.blink;

  return [expr, { smile: smileR, mouth: moR, brow: browR, ear: earAvg, head: headR }];
}

// ============================================================
// GAME OBJECTS
// ============================================================

class Star {
  constructor(scattered) {
    this.reset(scattered);
  }
  reset(scattered) {
    this.x = scattered ? rand(0, W) : W + rand(0, 40);
    this.y = rand(0, H);
    this.speed = rand(0.3, 2.5);
    this.size = randInt(1, 3);
    this.brightness = randInt(100, 255);
  }
  update() {
    this.x -= this.speed;
    if (this.x < 0) this.reset(false);
  }
  draw(ctx) {
    const c = this.brightness;
    ctx.fillStyle = `rgb(${c},${c},${c})`;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
  }
}

class Particle {
  constructor(x, y, color) {
    this.x = x; this.y = y;
    const a = rand(0, Math.PI * 2), s = rand(1, 6);
    this.vx = Math.cos(a) * s;
    this.vy = Math.sin(a) * s;
    this.life = this.maxLife = randInt(15, 35);
    this.color = color;
    this.size = randInt(2, 5);
  }
  update() {
    this.x += this.vx; this.y += this.vy;
    this.vy += 0.15; this.life--;
  }
  draw(ctx) {
    const a = this.life / this.maxLife;
    ctx.globalAlpha = a;
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, Math.max(1, this.size * a), 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  get dead() { return this.life <= 0; }
}

class Bullet {
  constructor(x, y) {
    this.x = x; this.y = y;
    this.speed = 14; this.w = 22; this.h = 6;
    this.alive = true;
  }
  update() {
    this.x += this.speed;
    if (this.x > W + 50) this.alive = false;
  }
  draw(ctx) {
    ctx.fillStyle = COL.yellow;
    ctx.beginPath();
    ctx.roundRect(this.x, this.y - 2, this.w, this.h, 3);
    ctx.fill();
    ctx.fillStyle = COL.white;
    ctx.beginPath();
    ctx.arc(this.x + this.w, this.y + 1, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  get rect() { return { x: this.x, y: this.y - 3, w: this.w, h: this.h }; }
}

class Player {
  constructor() {
    this.x = 160; this.y = H / 2;
    this.w = 60; this.h = 36;
    this.speed = 2; // starts slow, increases with level
    this.health = 100;
    this.score = 0; this.bullets = [];
    this.shootCd = 0; this.invincible = 0;
    this.trail = [];
  }
  setSpeedForLevel(level) {
    // Level 1 = 2, Level 2 = 2.5, ... capped at 7
    this.speed = Math.min(7, 2 + (level - 1) * 0.5);
  }
  move(dx, dy) {
    this.x = clamp(this.x + dx * this.speed, this.w / 2 + 10, W / 2);
    this.y = clamp(this.y + dy * this.speed, this.h / 2 + 10, H - this.h / 2 - 50);
  }
  shoot() {
    if (this.shootCd <= 0) {
      this.bullets.push(new Bullet(this.x + this.w / 2, this.y));
      this.shootCd = 12;
    }
  }
  update() {
    if (this.shootCd > 0) this.shootCd--;
    if (this.invincible > 0) this.invincible--;
    this.trail.push([this.x, this.y]);
    if (this.trail.length > 12) this.trail.shift();
    this.bullets = this.bullets.filter(b => { b.update(); return b.alive; });
  }
  draw(ctx) {
    // Trail
    for (let i = 0; i < this.trail.length; i++) {
      const r = i / this.trail.length;
      ctx.fillStyle = `rgb(${30 * r | 0},${100 * r | 0},${255 * r | 0})`;
      ctx.beginPath();
      ctx.arc(this.trail[i][0], this.trail[i][1], Math.max(1, 5 * r), 0, Math.PI * 2);
      ctx.fill();
    }
    if (this.invincible % 4 < 2) {
      // Ship body
      ctx.fillStyle = COL.cyan;
      ctx.beginPath();
      ctx.moveTo(this.x + this.w / 2, this.y);
      ctx.lineTo(this.x - this.w / 2, this.y - this.h / 2);
      ctx.lineTo(this.x - this.w / 4, this.y);
      ctx.lineTo(this.x - this.w / 2, this.y + this.h / 2);
      ctx.closePath();
      ctx.fill();
      // Cockpit
      ctx.fillStyle = '#96e6ff';
      ctx.beginPath();
      ctx.arc(this.x + this.w / 6, this.y, 8, 0, Math.PI * 2);
      ctx.fill();
      // Engine glow
      ctx.fillStyle = COL.orange;
      ctx.beginPath();
      ctx.arc(this.x - this.w / 2, this.y, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = COL.yellow;
      ctx.beginPath();
      ctx.arc(this.x - this.w / 2, this.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    this.bullets.forEach(b => b.draw(ctx));
  }
  get rect() {
    return { x: this.x - this.w / 2, y: this.y - this.h / 2, w: this.w, h: this.h };
  }
}

class Enemy {
  static COLORS = [COL.red, COL.orange, COL.purple, '#c832c8'];
  constructor(level) {
    this.x = W + 60;
    this.y = rand(60, H - 100);
    this.w = 52; this.h = 38;
    this.speed = rand(2.0, 2.8 + level * 0.3);
    this.maxHp = 1 + Math.floor(level / 4);
    this.hp = this.maxHp;
    this.points = 10 + level * 5;
    this.color = Enemy.COLORS[randInt(0, Enemy.COLORS.length - 1)];
  }
  update() { this.x -= this.speed; }
  draw(ctx) {
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.moveTo(this.x - this.w / 2, this.y);
    ctx.lineTo(this.x + this.w / 2, this.y - this.h / 2);
    ctx.lineTo(this.x + this.w / 4, this.y);
    ctx.lineTo(this.x + this.w / 2, this.y + this.h / 2);
    ctx.closePath();
    ctx.fill();
    // engine
    ctx.fillStyle = '#ffb432';
    ctx.beginPath();
    ctx.arc(this.x + this.w / 2, this.y, 7, 0, Math.PI * 2);
    ctx.fill();
    // HP bar
    if (this.maxHp > 1) {
      const bw = 44, filled = bw * this.hp / this.maxHp;
      ctx.fillStyle = COL.red;
      ctx.fillRect(this.x - bw / 2, this.y - this.h / 2 - 8, bw, 5);
      ctx.fillStyle = COL.green;
      ctx.fillRect(this.x - bw / 2, this.y - this.h / 2 - 8, filled, 5);
    }
  }
  get rect() {
    return { x: this.x - this.w / 2, y: this.y - this.h / 2, w: this.w, h: this.h };
  }
  get offScreen() { return this.x < -80; }
}

// ============================================================
// OBSTACLES (Progressive difficulty)
// ============================================================

class Asteroid {
  /** Spinning rock – indestructible, must dodge */
  constructor(level) {
    this.x = W + 40;
    this.y = rand(50, H - 90);
    this.radius = rand(18, 28 + level);
    this.speed = rand(1.2, 2.0 + level * 0.15);
    this.angle = 0;
    this.spin = rand(-0.04, 0.04);
    this.vertices = [];
    const n = randInt(7, 12);
    for (let i = 0; i < n; i++) {
      this.vertices.push(rand(0.7, 1.3));
    }
  }
  update() {
    this.x -= this.speed;
    this.angle += this.spin;
  }
  draw(ctx) {
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(this.angle);
    ctx.fillStyle = '#5a5a6e';
    ctx.strokeStyle = '#8888a0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const n = this.vertices.length;
    for (let i = 0; i <= n; i++) {
      const a = (i % n) / n * Math.PI * 2;
      const r = this.radius * this.vertices[i % n];
      const px = Math.cos(a) * r, py = Math.sin(a) * r;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  get rect() {
    const r = this.radius * 0.8;
    return { x: this.x - r, y: this.y - r, w: r * 2, h: r * 2 };
  }
  get offScreen() { return this.x < -60; }
}

class SpaceMine {
  /** Blinking mine – hurts on contact, can be shot (1 hp) */
  constructor(level) {
    this.x = W + 30;
    this.y = rand(50, H - 90);
    this.radius = 14;
    this.speed = rand(0.8, 1.5 + level * 0.1);
    this.hp = 1;
    this.timer = 0;
    this.points = 20 + level * 3;
  }
  update() {
    this.x -= this.speed;
    this.timer++;
  }
  draw(ctx) {
    const blink = Math.sin(this.timer * 0.2) > 0;
    // Spikes
    ctx.strokeStyle = '#ff6666';
    ctx.lineWidth = 2;
    for (let i = 0; i < 8; i++) {
      const a = i / 8 * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(this.x + Math.cos(a) * this.radius, this.y + Math.sin(a) * this.radius);
      ctx.lineTo(this.x + Math.cos(a) * (this.radius + 8), this.y + Math.sin(a) * (this.radius + 8));
      ctx.stroke();
    }
    // Body
    ctx.fillStyle = blink ? '#cc3333' : '#881111';
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ff4444';
    ctx.stroke();
    // Center blink light
    ctx.fillStyle = blink ? '#ff0000' : '#440000';
    ctx.beginPath();
    ctx.arc(this.x, this.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  get rect() {
    return { x: this.x - this.radius, y: this.y - this.radius, w: this.radius * 2, h: this.radius * 2 };
  }
  get offScreen() { return this.x < -40; }
}

class LaserGate {
  /** Horizontal laser beam – opens/closes periodically, hurts on contact */
  constructor(level) {
    this.x = W + 10;
    this.gapY = rand(100, H - 150);
    this.gapH = clamp(180 - level * 8, 80, 180);
    this.speed = rand(1.5, 2.5);
    this.w = 8;
    this.timer = 0;
    this.active = true;
    this.cycleDuration = clamp(180 - level * 5, 60, 180);
  }
  update() {
    this.x -= this.speed;
    this.timer++;
    // Toggle on/off
    this.active = (this.timer % this.cycleDuration) < (this.cycleDuration * 0.65);
  }
  draw(ctx) {
    if (!this.active) {
      // Draw dim inactive line
      ctx.strokeStyle = 'rgba(255,50,50,0.15)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(this.x, 0);
      ctx.lineTo(this.x, this.gapY);
      ctx.moveTo(this.x, this.gapY + this.gapH);
      ctx.lineTo(this.x, H - 40);
      ctx.stroke();
      return;
    }
    // Active beam glow
    for (let i = 3; i >= 0; i--) {
      ctx.strokeStyle = `rgba(255,50,50,${0.08 + i * 0.07})`;
      ctx.lineWidth = this.w + i * 6;
      ctx.beginPath();
      ctx.moveTo(this.x, 0);
      ctx.lineTo(this.x, this.gapY);
      ctx.moveTo(this.x, this.gapY + this.gapH);
      ctx.lineTo(this.x, H - 40);
      ctx.stroke();
    }
    // Core beam
    ctx.strokeStyle = '#ff3232';
    ctx.lineWidth = this.w;
    ctx.beginPath();
    ctx.moveTo(this.x, 0);
    ctx.lineTo(this.x, this.gapY);
    ctx.moveTo(this.x, this.gapY + this.gapH);
    ctx.lineTo(this.x, H - 40);
    ctx.stroke();
  }
  collidesPlayer(pr) {
    if (!this.active) return false;
    const beamRect1 = { x: this.x - this.w / 2, y: 0, w: this.w, h: this.gapY };
    const beamRect2 = { x: this.x - this.w / 2, y: this.gapY + this.gapH, w: this.w, h: H - this.gapY - this.gapH };
    return rectOverlap(pr, beamRect1) || rectOverlap(pr, beamRect2);
  }
  get offScreen() { return this.x < -20; }
}

// ============================================================
// MAIN GAME
// ============================================================

class FaceGame {
  constructor(canvas, camPreview, video) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.camCanvas = camPreview;
    this.camCtx = camPreview.getContext('2d');
    this.video = video;

    canvas.width = W;
    canvas.height = H;

    // Handle window resize
    window.addEventListener('resize', () => {
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = W;
      canvas.height = H;
    });

    this.state = 'MENU';
    this.expr = {};
    this.metrics = {};
    this.blinkCd = 0;
    this.landmarker = null;
    this.frameTs = 0;
    this.keys = {};

    this._initGame();
    this._bindKeys();
  }

  _initGame() {
    this.player = new Player();
    this.enemies = [];
    this.asteroids = [];
    this.mines = [];
    this.laserGates = [];
    this.particles = [];
    this.stars = Array.from({ length: 120 }, () => new Star(true));
    this.level = 1;
    this.spawnTimer = 0;
    this.spawnRate = 90;
    this.obstacleTimer = 0;
    this.scoreToLevel = 150;
  }

  _bindKeys() {
    window.addEventListener('keydown', e => {
      this.keys[e.key.toLowerCase()] = true;
      if (e.key === ' ' && this.state === 'MENU') this.state = 'PLAYING';
      if (e.key.toLowerCase() === 'p') {
        if (this.state === 'PLAYING') this.state = 'PAUSED';
        else if (this.state === 'PAUSED') this.state = 'PLAYING';
      }
      if (e.key.toLowerCase() === 'r' && this.state === 'GAME_OVER') {
        this._initGame(); this.state = 'PLAYING';
      }
    });
    window.addEventListener('keyup', e => { this.keys[e.key.toLowerCase()] = false; });
  }

  // ---- MediaPipe setup ----
  async initMediaPipe() {
    const setStatus = (msg) => {
      const el = document.getElementById('loading-status');
      if (el) el.textContent = msg;
    };
    setStatus('Loading AI face model…');
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );
    this.landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.65,
      minFacePresenceConfidence: 0.65,
      minTrackingConfidence: 0.65,
    });
    setStatus('Starting camera…');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
    this.video.srcObject = stream;
    await this.video.play();

    // Hide loading, show game
    document.getElementById('loading-screen').style.display = 'none';
    document.getElementById('game-container').style.display = 'block';
  }

  // ---- Camera processing ----
  processCamera() {
    if (!this.landmarker || this.video.readyState < 2) return;
    this.frameTs += 33;
    const result = this.landmarker.detectForVideo(this.video, this.frameTs);
    if (result.faceLandmarks && result.faceLandmarks.length > 0) {
      const lms = result.faceLandmarks[0];
      [this.expr, this.metrics] = detectExpressions(lms, 640, 480);
    } else {
      this.expr = {};
    }
    // Draw camera preview
    this.camCtx.save();
    this.camCtx.scale(-1, 1);
    this.camCtx.drawImage(this.video, -160, 0, 160, 120);
    this.camCtx.restore();
  }

  // ---- Expression → Actions ----
  handleExpressions() {
    if (!this.expr || Object.keys(this.expr).length === 0) return;
    let dx = 0, dy = 0;
    if (this.expr.headLeft) dx = -1;
    if (this.expr.headRight) dx = 1;
    if (this.expr.eyebrowRaise) dy = -1;
    if (this.expr.smile && !this.expr.mouthOpen) dy = 1;
    this.player.move(dx, dy);
    if (this.expr.mouthOpen) this.player.shoot();
    if (this.blinkCd > 0) this.blinkCd--;

    // Update control bar highlights
    this._setCtrl('ctrl-smile', this.expr.smile);
    this._setCtrl('ctrl-brows', this.expr.eyebrowRaise);
    this._setCtrl('ctrl-mouth', this.expr.mouthOpen);
    this._setCtrl('ctrl-head', this.expr.headLeft || this.expr.headRight);
  }

  _setCtrl(id, active) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', !!active);
  }

  // ---- Spawning ----
  spawnEnemies() {
    this.spawnTimer++;
    if (this.spawnTimer >= this.spawnRate) {
      this.enemies.push(new Enemy(this.level));
      this.spawnTimer = 0;
      this.spawnRate = Math.max(25, 90 - this.level * 6);
    }
  }

  spawnObstacles() {
    this.obstacleTimer++;
    // Asteroids from level 2+
    if (this.level >= 2 && this.obstacleTimer % Math.max(100, 220 - this.level * 15) === 0) {
      this.asteroids.push(new Asteroid(this.level));
    }
    // Mines from level 3+
    if (this.level >= 3 && this.obstacleTimer % Math.max(120, 250 - this.level * 12) === 0) {
      this.mines.push(new SpaceMine(this.level));
    }
    // Laser gates from level 5+
    if (this.level >= 5 && this.obstacleTimer % Math.max(200, 400 - this.level * 15) === 0) {
      this.laserGates.push(new LaserGate(this.level));
    }
  }

  // ---- Collisions ----
  checkCollisions() {
    const pr = this.player.rect;

    // Bullets vs enemies
    for (let ei = this.enemies.length - 1; ei >= 0; ei--) {
      const e = this.enemies[ei];
      for (let bi = this.player.bullets.length - 1; bi >= 0; bi--) {
        const b = this.player.bullets[bi];
        if (rectOverlap(b.rect, e.rect)) {
          e.hp--;
          this.player.bullets.splice(bi, 1);
          if (e.hp <= 0) {
            this.player.score += e.points;
            this._explode(e.x, e.y, e.color);
            this.enemies.splice(ei, 1);
            const nl = 1 + Math.floor(this.player.score / this.scoreToLevel);
            if (nl > this.level) {
              this.level = nl;
              this.player.setSpeedForLevel(nl);
            }
          }
          break;
        }
      }
    }

    // Bullets vs mines
    for (let mi = this.mines.length - 1; mi >= 0; mi--) {
      const m = this.mines[mi];
      for (let bi = this.player.bullets.length - 1; bi >= 0; bi--) {
        if (rectOverlap(this.player.bullets[bi].rect, m.rect)) {
          m.hp--;
          this.player.bullets.splice(bi, 1);
          if (m.hp <= 0) {
            this.player.score += m.points;
            this._explode(m.x, m.y, COL.red);
            this.mines.splice(mi, 1);
          }
          break;
        }
      }
    }

    // Enemies vs player
    for (let ei = this.enemies.length - 1; ei >= 0; ei--) {
      const e = this.enemies[ei];
      if (rectOverlap(e.rect, pr) && this.player.invincible <= 0) {
        this.player.health -= 20;
        this.player.invincible = 60;
        this._explode(e.x, e.y, COL.orange);
        this.enemies.splice(ei, 1);
      }
    }

    // Asteroids vs player (no damage to asteroid)
    for (const a of this.asteroids) {
      if (rectOverlap(a.rect, pr) && this.player.invincible <= 0) {
        this.player.health -= 25;
        this.player.invincible = 60;
        this._explode(this.player.x, this.player.y, COL.grey);
      }
    }

    // Mines vs player
    for (let mi = this.mines.length - 1; mi >= 0; mi--) {
      const m = this.mines[mi];
      if (rectOverlap(m.rect, pr) && this.player.invincible <= 0) {
        this.player.health -= 30;
        this.player.invincible = 60;
        this._explode(m.x, m.y, COL.red);
        this.mines.splice(mi, 1);
      }
    }

    // Laser gates vs player
    for (const lg of this.laserGates) {
      if (lg.collidesPlayer(pr) && this.player.invincible <= 0) {
        this.player.health -= 15;
        this.player.invincible = 40;
        this._explode(this.player.x, this.player.y, COL.red);
      }
    }

    // Enemy escapes (no HP penalty — just remove)
    for (let ei = this.enemies.length - 1; ei >= 0; ei--) {
      if (this.enemies[ei].offScreen) {
        this.enemies.splice(ei, 1);
      }
    }

    // Clean up off-screen obstacles
    this.asteroids = this.asteroids.filter(a => !a.offScreen);
    this.mines = this.mines.filter(m => !m.offScreen);
    this.laserGates = this.laserGates.filter(lg => !lg.offScreen);

    if (this.player.health <= 0) {
      this.player.health = 0;
      this.state = 'GAME_OVER';
    }
  }

  _explode(x, y, color) {
    for (let i = 0; i < 20; i++) this.particles.push(new Particle(x, y, color));
  }

  // ---- Drawing ----
  drawStars() {
    this.stars.forEach(s => { s.update(); s.draw(this.ctx); });
  }

  drawHUD() {
    const ctx = this.ctx;
    // Health bar
    const hbW = 180;
    ctx.fillStyle = '#500000';
    ctx.beginPath(); ctx.roundRect(15, 12, hbW, 18, 5); ctx.fill();
    ctx.fillStyle = COL.green;
    ctx.beginPath(); ctx.roundRect(15, 12, hbW * Math.max(0, this.player.health) / 100, 18, 5); ctx.fill();
    ctx.strokeStyle = COL.white; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(15, 12, hbW, 18, 5); ctx.stroke();
    ctx.fillStyle = COL.white;
    ctx.font = '13px Inter, sans-serif';
    ctx.fillText(`HP: ${this.player.health}`, hbW + 22, 26);

    // Score
    ctx.font = 'bold 18px Orbitron, sans-serif';
    ctx.fillStyle = COL.yellow;
    ctx.textAlign = 'center';
    ctx.fillText(`Score: ${this.player.score}`, W / 2, 28);
    ctx.textAlign = 'left';

    // Level
    ctx.fillStyle = COL.cyan;
    ctx.font = 'bold 16px Orbitron, sans-serif';
    ctx.fillText(`Lvl ${this.level}`, W - 80, 26);

    // Debug metrics (top right, small)
    ctx.font = '11px Inter, monospace';
    ctx.fillStyle = '#555';
    const dm = [
      `smile: ${(this.metrics.smile || 0).toFixed(3)}  (>${THRESHOLDS.smile})`,
      `mouth: ${(this.metrics.mouth || 0).toFixed(3)}  (>${THRESHOLDS.mouthOpen})`,
      `brow:  ${(this.metrics.brow || 0).toFixed(3)}  (>${THRESHOLDS.eyebrow})`,
      `ear:   ${(this.metrics.ear || 0).toFixed(3)}  (<${THRESHOLDS.blink})`,
      `head:  ${(this.metrics.head || 0).toFixed(3)}  (±${THRESHOLDS.headTurn})`,
    ];
    dm.forEach((t, i) => ctx.fillText(t, W - 195, 48 + i * 16));
  }

  drawMenu() {
    const ctx = this.ctx;
    ctx.fillStyle = COL.darkBlue;
    ctx.fillRect(0, 0, W, H);
    this.drawStars();

    ctx.textAlign = 'center';
    ctx.font = 'bold 46px Orbitron, sans-serif';
    ctx.fillStyle = COL.cyan;
    ctx.fillText('FACE SPACE SHOOTER', W / 2, 100);

    ctx.font = 'bold 22px Orbitron, sans-serif';
    ctx.fillStyle = COL.yellow;
    ctx.fillText('Control your ship with facial expressions!', W / 2, 150);

    const ctrls = [
      ['😊 SMILE', 'Move ship DOWN'],
      ['🤨 RAISE EYEBROWS', 'Move ship UP'],
      ['😮 OPEN MOUTH', 'SHOOT laser'],
      ['↩ TURN HEAD LEFT', 'Move ship LEFT'],
      ['↪ TURN HEAD RIGHT', 'Move ship RIGHT'],
      ['⏸ Press P', 'PAUSE / RESUME'],
    ];
    ctx.font = '18px Inter, sans-serif';
    ctrls.forEach(([exp, act], i) => {
      const y = 210 + i * 36;
      ctx.fillStyle = COL.orange;
      ctx.textAlign = 'right';
      ctx.fillText(exp, W / 2 - 20, y);
      ctx.fillStyle = COL.white;
      ctx.textAlign = 'left';
      ctx.fillText(act, W / 2 + 20, y);
    });

    ctx.textAlign = 'center';
    ctx.font = 'bold 26px Orbitron, sans-serif';
    ctx.fillStyle = COL.green;
    const pulse = 0.7 + 0.3 * Math.sin(Date.now() / 300);
    ctx.globalAlpha = pulse;
    ctx.fillText('>> Press SPACE to START <<', W / 2, H - 60);
    ctx.globalAlpha = 1;
    ctx.textAlign = 'left';
  }

  drawPaused() {
    const ctx = this.ctx;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(0, 0, W, H);
    ctx.textAlign = 'center';
    ctx.font = 'bold 48px Orbitron, sans-serif';
    ctx.fillStyle = COL.yellow;
    ctx.fillText('PAUSED', W / 2, H / 2 - 30);
    ctx.font = '20px Inter, sans-serif';
    ctx.fillStyle = COL.white;
    ctx.fillText('Press P to resume', W / 2, H / 2 + 20);
    ctx.textAlign = 'left';
  }

  drawGameOver() {
    const ctx = this.ctx;
    ctx.fillStyle = COL.darkBlue;
    ctx.fillRect(0, 0, W, H);
    this.drawStars();

    ctx.textAlign = 'center';
    ctx.font = 'bold 48px Orbitron, sans-serif';
    ctx.fillStyle = COL.red;
    ctx.fillText('GAME OVER', W / 2, 180);

    ctx.font = 'bold 28px Orbitron, sans-serif';
    ctx.fillStyle = COL.yellow;
    ctx.fillText(`Final Score: ${this.player.score}`, W / 2, 270);

    ctx.fillStyle = COL.cyan;
    ctx.fillText(`Level Reached: ${this.level}`, W / 2, 330);

    ctx.font = '18px Inter, sans-serif';
    ctx.fillStyle = COL.white;
    ctx.fillText('Press R to Play Again  |  Press Q to Quit', W / 2, 430);
    ctx.textAlign = 'left';
  }

  // ---- Main loop ----
  tick() {
    this.processCamera();

    if (this.state === 'MENU') {
      this.drawMenu();
    } else if (this.state === 'PLAYING') {
      this.handleExpressions();
      this.player.update();
      this.spawnEnemies();
      this.spawnObstacles();
      this.enemies.forEach(e => e.update());
      this.asteroids.forEach(a => a.update());
      this.mines.forEach(m => m.update());
      this.laserGates.forEach(lg => lg.update());
      this.particles = this.particles.filter(p => { p.update(); return !p.dead; });
      this.checkCollisions();

      // Draw
      const ctx = this.ctx;
      ctx.fillStyle = COL.darkBlue;
      ctx.fillRect(0, 0, W, H);
      this.drawStars();
      this.laserGates.forEach(lg => lg.draw(ctx));
      this.asteroids.forEach(a => a.draw(ctx));
      this.mines.forEach(m => m.draw(ctx));
      this.player.draw(ctx);
      this.enemies.forEach(e => e.draw(ctx));
      this.particles.forEach(p => p.draw(ctx));
      this.drawHUD();

      // Level-up flash
    } else if (this.state === 'PAUSED') {
      this.drawPaused();
    } else if (this.state === 'GAME_OVER') {
      this.drawGameOver();
    }

    requestAnimationFrame(() => this.tick());
  }

  start() {
    this.tick();
  }
}

// ============================================================
// BOOTSTRAP
// ============================================================
(async () => {
  const canvas = document.getElementById('gameCanvas');
  const camPreview = document.getElementById('camPreview');
  const video = document.getElementById('webcam');

  camPreview.width = 160;
  camPreview.height = 120;

  const game = new FaceGame(canvas, camPreview, video);

  try {
    await game.initMediaPipe();
  } catch (err) {
    const el = document.getElementById('loading-status');
    if (el) el.textContent = `Error: ${err.message}. Make sure camera is allowed.`;
    console.error(err);
    return;
  }

  game.start();
})();
