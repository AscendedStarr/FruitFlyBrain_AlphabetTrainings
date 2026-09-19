/* Fly3D — a hand-rolled software 3D renderer for a fruit fly.
 *
 * There is no three.js here and no build step. The whole thing is a
 * perspective projection plus painter's-algorithm depth sorting onto a 2D
 * canvas. Wireframe would be unreadable at this size, so every body part is
 * drawn as a shaded solid: ellipsoids for the head, thorax, abdomen and the
 * two compound eyes; gradient-filled polygons with veins for the wings;
 * tapered segments for the six legs and the two antennae.
 *
 * Frame of reference is "body units": +x forward (toward the head), +y to the
 * fly's right, +z up. The fly is ~1.5 units long, i.e. about 1.5 mm of real
 * Drosophila. Proportions are taken from the animal: the abdomen is roughly
 * half the body length, the thorax is the hub the wings and legs hang off,
 * and the head is dominated by two enormous compound eyes.
 *
 * The fly is never idle. It walks on the spot with a real tripod gait
 * ({L1,R2,L3} stepping together against {R1,L2,R3}), its wings beat with a
 * slow amplitude drift and occasional flicks, the antennae sweep, the
 * abdomen breathes, and the whole animal yaws so the 3D structure is legible.
 */
(function (global) {
'use strict';

const TAU = Math.PI * 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* ======================================================================= */
/* anatomy — static rest geometry                                          */
/* ======================================================================= */

/* The wing outline, as offsets from the hinge in the wing's own plane.
 * x runs backward along the wing, y outward from the body. A real Drosophila
 * wing is about as long as the body, so this is deliberately large. */
const WING_OUTLINE = [
  [ 0.03, -0.035], [ 0.07,  0.055], [-0.07,  0.205], [-0.33,  0.330],
  [-0.67,  0.380], [-0.99,  0.330], [-0.98,  0.180], [-0.66,  0.055],
  [-0.27, -0.050],
];

/* Wing veins, as polylines in the same local plane. */
const WING_VEINS = [
  [[0.00, 0.010], [-0.30, 0.300], [-0.65, 0.360], [-0.97, 0.310]],
  [[0.00, 0.010], [-0.34, 0.250], [-0.66, 0.300], [-0.95, 0.255]],
  [[0.00, 0.000], [-0.30, 0.160], [-0.62, 0.190], [-0.90, 0.190]],
  [[0.00, -0.005], [-0.26, 0.070], [-0.55, 0.060], [-0.70, 0.050]],
  [[-0.30, 0.300], [-0.31, 0.160]],
];

/* One leg per row x two sides. Attach point is on the thorax; the knee and
 * foot are offsets from it, mirrored by side. `tripod` is the gait group on
 * the fly's right; the left side takes the opposite group. */
const LEGS = [
  { ax: 0.205, ay: 0.150, az: -0.085, kdx:  0.075, kout: 0.115, kdz: -0.235,
    fdx: -0.055, fdz: -0.285, tripod: 0, hw: 0.023 },
  { ax: 0.055, ay: 0.185, az: -0.105, kdx:  0.010, kout: 0.155, kdz: -0.260,
    fdx: -0.020, fdz: -0.300, tripod: 1, hw: 0.025 },
  { ax:-0.115, ay: 0.175, az: -0.105, kdx: -0.075, kout: 0.150, kdz: -0.250,
    fdx:  0.035, fdz: -0.315, tripod: 0, hw: 0.027 },
];

const WING_HINGE = [0.16, 0.13, 0.15];
const ANTENNA = [
  { from: [0.570,  0.075, 0.150], bend: [0.715,  0.170, 0.245], tip: [0.855,  0.250, 0.280] },
  { from: [0.570, -0.075, 0.150], bend: [0.715, -0.170, 0.245], tip: [0.855, -0.250, 0.280] },
];

/* Bristles: the stiff macrochaetae a Drosophila is covered in. Purely
 * decorative, but they sell the silhouette. */
const BRISTLES = [
  { p: [0.16, 0.10, 0.20], d: [ 0.06,  0.16, 0.14] },
  { p: [0.04, 0.13, 0.18], d: [ 0.02,  0.18, 0.13] },
  { p: [0.02, 0.05, 0.22], d: [-0.02,  0.06, 0.16] },
  { p: [0.16, -0.10, 0.20], d: [ 0.06, -0.16, 0.14] },
  { p: [0.04, -0.13, 0.18], d: [ 0.02, -0.18, 0.13] },
  { p: [0.02, -0.05, 0.22], d: [-0.02, -0.06, 0.16] },
  { p: [0.40, 0.08, 0.16], d: [ 0.05,  0.10, 0.12] },
  { p: [0.40, -0.08, 0.16], d: [ 0.05, -0.10, 0.12] },
];

const COLOURS = {
  abdomen: ['#3a2816', '#8a6640', '#d2b184'],
  thorax:  ['#3a2918', '#9c7a51', '#e0c69c'],
  head:    ['#3a2918', '#9c7a51', '#e0c69c'],
  eye:     ['#3d1010', '#8f3030', '#e79b97'],
};

/* ======================================================================= */
/* projection                                                              */
/* ======================================================================= */

/* Rotate a body-frame point into camera space and apply perspective.
 * Camera sits on -y looking toward +y; `d` grows with distance. */
function project(p, v) {
  const x =  p[0] * v.cosYaw - p[1] * v.sinYaw;
  const y =  p[0] * v.sinYaw + p[1] * v.cosYaw;
  const z =  p[2];
  const y2 = y * v.cosPitch - z * v.sinPitch;
  const z2 = y * v.sinPitch + z * v.cosPitch;
  const d  = v.camDist - y2;
  const f  = v.focal / Math.max(0.5, d) * v.scale;
  return { x: v.cx + x * f, y: v.cy - z2 * f, d: d, f: f };
}

/* How far a point sits behind the fly's centre, 0 (near) .. 1 (far).
 * Used to fade distant parts, which is the cheapest depth cue there is. */
function haze(d, v) {
  return clamp((d - v.dNear) / Math.max(0.001, v.dFar - v.dNear), 0, 1);
}

/* The projection of an ellipsoid is an ellipse. Take the three scaled axis
 * vectors, project each as a small offset, and the screen-space covariance
 * matrix is their outer-product sum. Eigendecompose the 2x2 and you have the
 * ellipse's semi-axes and rotation. Exact enough at these radii, and it
 * means the eye bulges correctly as the fly yaws. */
function ellipsoidScreen(c, r, v) {
  const C  = project(c, v);
  const ax = project([c[0] + r[0], c[1], c[2]], v);
  const ay = project([c[0], c[1] + r[1], c[2]], v);
  const az = project([c[0], c[1], c[2] + r[2]], v);
  const ux = ax.x - C.x, uy = ax.y - C.y;
  const vx = ay.x - C.x, vy = ay.y - C.y;
  const wx = az.x - C.x, wy = az.y - C.y;
  const A = ux * ux + vx * vx + wx * wx;
  const B = ux * uy + vx * vy + wx * wy;
  const D = uy * uy + vy * vy + wy * wy;
  const tr = A + D, det = A * D - B * B;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = Math.max(0.01, tr / 2 + disc);
  const l2 = Math.max(0.01, tr / 2 - disc);
  return {
    C: C, ra: Math.sqrt(l1), rb: Math.sqrt(l2),
    ang: 0.5 * Math.atan2(2 * B, A - D), d: C.d, f: C.f,
  };
}

/* ======================================================================= */
/* drawing primitives                                                      */
/* ======================================================================= */

function shade(cols, near) {
  // `near` is 0..1, 0 = closest to the camera
  if (near < 0.02) return cols;
  const k = 1 - 0.30 * near;
  const mix = (h) => {
    const n = parseInt(h.slice(1), 16);
    const r = Math.round(((n >> 16) & 255) * k);
    const g = Math.round(((n >> 8) & 255) * k);
    const b = Math.round((n & 255) * k * 0.98);   // cools slightly with distance
    return `rgb(${r},${g},${b})`;
  };
  return [mix(cols[0]), mix(cols[1]), mix(cols[2])];
}

/* A lit sphere-ish blob: radial gradient offset toward an upper-left key
 * light, plus a thin dark rim. The rim is what keeps the head, thorax and
 * eyes legible as separate solids instead of melting into one mass. */
function drawEllipsoid(ctx, e, cols) {
  const ra = e.ra, rb = e.rb;
  if (ra < 0.4 || rb < 0.4) return;
  ctx.save();
  ctx.translate(e.C.x, e.C.y);
  ctx.rotate(e.ang);
  const g = ctx.createRadialGradient(
    -ra * 0.34, -rb * 0.38, Math.max(0.6, Math.min(ra, rb) * 0.06),
    -ra * 0.06, -rb * 0.08, Math.max(ra, rb) * 1.30);
  g.addColorStop(0.00, cols[2]);
  g.addColorStop(0.42, cols[1]);
  g.addColorStop(1.00, cols[0]);
  ctx.beginPath();
  ctx.ellipse(0, 0, ra, rb, 0, 0, TAU);
  ctx.fillStyle = g;
  ctx.fill();
  ctx.strokeStyle = 'rgba(20,13,6,0.42)';
  ctx.lineWidth = clamp(Math.min(ra, rb) * 0.045, 0.6, 2.0);
  ctx.stroke();
  ctx.restore();
}

function drawTapered(ctx, p, q, hwA, hwB, col) {
  const dx = q.x - p.x, dy = q.y - p.y;
  const L = Math.hypot(dx, dy) || 1;
  const nx = -dy / L, ny = dx / L;
  ctx.beginPath();
  ctx.moveTo(p.x + nx * hwA, p.y + ny * hwA);
  ctx.lineTo(q.x + nx * hwB, q.y + ny * hwB);
  ctx.lineTo(q.x - nx * hwB, q.y - ny * hwB);
  ctx.lineTo(p.x - nx * hwA, p.y - ny * hwA);
  ctx.closePath();
  ctx.fillStyle = col;
  ctx.fill();
  if (hwA > 0.9) { ctx.beginPath(); ctx.arc(p.x, p.y, hwA, 0, TAU); ctx.fill(); }
  if (hwB > 0.9) { ctx.beginPath(); ctx.arc(q.x, q.y, hwB, 0, TAU); ctx.fill(); }
}

/* ======================================================================= */
/* the renderer                                                            */
/* ======================================================================= */

function Fly3D(canvas, opts) {
  const ctx = canvas.getContext('2d');
  const hud = (opts && opts.hud) || null;
  const reduced = global.matchMedia &&
    global.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let W = 0, H = 0, S = 1;
  let hudAnchor = null;      // stage-space point the leader line runs to
  let t = 0;                 // seconds of animation time
  let last = 0;
  let mood = { dir: 0, age: 99 };
  let raf = 0;
  let lit = 0;               // smoothed brain glow, 0..1

  /* ---------------------------------------------------------------- sizing */
  function resize() {
    const dpr = Math.min(2, global.devicePixelRatio || 1);
    W = Math.max(120, canvas.clientWidth);
    H = Math.max(120, canvas.clientHeight);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    S = Math.min(W * 0.56, H * 1.34);
    if (hud) {
      const cr = canvas.getBoundingClientRect();
      const hr = hud.getBoundingClientRect();
      hudAnchor = { x: hr.left - cr.left, y: hr.top + hr.height / 2 - cr.top };
    }
  }

  /* --------------------------------------------------------- animation state
   * Everything below is a function of `t`, so the fly is deterministic given
   * the clock and never needs to be ticked to stay consistent. */
  function animate() {
    const speed = reduced ? 0.35 : 1;

    // A walking fly yaws gently; a faster, larger wander when excited.
    const yaw = 0.42 * Math.sin(t * 0.42 * speed) + 0.10 * Math.sin(t * 1.13 * speed);

    // Wingbeat. Drosophila run at ~200 Hz; that is a blur at 60 fps, so this
    // is the same motion slowed to a legible 6 Hz.
    const beat = t * TAU * 6.0 * speed;
    const drift = 0.5 + 0.5 * Math.sin(t * 0.7 * speed + 1.1);
    let amp = 0.55 + 0.30 * drift;

    // Every few seconds, two quick flicks of the wings — the fly is not idle.
    const flickPhase = (t * 0.19 * speed) % 1;
    if (flickPhase < 0.05) amp += 0.55 * Math.sin(flickPhase / 0.05 * Math.PI);

    // Reward makes it beat harder; punishment makes it recoil.
    const kick = mood.dir * Math.exp(-mood.age * 2.6);
    if (mood.dir > 0) amp += 0.60 * Math.exp(-mood.age * 3.4);

    // Tripod gait at ~1.8 strides/s.
    const gait = (t * 1.8 * speed) % 1;

    return {
      yaw: yaw,
      beat: beat,
      amp: amp,
      gait: gait,
      kick: kick,
      bob: 0.022 * Math.sin(t * TAU * 1.8 * speed)
           + 0.014 * Math.sin(beat * 0.5)
           - 0.030 * kick,
      ant: 0.10 * Math.sin(t * 1.9 * speed + 0.7),
      abScale: 1 + 0.024 * Math.sin(t * TAU * 1.35 * speed),
      pitch: 0.38 + 0.10 * kick,
    };
  }

  /* -------------------------------------------------------------- body parts */
  function parts(a) {
    const b = a.bob;
    return [
      { c: [-0.44, 0, -0.020 + b * 0.6], r: [0.44 * a.abScale, 0.215, 0.185],
        cols: COLOURS.abdomen, stripes: true },
      { c: [ 0.10, 0,  0.020 + b], r: [0.27, 0.235, 0.225],
        cols: COLOURS.thorax },
      { c: [ 0.47, 0,  0.040 + b], r: [0.17, 0.200, 0.175],
        cols: COLOURS.head },
      { c: [ 0.53,  0.180, 0.080 + b], r: [0.125, 0.095, 0.130],
        cols: COLOURS.eye, eye: true },
      { c: [ 0.53, -0.180, 0.080 + b], r: [0.125, 0.095, 0.130],
        cols: COLOURS.eye, eye: true },
    ];
  }

  /* ------------------------------------------------------------------ legs */
  function legJoints(a) {
    const out = [];
    const stride = 0.115, duty = 0.62;
    for (let s = 0; s < LEGS.length; s++) {
      const L = LEGS[s];
      for (let side = 0; side < 2; side++) {
        const sgn = side ? 1 : -1;             // -1 = fly's right
        const grp = (L.tripod === 1) === (sgn > 0) ? 0 : 0.5;
        let p = (a.gait + grp) % 1;
        let xoff, zoff, flex;
        if (p < duty) {                        // stance: foot planted, body passes over
          const u = p / duty;
          xoff = stride * (0.5 - u);
          zoff = 0;
          flex = 0.010 * u;
        } else {                               // swing: foot lifts and reaches forward
          const u = (p - duty) / (1 - duty);
          xoff = -stride * 0.5 + stride * u;
          zoff = 0.055 * Math.sin(Math.PI * u);
          flex = 0.010 * (1 - u) + 0.030 * Math.sin(Math.PI * u);
        }
        // the middle leg has the least reach; the hind leg the most
        const sc = s === 0 ? 0.85 : (s === 1 ? 0.75 : 1.05);
        xoff *= sc;

        const attach = [L.ax, sgn * L.ay, L.az + a.bob];
        const knee = [L.ax + L.kdx + xoff * 0.45,
                      sgn * (L.ay + L.kout),
                      L.az + L.kdz + a.bob + zoff * 0.55 + flex];
        const foot = [knee[0] + L.fdx + xoff * 0.55,
                      knee[1] * 1.05,
                      knee[2] + L.fdz + zoff * 0.45];
        out.push({ attach: attach, knee: knee, foot: foot, hw: L.hw });
      }
    }
    return out;
  }

  /* ----------------------------------------------------------------- wings */
  function wingPoints(side, a) {
    const sgn = side ? 1 : -1;
    const hinge = [WING_HINGE[0], sgn * WING_HINGE[1], WING_HINGE[2] + a.bob];
    const flap = a.amp * Math.sin(a.beat + (side ? 0 : Math.PI));
    const twist = 0.30 * Math.cos(a.beat + (side ? 0 : Math.PI));
    const cf = Math.cos(flap), sf = Math.sin(flap);
    const ct = Math.cos(twist), st = Math.sin(twist);
    const map = (o) => {
      // the wing's own plane, in body coords
      let ux = o[0], uy = sgn * o[1], uz = 0.008 - 0.02 * Math.min(0, o[0]);
      // angle of attack: rotate the wing about the vertical through the hinge
      const rx = ux * ct - uy * st;
      const ry = ux * st + uy * ct;
      // and sweep it up and down about the hinge's fore-aft axis
      const py = ry, pz = uz;
      return [hinge[0] + rx,
              hinge[1] + py * cf - pz * sf,
              hinge[2] + py * sf + pz * cf];
    };
    return {
      hinge: hinge,
      outline: WING_OUTLINE.map(map),
      veins: WING_VEINS.map((v) => v.map(map)),
    };
  }

  /* ------------------------------------------------------------- the scene */
  function gather(a) {
    const prims = [];

    // wings first so they participate in the same depth sort as everything else
    for (let side = 0; side < 2; side++) {
      const w = wingPoints(side, a);
      const pts = w.outline.map((p) => project(p, a.view));
      const depth = pts.reduce((s, p) => s + p.d, 0) / pts.length;
      prims.push({ kind: 'wing', pts: pts, w: w, depth: depth, a: a });
    }

    for (const leg of legJoints(a)) {
      const p0 = project(leg.attach, a.view);
      const p1 = project(leg.knee, a.view);
      const p2 = project(leg.foot, a.view);
      prims.push({
        kind: 'leg', p0: p0, p1: p1, p2: p2, hw: leg.hw,
        depth: (p0.d + p1.d + p2.d) / 3,
      });
    }

    for (const ant of ANTENNA) {
      const bend = [ant.bend[0], ant.bend[1], ant.bend[2] + a.bob + a.ant * 0.05];
      const tip  = [ant.tip[0],  ant.tip[1],  ant.tip[2]  + a.bob + a.ant * 0.10];
      const p0 = project(ant.from, a.view);
      const p1 = project(bend, a.view);
      const p2 = project(tip, a.view);
      prims.push({ kind: 'ant', p0: p0, p1: p1, p2: p2,
                   depth: (p0.d + p1.d + p2.d) / 3 });
    }

    for (const bp of parts(a)) {
      const e = ellipsoidScreen(bp.c, bp.r, a.view);
      prims.push({ kind: 'blob', e: e, cols: bp.cols,
                   stripes: !!bp.stripes, eye: !!bp.eye,
                   bodyX: bp.c, depth: e.d });
    }

    for (const br of BRISTLES) {
      const p0 = project(br.p, a.view);
      const q = [br.p[0] + br.d[0], br.p[1] + br.d[1], br.p[2] + br.d[2] + a.bob];
      const p1 = project(q, a.view);
      prims.push({ kind: 'bristle', p0: p0, p1: p1, depth: (p0.d + p1.d) / 2 });
    }

    prims.sort((x, y) => y.depth - x.depth);   // far first
    return prims;
  }

  /* ------------------------------------------------------------- the paint */
  function paint(prims, a) {
    const v = a.view;
    for (const pr of prims) {
      const n = haze(pr.depth, v);              // 0 = near, 1 = far
      switch (pr.kind) {
        case 'blob': {
          drawEllipsoid(ctx, pr.e, shade(pr.cols, n));
          if (pr.stripes) stripes(pr, v, n);
          break;
        }
        case 'wing': {
          wing(pr, n, a);
          break;
        }
        case 'leg': {
          const c = `rgba(${Math.round(74 - 16 * n)},${Math.round(53 - 12 * n)},` +
                    `${Math.round(36 - 8 * n)},1)`;
          drawTapered(ctx, pr.p0, pr.p1, pr.hw * pr.p0.f, pr.hw * pr.p1.f * 0.72, c);
          drawTapered(ctx, pr.p1, pr.p2, pr.hw * pr.p1.f * 0.72,
                      pr.hw * pr.p2.f * 0.40, c);
          break;
        }
        case 'ant': {
          const c = 'rgba(44,30,20,1)';
          drawTapered(ctx, pr.p0, pr.p1, 0.014 * pr.p0.f, 0.011 * pr.p1.f, c);
          drawTapered(ctx, pr.p1, pr.p2, 0.011 * pr.p1.f, 0.018 * pr.p2.f, c);
          break;
        }
        case 'bristle': {
          ctx.strokeStyle = `rgba(28,20,12,${0.75 - 0.3 * n})`;
          ctx.lineWidth = Math.max(0.6, 0.008 * pr.p0.f);
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(pr.p0.x, pr.p0.y);
          ctx.lineTo(pr.p1.x, pr.p1.y);
          ctx.stroke();
          break;
        }
      }
    }
  }

  /* Abdominal tergites: dark bands running across the abdomen. Drawn by
   * clipping to the abdomen's screen ellipse and stroking wide lines
   * perpendicular to the projected body axis. */
  function stripes(pr, v, n) {
    const e = pr.e, c = pr.bodyX;
    const along = project([c[0] + 0.1, c[1], c[2]], v);
    let ux = along.x - e.C.x, uy = along.y - e.C.y;
    const L = Math.hypot(ux, uy) || 1;
    ux /= L; uy /= L;
    ctx.save();
    ctx.beginPath();
    ctx.translate(e.C.x, e.C.y);
    ctx.rotate(e.ang);
    ctx.ellipse(0, 0, e.ra, e.rb, 0, 0, TAU);
    ctx.restore();
    ctx.save();
    ctx.clip();
    ctx.strokeStyle = `rgba(52,34,18,${0.55 - 0.2 * n})`;
    ctx.lineCap = 'butt';
    for (const o of [-0.235, -0.075, 0.095, 0.255]) {
      const cm = project([c[0] + o, c[1], c[2]], v);
      ctx.lineWidth = Math.max(0.8, 0.052 * cm.f);
      ctx.beginPath();
      ctx.moveTo(cm.x - uy * 400, cm.y + ux * 400);
      ctx.lineTo(cm.x + uy * 400, cm.y - ux * 400);
      ctx.stroke();
    }
    ctx.restore();
  }

  /* A wing: a translucent membrane with veins, filled with a gradient that
   * runs root -> tip so the membrane reads as thin. */
  function wing(pr, n, a) {
    const pts = pr.pts;
    const root = project(pr.w.hinge, a.view);
    const tipP = project(pr.w.outline[5], a.view);
    const alpha = 0.94 - 0.26 * n;

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();

    const g = ctx.createLinearGradient(root.x, root.y, tipP.x, tipP.y);
    g.addColorStop(0, `rgba(246,251,255,${alpha})`);
    g.addColorStop(0.45, `rgba(208,230,250,${alpha * 0.86})`);
    g.addColorStop(1, `rgba(166,202,236,${alpha * 0.66})`);
    ctx.fillStyle = g;
    ctx.fill();
    ctx.strokeStyle = `rgba(112,150,190,${0.75 - 0.25 * n})`;
    ctx.lineWidth = 1.1;
    ctx.stroke();

    ctx.save();
    ctx.clip();
    ctx.strokeStyle = `rgba(132,170,208,${0.70 - 0.25 * n})`;
    ctx.lineWidth = 0.75;
    for (const vein of pr.w.veins) {
      ctx.beginPath();
      vein.forEach((p, i) => {
        const q = project(p, a.view);
        if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
      });
      ctx.stroke();
    }
    // a glint along the leading edge
    ctx.strokeStyle = `rgba(255,255,255,${0.30 - 0.18 * n})`;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const q = pts[i];
      if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    }
    ctx.stroke();
    ctx.restore();
  }

  /* ------------------------------------------------------------- the brain
   * The mushroom body sits inside the head. It is drawn as an additive glow
   * whose colour and intensity track the last dopamine event, plus a short
   * leader line out to the labelled callout. */
  function brain(a) {
    const head = projectionOfHead(a);
    const pulse = lit;
    const tint = mood.dir > 0 ? [122, 240, 160]
               : mood.dir < 0 ? [130, 176, 235]
               : [255, 214, 130];
    const r = Math.max(10, 0.30 * head.f);
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, r);
    const A = 0.20 + 0.62 * pulse;
    g.addColorStop(0, `rgba(${tint[0]},${tint[1]},${tint[2]},${A})`);
    g.addColorStop(0.35, `rgba(${tint[0]},${tint[1]},${tint[2]},${A * 0.34})`);
    g.addColorStop(1, `rgba(${tint[0]},${tint[1]},${tint[2]},0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(head.x, head.y, r, 0, TAU);
    ctx.fill();
    ctx.restore();

    if (hudAnchor) {
      ctx.save();
      ctx.setLineDash([2.5, 3]);
      ctx.strokeStyle = 'rgba(120,102,168,0.75)';
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      ctx.moveTo(head.x + r * 0.55, head.y);
      ctx.lineTo(hudAnchor.x - 2, hudAnchor.y);
      ctx.stroke();
      ctx.restore();
    }
  }

  function projectionOfHead(a) {
    return project([0.47, 0, 0.04 + a.bob], a.view);
  }

  /* Shared per-frame view setup, so anything drawn outside `gather` lands in
   * exactly the same camera. */
  function makeView(a) {
    const v = {
      cx: W * 0.40, cy: H * 0.50,
      camDist: 3.55, focal: 2.15, scale: S,
      yaw: a.yaw, pitch: a.pitch,
      cosYaw: Math.cos(a.yaw), sinYaw: Math.sin(a.yaw),
      cosPitch: Math.cos(a.pitch), sinPitch: Math.sin(a.pitch),
    };
    // depth range of the animal, for the haze term
    const probe = [
      project([-0.9, -0.45, 0], v).d, project([0.9, 0.45, 0], v).d,
    ];
    v.dNear = Math.min(probe[0], probe[1]);
    v.dFar = Math.max(probe[0], probe[1]);
    return v;
  }

  /* ------------------------------------------------------------------ frame */
  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (!last) last = now;
    let dt = (now - last) / 1000;
    last = now;
    if (dt > 0.25) dt = 0.25;              // tab was hidden; do not lurch
    t += dt;
    mood.age += dt;
    if (mood.age > 2.4) mood.dir = 0;        // let the tint fade back to amber
    // smooth the brain glow toward its target
    const target = mood.dir !== 0 ? Math.exp(-mood.age * 1.5)
                                  : 0.16 + 0.10 * Math.sin(t * 2.2);
    lit += (target - lit) * Math.min(1, dt * 9);

    if (canvas.width === 0) resize();
    const a = animate();
    a.view = makeView(a);
    a.mood = mood;

    ctx.clearRect(0, 0, W, H);
    paint(gather(a), a);
    brain(a);
  }

  /* =================================================================== API */
  const api = {
    start: function () {
      resize();
      if (!raf) { last = 0; raf = requestAnimationFrame(frame); }
    },
    stop: function () {
      if (raf) { cancelAnimationFrame(raf); raf = 0; }
    },
    resize: resize,
    /* direction: +1 sucrose, -1 bitter, 0 neutral */
    pulse: function (direction) {
      mood = { dir: direction, age: 0 };
    },
  };

  return api;
}

/* ------------------------------------------------------------------ mount */
function mount(canvas, opts) {
  const inst = Fly3D(canvas, opts);
  inst.start();
  if (global.ResizeObserver) {
    new ResizeObserver(() => inst.resize()).observe(canvas);
  } else {
    global.addEventListener('resize', () => inst.resize());
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) inst.stop();
    else inst.start();
  });
  return inst;
}

global.Fly3D = { mount: mount };

})(window);
