/* BrainView — a causal map of the fly brain: which part answers, what lights
 * up for sugar, and what gets inflamed by punishment.
 *
 * The picture is organised as a signal chain, because that is the question the
 * panel is supposed to answer:
 *
 *     1 IN     antennal lobe        receptors -> projection neurons
 *     2 CODE   mushroom body        Kenyon cells, the sparse code
 *     3 DECIDE MBON                 the vote: 27 cells, one per class
 *     4 LEARN  PAM / DAN -> MB      the reward stamps (or un-stamps) a row
 *
 * Green/cyan/amber tracts carry the forward sweep from smell to answer, with
 * travelling pulses whose rate follows the activity. The reward pathway is
 * drawn as the *loop back* onto the mushroom body, because that is what it is:
 * sugar GRNs (Gr5a/Gr64f, tarsi + labellum) -> subesophageal zone -> PAM/DAN
 * dopaminergic neurons -> the KC -> MBON synapse.
 *
 * Sugar and bitter are given deliberately opposite renderings, because the
 * model (and the fly) treats them as opposite polarities:
 *   sucrose  warm gold, sweeps outward along the PAM/DAN fan, then a bright
 *            expanding ring on the row that earned the credit. A rush.
 *   bitter   crimson, a blotchy jagged spread that lingers, plus a
 *            cross-out on the row that answered. Inflammation.
 *
 * HONESTY: only AL, MB/KC, MBON and PAM/DAN have real spikes behind them. The
 * optic lobes, lateral horn, central complex, subesophageal zone and taste
 * organ are drawn as faint anatomical context - this model does not implement
 * them. The style echoes FlyWire whole-brain renders, but no FlyWire data is
 * used; it is unreachable without authenticated neuPrint or Codex access.
 */
(function (global) {
'use strict';

const TAU = Math.PI * 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* An anatomical outline in arbitrary units, centred on the brain's midline,
 * with +x to the fly's right and +y forward. Roughly after the FAFB volume:
 * two large optic lobes flanking a central mass, the mushroom bodies sitting
 * dorso-lateral and just inboard of the optic lobes, the antennal lobes
 * anterior-ventral, the central complex on the midline. */
const BRAIN_OUTLINE = [
  [-0.30,  0.66], [ 0.30,  0.66], [ 0.72,  0.60], [ 1.08,  0.42],
  [ 1.34,  0.06], [ 1.36, -0.32], [ 1.16, -0.60], [ 0.78, -0.74],
  [ 0.36, -0.70], [ 0.14, -0.52], [ 0.00, -0.46], [-0.14, -0.52],
  [-0.36, -0.70], [-0.78, -0.74], [-1.16, -0.60], [-1.36, -0.32],
  [-1.34,  0.06], [-1.08,  0.42], [-0.72,  0.60],
];

/* Each region: an ellipse in brain coordinates, and whether the model
 * actually simulates it. */
const REGIONS = [
  { id: 'optic',  name: 'optic lobe',        c: [ 1.02, -0.06], r: [0.34, 0.50],
    rot: -0.18, simulated: false },
  { id: 'optic',  name: 'optic lobe',        c: [-1.02, -0.06], r: [0.34, 0.50],
    rot:  0.18, simulated: false },
  { id: 'lh',     name: 'lateral horn',      c: [ 0.70,  0.12], r: [0.20, 0.26],
    rot: 0, simulated: false },
  { id: 'lh',     name: 'lateral horn',      c: [-0.70,  0.12], r: [0.20, 0.26],
    rot: 0, simulated: false },
  { id: 'cx',     name: 'central complex',   c: [ 0.00,  0.02], r: [0.30, 0.17],
    rot: 0, simulated: false },
  // the mushroom body calyx sits inboard and posterior to the lateral horn
  { id: 'kc',     name: 'mushroom body',     c: [ 0.50, -0.30], r: [0.26, 0.22],
    rot: 0.30, simulated: true },
  { id: 'kc',     name: 'mushroom body',     c: [-0.50, -0.30], r: [0.26, 0.22],
    rot: -0.30, simulated: true },
  { id: 'al',     name: 'antennal lobe',     c: [ 0.34,  0.46], r: [0.19, 0.15],
    rot: 0, simulated: true },
  { id: 'al',     name: 'antennal lobe',     c: [-0.34,  0.46], r: [0.19, 0.15],
    rot: 0, simulated: true },
  // The reward path enters here. The SEZ and the taste organ are not part of
  // the model - they are drawn so the sugar/bitter input has somewhere to come
  // from, and they light only because a reward was actually delivered.
  { id: 'sez',    name: 'subesophageal zone', c: [ 0.00, -0.62], r: [0.28, 0.14],
    rot: 0, simulated: false },
  { id: 'taste',  name: 'sugar / bitter GRNs', c: [ 0.00, -0.80], r: [0.20, 0.08],
    rot: 0, simulated: false },
];

const PALETTE = {
  al:    [111, 224, 163],   // antennal lobe / projection neurons - green
  kc:    [ 90, 200, 255],   // Kenyon cells - cyan
  mbon:  [255, 214, 130],   // MBON output - amber
  dan:   [255, 122, 158],   // PAM/DAN dopamine, tonic - magenta
  sugar: [255, 206,  92],   // sucrose / glucose reward - warm gold
  bitter:[255,  74,  92],   // punishment - crimson
  out:   [186, 230, 255],   // descending output (the answer leaving)
  off:   [104, 122, 150],
};

/* The signal chain, top to bottom. `role` is what the stage does; `sub` is
 * what the model has behind it, so the legend doubles as documentation. */
const STAGES = [
  { id: 'al',   role: '1 IN',     sub: 'receptors -> PNs' },
  { id: 'kc',   role: '2 CODE',   sub: 'Kenyon cells (sparse)' },
  { id: 'mbon', role: '3 DECIDE', sub: '27 output cells, one per class' },
  { id: 'dan',  role: '4 LEARN',  sub: 'sugar/bitter -> stamps a row' },
];


function rgba(c, a) {
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

function BrainView(canvas) {
  const ctx = canvas.getContext('2d');
  let W = 0, H = 0, S = 1, bandH = 1, brainCX = 0, brainCY = 0, stripTop = 0;
  let t = 0, last = 0, raf = 0;
  let act = { al: 0, kc: 0, mbon: 0, dan: 0 };   // target activity, 0..1
  let lit = { al: 0, kc: 0, mbon: 0, dan: 0 };   // smoothed, drawn
  let phase = 0;

  // What the fly argued, and what the reward did about it. These come straight
  // from the server's `flow` / `reward` blocks - nothing is inferred here.
  let counts = [];                 // per-MBON spike tally, one per class
  let labels = [];                 // class characters, same order
  let win = -1, runner = -1, margin = 0;
  let rew = null;                  // last reward block, or null
  let rewT = 0;                    // seconds since that reward landed

  // slow per-region desynchronisation so the glow looks like many cells
  // firing rather than one lamp turning on
  const wobble = { al: 0.0, kc: 1.7, mbon: 3.4, dan: 5.1 };

  function resize() {
    const dpr = Math.min(2, global.devicePixelRatio || 1);
    W = Math.max(120, canvas.clientWidth);
    H = Math.max(80, canvas.clientHeight);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // the brain gets the top band; the MBON vote strip gets the bottom one
    const padTop = 8, padBot = 4;
    stripTop = H - Math.max(50, Math.min(78, H * 0.32));
    bandH = Math.max(60, stripTop - padTop - 6);
    S = Math.min(W / 3.12, bandH / 1.74);
    brainCX = W / 2;
    // the reward pathway hangs below the brain (the SEZ and the taste organ),
    // so the volume sits slightly above the band's centre
    brainCY = padTop + bandH * 0.50;
  }

  /* brain coords -> screen. +y is forward, which is up, hence the flip. */
  function toScreen(p) {
    return { x: brainCX + p[0] * S, y: brainCY - p[1] * S };
  }

  function outlinePath() {
    ctx.beginPath();
    BRAIN_OUTLINE.forEach((p, i) => {
      const q = toScreen(p);
      if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    });
    ctx.closePath();
  }

  /* A quadratic curve between three brain-space points, with pulses running
   * along it. This is what makes "which part gets used" visible: traffic on
   * the tract is the activity, and the direction says where it is going. */
  function tract(p0, p1, p2, colour, drive, speed, opts) {
    const o = opts || {};
    const a = clamp(drive, 0, 1);
    const A = toScreen(p0), B = toScreen(p1), C = toScreen(p2);
    const at = (u) => ({
      x: (1 - u) * (1 - u) * A.x + 2 * (1 - u) * u * B.x + u * u * C.x,
      y: (1 - u) * (1 - u) * A.y + 2 * (1 - u) * u * B.y + u * u * C.y,
    });

    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.quadraticCurveTo(B.x, B.y, C.x, C.y);
    ctx.strokeStyle = rgba(colour, (o.base || 0.10) + (o.gain || 0.42) * a);
    ctx.lineWidth = (o.w0 || 0.8) + (o.wg || 1.4) * a;
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.stroke();
    ctx.setLineDash([]);

    // travelling pulses - fewer, slower when quiet
    const n = o.pulses || 5;
    if (a > 0.04) {
      // `sweep` is set for the reward pathway, where the burst has a real
      // conduction delay and must visibly propagate; forward tracts leave it
      // undefined and their pulses circulate continuously.
      const swept = o.sweep !== undefined;
      const head = swept ? o.sweep : 1;
      for (let i = 0; i < n; i++) {
        // NB: the modulo wraps, so u is not monotonic in i - this has to be a
        // `continue`, not a `break`.
        const u = (phase * (o.rate || speed) + i / n) % 1;
        if (swept && u > head) continue;           // still propagating
        const fade = swept ? clamp(1 - (head - u) / 0.25, 0, 1) : 1;
        const r = (o.pr || 1.6) * (0.6 + 0.9 * a) * fade;
        if (r <= 0.2) continue;
        const q = at(u);
        ctx.beginPath();
        ctx.arc(q.x, q.y, r, 0, TAU);
        ctx.fillStyle = rgba(colour, (0.15 + 0.75 * a) * fade);
        ctx.fill();
      }
    }
  }


  /* How strongly the reward pathway is lit right now, and with what sign.
   * `env` is the burst envelope (sugar fades fast, bitter lingers, which is
   * what makes punishment read as inflammation rather than a flash). */
  function rewardEnv() {
    if (!rew || !rew.delivered || !rew.sign) return 0;
    const tau = rew.sign > 0 ? 0.85 : 2.1;
    return clamp(Math.exp(-rewT / tau), 0, 1);
  }

  /* How far up the reward pathway the burst has travelled, 0..1. The model has
   * a real conduction delay (`dopa_delay` timesteps) plus a phasic tau, so the
   * sweep is not decoration - it is how long the neuromodulator takes to
   * arrive at the mushroom body. */
  function rewardSweep() {
    if (!rew || !rew.delivered) return 0;
    const lag = 0.14 + 0.10 * (rew.delay || 0) + 0.12 * (rew.tau || 3);
    return clamp(rewT / lag, 0, 1);
  }

  /* A region is drawn as a cluster of small somata rather than a solid blob,
   * which is what makes it read as tissue rather than a shape. */
  function drawRegion(rg) {
    const simulated = !!rg.simulated;
    let a = simulated ? lit[rg.id] : 0;
    let base = simulated ? PALETTE[rg.id] : PALETTE.off;

    // The taste organ and the SEZ are not simulated, but they light with the
    // reward that was actually delivered - gold for sugar, crimson for bitter.
    // They are the only unsimulated regions that ever brighten, and only ever
    // as a direct consequence of a real burst.
    const isRewardNode = (rg.id === 'sez' || rg.id === 'taste');
    let ring = 0;
    if (isRewardNode) {
      a = rewardEnv() * rewardSweep();
      base = rew && rew.sign < 0 ? PALETTE.bitter : PALETTE.sugar;
      ring = a;
    }
    // the mushroom body lobes get washed with the reward colour when the burst
    // arrives, because that is where the neuromodulator actually acts
    const isLobe = (rg.id === 'kc');
    const wash = isLobe ? rewardEnv() * rewardSweep() : 0;

    const c = toScreen(rg.c);
    const ra = rg.r[0] * S, rb = rg.r[1] * S;
    const rot = rg.rot;

    // faint anatomical fill, so unsimulated regions still read as structure
    ctx.save();
    ctx.translate(c.x, c.y);
    ctx.rotate(rot);
    ctx.beginPath();
    ctx.ellipse(0, 0, ra, rb, 0, 0, TAU);
    ctx.fillStyle = rgba(base, simulated ? 0.05 + 0.13 * a
                                        : (isRewardNode ? 0.04 + 0.32 * a : 0.035));
    ctx.fill();
    ctx.strokeStyle = rgba(base, (simulated ? 0.30 + 0.50 * a : 0.20) + 0.5 * ring);
    ctx.lineWidth = (simulated ? 0.9 + 0.7 * a : 0.7) + 0.9 * ring;
    if (!simulated && !isRewardNode) ctx.setLineDash([2, 2.6]);
    ctx.stroke();
    ctx.setLineDash([]);

    // The winner. This is the cell group the answer came out of, so it gets a
    // solid amber ring instead of the usual faint outline.
    if (isLobe && win >= 0 && decidable()) {
      ctx.beginPath();
      ctx.ellipse(0, 0, ra * 1.30, rb * 1.30, 0, 0, TAU);
      ctx.strokeStyle = rgba(PALETTE.mbon, 0.16 + 0.62 * lit.mbon);
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    // ...and the punishment ring: jagged, red, and it lingers
    if (isLobe && rew && rew.punished !== null && rew.punished !== undefined) {
      const e = rewardEnv();
      if (e > 0.02) {
        ctx.beginPath();
        const N = 40;
        for (let i = 0; i <= N; i++) {
          const ang = (i / N) * TAU;
          const jag = 1.42 + 0.16 * Math.sin(ang * 9 + t * 14)
                      + 0.09 * Math.sin(ang * 23 - t * 9);
          const px = Math.cos(ang) * ra * jag;
          const py = Math.sin(ang) * rb * jag;
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = rgba(PALETTE.bitter, 0.20 + 0.68 * e);
        ctx.lineWidth = 1.1 + 1.3 * e;
        ctx.stroke();
      }
    }
    ctx.restore();

    if (simulated) {
      ctx.save();
      ctx.translate(c.x, c.y);
      ctx.rotate(rot);
      ctx.globalCompositeOperation = 'lighter';
      // somata: a jittered lattice, each dot's brightness driven by the same
      // activity value but offset in time, so the region sparkles
      const n = 26;
      for (let i = 0; i < n; i++) {
        const ang = (i * 2.39996) + wobble[rg.id];   // golden-angle packing
        const rad = Math.sqrt((i + 0.5) / n);
        const px = Math.cos(ang) * rad * ra * 0.86;
        const py = Math.sin(ang) * rad * rb * 0.86;
        const flick = 0.45 + 0.55 * Math.sin(t * 6.2 + i * 1.7 + wobble[rg.id]);
        const v = clamp(a * flick, 0, 1);
        ctx.beginPath();
        ctx.arc(px, py, Math.max(0.9, (rg.id === 'kc' ? 1.5 : 1.9) * (0.7 + 0.6 * v)),
                0, TAU);
        ctx.fillStyle = rgba(base, 0.10 + 0.80 * v);
        ctx.fill();
      }
      ctx.globalCompositeOperation = 'source-over';
      ctx.restore();

      // a soft halo when the region is hot
      if (a > 0.05) {
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        const HR = Math.max(ra, rb) * 1.7;
        const g = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, HR);
        g.addColorStop(0, rgba(base, 0.26 * a));
        g.addColorStop(1, rgba(base, 0));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(c.x, c.y, HR, 0, TAU);
        ctx.fill();
        ctx.restore();
      }
    }

    // the reward wash on the lobes: the neuromodulator arriving where it acts
    if (wash > 0.02) {
      const col = rew && rew.sign < 0 ? PALETTE.bitter : PALETTE.sugar;
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      const HR = Math.max(ra, rb) * 2.3;
      const g = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, HR);
      g.addColorStop(0, rgba(col, 0.42 * wash));
      g.addColorStop(1, rgba(col, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(c.x, c.y, HR, 0, TAU);
      ctx.fill();
      ctx.restore();
    }

    // and the taste organ / SEZ get a halo of their own, because they are the
    // only unsimulated regions that light up - and only ever because a real
    // reward was delivered
    if (isRewardNode && a > 0.04) {
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      const HR = Math.max(ra, rb) * 2.6;
      const g = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, HR);
      g.addColorStop(0, rgba(base, 0.40 * a));
      g.addColorStop(1, rgba(base, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(c.x, c.y, HR, 0, TAU);
      ctx.fill();
      ctx.restore();
    }
  }

  /* Was the decision at all close? A margin near zero means the fly was
   * guessing, and the picture should not pretend otherwise. */
  function decidable() { return counts.length > 0 && countMax() > 0; }
  function countMax() {
    let m = 0;
    for (const v of counts) if (v > m) m = v;
    return m;
  }


  /* The signal chain, listed as the four things the model actually does. The
   * dim entries are the unsimulated context regions, marked as such. */
  function legend() {
    const x = 7, y0 = 12, dy = 15;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    STAGES.forEach((it, i) => {
      const y = y0 + i * dy;
      const a = lit[it.id];
      ctx.beginPath();
      ctx.arc(x + 3.5, y, 3.2, 0, TAU);
      ctx.fillStyle = rgba(PALETTE[it.id], 0.20 + 0.80 * a);
      ctx.fill();
      ctx.font = 'bold 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = `rgba(${PALETTE[it.id].join(',')},${0.55 + 0.45 * a})`;
      ctx.fillText(it.role, x + 11, y);
      const w = ctx.measureText(it.role).width;
      ctx.font = '8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = `rgba(140,156,184,${0.42 + 0.40 * a})`;
      ctx.fillText(it.sub, x + 14 + w, y);
    });

    // reward key: the two polarities, each shown at its current intensity
    const y = y0 + STAGES.length * dy + 2;
    const e = rewardEnv(), sw = rewardSweep();
    const pairs = [
      { c: PALETTE.sugar,  txt: 'sucrose', a: rew && rew.sign > 0 ? e * sw : 0 },
      { c: PALETTE.bitter, txt: 'bitter',  a: rew && rew.sign < 0 ? e * sw : 0 },
    ];
    let px = x + 3.5;
    for (const p of pairs) {
      ctx.beginPath();
      ctx.arc(px, y, 3.2, 0, TAU);
      ctx.fillStyle = rgba(p.c, 0.16 + 0.84 * p.a);
      ctx.fill();
      ctx.font = '8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = rgba(p.c, 0.38 + 0.55 * p.a);
      ctx.fillText(p.txt, px + 7, y);
      px += 12 + ctx.measureText(p.txt).width;
    }
    ctx.fillStyle = 'rgba(120,136,164,0.55)';
    ctx.font = '8px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText('· dashed = not simulated', px + 4, y);
  }

  /* The forward sweep: smell in through the antennal lobe, across the calyx
   * into the mushroom body, then out to the vote. Traffic on these tracts is
   * proportional to the real activity of the stage it comes from, so "which
   * part is being used" is answered by where the pulses are. */
  function tracts() {
    for (const sgn of [1, -1]) {
      // AL -> MB calyx. Green, driven by projection-neuron rate.
      tract([sgn * 0.34, 0.40], [sgn * 0.52, 0.06], [sgn * 0.50, -0.16],
            PALETTE.al, lit.al, 0.55,
            { base: 0.08, gain: 0.40, w0: 0.9, wg: 1.3, pulses: 4, rate: 0.5, pr: 1.5 });
      // MB -> MBON. Cyan into amber, driven by the Kenyon-cell code.
      tract([sgn * 0.50, -0.40], [sgn * 0.36, -0.58], [sgn * 0.18, -0.60],
            PALETTE.mbon, lit.kc, 0.55,
            { base: 0.07, gain: 0.38, w0: 0.9, wg: 1.4, pulses: 4, rate: 0.5, pr: 1.6 });
      // The answer leaving. Pulses only when there is a decisive vote.
      tract([sgn * 0.18, -0.60], [sgn * 0.07, -0.66], [0.0, -0.74],
            PALETTE.out, lit.mbon * (margin > 0 ? 1 : 0.5), 0.7,
            { base: 0.05, gain: 0.30, w0: 0.7, wg: 0.9, pulses: 3, rate: 0.65, pr: 1.3 });
    }
  }


  /* The reward pathway, drawn as the loop it is: taste organ -> SEZ ->
   * PAM/DAN -> back onto the mushroom-body lobes, where the KC -> MBON synapse
   * is the only plastic thing in the circuit. Sugar and bitter get opposite
   * renderings - a gold outward rush versus a crimson bloated spread - so the
   * two are never mistaken for each other the way a single magenta fan made
   * them. */
  function rewardPath() {
    if (!rew || !rew.delivered || !rew.sign) return;
    const env = rewardEnv();
    if (env <= 0.01) return;
    const sweep = rewardSweep();
    const sugar = rew.sign > 0;
    const col = sugar ? PALETTE.sugar : PALETTE.bitter;

    ctx.save();
    // sugar glows additively; bitter is drawn normally so it looks like it is
    // staining the tissue rather than shining through it
    if (sugar) ctx.globalCompositeOperation = 'lighter';

    for (const sgn of [1, -1]) {
      // taste organ -> SEZ, the input itself
      tract([sgn * 0.05, -0.80], [sgn * 0.10, -0.72], [sgn * 0.04, -0.64],
            col, env, 0.9,
            { base: 0.10, gain: 0.60, w0: 1.0, wg: 2.2, pulses: 4, rate: 0.8,
              pr: 2.0, sweep: clamp(sweep / 0.28, 0, 1) });

      // SEZ -> PAM/DAN -> mushroom body. This is the burst travelling.
      tract([sgn * 0.10, -0.60], [sgn * 0.34, -0.50], [sgn * 0.50, -0.30],
            col, env, 0.75,
            { base: 0.10, gain: 0.62, w0: 0.9, wg: 2.4, pulses: 6, rate: 0.72,
              pr: 2.1, sweep: clamp((sweep - 0.22) / 0.78, 0, 1) });

      // terminal boutons on the lobe: where the neuromodulator is released
      const p0 = toScreen([sgn * 0.10, -0.60]);
      const p1 = toScreen([sgn * 0.34, -0.50]);
      const p2 = toScreen([sgn * 0.50, -0.30]);
      const arrived = clamp((sweep - 0.8) / 0.2, 0, 1);
      if (arrived > 0.01) {
        for (let i = 0; i < 5; i++) {
          const u = 0.62 + i * 0.085;
          const mx = (1 - u) * (1 - u) * p0.x + 2 * (1 - u) * u * p1.x + u * u * p2.x;
          const my = (1 - u) * (1 - u) * p0.y + 2 * (1 - u) * u * p1.y + u * u * p2.y;
          // sugar terminals pop; bitter ones jerk, which is what makes them
          // read as irritated rather than bright
          const f = sugar
            ? 0.5 + 0.5 * Math.sin(t * 9 + i * 1.9)
            : 0.5 + 0.5 * Math.sin(t * 26 + i * 3.7);
          const jit = sugar ? 0 : Math.sin(t * 21 + i * 2.3) * 1.3;
          ctx.beginPath();
          ctx.arc(mx + jit, my + jit * 0.5,
                  1.4 + 2.4 * env * arrived * (0.5 + f), 0, TAU);
          ctx.fillStyle = rgba(col, (0.15 + 0.80 * env * arrived) * (0.5 + 0.5 * f));
          ctx.fill();
        }
      }
    }

    /* Sucrose on the credited row: an expanding gold ring - the memory being
     * written. `credited` is the row the server actually called
     * `apply_dopamine(..., credit=row)` on, so this is not a guess. */
    if (sugar && rew.credited !== null && rew.credited !== undefined) {
      const c = toScreen([0.0, -0.40]);
      const u = clamp((sweep - 0.75) / 0.25, 0, 1);
      for (let r = 0; r < 2; r++) {
        const rad = (0.10 + 0.62 * ((u + r * 0.5) % 1)) * S;
        const fade = env * (1 - ((u + r * 0.5) % 1));
        if (fade <= 0.01) continue;
        ctx.beginPath();
        ctx.ellipse(c.x, c.y, rad, rad * 0.55, 0, 0, TAU);
        ctx.strokeStyle = rgba(PALETTE.sugar, 0.10 + 0.55 * fade);
        ctx.lineWidth = 0.9 + 2.0 * fade;
        ctx.stroke();
      }
    }

    /* Bitter on the row that answered wrong: the tissue around it swells and
     * blotches, and the swelling keeps growing for as long as the burst lasts.
     * This is the "inflamed" reading - it should look unhealthy, not pretty. */
    if (!sugar) {
      // capped so the swelling stays on the brain and never bleeds into the
      // vote strip below it
      const grow = clamp(0.4 + rewT * 0.45, 0.4, 1.75);
      for (const sgn of [1, -1]) {
        const c = toScreen([sgn * 0.50, -0.30]);
        ctx.save();
        ctx.translate(c.x, c.y);
        ctx.rotate(Math.sin(t * 0.9) * 0.05);
        ctx.beginPath();
        const N = 54;
        for (let i = 0; i <= N; i++) {
          const ang = (i / N) * TAU;
          const wob = 1.0 + 0.20 * Math.sin(ang * 5 + t * 3.1)
                          + 0.11 * Math.sin(ang * 13 - t * 5.7)
                          + 0.06 * Math.sin(ang * 29 + t * 11.0);
          const rad = S * 0.30 * grow * wob;
          const px = Math.cos(ang) * rad;
          const py = Math.sin(ang) * rad * 0.72;
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fillStyle = rgba(PALETTE.bitter, 0.055 + 0.16 * env);
        ctx.fill();
        ctx.strokeStyle = rgba(PALETTE.bitter, 0.16 + 0.52 * env);
        ctx.lineWidth = 0.8 + 1.1 * env;
        ctx.stroke();

        // speckle: inflamed tissue is granular, not smooth
        for (let i = 0; i < 16; i++) {
          const ang = i * 2.39996 + t * 0.7;
          const rad = Math.sqrt((i + 0.5) / 16) * S * 0.26 * grow;
          const px = Math.cos(ang) * rad;
          const py = Math.sin(ang) * rad * 0.72;
          const f = 0.5 + 0.5 * Math.sin(t * 17 + i * 2.7);
          ctx.beginPath();
          ctx.arc(px, py, 0.8 + 1.5 * env * f, 0, TAU);
          ctx.fillStyle = rgba(PALETTE.bitter, (0.20 + 0.55 * env) * f);
          ctx.fill();
        }
        ctx.restore();
      }
    }
    ctx.restore();
  }


  /* The vote. 27 output cells, one per class, drawn as a bar chart under the
   * brain. This is the most direct answer to "which part gets used for the
   * response": the tallest bar IS the answer, and the runner-up shows what it
   * beat. The row the reward stamped (gold) or suppressed (crimson) is marked
   * on the same axis, so cause and effect share one picture. */
  function mbonStrip() {
    const pad = 8;
    const top = stripTop;
    const sepY = top - 2;

    ctx.beginPath();
    ctx.moveTo(pad, sepY);
    ctx.lineTo(W - pad, sepY);
    ctx.strokeStyle = 'rgba(120,140,175,0.20)';
    ctx.lineWidth = 0.7;
    ctx.stroke();

    ctx.textBaseline = 'top';

    if (!counts.length) {
      ctx.font = '8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.textAlign = 'left';
      ctx.fillStyle = 'rgba(132,148,176,0.66)';
      ctx.fillText('the vote — present a letter to see the 27 output cells', pad, top + 2);
      return;
    }

    const max = countMax() || 1;
    const headH = 11, footH = 12, letterH = 11;
    const barsTop = top + headH;
    const barsBot = H - footH - letterH;
    const barH = Math.max(10, barsBot - barsTop);
    const n = counts.length;
    const cw = (W - pad * 2) / n;
    const bw = Math.max(2, Math.min(11, cw - 2.6));

    // heading + what the fly argued
    ctx.textAlign = 'left';
    ctx.font = '8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = 'rgba(140,156,184,0.70)';
    ctx.fillText('the vote — 27 output cells, one per class', pad, top + 1);

    ctx.textAlign = 'right';
    if (win >= 0) {
      const wl = labels[win] === ' ' ? 'blank' : labels[win];
      const rl = labels[runner] === ' ' ? 'blank' : labels[runner];
      const tight = margin < 0.12;
      ctx.fillStyle = tight ? 'rgba(255,196,120,0.90)' : 'rgba(255,214,130,0.92)';
      const txt = tight
        ? `too close to call: '${wl}' edged '${rl}' by ${(margin * 100).toFixed(0)}%`
        : `argued '${wl}' over '${rl}' by ${(margin * 100).toFixed(0)}%`;
      ctx.fillText(txt, W - pad, top + 1);
    }

    for (let i = 0; i < n; i++) {
      const v = counts[i] / max;
      const cx = pad + i * cw + cw / 2;
      const h = Math.max(0, v * barH);
      const x0 = cx - bw / 2;

      const isWin = i === win;
      const isRun = i === runner;
      const credited = rew && rew.credited === i;
      const punished = rew && rew.punished === i;

      // track
      ctx.fillStyle = 'rgba(96,112,146,0.13)';
      ctx.fillRect(x0, barsTop, bw, barH);

      // fill
      const a = 0.18 + 0.72 * v;
      ctx.fillStyle = rgba(PALETTE.mbon, isWin ? Math.min(1, a + 0.25) : a * 0.8);
      ctx.fillRect(x0, barsBot - h, bw, h);

      if (isWin || isRun) {
        ctx.strokeStyle = rgba(PALETTE.mbon, isWin ? 0.85 : 0.34);
        ctx.lineWidth = 0.9;
        ctx.strokeRect(x0 - 0.5, barsTop - 0.5, bw + 1, barH + 1);
      }

      // the plasticity mark, on the same axis as the vote it acted on
      if (credited) {
        const e = rewardEnv();
        ctx.beginPath();
        ctx.arc(cx, barsTop - 4, 2.4 + 1.4 * e, 0, TAU);
        ctx.fillStyle = rgba(PALETTE.sugar, 0.55 + 0.45 * e);
        ctx.fill();
      }
      if (punished) {
        const e = rewardEnv();
        ctx.beginPath();
        ctx.arc(cx, barsTop - 4, 2.2 + 1.2 * e, 0, TAU);
        ctx.fillStyle = rgba(PALETTE.bitter, 0.55 + 0.45 * e);
        ctx.fill();
        // the cross-out: this row was suppressed, not reinforced
        const r = 3.0 + e;
        ctx.beginPath();
        ctx.moveTo(cx - r, barsTop - 4 - r + 1);
        ctx.lineTo(cx + r, barsTop - 4 + r - 1);
        ctx.moveTo(cx + r, barsTop - 4 - r + 1);
        ctx.lineTo(cx - r, barsTop - 4 + r - 1);
        ctx.strokeStyle = rgba(PALETTE.bitter, 0.30 + 0.60 * e);
        ctx.lineWidth = 1.1;
        ctx.stroke();
      }

      // the class letter under each bar - the whole alphabet voting at once
      ctx.textAlign = 'center';
      ctx.font = (isWin ? 'bold ' : '') + '7.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      const ch = labels[i] === undefined ? String(i) : (labels[i] === ' ' ? '_' : labels[i]);
      ctx.fillStyle = isWin
        ? 'rgba(255,226,168,0.98)'
        : `rgba(138,154,182,${0.42 + 0.36 * v})`;
      ctx.fillText(ch, cx, barsBot + 1);
    }

    // the outcome line: what the reward did, in words, with the sign kept
    const y = H - footH + 1;
    ctx.textAlign = 'left';
    ctx.font = '8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    if (!rew || !rew.delivered) {
      ctx.fillStyle = 'rgba(126,142,170,0.70)';
      ctx.fillText('no reward — nothing was written to the weights', pad, y);
    } else if (rew.sign > 0) {
      const L = rew.credited_label === ' ' ? 'blank' : rew.credited_label;
      ctx.fillStyle = 'rgba(255,206,92,0.95)';
      ctx.fillText(`sucrose → SEZ → PAM/DAN → MB:  row '${L}' reinforced`
                   + `  (RPE ${rew.rpe >= 0 ? '+' : ''}${rew.rpe.toFixed(2)},`
                   + ` baseline ${rew.baseline.toFixed(2)})`, pad, y);
    } else {
      const L = rew.punished_label === ' ' ? 'blank' : rew.punished_label;
      ctx.fillStyle = 'rgba(255,110,124,0.95)';
      ctx.fillText(`bitter → SEZ → DAN → MB:  row '${L}' suppressed`
                   + `  (RPE ${rew.rpe.toFixed(2)}, baseline ${rew.baseline.toFixed(2)})`,
                   pad, y);
    }
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (!last) last = now;
    let dt = (now - last) / 1000;
    last = now;
    if (dt > 0.25) dt = 0.25;
    t += dt;
    phase += dt;
    if (rew && rew.delivered) rewT += dt;

    if (canvas.width === 0) resize();

    // smooth toward the current activity
    const k = Math.min(1, dt * 5.5);
    for (const id of ['al', 'kc', 'mbon', 'dan']) {
      lit[id] += (act[id] - lit[id]) * k;
    }

    ctx.clearRect(0, 0, W, H);

    // the volume itself: a dark tissue fill with a bright boundary. The
    // forward sweep is drawn inside the clip so it never spills out of the
    // brain; the reward loop is drawn outside it, because the sugar receptors
    // are on the tarsi and the SEZ sits ventral - that pathway really does
    // come from outside the brain.
    ctx.save();
    outlinePath();
    const bg = ctx.createRadialGradient(W / 2, brainCY, 0, W / 2, brainCY,
                                        Math.max(W, H) * 0.62);
    bg.addColorStop(0, 'rgba(26,30,48,0.92)');
    bg.addColorStop(1, 'rgba(12,15,26,0.92)');
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.clip();
    tracts();
    ctx.restore();

    ctx.save();
    outlinePath();
    ctx.strokeStyle = 'rgba(120,180,230,0.34)';
    ctx.lineWidth = 1.1;
    ctx.stroke();
    ctx.restore();

    // midline
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = 'rgba(120,140,175,0.28)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    const a0 = toScreen([0, 0.62]), a1 = toScreen([0, -0.50]);
    ctx.moveTo(a0.x, a0.y);
    ctx.lineTo(a1.x, a1.y);
    ctx.stroke();
    ctx.restore();

    // reward loop over the top: it ends on the mushroom body, which is where
    // the only plastic synapses in the circuit are
    rewardPath();

    for (const rg of REGIONS) drawRegion(rg);

    legend();
    mbonStrip();

    // anterior / posterior cue
    ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(120,138,168,0.62)';
    const ant = toScreen([0, 0.66]);
    ctx.fillText('anterior', W - 8, ant.y + 3);
    ctx.textAlign = 'left';
    const pos = toScreen([0, -0.76]);
    ctx.fillText('posterior', 8, pos.y + 2);
  }


  const api = {
    start: function () {
      resize();
      if (!raf) { last = 0; raf = requestAnimationFrame(frame); }
    },
    stop: function () { if (raf) { cancelAnimationFrame(raf); raf = 0; } },
    resize: resize,

    /* The class characters, in the same order as the vote vector. */
    setClasses: function (list) {
      if (Array.isArray(list) && list.length) labels = list.slice();
    },

    /* Take a whole `/api/present` response. Everything drawn is a field the
     * server computed - the winning MBON, the margin, and the row the
     * dopamine burst was aimed at - so nothing on this panel is inferred. */
    set: function (r) {
      if (!r) return;
      const rg = r.regions;
      if (rg) {
        act.al = clamp(rg.al || 0, 0, 1);
        act.kc = clamp(rg.kc || 0, 0, 1);
        act.mbon = clamp(rg.mbon || 0, 0, 1);
        act.dan = clamp(rg.dan || 0, 0, 1);
      }

      if (Array.isArray(r.mbon) && r.mbon.length) {
        counts = r.mbon.slice();
      }
      if (r.flow) {
        if (r.flow.top && r.flow.top.length && !labels.length) {
          // no class list yet: fall back to whatever the server named
          const n = counts.length || 27;
          const guess = new Array(n).fill('?');
          for (const e of r.flow.top) guess[e.i] = e.label;
          labels = guess;
        }
        const idx = [];
        for (let i = 0; i < counts.length; i++) idx.push(i);
        idx.sort((a, b) => counts[b] - counts[a]);
        win = idx.length ? idx[0] : -1;
        runner = idx.length > 1 ? idx[1] : -1;
        margin = Number(r.flow.margin) || 0;
      }

      // A delivered reward restarts the sweep and holds the sign, so sugar and
      // bitter drive visibly different renders. Without this the two were
      // indistinguishable: both just pinned the magenta fan on.
      //
      // A no-reward update must not erase a burst that is still animating -
      // the phrase-level sugar lands and is immediately followed by a plain
      // `present`, which used to cut the flash off before it was visible.
      const rw = r.reward;
      if (rw && rw.delivered && rw.sign) {
        rew = rw;
        rewT = 0;
        act.dan = rw.sign > 0 ? 1 : 0.9;
      } else if (!(rw && !rw.delivered && rew && rew.delivered && rewT < 1.2)) {
        if (rw) rew = rw;   // keep the last credit/punish marks and the baseline
      }
    },
    clear: function () {
      act = { al: 0, kc: 0, mbon: 0, dan: 0 };
      counts = []; win = -1; runner = -1; margin = 0;
      rew = null; rewT = 0;
    },
  };
  return api;
}

function mount(canvas) {
  const inst = BrainView(canvas);
  inst.start();
  if (global.ResizeObserver) {
    new ResizeObserver(() => inst.resize()).observe(canvas);
  } else {
    global.addEventListener('resize', () => inst.resize());
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) inst.stop(); else inst.start();
  });
  return inst;
}

global.BrainView = { mount: mount };

})(window);
