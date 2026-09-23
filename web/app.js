/* FlyBrain live UI. No frameworks: fetch + canvas. */

const CMAP = {
  magma: [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],
          [229,80,100],[251,135,97],[254,194,135],[252,253,191]],
  viridis: [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
            [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]],
  diverging: [[33,102,172],[103,169,207],[209,229,240],[247,247,247],
              [253,219,199],[239,138,98],[178,24,43]],
};

const HEIGHTS = { rec: 180, pn: 70, kc: 70, mbon: 240, dopa: 90, curve: 120, w: 150 };

const state = {
  classes: [],       // the fly's vocabulary: A-Z + space. what it can say.
  digits: [],        // 0-9: presentable input it has no output cell for
  glyphs: {},        // every drawable glyph, digits included
  learn: false,
  tab: 'letters',
  last: null,
  history: [],
  proof: null,
};

/* ------------------------------------------------------------------ helpers */

function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function prep(cv, h) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(80, cv.clientWidth || 400);
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function blank(cv, h) {
  const { ctx, w } = prep(cv, h);
  ctx.fillStyle = '#0a0d13';
  ctx.fillRect(0, 0, w, h);
  return { ctx, w, h };
}

function heat(cv, values, cols, cmap, vmin, vmax, h) {
  const { ctx, w } = blank(cv, h);
  const rows = Math.max(1, Math.ceil(values.length / cols));
  const off = document.createElement('canvas');
  off.width = cols; off.height = rows;
  const octx = off.getContext('2d');
  const img = octx.createImageData(cols, rows);
  const span = (vmax - vmin) || 1;
  for (let i = 0; i < values.length; i++) {
    const [r, g, b] = ramp(cmap, (values[i] - vmin) / span);
    img.data[i * 4] = r; img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, w, h);
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? undefined : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

function $(id) { return document.getElementById(id); }

/* -------------------------------------------------------------------- fly */

let pulseTimer = null;

function pulseFly(direction, msg) {
  const glow = $('mb-glow'), mb = $('mb');
  const warm = direction >= 0;

  // the 3D animal reacts too: wings beat harder for sugar, the body recoils
  // from bitter, and the brain glow inside the head flares
  if (state.fly) state.fly.pulse(direction);

  // restart the beat animation: force a reflow so the class re-applies
  mb.classList.remove('warm', 'cold', 'beat');
  if (direction !== 0) {
    void mb.getBoundingClientRect();
    mb.classList.add('beat', warm ? 'warm' : 'cold');
  }

  clearTimeout(pulseTimer);
  pulseTimer = setTimeout(
    () => mb.classList.remove('beat', 'warm', 'cold'), 1500);

  glow.style.transition = 'none';
  glow.setAttribute('opacity', direction === 0 ? '0' : '1');
  requestAnimationFrame(() => {
    glow.style.transition = 'opacity 1.1s ease-out';
    glow.setAttribute('opacity', '0');
  });

  const el = $('fly-state');
  el.textContent = msg;
  el.style.color =
    direction === 0 ? '#8b98ad' : (warm ? '#4ade80' : '#f87171');
}

let bubbleTimer = null;

/* what the fly says. shows the letter it read, or the whole phrase it managed.
   long text lingers, a single letter flashes and goes. */
function sayBubble(text, kind) {
  const el = $('bubble'), tx = $('bubble-text');
  if (!el || !tx) return;
  const said = (text || '').trim();
  tx.textContent = said === '' ? '…' : said;
  el.classList.remove('right', 'wrong');
  if (kind) el.classList.add(kind);
  // restart the pop-in even if the bubble is already up
  el.classList.remove('show');
  void el.getBoundingClientRect();
  el.classList.add('show');

  clearTimeout(bubbleTimer);
  bubbleTimer = setTimeout(() => el.classList.remove('show'),
                           said.length > 3 ? 3600 : 1500);
}

/* ------------------------------------------------------------------ panels */

function drawReceptor(rate, raster, tStim) {
  const h = HEIGHTS.rec;
  const { ctx, w } = blank($('cv-rec'), h);
  const pad = 8;
  const cell = (h - 2 * pad) / 7;          // 7 rows of a 5x7 glyph
  const gw = 5 * cell;

  // the glyph, brightness = mean firing rate on that receptor
  const lo = Math.min(...rate), hi = Math.max(...rate);
  const span = (hi - lo) || 1;
  for (let r = 0; r < 7; r++) {
    for (let c = 0; c < 5; c++) {
      const v = (rate[r * 5 + c] - lo) / span;
      const [rr, gg, bb] = ramp(CMAP.viridis, v);
      ctx.fillStyle = `rgb(${rr | 0},${gg | 0},${bb | 0})`;
      ctx.fillRect(pad + c * cell + 1, pad + r * cell + 1, cell - 2, cell - 2);
    }
  }

  // the raw Poisson spikes: 35 columns x t_stim rows
  const rx = pad + gw + 18;
  const availW = w - rx - pad;
  const px = Math.max(1, Math.min((h - 2 * pad) / tStim, availW / 35));
  ctx.fillStyle = '#8b98ad';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(`${tStim} steps`, rx, pad - 0.5);
  for (let p = 0; p < 35; p++) {
    for (let t = 0; t < tStim; t++) {
      if (!raster[t * 35 + p]) continue;
      ctx.fillStyle = '#7ee0c0';
      ctx.fillRect(rx + p * px, pad + 4 + t * px, Math.max(1, px - 0.5), Math.max(1, px - 0.5));
    }
  }
}

function drawMbon(counts, predicted) {
  const h = HEIGHTS.mbon;
  const { ctx, w } = blank($('cv-mbon'), h);
  const pad = 6;
  const labelW = 22;
  const rowH = (h - 2 * pad) / counts.length;
  const max = Math.max(1, ...counts);
  const barW = w - labelW - pad;
  // The row that the presented character *should* have won, or -1 when the
  // character has no output cell at all (a digit). Without this the bar chart
  // silently implied a target that does not exist.
  const target = state.classes.indexOf(predicted);
  ctx.font = '9px ui-monospace, monospace';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < counts.length; i++) {
    const y = pad + i * rowH;
    const isPred = state.classes[i] === predicted;
    const isTarget = i === target;
    const len = (counts[i] / max) * barW;
    ctx.fillStyle = isPred ? '#4ade80' : (isTarget ? '#f0a84a' : '#39445a');
    ctx.fillRect(labelW, y + 0.5, Math.max(len, counts[i] > 0 ? 1 : 0), rowH - 1);
    ctx.fillStyle = isPred ? '#cdeccd' : (isTarget ? '#f0c48a' : '#7b8798');
    ctx.fillText(JSON.stringify(state.classes[i]).replace(/"/g, ''), 4, y + rowH / 2);
    if (counts[i] > 0) {
      ctx.fillStyle = isPred ? '#4ade80' : '#5b6879';
      ctx.fillText(counts[i].toFixed(0), labelW + len + 4, y + rowH / 2);
    }
  }
}

function drawDopa(trace, label) {
  const h = HEIGHTS.dopa;
  const { ctx, w } = blank($('cv-dopa'), h);
  if (!trace) {
    ctx.fillStyle = '#5b6879';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('no reward delivered yet', 8, h / 2);
    return;
  }
  const pad = 8;
  const peak = Math.max(1, ...trace.map(Math.abs));
  const y0 = trace.every(v => v >= 0) ? h - pad - 4 : h / 2;
  ctx.strokeStyle = '#2a3346';
  ctx.beginPath(); ctx.moveTo(pad, y0); ctx.lineTo(w - pad, y0); ctx.stroke();

  const bw = (w - 2 * pad) / trace.length;
  for (let i = 0; i < trace.length; i++) {
    const v = trace[i] / peak;
    const bh = v * (h / 2 - pad);
    ctx.fillStyle = v >= 0 ? '#4ade80' : '#f87171';
    ctx.fillRect(pad + i * bw + 1, v >= 0 ? y0 - bh : y0, Math.max(1, bw - 2), Math.abs(bh));
  }
  ctx.fillStyle = label.startsWith('sucrose') ? '#4ade80' : '#f87171';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(label, pad, 12);
}

function drawCurve(hist) {
  const h = HEIGHTS.curve;
  const { ctx, w } = blank($('cv-curve'), h);
  const pad = 8;
  ctx.strokeStyle = '#222a38';
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  if (!hist || !hist.length) {
    ctx.fillStyle = '#5b6879';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('not trained yet', pad + 6, h / 2);
    return;
  }
  const n = hist.length;
  const X = i => pad + (n === 1 ? 0.5 : i / (n - 1)) * (w - 2 * pad);
  const Y = a => h - pad - a * (h - 2 * pad);

  ctx.strokeStyle = '#2f5d3f';
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(pad, Y(1 / 27)); ctx.lineTo(w - pad, Y(1 / 27)); ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#ffd166';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  hist.forEach((a, i) => i ? ctx.lineTo(X(i), Y(a)) : ctx.moveTo(X(i), Y(a)));
  ctx.stroke();
  ctx.fillStyle = '#ffd166';
  ctx.beginPath(); ctx.arc(X(n - 1), Y(hist[n - 1]), 2.5, 0, 7); ctx.fill();

  ctx.fillStyle = '#5b6879';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText('chance', pad + 4, Y(1 / 27) - 3);
  ctx.fillText(`${hist[hist.length - 1].toFixed(1)}%`, w - pad - 32, Y(hist[n - 1]) - 5);
}

function drawWeights(w) {
  if (!w || !w.length) return;
  const flat = w.flat();
  const vmax = Math.max(...flat.map(Math.abs)) || 1;
  heat($('cv-w'), flat, w[0].length, CMAP.diverging, -vmax, vmax, HEIGHTS.w);
}

/* ------------------------------------------------------------------ actions */

function setBusy(busy, pct) {
  document.querySelectorAll('button').forEach(b => { b.disabled = busy; });
  $('bar').style.width = busy ? `${pct || 8}%` : '0%';
}

async function present(char, silent) {
  try {
    const r = await api('/api/present', { char, learn: state.learn });
    state.last = r;
    render(r);
    // `silent` is for the re-read that follows training: the character was
    // already read once, so it must not be appended to the transcript twice.
    if (!silent) txPush(r);
  } catch (e) {
    log(`error: ${e.message}`);
  }
}

function render(r) {
  drawReceptor(r.receptor_rate, r.receptor_raster, r.t_stim);
  heat($('cv-pn'), r.pn, r.pn.length, CMAP.viridis,
       Math.min(...r.pn), Math.max(...r.pn), HEIGHTS.pn);
  heat($('cv-kc'), r.kc_act, r.kc_act.length, CMAP.magma, 0,
       Math.max(...r.kc_act) || 1, HEIGHTS.kc);
  drawMbon(r.mbon, r.predicted);
  drawDopa(r.burst, r.burst && r.burst.some(v => v !== 0)
    ? (r.correct ? 'sucrose burst' : 'bitter burst') : 'no reward');

  // The whole response goes to the brain view. It needs the vote vector (to
  // show which of the 27 output cells won), the flow block (the winner, the
  // runner-up, the margin) and the reward block with its sign kept, so sugar
  // and bitter can be drawn as different things.
  if (state.brain) state.brain.set(r);

  const nActive = r.kc_act.filter(v => v > 0).length;
  $('lbl-pn').textContent =
    `128 PNs · range ${Math.min(...r.pn).toFixed(2)}–${Math.max(...r.pn).toFixed(2)}`;
  $('lbl-kc').textContent =
    `${nActive}/512 active (${(r.kc_sparsity * 100).toFixed(1)}%) · top-k after APL`;
  $('lbl-mbon').textContent =
    `${r.mbon.reduce((a, b) => a + b, 0).toFixed(0)} spikes · argued ${r.predicted}`;

  const out = $('readout');
  // Three outcomes, not two. A digit is neither a hit nor a miss: it sits
  // outside the fly's vocabulary, and rendering it as a plain red "wrong"
  // would understate what is actually happening - the fly is not failing at a
  // task it has, it is being asked a question it has no cell to answer.
  out.className = 'readout ' +
    (r.known === false ? 'unknown' : (r.correct ? 'ok' : 'no'));
  if (r.known === false) {
    out.innerHTML = `${r.predicted === ' ' ? '␣' : r.predicted}
      <small class="warn">guessed &mdash; but <em>${r.char}</em> is not one of the
      27 characters this fly can say, so this answer cannot be right</small>`;
  } else {
    out.innerHTML = `${r.predicted === ' ' ? '␣' : r.predicted}
      <small>${r.correct ? 'correct' : `presented ${r.char === ' ' ? '␣' : r.char}`}
      ${r.learned ? '· plasticity applied' : ''}</small>`;
  }

  $('acc-badge').textContent = `accuracy ${(r.history.length ? state.accuracy || 0 : 0).toFixed(0)}%`;
  $('spar-badge').textContent = `KC code ${(r.kc_sparsity * 100).toFixed(1)}% active`;
  $('dopa-badge').textContent = `dopamine baseline ${r.dopa_baseline >= 0 ? '+' : ''}${r.dopa_baseline.toFixed(3)}`;

  if (r.learned) {
    pulseFly(r.pulse, r.correct
      ? `sucrose — ${r.char} read correctly`
      : `bitter — ${r.char} misread as ${r.predicted}`);
  } else if (r.known === false) {
    pulseFly(-1, `bitter — no output cell for '${r.char}'`);
  } else {
    pulseFly(0, `${r.char} presented (no learning)`);
  }

  // the animal says what it just heard
  sayBubble(r.predicted === ' ' ? '␣' : r.predicted,
            r.correct ? 'right' : 'wrong');
}

function log(msg) {
  const el = $('log');
  const d = document.createElement('div');
  d.textContent = msg;
  el.prepend(d);
  while (el.children.length > 40) el.lastChild.remove();
}

async function train(epochs) {
  setBusy(true, 20);
  log(`training ${epochs} epochs…`);
  try {
    const r = await api('/api/train', { epochs });
    state.accuracy = r.accuracy * 100;
    state.history = r.history.map(h => h.accuracy);
    $('acc-badge').textContent = `accuracy ${(r.accuracy * 100).toFixed(1)}%`;
    $('acc-badge').className = 'badge ' + (r.accuracy > 0.1 ? 'on' : 'off');
    drawCurve(state.history);
    drawWeights((await api('/api/weights')).weights);
    log(`trained ${epochs} epochs → ${(r.accuracy * 100).toFixed(1)}%`);
    if (state.last) present(state.last.char, true);
  } catch (e) {
    log(`error: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

async function say() {
  const phrase = $('phrase').value || 'PHILIPPE DELAMBRE';
  setBusy(true, 30);
  try {
    const r = await api('/api/say', { phrase, learn: state.learn });
    // Four render states, because there are four things that can happen to a
    // character: read correctly, read wrongly, presented-but-unanswerable (a
    // digit), or not presentable at all (no glyph). Collapsing the last two
    // into "correct" is what made digits look recognised.
    $('said').innerHTML = r.results.map(e => {
      const ch = e.char === ' ' ? '&nbsp;' : e.char;
      const title = e.skipped
        ? 'no 5x7 glyph — never presented'
        : (e.known === false
            ? `presented, but the fly has no '${e.char}' to answer with`
            : (e.correct ? 'read correctly' : `misread as ${e.predicted}`));
      const cls = e.skipped ? 'skip'
        : (e.known === false ? 'unknown' : (e.correct ? 'hit' : 'miss'));
      return `<span class="${cls}" title="${title}">${ch}</span>`;
    }).join('');

    const bits = [`${r.letters_correct}/${r.letters} letters read`];
    if (r.unknown_chars) bits.push(`${r.unknown_chars} not in vocabulary`);
    if (r.skipped) bits.push(`${r.skipped} skipped (${r.skipped_chars})`);
    log(`said "${r.said}" — ${bits.join(', ')}`);
    // the phrase was read, so it belongs in the running text like any other
    // read - per character, in order, with the same verdicts
    r.results.forEach(txPush);

    if (state.brain) state.brain.set(r);
    pulseFly(r.correct ? 1 : -1, r.correct
      ? 'sucrose — it said the whole phrase!'
      : (r.unknown_chars
          ? `bitter — ${r.unknown_chars} character(s) outside the vocabulary`
          : 'bitter — the phrase came out wrong'));
    drawDopa(r.burst, r.correct ? 'sucrose burst (phrase)' : 'bitter burst (phrase)');
    sayBubble(r.said, r.correct ? 'right' : 'wrong');
    // NB: this used to call present(lastLetter) here, which overwrote the brain
    // panel with a single letter and threw away the phrase it had just drawn.
    if (r.correct) log('perfect read — the fly said the whole phrase');
  } catch (e) {
    log(`error: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------- what it has read so far */

/* The vote strip says which of the 27 output cells won *this* presentation.
   This accumulates those wins into text, which is the only place in the UI
   where the fly's output is allowed to become prose. Because it looks like
   prose, it is the easiest place in the whole page to accidentally lie, so:

     - a character the fly was never shown (no glyph in the 5x7 font) is NOT
       added as though it had been read. it is drawn greyed and struck through,
       because it was on the page and the fly did not read it. it is counted
       separately and left out of the accuracy figure entirely.
     - a digit is added as the letter the fly answered, in amber, because that
       letter is genuinely what it said - it simply cannot be right.
     - a misread letter is added as the letter the fly said, in red and struck
       through, with the character that was actually on the page in the tooltip.

   So the paragraph is a transcription of the fly, per-character fidelity
   included. Nothing is silently corrected on its behalf. */

const TX_MAX = 1400;   // characters kept in the DOM; the counters cover everything

const TX = {
  out: [],        // every character the fly has said, oldest first
  word: [],       // the characters since its last space
  lastWord: '',   // the word it just finished, kept visible after the space
  hit: 0, miss: 0, unknown: 0, skip: 0, words: 0,
};

function txClass(r) {
  if (r.skipped) return 'skip';
  if (r.known === false) return 'unknown';
  return r.correct ? 'hit' : 'miss';
}

function txTitle(r) {
  if (r.skipped) {
    return `the page had "${r.char}", but the 5x7 font has no glyph for it — `
      + 'it was never shown, so it did not read it';
  }
  const said = r.predicted === ' ' ? 'a space' : `"${r.predicted}"`;
  if (r.known === false) {
    return `the page had "${r.char}"; the fly said ${said} — it has no output `
      + `cell for "${r.char}", so this answer cannot be right`;
  }
  if (r.correct) return `the fly said ${said} — correct`;
  return `the page had "${r.char}"; the fly said ${said}`;
}

/* The character to draw. For anything the fly spoke, that is its answer, not
   the page's character - this display is the fly's output, not the source. */
function txChar(r) {
  if (r.skipped) return r.char;
  return r.predicted === undefined || r.predicted === null ? ' ' : r.predicted;
}

function txPush(r) {
  if (!r) return;
  if (!$('tx-para')) return;

  const cls = txClass(r);
  TX[cls]++;

  const spoken = !r.skipped;
  const spaced = spoken && r.predicted === ' ';

  TX.out.push({ ch: txChar(r), cls, title: txTitle(r), spoken });

  /* A space ends a word. It is the fly's own space that counts, not the
     page's: if it says a space in the middle of a word, the word is over,
     because that is what it said. */
  if (spaced) {
    TX.words++;
    TX.lastWord = TX.word.join('');
    TX.word.length = 0;
  } else if (spoken) {
    TX.word.push(txChar(r));
  }

  txPaint();
}

function txPaint() {
  if (!TX.out.length) {
    $('tx-word').innerHTML = '<i class="tx-blank">&mdash;</i>';
    $('tx-para').innerHTML = '<i class="tx-blank">nothing read yet</i>';
    $('tx-stat').innerHTML = '';
    return;
  }

  // the word in progress; once it is finished the completed word stays up
  // (dimmed) until the next letter arrives, so a space does not blank the panel
  $('tx-word').innerHTML = TX.word.length
    ? esc(TX.word.join(''))
    : `<i class="tx-blank">${TX.lastWord ? esc(TX.lastWord) : '&mdash;'}</i>`;

  /* Only the tail is kept in the DOM: a book is millions of characters and the
     paragraph is meant to be read, not scrolled through forever. Everything
     below the box is counted over all of it, not over what is on screen. */
  const cut = Math.max(0, TX.out.length - TX_MAX);
  const tail = TX.out.slice(cut);
  const html = (cut ? '<i class="ell">…</i>' : '')
    + tail.map((c, k) => {
      const last = (k === tail.length - 1) ? ' now' : '';
      const body = c.ch === ' ' ? '&nbsp;' : esc(c.ch);
      return `<span class="${c.cls}${last}" title="${esc(c.title)}">${body}</span>`;
    }).join('');

  const box = $('tx-para');
  box.innerHTML = html;
  box.scrollTop = box.scrollHeight;    // follow the fly

  const shown = TX.hit + TX.miss + TX.unknown;
  const pct = shown ? (100 * TX.hit / shown).toFixed(1) : '0.0';
  // words in progress count too, otherwise the tally says 4 while 5 are on screen
  const nWords = TX.words + (TX.word.length ? 1 : 0);
  $('tx-stat').innerHTML =
    `<span class="hit">${TX.hit.toLocaleString()} right</span> &middot; `
    + `<span class="miss">${TX.miss.toLocaleString()} wrong</span> &middot; `
    + `<span class="unk">${TX.unknown.toLocaleString()} unanswerable</span> &middot; `
    + `<span class="un">${TX.skip.toLocaleString()} stepped over</span> &middot; `
    + `${nWords.toLocaleString()} word(s)`
    + `<br>of the ${shown.toLocaleString()} characters it could actually be `
    + `shown, it read <b>${pct}%</b> correctly`;
}

function txReset() {
  TX.out.length = 0;
  TX.word.length = 0;
  TX.lastWord = '';
  TX.hit = TX.miss = TX.unknown = TX.skip = TX.words = 0;
  txPaint();
}

/* --------------------------------------------------------- the digit control */

/* Present one digit. `known` is false, so the readout says so plainly and the
   reward is bitter - the fly is not being graded on something it knows. */
async function presentDigit(d) {
  try {
    const r = await api('/api/present', { char: d, learn: state.learn });
    state.last = r;
    render(r);
    txPush(r);
    document.querySelectorAll('#digits button').forEach(
      b => b.classList.toggle('on', b.dataset.digit === d));
  } catch (e) {
    log(`error: ${e.message}`);
  }
}

/* Show the fly all ten digits and print the whole confusion table. This is the
   reproducibility artifact for "the fly does not know digits": the numbers came
   from live forward passes on the loaded checkpoint, not from this file. */
async function runDigitProof() {
  setBusy(true, 25);
  log('running the digit test on all 10 digits…');
  try {
    const r = await api('/api/digit_proof', { learn: state.learn });
    state.proof = r;
    drawProof(r);
    log(`digit test: ${r.n_correct}/${r.n} correct, `
      + `alphabet control ${(r.alphabet_accuracy * 100).toFixed(1)}%`);
  } catch (e) {
    log(`error: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

function drawProof(r) {
  /* Every number below comes from the server's own forward passes on the loaded
     checkpoint - including the letters (r.margin_letters), which used to be a
     hardcoded 0.37 typed into this file. Nothing on this panel is asserted. */
  const ties = (r.near_ties !== undefined)
    ? r.near_ties : r.digits.filter(d => d.margin < 0.05).length;
  const collapse = (r.margin_letters > 0)
    ? (r.margin_digits / r.margin_letters)
    : 0;

  $('verdict').innerHTML = `
    digits recognised: <b>${r.n_correct} / ${r.n}</b>
    &nbsp;·&nbsp; alphabet control:
    <span class="ok">${r.alphabet_hits}/${r.alphabet_presented}</span>
    (${(r.alphabet_accuracy * 100).toFixed(1)}% on holdout)<br>
    every answer above is one of the <b>${r.n_classes}</b> output letters, because
    those are the only cells there are. chance is
    ${(r.chance * 100).toFixed(1)}%, and the fly did worse than that on digits -
    it cannot do anything else.<br>
    mean decision margin fell from <code>${r.margin_letters.toFixed(3)}</code> on
    letters (min <code>${r.margin_letters_min.toFixed(3)}</code>) to
    <code>${r.margin_digits.toFixed(3)}</code> here (min
    <code>${r.margin_digits_min.toFixed(3)}</code>) &mdash; a
    <code>${collapse.toFixed(2)}&times;</code> collapse, with ${ties}/${r.n}
    near-ties. the fly is not confidently wrong, it is <i>confused</i>, which is
    what an out-of-vocabulary pattern looks like.<br>
    epoch ${r.epoch}, ${r.looks} looks/answer, plasticity
    ${r.learn ? 'on (weights were updated)' : 'off (frozen — pure inference)'}.`;

  const rows = r.digits.map(d => {
    const ink = d.bitmap.join('').split('#').length - 1;
    const tie = d.margin < 0.05;
    return `<tr class="${tie ? 'tie' : ''}">
      <td class="digit">${d.digit}</td>
      <td class="num">${ink}/35</td>
      <td class="guess">${d.guess === ' ' ? '␣' : d.guess}</td>
      <td class="num">${d.margin.toFixed(3)}${tie ? ' <span class="tie-mark">tie</span>' : ''}</td>
      <td class="num">${d.runner}</td>
    </tr>`;
  }).join('');

  $('proof').innerHTML = `
    <table class="proof">
      <thead><tr>
        <th>digit</th><th>ink</th><th>fly said</th><th>margin</th><th>runner-up</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/* ---------------------------------------------------------------- documents */

/* The fly reads a document the only way it can: one character at a time, in
   order, as if each one had been pressed on the button grid. Everything below
   decides *which* character is next; /api/present does the rest, so the brain
   panel animates for every character exactly as it does for a button.

   A character with no 5x7 glyph is never sent to the server. It is recorded as
   stepped over. That is not an optimisation - it is the point. Presenting
   punctuation would either throw or get an answer, and an answer is a lie. */

const DOC = {
  stream: '',
  pages: [],
  meta: null,
  i: 0,
  running: false,
  stopAt: null,
  tally: { hit: 0, miss: 0, unknown: 0, skip: 0 },
};

/* Milliseconds between characters. The last two are "as fast as the server
   answers" - one step is a full forward pass, so that is the real ceiling. */
const DOC_DELAY = [1500, 800, 400, 200, 90, 30, 0];
const DOC_FEED = 240;

function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function b64encode(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  bytes.forEach(b => { bin += String.fromCharCode(b); });
  return btoa(bin);
}

function readFileAsB64(file) {
  return new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(String(fr.result).split(',', 2)[1] || '');
    fr.onerror = () => rej(new Error('the browser could not read that file'));
    fr.readAsDataURL(file);
  });
}

/* Binary search, because this runs once per character and a 5000-page book
   would make a linear scan from the top O(n) per step. */
function pageAt(i) {
  const p = DOC.pages;
  if (!p.length) return null;
  let lo = 0, hi = p.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (p[mid].start <= i) lo = mid; else hi = mid - 1;
  }
  return { page: p[lo], index: lo };
}

/* Escape the readable runs, wrap the unreadable ones. Doing it in this order
   matters: escaping first would turn "&" into "&amp;" and then mark the "&" of
   the entity as an unreadable character. */
function markChunk(s) {
  let out = '', buf = '';
  for (const ch of s) {
    if (/[A-Za-z0-9 ]/.test(ch)) {
      buf += ch;
    } else {
      if (buf) { out += esc(buf); buf = ''; }
      out += `<span class="un">${esc(ch)}</span>`;
    }
  }
  return out + esc(buf);
}

function docViewer() {
  const el = $('doc-viewer');
  const at = pageAt(DOC.i);
  if (!DOC.stream || !at) { el.textContent = ''; return; }

  const p = at.page;
  const WIN = 4000;
  let from = Math.max(p.start, DOC.i - (WIN >> 1));
  let to = Math.min(p.end, from + WIN);
  from = Math.max(p.start, to - WIN);

  const before = markChunk(DOC.stream.slice(from, DOC.i));
  const after = markChunk(DOC.stream.slice(DOC.i + 1, to));
  const cur = DOC.stream.slice(DOC.i, DOC.i + 1);
  const curHtml = (cur === '')
    ? ''
    : `<span class="cur${cur === ' ' ? ' blank' : ''}">${cur === ' ' ? '' : esc(cur)}</span>`;

  el.innerHTML = (from > p.start ? '<i class="ell">…</i>' : '') + before + curHtml
    + after + (to < p.end ? '<i class="ell">…</i>' : '');

  const mark = el.querySelector('.cur');
  if (mark) {
    const top = mark.offsetTop - el.clientHeight / 2 + mark.offsetHeight / 2;
    if (Math.abs(el.scrollTop - top) > el.clientHeight / 2) {
      el.scrollTop = Math.max(0, top);
    }
  }
}

function docWhere(r) {
  const at = pageAt(DOC.i);
  if (!at) return;
  const pct = DOC.stream.length ? (100 * DOC.i / DOC.stream.length) : 0;
  let html = `<b>${esc(at.page.label)}</b> &mdash; page ${at.index + 1} of
    ${DOC.pages.length} &middot; character ${DOC.i.toLocaleString()} of
    ${DOC.stream.length.toLocaleString()} (${pct.toFixed(1)}%)`;

  if (r) {
    const label = r.char === ' ' ? '␣' : esc(r.char);
    let verdict;
    if (r.skipped) {
      verdict = '<span class="un">not shown &mdash; the font has no glyph for it</span>';
    } else if (r.known === false) {
      verdict = `<span class="unk">shown, but the fly has no '${esc(r.char)}' to
        answer with &mdash; it said '${esc(r.predicted)}' instead</span>`;
    } else if (r.correct) {
      verdict = '<span class="hit">read correctly</span>';
    } else {
      verdict = `<span class="miss">misread as '${esc(r.predicted)}'</span>`;
    }
    html += `<br>reading <code>${label}</code> &rarr; ${verdict}`;
  }
  $('doc-where').innerHTML = html;
}

function docTally() {
  const t = DOC.tally;
  const shown = t.hit + t.miss + t.unknown;
  const pct = shown ? (100 * t.hit / shown).toFixed(1) : '0.0';
  $('doc-tally').innerHTML = `
    <span class="hit">${t.hit.toLocaleString()} read right</span> &middot;
    <span class="miss">${t.miss.toLocaleString()} misread</span> &middot;
    <span class="unk">${t.unknown.toLocaleString()} unanswerable</span> &middot;
    <span class="un">${t.skip.toLocaleString()} stepped over</span>
    <br>of the ${shown.toLocaleString()} characters it could actually be shown,
    it read <b>${pct}%</b> correctly.`;
}

function docFeedPush(r) {
  const el = $('doc-feed');
  let cls, title;
  if (r.skipped) {
    cls = 'skip';
    title = 'no glyph in the 5x7 font - never presented';
  } else if (r.known === false) {
    cls = 'unknown';
    title = `presented, but not in the 27-cell vocabulary; it said "${r.predicted}"`;
  } else if (r.correct) {
    cls = 'hit';
    title = 'read correctly';
  } else {
    cls = 'miss';
    title = `misread as "${r.predicted}"`;
  }
  const d = document.createElement('span');
  d.className = 'cell ' + cls;
  d.textContent = r.char === ' ' ? '␣' : r.char;
  d.title = title;
  el.prepend(d);
  while (el.children.length > DOC_FEED) el.lastChild.remove();
}

/* One character. Returns false if the step could not be taken, which stops the
   pump rather than spinning on a broken connection. */
async function docStep() {
  if (!DOC.stream || DOC.i >= DOC.stream.length) return false;

  const raw = DOC.stream[DOC.i];
  const ch = raw.toUpperCase();
  const presentable = !!state.glyphs[ch] && ch !== '\n' && ch !== '\t';

  let rec = { char: raw, skipped: true };
  if (presentable) {
    try {
      rec = await api('/api/present', { char: ch, learn: state.learn });
      state.last = rec;
      render(rec);
    } catch (e) {
      log(`document stopped at character ${DOC.i}: ${e.message}`);
      return false;
    }
  }

  DOC.i++;
  if (rec.skipped) DOC.tally.skip++;
  else if (rec.known === false) DOC.tally.unknown++;
  else if (rec.correct) DOC.tally.hit++;
  else DOC.tally.miss++;

  /* A character with no glyph was never shown, so it is NOT counted as read -
     but it still goes into the running text, greyed and struck through, rather
     than vanishing. Otherwise the paragraph would silently lose the commas and
     full stops and stop looking like the page it came from. */
  txPush(rec);
  docFeedPush(rec);
  docViewer();
  docWhere(rec);
  docTally();
  $('doc-bar').style.width = `${100 * DOC.i / DOC.stream.length}%`;
  return true;
}

function setDocRunLabel() {
  const b = $('btn-doc-run');
  if (!b) return;
  b.textContent = DOC.running ? 'pause' : 'read';
  b.classList.toggle('running', DOC.running);
}

async function docPump() {
  try {
    while (DOC.running) {
      if (DOC.i >= DOC.stream.length) {
        log('end of document');
        break;
      }
      if (DOC.stopAt !== null && DOC.i >= DOC.stopAt) break;
      if (!(await docStep())) break;
      const d = DOC_DELAY[+$('doc-speed').value] || 0;
      if (d) await new Promise(r => setTimeout(r, d));
    }
  } finally {
    DOC.running = false;
    DOC.stopAt = null;
    setDocRunLabel();
  }
}

function docRunTo(kind) {
  if (!DOC.stream) { log('load a document first'); return; }
  if (DOC.i >= DOC.stream.length) docRewind();
  if (kind === 'word') {
    const sp = DOC.stream.indexOf(' ', DOC.i);
    DOC.stopAt = sp === -1 ? DOC.stream.length : sp + 1;
  } else {
    const at = pageAt(DOC.i);
    DOC.stopAt = at ? at.page.end : DOC.stream.length;
  }
  DOC.running = true;
  setDocRunLabel();
  docPump();
}

function toggleDocRun() {
  if (!DOC.stream) { log('load a document first'); return; }
  if (DOC.running) { DOC.running = false; setDocRunLabel(); return; }
  if (DOC.i >= DOC.stream.length) docRewind();
  DOC.stopAt = null;
  DOC.running = true;
  setDocRunLabel();
  docPump();
}

function docRewind() {
  DOC.running = false;
  DOC.stopAt = null;
  DOC.i = 0;
  DOC.tally = { hit: 0, miss: 0, unknown: 0, skip: 0 };
  $('doc-feed').innerHTML = '';
  // a rewind starts a fresh reading, so the running text starts fresh too -
  // otherwise the paragraph would be two passes glued together
  txReset();
  $('doc-bar').style.width = '0%';
  docViewer();
  docWhere(null);
  docTally();
  setDocRunLabel();
}

function docMeta(d) {
  const c = d.counts;
  const un = d.unreadable_chars.map(ch => `<code>${esc(ch)}</code>`).join(' ');
  $('doc-meta').innerHTML = `
    <div class="doc-card">
      <b>${esc(d.title)}</b> <span class="tag">${esc(d.kind)}</span><br>
      ${d.chars.toLocaleString()} characters &middot; ${d.pages.length} page(s)<br>
      letters <b>${c.letter.toLocaleString()}</b> &middot;
      digits <b>${c.digit.toLocaleString()}</b> &middot;
      space <b>${c.space.toLocaleString()}</b> &middot;
      <span class="un">no glyph ${d.n_unreadable.toLocaleString()}</span>
      ${(un && d.n_unreadable) ? `<br><span class="un">no glyph for:</span> ${un}` : ''}
      ${d.folded ? '<br><i>every character with no glyph was folded to a space</i>' : ''}
    </div>`;
}

function docSpeedNote() {
  const d = DOC_DELAY[+$('doc-speed').value] || 0;
  $('doc-speed-note').textContent = d ? `${d} ms/letter` : 'as fast as it can';
}

async function docLoad(name, b64) {
  $('doc-meta').innerHTML = `<div class="doc-card">parsing ${esc(name)}…</div>`;
  log(`parsing ${name}…`);
  try {
    const d = await api('/api/document',
      { name, data: b64, fold: $('doc-fold').checked });
    DOC.stream = d.stream;
    DOC.pages = d.pages;
    DOC.meta = d;
    $('doc-controls').hidden = false;
    docMeta(d);
    docRewind();
    setTab('document');
    log(`${d.title} (${d.kind}): ${d.chars.toLocaleString()} characters, `
      + `${d.pages.length} page(s), ${d.n_unreadable.toLocaleString()} with no glyph`);
    if (d.n_unreadable && !d.folded) {
      log(`${d.n_unreadable.toLocaleString()} characters have no glyph and will be stepped over`);
    }
  } catch (e) {
    $('doc-meta').innerHTML = `<div class="doc-error">${esc(e.message)}</div>`;
    log(`could not read ${name}: ${e.message}`);
  }
}

function setTab(t) {
  state.tab = t;
  $('pane-letters').hidden = t !== 'letters';
  $('pane-document').hidden = t !== 'document';
  $('tab-letters').classList.toggle('on', t === 'letters');
  $('tab-document').classList.toggle('on', t === 'document');
}

function wireDocument() {
  $('tab-letters').onclick = () => setTab('letters');
  $('tab-document').onclick = () => setTab('document');

  // The running text is shared by both tabs: it follows whatever the fly reads,
  // whether that is one letter button, a typed phrase, or a page of a book.
  $('btn-tx-clear').onclick = () => {
    txReset();
    log('cleared the running text');
  };

  const drop = $('doc-drop');
  const file = $('doc-file');
  file.onchange = async e => {
    const f = e.target.files && e.target.files[0];
    if (f) docLoad(f.name, await readFileAsB64(f));
  };
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.remove('over');
  }));
  drop.addEventListener('drop', async e => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) docLoad(f.name, await readFileAsB64(f));
  });

  $('btn-doc-paste').onclick = () => {
    const text = $('doc-paste').value;
    if (!text.trim()) { log('nothing to read'); return; }
    docLoad('pasted.txt', b64encode(text));
  };
  $('doc-paste').addEventListener('keydown', e => {
    if (e.key === 'Enter') $('btn-doc-paste').click();
  });

  $('btn-doc-run').onclick = toggleDocRun;
  $('btn-doc-step').onclick = async () => {
    DOC.running = false; setDocRunLabel();
    await docStep();
  };
  $('btn-doc-word').onclick = () => docRunTo('word');
  $('btn-doc-page').onclick = () => docRunTo('page');
  $('btn-doc-rewind').onclick = () => { log('rewound to the start'); docRewind(); };
  $('doc-speed').oninput = docSpeedNote;
  docSpeedNote();

  $('doc-fold').onchange = () => log($('doc-fold').checked
    ? 'characters with no glyph will be folded to spaces — reload the file to apply'
    : 'characters with no glyph will be stepped over — reload the file to apply');
}

/* --------------------------------------------------------------------- init */

function buildLetters() {
  const el = $('letters');
  state.classes.forEach(c => {
    const b = document.createElement('button');
    b.textContent = c === ' ' ? '␣' : c;
    b.title = `present ${c === ' ' ? 'space' : c}`;
    b.onclick = () => present(c);
    if (c === ' ') b.classList.add('hot');
    el.appendChild(b);
  });
}

/* The ten digits. Deliberately outside the `letters` grid: they are a different
   kind of thing, and the CSS marks them as such so the distinction is visible
   before anything is clicked. */
function buildDigits() {
  const el = $('digits');
  if (!el) return;
  (state.digits.length ? state.digits : ['0','1','2','3','4','5','6','7','8','9'])
    .forEach(d => {
      const b = document.createElement('button');
      b.textContent = d;
      b.dataset.digit = d;
      b.title = `present ${d} — the fly has no '${d}' to answer with`;
      b.onclick = () => presentDigit(d);
      el.appendChild(b);
    });
}

function clearPanels() {
  blank($('cv-rec'), HEIGHTS.rec);
  blank($('cv-pn'), HEIGHTS.pn);
  blank($('cv-kc'), HEIGHTS.kc);
  blank($('cv-mbon'), HEIGHTS.mbon);
}

async function init() {
  const s = await api('/api/state');
  state.classes = s.classes;
  state.digits = s.digits || [];
  state.glyphs = s.glyphs;
  state.accuracy = s.accuracy * 100;
  buildLetters();
  buildDigits();
  wireDocument();
  setTab('letters');

  if (window.Fly3D) {
    state.fly = window.Fly3D.mount($('cv-fly3d'), { hud: $('brain-hud') });
  }
  if (window.BrainView) {
    state.brain = window.BrainView.mount($('cv-brain'));
    state.brain.setClasses(state.classes);
  }

  clearPanels();
  drawDopa(null, '');
  drawCurve(s.history.map(h => h.accuracy));
  $('acc-badge').textContent = `accuracy ${(s.accuracy * 100).toFixed(1)}%`;
  log(`${s.plastic_synapses.toLocaleString()} plastic KC→MBON synapses ready`);
  log(`${s.n_classes || state.classes.length} output cells — the fly's whole vocabulary`);
  log(`${state.digits.length} digits exist as input only; no output cell for them`);
  if (s.history.length) log(`${s.history.length} epochs of prior training loaded`);
  drawWeights((await api('/api/weights')).weights);

  $('learn-toggle').onchange = e => {
    state.learn = e.target.checked;
    $('learn-toggle').nextElementSibling.textContent =
      state.learn ? 'plasticity on' : 'plasticity off (inference)';
    log(state.learn ? 'dopamine will now modify synapses' : 'frozen: no learning');
  };
  document.querySelectorAll('button[data-epochs]').forEach(b => {
    b.onclick = () => train(parseInt(b.dataset.epochs, 10));
  });
  $('btn-say').onclick = say;
  if ($('btn-proof')) $('btn-proof').onclick = runDigitProof;
  $('btn-reset').onclick = async () => {
    setBusy(true);
    try {
      const s2 = await api('/api/reset', {});
      state.accuracy = s2.accuracy * 100;
      drawCurve([]); clearPanels(); drawDopa(null, '');
      if (state.brain) state.brain.clear();
      $('proof').innerHTML = '';
      $('verdict').innerHTML = '';
      state.proof = null;
      txReset();
      drawWeights((await api('/api/weights')).weights);
      log('circuit reset to a naive fly');
    } finally { setBusy(false); }
  };

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT') return;
    // The document tab owns the arrow keys and space, so that reading a book
    // letter by letter is not fighting the single-letter shortcuts.
    if (state.tab === 'document') {
      if (e.key === 'ArrowRight') { e.preventDefault(); docStep(); }
      else if (e.key === ' ') { e.preventDefault(); toggleDocRun(); }
      return;
    }
    const k = e.key.toUpperCase();
    if (state.classes.includes(k)) present(k);
    else if (/^[0-9]$/.test(k) && state.digits.includes(k)) presentDigit(k);
  });

  let t;
  window.addEventListener('resize', () => {
    clearTimeout(t);
    t = setTimeout(() => { if (state.last) render(state.last); }, 150);
  });

  log('ready — click a letter, type a letter, or press “say it”');
}

init().catch(e => log(`fatal: ${e.message}`));
