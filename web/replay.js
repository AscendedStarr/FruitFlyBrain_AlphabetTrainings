/* replay.js - serve the FlyBrain UI from a recording instead of a server.
 *
 * web/app.js talks to Python: every panel is filled by `fetch('/api/...')`.
 * GitHub Pages runs no Python, so the published page cannot call a brain. This
 * file makes it call a recording instead.
 *
 * `scripts/make_replay.py` drives the real `Session` from `scripts/serve.py`
 * against the tracked checkpoint and writes every response to `replay.json`.
 * Here we intercept `window.fetch`, match the request against those recorded
 * routes, and return the recorded body. No arithmetic happens in this file.
 *
 * Three deliberate constraints, so the page cannot overclaim:
 *   1. It is inert unless the page sets `window.__FLYBRAIN_REPLAY__`. The local
 *      dev server never sets it, so `python scripts/serve.py` still runs live.
 *   2. Anything not recorded returns a real error object, which `api()` in
 *      app.js already turns into a visible "error: ..." line in the log. It
 *      never falls back to a plausible-looking default.
 *   3. Training, reset and plasticity are refused, because all three write to
 *      the weights and a recording cannot. Their controls are dimmed and their
 *      routes answer with an explanation.
 */
(function () {
  'use strict';

  var REPLAY_URL = window.__FLYBRAIN_REPLAY__;
  if (!REPLAY_URL) return;            // live server: do nothing at all.

  var realFetch = window.fetch.bind(window);

  /* Load the recording once. Every API call awaits this promise, so the UI's
     very first request blocks on the JSON rather than racing it. */
  var ready = realFetch(REPLAY_URL, { cache: 'no-cache' }).then(function (r) {
    if (!r.ok) throw new Error('replay.json ' + r.status);
    return r.json();
  });
  ready.catch(function () { /* surfaced per-request below */ });

  function respond(obj) {
    return new Response(JSON.stringify(obj), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  function note(text) {
    // Show it where the user is already looking, if the log panel exists yet.
    var el = document.getElementById('log');
    if (el) {
      el.textContent += (el.textContent ? '\n' : '') + '· ' + text;
      el.scrollTop = el.scrollHeight;
      return;
    }
    // Otherwise queue it; boot() drains the queue once the panel is up.
    queued.push(text);
  }
  var queued = [];

  /* ------------------------------------------------------------ route match */

  function routeKey(path, body) {
    if (path === '/api/state' || path === '/api/weights') {
      return 'GET ' + path;
    }
    if (path === '/api/digit_proof') return 'POST /api/digit_proof';
    if (path === '/api/present' && body && typeof body.char === 'string') {
      // The server upper-cases before encoding, so the key must too.
      return 'POST /api/present ' + body.char.toUpperCase();
    }
    if (path === '/api/say' && body && typeof body.phrase === 'string') {
      // Session.say upper-cases each character anyway; matching the recorded
      // key case-insensitively mirrors exactly what the server would do.
      return 'POST /api/say ' + body.phrase.trim().toUpperCase();
    }
    return null;
  }

  var WRITES = {
    '/api/train': 'training',
    '/api/reset': 'resetting the brain',
    '/api/document': 'reading a document',
  };

  function refusal(path) {
    return {
      error: (WRITES[path] || 'that') + ' is not available in the published '
        + 'demo: this page replays a recording of the real network, and a '
        + 'recording cannot change the weights. run "python scripts/serve.py" '
        + 'from a clone for the live version.',
    };
  }

  /* -------------------------------------------------------------- the shim */

  window.fetch = function (input, init) {
    var url = typeof input === 'string' ? input
      : (input && input.url) || String(input);
    var path = url.split('?')[0];

    if (path.indexOf('/api/') !== 0) return realFetch(input, init);

    if (WRITES[path]) {
      return Promise.resolve(respond(refusal(path)));
    }

    var body = null;
    if (init && init.body) {
      try { body = JSON.parse(init.body); } catch (_) { body = null; }
    }

    /* The recording only contains `learn=False` responses. Honouring a
       `learn=True` request by returning them would leave the toggle saying
       "plasticity on" while the weights provably did not move, so refuse. */
    if (body && body.learn === true) {
      return Promise.resolve(respond({
        error: 'plasticity is off in the recording: the captured responses '
          + 'were all inference-only (learn=false), so returning them for a '
          + 'learning request would misreport what the network did.',
      }));
    }

    var key = routeKey(path, body);
    return ready.then(function (data) {
      var hit = key === null ? undefined : data.routes[key];
      if (hit === undefined) {
        note('not in the recording: ' + (key || path));
        return {
          error: 'this request was not part of the recording '
            + '(' + (key || path) + '). the static demo replays a fixed set '
            + 'of real responses; anything outside it needs the live server.',
        };
      }
      return hit;
    }).then(respond, function (err) {
      return respond({ error: 'could not load replay.json: ' + err.message });
    });
  };

  /* --------------------------------------------------------------- the UI */

  var CSS =
    '.replay-bar{position:sticky;top:0;z-index:60;display:flex;gap:10px;'
    + 'align-items:baseline;flex-wrap:wrap;padding:9px 14px;'
    + 'background:#14141a;border-bottom:1px solid #2a2a35;'
    + 'font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#b9b9c8}'
    + '.replay-bar b{color:#e8e8f0;font-weight:600}'
    + '.replay-bar .tag{background:#232333;border:1px solid #3a3a4d;'
    + 'border-radius:3px;padding:1px 6px;color:#8fd6a0}'
    + '.replay-bar a{color:#7fc8ff}'
    + '.replay-off{opacity:.4;pointer-events:none!important;'
    + 'filter:grayscale(1)}';

  function boot() {
    var data = null;

    var style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    var bar = document.createElement('div');
    bar.className = 'replay-bar';
    bar.innerHTML = '<b>recording, not a live brain</b>'
      + '<span>every number on this page is a real forward pass through the '
      + 'trained checkpoint, captured once and replayed from '
      + '<code>replay.json</code></span>';
    document.body.insertBefore(bar, document.body.firstChild);

    ready.then(function (d) {
      data = d;
      var m = d.meta;
      bar.innerHTML += '<span class="tag">epoch ' + m.epoch + '</span>'
        + '<span class="tag">' + m.verified.alphabet_hits + '/'
        + m.verified.alphabet_presented + ' letters</span>'
        + '<span class="tag">' + m.verified.digit_correct + '/'
        + m.verified.digit_n + ' digits</span>'
        + '<span class="tag">plasticity off</span>'
        + '<span>checkpoint <code>' + m.checkpoint + '</code> sha256 '
        + '<code>' + m.checkpoint_sha256.slice(0, 12) + '…</code>, '
        + 'recorded ' + m.generated_utc + '</span>'
        + '<span>the phrases that replay: <code>'
        + m.phrases.join('</code>, <code>') + '</code></span>';

      // Drop the queued explanation into the log panel now that it exists.
      if (queued.length) {
        queued.forEach(note);
        queued = [];
      }
      note('replay mode: inference only. plasticity, training, reset and the '
        + 'document reader each need the live server, so they are unavailable '
        + 'here - run python scripts/serve.py for those.');

      // The phrase box still accepts free text, but only the recorded set
      // resolves. Offer them rather than leaving the reader to guess.
      var phrase = document.getElementById('phrase');
      if (phrase) {
        var dl = document.createElement('datalist');
        dl.id = 'replay-phrases';
        m.phrases.forEach(function (p) {
          var o = document.createElement('option');
          o.value = p;
          dl.appendChild(o);
        });
        document.body.appendChild(dl);
        phrase.setAttribute('list', 'replay-phrases');
        phrase.value = m.phrases[0];
        phrase.title = 'only the recorded phrases replay in the static demo';
      }
    }).catch(function (err) {
      bar.innerHTML = '<b style="color:#ff8080">replay.json failed to load</b>'
        + '<span>' + err.message + '</span>';
    });

    /* Dim the controls that a recording genuinely cannot honour. `setBusy()`
       in app.js rewrites `button.disabled` on every request, so this uses a
       class (pointer-events) instead - and the routes above refuse anyway, so
       a keyboard-triggered click still cannot fake a result. */
    var dim = function (el) {
      if (!el) return;
      el.classList.add('replay-off');
      el.title = 'not available in the recording - see the note at the top';
      /* A checkbox inside a <label> still toggles when the label is clicked,
         and `setBusy()` only ever touches `button.disabled`, so disabling the
         input outright is the one form of this that actually sticks. */
      if (el.tagName === 'INPUT') el.disabled = true;
    };
    document.querySelectorAll('button[data-epochs]').forEach(dim);
    dim(document.getElementById('btn-reset'));
    dim(document.getElementById('learn-toggle'));
    dim(document.getElementById('btn-doc-run'));
    dim(document.getElementById('btn-doc-step'));
    dim(document.getElementById('btn-doc-word'));
    dim(document.getElementById('btn-doc-page'));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
