/* Attention, explained slowly — interactive explainer.
   One consistent toy model is used everywhere: 4 tokens, 2-D queries/keys,
   values drawn as colours so "weighted average of values" is literally visible. */

const TOKENS = [
  { word: 'the',    emb: [0.1, -0.2, 0.3, 0.0],  q: [0.2, 0.1],   k: [0.2, -0.4], color: [154, 164, 183] },
  { word: 'fluffy', emb: [0.8, 0.5, -0.1, 0.6],  q: [1.3, -0.1],  k: [1.2, 0.6],  color: [192, 132, 252] },
  { word: 'cat',    emb: [0.9, -0.4, 0.7, 0.2],  q: [0.6, 0.8],   k: [1.4, -0.3], color: [251, 146, 60]  },
  { word: 'slept',  emb: [-0.3, 0.6, 0.2, -0.5], q: [1.5, -0.2],  k: [0.3, 1.1],  color: [96, 165, 250]  },
];
const FOCUS = 3; // "slept"
const SQRT_DK = Math.SQRT2; // dk = 2

const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const rgb = (c) => `rgb(${c.map(Math.round).join(',')})`;
const fmt = (v) => (v >= 0 ? v.toFixed(2) : '−' + Math.abs(v).toFixed(2));

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, v) => a + v, 0);
  return exps.map((e) => e / sum);
}

function blend(weights) {
  const out = [0, 0, 0];
  weights.forEach((w, i) => TOKENS[i].color.forEach((c, ch) => (out[ch] += w * c)));
  return out;
}

const rawScores = TOKENS.map((t) => dot(TOKENS[FOCUS].q, t.k));
const scaledScores = rawScores.map((s) => s / SQRT_DK);
const weights = softmax(scaledScores);

/* ===================== Stepper ===================== */
(function stepper() {
  const grid = document.getElementById('stageGrid');
  const stage = document.getElementById('stage');
  const N = TOKENS.length;

  function rowLabel(text, cls) {
    const el = document.createElement('div');
    el.className = `row-label ${cls}`;
    el.textContent = text;
    grid.appendChild(el);
  }
  function cell(cls) {
    const el = document.createElement('div');
    el.className = `cell ${cls}`;
    grid.appendChild(el);
    return el;
  }

  // Row: words
  rowLabel('word', 'r1');
  TOKENS.forEach((t, i) => {
    const c = cell('r1');
    const chip = document.createElement('span');
    chip.className = 'word-chip' + (i === FOCUS ? ' focus' : '');
    chip.textContent = t.word;
    c.appendChild(chip);
  });

  // Row: embeddings
  rowLabel('embedding', 'r1');
  TOKENS.forEach((t) => {
    const c = cell('r1');
    const bars = document.createElement('div');
    bars.className = 'emb-bars';
    for (const v of t.emb) {
      const bar = document.createElement('i');
      bar.style.height = `${6 + Math.abs(v) * 30}px`;
      bar.style.background = v >= 0 ? '#6090ff' : '#ff6b72';
      bars.appendChild(bar);
    }
    c.appendChild(bars);
  });

  // Row: q / k / v
  rowLabel('q · k · v', 'r2');
  TOKENS.forEach((t, i) => {
    const c = cell('r2');
    const wrap = document.createElement('div');
    wrap.className = 'qkv-mini';
    const mq = document.createElement('span');
    mq.className = 'mini-chip mq';
    mq.textContent = `q ${fmt(t.q[0])}, ${fmt(t.q[1])}`;
    if (i === FOCUS) mq.style.boxShadow = '0 0 0 2px rgba(255,209,102,0.25)';
    const mk = document.createElement('span');
    mk.className = 'mini-chip mk';
    mk.textContent = `k ${fmt(t.k[0])}, ${fmt(t.k[1])}`;
    const mv = document.createElement('span');
    mv.className = 'mini-chip mv';
    mv.textContent = 'v';
    mv.style.background = rgb(t.color);
    wrap.append(mq, mk, mv);
    c.appendChild(wrap);
  });

  // Row: scores (text swaps between raw and scaled at step 4)
  rowLabel('score', 'r3');
  const scoreBadges = TOKENS.map(() => {
    const c = cell('r3');
    const b = document.createElement('div');
    b.className = 'score-badge';
    c.appendChild(b);
    return b;
  });

  // Row: softmax weights
  rowLabel('weight', 'r4');
  TOKENS.forEach((t, i) => {
    const c = cell('r4');
    const wrap = document.createElement('div');
    wrap.className = 'weight-cell';
    const track = document.createElement('div');
    track.className = 'weight-track';
    const fill = document.createElement('i');
    fill.style.width = `${weights[i] * 100}%`;
    track.appendChild(fill);
    const pct = document.createElement('span');
    pct.className = 'weight-pct';
    pct.textContent = `${(weights[i] * 100).toFixed(0)}%`;
    wrap.append(track, pct);
    c.appendChild(wrap);
  });

  // Output row (step 6)
  const out = document.getElementById('stageOutput');
  const swatch = document.createElement('span');
  swatch.className = 'out-swatch';
  swatch.style.background = rgb(blend(weights));
  const txt = document.createElement('span');
  txt.innerHTML = `new vector for <b>“slept”</b> &nbsp;=&nbsp; ${TOKENS
    .map((t, i) => `<b>${(weights[i] * 100).toFixed(0)}%</b>·<span style="color:${rgb(t.color)}">v<sub>${t.word}</sub></span>`)
    .join(' + ')}`;
  out.append(swatch, txt);

  const STEPS = [
    {
      title: 'Words become lists of numbers',
      html: `Before anything else, each word is replaced by its <b>embedding</b>: a list of numbers looked up in a big learned table. Words with related meanings get similar lists. Real models use 768+ numbers per word; our toy uses 4, drawn as bars (blue = positive, red = negative). <i>That's the entire step — words in, numbers out.</i>
      <span class="why"><b>Why it matters:</b> everything that follows is plain arithmetic on these numbers. There is no “understanding” hidden anywhere else.</span>`,
    },
    {
      title: 'Each word gets its three role vectors',
      html: `The embedding of each word is multiplied by three learned matrices — \\(W_Q\\), \\(W_K\\), \\(W_V\\) — producing the <b style="color:var(--warn)">query</b>, <b style="color:var(--success)">key</b>, and <b style="color:#bcaaff">value</b> for that word. In our toy model, queries and keys are 2-dimensional (two numbers each, shown in the chips), and we draw each <b>value</b> as a colour — so that later, when values get mixed, you can literally see the mixing.
      <span class="why"><b>Why it matters:</b> the matrices are the only thing training changes. Learning “how to attend” means learning good \\(W_Q\\), \\(W_K\\), \\(W_V\\).</span>`,
    },
    {
      title: '“slept” compares its query with every key',
      html: `Now the word <b style="color:var(--warn)">“slept”</b> asks its question. Its query \\(q = (1.50, -0.20)\\) is compared with <b>every</b> key — including its own — using a dot product: multiply the first numbers together, multiply the second numbers together, add. For “cat”: \\(1.50 \\times 1.40 + (-0.20) \\times (-0.30) = 2.16\\). That's the whole computation. Higher score = better match between what “slept” seeks and what that word offers.
      <span class="why"><b>Why a dot product?</b> It is large when two vectors point the same way, near zero when unrelated, negative when opposed — a cheap, differentiable “compatibility meter”. (You can play with this geometrically in the next section.)</span>`,
    },
    {
      title: 'Divide every score by √dₖ — a small but crucial fix',
      html: `Each score is divided by \\(\\sqrt{d_k}\\), the square root of the key length. Here \\(d_k = 2\\), so we divide by \\(\\sqrt{2} \\approx 1.41\\): the “cat” score drops from 2.16 to 1.53. Boring? Yes — but skip it and big models break.
      <span class="why"><b>Why:</b> with hundreds of dimensions, dot products of random vectors grow large simply because many terms are summed. Huge scores push the next step (softmax) into saturation — one weight ≈ 100%, all others ≈ 0% — and the gradients needed for learning vanish. Dividing by \\(\\sqrt{d_k}\\) keeps the scores in a healthy range at any size.</span>`,
    },
    {
      title: 'Softmax: scores become shares of a 100% budget',
      html: `The scaled scores are still arbitrary numbers. <b>Softmax</b> converts them into percentages: exponentiate each score (making everything positive, and gaps more dramatic), then divide by the total. Result: “slept” spends <b>45%</b> of its attention on “cat”, <b>32%</b> on “fluffy”, and little on “the” or itself. The four weights always sum to exactly 100%.
      <span class="why"><b>Why softmax and not the raw scores?</b> Percentages are comparable across words and layers, never negative, and emphasise the strongest matches while never fully silencing the rest — every word keeps at least a whisper of influence.</span>`,
    },
    {
      title: 'Blend the values — the payoff',
      html: `The finale: multiply each word's <b style="color:#bcaaff">value</b> by its weight and add everything up. Watch the output swatch: it is a warm blend dominated by <span style="color:rgb(251,146,60)"><b>cat-orange</b></span> (44%) and <span style="color:rgb(192,132,252)"><b>fluffy-purple</b></span> (32%) — the new representation of “slept” now literally <i>contains</i> information about who slept. That enriched vector is what the next layer receives.
      <span class="why"><b>The big picture:</b> this exact six-step dance runs for every word at once, in every head, in every layer. That's all attention is: ask (Q), match (K), mix (V).</span>`,
    },
  ];

  const dots = document.getElementById('stepDots');
  STEPS.forEach(() => {
    const d = document.createElement('span');
    d.className = 'step-dot';
    dots.appendChild(d);
  });

  const titleEl = document.getElementById('stepTitle');
  const explainEl = document.getElementById('stepExplain');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  const urlStep = parseInt(new URLSearchParams(location.search).get('step'), 10);
  let step = urlStep >= 1 && urlStep <= STEPS.length ? urlStep : 1;

  function render() {
    stage.className = 'stage ' + Array.from({ length: step }, (_, i) => `ge${i + 1}`).join(' ');
    titleEl.innerHTML = `<small>Step ${step} of ${STEPS.length}</small>${STEPS[step - 1].title}`;
    explainEl.innerHTML = STEPS[step - 1].html;
    scoreBadges.forEach((b, i) => {
      const showScaled = step >= 4;
      const v = showScaled ? scaledScores[i] : rawScores[i];
      b.innerHTML = `${fmt(v)}<small>${showScaled ? 'q·k ÷ √2' : 'q · k'}</small>`;
      b.style.borderColor = i === 2 && step >= 3 ? 'rgba(251,146,60,0.6)' : '';
    });
    [...dots.children].forEach((d, i) => {
      d.className = 'step-dot' + (i + 1 < step ? ' done' : i + 1 === step ? ' current' : '');
    });
    prevBtn.disabled = step === 1;
    nextBtn.disabled = step === STEPS.length;
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([titleEl, explainEl]);
  }

  prevBtn.addEventListener('click', () => { if (step > 1) { step--; render(); } });
  nextBtn.addEventListener('click', () => { if (step < STEPS.length) { step++; render(); } });
  render();
})();

/* ===================== Playground: draggable query ===================== */
(function playground() {
  const svg = document.getElementById('plane');
  const NS = 'http://www.w3.org/2000/svg';
  const C = 180, SCALE = 75; // centre and px-per-unit
  const toPx = (v) => [C + v[0] * SCALE, C - v[1] * SCALE];
  const toVec = (x, y) => [(x - C) / SCALE, (C - y) / SCALE];

  function el(tag, attrs, parent = svg) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    parent.appendChild(e);
    return e;
  }

  // grid + axes
  for (let u = -2; u <= 2; u++) {
    const p = C + u * SCALE;
    if (p < 0 || p > 360) continue;
    el('line', { x1: p, y1: 0, x2: p, y2: 360, stroke: '#1a2235', 'stroke-width': 1 });
    el('line', { x1: 0, y1: p, x2: 360, y2: p, stroke: '#1a2235', 'stroke-width': 1 });
  }
  el('line', { x1: 0, y1: C, x2: 360, y2: C, stroke: '#263149', 'stroke-width': 1.4 });
  el('line', { x1: C, y1: 0, x2: C, y2: 360, stroke: '#263149', 'stroke-width': 1.4 });

  const defs = el('defs', {});
  function marker(id, color) {
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', fill: color }, m);
  }

  // key arrows
  TOKENS.forEach((t, i) => {
    const color = rgb(t.color);
    marker(`mk${i}`, color);
    const [x, y] = toPx(t.k);
    el('line', { x1: C, y1: C, x2: x, y2: y, stroke: color, 'stroke-width': 2.4, 'marker-end': `url(#mk${i})` });
    el('text', {
      x: x + (t.k[0] >= 0 ? 8 : -8), y: y + (t.k[1] >= 0 ? -8 : 14),
      fill: color, 'font-size': 13, 'font-weight': 700,
      'text-anchor': t.k[0] >= 0 ? 'start' : 'end', 'font-family': 'Inter, sans-serif',
    }).textContent = `k ${t.word}`;
  });

  // query arrow (draggable)
  marker('mq', '#ffd166');
  const qLine = el('line', { x1: C, y1: C, stroke: '#ffd166', 'stroke-width': 3, 'marker-end': 'url(#mq)' });
  const qHandle = el('circle', { r: 13, fill: 'rgba(255,209,102,0.15)', stroke: '#ffd166', 'stroke-width': 1.5, cursor: 'grab' });
  const qLabel = el('text', { fill: '#ffd166', 'font-size': 13.5, 'font-weight': 800, 'font-family': 'Inter, sans-serif' });
  qLabel.textContent = 'q (drag me)';

  // score table
  const tbody = document.querySelector('#scoreTable tbody');
  const rows = TOKENS.map((t) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td style="color:${rgb(t.color)}">${t.word}</td><td class="tdot"></td>` +
      `<td class="tbar"><div class="tbar-track"><i></i></div></td><td class="tpct"></td>`;
    tbody.appendChild(tr);
    return { dot: tr.querySelector('.tdot'), bar: tr.querySelector('.tbar-track i'), pct: tr.querySelector('.tpct') };
  });
  const blendSwatch = document.getElementById('blendSwatch');
  const scaleToggle = document.getElementById('scaleToggle');

  let q = [...TOKENS[FOCUS].q];

  function update() {
    const [x, y] = toPx(q);
    qLine.setAttribute('x2', x);
    qLine.setAttribute('y2', y);
    qHandle.setAttribute('cx', x);
    qHandle.setAttribute('cy', y);
    qLabel.setAttribute('x', x + (q[0] >= 0 ? 14 : -14));
    qLabel.setAttribute('y', y + (q[1] >= 0 ? -12 : 20));
    qLabel.setAttribute('text-anchor', q[0] >= 0 ? 'start' : 'end');

    const scores = TOKENS.map((t) => dot(q, t.k));
    const w = softmax(scaleToggle.checked ? scores.map((s) => s / SQRT_DK) : scores);
    rows.forEach((r, i) => {
      r.dot.textContent = fmt(scores[i]);
      r.bar.style.width = `${w[i] * 100}%`;
      r.pct.textContent = `${(w[i] * 100).toFixed(0)}%`;
    });
    blendSwatch.style.background = rgb(blend(w));
  }

  let dragging = false;
  function pointToVec(ev) {
    const rect = svg.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * 360;
    const y = ((ev.clientY - rect.top) / rect.height) * 360;
    const v = toVec(x, y);
    return [Math.max(-2.1, Math.min(2.1, v[0])), Math.max(-2.1, Math.min(2.1, v[1]))];
  }
  svg.addEventListener('pointerdown', (ev) => {
    dragging = true;
    svg.setPointerCapture(ev.pointerId);
    q = pointToVec(ev);
    update();
  });
  svg.addEventListener('pointermove', (ev) => {
    if (!dragging) return;
    q = pointToVec(ev);
    update();
  });
  svg.addEventListener('pointerup', () => (dragging = false));
  scaleToggle.addEventListener('change', update);
  update();
})();

/* ===================== Full attention matrix ===================== */
(function heatmap() {
  const hm = document.getElementById('heatmap');
  const corner = document.createElement('div');
  corner.className = 'hm-corner';
  corner.textContent = 'query ↓ key →';
  hm.appendChild(corner);
  TOKENS.forEach((t) => {
    const l = document.createElement('div');
    l.className = 'hm-label col';
    l.textContent = t.word;
    hm.appendChild(l);
  });
  TOKENS.forEach((tq) => {
    const l = document.createElement('div');
    l.className = 'hm-label row';
    l.textContent = tq.word;
    hm.appendChild(l);
    const w = softmax(TOKENS.map((tk) => dot(tq.q, tk.k) / SQRT_DK));
    w.forEach((v) => {
      const cell = document.createElement('div');
      cell.className = 'hm-cell';
      cell.style.background = `rgba(96, 144, 255, ${(v * 1.6).toFixed(3)})`;
      cell.style.color = v > 0.3 ? '#fff' : 'var(--muted)';
      cell.textContent = `${(v * 100).toFixed(0)}%`;
      hm.appendChild(cell);
    });
  });
})();

/* ===================== The "it" demo ===================== */
(function itDemo() {
  const VARIANTS = {
    tired: {
      last: 'tired',
      weights: { The: 0.01, animal: 0.58, "didn't": 0.02, cross: 0.05, the: 0.01, street: 0.09, because: 0.03, it: 0, was: 0.02, too: 0.03, tired: 0.14, '.': 0.02 },
      verdict: 'An animal can be <b>tired</b>, a street can’t — so “it” attends to <b>animal</b> (58%).',
    },
    wide: {
      last: 'wide',
      weights: { The: 0.01, animal: 0.10, "didn't": 0.02, cross: 0.06, the: 0.02, street: 0.55, because: 0.03, it: 0, was: 0.02, too: 0.03, wide: 0.15, '.': 0.01 },
      verdict: 'A street can be <b>wide</b>, an animal can’t — so the same mechanism now picks <b>street</b> (55%).',
    },
  };
  const ORDER = ['The', 'animal', "didn't", 'cross', 'the', 'street', 'because', 'it', 'was', 'too', 'LAST', '.'];

  const sentence = document.getElementById('itSentence');
  const verdict = document.getElementById('itVerdict');
  const buttons = document.querySelectorAll('.it-btn');

  function render(name) {
    const v = VARIANTS[name];
    sentence.innerHTML = '';
    for (const slot of ORDER) {
      const word = slot === 'LAST' ? v.last : slot;
      const w = v.weights[word] ?? 0;
      const span = document.createElement('span');
      span.className = 'it-word' + (word === 'it' ? ' query-word' : '') + (w >= 0.05 ? ' show-pct' : '');
      if (word !== 'it') {
        span.style.background = `rgba(96, 144, 255, ${Math.min(0.85, w * 1.45).toFixed(3)})`;
        if (w > 0.3) span.style.fontWeight = '700';
      }
      span.innerHTML = `${word}<span class="pct">${w >= 0.05 ? (w * 100).toFixed(0) + '%' : ''}</span>`;
      sentence.appendChild(span);
    }
    verdict.innerHTML = v.verdict;
    buttons.forEach((b) => b.classList.toggle('active', b.dataset.variant === name));
  }

  buttons.forEach((b) => b.addEventListener('click', () => render(b.dataset.variant)));
  render('tired');
})();

/* ===================== Multi-head minis ===================== */
(function multiHead() {
  const WORDS = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
  const n = WORDS.length;

  function normalizeRows(m) {
    return m.map((row) => {
      const s = row.reduce((a, v) => a + v, 0) || 1;
      return row.map((v) => v / s);
    });
  }

  const HEADS = [
    {
      name: 'Head 1 — “look one word back”',
      desc: 'A positional head: every word attends mostly to its predecessor. Useful for local word order.',
      matrix: normalizeRows(WORDS.map((_, i) => WORDS.map((_, j) => (j === i - 1 ? 1 : j === i ? 0.25 : 0.05)))),
    },
    {
      name: 'Head 2 — “find the nouns”',
      desc: 'A content head: all words route attention to the nouns “cat” and “mat”, wherever they sit.',
      matrix: normalizeRows(WORDS.map(() => WORDS.map((_, j) => (j === 1 || j === 5 ? 1 : 0.12)))),
    },
    {
      name: 'Head 3 — “verb seeks its subject”',
      desc: 'A syntactic head: “sat” binds strongly to “cat” (who sat?) and “mat” (sat where?).',
      matrix: normalizeRows(WORDS.map((_, i) =>
        WORDS.map((_, j) => (i === 2 ? (j === 1 ? 1 : j === 5 ? 0.7 : 0.06) : j === i ? 0.6 : 0.12)))),
    },
  ];

  const row = document.getElementById('headsRow');
  for (const head of HEADS) {
    const card = document.createElement('div');
    card.className = 'head-card';
    const grid = document.createElement('div');
    grid.className = 'head-grid';
    grid.appendChild(Object.assign(document.createElement('span'), { className: 'hg-lab' }));
    for (const w of WORDS) {
      const l = document.createElement('span');
      l.className = 'hg-lab';
      l.textContent = w;
      grid.appendChild(l);
    }
    head.matrix.forEach((r, i) => {
      const l = document.createElement('span');
      l.className = 'hg-lab';
      l.textContent = WORDS[i];
      grid.appendChild(l);
      for (const v of r) {
        const c = document.createElement('span');
        c.className = 'hg-cell';
        c.style.background = `rgba(96, 144, 255, ${Math.min(0.95, v * 1.5).toFixed(3)})`;
        grid.appendChild(c);
      }
    });
    const title = document.createElement('b');
    title.textContent = head.name;
    const desc = document.createElement('small');
    desc.textContent = head.desc;
    card.append(title, grid, desc);
    row.appendChild(card);
  }
})();
