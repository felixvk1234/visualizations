/* ============================================================
   The Transformer, End to End — interactions
   ============================================================ */

const WORDS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
const IDS = [1996, 4937, 2938, 2006, 1996, 13523];

/* ===================== Stage 1: tokenization ===================== */
(() => {
  const wordRow = document.getElementById('tokWords');
  const idRow = document.getElementById('tokIds');
  WORDS.forEach((w, i) => {
    wordRow.insertAdjacentHTML('beforeend', `<span class="tok-chip" data-i="${i}">${w}</span>`);
    idRow.insertAdjacentHTML('beforeend', `<span class="tok-chip id" data-i="${i}">${IDS[i]}</span>`);
  });
  const chips = document.querySelectorAll('.tok-chip');
  let i = 0;
  setInterval(() => {
    chips.forEach(c => c.classList.toggle('hl', +c.dataset.i === i % WORDS.length));
    i++;
  }, 950);
})();

/* ===================== Stage 2: embeddings ===================== */
(() => {
  const DIMS = 8;
  // deterministic pseudo-vector per word; "The" and "the" share a seed
  const seeds = { The: 11, cat: 42, sat: 77, on: 23, the: 11, mat: 58 };
  const row = document.getElementById('embedRow');
  const caption = document.getElementById('embedCaption');
  const toggle = document.getElementById('posToggle');

  const vec = seed => {
    let s = seed;
    return Array.from({ length: DIMS }, () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    });
  };
  const color = v => `hsl(${215 + v * 110}, 70%, ${32 + v * 32}%)`;

  const base = WORDS.map(w => vec(seeds[w]));
  // sinusoidal-ish positional offsets
  const pos = WORDS.map((_, p) =>
    Array.from({ length: DIMS }, (_, d) => 0.5 + 0.45 * Math.sin(p / Math.pow(40, d / DIMS) + d))
  );

  WORDS.forEach((w, i) => {
    const cells = base[i].map(() => `<div class="embed-cell"></div>`).join('');
    row.insertAdjacentHTML('beforeend',
      `<div class="embed-col ${w.toLowerCase() === 'the' ? 'same' : ''}" data-i="${i}">
         <div class="embed-vec">${cells}</div>
         <div class="embed-label">${w}</div>
       </div>`);
  });

  let withPos = false;
  function render() {
    document.querySelectorAll('.embed-col').forEach(col => {
      const i = +col.dataset.i;
      col.querySelectorAll('.embed-cell').forEach((cell, d) => {
        const v = withPos ? (base[i][d] + pos[i][d]) / 2 : base[i][d];
        cell.style.background = color(Math.min(1, Math.max(0, v)));
      });
    });
  }
  render();

  toggle.addEventListener('click', () => {
    withPos = !withPos;
    render();
    toggle.textContent = withPos ? 'Remove positional encoding' : 'Add positional encoding';
    caption.innerHTML = withPos
      ? 'Positions added — the two <b>“the”</b> columns now differ, so the model knows one is word 1 and the other word 5.'
      : 'Each column is one token’s embedding. The two <b>“the”</b> columns are identical — the model can’t tell them apart yet.';
  });
})();

/* ===================== Stage 3: attention arcs ===================== */
(() => {
  const svg = document.getElementById('attnSvg');
  const NS = 'http://www.w3.org/2000/svg';
  const hint = document.getElementById('attnHint');
  // illustrative attention weights [query][key]
  const W = [
    [0.50, 0.30, 0.05, 0.05, 0.05, 0.05],
    [0.20, 0.40, 0.25, 0.03, 0.02, 0.10],
    [0.05, 0.45, 0.15, 0.05, 0.02, 0.28],
    [0.02, 0.08, 0.35, 0.20, 0.05, 0.30],
    [0.05, 0.03, 0.04, 0.08, 0.30, 0.50],
    [0.05, 0.15, 0.25, 0.15, 0.20, 0.20],
  ];
  const HINTS = [
    '“The” attends to itself and “cat” — determiners bind to their noun.',
    '“cat” attends to “sat” — subject linking to its verb.',
    '“sat” attends strongly to “cat” (who sat?) and “mat” (sat where?).',
    '“on” attends to “sat” and “mat” — prepositions connect verb and object.',
    '“the” attends ahead to “mat” — its noun.',
    '“mat” attends back across the clause it completes.',
  ];
  const xs = WORDS.map((_, i) => 80 + i * 108);
  const topY = 42, botY = 205;
  let selected = 2;

  function el(name, attrs, text) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function draw() {
    svg.innerHTML = '';
    W[selected].forEach((w, j) => {
      const x1 = xs[selected], x2 = xs[j];
      svg.appendChild(el('path', {
        d: `M ${x1} ${topY + 17} C ${x1} 125, ${x2} 125, ${x2} ${botY - 17}`,
        fill: 'none',
        stroke: '#ffd166',
        'stroke-width': Math.max(0.8, w * 13),
        'stroke-opacity': Math.max(0.22, Math.min(1, w * 1.7)),
        'stroke-linecap': 'round',
      }));
      if (w >= 0.15) {
        svg.appendChild(el('text', {
          x: x2, y: botY - 26, 'text-anchor': 'middle',
          fill: '#ffd166', 'font-size': 11, 'font-weight': 700,
          'font-family': 'Inter, sans-serif',
        }, w.toFixed(2)));
      }
    });
    WORDS.forEach((word, i) => {
      [[topY, true], [botY, false]].forEach(([y, isTop]) => {
        const sel = isTop && i === selected;
        const g = el('g', isTop ? { class: 'clickable' } : {});
        g.appendChild(el('rect', {
          x: xs[i] - 36, y: y - 17, width: 72, height: 33, rx: 9,
          fill: sel ? 'rgba(96,144,255,0.25)' : '#151c2c',
          stroke: sel ? '#6090ff' : '#263149',
          'stroke-width': 1.5,
        }));
        g.appendChild(el('text', {
          x: xs[i], y: y + 4, 'text-anchor': 'middle',
          fill: sel ? '#b9ccff' : '#eef1f7',
          'font-size': 14, 'font-weight': 700,
          'font-family': 'Inter, sans-serif',
        }, word));
        if (isTop) g.addEventListener('click', () => { selected = i; draw(); });
        svg.appendChild(g);
      });
    });
    svg.appendChild(el('text', {
      x: 8, y: topY + 3, fill: '#9aa4b8', 'font-size': 9.5,
      'font-family': 'Inter, sans-serif', 'font-weight': 700,
    }, 'QUERIES'));
    svg.appendChild(el('text', {
      x: 8, y: botY + 3, fill: '#9aa4b8', 'font-size': 9.5,
      'font-family': 'Inter, sans-serif', 'font-weight': 700,
    }, 'KEYS'));
    hint.innerHTML = HINTS[selected] + ' <b>Weights always sum to 100%.</b>';
  }
  draw();
})();

/* ===================== Stage 4: architecture trace ===================== */
(() => {
  const blocks = document.querySelectorAll('#arch .arch-block');
  const btn = document.getElementById('archPlay');
  let running = false;
  btn.addEventListener('click', () => {
    if (running) return;
    running = true;
    // pass through the repeated block twice to suggest stacking
    const order = [0, 1, 2, 1, 2, 3];
    let i = 0;
    const iv = setInterval(() => {
      blocks.forEach(b => b.classList.remove('active'));
      if (i >= order.length) {
        clearInterval(iv);
        running = false;
        return;
      }
      document.querySelector(`#arch .arch-block[data-step="${order[i]}"]`).classList.add('active');
      i++;
    }, 650);
  });
})();

/* ===================== Stage 5: generation loop ===================== */
(() => {
  const sentenceEl = document.getElementById('genSentence');
  const rowsEl = document.getElementById('probRows');
  const btn = document.getElementById('genBtn');
  const resetBtn = document.getElementById('genReset');
  const caption = document.getElementById('genCaption');

  const PROMPT = ['The', 'cat', 'sat', 'on', 'the'];
  // predetermined prediction rounds: [winner, [word, prob] x5]
  const ROUNDS = [
    ['mat', [['mat', 0.62], ['floor', 0.14], ['couch', 0.09], ['rug', 0.07], ['table', 0.04]]],
    ['and', [['and', 0.38], ['.', 0.27], [',', 0.18], ['next', 0.06], ['while', 0.04]]],
    ['purred', [['purred', 0.41], ['slept', 0.25], ['watched', 0.12], ['yawned', 0.09], ['waited', 0.05]]],
    ['.', [['.', 0.71], ['softly', 0.11], ['loudly', 0.08], [',', 0.05], ['all', 0.03]]],
  ];

  let generated = [];
  let round = 0;
  let busy = false;

  function renderSentence(cursor) {
    sentenceEl.innerHTML =
      PROMPT.map(w => `<span class="gen-word">${w}</span>`).join('') +
      generated.map((w, i) =>
        `<span class="gen-word ${i === generated.length - 1 ? 'new' : ''}">${w}</span>`).join('') +
      (cursor ? '<span class="gen-word cursor">___</span>' : '');
  }

  function renderBars(preds, winner) {
    rowsEl.innerHTML = preds.map(([w, p]) =>
      `<div class="prob-item ${w === winner ? 'winner' : ''}">
         <div class="prob-word">${w}</div>
         <div class="prob-track"><i data-p="${p}"></i></div>
         <div class="prob-val">${Math.round(p * 100)}%</div>
       </div>`).join('');
    requestAnimationFrame(() => requestAnimationFrame(() => {
      rowsEl.querySelectorAll('i').forEach(b => { b.style.width = (b.dataset.p * 100) + '%'; });
    }));
  }

  function reset() {
    generated = [];
    round = 0;
    busy = false;
    btn.disabled = false;
    rowsEl.innerHTML = '';
    renderSentence(true);
    caption.textContent = 'The model predicts a distribution, the most likely token is appended, and the loop starts over.';
  }

  btn.addEventListener('click', () => {
    if (busy || round >= ROUNDS.length) return;
    busy = true;
    const [winner, preds] = ROUNDS[round];
    renderSentence(true);
    renderBars(preds, winner);
    caption.innerHTML = `Forward pass ${round + 1}: the model assigns <b>${Math.round(preds[0][1] * 100)}%</b> to “${winner}” — sampled and appended.`;
    setTimeout(() => {
      generated.push(winner);
      round++;
      const done = round >= ROUNDS.length;
      renderSentence(!done);
      if (done) {
        caption.innerHTML = '<b>“The cat sat on the mat and purred.”</b> — four forward passes, four tokens. That is all text generation is.';
        btn.disabled = true;
      }
      busy = false;
    }, 1400);
  });

  resetBtn.addEventListener('click', reset);
  reset();
})();
