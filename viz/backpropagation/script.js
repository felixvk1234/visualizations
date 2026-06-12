/* ============================================================
   Backpropagation, Explained Simply — interactions
   1. Loss valley with draggable ball (gradient = slope)
   2. Step-through of one real backprop pass on a 2-2-1 net
   3. Live training of a 1-6-1 net fitting a curve
   ============================================================ */

const SVGNS = 'http://www.w3.org/2000/svg';
function el(name, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
const fmt = (v, d = 2) => (v >= 0 ? '+' : '') + v.toFixed(d);

/* ===================== 1. THE LOSS VALLEY ===================== */
(() => {
  const svg = document.getElementById('hillSvg');
  const W0 = 1.1;                       // valley bottom
  const L = w => 0.1 + 0.5 * (w - W0) ** 2;
  const slope = w => (w - W0);
  const WMIN = -2.2, WMAX = 4.4, LMAX = 4.0;

  // svg coords
  const PX = w => 30 + (w - WMIN) / (WMAX - WMIN) * 410;
  const PY = l => 270 - Math.min(l, LMAX) / LMAX * 245;
  const fromPX = px => WMIN + (px - 30) / 410 * (WMAX - WMIN);

  let w = 3.4;
  let lr = 0.8;
  let trail = [];
  let auto = null;

  const roW = document.getElementById('roW');
  const roL = document.getElementById('roL');
  const roG = document.getElementById('roG');
  const roDir = document.getElementById('roDir');
  const cap = document.getElementById('hillCaption');
  const autoBtn = document.getElementById('hillAuto');

  function draw() {
    svg.innerHTML = '';
    // curve
    let d = '';
    for (let px = 30; px <= 440; px += 4) {
      const y = PY(L(fromPX(px)));
      d += (d ? ' L ' : 'M ') + px + ' ' + y;
    }
    svg.appendChild(el('path', { d, fill: 'none', stroke: '#6090ff', 'stroke-width': 2.5 }));
    // axes labels
    svg.appendChild(el('text', { x: 235, y: 293, 'text-anchor': 'middle', fill: '#9aa4b8', 'font-size': 11, 'font-family': 'Inter, sans-serif', 'font-weight': 600 }, 'weight w →'));
    svg.appendChild(el('text', { x: 14, y: 150, fill: '#9aa4b8', 'font-size': 11, 'font-family': 'Inter, sans-serif', 'font-weight': 600, transform: 'rotate(-90 14 150)', 'text-anchor': 'middle' }, 'loss L'));
    // valley marker
    svg.appendChild(el('text', { x: PX(W0), y: PY(0.1) + 18, 'text-anchor': 'middle', fill: '#55d68b', 'font-size': 10.5, 'font-family': 'Inter, sans-serif', 'font-weight': 700 }, '★ goal'));
    // trail ghosts
    trail.forEach((tw, i) => {
      svg.appendChild(el('circle', {
        cx: PX(tw), cy: PY(L(tw)), r: 5,
        fill: '#ffd166', opacity: 0.12 + 0.3 * (i / trail.length),
      }));
    });
    // tangent line at ball (slope converted from data units to pixels; SVG y points down)
    const g = slope(w), bx = PX(w), by = PY(L(w));
    const scaleX = 410 / (WMAX - WMIN), scaleY = 245 / LMAX;
    const dxPix = 46, dyPix = -g * scaleY / scaleX * dxPix;
    svg.appendChild(el('line', {
      x1: bx - dxPix, y1: by - dyPix, x2: bx + dxPix, y2: by + dyPix,
      stroke: '#ffd166', 'stroke-width': 2, 'stroke-dasharray': '5 4', opacity: 0.85,
    }));
    // downhill arrow
    const dir = g > 0 ? -1 : 1;
    if (Math.abs(g) > 0.04) {
      svg.appendChild(el('path', {
        d: `M ${bx + dir * 18} ${by - 24} l ${dir * 22} 0 l ${-dir * 7} -6 m ${dir * 7} 6 l ${-dir * 7} 6`,
        stroke: '#55d68b', 'stroke-width': 2.5, fill: 'none', 'stroke-linecap': 'round',
      }));
    }
    // ball
    svg.appendChild(el('circle', { cx: bx, cy: by, r: 10, fill: '#ffd166', stroke: '#0b0e14', 'stroke-width': 2, style: 'cursor:grab' }));

    roW.textContent = w.toFixed(2);
    roL.textContent = L(w).toFixed(2);
    roG.textContent = fmt(g, 2);
    roDir.textContent = Math.abs(g) < 0.04 ? '— stay (slope ≈ 0)' : g > 0 ? '← left (downhill)' : '→ right (downhill)';
  }

  function setCaption(html) { cap.innerHTML = html; }

  function step() {
    const g = slope(w);
    trail.push(w);
    if (trail.length > 12) trail.shift();
    let next = w - lr * g;
    let diverged = false;
    if (next < WMIN || next > WMAX) { next = Math.max(WMIN, Math.min(WMAX, next)); diverged = true; }
    // animate
    const from = w, to = next, t0 = performance.now();
    function anim(t) {
      const u = Math.min(1, (t - t0) / 350);
      w = from + (to - from) * (1 - Math.pow(1 - u, 3));
      draw();
      if (u < 1) requestAnimationFrame(anim);
      else {
        if (diverged) setCaption('<b style="color:#ff6b72">Diverged!</b> The step was so large it jumped clean over the valley and ended up higher. Pick a smaller η and try again.');
        else if (Math.abs(slope(w)) < 0.05) setCaption('<b style="color:#55d68b">Bottom reached.</b> The slope is ≈ 0, so the update stops moving — the weight has converged.');
        else if (lr >= 2) setCaption('Overshot the bottom — with η = 2.1 every step lands <b>higher on the other side</b>. Watch the loss grow.');
        else setCaption(`Moved against the slope: w went ${to < from ? 'left' : 'right'}, loss went <b>down</b>.`);
      }
    }
    requestAnimationFrame(anim);
  }

  function stopAuto() {
    if (auto) { clearInterval(auto); auto = null; autoBtn.textContent = '▶ Walk down'; }
  }

  document.getElementById('hillStep').addEventListener('click', () => { stopAuto(); step(); });
  autoBtn.addEventListener('click', () => {
    if (auto) { stopAuto(); return; }
    autoBtn.textContent = '⏸ Stop';
    auto = setInterval(() => {
      if (Math.abs(slope(w)) < 0.05 && lr < 2) { stopAuto(); return; }
      step();
    }, 480);
  });
  document.querySelectorAll('.lr-chips .chip').forEach(c => {
    c.addEventListener('click', () => {
      document.querySelectorAll('.lr-chips .chip').forEach(x => x.classList.remove('active'));
      c.classList.add('active');
      lr = parseFloat(c.dataset.lr);
      setCaption(lr >= 2
        ? 'Reckless mode armed. Take a few steps and watch it <b>overshoot</b>.'
        : lr <= 0.2
          ? 'Timid mode: safe, but it will take <b>many</b> steps to reach the bottom.'
          : 'A good step size: fast progress, no overshooting.');
    });
  });

  // dragging
  function pointerW(ev) {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * 460;
    return Math.max(WMIN, Math.min(WMAX, fromPX(px)));
  }
  let dragging = false;
  svg.addEventListener('pointerdown', ev => { dragging = true; stopAuto(); trail = []; w = pointerW(ev); draw(); svg.setPointerCapture(ev.pointerId); });
  svg.addEventListener('pointermove', ev => { if (dragging) { w = pointerW(ev); draw(); } });
  svg.addEventListener('pointerup', () => { dragging = false; });

  draw();
})();

/* ===================== 2. THE BACKWARD PASS, STEP BY STEP ===================== */
(() => {
  const svg = document.getElementById('bpSvg');
  const dotsEl = document.getElementById('bpDots');
  const titleEl = document.getElementById('bpTitle');
  const explainEl = document.getElementById('bpExplain');
  const prevBtn = document.getElementById('bpPrev');
  const nextBtn = document.getElementById('bpNext');

  /* --- the actual math, computed live --- */
  const x = [1.0, 0.5];
  const y = 1.0;
  const LR = 0.5;
  const W1 = [[0.6, -0.4], [0.3, 0.8]];   // W1[input][hidden]
  const W2 = [0.5, -0.3];                  // hidden -> output

  function forward(W1_, W2_) {
    const hin = [0, 1].map(j => x[0] * W1_[0][j] + x[1] * W1_[1][j]);
    const h = hin.map(Math.tanh);
    const yhat = h[0] * W2_[0] + h[1] * W2_[1];
    return { hin, h, yhat, loss: 0.5 * (yhat - y) ** 2 };
  }
  const f = forward(W1, W2);
  const e = f.yhat - y;                               // dL/dŷ
  const gW2 = [e * f.h[0], e * f.h[1]];               // blame for output weights
  const gh = [e * W2[0], e * W2[1]];                  // blame for hidden activations
  const ghin = [gh[0] * (1 - f.h[0] ** 2), gh[1] * (1 - f.h[1] ** 2)];
  const gW1 = [[x[0] * ghin[0], x[0] * ghin[1]], [x[1] * ghin[0], x[1] * ghin[1]]];
  const W1n = W1.map((row, i) => row.map((w, j) => w - LR * gW1[i][j]));
  const W2n = W2.map((w, j) => w - LR * gW2[j]);
  const f2 = forward(W1n, W2n);

  /* --- layout --- */
  const IN = [{ x: 95, y: 75 }, { x: 95, y: 195 }];
  const HID = [{ x: 340, y: 75 }, { x: 340, y: 195 }];
  const OUT = { x: 585, y: 135 };
  const R = 27;
  // edges: list of {from, to, w, wNew, g, key}
  const EDGES = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++)
    EDGES.push({ from: IN[i], to: HID[j], w: W1[i][j], wNew: W1n[i][j], g: gW1[i][j] });
  for (let j = 0; j < 2; j++)
    EDGES.push({ from: HID[j], to: OUT, w: W2[j], wNew: W2n[j], g: gW2[j] });

  const STEPS = [
    {
      kicker: 'Step 0 of 4', title: 'The setup',
      explain: `A tiny but real network: <b>2 inputs, 2 hidden neurons, 1 output — six weights</b>, shown as connections (<b style="color:#6090ff">blue</b> = positive, <b style="color:#ff6b72">red</b> = negative, thickness = size). The inputs are ${x[0].toFixed(1)} and ${x[1].toFixed(1)}, and we want the output to be <b>${y.toFixed(2)}</b>. Every number you'll see is computed live on this page.`,
    },
    {
      kicker: 'Step 1 of 4', title: 'Forward pass — make a guess',
      explain: `The signal flows <b>left to right</b>: each neuron sums its inputs × weights and squashes the result. The network guesses <b>ŷ = ${f.yhat.toFixed(2)}</b>. Notice the bottom hidden neuron computes exactly <b>0.00</b> — its two inputs cancelled out. Remember it.`,
    },
    {
      kicker: 'Step 2 of 4', title: 'Compare — measure the error',
      explain: `We wanted <b>${y.toFixed(2)}</b>, we got <b>${f.yhat.toFixed(2)}</b> — too low by ${Math.abs(e).toFixed(2)}. Squashing that miss into one number gives the loss: <b>L = ½(ŷ − y)² = ${f.loss.toFixed(3)}</b>. The whole point of what follows is to make this number smaller.`,
    },
    {
      kicker: 'Step 3 of 4', title: 'Backward pass — blame flows right to left',
      explain: `The error sweeps <b>backward</b>, and each weight receives its <b>blame</b> (its gradient): how much the loss would change if that weight were nudged up. It's just the chain rule — <i>blame arriving from the right × what flowed in from the left</i>. Look at the bottom-right connection: its blame is <b>exactly 0</b>, because the neuron behind it output 0.00 — a weight that contributed nothing learns nothing this round.`,
    },
    {
      kicker: 'Step 4 of 4', title: 'Update — nudge all six weights at once',
      explain: `Every weight takes one small step against its blame (η = ${LR}): <b>w ← w − η·blame</b>. Run the forward pass again with the new weights and the guess improves from ${f.yhat.toFixed(2)} to <b>${f2.yhat.toFixed(2)}</b>, the loss drops from ${f.loss.toFixed(3)} to <b style="color:#55d68b">${f2.loss.toFixed(3)}</b>. That's one learning step. Training = this, repeated thousands of times.`,
    },
  ];

  let step = 0;
  let animToken = 0;

  const edgeColor = w => (w >= 0 ? '#6090ff' : '#ff6b72');
  const edgeWidth = w => 1.5 + Math.min(5, Math.abs(w) * 5);
  const trim = (a, b, r1, r2) => {
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy);
    return {
      x1: a.x + dx / len * r1, y1: a.y + dy / len * r1,
      x2: b.x - dx / len * r2, y2: b.y - dy / len * r2,
    };
  };

  function node(p, label, value, opts = {}) {
    const g = el('g');
    g.appendChild(el('circle', {
      cx: p.x, cy: p.y, r: R,
      fill: opts.fill || '#151c2c',
      stroke: opts.stroke || '#3a4666',
      'stroke-width': opts.strokeWidth || 1.5,
    }));
    if (value != null) {
      g.appendChild(el('text', {
        x: p.x, y: p.y + 5, 'text-anchor': 'middle',
        fill: opts.valueColor || '#eef1f7', 'font-size': 13.5, 'font-weight': 700,
        'font-family': 'Inter, sans-serif', 'font-variant-numeric': 'tabular-nums',
      }, value));
    }
    g.appendChild(el('text', {
      x: p.x, y: p.y + R + 16, 'text-anchor': 'middle',
      fill: '#9aa4b8', 'font-size': 10.5, 'font-weight': 700, 'font-family': 'Inter, sans-serif',
    }, label));
    svg.appendChild(g);
  }

  function pulse(edges, color, reverse, token) {
    const t0 = performance.now();
    const DUR = 750;
    const layer = el('g');
    svg.appendChild(layer);
    function anim(t) {
      if (token !== animToken) { layer.remove(); return; }
      const u = Math.min(1, (t - t0) / DUR);
      layer.innerHTML = '';
      edges.forEach(ed => {
        const { x1, y1, x2, y2 } = trim(ed.from, ed.to, R, R);
        const a = reverse ? { x: x2, y: y2 } : { x: x1, y: y1 };
        const b = reverse ? { x: x1, y: y1 } : { x: x2, y: y2 };
        layer.appendChild(el('circle', {
          cx: a.x + (b.x - a.x) * u, cy: a.y + (b.y - a.y) * u,
          r: 5, fill: color, opacity: 0.95,
        }));
      });
      if (u < 1) requestAnimationFrame(anim);
      else layer.remove();
    }
    requestAnimationFrame(anim);
  }

  function render() {
    animToken++;
    const token = animToken;
    svg.innerHTML = '';
    const s = step;
    const showActs = s >= 1;
    const useNew = s >= 4;

    // edges
    EDGES.forEach(ed => {
      const w = useNew ? ed.wNew : ed.w;
      const { x1, y1, x2, y2 } = trim(ed.from, ed.to, R, R);
      svg.appendChild(el('line', {
        x1, y1, x2, y2,
        stroke: edgeColor(w), 'stroke-width': edgeWidth(w),
        'stroke-opacity': s === 3 ? 0.35 : 0.85, 'stroke-linecap': 'round',
      }));
      // weight label at 38% along the edge
      const lx = x1 + (x2 - x1) * 0.38, ly = y1 + (y2 - y1) * 0.38 - 7;
      svg.appendChild(el('text', {
        x: lx, y: ly, 'text-anchor': 'middle', fill: edgeColor(w),
        'font-size': 11, 'font-weight': 700, 'font-family': 'Inter, sans-serif',
        'font-variant-numeric': 'tabular-nums',
      }, (useNew ? '' : 'w ') + w.toFixed(2)));
      // blame badge at 62% (step 3 only)
      if (s === 3) {
        const bx = x1 + (x2 - x1) * 0.62, by = y1 + (y2 - y1) * 0.62 + 2;
        const zero = Math.abs(ed.g) < 1e-9;
        const bg = el('g');
        bg.appendChild(el('rect', {
          x: bx - 33, y: by - 11, width: 66, height: 20, rx: 6,
          fill: zero ? '#1a1f2e' : 'rgba(255,209,102,0.13)',
          stroke: zero ? '#3a4666' : '#ffd166', 'stroke-width': 1,
        }));
        bg.appendChild(el('text', {
          x: bx, y: by + 3.5, 'text-anchor': 'middle',
          fill: zero ? '#9aa4b8' : '#ffd166', 'font-size': 10.5, 'font-weight': 700,
          'font-family': 'Inter, sans-serif', 'font-variant-numeric': 'tabular-nums',
        }, 'blame ' + fmt(ed.g, 2)));
        svg.appendChild(bg);
      }
    });

    // nodes
    const acts = useNew ? f2 : f;
    node(IN[0], 'input x₁', x[0].toFixed(2));
    node(IN[1], 'input x₂', x[1].toFixed(2));
    node(HID[0], 'hidden h₁', showActs ? acts.h[0].toFixed(2) : '·');
    node(HID[1], 'hidden h₂', showActs ? acts.h[1].toFixed(2) : '·');
    node(OUT, 'output ŷ', showActs ? acts.yhat.toFixed(2) : '?', {
      stroke: s === 2 ? '#ff6b72' : s >= 4 ? '#55d68b' : '#3a4666',
      strokeWidth: s >= 2 ? 2.5 : 1.5,
      valueColor: s === 2 ? '#ff6b72' : s >= 4 ? '#55d68b' : '#eef1f7',
    });

    // target + loss readout (steps >= 2)
    if (s >= 2) {
      const g = el('g');
      g.appendChild(el('text', {
        x: OUT.x, y: OUT.y - R - 30, 'text-anchor': 'middle',
        fill: '#55d68b', 'font-size': 11.5, 'font-weight': 700, 'font-family': 'Inter, sans-serif',
      }, `target y = ${y.toFixed(2)}`));
      g.appendChild(el('text', {
        x: OUT.x, y: OUT.y - R - 14, 'text-anchor': 'middle',
        fill: s >= 4 ? '#55d68b' : '#ff6b72', 'font-size': 12, 'font-weight': 800,
        'font-family': 'Inter, sans-serif', 'font-variant-numeric': 'tabular-nums',
      }, s >= 4 ? `loss ${f.loss.toFixed(3)} → ${f2.loss.toFixed(3)} ✓` : `loss = ${f.loss.toFixed(3)}`));
      svg.appendChild(g);
    }

    // direction banner
    if (s === 1 || s === 3) {
      svg.appendChild(el('text', {
        x: 340, y: 22, 'text-anchor': 'middle',
        fill: s === 1 ? '#6090ff' : '#ffd166', 'font-size': 11.5, 'font-weight': 800,
        'font-family': 'Inter, sans-serif', 'letter-spacing': '0.08em',
      }, s === 1 ? 'SIGNAL →' : '← BLAME'));
    }

    // animations
    if (s === 1) {
      pulse(EDGES.slice(0, 4), '#6090ff', false, token);
      setTimeout(() => { if (token === animToken) pulse(EDGES.slice(4), '#6090ff', false, token); }, 780);
    } else if (s === 3) {
      pulse(EDGES.slice(4), '#ffd166', true, token);
      setTimeout(() => { if (token === animToken) pulse(EDGES.slice(0, 4), '#ffd166', true, token); }, 780);
    }

    // chrome
    titleEl.innerHTML = `<small>${STEPS[s].kicker}</small>${STEPS[s].title}`;
    explainEl.innerHTML = STEPS[s].explain;
    dotsEl.innerHTML = '';
    STEPS.forEach((_, i) => {
      const d = document.createElement('span');
      d.className = 'step-dot' + (i === s ? ' current' : i < s ? ' done' : '');
      dotsEl.appendChild(d);
    });
    prevBtn.disabled = s === 0;
    nextBtn.disabled = s === STEPS.length - 1;
  }

  prevBtn.addEventListener('click', () => { if (step > 0) { step--; render(); } });
  nextBtn.addEventListener('click', () => { if (step < STEPS.length - 1) { step++; render(); } });
  render();
})();

/* ===================== 3. WATCH IT LEARN ===================== */
(() => {
  const fitSvg = document.getElementById('fitSvg');
  const lossSvg = document.getElementById('lossSvg');
  const netSvg = document.getElementById('netSvg');
  const stepsEl = document.getElementById('trSteps');
  const lossEl = document.getElementById('trLoss');
  const playBtn = document.getElementById('trPlay');

  const H = 6;                      // hidden neurons
  const LR = 0.04;
  const target = x => 0.62 * Math.sin(2.6 * x) + 0.18 * x;
  const XS = Array.from({ length: 33 }, (_, i) => -1 + 2 * i / 32);
  const YS = XS.map(target);

  let w1, b1, w2, b2, steps, lossHist, playing = false, raf = null;

  function init() {
    const r = () => (Math.random() - 0.5) * 1.6;
    w1 = Array.from({ length: H }, r);
    b1 = Array.from({ length: H }, r);
    w2 = Array.from({ length: H }, r);
    b2 = 0;
    steps = 0;
    lossHist = [];
  }

  function predict(x) {
    let out = b2;
    for (let j = 0; j < H; j++) out += w2[j] * Math.tanh(w1[j] * x + b1[j]);
    return out;
  }

  function trainSteps(n) {
    for (let s = 0; s < n; s++) {
      const gw1 = new Array(H).fill(0), gb1 = new Array(H).fill(0), gw2 = new Array(H).fill(0);
      let gb2 = 0, loss = 0;
      for (let i = 0; i < XS.length; i++) {
        const x = XS[i];
        const hin = new Array(H), h = new Array(H);
        let yhat = b2;
        for (let j = 0; j < H; j++) {
          hin[j] = w1[j] * x + b1[j];
          h[j] = Math.tanh(hin[j]);
          yhat += w2[j] * h[j];
        }
        const e = yhat - YS[i];
        loss += 0.5 * e * e;
        gb2 += e;
        for (let j = 0; j < H; j++) {
          gw2[j] += e * h[j];
          const gh = e * w2[j] * (1 - h[j] * h[j]);
          gw1[j] += gh * x;
          gb1[j] += gh;
        }
      }
      const n_ = XS.length;
      for (let j = 0; j < H; j++) {
        w1[j] -= LR * gw1[j] / n_;
        b1[j] -= LR * gb1[j] / n_;
        w2[j] -= LR * gw2[j] / n_;
      }
      b2 -= LR * gb2 / n_;
      steps++;
      if (steps % 20 === 0) lossHist.push(loss / n_);
      if (lossHist.length > 220) lossHist.shift();
    }
  }

  const FX = x => 30 + (x + 1) / 2 * 410;
  const FY = y => 150 - y * 115;

  function pathOf(fn) {
    let d = '';
    for (let px = 0; px <= 100; px++) {
      const x = -1 + 2 * px / 100;
      d += (d ? ' L ' : 'M ') + FX(x).toFixed(1) + ' ' + FY(fn(x)).toFixed(1);
    }
    return d;
  }

  function draw() {
    // fit plot
    fitSvg.innerHTML = '';
    fitSvg.appendChild(el('line', { x1: 30, y1: 150, x2: 440, y2: 150, stroke: '#263149', 'stroke-width': 1 }));
    fitSvg.appendChild(el('path', { d: pathOf(target), fill: 'none', stroke: '#9aa4b8', 'stroke-width': 2, 'stroke-dasharray': '6 5' }));
    XS.filter((_, i) => i % 2 === 0).forEach((x, i) => {
      fitSvg.appendChild(el('circle', { cx: FX(x), cy: FY(target(x)), r: 2.5, fill: '#9aa4b8', opacity: 0.6 }));
    });
    fitSvg.appendChild(el('path', { d: pathOf(predict), fill: 'none', stroke: '#55d68b', 'stroke-width': 2.8 }));

    // loss sparkline
    lossSvg.innerHTML = '';
    if (lossHist.length > 1) {
      const max = Math.max(...lossHist, 1e-6);
      let d = '';
      lossHist.forEach((l, i) => {
        const px = 6 + i / (lossHist.length - 1) * 188;
        const py = 62 - (l / max) * 52;
        d += (d ? ' L ' : 'M ') + px.toFixed(1) + ' ' + py.toFixed(1);
      });
      lossSvg.appendChild(el('path', { d, fill: 'none', stroke: '#ffd166', 'stroke-width': 2 }));
    }

    // live network
    netSvg.innerHTML = '';
    const inP = { x: 25, y: 75 };
    const hidP = Array.from({ length: H }, (_, j) => ({ x: 100, y: 15 + j * 24 }));
    const outP = { x: 175, y: 75 };
    hidP.forEach((p, j) => {
      netSvg.appendChild(el('line', {
        x1: inP.x, y1: inP.y, x2: p.x, y2: p.y,
        stroke: w1[j] >= 0 ? '#6090ff' : '#ff6b72',
        'stroke-width': Math.min(4.5, 0.4 + Math.abs(w1[j]) * 1.4), 'stroke-opacity': 0.8,
      }));
      netSvg.appendChild(el('line', {
        x1: p.x, y1: p.y, x2: outP.x, y2: outP.y,
        stroke: w2[j] >= 0 ? '#6090ff' : '#ff6b72',
        'stroke-width': Math.min(4.5, 0.4 + Math.abs(w2[j]) * 1.4), 'stroke-opacity': 0.8,
      }));
    });
    [inP, ...hidP, outP].forEach(p => {
      netSvg.appendChild(el('circle', { cx: p.x, cy: p.y, r: 6, fill: '#151c2c', stroke: '#3a4666', 'stroke-width': 1.2 }));
    });

    stepsEl.textContent = steps.toLocaleString();
    lossEl.textContent = lossHist.length ? lossHist[lossHist.length - 1].toFixed(4) : '—';
  }

  function loop() {
    trainSteps(25);
    draw();
    if (playing) raf = requestAnimationFrame(loop);
  }

  playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.textContent = playing ? '⏸ Pause' : '▶ Train';
    if (playing) loop();
    else cancelAnimationFrame(raf);
  });
  document.getElementById('trStep').addEventListener('click', () => { trainSteps(100); draw(); });
  document.getElementById('trReset').addEventListener('click', () => {
    playing = false;
    playBtn.textContent = '▶ Train';
    cancelAnimationFrame(raf);
    init();
    draw();
  });

  init();
  draw();
})();
