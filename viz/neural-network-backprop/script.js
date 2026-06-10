/* ============================================================
   Feedforward Neural Network — interactive explainer
   ============================================================ */

const SVG_NS = 'http://www.w3.org/2000/svg';
const net = document.getElementById('net');
const W = 900, H = 460;

// state
let nLayers = 4;
let hidden = 4;       // neurons per hidden layer
let nInputs = 3;      // neurons in the input layer
let nOutputs = 2;     // neurons in the output layer
let layers = [];      // array of arrays of node objects

// activation functions; the chosen one is applied at the output layer
const NET_ACTS = {
  sigmoid: z => 1 / (1 + Math.exp(-z)),
  tanh: z => Math.tanh(z),
  relu: z => Math.max(0, z),
  leakyrelu: z => z >= 0 ? z : 0.01 * z,
  elu: z => z >= 0 ? z : (Math.exp(z) - 1),
  gelu: z => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z * z * z))),
  swish: z => z / (1 + Math.exp(-z)),
  softplus: z => Math.log(1 + Math.exp(z)),
  linear: z => z,
};
const ACT_NAMES = {
  sigmoid: 'sigmoid', tanh: 'tanh', relu: 'ReLU', leakyrelu: 'LeakyReLU',
  elu: 'ELU', gelu: 'GELU', swish: 'Swish', softplus: 'Softplus', linear: 'linear',
};
let activationFn = NET_ACTS.sigmoid;

/* color scale for weights: blue (negative) → grey (0) → red (positive) */
function weightColor(w) {
  const t = Math.max(-1, Math.min(1, w)); // clamp to [-1,1]
  const mag = Math.abs(t);
  // interpolate from neutral grey toward the signed hue
  const neutral = [90, 99, 122];          // #5a637a-ish
  const pos = [255, 99, 99];              // red  (#ff6363)
  const neg = [91, 140, 255];            // blue (#5b8cff)
  const target = t >= 0 ? pos : neg;
  const c = neutral.map((n, k) => Math.round(n + (target[k] - n) * mag));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}
function weightWidth(w) { return 1 + Math.abs(Math.max(-1, Math.min(1, w))) * 2.5; }

/* deterministic pseudo-random so weights/inputs are stable across rebuilds */
let seed = 1;
function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
function resetSeed() { seed = 1; }

/* ---------- the architecture ---------- */
function layerSizes() {
  const sizes = [];
  for (let l = 0; l < nLayers; l++) {
    if (l === 0) sizes.push(nInputs);                 // input
    else if (l === nLayers - 1) sizes.push(nOutputs); // output
    else sizes.push(hidden);                          // hidden
  }
  return sizes;
}

function buildNetwork() {
  net.innerHTML = '';
  layers = [];
  resetSeed();
  const sizes = layerSizes();
  const padX = 100, padY = 64;
  const usableW = W - padX * 2;

  // node objects with positions, a value, and incoming weights
  const colH = H - padY * 2;
  const maxCount = Math.max(...sizes);
  // shared vertical spacing so every column uses the same scale and is centered
  const gap = maxCount <= 1 ? 0 : Math.min(72, colH / (maxCount - 1));
  sizes.forEach((count, l) => {
    const x = padX + (usableW * l) / (sizes.length - 1);
    const blockH = (count - 1) * gap;
    const startY = H / 2 - blockH / 2; // center this column vertically
    const nodes = [];
    for (let i = 0; i < count; i++) {
      const y = startY + gap * i;
      // input layer: a value in [0,1]; others computed during propagation
      const value = l === 0 ? +(rnd().toFixed(2)) : 0;
      // every non-input neuron has a bias term b
      const bias = l === 0 ? 0 : +((rnd() * 2 - 1).toFixed(2));
      nodes.push({ x, y, el: null, valEl: null, value, bias, weights: [] });
    }
    layers.push(nodes);
  });

  // assign a weight to every incoming connection
  for (let l = 1; l < layers.length; l++) {
    layers[l].forEach(node => {
      node.weights = layers[l - 1].map(() => +((rnd() * 2 - 1).toFixed(2))); // [-1,1]
    });
  }

  drawBiasNodes();
  drawEdges();
  drawNodes();
  drawLabels();
  computeForward(false); // fill in values without animation
  updateParamCount();
}

/* ---------- bias units: one "+1" node per layer that feeds the next ---------- */
let biasNodes = []; // biasNodes[l] = {x, y} feeding layer l+1
function drawBiasNodes() {
  biasNodes = [];
  for (let l = 0; l < layers.length - 1; l++) {
    const col = layers[l];
    const bx = col[0].x;
    const by = 26; // just above the top neuron of the column
    biasNodes[l] = { x: bx, y: by };

    // edges from this bias unit to every neuron in the next layer
    layers[l + 1].forEach((b, i) => {
      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', bx); line.setAttribute('y1', by);
      line.setAttribute('x2', b.x); line.setAttribute('y2', b.y);
      line.setAttribute('class', 'edge bias-edge');
      line.dataset.to = `${l + 1}-${i}`;
      line.dataset.bias = '1';
      net.appendChild(line);

      // bias-value label near the receiving neuron
      const t = 0.72;
      const wl = document.createElementNS(SVG_NS, 'text');
      wl.setAttribute('x', bx + (b.x - bx) * t);
      wl.setAttribute('y', by + (b.y - by) * t - 2);
      wl.setAttribute('text-anchor', 'middle');
      wl.setAttribute('class', 'weight-label bias-label');
      wl.dataset.to = `${l + 1}-${i}`;
      wl.textContent = layers[l + 1][i].bias.toFixed(2);
      net.appendChild(wl);
    });

    // the bias node itself (drawn after edges)
    const c = document.createElementNS(SVG_NS, 'circle');
    c.setAttribute('cx', bx); c.setAttribute('cy', by);
    c.setAttribute('r', 13);
    c.setAttribute('class', 'bias-node');
    net.appendChild(c);
    const tx = document.createElementNS(SVG_NS, 'text');
    tx.setAttribute('x', bx); tx.setAttribute('y', by);
    tx.setAttribute('text-anchor', 'middle');
    tx.setAttribute('dominant-baseline', 'central');
    tx.setAttribute('class', 'bias-node-label');
    // label b1, b2, ... (the bias unit feeding layer l+1)
    const main = document.createElementNS(SVG_NS, 'tspan');
    main.textContent = 'b';
    const sub = document.createElementNS(SVG_NS, 'tspan');
    sub.setAttribute('baseline-shift', 'sub');
    sub.setAttribute('font-size', '8');
    sub.textContent = (l + 1);
    tx.appendChild(main);
    tx.appendChild(sub);
    net.appendChild(tx);
  }
}

/* ---------- parameter counter (weights + biases) ---------- */
function updateParamCount() {
  let weightsN = 0, biasN = 0;
  const wTerms = [], bTerms = [];
  for (let l = 1; l < layers.length; l++) {
    const nPrev = layers[l - 1].length, nCur = layers[l].length;
    weightsN += nPrev * nCur;
    biasN += nCur;
    wTerms.push(`${nPrev}×${nCur}`);
    bTerms.push(`${nCur}`);
  }
  const total = weightsN + biasN;
  const el = document.getElementById('paramCount');
  if (!el) return;
  el.innerHTML =
    `<div class="pc-line"><b>${total}</b> trainable parameters &nbsp;=&nbsp; ${weightsN} weights + ${biasN} biases</div>
     <div class="pc-formula">
       weights = (${wTerms.join(') + (')}) = ${weightsN}<br>
       biases &nbsp;= ${bTerms.join(' + ')} = ${biasN}<br>
       <span class="pc-tot">total = ${weightsN} + ${biasN} = ${total}</span>
     </div>`;
}

function drawEdges() {
  for (let l = 0; l < layers.length - 1; l++) {
    layers[l].forEach((a, j) => {
      layers[l + 1].forEach((b, i) => {
        const w = layers[l + 1][i].weights[j];
        const line = document.createElementNS(SVG_NS, 'line');
        line.setAttribute('x1', a.x); line.setAttribute('y1', a.y);
        line.setAttribute('x2', b.x); line.setAttribute('y2', b.y);
        line.setAttribute('class', 'edge');
        line.dataset.from = `${l}-${j}`;
        line.dataset.to = `${l + 1}-${i}`;
        net.appendChild(line);

        // weight label at the midpoint, nudged to reduce overlap
        const t = +(0.35 + 0.3 * (i / Math.max(1, layers[l + 1].length - 1)));
        const mx = a.x + (b.x - a.x) * t;
        const my = a.y + (b.y - a.y) * t;
        const wl = document.createElementNS(SVG_NS, 'text');
        wl.setAttribute('x', mx); wl.setAttribute('y', my - 2);
        wl.setAttribute('text-anchor', 'middle');
        wl.setAttribute('class', 'weight-label');
        wl.dataset.to = `${l + 1}-${i}`;
        wl.dataset.from = `${l}-${j}`;
        wl.textContent = w.toFixed(2);
        net.appendChild(wl);
      });
    });
  }
}

function drawNodes() {
  layers.forEach((nodes, l) => {
    nodes.forEach((n, i) => {
      const c = document.createElementNS(SVG_NS, 'circle');
      c.setAttribute('cx', n.x); c.setAttribute('cy', n.y);
      c.setAttribute('r', 18);
      c.setAttribute('class', 'node');
      c.dataset.layer = l; c.dataset.idx = i;
      net.appendChild(c);
      n.el = c;

      // value text inside the node
      const v = document.createElementNS(SVG_NS, 'text');
      v.setAttribute('x', n.x); v.setAttribute('y', n.y);
      v.setAttribute('text-anchor', 'middle');
      v.setAttribute('dominant-baseline', 'central');
      v.setAttribute('class', 'node-val');
      v.textContent = n.value.toFixed(2);
      net.appendChild(v);
      n.valEl = v;

      c.addEventListener('mouseenter', () => onNodeHover(l, i));
      c.addEventListener('mouseleave', onNodeLeave);
      c.addEventListener('click', (e) => { e.stopPropagation(); onNodeClick(l, i); });
    });
  });
}

function drawLabels() {
  layers.forEach((nodes, l) => {
    const t = document.createElementNS(SVG_NS, 'text');
    t.setAttribute('x', nodes[0].x);
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('class', 'layer-label');
    let lbl;
    const hiddenCount = layers.length - 2;
    if (l === 0) lbl = 'Input layer';
    else if (l === layers.length - 1) lbl = 'Output layer';
    else lbl = hiddenCount === 1 ? 'Hidden layer' : `Hidden layer ${l}`;
    // first line: name, second line: l-index
    const name = document.createElementNS(SVG_NS, 'tspan');
    name.setAttribute('x', nodes[0].x);
    name.textContent = lbl;
    const idx = document.createElementNS(SVG_NS, 'tspan');
    idx.setAttribute('x', nodes[0].x);
    idx.setAttribute('dy', '15');
    idx.setAttribute('class', 'layer-idx');
    idx.textContent = `l = ${l + 1}`;
    t.appendChild(name);
    t.appendChild(idx);
    t.setAttribute('y', H - 30);
    net.appendChild(t);
  });
}

/* ---------- forward pass: compute every neuron value ---------- */
function computeForward(animate) {
  for (let l = 1; l < layers.length; l++) {
    const isOutput = l === layers.length - 1;
    layers[l].forEach(node => {
      let xi = node.bias;
      layers[l - 1].forEach((prev, j) => { xi += node.weights[j] * prev.value; });
      // activation σ is applied only on the final (output) layer;
      // hidden layers pass the raw weighted sum ξ through.
      node.value = isOutput ? activationFn(xi) : xi;
    });
  }
  if (!animate) {
    layers.forEach(nodes => nodes.forEach(n => { n.valEl.textContent = n.value.toFixed(2); }));
  }
}

let pinned = null; // {l, i} of the clicked node whose calculation is shown

/* ---------- hover: just light up the connections feeding a neuron ---------- */
function onNodeHover(l, i) {
  const node = layers[l][i];
  net.querySelectorAll('.node.hover, .edge.hover').forEach(e => e.classList.remove('hover'));
  node.el.classList.add('hover');
  if (l > 0) {
    net.querySelectorAll(`.edge[data-to="${l}-${i}"]`).forEach(e => {
      e.classList.add('hover');
      if (e.dataset.from) layers[l - 1][parseInt(e.dataset.from.split('-')[1])].el.classList.add('hover');
    });
  }
}

function onNodeLeave() {
  net.querySelectorAll('.node.hover, .edge.hover').forEach(e => e.classList.remove('hover'));
}

/* ---------- click: pin a neuron, reveal weights + fill the panel below ---------- */
const calcPanel = document.getElementById('calcPanel');
const calcEmpty = calcPanel.querySelector('.calc-empty');
const calcContent = calcPanel.querySelector('.calc-content');
const calcTitle = document.getElementById('calcTitle');
const calcSum = document.getElementById('calcSum');
const calcAct = document.getElementById('calcAct');

function onNodeClick(l, i) {
  if (pinned && pinned.l === l && pinned.i === i) { clearPinned(); return; }
  clearPinned();
  pinned = { l, i };
  const node = layers[l][i];
  node.el.classList.add('active');

  const layerName = l === 0 ? 'Input layer'
    : (l === layers.length - 1 ? 'Output layer'
    : (layers.length - 2 === 1 ? 'Hidden layer' : `Hidden layer ${l}`));

  calcEmpty.hidden = true;
  calcContent.hidden = false;

  if (l === 0) {
    calcTitle.textContent = `${layerName} · neuron ${i + 1}`;
    calcSum.innerHTML = `This is an <b>input</b> neuron — its value is given, not computed.`;
    calcAct.innerHTML = `value &nbsp; x = <b>${node.value.toFixed(2)}</b>`;
    return;
  }

  // highlight incoming edges (weights + the bias edge), colour them by weight,
  // and reveal their labels
  net.querySelectorAll(`.edge[data-to="${l}-${i}"]`).forEach(e => {
    e.classList.add('active');
    if (e.dataset.from) {
      const j = parseInt(e.dataset.from.split('-')[1]);
      layers[l - 1][j].el.classList.add('active');
      // intensity colour scale applied only to this neuron's incoming weights
      e.style.stroke = weightColor(node.weights[j]);
      e.style.strokeWidth = weightWidth(node.weights[j]);
    } else {
      e.classList.add('bias-active'); // bias edge
    }
  });
  net.querySelectorAll(`.weight-label[data-to="${l}-${i}"]`).forEach(w => w.classList.add('shown'));

  const parts = node.weights.map((w, j) =>
    `(${w.toFixed(2)} × ${layers[l - 1][j].value.toFixed(2)})`);
  const b = node.bias;
  const xi = node.weights.reduce((s, w, j) => s + w * layers[l - 1][j].value, 0) + b;
  const isOutput = l === layers.length - 1;
  const biasStr = `${b < 0 ? '−' : '+'} ${Math.abs(b).toFixed(2)}`;

  calcTitle.textContent = `${layerName} · neuron ${i + 1}`;
  calcSum.innerHTML = `<span class="ck">Weighted sum</span> &nbsp; ξ = ${parts.join(' + ')} ${biasStr} <span class="bias-tag">(bias b<sub>${l}</sub>)</span> = <b>${xi.toFixed(2)}</b>`;
  if (isOutput) {
    const key = netActSelect.value;
    // σ conventionally means sigmoid; use the actual function name otherwise
    const fn = key === 'sigmoid' ? 'σ' : (ACT_NAMES[key] || key);
    calcAct.innerHTML = `<span class="ck">Activation</span> &nbsp; x = ${fn}(ξ) = ${fn}(${xi.toFixed(2)}) = <b>${node.value.toFixed(2)}</b>`;
  } else {
    calcAct.innerHTML = `<span class="ck">Output</span> &nbsp; this is a hidden layer, so the raw sum is passed on: x = <b>${node.value.toFixed(2)}</b>`;
  }
}

function clearPinned() {
  pinned = null;
  net.querySelectorAll('.node.active').forEach(n => n.classList.remove('active'));
  net.querySelectorAll('.edge.active').forEach(e => {
    e.classList.remove('active');
    e.style.stroke = ''; e.style.strokeWidth = ''; // back to neutral
  });
  net.querySelectorAll('.edge.bias-active').forEach(e => e.classList.remove('bias-active'));
  net.querySelectorAll('.weight-label.shown').forEach(e => e.classList.remove('shown'));
  calcContent.hidden = true;
  calcEmpty.hidden = false;
}

// click on empty SVG space clears the selection
net.addEventListener('click', () => clearPinned());

function clearHighlight() {
  clearPinned();
  net.querySelectorAll('.node.hover, .edge.hover').forEach(e => e.classList.remove('hover'));
}

/* ---------- forward propagation animation ---------- */
let playing = false;
const playBtn = document.getElementById('play');

function propagate() {
  if (playing) return;
  playing = true;
  playBtn.disabled = true;
  clearHighlight();
  computeForward(true); // recompute values (they're already set, but stay consistent)

  const stepDelay = 800;
  layers.forEach((nodes, l) => {
    setTimeout(() => {
      nodes.forEach(n => {
        n.el.classList.add('active');
        n.valEl.textContent = n.value.toFixed(2); // reveal computed value
      });
      if (l > 0) {
        net.querySelectorAll('.edge').forEach(e => { if (e.dataset.to.startsWith(`${l}-`)) e.classList.add('active'); });
      }
      if (l < layers.length - 1) firePulses(l);
    }, l * stepDelay);
  });

  setTimeout(() => {
    clearHighlight();
    playing = false;
    playBtn.disabled = false;
  }, layers.length * stepDelay + 400);
}

function firePulses(l) {
  layers[l].forEach((a) => {
    layers[l + 1].forEach((b) => {
      const p = document.createElementNS(SVG_NS, 'circle');
      p.setAttribute('r', 4);
      p.setAttribute('class', 'pulse');
      p.setAttribute('cx', a.x); p.setAttribute('cy', a.y);
      net.appendChild(p);
      const anim = p.animate(
        [{ transform: `translate(0,0)` },
         { transform: `translate(${b.x - a.x}px, ${b.y - a.y}px)` }],
        { duration: 650, easing: 'ease-in-out' }
      );
      anim.onfinish = () => p.remove();
    });
  });
}

playBtn.addEventListener('click', propagate);

/* ---------- sliders ---------- */
const layersSlider = document.getElementById('layers');
const neuronsSlider = document.getElementById('neurons');
const layersOut = document.getElementById('layersOut');
const neuronsOut = document.getElementById('neuronsOut');

layersSlider.addEventListener('input', () => {
  nLayers = +layersSlider.value; layersOut.textContent = nLayers; buildNetwork();
});
neuronsSlider.addEventListener('input', () => {
  hidden = +neuronsSlider.value; neuronsOut.textContent = hidden; buildNetwork();
});

const inputsSlider = document.getElementById('inputs');
const outputsSlider = document.getElementById('outputs');
const inputsOut = document.getElementById('inputsOut');
const outputsOut = document.getElementById('outputsOut');
inputsSlider.addEventListener('input', () => {
  nInputs = +inputsSlider.value; inputsOut.textContent = nInputs; buildNetwork();
});
outputsSlider.addEventListener('input', () => {
  nOutputs = +outputsSlider.value; outputsOut.textContent = nOutputs; buildNetwork();
});

const netActSelect = document.getElementById('netAct');
netActSelect.addEventListener('change', () => {
  activationFn = NET_ACTS[netActSelect.value];
  computeForward(false);             // recompute output-layer values with new σ
  if (pinned) { const p = pinned; clearPinned(); onNodeClick(p.l, p.i); } // refresh panel
});

/* ---------- randomize weights & inputs ---------- */
const randomBtn = document.getElementById('randomize');
randomBtn.addEventListener('click', () => {
  // fresh random input values
  layers[0].forEach(n => { n.value = +Math.random().toFixed(2); });
  // fresh random weights + bias on every non-input neuron
  for (let l = 1; l < layers.length; l++) {
    layers[l].forEach(node => {
      node.weights = node.weights.map(() => +((Math.random() * 2 - 1).toFixed(2)));
      node.bias = +((Math.random() * 2 - 1).toFixed(2));
    });
  }
  // update on-screen weight + bias labels and input node values
  net.querySelectorAll('.weight-label').forEach(wl => {
    const l = +wl.dataset.to.split('-')[0];
    const i = +wl.dataset.to.split('-')[1];
    if (wl.dataset.from) {
      const j = +wl.dataset.from.split('-')[1];
      wl.textContent = layers[l][i].weights[j].toFixed(2);
    } else {
      wl.textContent = layers[l][i].bias.toFixed(2); // bias label
    }
  });
  layers[0].forEach(n => { n.valEl.textContent = n.value.toFixed(2); });
  computeForward(false);
  if (pinned) { const p = pinned; clearPinned(); onNodeClick(p.l, p.i); }
});

/* ---------- init ---------- */
buildNetwork();
