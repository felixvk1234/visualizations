/* Diffusion networks interactive explainer
   Every noisy image is computed live from cat.jpg's real pixels using the
   DDPM closed form  x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps  with a cosine
   noise schedule. Noise draws are seeded, so the page is deterministic. */

const IMG_SRC = 'cat.jpg';
const S = 256;           // working resolution
const T = 1000;          // diffusion timesteps

/* ---------- Cosine noise schedule (Nichol & Dhariwal, 2021) ---------- */
function alphaBar(t, s = 0.008) {
  const f = Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2);
  return f * f;
}

/* ---------- Seeded Gaussian noise (mulberry32 + Box-Muller) ---------- */
function gaussianField(seed, n) {
  let a = seed >>> 0;
  const rand = () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rand(), 1e-12), v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

const EPS = gaussianField(42, S * S * 3);      // the "true" noise ε
const EPS_WRONG = gaussianField(1337, S * S * 3); // an untrained network's guess

/* ---------- Load the photo's pixels (with file:// fallback) ---------- */
function loadPixels() {
  return new Promise((resolve) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    canvas.width = S;
    canvas.height = S;
    const ctx = canvas.getContext('2d');
    const finish = () => {
      const data = ctx.getImageData(0, 0, S, S).data;
      const x0 = new Float32Array(S * S * 3); // scaled to [-1, 1]
      for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
        x0[j] = data[i] / 127.5 - 1;
        x0[j + 1] = data[i + 1] / 127.5 - 1;
        x0[j + 2] = data[i + 2] / 127.5 - 1;
      }
      resolve(x0);
    };
    img.onload = () => {
      try {
        ctx.drawImage(img, 0, 0, S, S);
        finish();
      } catch (e) {
        drawFallbackCat(ctx); // canvas tainted (e.g. file://) — synthetic cat
        finish();
      }
    };
    img.onerror = () => { drawFallbackCat(ctx); finish(); };
    img.src = IMG_SRC;
  });
}

/* Shapes drawn on canvas never taint it, so the demos still work offline. */
function drawFallbackCat(ctx) {
  ctx.fillStyle = '#e8b32a';                               // yellow backdrop
  ctx.fillRect(0, 0, S, S);
  ctx.fillStyle = '#c97f3d';                               // body
  ctx.beginPath(); ctx.ellipse(140, 165, 72, 52, 0, 0, 7); ctx.fill();
  ctx.beginPath(); ctx.ellipse(92, 110, 34, 32, 0, 0, 7); ctx.fill(); // head
  ctx.beginPath();                                          // ears
  ctx.moveTo(68, 92); ctx.lineTo(74, 62); ctx.lineTo(92, 84);
  ctx.moveTo(112, 86); ctx.lineTo(122, 60); ctx.lineTo(96, 80);
  ctx.fill();
  ctx.strokeStyle = '#c97f3d'; ctx.lineWidth = 13; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(205, 140); ctx.quadraticCurveTo(235, 95, 215, 62); ctx.stroke(); // tail
  ctx.fillStyle = '#f3e3c8';                                // chest
  ctx.beginPath(); ctx.ellipse(105, 150, 22, 30, 0.3, 0, 7); ctx.fill();
  ctx.fillStyle = '#274227';                                // eyes
  ctx.beginPath(); ctx.ellipse(82, 106, 4, 5, 0, 0, 7); ctx.ellipse(102, 106, 4, 5, 0, 0, 7); ctx.fill();
}

/* ---------- Renderers ---------- */
const work = document.createElement('canvas');
work.width = S;
work.height = S;
const workCtx = work.getContext('2d');
const workImg = workCtx.createImageData(S, S);

function paintNoisy(x0, t, targetCanvas) {
  const ab = alphaBar(t);
  const sig = Math.sqrt(ab), noi = Math.sqrt(1 - ab);
  const d = workImg.data;
  for (let j = 0, i = 0; j < x0.length; j += 3, i += 4) {
    d[i]     = (sig * x0[j]     + noi * EPS[j]     + 1) * 127.5;
    d[i + 1] = (sig * x0[j + 1] + noi * EPS[j + 1] + 1) * 127.5;
    d[i + 2] = (sig * x0[j + 2] + noi * EPS[j + 2] + 1) * 127.5;
    d[i + 3] = 255;
  }
  blit(targetCanvas);
}

function paintField(field, targetCanvas) { // visualize a noise field in grayscale
  const d = workImg.data;
  for (let j = 0, i = 0; j < field.length; j += 3, i += 4) {
    const v = (field[j] * 0.3 + field[j + 1] * 0.4 + field[j + 2] * 0.3) * 55 + 127.5;
    d[i] = d[i + 1] = d[i + 2] = v;
    d[i + 3] = 255;
  }
  blit(targetCanvas);
}

function paintRaw(px, targetCanvas) { // paint a [-1,1] pixel array directly
  const d = workImg.data;
  for (let j = 0, i = 0; j < px.length; j += 3, i += 4) {
    d[i]     = (px[j]     + 1) * 127.5;
    d[i + 1] = (px[j + 1] + 1) * 127.5;
    d[i + 2] = (px[j + 2] + 1) * 127.5;
    d[i + 3] = 255;
  }
  blit(targetCanvas);
}

function blit(targetCanvas) {
  workCtx.putImageData(workImg, 0, 0);
  const ctx = targetCanvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(work, 0, 0, targetCanvas.width, targetCanvas.height);
}

/* ===================================================================== */
loadPixels().then((x0) => {
  /* ---------- Pipeline overview thumbnails ---------- */
  document.querySelectorAll('.pipe-canvas').forEach((c) => {
    paintNoisy(x0, +c.dataset.t, c);
  });

  /* ---------- Step 1: forward-process lab ---------- */
  const noiseCanvas = document.getElementById('noiseCanvas');
  const tSlider = document.getElementById('tSlider');
  const tLabel = document.getElementById('tLabel');
  const roSignal = document.getElementById('roSignal');
  const roNoise = document.getElementById('roNoise');
  const roAlpha = document.getElementById('roAlpha');
  const roVisible = document.getElementById('roVisible');
  const mixSignal = document.getElementById('mixSignal');

  const schedCanvas = document.getElementById('scheduleCanvas');

  function drawSchedule(tNow) {
    const ctx = schedCanvas.getContext('2d');
    const W = schedCanvas.width, H = schedCanvas.height;
    const L = 52, R = 16, Tp = 14, B = 34; // margins
    ctx.clearRect(0, 0, W, H);
    const px = (t) => L + (t / T) * (W - L - R);
    const py = (v) => Tp + (1 - v) * (H - Tp - B);

    ctx.strokeStyle = '#263149';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const v of [0, 0.5, 1]) { ctx.moveTo(px(0), py(v)); ctx.lineTo(px(T), py(v)); }
    ctx.stroke();

    ctx.fillStyle = '#9aa4b8';
    ctx.font = '600 12px Inter, sans-serif';
    ctx.textAlign = 'right';
    for (const v of [0, 0.5, 1]) ctx.fillText(v.toFixed(1), L - 8, py(v) + 4);
    ctx.textAlign = 'center';
    for (const t of [0, 250, 500, 750, 1000]) ctx.fillText(t, px(t), H - 12);
    ctx.fillText('timestep t', (L + W - R) / 2, H - 0.5);
    ctx.save();
    ctx.translate(13, (Tp + H - B) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('ᾱ(t)', 0, 0);
    ctx.restore();

    ctx.strokeStyle = '#6090ff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let t = 0; t <= T; t += 5) {
      const x = px(t), y = py(alphaBar(t));
      t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#ffd166';
    ctx.beginPath();
    ctx.arc(px(tNow), py(alphaBar(tNow)), 6, 0, 7);
    ctx.fill();
  }

  function visibility(sigPct) {
    if (sigPct > 85) return 'Completely';
    if (sigPct > 60) return 'Easily';
    if (sigPct > 35) return 'Faintly';
    if (sigPct > 15) return 'Barely';
    return 'Gone';
  }

  function updateForward() {
    const t = +tSlider.value;
    const ab = alphaBar(t);
    const sigPct = Math.sqrt(ab) * 100;
    paintNoisy(x0, t, noiseCanvas);
    tLabel.textContent = `t = ${t}`;
    roSignal.textContent = `${sigPct.toFixed(0)}%`;
    roNoise.textContent = `${(Math.sqrt(1 - ab) * 100).toFixed(0)}%`;
    roAlpha.textContent = ab.toFixed(3);
    roVisible.textContent = visibility(sigPct);
    mixSignal.style.width = `${sigPct}%`;
    drawSchedule(t);
  }

  tSlider.addEventListener('input', updateForward);
  updateForward();

  /* ---------- Step 2: training lab ---------- */
  const TRAIN_T = 600;
  const trainInput = document.getElementById('trainInput');
  const trainPred = document.getElementById('trainPred');
  const trainRecon = document.getElementById('trainRecon');
  const trainSlider = document.getElementById('trainSlider');
  const trainLabel = document.getElementById('trainLabel');
  const lossLabel = document.getElementById('lossLabel');
  const lossBar = document.getElementById('lossBar');
  document.getElementById('trainTLabel').textContent = `t = ${TRAIN_T}`;

  paintNoisy(x0, TRAIN_T, trainInput);

  const epsHat = new Float32Array(S * S * 3);
  const recon = new Float32Array(S * S * 3);

  function updateTraining() {
    const p = +trainSlider.value / 100; // 0 = untrained, 1 = converged
    const ab = alphaBar(TRAIN_T);
    const sig = Math.sqrt(ab), noi = Math.sqrt(1 - ab);
    for (let j = 0; j < epsHat.length; j++) {
      epsHat[j] = p * EPS[j] + (1 - p) * EPS_WRONG[j]; // simulated prediction
      const xt = sig * x0[j] + noi * EPS[j];
      recon[j] = Math.max(-1, Math.min(1, (xt - noi * epsHat[j]) / sig));
    }
    paintField(epsHat, trainPred);
    paintRaw(recon, trainRecon);

    trainLabel.textContent = `${trainSlider.value}%`;
    const loss = (1 - p) * (1 - p); // ‖ε−ε̂‖² shrinks quadratically with the blend
    lossBar.style.width = `${Math.max(loss * 100, 1)}%`;
    lossLabel.textContent = loss > 0.5 ? 'high' : loss > 0.12 ? 'falling' : loss > 0.01 ? 'low' : '≈ 0';
    lossLabel.style.color = loss > 0.5 ? '#ff6b72' : loss > 0.12 ? '#ffd166' : '#55d68b';
  }

  trainSlider.addEventListener('input', updateTraining);
  updateTraining();

  /* ---------- Step 3: sampling animation ---------- */
  const sampleCanvas = document.getElementById('sampleCanvas');
  const sampleBtn = document.getElementById('sampleBtn');
  const sampleStep = document.getElementById('sampleStep');
  const sampleDone = document.getElementById('sampleDone');
  const sampleBar = document.getElementById('sampleBar');

  paintNoisy(x0, T, sampleCanvas);

  const DURATION = 6000; // ms for the full reverse trajectory
  let running = false;

  function runSample() {
    if (running) return;
    running = true;
    sampleBtn.disabled = true;
    const start = performance.now();

    function frame(now) {
      const u = Math.min((now - start) / DURATION, 1);
      // ease-in so the early steps (mostly noise) pass quickly on screen
      // and the structure-forming late steps get more screen time
      const t = Math.round(T * (1 - u * u));
      paintNoisy(x0, t, sampleCanvas);
      sampleStep.textContent = `t = ${t}`;
      sampleDone.textContent = `${T - t} / ${T}`;
      sampleBar.style.width = `${(1 - t / T) * 100}%`;
      if (u < 1) {
        requestAnimationFrame(frame);
      } else {
        running = false;
        sampleBtn.disabled = false;
        sampleBtn.textContent = '↻ Generate again';
      }
    }
    requestAnimationFrame(frame);
  }

  sampleBtn.addEventListener('click', runSample);
});
