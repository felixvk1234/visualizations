/* Vision Transformer interactive explainer
   All patch visuals are CSS slices of dog.jpg; embedding bars and the
   attention demo are computed from the photo's real pixel statistics. */

const IMG_SRC = 'dog.jpg';

function bgPos(row, col, n) {
  const p = n > 1 ? 100 / (n - 1) : 0;
  return `${col * p}% ${row * p}%`;
}

/* ---------- Pipeline overview: mini patch grid + token row ---------- */
(function buildOverview() {
  const grid = document.getElementById('pipeGrid');
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.style.backgroundPosition = bgPos(r, c, 4);
      grid.appendChild(cell);
    }
  }

  const row = document.getElementById('pipeTokens');
  const cls = document.createElement('div');
  cls.className = 'tok cls';
  cls.title = '[CLS] token';
  row.appendChild(cls);
  for (let i = 0; i < 4; i++) {
    const t = document.createElement('div');
    t.className = 'tok';
    t.style.backgroundPosition = bgPos(Math.floor(i / 4), i % 4, 4);
    row.appendChild(t);
  }
  const dots = document.createElement('div');
  dots.className = 'tok dots';
  dots.textContent = '…';
  row.appendChild(dots);
  const last = document.createElement('div');
  last.className = 'tok';
  last.style.backgroundPosition = bgPos(3, 3, 4);
  row.appendChild(last);
})();

/* ---------- Step 1: patchify lab ---------- */
(function patchLab() {
  const stage = document.getElementById('patchStage');
  const slider = document.getElementById('gridSlider');
  const explodeBtn = document.getElementById('explodeBtn');
  const gridLabel = document.getElementById('gridLabel');
  const roSide = document.getElementById('roSide');
  const roCount = document.getElementById('roCount');
  const roPixels = document.getElementById('roPixels');

  function build(n) {
    stage.innerHTML = '';
    stage.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    stage.style.gridTemplateRows = `repeat(${n}, 1fr)`;
    const showIdx = n <= 8;
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const patch = document.createElement('div');
        patch.className = 'patch';
        patch.style.backgroundSize = `${n * 100}% ${n * 100}%`;
        patch.style.backgroundPosition = bgPos(r, c, n);
        if (showIdx) {
          const idx = document.createElement('span');
          idx.className = 'idx';
          idx.textContent = r * n + c + 1;
          patch.appendChild(idx);
        }
        stage.appendChild(patch);
      }
    }
    gridLabel.textContent = `${n} × ${n}`;
    roSide.textContent = n;
    roCount.textContent = `${n * n} tokens`;
    const px = 224 / n;
    const pxLabel = Number.isInteger(px) ? px : `≈${Math.round(px)}`;
    roPixels.textContent = `${pxLabel} × ${pxLabel}`;
  }

  slider.addEventListener('input', () => build(+slider.value));
  explodeBtn.addEventListener('click', () => {
    const on = stage.classList.toggle('exploded');
    explodeBtn.classList.toggle('active', on);
    explodeBtn.textContent = on ? 'Reassemble image' : 'Explode patches';
  });
  build(+slider.value);
})();

/* ---------- Pixel statistics (shared by embeddings + attention) ---------- */
function loadPatchStats(n) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const S = 224;
        const canvas = document.createElement('canvas');
        canvas.width = S;
        canvas.height = S;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, S, S);
        const data = ctx.getImageData(0, 0, S, S).data;
        const cell = S / n;
        const stats = [];
        for (let r = 0; r < n; r++) {
          for (let c = 0; c < n; c++) {
            let rs = 0, gs = 0, bs = 0, count = 0, lumVar = 0;
            const lums = [];
            for (let y = Math.floor(r * cell); y < Math.floor((r + 1) * cell); y += 2) {
              for (let x = Math.floor(c * cell); x < Math.floor((c + 1) * cell); x += 2) {
                const i = (y * S + x) * 4;
                rs += data[i]; gs += data[i + 1]; bs += data[i + 2];
                lums.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
                count++;
              }
            }
            const mr = rs / count, mg = gs / count, mb = bs / count;
            const mLum = lums.reduce((a, v) => a + v, 0) / lums.length;
            for (const l of lums) lumVar += (l - mLum) ** 2;
            stats.push({ r: mr, g: mg, b: mb, lum: mLum, tex: Math.sqrt(lumVar / lums.length), row: r, col: c });
          }
        }
        resolve(stats);
      } catch (e) {
        resolve(fallbackStats(n)); // canvas blocked (e.g. file://) — synthetic stats
      }
    };
    img.onerror = () => resolve(fallbackStats(n));
    img.src = IMG_SRC;
  });
}

function fallbackStats(n) {
  const stats = [];
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const cx = (c - (n - 1) / 2) / n, cy = (r - (n - 1) / 2) / n;
      const centre = Math.exp(-(cx * cx + cy * cy) * 6); // dog ≈ image centre
      stats.push({
        r: 90 + centre * 130, g: 110 + centre * 90, b: 60 + centre * 60,
        lum: 80 + centre * 120, tex: 20 + centre * 30, row: r, col: c,
      });
    }
  }
  return stats;
}

/* ---------- Step 2: embedding strip ---------- */
loadPatchStats(4).then((stats) => {
  const strip = document.getElementById('embedStrip');
  const N = 4;

  function vecBars(values) {
    const vec = document.createElement('div');
    vec.className = 'vec';
    for (const v of values) {
      const bar = document.createElement('i');
      const h = 5 + Math.abs(v) * 21;
      bar.style.height = `${h.toFixed(1)}px`;
      bar.style.background = v >= 0 ? '#6090ff' : '#ff6b72';
      vec.appendChild(bar);
    }
    return vec;
  }

  // Pseudo-embedding: 8 deterministic mixtures of the patch's colour/texture stats.
  function embed(s) {
    const r = s.r / 255 - 0.5, g = s.g / 255 - 0.5, b = s.b / 255 - 0.5;
    const l = s.lum / 255 - 0.5, t = s.tex / 80 - 0.5;
    return [r, g, b, l, t, r - g, g - b, (r + b) / 2 - g].map((v) => Math.max(-1, Math.min(1, v * 1.8)));
  }

  function card(label, statsOrNull) {
    const el = document.createElement('div');
    el.className = 'embed-card' + (statsOrNull ? '' : ' cls-card');
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    if (statsOrNull) {
      thumb.style.backgroundSize = `${N * 100}% ${N * 100}%`;
      thumb.style.backgroundPosition = bgPos(statsOrNull.row, statsOrNull.col, N);
    } else {
      thumb.innerHTML = '<span class="cls-chip">CLS</span>';
    }
    const pos = document.createElement('span');
    pos.className = 'pos';
    pos.textContent = label;
    el.appendChild(thumb);
    el.appendChild(vecBars(statsOrNull ? embed(statsOrNull) : [0.4, -0.3, 0.6, -0.5, 0.2, 0.5, -0.4, 0.3]));
    el.appendChild(pos);
    return el;
  }

  strip.appendChild(card('pos 0', null));
  for (let i = 0; i < 8; i++) strip.appendChild(card(`pos ${i + 1}`, stats[i]));
  const dots = document.createElement('span');
  dots.className = 'embed-ellipsis';
  dots.textContent = '…';
  strip.appendChild(dots);
  strip.appendChild(card('pos 16', stats[15]));
});

/* ---------- Step 3: attention lab ---------- */
loadPatchStats(7).then((stats) => {
  const grid = document.getElementById('attnGrid');
  const readout = document.getElementById('attnReadout');
  const N = 7;
  const patches = [];

  for (let i = 0; i < N * N; i++) {
    const p = document.createElement('div');
    p.className = 'apatch';
    p.style.backgroundPosition = bgPos(stats[i].row, stats[i].col, N);
    const w = document.createElement('span');
    w.className = 'w';
    p.appendChild(w);
    p.addEventListener('mouseenter', () => select(i));
    p.addEventListener('click', () => select(i));
    grid.appendChild(p);
    patches.push(p);
  }

  // One simulated attention head: similarity in colour + texture, mild distance penalty.
  function attention(q) {
    const s = stats[q];
    const scores = stats.map((k) => {
      const colour = Math.hypot(s.r - k.r, s.g - k.g, s.b - k.b) / 255;
      const texture = Math.abs(s.tex - k.tex) / 80;
      const dist = Math.hypot(s.row - k.row, s.col - k.col) / (N * Math.SQRT2);
      return -(colour * 6 + texture * 2.5 + dist * 1.8);
    });
    const max = Math.max(...scores);
    const exps = scores.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, v) => a + v, 0);
    return exps.map((v) => v / sum);
  }

  function select(q) {
    const w = attention(q);
    const wMax = Math.max(...w);
    const ranked = w.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    const topSet = new Set(ranked.slice(0, 6).map(([, i]) => i));

    grid.classList.add('active');
    patches.forEach((p, i) => {
      p.classList.toggle('query', i === q);
      p.classList.toggle('top', topSet.has(i) && i !== q);
      p.style.setProperty('--dim', i === q ? 0 : (1 - w[i] / wMax) * 0.88);
      p.querySelector('.w').textContent = `${(w[i] * 100).toFixed(0)}%`;
    });

    const top3 = ranked.filter(([, i]) => i !== q).slice(0, 3);
    readout.innerHTML =
      `<p>Query: <b>patch ${q + 1}</b> (row ${stats[q].row + 1}, column ${stats[q].col + 1}). ` +
      `Its strongest attention weights:</p>` +
      `<ol class="attn-top-list">` +
      top3.map(([v, i]) => `<li>Patch ${i + 1} — <b>${(v * 100).toFixed(1)}%</b></li>`).join('') +
      `</ol>`;
  }

  select(24); // start on the centre patch (the puppy's face)
});

/* ---------- Step 5: animated probability bars ---------- */
(function probBars() {
  const bars = document.querySelectorAll('#probBars .prob-bar');
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const bar of bars) {
        bar.querySelector('.prob-track i').style.width = `${bar.dataset.p}%`;
      }
      observer.disconnect();
    }
  }, { threshold: 0.4 });
  observer.observe(document.getElementById('probBars'));
})();
