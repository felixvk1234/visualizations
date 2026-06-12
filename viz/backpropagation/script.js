// ============ UTILITIES ============
const sigmoid = x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
const sigmoid_prime = x => {
  const s = sigmoid(x);
  return s * (1 - s);
};
const relu = x => Math.max(0, x);
const relu_prime = x => (x > 0 ? 1 : 0);
const tanh_fn = x => Math.tanh(x);
const tanh_prime = x => 1 - Math.tanh(x) ** 2;

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const rand = () => (Math.random() - 0.5) * 0.5;

// ============ SIMPLE NEURAL NETWORK (STATIC) ============
class SimpleNetwork {
  constructor(layerSizes = [1, 3, 3, 1]) {
    this.layers = layerSizes.length;
    this.layerSizes = layerSizes;
    this.weights = [];
    this.biases = [];
    this.activations = [];
    this.preActivations = [];
    this.initWeights();
    this.lossHistory = [];
  }

  initWeights() {
    this.weights = [];
    this.biases = [];
    this.activations = [];
    this.preActivations = [];
    this.deltas = [];
    this.gradients = [];
    for (let i = 0; i < this.layers - 1; i++) {
      const w = Array(this.layerSizes[i + 1])
        .fill(0)
        .map(() => Array(this.layerSizes[i]).fill(0).map(rand));
      const b = Array(this.layerSizes[i + 1]).fill(0).map(rand);
      this.weights.push(w);
      this.biases.push(b);
    }
  }

  forward(x) {
    this.activations = [x.slice()];
    this.preActivations = [];

    for (let l = 0; l < this.layers - 1; l++) {
      const z = this.weights[l].map((row, i) => {
        const sum = row.reduce((acc, w, j) => acc + w * x[j], 0) + this.biases[l][i];
        return sum;
      });
      this.preActivations.push(z);

      // Apply activation: ReLU for hidden, sigmoid for output
      if (l < this.layers - 2) {
        x = z.map(relu);
      } else {
        x = z.map(sigmoid);
      }
      this.activations.push(x);
    }

    return x;
  }

  backward(target, learningRate = 0.1) {
    const deltas = Array(this.layers - 1);
    const L = this.layers - 1;

    // Output layer delta
    deltas[L - 1] = this.activations[L].map((a, i) => {
      const dLda = a - target[i];
      const dadz = sigmoid_prime(this.preActivations[L - 1][i]);
      return dLda * dadz;
    });

    // Backprop through hidden layers
    for (let l = L - 2; l >= 0; l--) {
      const delta = Array(this.layerSizes[l + 1]).fill(0);
      const nextDelta = deltas[l + 1];

      for (let i = 0; i < this.layerSizes[l + 1]; i++) {
        let sum = 0;
        for (let j = 0; j < this.layerSizes[l + 2]; j++) {
          sum += this.weights[l + 1][j][i] * nextDelta[j];
        }
        delta[i] = sum * relu_prime(this.preActivations[l][i]);
      }
      deltas[l] = delta;
    }

    this.deltas = deltas.map(delta => delta.slice());
    this.gradients = this.weights.map((layer, l) =>
      layer.map((row, i) =>
        row.map((_, j) => deltas[l][i] * this.activations[l][j])
      )
    );

    // Update weights
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < this.weights[l].length; i++) {
        for (let j = 0; j < this.weights[l][i].length; j++) {
          const grad = deltas[l][i] * this.activations[l][j];
          this.weights[l][i][j] -= learningRate * grad;
        }
        this.biases[l][i] -= learningRate * deltas[l][i];
      }
    }
  }

  loss(output, target) {
    let l = 0;
    for (let i = 0; i < output.length; i++) {
      l += 0.5 * (output[i] - target[i]) ** 2;
    }
    return l;
  }
}

// ============ SIMPLE RNN (DYNAMIC) ============
class SimpleRNN {
  constructor(inputSize = 1, hiddenSize = 3, outputSize = 1) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // Weight matrices
    this.Wxh = Array(hiddenSize).fill(0).map(() => Array(inputSize).fill(0).map(rand)); // input -> hidden
    this.Whh = Array(hiddenSize).fill(0).map(() => Array(hiddenSize).fill(0).map(rand)); // hidden -> hidden
    this.Why = Array(outputSize).fill(0).map(() => Array(hiddenSize).fill(0).map(rand)); // hidden -> output

    this.bh = Array(hiddenSize).fill(0);
    this.by = Array(outputSize).fill(0);

    this.lossHistory = [];
    this.inputs = [];
    this.gradientMagnitudes = [];
  }

  forward(inputs) {
    const T = inputs.length;
    const h = [Array(this.hiddenSize).fill(0)]; // h_0 = zero
    const y = [];

    for (let t = 0; t < T; t++) {
      const ht = Array(this.hiddenSize).fill(0);
      for (let i = 0; i < this.hiddenSize; i++) {
        let sum = this.bh[i];
        for (let j = 0; j < this.inputSize; j++) {
          sum += this.Wxh[i][j] * inputs[t][j];
        }
        for (let j = 0; j < this.hiddenSize; j++) {
          sum += this.Whh[i][j] * h[t][j];
        }
        ht[i] = Math.tanh(sum);
      }
      h.push(ht);

      const yt = Array(this.outputSize).fill(0);
      for (let i = 0; i < this.outputSize; i++) {
        let sum = this.by[i];
        for (let j = 0; j < this.hiddenSize; j++) {
          sum += this.Why[i][j] * ht[j];
        }
        yt[i] = sigmoid(sum);
      }
      y.push(yt);
    }

    this.h = h;
    this.y = y;
    this.inputs = inputs.map(input => input.slice());
    return { h, y };
  }

  loss(outputs, targets) {
    let l = 0;
    for (let t = 0; t < outputs.length; t++) {
      for (let i = 0; i < outputs[t].length; i++) {
        l += 0.5 * (outputs[t][i] - targets[t][i]) ** 2;
      }
    }
    return l;
  }

  backward(targets, learningRate = 0.1) {
    const T = targets.length;
    this.gradientMagnitudes = Array(T).fill(0);
    const dWxh = Array(this.hiddenSize).fill(0).map(() => Array(this.inputSize).fill(0));
    const dWhh = Array(this.hiddenSize).fill(0).map(() => Array(this.hiddenSize).fill(0));
    const dWhy = Array(this.outputSize).fill(0).map(() => Array(this.hiddenSize).fill(0));
    const dbh = Array(this.hiddenSize).fill(0);
    const dby = Array(this.outputSize).fill(0);

    let dh_next = Array(this.hiddenSize).fill(0);
    let gradNorm = 0;

    for (let t = T - 1; t >= 0; t--) {
      // Output pre-activation gradient: MSE derivative through sigmoid.
      const dy = Array(this.outputSize).fill(0);
      for (let i = 0; i < this.outputSize; i++) {
        const y = this.y[t][i];
        dy[i] = (y - targets[t][i]) * y * (1 - y);
      }

      // Why gradients
      for (let i = 0; i < this.outputSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dWhy[i][j] += dy[i] * this.h[t + 1][j];
        }
        dby[i] += dy[i];
      }

      // Hidden state gradient
      const dh = dh_next.slice();
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.outputSize; j++) {
          dh[i] += dy[j] * this.Why[j][i];
        }
      }

      // Through tanh.
      const dhRaw = Array(this.hiddenSize).fill(0);
      for (let i = 0; i < this.hiddenSize; i++) {
        dhRaw[i] = dh[i] * (1 - this.h[t + 1][i] ** 2);
        gradNorm += dhRaw[i] ** 2;
      }
      this.gradientMagnitudes[t] = Math.sqrt(dhRaw.reduce((sum, value) => sum + value ** 2, 0));

      // Wxh, Whh gradients
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.inputSize; j++) {
          dWxh[i][j] += dhRaw[i] * this.inputs[t][j];
        }
        dbh[i] += dhRaw[i];
      }

      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dWhh[i][j] += dhRaw[i] * this.h[t][j];
        }
      }

      dh_next = Array(this.hiddenSize).fill(0);
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dh_next[i] += this.Whh[j][i] * dhRaw[j];
        }
      }
    }

    // Update parameters
    const lr = learningRate;
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.Wxh[i][j] -= lr * dWxh[i][j];
      }
      for (let j = 0; j < this.hiddenSize; j++) {
        this.Whh[i][j] -= lr * dWhh[i][j];
      }
      this.bh[i] -= lr * dbh[i];
    }

    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.Why[i][j] -= lr * dWhy[i][j];
      }
      this.by[i] -= lr * dby[i];
    }

    return Math.sqrt(gradNorm);
  }
}

// ============ VISUALIZATIONS ============
const SVG_NS = 'http://www.w3.org/2000/svg';

function appendSvg(svg, tag, attrs = {}, text = '') {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([name, value]) => element.setAttribute(name, value));
  if (text) element.textContent = text;
  svg.appendChild(element);
  return element;
}

function layerY(index, count, height, margin) {
  return count === 1
    ? height / 2
    : margin + (height - 2 * margin) * (index / (count - 1));
}

function drawStaticNetwork(svg, net, showGradients = false) {
  svg.innerHTML = '';

  const layers = net.layerSizes;
  const layerCount = layers.length;
  const height = 300;
  const width = 400;
  const margin = 42;
  const layerWidth = (width - 2 * margin) / (layerCount - 1);
  const allGradients = net.gradients.flat(2).map(Math.abs);
  const maxGradient = Math.max(...allGradients, 0.0001);

  for (let l = 0; l < layerCount - 1; l++) {
    for (let i = 0; i < layers[l]; i++) {
      for (let j = 0; j < layers[l + 1]; j++) {
        const gradient = Math.abs(net.gradients[l]?.[j]?.[i] || 0);
        const intensity = showGradients ? gradient / maxGradient : 0.35;
        appendSvg(svg, 'line', {
          x1: margin + l * layerWidth,
          y1: layerY(i, layers[l], height, margin),
          x2: margin + (l + 1) * layerWidth,
          y2: layerY(j, layers[l + 1], height, margin),
          stroke: showGradients ? '#ff6b6b' : '#5b8cff',
          'stroke-width': showGradients ? 0.8 + intensity * 3 : 1,
          opacity: showGradients ? 0.12 + intensity * 0.88 : 0.32,
        });
      }
    }
  }

  for (let l = 0; l < layerCount; l++) {
    for (let i = 0; i < layers[l]; i++) {
      const x = margin + l * layerWidth;
      const y = layerY(i, layers[l], height, margin);
      const value = showGradients
        ? (l === 0 ? 0 : net.deltas[l - 1]?.[i] || 0)
        : net.activations[l]?.[i] || 0;
      const intensity = clamp(Math.abs(value) * (showGradients ? 8 : 1), 0, 1);

      appendSvg(svg, 'circle', {
        cx: x,
        cy: y,
        r: 10,
        fill: showGradients ? '#ff6b6b' : '#5b8cff',
        stroke: showGradients ? '#ffc1c1' : '#b8caff',
        'stroke-width': 1,
        opacity: 0.25 + intensity * 0.75,
      });
      appendSvg(svg, 'text', {
        x,
        y: y + 22,
        'text-anchor': 'middle',
        'font-size': 9,
        fill: '#8b93a7',
      }, value.toFixed(2));
    }

    appendSvg(svg, 'text', {
      x: margin + l * layerWidth,
      y: 294,
      'text-anchor': 'middle',
      'font-size': 9,
      fill: '#8b93a7',
    }, l === 0 ? 'input' : l === layerCount - 1 ? 'output' : `hidden ${l}`);
  }
}

function drawDynamicNetwork(svg, net, sequenceLength, showGradients = false) {
  svg.innerHTML = '';

  const width = 500;
  const left = 42;
  const right = 42;
  const spacing = (width - left - right) / Math.max(1, sequenceLength - 1);
  const inputY = 275;
  const hiddenY = 170;
  const outputY = 62;
  const maxGradient = Math.max(...(net.gradientMagnitudes || []), 0.0001);

  appendSvg(svg, 'text', {
    x: 250,
    y: 20,
    'text-anchor': 'middle',
    'font-size': 11,
    fill: showGradients ? '#ff9c9c' : '#8b93a7',
  }, showGradients ? 'gradients flow backward through shared recurrent weights' : 'the same weights are reused at every timestep');

  for (let t = 0; t < sequenceLength; t++) {
    const x = left + spacing * t;
    const nextX = left + spacing * (t + 1);

    appendSvg(svg, 'line', {
      x1: x,
      y1: inputY - 18,
      x2: x,
      y2: hiddenY + 25,
      stroke: showGradients ? '#4a5268' : '#5b8cff',
      'stroke-width': 1.5,
      opacity: showGradients ? 0.35 : 0.7,
    });
    appendSvg(svg, 'line', {
      x1: x,
      y1: hiddenY - 25,
      x2: x,
      y2: outputY + 18,
      stroke: showGradients ? '#4a5268' : '#5b8cff',
      'stroke-width': 1.5,
      opacity: showGradients ? 0.35 : 0.7,
    });

    if (t < sequenceLength - 1) {
      const magnitude = net.gradientMagnitudes?.[t + 1] || 0;
      const intensity = magnitude / maxGradient;
      appendSvg(svg, 'line', {
        x1: showGradients ? nextX - 26 : x + 26,
        y1: hiddenY,
        x2: showGradients ? x + 26 : nextX - 26,
        y2: hiddenY,
        stroke: showGradients ? '#ff6b6b' : '#7c5cff',
        'stroke-width': showGradients ? 1 + intensity * 4 : 2,
        opacity: showGradients ? 0.2 + intensity * 0.8 : 0.8,
        'stroke-dasharray': showGradients ? '5 3' : 'none',
      });
    }

    const input = net.inputs[t]?.[0] ?? 0;
    const output = net.y[t]?.[0] ?? 0;
    const hidden = net.h[t + 1]
      ? net.h[t + 1].reduce((sum, value) => sum + value, 0) / net.hiddenSize
      : 0;
    const gradientIntensity = clamp((net.gradientMagnitudes?.[t] || 0) / maxGradient, 0, 1);

    appendSvg(svg, 'circle', {
      cx: x, cy: inputY, r: 18, fill: '#26345a', stroke: '#5b8cff', 'stroke-width': 1.5,
    });
    appendSvg(svg, 'circle', {
      cx: x, cy: hiddenY, r: 26,
      fill: showGradients ? '#5a2630' : '#30275a',
      stroke: showGradients ? '#ff6b6b' : '#7c5cff',
      'stroke-width': showGradients ? 1.5 + gradientIntensity * 2 : 2,
      opacity: showGradients ? 0.45 + gradientIntensity * 0.55 : 1,
    });
    appendSvg(svg, 'circle', {
      cx: x, cy: outputY, r: 18, fill: '#26345a', stroke: '#5b8cff', 'stroke-width': 1.5,
    });

    [
      [inputY + 4, `x=${input.toFixed(2)}`, 9],
      [hiddenY - 2, `h${t + 1}`, 11],
      [hiddenY + 11, `${net.hiddenSize} units`, 8],
      [outputY + 4, `y=${output.toFixed(2)}`, 9],
      [325, `t=${t + 1}`, 10],
    ].forEach(([y, text, size]) => appendSvg(svg, 'text', {
      x, y, 'text-anchor': 'middle', 'font-size': size, fill: '#e6e9f0',
    }, text));

    appendSvg(svg, 'text', {
      x,
      y: hiddenY + 40,
      'text-anchor': 'middle',
      'font-size': 8,
      fill: '#8b93a7',
    }, showGradients ? `|grad|=${(net.gradientMagnitudes?.[t] || 0).toFixed(3)}` : `mean=${hidden.toFixed(2)}`);
  }
}

// ============ INTERACTION ============
const staticCanvas = document.getElementById('staticNet');
const staticGradCanvas = document.getElementById('staticGradNet');
const dynamicCanvas = document.getElementById('dynamicNet');
const dynamicGradCanvas = document.getElementById('dynamicGradNet');
const STATIC_INPUT = [1];

let staticNet = new SimpleNetwork([1, 3, 1]);
let staticLossHistory = [];
let staticStepCount = 0;
let dynamicNet = new SimpleRNN(1, 3, 1);
let dynamicLossHistory = [];
let dynamicStepCount = 0;

function staticLayerSizes() {
  const layerCount = parseInt(document.getElementById('staticLayers').value);
  const hiddenSize = parseInt(document.getElementById('staticNeurons').value);
  return [1, ...Array(Math.max(0, layerCount - 2)).fill(hiddenSize), 1];
}

function resetStaticMetrics() {
  staticLossHistory = [];
  staticStepCount = 0;
  document.getElementById('staticLossValue').textContent = '—';
  document.getElementById('staticPredValue').textContent = '—';
  document.getElementById('staticStepCount').textContent = '0';
  document.getElementById('staticBackpropDetail').innerHTML = '';
  document.getElementById('staticCoachTitle').textContent = 'Ready to make a prediction';
  document.getElementById('staticCoach').innerHTML =
    'Click <b>Make a prediction</b>. The signal will travel from left to right through the blue network.';
  updateStaticChart();
}

function updateStaticComparison(prediction = null) {
  const target = parseFloat(document.getElementById('staticTarget').value);
  document.getElementById('staticTargetText').textContent = target.toFixed(2);
  document.getElementById('staticTargetBar').style.width = `${target * 100}%`;

  if (prediction === null) {
    document.getElementById('staticPredictionText').textContent = '—';
    document.getElementById('staticPredictionBar').style.width = '0%';
    document.getElementById('staticGapText').textContent = 'Make a prediction to see the mistake.';
    return;
  }

  const gap = Math.abs(target - prediction);
  document.getElementById('staticPredictionText').textContent = prediction.toFixed(3);
  document.getElementById('staticPredictionBar').style.width = `${clamp(prediction, 0, 1) * 100}%`;
  document.getElementById('staticGapText').textContent =
    gap < 0.02 ? 'Very close. The network has nearly reached your goal.'
      : `The prediction is ${gap.toFixed(3)} away from your goal.`;
}

function rebuildStaticNetwork() {
  staticNet = new SimpleNetwork(staticLayerSizes());
  staticNet.forward(STATIC_INPUT);
  resetStaticMetrics();
  updateStaticComparison();
  document.getElementById('staticForwardDetail').innerHTML = '';
  drawStaticNetwork(staticCanvas, staticNet);
  drawStaticNetwork(staticGradCanvas, staticNet, true);
}

function runStaticForward() {
  const output = staticNet.forward(STATIC_INPUT);
  const target = parseFloat(document.getElementById('staticTarget').value);
  const loss = staticNet.loss(output, [target]);
  drawStaticNetwork(staticCanvas, staticNet);
  updateStaticComparison(output[0]);
  document.getElementById('staticLossValue').textContent = loss.toFixed(4);
  document.getElementById('staticPredValue').textContent = output[0].toFixed(3);
  document.getElementById('staticCoachTitle').textContent = 'The network made a guess';
  document.getElementById('staticCoach').innerHTML =
    `Its prediction is <b>${output[0].toFixed(3)}</b>, while your goal is <b>${target.toFixed(2)}</b>. ` +
    'Now click <b>Learn from the mistake</b>.';
  document.getElementById('staticForwardDetail').innerHTML =
    `<strong>Forward pass:</strong> the input moved left to right through the network. No weights changed yet.`;
}

function confirmButtonAction(buttonId, label) {
  const button = document.getElementById(buttonId);
  button.originalLabel ||= button.textContent;
  button.textContent = label;
  button.classList.add('btn-confirmed');
  clearTimeout(button.confirmTimeout);
  button.confirmTimeout = setTimeout(() => {
    button.textContent = button.originalLabel;
    button.classList.remove('btn-confirmed');
  }, 700);
}

function trainStaticStep() {
  const target = parseFloat(document.getElementById('staticTarget').value);
  const learningRate = parseFloat(document.getElementById('staticLR').value);

  const before = staticNet.forward(STATIC_INPUT)[0];
  staticNet.backward([target], learningRate);
  const after = staticNet.forward(STATIC_INPUT)[0];
  const loss = staticNet.loss([after], [target]);
  staticLossHistory.push(loss);
  staticStepCount++;

  document.getElementById('staticLossValue').textContent = loss.toFixed(4);
  document.getElementById('staticPredValue').textContent = after.toFixed(3);
  document.getElementById('staticStepCount').textContent = staticStepCount;
  document.getElementById('staticCoachTitle').textContent =
    Math.abs(target - after) < Math.abs(target - before) ? 'The prediction moved closer' : 'The network made a small adjustment';
  document.getElementById('staticCoach').innerHTML =
    `Backpropagation assigned responsibility for the mistake and adjusted the connections. ` +
    `The prediction moved from <b>${before.toFixed(3)}</b> to <b>${after.toFixed(3)}</b>.`;
  updateStaticComparison(after);
  document.getElementById('staticForwardDetail').innerHTML =
    `<strong>Adjustment:</strong> prediction changed by ${(after - before).toFixed(6)}. ` +
    `Small changes are normal; learning happens through repetition.`;
  document.getElementById('staticBackpropDetail').innerHTML =
    `<strong>Update ${staticStepCount}:</strong> prediction ${before.toFixed(6)} → ${after.toFixed(6)} toward target ${target.toFixed(2)}.`;

  drawStaticNetwork(staticCanvas, staticNet);
  drawStaticNetwork(staticGradCanvas, staticNet, true);
  updateStaticChart();
  confirmButtonAction('staticBackprop', 'Updated weights');
}

function sequenceForTask(task, length) {
  if (task === 'counting') {
    const pattern = [0.15, 0.45, 0.75];
    return {
      inputs: Array.from({ length }, (_, t) => [pattern[t % pattern.length]]),
      targets: Array.from({ length }, (_, t) => [pattern[(t + 1) % pattern.length]]),
    };
  }

  if (task === 'addition') {
    const values = Array.from({ length }, (_, t) => 0.06 + (t % 3) * 0.03);
    let sum = 0;
    return {
      inputs: values.map(value => [value]),
      targets: values.map(value => {
        sum += value;
        return [sum];
      }),
    };
  }

  return {
    inputs: Array.from({ length }, (_, t) => [(Math.sin(t * 0.7) + 1) / 2]),
    targets: Array.from({ length }, (_, t) => [(Math.sin((t + 1) * 0.7) + 1) / 2]),
  };
}

function currentSequence() {
  const task = document.getElementById('dynamicTaskSelect').value;
  const length = parseInt(document.getElementById('dynamicSteps').value);
  return { task, ...sequenceForTask(task, length) };
}

function drawCurrentDynamic(showGradients = dynamicNet.gradientMagnitudes.length > 0) {
  const length = parseInt(document.getElementById('dynamicSteps').value);
  drawDynamicNetwork(dynamicCanvas, dynamicNet, length);
  drawDynamicNetwork(dynamicGradCanvas, dynamicNet, length, showGradients);
}

function resetDynamicMetrics() {
  dynamicLossHistory = [];
  dynamicStepCount = 0;
  document.getElementById('dynamicLossValue').textContent = '—';
  document.getElementById('dynamicGradNorm').textContent = '—';
  document.getElementById('dynamicStepCount').textContent = '0';
  document.getElementById('dynamicBPTTDetail').innerHTML = '';
  document.getElementById('dynamicCoachTitle').textContent = 'Ready to predict a sequence';
  document.getElementById('dynamicCoach').innerHTML =
    'Each purple memory circle receives the current input and a message from the previous timestep.';
  updateDynamicChart();
}

function rebuildDynamicNetwork() {
  const hiddenSize = parseInt(document.getElementById('dynamicHidden').value);
  dynamicNet = new SimpleRNN(1, hiddenSize, 1);
  const { inputs } = currentSequence();
  dynamicNet.forward(inputs);
  resetDynamicMetrics();
  drawCurrentDynamic(false);
}

function runDynamicForward() {
  const { task, inputs, targets } = currentSequence();
  const { y } = dynamicNet.forward(inputs);
  const loss = dynamicNet.loss(y, targets);
  drawCurrentDynamic(false);
  document.getElementById('dynamicLossValue').textContent = loss.toFixed(4);
  document.getElementById('dynamicCoachTitle').textContent = 'The network predicted every timestep';
  document.getElementById('dynamicCoach').innerHTML =
    `The purple memory moved forward through the sequence. The network's total mistake is <b>${loss.toFixed(4)}</b>. ` +
    'Now let feedback travel backward through time.';
  document.getElementById('dynamicBPTTDetail').innerHTML =
    `<strong>Predictions:</strong> ${y.map(v => v[0].toFixed(2)).join(' → ')}<br>` +
    `<strong>Goals:</strong> ${targets.map(v => v[0].toFixed(2)).join(' → ')}`;
}

function trainDynamicStep() {
  const { task, inputs, targets } = currentSequence();
  const learningRate = parseFloat(document.getElementById('dynamicLR').value);

  dynamicNet.forward(inputs);
  const gradNorm = dynamicNet.backward(targets, learningRate);
  const outputs = dynamicNet.forward(inputs).y;
  const loss = dynamicNet.loss(outputs, targets);
  dynamicLossHistory.push(loss);
  dynamicStepCount++;

  document.getElementById('dynamicLossValue').textContent = loss.toFixed(4);
  document.getElementById('dynamicGradNorm').textContent = gradNorm.toFixed(4);
  document.getElementById('dynamicStepCount').textContent = dynamicStepCount;
  document.getElementById('dynamicCoachTitle').textContent = 'The shared connections were adjusted';
  document.getElementById('dynamicCoach').innerHTML =
    `Feedback traveled from later timesteps toward earlier memories. All ${targets.length} timesteps helped update the same shared connections.`;
  document.getElementById('dynamicBPTTDetail').innerHTML =
    `<strong>Learning step ${dynamicStepCount}:</strong> sequence mistake is now ${loss.toFixed(4)}.`;

  drawCurrentDynamic(true);
  updateDynamicChart();
  confirmButtonAction('dynamicBPTT', 'Updated shared weights');
}

document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.mode-content').forEach(m => m.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.mode + 'Mode').classList.add('active');
    updateStaticChart();
    updateDynamicChart();
  });
});

document.getElementById('staticLayers').addEventListener('input', e => {
  document.getElementById('staticLayersOut').textContent = e.target.value;
  rebuildStaticNetwork();
});
document.getElementById('staticNeurons').addEventListener('input', e => {
  document.getElementById('staticNeuronsOut').textContent = e.target.value;
  rebuildStaticNetwork();
});
document.getElementById('staticForward').addEventListener('click', runStaticForward);
document.getElementById('staticBackprop').addEventListener('click', trainStaticStep);
document.getElementById('staticTrain').addEventListener('click', () => {
  for (let i = 0; i < 20; i++) trainStaticStep();
});
document.getElementById('staticReset').addEventListener('click', rebuildStaticNetwork);

document.getElementById('dynamicForward').addEventListener('click', runDynamicForward);
document.getElementById('dynamicBPTT').addEventListener('click', trainDynamicStep);
document.getElementById('dynamicTrain').addEventListener('click', () => {
  for (let i = 0; i < 5; i++) trainDynamicStep();
});
document.getElementById('dynamicReset').addEventListener('click', rebuildDynamicNetwork);
document.getElementById('dynamicTaskSelect').addEventListener('change', rebuildDynamicNetwork);

// ============ CHARTS ============
function drawLossChart(canvas, history, color) {
  canvas.width = 900;
  canvas.height = 250;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const pad = { left: 52, right: 20, top: 20, bottom: 32 };

  ctx.fillStyle = '#131826';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#232b40';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  ctx.fillStyle = '#8b93a7';
  ctx.font = '12px Inter, sans-serif';
  if (history.length === 0) {
    ctx.fillText('Repeat learning to see the mistake shrink.', pad.left, height / 2);
    return;
  }

  const maxLoss = Math.max(...history, 0.0001) * 1.08;
  ctx.fillText(maxLoss.toFixed(3), 8, pad.top + 4);
  ctx.fillText('0', 36, height - pad.bottom + 4);
  ctx.fillText(`step ${history.length}`, width - 72, height - 8);

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  history.forEach((loss, i) => {
    const x = pad.left + (i / Math.max(1, history.length - 1)) * (width - pad.left - pad.right);
    const y = height - pad.bottom - (loss / maxLoss) * (height - pad.top - pad.bottom);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (history.length === 1) ctx.lineTo(pad.left + 1, height - pad.bottom - (history[0] / maxLoss) * (height - pad.top - pad.bottom));
  ctx.stroke();
}

function updateStaticChart() {
  drawLossChart(document.getElementById('staticLossChart'), staticLossHistory, '#5b8cff');
}

function updateDynamicChart() {
  drawLossChart(document.getElementById('dynamicLossChart'), dynamicLossHistory, '#ff6b6b');
}

function drawGradientFlow() {
  const canvas = document.getElementById('gradientFlow');
  canvas.width = 900;
  canvas.height = 250;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const tau = parseFloat(document.getElementById('timeConstant').value);
  const steps = 16;
  const pad = 36;

  ctx.fillStyle = '#131826';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#232b40';
  for (let i = 0; i <= 4; i++) {
    const y = pad + ((height - pad * 2) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  ctx.strokeStyle = '#ff6b6b';
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let t = 0; t < steps; t++) {
    const magnitude = Math.exp(-(steps - 1 - t) / tau);
    const x = pad + (t / (steps - 1)) * (width - pad * 2);
    const y = height - pad - magnitude * (height - pad * 2);
    if (t === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = '#8b93a7';
  ctx.font = '12px Inter, sans-serif';
  ctx.fillText('early timestep', pad, height - 10);
  ctx.textAlign = 'right';
  ctx.fillText('loss / latest timestep', width - pad, height - 10);
  ctx.textAlign = 'left';
  ctx.fillText('gradient magnitude', pad, 18);
}

document.getElementById('visualizeGradient').addEventListener('click', drawGradientFlow);
document.getElementById('timeConstant').addEventListener('input', e => {
  document.getElementById('timeConstantOut').textContent = parseFloat(e.target.value).toFixed(1);
  drawGradientFlow();
});
document.getElementById('staticTarget').addEventListener('input', e => {
  document.getElementById('staticTargetOut').textContent = e.target.value;
  updateStaticComparison(staticNet.activations.at(-1)?.[0] ?? null);
});
document.getElementById('staticLR').addEventListener('input', e => {
  document.getElementById('staticLROut').textContent = e.target.value;
});
document.getElementById('dynamicLR').addEventListener('input', e => {
  document.getElementById('dynamicLROut').textContent = e.target.value;
});
document.getElementById('dynamicSteps').addEventListener('input', e => {
  document.getElementById('dynamicStepsOut').textContent = e.target.value;
  rebuildDynamicNetwork();
});
document.getElementById('dynamicHidden').addEventListener('input', e => {
  document.getElementById('dynamicHiddenOut').textContent = e.target.value;
  rebuildDynamicNetwork();
});

rebuildStaticNetwork();
rebuildDynamicNetwork();
drawGradientFlow();
