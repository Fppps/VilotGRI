/**
 * Sparse Hidden Layer representing 200,000 real neurons.
 * Low-memory architecture: K=4 synapses per neuron via pseudo-random prime dispersion.
 * Total parameters per layer: 800k weights + 200k biases = 1M float values (~4.8 MB).
 */
class Sparse200kLayer {
  constructor(layerIndex, prevSize = 200000, currentSize = 200000, fanIn = 4, isResidual = null) {
    this.layerIndex = layerIndex;
    this.size = currentSize; // 200,000 neurons
    this.prevSize = prevSize;
    this.fanIn = fanIn;
    // Layer 0 receives embedding input; Layers 1-14 enable residual skips
    this.isResidual = isResidual !== null ? isResidual : (layerIndex > 0);

    // 200,000 explicit neuron activations
    this.activations = new Float32Array(this.size);
    // Synaptic weights: fanIn (4) per neuron
    this.weights = new Float32Array(this.size * this.fanIn);
    this.biases = new Float32Array(this.size);

    // Eligibility traces for Policy Gradient updates
    this.weightGradients = new Float32Array(this.size * this.fanIn);

    this.initWeights();
  }

  initWeights() {
    const scale = Math.sqrt(2.0 / this.fanIn);
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = (Math.random() * 2 - 1) * scale;
    }
  }

  forward(prevActivations, promptContext = null, contextScale = 0.0) {
    const act = this.activations;
    const w = this.weights;
    const b = this.biases;
    const pSize = this.prevSize;
    const lIdx = this.layerIndex;
    const isRes = this.isResidual;

    let activeMean = 0;

    // Compute activation across all 200,000 neurons
    for (let i = 0; i < this.size; i++) {
      let sum = b[i];
      const baseWeightIdx = i * 4;

      // Deterministic prime hash connectivity matching WGSL shaders
      const inIdx0 = (i * 7919 + lIdx * 101) % pSize;
      const inIdx1 = (i * 7919 + 31 + lIdx * 101) % pSize;
      const inIdx2 = (i * 7919 + 67 + lIdx * 101) % pSize;
      const inIdx3 = (i * 7919 + 109 + lIdx * 101) % pSize;

      sum += w[baseWeightIdx]     * prevActivations[inIdx0];
      sum += w[baseWeightIdx + 1] * prevActivations[inIdx1];
      sum += w[baseWeightIdx + 2] * prevActivations[inIdx2];
      sum += w[baseWeightIdx + 3] * prevActivations[inIdx3];

      // Match the WebGPU in-kernel layer-7 prompt reinjection.
      if (
        this.layerIndex === 7 &&
        promptContext &&
        promptContext.length > 0 &&
        contextScale !== 0
      ) {
        const contextIdx = i % Math.min(64, promptContext.length);
        sum += contextScale * promptContext[contextIdx];
      }

      // LeakyReLU activation (slope 0.02)
      let val = sum > 0 ? sum : 0.02 * sum;

      // Residual skip preserves signal across 15 compute steps
      if (isRes && i < prevActivations.length) {
        val += 0.5 * prevActivations[i];
      }

      act[i] = val;
      activeMean += Math.abs(val);
    }

    return activeMean / this.size;
  }

  clearGradients() {
    this.weightGradients.fill(0);
  }

  accumulateGradient(neuronErrors, prevActivations) {
    const pSize = this.prevSize;
    const lIdx = this.layerIndex;

    for (let i = 0; i < neuronErrors.length; i++) {
      const err = neuronErrors[i];
      if (Math.abs(err) < 1e-6) continue;

      const baseWeightIdx = i * 4;
      const inIdx0 = (i * 7919 + lIdx * 101) % pSize;
      const inIdx1 = (i * 7919 + 31 + lIdx * 101) % pSize;
      const inIdx2 = (i * 7919 + 67 + lIdx * 101) % pSize;
      const inIdx3 = (i * 7919 + 109 + lIdx * 101) % pSize;

      this.weightGradients[baseWeightIdx]     += err * prevActivations[inIdx0];
      this.weightGradients[baseWeightIdx + 1] += err * prevActivations[inIdx1];
      this.weightGradients[baseWeightIdx + 2] += err * prevActivations[inIdx2];
      this.weightGradients[baseWeightIdx + 3] += err * prevActivations[inIdx3];
    }
  }

  applyReward(reward, lr = 0.008) {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] += lr * reward * this.weightGradients[i];
      // Synaptic weight decay
      this.weights[i] *= 0.9999;
      this.weightGradients[i] = 0;
    }
  }

  // Weight persistence helpers for IndexedDB
  getWeights() {
    return {
      weights: new Float32Array(this.weights),
      biases: new Float32Array(this.biases)
    };
  }

  setWeights(data) {
    if (data.weights && data.weights.length === this.weights.length) {
      this.weights.set(data.weights);
    }
    if (data.biases && data.biases.length === this.biases.length) {
      this.biases.set(data.biases);
    }
  }
}

if (typeof module !== "undefined") module.exports = Sparse200kLayer;