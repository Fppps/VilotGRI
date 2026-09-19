/**
 * RSL Engine with Natural Z-Score Confidence.
 * Operates purely on active dynamic vocabulary entries with session-only learning.
 */
class RLEngine {
  constructor(network, temperature = 1.05) {
    this.network = network;
    this.temperature = temperature;
    this.trajectory = [];
    this.episodeCount = 0;
    this.cumulativeReward = 0;
    this.lastSelfReward = 0;
    this.mode = "rsl"; 
    this.userTokenCountAverage = 16; 

    // Deep Learning controls
    this.isDeepLearning = true;
    this.turnCounter = 0;
  }

  setMode(newMode) {
    if (newMode === "rsl" || newMode === "user") {
      this.mode = newMode;
    }
  }

  setDeepLearning(enabled) {
    this.isDeepLearning = Boolean(enabled);
  }

  shouldUpdateDeepSynapses() {
    this.turnCounter++;
    // Deep learning updates every turn; periodic learning updates every fifth turn.
    return this.isDeepLearning || this.turnCounter % 5 === 0;
  }

  softmax(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }

    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      const e = Math.exp((logits[i] - max) / this.temperature);
      exps[i] = e;
      sum += e;
    }
    for (let i = 0; i < logits.length; i++) {
      exps[i] /= (sum || 1.0);
    }
    return exps;
  }

  sampleAction(probs) {
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return i;
    }
    return probs.length - 1;
  }

  async generate(promptTokens, onWordSampled = null) {
    this.trajectory = [];
    const generated = [];
    let state = [...promptTokens];

    this.userTokenCountAverage = Math.round(
      this.userTokenCountAverage * 0.8 + promptTokens.length * 0.2
    );

    // RSL: up to 360 tokens with prompt length adaptation; RL: 256 tokens max
    const maxTokens = this.mode === "rsl" ? 360 : 256;
    const minTokens = this.mode === "rsl" 
      ? Math.min(64, Math.max(12, this.userTokenCountAverage * 2))
      : 8;

    let totalZConfidence = 0;

    for (let step = 0; step < maxTokens; step++) {
      const logits = await this.network.forwardSequence(state);

      for (let i = 0; i < logits.length; i++) {
        if (!this.network.vocab || !this.network.vocab.hasToken(i)) {
          logits[i] = -1e9;
        }
      }

      let logitSum = 0;
      let logitSqSum = 0;
      let validCount = 0;

      for (let i = 0; i < logits.length; i++) {
        if (logits[i] > -1e8) {
          logitSum += logits[i];
          logitSqSum += logits[i] * logits[i];
          validCount++;
        }
      }

      const meanLogit = validCount > 0 ? (logitSum / validCount) : 0;
      const varLogit = validCount > 0 ? Math.max(1e-5, (logitSqSum / validCount) - (meanLogit * meanLogit)) : 1;
      const stdLogit = Math.sqrt(varLogit);

      const probs = this.softmax(logits);
      const actionToken = this.sampleAction(probs);
      const actionProb = probs[actionToken] || 0.0;

      // Calculate confidence naturally via logistic z-score
      const actionLogit = logits[actionToken] > -1e8 ? logits[actionToken] : meanLogit;
      const zScore = (actionLogit - meanLogit) / stdLogit;
      const tokenConfidence = 1.0 / (1.0 + Math.exp(-zScore));
      totalZConfidence += tokenConfidence;

      if (onWordSampled && this.network.vocab) {
        const wordString = this.network.vocab.idToToken[actionToken];
        if (wordString) onWordSampled(wordString, step);
      }

      const latent = this.network.currentLatent;
      for (let v = 0; v < this.network.vocabSize; v++) {
        const delta = (v === actionToken ? 1.0 : 0.0) - probs[v];
        const offset = v * this.network.poolDim;
        for (let p = 0; p < this.network.poolDim; p++) {
          this.network.headGradients[offset + p] += delta * latent[p];
        }
      }

      this.trajectory.push({ token: actionToken, prob: actionProb });
      generated.push(actionToken);
      state.push(actionToken);

      const word = this.network.vocab ? this.network.vocab.idToToken[actionToken] : null;
      if (step >= minTokens && word && /[.?!]$/.test(word)) {
        break;
      }
    }

    const avgConfidence = generated.length > 0 ? (totalZConfidence / generated.length) : 0.5;
    const confidencePct = (avgConfidence * 100).toFixed(1);

    if (this.mode === "rsl") {
      this.lastSelfReward = this.evaluateAndSelfLearn(promptTokens, generated);
    } else {
      this.lastSelfReward = 0.0;
    }

    return {
      tokens: generated,
      confidence: confidencePct
    };
  }

  evaluateAndSelfLearn(promptTokens, generated) {
    if (generated.length === 0) return 0;

    const uniqueTokens = new Set(generated);
    const diversityRatio = uniqueTokens.size / generated.length;

    let logProbSum = 0;
    for (const step of this.trajectory) {
      logProbSum += Math.log(step.prob + 1e-7);
    }
    const avgConfidence = Math.exp(logProbSum / this.trajectory.length);

    let selfReward = (diversityRatio - 0.5) * 0.4 + (avgConfidence - 0.2) * 0.2;
    if (diversityRatio < 0.4) selfReward -= 0.3;

    // Deep Learning switch applies strictly during RSL evaluation
    const updateDeep = this.shouldUpdateDeepSynapses();
    this.network.applyPolicyReward(selfReward, 0.004, updateDeep);
    return selfReward;
  }

  train(userRewardSignal) {
    const effectiveReturn = this.mode === "rsl" 
      ? userRewardSignal + (this.lastSelfReward * 0.5)
      : userRewardSignal;

    // In RL mode, updates remain on the policy head without triggering the RSL deep dispatch
    const updateDeep = (this.mode === "rsl") ? this.shouldUpdateDeepSynapses() : false;
    this.network.applyPolicyReward(effectiveReturn, 0.008, updateDeep);
    this.episodeCount++;
    this.cumulativeReward += effectiveReturn;
    this.trajectory = [];
    return effectiveReturn;
  }
}