/**
 * VilotNI 1.5 - High Performance / Lowest Time RLEngine
 *
 * Deep mode uses one neural forward, one-time latent reasoning, dual-centroid routing, a compact POS-balanced candidate pool, dynamic word-pattern scoring, reward-learned token/POS n-grams, universal grammar, quality-floor RSL refinement, adaptive generation length, and an optional 60-pass ceiling.
 * It spends a tiny amount of CPU work selecting a better token from the SAME
 * neural forward pass, suppresses morphological repetition, then exits once a clean prompt-anchored clause
 * has been formed. Low mode keeps the cheaper stochastic sampler.
 *
 * Goal:
 *   Deep = one fast neural pass + one-time latent setup + pruned decoding + only-as-needed RSL refinement + protected grammar quality floor.
 *   Low  = simple one-path sampling.
 */

class RLEngine {
  constructor(network) {
    this.network = network;
    this.trajectory = [];
    this.episodeCount = 0;
    this.cumulativeReward = 0;
    this.lastSelfReward = 0;
    this.mode = "rsl";
    this.isDeepLearning = true;
    this.turnCounter = 0;

    this.maxTokenCeiling = 360;
    this.normalizedEmbeddings = null;
    this.initNormalizedEmbeddings();

    // Lightweight RSL word-pattern memory. It learns token/POS bigrams and
    // trigrams from rewarded trajectories. This is generic sequence learning,
    // not prompt-specific or answer-template logic.
    this.tokenPatternMemory = new Map();
    this.posPatternMemory = new Map();
    this.maxPatternEntries = 4096;
    this.loadPatternMemory();

    // Conservative English-ish POS matrix. The matrix is a preference system,
    // not a language model replacement. Prompt anchoring + network logits still
    // choose the actual words.
    this.transitionMatrix = {
      "START->Det":        { legal: true, weight: 2.4 },
      "START->Adj":        { legal: true, weight: 3.2 },
      "START->Noun":       { legal: true, weight: 3.8 },
      "START->ProperNoun": { legal: true, weight: 4.0 },
      "START->Pronoun":    { legal: true, weight: 3.2 },
      "START->Num":        { legal: true, weight: 2.0 },

      "Det->Adj":          { legal: true, weight: 3.2 },
      "Det->Noun":         { legal: true, weight: 4.1 },
      "Det->ProperNoun":   { legal: true, weight: 3.6 },
      "Det->Num":          { legal: true, weight: 2.8 },

      "Adj->Adj":          { legal: true, weight: 1.4 },
      "Adj->Noun":         { legal: true, weight: 4.4 },
      "Adj->ProperNoun":   { legal: true, weight: 3.7 },
      "Adj->Prep":         { legal: true, weight: 1.8 },
      "Adj->Conj":         { legal: true, weight: 1.5 },

      // Noun compounds matter for prompts like "AI model chips".
      "Noun->Noun":        { legal: true, weight: 2.8 },
      "Noun->ProperNoun":  { legal: true, weight: 2.0 },
      "Noun->Modal":       { legal: true, weight: 3.8 },
      "Noun->Aux":         { legal: true, weight: 3.5 },
      "Noun->Verb":        { legal: true, weight: 4.2 },
      "Noun->Adv":         { legal: true, weight: 1.8 },
      "Noun->Prep":        { legal: true, weight: 2.2 },
      "Noun->Conj":        { legal: true, weight: 1.6 },

      "ProperNoun->Noun":  { legal: true, weight: 2.8 },
      "ProperNoun->Modal": { legal: true, weight: 3.8 },
      "ProperNoun->Aux":   { legal: true, weight: 3.5 },
      "ProperNoun->Verb":  { legal: true, weight: 4.2 },
      "ProperNoun->Prep":  { legal: true, weight: 2.2 },
      "ProperNoun->Adv":   { legal: true, weight: 1.7 },
      "ProperNoun->Conj":  { legal: true, weight: 1.6 },

      "Pronoun->Modal":    { legal: true, weight: 4.0 },
      "Pronoun->Aux":      { legal: true, weight: 3.8 },
      "Pronoun->Verb":     { legal: true, weight: 4.0 },
      "Pronoun->Adv":      { legal: true, weight: 1.5 },
      "Pronoun->Prep":     { legal: true, weight: 1.6 },
      "Pronoun->Conj":     { legal: true, weight: 1.4 },

      "Modal->Verb":       { legal: true, weight: 4.8 },
      "Modal->Aux":        { legal: true, weight: 2.8 },
      "Modal->Adv":        { legal: true, weight: 1.8 },
      "Aux->Verb":         { legal: true, weight: 3.4 },
      "Aux->Adj":          { legal: true, weight: 3.4 },
      "Aux->Noun":         { legal: true, weight: 2.8 },
      "Aux->ProperNoun":   { legal: true, weight: 2.8 },
      "Aux->Pronoun":      { legal: true, weight: 2.4 },
      "Aux->Det":          { legal: true, weight: 3.5 },
      "Aux->Prep":         { legal: true, weight: 3.0 },
      "Aux->Num":          { legal: true, weight: 2.2 },
      "Aux->Adv":          { legal: true, weight: 1.8 },
      "Aux->Conj":         { legal: true, weight: 1.2 },

      "Verb->Det":         { legal: true, weight: 3.0 },
      "Verb->Adj":         { legal: true, weight: 2.6 },
      "Verb->Noun":        { legal: true, weight: 4.0 },
      "Verb->ProperNoun":  { legal: true, weight: 3.6 },
      "Verb->Pronoun":     { legal: true, weight: 2.5 },
      "Verb->Adv":         { legal: true, weight: 2.2 },
      "Verb->Prep":        { legal: true, weight: 2.8 },
      "Verb->Conj":        { legal: true, weight: 1.2 },
      "Verb->Num":         { legal: true, weight: 2.2 },

      "Adv->Verb":         { legal: true, weight: 2.6 },
      "Adv->Adj":          { legal: true, weight: 2.0 },
      "Adv->Adv":          { legal: true, weight: 0.8 },
      "Adv->Prep":         { legal: true, weight: 1.3 },
      "Adv->Modal":        { legal: true, weight: 1.3 },
      "Adv->Aux":          { legal: true, weight: 1.3 },
      "Adv->Noun":         { legal: true, weight: 1.2 },
      "Adv->ProperNoun":   { legal: true, weight: 1.2 },
      "Adv->Conj":         { legal: true, weight: 1.0 },

      "Prep->Det":         { legal: true, weight: 3.6 },
      "Prep->Adj":         { legal: true, weight: 2.7 },
      "Prep->Noun":        { legal: true, weight: 4.1 },
      "Prep->ProperNoun":  { legal: true, weight: 4.1 },
      "Prep->Pronoun":     { legal: true, weight: 3.0 },
      "Prep->Num":         { legal: true, weight: 3.1 },

      "Conj->Det":         { legal: true, weight: 2.8 },
      "Conj->Adj":         { legal: true, weight: 2.0 },
      "Conj->Noun":        { legal: true, weight: 3.2 },
      "Conj->ProperNoun":  { legal: true, weight: 3.2 },
      "Conj->Pronoun":     { legal: true, weight: 3.0 },
      "Conj->Adv":         { legal: true, weight: 1.6 },
      "Conj->Verb":        { legal: true, weight: 1.3 },

      "Num->Noun":         { legal: true, weight: 3.7 },
      "Num->ProperNoun":   { legal: true, weight: 2.6 },
      "Num->Prep":         { legal: true, weight: 2.0 },
      "Num->Conj":         { legal: true, weight: 1.3 }
    };
  }

  initNormalizedEmbeddings() {
    if (!this.network || !this.network.embeddings) return;
    const vocabSize = this.network.vocabSize;
    const embDim = this.network.embDim;
    this.normalizedEmbeddings = new Float32Array(vocabSize * embDim);

    for (let i = 0; i < vocabSize; i++) {
      const off = i * embDim;
      let normSq = 0;
      for (let d = 0; d < embDim; d++) {
        const x = this.network.embeddings[off + d];
        normSq += x * x;
      }
      const invNorm = normSq > 0 ? 1.0 / Math.sqrt(normSq) : 1.0;
      for (let d = 0; d < embDim; d++) {
        this.normalizedEmbeddings[off + d] = this.network.embeddings[off + d] * invNorm;
      }
    }
  }

  setMode(newMode) {
    if (newMode === "rsl" || newMode === "user" || newMode === "rl") {
      this.mode = newMode;
    }
  }

  shouldUpdateDeepSynapses() {
    this.turnCounter++;
    return this.isDeepLearning || (this.turnCounter % 5 === 0);
  }

  getCanonicalType(tokenId) {
    const rawType = this.network.vocab?.idToType?.[tokenId];
    if (!rawType) return "Other";
    if (rawType.includes("Modal")) return "Modal";
    if (rawType.includes("Auxiliary")) return "Aux";
    if (rawType.includes("Verb")) return "Verb";
    if (rawType.includes("Proper Noun")) return "ProperNoun";
    if (rawType.includes("Noun")) return "Noun";
    if (rawType.includes("Pronoun")) return "Pronoun";
    if (rawType.includes("Adjective")) return "Adj";
    if (rawType.includes("Adverb")) return "Adv";
    if (rawType.includes("Determiner")) return "Det";
    if (rawType.includes("Preposition")) return "Prep";
    if (rawType.includes("Conjunction")) return "Conj";
    if (rawType.includes("Punctuation")) return "Punct";
    if (rawType.includes("Number")) return "Num";
    return rawType;
  }

  stemWord(word) {
    return String(word || "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "")
      .replace(/(ing|edly|edly|ed|es|s)$/i, "")
      .slice(0, 8);
  }

  isContentType(type) {
    return ["Noun", "ProperNoun", "Verb", "Adj", "Adv", "Num"].includes(type);
  }

  loadPatternMemory() {
    if (typeof localStorage === "undefined") return;

    try {
      const raw = localStorage.getItem("vilotni15_rsl_word_patterns_v1");
      if (!raw) return;

      const parsed = JSON.parse(raw);

      for (const [k, v] of parsed.token || []) {
        if (Number.isFinite(v)) this.tokenPatternMemory.set(k, v);
      }

      for (const [k, v] of parsed.pos || []) {
        if (Number.isFinite(v)) this.posPatternMemory.set(k, v);
      }
    } catch (_) {
      // Persistence is optional. Generation must never depend on storage.
    }
  }

  savePatternMemory() {
    if (typeof localStorage === "undefined") return;

    try {
      localStorage.setItem(
        "vilotni15_rsl_word_patterns_v1",
        JSON.stringify({
          token: Array.from(this.tokenPatternMemory.entries()),
          pos: Array.from(this.posPatternMemory.entries())
        })
      );
    } catch (_) {
      // Ignore quota/privacy-mode failures.
    }
  }

  trimPatternMap(map) {
    if (map.size <= this.maxPatternEntries) return;

    const entries = Array.from(map.entries())
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, this.maxPatternEntries);

    map.clear();
    for (const [k, v] of entries) map.set(k, v);
  }

  patternMemoryBonus(generated, candidateToken) {
    let bonus = 0.0;
    const n = generated.length;

    if (n >= 1) {
      const tokenBigram = `${generated[n - 1]}>${candidateToken}`;
      bonus += (this.tokenPatternMemory.get(tokenBigram) || 0) * 0.75;

      const a = this.getCanonicalType(generated[n - 1]);
      const b = this.getCanonicalType(candidateToken);
      bonus += (this.posPatternMemory.get(`${a}>${b}`) || 0) * 0.55;
    }

    if (n >= 2) {
      const tokenTrigram =
        `${generated[n - 2]}>${generated[n - 1]}>${candidateToken}`;
      bonus += (this.tokenPatternMemory.get(tokenTrigram) || 0) * 1.05;

      const a = this.getCanonicalType(generated[n - 2]);
      const b = this.getCanonicalType(generated[n - 1]);
      const c = this.getCanonicalType(candidateToken);
      bonus += (
        this.posPatternMemory.get(`${a}>${b}>${c}`) || 0
      ) * 0.80;
    }

    return Math.max(-4.0, Math.min(4.0, bonus));
  }

  learnTrajectoryPatterns(reward) {
    if (!this.trajectory?.length || !Number.isFinite(reward)) return;

    const signed = Math.max(-1, Math.min(1, reward));
    if (Math.abs(signed) < 0.05) return;

    const tokens = this.trajectory
      .map(x => x?.token)
      .filter(Number.isFinite);

    if (tokens.length < 2) return;

    const step = signed * 0.20;

    const update = (map, key, delta) => {
      const next = Math.max(
        -2.5,
        Math.min(4.0, (map.get(key) || 0) + delta)
      );
      map.set(key, next);
    };

    for (let i = 1; i < tokens.length; i++) {
      update(
        this.tokenPatternMemory,
        `${tokens[i - 1]}>${tokens[i]}`,
        step * 0.60
      );

      const p0 = this.getCanonicalType(tokens[i - 1]);
      const p1 = this.getCanonicalType(tokens[i]);

      update(
        this.posPatternMemory,
        `${p0}>${p1}`,
        step * 0.45
      );

      if (i >= 2) {
        update(
          this.tokenPatternMemory,
          `${tokens[i - 2]}>${tokens[i - 1]}>${tokens[i]}`,
          step
        );

        const p2 = this.getCanonicalType(tokens[i - 2]);

        update(
          this.posPatternMemory,
          `${p2}>${p0}>${p1}`,
          step * 0.80
        );
      }
    }

    this.trimPatternMap(this.tokenPatternMemory);
    this.trimPatternMap(this.posPatternMemory);
    this.savePatternMemory();
  }

  buildLocalPatternContext(generated) {
    const recent = generated.slice(-32);
    const seenBigrams = new Set();
    const seenTrigrams = new Set();
    const contentTokens = [];

    let maxFunctionRun = 0;
    let functionRun = 0;

    for (let i = 0; i < recent.length; i++) {
      const tok = recent[i];
      const type = this.getCanonicalType(tok);

      if (this.isContentType(type)) {
        contentTokens.push(tok);
      }

      const isFunction =
        ["Det", "Prep", "Conj", "Modal", "Aux"].includes(type);

      if (isFunction) {
        functionRun++;
        maxFunctionRun = Math.max(maxFunctionRun, functionRun);
      } else if (type !== "Punct") {
        functionRun = 0;
      }

      if (i >= 1) {
        seenBigrams.add(`${recent[i - 1]}>${tok}`);
      }

      if (i >= 2) {
        seenTrigrams.add(
          `${recent[i - 2]}>${recent[i - 1]}>${tok}`
        );
      }
    }

    let recentContentCentroid = null;

    if (contentTokens.length && this.normalizedEmbeddings) {
      const embDim = this.network.embDim;
      recentContentCentroid = new Float32Array(embDim);

      const tail = contentTokens.slice(-5);

      for (const tok of tail) {
        const off = (tok % this.network.vocabSize) * embDim;
        for (let d = 0; d < embDim; d++) {
          recentContentCentroid[d] += this.normalizedEmbeddings[off + d];
        }
      }

      for (let d = 0; d < embDim; d++) {
        recentContentCentroid[d] /= tail.length;
      }

      this.normalizeVector(recentContentCentroid);
    }

    return {
      seenBigrams,
      seenTrigrams,
      recentContentCentroid,
      maxFunctionRun
    };
  }

  dynamicWordPatternScore(
    candidateToken,
    generated,
    contextModel,
    sentenceState,
    patternContext
  ) {
    const type = this.getCanonicalType(candidateToken);
    const n = generated.length;
    let score = this.patternMemoryBonus(generated, candidateToken);

    // Prevent exact local phrase loops before they are emitted.
    if (n >= 1) {
      const bigram = `${generated[n - 1]}>${candidateToken}`;
      if (patternContext.seenBigrams.has(bigram)) score -= 6.5;
    }

    if (n >= 2) {
      const trigram =
        `${generated[n - 2]}>${generated[n - 1]}>${candidateToken}`;
      if (patternContext.seenTrigrams.has(trigram)) score -= 13.0;
    }

    // Keep content words on the same local semantic thread.
    if (
      this.isContentType(type) &&
      patternContext.recentContentCentroid
    ) {
      const localSimilarity = this.dotTokenVector(
        candidateToken,
        patternContext.recentContentCentroid
      );

      if (localSimilarity > 0.10) {
        score += Math.min(1.8, localSimilarity * 2.2);
      } else if (
        localSimilarity < -0.10 &&
        sentenceState.predicateReady
      ) {
        score -= Math.min(2.4, Math.abs(localSimilarity) * 2.8);
      }
    }

    // Once a proposition is complete, endless grammar-glue continuations are
    // less useful than finishing the sentence.
    if (
      this.clauseComplete(sentenceState) &&
      ["Conj", "Prep", "Adv"].includes(type)
    ) {
      score -= Math.max(
        0,
        patternContext.maxFunctionRun - 1
      ) * 0.8;
    }

    return score;
  }

  normalizeVector(vec) {
    let normSq = 0;
    for (let i = 0; i < vec.length; i++) normSq += vec[i] * vec[i];
    if (normSq <= 1e-12) return vec;

    const inv = 1.0 / Math.sqrt(normSq);
    for (let i = 0; i < vec.length; i++) vec[i] *= inv;
    return vec;
  }

  blendVectors(a, b, wa = 0.5, wb = 0.5) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < out.length; i++) {
      out[i] = a[i] * wa + b[i] * wb;
    }
    return this.normalizeVector(out);
  }

  dotTokenVector(tokenId, vector) {
    if (!vector || !this.normalizedEmbeddings) return 0.0;
    const embDim = this.network.embDim;
    const off = (tokenId % this.network.vocabSize) * embDim;

    let dot = 0;
    for (let d = 0; d < embDim; d++) {
      dot += this.normalizedEmbeddings[off + d] * vector[d];
    }
    return Math.max(-1.0, Math.min(1.0, dot));
  }

  /**
   * Dual-Centroid Semantic Routing
   *
   * topicCentroid  = entities / concepts / descriptive content
   * intentCentroid = actions / relations / requested operation
   *
   * The routing weights change with clause state instead of collapsing the
   * entire prompt into one average semantic vector.
   */
  getRoutedSemanticScore(tokenId, contextModel, sentenceState = null) {
    const topic = contextModel?.topicScores?.[tokenId]
      ?? contextModel?.semanticScores?.[tokenId]
      ?? 0.0;

    const intent = contextModel?.intentScores?.[tokenId]
      ?? contextModel?.semanticScores?.[tokenId]
      ?? 0.0;

    let topicWeight = 0.60;
    let intentWeight = 0.40;

    if (sentenceState) {
      if (!sentenceState.subjectReady) {
        // Find the thing we are talking about.
        topicWeight = 0.72;
        intentWeight = 0.28;
      } else if (!sentenceState.predicateReady) {
        // Once a subject exists, aggressively route toward the action/relation.
        topicWeight = 0.25;
        intentWeight = 0.75;
      } else {
        // After the predicate, route back toward useful content/complements.
        topicWeight = 0.64;
        intentWeight = 0.36;
      }
    }

    const dual = topic * topicWeight + intent * intentWeight;

    // Latent reasoning is a third signal, not a replacement for the two
    // explicit centroids. This keeps routing interpretable and stable.
    const latent = contextModel?.reasoningScores?.[tokenId];
    if (Number.isFinite(latent)) {
      return Math.max(-1.0, Math.min(1.0, dual * 0.68 + latent * 0.32));
    }

    return Math.max(-1.0, Math.min(1.0, dual));
  }

  /**
   * Latent Chain-of-Thought Reasoning
   *
   * No textual chain is generated or exposed. Deep mode performs several tiny
   * 64-dimensional refinement hops over a compact candidate neighborhood from
   * the SAME neural forward pass. Each hop updates an internal reasoning
   * centroid, then generation uses that latent state for semantic routing.
   *
   * This adds reasoning depth without adding another 3M-neuron WebGPU pass.
   */
  runLatentReasoning(rawLogits, contextModel, maxHops = 4) {
    const vocab = this.network.vocab;
    const vocabSize = this.network.vocabSize;
    const embDim = this.network.embDim;

    if (!rawLogits || !contextModel || !this.normalizedEmbeddings) {
      return {
        centroid: contextModel?.centroid || null,
        scores: null,
        stability: 0.0,
        hopsUsed: 0
      };
    }

    const topicCentroid = contextModel.topicCentroid || contextModel.centroid;
    const intentCentroid = contextModel.intentCentroid || contextModel.centroid;

    let state = this.blendVectors(topicCentroid, intentCentroid, 0.58, 0.42);

    // Build a compact reasoning neighborhood once. Raw logits provide neural
    // evidence; the two centroids provide semantic evidence.
    const pool = [];
    let maxRaw = -Infinity;
    for (let i = 0; i < rawLogits.length; i++) {
      if (rawLogits[i] > maxRaw) maxRaw = rawLogits[i];
    }

    for (let i = 0; i < vocabSize; i++) {
      if (!vocab?.hasToken?.(i)) continue;

      const type = this.getCanonicalType(i);
      if (type === "Punct" || type === "Other") continue;

      const topic = contextModel.topicScores?.[i] ?? 0.0;
      const intent = contextModel.intentScores?.[i] ?? 0.0;
      const semantic = Math.max(topic, intent, (topic + intent) * 0.5);

      const neuralSupport = Math.exp(
        Math.max(-8.0, Math.min(0.0, rawLogits[i] - maxRaw))
      );

      const rankScore = semantic * 2.1 + neuralSupport * 1.4;

      if (pool.length < 48) {
        pool.push({ token: i, rankScore, neuralSupport, topic, intent });
        pool.sort((a, b) => b.rankScore - a.rankScore);
      } else if (rankScore > pool[pool.length - 1].rankScore) {
        pool[pool.length - 1] = {
          token: i,
          rankScore,
          neuralSupport,
          topic,
          intent
        };
        pool.sort((a, b) => b.rankScore - a.rankScore);
      }
    }

    if (!pool.length) {
      return {
        centroid: state,
        scores: null,
        stability: 1.0,
        hopsUsed: 0
      };
    }

    let stability = 0.0;
    let hopsUsed = 0;

    for (let hop = 0; hop < maxHops; hop++) {
      const accumulator = new Float32Array(embDim);
      let weightSum = 0.0;

      for (const item of pool) {
        const stateSim = this.dotTokenVector(item.token, state);
        const dualSupport = item.topic * 0.55 + item.intent * 0.45;

        // A bounded attention-like weight. It is intentionally cheap and uses
        // no textual intermediate reasoning.
        const logWeight =
          stateSim * 1.55 +
          dualSupport * 1.05 +
          Math.log(item.neuralSupport + 1e-5);

        const weight = Math.exp(Math.max(-8.0, Math.min(5.0, logWeight)));
        const off = item.token * embDim;

        for (let d = 0; d < embDim; d++) {
          accumulator[d] += this.normalizedEmbeddings[off + d] * weight;
        }
        weightSum += weight;
      }

      if (weightSum > 0) {
        for (let d = 0; d < embDim; d++) accumulator[d] /= weightSum;
        this.normalizeVector(accumulator);
      }

      const nextState = new Float32Array(embDim);
      for (let d = 0; d < embDim; d++) {
        nextState[d] =
          state[d] * 0.50 +
          accumulator[d] * 0.28 +
          topicCentroid[d] * 0.13 +
          intentCentroid[d] * 0.09;
      }
      this.normalizeVector(nextState);

      let dot = 0;
      for (let d = 0; d < embDim; d++) dot += state[d] * nextState[d];
      stability = Math.max(-1.0, Math.min(1.0, dot));

      state = nextState;
      hopsUsed = hop + 1;

      // The latent thought has converged. Do not burn CPU on redundant hops.
      if (hop >= 1 && stability >= 0.9985) break;
    }

    // One full vocabulary projection after the latent chain converges.
    const reasoningScores = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      if (!vocab?.hasToken?.(i)) continue;
      reasoningScores[i] = this.dotTokenVector(i, state);
    }

    return {
      centroid: state,
      scores: reasoningScores,
      stability,
      hopsUsed
    };
  }

  tokenWord(tokenId) {
    return this.network.vocab?.idToToken?.[tokenId] || "";
  }

  sentenceCaseToken(tokenId) {
    const word = this.tokenWord(tokenId);
    if (!word || !/^[a-z]/.test(word)) return tokenId;
    const title = word.charAt(0).toUpperCase() + word.slice(1);
    const alt = this.network.vocab?.tokenToId?.get?.(title);
    return Number.isFinite(alt) ? alt : tokenId;
  }

  /**
   * Structural question analysis.
   *
   * No hardcoded prompts and no "how -> by" / "why -> because" tables.
   * The parser only detects that the input is interrogative and identifies
   * which token behaves like the query operator. Its meaning is left to the
   * learned embeddings, dual centroids, latent reasoning, and RSL search.
   */
  analyzePromptStructure(cleanPromptTokens) {
    const words = cleanPromptTokens.map(t => this.tokenWord(t));
    const types = cleanPromptTokens.map(t => this.getCanonicalType(t));

    const questionMarkId = this.network.vocab?.tokenToId?.get?.("?");
    const hasQuestionMark = Number.isFinite(questionMarkId)
      ? cleanPromptTokens.includes(questionMarkId)
      : words.includes("?");

    // Interrogative inversion is structural: an initial function/content token
    // followed shortly by an auxiliary/modal is usually a question operator.
    let operatorIndex = -1;
    let helperIndex = -1;

    for (let i = 0; i < Math.min(types.length, 5); i++) {
      if (["Aux", "Modal"].includes(types[i])) {
        helperIndex = i;
        break;
      }
    }

    if (helperIndex > 0) {
      for (let i = 0; i < helperIndex; i++) {
        if (["Pronoun", "Adv", "Adj", "Det", "Noun", "ProperNoun"].includes(types[i])) {
          operatorIndex = i;
          break;
        }
      }
    }

    const startsWithHelper = ["Aux", "Modal"].includes(types[0]);
    const isQuestion = hasQuestionMark || startsWithHelper || operatorIndex >= 0;

    return {
      isQuestion,
      hasQuestionMark,
      operatorIndex,
      operatorToken: operatorIndex >= 0 ? cleanPromptTokens[operatorIndex] : null,
      helperIndex,
      helperToken: helperIndex >= 0 ? cleanPromptTokens[helperIndex] : null
    };
  }

  /**
   * Generic answer pressure for interrogative prompts.
   *
   * The query operator's learned embedding contributes through the intent
   * centroid. Generation is simply discouraged from echoing that operator and
   * rewarded for adding novel, semantically routed content after a predicate.
   */
  questionRelationBonus(tokenId, sentenceState, contextModel) {
    const profile = contextModel?.questionProfile;
    if (!profile?.isQuestion) return 0.0;

    if (tokenId === profile.operatorToken) {
      return sentenceState.isStart ? -16.0 : -4.0;
    }

    const type = this.getCanonicalType(tokenId);
    const exactPrompt = contextModel?.promptTokenSet?.has(tokenId);

    if (sentenceState.predicateReady && this.isContentType(type) && !exactPrompt) {
      const routed = this.getRoutedSemanticScore(tokenId, contextModel, sentenceState);
      return Math.max(0.0, routed) * 2.6;
    }

    return 0.0;
  }

  countNovelContent(tokens, contextModel) {
    let count = 0;
    for (const tok of tokens) {
      const type = this.getCanonicalType(tok);
      if (!this.isContentType(type)) continue;

      const word = this.tokenWord(tok);
      const stem = this.stemWord(word);
      const exact = contextModel?.promptContentSet?.has(tok);
      const stemMatch = stem.length >= 3 && contextModel?.promptStems?.has(stem);

      if (!exact && !stemMatch) count++;
    }
    return count;
  }

  answerRelationSatisfied(tokens, sentenceState, contextModel) {
    if (!this.clauseComplete(sentenceState)) return false;

    const profile = contextModel?.questionProfile;
    if (!profile?.isQuestion) return true;

    // A question answer should contribute something beyond simply parroting
    // the prompt. The learned semantic router decides WHAT that contribution
    // is; this function only checks that one exists.
    const novelContent = this.countNovelContent(tokens, contextModel);
    const contentCount = tokens.reduce((count, tok) => {
      return count + (this.isContentType(this.getCanonicalType(tok)) ? 1 : 0);
    }, 0);

    const requiredNovel = Math.max(
      1,
      Math.min(3, Math.ceil(contentCount * 0.18))
    );

    return novelContent >= requiredNovel;
  }

  resetSentenceState(sentenceState) {
    sentenceState.isStart = true;
    sentenceState.subjectReady = false;
    sentenceState.predicateReady = false;
    sentenceState.subjectToken = null;
    sentenceState.verbCount = 0;
    sentenceState.nounCount = 0;
    sentenceState.objectCount = 0;
    sentenceState.prepCount = 0;
    sentenceState.isPluralSubject = false;
  }

  inferRequestedLength(cleanPromptTokens) {
    const words = cleanPromptTokens.map(t => this.tokenWord(t).toLowerCase());

    for (let i = 0; i < words.length; i++) {
      if (!/^\d+$/.test(words[i])) continue;

      const n = Number(words[i]);
      if (!Number.isFinite(n) || n <= 0) continue;

      const nearby = words.slice(Math.max(0, i - 2), Math.min(words.length, i + 3));
      if (nearby.some(w => ["word", "words", "token", "tokens"].includes(w))) {
        return Math.max(1, Math.min(this.maxTokenCeiling, Math.round(n)));
      }
    }

    return null;
  }

  estimateAnswerLength(cleanPromptTokens, contextModel) {
    const explicit = this.inferRequestedLength(cleanPromptTokens);
    if (explicit !== null) return explicit;

    const promptLen = Math.max(1, cleanPromptTokens.length);
    const contentCount = contextModel?.promptContentSet?.size || 0;

    let structuralSignals = 0;

    for (const tok of cleanPromptTokens) {
      const type = this.getCanonicalType(tok);
      if (["Conj", "Prep", "Aux", "Modal"].includes(type)) {
        structuralSignals++;
      }
    }

    // Compact by default. Complexity grows the target continuously.
    let target =
      6 +
      Math.sqrt(promptLen) * 1.9 +
      contentCount * 1.15 +
      Math.sqrt(structuralSignals + 1) * 1.8;

    if (contextModel?.questionProfile?.isQuestion) {
      target += Math.sqrt(contentCount + 1) * 1.5;
    }

    // Large prompts still scale all the way toward the 360-token ceiling.
    target += Math.max(0, promptLen - 28) * 0.90;
    target += Math.max(0, promptLen - 80) * 1.10;
    target += Math.max(0, promptLen - 160) * 1.30;

    return Math.max(
      8,
      Math.min(this.maxTokenCeiling, Math.round(target))
    );
  }

  getAdaptiveRSLPassPlan(inputTokenCount, targetLength) {
    const n = Math.max(1, inputTokenCount);
    const out = Math.max(1, targetLength);

    const inputPressure = 1.0 - Math.exp(-n / 30.0);
    const outputPressure = 1.0 - Math.exp(-out / 140.0);

    // RSL is allowed to stop after pass 1. Complexity grants more SEARCH ROOM,
    // it never forces the model to spend that room.
    const minimumPasses = 1;

    const preferredPasses = Math.max(
      1,
      Math.min(
        6,
        Math.ceil(
          1 +
          inputPressure * 2.2 +
          outputPressure * 1.8
        )
      )
    );

    return {
      minimumPasses,
      preferredPasses,
      maxPasses: 60,
      inputPressure,
      outputPressure
    };
  }

  getAdaptiveLatentHops(inputTokenCount, targetLength) {
    const n = Math.max(1, inputTokenCount);
    const out = Math.max(1, targetLength);

    // Cheap prompts get one latent refinement. Complexity can grow this to 4.
    const pressure =
      (1.0 - Math.exp(-n / 34.0)) * 0.62 +
      (1.0 - Math.exp(-out / 150.0)) * 0.38;

    return Math.max(1, Math.min(4, 1 + Math.floor(pressure * 4.0)));
  }

  getAdaptiveGenerationBudget(inputTokenCount, targetLength, contextModel) {
    const explicit = contextModel?.explicitRequestedLength;

    if (Number.isFinite(explicit)) {
      return Math.max(
        1,
        Math.min(
          this.maxTokenCeiling,
          Math.ceil(explicit + Math.min(12, Math.sqrt(explicit) + 2))
        )
      );
    }

    // 360 is AVAILABLE, but a 5-token question should not silently receive a
    // 360-token CPU budget. The budget grows continuously with requested work.
    const slack =
      5 +
      Math.sqrt(Math.max(1, inputTokenCount)) * 1.8 +
      Math.sqrt(Math.max(1, targetLength)) * 1.7;

    return Math.max(
      10,
      Math.min(
        this.maxTokenCeiling,
        Math.ceil(targetLength + slack)
      )
    );
  }

  deterministicPassVariation(tokenId, passIndex) {
    if (!passIndex) return 0.0;
    const x = Math.sin(
      (tokenId + 1) * 12.9898 +
      (passIndex + 1) * 78.233
    ) * 43758.5453;
    return (x - Math.floor(x)) - 0.5;
  }

  refineRSLContext(contextModel, candidate, passIndex) {
    const tokens = candidate?.tokens || [];
    if (!tokens.length || !contextModel?.reasoningCentroid) return;

    // Do not learn from a sloppy candidate during inference. This prevents a
    // bad pass from reinforcing its own word salad on the next pass.
    if (
      (candidate.fluencyScore ?? 0) < 0.58 ||
      (candidate.syntaxScore ?? 0) < 0.60
    ) {
      return;
    }

    const embDim = this.network.embDim;
    const candidateCentroid = new Float32Array(embDim);
    let count = 0;

    for (const tok of tokens) {
      const type = this.getCanonicalType(tok);
      if (!this.isContentType(type)) continue;

      const off = (tok % this.network.vocabSize) * embDim;
      for (let d = 0; d < embDim; d++) {
        candidateCentroid[d] += this.normalizedEmbeddings[off + d];
      }
      count++;
    }

    if (!count) return;

    for (let d = 0; d < embDim; d++) {
      candidateCentroid[d] /= count;
    }
    this.normalizeVector(candidateCentroid);

    const old = contextModel.reasoningCentroid;
    const topic = contextModel.topicCentroid || contextModel.centroid;
    const intent = contextModel.intentCentroid || contextModel.centroid;

    // Much smaller adaptation than before. The candidate can nudge reasoning,
    // but the original input remains the dominant semantic anchor.
    const candidateRate = 0.055 / Math.sqrt(passIndex + 1);
    const anchorRate = 0.11;
    const retainRate = Math.max(
      0,
      1.0 - candidateRate - anchorRate
    );

    const next = new Float32Array(embDim);
    for (let d = 0; d < embDim; d++) {
      next[d] =
        old[d] * retainRate +
        candidateCentroid[d] * candidateRate +
        topic[d] * 0.065 +
        intent[d] * 0.045;
    }
    this.normalizeVector(next);

    contextModel.reasoningCentroid = next;

    const scores = contextModel.reasoningScores
      ? new Float32Array(contextModel.reasoningScores)
      : new Float32Array(this.network.vocabSize);

    const ids = contextModel.deepCandidatePool?.length
      ? contextModel.deepCandidatePool
      : Int32Array.from({ length: this.network.vocabSize }, (_, i) => i);

    for (const i of ids) {
      if (!this.network.vocab?.hasToken?.(i)) continue;
      scores[i] = this.dotTokenVector(i, next);
    }

    contextModel.reasoningScores = scores;
  }

  shouldStopRSLSearch(history, best, passIndex, plan) {
    const passesDone = passIndex + 1;
    if (!best) return false;

    const strongLanguage =
      (best.syntaxScore ?? 0) >= 0.80 &&
      (best.clauseOrderScore ?? 0) >= 0.70 &&
      (best.fluencyScore ?? 0) >= 0.70;

    const strongRelevance =
      (best.contentPrecision ?? 0) >= 0.58 &&
      (best.promptCoverage ?? 0) >= 0.26;

    const confidence = Number.isFinite(best.confidenceValue)
      ? best.confidenceValue
      : 0.0;

    // Fastest path: if the FIRST result is already clean, relevant, and the
    // model agrees with it, ship it immediately.
    if (
      passesDone === 1 &&
      strongLanguage &&
      strongRelevance &&
      confidence >= 0.58
    ) {
      return true;
    }

    if (passesDone < plan.minimumPasses) return false;

    const recent = history.slice(-3);

    // After two or more passes, stop as soon as extra work stops buying quality.
    if (recent.length >= 2) {
      const gain = recent[recent.length - 1] - recent[0];
      const spread = Math.max(...recent) - Math.min(...recent);

      if (strongLanguage && strongRelevance && gain <= 0.0030) {
        return true;
      }

      if (passesDone >= plan.preferredPasses && spread <= 0.0025) {
        return true;
      }

      // Beyond the preferred budget, only continue when quality remains weak
      // AND measurable improvement still exists.
      if (passesDone >= plan.preferredPasses) {
        const stillWeak = !strongLanguage || !strongRelevance;
        const stillImproving = gain > 0.0020;

        if (!stillWeak || !stillImproving) return true;
      }
    }

    return passesDone >= plan.maxPasses;
  }

  shouldStopDeepAnswer(
    generated,
    sentenceState,
    contextModel,
    targetLength,
    sentenceCount
  ) {
    const explicitLength = contextModel?.explicitRequestedLength;

    if (explicitLength !== null && explicitLength !== undefined) {
      return generated.length >= Math.max(1, explicitLength);
    }

    const answered = this.answerRelationSatisfied(
      generated,
      sentenceState,
      contextModel
    );

    if (!answered) return false;

    const coverage = this.promptCoverage(generated, contextModel);
    const lengthRatio =
      generated.length / Math.max(1, targetLength);

    if (
      sentenceCount >= 1 &&
      coverage >= 0.48 &&
      generated.length >= 8 &&
      lengthRatio >= 0.48
    ) {
      return true;
    }

    if (
      sentenceCount >= 1 &&
      coverage >= 0.72 &&
      generated.length >= 7
    ) {
      return true;
    }

    return generated.length >= targetLength;
  }

  /**
   * Universal prompt-clause planner.
   *
   * Applies to every prompt. There are no hardcoded question words, brand
   * names, example phrases, or canned response templates in this planner.
   *
   * It assembles one compact clause from:
   *   - POS metadata
   *   - prompt token order
   *   - semantic relevance
   *   - one neural logit vector
   *   - the existing clause/transition model
   *
   * Prompt tokens are preferred. A small set of high-scoring neural tokens may
   * fill missing grammatical roles without requiring another neural forward.
   */
  buildUniversalPromptFrame(cleanPromptTokens, rawLogits, contextModel) {
    const vocab = this.network.vocab;
    const vocabSize = this.network.vocabSize;

    const promptSeq = cleanPromptTokens
      .filter(tok => this.getCanonicalType(tok) !== "Punct");

    if (!rawLogits || !vocab || vocabSize <= 0) return null;

    const candidateMap = new Map();

    // Prompt tokens retain their original source positions. For questions,
    // suppress the leading interrogative and do-support helper so Deep forms a
    // declarative answer rather than simply rebuilding the question.
    const qProfile = contextModel?.questionProfile;

    for (let i = 0; i < promptSeq.length; i++) {
      const tok = promptSeq[i];
      if (!vocab.hasToken(tok)) continue;

      if (qProfile?.isQuestion && tok === qProfile.operatorToken) {
        continue;
      }

      // An inverted helper before the subject belongs to the question form.
      // We let the neural candidate pool reintroduce it only if the answer
      // actually needs it.
      if (
        qProfile?.isQuestion &&
        qProfile.helperIndex >= 0 &&
        i === qProfile.helperIndex
      ) {
        continue;
      }

      if (!candidateMap.has(tok)) {
        candidateMap.set(tok, {
          token: tok,
          source: "prompt",
          promptIndex: i
        });
      }
    }

    // Add a compact neural candidate set from the already-pruned Deep pool.
    // The full vocabulary was scanned once when the pool was built.
    const neuralTop = [];
    const sourceIds = contextModel?.deepCandidatePool?.length
      ? contextModel.deepCandidatePool
      : Int32Array.from({ length: vocabSize }, (_, i) => i);

    for (const i of sourceIds) {
      if (!vocab.hasToken(i)) continue;

      const type = this.getCanonicalType(i);
      if (type === "Punct" || type === "Other") continue;

      let score = rawLogits[i];
      score += this.getRoutedSemanticScore(i, contextModel, null) * 1.15;

      if (neuralTop.length < 28) {
        neuralTop.push({ token: i, score });
        neuralTop.sort((a, b) => b.score - a.score);
      } else if (score > neuralTop[neuralTop.length - 1].score) {
        neuralTop[neuralTop.length - 1] = { token: i, score };
        neuralTop.sort((a, b) => b.score - a.score);
      }
    }

    for (const entry of neuralTop) {
      if (!candidateMap.has(entry.token)) {
        candidateMap.set(entry.token, {
          token: entry.token,
          source: "neural",
          promptIndex: -1
        });
      }
    }

    const candidates = Array.from(candidateMap.values());
    if (!candidates.length) return null;

    const freshState = () => ({
      isStart: true,
      subjectReady: false,
      predicateReady: false,
      verbCount: 0,
      nounCount: 0,
      prepCount: 0,
      objectCount: 0,
      isPluralSubject: false
    });

    const cloneState = s => ({
      isStart: s.isStart,
      subjectReady: s.subjectReady,
      predicateReady: s.predicateReady,
      verbCount: s.verbCount,
      nounCount: s.nounCount,
      prepCount: s.prepCount,
      objectCount: s.objectCount,
      isPluralSubject: s.isPluralSubject
    });

    const startTypeBonus = type => {
      if (type === "ProperNoun" || type === "Pronoun") return 3.4;
      if (type === "Noun") return 3.1;
      if (type === "Det") return 2.7;
      if (type === "Adj") return 2.3;
      if (type === "Adv") return 0.6;
      if (type === "Verb") return 0.3;
      return -3.0;
    };

    const roleBonus = (type, state) => {
      if (!state.subjectReady) {
        if (["Noun", "ProperNoun", "Pronoun"].includes(type)) return 4.2;
        if (["Det", "Adj"].includes(type)) return 2.1;
        return -1.2;
      }

      if (!state.predicateReady) {
        if (["Verb", "Modal", "Aux"].includes(type)) return 4.8;
        if (["Noun", "ProperNoun", "Adj"].includes(type) && state.nounCount < 3) return 1.2;
        return -1.0;
      }

      if (state.objectCount === 0) {
        if (["Noun", "ProperNoun", "Pronoun"].includes(type)) return 3.3;
        if (["Det", "Adj", "Prep", "Adv"].includes(type)) return 1.6;
        if (["Verb", "Modal", "Aux"].includes(type)) return -4.5;
      }

      if (state.objectCount >= 1) {
        if (type === "Prep") return 1.3;
        if (["Adj", "Adv"].includes(type)) return 0.6;
        if (["Verb", "Modal", "Aux"].includes(type)) return -5.5;
      }

      return 0.0;
    };

    // General transition logic for the beam. The only extension over the base
    // DFA is allowing short nominal compounds before the predicate.
    const beamTransitionLegal = (beam, word, type) => {
      const state = beam.state;
      const prevToken = beam.tokens.length
        ? beam.tokens[beam.tokens.length - 1]
        : null;
      const prevType = prevToken !== null
        ? this.getCanonicalType(prevToken)
        : null;

      if (state.isStart) {
        return ["Det", "Adj", "Noun", "ProperNoun", "Pronoun", "Adv", "Verb"].includes(type);
      }

      if (
        !state.predicateReady &&
        state.nounCount < 3 &&
        ["Noun", "ProperNoun", "Adj"].includes(prevType) &&
        ["Noun", "ProperNoun"].includes(type)
      ) {
        return true;
      }

      if (
        state.predicateReady &&
        prevType !== "Conj" &&
        ["Verb", "Modal", "Aux"].includes(type)
      ) {
        return false;
      }

      return this.isTransitionLegal(beam.tokens, word, type, state, true);
    };

    const maxWords = Math.max(
      4,
      Math.min(14, Math.max(7, Math.round(promptSeq.length * 0.80)))
    );
    const beamWidth = 10;

    let beams = [{
      tokens: [],
      state: freshState(),
      score: 0,
      used: new Set(),
      lastPromptIndex: -1,
      promptUsed: 0
    }];

    const completed = [];

    for (let depth = 0; depth < maxWords; depth++) {
      const expanded = [];

      for (const beam of beams) {
        for (const cand of candidates) {
          if (beam.used.has(cand.token)) continue;

          const word = this.tokenWord(cand.token);
          const type = this.getCanonicalType(cand.token);
          if (!word || type === "Punct" || type === "Other") continue;
          if (!beamTransitionLegal(beam, word, type)) continue;

          const nextState = cloneState(beam.state);
          const nextTokens = [...beam.tokens, cand.token];

          let score = beam.score;

          // Neural support from the current deep pass.
          score += Math.max(-6, Math.min(6, rawLogits[cand.token])) * 0.28;

          // Dual-centroid + latent semantic support. The routing changes as
          // the clause moves from subject -> predicate -> complement.
          score += this.getRoutedSemanticScore(
            cand.token,
            contextModel,
            beam.state
          ) * 1.45;

          // Universal grammatical-role pressure.
          score += beam.tokens.length === 0
            ? startTypeBonus(type)
            : roleBonus(type, beam.state);

          // A question changes what constitutes a useful answer. "How" routes
          // toward mechanisms, "why" toward causes, "where" toward locations,
          // etc., without hardcoding the factual content.
          score += this.questionRelationBonus(
            cand.token,
            beam.state,
            contextModel
          );

          // Existing POS transition matrix still matters.
          if (beam.tokens.length) {
            const prevType = this.getCanonicalType(
              beam.tokens[beam.tokens.length - 1]
            );
            score += this.getTransitionRule(prevType, type).weight * 0.75;
          }

          let nextPromptUsed = beam.promptUsed;
          let nextLastPromptIndex = beam.lastPromptIndex;

          // Prefer prompt evidence over invented neural filler.
          if (cand.source === "prompt") {
            score += 3.0;
            nextPromptUsed++;

            if (beam.lastPromptIndex >= 0) {
              const jump = cand.promptIndex - beam.lastPromptIndex;

              // Prompt order helps, but reordering is allowed whenever grammar
              // makes the original order unsuitable for an answer clause.
              if (jump >= 0 && jump <= 3) score += 1.0;
              else if (jump < 0) {
                score -= Math.min(1.2, Math.abs(jump) * 0.15);
              } else {
                score -= Math.min(0.8, jump * 0.08);
              }
            }

            nextLastPromptIndex = cand.promptIndex;
          } else {
            score -= 0.8;
          }

          // Generic morphological-family suppression.
          const stem = this.stemWord(word);
          if (stem.length >= 3) {
            let sameStem = 0;
            for (const usedTok of beam.tokens) {
              if (this.stemWord(this.tokenWord(usedTok)) === stem) {
                sameStem++;
              }
            }
            if (sameStem > 0) {
              score -= 8.0 + sameStem * 4.0;
            }
          }

          this.updateSentenceState(
            nextState,
            cand.token,
            word,
            type
          );

          const nextUsed = new Set(beam.used);
          nextUsed.add(cand.token);

          const nextBeam = {
            tokens: nextTokens,
            state: nextState,
            score,
            used: nextUsed,
            lastPromptIndex: nextLastPromptIndex,
            promptUsed: nextPromptUsed
          };

          if (
            this.answerRelationSatisfied(nextTokens, nextState, contextModel) &&
            nextTokens.length >= 3
          ) {
            const promptRatio =
              nextPromptUsed / Math.max(1, nextTokens.length);

            const completeScore =
              score +
              promptRatio * 4.0 +
              (nextState.objectCount > 0 ? 1.2 : 0.0) -
              Math.max(0, nextTokens.length - 9) * 0.35;

            completed.push({
              ...nextBeam,
              score: completeScore
            });
          }

          expanded.push(nextBeam);
        }
      }

      if (!expanded.length) break;

      expanded.sort((a, b) => b.score - a.score);
      beams = expanded.slice(0, beamWidth);

      // Anytime stop when a short prompt-heavy clause is decisively ahead.
      if (completed.length) {
        completed.sort((a, b) => b.score - a.score);

        const bestComplete = completed[0];
        const bestActive = beams[0];

        if (
          bestComplete.tokens.length >= 4 &&
          bestComplete.promptUsed >= 2 &&
          bestComplete.score >= bestActive.score + 2.0
        ) {
          break;
        }
      }
    }

    if (!completed.length) return null;

    completed.sort((a, b) => b.score - a.score);
    const winner = completed[0];

    const tokens = [...winner.tokens];
    if (tokens.length) {
      tokens[0] = this.sentenceCaseToken(tokens[0]);
    }

    return {
      kind: "universal-pos-beam",
      tokens,
      complete: true,
      sourceTokenCount: winner.promptUsed,
      plannerScore: winner.score
    };
  }

  buildFrameTrajectory(frameTokens, rawLogits) {
    if (!frameTokens?.length) return [];
    let maxLogit = -Infinity;
    if (rawLogits) {
      for (let i = 0; i < rawLogits.length; i++) {
        if (rawLogits[i] > maxLogit) maxLogit = rawLogits[i];
      }
    }

    const trajectory = [];
    for (const tok of frameTokens) {
      let prob = 0.70;
      if (rawLogits && Number.isFinite(rawLogits[tok]) && Number.isFinite(maxLogit)) {
        // Prompt-frame support: a token that is close to the strongest neural
        // logit gets high support, while weakly supported copied tokens remain
        // modest. This is evidence, not an automatic confidence bonus.
        const delta = Math.max(-8.0, Math.min(0.0, rawLogits[tok] - maxLogit));
        prob = Math.max(0.50, Math.min(0.88, 0.58 + 0.30 * Math.exp(delta)));
      }
      trajectory.push({ token: tok, prob, margin: 0.0, scaffold: true });
    }
    return trajectory;
  }

  extractContextModel(cleanPromptTokens) {
    const embDim = this.network.embDim;
    const vocabSize = this.network.vocabSize;
    const vocab = this.network.vocab;
    if (!this.normalizedEmbeddings) this.initNormalizedEmbeddings();

    const centroid = new Float32Array(embDim);
    const topicCentroid = new Float32Array(embDim);
    const intentCentroid = new Float32Array(embDim);

    let topicWeightSum = 0.0;
    let intentWeightSum = 0.0;

    const questionProfile = this.analyzePromptStructure(cleanPromptTokens);
    const promptTokenSet = new Set(cleanPromptTokens);
    const promptContentSet = new Set();
    const promptContentOrder = new Map();
    const promptContentTokens = [];
    const promptStems = new Set();
    const promptTransitions = new Map();

    for (let i = 0; i < cleanPromptTokens.length; i++) {
      const tok = cleanPromptTokens[i];
      const off = (tok % vocabSize) * embDim;
      const type = this.getCanonicalType(tok);
      const word = vocab?.idToToken?.[tok] || "";

      // Mild recency gives the end of long prompts slightly more influence
      // without allowing it to erase the beginning.
      const posWeight =
        0.90 + 0.10 * (i / Math.max(1, cleanPromptTokens.length - 1));

      for (let d = 0; d < embDim; d++) {
        centroid[d] += this.normalizedEmbeddings[off + d] * posWeight;
      }

      // Topic centroid: "what is this about?"
      let topicWeight = 0.0;
      if (["Noun", "ProperNoun", "Adj", "Num"].includes(type)) {
        topicWeight = 1.0;
      } else if (type === "Pronoun") {
        topicWeight = 0.35;
      }

      if (topicWeight > 0) {
        const w = topicWeight * posWeight;
        for (let d = 0; d < embDim; d++) {
          topicCentroid[d] += this.normalizedEmbeddings[off + d] * w;
        }
        topicWeightSum += w;
      }

      // Intent centroid: "what action / relation is being requested?"
      let intentWeight = 0.0;
      if (type === "Verb") intentWeight = 1.0;
      else if (type === "Modal" || type === "Aux") intentWeight = 0.85;
      else if (type === "Adv") intentWeight = 0.60;
      else if (type === "Prep") intentWeight = 0.35;

      if (intentWeight > 0) {
        const w = intentWeight * posWeight;
        for (let d = 0; d < embDim; d++) {
          intentCentroid[d] += this.normalizedEmbeddings[off + d] * w;
        }
        intentWeightSum += w;
      }

      if (this.isContentType(type)) {
        promptContentSet.add(tok);
        if (!promptContentOrder.has(tok)) {
          promptContentOrder.set(tok, promptContentTokens.length);
          promptContentTokens.push(tok);
        }

        const stem = this.stemWord(word);
        if (stem.length >= 3) promptStems.add(stem);
      }

      if (i + 1 < cleanPromptTokens.length) {
        const next = cleanPromptTokens[i + 1];
        let set = promptTransitions.get(tok);

        if (!set) {
          set = new Set();
          promptTransitions.set(tok, set);
        }
        set.add(next);
      }
    }

    this.normalizeVector(centroid);

    // If one route has no POS evidence, fall back to the general prompt
    // centroid. That keeps one-word or unusual prompts valid.
    if (topicWeightSum > 0) this.normalizeVector(topicCentroid);
    else topicCentroid.set(centroid);

    if (intentWeightSum > 0) this.normalizeVector(intentCentroid);
    else intentCentroid.set(centroid);

    // Compute both semantic routes once per prompt.
    const topicScores = new Float32Array(vocabSize);
    const intentScores = new Float32Array(vocabSize);
    const semanticScores = new Float32Array(vocabSize);

    for (let i = 0; i < vocabSize; i++) {
      if (!vocab?.hasToken?.(i)) continue;

      const topic = this.dotTokenVector(i, topicCentroid);
      const intent = this.dotTokenVector(i, intentCentroid);

      topicScores[i] = topic;
      intentScores[i] = intent;

      // Static fallback blend. Generation uses dynamic routing based on clause
      // state through getRoutedSemanticScore().
      semanticScores[i] = Math.max(
        -1.0,
        Math.min(1.0, topic * 0.58 + intent * 0.42)
      );
    }

    return {
      centroid,
      topicCentroid,
      intentCentroid,
      topicScores,
      intentScores,
      semanticScores,
      reasoningCentroid: null,
      reasoningScores: null,
      reasoningStability: 0.0,
      reasoningHops: 0,
      questionProfile,
      explicitRequestedLength: this.inferRequestedLength(cleanPromptTokens),
      promptTokenSet,
      promptContentSet,
      promptContentOrder,
      promptContentTokens,
      promptStems,
      promptTransitions
    };
  }

  getTransitionRule(prevType, curType) {
    if (!prevType) return { legal: true, weight: 0.0 };
    const key = `${prevType}->${curType}`;
    return this.transitionMatrix[key] || { legal: false, weight: -100.0 };
  }

  clauseComplete(sentenceState) {
    return sentenceState.subjectReady && sentenceState.predicateReady;
  }

  promptCoverage(generated, contextModel) {
    const wanted = contextModel.promptContentSet;
    if (!wanted || wanted.size === 0) return 0.0;
    let hit = 0;
    const seen = new Set();
    for (const tok of generated) {
      if (wanted.has(tok) && !seen.has(tok)) {
        seen.add(tok);
        hit++;
      }
    }
    return hit / wanted.size;
  }

  isTransitionLegal(generatedTokens, curWord, curType, sentenceState, deepMode) {
    const lowerCur = curWord.toLowerCase();
    const prevToken = generatedTokens.length > 0 ? generatedTokens[generatedTokens.length - 1] : null;
    const prevWord = prevToken !== null ? (this.network.vocab?.idToToken[prevToken] || "").toLowerCase() : "";
    const prevType = prevToken !== null ? this.getCanonicalType(prevToken) : null;

    if (curType === "Punct") {
      if (![".", "!", "?"].includes(curWord)) return false;
      if (!this.clauseComplete(sentenceState)) return false;
      if (["Prep", "Det", "Modal", "Conj"].includes(prevType)) return false;
      if (prevType === "Aux" && ["be", "is", "am", "are", "been"].includes(prevWord)) return false;
      return true;
    }

    if (sentenceState.isStart) {
      // Answers should normally start declaratively, not with the question's
      // interrogative token.
      return this.getTransitionRule("START", curType).legal;
    }

    // "to" can be an infinitive marker as well as a preposition.
    if (prevWord === "to" && curType === "Verb") {
      return true;
    }

    if (curType === "Prep" && (sentenceState.prepCount >= 3 || prevType === "Prep")) return false;
    // Article phonetics and basic number agreement.
    if (prevWord === "a" && (/^[aeiou]/i.test(curWord) || /^(ai|apu|npu|cpu|gpu)/i.test(curWord))) {
      return false;
    }
    if (prevWord === "an" && !/^[aeiou]/i.test(curWord) && !/^(ai|apu|npu|cpu|gpu)/i.test(curWord)) {
      return false;
    }

    if (prevType === "Det" && ["Noun", "ProperNoun"].includes(curType)) {
      const singularDets = new Set(["a", "an", "this", "each", "every", "another", "one"]);
      const pluralDets = new Set(["these", "those", "many", "several", "both", "few"]);
      const looksPlural = lowerCur.endsWith("s") && !lowerCur.endsWith("ss");

      if (singularDets.has(prevWord) && looksPlural) return false;
      if (pluralDets.has(prevWord) && !looksPlural) return false;
    }

    // Modal and do-support auxiliaries require a base-like verb.
    if (prevType === "Modal") {
      if (curType !== "Verb" && curType !== "Aux" && curType !== "Adv") return false;
      if (curType === "Verb" &&
          (lowerCur.endsWith("ing") || lowerCur.endsWith("ed") ||
           (lowerCur.endsWith("s") && !lowerCur.endsWith("ss")))) {
        return false;
      }
    }

    if (prevType === "Aux" && ["do", "does", "did"].includes(prevWord)) {
      if (curType !== "Verb" && curType !== "Adv") return false;
      if (curType === "Verb" &&
          (lowerCur.endsWith("ing") || lowerCur.endsWith("ed") ||
           (lowerCur.endsWith("s") && !lowerCur.endsWith("ss")))) {
        return false;
      }
    }

    if (prevType === "Aux" && curType === "Aux") return false;
    if (prevType === "Det" && curType === "Det") return false;
    if (prevType === "Conj" && curType === "Conj") return false;

    // Avoid chains of grammatical glue with no content between them.
    const functionTypes = new Set(["Det", "Prep", "Conj", "Modal", "Aux"]);
    if (functionTypes.has(prevType) && functionTypes.has(curType)) {
      const permitted =
        (prevType === "Modal" && curType === "Aux") ||
        (prevType === "Conj" && ["Det", "Aux", "Modal"].includes(curType));

      if (!permitted) return false;
    }

    // Prepositions need an object-like continuation, except "to + Verb".
    if (
      prevType === "Prep" &&
      prevWord !== "to" &&
      !["Det", "Adj", "Noun", "ProperNoun", "Pronoun", "Num"].includes(curType)
    ) {
      return false;
    }

    // Strong subject/verb agreement checks for simple present forms. These are
    // deliberately conservative; ambiguous verbs are left to neural scoring.
    if (
      sentenceState.subjectReady &&
      !sentenceState.predicateReady &&
      curType === "Verb"
    ) {
      const verbHasS = lowerCur.endsWith("s") && !lowerCur.endsWith("ss");
      const subjectWord = sentenceState.subjectToken !== null
        ? this.tokenWord(sentenceState.subjectToken).toLowerCase()
        : "";

      const explicitlyNonThird = new Set(["i", "you", "we", "they"]);
      const explicitlyThird = new Set(["he", "she", "it"]);

      if ((sentenceState.isPluralSubject || explicitlyNonThird.has(subjectWord)) && verbHasS) {
        return false;
      }
      if (explicitlyThird.has(subjectWord) && !verbHasS) {
        return false;
      }
    }

    // Conjunctions should not be generic escape hatches in unfinished clauses.
    if (
      deepMode &&
      curType === "Conj" &&
      !this.clauseComplete(sentenceState)
    ) {
      return false;
    }

    // Prevent three-deep modifier chains and immediate Prep-object-Prep loops.
    if (deepMode && generatedTokens.length >= 2) {
      const t1 = this.getCanonicalType(
        generatedTokens[generatedTokens.length - 1]
      );
      const t2 = this.getCanonicalType(
        generatedTokens[generatedTokens.length - 2]
      );

      const modifier = t => t === "Adj" || t === "Adv";

      if (
        modifier(t1) &&
        modifier(t2) &&
        modifier(curType)
      ) {
        return false;
      }

      if (
        t2 === "Prep" &&
        ["Noun", "ProperNoun", "Pronoun", "Num"].includes(t1) &&
        curType === "Prep"
      ) {
        return false;
      }
    }

    // Before the predicate, allow compact noun compounds but prevent the model
    // from stacking an arbitrary river of nouns before ever reaching a verb.
    if (
      deepMode &&
      !sentenceState.predicateReady &&
      sentenceState.nounCount >= 4 &&
      ["Noun", "ProperNoun"].includes(curType)
    ) {
      return false;
    }

    // Deep mode is deliberately simple-clause biased. Once a predicate exists,
    // do not start another finite verb/modal/aux chain unless a conjunction has
    // explicitly opened a new clause. This kills outputs like
    // "chips accelerate model accelerates accelerating accelerated".
    if (
      deepMode &&
      sentenceState.predicateReady &&
      prevType !== "Conj" &&
      ["Verb", "Modal", "Aux"].includes(curType)
    ) {
      return false;
    }

    // Avoid endless noun piles after the predicate. Two object/complement nouns
    // are enough for short declarative answers; after that prefer a modifier,
    // prepositional phrase, or punctuation.
    if (
      deepMode &&
      sentenceState.predicateReady &&
      sentenceState.objectCount >= 2 &&
      ["Noun", "ProperNoun"].includes(prevType) &&
      ["Noun", "ProperNoun"].includes(curType)
    ) {
      return false;
    }

    return this.getTransitionRule(prevType, curType).legal;
  }

  roleBias(curType, sentenceState) {
    if (!sentenceState.subjectReady) {
      if (curType === "Noun" || curType === "ProperNoun" || curType === "Pronoun") return 4.5;
      if (curType === "Adj" || curType === "Det") return 1.8;
      return -2.0;
    }

    if (!sentenceState.predicateReady) {
      if (curType === "Verb") return 5.0;
      if (curType === "Modal" || curType === "Aux") return 3.8;
      if (curType === "Adv") return 0.5;
      if (curType === "Noun") return 0.4; // noun compound such as "model chips"
      return -1.0;
    }

    if (sentenceState.objectCount === 0) {
      if (curType === "Noun" || curType === "ProperNoun") return 3.4;
      if (curType === "Det" || curType === "Adj") return 1.8;
      if (curType === "Prep" || curType === "Adv") return 0.9;
      if (["Verb", "Modal", "Aux"].includes(curType)) return -4.0;
    }

    if (sentenceState.objectCount >= 1) {
      if (curType === "Punct") return 4.2;
      if (curType === "Prep") return 1.8;
      if (curType === "Adv" || curType === "Adj") return 0.8;
      if (["Verb", "Modal", "Aux"].includes(curType)) return -5.0;
    }

    if (curType === "Punct") return 3.0;
    return 0.0;
  }

  nextRoleLookahead(curType, sentenceState) {
    // Cheap one-step structural lookahead. No extra neural forward pass.
    if (!sentenceState.subjectReady) {
      return ["Noun", "ProperNoun", "Pronoun"].includes(curType) ? 1.2 : 0.0;
    }
    if (!sentenceState.predicateReady) {
      return ["Verb", "Modal", "Aux"].includes(curType) ? 1.5 : 0.0;
    }
    if (sentenceState.objectCount === 0) {
      return ["Noun", "ProperNoun", "Det", "Adj", "Prep"].includes(curType) ? 0.8 : 0.0;
    }
    return curType === "Punct" ? 1.0 : 0.0;
  }

  getStemCounts(tokens) {
    const counts = new Map();
    const vocab = this.network.vocab;
    for (const tok of tokens) {
      const type = this.getCanonicalType(tok);
      if (!this.isContentType(type)) continue;
      const word = vocab?.idToToken?.[tok] || "";
      const stem = this.stemWord(word);
      if (stem.length < 3) continue;
      counts.set(stem, (counts.get(stem) || 0) + 1);
    }
    return counts;
  }

  buildDeepCandidatePool(rawLogits, contextModel, cleanPromptTokens) {
    const vocab = this.network.vocab;
    const vocabSize = this.network.vocabSize;

    // Keep a small top set per grammatical family so pruning never means
    // "top logits are all nouns, therefore no verb can be produced".
    const familyLimit = 18;
    const families = new Map();

    const familyOf = type => {
      if (type === "ProperNoun") return "Noun";
      if (type === "Modal") return "Aux";
      return type;
    };

    const insertTop = (family, token, score) => {
      let arr = families.get(family);
      if (!arr) {
        arr = [];
        families.set(family, arr);
      }

      if (arr.length < familyLimit) {
        arr.push({ token, score });
        arr.sort((a, b) => b.score - a.score);
      } else if (score > arr[arr.length - 1].score) {
        arr[arr.length - 1] = { token, score };
        arr.sort((a, b) => b.score - a.score);
      }
    };

    for (let i = 0; i < vocabSize; i++) {
      if (!vocab?.hasToken?.(i)) continue;

      const type = this.getCanonicalType(i);
      if (type === "Other") continue;

      const semantic = this.getRoutedSemanticScore(
        i,
        contextModel,
        null
      );

      const score = rawLogits[i] + semantic * 1.25;
      insertTop(familyOf(type), i, score);
    }

    const pool = new Set();

    // Prompt tokens must always survive pruning.
    for (const tok of cleanPromptTokens) {
      if (vocab?.hasToken?.(tok)) pool.add(tok);
    }

    for (const arr of families.values()) {
      for (const item of arr) pool.add(item.token);
    }

    // Punctuation is cheap and grammatically essential.
    for (const p of [".", ",", "!", "?", ";", ":"]) {
      const id = vocab?.tokenToId?.get?.(p);
      if (Number.isFinite(id)) pool.add(id);
    }

    return Int32Array.from(pool);
  }

  buildScores(rawLogits, generated, contextModel, sentenceState, deepMode, step, targetLength) {
    const vocabSize = this.network.vocabSize;
    const vocab = this.network.vocab;
    const prevToken = generated.length ? generated[generated.length - 1] : null;
    const prevType = prevToken !== null ? this.getCanonicalType(prevToken) : null;

    const scores = new Float32Array(vocabSize);
    scores.fill(-1e9);

    const recent = generated.slice(-10);
    const recentCounts = new Map();
    for (const t of recent) recentCounts.set(t, (recentCounts.get(t) || 0) + 1);

    const stemCounts = this.getStemCounts(generated.slice(-40));
    const generatedSet = new Set(generated.slice(-48));
    const coverageNow = this.promptCoverage(generated, contextModel);
    const firstPromptContent = contextModel.promptContentTokens?.[0] ?? null;
    const promptContentCount = Math.max(1, contextModel.promptContentTokens?.length || 0);

    const patternContext = deepMode
      ? this.buildLocalPatternContext(generated)
      : null;

    const candidateIds =
      deepMode && contextModel?.deepCandidatePool?.length
        ? contextModel.deepCandidatePool
        : null;

    const scanCount = candidateIds ? candidateIds.length : vocabSize;

    for (let scanIndex = 0; scanIndex < scanCount; scanIndex++) {
      const i = candidateIds ? candidateIds[scanIndex] : scanIndex;

      if (!vocab?.hasToken?.(i)) continue;
      const word = vocab.idToToken[i] || "";
      const curType = this.getCanonicalType(i);
      if (!this.isTransitionLegal(generated, word, curType, sentenceState, deepMode)) continue;

      let score = rawLogits[i];
      score += this.getTransitionRule(prevType, curType).weight;

      const routedSemantic = this.getRoutedSemanticScore(
        i,
        contextModel,
        sentenceState
      );
      score += routedSemantic * (deepMode ? 1.55 : 0.70);

      const count = recentCounts.get(i) || 0;
      if (count) score -= count * count * (deepMode ? 8.0 : 3.0);
      if (i === prevToken) score -= 22.0;

      const stem = this.stemWord(word);
      const exactPrompt = contextModel.promptContentSet.has(i);
      const stemPrompt = stem.length >= 3 && contextModel.promptStems.has(stem);
      const sameStemCount = stem.length >= 3 ? (stemCounts.get(stem) || 0) : 0;

      if (deepMode) {
        // 1) Prompt-order bias: prefer the first useful content words from the
        // prompt at sentence start. This helps "AI model chips ..." stay in the
        // natural prompt order instead of starting from a random final noun.
        if (sentenceState.isStart && exactPrompt) {
          const order = contextModel.promptContentOrder?.get(i) ?? promptContentCount;
          const orderBonus = Math.max(0, 4.8 - order * 0.65);
          score += orderBonus;
          if (i === firstPromptContent) score += 3.2;
        }

        // 2) Prompt-anchor quota: prompt words are valuable once, not forever.
        // Once used, their exact token/stem loses its bonus and starts taking a
        // repetition penalty. This prevents prompt echo loops.
        if (exactPrompt && !generatedSet.has(i) && sameStemCount === 0) {
          score += coverageNow < 0.55 ? 5.2 : 2.0;
        } else if (stemPrompt && sameStemCount === 0) {
          score += coverageNow < 0.55 ? 2.8 : 1.0;
        }

        // 3) Morphological-family suppression. One accelerate-family word per
        // clause is normally enough. Same-family forms get hammered.
        if (sameStemCount > 0 && this.isContentType(curType)) {
          score -= 9.0 + (sameStemCount - 1) * 5.0;
          if (exactPrompt || stemPrompt) score -= 2.0;
        }

        // Keep off-topic content from sneaking back in after prompt coverage.
        if (!exactPrompt && !stemPrompt && this.isContentType(curType)) {
          if (routedSemantic < 0.10) score -= 2.8;
          if (coverageNow >= 0.45 && routedSemantic < 0.20) score -= 1.5;
        }

        if (prevToken !== null && contextModel.promptTransitions.get(prevToken)?.has(i) && !generatedSet.has(i)) {
          score += 4.6;
        }

        score += this.roleBias(curType, sentenceState);
        score += this.nextRoleLookahead(curType, sentenceState);
        score += this.questionRelationBonus(i, sentenceState, contextModel);

        score += this.dynamicWordPatternScore(
          i,
          generated,
          contextModel,
          sentenceState,
          patternContext
        );

        if (!sentenceState.isStart && /^[A-Z][a-z]/.test(word) && curType !== "ProperNoun") score -= 2.0;
      }

      if (curType === "Punct") {
        const coverage = this.promptCoverage(generated, contextModel);
        const answered = this.answerRelationSatisfied(
          generated,
          sentenceState,
          contextModel
        );
        const goodStop =
          answered &&
          step >= 3 &&
          coverage >= (deepMode ? 0.20 : 0.15);

        if (word === ".") score += goodStop ? 8.5 : -10.0;
        else score += goodStop ? 0.2 : -10.0;

        // The user asked a question; the model is answering it, not echoing it.
        if (contextModel?.questionProfile?.isQuestion && word === "?") {
          score -= 12.0;
        }
      }

      // Once Deep has subject + predicate + at least one complement, ending the
      // sentence should beat inventing another clause unless a clearly relevant
      // prepositional/modifier continuation is much stronger.
      if (deepMode && this.clauseComplete(sentenceState) && sentenceState.objectCount >= 1 && step >= 4) {
        const clauseLengthPressure =
          Math.min(8.0, Math.max(0, step - 7) * 0.65);

        if (curType === "Punct" && word === ".") {
          score += 10.0 + clauseLengthPressure;
        } else if (["Verb", "Modal", "Aux"].includes(curType)) {
          score -= 9.0;
        } else if (
          this.isContentType(curType) &&
          !exactPrompt &&
          !stemPrompt
        ) {
          score -= 2.8 + Math.max(0, step - targetLength) * 0.35;
        } else if (
          ["Conj", "Prep"].includes(curType) &&
          step >= targetLength
        ) {
          score -= 3.0;
        }
      }

      scores[i] = score;
    }

    return scores;
  }

  selectLowToken(scores, temperature) {
    let maxLogit = -1e9;
    for (let i = 0; i < scores.length; i++) if (scores[i] > maxLogit) maxLogit = scores[i];

    let expSum = 0;
    const probs = new Float32Array(scores.length);
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] <= -1e8) continue;
      const p = Math.exp((scores[i] - maxLogit) / temperature);
      probs[i] = p;
      expSum += p;
    }

    if (expSum <= 0) return { token: -1, prob: 0.0 };
    const r = Math.random() * expSum;
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return { token: i, prob: probs[i] / expSum };
    }
    return { token: -1, prob: 0.0 };
  }

  selectDeepToken(scores) {
    // Deep does a tiny top-K rerank from the SAME neural pass instead of
    // generating 3 complete rollouts. This is the efficiency win.
    const K = 8;
    const topIds = new Int32Array(K).fill(-1);
    const topVals = new Float32Array(K).fill(-1e9);

    for (let i = 0; i < scores.length; i++) {
      const s = scores[i];
      if (s <= topVals[K - 1]) continue;
      let pos = K - 1;
      while (pos > 0 && s > topVals[pos - 1]) {
        topVals[pos] = topVals[pos - 1];
        topIds[pos] = topIds[pos - 1];
        pos--;
      }
      topVals[pos] = s;
      topIds[pos] = i;
    }

    if (topIds[0] < 0) return { token: -1, prob: 0.0, margin: 0.0 };

    // Deterministic best candidate. Margin is a cheap certainty estimate.
    const margin = K > 1 && topIds[1] >= 0 ? Math.max(0, topVals[0] - topVals[1]) : 8.0;
    const pseudoProb = 1.0 / (1.0 + Math.exp(-margin));
    return { token: topIds[0], prob: pseudoProb, margin };
  }

  updateSentenceState(sentenceState, token, word, curType) {
    const lower = word.toLowerCase();

    // A conjunction after a complete proposition opens a fresh clause while
    // remaining in the same sentence ("X happens because Y changes").
    if (curType === "Conj" && this.clauseComplete(sentenceState)) {
      sentenceState.isStart = true;
      sentenceState.subjectReady = false;
      sentenceState.predicateReady = false;
      sentenceState.subjectToken = null;
      sentenceState.verbCount = 0;
      sentenceState.nounCount = 0;
      sentenceState.objectCount = 0;
      sentenceState.prepCount = 0;
      sentenceState.isPluralSubject = false;
      return;
    }

    if (!sentenceState.subjectReady && ["Noun", "ProperNoun", "Pronoun"].includes(curType)) {
      sentenceState.subjectReady = true;
      sentenceState.subjectToken = token;
      sentenceState.isPluralSubject = lower.endsWith("s") && !lower.endsWith("ss");
      sentenceState.nounCount++;
    } else if (["Noun", "ProperNoun", "Pronoun"].includes(curType)) {
      sentenceState.nounCount++;

      // Before the predicate, the newest noun in a noun compound is usually
      // the grammatical head ("AI model chips" -> plural head "chips").
      if (!sentenceState.predicateReady) {
        sentenceState.subjectToken = token;
        sentenceState.isPluralSubject = lower.endsWith("s") && !lower.endsWith("ss");
      } else {
        sentenceState.objectCount++;
      }
    }

    if (curType === "Verb") {
      sentenceState.verbCount++;
      if (sentenceState.subjectReady) sentenceState.predicateReady = true;
    } else if (curType === "Aux") {
      sentenceState.verbCount++;
      // Copular/auxiliary constructions can be predicates once followed by a
      // complement; don't immediately call them complete here.
    } else if (curType === "Prep") {
      sentenceState.prepCount++;
    }

    // "is/are/be + noun/adjective" style predicate completion.
    if (sentenceState.subjectReady && sentenceState.verbCount > 0 && ["Adj", "Noun", "ProperNoun"].includes(curType)) {
      sentenceState.predicateReady = true;
    }

    sentenceState.isStart = false;
  }

  async executeRollout(
    cleanPromptTokens,
    contextModel,
    targetLength,
    maxTokens,
    temperature,
    deepMode,
    sharedRawLogits = null
  ) {
    const trajectory = [];
    const generated = [];
    let step = 0;
    let rawLogits = null;

    const sentenceState = {
      isStart: true,
      subjectReady: false,
      predicateReady: false,
      subjectToken: null,
      verbCount: 0,
      nounCount: 0,
      objectCount: 0,
      prepCount: 0,
      isPluralSubject: false
    };

    // HIGH-PERFORMANCE DEEP PATH:
    // One real neural forward pass for the whole answer. No cache, no canned
    // answer, no second model. The neural logits are prompt-derived once, then
    // the POS/clause planner spends cheap CPU work assembling the response.
    // This removes the dominant latency source: a 15-layer WebGPU dispatch +
    // GPU->CPU readback for every generated token.
    if (deepMode) {
      rawLogits = sharedRawLogits ||
        await this.network.forwardSequence(cleanPromptTokens);

      // Initialize latent reasoning ONCE. Later RSL passes preserve and refine
      // this state instead of throwing it away and rebuilding it.
      if (!contextModel.reasoningInitialized) {
        const latentReasoning = this.runLatentReasoning(
          rawLogits,
          contextModel,
          contextModel.latentHopBudget || 1
        );

        contextModel.reasoningCentroid = latentReasoning.centroid;
        contextModel.reasoningScores = latentReasoning.scores;
        contextModel.reasoningStability = latentReasoning.stability;
        contextModel.reasoningHops = latentReasoning.hopsUsed;
        contextModel.reasoningInitialized = true;

        // Build the expensive vocabulary-pruning structure once per prompt.
        contextModel.deepCandidatePool = this.buildDeepCandidatePool(
          rawLogits,
          contextModel,
          cleanPromptTokens
        );
      }

      // First try the universal grammatical planner. It now routes semantics
      // through topic + intent centroids plus the refined latent state.
      // nothing after the single neural pass and directly fixes the biggest
      // current failure: knowing the right words but placing them in nonsense
      // order.
      const frame = this.buildUniversalPromptFrame(
        cleanPromptTokens,
        rawLogits,
        contextModel
      );

      if (frame?.complete && frame.tokens.length >= 3) {
        for (const tok of frame.tokens) {
          if (step >= maxTokens) break;

          const word = this.tokenWord(tok);
          const curType = this.getCanonicalType(tok);
          generated.push(tok);
          this.updateSentenceState(sentenceState, tok, word, curType);
          step++;
        }

        trajectory.push(...this.buildFrameTrajectory(frame.tokens, rawLogits));
      }

      // Continue from the frame (or from an empty frame) using the SAME neural
      // logits. Deep can now produce multiple sentences and may consume any
      // amount up to the 360-token ceiling when the prompt warrants it.
      let sentenceCount = 0;

      while (step < maxTokens) {
        const scores = this.buildScores(
          rawLogits,
          generated,
          contextModel,
          sentenceState,
          true,
          step,
          targetLength
        );

        const picked = this.selectDeepToken(scores);
        if (picked.token < 0) break;

        const sampledToken = picked.token;
        const word = this.network.vocab?.idToToken?.[sampledToken] || "";
        const curType = this.getCanonicalType(sampledToken);

        trajectory.push({
          token: sampledToken,
          prob: picked.prob,
          margin: picked.margin || 0.0
        });

        generated.push(sampledToken);
        this.updateSentenceState(
          sentenceState,
          sampledToken,
          word,
          curType
        );
        step++;

        if ([".", "!", "?"].includes(word)) {
          sentenceCount++;

          const stop = this.shouldStopDeepAnswer(
            generated,
            sentenceState,
            contextModel,
            targetLength,
            sentenceCount
          );

          if (stop || step >= maxTokens) break;

          this.resetSentenceState(sentenceState);
          continue;
        }

        // If an explicit length request reaches its ceiling mid-sentence, let
        // the final punctuation append below close it cleanly.
        if (
          contextModel?.explicitRequestedLength !== null &&
          contextModel?.explicitRequestedLength !== undefined &&
          step >= contextModel.explicitRequestedLength
        ) {
          break;
        }
      }
    } else {
      // LOW PATH keeps the original autoregressive behavior. It asks the neural
      // network to refresh its logits after each generated token.
      let state = [...cleanPromptTokens];

      while (step < maxTokens) {
        const rawLogits = await this.network.forwardSequence(state);
        const scores = this.buildScores(rawLogits, generated, contextModel, sentenceState, false, step, targetLength);
        const picked = this.selectLowToken(scores, temperature);
        if (picked.token < 0) break;

        const sampledToken = picked.token;
        const word = this.network.vocab?.idToToken?.[sampledToken] || "";
        const curType = this.getCanonicalType(sampledToken);

        trajectory.push({ token: sampledToken, prob: picked.prob, margin: 0.0 });
        generated.push(sampledToken);
        state.push(sampledToken);
        this.updateSentenceState(sentenceState, sampledToken, word, curType);
        step++;

        if ([".", "!", "?"].includes(word)) {
          if (step >= targetLength) break;

          sentenceState.isStart = true;
          sentenceState.subjectReady = false;
          sentenceState.predicateReady = false;
          sentenceState.subjectToken = null;
          sentenceState.verbCount = 0;
          sentenceState.nounCount = 0;
          sentenceState.objectCount = 0;
          sentenceState.prepCount = 0;
          sentenceState.isPluralSubject = false;
        }
      }
    }

    const lastWord = this.network.vocab?.idToToken?.[generated[generated.length - 1]] || "";
    if (![".", "!", "?"].includes(lastWord)) {
      const periodId = this.network.vocab?.tokenToId?.get?.(".") ?? 432;
      generated.push(periodId);
    }

    return {
      tokens: generated,
      trajectory,
      rawLogits: deepMode ? rawLogits : null
    };
  }

  verifyCandidate(candidate, contextModel, targetLength) {
    const tokens = candidate.tokens || [];
    if (!tokens.length) {
      return {
        score: -1.0,
        semanticSim: 0.0,
        syntaxScore: 0.0,
        clauseOrderScore: 0.0,
        fluencyScore: 0.0,
        globalCoherenceScore: 0.0,
        sequenceIntegrityScore: 0.0,
        clauseIntegrityScore: 0.0,
        answerCompletionScore: 0.0,
        promptCoverage: 0.0,
        contentPrecision: 0.0,
        dualRouteCoherence: 0.0,
        topicRouteSupport: 0.0,
        intentRouteSupport: 0.0,
        latentStability: 0.0,
        repetitionScore: 0.0,
        stemDiversityScore: 0.0,
        rolloutCertainty: 0.0
      };
    }

    let semantic = 0;
    let contentCount = 0;
    let relevantContentCount = 0;
    let subjects = 0;
    let verbs = 0;
    let prevType = null;
    let illegal = 0;
    let firstSubjectIndex = -1;
    let firstPredicateIndex = -1;
    let legalTransitions = 0;
    let transitionCount = 0;
    let transitionStrength = 0.0;
    let maxFunctionRun = 0;
    let functionRun = 0;
    let nonPunctCount = 0;
    let functionTokenCount = 0;

    // Global-coherence bookkeeping. These are linear in output length and do
    // not touch the expensive neural inference path.
    const lexicalSequence = [];
    const posSequence = [];
    const clauseSignatures = [];

    let clauseTypes = [];
    let clauseHasNominal = false;
    let clauseHasPredicate = false;
    let clauseContentCount = 0;
    let validClauses = 0;
    let totalClauses = 0;

    const finishClause = () => {
      if (!clauseTypes.length) return;

      const meaningful = clauseContentCount > 0 || clauseHasPredicate;
      if (meaningful) {
        totalClauses++;

        // A normal declarative clause needs a predicate plus an argument-like
        // element. A leading verb is allowed as a generic imperative shape.
        const leadingVerb =
          clauseTypes[0] === "Verb" ||
          clauseTypes[0] === "Aux" ||
          clauseTypes[0] === "Modal";

        if (
          clauseHasPredicate &&
          (clauseHasNominal || leadingVerb)
        ) {
          validClauses++;
        }

        // Keep only structure, never prompt-specific words.
        clauseSignatures.push(clauseTypes.slice(0, 12).join(">"));
      }

      clauseTypes = [];
      clauseHasNominal = false;
      clauseHasPredicate = false;
      clauseContentCount = 0;
    };

    const contentStems = [];
    for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
      const tok = tokens[tokenIndex];
      const type = this.getCanonicalType(tok);
      const word = this.network.vocab?.idToToken?.[tok] || "";
      const lowerWord = word.toLowerCase();

      if (type !== "Punct") {
        nonPunctCount++;
        posSequence.push(type);

        const lexicalStem = this.stemWord(lowerWord);
        lexicalSequence.push(
          lexicalStem.length >= 2 ? lexicalStem : lowerWord
        );

        const isFunctionToken =
          ["Det", "Prep", "Conj", "Modal", "Aux"].includes(type);
        if (isFunctionToken) functionTokenCount++;

        // A conjunction closes the preceding proposition and opens another.
        if (type === "Conj" && clauseTypes.length) {
          finishClause();
        }

        clauseTypes.push(type);

        if (["Noun", "ProperNoun", "Pronoun"].includes(type)) {
          clauseHasNominal = true;
        }
        if (["Verb", "Aux", "Modal"].includes(type)) {
          clauseHasPredicate = true;
        }
        if (this.isContentType(type)) {
          clauseContentCount++;
        }
      } else {
        finishClause();
      }

      if (this.isContentType(type)) {
        const routedSupport = this.getRoutedSemanticScore(
          tok,
          contextModel,
          null
        );
        const semanticSupport = Math.max(0, routedSupport);
        semantic += semanticSupport;
        contentCount++;

        const stem = this.stemWord(word);
        if (stem.length >= 3) contentStems.push(stem);

        const exactPrompt = contextModel.promptContentSet?.has(tok);
        const stemPrompt = stem.length >= 3 && contextModel.promptStems?.has(stem);
        if (exactPrompt || stemPrompt || semanticSupport >= 0.20) {
          relevantContentCount++;
        }
      }

      if (["Noun", "ProperNoun", "Pronoun"].includes(type)) {
        subjects++;
        if (firstSubjectIndex < 0) firstSubjectIndex = tokenIndex;
      }
      if (type === "Verb" || type === "Aux" || type === "Modal") {
        if (type !== "Modal") verbs++;
        if (firstPredicateIndex < 0) firstPredicateIndex = tokenIndex;
      }

      if (type !== "Punct") {
        if (prevType) {
          const transition = this.getTransitionRule(prevType, type);
          transitionCount++;

          if (transition.legal) {
            legalTransitions++;
            // Most valid matrix weights live roughly in the 0-5 range.
            transitionStrength += Math.max(
              0,
              Math.min(1, transition.weight / 5.0)
            );
          } else {
            illegal++;
          }
        }

        const isFunction = ["Det", "Prep", "Conj", "Modal", "Aux"].includes(type);
        if (isFunction) {
          functionRun++;
          maxFunctionRun = Math.max(maxFunctionRun, functionRun);
        } else {
          functionRun = 0;
        }

        prevType = type;
      } else {
        functionRun = 0;
      }
    }

    // Close a final unterminated structural segment for analysis.
    finishClause();

    const repeatedNgramPenalty = (sequence, n) => {
      if (sequence.length < n * 2) return 0.0;

      const counts = new Map();
      const total = sequence.length - n + 1;

      for (let i = 0; i <= sequence.length - n; i++) {
        const key = sequence.slice(i, i + n).join("|");
        counts.set(key, (counts.get(key) || 0) + 1);
      }

      let repeatedOccurrences = 0;
      let maxExcess = 0;

      for (const count of counts.values()) {
        if (count > 1) {
          const excess = count - 1;
          repeatedOccurrences += excess;
          maxExcess = Math.max(maxExcess, excess);
        }
      }

      const repeatRatio = repeatedOccurrences / Math.max(1, total);

      return Math.max(
        0,
        Math.min(
          1,
          repeatRatio * 1.65 +
          Math.min(0.45, maxExcess * 0.12)
        )
      );
    };

    const lexical3Penalty = repeatedNgramPenalty(lexicalSequence, 3);
    const lexical4Penalty = repeatedNgramPenalty(lexicalSequence, 4);
    const pos4Penalty = repeatedNgramPenalty(posSequence, 4);

    // Repeating an entire clause shape is a strong degeneration signal even
    // when the words are slightly different.
    let repeatedClauseShapes = 0;
    if (clauseSignatures.length > 1) {
      const shapeCounts = new Map();
      for (const signature of clauseSignatures) {
        shapeCounts.set(signature, (shapeCounts.get(signature) || 0) + 1);
      }
      for (const count of shapeCounts.values()) {
        if (count > 1) repeatedClauseShapes += count - 1;
      }
    }

    const clauseShapePenalty = clauseSignatures.length
      ? Math.min(
          1,
          repeatedClauseShapes / Math.max(1, clauseSignatures.length - 1)
        )
      : 0.0;

    const degenerationPenalty = Math.max(
      lexical3Penalty,
      lexical4Penalty,
      pos4Penalty * 0.72,
      clauseShapePenalty * 0.90
    );

    const sequenceIntegrityScore = Math.max(
      0,
      Math.min(1, 1.0 - degenerationPenalty)
    );

    const clauseIntegrityScore = totalClauses
      ? Math.max(0, Math.min(1, validClauses / totalClauses))
      : 0.0;

    // Extremely glue-heavy output is often syntactically legal word salad.
    // This is deliberately a broad linguistic check, not a prompt rule.
    const functionRatio = nonPunctCount
      ? functionTokenCount / nonPunctCount
      : 0.0;

    const functionBalanceScore = Math.max(
      0,
      Math.min(
        1,
        1.0 - Math.max(0, functionRatio - 0.48) / 0.32
      )
    );

    const semanticSim = contentCount ? semantic / contentCount : 0.0;
    const contentPrecision = contentCount
      ? Math.max(0, Math.min(1, relevantContentCount / contentCount))
      : 0.0;
    const promptCoverage = this.promptCoverage(tokens, contextModel);

    const questionProfile = contextModel?.questionProfile;
    let operatorEchoCount = 0;

    if (
      questionProfile?.isQuestion &&
      Number.isFinite(questionProfile.operatorToken)
    ) {
      for (const tok of tokens) {
        if (tok === questionProfile.operatorToken) operatorEchoCount++;
      }
    }

    const operatorEchoScore = questionProfile?.isQuestion
      ? 1.0 / (1.0 + operatorEchoCount)
      : 1.0;

    const novelContentCount = this.countNovelContent(tokens, contextModel);
    const novelContentScore = contentCount
      ? Math.max(
          0,
          Math.min(
            1,
            novelContentCount / Math.max(1, contentCount * 0.22)
          )
        )
      : 0.0;

    // A question is considered "answered" only when the response forms a real
    // proposition, stays connected to the prompt, and adds non-echoed content.
    // No how/why/etc. answer templates are used.
    const answerCompletionScore = questionProfile?.isQuestion
      ? Math.max(
          0,
          Math.min(
            1,
            (
              clauseIntegrityScore * 0.32 +
              sequenceIntegrityScore * 0.24 +
              novelContentScore * 0.20 +
              Math.min(1, promptCoverage * 1.6) * 0.14 +
              operatorEchoScore * 0.10
            )
          )
        )
      : Math.max(
          0,
          Math.min(
            1,
            clauseIntegrityScore * 0.55 +
            sequenceIntegrityScore * 0.45
          )
        );

    // Global coherence operates above adjacent POS legality. This is what
    // catches output that is locally legal but globally scrambled.
    const globalCoherenceScore = Math.max(
      0,
      Math.min(
        1,
        clauseIntegrityScore * 0.38 +
        sequenceIntegrityScore * 0.34 +
        functionBalanceScore * 0.16 +
        operatorEchoScore * 0.12
      )
    );

    // Dual-centroid coherence asks whether the response remained connected to
    // BOTH what the prompt is about and what it is trying to do.
    let topicSupportSum = 0.0;
    let intentSupportSum = 0.0;
    let routeCount = 0;

    for (const tok of tokens) {
      const type = this.getCanonicalType(tok);
      if (!this.isContentType(type)) continue;

      topicSupportSum += Math.max(0, contextModel.topicScores?.[tok] || 0);
      intentSupportSum += Math.max(0, contextModel.intentScores?.[tok] || 0);
      routeCount++;
    }

    const topicRouteSupport = routeCount
      ? topicSupportSum / routeCount
      : 0.0;

    const intentRouteSupport = routeCount
      ? intentSupportSum / routeCount
      : 0.0;

    const dualRouteCoherence = Math.sqrt(
      Math.max(0, topicRouteSupport) *
      Math.max(0, intentRouteSupport)
    );

    const latentStability = Math.max(
      0,
      Math.min(1, contextModel.reasoningStability || 0)
    );

    const terminal = this.network.vocab?.idToToken?.[tokens[tokens.length - 1]] || "";
    const closed = [".", "!", "?"].includes(terminal);
    const syntaxScore = Math.max(0, Math.min(1,
      (closed ? 0.25 : 0) +
      (subjects > 0 ? 0.30 : 0) +
      (verbs > 0 ? 0.35 : 0) +
      (illegal === 0 ? 0.10 : Math.max(0, 0.10 - illegal * 0.04))
    ));

    // Sentence-shape quality is intentionally stricter than basic POS legality.
    // It rewards a subject before the predicate and penalizes predicate pileups.
    let clauseOrderScore = 0.0;
    if (firstSubjectIndex >= 0) clauseOrderScore += 0.25;
    if (firstPredicateIndex >= 0) clauseOrderScore += 0.25;
    if (
      firstSubjectIndex >= 0 &&
      firstPredicateIndex > firstSubjectIndex
    ) clauseOrderScore += 0.30;
    if (closed) clauseOrderScore += 0.10;
    if (verbs === 1) clauseOrderScore += 0.10;
    else if (verbs > 2) clauseOrderScore -= Math.min(0.25, (verbs - 2) * 0.08);
    clauseOrderScore = Math.max(0, Math.min(1, clauseOrderScore));

    const legalRatio = transitionCount
      ? legalTransitions / transitionCount
      : 1.0;

    const meanTransitionStrength = legalTransitions
      ? transitionStrength / legalTransitions
      : 0.0;

    // Fluency is intentionally local. A response can have nouns + a verb + a
    // period and still be word salad if its adjacent grammatical transitions
    // are poor. Long runs of function words are also penalized.
    const functionRunPenalty = Math.max(
      0,
      Math.min(0.35, (maxFunctionRun - 2) * 0.10)
    );

    const fluencyScore = Math.max(
      0,
      Math.min(
        1,
        legalRatio * 0.58 +
        meanTransitionStrength * 0.30 +
        clauseOrderScore * 0.12 -
        functionRunPenalty
      )
    );

    const unique = new Set(tokens).size;
    const repetitionScore = Math.min(1, unique / Math.max(1, tokens.length) * 1.15);

    const uniqueStems = new Set(contentStems).size;
    const stemDiversityScore = contentStems.length
      ? Math.max(0, Math.min(1, uniqueStems / contentStems.length))
      : 1.0;

    const probs = candidate.trajectory?.map(x => x.prob || 0) || [];
    const rolloutCertainty = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0.0;

    const scaffoldSteps = candidate.trajectory?.filter(x => x.scaffold).length || 0;
    const frameUsage = tokens.length
      ? Math.max(0, Math.min(1, scaffoldSteps / Math.max(1, tokens.length - 1)))
      : 0.0;

    // Roughness is now explicitly punished. A response full of
    // accelerate/accelerates/accelerating/accelerated can no longer score as if
    // it were diverse, fluent output.
    const score =
      promptCoverage * 0.15 +
      contentPrecision * 0.14 +
      semanticSim * 0.10 +
      dualRouteCoherence * 0.10 +
      syntaxScore * 0.12 +
      clauseOrderScore * 0.11 +
      fluencyScore * 0.16 +
      repetitionScore * 0.04 +
      stemDiversityScore * 0.04 +
      rolloutCertainty * 0.02 +
      latentStability * 0.01 +
      frameUsage * 0.01;

    return {
      score,
      semanticSim,
      syntaxScore,
      clauseOrderScore,
      fluencyScore,
      globalCoherenceScore,
      sequenceIntegrityScore,
      clauseIntegrityScore,
      answerCompletionScore,
      promptCoverage,
      contentPrecision,
      dualRouteCoherence,
      topicRouteSupport,
      intentRouteSupport,
      latentStability,
      repetitionScore,
      stemDiversityScore,
      rolloutCertainty,
      frameUsage
    };
  }

  candidateQualityVector(candidate) {
    return {
      score: candidate?.score ?? -1.0,
      syntax: candidate?.syntaxScore ?? 0.0,
      order: candidate?.clauseOrderScore ?? 0.0,
      fluency: candidate?.fluencyScore ?? 0.0,
      precision: candidate?.contentPrecision ?? 0.0,
      coverage: candidate?.promptCoverage ?? 0.0,
      semantic: candidate?.semanticSim ?? 0.0,
      repetition: candidate?.repetitionScore ?? 0.0
    };
  }

  isSafeRSLUpgrade(candidate, currentBest) {
    if (!candidate) return false;
    if (!currentBest) return true;

    const next = this.candidateQualityVector(candidate);
    const base = this.candidateQualityVector(currentBest);

    // Never trade a meaningful amount of readability for a tiny composite gain.
    if (next.fluency < base.fluency - 0.025) return false;
    if (next.syntax < base.syntax - 0.025) return false;
    if (next.order < base.order - 0.035) return false;
    if (next.precision < base.precision - 0.045) return false;
    if (next.repetition < base.repetition - 0.050) return false;

    // A candidate that is visibly cleaner may replace the baseline even when
    // its scalar score is nearly tied. Otherwise demand a real improvement.
    const languageGain =
      (next.fluency - base.fluency) * 0.50 +
      (next.syntax - base.syntax) * 0.25 +
      (next.order - base.order) * 0.25;

    const relevanceGain =
      (next.precision - base.precision) * 0.45 +
      (next.coverage - base.coverage) * 0.30 +
      (next.semantic - base.semantic) * 0.25;

    return (
      next.score >= base.score + 0.003 ||
      languageGain >= 0.025 ||
      relevanceGain >= 0.030
    );
  }

  computeConfidence(candidate) {
    if (!candidate || !candidate.tokens?.length) return 0.0;

    const clamp = x =>
      Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));

    const relevanceEvidence = clamp(
      clamp(candidate.promptCoverage) * 0.44 +
      clamp(candidate.contentPrecision) * 0.34 +
      clamp(candidate.dualRouteCoherence) * 0.22
    );

    const languageEvidence = clamp(
      clamp(candidate.syntaxScore) * 0.24 +
      clamp(candidate.clauseOrderScore) * 0.22 +
      clamp(candidate.fluencyScore) * 0.30 +
      clamp(candidate.globalCoherenceScore) * 0.24
    );

    const semanticEvidence = clamp(
      clamp(candidate.semanticSim) * 0.42 +
      clamp(candidate.dualRouteCoherence) * 0.32 +
      clamp(candidate.contentPrecision) * 0.26
    );

    const consistencyEvidence = clamp(
      clamp(candidate.repetitionScore) * 0.34 +
      clamp(candidate.stemDiversityScore) * 0.30 +
      clamp(candidate.sequenceIntegrityScore) * 0.36
    );

    const answerEvidence = clamp(candidate.answerCompletionScore);
    const neuralEvidence = clamp(candidate.rolloutCertainty);
    const reasoningEvidence = clamp(candidate.latentStability);
    const coherenceEvidence = clamp(candidate.globalCoherenceScore);
    const sequenceEvidence = clamp(candidate.sequenceIntegrityScore);

    // Raw evidence estimate.
    const base =
      relevanceEvidence * 0.23 +
      languageEvidence * 0.23 +
      semanticEvidence * 0.16 +
      answerEvidence * 0.13 +
      consistencyEvidence * 0.10 +
      neuralEvidence * 0.09 +
      reasoningEvidence * 0.06;

    // Cross-channel agreement. Strong answers should agree across dimensions;
    // weak answers should not receive a free calibration boost.
    const core = [
      relevanceEvidence,
      languageEvidence,
      semanticEvidence,
      answerEvidence
    ];

    const mean =
      core.reduce((sum, value) => sum + value, 0) / core.length;

    let variance = 0.0;
    for (const value of core) {
      const diff = value - mean;
      variance += diff * diff;
    }
    variance /= core.length;

    const disagreement = Math.min(1, Math.sqrt(variance) * 1.45);

    const integrity =
      coherenceEvidence * 0.45 +
      sequenceEvidence * 0.30 +
      consistencyEvidence * 0.25;

    // Smoothly punish broken structure without stacking multiple harsh gates.
    const qualityFactor = clamp(
      1.0 -
      disagreement * 0.13 -
      (1.0 - integrity) * 0.10
    );

    const rawConfidence = clamp(base * qualityFactor);

    // ------------------------------------------------------------
    // High-confidence calibration
    // ------------------------------------------------------------
    // The old confidence system compressed even genuinely strong answers into
    // the 60-80% band. This calibration expands the upper range ONLY when the
    // response has strong, mutually-agreeing evidence.
    //
    // It is not a blanket +20% boost. Weak/word-salad answers stay low because
    // they fail the eligibility gate below.
    const highQualityGate = clamp(
      Math.min(
        relevanceEvidence,
        languageEvidence,
        semanticEvidence,
        answerEvidence,
        integrity
      )
    );

    let calibrated = rawConfidence;

    // Strong answers can enter the "inhuman" 90%+ range.
    if (highQualityGate >= 0.72 && rawConfidence >= 0.72) {
      const strength = clamp(
        (rawConfidence - 0.72) / 0.20
      );

      // 0.72 -> ~0.84
      // 0.80 -> ~0.90
      // 0.86 -> ~0.94
      // 0.92+ -> upper 90s
      const target =
        0.84 +
        0.14 * Math.pow(strength, 0.72);

      calibrated = Math.max(
        rawConfidence,
        target
      );
    }

    // Extra promotion for very strong agreement across all major channels.
    if (
      highQualityGate >= 0.80 &&
      rawConfidence >= 0.78 &&
      disagreement <= 0.10
    ) {
      const eliteStrength = clamp(
        (rawConfidence - 0.78) / 0.16
      );

      const eliteTarget =
        0.90 +
        0.085 * Math.pow(eliteStrength, 0.70);

      calibrated = Math.max(
        calibrated,
        eliteTarget
      );
    }

    return clamp(calibrated);
  }

  // ========================================================
  // ANYTIME PIPELINE
  // Deep = ONE neural rollout with stronger per-token planning.
  // Low  = ONE simpler stochastic rollout.
  // ========================================================
  async generate(rawPromptTokens, onProgress = null) {
    const cleanPromptTokens = rawPromptTokens
      .map(t => (typeof t === "object" && t !== null ? t.id : t))
      .filter(Number.isFinite);

    if (!cleanPromptTokens.length) {
      return { tokens: [], confidence: "0.0" };
    }

    const deepMode = this.mode === "rsl" && this.isDeepLearning;
    const promptLen = cleanPromptTokens.length;
    const contextModel = this.extractContextModel(cleanPromptTokens);

    const targetLength = deepMode
      ? this.estimateAnswerLength(cleanPromptTokens, contextModel)
      : Math.max(
          8,
          Math.min(
            this.maxTokenCeiling,
            Math.round(8 + Math.sqrt(promptLen) * 5 + promptLen * 0.55)
          )
        );

    const maxTokens = deepMode
      ? this.getAdaptiveGenerationBudget(
          promptLen,
          targetLength,
          contextModel
        )
      : Math.min(
          this.maxTokenCeiling,
          Math.max(targetLength, Math.round(targetLength * 1.20))
        );

    contextModel.latentHopBudget = deepMode
      ? this.getAdaptiveLatentHops(promptLen, targetLength)
      : 1;

    if (!deepMode) {
      const candidate = await this.executeRollout(
        cleanPromptTokens,
        contextModel,
        targetLength,
        maxTokens,
        0.42,
        false,
        null
      );

      const evaluation = this.verifyCandidate(
        candidate,
        contextModel,
        targetLength
      );
      const winner = { ...candidate, ...evaluation };

      if (onProgress) onProgress(1, 1, winner.score);

      this.trajectory = winner.trajectory || [];
      const finalConf = this.computeConfidence(winner);

      return {
        tokens: winner.tokens,
        confidence: (finalConf * 100).toFixed(1)
      };
    }

    // ================================================================
    // RSL DEEP: INPUT-ADAPTIVE ANYTIME SEARCH, UP TO 60 PASSES
    // ================================================================
    const plan = this.getAdaptiveRSLPassPlan(
      promptLen,
      targetLength
    );

    let best = null;
    let sharedRawLogits = null;
    let passesUsed = 0;
    const bestHistory = [];

    for (let pass = 0; pass < plan.maxPasses; pass++) {
      passesUsed = pass + 1;
      contextModel.searchPass = pass;

      // High-performance path: only the first pass pays for the full 3M-neuron
      // forward. Later passes reuse it and refine the latent/semantic state.
      // This preserves the 60-pass capability without multiplying GPU latency.
      if (pass === 0) {
        sharedRawLogits = null;
      }

      const candidate = await this.executeRollout(
        cleanPromptTokens,
        contextModel,
        targetLength,
        maxTokens,
        0.20,
        true,
        sharedRawLogits
      );

      if (!sharedRawLogits && candidate.rawLogits) {
        sharedRawLogits = candidate.rawLogits;
      }

      const evaluation = this.verifyCandidate(
        candidate,
        contextModel,
        targetLength
      );

      const entry = {
        ...candidate,
        ...evaluation
      };
      entry.confidenceValue = this.computeConfidence(entry);

      // Pass 0 is the protected quality floor. Later RSL passes are allowed to
      // replace it only when they are a safe upgrade, so thinking longer cannot
      // casually destroy grammar or relevance.
      if (this.isSafeRSLUpgrade(entry, best)) {
        best = entry;
      }

      bestHistory.push(best?.score ?? entry.score);

      if (onProgress) {
        onProgress(
          pass + 1,
          Math.max(plan.preferredPasses, pass + 1),
          best.score
        );
      }

      // Decide whether another pass is worth paying for BEFORE refining.
      if (
        this.shouldStopRSLSearch(
          bestHistory,
          best,
          pass,
          plan
        )
      ) {
        break;
      }

      // Only refine if another pass will actually run.
      this.refineRSLContext(
        contextModel,
        best,
        pass
      );
    }

    const winner = best || {
      tokens: [],
      trajectory: [],
      confidenceValue: 0.0
    };

    this.trajectory = winner.trajectory || [];

    return {
      tokens: winner.tokens,
      confidence: (
        Number.isFinite(winner.confidenceValue)
          ? winner.confidenceValue * 100
          : this.computeConfidence(winner) * 100
      ).toFixed(1),
      passesUsed,
      preferredPasses: plan.preferredPasses,
      generationBudget: maxTokens
    };
  }

  train(userRewardSignal) {
    this.episodeCount++;
    this.cumulativeReward += userRewardSignal;

    this.learnTrajectoryPatterns(userRewardSignal);

    return userRewardSignal;
  }
}

if (typeof window !== "undefined") {
  window.RLEngine = RLEngine;
}
