/**
 * VilotNI 2.5 - StateProcessor.js
 *
 * ADAPTIVE MULTI-CHANNEL / HIGHER-ORDER STATE PROCESSOR
 *
 * Purpose:
 *   Resolve several near-confidence decoder or semantic possibilities using a
 *   bounded continuous-state graph before VilotNI collapses to one choice.
 *
 * This remains an ordinary classical CPU/GPU algorithm. "Multi-state" here
 * means that every candidate carries several simultaneous continuous channels,
 * not that the hardware is quantum or has infinite precision.
 *
 * Candidate channels:
 *   activation, certainty, uncertainty, momentum, consensus, contradiction,
 *   centrality, reliability, context, semantics, priors, robustness.
 *
 * Main algorithms:
 *   1. strict confidence-window gating (the 2% rule is still hard)
 *   2. graded attraction/repulsion graph instead of binary compatible/not
 *   3. higher-order triadic consensus and coalition tension
 *   4. graph-centrality estimation across the active candidate lattice
 *   5. uncertainty/certainty dynamics coupled to support and contradiction
 *   6. damped momentum propagation with adaptive step size
 *   7. entropy-adaptive sharpening and bounded raw-confidence displacement
 *   8. free-energy tracking, oscillation detection, and best-basin rollback
 *   9. perturbation-style collapse robustness and winner stability
 *  10. multi-signal priors for knowledge, focus, transition, reliability,
 *      context, semantics, and novelty
 *
 * Performance strategy:
 *   - default maximum of eight candidates
 *   - typed-array storage for all hot numeric channels
 *   - pair and triple indices are precomputed once
 *   - no object allocation inside pair/triple propagation loops
 *   - complexity is tiny for the normal four-branch word collapse
 *   - no additional decoder pass is introduced
 */
(() => {

  'use strict';


  const EPSILON = 1e-7;


  class VilotStateProcessor {

    constructor(options = {
}
) {

      this.options = {

        confidenceWindow:
          Math.max(0.001, Number(options.confidenceWindow) || 0.02),

        reinforcementStrength:
          this.clamp01(options.reinforcementStrength ?? 0.44),

        cancellationStrength:
          this.clamp01(options.cancellationStrength ?? 0.24),

        sharpenPower:
          Math.max(1, Number(options.sharpenPower) || 1.18),

        propagationPasses:
          Math.max(2, Math.min(6, options.propagationPasses | 0 || 4)),

        maxPropagationPasses:
          Math.max(3, Math.min(8, options.maxPropagationPasses | 0 || 6)),

        entropyPassBoost:
          Math.max(0, Math.min(3, options.entropyPassBoost | 0 || 1)),

        consensusWeight:
          this.clamp01(options.consensusWeight ?? 0.12),

        transitionWeight:
          this.clamp01(options.transitionWeight ?? 0.035),

        knowledgeWeight:
          this.clamp01(options.knowledgeWeight ?? 0.035),

        focusWeight:
          this.clamp01(options.focusWeight ?? 0.025),

        stabilityWeight:
          this.clamp01(options.stabilityWeight ?? 0.05),

        reliabilityWeight:
          this.clamp01(options.reliabilityWeight ?? 0.055),

        contextWeight:
          this.clamp01(options.contextWeight ?? 0.045),

        semanticWeight:
          this.clamp01(options.semanticWeight ?? 0.055),

        certaintyWeight:
          this.clamp01(options.certaintyWeight ?? 0.07),

        centralityWeight:
          this.clamp01(options.centralityWeight ?? 0.045),

        robustnessWeight:
          this.clamp01(options.robustnessWeight ?? 0.05),

        higherOrderWeight:
          this.clamp01(options.higherOrderWeight ?? 0.075),

        contradictionWeight:
          this.clamp01(options.contradictionWeight ?? 0.06),

        noveltyWeight:
          this.clamp01(options.noveltyWeight ?? 0.01),

        stateLiftCap:
          Math.max(0.01, Math.min(0.18, Number(options.stateLiftCap) || 0.075)),

        stateDropCap:
          Math.max(0.01, Math.min(0.25, Number(options.stateDropCap) || 0.12)),

        convergenceEpsilon:
          Math.max(0.00001, Math.min(0.01, Number(options.convergenceEpsilon) || 0.00035)),

        energyEpsilon:
          Math.max(0.000001, Math.min(0.005, Number(options.energyEpsilon) || 0.00002)),

        entropyTemperature:
          Math.max(0.005, Math.min(0.15, Number(options.entropyTemperature) || 0.025)),

        momentum:
          this.clamp01(options.momentum ?? 0.32),

        damping:
          this.clamp01(options.damping ?? 0.70),

        adaptiveStepMin:
          Math.max(0.05, Math.min(0.5, Number(options.adaptiveStepMin) || 0.18)),

        adaptiveStepMax:
          Math.max(0.2, Math.min(0.9, Number(options.adaptiveStepMax) || 0.56)),

        uncertaintyDecay:
          this.clamp01(options.uncertaintyDecay ?? 0.18),

        supportCertaintyGain:
          this.clamp01(options.supportCertaintyGain ?? 0.22),

        conflictUncertaintyGain:
          this.clamp01(options.conflictUncertaintyGain ?? 0.26),

        centralityIterations:
          Math.max(2, Math.min(6, options.centralityIterations | 0 || 3)),

        perturbationRadius:
          Math.max(0.001, Math.min(0.03, Number(options.perturbationRadius) || 0.006)),

        rollbackOnEnergyIncrease:
          options.rollbackOnEnergyIncrease !== false,

        maxStates:
          Math.max(4, Math.min(16, options.maxStates | 0 || 8)),

        amplitudeAmplificationEnabled:
          options.amplitudeAmplificationEnabled !== false,

        amplitudeWeight:
          this.clamp01(options.amplitudeWeight ?? 0.085),

        amplitudeSeedGain:
          Math.max(0, Math.min(0.08, Number(options.amplitudeSeedGain) || 0.035)),

        amplitudeCoupling:
          Math.max(0, Math.min(0.35, Number(options.amplitudeCoupling) || 0.11)),

        amplitudeMarkThreshold:
          this.clamp01(options.amplitudeMarkThreshold ?? 0.56),

        amplitudeRoundsMin:
          Math.max(1, Math.min(4, options.amplitudeRoundsMin | 0 || 1)),

        amplitudeRoundsMax:
          Math.max(1, Math.min(8, options.amplitudeRoundsMax | 0 || 4)),

        vectorFeatureCount: 12
      
}
;


      if (this.options.adaptiveStepMax < this.options.adaptiveStepMin) {

        this.options.adaptiveStepMax = this.options.adaptiveStepMin;

      
}


      const n = this.options.maxStates;

      const nn = n * n;

      const pairMax = (n * (n - 1)) >> 1;

      const tripleMax = n >= 3 ? (n * (n - 1) * (n - 2)) / 6 : 0;

      this.matrix = globalThis.VilotMatrixEngine
        ? new globalThis.VilotMatrixEngine({ matrixMaxRows: Math.max(64, n), matrixMaxCols: 16, matrixTopK: 16 })
        : null;
      this._amplitudeProbability = new Float32Array(n);
      this._queryProjection = new Float32Array(n);
      this._amplitudeMarked = new Uint8Array(n);
      this._stateQueryVector = new Float32Array(this.options.vectorFeatureCount);


      // Primary candidate channels.
      this._raw = new Float32Array(n);

      this._current = new Float32Array(n);

      this._next = new Float32Array(n);

      this._bestState = new Float32Array(n);


      this._certainty = new Float32Array(n);

      this._nextCertainty = new Float32Array(n);

      this._bestCertainty = new Float32Array(n);

      this._uncertainty = new Float32Array(n);


      this._velocity = new Float32Array(n);

      this._nextVelocity = new Float32Array(n);

      this._previousDelta = new Float32Array(n);

      this._probability = new Float32Array(n);


      // Evidence/prior channels.
      this._knowledge = new Float32Array(n);

      this._focus = new Float32Array(n);

      this._transition = new Float32Array(n);

      this._reliability = new Float32Array(n);

      this._context = new Float32Array(n);

      this._semantic = new Float32Array(n);

      this._novelty = new Float32Array(n);

      this._certaintyInput = new Float32Array(n);


      // Derived graph/dynamics channels.
      this._reinforcement = new Float32Array(n);

      this._inhibition = new Float32Array(n);

      this._support = new Float32Array(n);

      this._conflict = new Float32Array(n);

      this._higherSupport = new Float32Array(n);

      this._higherTension = new Float32Array(n);

      this._centrality = new Float32Array(n);

      this._centralityNext = new Float32Array(n);

      this._stability = new Float32Array(n);

      this._robustness = new Float32Array(n);

      this._score = new Float32Array(n);


      // Pair matrices.
      this._coherence = new Float32Array(nn);

      this._compatibility = new Float32Array(nn);

      this._attraction = new Float32Array(nn);

      this._repulsion = new Float32Array(nn);


      // Precomputed unique pair indices.
      this._pairA = new Uint8Array(pairMax);

      this._pairB = new Uint8Array(pairMax);

      this._pairCount = 0;


      for (let i = 0;
 i < n;
 i++) {

        for (let j = i + 1;
 j < n;
 j++) {

          this._pairA[this._pairCount] = i;

          this._pairB[this._pairCount] = j;

          this._pairCount++;

        
}

      
}


      // Precomputed unique triplets. For n=8 this is only 56 triples.
      this._tripleA = new Uint8Array(tripleMax);

      this._tripleB = new Uint8Array(tripleMax);

      this._tripleC = new Uint8Array(tripleMax);

      this._tripleCount = 0;


      for (let i = 0;
 i < n;
 i++) {

        for (let j = i + 1;
 j < n;
 j++) {

          for (let k = j + 1;
 k < n;
 k++) {

            this._tripleA[this._tripleCount] = i;

            this._tripleB[this._tripleCount] = j;

            this._tripleC[this._tripleCount] = k;

            this._tripleCount++;

          
}

        
}

      
}


      this.totalCollapses = 0;

      this.totalOperations = 0;

      this.totalProcessorMs = 0;

      this.totalPropagationPasses = 0;

      this.totalTripletsEvaluated = 0;

      this.totalOscillationEvents = 0;

      this.totalRollbacks = 0;

      this.lastDiagnostics = null;

    
}


    clamp01(value) {

      const n = Number(value);

      if (!Number.isFinite(n)) return 0;

      if (n <= 0) return 0;

      if (n >= 1) return 1;

      return n;

    
}


    fromPercent(value) {

      return this.clamp01(Number(value) / 100);

    
}


    toPercent(value) {

      return this.clamp01(value) * 100;

    
}


    distance(a, b) {

      return Math.abs(this.clamp01(a) - this.clamp01(b));

    
}


    withinWindow(a, b, window = this.options.confidenceWindow) {

      return this.distance(a, b) <= Math.max(EPSILON, Number(window) || 0);

    
}


    coherence(a, b, window = this.options.confidenceWindow) {

      const width = Math.max(EPSILON, Number(window) || EPSILON);

      const d = this.distance(a, b);


      // Hard rule: outside the confidence window, there is no interference.
      if (d > width) return 0;


      const x = d / width;

      return 0.5 + 0.5 * Math.cos(Math.PI * x);

    
}


    softNot(a) {

      return 1 - this.clamp01(a);

    
}


    softAnd(a, b) {

      return this.clamp01(a) * this.clamp01(b);

    
}


    softOr(a, b) {

      a = this.clamp01(a);

      b = this.clamp01(b);

      return 1 - ((1 - a) * (1 - b));

    
}


    blend(a, b, weight = 0.5) {

      const x = this.clamp01(a);

      const y = this.clamp01(b);

      const w = this.clamp01(weight);

      return x + (y - x) * w;

    
}


    reinforce(a, b, strength = this.options.reinforcementStrength, coherence = 1) {

      const x = this.clamp01(a);

      const y = this.clamp01(b);

      const k = this.clamp01(strength) * this.clamp01(coherence);

      return this.blend(x, this.softOr(x, y), k);

    
}


    cancel(a, b, strength = this.options.cancellationStrength, coherence = 1) {

      const x = this.clamp01(a);

      const y = this.clamp01(b);

      const k = this.clamp01(strength) * this.clamp01(coherence);

      return this.clamp01(x * (1 - y * k));

    
}


    sharpen(value, power = this.options.sharpenPower) {

      const x = this.clamp01(value);

      if (x <= 0 || x >= 1) return x;


      const p = Math.max(1, Number(power) || 1);

      const hi = Math.pow(x, p);

      const lo = Math.pow(1 - x, p);

      const denom = hi + lo;


      return denom > EPSILON ? hi / denom : x;

    
}


    _entropy(n) {

      if (n <= 1) return 0;


      let mean = 0;

      for (let i = 0;
 i < n;
 i++) mean += this._current[i];

      mean /= n;


      const temperature = Math.max(EPSILON, this.options.entropyTemperature);

      let z = 0;


      for (let i = 0;
 i < n;
 i++) {

        const centered = (this._current[i] - mean) / temperature;

        const exp = Math.exp(Math.max(-24, Math.min(24, centered)));

        this._probability[i] = exp;

        z += exp;

      
}


      if (z <= EPSILON) return 0;


      let entropy = 0;

      for (let i = 0;
 i < n;
 i++) {

        const p = this._probability[i] / z;

        if (p > EPSILON) entropy -= p * Math.log(p);

      
}


      return this.clamp01(entropy / Math.log(n));

    
}


    _variance(n) {

      if (!n) return 0;


      let mean = 0;

      for (let i = 0;
 i < n;
 i++) mean += this._current[i];

      mean /= n;


      let variance = 0;

      for (let i = 0;
 i < n;
 i++) {

        const d = this._current[i] - mean;

        variance += d * d;

      
}


      return variance / n;

    
}


    _meanUncertainty(n) {

      if (!n) return 0;

      let total = 0;

      for (let i = 0;
 i < n;
 i++) total += 1 - this._certainty[i];

      return total / n;

    
}


    _adaptiveSharpenPower(entropy, variance, conflictMean, passIndex) {

      const entropyBoost = entropy * 0.34;

      const conflictBoost = conflictMean * 0.16;

      const varianceRelief = Math.min(0.16, variance * 3.2);

      const passBoost = passIndex * 0.028;


      return Math.max(
        1,
        this.options.sharpenPower +
          entropyBoost +
          conflictBoost +
          passBoost -
          varianceRelief
      );

    
}


    _adaptiveStep(entropy, variance, conflictMean, uncertaintyMean) {

      const minStep = this.options.adaptiveStepMin;

      const maxStep = this.options.adaptiveStepMax;

      const span = maxStep - minStep;


      const drive = this.clamp01(
        entropy * 0.42 +
        uncertaintyMean * 0.25 +
        (1 - Math.min(1, variance * 8)) * 0.18 +
        (1 - conflictMean) * 0.15
      );


      return minStep + span * drive;

    
}


    _buildPairGraph(list, n, compatible, window) {

      const stride = this.options.maxStates;

      const matrixSpan = stride * stride;


      this._coherence.fill(0, 0, matrixSpan);

      this._compatibility.fill(0, 0, matrixSpan);

      this._attraction.fill(0, 0, matrixSpan);

      this._repulsion.fill(0, 0, matrixSpan);


      let coherentPairs = 0;

      let compatiblePairs = 0;

      let incompatiblePairs = 0;

      let ambiguousPairs = 0;

      let attractionTotal = 0;

      let repulsionTotal = 0;

      let operations = 0;


      const pairCount = (n * (n - 1)) >> 1;


      for (let p = 0;
 p < pairCount;
 p++) {

        const i = this._pairA[p];

        const j = this._pairB[p];

        const a = this._raw[i];

        const b = this._raw[j];

        const coh = this.coherence(a, b, window);


        operations += 2;

        if (coh <= 0) continue;


        coherentPairs++;


        const compat = this.clamp01(compatible(list[i], list[j]));

        const ij = i * stride + j;

        const ji = j * stride + i;


        this._coherence[ij] = coh;

        this._coherence[ji] = coh;

        this._compatibility[ij] = compat;

        this._compatibility[ji] = compat;


        // Compatibility is now graded. A 0.35 relation does not behave the
        // same as a hard contradiction, and a 0.75 relation is not identical
        // to exact agreement.
        const attraction =
          coh *
          compat *
          (0.42 + 0.58 * Math.sqrt(Math.max(0, a * b)));


        const repulsion =
          coh *
          (1 - compat) *
          (0.38 + 0.62 * Math.min(a, b));


        this._attraction[ij] = attraction;

        this._attraction[ji] = attraction;

        this._repulsion[ij] = repulsion;

        this._repulsion[ji] = repulsion;


        attractionTotal += attraction;

        repulsionTotal += repulsion;


        if (compat >= 0.62) compatiblePairs++;

        else if (compat <= 0.28) incompatiblePairs++;

        else ambiguousPairs++;


        operations += 8;

      
}


      return {

        coherentPairs,
        compatiblePairs,
        incompatiblePairs,
        ambiguousPairs,
        attractionTotal,
        repulsionTotal,
        operations
      
}
;

    
}


    _computeCentrality(n) {

      if (n <= 1) {

        if (n === 1) this._centrality[0] = 1;

        return 0;

      
}


      const stride = this.options.maxStates;

      let hasEdges = false;


      for (let i = 0;
 i < n;
 i++) {

        this._centrality[i] = 1 / n;

        const row = i * stride;

        for (let j = 0;
 j < n;
 j++) {

          if (this._attraction[row + j] > EPSILON) {

            hasEdges = true;

            break;

          
}

        
}

      
}


      if (!hasEdges) {

        for (let i = 0;
 i < n;
 i++) this._centrality[i] = 0.5;

        return 0;

      
}


      let operations = 0;


      for (let pass = 0;
 pass < this.options.centralityIterations;
 pass++) {

        let maxValue = EPSILON;


        for (let i = 0;
 i < n;
 i++) {

          const row = i * stride;

          let value = 0.04 + this._reliability[i] * 0.04;


          for (let j = 0;
 j < n;
 j++) {

            if (i === j) continue;

            value +=
              (
                this._attraction[row + j] +
                this._coherence[row + j] * 0.08
              ) *
              this._centrality[j];

            operations += 4;

          
}


          this._centralityNext[i] = value;

          if (value > maxValue) maxValue = value;

        
}


        for (let i = 0;
 i < n;
 i++) {

          this._centrality[i] = this.clamp01(this._centralityNext[i] / maxValue);

        
}

      
}


      return operations;

    
}


    _buildHigherOrderGraph(n) {

      this._higherSupport.fill(0, 0, n);

      this._higherTension.fill(0, 0, n);


      if (n < 3) {

        return {

          triplets: 0,
          activeTriplets: 0,
          synergyTotal: 0,
          tensionTotal: 0,
          operations: 0
        
}
;

      
}


      const stride = this.options.maxStates;

      const tripleCount = (n * (n - 1) * (n - 2)) / 6;

      let activeTriplets = 0;

      let synergyTotal = 0;

      let tensionTotal = 0;

      let operations = 0;


      for (let t = 0;
 t < tripleCount;
 t++) {

        const i = this._tripleA[t];

        const j = this._tripleB[t];

        const k = this._tripleC[t];


        const ij = i * stride + j;

        const ik = i * stride + k;

        const jk = j * stride + k;


        const aij = this._attraction[ij];

        const aik = this._attraction[ik];

        const ajk = this._attraction[jk];

        const rij = this._repulsion[ij];

        const rik = this._repulsion[ik];

        const rjk = this._repulsion[jk];


        const coherentEdges =
          (this._coherence[ij] > 0 ? 1 : 0) +
          (this._coherence[ik] > 0 ? 1 : 0) +
          (this._coherence[jk] > 0 ? 1 : 0);


        if (coherentEdges < 2) {

          operations += 3;

          continue;

        
}


        activeTriplets++;


        // Full three-way agreement gets geometric synergy. An open triad still
        // gets weaker support, which helps two agreeing alternatives stabilize
        // without pretending the third edge exists.
        let synergy = 0;

        if (aij > 0 && aik > 0 && ajk > 0) {

          synergy = Math.cbrt(aij * aik * ajk);

        
}
 else {

          const first = Math.max(aij, aik, ajk);

          const second = aij + aik + ajk - first - Math.min(aij, aik, ajk);

          synergy = Math.sqrt(Math.max(0, first * second)) * 0.24;

        
}


        const tensionI = ajk * ((rij + rik) * 0.5);

        const tensionJ = aik * ((rij + rjk) * 0.5);

        const tensionK = aij * ((rik + rjk) * 0.5);


        this._higherSupport[i] += synergy;

        this._higherSupport[j] += synergy;

        this._higherSupport[k] += synergy;


        this._higherTension[i] += tensionI;

        this._higherTension[j] += tensionJ;

        this._higherTension[k] += tensionK;


        synergyTotal += synergy * 3;

        tensionTotal += tensionI + tensionJ + tensionK;

        operations += 20;

      
}


      const norm = Math.max(1, n - 2);

      for (let i = 0;
 i < n;
 i++) {

        this._higherSupport[i] = this.clamp01(this._higherSupport[i] / norm);

        this._higherTension[i] = this.clamp01(this._higherTension[i] / norm);

      
}


      return {

        triplets: tripleCount,
        activeTriplets,
        synergyTotal,
        tensionTotal,
        operations
      
}
;

    
}


    _freeEnergy(n) {

      if (!n) return 0;


      const stride = this.options.maxStates;

      let energy = 0;


      for (let i = 0;
 i < n;
 i++) {

        const displacement = this._current[i] - this._raw[i];

        energy += displacement * displacement * 0.34;

        energy += (1 - this._certainty[i]) * 0.055;

        energy -= this._semantic[i] * this._current[i] * 0.025;

        energy -= this._reliability[i] * this._current[i] * 0.02;

        energy -= this._higherSupport[i] * this._current[i] * 0.035;

        energy += this._higherTension[i] * this._current[i] * 0.05;

      
}


      const pairCount = (n * (n - 1)) >> 1;

      for (let p = 0;
 p < pairCount;
 p++) {

        const i = this._pairA[p];

        const j = this._pairB[p];

        const ij = i * stride + j;

        const product = this._current[i] * this._current[j];


        energy -= this._attraction[ij] * product * 0.23;

        energy += this._repulsion[ij] * product * 0.31;

      
}


      return energy / n;

    
}


    _propagate(n, passIndex) {

      const stride = this.options.maxStates;

      const entropy = this._entropy(n);

      const variance = this._variance(n);

      const uncertaintyMean = this._meanUncertainty(n);


      let conflictMean = 0;

      for (let i = 0;
 i < n;
 i++) conflictMean += this._conflict[i];

      conflictMean /= Math.max(1, n);


      const adaptivePower = this._adaptiveSharpenPower(
        entropy,
        variance,
        conflictMean,
        passIndex
      );


      const step = this._adaptiveStep(
        entropy,
        variance,
        conflictMean,
        uncertaintyMean
      );


      let operations = 0;

      let constructiveTotal = 0;

      let destructiveTotal = 0;

      let oscillationEvents = 0;


      for (let i = 0;
 i < n;
 i++) {

        let positive = 0;

        let negative = 0;

        let attractionMass = 0;

        let repulsionMass = 0;

        const row = i * stride;


        for (let j = 0;
 j < n;
 j++) {

          if (i === j) continue;


          const attraction = this._attraction[row + j];

          const repulsion = this._repulsion[row + j];


          if (attraction > 0) {

            positive += attraction * this._current[j] * this._certainty[j];

            attractionMass += attraction;

          
}


          if (repulsion > 0) {

            negative += repulsion * this._current[j] * (0.5 + 0.5 * this._certainty[j]);

            repulsionMass += repulsion;

          
}


          operations += 6;

        
}


        const norm = Math.max(EPSILON, n - 1);

        const pairSupport = this.clamp01(attractionMass / norm);

        const pairConflict = this.clamp01(repulsionMass / norm);


        const support = this.clamp01(
          pairSupport * 0.72 +
          this._higherSupport[i] * 0.28
        );


        const conflict = this.clamp01(
          pairConflict * 0.74 +
          this._higherTension[i] * 0.26
        );


        this._support[i] = support;

        this._conflict[i] = conflict;

        this._reinforcement[i] = positive;

        this._inhibition[i] = negative;


        let target = this._current[i];


        if (positive > 0) {

          const resonance = this.clamp01(positive / norm);

          const before = target;

          target = this.reinforce(
            target,
            resonance,
            this.options.reinforcementStrength,
            support
          );

          constructiveTotal += Math.max(0, target - before);

        
}


        if (negative > 0) {

          const inhibition = this.clamp01(negative / norm);

          const before = target;

          target = this.cancel(
            target,
            inhibition,
            this.options.cancellationStrength,
            0.55 + conflict * 0.45
          );

          destructiveTotal += Math.max(0, before - target);

        
}


        if (this._higherSupport[i] > 0) {

          target = this.reinforce(
            target,
            this._higherSupport[i],
            this.options.higherOrderWeight,
            this._higherSupport[i]
          );

        
}


        if (this._higherTension[i] > 0) {

          target = this.cancel(
            target,
            this._higherTension[i],
            this.options.contradictionWeight,
            0.65 + this._higherTension[i] * 0.35
          );

        
}


        const priorWeightTotal =
          this.options.knowledgeWeight +
          this.options.focusWeight +
          this.options.transitionWeight +
          this.options.reliabilityWeight +
          this.options.contextWeight +
          this.options.semanticWeight +
          this.options.noveltyWeight +
        this.options.amplitudeWeight;


        const prior = priorWeightTotal > EPSILON
          ? this.clamp01(
              (
                this._knowledge[i] * this.options.knowledgeWeight +
                this._focus[i] * this.options.focusWeight +
                this._transition[i] * this.options.transitionWeight +
                this._reliability[i] * this.options.reliabilityWeight +
                this._context[i] * this.options.contextWeight +
                this._semantic[i] * this.options.semanticWeight +
                this._novelty[i] * this.options.noveltyWeight +
                this._amplitudeProbability[i] * this.options.amplitudeWeight
              ) /
              priorWeightTotal
            )
          : 0;


        const priorPull = Math.min(
          0.20,
          0.035 + prior * 0.12 + this._certainty[i] * 0.035
        );


        target = this.blend(
          target,
          this.softOr(target, prior),
          priorPull
        );


        const acceleration = target - this._current[i];

        let velocity =
          this._velocity[i] * this.options.damping +
          acceleration * (1 - this.options.damping);


        velocity *= 1 + this.options.momentum * this._certainty[i] * 0.35;


        const delta = velocity * step;


        if (
          this._previousDelta[i] * delta < -this.options.convergenceEpsilon * 0.25 &&
          Math.abs(delta) > this.options.convergenceEpsilon
        ) {

          oscillationEvents++;

          velocity *= 0.58;

        
}


        let state = this._current[i] + velocity * step;


        // Certainty determines how far resonance can move the candidate away
        // from its original decoder confidence. Raw confidence stays dominant.
        state = this.blend(
          this._raw[i],
          state,
          0.46 + this._certainty[i] * 0.42
        );


        state = this.sharpen(state, adaptivePower);


        const lowerBound = Math.max(0, this._raw[i] - this.options.stateDropCap);

        const upperBound = Math.min(1, this._raw[i] + this.options.stateLiftCap);


        this._next[i] = Math.max(
          lowerBound,
          Math.min(upperBound, this.clamp01(state))
        );


        const certaintyEvidence = this.clamp01(
          support * this.options.supportCertaintyGain +
          this._reliability[i] * 0.12 +
          this._semantic[i] * 0.10 +
          this._context[i] * 0.07 +
          this._centrality[i] * 0.06 +
          this._certaintyInput[i] * 0.10
        );


        const uncertaintyPressure = this.clamp01(
          conflict * this.options.conflictUncertaintyGain +
          this._higherTension[i] * 0.13 +
          entropy * 0.055
        );


        let certainty = this._certainty[i];

        certainty += certaintyEvidence * (1 - certainty) * 0.42;

        certainty -= uncertaintyPressure * certainty * 0.44;

        certainty = this.blend(
          certainty,
          this._certaintyInput[i],
          this.options.uncertaintyDecay * 0.24
        );


        this._nextCertainty[i] = this.clamp01(certainty);

        this._nextVelocity[i] = velocity;

        this._previousDelta[i] = delta;


        operations += 34;

      
}


      let maxDelta = 0;

      let maxCertaintyDelta = 0;


      for (let i = 0;
 i < n;
 i++) {

        const delta = Math.abs(this._next[i] - this._current[i]);

        const certaintyDelta = Math.abs(this._nextCertainty[i] - this._certainty[i]);


        if (delta > maxDelta) maxDelta = delta;

        if (certaintyDelta > maxCertaintyDelta) maxCertaintyDelta = certaintyDelta;


        this._current[i] = this._next[i];

        this._certainty[i] = this._nextCertainty[i];

        this._uncertainty[i] = 1 - this._certainty[i];

        this._velocity[i] = this._nextVelocity[i];

      
}


      return {

        entropy,
        variance,
        uncertaintyMean,
        conflictMean,
        adaptivePower,
        step,
        operations,
        constructiveTotal,
        destructiveTotal,
        oscillationEvents,
        maxDelta,
        maxCertaintyDelta
      
}
;

    
}


    _collapseStability(n, candidateIndex) {

      if (candidateIndex < 0 || n <= 1) return n === 1 ? 1 : 0;


      const winner = this._current[candidateIndex];

      let nearest = 0;


      for (let i = 0;
 i < n;
 i++) {

        if (i === candidateIndex) continue;

        nearest = Math.max(nearest, this._current[i]);

      
}


      const margin = Math.max(0, winner - nearest);

      const marginScore = this.clamp01(
        margin / Math.max(EPSILON, this.options.perturbationRadius * 3)
      );


      return this.clamp01(
        marginScore * 0.38 +
        this._support[candidateIndex] * 0.24 +
        this._certainty[candidateIndex] * 0.16 +
        this._centrality[candidateIndex] * 0.12 +
        (1 - this._conflict[candidateIndex]) * 0.10
      );

    
}


    _collapseRobustness(n, candidateIndex) {

      if (candidateIndex < 0) return 0;

      if (n <= 1) return 1;


      const current = this._current[candidateIndex];

      let nearest = 0;


      for (let i = 0;
 i < n;
 i++) {

        if (i === candidateIndex) continue;

        nearest = Math.max(nearest, this._current[i]);

      
}


      const margin = Math.max(0, current - nearest);

      const perturbationScore = this.clamp01(
        margin / Math.max(EPSILON, this.options.perturbationRadius * 2.5)
      );


      return this.clamp01(
        perturbationScore * 0.48 +
        this._certainty[candidateIndex] * 0.17 +
        this._support[candidateIndex] * 0.15 +
        this._centrality[candidateIndex] * 0.10 +
        (1 - this._conflict[candidateIndex]) * 0.10
      );

    
}



    _defaultStateQueryVector(options = {}) {
      const supplied = options.queryVector;
      if (supplied && supplied.length) {
        for (let i = 0; i < this._stateQueryVector.length; i++) this._stateQueryVector[i] = Number(supplied[i]) || 0;
        return this._stateQueryVector;
      }
      // Columns: raw, knowledge, focus, transition, reliability, context,
      // semantic, certainty, novelty, support, centrality, non-conflict.
      const defaults = [0.34, 0.08, 0.08, 0.05, 0.08, 0.07, 0.12, 0.08, 0.025, 0.08, 0.05, 0.07];
      for (let i = 0; i < defaults.length; i++) this._stateQueryVector[i] = defaults[i];
      return this._stateQueryVector;
    }

    _runAmplitudeAmplification(n, options = {}) {
      if (!this.options.amplitudeAmplificationEnabled || !this.matrix || n < 2) {
        const uniform = n ? 1 / n : 0;
        for (let i = 0; i < n; i++) {
          this._amplitudeProbability[i] = uniform;
          this._queryProjection[i] = 0.5;
          this._amplitudeMarked[i] = 0;
        }
        return { enabled: false, rounds: 0, markedCount: 0, bestIndex: -1, bestProbability: uniform, processorMs: 0 };
      }

      const columns = this.options.vectorFeatureCount;
      this.matrix.ensure(n, columns);
      for (let i = 0; i < n; i++) {
        this.matrix.setRow(i, i, [
          this._raw[i],
          this._knowledge[i],
          this._focus[i],
          this._transition[i],
          this._reliability[i],
          this._context[i],
          this._semantic[i],
          this._certaintyInput[i],
          this._novelty[i],
          this.clamp01(this._support[i]),
          this._centrality[i],
          1 - this.clamp01(this._conflict[i])
        ]);
      }

      let bestRaw = 0;
      for (let i = 0; i < n; i++) if (this._raw[i] > bestRaw) bestRaw = this._raw[i];
      const eligibleMask = new Uint8Array(n);
      for (let i = 0; i < n; i++) eligibleMask[i] = Math.abs(bestRaw - this._raw[i]) <= this.options.confidenceWindow + EPSILON ? 1 : 0;

      const result = this.matrix.amplitudeAmplify({
        rows: n,
        queryVector: this._defaultStateQueryVector(options),
        prior: this._raw,
        couplingMatrix: this._compatibility,
        coherenceMatrix: this._coherence,
        couplingStride: this.options.maxStates,
        eligibleMask,
        coupling: this.options.amplitudeCoupling,
        markThreshold: this.options.amplitudeMarkThreshold,
        roundsMin: this.options.amplitudeRoundsMin,
        roundsMax: this.options.amplitudeRoundsMax
      });

      for (let i = 0; i < n; i++) {
        this._amplitudeProbability[i] = Number(result.probabilities?.[i]) || 0;
        this._queryProjection[i] = Number(result.projections?.[i]) || 0;
        this._amplitudeMarked[i] = result.marked?.[i] ? 1 : 0;
      }
      return {
        enabled: true,
        rounds: result.rounds || 0,
        markedCount: result.markedCount || 0,
        oracleThreshold: Number(result.oracleThreshold || 0),
        bestIndex: Number.isFinite(result.bestIndex) ? result.bestIndex : -1,
        bestProbability: Number(result.bestProbability || 0),
        processorMs: Number(result.processorMs || 0),
        featureRows: n,
        featureColumns: columns
      };
    }

    collapseCandidates(candidates, options = {
}
) {

      const started = performance.now();


      const list = Array.isArray(candidates)
        ? candidates.slice(0, this.options.maxStates)
        : [];


      const n = list.length;


      if (!n) {

        return {

          selected: null,
          state: 0,
          support: 0,
          score: 0,
          diagnostics: {

            mode: 'adaptive-multichannel-matrix-amplitude-state-lattice',
            representation: 'Float32Array',
            processorMs: 0,
            operations: 0,
            stateCount: 0,
            range: [0, 1]
          
}

        
}
;

      
}


      const getState = typeof options.getState === 'function'
        ? options.getState
        : candidate => candidate?.state ?? 0;


      const compatible = typeof options.compatible === 'function'
        ? options.compatible
        : () => 1;


      const knowledgeScore = typeof options.knowledgeScore === 'function'
        ? options.knowledgeScore
        : () => 0;


      const focusScore = typeof options.focusScore === 'function'
        ? options.focusScore
        : () => 0.5;


      const transitionScore = typeof options.transitionScore === 'function'
        ? options.transitionScore
        : () => 0.5;


      const reliabilityScore = typeof options.reliabilityScore === 'function'
        ? options.reliabilityScore
        : () => 0.5;


      const contextScore = typeof options.contextScore === 'function'
        ? options.contextScore
        : () => 0.5;


      const semanticScore = typeof options.semanticScore === 'function'
        ? options.semanticScore
        : () => 0.5;


      const noveltyScore = typeof options.noveltyScore === 'function'
        ? options.noveltyScore
        : () => 0.5;


      const certaintyScore = typeof options.certaintyScore === 'function'
        ? options.certaintyScore
        : candidate => getState(candidate);


      const window = Math.max(
        EPSILON,
        Number(options.confidenceWindow) || this.options.confidenceWindow
      );


      const minAgreement = Math.max(
        1,
        Math.min(n, options.minAgreement | 0 || 2)
      );


      const mixVariant = Math.max(0, options.mixVariant | 0 || 0);


      let operations = 0;


      for (let i = 0;
 i < n;
 i++) {

        const raw = this.clamp01(getState(list[i]));


        this._raw[i] = raw;

        this._current[i] = raw;

        this._next[i] = raw;

        this._bestState[i] = raw;

        this._velocity[i] = 0;

        this._nextVelocity[i] = 0;

        this._previousDelta[i] = 0;

        this._reinforcement[i] = 0;

        this._inhibition[i] = 0;

        this._support[i] = 0;

        this._conflict[i] = 0;

        this._higherSupport[i] = 0;

        this._higherTension[i] = 0;


        this._knowledge[i] = this.clamp01(knowledgeScore(list[i]));

        this._focus[i] = this.clamp01(focusScore(list[i]));

        this._transition[i] = this.clamp01(transitionScore(list[i]));

        this._reliability[i] = this.clamp01(reliabilityScore(list[i]));

        this._context[i] = this.clamp01(contextScore(list[i]));

        this._semantic[i] = this.clamp01(semanticScore(list[i]));

        this._novelty[i] = this.clamp01(noveltyScore(list[i]));

        this._certaintyInput[i] = this.clamp01(certaintyScore(list[i]));


        const initialCertainty = this.clamp01(
          raw * 0.40 +
          this._reliability[i] * 0.24 +
          this._certaintyInput[i] * 0.20 +
          this._semantic[i] * 0.10 +
          this._context[i] * 0.06
        );


        this._certainty[i] = initialCertainty;

        this._nextCertainty[i] = initialCertainty;

        this._bestCertainty[i] = initialCertainty;

        this._uncertainty[i] = 1 - initialCertainty;

        this._centrality[i] = 0;

        this._stability[i] = 0;

        this._robustness[i] = 0;

        this._score[i] = 0;


        operations += 16;

      
}


      const pairStats = this._buildPairGraph(list, n, compatible, window);

      operations += pairStats.operations;


      operations += this._computeCentrality(n);


      const higherStats = this._buildHigherOrderGraph(n);

      operations += higherStats.operations;

      const amplitudeStats = this._runAmplitudeAmplification(n, options);
      const uniformAmplitudeProbability = 1 / Math.max(1, n);
      if (amplitudeStats.enabled) {
        for (let i = 0; i < n; i++) {
          const shift = (this._amplitudeProbability[i] - uniformAmplitudeProbability) * this.options.amplitudeSeedGain;
          const lowerBound = Math.max(0, this._raw[i] - this.options.stateDropCap);
          const upperBound = Math.min(1, this._raw[i] + this.options.stateLiftCap);
          this._current[i] = Math.max(lowerBound, Math.min(upperBound, this._current[i] + shift));
          this._next[i] = this._current[i];
          this._bestState[i] = this._current[i];
        }
      }
      operations += amplitudeStats.enabled ? n * this.options.vectorFeatureCount : 0;


      const initialEntropy = this._entropy(n);

      const initialUncertainty = this._meanUncertainty(n);

      const initialEnergy = this._freeEnergy(n);


      let bestEnergy = initialEnergy;

      let previousEnergy = initialEnergy;


      for (let i = 0;
 i < n;
 i++) {

        this._bestState[i] = this._current[i];

        this._bestCertainty[i] = this._certainty[i];

      
}


      const passCount = Math.min(
        this.options.maxPropagationPasses,
        this.options.propagationPasses +
          (initialEntropy > 0.82 ? this.options.entropyPassBoost : 0) +
          (initialUncertainty > 0.38 ? 1 : 0)
      );


      const passDiagnostics = [];

      let constructiveTotal = 0;

      let destructiveTotal = 0;

      let oscillationEvents = 0;

      let energyIncreaseEvents = 0;

      let executedPasses = 0;

      let converged = false;


      for (let pass = 0;
 pass < passCount;
 pass++) {

        const stats = this._propagate(n, pass);

        executedPasses++;

        operations += stats.operations;

        constructiveTotal += stats.constructiveTotal;

        destructiveTotal += stats.destructiveTotal;

        oscillationEvents += stats.oscillationEvents;


        const energy = this._freeEnergy(n);

        const energyDelta = energy - previousEnergy;


        if (energy < bestEnergy - this.options.energyEpsilon) {

          bestEnergy = energy;

          for (let i = 0;
 i < n;
 i++) {

            this._bestState[i] = this._current[i];

            this._bestCertainty[i] = this._certainty[i];

          
}

        
}


        if (energyDelta > this.options.energyEpsilon * 4) {

          energyIncreaseEvents++;

          for (let i = 0;
 i < n;
 i++) {

            this._velocity[i] *= 0.52;

          
}

        
}


        passDiagnostics.push({

          pass,
          entropy: Number(stats.entropy.toFixed(6)),
          variance: Number(stats.variance.toFixed(8)),
          uncertainty: Number(stats.uncertaintyMean.toFixed(6)),
          conflict: Number(stats.conflictMean.toFixed(6)),
          sharpenPower: Number(stats.adaptivePower.toFixed(5)),
          adaptiveStep: Number(stats.step.toFixed(5)),
          maxDelta: Number(stats.maxDelta.toFixed(7)),
          maxCertaintyDelta: Number(stats.maxCertaintyDelta.toFixed(7)),
          energy: Number(energy.toFixed(8)),
          energyDelta: Number(energyDelta.toFixed(8)),
          oscillations: stats.oscillationEvents
        
}
);


        const smallStateDelta = stats.maxDelta <= this.options.convergenceEpsilon;

        const smallCertaintyDelta = stats.maxCertaintyDelta <= this.options.convergenceEpsilon * 1.5;

        const smallEnergyDelta = Math.abs(energyDelta) <= this.options.energyEpsilon;


        previousEnergy = energy;


        if (pass >= 1 && smallStateDelta && smallCertaintyDelta && smallEnergyDelta) {

          converged = true;

          break;

        
}

      
}


      const energyBeforeRollback = this._freeEnergy(n);

      let rolledBack = false;


      if (
        this.options.rollbackOnEnergyIncrease &&
        bestEnergy + this.options.energyEpsilon < energyBeforeRollback
      ) {

        rolledBack = true;

        for (let i = 0;
 i < n;
 i++) {

          this._current[i] = this._bestState[i];

          this._certainty[i] = this._bestCertainty[i];

          this._uncertainty[i] = 1 - this._certainty[i];

        
}

      
}


      const finalEnergy = this._freeEnergy(n);

      const finalEntropy = this._entropy(n);

      const finalUncertainty = this._meanUncertainty(n);


      let bestIndex = -1;

      let bestScore = -Infinity;

      let secondScore = -Infinity;

      const rows = [];

      const stride = this.options.maxStates;


      const positiveWeightTotal =
        0.34 +
        0.18 +
        this.options.consensusWeight +
        this.options.certaintyWeight +
        this.options.centralityWeight +
        this.options.robustnessWeight +
        this.options.stabilityWeight +
        this.options.reliabilityWeight +
        this.options.contextWeight +
        this.options.semanticWeight +
        this.options.knowledgeWeight +
        this.options.focusWeight +
        this.options.transitionWeight +
        this.options.higherOrderWeight +
        this.options.noveltyWeight +
        this.options.amplitudeWeight;


      for (let i = 0;
 i < n;
 i++) {

        let agreementCount = 1;

        let weightedAgreement = 0;


        for (let j = 0;
 j < n;
 j++) {

          if (i === j) continue;

          const attraction = this._attraction[i * stride + j];

          const compat = this._compatibility[i * stride + j];

          const coh = this._coherence[i * stride + j];


          if (attraction > 0.035 && compat >= 0.30) agreementCount++;

          weightedAgreement += compat * coh;

        
}


        const support = this.clamp01(this._support[i]);

        const conflict = this.clamp01(this._conflict[i]);

        const stability = this._collapseStability(n, i);

        const robustness = this._collapseRobustness(n, i);


        this._stability[i] = stability;

        this._robustness[i] = robustness;


        const numerator =
          this._raw[i] * 0.34 +
          this._current[i] * 0.18 +
          support * this.options.consensusWeight +
          this._certainty[i] * this.options.certaintyWeight +
          this._centrality[i] * this.options.centralityWeight +
          robustness * this.options.robustnessWeight +
          stability * this.options.stabilityWeight +
          this._reliability[i] * this.options.reliabilityWeight +
          this._context[i] * this.options.contextWeight +
          this._semantic[i] * this.options.semanticWeight +
          this._knowledge[i] * this.options.knowledgeWeight +
          this._focus[i] * this.options.focusWeight +
          this._transition[i] * this.options.transitionWeight +
          this._higherSupport[i] * this.options.higherOrderWeight +
          this._novelty[i] * this.options.noveltyWeight +
          this._amplitudeProbability[i] * this.options.amplitudeWeight -
          conflict * this.options.contradictionWeight -
          this._higherTension[i] * this.options.contradictionWeight * 0.72;


        const score = this.clamp01(
          numerator / Math.max(EPSILON, positiveWeightTotal)
        );


        this._score[i] = score;


        const eligible = agreementCount >= minAgreement;


        rows.push({

          index: i,
          rawState: Number(this._raw[i].toFixed(6)),
          evolvedState: Number(this._current[i].toFixed(6)),
          certainty: Number(this._certainty[i].toFixed(6)),
          uncertainty: Number((1 - this._certainty[i]).toFixed(6)),
          support: Number(support.toFixed(6)),
          conflict: Number(conflict.toFixed(6)),
          higherOrderSupport: Number(this._higherSupport[i].toFixed(6)),
          higherOrderTension: Number(this._higherTension[i].toFixed(6)),
          centrality: Number(this._centrality[i].toFixed(6)),
          agreementCount,
          weightedAgreement: Number(weightedAgreement.toFixed(6)),
          knowledge: Number(this._knowledge[i].toFixed(6)),
          focus: Number(this._focus[i].toFixed(6)),
          transition: Number(this._transition[i].toFixed(6)),
          reliability: Number(this._reliability[i].toFixed(6)),
          context: Number(this._context[i].toFixed(6)),
          semantic: Number(this._semantic[i].toFixed(6)),
          novelty: Number(this._novelty[i].toFixed(6)),
          stability: Number(stability.toFixed(6)),
          robustness: Number(robustness.toFixed(6)),
          amplitudeProbability: Number(this._amplitudeProbability[i].toFixed(6)),
          queryProjection: Number(this._queryProjection[i].toFixed(6)),
          oracleMarked: Boolean(this._amplitudeMarked[i]),
          score: Number(score.toFixed(6)),
          eligible
        
}
);


        operations += 24;


        if (!eligible) continue;


        if (score > bestScore + EPSILON) {

          secondScore = bestScore;

          bestScore = score;

          bestIndex = i;

        
}
 else if (score > secondScore) {

          secondScore = score;

        
}
 else if (Math.abs(score - bestScore) <= EPSILON) {

          const currentTie = bestIndex < 0 ? Infinity : (bestIndex + mixVariant) % n;

          const newTie = (i + mixVariant) % n;


          if (newTie < currentTie) {

            secondScore = bestScore;

            bestScore = score;

            bestIndex = i;

          
}

        
}

      
}


      const selected = bestIndex >= 0 ? list[bestIndex] : null;

      const selectedState = bestIndex >= 0 ? this.clamp01(this._current[bestIndex]) : 0;

      const scoreMargin =
        bestIndex >= 0 && Number.isFinite(secondScore)
          ? Math.max(0, bestScore - secondScore)
          : 0;


      const elapsed = performance.now() - started;


      this.totalCollapses++;

      this.totalOperations += operations;

      this.totalProcessorMs += elapsed;

      this.totalPropagationPasses += executedPasses;

      this.totalTripletsEvaluated += higherStats.triplets;

      this.totalOscillationEvents += oscillationEvents;

      if (rolledBack) this.totalRollbacks++;


      const graphPairMax = Math.max(1, (n * (n - 1)) >> 1);


      const diagnostics = {

        mode: 'adaptive-multichannel-matrix-amplitude-state-lattice',
        representation: 'Float32Array',
        range: [0, 1],
        channels: [
          'activation',
          'query-projection',
          'amplitude-probability',
          'certainty',
          'uncertainty',
          'momentum',
          'consensus',
          'contradiction',
          'centrality',
          'reliability',
          'context',
          'semantic',
          'prior',
          'robustness'
        ],
        stateCount: n,
        matrixVectorState: amplitudeStats,
        stateVector: Array.from(this._current.slice(0, n), value => Number(value.toFixed(6))),
        amplitudeVector: Array.from(this._amplitudeProbability.slice(0, n), value => Number(value.toFixed(6))),
        queryProjectionVector: Array.from(this._queryProjection.slice(0, n), value => Number(value.toFixed(6))),
        featureMatrixShape: [n, this.options.vectorFeatureCount],
        maxStates: this.options.maxStates,
        confidenceWindow: window,
        minAgreement,
        propagationPasses: executedPasses,
        requestedPropagationPasses: passCount,
        converged,
        rolledBack,
        coherentPairs: pairStats.coherentPairs,
        compatiblePairs: pairStats.compatiblePairs,
        incompatiblePairs: pairStats.incompatiblePairs,
        ambiguousPairs: pairStats.ambiguousPairs,
        graphDensity: Number((pairStats.coherentPairs / graphPairMax).toFixed(6)),
        attractionTotal: Number(pairStats.attractionTotal.toFixed(6)),
        repulsionTotal: Number(pairStats.repulsionTotal.toFixed(6)),
        higherOrderTriplets: higherStats.triplets,
        activeHigherOrderTriplets: higherStats.activeTriplets,
        higherOrderSynergy: Number(higherStats.synergyTotal.toFixed(6)),
        higherOrderTension: Number(higherStats.tensionTotal.toFixed(6)),
        centralityIterations: this.options.centralityIterations,
        initialEntropy: Number(initialEntropy.toFixed(6)),
        finalEntropy: Number(finalEntropy.toFixed(6)),
        initialUncertainty: Number(initialUncertainty.toFixed(6)),
        finalUncertainty: Number(finalUncertainty.toFixed(6)),
        initialEnergy: Number(initialEnergy.toFixed(8)),
        bestEnergy: Number(bestEnergy.toFixed(8)),
        finalEnergy: Number(finalEnergy.toFixed(8)),
        energyIncreaseEvents,
        oscillationEvents,
        constructiveTotal: Number(constructiveTotal.toFixed(6)),
        destructiveTotal: Number(destructiveTotal.toFixed(6)),
        selectedIndex: bestIndex,
        selectedState: Number(selectedState.toFixed(6)),
        selectedCertainty:
          bestIndex >= 0
            ? Number(this._certainty[bestIndex].toFixed(6))
            : 0,
        selectedScore:
          bestIndex >= 0
            ? Number(bestScore.toFixed(6))
            : 0,
        scoreMargin: Number(scoreMargin.toFixed(6)),
        operations,
        processorMs: elapsed,
        passDiagnostics,
        rows
      
}
;


      this.lastDiagnostics = diagnostics;


      return {

        selected,
        state: selectedState,
        certainty:
          bestIndex >= 0
            ? this._certainty[bestIndex]
            : 0,
        uncertainty:
          bestIndex >= 0
            ? 1 - this._certainty[bestIndex]
            : 1,
        support:
          bestIndex >= 0
            ? rows[bestIndex].agreementCount
            : 0,
        score:
          bestIndex >= 0
            ? bestScore
            : 0,
        diagnostics
      
}
;

    
}


    propositionVector(structure = {}) {
      const clauses = structure?.clauses || [];
      const propositions = structure?.propositions || [];
      const refs = structure?.references || [];
      const features = structure?.features || {};
      return [
        Math.min(1, clauses.length / 6), Math.min(1, propositions.length / 10),
        Math.min(1, refs.length / 6), Number(structure?.structuralComplexity || 0),
        features.conditional ? 1 : 0, features.temporal ? 1 : 0,
        features.comparison ? 1 : 0, features.negation ? 1 : 0,
        features.coordination ? 1 : 0, features.embeddedQuestion ? 1 : 0,
        Math.min(1, (structure?.requestedRoles || []).length / 4),
        Math.min(1, (structure?.unresolved || []).length / 4)
      ];
    }

    status() {

      return {

        ready: true,
        mode: 'adaptive-multichannel-higher-order-state-lattice',
        representation: 'Float32Array',
        range: [0, 1],
        channelsPerCandidate: 12,
        confidenceWindow: this.options.confidenceWindow,
        maxStates: this.options.maxStates,
        reinforcementStrength: this.options.reinforcementStrength,
        cancellationStrength: this.options.cancellationStrength,
        sharpenPower: this.options.sharpenPower,
        propagationPasses: this.options.propagationPasses,
        maxPropagationPasses: this.options.maxPropagationPasses,
        entropyPassBoost: this.options.entropyPassBoost,
        consensusWeight: this.options.consensusWeight,
        transitionWeight: this.options.transitionWeight,
        knowledgeWeight: this.options.knowledgeWeight,
        focusWeight: this.options.focusWeight,
        stabilityWeight: this.options.stabilityWeight,
        reliabilityWeight: this.options.reliabilityWeight,
        contextWeight: this.options.contextWeight,
        semanticWeight: this.options.semanticWeight,
        certaintyWeight: this.options.certaintyWeight,
        centralityWeight: this.options.centralityWeight,
        robustnessWeight: this.options.robustnessWeight,
        higherOrderWeight: this.options.higherOrderWeight,
        contradictionWeight: this.options.contradictionWeight,
        noveltyWeight: this.options.noveltyWeight,
        momentum: this.options.momentum,
        damping: this.options.damping,
        adaptiveStepMin: this.options.adaptiveStepMin,
        adaptiveStepMax: this.options.adaptiveStepMax,
        stateLiftCap: this.options.stateLiftCap,
        stateDropCap: this.options.stateDropCap,
        convergenceEpsilon: this.options.convergenceEpsilon,
        energyEpsilon: this.options.energyEpsilon,
        entropyTemperature: this.options.entropyTemperature,
        amplitudeAmplificationEnabled: this.options.amplitudeAmplificationEnabled,
        amplitudeWeight: this.options.amplitudeWeight,
        amplitudeCoupling: this.options.amplitudeCoupling,
        amplitudeMarkThreshold: this.options.amplitudeMarkThreshold,
        amplitudeRounds: [this.options.amplitudeRoundsMin, this.options.amplitudeRoundsMax],
        vectorFeatureCount: this.options.vectorFeatureCount,
        matrixEngine: this.matrix?.status?.() || null,
        centralityIterations: this.options.centralityIterations,
        perturbationRadius: this.options.perturbationRadius,
        rollbackOnEnergyIncrease: this.options.rollbackOnEnergyIncrease,
        totalCollapses: this.totalCollapses,
        totalOperations: this.totalOperations,
        totalPropagationPasses: this.totalPropagationPasses,
        totalTripletsEvaluated: this.totalTripletsEvaluated,
        totalOscillationEvents: this.totalOscillationEvents,
        totalRollbacks: this.totalRollbacks,
        totalProcessorMs: this.totalProcessorMs,
        averageProcessorMs:
          this.totalCollapses
            ? this.totalProcessorMs / this.totalCollapses
            : 0,
        averagePropagationPasses:
          this.totalCollapses
            ? this.totalPropagationPasses / this.totalCollapses
            : 0,
        lastDiagnostics: this.lastDiagnostics
      
}
;

    
}

  
}


  globalThis.VilotStateProcessor = VilotStateProcessor;

}
)();

