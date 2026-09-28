/**
 * VilotNI 2.5 - LearningEvaluator.js
 *
 * Measures whether learning actually improves prediction, consistency,
 * calibration, output quality, and trusted relation retention. It can signal a
 * regression so LearningController can roll back untrusted mutations from the
 * affected cycle.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));

  const normalize = value =>
    String(value || '')
      .toLowerCase()
      .replace(/[’]/g, "'")
      .replace(/[^a-z0-9'+.\- ]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

  class VilotLearningEvaluator {
    constructor(processor, options = {}) {
      this.processor = processor;
      this.options = {
        alpha: clamp(options.learningEvaluatorEWMAAlpha ?? 0.10, 0.02, 0.5),
        promptHistoryMax: Math.max(64, options.learningEvaluatorPromptHistoryMax | 0 || 512),
        regressionCaseMax: Math.max(64, options.learningRegressionCaseMax | 0 || 512),
        regressionTolerance: clamp(options.learningRegressionTolerance ?? 0.035, 0.005, 0.25),
        regressionMinCases: Math.max(4, options.learningRegressionMinCases | 0 || 8)
      };
      this.outcomes = 0;
      this.predictions = 0;
      this.replays = 0;
      this.meanPredictionError = 0;
      this.consistency = 0.5;
      this.calibrationError = 0.5;
      this.outputQuality = 0.5;
      this.learningDelta = 0;
      this.promptHistory = new Map();
      this.regressionCases = new Map();
      this.regressionRuns = 0;
      this.regressions = 0;
      this.rollbacks = 0;
      this.lastRegression = null;
      this.last = null;
    }

    _tokens(text) {
      const rows = this.processor?.tokenize?.(String(text || ''), 128) || [];
      if (rows.length) {
        return Array.from(new Set(rows.filter(row => row?.pos !== 'Punct').map(row => normalize(row?.lower || row?.surface)).filter(Boolean)));
      }
      return Array.from(new Set(normalize(text).split(/\s+/).filter(Boolean)));
    }

    _promptKey(prompt, analysisSummary = null) {
      const semanticWords = Array.from(new Set([...(analysisSummary?.contentWords || []), ...(analysisSummary?.entityNames || [])])).map(normalize).filter(Boolean);
      if (semanticWords.length) {
        const intent = normalize(analysisSummary?.intent || 'content');
        const slot = normalize(analysisSummary?.requestedSlot || 'content');
        return [`intent:${intent}`, `slot:${slot}`, ...Array.from(new Set(semanticWords)).sort().slice(0, 24)].join('|');
      }
      return this._tokens(prompt).sort().slice(0, 24).join('|');
    }

    _similarity(a, b) {
      const A = new Set(this._tokens(a));
      const B = new Set(this._tokens(b));
      if (!A.size || !B.size) return 0;
      let intersection = 0;
      for (const token of A) if (B.has(token)) intersection++;
      const union = A.size + B.size - intersection;
      return union ? intersection / union : 0;
    }

    _ewma(oldValue, next, first = false) {
      if (first) return next;
      const a = this.options.alpha;
      return oldValue * (1 - a) + next * a;
    }

    observePrediction(errorResult) {
      if (!errorResult) return null;
      const value = clamp(Math.abs(Number(errorResult.signedError) || 0));
      this.predictions++;
      this.meanPredictionError = this._ewma(this.meanPredictionError, value, this.predictions === 1);
      return this.meanPredictionError;
    }

    observeOutcome(event = {}) {
      const prompt = String(event.promptText || '');
      const response = String(event.responseText || event.text || '');
      const confidence = clamp((Number(event.confidence) || 0) / 100);
      const quality = clamp(event.quality ?? confidence);
      const calibration = Math.abs(confidence - quality);
      let consistency = 0.5;
      const key = this._promptKey(prompt, event.analysisSummary || null);
      if (key && response) {
        const previous = this.promptHistory.get(key);
        consistency = previous?.response ? this._similarity(previous.response, response) : 0.75;
        this.promptHistory.set(key, { response, confidence, quality, at: Date.now() });
        if (this.promptHistory.size > this.options.promptHistoryMax) {
          const first = this.promptHistory.keys().next();
          if (!first.done) this.promptHistory.delete(first.value);
        }
      }
      this.outcomes++;
      this.consistency = this._ewma(this.consistency, consistency, this.outcomes === 1);
      this.calibrationError = this._ewma(this.calibrationError, calibration, this.outcomes === 1);
      this.outputQuality = this._ewma(this.outputQuality, quality, this.outcomes === 1);
      this.last = { at: Date.now(), confidence, quality, calibrationError: calibration, consistency, semanticKey: key || null };
      return this.last;
    }

    observeReplay(beforeError, afterError) {
      const before = clamp(beforeError ?? 0);
      const after = clamp(afterError ?? 0);
      const delta = before - after;
      this.replays++;
      this.learningDelta = this._ewma(this.learningDelta, delta, this.replays === 1);
      return delta;
    }

    registerExpectedRelation(from, relation, to, metadata = {}) {
      const a = normalize(from), r = normalize(relation), b = normalize(to);
      if (!a || !r || !b) return false;
      const key = `${a}|${r}|${b}`;
      const old = this.regressionCases.get(key) || { key, from: a, relation: r, to: b, weight: 0.5, observations: 0 };
      old.weight = Math.max(old.weight, clamp(metadata.weight ?? metadata.trust ?? 0.85));
      old.observations++;
      old.lastSeen = Date.now();
      this.regressionCases.set(key, old);
      if (this.regressionCases.size > this.options.regressionCaseMax) {
        const victims = Array.from(this.regressionCases.values()).sort((x, y) => (x.lastSeen || 0) - (y.lastSeen || 0));
        for (const victim of victims.slice(0, this.regressionCases.size - this.options.regressionCaseMax)) this.regressionCases.delete(victim.key);
      }
      return true;
    }

    benchmark(graph) {
      const cases = Array.from(this.regressionCases.values());
      if (cases.length < this.options.regressionMinCases) {
        return { ready: false, cases: cases.length, score: null, coverage: null, confidence: null, contested: null };
      }
      let weightSum = 0, scoreSum = 0, coverageSum = 0, confidenceSum = 0, contestedSum = 0;
      for (const item of cases) {
        const edge = graph?.getEdge?.(item.from, item.relation, item.to);
        const w = clamp(item.weight || 0.5, 0.1, 1);
        weightSum += w;
        if (edge) {
          const coverage = edge.status === 'consolidated' ? 1 : edge.status === 'provisional' ? 0.7 : edge.status === 'observed' ? 0.4 : 0.2;
          const contested = edge.status === 'contested' ? 1 : clamp(edge.contradictionPressure ?? 0);
          const local = clamp(coverage * 0.36 + clamp(edge.confidence) * 0.44 + (1 - contested) * 0.20);
          scoreSum += local * w;
          coverageSum += coverage * w;
          confidenceSum += clamp(edge.confidence) * w;
          contestedSum += contested * w;
        }
      }
      const denom = Math.max(1e-9, weightSum);
      return {
        ready: true,
        cases: cases.length,
        score: scoreSum / denom,
        coverage: coverageSum / denom,
        confidence: confidenceSum / denom,
        contested: contestedSum / denom
      };
    }

    compareRegression(before, after, metadata = {}) {
      this.regressionRuns++;
      if (!before?.ready || !after?.ready) {
        this.lastRegression = { at: Date.now(), tested: false, reason: 'insufficient-regression-cases', before, after };
        return this.lastRegression;
      }
      const delta = after.score - before.score;
      const regression = delta < -this.options.regressionTolerance;
      if (regression) this.regressions++;
      this.lastRegression = {
        at: Date.now(), tested: true, regression, delta, tolerance: this.options.regressionTolerance,
        before, after, mutationCount: Number(metadata.mutationCount) || 0,
        untrustedMutationCount: Number(metadata.untrustedMutationCount) || 0
      };
      return this.lastRegression;
    }

    noteRollback(count = 1) {
      this.rollbacks += Math.max(0, count | 0 || 0);
      if (this.lastRegression) this.lastRegression.rollbackCount = Math.max(0, count | 0 || 0);
    }

    reset() {
      this.outcomes = 0;
      this.predictions = 0;
      this.replays = 0;
      this.meanPredictionError = 0;
      this.consistency = 0.5;
      this.calibrationError = 0.5;
      this.outputQuality = 0.5;
      this.learningDelta = 0;
      this.promptHistory.clear();
      this.regressionCases.clear();
      this.regressionRuns = 0;
      this.regressions = 0;
      this.rollbacks = 0;
      this.lastRegression = null;
      this.last = null;
      return true;
    }

    exportState() {
      return {
        schemaVersion: 2,
        outcomes: this.outcomes,
        predictions: this.predictions,
        replays: this.replays,
        meanPredictionError: this.meanPredictionError,
        consistency: this.consistency,
        calibrationError: this.calibrationError,
        outputQuality: this.outputQuality,
        learningDelta: this.learningDelta,
        promptHistory: Array.from(this.promptHistory.entries()).slice(-this.options.promptHistoryMax),
        regressionCases: Array.from(this.regressionCases.entries()).slice(-this.options.regressionCaseMax),
        regressionRuns: this.regressionRuns,
        regressions: this.regressions,
        rollbacks: this.rollbacks,
        lastRegression: this.lastRegression,
        last: this.last
      };
    }

    importState(state) {
      if (!state || ![1, 2].includes(Number(state.schemaVersion))) return false;
      this.outcomes = Math.max(0, Number(state.outcomes) || 0);
      this.predictions = Math.max(0, Number(state.predictions) || 0);
      this.replays = Math.max(0, Number(state.replays) || 0);
      this.meanPredictionError = clamp(state.meanPredictionError ?? 0);
      this.consistency = clamp(state.consistency ?? 0.5);
      this.calibrationError = clamp(state.calibrationError ?? 0.5);
      this.outputQuality = clamp(state.outputQuality ?? 0.5);
      this.learningDelta = Math.max(-1, Math.min(1, Number(state.learningDelta) || 0));
      this.promptHistory = new Map(Array.isArray(state.promptHistory) ? state.promptHistory : []);
      this.regressionCases = new Map(Array.isArray(state.regressionCases) ? state.regressionCases : []);
      this.regressionRuns = Math.max(0, Number(state.regressionRuns) || 0);
      this.regressions = Math.max(0, Number(state.regressions) || 0);
      this.rollbacks = Math.max(0, Number(state.rollbacks) || 0);
      this.lastRegression = state.lastRegression || null;
      this.last = state.last || null;
      return true;
    }

    status() {
      return {
        ready: true,
        role: 'learning-quality-and-regression-evaluator',
        outcomes: this.outcomes,
        predictions: this.predictions,
        replays: this.replays,
        meanPredictionError: Number(this.meanPredictionError.toFixed(5)),
        consistency: Number(this.consistency.toFixed(5)),
        calibrationError: Number(this.calibrationError.toFixed(5)),
        outputQuality: Number(this.outputQuality.toFixed(5)),
        compositeUtility: Number((
          this.outputQuality * 0.36 +
          this.consistency * 0.24 +
          (1 - this.calibrationError) * 0.20 +
          (1 - this.meanPredictionError) * 0.20
        ).toFixed(5)),
        learningDelta: Number(this.learningDelta.toFixed(5)),
        regressionCases: this.regressionCases.size,
        regressionRuns: this.regressionRuns,
        regressions: this.regressions,
        rollbacks: this.rollbacks,
        lastRegression: this.lastRegression,
        last: this.last
      };
    }
  }

  globalThis.VilotLearningEvaluator = VilotLearningEvaluator;
})();
