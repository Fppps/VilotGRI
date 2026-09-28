/**
 * VilotNI 2.5 - PredictionError.js
 *
 * Computes expected-vs-observed relation error.
 * Learning gain is driven by surprise rather than raw repetition alone.
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

  class VilotPredictionError {
    constructor(graph, rsl, processor, options = {}) {
      this.graph = graph;
      this.rsl = rsl;
      this.processor = processor;

      this.options = {
        priorExpectation: clamp(options.learningPredictionPrior ?? 0.18),
        ewmaAlpha: clamp(options.learningPredictionEWMAAlpha ?? 0.12, 0.02, 0.6)
      };

      this.evaluations = 0;
      this.meanAbsError = 0;
      this.meanSignedError = 0;
      this.byRelation = new Map();
      this.last = null;
    }

    _terms(value) {
      const text = normalize(value);
      const fromProcessor =
        this.processor?.tokenize?.(text, 12) || [];

      const terms =
        fromProcessor.length
          ? fromProcessor
              .filter(row => row?.pos !== 'Punct')
              .map(row => normalize(row?.lower || row?.surface))
              .filter(Boolean)
          : text.split(/\s+/).filter(Boolean);

      return Array.from(new Set(terms)).slice(0, 6);
    }

    _lexicalExpectation(from, to) {
      const a = this._terms(from);
      const b = this._terms(to);

      let rslBest = 0;
      let processorBest = 0;

      for (const x of a) {
        for (const y of b) {
          if (x === y) continue;

          const rsl =
            Number(this.rsl?.correlationScore?.(x, y)) || 0;

          const proc =
            Number(
              this.processor?.associationScore?.(
                x,
                y,
                { bidirectional: true }
              )
            ) || 0;

          if (rsl > rslBest) rslBest = rsl;
          if (proc > processorBest) processorBest = proc;
        }
      }

      return {
        rsl: clamp(rslBest),
        processor: clamp(processorBest)
      };
    }

    evaluate(from, relation, to, metadata = {}) {
      const graphScore =
        Number(
          this.graph?.score?.(
            from,
            relation,
            to
          )
        ) || 0;

      const lexical =
        this._lexicalExpectation(
          from,
          to
        );

      const hasGraph =
        graphScore > 0;

      const expected =
        hasGraph
          ? clamp(
              graphScore * 0.58 +
              lexical.rsl * 0.27 +
              lexical.processor * 0.15
            )
          : clamp(
              this.options.priorExpectation * 0.55 +
              lexical.rsl * 0.30 +
              lexical.processor * 0.15
            );

      const observed =
        clamp(
          metadata.observed ??
          metadata.strength ??
          metadata.support ??
          1
        );

      const signedError =
        observed - expected;

      const absError =
        Math.abs(signedError);

      const trust =
        clamp(metadata.trust ?? 0.5);

      const novelty =
        clamp(
          metadata.novelty ??
          (hasGraph ? 0.25 : 1)
        );

      const gain =
        clamp(
          absError *
          (0.52 + trust * 0.33 + novelty * 0.15),
          0.01,
          1
        );

      this.evaluations++;

      const alpha =
        this.options.ewmaAlpha;

      this.meanAbsError =
        this.evaluations === 1
          ? absError
          : this.meanAbsError * (1 - alpha) + absError * alpha;

      this.meanSignedError =
        this.evaluations === 1
          ? signedError
          : this.meanSignedError * (1 - alpha) + signedError * alpha;

      const relationKey =
        normalize(relation) ||
        'unknown';

      const stat =
        this.byRelation.get(relationKey) || {
          count: 0,
          meanAbsError: 0,
          meanSignedError: 0
        };

      stat.count++;

      const a =
        Math.min(
          0.5,
          2 / (stat.count + 2)
        );

      stat.meanAbsError =
        stat.count === 1
          ? absError
          : stat.meanAbsError * (1 - a) + absError * a;

      stat.meanSignedError =
        stat.count === 1
          ? signedError
          : stat.meanSignedError * (1 - a) + signedError * a;

      this.byRelation.set(relationKey, stat);

      this.last = {
        at: Date.now(),
        from: normalize(from),
        relation: relationKey,
        to: normalize(to),
        expected,
        observed,
        signedError,
        absError,
        gain,
        trust,
        novelty
      };

      return this.last;
    }

    reset() {
      this.evaluations = 0;
      this.meanAbsError = 0;
      this.meanSignedError = 0;
      this.byRelation.clear();
      this.last = null;
      return true;
    }

    exportState() {
      return {
        schemaVersion: 1,
        evaluations: this.evaluations,
        meanAbsError: this.meanAbsError,
        meanSignedError: this.meanSignedError,
        byRelation: Array.from(this.byRelation.entries()).slice(-256),
        last: this.last
      };
    }

    importState(state) {
      if (!state || Number(state.schemaVersion) !== 1) return false;

      this.evaluations = Math.max(0, Number(state.evaluations) || 0);
      this.meanAbsError = clamp(state.meanAbsError ?? 0);
      this.meanSignedError =
        Math.max(-1, Math.min(1, Number(state.meanSignedError) || 0));

      this.byRelation = new Map(
        Array.isArray(state.byRelation)
          ? state.byRelation
          : []
      );

      this.last = state.last || null;
      return true;
    }

    status() {
      return {
        ready: true,
        role: 'expected-vs-observed-learning-signal',
        evaluations: this.evaluations,
        meanAbsError: Number(this.meanAbsError.toFixed(5)),
        meanSignedError: Number(this.meanSignedError.toFixed(5)),
        relationTypes: this.byRelation.size,
        last: this.last
      };
    }
  }

  globalThis.VilotPredictionError = VilotPredictionError;
})();
