/**
 * VilotNI 2.5 - Generalization.js
 *
 * Detects repeated relation structure across multiple concepts.
 * It creates hypothesis candidates only. It never writes generalized claims
 * directly into factual knowledge.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));

  const normalize = value =>
    String(value || '')
      .toLowerCase()
      .replace(/[’]/g, "'")
      .replace(/[^a-z0-9'+.:\- ]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 192);

  class VilotGeneralization {
    constructor(options = {}) {
      this.options = {
        minSubjects: Math.max(3, options.learningGeneralizationMinSubjects | 0 || 3),
        minMeanConfidence: clamp(options.learningGeneralizationMinConfidence ?? 0.72),
        maxPatterns: Math.max(256, options.learningGeneralizationMaxPatterns | 0 || 4096)
      };

      this.patterns = new Map();
      this.emitted = 0;
      this.last = null;
    }

    _key(domain, relation, to) {
      return `${normalize(domain)}|${normalize(relation)}|${normalize(to)}`;
    }

    observe(edge, domain) {
      if (!edge?.from || !edge?.relation || !edge?.to) return null;

      const d = normalize(domain || edge?.metadata?.domain);
      if (!d) return null;

      const key = this._key(d, edge.relation, edge.to);

      if (!this.patterns.has(key) && this.patterns.size >= this.options.maxPatterns) {
        const first = this.patterns.keys().next();
        if (!first.done) this.patterns.delete(first.value);
      }

      const pattern =
        this.patterns.get(key) || {
          key,
          domain: d,
          relation: normalize(edge.relation),
          to: normalize(edge.to),
          subjects: {},
          totalConfidence: 0,
          observations: 0,
          emittedConfidence: 0,
          lastSeen: Date.now()
        };

      const subject = normalize(edge.from);

      if (!pattern.subjects[subject]) {
        pattern.subjects[subject] = 0;
      }

      pattern.subjects[subject] =
        Math.max(
          pattern.subjects[subject],
          clamp(edge.confidence)
        );

      pattern.observations++;
      pattern.totalConfidence += clamp(edge.confidence);
      pattern.lastSeen = Date.now();

      this.patterns.set(key, pattern);

      const subjectCount =
        Object.keys(pattern.subjects).length;

      const meanConfidence =
        Object.values(pattern.subjects)
          .reduce((a, b) => a + Number(b || 0), 0) /
        Math.max(1, subjectCount);

      if (
        subjectCount < this.options.minSubjects ||
        meanConfidence < this.options.minMeanConfidence
      ) {
        return null;
      }

      const candidateConfidence =
        clamp(
          meanConfidence *
          Math.min(
            0.88,
            0.62 +
            subjectCount * 0.05
          )
        );

      if (
        candidateConfidence <=
        pattern.emittedConfidence + 0.04
      ) {
        return null;
      }

      pattern.emittedConfidence =
        candidateConfidence;

      const candidate = {
        from: `domain:${d}`,
        relation: pattern.relation,
        to: pattern.to,
        confidence: candidateConfidence,
        supportCount: subjectCount,
        sourceSubjects: Object.keys(pattern.subjects).slice(0, 16),
        inferred: true,
        provenance: 'generalization'
      };

      this.emitted++;
      this.last = candidate;
      return candidate;
    }

    reset() {
      this.patterns.clear();
      this.emitted = 0;
      this.last = null;
      return true;
    }

    exportState() {
      return {
        schemaVersion: 1,
        emitted: this.emitted,
        patterns: Array.from(this.patterns.values()).slice(-this.options.maxPatterns),
        last: this.last
      };
    }

    importState(state) {
      if (!state || Number(state.schemaVersion) !== 1) return false;

      this.patterns.clear();

      for (const pattern of Array.isArray(state.patterns) ? state.patterns : []) {
        if (!pattern?.key) continue;
        this.patterns.set(String(pattern.key), {
          ...pattern,
          subjects: { ...(pattern.subjects || {}) }
        });
      }

      this.emitted = Math.max(0, Number(state.emitted) || 0);
      this.last = state.last || null;
      return true;
    }

    status() {
      return {
        ready: true,
        role: 'cross-example-structural-generalization',
        patterns: this.patterns.size,
        emittedHypotheses: this.emitted,
        writesFactsDirectly: false,
        last: this.last
      };
    }
  }

  globalThis.VilotGeneralization = VilotGeneralization;
})();
