/**
 * VilotNI 2.5 - HypothesisStore.js
 *
 * Keeps inferred relations separate from factual knowledge.
 * Hypotheses can gain or lose support. Promotion requires trusted evidence.
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

  class VilotHypothesisStore {
    constructor(options = {}) {
      this.options = {
        maxHypotheses: Math.max(512, options.learningHypothesisMax | 0 || 8192),
        promotionThreshold: clamp(options.learningHypothesisPromotionThreshold ?? 0.82),
        rejectionThreshold: clamp(options.learningHypothesisRejectionThreshold ?? 0.26),
        trustedThreshold: clamp(options.learningTrustedSourceThreshold ?? 0.85)
      };

      this.items = new Map();
      this.observations = 0;
      this.promotions = 0;
      this.rejections = 0;
      this.last = null;
    }

    _key(from, relation, to) {
      return `${normalize(from)}|${normalize(relation)}|${normalize(to)}`;
    }

    _new(from, relation, to) {
      return {
        key: this._key(from, relation, to),
        from: normalize(from),
        relation: normalize(relation),
        to: normalize(to),
        alpha: 1,
        beta: 1,
        confidence: 0.5,
        confirmations: 0,
        contradictions: 0,
        trustedSupport: 0,
        sourceKinds: {},
        inferredSupport: 0,
        directSupport: 0,
        status: 'provisional',
        firstSeen: Date.now(),
        lastSeen: Date.now(),
        predictionError: 0,
        metadata: {}
      };
    }

    _recompute(item) {
      item.confidence =
        clamp(
          item.alpha /
          Math.max(1e-9, item.alpha + item.beta)
        );

      if (item.status === 'promoted') {
        return item;
      }

      if (item.confidence <= this.options.rejectionThreshold) {
        item.status = 'rejected';
      } else if (
        item.confidence >= this.options.promotionThreshold &&
        item.trustedSupport >= this.options.trustedThreshold
      ) {
        item.status = 'promotable';
      } else if (
        item.contradictions > 0 &&
        item.contradictions >= item.confirmations
      ) {
        item.status = 'contested';
      } else {
        item.status = 'provisional';
      }

      return item;
    }

    _evictIfNeeded() {
      if (this.items.size < this.options.maxHypotheses) return;

      let weakestKey = null;
      let weakestScore = Infinity;

      for (const [key, item] of this.items) {
        if (item.status === 'promoted') continue;

        const score =
          item.confidence * 0.45 +
          Math.min(1, item.confirmations / 8) * 0.25 +
          item.trustedSupport * 0.20 +
          Math.min(1, Math.abs(item.predictionError)) * 0.10;

        if (score < weakestScore) {
          weakestScore = score;
          weakestKey = key;
        }
      }

      if (weakestKey) this.items.delete(weakestKey);
    }

    observe(from, relation, to, metadata = {}) {
      const a = normalize(from);
      const r = normalize(relation);
      const b = normalize(to);

      if (!a || !r || !b || a === b) return null;

      const key = this._key(a, r, b);

      if (!this.items.has(key)) {
        this._evictIfNeeded();
        this.items.set(key, this._new(a, r, b));
      }

      const item = this.items.get(key);

      const trust =
        clamp(metadata.trust ?? 0.4);

      const support =
        clamp(metadata.support ?? metadata.strength ?? 1);

      const polarity =
        Number(metadata.polarity ?? 1) >= 0
          ? 1
          : -1;

      const evidence =
        support *
        trust *
        4;

      if (polarity >= 0) {
        item.alpha += evidence;
        item.confirmations++;
      } else {
        item.beta += evidence;
        item.contradictions++;
      }

      if (
        trust >= this.options.trustedThreshold &&
        polarity >= 0
      ) {
        item.trustedSupport =
          Math.max(
            item.trustedSupport,
            trust
          );
      }

      const provenance =
        String(
          metadata.provenance ||
          'unknown'
        );

      item.sourceKinds[provenance] =
        (item.sourceKinds[provenance] || 0) + 1;

      if (metadata.inferred === true) {
        item.inferredSupport += evidence;
      } else {
        item.directSupport += evidence;
      }

      item.predictionError =
        Number(
          metadata.predictionError
        ) || item.predictionError || 0;

      if (metadata.domain) {
        item.metadata.domain =
          normalize(metadata.domain);
      }

      if (metadata.entryId) {
        item.metadata.entryId =
          String(metadata.entryId);
      }

      item.lastSeen = Date.now();
      this.observations++;

      this.last =
        this._recompute(item);

      return this.last;
    }

    get(from, relation, to) {
      return this.items.get(this._key(from, relation, to)) || null;
    }

    promotable(limit = 16) {
      return Array.from(this.items.values())
        .filter(item => item.status === 'promotable')
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, Math.max(1, limit | 0 || 16));
    }

    promote(itemOrKey) {
      const item =
        typeof itemOrKey === 'string'
          ? this.items.get(itemOrKey)
          : itemOrKey;

      if (!item || item.status === 'promoted') return false;

      item.status = 'promoted';
      item.promotedAt = Date.now();
      this.promotions++;
      return true;
    }

    markContested(itemOrKey, reason = 'contradiction') {
      const item =
        typeof itemOrKey === 'string'
          ? this.items.get(itemOrKey)
          : itemOrKey;

      if (!item) return false;

      item.status = 'contested';
      item.metadata.contestedReason = String(reason || 'contradiction');
      item.metadata.contestedAt = Date.now();
      return true;
    }

    candidates(limit = 16, statuses = ['provisional', 'promotable', 'contested']) {
      const allowed = new Set(
        Array.isArray(statuses)
          ? statuses.map(value => String(value || ''))
          : [String(statuses || '')]
      );

      return Array.from(this.items.values())
        .filter(item => allowed.has(item.status))
        .sort((a, b) => {
          const scoreA =
            a.confidence * 0.46 +
            Math.min(1, a.confirmations / 8) * 0.22 +
            a.trustedSupport * 0.18 +
            Math.min(1, Math.abs(a.predictionError || 0)) * 0.14;

          const scoreB =
            b.confidence * 0.46 +
            Math.min(1, b.confirmations / 8) * 0.22 +
            b.trustedSupport * 0.18 +
            Math.min(1, Math.abs(b.predictionError || 0)) * 0.14;

          return scoreB - scoreA;
        })
        .slice(0, Math.max(1, limit | 0 || 16));
    }

    reset() {
      this.items.clear();
      this.observations = 0;
      this.promotions = 0;
      this.rejections = 0;
      this.last = null;
      return true;
    }

    exportState() {
      const rows =
        Array.from(this.items.values())
          .sort((a, b) => {
            const scoreA =
              a.confidence +
              Math.min(1, a.confirmations / 10) * 0.25;

            const scoreB =
              b.confidence +
              Math.min(1, b.confirmations / 10) * 0.25;

            return scoreB - scoreA;
          })
          .slice(0, this.options.maxHypotheses);

      return {
        schemaVersion: 1,
        observations: this.observations,
        promotions: this.promotions,
        rejections: this.rejections,
        items: rows.map(item => ({
          ...item,
          sourceKinds: { ...item.sourceKinds },
          metadata: { ...item.metadata }
        }))
      };
    }

    importState(state) {
      if (!state || Number(state.schemaVersion) !== 1) return false;

      this.items.clear();

      for (const raw of Array.isArray(state.items) ? state.items : []) {
        if (!raw?.from || !raw?.relation || !raw?.to) continue;

        const item = this._new(raw.from, raw.relation, raw.to);
        Object.assign(item, raw);
        item.sourceKinds = { ...(raw.sourceKinds || {}) };
        item.metadata = { ...(raw.metadata || {}) };
        this.items.set(item.key || this._key(item.from, item.relation, item.to), item);
      }

      this.observations = Math.max(0, Number(state.observations) || 0);
      this.promotions = Math.max(0, Number(state.promotions) || 0);
      this.rejections = Math.max(0, Number(state.rejections) || 0);

      return true;
    }

    status() {
      let provisional = 0;
      let promotable = 0;
      let promoted = 0;
      let contested = 0;
      let rejected = 0;

      for (const item of this.items.values()) {
        if (item.status === 'promotable') promotable++;
        else if (item.status === 'promoted') promoted++;
        else if (item.status === 'contested') contested++;
        else if (item.status === 'rejected') rejected++;
        else provisional++;
      }

      return {
        ready: true,
        role: 'provisional-hypothesis-separate-from-facts',
        hypotheses: this.items.size,
        provisional,
        promotable,
        promoted,
        contested,
        rejected,
        observations: this.observations,
        promotions: this.promotions,
        last: this.last
      };
    }
  }

  globalThis.VilotHypothesisStore = VilotHypothesisStore;
})();
