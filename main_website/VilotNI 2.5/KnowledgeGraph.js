/**
 * VilotNI 2.5 - KnowledgeGraph.js
 *
 * Persistent relation graph for learned concepts.
 *
 * Important:
 * - A graph edge can be observed, provisional, consolidated, or contested.
 * - Model-generated text alone is never enough to create a consolidated factual
 *   relation.
 * - Positive and negative evidence are tracked separately.
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

  class VilotKnowledgeGraph {
    constructor(options = {}, policy = null) {
      this.policy = policy || null;
      this.options = {
        maxEdges: Math.max(2048, options.learningGraphMaxEdges | 0 || 16384),
        evidenceScale: Math.max(1, Number(options.learningGraphEvidenceScale) || 4),
        consolidationThreshold: clamp(options.learningGraphConsolidationThreshold ?? 0.82),
        contestedThreshold: clamp(options.learningGraphContestedThreshold ?? 0.26),
        trustedThreshold: clamp(options.learningTrustedSourceThreshold ?? 0.85)
      };

      this.edges = new Map();
      this.observations = 0;
      this.consolidations = 0;
      this.contested = 0;
      this.lastObservation = null;
      this.revision = 0;
    }

    _key(from, relation, to) {
      return `${normalize(from)}|${normalize(relation)}|${normalize(to)}`;
    }

    _sourceTrust(provenance) {
      if (this.policy?.sourceTrust) {
        return this.policy.sourceTrust(provenance);
      }
      switch (String(provenance || 'unknown')) {
        case 'traininginfo': return 0.98;
        case 'entity_seed': return 0.97;
        case 'verified_source': return 0.96;
        case 'user': return 0.56;
        case 'correlation': return 0.48;
        case 'background_replay': return 0.52;
        case 'generalization': return 0.34;
        case 'cognitive_inference': return 0.30;
        case 'model_response': return 0.20;
        case 'failed_response': return 0.12;
        default: return 0.40;
      }
    }

    _newEdge(from, relation, to) {
      return {
        key: this._key(from, relation, to),
        from: normalize(from),
        relation: normalize(relation),
        to: normalize(to),
        positiveEvidence: 0,
        negativeEvidence: 0,
        trustedEvidence: 0,
        confirmations: 0,
        contradictions: 0,
        sourceKinds: {},
        provenanceWeights: {},
        confidence: 0.5,
        contradictionPressure: 0,
        status: 'observed',
        firstSeen: Date.now(),
        lastSeen: Date.now(),
        usefulness: 0.5,
        retrievals: 0,
        metadata: {}
      };
    }

    _recompute(edge) {
      const alpha = 1 + edge.positiveEvidence;
      const beta = 1 + edge.negativeEvidence;
      const confidence = alpha / Math.max(1e-9, alpha + beta);

      edge.confidence = clamp(confidence);
      edge.contradictionPressure = clamp(
        edge.negativeEvidence /
        Math.max(1e-9, edge.positiveEvidence + edge.negativeEvidence)
      );

      const wasConsolidated = edge.status === 'consolidated';
      const wasContested = edge.status === 'contested';

      if (edge.contradictionPressure >= this.options.contestedThreshold) {
        edge.status = 'contested';
      } else if (
        edge.confidence >= this.options.consolidationThreshold &&
        edge.trustedEvidence >= this.options.trustedThreshold
      ) {
        edge.status = 'consolidated';
      } else if (edge.confidence >= 0.68) {
        edge.status = 'provisional';
      } else {
        edge.status = 'observed';
      }

      if (!wasConsolidated && edge.status === 'consolidated') {
        this.consolidations++;
      }

      if (!wasContested && edge.status === 'contested') {
        this.contested++;
      }

      return edge;
    }

    _evictIfNeeded() {
      if (this.edges.size < this.options.maxEdges) return;

      let weakestKey = null;
      let weakestScore = Infinity;

      for (const [key, edge] of this.edges) {
        const score =
          edge.confidence * 0.45 +
          clamp(edge.usefulness) * 0.20 +
          Math.min(1, edge.retrievals / 12) * 0.15 +
          Math.min(1, edge.confirmations / 8) * 0.20;

        if (
          edge.status !== 'consolidated' &&
          score < weakestScore
        ) {
          weakestScore = score;
          weakestKey = key;
        }
      }

      if (!weakestKey) {
        for (const [key, edge] of this.edges) {
          const score = edge.confidence + edge.retrievals * 0.01;
          if (score < weakestScore) {
            weakestScore = score;
            weakestKey = key;
          }
        }
      }

      if (weakestKey) this.edges.delete(weakestKey);
    }

    observeEdge(from, relation, to, metadata = {}) {
      const a = normalize(from);
      const r = normalize(relation);
      const b = normalize(to);

      if (!a || !r || !b || a === b) return null;

      const key = this._key(a, r, b);

      if (!this.edges.has(key)) {
        this._evictIfNeeded();
        this.edges.set(key, this._newEdge(a, r, b));
      }

      const edge = this.edges.get(key);
      const provenance = String(metadata.provenance || 'unknown');

      const trust = clamp(
        metadata.trust ??
        this._sourceTrust(provenance)
      );

      const strength = clamp(metadata.strength ?? metadata.support ?? 1);
      const polarity = Number(metadata.polarity ?? 1) >= 0 ? 1 : -1;
      const mass = strength * trust * this.options.evidenceScale;

      if (polarity >= 0) {
        edge.positiveEvidence += mass;
        edge.confirmations++;
      } else {
        edge.negativeEvidence += mass;
        edge.contradictions++;
      }

      if (trust >= this.options.trustedThreshold && polarity >= 0) {
        edge.trustedEvidence = Math.max(edge.trustedEvidence, trust);
      }

      edge.sourceKinds[provenance] =
        (edge.sourceKinds[provenance] || 0) + 1;

      edge.provenanceWeights[provenance] =
        Math.max(
          Number(edge.provenanceWeights[provenance]) || 0,
          trust
        );

      edge.usefulness = clamp(
        edge.usefulness * 0.82 +
        clamp(metadata.usefulness ?? 0.5) * 0.18
      );

      edge.lastSeen = Date.now();

      if (metadata.domain) {
        edge.metadata.domain = normalize(metadata.domain);
      }

      if (metadata.entryId) {
        edge.metadata.entryId = String(metadata.entryId);
      }

      for (const [source, target] of [
        ['validFrom', 'validFrom'],
        ['validUntil', 'validUntil'],
        ['lastVerified', 'lastVerified'],
        ['timeless', 'timeless']
      ]) {
        if (metadata[source] != null) edge.metadata[target] = metadata[source];
      }

      this.observations++;
      this.revision = (this.revision + 1) >>> 0;
      this.lastObservation = {
        at: edge.lastSeen,
        from: edge.from,
        relation: edge.relation,
        to: edge.to,
        provenance,
        trust,
        polarity
      };

      return this._recompute(edge);
    }

    getEdge(from, relation, to) {
      return this.edges.get(this._key(from, relation, to)) || null;
    }

    score(from, relation, to) {
      const edge = this.getEdge(from, relation, to);
      if (!edge) return 0;

      edge.retrievals++;

      return clamp(
        edge.confidence *
        (1 - edge.contradictionPressure * 0.75)
      );
    }

    snapshotEdge(edgeOrKey) {
      const edge =
        typeof edgeOrKey === 'string'
          ? this.edges.get(edgeOrKey)
          : edgeOrKey;
      if (!edge) return null;
      return JSON.parse(JSON.stringify(edge));
    }

    restoreEdge(snapshot) {
      if (!snapshot?.from || !snapshot?.relation || !snapshot?.to) return false;
      const key = snapshot.key || this._key(snapshot.from, snapshot.relation, snapshot.to);
      this.edges.set(key, {
        ...snapshot,
        sourceKinds: { ...(snapshot.sourceKinds || {}) },
        provenanceWeights: { ...(snapshot.provenanceWeights || {}) },
        metadata: { ...(snapshot.metadata || {}) }
      });
      this.revision = (this.revision + 1) >>> 0;
      return true;
    }

    removeEdge(edgeOrKey) {
      const key = typeof edgeOrKey === 'string' ? edgeOrKey : edgeOrKey?.key;
      if (!key) return false;
      const removed = this.edges.delete(key);
      if (removed) this.revision = (this.revision + 1) >>> 0;
      return removed;
    }

    markContested(edgeOrKey, reason = 'contradiction') {
      const edge =
        typeof edgeOrKey === 'string'
          ? this.edges.get(edgeOrKey)
          : edgeOrKey;

      if (!edge) return false;

      edge.status = 'contested';
      edge.metadata.contestedReason = String(reason || 'contradiction');
      edge.metadata.contestedAt = Date.now();
      this.revision = (this.revision + 1) >>> 0;
      return true;
    }

    topEdges(limit = 64, predicate = null) {
      const rows = Array.from(this.edges.values());

      const filtered =
        typeof predicate === 'function'
          ? rows.filter(predicate)
          : rows;

      filtered.sort((a, b) => {
        const scoreA =
          a.confidence * 0.55 +
          a.usefulness * 0.20 +
          Math.min(1, a.retrievals / 16) * 0.10 +
          Math.min(1, a.confirmations / 10) * 0.15;

        const scoreB =
          b.confidence * 0.55 +
          b.usefulness * 0.20 +
          Math.min(1, b.retrievals / 16) * 0.10 +
          Math.min(1, b.confirmations / 10) * 0.15;

        return scoreB - scoreA;
      });

      return filtered.slice(0, Math.max(1, limit | 0 || 64));
    }

    related(node, limit = 16) {
      const key = normalize(node);

      return this.topEdges(
        Math.max(1, limit | 0 || 16),
        edge =>
          edge.from === key ||
          edge.to === key
      );
    }

    reset() {
      this.edges.clear();
      this.observations = 0;
      this.consolidations = 0;
      this.contested = 0;
      this.lastObservation = null;
      this.revision = (this.revision + 1) >>> 0;
      return true;
    }

    exportState() {
      return {
        schemaVersion: 1,
        observations: this.observations,
        consolidations: this.consolidations,
        contested: this.contested,
        revision: this.revision,
        edges: this.topEdges(this.options.maxEdges).map(edge => ({
          ...edge,
          sourceKinds: { ...edge.sourceKinds },
          provenanceWeights: { ...edge.provenanceWeights },
          metadata: { ...edge.metadata }
        }))
      };
    }

    importState(state) {
      if (!state || Number(state.schemaVersion) !== 1) return false;

      this.edges.clear();

      for (const raw of Array.isArray(state.edges) ? state.edges : []) {
        if (!raw?.from || !raw?.relation || !raw?.to) continue;
        const edge = this._newEdge(raw.from, raw.relation, raw.to);
        Object.assign(edge, raw);
        edge.sourceKinds = { ...(raw.sourceKinds || {}) };
        edge.provenanceWeights = { ...(raw.provenanceWeights || {}) };
        edge.metadata = { ...(raw.metadata || {}) };
        this.edges.set(edge.key || this._key(edge.from, edge.relation, edge.to), edge);
      }

      this.observations = Math.max(0, Number(state.observations) || 0);
      this.consolidations = Math.max(0, Number(state.consolidations) || 0);
      this.contested = Math.max(0, Number(state.contested) || 0);
      this.revision = Math.max(0, Number(state.revision) || this.edges.size) >>> 0;

      return true;
    }

    status() {
      let consolidated = 0;
      let provisional = 0;
      let contested = 0;

      for (const edge of this.edges.values()) {
        if (edge.status === 'consolidated') consolidated++;
        else if (edge.status === 'contested') contested++;
        else if (edge.status === 'provisional') provisional++;
      }

      return {
        ready: true,
        role: 'persistent-provenance-aware-knowledge-graph',
        edges: this.edges.size,
        observations: this.observations,
        consolidated,
        provisional,
        contested,
        revision: this.revision,
        lastObservation: this.lastObservation
      };
    }
  }

  globalThis.VilotKnowledgeGraph = VilotKnowledgeGraph;
})();
