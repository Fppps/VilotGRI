/**
 * VilotNI 2.5 - GraphReasoner.js
 *
 * Bounded relation-aware graph reasoning. Multi-hop paths are useful as
 * relevance evidence, but only explicitly safe relation compositions create a
 * derived factual proposition.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));

  const normalize = value =>
    String(value || '')
      .toLowerCase()
      .replace(/[’]/g, "'")
      .replace(/[^a-z0-9'+.: -]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

  class VilotGraphReasoner {
    constructor(graph, temporalKnowledge = null, options = {}, policy = null) {
      this.graph = graph;
      this.temporal = temporalKnowledge;
      this.policy = policy || null;
      this.options = {
        maxDepth: Math.max(1, Math.min(4, options.graphReasonerMaxDepth | 0 || 3)),
        maxFrontier: Math.max(8, Math.min(128, options.graphReasonerMaxFrontier | 0 || 32)),
        maxPaths: Math.max(4, Math.min(64, options.graphReasonerMaxPaths | 0 || 16)),
        scanLimit: Math.max(64, Math.min(1024, options.graphReasonerScanLimit | 0 || 256)),
        minEdgeConfidence: clamp(options.graphReasonerMinEdgeConfidence ?? 0.72),
        minDerivedConfidence: clamp(options.graphReasonerMinDerivedConfidence ?? 0.70)
      };
      this.safeTransitive = new Set(
        this.policy?.status?.().safeTransitiveRelations || [
          'part of',
          'is a',
          'subtype of',
          'located in'
        ]
      );
      this.reasonings = 0;
      this.last = null;
      this.pathRegistry = new Map();
    }

    _eligible(edge, workspace = null) {
      const authority = this.policy?.edgeAuthority?.(
        edge,
        this.temporal,
        {
          minConfidence: this.options.minEdgeConfidence,
          maxContradiction: 0.20
        }
      );
      if (authority ? !authority.factual : (!edge || edge.status !== 'consolidated' || edge.confidence < this.options.minEdgeConfidence || edge.contradictionPressure >= 0.20 || !(this.temporal?.assess?.(edge) || { active: true }).active)) return false;
      if (workspace?.allowsEdge && !workspace.allowsEdge(edge)) return false;
      return true;
    }

    _edgeScore(edge) {
      const temporal = this.temporal?.assess?.(edge) || { freshness: 1 };
      return clamp(
        edge.confidence * 0.62 +
        (1 - edge.contradictionPressure) * 0.18 +
        clamp(edge.usefulness ?? 0.5) * 0.08 +
        temporal.freshness * 0.12
      );
    }

    _tokenMatch(node, seeds) {
      const n = normalize(node);
      if (!n) return 0;
      let best = 0;
      for (const seedRaw of seeds || []) {
        const seed = normalize(seedRaw);
        if (!seed) continue;
        if (n === seed) return 1;
        if (n.includes(seed) || seed.includes(n)) best = Math.max(best, 0.82);
        const a = new Set(n.split(/\s+/));
        const b = new Set(seed.split(/\s+/));
        let hit = 0;
        for (const t of a) if (b.has(t)) hit++;
        const union = a.size + b.size - hit;
        if (union) best = Math.max(best, hit / union);
      }
      return best;
    }

    _seedEdges(seeds, workspace, deadlineAt) {
      const out = new Map();
      for (const seed of seeds.slice(0, 12)) {
        if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt) break;
        for (const edge of this.graph?.related?.(seed, 24) || []) {
          if (this._eligible(edge, workspace)) out.set(edge.key, edge);
        }
      }
      if (out.size < 6) {
        for (const edge of this.graph?.topEdges?.(this.options.scanLimit) || []) {
          if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt) break;
          if (!this._eligible(edge, workspace)) continue;
          const relevance = Math.max(this._tokenMatch(edge.from, seeds), this._tokenMatch(edge.to, seeds));
          if (relevance >= 0.28) out.set(edge.key, edge);
          if (out.size >= this.options.maxFrontier) break;
        }
      }
      return Array.from(out.values());
    }

    _adjacent(node, workspace, deadlineAt) {
      const rows = this.graph?.related?.(node, this.options.maxFrontier) || [];
      const eligible = [];
      for (const edge of rows) {
        if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt) break;
        if (this._eligible(edge, workspace)) eligible.push(edge);
      }
      return eligible;
    }

    _pathId(path) {
      const raw = path.map(edge => edge.key).join('>');
      let h = 2166136261 >>> 0;
      for (let i = 0; i < raw.length; i++) {
        h ^= raw.charCodeAt(i);
        h = Math.imul(h, 16777619) >>> 0;
      }
      return `GR:${h.toString(16)}`;
    }

    _derive(path) {
      if (!path?.length || path.length < 2) return null;
      const relation = this.policy?.canonicalRelation?.(path[0].relation) || normalize(path[0].relation);
      if (this.policy?.isSafeTransitive ? !this.policy.isSafeTransitive(relation) : !this.safeTransitive.has(relation)) return null;
      if (!path.every(edge => normalize(edge.relation) === relation)) return null;

      let confidence = 1;
      let temporalFreshness = 1;
      for (const edge of path) {
        confidence *= clamp(edge.confidence);
        temporalFreshness = Math.min(
          temporalFreshness,
          this.temporal?.assess?.(edge)?.freshness ?? 1
        );
      }
      confidence = Math.pow(confidence, 1 / path.length) * 0.90 * temporalFreshness;
      if (confidence < this.options.minDerivedConfidence) return null;

      return {
        id: this._pathId(path),
        from: path[0].from,
        relation,
        to: path[path.length - 1].to,
        confidence: clamp(confidence),
        status: 'reasoned',
        responseEligible: true,
        provenance: 'graph_reasoning',
        path: path.map(edge => edge.key),
        depth: path.length,
        rule: `transitive:${relation}`
      };
    }

    reason(seedsInput, options = {}) {
      const started = performance.now();
      const seeds = Array.from(new Set((Array.isArray(seedsInput) ? seedsInput : [seedsInput]).map(normalize).filter(Boolean))).slice(0, 16);
      const deadlineAt = Number.isFinite(Number(options.deadlineAt)) ? Number(options.deadlineAt) : Infinity;
      const workspace = options.counterfactualWorkspace || null;
      const maxDepth = Math.max(1, Math.min(this.options.maxDepth, options.maxDepth | 0 || this.options.maxDepth));
      const initial = this._seedEdges(seeds, workspace, deadlineAt);
      const paths = [];
      const derived = [];
      const seenSignatures = new Set();
      let frontier = initial.map(edge => ({
        node: edge.to,
        path: [edge],
        score: this._edgeScore(edge)
      }));

      for (const edge of initial) {
        const sig = edge.key;
        if (!seenSignatures.has(sig)) {
          seenSignatures.add(sig);
          paths.push({ id: this._pathId([edge]), path: [edge.key], score: this._edgeScore(edge), depth: 1, responseEligible: true });
        }
      }

      for (let depth = 2; depth <= maxDepth && frontier.length; depth++) {
        if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt) break;
        const next = [];
        frontier.sort((a, b) => b.score - a.score);
        for (const state of frontier.slice(0, this.options.maxFrontier)) {
          if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt) break;
          for (const edge of this._adjacent(state.node, workspace, deadlineAt)) {
            if (state.path.some(old => old.key === edge.key)) continue;
            const path = [...state.path, edge];
            const signature = path.map(row => row.key).join('>');
            if (seenSignatures.has(signature)) continue;
            seenSignatures.add(signature);
            const score = clamp(state.score * 0.64 + this._edgeScore(edge) * 0.36);
            const id = this._pathId(path);
            const row = { id, path: path.map(item => item.key), score, depth, responseEligible: true };
            paths.push(row);
            this.pathRegistry.set(id, { ...row, edges: path, at: Date.now() });
            const proposition = this._derive(path);
            if (proposition) derived.push(proposition);
            next.push({ node: edge.to, path, score });
            if (paths.length >= this.options.maxPaths * 4) break;
          }
          if (paths.length >= this.options.maxPaths * 4) break;
        }
        frontier = next;
      }

      const bestPaths = paths.sort((a, b) => b.score - a.score).slice(0, this.options.maxPaths);
      const bestDerived = derived.sort((a, b) => b.confidence - a.confidence).slice(0, this.options.maxPaths);
      for (const item of bestDerived) {
        const reg = this.pathRegistry.get(item.id);
        if (reg) reg.derived = item;
      }
      while (this.pathRegistry.size > 128) this.pathRegistry.delete(this.pathRegistry.keys().next().value);

      const result = {
        seeds,
        paths: bestPaths,
        derived: bestDerived,
        diagnostics: {
          reasoningMs: performance.now() - started,
          seedCount: seeds.length,
          initialEdges: initial.length,
          pathCount: bestPaths.length,
          derivedCount: bestDerived.length,
          maxDepth,
          deadlineReached: Number.isFinite(deadlineAt) && performance.now() >= deadlineAt
        }
      };
      this.reasonings++;
      this.last = result.diagnostics;
      return result;
    }

    verifyPath(id) {
      const row = this.pathRegistry.get(String(id || ''));
      if (!row) return null;
      const stillValid = (row.edges || []).every(edge => this._eligible(edge));
      return { ...row, valid: stillValid };
    }

    status() {
      return {
        ready: true,
        role: 'bounded-relation-aware-knowledge-graph-reasoner',
        reasonings: this.reasonings,
        safeTransitiveRelations: Array.from(this.safeTransitive),
        registrySize: this.pathRegistry.size,
        last: this.last
      };
    }
  }

  globalThis.VilotGraphReasoner = VilotGraphReasoner;
})();
