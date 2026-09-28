/**
 * VilotNI 2.5 - KnowledgePolicy.js
 *
 * Shared authority policy for learned knowledge. KnowledgeGraph,
 * LearningController, GraphReasoner, LearnedKnowledgeBridge, and ClaimVerifier
 * all use the same provenance weights, relation canonicalization, temporal /
 * contradiction eligibility rules, and safe-transitive whitelist.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));

  const normalize = value =>
    String(value || '')
      .toLowerCase()
      .replace(/[’]/g, "'")
      .replace(/_/g, ' ')
      .replace(/[^a-z0-9'+.:\- ]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 192);

  class VilotKnowledgePolicy {
    constructor(options = {}) {
      this.options = {
        factualMinConfidence: clamp(options.knowledgePolicyFactualMinConfidence ?? 0.78),
        maxContradiction: clamp(options.knowledgePolicyMaxContradiction ?? 0.20),
        structuredSourceThreshold: clamp(options.knowledgePolicyStructuredSourceThreshold ?? 0.58),
        branchSourceThreshold: clamp(options.knowledgePolicyBranchSourceThreshold ?? 0.78),
        minimumBranchSupport: Math.max(1, options.knowledgePolicyMinimumBranchSupport | 0 || 2)
      };

      this.safeTransitive = new Set([
        'part of',
        'is a',
        'subtype of',
        'located in'
      ]);
    }

    canonicalRelation(value) {
      return normalize(value);
    }

    sourceTrust(provenance, explicit = null) {
      if (Number.isFinite(Number(explicit))) return clamp(explicit);

      switch (normalize(provenance)) {
        case 'traininginfo': return 0.98;
        case 'entity seed': return 0.97;
        case 'entity_seed': return 0.97;
        case 'architecture_seed': return 0.995;
        case 'verified source': return 0.96;
        case 'verified_source': return 0.96;
        case 'user': return 0.56;
        case 'background replay': return 0.52;
        case 'background_replay': return 0.52;
        case 'correlation': return 0.48;
        case 'generalization': return 0.34;
        case 'cognitive inference': return 0.30;
        case 'cognitive_inference': return 0.30;
        case 'model response': return 0.20;
        case 'model_response': return 0.20;
        case 'failed response': return 0.12;
        case 'failed_response': return 0.12;
        default: return 0.40;
      }
    }

    isTrustedFactualSource(provenance) {
      const source = normalize(provenance);
      return source === 'traininginfo' || source === 'entity seed' || source === 'entity_seed' || source === 'architecture seed' || source === 'architecture_seed' || source === 'verified source' || source === 'verified_source';
    }

    isSafeTransitive(relation) {
      return this.safeTransitive.has(this.canonicalRelation(relation));
    }

    edgeAuthority(edge, temporal = null, overrides = {}) {
      if (!edge) {
        return { factual: false, reason: 'missing-edge', confidence: 0, freshness: 0, contradictionPressure: 1 };
      }

      const minConfidence = clamp(overrides.minConfidence ?? this.options.factualMinConfidence);
      const maxContradiction = clamp(overrides.maxContradiction ?? this.options.maxContradiction);
      const temporalState = temporal?.assess?.(edge) || { active: true, freshness: 1 };
      const confidence = clamp(edge.confidence ?? 0);
      const contradictionPressure = clamp(edge.contradictionPressure ?? 0);
      const consolidated = edge.status === 'consolidated';
      const active = temporalState.active !== false;
      const factual = consolidated && active && confidence >= minConfidence && contradictionPressure < maxContradiction;

      let reason = 'eligible';
      if (!consolidated) reason = `status:${edge.status || 'unknown'}`;
      else if (!active) reason = 'temporally-inactive';
      else if (confidence < minConfidence) reason = 'confidence-below-policy';
      else if (contradictionPressure >= maxContradiction) reason = 'contradiction-pressure';

      return {
        factual,
        reason,
        confidence,
        freshness: clamp(temporalState.freshness ?? 1),
        contradictionPressure,
        relation: this.canonicalRelation(edge.relation)
      };
    }

    structuredSourceAllowed(sourceType, trust, support = 0) {
      const source = normalize(sourceType);
      const t = clamp(trust);
      const s = Math.max(0, Number(support) || 0);

      if (source.includes('training') || source.includes('entity') || source === 'model-general' || source === 'structured') {
        return {
          allowed: t >= this.options.structuredSourceThreshold,
          threshold: this.options.structuredSourceThreshold,
          kind: 'structured'
        };
      }

      return {
        allowed:
          t >= this.options.branchSourceThreshold &&
          s >= this.options.minimumBranchSupport,
        threshold: this.options.branchSourceThreshold,
        minimumSupport: this.options.minimumBranchSupport,
        kind: 'branch-supported'
      };
    }

    status() {
      return {
        ready: true,
        role: 'shared-knowledge-authority-policy',
        factualMinConfidence: this.options.factualMinConfidence,
        maxContradiction: this.options.maxContradiction,
        safeTransitiveRelations: Array.from(this.safeTransitive),
        sharedBy: [
          'KnowledgeGraph',
          'LearningController',
          'GraphReasoner',
          'LearnedKnowledgeBridge',
          'ClaimVerifier'
        ]
      };
    }
  }

  globalThis.VilotKnowledgePolicy = VilotKnowledgePolicy;
})();
