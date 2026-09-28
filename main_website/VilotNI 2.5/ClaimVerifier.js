/**
 * VilotNI 2.5 - ClaimVerifier.js
 *
 * Verifies semantic clauses against their actual provenance before realization.
 * It can reject or lower a clause; it never upgrades confidence.
 */
(() => {
  'use strict';
  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));
  const normalize = value => String(value || '').toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9'+.: -]+/g, ' ').replace(/\s+/g, ' ').trim();

  class VilotClaimVerifier {
    constructor(graph, temporalKnowledge, graphReasoner, uncertaintyState = null, options = {}, policy = null) {
      this.graph = graph;
      this.temporal = temporalKnowledge;
      this.reasoner = graphReasoner;
      this.uncertainty = uncertaintyState;
      this.policy = policy || null;
      this.options = {
        minVerificationScore: clamp(options.claimVerifierMinScore ?? 0.58),
        trustedSourceThreshold: clamp(options.claimVerifierTrustedThreshold ?? 0.78),
        maxDirectionMismatch: clamp(options.claimVerifierDirectionMismatch ?? 0.25)
      };
      this.verifications = 0;
      this.rejections = 0;
      this.last = null;
    }

    _overlap(a, b) {
      const A = new Set(normalize(a).split(/\s+/).filter(Boolean));
      const B = new Set(normalize(b).split(/\s+/).filter(Boolean));
      if (!A.size || !B.size) return 0;
      let hit = 0;
      for (const t of A) if (B.has(t)) hit++;
      return hit / Math.max(A.size, B.size);
    }

    _verifyClause(clause) {
      const sourceType = String(clause?.sourceType || '').toLowerCase();
      const verification = clause?.verification || null;
      const trust = clamp(clause?.trust ?? 0);
      let score = trust;
      let contradictionPressure = 0;
      let freshness = 0.9;
      let reason = 'trusted-semantic-source';
      let valid = trust >= this.options.trustedSourceThreshold;

      if (verification?.kind === 'knowledge_graph' || String(clause?.sourceEntryId || '').startsWith('KG:')) {
        const key = verification?.key || String(clause.sourceEntryId).slice(3);
        const edge = this.graph?.edges?.get?.(key) || null;
        if (!edge) return { valid: false, score: 0, reason: 'missing-learned-edge', contradictionPressure: 1, freshness: 0 };
        const temporal = this.temporal?.assess?.(edge) || { active: true, freshness: 1 };
        const authority = this.policy?.edgeAuthority?.(
          edge,
          this.temporal,
          {
            minConfidence: this.options.minVerificationScore,
            maxContradiction: 0.25
          }
        );
        contradictionPressure = clamp(edge.contradictionPressure ?? 0);
        freshness = clamp(temporal.freshness ?? 1);
        const direction = Math.min(this._overlap(clause.subject, edge.from), this._overlap(clause.value, edge.to));
        score = clamp(edge.confidence * 0.55 + (1 - contradictionPressure) * 0.20 + freshness * 0.15 + direction * 0.10);
        valid = (authority ? authority.factual : (edge.status === 'consolidated' && temporal.active && contradictionPressure < 0.25)) && direction >= this.options.maxDirectionMismatch;
        reason = valid ? 'consolidated-learned-edge' : (authority?.reason || 'learned-edge-not-verified');
      } else if (verification?.kind === 'graph_reasoning' || String(clause?.sourceEntryId || '').startsWith('GR:')) {
        const id = verification?.pathId || clause.sourceEntryId;
        const path = this.reasoner?.verifyPath?.(id);
        valid = Boolean(path?.valid && path?.derived?.responseEligible);
        score = valid ? clamp(path.derived.confidence * 0.90) : 0;
        freshness = valid ? 0.88 : 0;
        reason = valid ? `verified-${path.derived.rule}` : 'unverified-reasoning-path';
      } else {
        const allowed = this.policy?.structuredSourceAllowed?.(
          sourceType,
          trust,
          Number(clause?.support || 0)
        );
        if (allowed) {
          valid = allowed.allowed;
          score = clamp(
            allowed.kind === 'structured'
              ? trust
              : trust * 0.70 + clamp((Number(clause?.support) || 0) / 4) * 0.30
          );
          reason = valid ? `${allowed.kind}-source` : `insufficient-${allowed.kind}-authority`;
        } else if (sourceType.includes('training') || sourceType === 'model-general' || sourceType === 'structured') {
          valid = trust >= this.options.minVerificationScore;
          score = clamp(trust);
          reason = valid ? 'structured-source' : 'low-trust-structured-source';
        } else {
          valid = trust >= this.options.trustedSourceThreshold && Number(clause?.support || 0) >= 2;
          score = clamp(trust * 0.70 + clamp((Number(clause?.support) || 0) / 4) * 0.30);
          reason = valid ? 'branch-supported-source' : 'insufficient-provenance';
        }
      }

      if (clause?.counterfactual) {
        // A hypothetical clause is allowed as hypothetical reasoning, never as a factual claim.
        valid = valid || score >= 0.50;
        reason = `counterfactual:${reason}`;
      }

      return { valid, score, reason, contradictionPressure, freshness };
    }

    verifyPromptStructure(text, analysis = {}) {
      const directives = new Set(['explain','describe','show','tell','analyze','compare','discuss','trace','evaluate']);
      const clauses = (analysis?.complexLanguage?.clauses || []).filter(clause => !directives.has(String(clause?.predicate || '').toLowerCase()));
      const refs = analysis?.referenceLinks || [];
      const response = String(text || '').toLowerCase();
      if (!clauses.length) return {score:0.5,clauseCoverage:0.5,referenceCoverage:0.5,missing:[]};
      const missing=[]; let clauseHit=0;
      for(const clause of clauses.slice(0,12)){
        const terms=[clause.subject,clause.predicate,clause.object].flatMap(v=>String(v||'').toLowerCase().split(/\s+/)).filter(w=>w.length>2);
        const ratio=terms.length?terms.filter(t=>response.includes(t)).length/terms.length:0.5;
        if(ratio>=0.30)clauseHit++; else missing.push(clause.id);
      }
      const clauseCoverage=clauseHit/Math.max(1,Math.min(12,clauses.length));
      const resolved=refs.filter(r=>r.resolved);
      const referenceCoverage=refs.length?resolved.length/refs.length:1;
      const propositions=analysis?.propositionGraph?.propositions||[];
      let propositionHit=0;
      for(const prop of propositions.slice(0,20)){
        const terms=[prop.subject,prop.relation,prop.object].flatMap(v=>String(v||'').toLowerCase().split(/\s+/)).filter(w=>w.length>2);
        const ratio=terms.length?terms.filter(t=>response.includes(t)).length/terms.length:0.5;
        if(ratio>=0.28) propositionHit++;
      }
      const propositionCoverage=propositions.length?propositionHit/Math.min(20,propositions.length):0.5;
      const mechanism=analysis?.mechanismGraph||null;
      const mechanismTerms=(mechanism?.stages||[]).flatMap(s=>[s.from,s.relation,s.to]).flatMap(v=>String(v||'').toLowerCase().split(/\s+/)).filter(w=>w.length>2);
      const mechanismCoverage=mechanism?.active&&mechanismTerms.length?mechanismTerms.filter(t=>response.includes(t)).length/mechanismTerms.length:1;
      const score=Math.max(0,Math.min(1,clauseCoverage*.50+referenceCoverage*.16+propositionCoverage*.22+mechanismCoverage*.12));
      return {score,clauseCoverage,referenceCoverage,propositionCoverage,mechanismCoverage,missing};
    }

    verifyPlan(plan, context = {}) {
      const started = performance.now();
      const clauses = plan?.clauses || [];
      const verified = [];
      const rejected = [];
      let contradiction = 0;
      let freshness = 0;

      for (const clause of clauses) {
        const result = this._verifyClause(clause);
        const row = {
          ...clause,
          verificationScore: result.score,
          verificationReason: result.reason,
          verified: result.valid,
          confidence: Math.min(Number(clause.confidence) || 0, result.score * 100)
        };
        contradiction += result.contradictionPressure;
        freshness += result.freshness;
        if (result.valid && result.score >= this.options.minVerificationScore) verified.push(row);
        else rejected.push({ clause: row, ...result });
      }

      const diagnostics = {
        verifierMs: performance.now() - started,
        totalClauses: clauses.length,
        verifiedClauses: verified.length,
        rejectedClauses: rejected.length,
        meanContradictionPressure: clauses.length ? contradiction / clauses.length : 0,
        meanFreshness: clauses.length ? freshness / clauses.length : 0.9,
        reasons: rejected.slice(0, 12).map(row => ({ type: row.clause?.type, reason: row.reason, score: row.score }))
      };
      this.verifications++;
      this.rejections += rejected.length;
      this.last = diagnostics;

      const out = {
        ...plan,
        clauses: verified,
        confidence: verified.length
          ? verified.reduce((sum, clause) => sum + (Number(clause.confidence) || 0), 0) / verified.length
          : 0,
        verification: diagnostics,
        diagnostics: { ...(plan?.diagnostics || {}), claimVerifier: diagnostics }
      };
      if (context.counterfactualWorkspace?.active) {
        out.counterfactual = true;
        out.counterfactualId = context.counterfactualWorkspace.id;
      }
      return out;
    }

    status() {
      return { ready: true, role: 'semantic-claim-provenance-and-direction-verifier', verifications: this.verifications, rejections: this.rejections, last: this.last };
    }
  }

  globalThis.VilotClaimVerifier = VilotClaimVerifier;
})();
