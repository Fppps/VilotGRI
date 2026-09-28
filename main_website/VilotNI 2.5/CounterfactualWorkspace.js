/**
 * VilotNI 2.5 - CounterfactualWorkspace.js
 *
 * Ephemeral hypothetical sandbox. It never writes facts into KnowledgeGraph.
 */
(() => {
  'use strict';
  const normalize = value => String(value || '').toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9'+.: -]+/g, ' ').replace(/\s+/g, ' ').trim();

  class VilotCounterfactualWorkspace {
    constructor(processor, graphReasoner = null, options = {}) {
      this.processor = processor;
      this.reasoner = graphReasoner;
      this.options = {
        maxAssumptionChars: Math.max(128, options.counterfactualMaxAssumptionChars | 0 || 768),
        maxDisabledConcepts: Math.max(2, Math.min(16, options.counterfactualMaxDisabledConcepts | 0 || 8))
      };
      this.sessions = 0;
      this.last = null;
    }

    _tokens(text) {
      const rows = this.processor?.tokenize?.(String(text || ''), 64) || [];
      return rows.filter(row => row?.pos !== 'Punct').map(row => normalize(row?.lower || row?.surface)).filter(Boolean);
    }

    analyze(promptText, analysis = null) {
      const raw = String(promptText || '');
      const lower = raw.toLowerCase();
      const structuralConditional = Boolean(analysis?.complexLanguage?.features?.conditional || analysis?.propositionGraph?.propositions?.some?.(p => /condition|if|unless/.test(String(p?.type || p?.relation || ''))));
      const active = structuralConditional || /\b(what if|suppose|assuming|assume|imagine|hypothetically|if .* were|if .* was|if )\b/.test(lower);
      if (!active) {
        this.last = { active: false };
        return this.last;
      }

      const marker = lower.match(/\b(what if|suppose|assuming|assume|imagine|hypothetically)\b/);
      const start = marker ? marker.index + marker[0].length : 0;
      const assumptionText = raw.slice(start).trim().slice(0, this.options.maxAssumptionChars);
      const negated = /\b(not|never|without|stopped|stop|removed|disabled|ceased|cease|no longer)\b/.test(assumptionText.toLowerCase());
      const content = Array.from(new Set([...(analysis?.contentWords || []), ...this._tokens(assumptionText)])).map(normalize).filter(Boolean);
      const disabledConcepts = negated ? content.slice(0, this.options.maxDisabledConcepts) : [];

      const propositions = (analysis?.propositionGraph?.propositions || []).slice(0, 24);
      const assumptionProps = propositions.filter(p => /condition|if|unless|counterfactual/.test(String(p?.type || p?.relation || '')) || String(p?.condition || '').trim());
      const baselineProps = propositions.filter(p => !assumptionProps.includes(p));
      const workspace = {
        active: true,
        id: `CF:${Date.now()}:${++this.sessions}`,
        assumptionText,
        negated,
        contentWords: content.slice(0, 24),
        disabledConcepts,
        baseline: baselineProps.map(p => ({ subject:p.subject, relation:p.relation, object:p.object, polarity:p.polarity, certainty:p.certainty })),
        modified: assumptionProps.map(p => ({ subject:p.subject, relation:p.relation, object:p.object, polarity:p.polarity, certainty:p.certainty })),
        comparisonRequired: true,
        factualAuthority: false,
        persistenceAllowed: false,
        createdAt: Date.now(),
        allowsEdge(edge) {
          if (!negated || !disabledConcepts.length) return true;
          const hay = normalize(`${edge?.from || ''} ${edge?.to || ''}`);
          return !disabledConcepts.some(term => term && hay.includes(term));
        }
      };
      this.last = { active: true, id: workspace.id, assumptionText, negated, disabledConcepts, contentWords: workspace.contentWords, baselineCount: workspace.baseline.length, modifiedCount: workspace.modified.length, comparisonRequired: true };
      return workspace;
    }

    tagPlan(plan, workspace) {
      if (!workspace?.active || !plan) return plan;
      return {
        ...plan,
        counterfactual: true,
        counterfactualId: workspace.id,
        clauses: (plan.clauses || []).map(clause => ({ ...clause, counterfactual: true, factualAuthority: false }))
      };
    }

    status() {
      return { ready: true, role: 'ephemeral-counterfactual-reasoning-workspace', sessions: this.sessions, writesFactualMemory: false, last: this.last };
    }
  }

  globalThis.VilotCounterfactualWorkspace = VilotCounterfactualWorkspace;
})();
