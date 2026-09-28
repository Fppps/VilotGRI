/**
 * VilotNI 2.5 - UncertaintyState.js
 * System 34 foundation capacity expansion.
 *
 * Uncertainty is tracked as a vector of causes instead of one opaque scalar.
 * The class can score a whole response context, individual propositions, and
 * learned knowledge edges. Confidence is only capped by uncertainty; this
 * module never raises factual confidence.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));
  const mean = rows => rows?.length ? rows.reduce((sum, value) => sum + Number(value || 0), 0) / rows.length : 0;
  const norm = value => String(value || '').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g, ' ').replace(/\s+/g, ' ').trim();

  class VilotUncertaintyState {
    constructor(evaluator = null, options = {}, ruleStore = null, temporalKnowledge = null) {
      this.evaluator = evaluator;
      this.ruleStore = ruleStore;
      this.temporalKnowledge = temporalKnowledge;
      this.options = {
        minConfidenceCeiling: clamp(options.uncertaintyMinConfidenceCeiling ?? 0.55),
        contradictionPenalty: clamp(options.uncertaintyContradictionPenalty ?? 0.22),
        reasoningPenalty: clamp(options.uncertaintyReasoningPenalty ?? 0.08),
        propositionFloor: clamp(options.uncertaintyPropositionFloor ?? 0.10),
        maxTrackedPropositions: Math.max(8, Math.min(96, options.uncertaintyMaxTrackedPropositions | 0 || 48))
      };
      this.assessments = 0;
      this.propositionAssessments = 0;
      this.last = null;
      this.lastPropositions = [];
    }

    _rules() {
      return this.ruleStore?.uncertaintyRules?.() || this.ruleStore?.get?.('uncertaintyRules') || {};
    }

    _structuredRules() {
      return this.ruleStore?.structuredReasoningRules?.() || {};
    }

    _statusPrior(status) {
      const table = this._rules()?.Epistemic_Status || {};
      const key = norm(status || 'unknown').replace(/\s+/g, '_');
      const value = table[key];
      return Number.isFinite(Number(value)) ? clamp(value) : 0.55;
    }

    _referenceMetrics(context = {}) {
      const analysis = context.analysis || {};
      const complex = analysis.complexLanguage || {};
      const graph = analysis.propositionGraph || context.propositionGraph || {};
      const refs = complex.references || analysis.referenceLinks || [];
      const unresolved = complex.unresolved || graph.unresolved || [];
      if (!refs.length && !unresolved.length) return { coverage: 1, ambiguity: 0, count: 0, unresolved: 0 };
      const resolved = refs.filter(row => row?.resolved || row?.antecedent).length;
      const candidateAmbiguity = refs.map(row => {
        const candidates = Array.isArray(row?.candidates) ? row.candidates : [];
        if (candidates.length <= 1) return 0;
        const scores = candidates.map(c => Number(c?.score ?? c?.confidence ?? 0)).filter(Number.isFinite).sort((a, b) => b - a);
        if (scores.length < 2) return 0.35;
        return clamp(1 - Math.max(0, scores[0] - scores[1]) * 2.5);
      });
      const coverage = refs.length ? clamp(resolved / refs.length) : clamp(1 - unresolved.length * 0.18);
      return {
        coverage,
        ambiguity: clamp(Math.max(unresolved.length ? 0.35 : 0, mean(candidateAmbiguity))),
        count: refs.length,
        unresolved: unresolved.length
      };
    }

    _propositionMetrics(context = {}) {
      const graph = context.propositionGraph || context.analysis?.propositionGraph || context.plan?.propositionGraph || null;
      if (!graph) return { coverage: clamp(context.verification?.propositionCoverage ?? 0.75), certainty: 0.65, count: 0, inferenceDepth: 0 };
      const props = Array.isArray(graph.propositions) ? graph.propositions : [];
      const complete = props.map(p => [p?.subject, p?.relation, p?.object].filter(Boolean).length / 3);
      const certainty = props.map(p => clamp(p?.certainty ?? 0.62));
      const deps = Array.isArray(graph.dependencies) ? graph.dependencies : [];
      const inferred = props.filter(p => /infer|derive|hypoth|predict|general/i.test(String(p?.source || p?.epistemicStatus || ''))).length;
      return {
        coverage: clamp((graph.metrics?.propositionCompleteness ?? mean(complete)) || 0.75),
        certainty: clamp(mean(certainty) || 0.65),
        count: props.length,
        inferenceDepth: props.length ? clamp((deps.length + inferred) / Math.max(1, props.length * 2)) : 0
      };
    }

    _mechanismMetrics(context = {}) {
      const graph = context.analysis?.mechanismGraph || context.mechanismGraph || context.plan?.mechanismGraph || null;
      if (!graph?.active) return { active: false, completeness: 1, missing: [] };
      return {
        active: true,
        completeness: clamp(graph.completeness ?? graph.metrics?.completeness ?? 0.5),
        missing: Array.isArray(graph.missing) ? graph.missing.slice(0, 12) : []
      };
    }

    _temporalMetrics(context = {}) {
      const graph = context.propositionGraph || context.analysis?.propositionGraph || context.plan?.propositionGraph || null;
      const direct = context.temporalAssessment || this.temporalKnowledge?.analyzePropositionGraph?.(graph) || null;
      if (!direct) return { consistency: clamp(context.verification?.temporalConsistency ?? 0.88), ambiguity: 0.08, contradictions: 0 };
      return {
        consistency: clamp(direct.consistency ?? 1),
        ambiguity: clamp(direct.ambiguity ?? 0),
        contradictions: Number(direct.contradictions?.length || direct.contradictionCount || 0)
      };
    }

    _weightedReliability(dimensions = {}) {
      const configured = this._rules()?.Dimensions || {};
      const fallbackWeights = {
        evidenceCoverage: 0.15, semanticAgreement: 0.11, branchAgreement: 0.10,
        contradictionSafety: 0.12, knowledgeCertainty: 0.08, calibrationReliability: 0.08,
        freshness: 0.06, referenceResolution: 0.08, temporalConsistency: 0.06,
        propositionCoverage: 0.08, mechanismCompleteness: 0.05, driftSafety: 0.03
      };
      let weighted = 0;
      let total = 0;
      for (const [name, fallback] of Object.entries(fallbackWeights)) {
        const weight = clamp(configured?.[name]?.weight ?? fallback, 0, 1);
        if (!weight) continue;
        weighted += clamp(dimensions[name] ?? 0.5) * weight;
        total += weight;
      }
      return clamp(total ? weighted / total : 0.5);
    }

    _stateFrom(result) {
      if (result.contradictionPressure > 0.30) return 'contested';
      if (result.referenceResolution < 0.58) return 'reference-ambiguous';
      if (result.temporalConsistency < 0.58) return 'temporally-ambiguous';
      if (result.evidenceCoverage < 0.42) return 'insufficient-evidence';
      if (result.freshness < 0.62) return 'stale';
      if (result.reasoningDependence > 0.65) return 'inference-heavy';
      if (result.reliability >= 0.84) return 'strong';
      if (result.reliability >= 0.68) return 'moderate';
      return 'uncertain';
    }

    assess(context = {}) {
      const branches = context.branches || [];
      const plan = context.plan || {};
      const verification = context.verification || {};
      const knowledgeRows = context.knowledge?.rows || [];
      const learnedRows = context.learnedKnowledge?.rows || context.knowledge?.learnedKnowledge?.rows || [];
      const totalRows = knowledgeRows.length + learnedRows.length;

      const branchConfidences = branches.map(b => (Number(b?.confidence) || 0) / 100).filter(Number.isFinite);
      let branchAgreement = 0.5;
      if (branchConfidences.length) {
        const avg = mean(branchConfidences);
        const variance = mean(branchConfidences.map(v => (v - avg) ** 2));
        branchAgreement = clamp(1 - Math.sqrt(variance) * 3.2);
      }

      const evidenceCoverage = clamp(
        verification.totalClauses
          ? verification.verifiedClauses / Math.max(1, verification.totalClauses)
          : verification.evidenceCoverage ?? (totalRows ? 0.72 : 0.18)
      );
      const semanticAgreement = clamp(plan.agreementRatio ?? context.composition?.diagnostics?.semanticAgreement ?? 0.5);
      const contradictionPressure = clamp(verification.meanContradictionPressure ?? context.propositionContradictions?.pressure ?? 0);
      const contradictionSafety = clamp(1 - contradictionPressure);
      const trustValues = [...knowledgeRows, ...learnedRows]
        .map(row => Number(row?.entry?.Trust ?? row?.edge?.confidence ?? row?.confidence))
        .filter(Number.isFinite)
        .map(clamp);
      const knowledgeCertainty = trustValues.length ? clamp(mean(trustValues)) : 0.28;
      const knowledgeUncertainty = clamp(1 - knowledgeCertainty);
      const reasoningDependence = clamp(
        learnedRows.length
          ? learnedRows.filter(row => row.reasoned || row.derived || row?.edge?.provenanceWeights?.generalization).length / learnedRows.length
          : 0
      );
      const calibrationReliability = clamp(1 - (Number(this.evaluator?.calibrationError) || 0.18));
      const outOfDomain = clamp(totalRows ? 0.12 + (1 - evidenceCoverage) * 0.30 : 0.85);
      const freshnessValues = [...knowledgeRows, ...learnedRows].map(row => this.temporalKnowledge?.assess?.(row?.edge || row?.entry || row)?.freshness).filter(Number.isFinite);
      const freshness = clamp(verification.meanFreshness ?? (freshnessValues.length ? mean(freshnessValues) : 0.90));
      const hypothetical = Boolean(context.counterfactual?.active);

      const refs = this._referenceMetrics(context);
      const propositions = this._propositionMetrics(context);
      const mechanism = this._mechanismMetrics(context);
      const temporal = this._temporalMetrics(context);
      const driftSafety = clamp(context.reasoningQuality?.driftSafety ?? verification.driftSafety ?? 0.82);
      const multiPartCoverage = clamp(context.reasoningQuality?.multiPartCoverage ?? verification.multiPartCoverage ?? 0.82);

      const dimensions = {
        evidenceCoverage,
        semanticAgreement,
        branchAgreement,
        contradictionSafety,
        knowledgeCertainty,
        calibrationReliability,
        freshness,
        referenceResolution: refs.coverage,
        temporalConsistency: temporal.consistency,
        propositionCoverage: propositions.coverage,
        mechanismCompleteness: mechanism.completeness,
        driftSafety
      };

      let reliability = this._weightedReliability(dimensions);
      reliability = clamp(
        reliability
        - reasoningDependence * this.options.reasoningPenalty
        - refs.ambiguity * 0.05
        - temporal.ambiguity * 0.04
        - (hypothetical ? 0.025 : 0)
      );

      const ceiling = clamp(Math.max(this.options.minConfidenceCeiling, reliability));
      const result = {
        ...dimensions,
        contradictionPressure,
        knowledgeUncertainty,
        outOfDomain,
        reasoningDependence,
        hypothetical,
        referenceAmbiguity: refs.ambiguity,
        unresolvedReferences: refs.unresolved,
        temporalAmbiguity: temporal.ambiguity,
        temporalContradictions: temporal.contradictions,
        propositionCertainty: propositions.certainty,
        propositionCount: propositions.count,
        inferenceDepth: propositions.inferenceDepth,
        mechanismActive: mechanism.active,
        mechanismMissing: mechanism.missing,
        multiPartCoverage,
        reliability,
        confidenceCeilingPct: ceiling * 100
      };
      result.state = this._stateFrom(result);
      result.reasons = this._reasons(result);

      this.assessments++;
      this.last = result;
      return result;
    }

    _reasons(result) {
      const reasons = [];
      const dims = this._rules()?.Dimensions || {};
      const checks = [
        ['evidenceCoverage', 'evidence-coverage'], ['semanticAgreement', 'semantic-disagreement'], ['branchAgreement', 'branch-disagreement'],
        ['contradictionSafety', 'contradiction-pressure'], ['knowledgeCertainty', 'weak-knowledge-support'], ['calibrationReliability', 'calibration-gap'],
        ['freshness', 'stale-evidence'], ['referenceResolution', 'reference-ambiguity'], ['temporalConsistency', 'temporal-ambiguity'],
        ['propositionCoverage', 'proposition-gap'], ['mechanismCompleteness', 'mechanism-gap'], ['driftSafety', 'semantic-drift']
      ];
      for (const [metric, reason] of checks) {
        const good = clamp(dims?.[metric]?.good ?? 0.72);
        if (clamp(result?.[metric]) < good) reasons.push({ reason, metric, value: clamp(result?.[metric]), target: good, deficit: good - clamp(result?.[metric]) });
      }
      return reasons.sort((a, b) => b.deficit - a.deficit).slice(0, 8);
    }

    assessPropositions(graph, context = {}) {
      const props = Array.isArray(graph?.propositions) ? graph.propositions.slice(0, this.options.maxTrackedPropositions) : [];
      const penalties = this._rules()?.Proposition_Penalties || {};
      const dependencies = Array.isArray(graph?.dependencies) ? graph.dependencies : [];
      const unresolved = new Set((graph?.unresolved || []).map(norm));
      const temporal = this.temporalKnowledge?.analyzePropositionGraph?.(graph) || null;
      const temporalByProp = temporal?.propositionStates || {};
      const rows = props.map((p, index) => {
        let confidence = clamp(p?.certainty ?? this._statusPrior(p?.epistemicStatus || p?.source));
        const reasons = [];
        const penalize = (name, fallback) => {
          const amount = clamp(penalties?.[name] ?? fallback, 0, 0.8);
          confidence *= (1 - amount);
          reasons.push({ type: name, penalty: amount });
        };
        if (!p?.subject) penalize('missing_subject', 0.18);
        if (!p?.relation) penalize('missing_relation', 0.22);
        if (!p?.object) penalize('missing_object', 0.16);
        const terms = [p?.subject, p?.object].map(norm).filter(Boolean);
        if (terms.some(term => unresolved.has(term))) penalize('unresolved_reference', 0.24);
        const temporalState = temporalByProp[p?.id] || null;
        if (temporalState?.ambiguous) penalize('temporal_ambiguity', 0.14);
        if (temporalState?.contradicted) penalize('contradiction', 0.28);
        const incoming = dependencies.filter(d => d?.from === p?.id || d?.to === p?.id).length;
        if (incoming > 1 && /infer|derive|reason/i.test(String(p?.source || ''))) {
          const each = clamp(penalties?.inference_depth_each ?? 0.055, 0, 0.2);
          confidence *= Math.pow(1 - each, Math.min(6, incoming - 1));
          reasons.push({ type: 'inference_depth', depth: incoming, penaltyEach: each });
        }
        confidence = clamp(Math.max(this.options.propositionFloor, confidence));
        return {
          id: p?.id || `p-${index + 1}`,
          subject: p?.subject || '', relation: p?.relation || '', object: p?.object || '',
          confidence, uncertainty: clamp(1 - confidence), reasons,
          epistemicStatus: p?.epistemicStatus || p?.source || 'unknown'
        };
      });
      this.propositionAssessments += rows.length;
      this.lastPropositions = rows;
      return rows;
    }

    capConfidence(confidencePct, state) {
      const raw = Math.max(0, Number(confidencePct) || 0);
      const cap = Math.max(0, Number(state?.confidenceCeilingPct) || 100);
      return Math.min(raw, cap);
    }

    assessKnowledgeTarget(edge) {
      if (!edge) return { state: 'unknown', uncertainty: 1, contradictionPressure: 0, verificationLevel: 0, freshness: 0.5, inferenceDependence: 1 };
      const contradictionPressure = clamp(edge.contradictionPressure ?? 0);
      const temporal = this.temporalKnowledge?.assess?.(edge) || { freshness: 0.9, active: true };
      const status = edge.status === 'consolidated' ? 'known' : edge.status === 'provisional' ? 'observed' : edge.status || 'observed';
      const prior = this._statusPrior(status);
      const rawConfidence = clamp(edge.confidence ?? prior);
      const verificationLevel = edge.status === 'consolidated' ? clamp(edge.trustedEvidence ?? rawConfidence) : edge.status === 'provisional' ? 0.35 : 0.2;
      const inferenceDependence = edge.provenanceWeights?.generalization ? 0.7 : edge.provenanceWeights?.cognitive_inference ? 0.8 : edge.reasoned ? 0.65 : 0.2;
      const uncertainty = clamp(1 - rawConfidence * prior * (1 - contradictionPressure) * clamp(temporal.freshness ?? 0.9));
      return {
        state: edge.status === 'contested' ? 'contested' : status,
        uncertainty,
        contradictionPressure,
        verificationLevel,
        freshness: clamp(temporal.freshness ?? 0.9),
        inferenceDependence,
        active: temporal.active !== false
      };
    }

    status() {
      return {
        ready: true,
        role: 'proposition-localized-multi-dimensional-epistemic-uncertainty-state',
        assessments: this.assessments,
        propositionAssessments: this.propositionAssessments,
        ruleDriven: Boolean(this._rules()?.Schema_Version),
        last: this.last,
        lastPropositions: this.lastPropositions.slice(0, 12)
      };
    }
  }


  // System 34 deep uncertainty extension. Uncertainty is localized to the exact
  // linguistic or reasoning dependency that introduced it, then propagated.
  const V34_UNCERTAINTY_DIMENSIONS = Object.freeze([
    { id:'token_identity', family:'surface', weight:0.45, localize:true, propagate:true },
    { id:'word_sense', family:'semantic', weight:0.72, localize:true, propagate:true },
    { id:'entity_resolution', family:'semantic', weight:0.78, localize:true, propagate:true },
    { id:'reference_resolution', family:'reference', weight:0.96, localize:true, propagate:true },
    { id:'reference_margin', family:'reference', weight:0.82, localize:true, propagate:true },
    { id:'clause_attachment', family:'syntax', weight:0.82, localize:true, propagate:true },
    { id:'negation_scope', family:'syntax', weight:0.92, localize:true, propagate:true },
    { id:'condition_scope', family:'syntax', weight:0.90, localize:true, propagate:true },
    { id:'proposition_subject', family:'proposition', weight:0.88, localize:true, propagate:true },
    { id:'proposition_relation', family:'proposition', weight:0.94, localize:true, propagate:true },
    { id:'proposition_object', family:'proposition', weight:0.82, localize:true, propagate:true },
    { id:'proposition_dependency', family:'proposition', weight:0.90, localize:true, propagate:true },
    { id:'relation_direction', family:'reasoning', weight:0.94, localize:true, propagate:true },
    { id:'causal_direction', family:'reasoning', weight:0.98, localize:true, propagate:true },
    { id:'mechanism_stage', family:'mechanism', weight:0.92, localize:true, propagate:true },
    { id:'mechanism_order', family:'mechanism', weight:0.94, localize:true, propagate:true },
    { id:'mechanism_missing_stage', family:'mechanism', weight:0.86, localize:true, propagate:true },
    { id:'temporal_identity', family:'temporal', weight:0.88, localize:true, propagate:true },
    { id:'temporal_order', family:'temporal', weight:0.96, localize:true, propagate:true },
    { id:'temporal_overlap', family:'temporal', weight:0.82, localize:true, propagate:true },
    { id:'temporal_duration', family:'temporal', weight:0.78, localize:true, propagate:true },
    { id:'temporal_frequency', family:'temporal', weight:0.74, localize:true, propagate:true },
    { id:'evidence_coverage', family:'evidence', weight:0.94, localize:true, propagate:true },
    { id:'evidence_independence', family:'evidence', weight:0.72, localize:true, propagate:true },
    { id:'source_reliability', family:'evidence', weight:0.90, localize:true, propagate:true },
    { id:'freshness', family:'evidence', weight:0.76, localize:true, propagate:true },
    { id:'branch_agreement', family:'cognition', weight:0.86, localize:true, propagate:true },
    { id:'branch_diversity', family:'cognition', weight:0.62, localize:true, propagate:true },
    { id:'candidate_margin', family:'cognition', weight:0.82, localize:true, propagate:true },
    { id:'cognitive_progress', family:'cognition', weight:0.74, localize:true, propagate:true },
    { id:'contradiction_pressure', family:'verification', weight:0.98, localize:true, propagate:true },
    { id:'claim_support', family:'verification', weight:0.96, localize:true, propagate:true },
    { id:'multi_part_coverage', family:'planning', weight:0.92, localize:true, propagate:true },
    { id:'semantic_drift', family:'verification', weight:0.96, localize:true, propagate:true },
    { id:'surface_integrity', family:'generation', weight:0.68, localize:true, propagate:true },
    { id:'terminal_integrity', family:'generation', weight:0.72, localize:true, propagate:true },
    { id:'calibration_history', family:'calibration', weight:0.88, localize:true, propagate:true },
    { id:'out_of_domain', family:'calibration', weight:0.82, localize:true, propagate:true },
    { id:'counterfactual_distance', family:'reasoning', weight:0.74, localize:true, propagate:true },
    { id:'hypothesis_depth', family:'reasoning', weight:0.72, localize:true, propagate:true },
  ]);

  const v34UClamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(Number(value)) ? Number(value) : 0));
  const v34UNorm = value => String(value ?? '').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const v34UArray = value => Array.isArray(value) ? value : [];
  const v34UObj = value => value && typeof value === 'object' ? value : {};
  const v34UMean = rows => rows.length ? rows.reduce((sum, value) => sum + Number(value || 0), 0) / rows.length : 0;

  class V34UncertaintyEngine {
    constructor(owner) {
      this.owner = owner;
      this.history = [];
      this.calibration = new Map();
      this.last = null;
    }

    dimensionSpec(id) {
      return V34_UNCERTAINTY_DIMENSIONS.find(row => row.id === id) || { id, family:'general', weight:0.5, localize:true, propagate:true };
    }

    normalizeProbability(value, fallback = 0.5) {
      const number = Number(value);
      if (!Number.isFinite(number)) return v34UClamp(fallback);
      if (number > 1 && number <= 100) return v34UClamp(number / 100);
      return v34UClamp(number);
    }

    confidenceFromMargin(first, second) {
      const a = this.normalizeProbability(first, 0);
      const b = this.normalizeProbability(second, 0);
      return v34UClamp(0.5 + Math.max(0, a - b) * 0.7);
    }

    entropy(distribution = []) {
      const values = v34UArray(distribution).map(row => this.normalizeProbability(typeof row === 'number' ? row : row?.score, 0)).filter(x => x > 0);
      const total = values.reduce((a,b) => a+b, 0);
      if (!total || values.length <= 1) return 0;
      let h = 0;
      for (const value of values) {
        const p = value / total;
        h -= p * Math.log2(Math.max(1e-12, p));
      }
      return v34UClamp(h / Math.log2(values.length));
    }

    referenceReport(context = {}) {
      const refs = v34UArray(context.referenceLinks || context.references || context.complexLanguage?.references || context.analysis?.referenceLinks);
      const items = refs.map((ref, index) => {
        const candidates = v34UArray(ref.candidates || ref.hypotheses || ref.alternatives);
        const scores = candidates.map(c => this.normalizeProbability(c.score ?? c.confidence ?? c.weight, 0));
        const sorted = scores.slice().sort((a,b)=>b-a);
        const explicit = this.normalizeProbability(ref.confidence ?? ref.score, ref.resolved ? 0.9 : 0.45);
        const ambiguity = candidates.length > 1 ? this.entropy(scores) : (ref.resolved ? 0.08 : 0.75);
        const marginConfidence = sorted.length > 1 ? this.confidenceFromMargin(sorted[0], sorted[1]) : explicit;
        const confidence = v34UClamp(explicit * 0.55 + marginConfidence * 0.25 + (1 - ambiguity) * 0.20);
        return {
          id: ref.id || `reference:${index}`,
          surface: String(ref.surface || ref.text || ref.reference || '').slice(0,120),
          resolved: Boolean(ref.resolved || ref.antecedent || ref.target),
          confidence,
          ambiguity,
          candidateCount: candidates.length,
          selected: ref.antecedent || ref.target || candidates[0]?.id || null,
          dependencies: v34UArray(ref.dependencies).slice(0,12),
          reason: !ref.resolved ? 'unresolved-reference' : ambiguity > 0.55 ? 'ambiguous-reference' : null
        };
      });
      const confidence = items.length ? v34UMean(items.map(x => x.confidence)) : 1;
      return { family:'reference', items, confidence, uncertainty:1-confidence, unresolved:items.filter(x=>!x.resolved).length, ambiguous:items.filter(x=>x.ambiguity>0.55).length };
    }

    propositionReport(context = {}) {
      const graph = context.propositionGraph || context.analysis?.propositionGraph || {};
      const propositions = v34UArray(graph.propositions || context.propositions);
      const items = propositions.map((p,index) => {
        const subject = String(p.subject || p.agent || '').trim();
        const relation = String(p.relation || p.predicate || '').trim();
        const object = String(p.object || p.patient || p.value || '').trim();
        const base = this.normalizeProbability(p.confidence ?? p.certainty, 0.76);
        const completeness = (Number(Boolean(subject)) + Number(Boolean(relation)) + Number(Boolean(object))) / 3;
        const sourceSupport = this.normalizeProbability(p.support ?? p.evidenceConfidence, p.source ? 0.78 : 0.60);
        const refPenalty = v34UArray(p.references).some(r => r?.resolved === false) ? 0.18 : 0;
        const conditionPenalty = p.condition && p.conditionResolved === false ? 0.12 : 0;
        const confidence = v34UClamp(base * 0.46 + completeness * 0.24 + sourceSupport * 0.30 - refPenalty - conditionPenalty);
        return {
          id:p.id || `proposition:${index}`, subject, relation, object,
          confidence, completeness, sourceSupport,
          dependencies:v34UArray(p.dependencies || p.dependsOn).slice(0,16),
          polarity:p.polarity === false ? 'negative' : 'positive',
          hypothetical:Boolean(p.hypothetical),
          temporalScope:p.time || p.temporal || null
        };
      });
      const confidence = items.length ? v34UMean(items.map(x=>x.confidence)) : 1;
      return { family:'proposition', items, confidence, uncertainty:1-confidence, incomplete:items.filter(x=>x.completeness<0.67).length };
    }

    mechanismReport(context = {}) {
      const graph = context.mechanismGraph || context.analysis?.mechanismGraph || {};
      const stages = v34UArray(graph.stages);
      const items = stages.map((stage,index) => {
        const explicit = this.normalizeProbability(stage.confidence ?? stage.score, 0.72);
        const supported = this.normalizeProbability(stage.support ?? stage.evidenceConfidence, stage.evidence ? 0.78 : 0.60);
        const hasAction = Boolean(stage.operation || stage.action || stage.relation || stage.label);
        const hasInput = Boolean(stage.input || stage.from || stage.precondition || index === 0);
        const hasOutput = Boolean(stage.output || stage.to || index === stages.length-1);
        const structural = (Number(hasAction)+Number(hasInput)+Number(hasOutput))/3;
        const confidence = v34UClamp(explicit*0.48 + supported*0.30 + structural*0.22);
        return { id:stage.id || `stage:${index}`, index, confidence, structural, supported, missing:[!hasAction?'operation':null,!hasInput?'input':null,!hasOutput?'output':null].filter(Boolean) };
      });
      const declared = this.normalizeProbability(graph.completeness, stages.length ? 0.68 : 1);
      const stageConfidence = items.length ? v34UMean(items.map(x=>x.confidence)) : declared;
      const confidence = v34UClamp(declared*0.45 + stageConfidence*0.55);
      return { family:'mechanism', items, confidence, uncertainty:1-confidence, missing:v34UArray(graph.missing), active:Boolean(graph.active || stages.length) };
    }

    temporalReport(context = {}) {
      const assessment = context.temporalAssessment || context.analysis?.temporalAssessment || context.temporal || {};
      const relations = v34UArray(assessment.relations);
      const contradictions = v34UArray(assessment.contradictions);
      const ambiguity = this.normalizeProbability(assessment.ambiguity, relations.length ? 0.15 : 0.35);
      const consistency = this.normalizeProbability(assessment.consistency, contradictions.length ? 0.45 : 0.9);
      const relationItems = relations.map((row,index)=>({
        id:row.id || `temporal:${index}`, from:row.from || row.left || null, to:row.to || row.right || null,
        relation:v34UNorm(row.relation || row.type), confidence:this.normalizeProbability(row.confidence,0.72), inferred:Boolean(row.inferred)
      }));
      const relationConfidence = relationItems.length ? v34UMean(relationItems.map(x=>x.confidence)) : consistency;
      const confidence = v34UClamp(consistency*0.55 + relationConfidence*0.25 + (1-ambiguity)*0.20 - Math.min(0.35, contradictions.length*0.10));
      return { family:'temporal', items:relationItems, confidence, uncertainty:1-confidence, ambiguity, consistency, contradictions:contradictions.slice(0,24) };
    }

    evidenceReport(context = {}) {
      const evidence = v34UArray(context.evidence || context.analysis?.evidence || context.cognitivePacket?.evidence);
      const items = evidence.map((row,index)=>{
        const support = this.normalizeProbability(row.confidence ?? row.score ?? row.trust, 0.62);
        const freshness = this.normalizeProbability(row.freshness, 0.86);
        const direct = row.direct === true || v34UNorm(row.kind).includes('direct');
        const independence = this.normalizeProbability(row.independence, direct ? 0.82 : 0.62);
        return { id:row.id || `evidence:${index}`, support, freshness, independence, direct, source:row.source || null, confidence:v34UClamp(support*0.55+freshness*0.20+independence*0.25) };
      });
      const confidence = items.length ? v34UMean(items.map(x=>x.confidence)) : 0.58;
      return { family:'evidence', items, confidence, uncertainty:1-confidence, direct:items.filter(x=>x.direct).length };
    }

    branchReport(context = {}) {
      const branches = v34UArray(context.branches || context.reasoningBranches || context.cognitiveState?.branches || context.analysis?.branches);
      if (!branches.length) return { family:'branches', items:[], confidence:0.7, uncertainty:0.3, disagreement:0.3 };
      const items = branches.map((row,index)=>({
        id:row.id || `branch:${index}`, confidence:this.normalizeProbability(row.confidence ?? row.score ?? row.amplitude,0.62),
        conclusion:v34UNorm(row.conclusion || row.answer || row.text || '').slice(0,180),
        relation:v34UNorm(row.relation || row.mode || row.type)
      }));
      const mean=v34UMean(items.map(x=>x.confidence));
      const spread=Math.sqrt(v34UMean(items.map(x=>(x.confidence-mean)*(x.confidence-mean))));
      const uniqueConclusions=new Set(items.map(x=>x.conclusion).filter(Boolean)).size;
      const disagreement=v34UClamp(spread*1.8 + (uniqueConclusions>1 ? Math.min(0.55,(uniqueConclusions-1)*0.12):0));
      return { family:'branches', items, confidence:v34UClamp(mean*(1-disagreement*0.35)), uncertainty:v34UClamp(1-mean+disagreement*0.35), disagreement };
    }

    dependencyGraph(reports = {}) {
      const nodes=new Map();
      const edges=[];
      const add=(id, family, confidence, meta={})=>{
        if(!id) return;
        nodes.set(id,{id,family,confidence:this.normalizeProbability(confidence,0.5),...meta});
      };
      for(const item of reports.references?.items||[]) add(item.id,'reference',item.confidence,item);
      for(const item of reports.propositions?.items||[]) add(item.id,'proposition',item.confidence,item);
      for(const item of reports.mechanisms?.items||[]) add(item.id,'mechanism',item.confidence,item);
      for(const item of reports.temporal?.items||[]) add(item.id,'temporal',item.confidence,item);
      for(const p of reports.propositions?.items||[]) {
        for(const dep of p.dependencies||[]) {
          const target=typeof dep==='string'?dep:dep?.id;
          if(target) edges.push({from:target,to:p.id,type:'proposition-dependency',discount:0.90});
        }
      }
      return {nodes,edges};
    }

    propagate(graph) {
      const nodes=new Map(Array.from(graph.nodes.entries()).map(([k,v])=>[k,{...v}]));
      const edges=v34UArray(graph.edges);
      for(let pass=0;pass<5;pass++) {
        let changed=false;
        for(const edge of edges) {
          const from=nodes.get(edge.from), to=nodes.get(edge.to);
          if(!from||!to) continue;
          const next=Math.min(to.confidence, from.confidence*this.normalizeProbability(edge.discount,0.9));
          if(next < to.confidence-0.002) { to.confidence=next; changed=true; }
        }
        if(!changed) break;
      }
      return {nodes,edges};
    }

    calibrate(label, predicted, observed) {
      const key=v34UNorm(label)||'general';
      const row=this.calibration.get(key)||{n:0,predicted:0,observed:0,error:0};
      const p=this.normalizeProbability(predicted,0.5), o=this.normalizeProbability(observed,0.5);
      row.n++; row.predicted += (p-row.predicted)/row.n; row.observed += (o-row.observed)/row.n; row.error += (Math.abs(p-o)-row.error)/row.n;
      this.calibration.set(key,row);
      return {...row};
    }

    calibrationPenalty(label) {
      const row=this.calibration.get(v34UNorm(label)||'general');
      if(!row||row.n<4) return 0;
      return v34UClamp(row.error*0.55,0,0.22);
    }

    summarize(context={}) {
      const reports={
        references:this.referenceReport(context), propositions:this.propositionReport(context), mechanisms:this.mechanismReport(context),
        temporal:this.temporalReport(context), evidence:this.evidenceReport(context), branches:this.branchReport(context)
      };
      const graph=this.propagate(this.dependencyGraph(reports));
      const families=Object.values(reports);
      const weighted=[];
      const familyWeights={references:0.18,propositions:0.19,mechanisms:0.16,temporal:0.13,evidence:0.18,branches:0.16};
      for(const [name,row] of Object.entries(reports)) weighted.push([row.confidence,familyWeights[name]||0.1]);
      const denom=weighted.reduce((s,r)=>s+r[1],0)||1;
      let confidence=weighted.reduce((s,r)=>s+r[0]*r[1],0)/denom;
      const contradictionCount=v34UArray(context.contradictions||context.analysis?.contradictions||context.temporalAssessment?.contradictions).length;
      confidence=v34UClamp(confidence-Math.min(0.28,contradictionCount*0.06)-this.calibrationPenalty('overall'));
      const localized=[];
      for(const [family,row] of Object.entries(reports)) {
        if(row.uncertainty>0.25) localized.push({family,uncertainty:row.uncertainty,confidence:row.confidence});
        for(const item of row.items||[]) if(Number(item.confidence)<0.62) localized.push({family,id:item.id,uncertainty:1-Number(item.confidence),reason:item.reason||'low-local-confidence'});
      }
      localized.sort((a,b)=>b.uncertainty-a.uncertainty);
      const result={confidence,uncertainty:1-confidence,reports,localized:localized.slice(0,32),dependencyNodes:Array.from(graph.nodes.values()).slice(0,128)};
      this.last=result; this.history.push({at:Date.now(),confidence,localized:result.localized.slice(0,8)}); if(this.history.length>128)this.history.shift();
      return result;
    }
  }

  const v34OriginalUncertaintyAssess = VilotUncertaintyState.prototype.assess;
  VilotUncertaintyState.prototype.assess = function(context = {}) {
    const base = v34OriginalUncertaintyAssess.call(this, context) || {};
    this.v34Engine ||= new V34UncertaintyEngine(this);
    const deep = this.v34Engine.summarize(context);
    const baseReliability = v34UClamp(base.reliability ?? base.confidence ?? 0.72);
    const combined = v34UClamp(baseReliability * 0.46 + deep.confidence * 0.54);
    const result = { ...base, reliability:combined, confidence:combined, uncertainty:1-combined, localizedUncertainty:deep.localized, uncertaintyLayers:deep.reports, dependencyUncertainty:deep.dependencyNodes, system34:true };
    this.last = result;
    return result;
  };

  VilotUncertaintyState.prototype.assessReferences = function(context={}) { this.v34Engine ||= new V34UncertaintyEngine(this); return this.v34Engine.referenceReport(context); };
  VilotUncertaintyState.prototype.assessTemporal = function(context={}) { this.v34Engine ||= new V34UncertaintyEngine(this); return this.v34Engine.temporalReport(context); };
  VilotUncertaintyState.prototype.assessMechanismStages = function(context={}) { this.v34Engine ||= new V34UncertaintyEngine(this); return this.v34Engine.mechanismReport(context); };
  VilotUncertaintyState.prototype.assessReasoningBranches = function(context={}) { this.v34Engine ||= new V34UncertaintyEngine(this); return this.v34Engine.branchReport(context); };
  VilotUncertaintyState.prototype.calibrateDimension = function(label,predicted,observed) { this.v34Engine ||= new V34UncertaintyEngine(this); return this.v34Engine.calibrate(label,predicted,observed); };
  VilotUncertaintyState.prototype.uncertaintyDimensions = function() { return V34_UNCERTAINTY_DIMENSIONS.map(row=>({...row})); };



  const V34_UNCERTAINTY_SCENARIOS = Object.freeze([
    {
      id: "reference-scenario-1",
      family: "reference",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "temporal-scenario-2",
      family: "temporal",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "proposition-scenario-3",
      family: "proposition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "mechanism-scenario-4",
      family: "mechanism",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "evidence-scenario-5",
      family: "evidence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "branch-scenario-6",
      family: "branch",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "calibration-scenario-7",
      family: "calibration",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "contradiction-scenario-8",
      family: "contradiction",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "word-sense-scenario-9",
      family: "word-sense",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "scope-scenario-10",
      family: "scope",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "reference-scenario-11",
      family: "reference",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "temporal-scenario-12",
      family: "temporal",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "proposition-scenario-13",
      family: "proposition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "mechanism-scenario-14",
      family: "mechanism",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "evidence-scenario-15",
      family: "evidence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "branch-scenario-16",
      family: "branch",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "calibration-scenario-17",
      family: "calibration",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "contradiction-scenario-18",
      family: "contradiction",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "word-sense-scenario-19",
      family: "word-sense",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "scope-scenario-20",
      family: "scope",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "reference-scenario-21",
      family: "reference",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "temporal-scenario-22",
      family: "temporal",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "proposition-scenario-23",
      family: "proposition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "mechanism-scenario-24",
      family: "mechanism",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "evidence-scenario-25",
      family: "evidence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "branch-scenario-26",
      family: "branch",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "calibration-scenario-27",
      family: "calibration",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "contradiction-scenario-28",
      family: "contradiction",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "word-sense-scenario-29",
      family: "word-sense",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "scope-scenario-30",
      family: "scope",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "reference-scenario-31",
      family: "reference",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "temporal-scenario-32",
      family: "temporal",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "proposition-scenario-33",
      family: "proposition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "mechanism-scenario-34",
      family: "mechanism",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "evidence-scenario-35",
      family: "evidence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "branch-scenario-36",
      family: "branch",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "calibration-scenario-37",
      family: "calibration",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "contradiction-scenario-38",
      family: "contradiction",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "word-sense-scenario-39",
      family: "word-sense",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "scope-scenario-40",
      family: "scope",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "reference-scenario-41",
      family: "reference",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
  ]);
  VilotUncertaintyState.prototype.uncertaintyScenarios = function() { return V34_UNCERTAINTY_SCENARIOS.map(row => ({ ...row })); };

  globalThis.VilotUncertaintyState = VilotUncertaintyState;
})();
