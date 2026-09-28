/**
 * VilotNI 2.5 - LearnedKnowledgeBridge.js
 *
 * Bridges consolidated learned graph knowledge into foreground retrieval and
 * semantic composition. Provisional hypotheses remain diagnostic-only.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));
  const normalize = value =>
    String(value || '').toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9'+.: -]+/g, ' ').replace(/\s+/g, ' ').trim();

  class VilotLearnedKnowledgeBridge {
    constructor(graph, hypotheses, temporalKnowledge, processor, graphReasoner = null, options = {}, policy = null) {
      this.graph = graph;
      this.hypotheses = hypotheses;
      this.temporal = temporalKnowledge;
      this.processor = processor;
      this.reasoner = graphReasoner;
      this.policy = policy || null;
      this.options = {
        resultLimit: Math.max(4, Math.min(48, options.learnedKnowledgeResultLimit | 0 || 12)),
        scanLimit: Math.max(64, Math.min(1024, options.learnedKnowledgeScanLimit | 0 || 256)),
        minConfidence: clamp(options.learnedKnowledgeMinConfidence ?? 0.78),
        minRelevance: clamp(options.learnedKnowledgeMinRelevance ?? 0.18),
        enableReasoning: options.learnedKnowledgeReasoning !== false
      };
      this.searches = 0;
      this.last = null;
    }

    _tokens(value) {
      const rows = this.processor?.tokenize?.(String(value || ''), 64) || [];
      if (rows.length) return Array.from(new Set(rows.filter(x => x?.pos !== 'Punct').map(x => normalize(x?.lower || x?.surface)).filter(Boolean)));
      return Array.from(new Set(normalize(value).split(/\s+/).filter(Boolean)));
    }

    _relevance(edge, terms) {
      if (!terms.length) return 0.25;
      const hay = new Set(this._tokens(`${edge.from} ${edge.relation} ${edge.to}`));
      let exact = 0;
      for (const term of terms) if (hay.has(term)) exact++;
      let assoc = 0;
      for (const term of terms.slice(0, 8)) {
        for (const node of this._tokens(`${edge.from} ${edge.to}`).slice(0, 8)) {
          assoc = Math.max(assoc, Number(this.processor?.associationScore?.(term, node, { bidirectional: true })) || 0);
        }
      }
      return clamp((exact / Math.max(1, terms.length)) * 0.72 + assoc * 0.28);
    }

    _eligible(edge) {
      const authority = this.policy?.edgeAuthority?.(
        edge,
        this.temporal,
        {
          minConfidence: this.options.minConfidence,
          maxContradiction: 0.20
        }
      );
      if (authority) return authority.factual;
      if (!edge || edge.status !== 'consolidated') return false;
      if (edge.confidence < this.options.minConfidence) return false;
      if (edge.contradictionPressure >= 0.20) return false;
      const temporal = this.temporal?.assess?.(edge) || { active: true };
      return temporal.active;
    }

    _entryFromRelation(item, derived = false) {
      const edge = item.edge || item;
      const relation = this.policy?.canonicalRelation?.(edge.relation) || normalize(edge.relation);
      const subject = String(edge.from || 'concept');
      const target = String(edge.to || '');
      const trust = clamp(derived ? edge.confidence * 0.88 : edge.confidence * 0.96, 0, 0.95);
      const entry = {
        ID: derived ? edge.id : `KG:${edge.key}`,
        Subject: subject,
        Aliases: [],
        Trust: trust,
        Source_Type: derived ? 'graph-reasoned' : 'learned-graph-consolidated',
        Decoder_Knowledge_Eligible: true,
        Response_Use: { General: true, Definition: true, How: true, Yes_No: true },
        _verification: derived
          ? { kind: 'graph_reasoning', pathId: edge.id, relation, from: subject, to: target, confidence: edge.confidence }
          : { kind: 'knowledge_graph', key: edge.key, relation, from: subject, to: target, confidence: edge.confidence, status: edge.status }
      };

      const addArray = (field) => { entry[field] = [target]; };
      switch (relation) {
        case 'is a':
        case 'is_a':
        case 'definition':
        case 'identity':
          entry.Description = target;
          entry.Long_Definition = target;
          break;
        case 'purpose': entry.Purpose = target; break;
        case 'mechanism': entry.Mechanism = target; break;
        case 'property': addArray('Properties'); break;
        case 'capability': addArray('Capabilities'); break;
        case 'limitation': addArray('Limitations'); break;
        case 'example': addArray('Examples'); break;
        case 'contrasts with':
        case 'contrasts_with':
        case 'contrast': addArray('Contrasts_With'); break;
        case 'part of':
        case 'part_of': addArray('Part_Of'); break;
        case 'has part':
        case 'has_part': addArray('Has_Parts'); break;
        case 'causes': addArray('Causes'); break;
        case 'effect': addArray('Effects'); break;
        case 'evaluated by':
        case 'evaluated_by': addArray('Evaluation_Criteria'); break;
        case 'failure mode':
        case 'failure_mode': addArray('Failure_Modes'); break;
        case 'tradeoff': addArray('Tradeoffs'); break;
        default: addArray('Related_To'); break;
      }
      return entry;
    }

    search(promptText, analysis = null, plan = null, options = {}) {
      const started = performance.now();
      const deadlineAt = Number.isFinite(Number(options.deadlineAt)) ? Number(options.deadlineAt) : Infinity;
      const limit = Math.max(2, Math.min(this.options.resultLimit, options.limit | 0 || this.options.resultLimit));
      const terms = Array.from(new Set([
        ...this._tokens(promptText),
        ...(analysis?.contentWords || []).map(normalize),
        ...(plan?.focusSeeds || []).map(normalize),
        ...(plan?.subjectWords || []).map(normalize)
      ].filter(Boolean))).slice(0, 24);

      const candidates = new Map();
      for (const term of terms.slice(0, 12)) {
        if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt) break;
        for (const edge of this.graph?.related?.(term, 24) || []) {
          if (this._eligible(edge)) candidates.set(edge.key, edge);
        }
      }
      if (candidates.size < limit) {
        for (const edge of this.graph?.topEdges?.(this.options.scanLimit) || []) {
          if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt) break;
          if (!this._eligible(edge)) continue;
          const relevance = this._relevance(edge, terms);
          if (relevance >= this.options.minRelevance) candidates.set(edge.key, edge);
          if (candidates.size >= this.options.scanLimit) break;
        }
      }

      const scored = Array.from(candidates.values()).map(edge => {
        const relevance = this._relevance(edge, terms);
        const temporal = this.temporal?.assess?.(edge) || { freshness: 1, active: true };
        const sourceTrust = Math.max(0, ...Object.values(edge.provenanceWeights || {}).map(Number).filter(Number.isFinite));
        const score = clamp(
          edge.confidence * 0.38 + relevance * 0.30 + temporal.freshness * 0.12 +
          clamp(edge.usefulness ?? 0.5) * 0.08 + clamp(sourceTrust) * 0.07 +
          (1 - edge.contradictionPressure) * 0.05
        );
        return { edge, relevance, temporal, score };
      }).sort((a, b) => b.score - a.score).slice(0, limit);

      let reasoning = { paths: [], derived: [], diagnostics: null };
      if (this.options.enableReasoning && this.reasoner && (!Number.isFinite(deadlineAt) || performance.now() < deadlineAt)) {
        reasoning = this.reasoner.reason(terms, {
          deadlineAt,
          maxDepth: options.reasoningDepth || 2,
          counterfactualWorkspace: options.counterfactualWorkspace || null
        });
      }

      const rows = scored.map(item => ({
        entry: this._entryFromRelation(item),
        score: item.score * 8,
        learned: true,
        verification: item.edge.key,
        edge: item.edge
      }));

      for (const derived of (reasoning.derived || []).slice(0, Math.max(0, limit - rows.length))) {
        rows.push({
          entry: this._entryFromRelation(derived, true),
          score: derived.confidence * 6.5,
          learned: true,
          reasoned: true,
          verification: derived.id,
          edge: derived
        });
      }

      const lexicalTerms = Array.from(new Set(rows.flatMap(row => this._tokens(`${row.entry.Subject} ${row.edge?.to || ''}`)))).slice(0, 48);
      const hypotheses = this.hypotheses?.candidates?.(8, ['provisional', 'promotable', 'contested']) || [];

      const result = {
        rows,
        edges: scored.map(x => x.edge),
        lexicalTerms,
        hypotheses,
        reasoning,
        diagnostics: {
          bridgeMs: performance.now() - started,
          resultCount: rows.length,
          consolidatedCount: scored.length,
          reasonedCount: reasoning.derived?.length || 0,
          provisionalDiagnosticCount: hypotheses.length,
          responseUsesProvisionalHypotheses: false,
          deadlineReached: Number.isFinite(deadlineAt) && performance.now() >= deadlineAt
        }
      };
      this.searches++;
      this.last = result.diagnostics;
      return result;
    }

    status() {
      return {
        ready: true,
        role: 'consolidated-learned-knowledge-to-response-bridge',
        searches: this.searches,
        provisionalHypothesesAreFactualEvidence: false,
        graphReasoningEnabled: Boolean(this.reasoner),
        last: this.last
      };
    }
  }

  globalThis.VilotLearnedKnowledgeBridge = VilotLearnedKnowledgeBridge;
})();
