/**
 * VilotNI 2.5 - SelfImprovement.js
 * System 34 foundation capacity expansion.
 *
 * Self-improvement now diagnoses stage-specific failures, schedules targeted
 * replay curricula, evaluates target and control behavior, and promotes only
 * bounded strategy-profile changes. Runtime source files are never rewritten.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));
  const bound = (x, lo, hi) => Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : lo));
  const norm = value => String(value || '').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g, ' ').replace(/\s+/g, ' ').trim();

  class VilotSelfImprovement {
    constructor(deps = {}, options = {}) {
      this.evaluator = deps.evaluator || null;
      this.predictionError = deps.predictionError || null;
      this.hypotheses = deps.hypotheses || null;
      this.replay = deps.replay || null;
      this.graph = deps.graph || null;
      this.ruleStore = deps.ruleStore || null;
      this.runtimeOptions = deps.runtimeOptions || options;
      this.options = {
        enabled: options.selfImprovementEnabled !== false,
        proposalEvery: Math.max(4, options.selfImprovementProposalEvery | 0 || 8),
        evaluationWindow: Math.max(4, options.selfImprovementEvaluationWindow | 0 || 10),
        rollbackTolerance: bound(options.selfImprovementRollbackTolerance ?? 0.025, 0.005, 0.15),
        step: bound(options.selfImprovementStep ?? 0.015, 0.002, 0.05),
        alpha: bound(options.selfImprovementEWMAAlpha ?? 0.16, 0.03, 0.5),
        diagnosisHistory: Math.max(32, Math.min(512, options.selfImprovementDiagnosisHistory | 0 || 192)),
        failureReplayThreshold: clamp(options.selfImprovementFailureReplayThreshold ?? 0.08),
        replayCooldownResponses: Math.max(1, options.selfImprovementReplayCooldownResponses | 0 || 2)
      };
      this.profile = this._defaultProfile();
      this.metrics = this._defaultMetrics();
      this.responses = 0;
      this.proposals = 0;
      this.accepted = 0;
      this.rollbacks = 0;
      this.pending = null;
      this.history = [];
      this.diagnoses = [];
      this.diagnosisCounts = {};
      this.cognitiveTraces = 0;
      this.lastCognitiveTrace = null;
      this.lastReplayScheduledAt = -999;
      this.replaySchedules = 0;
      this.dirty = false;
      this.last = null;
      this._publish();
    }

    _policy() {
      const value = this.ruleStore?.selfImprovementPolicy?.() || this.ruleStore?.safeDataset?.('selfImprovementPolicy') || this.ruleStore?.get?.('selfImprovementPolicy') || {};
      const policy = value && typeof value === 'object' ? value : {};
      if (!policy.Diagnosis_Targets || typeof policy.Diagnosis_Targets !== 'object' || Array.isArray(policy.Diagnosis_Targets)) policy.Diagnosis_Targets = {};
      return policy;
    }
    _replayPolicy() {
      const value = this.ruleStore?.replayPolicy?.() || this.ruleStore?.safeDataset?.('replayPolicy') || this.ruleStore?.get?.('replayPolicy') || {};
      const policy = value && typeof value === 'object' ? value : {};
      if (!policy.Curricula || typeof policy.Curricula !== 'object' || Array.isArray(policy.Curricula)) policy.Curricula = {};
      return policy;
    }
    _diagnosisTarget(type) {
      return this.ruleStore?.diagnosisTarget?.(type) || this._policy().Diagnosis_Targets?.[type] || this._policy().Fallback_Diagnosis || { metrics:['reasoning'], profiles:{ reasoningScale:0.1 }, replay:type || 'general' };
    }

    // System 37 runtime-state migration guard. Persisted self-improvement state
    // from older builds may contain a pending experiment without the newer
    // keyed containers (diagnoses, targetSamples, reasons). Dynamic diagnosis
    // names such as "semantic-retrieval" must never index an undefined map.
    _ensureRuntimeState() {
      if (!this.metrics || typeof this.metrics !== 'object' || Array.isArray(this.metrics)) this.metrics = this._defaultMetrics();
      if (!this.profile || typeof this.profile !== 'object' || Array.isArray(this.profile)) this.profile = this._defaultProfile();
      if (!Array.isArray(this.history)) this.history = [];
      if (!Array.isArray(this.diagnoses)) this.diagnoses = [];
      if (!this.diagnosisCounts || typeof this.diagnosisCounts !== 'object' || Array.isArray(this.diagnosisCounts)) this.diagnosisCounts = {};

      if (this.pending != null) {
        if (typeof this.pending !== 'object' || Array.isArray(this.pending)) {
          this.pending = null;
        } else {
          if (!this.pending.diagnoses || typeof this.pending.diagnoses !== 'object' || Array.isArray(this.pending.diagnoses)) this.pending.diagnoses = {};
          if (!Array.isArray(this.pending.targetSamples)) this.pending.targetSamples = [];
          if (!Array.isArray(this.pending.reasons)) this.pending.reasons = [];
          if (!this.pending.previous || typeof this.pending.previous !== 'object' || Array.isArray(this.pending.previous)) this.pending.previous = { ...this.profile };
          if (!this.pending.proposed || typeof this.pending.proposed !== 'object' || Array.isArray(this.pending.proposed)) this.pending.proposed = { ...this.profile };
          this.pending.observed = Math.max(0, Number(this.pending.observed) || 0);
          this.pending.evaluationWindow = Math.max(4, Number(this.pending.evaluationWindow) || this.options.evaluationWindow);
          this.pending.baselineUtility = Number.isFinite(Number(this.pending.baselineUtility)) ? Number(this.pending.baselineUtility) : Number(this.metrics.utility ?? 0.5);
          this.pending.baselineTarget = Number.isFinite(Number(this.pending.baselineTarget)) ? Number(this.pending.baselineTarget) : 0.5;
          this.pending.baselineControlRate = Number.isFinite(Number(this.pending.baselineControlRate)) ? Number(this.pending.baselineControlRate) : 1;
        }
      }
      return this;
    }

    _defaultProfile() {
      return {
        grammarScale: 1, semanticScale: 1, reasoningScale: 1, transitionScale: 1,
        structurePenaltyScale: 1, terminalScale: 1, relationScale: 1, causalScale: 1,
        contextScale: 1, reflectionScale: 1, structureReasoningScale: 1, referenceScale: 1,
        propositionScale: 1, mechanismScale: 1, counterfactualScale: 1, contradictionScale: 1
      };
    }

    _defaultMetrics() {
      return {
        quality: 0.5, grammar: 0.5, semantic: 0.5, reasoning: 0.5, context: 0.5,
        contradictionSafety: 0.5, latencyScore: 0.5, calibration: 0.5, cognitiveProgress: 0.5,
        structureCoverage: 0.5, referenceResolution: 0.5, propositionCoverage: 0.5,
        mechanismCompleteness: 0.5, multiPartCoverage: 0.5, driftSafety: 0.5,
        temporalConsistency: 0.5, uncertaintyLocalization: 0.5, utility: 0.5
      };
    }

    _publish() {
      if (!this.runtimeOptions) return;
      this.runtimeOptions.selfImprovementProfile = {
        ...this.profile,
        generation: this.accepted - this.rollbacks,
        updatedAt: Date.now(),
        dominantDiagnosis: this.diagnoses.at(-1)?.type || null,
        pendingExperiment: this.pending?.id || null
      };
    }

    _ewma(oldValue, next, first = false) {
      return first ? next : oldValue * (1 - this.options.alpha) + next * this.options.alpha;
    }

    _utility(m = this.metrics) {
      return clamp(
        m.quality * 0.17 + m.grammar * 0.11 + m.semantic * 0.13 + m.reasoning * 0.13 + m.context * 0.07 +
        m.contradictionSafety * 0.06 + m.cognitiveProgress * 0.04 + m.structureCoverage * 0.05 + m.referenceResolution * 0.05 +
        m.propositionCoverage * 0.06 + m.mechanismCompleteness * 0.06 + m.multiPartCoverage * 0.03 + m.driftSafety * 0.04 +
        m.temporalConsistency * 0.04 + m.uncertaintyLocalization * 0.03 + m.latencyScore * 0.02 + (1 - m.calibration) * 0.04
      );
    }

    _diagnosticCandidates(current) {
      const rows = [
        ['grammar', current.grammar, 0.82, 'surface-grammar'],
        ['semantic', current.semantic, 0.78, 'semantic-retrieval'],
        ['reasoning', current.reasoning, 0.74, 'reasoning-selection'],
        ['context', current.context, 0.72, 'conversation-focus'],
        ['contradictionSafety', current.contradictionSafety, 0.82, 'claim-consistency'],
        ['cognitiveProgress', current.cognitiveProgress, 0.66, 'cognitive-convergence'],
        ['structureCoverage', current.structureCoverage, 0.74, 'complex-language-coverage'],
        ['referenceResolution', current.referenceResolution, 0.80, 'reference-or-dependency-resolution'],
        ['propositionCoverage', current.propositionCoverage, 0.74, 'proposition-coverage'],
        ['mechanismCompleteness', current.mechanismCompleteness, 0.72, 'mechanism-completeness'],
        ['multiPartCoverage', current.multiPartCoverage, 0.74, 'multi-part-request-coverage'],
        ['driftSafety', current.driftSafety, 0.76, 'semantic-drift'],
        ['temporalConsistency', current.temporalConsistency, 0.78, 'temporal-reasoning'],
        ['uncertaintyLocalization', current.uncertaintyLocalization, 0.70, 'uncertainty-localization']
      ].map(([metric, value, target, type]) => ({ metric, value, target, type, deficit: Math.max(0, target - value) }));
      if (current.calibration > 0.14) rows.push({ metric: 'calibration', value: 1 - current.calibration, target: 0.86, type: 'confidence-calibration', deficit: current.calibration - 0.14 });
      if (current.latencyScore < 0.55) rows.push({ metric: 'latencyScore', value: current.latencyScore, target: 0.55, type: 'search-efficiency', deficit: 0.55 - current.latencyScore });
      return rows.sort((a, b) => b.deficit - a.deficit);
    }

    _diagnose(event, current) {
      const ranked = this._diagnosticCandidates(current);
      const best = ranked[0];
      const type = best?.deficit > 0.025 ? best.type : 'balanced-response';
      const row = {
        at: Date.now(),
        responseIndex: this.responses,
        type,
        metric: best?.metric || 'none',
        deficit: Number(best?.deficit || 0),
        pressure: clamp((best?.deficit || 0) / Math.max(0.01, best?.target || 1)),
        alternatives: ranked.slice(1, 4).filter(r => r.deficit > 0.02).map(r => ({ type: r.type, metric: r.metric, deficit: r.deficit })),
        prompt: String(event.promptText || '').slice(0, 240),
        reasoningMode: event.reasoning?.selectedMode || event.reasoning?.selectedState?.mode || null,
        cognitiveCycles: Number(event.cognitiveState?.cycleCount || 0),
        unresolved: (event.cognitiveState?.unresolved || event.analysis?.complexLanguage?.unresolved || []).slice(0, 12),
        structure: {
          complexity: Number(event.structuralComplexity || event.analysis?.structuralComplexity || 0),
          clauseCount: Number(event.clauseCount || event.analysis?.complexLanguage?.clauseCount || 0),
          propositionCount: Number(event.propositionCount || event.analysis?.propositionGraph?.propositions?.length || 0),
          mechanismActive: Boolean(event.mechanismActive || event.analysis?.mechanismGraph?.active)
        }
      };
      this.diagnoses.push(row);
      if (this.diagnoses.length > this.options.diagnosisHistory) this.diagnoses.splice(0, this.diagnoses.length - this.options.diagnosisHistory);
      if (!this.diagnosisCounts || typeof this.diagnosisCounts !== 'object' || Array.isArray(this.diagnosisCounts)) this.diagnosisCounts = {};
      this.diagnosisCounts[type] = (Number(this.diagnosisCounts[type]) || 0) + 1;
      return row;
    }

    _scheduleReplay(diagnosis, event, current) {
      if (!this.replay?.scheduleFailure || diagnosis.type === 'balanced-response') return null;
      if (diagnosis.deficit < this.options.failureReplayThreshold) return null;
      if (this.responses - this.lastReplayScheduledAt < this.options.replayCooldownResponses) return null;
      const prediction = this.predictionError?.status?.() || {};
      const uncertainty = event.uncertaintyState || event.reasoning?.uncertainty || {};
      const scheduled = this.replay.scheduleFailure(diagnosis, {
        promptText: event.promptText || '',
        responseText: event.responseText || '',
        reasoningMode: diagnosis.reasoningMode,
        structure: diagnosis.structure,
        targetMetric: diagnosis.metric,
        currentScore: current?.[diagnosis.metric]
      }, {
        predictionError: clamp(prediction.meanAbsError ?? 0.35),
        uncertainty: clamp(1 - Number(uncertainty.reliability ?? current.uncertaintyLocalization ?? 0.5)),
        usefulness: 0.84,
        novelty: clamp(event.cognitiveState?.novelty ?? 0.55),
        contradictionPressure: clamp(1 - current.contradictionSafety),
        diagnosisPressure: diagnosis.pressure
      });
      if (scheduled) {
        this.lastReplayScheduledAt = this.responses;
        this.replaySchedules++;
      }
      return scheduled;
    }

    observeCognitiveTrace(trace = {}) {
      this.cognitiveTraces++;
      this.lastCognitiveTrace = {
        at: Date.now(),
        cycleCount: Number(trace.cycleCount || 0),
        progress: clamp(trace.progress ?? 0.5),
        converged: Boolean(trace.converged),
        unresolved: (trace.unresolved || []).slice(0, 12),
        relationCount: (trace.relations || []).length,
        activeConceptCount: (trace.activeConcepts || []).length,
        hypothesisCount: (trace.hypotheses || []).length
      };
      return this.lastCognitiveTrace;
    }

    _currentMetrics(event = {}) {
      const ch = event.channels || event.diagnostics?.channels || {};
      const grammar = clamp(event.grammar ?? ch.grammarLegality ?? event.syntax?.smoothedGrammar ?? 0.5);
      const semantic = clamp(event.semantic ?? ch.semanticSupport ?? ch.knowledgeSupport ?? event.rankFocus ?? 0.5);
      const reasoning = clamp(event.reasoningAlignment ?? event.reasoning?.alignment ?? 0.5);
      const context = clamp(event.contextAlignment ?? event.conversationAlignment ?? event.reasoning?.conversationAlignment ?? 0.5);
      const contradictionSafety = clamp(event.contradictionSafety ?? event.reasoningQuality?.contradictionSafety ?? (1 - Number(event.contradictionScore || 0)));
      const cognitiveProgress = clamp(event.cognitiveState?.progress ?? event.reasoning?.cognitiveProgress ?? 0.5);
      const structureCoverage = clamp(event.structureVerification?.score ?? (Number(event.structuralComplexity || 0) >= 0.15 ? 0.5 : 0.82));
      const referenceResolution = clamp(event.structureVerification?.referenceCoverage ?? event.uncertaintyState?.referenceResolution ?? 0.84);
      const rq = event.reasoningQuality || {};
      const propositionCoverage = clamp(rq.propositionCoverage ?? event.structureVerification?.propositionCoverage ?? structureCoverage);
      const mechanismCompleteness = clamp(rq.mechanismCompleteness ?? event.uncertaintyState?.mechanismCompleteness ?? (event.mechanismActive ? 0.5 : 0.92));
      const multiPartCoverage = clamp(rq.multiPartCoverage ?? event.structureVerification?.multiPartCoverage ?? 0.84);
      const driftSafety = clamp(rq.driftSafety ?? event.structureVerification?.driftSafety ?? 0.84);
      const temporalConsistency = clamp(event.temporalConsistency ?? event.uncertaintyState?.temporalConsistency ?? rq.temporalConsistency ?? 0.86);
      const uncertaintyReasons = event.uncertaintyState?.reasons || [];
      const uncertaintyLocalization = clamp(event.uncertaintyLocalization ?? (event.uncertaintyState ? (uncertaintyReasons.length ? 0.78 : 0.88) : 0.60));
      const quality = clamp(event.quality ?? ((Number(event.confidence) || 0) / 100) * 0.20 + grammar * 0.15 + semantic * 0.16 + reasoning * 0.14 + context * 0.05 + structureCoverage * 0.06 + referenceResolution * 0.05 + propositionCoverage * 0.06 + mechanismCompleteness * 0.05 + multiPartCoverage * 0.02 + driftSafety * 0.03 + temporalConsistency * 0.03);
      const target = Math.max(1, Number(event.latencyTargetMs) || 40);
      const latency = Math.max(0, Number(event.totalMs) || 0);
      const latencyScore = clamp(target / Math.max(target, latency || target));
      const calibration = clamp(Math.abs(((Number(event.confidence) || 0) / 100) - quality));
      return { quality, grammar, semantic, reasoning, context, contradictionSafety, latencyScore, calibration, cognitiveProgress, structureCoverage, referenceResolution, propositionCoverage, mechanismCompleteness, multiPartCoverage, driftSafety, temporalConsistency, uncertaintyLocalization };
    }

    observeResponse(event = {}) {
      this._ensureRuntimeState();
      const current = this._currentMetrics(event);
      const first = this.responses === 0;
      this.responses++;
      for (const key of Object.keys(current)) this.metrics[key] = this._ewma(this.metrics[key], current[key], first);
      this.metrics.utility = this._utility();
      const diagnosis = this._diagnose(event, current);
      const replayItem = this._scheduleReplay(diagnosis, event, current);
      if (event.cognitiveState) this.observeCognitiveTrace(event.cognitiveState);

      if (this.pending) {
        this._ensureRuntimeState();
        this.pending.observed++;
        this.pending.diagnoses[diagnosis.type] = (Number(this.pending.diagnoses[diagnosis.type]) || 0) + 1;
        const targetMetric = this.pending.targetMetric;
        if (targetMetric && Number.isFinite(current[targetMetric])) {
          this.pending.targetSamples.push(current[targetMetric]);
        }
        if (this.pending.observed >= this.pending.evaluationWindow) this._finishPending();
      }

      this.last = { at: Date.now(), ...this.metrics, diagnosis, replayScheduled: replayItem?.key || null, profile: { ...this.profile } };
      return this.last;
    }

    _mean(rows) { return rows?.length ? rows.reduce((a, b) => a + Number(b || 0), 0) / rows.length : 0; }

    _finishPending() {
      if (!this.pending) return null;
      const pending = this.pending;
      const utilityDelta = this.metrics.utility - pending.baselineUtility;
      const targetEnd = this._mean(pending.targetSamples);
      const targetGain = pending.targetSamples.length ? targetEnd - pending.baselineTarget : 0;
      const replayOutcome = pending.curriculum ? this.replay?.curriculumOutcome?.(pending.curriculum) : null;
      const controlRegression = replayOutcome?.controlSuccessRate == null ? 0 : Math.max(0, pending.baselineControlRate - replayOutcome.controlSuccessRate);
      const promotion = this._policy()?.Promotion || {};
      const targetGainMin = Number(promotion.target_gain_min ?? 0.012);
      const controlRegressionMax = Number(promotion.control_regression_max ?? this.options.rollbackTolerance);
      const overallRegressionMax = Number(promotion.overall_utility_regression_max ?? this.options.rollbackTolerance);

      const badUtility = utilityDelta < -overallRegressionMax;
      const badControls = controlRegression > controlRegressionMax;
      const targetFailed = pending.targetSamples.length >= 3 && targetGain < -targetGainMin;
      if (badUtility || badControls || targetFailed) {
        this.profile = { ...pending.previous };
        this.rollbacks++;
        pending.result = 'rolled-back';
      } else {
        this.accepted++;
        pending.result = targetGain >= targetGainMin ? 'accepted-target-improved' : 'accepted-stable';
      }
      pending.finishedAt = Date.now();
      pending.endingUtility = this.metrics.utility;
      pending.utilityDelta = utilityDelta;
      pending.targetEnd = targetEnd;
      pending.targetGain = targetGain;
      pending.controlRegression = controlRegression;
      pending.replayOutcome = replayOutcome;
      this.history.push({ ...pending });
      if (this.history.length > 128) this.history.splice(0, this.history.length - 128);
      this.pending = null;
      this._publish();
      this.dirty = true;
      return pending;
    }

    _recentDiagnosis(type, window = 24) {
      return this.diagnoses.slice(-window).filter(row => row.type === type).length;
    }

    _profileBounds(key) {
      const configured = this._policy()?.Bounds?.[key];
      if (Array.isArray(configured) && configured.length >= 2) return [Number(configured[0]), Number(configured[1])];
      const defaults = {
        grammarScale:[0.88,1.24], semanticScale:[0.88,1.24], reasoningScale:[0.88,1.30], transitionScale:[0.88,1.30],
        structurePenaltyScale:[0.88,1.30], terminalScale:[0.92,1.16], relationScale:[0.86,1.32], causalScale:[0.86,1.32],
        contextScale:[0.86,1.30], reflectionScale:[0.86,1.32], structureReasoningScale:[0.86,1.36], referenceScale:[0.86,1.36],
        propositionScale:[0.86,1.36], mechanismScale:[0.86,1.38], counterfactualScale:[0.86,1.30], contradictionScale:[0.86,1.30]
      };
      return defaults[key] || [0.86, 1.34];
    }

    _applyDiagnosisTarget(next, diagnosisType, reasons) {
      const target = this._diagnosisTarget(diagnosisType);
      if (!target?.profiles) return false;
      const step = this.options.step;
      let changed = false;
      for (const [key, scale] of Object.entries(target.profiles)) {
        if (!Object.prototype.hasOwnProperty.call(next, key)) continue;
        next[key] += step * Number(scale || 0);
        changed = true;
      }
      if (changed) reasons.push(diagnosisType);
      return changed;
    }

    _dominantDeficit() {
      const candidates = [
        ['surface-grammar', 0.82 - this.metrics.grammar],
        ['semantic-retrieval', 0.78 - this.metrics.semantic],
        ['reasoning-selection', 0.74 - this.metrics.reasoning],
        ['conversation-focus', 0.72 - this.metrics.context],
        ['complex-language-coverage', 0.74 - this.metrics.structureCoverage],
        ['reference-or-dependency-resolution', 0.80 - this.metrics.referenceResolution],
        ['proposition-coverage', 0.74 - this.metrics.propositionCoverage],
        ['mechanism-completeness', 0.72 - this.metrics.mechanismCompleteness],
        ['multi-part-request-coverage', 0.74 - this.metrics.multiPartCoverage],
        ['semantic-drift', 0.76 - this.metrics.driftSafety],
        ['temporal-reasoning', 0.78 - this.metrics.temporalConsistency],
        ['confidence-calibration', this.metrics.calibration - 0.14]
      ].sort((a, b) => b[1] - a[1]);
      return candidates[0]?.[1] > 0.02 ? candidates[0] : null;
    }

    _propose() {
      if (!this.options.enabled || this.pending || this.responses < this.options.proposalEvery || this.responses % this.options.proposalEvery !== 0) return null;
      const next = { ...this.profile }, reasons = [];
      const dominant = this._dominantDeficit();
      if (dominant) this._applyDiagnosisTarget(next, dominant[0], reasons);

      // Repeated diagnoses are allowed to add a smaller secondary adjustment.
      const recent = this.diagnoses.slice(-24).reduce((m, row) => (m[row.type] = (m[row.type] || 0) + 1, m), {});
      const secondaries = Object.entries(recent).sort((a, b) => b[1] - a[1]).filter(([type, count]) => count >= 3 && type !== dominant?.[0]).slice(0, 2);
      for (const [type] of secondaries) {
        const target = this._diagnosisTarget(type);
        if (!target?.profiles) continue;
        for (const [key, scale] of Object.entries(target.profiles)) if (Object.prototype.hasOwnProperty.call(next, key)) next[key] += this.options.step * Number(scale || 0) * 0.35;
        reasons.push(`${type}:secondary`);
      }

      const prediction = this.predictionError?.status?.() || {};
      const hypothesis = this.hypotheses?.status?.() || {};
      const replay = this.replay?.status?.() || {};
      if (Number(prediction.meanAbsError) > 0.22) {
        next.semanticScale += this.options.step * 0.35;
        next.relationScale += this.options.step * 0.25;
        reasons.push('prediction-error-pressure');
      }
      if (Number(hypothesis.contested) > 0 && Number(hypothesis.contested) >= Math.max(2, Number(hypothesis.promotable) || 0)) {
        next.structurePenaltyScale += this.options.step * 0.30;
        next.contradictionScale += this.options.step * 0.25;
        reasons.push('hypothesis-contradiction-pressure');
      }
      if (!reasons.length) {
        const evaluator = this.evaluator?.status?.() || {};
        if (Number(evaluator.learningDelta) > 0.01 || (Number(replay.due) > 0 && Number(evaluator.learningDelta) >= 0)) {
          next.semanticScale += this.options.step * 0.20;
          next.reflectionScale += this.options.step * 0.15;
          reasons.push('positive-learning-reinforcement');
        } else return null;
      }

      for (const key of Object.keys(next)) {
        const [lo, hi] = this._profileBounds(key);
        next[key] = bound(next[key], lo, hi);
      }
      const changed = Object.keys(next).some(key => Math.abs(next[key] - this.profile[key]) > 1e-9);
      if (!changed) return null;

      const primaryType = dominant?.[0] || this.diagnoses.at(-1)?.type || 'general';
      const policyTarget = this._diagnosisTarget(primaryType) || {};
      const targetMetric = policyTarget.metrics?.[0] || this.diagnoses.at(-1)?.metric || null;
      const curriculum = policyTarget.replay || primaryType;
      const replayOutcome = this.replay?.curriculumOutcome?.(curriculum) || null;
      const promotion = this._policy()?.Promotion || {};
      const evaluationWindow = Math.max(this.options.evaluationWindow, Number(promotion.evaluation_window) || 0);

      this.pending = {
        id: `self-${++this.proposals}`,
        at: Date.now(),
        previous: { ...this.profile },
        proposed: { ...next },
        baselineUtility: this.metrics.utility,
        baselineTarget: targetMetric ? Number(this.metrics[targetMetric] ?? 0.5) : 0.5,
        baselineControlRate: replayOutcome?.controlSuccessRate ?? 1,
        targetMetric,
        curriculum,
        targetSamples: [],
        observed: 0,
        evaluationWindow,
        reasons,
        diagnoses: {}
      };
      this.profile = next;
      this._publish();
      this.dirty = true;
      return this.pending;
    }

    diagnoseLanguageStructure(event = {}) {
      const analysis = event.analysis || {};
      const verification = event.structureVerification || {};
      const complexity = Number(analysis.structuralComplexity || 0);
      const clauseCoverage = clamp(verification.clauseCoverage ?? verification.score ?? (complexity < 0.15 ? 0.90 : 0.5));
      const referenceCoverage = clamp(verification.referenceCoverage ?? analysis.propositionGraph?.metrics?.referenceCoverage ?? 0.82);
      const propositionCoverage = clamp(verification.propositionCoverage ?? analysis.propositionGraph?.metrics?.propositionCompleteness ?? 0.76);
      const temporal = analysis.temporalAssessment || analysis.temporal || null;
      const temporalConsistency = clamp(verification.temporalConsistency ?? temporal?.consistency ?? 0.86);
      const unresolved = (analysis.complexLanguage?.unresolved || []).length;
      const mechanism = analysis.mechanismGraph || {};
      const mechanismCompleteness = clamp(mechanism.active ? mechanism.completeness ?? 0.5 : 1);

      const candidates = [
        { type:'complex-language-coverage', metric:'clauseCoverage', value:clauseCoverage, target:0.72, targetProfile:'structureReasoningScale' },
        { type:'reference-or-dependency-resolution', metric:'referenceCoverage', value:referenceCoverage, target:0.80, targetProfile:'referenceScale' },
        { type:'proposition-coverage', metric:'propositionCoverage', value:propositionCoverage, target:0.74, targetProfile:'propositionScale' },
        { type:'temporal-reasoning', metric:'temporalConsistency', value:temporalConsistency, target:0.78, targetProfile:'structureReasoningScale' },
        { type:'mechanism-completeness', metric:'mechanismCompleteness', value:mechanismCompleteness, target:0.72, targetProfile:'mechanismScale' }
      ].map(row => ({ ...row, deficit: Math.max(0, row.target - row.value) }));
      if (unresolved) candidates.push({ type:'reference-or-dependency-resolution', metric:'unresolved', value:0, target:1, deficit:Math.min(0.4, unresolved * 0.08), targetProfile:'referenceScale' });
      candidates.sort((a, b) => b.deficit - a.deficit);
      const best = candidates[0];
      if (complexity < 0.15 && (!mechanism.active || mechanismCompleteness >= 0.72) && !unresolved) return null;
      return {
        type: best?.deficit > 0.025 ? best.type : 'complex-language-stable',
        complexity,
        clauseCoverage,
        referenceCoverage,
        propositionCoverage,
        temporalConsistency,
        mechanismCompleteness,
        unresolved,
        target: best?.targetProfile || 'grammarScale',
        pressure: clamp((best?.deficit || 0) + complexity * (1 - clauseCoverage) * 0.35),
        deficits: candidates.slice(0, 5).map(row => ({ type: row.type, metric: row.metric, deficit: row.deficit }))
      };
    }

    async backgroundTick(phase) {
      if (!this.options.enabled) return null;
      if (phase === 'ACTIVE' && this.responses % Math.max(8, this.options.proposalEvery) !== 0) return null;
      return this._propose();
    }

    exportState() {
      this._ensureRuntimeState();
      return {
        schemaVersion: 5,
        profile: { ...this.profile }, metrics: { ...this.metrics }, responses: this.responses,
        proposals: this.proposals, accepted: this.accepted, rollbacks: this.rollbacks, pending: this.pending,
        history: this.history.slice(-128), diagnoses: this.diagnoses.slice(-this.options.diagnosisHistory),
        diagnosisCounts: { ...this.diagnosisCounts }, cognitiveTraces: this.cognitiveTraces,
        lastCognitiveTrace: this.lastCognitiveTrace, replaySchedules: this.replaySchedules,
        lastReplayScheduledAt: this.lastReplayScheduledAt, last: this.last
      };
    }

    importState(state) {
      if (!state || ![1,2,3,4,5].includes(Number(state.schemaVersion))) return false;
      this.profile = { ...this._defaultProfile(), ...(state.profile || {}) };
      this.metrics = { ...this._defaultMetrics(), ...(state.metrics || {}) };
      this.responses = Math.max(0, Number(state.responses) || 0);
      this.proposals = Math.max(0, Number(state.proposals) || 0);
      this.accepted = Math.max(0, Number(state.accepted) || 0);
      this.rollbacks = Math.max(0, Number(state.rollbacks) || 0);
      this.pending = state.pending || null;
      this._ensureRuntimeState();
      this.history = Array.isArray(state.history) ? state.history.slice(-128) : [];
      this.diagnoses = Array.isArray(state.diagnoses) ? state.diagnoses.slice(-this.options.diagnosisHistory) : [];
      this.diagnosisCounts = state.diagnosisCounts && typeof state.diagnosisCounts === 'object' ? { ...state.diagnosisCounts } : {};
      this.cognitiveTraces = Math.max(0, Number(state.cognitiveTraces) || 0);
      this.lastCognitiveTrace = state.lastCognitiveTrace || null;
      this.replaySchedules = Math.max(0, Number(state.replaySchedules) || 0);
      this.lastReplayScheduledAt = Number(state.lastReplayScheduledAt ?? -999);
      this.last = state.last || null;
      this._publish();
      return true;
    }

    reset() {
      this.profile = this._defaultProfile();
      this.metrics = this._defaultMetrics();
      this.responses = this.proposals = this.accepted = this.rollbacks = 0;
      this.pending = null;
      this.history = [];
      this.diagnoses = [];
      this.diagnosisCounts = {};
      this.cognitiveTraces = 0;
      this.lastCognitiveTrace = null;
      this.replaySchedules = 0;
      this.lastReplayScheduledAt = -999;
      this._publish();
      this.dirty = true;
      return true;
    }

    status() {
      this._ensureRuntimeState();
      return {
        ready: true,
        role: 'stage-diagnostic-replay-curriculum-control-guarded-bounded-self-improvement',
        enabled: this.options.enabled,
        responses: this.responses,
        proposals: this.proposals,
        accepted: this.accepted,
        rollbacks: this.rollbacks,
        replaySchedules: this.replaySchedules,
        profile: { ...this.profile },
        metrics: { ...this.metrics },
        pending: this.pending,
        last: this.last,
        diagnosisCounts: { ...this.diagnosisCounts },
        lastDiagnosis: this.diagnoses.at(-1) || null,
        cognitiveTraces: this.cognitiveTraces,
        lastCognitiveTrace: this.lastCognitiveTrace,
        ruleDriven: Boolean(this._policy()?.Schema_Version),
        safeguards: { sourceRewrite:false, modelBreadthReduction:false, branchReduction:false, knowledgeSourceRemoval:false, rollback:true, controlRegressionCheck:true }
      };
    }
  }


  const V34_FAILURE_TYPES=Object.freeze([
    { type:'surface-grammar', stage:'generation', metric:'grammar', target:0.82, replay:'surface-grammar' },
    { type:'semantic-retrieval', stage:'retrieval', metric:'semantic', target:0.78, replay:'semantic-retrieval' },
    { type:'reasoning-selection', stage:'reasoning', metric:'reasoning', target:0.74, replay:'reasoning-selection' },
    { type:'conversation-focus', stage:'context', metric:'context', target:0.72, replay:'conversation-focus' },
    { type:'claim-consistency', stage:'verification', metric:'contradictionSafety', target:0.82, replay:'claim-consistency' },
    { type:'cognitive-convergence', stage:'cognition', metric:'cognitiveProgress', target:0.66, replay:'cognitive-convergence' },
    { type:'complex-language-coverage', stage:'language', metric:'structureCoverage', target:0.74, replay:'complex-language-coverage' },
    { type:'reference-or-dependency-resolution', stage:'reference', metric:'referenceResolution', target:0.80, replay:'reference-or-dependency-resolution' },
    { type:'proposition-coverage', stage:'representation', metric:'propositionCoverage', target:0.74, replay:'proposition-coverage' },
    { type:'mechanism-completeness', stage:'mechanism', metric:'mechanismCompleteness', target:0.72, replay:'mechanism-completeness' },
    { type:'multi-part-request-coverage', stage:'planning', metric:'multiPartCoverage', target:0.74, replay:'multi-part-request-coverage' },
    { type:'semantic-drift', stage:'verification', metric:'driftSafety', target:0.76, replay:'semantic-drift' },
    { type:'temporal-reasoning', stage:'temporal', metric:'temporalConsistency', target:0.78, replay:'temporal-reasoning' },
    { type:'uncertainty-localization', stage:'calibration', metric:'uncertaintyLocalization', target:0.70, replay:'uncertainty-localization' },
    { type:'confidence-calibration', stage:'calibration', metric:'calibration', target:0.86, replay:'confidence-calibration' },
    { type:'causal-reasoning', stage:'reasoning', metric:'reasoning', target:0.76, replay:'causal-reasoning' },
    { type:'comparison-reasoning', stage:'reasoning', metric:'reasoning', target:0.76, replay:'comparison-reasoning' },
    { type:'counterfactual-reasoning', stage:'reasoning', metric:'reasoning', target:0.74, replay:'counterfactual-reasoning' },
    { type:'replay-curriculum', stage:'learning', metric:'reasoning', target:0.72, replay:'replay-curriculum' },
    { type:'control-regression', stage:'learning', metric:'calibration', target:0.82, replay:'control-regression' },
  ]);
  const v34IClamp=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(v))?Number(v):0));
  const v34INorm=v=>String(v??'').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g,' ').replace(/\s+/g,' ').trim();
  class V34ImprovementEngine {
    constructor(owner){this.owner=owner;this.hypotheses=[];this.experiments=[];this.promotions=[];this.rollbacks=[];this.last=null;}
    type(type){return V34_FAILURE_TYPES.find(x=>x.type===type)||{type:type||'general',stage:'unknown',metric:'reasoning',target:0.72,replay:type||'general'};}
    policyTarget(type){
      const policy=this.owner?._policy?.()||{};const table=policy&&typeof policy.Diagnosis_Targets==='object'?policy.Diagnosis_Targets:{};return table[type]||policy.Fallback_Diagnosis||{metrics:['reasoning'],profiles:{reasoningScale:0.1},replay:type||'general'};
    }
    metric(event,name,fallback=0.5){const pools=[event?.metrics,event?.evaluation,event?.quality,event?.diagnostics,event?.uncertaintyState,event?.analysis?.quality];for(const pool of pools)if(Number.isFinite(Number(pool?.[name])))return v34IClamp(pool[name]);return v34IClamp(fallback);}
    traceSignals(event={}){
      const trace=event.cognitiveTrace||event.trace||event.cognitiveState||{};const analysis=event.analysis||{};return{
        unresolved:(trace.unresolved||analysis.complexLanguage?.unresolved||[]).length,
        contradictions:(event.contradictions||analysis.contradictions||event.uncertaintyState?.contradictions||[]).length,
        references:(analysis.referenceLinks||analysis.complexLanguage?.references||[]).length,
        unresolvedReferences:(analysis.referenceLinks||analysis.complexLanguage?.references||[]).filter(r=>r?.resolved===false).length,
        propositions:(analysis.propositionGraph?.propositions||analysis.propositions||[]).length,
        mechanismStages:(analysis.mechanismGraph?.stages||[]).length,
        mechanismMissing:(analysis.mechanismGraph?.missing||[]).length,
        temporalContradictions:(analysis.temporalAssessment?.contradictions||[]).length,
        temporalRelations:(analysis.temporalAssessment?.relations||[]).length,
        branchCount:(trace.branches||event.branches||[]).length,
        branchDisagreement:Number(event.uncertaintyState?.uncertaintyLayers?.branches?.disagreement||0),
        localizedUncertainty:(event.uncertaintyState?.localizedUncertainty||[]).length,
        repairs:(event.syntax?.repairs||event.surface?.repairs||[]).length
      };
    }
    rootCauses(diagnosis,event={}){
      const signals=this.traceSignals(event);const causes=[];const push=(cause,score,evidence)=>causes.push({cause,score:v34IClamp(score),evidence});
      switch(diagnosis.type){
        case'reference-or-dependency-resolution':push('insufficient-antecedent-margin',signals.unresolvedReferences?0.92:0.42,signals.unresolvedReferences);push('dependency-alignment-weak',signals.unresolved?0.76:0.38,signals.unresolved);push('semantic-class-mismatch',0.48,signals.references);break;
        case'temporal-reasoning':push('temporal-conflict',signals.temporalContradictions?0.96:0.30,signals.temporalContradictions);push('missing-event-order',signals.temporalRelations<2?0.72:0.34,signals.temporalRelations);push('aspect-or-frequency-unmodeled',0.46,signals.temporalRelations);break;
        case'mechanism-completeness':push('missing-stage',signals.mechanismMissing?0.94:0.44,signals.mechanismMissing);push('stage-order-weak',signals.mechanismStages>2?0.58:0.34,signals.mechanismStages);push('input-output-link-weak',0.52,signals.mechanismStages);break;
        case'claim-consistency':push('contradiction-unresolved',signals.contradictions?0.96:0.38,signals.contradictions);push('scope-alignment-failed',0.54,signals.propositions);break;
        case'cognitive-convergence':push('premature-stop',signals.unresolved?0.86:0.42,signals.unresolved);push('branch-disagreement',signals.branchDisagreement,signals.branchCount);push('low-gain-loop',0.50,signals.branchCount);break;
        case'uncertainty-localization':push('uncertainty-collapsed-to-global',signals.localizedUncertainty<1?0.84:0.28,signals.localizedUncertainty);push('reference-ambiguity-hidden',signals.unresolvedReferences?0.76:0.30,signals.unresolvedReferences);break;
        default:push('target-metric-deficit',diagnosis.deficit??0.5,diagnosis.metric);push('cross-stage-dependency',Math.min(0.72,(signals.unresolved+signals.contradictions)*0.08+0.28),signals);break;
      }
      return causes.sort((a,b)=>b.score-a.score).slice(0,8);
    }
    generateHypotheses(diagnosis,event={}){
      const causes=this.rootCauses(diagnosis,event);const target=this.policyTarget(diagnosis.type);const hypotheses=[];
      for(const cause of causes){for(const [profile,scale] of Object.entries(target.profiles||{})){hypotheses.push({id:`h:${diagnosis.type}:${cause.cause}:${profile}`,diagnosis:diagnosis.type,cause:cause.cause,profile,deltaDirection:Number(scale)>=0?'increase':'decrease',scale:Number(scale||0),score:v34IClamp(cause.score*(0.7+Math.min(0.3,Math.abs(Number(scale||0))*0.2))),evidence:cause.evidence,status:'proposed'});}}
      hypotheses.sort((a,b)=>b.score-a.score);this.hypotheses.push(...hypotheses.slice(0,12));if(this.hypotheses.length>512)this.hypotheses.splice(0,this.hypotheses.length-512);return hypotheses.slice(0,12);
    }
    experimentFor(hypothesis,diagnosis,event={}){
      const target=this.policyTarget(diagnosis.type);const controls=['surface-grammar','semantic-retrieval','reasoning-selection','reference-or-dependency-resolution','temporal-reasoning','mechanism-completeness'].filter(x=>x!==diagnosis.type).slice(0,3);
      return{id:`exp:${Date.now()}:${this.experiments.length}`,hypothesis:hypothesis.id,diagnosis:diagnosis.type,targetMetric:(target.metrics||[diagnosis.metric||'reasoning'])[0],profile:hypothesis.profile,proposedScale:hypothesis.scale,replayCurriculum:target.replay||diagnosis.type,controlCurricula:controls,minimumTargetGain:Number(this.owner?._policy?.()?.Promotion?.target_gain_min??0.012),maximumControlRegression:Number(this.owner?._policy?.()?.Promotion?.control_regression_max??0.025),status:'queued',createdAt:Date.now()};
    }
    designExperiments(diagnosis,event={}){const hypotheses=this.generateHypotheses(diagnosis,event);const experiments=hypotheses.slice(0,4).map(h=>this.experimentFor(h,diagnosis,event));this.experiments.push(...experiments);if(this.experiments.length>256)this.experiments.splice(0,this.experiments.length-256);return experiments;}
    evaluateExperiment(experiment,result={}){
      const targetBefore=Number(result.targetBefore??0.5),targetAfter=Number(result.targetAfter??targetBefore),controls=v34INorm(result.controls)===''?(Array.isArray(result.controls)?result.controls:[]):result.controls;const controlRows=Array.isArray(controls)?controls:[];
      const targetGain=targetAfter-targetBefore;let worstControlRegression=0;for(const row of controlRows)worstControlRegression=Math.max(worstControlRegression,Number(row.before??0)-Number(row.after??row.before??0));
      const promote=targetGain>=experiment.minimumTargetGain&&worstControlRegression<=experiment.maximumControlRegression;const decision={...experiment,targetGain,worstControlRegression,status:promote?'promoted':'rejected',evaluatedAt:Date.now()};
      if(promote)this.promotions.push(decision);else this.rollbacks.push({...decision,status:'rolled-back'});return decision;
    }
    scheduleExperiment(experiment,event={}){
      const replay=this.owner?.replay;if(!replay?.schedule)return null;const payload={promptText:event.promptText||event.prompt||'',experiment,diagnosis:experiment.diagnosis,replayKind:'diagnostic'};
      return replay.schedule(`experiment:${experiment.id}`,payload,{curriculum:experiment.replayCurriculum,replayKind:'diagnostic',diagnosisPressure:0.88,uncertainty:0.72,usefulness:0.92});
    }
    diagnose(event,current){
      const candidates=V34_FAILURE_TYPES.map(spec=>{const value=Number(current?.[spec.metric]);const safe=Number.isFinite(value)?value:this.metric(event,spec.metric,0.5);return{...spec,value:safe,deficit:Math.max(0,spec.target-safe)};}).sort((a,b)=>b.deficit-a.deficit);
      const best=candidates[0];return best&&best.deficit>0.015?best:{type:'balanced-response',stage:'none',metric:'reasoning',value:current?.reasoning??0.8,target:0.74,deficit:0};
    }
    inspect(event,current,diagnosis){
      const rootCauses=this.rootCauses(diagnosis,event);const experiments=diagnosis.type==='balanced-response'?[]:this.designExperiments(diagnosis,event);for(const exp of experiments.slice(0,2))this.scheduleExperiment(exp,event);
      const result={diagnosis,rootCauses,experiments,traceSignals:this.traceSignals(event)};this.last=result;return result;
    }
    status(){return{hypotheses:this.hypotheses.length,experiments:this.experiments.length,promotions:this.promotions.length,rollbacks:this.rollbacks.length,last:this.last};}
  }

  const v34OriginalObserveResponse=VilotSelfImprovement.prototype.observeResponse;
  VilotSelfImprovement.prototype.observeResponse=function(event={}){
    const result=v34OriginalObserveResponse.call(this,event);this.v34Engine||=new V34ImprovementEngine(this);const current=this._currentMetrics?.(event)||this.metrics||{};const diagnosis=result?.diagnosis||this.diagnoses?.at(-1)||this.v34Engine.diagnose(event,current);const deep=this.v34Engine.inspect(event,current,diagnosis);if(this.last&&typeof this.last==='object')this.last={...this.last,system34Diagnosis:deep};return result;
  };
  VilotSelfImprovement.prototype.deepDiagnose=function(event={}){this.v34Engine||=new V34ImprovementEngine(this);const current=this._currentMetrics?.(event)||this.metrics||{};const diagnosis=this.v34Engine.diagnose(event,current);return this.v34Engine.inspect(event,current,diagnosis);};
  VilotSelfImprovement.prototype.evaluateImprovementExperiment=function(experiment,result={}){this.v34Engine||=new V34ImprovementEngine(this);return this.v34Engine.evaluateExperiment(experiment,result);};
  const v34OriginalSelfStatus=VilotSelfImprovement.prototype.status;
  VilotSelfImprovement.prototype.status=function(){const base=v34OriginalSelfStatus.call(this);this.v34Engine||=new V34ImprovementEngine(this);return{...base,system34:this.v34Engine.status()};};



  const V34_IMPROVEMENT_SCENARIOS = Object.freeze([
    {
      id: "diagnosis-scenario-1",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "root-cause-scenario-2",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "hypothesis-scenario-3",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "experiment-scenario-4",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "target-replay-scenario-5",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "control-replay-scenario-6",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "promotion-scenario-7",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "rollback-scenario-8",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "monitoring-scenario-9",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "regression-scenario-10",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "diagnosis-scenario-11",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "root-cause-scenario-12",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "hypothesis-scenario-13",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "experiment-scenario-14",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "target-replay-scenario-15",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "control-replay-scenario-16",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "promotion-scenario-17",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "rollback-scenario-18",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "monitoring-scenario-19",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "regression-scenario-20",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "diagnosis-scenario-21",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "root-cause-scenario-22",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "hypothesis-scenario-23",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "experiment-scenario-24",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "target-replay-scenario-25",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "control-replay-scenario-26",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "promotion-scenario-27",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "rollback-scenario-28",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "monitoring-scenario-29",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "regression-scenario-30",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "diagnosis-scenario-31",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "root-cause-scenario-32",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "hypothesis-scenario-33",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "experiment-scenario-34",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "target-replay-scenario-35",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "control-replay-scenario-36",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "promotion-scenario-37",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "rollback-scenario-38",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "monitoring-scenario-39",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "regression-scenario-40",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "diagnosis-scenario-41",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "root-cause-scenario-42",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "hypothesis-scenario-43",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "experiment-scenario-44",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "target-replay-scenario-45",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "control-replay-scenario-46",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "promotion-scenario-47",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "rollback-scenario-48",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "monitoring-scenario-49",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "regression-scenario-50",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "diagnosis-scenario-51",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "root-cause-scenario-52",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "hypothesis-scenario-53",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "experiment-scenario-54",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "target-replay-scenario-55",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "control-replay-scenario-56",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "promotion-scenario-57",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "rollback-scenario-58",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "monitoring-scenario-59",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "regression-scenario-60",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "diagnosis-scenario-61",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "root-cause-scenario-62",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "hypothesis-scenario-63",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "experiment-scenario-64",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "target-replay-scenario-65",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "control-replay-scenario-66",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "promotion-scenario-67",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "rollback-scenario-68",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "monitoring-scenario-69",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "regression-scenario-70",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "diagnosis-scenario-71",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "root-cause-scenario-72",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "hypothesis-scenario-73",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "experiment-scenario-74",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "target-replay-scenario-75",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "control-replay-scenario-76",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "promotion-scenario-77",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "rollback-scenario-78",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "monitoring-scenario-79",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "regression-scenario-80",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "diagnosis-scenario-81",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "root-cause-scenario-82",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "hypothesis-scenario-83",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "experiment-scenario-84",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "target-replay-scenario-85",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "control-replay-scenario-86",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "promotion-scenario-87",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "rollback-scenario-88",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "monitoring-scenario-89",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "regression-scenario-90",
      family: "regression",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "diagnosis-scenario-91",
      family: "diagnosis",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "root-cause-scenario-92",
      family: "root-cause",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "hypothesis-scenario-93",
      family: "hypothesis",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "experiment-scenario-94",
      family: "experiment",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "target-replay-scenario-95",
      family: "target-replay",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "control-replay-scenario-96",
      family: "control-replay",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "promotion-scenario-97",
      family: "promotion",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "rollback-scenario-98",
      family: "rollback",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "monitoring-scenario-99",
      family: "monitoring",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
  ]);
  VilotSelfImprovement.prototype.improvementScenarios = function() { return V34_IMPROVEMENT_SCENARIOS.map(row => ({ ...row })); };

  globalThis.VilotSelfImprovement = VilotSelfImprovement;
})();
