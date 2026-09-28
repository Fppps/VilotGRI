/**
 * VilotNI 2.5 - Cognitive.js
 *
 * Autonomous background cognition coordinator.
 *
 * VilotNI 2 already had autonomous background learning through Worker scheduling,
 * Correlation.backgroundTick(), LanguageState.backgroundTick(), WordState and RSL
 * maintenance. Cognitive.js does not throw that work away. It coordinates it and
 * adds user-independent TrainingInfo self-study, relation consolidation, internal
 * reflection, novelty/attention tracking, and persistent cognitive state.
 *
 * It never emits a user response and never learns an invented answer as a fact.
 */
(() => {

  'use strict';


  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));


  const normalize = value =>
    String(value || '')
      .toLowerCase()
      .replace(/[’]/g, "'")
      .replace(/[^a-z0-9'+.-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();


  class VilotCognitive {

    constructor(dependencies = {
}
, options = {
}
) {

      this.rsl = dependencies.rsl || null;

      this.processor = dependencies.processor || null;

      this.memory = dependencies.memory || null;

      this.correlation = dependencies.correlation || null;

      this.languageState = dependencies.languageState || null;

      this.wordState = dependencies.wordState || null;

      this.phraseState = dependencies.phraseState || null;

      this.wordMatrix = dependencies.wordMatrix || null;

      this.trainingKnowledge = dependencies.trainingKnowledge || null;

      this.entityResolver = dependencies.entityResolver || null;

      this.selfImprovement = dependencies.selfImprovement || null;

      this.learningController = dependencies.learningController || null;

      this.knowledgeGraph = dependencies.knowledgeGraph || null;

      this.predictionError = dependencies.predictionError || null;

      this.hypothesisStore = dependencies.hypothesisStore || null;

      this.replayScheduler = dependencies.replayScheduler || null;

      this.contradictionResolver = dependencies.contradictionResolver || null;

      this.learningEvaluator = dependencies.learningEvaluator || null;

      this.graphReasoner = dependencies.graphReasoner || null;

      this.learnedKnowledgeBridge = dependencies.learnedKnowledgeBridge || null;

      this.uncertaintyState = dependencies.uncertaintyState || null;

      this.temporalKnowledge = dependencies.temporalKnowledge || null;

      this.trace = dependencies.trace || null;


      this.options = {

        enabled: options.cognitiveEnabled !== false,

        activeBudgetMs: Math.max(
          0.5,
          Number(options.cognitiveActiveBudgetMs) || 1.5
        ),

        warmBudgetMs: Math.max(
          1,
          Number(options.cognitiveWarmBudgetMs) || 4
        ),

        idleBudgetMs: Math.max(
          2,
          Number(options.cognitiveIdleBudgetMs) || 10
        ),

        activeConcepts: Math.max(
          0,
          options.cognitiveActiveConcepts | 0 || 1
        ),

        warmConcepts: Math.max(
          1,
          options.cognitiveWarmConcepts | 0 || 2
        ),

        idleConcepts: Math.max(
          1,
          options.cognitiveIdleConcepts | 0 || 5
        ),

        activeStudyEvery: Math.max(
          2,
          options.cognitiveActiveStudyEvery | 0 || 8
        ),

        warmStudyEvery: Math.max(
          1,
          options.cognitiveWarmStudyEvery | 0 || 3
        ),

        idleStudyEvery: Math.max(
          1,
          options.cognitiveIdleStudyEvery | 0 || 1
        ),

        conceptLookahead: Math.max(
          4,
          Math.min(64, options.cognitiveConceptLookahead | 0 || 16)
        ),

        maxConceptChars: Math.max(
          256,
          options.cognitiveMaxConceptChars | 0 || 1600
        ),

        maxRelationTargets: Math.max(
          2,
          Math.min(24, options.cognitiveMaxRelationTargets | 0 || 10)
        ),

        baseStudyGain: clamp(
          options.cognitiveBaseStudyGain ?? 0.18
        ),

        relationGain: clamp(
          options.cognitiveRelationGain ?? 0.12
        ),

        attentionDecay: Math.max(
          0.90,
          Math.min(0.9999, Number(options.cognitiveAttentionDecay) || 0.992)
        ),

        attentionCapacity: Math.max(
          64,
          Math.min(4096, options.cognitiveAttentionCapacity | 0 || 512)
        ),

        thoughtHistoryCapacity: Math.max(
          8,
          Math.min(256, options.cognitiveThoughtHistoryCapacity | 0 || 48)
        ),

        goalCapacity: Math.max(32, Math.min(1024, options.cognitiveGoalCapacity | 0 || 256)),
        goalHarvestLimit: Math.max(4, Math.min(64, options.cognitiveGoalHarvestLimit | 0 || 20)),
        activeGoalLimit: Math.max(0, Math.min(2, options.cognitiveActiveGoalLimit | 0 || 0)),
        warmGoalLimit: Math.max(1, Math.min(6, options.cognitiveWarmGoalLimit | 0 || 2)),
        idleGoalLimit: Math.max(1, Math.min(8, options.cognitiveIdleGoalLimit | 0 || 4)),
        deepIdleGoalLimit: Math.max(2, Math.min(12, options.cognitiveDeepIdleGoalLimit | 0 || 6)),
        deepIdleAfterCycles: Math.max(3, options.cognitiveDeepIdleAfterCycles | 0 || 8),
        deepIdleBudgetMs: Math.max(4, Number(options.cognitiveDeepIdleBudgetMs) || 22),
        goalMinPriority: clamp(options.cognitiveGoalMinPriority ?? 0.18),
        curiosityMinUncertainty: clamp(options.cognitiveCuriosityMinUncertainty ?? 0.42),
        staleReviewFreshness: clamp(options.cognitiveStaleReviewFreshness ?? 0.68),
        workspaceHistoryCapacity: Math.max(8, Math.min(128, options.cognitiveWorkspaceHistoryCapacity | 0 || 32)),
        foregroundThinkingCycles: Math.max(2, Math.min(8, options.cognitiveForegroundThinkingCycles | 0 || 4)),
        foregroundRelationLimit: Math.max(12, Math.min(96, options.cognitiveForegroundRelationLimit | 0 || 48)),
        foregroundConvergenceEpsilon: Math.max(0.005, Math.min(0.12, Number(options.cognitiveForegroundConvergenceEpsilon) || 0.03))
      
}
;


      this.cycles = 0;

      this.autonomousCycles = 0;

      this.conceptStudies = 0;

      this.relationConsolidations = 0;

      this.reflections = 0;

      this.interruptions = 0;

      this.totalMs = 0;

      this.lastCycleMs = 0;

      this.lastPhase = 'BOOTING';


      this.conceptCursor = 0;

      this.studyCounts = new Map();

      this.attention = new Map();

      this.thoughtHistory = [];

      this.lastThought = null;

      this.lastStudy = null;

      this.lastCorrelationTick = null;


      this.goalQueue = new Map();

      this.goalSequence = 0;

      this.goalsCompleted = 0;

      this.goalsFailed = 0;

      this.goalsDeferred = 0;

      this.curiosityQuestions = 0;

      this.contradictionInvestigations = 0;

      this.hypothesisInvestigations = 0;

      this.graphExplorations = 0;

      this.memoryReviews = 0;

      this.idleStreak = 0;

      this.executiveMode = 'ACTIVE';

      this.workspace = {

        currentGoal: null,
        focus: null,
        hypothesis: null,
        evidence: [],
        counterevidence: [],
        prediction: null,
        result: null,
        confidenceChange: 0,
        nextAction: null,
        knowledgeState: null,
        updatedAt: 0
      
}
;

      this.workspaceHistory = [];

      this.lastExecutiveCycle = null;
      this.foregroundThoughtCycles = 0;
      this.foregroundThoughtSessions = 0;
      this.foregroundConvergences = 0;
      this.lastForegroundThought = null;

      this.dirty = false;

    
}


    _array(value) {

      if (Array.isArray(value)) return value;

      if (value == null || value === '') return [];

      return [value];

    
}


    _phaseBudget(phase) {

      if (phase === 'DEEP_IDLE') return this.options.deepIdleBudgetMs;

      if (phase === 'IDLE') return this.options.idleBudgetMs;

      if (phase === 'WARM') return this.options.warmBudgetMs;

      return this.options.activeBudgetMs;

    
}


    _phaseConceptLimit(phase) {

      if (phase === 'IDLE') return this.options.idleConcepts;

      if (phase === 'WARM') return this.options.warmConcepts;

      return this.options.activeConcepts;

    
}


    _phaseStudyEvery(phase) {

      if (phase === 'IDLE') return this.options.idleStudyEvery;

      if (phase === 'WARM') return this.options.warmStudyEvery;

      return this.options.activeStudyEvery;

    
}


    _phaseGain(phase) {

      const multiplier =
        phase === 'IDLE' ? 1 :
        phase === 'WARM' ? 0.72 : 0.35;


      return clamp(
        this.options.baseStudyGain * multiplier,
        0.01,
        0.5
      );

    
}


    _interrupted(epochAtStart, getForegroundEpoch) {

      if (typeof getForegroundEpoch !== 'function') return false;


      const changed =
        getForegroundEpoch() !== epochAtStart;


      if (changed) this.interruptions++;

      return changed;

    
}


    _entryKey(entry) {

      return normalize(
        entry?.Subject ||
        entry?.ID ||
        ''
      );

    
}


    _trust(entry) {

      return clamp(entry?.Trust ?? 0.72);

    
}


    _eligible(entry) {

      if (!entry || typeof entry !== 'object') return false;

      if (entry.Decoder_Knowledge_Eligible === false) return false;


      const use = entry.Response_Use;

      if (!use || typeof use !== 'object') return true;


      return (
        use.General !== false ||
        use.Definition === true ||
        use.How === true ||
        use.Yes_No === true
      );

    
}


    _conceptUtility(entry) {

      if (!this._eligible(entry)) return -Infinity;


      const key = this._entryKey(entry);

      if (!key) return -Infinity;


      const studied = this.studyCounts.get(key) || 0;

      const attention = this.attention.get(key) || 0;


      const keywordCount = this._array(entry?.Keywords).length;

      const relationCount = this._array(entry?.Related_To).length;


      const detailLength =
        String(entry?.Description || '').length +
        String(entry?.Long_Definition || '').length +
        String(entry?.Mechanism || '').length;


      const novelty =
        1 / (1 + studied * 0.34);


      return (
        this._trust(entry) * 0.34 +
        novelty * 0.30 +
        Math.min(1, keywordCount / 10) * 0.10 +
        Math.min(1, relationCount / 8) * 0.10 +
        Math.min(1, detailLength / 1000) * 0.08 +
        attention * 0.08
      );

    
}


    _selectConcept() {

      const entries =
        Array.isArray(this.trainingKnowledge?.entries)
          ? this.trainingKnowledge.entries
          : [];


      if (!entries.length) return null;


      const lookahead =
        Math.min(entries.length, this.options.conceptLookahead);


      let best = null;

      let bestIndex = -1;

      let bestUtility = -Infinity;


      for (let offset = 0;
 offset < lookahead;
 offset++) {

        const index =
          (this.conceptCursor + offset) % entries.length;


        const entry = entries[index];

        const utility = this._conceptUtility(entry);


        if (utility > bestUtility) {

          best = entry;

          bestIndex = index;

          bestUtility = utility;

        
}

      
}


      this.conceptCursor =
        ((bestIndex >= 0 ? bestIndex : this.conceptCursor) + 1) %
        entries.length;


      return best;

    
}


    _conceptText(entry) {

      if (!entry) return '';


      const subject =
        String(entry?.Subject || '').trim();


      const keywords =
        this._array(entry?.Keywords)
          .map(value => String(value || '').trim())
          .filter(Boolean);


      const related =
        this._array(entry?.Related_To)
          .map(value => String(value || '').trim())
          .filter(Boolean);


      const aliases =
        this._array(entry?.Aliases)
          .map(value => String(value || '').trim())
          .filter(Boolean);


      const parts = [
        subject,
        String(entry?.Description || '').trim(),
        String(entry?.Long_Definition || '').trim(),
        String(entry?.Purpose || '').trim(),
        String(entry?.Mechanism || '').trim(),
        ...this._array(entry?.Capabilities),
        ...this._array(entry?.Limitations),
        ...this._array(entry?.Examples),
        ...this._array(entry?.Contrasts_With),
        ...this._array(entry?.Part_Of),
        ...this._array(entry?.Has_Parts),
        ...this._array(entry?.Causes),
        ...this._array(entry?.Effects),
        ...this._array(entry?.Properties),
        entry?.AI_Subdomain,
        ...this._array(entry?.Evaluation_Criteria),
        ...this._array(entry?.Failure_Modes),
        ...this._array(entry?.Tradeoffs)
      ];


      if (aliases.length) {

        parts.push(`${subject} aliases ${aliases.join(' ')}`);

      
}


      if (keywords.length) {

        parts.push(`${subject} keywords ${keywords.join(' ')}`);

      
}


      if (related.length) {

        parts.push(`${subject} related to ${related.join(' ')}`);

      
}


      return parts
        .filter(Boolean)
        .join('. ')
        .slice(0, this.options.maxConceptChars);

    
}


    _touchAttention(key, value) {

      if (!key) return;


      const previous =
        this.attention.get(key) || 0;


      this.attention.set(
        key,
        clamp(previous * 0.78 + clamp(value) * 0.22)
      );


      if (this.attention.size > this.options.attentionCapacity) {

        let weakestKey = null;

        let weakestValue = Infinity;


        for (const [candidateKey, score] of this.attention) {

          if (score < weakestValue) {

            weakestValue = score;

            weakestKey = candidateKey;

          
}

        
}


        if (weakestKey) this.attention.delete(weakestKey);

      
}

    
}


    _decayAttention() {

      for (const [key, value] of this.attention) {

        const next = value * this.options.attentionDecay;


        if (next < 0.005) {

          this.attention.delete(key);

        
}
 else {

          this.attention.set(key, next);

        
}

      
}

    
}


    _pushThought(thought) {

      this.lastThought = thought;

      this.thoughtHistory.push(thought);


      if (
        this.thoughtHistory.length >
        this.options.thoughtHistoryCapacity
      ) {

        this.thoughtHistory.splice(
          0,
          this.thoughtHistory.length -
          this.options.thoughtHistoryCapacity
        );

      
}

    
}


    _reflect(entry) {

      const subject = this._entryKey(entry);

      if (!subject) return null;


      const candidates =
        Array.from(
          new Set([
            ...this._array(entry?.Related_To),
            ...this._array(entry?.Keywords),
            ...this._array(entry?.Aliases)
          ]
            .map(normalize)
            .filter(Boolean)
          )
        )
          .slice(0, this.options.maxRelationTargets);


      let strongest = null;

      let strongestSupport = 0;

      let totalSupport = 0;


      for (const target of candidates) {

        const support = clamp(
          this.rsl?.correlationScore?.(subject, target) ||
          this.processor?.associationScore?.(
            subject,
            target,
            {
 bidirectional: true 
}

          ) ||
          0
        );


        totalSupport += support;


        if (support > strongestSupport) {

          strongestSupport = support;

          strongest = target;

        
}

      
}


      const meanSupport =
        candidates.length
          ? totalSupport / candidates.length
          : 0;


      const studied =
        this.studyCounts.get(subject) || 0;


      const novelty =
        1 / (1 + studied);


      const thought = {

        at: Date.now(),
        subject,
        strongestRelation: strongest,
        strongestSupport:
          Number(strongestSupport.toFixed(4)),
        meanSupport:
          Number(meanSupport.toFixed(4)),
        novelty:
          Number(novelty.toFixed(4)),
        trust:
          Number(this._trust(entry).toFixed(4)),
        source:
          'internal-traininginfo-reflection'
      
}
;


      this.reflections++;


      this._touchAttention(
        subject,
        clamp(
          novelty * 0.48 +
          (1 - meanSupport) * 0.34 +
          this._trust(entry) * 0.18
        )
      );


      this._pushThought(thought);

      return thought;

    
}


    _consolidateRelations(entry, gain) {

      const subject = this._entryKey(entry);

      if (!subject) return {
 updates: 0 
}
;


      const targets =
        Array.from(
          new Set([
            ...this._array(entry?.Keywords),
            ...this._array(entry?.Aliases),
            ...this._array(entry?.Related_To)
          ]
            .map(normalize)
            .filter(Boolean)
          )
        )
          .slice(0, this.options.maxRelationTargets);


      let updates = 0;


      for (const target of targets) {

        if (!target || target === subject) continue;


        const learned =
          this.rsl?.learnCorrelation?.(
            subject,
            target,
            {

              gain:
                gain *
                this.options.relationGain,
              source:
                'cognitive_traininginfo',
              symmetric: true,
              reverseGain: 0.72
            
}

          );


        if (learned !== false) updates++;

      
}


      this.relationConsolidations += updates;


      return {

        updates,
        targetCount: targets.length
      
}
;

    
}


    _studyConcept(entry, phase) {

      if (!entry) return null;


      const subject = this._entryKey(entry);

      const text = this._conceptText(entry);


      if (!subject || !text) return null;


      const trust = this._trust(entry);


      const gain = clamp(
        this._phaseGain(phase) *
        (0.70 + trust * 0.30),
        0.01,
        0.45
      );


      // Only existing TrainingInfo content is learned here. The advanced
      // learner receives the same trusted entry as structured evidence, but
      // only queues it; heavy graph/hypothesis/replay work remains background.
      const advancedQueued =
        this.learningController
          ?.observeTrustedKnowledge?.(
            entry,
            {

              trust,
              provenance: 'traininginfo'
            
}

          ) ||
        false;


      const learned =
        this.correlation?.observeText?.(
          text,
          {

            source: 'cognitive_traininginfo',
            confidence: trust * 100,
            accepted: true,
            episodic: false,
            countCycle: false,
            gain
          
}

        ) || null;


      const relation =
        this._consolidateRelations(
          entry,
          gain
        );


      const previousCount =
        this.studyCounts.get(subject) || 0;


      this.studyCounts.set(
        subject,
        Math.min(65535, previousCount + 1)
      );


      this.conceptStudies++;

      this.dirty = true;


      const reflection =
        this._reflect(entry);


      this.lastStudy = {

        at: Date.now(),
        phase,
        subject,
        gain:
          Number(gain.toFixed(5)),
        trust:
          Number(trust.toFixed(4)),
        tokens:
          Number(learned?.tokens) || 0,
        pairUpdates:
          Number(learned?.updates) || 0,
        relationUpdates:
          Number(relation?.updates) || 0,
        advancedLearningQueued:
          Boolean(advancedQueued),
        reflection
      
}
;


      return this.lastStudy;

    
}


    _executiveMode(phase) {

      if (phase === 'IDLE') {

        this.idleStreak++;

        return this.idleStreak >= this.options.deepIdleAfterCycles ? 'DEEP_IDLE' : 'IDLE';

      
}

      this.idleStreak = 0;

      return phase === 'WARM' ? 'WARM' : 'ACTIVE';

    
}


    _goalId(type, target = '', relation = '', to = '') {

      return [type, normalize(target), normalize(relation), normalize(to)].join('|');

    
}


    enqueueGoal(goal = {
}
) {

      const type = String(goal.type || 'study_concept');

      const target = normalize(goal.target || goal.subject || '');

      const relation = normalize(goal.relation || '');

      const to = normalize(goal.to || '');

      const id = goal.id || this._goalId(type, target, relation, to) || `goal:${++this.goalSequence}`;

      const old = this.goalQueue.get(id);

      const merged = {

        id,
        type,
        target,
        relation,
        to,
        priority: clamp(Math.max(Number(old?.priority) || 0, Number(goal.priority) || 0.5)),
        uncertainty: clamp(Math.max(Number(old?.uncertainty) || 0, Number(goal.uncertainty) || 0.5)),
        expectedGain: clamp(Math.max(Number(old?.expectedGain) || 0, Number(goal.expectedGain) || 0.5)),
        estimatedCostMs: Math.max(0.2, Number(goal.estimatedCostMs ?? old?.estimatedCostMs) || 2),
        attempts: Math.max(0, Number(old?.attempts) || 0),
        createdAt: old?.createdAt || Date.now(),
        lastWorkedAt: old?.lastWorkedAt || 0,
        status: old?.status || 'queued',
        source: String(goal.source || old?.source || 'cognitive'),
        metadata: {
 ...(old?.metadata || {
}
), ...(goal.metadata || {
}
) 
}

      
}
;

      this.goalQueue.set(id, merged);

      if (this.goalQueue.size > this.options.goalCapacity) {

        const victims = Array.from(this.goalQueue.values())
          .filter(item => item.status !== 'active')
          .sort((a, b) => this._goalUtility(a) - this._goalUtility(b));

        for (const victim of victims.slice(0, this.goalQueue.size - this.options.goalCapacity)) this.goalQueue.delete(victim.id);

      
}

      this.dirty = true;

      return merged;

    
}


    _goalUtility(goal) {

      const attemptPenalty = 1 / (1 + Math.max(0, goal?.attempts || 0) * 0.18);

      const value =
        clamp(goal?.priority ?? 0.5) * 0.36 +
        clamp(goal?.uncertainty ?? 0.5) * 0.24 +
        clamp(goal?.expectedGain ?? 0.5) * 0.40;

      return value * attemptPenalty / Math.max(0.4, Number(goal?.estimatedCostMs) || 2);

    
}


    _goalLimit(mode) {

      if (mode === 'DEEP_IDLE') return this.options.deepIdleGoalLimit;

      if (mode === 'IDLE') return this.options.idleGoalLimit;

      if (mode === 'WARM') return this.options.warmGoalLimit;

      return this.options.activeGoalLimit;

    
}


    _harvestGoals(mode) {

      let added = 0;

      const add = goal => {

        if (added >= this.options.goalHarvestLimit) return;

        if ((Number(goal.priority) || 0) < this.options.goalMinPriority) return;

        this.enqueueGoal(goal);

        added++;

      
}
;


      for (const edge of this.knowledgeGraph?.topEdges?.(48) || []) {

        if (added >= this.options.goalHarvestLimit) break;

        if (edge.status === 'contested' || edge.contradictionPressure >= 0.24) {

          const k = this.uncertaintyState?.assessKnowledgeTarget?.(edge) || {
}
;

          add({

            type: 'resolve_contradiction',
            target: edge.from,
            relation: edge.relation,
            to: edge.to,
            priority: 0.90,
            uncertainty: k.uncertainty ?? 0.8,
            expectedGain: 0.88,
            estimatedCostMs: 2.2,
            source: 'knowledge-graph'
          
}
);

        
}

      
}


      for (const item of this.hypothesisStore?.candidates?.(12, ['promotable', 'provisional', 'contested']) || []) {

        if (added >= this.options.goalHarvestLimit) break;

        add({

          type: 'test_hypothesis',
          target: item.from,
          relation: item.relation,
          to: item.to,
          priority: item.status === 'promotable' ? 0.88 : item.status === 'contested' ? 0.84 : 0.60,
          uncertainty: clamp(1 - Math.abs((item.confidence ?? 0.5) - 0.5) * 2),
          expectedGain: item.status === 'promotable' ? 0.90 : 0.68,
          estimatedCostMs: 2.8,
          source: 'hypothesis-store'
        
}
);

      
}


      const prediction = this.predictionError?.last;

      if (prediction && Math.abs(Number(prediction.absError) || 0) >= 0.18) {

        add({

          type: 'investigate_prediction_error',
          target: prediction.from,
          relation: prediction.relation,
          to: prediction.to,
          priority: clamp(0.55 + prediction.absError * 0.40),
          uncertainty: clamp(prediction.absError),
          expectedGain: clamp(0.55 + prediction.gain * 0.35),
          estimatedCostMs: 2.5,
          source: 'prediction-error',
          metadata: {
 prediction 
}

        
}
);

      
}


      for (const row of this.temporalKnowledge?.staleEdges?.(this.knowledgeGraph, 8) || []) {

        if (added >= this.options.goalHarvestLimit) break;

        if (row.temporal.freshness >= this.options.staleReviewFreshness) continue;

        add({

          type: 'refresh_stale',
          target: row.edge.from,
          relation: row.edge.relation,
          to: row.edge.to,
          priority: clamp(0.52 + (1 - row.temporal.freshness) * 0.36),
          uncertainty: clamp(1 - row.temporal.freshness),
          expectedGain: 0.66,
          estimatedCostMs: 3.2,
          source: 'temporal-knowledge'
        
}
);

      
}


      if (this.learningEvaluator?.lastRegression?.regression) {

        add({

          type: 'learning_regression',
          target: 'adaptive-learning',
          priority: 0.94,
          uncertainty: 0.78,
          expectedGain: 0.92,
          estimatedCostMs: 2,
          source: 'learning-evaluator',
          metadata: {
 regression: this.learningEvaluator.lastRegression 
}

        
}
);

      
}


      const attentionRows = Array.from(this.attention.entries()).sort((a, b) => b[1] - a[1]).slice(0, 6);

      for (const [subject, attention] of attentionRows) {

        if (added >= this.options.goalHarvestLimit) break;

        add({

          type: mode === 'DEEP_IDLE' ? 'graph_explore' : 'study_concept',
          target: subject,
          priority: clamp(0.35 + attention * 0.45),
          uncertainty: clamp(1 - attention * 0.45),
          expectedGain: clamp(0.40 + (1 - attention) * 0.35),
          estimatedCostMs: mode === 'DEEP_IDLE' ? 3.8 : 2.2,
          source: 'attention'
        
}
);

      
}


      if (!this.goalQueue.size && mode !== 'ACTIVE') {

        const entry = this._selectConcept();

        if (entry) {

          add({

            type: 'study_concept',
            target: this._entryKey(entry),
            priority: 0.48,
            uncertainty: 0.52,
            expectedGain: 0.55,
            estimatedCostMs: 2.2,
            source: 'traininginfo-fallback',
            metadata: {
 entryId: entry.ID || null 
}

          
}
);

        
}

      
}


      return added;

    
}


    _findTrainingEntry(subject) {

      const key = normalize(subject);

      if (!key) return null;

      const entries = this.trainingKnowledge?.entries || [];

      return entries.find(entry => normalize(entry?.Subject) === key || normalize(entry?.ID) === key) || null;

    
}


    _workspaceSnapshot() {

      const snapshot = JSON.parse(JSON.stringify(this.workspace));

      this.workspaceHistory.push(snapshot);

      if (this.workspaceHistory.length > this.options.workspaceHistoryCapacity) {

        this.workspaceHistory.splice(0, this.workspaceHistory.length - this.options.workspaceHistoryCapacity);

      
}

      return snapshot;

    
}


    _setWorkspace(goal, result = null, extras = {
}
) {

      let edge = null;

      if (goal?.target && goal?.relation && goal?.to) {

        edge = this.knowledgeGraph?.getEdge?.(goal.target, goal.relation, goal.to) || null;

      
}

      const knowledgeState = this.uncertaintyState?.assessKnowledgeTarget?.(edge) || null;

      this.workspace = {

        currentGoal: goal ? {
 id: goal.id, type: goal.type, target: goal.target, relation: goal.relation, to: goal.to 
}
 : null,
        focus: goal?.target || null,
        hypothesis: goal?.type === 'test_hypothesis' ? {
 from: goal.target, relation: goal.relation, to: goal.to 
}
 : null,
        evidence: extras.evidence || [],
        counterevidence: extras.counterevidence || [],
        prediction: extras.prediction || goal?.metadata?.prediction || null,
        result,
        confidenceChange: Number(extras.confidenceChange) || 0,
        nextAction: extras.nextAction || null,
        knowledgeState,
        updatedAt: Date.now()
      
}
;

      this._workspaceSnapshot();

      return this.workspace;

    
}


    _executeGoal(goal, mode) {

      if (!goal) return {
 success: false, reason: 'missing-goal' 
}
;

      goal.status = 'active';

      goal.attempts++;

      goal.lastWorkedAt = Date.now();

      let result = null;

      let success = true;

      let nextAction = null;


      if (goal.type === 'resolve_contradiction') {

        result = this.learningController?.runContradictionCheck?.() || this.contradictionResolver?.run?.() || null;

        this.contradictionInvestigations++;

        nextAction = result?.contested ? 'retest-contested-relations' : 'monitor';

      
}
 else if (goal.type === 'test_hypothesis') {

        result = this.learningController?.testHypotheses?.(1) || null;

        this.hypothesisInvestigations++;

        nextAction = result?.confirmed ? 'consolidate' : result?.contradicted ? 'lower-hypothesis' : 'collect-more-evidence';

      
}
 else if (goal.type === 'investigate_prediction_error') {

        const reasoning = this.graphReasoner?.reason?.([goal.target, goal.to].filter(Boolean), {
 maxDepth: mode === 'DEEP_IDLE' ? 3 : 2 
}
) || null;

        this.replayScheduler?.schedule?.(
          `cognitive-error:${goal.id}`,
          {
 subject: goal.target, relation: goal.relation, target: goal.to 
}
,
          {
 uncertainty: goal.uncertainty, usefulness: goal.priority, predictionError: goal.metadata?.prediction?.absError || goal.uncertainty, novelty: 0.5, contradictionPressure: 0 
}

        );

        result = reasoning;

        nextAction = reasoning?.derived?.length ? 'test-derived-relations' : 'replay-and-observe';

      
}
 else if (goal.type === 'refresh_stale') {

        const entry = this._findTrainingEntry(goal.target);

        if (entry) {

          this.learningController?.observeTrustedKnowledge?.(entry, {
 trust: entry.Trust ?? 0.96 
}
);

          result = {
 queuedTrustedRefresh: true, entryId: entry.ID || null 
}
;

          nextAction = 'verify-after-refresh';

        
}
 else {

          success = false;

          result = {
 queuedTrustedRefresh: false, reason: 'no-trusted-entry' 
}
;

          nextAction = 'remain-stale';

        
}

      
}
 else if (goal.type === 'graph_explore') {

        result = this.graphReasoner?.reason?.([goal.target], {
 maxDepth: mode === 'DEEP_IDLE' ? 3 : 2 
}
) || null;

        this.graphExplorations++;

        if (!(result?.derived?.length || result?.paths?.length)) {

          this.curiosityQuestions++;

          nextAction = 'unknown-no-supported-path';

        
}
 else {

          nextAction = result?.derived?.length ? 'queue-hypothesis-tests' : 'retain-as-relevance-only';

        
}

      
}
 else if (goal.type === 'memory_review') {

        result = this.memory?.retrieve?.(goal.target, {
}
, 6) || [];

        this.memoryReviews++;

        nextAction = result.length ? 'compare-memory-with-current-knowledge' : 'no-memory-evidence';

      
}
 else if (goal.type === 'learning_regression') {

        result = this.learningController?.benchmarkLearning?.() || this.learningEvaluator?.lastRegression || null;

        nextAction = 'prefer-trusted-replay-and-monitor';

      
}
 else {

        const entry = this._findTrainingEntry(goal.target) || this._selectConcept();

        if (entry) {

          result = this._studyConcept(entry, mode === 'DEEP_IDLE' ? 'IDLE' : mode);

          nextAction = 'reflect-and-retest';

        
}
 else {

          success = false;

          result = {
 reason: 'no-study-target' 
}
;

          nextAction = 'drop-goal';

        
}

      
}


      this._setWorkspace(goal, result, {
 nextAction 
}
);

      const thought = {

        at: Date.now(),
        source: 'cognitive-executive',
        goalType: goal.type,
        target: goal.target,
        relation: goal.relation || null,
        to: goal.to || null,
        success,
        nextAction,
        knowledgeState: this.workspace.knowledgeState
      
}
;

      this._pushThought(thought);


      if (success) {

        goal.status = 'complete';

        this.goalsCompleted++;

        this.goalQueue.delete(goal.id);

      
}
 else if (goal.attempts >= 3) {

        goal.status = 'failed';

        this.goalsFailed++;

        this.goalQueue.delete(goal.id);

      
}
 else {

        goal.status = 'queued';

        goal.priority = clamp(goal.priority * 0.82);

        this.goalsDeferred++;

      
}

      this.dirty = true;

      return {
 success, result, nextAction 
}
;

    
}


    thinkForeground(promptText = '', analysis = {}, options = {}) {
      const started = performance.now();
      const deadlineAt = Number.isFinite(Number(options.deadlineAt)) ? Number(options.deadlineAt) : Infinity;
      const maxCycles = Math.max(2, Math.min(this.options.foregroundThinkingCycles, Number(options.maxCycles) || this.options.foregroundThinkingCycles));
      const baseContext = this.foregroundContext(promptText, analysis);
      let packet = analysis.cognitivePacket || this.processor?.buildCognitivePacket?.(analysis, { correlation: this.correlation }) || { cycle:0, activeConcepts:analysis.contentWords||[], relations:[], unresolved:analysis.requestedSlots||[], metrics:{progress:.35} };
      const cycles=[]; let previousProgress=Number(packet?.metrics?.progress||0); let converged=Boolean(packet?.converged);
      for(let cycle=0; cycle<maxCycles && !converged; cycle++){
        if(performance.now()>=deadlineAt) break;
        const seeds=Array.from(new Set([...(packet.activeConcepts||[]),...(packet.seeds||[]),...(baseContext.attention||[]).slice(0,6).map(row=>row.concept)].map(normalize).filter(Boolean))).slice(0,40);
        const relations=this.correlation?.relationCandidates?.(seeds,this.options.foregroundRelationLimit)||[];
        const discoveries=Array.from(new Set(relations.flatMap(row=>[row.from,row.to]).map(normalize).filter(Boolean)));
        const strongest=relations[0]||null; const unresolvedBefore=(packet.unresolved||[]).slice();
        packet=this.processor?.advanceCognitivePacket?.(packet,{relations,concepts:discoveries,convergenceEpsilon:this.options.foregroundConvergenceEpsilon})||packet;
        const progress=Number(packet?.metrics?.progress||previousProgress), progressDelta=Math.abs(progress-previousProgress); previousProgress=progress;
        converged=Boolean(packet?.converged||(cycle>=1&&progressDelta<this.options.foregroundConvergenceEpsilon&&!relations.length));
        const thought={at:Date.now(),source:'foreground-cognitive-cycle',cycle:cycle+1,subject:normalize((packet.activeConcepts||[])[0]||(packet.seeds||[])[0]||''),strongestRelation:normalize(strongest?.relation||''),relationFrom:normalize(strongest?.from||''),relationTo:normalize(strongest?.to||''),strongestSupport:Number(strongest?.score??strongest?.strength??0),meanSupport:relations.length?relations.reduce((sum,row)=>sum+Number(row?.score??row?.strength??0),0)/relations.length:0,novelty:clamp(discoveries.filter(value=>!(packet.seeds||[]).includes(value)).length/12),trust:clamp(relations.length?relations.reduce((sum,row)=>sum+Number(row?.confidence??.5),0)/relations.length:.5),unresolved:(packet.unresolved||[]).slice(0,10),resolvedThisCycle:unresolvedBefore.filter(value=>!(packet.unresolved||[]).includes(value)),progress,progressDelta,converged};
        cycles.push(thought); this._pushThought(thought); for(const concept of discoveries.slice(0,12))this._touchAttention(concept,clamp(.36+thought.trust*.34)); this.foregroundThoughtCycles++;
      }
      this.foregroundThoughtSessions++; if(converged)this.foregroundConvergences++;
      const state={version:3,mode:'iterative-foreground-cognition',languageStructure:{clauses:analysis?.clauseGraph?.length||0,propositions:analysis?.propositions?.length||0,relations:(analysis?.propositionRelations||[]).slice(0,16),references:(analysis?.referenceLinks||[]).slice(0,12),structuralComplexity:Number(analysis?.structuralComplexity||0)},structuredReasoning:{propositionCoverage:Number(analysis?.propositionGraph?.metrics?.structuralCoverage||0),propositions:(analysis?.propositionGraph?.propositions||[]).slice(0,16),mechanismActive:Boolean(analysis?.mechanismGraph?.active),mechanismStages:(analysis?.mechanismGraph?.stages||[]).slice(0,16),mechanismCompleteness:Number(analysis?.mechanismGraph?.completeness||0),mechanismMissing:(analysis?.mechanismGraph?.missing||[]).slice(0,8),reasoningMemory:(analysis?.reasoningMemoryHints||[]).slice(0,4).map(row=>({mode:row.mode,score:row.score,successRate:row.successRate}))},prompt:String(promptText||'').slice(0,320),cycles,cycleCount:cycles.length,activeConcepts:(packet.activeConcepts||[]).slice(0,72),relations:(packet.relations||[]).slice(0,56),unresolved:(packet.unresolved||[]).slice(0,16),progress:Number(packet?.metrics?.progress||0),converged,background:baseContext,elapsedMs:performance.now()-started};
      this.workspace.focus=state.activeConcepts[0]||this.workspace.focus; this.workspace.evidence=state.relations.slice(0,12); this.workspace.nextAction=state.unresolved[0]||null; this.workspace.result=converged?'foreground-converged':'foreground-open'; this.workspace.updatedAt=Date.now(); this.lastForegroundThought=state; this.dirty=true; return state;
    }

    foregroundContext(promptText = '', analysis = {}) {

      const promptConcepts = Array.from(new Set([
        ...(analysis?.contentWords || []),
        ...(analysis?.entityConcepts || []),
        ...(analysis?.architectureConcepts || []),
        ...(analysis?.entities || []).map(row => row?.canonical),
        ...(analysis?.architectureComponents || []).map(row => row?.canonical)
      ].map(normalize).filter(Boolean)));

      const conceptTokens = new Set(
        promptConcepts.flatMap(value => value.split(/\s+/).filter(Boolean))
      );

      const attention = Array.from(this.attention.entries())
        .map(([concept, score]) => ({ concept, score: clamp(score) }))
        .sort((a, b) => {
          const aMatch = conceptTokens.has(a.concept) || promptConcepts.some(seed => a.concept.includes(seed) || seed.includes(a.concept));
          const bMatch = conceptTokens.has(b.concept) || promptConcepts.some(seed => b.concept.includes(seed) || seed.includes(b.concept));
          if (aMatch !== bMatch) return aMatch ? -1 : 1;
          return b.score - a.score;
        })
        .slice(0, 12);

      const relevance = thought => {
        const fields = [thought?.subject, thought?.strongestRelation, thought?.target, thought?.relation, thought?.to]
          .map(normalize)
          .filter(Boolean);
        if (!fields.length || !promptConcepts.length) return 0;
        let best = 0;
        for (const field of fields) {
          for (const seed of promptConcepts) {
            if (field === seed) best = Math.max(best, 1);
            else if (field.includes(seed) || seed.includes(field)) best = Math.max(best, 0.82);
            else {
              const A = new Set(field.split(/\s+/));
              const B = new Set(seed.split(/\s+/));
              let hit = 0;
              for (const token of A) if (B.has(token)) hit++;
              const union = A.size + B.size - hit;
              if (union) best = Math.max(best, hit / union);
            }
          }
        }
        return best;
      };

      const thoughts = this.thoughtHistory
        .map(row => ({ row, relevance: relevance(row) }))
        .filter(item => item.relevance > 0 || promptConcepts.length === 0)
        .sort((a, b) => b.relevance - a.relevance || (Number(b.row?.at) || 0) - (Number(a.row?.at) || 0))
        .slice(0, 10)
        .map(item => ({ ...item.row, relevance: Number(item.relevance.toFixed(4)) }));

      return {
        prompt: String(promptText || '').slice(0, 256),
        concepts: promptConcepts.slice(0, 32),
        attention,
        thoughts,
        workspace: JSON.parse(JSON.stringify(this.workspace || {})),
        lastThought: this.lastThought ? { ...this.lastThought } : null,
        executiveMode: this.executiveMode,
        queuedGoals: this.goalQueue?.size || 0,
        foreground: this.lastForegroundThought ? {
          cycleCount: this.lastForegroundThought.cycleCount,
          activeConcepts: (this.lastForegroundThought.activeConcepts || []).slice(0, 16),
          relations: (this.lastForegroundThought.relations || []).slice(0, 12),
          unresolved: (this.lastForegroundThought.unresolved || []).slice(0, 8),
          progress: this.lastForegroundThought.progress,
          converged: this.lastForegroundThought.converged
        } : null
      };
    }


    observeReasoningPlan(plan = {}) {

      if (!plan || !Array.isArray(plan.priorityTerms)) return false;

      const selected = plan.selectedState || null;
      const uncertainty = clamp(plan?.matrixState?.uncertainty ?? 0.5);

      for (const concept of (plan.priorityTerms || []).slice(0, 12)) {
        const key = normalize(concept);
        if (!key) continue;
        this._touchAttention(key, clamp(0.42 + uncertainty * 0.28));
      }

      this._pushThought({
        at: Date.now(),
        subject: normalize(plan.priorityTerms?.[0] || plan.seeds?.[0] || ''),
        strongestRelation: normalize(selected?.mode || ''),
        strongestSupport: Number(selected?.score || 0),
        meanSupport: Number(plan?.matrixState?.scoreMargin || 0),
        novelty: clamp((plan.unresolved?.length || 0) / 4),
        trust: clamp(selected?.certainty ?? 0.5),
        unresolved: (plan.unresolved || []).slice(0, 8),
        source: 'foreground-reasoning-plan'
      });

      if ((plan.unresolved || []).length) {
        const target = normalize(plan.priorityTerms?.[0] || plan.seeds?.[0] || 'reasoning-quality');
        this.enqueueGoal({
          type: 'study_concept',
          target,
          priority: clamp(0.48 + uncertainty * 0.34),
          uncertainty,
          expectedGain: clamp(0.55 + uncertainty * 0.28),
          estimatedCostMs: 2.0,
          source: 'foreground-reasoning-unresolved',
          metadata: { unresolved: (plan.unresolved || []).slice(0, 8) }
        });
      }

      return true;
    }


    observeForeground(event = {
}
) {

      const subject = normalize(event.subject || event.analysisSummary?.contentWords?.[0] || '');

      if (subject) {

        this._touchAttention(subject, clamp(event.uncertainty ?? 0.55));

      
}

      if (event.deadlineMiss) {

        this.enqueueGoal({

          type: 'study_concept', target: subject || 'response-quality', priority: 0.72,
          uncertainty: 0.66, expectedGain: 0.68, estimatedCostMs: 2.0, source: 'foreground-deadline-miss'
        
}
);

      
}

      if (event.uncertaintyState?.state === 'contested' && subject) {

        this.enqueueGoal({

          type: 'resolve_contradiction', target: subject, priority: 0.88,
          uncertainty: 0.82, expectedGain: 0.88, estimatedCostMs: 2.2, source: 'foreground-uncertainty'
        
}
);

      
}

      return true;

    
}


    async backgroundTick(
      phase = 'WARM',
      epochAtStart = 0,
      getForegroundEpoch = null
    ) {

      const started = performance.now();

      if (!this.options.enabled) return {
 enabled: false, phase, ms: 0 
}
;


      this.cycles++;

      this.autonomousCycles++;

      this.lastPhase = phase;

      const mode = this._executiveMode(phase);

      this.executiveMode = mode;


      // Existing autonomous learning remains coordinated here.
      this.rsl?.backgroundTick?.(phase);

      if (this._interrupted(epochAtStart, getForegroundEpoch)) {

        return this._finish(started, mode, 0, true);

      
}


      this.lastCorrelationTick = this.correlation?.backgroundTick?.(
        phase,
        epochAtStart,
        getForegroundEpoch
      ) || null;


      if (this._interrupted(epochAtStart, getForegroundEpoch)) {

        return this._finish(started, mode, 0, true);

      
}


      const budget = this._phaseBudget(mode);

      const studyEvery = this._phaseStudyEvery(phase);

      let studied = 0;

      let goalsExecuted = 0;


      this._harvestGoals(mode);


      const goalLimit = this._goalLimit(mode);

      while (
        goalsExecuted < goalLimit &&
        performance.now() - started < budget &&
        !this._interrupted(epochAtStart, getForegroundEpoch)
      ) {

        const queued = Array.from(this.goalQueue.values())
          .filter(goal => goal.status === 'queued')
          .sort((a, b) => this._goalUtility(b) - this._goalUtility(a));

        const goal = queued[0];

        if (!goal) break;

        this._executeGoal(goal, mode);

        goalsExecuted++;

      
}


      // Preserve the original broad TrainingInfo self-study path as a fallback
      // and as the ACTIVE-mode low-cost learning behavior.
      const conceptLimit = this._phaseConceptLimit(phase);

      if (
        conceptLimit > 0 &&
        this.cycles % studyEvery === 0 &&
        performance.now() - started < budget
      ) {

        for (let i = 0;
 i < conceptLimit;
 i++) {

          if (performance.now() - started >= budget) break;

          if (this._interrupted(epochAtStart, getForegroundEpoch)) break;

          const entry = this._selectConcept();

          if (!entry) break;

          if (this._studyConcept(entry, phase)) studied++;

        
}

      
}


      if (this.cycles % 8 === 0) this._decayAttention();


      this.lastExecutiveCycle = {

        at: Date.now(),
        phase,
        mode,
        harvestedGoals: this.goalQueue.size,
        goalsExecuted,
        studied,
        budgetMs: budget,
        usedMs: performance.now() - started,
        workspace: this.workspace
      
}
;


      return this._finish(
        started,
        mode,
        studied,
        this._interrupted(epochAtStart, getForegroundEpoch)
      );

    
}


    _finish(
      started,
      phase,
      studied,
      interrupted
    ) {

      const ms =
        performance.now() -
        started;


      this.lastCycleMs = ms;

      this.totalMs += ms;


      return {

        enabled: this.options.enabled,
        phase,
        cycle: this.cycles,
        studied,
        interrupted,
        ms,
        conceptStudies:
          this.conceptStudies,
        relationConsolidations:
          this.relationConsolidations,
        reflections:
          this.reflections,
        attentionSize:
          this.attention.size,
        lastThought:
          this.lastThought,
        lastStudy:
          this.lastStudy,
        correlation:
          this.lastCorrelationTick,
        executiveMode: this.executiveMode,
        queuedGoals: this.goalQueue.size,
        workspace: this.workspace,
        lastExecutiveCycle: this.lastExecutiveCycle
      
}
;

    
}


    exportState() {

      return {

        schemaVersion: 2,
        cycles: this.cycles,
        autonomousCycles:
          this.autonomousCycles,
        conceptStudies:
          this.conceptStudies,
        relationConsolidations:
          this.relationConsolidations,
        reflections:
          this.reflections,
        interruptions:
          this.interruptions,
        totalMs: this.totalMs,
        conceptCursor:
          this.conceptCursor,

        studyCounts:
          Array.from(
            this.studyCounts.entries()
          )
            .sort((a, b) => b[1] - a[1])
            .slice(
              0,
              this.options.attentionCapacity
            ),

        attention:
          Array.from(
            this.attention.entries()
          )
            .sort((a, b) => b[1] - a[1])
            .slice(
              0,
              this.options.attentionCapacity
            ),

        thoughtHistory:
          this.thoughtHistory.slice(
            -this.options
              .thoughtHistoryCapacity
          ),

        lastThought:
          this.lastThought,

        lastStudy:
          this.lastStudy,
        goals: Array.from(this.goalQueue.values()).slice(0, this.options.goalCapacity),
        goalSequence: this.goalSequence,
        goalsCompleted: this.goalsCompleted,
        goalsFailed: this.goalsFailed,
        goalsDeferred: this.goalsDeferred,
        curiosityQuestions: this.curiosityQuestions,
        contradictionInvestigations: this.contradictionInvestigations,
        hypothesisInvestigations: this.hypothesisInvestigations,
        graphExplorations: this.graphExplorations,
        memoryReviews: this.memoryReviews,
        idleStreak: this.idleStreak,
        executiveMode: this.executiveMode,
        workspace: this.workspace,
        workspaceHistory: this.workspaceHistory.slice(-this.options.workspaceHistoryCapacity),
        lastExecutiveCycle: this.lastExecutiveCycle,
        foregroundThoughtCycles: this.foregroundThoughtCycles,
        foregroundThoughtSessions: this.foregroundThoughtSessions,
        foregroundConvergences: this.foregroundConvergences,
        lastForegroundThought: this.lastForegroundThought
      
}
;

    
}


    importState(state) {

      if (
        !state ||
        ![1, 2].includes(Number(state.schemaVersion))
      ) {

        return false;

      
}


      this.cycles =
        Math.max(
          0,
          Number(state.cycles) || 0
        );


      this.autonomousCycles =
        Math.max(
          0,
          Number(state.autonomousCycles) || 0
        );


      this.conceptStudies =
        Math.max(
          0,
          Number(state.conceptStudies) || 0
        );


      this.relationConsolidations =
        Math.max(
          0,
          Number(
            state.relationConsolidations
          ) || 0
        );


      this.reflections =
        Math.max(
          0,
          Number(state.reflections) || 0
        );


      this.interruptions =
        Math.max(
          0,
          Number(state.interruptions) || 0
        );


      this.totalMs =
        Math.max(
          0,
          Number(state.totalMs) || 0
        );


      this.conceptCursor =
        Math.max(
          0,
          Number(state.conceptCursor) || 0
        ) >>> 0;


      this.studyCounts =
        new Map(
          Array.isArray(state.studyCounts)
            ? state.studyCounts
            : []
        );


      this.attention =
        new Map(
          Array.isArray(state.attention)
            ? state.attention
            : []
        );


      this.thoughtHistory =
        Array.isArray(state.thoughtHistory)
          ? state.thoughtHistory.slice(
              -this.options
                .thoughtHistoryCapacity
            )
          : [];


      this.lastThought =
        state.lastThought || null;


      this.lastStudy =
        state.lastStudy || null;


      this.goalQueue = new Map((Array.isArray(state.goals) ? state.goals : []).map(goal => [goal.id, goal]));

      this.goalSequence = Math.max(0, Number(state.goalSequence) || 0);

      this.goalsCompleted = Math.max(0, Number(state.goalsCompleted) || 0);

      this.goalsFailed = Math.max(0, Number(state.goalsFailed) || 0);

      this.goalsDeferred = Math.max(0, Number(state.goalsDeferred) || 0);

      this.curiosityQuestions = Math.max(0, Number(state.curiosityQuestions) || 0);

      this.contradictionInvestigations = Math.max(0, Number(state.contradictionInvestigations) || 0);

      this.hypothesisInvestigations = Math.max(0, Number(state.hypothesisInvestigations) || 0);

      this.graphExplorations = Math.max(0, Number(state.graphExplorations) || 0);

      this.memoryReviews = Math.max(0, Number(state.memoryReviews) || 0);

      this.idleStreak = Math.max(0, Number(state.idleStreak) || 0);

      this.executiveMode = String(state.executiveMode || 'ACTIVE');

      this.workspace = state.workspace || this.workspace;

      this.workspaceHistory = Array.isArray(state.workspaceHistory) ? state.workspaceHistory.slice(-this.options.workspaceHistoryCapacity) : [];

      this.lastExecutiveCycle = state.lastExecutiveCycle || null;
      this.foregroundThoughtCycles = Math.max(0, Number(state.foregroundThoughtCycles) || 0);
      this.foregroundThoughtSessions = Math.max(0, Number(state.foregroundThoughtSessions) || 0);
      this.foregroundConvergences = Math.max(0, Number(state.foregroundConvergences) || 0);
      this.lastForegroundThought = state.lastForegroundThought || null;


      this.dirty = false;

      return true;

    
}


    status() {

      return {

        ready: true,
        role:
          'autonomous-cognitive-executive-goal-and-metacognition-coordinator',
        enabled:
          this.options.enabled,
        cycles:
          this.cycles,
        autonomousCycles:
          this.autonomousCycles,
        conceptStudies:
          this.conceptStudies,
        relationConsolidations:
          this.relationConsolidations,
        reflections:
          this.reflections,
        interruptions:
          this.interruptions,
        totalMs:
          this.totalMs,
        averageCycleMs:
          this.cycles
            ? this.totalMs / this.cycles
            : 0,
        lastCycleMs:
          this.lastCycleMs,
        lastPhase:
          this.lastPhase,
        executiveMode: this.executiveMode,
        queuedGoals: this.goalQueue.size,
        goalsCompleted: this.goalsCompleted,
        goalsFailed: this.goalsFailed,
        goalsDeferred: this.goalsDeferred,
        curiosityQuestions: this.curiosityQuestions,
        contradictionInvestigations: this.contradictionInvestigations,
        hypothesisInvestigations: this.hypothesisInvestigations,
        graphExplorations: this.graphExplorations,
        memoryReviews: this.memoryReviews,
        workspace: this.workspace,
        lastExecutiveCycle: this.lastExecutiveCycle,
        foregroundThinking: {
          sessions: this.foregroundThoughtSessions,
          cycles: this.foregroundThoughtCycles,
          convergences: this.foregroundConvergences,
          last: this.lastForegroundThought ? { cycleCount: this.lastForegroundThought.cycleCount, progress: this.lastForegroundThought.progress, converged: this.lastForegroundThought.converged, unresolved: (this.lastForegroundThought.unresolved || []).slice(0, 8) } : null
        },
        attentionSize:
          this.attention.size,
        lastThought:
          this.lastThought,
        lastStudy:
          this.lastStudy,

        existingSystemsCoordinated: {

          rslBackground:
            Boolean(this.rsl),
          correlationBackground:
            Boolean(this.correlation),
          languageState:
            Boolean(this.languageState),
          wordState:
            Boolean(this.wordState),
          phraseState:
            Boolean(this.phraseState),
          wordMatrix:
            Boolean(this.wordMatrix),
          trainingInfoSelfStudy:
            Boolean(this.trainingKnowledge),
          entityKnowledge:
            Boolean(this.entityResolver),
          selfImprovement:
            Boolean(this.selfImprovement),
          adaptiveLearningController:
            Boolean(this.learningController)
        
}
,

        safeguards: {

          noUserPromptRequired: true,
          neverEmitsUserResponse: true,
          neverLearnsInventedAnswerAsFact: true,
          foregroundInterruptible: true,
          phaseTimeBudgeted: true
        
}

      
}
;

    
}

  
}


  globalThis.VilotCognitive =
    VilotCognitive;

}
)();

