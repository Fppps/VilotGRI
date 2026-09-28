/**
 * VilotNI 2 - Production cognitive worker.
 *
 * One unified six-million-neuron model. Words.json supplies lexical identities;
 * Lexicon/WordState/PhraseState/WordMatrix provide learned lexical behavior;
 * RSL resolves contextual POS and recurrent state; Correlation.js now combines
 * structural, lexical, correlation and curiosity learning between prompts. VilotNI 2.5 adds
 * four-branch output interference: TrainingInfo.json supplies keyword and structured-relation evidence
 * while Words.json remains the source of sayable words.
 */
'use strict';


const V25_BUILD_ID = '25-system-37-hotfix-1';


importScripts('./RuleStore.js?v=25-system-37-hotfix-1', './GrammarEngine.js?v=25-system-37-hotfix-1', './ComplexLanguage.js?v=25-system-37-hotfix-1', './PropositionGraph.js?v=25-system-37-hotfix-1', './MechanismGraph.js?v=25-system-37-hotfix-1', './ReasoningMemory.js?v=25-system-37-hotfix-1', './ReasoningQuality.js?v=25-system-37-hotfix-1', './Arena.js?v=25-system-37-hotfix-1', './Trace.js?v=25-system-37-hotfix-1', './Compute.js?v=25-system-37-hotfix-1', './Storage.js?v=25-system-37-hotfix-1', './RSL.js?v=25-system-37-hotfix-1', './Processor.js?v=25-system-37-hotfix-1', './Lexicon.js?v=25-system-37-hotfix-1', './WordState.js?v=25-system-37-hotfix-1', './PhraseState.js?v=25-system-37-hotfix-1', './WordMatrix.js?v=25-system-37-hotfix-1', './QuestionState.js?v=25-system-37-hotfix-1', './ClauseState.js?v=25-system-37-hotfix-1', './MorphologyState.js?v=25-system-37-hotfix-1', './KnowledgeState.js?v=25-system-37-hotfix-1', './LanguageState.js?v=25-system-37-hotfix-1', './Memory.js?v=25-system-37-hotfix-1', './Correlation.js?v=25-system-37-hotfix-1', './MatrixEngine.js?v=25-system-37-hotfix-1', './QueryPlanner.js?v=25-system-37-hotfix-1', './EvidenceSearch.js?v=25-system-37-hotfix-1', './CandidateRanker.js?v=25-system-37-hotfix-1', './Decoder.js?v=25-system-37-hotfix-1', './TrainingKnowledge.js?v=25-system-37-hotfix-1', './KnowledgePolicy.js?v=25-system-37-hotfix-1', './KnowledgeGraph.js?v=25-system-37-hotfix-1', './TemporalKnowledge.js?v=25-system-37-hotfix-1', './PredictionError.js?v=25-system-37-hotfix-1', './HypothesisStore.js?v=25-system-37-hotfix-1', './ReplayScheduler.js?v=25-system-37-hotfix-1', './Generalization.js?v=25-system-37-hotfix-1', './ContradictionResolver.js?v=25-system-37-hotfix-1', './LearningEvaluator.js?v=25-system-37-hotfix-1', './GraphReasoner.js?v=25-system-37-hotfix-1', './LearnedKnowledgeBridge.js?v=25-system-37-hotfix-1', './UncertaintyState.js?v=25-system-37-hotfix-1', './CounterfactualWorkspace.js?v=25-system-37-hotfix-1', './ClaimVerifier.js?v=25-system-37-hotfix-1', './LearningController.js?v=25-system-37-hotfix-1', './Cognitive.js?v=25-system-37-hotfix-1', './StateProcessor.js?v=25-system-37-hotfix-1', './SemanticComposer.js?v=25-system-37-hotfix-1', './ClausePlanner.js?v=25-system-37-hotfix-1', './IdeaFusion.js?v=25-system-37-hotfix-1', './KeywordSelector.js?v=25-system-37-hotfix-1', './SyntacticBridge.js?v=25-system-37-hotfix-1', './SyntaxSmoother.js?v=25-system-37-hotfix-1', './SurfaceRealizer.js?v=25-system-37-hotfix-1', './MixedSlotPlanner.js?v=25-system-37-hotfix-1', './OutputCoherence.js?v=25-system-37-hotfix-1', './SentenceMixer.js?v=25-system-37-hotfix-1', './InterferenceDecoder.js?v=25-system-37-hotfix-1', './LatencyProfiler.js?v=25-system-37-hotfix-1', './LatencyGovernor.js?v=25-system-37-hotfix-1', './ColdStartWarmup.js?v=25-system-37-hotfix-1', './FastLane.js?v=25-system-37-hotfix-1', './EntityResolver.js?v=25-system-37-hotfix-1', './ArchitectureKnowledge.js?v=25-system-37-hotfix-1', './ExplanationPlanner.js?v=25-system-37-hotfix-1', './SelfImprovement.js?v=25-system-37-hotfix-1', './ConversationalCognition.js?v=25-system-37-hotfix-1', './ReasoningCoordinator.js?v=25-system-37-hotfix-1');


const Phase = Object.freeze({

  BOOTING: 'BOOTING',
  ACTIVE: 'ACTIVE',
  WARM: 'WARM',
  IDLE: 'IDLE',
  SUSPENDED: 'SUSPENDED'
}
);


const runtime = {

  ready: false,
  phase: Phase.BOOTING,
  visible: true,
  arena: null,
  trace: null,
  compute: null,
  rsl: null,
  processor: null,
  lexicon: null,
  wordState: null,
  phraseState: null,
  wordMatrix: null,
  languageState: null,
  decoder: null,
  baseDecoder: null,
  trainingKnowledge: null,
  entityResolver: null,
  architectureKnowledge: null,
  explanationPlanner: null,
  propositionGraph: null,
  mechanismGraph: null,
  reasoningMemory: null,
  reasoningQuality: null,
  conversationalCognition: null,
  reasoningCoordinator: null,
  knowledgePolicy: null,
  knowledgeGraph: null,
  temporalKnowledge: null,
  graphReasoner: null,
  learnedKnowledgeBridge: null,
  uncertaintyState: null,
  counterfactualWorkspace: null,
  claimVerifier: null,
  predictionError: null,
  hypothesisStore: null,
  replayScheduler: null,
  generalization: null,
  contradictionResolver: null,
  learningEvaluator: null,
  learningController: null,
  selfImprovement: null,
  cognitive: null,
  stateProcessor: null,
  semanticComposer: null,
  clausePlanner: null,
  ideaFusion: null,
  keywordSelector: null,
  syntacticBridge: null,
  syntaxSmoother: null,
  surfaceRealizer: null,
  mixedSlotPlanner: null,
  outputCoherence: null,
  sentenceMixer: null,
  coldStartWarmup: null,
  storage: null,
  memory: null,
  correlation: null,
  matrix: null,
  fastLane: null,
  queryPlanner: null,
  evidenceSearch: null,
  candidateRanker: null,
  latencyProfiler: null,
  latencyGovernor: null,
  ruleStore: null,
  grammarEngine: null,
  initWarnings: [],
  dialogueContext: [],
  dialogueContextCapacity: 16,
  learningQueue: [],
  learningProcessed: 0,
  learningDropped: 0,
  tickCount: 0,
  foregroundEpoch: 0,
  lastForegroundAt: performance.now(),
  timer: null,
  foregroundBusy: false,
  lastHomeostasisAt: 0,
  options: {

    stateDim: 6000000,
    activeTickMs: 24,
    warmTickMs: 100,
    idleTickMs: 400,
    warmAfterMs: 1400,
    idleAfterMs: 12000,
    activeHomeostasisMs: 240,
    warmHomeostasisMs: 520,
    idleHomeostasisMs: 900,
    suspendWhenHidden: false,
    minResponseConfidence: 80,
    latencyTargetMs: 40,
    latencySafetyReserveMs: 4,
    latencyDecoderFinishReserveMs: 5,
    latencySemanticReserveMs: 3,
    latencyCoherenceReserveMs: 1,
    latencyMinimumStageReserveMs: 1,
    latencyEWMAAlpha: 0.22,
    latencyHardStopWithoutEligible: false,
    latencyHardDeadlineMode: false,
    latencyRSLMaxSteps: 192,
    latencyFocusedStaticPoolLimit: 256,
    latencyFocusedDynamicRSLLimit: 64,
    latencyFocusedCorrelationLimit: 64,
    latencyFocusedSuperpositionPoolLimit: 384,
    latencyFocusedKnowledgeLimit: 24,
    latencyFocusedTargetMax: 32,
    latencyFocusedTargetMin: 12,
    latencyDeadlineCheckStride: 8,
    preserveFullModelBreadth: true,
    interferenceFullSentenceBranches: true,
    interferenceSentenceConfidenceWindowPct: 2,
    interferenceSentenceSimilarityThreshold: 0.58,
    interferenceSentenceMinParticipants: 2,
    interferenceSentenceAmplitudeEnabled: true,
    interferenceSentenceAmplitudeCoupling: 0.12,
    interferenceSentenceAmplitudeThreshold: 0.54,
    keywordResponseMode: true,
    keywordDefinitionDirectOutput: false,
    keywordVectorMaxTerms: 96,
    keywordExactWeight: 1,
    keywordRelationWeight: 0.78,
    latencyProfilerSampleCapacity: 96,
    latencyProfilerMaxStages: 64,
    latencyTailPercentile: 0.95,
    latencyTailGuardFactor: 1.05,
    latencyTailMinSamples: 8,
    coldStartWarmupEnabled: true,
    coldStartWarmupGPU: true,
    coldStartWarmupModel: true,
    coldStartWarmupEvidenceDepth: 1,
    coldStartWarmupTrainingLimit: 6,
    maxForegroundSearchMs: 180, // speed mission threshold: widens search strategy, never terminates quality work
    learningQueueMax: 512,
    learningControllerEnabled: true,
    learningControllerQueueMax: 1024,
    learningControllerActiveBudgetMs: 1.5,
    learningControllerWarmBudgetMs: 5,
    learningControllerIdleBudgetMs: 14,
    learningControllerActiveEvents: 1,
    learningControllerWarmEvents: 4,
    learningControllerIdleEvents: 12,
    learningReplayActive: 0,
    learningReplayWarm: 1,
    learningReplayIdle: 4,
    learningGeneralizeEvery: 4,
    learningContradictionEvery: 3,
    learningHypothesisTestEvery: 3,
    learningHypothesisTestsActive: 0,
    learningHypothesisTestsWarm: 2,
    learningHypothesisTestsIdle: 6,
    learningPromptGroundingStudies: 3,
    learningPromptGroundingMinTrust: 0.75,
    learningRSLPromotionGain: 0.14,
    learningReplayGain: 0.20,
    learningMaxLexicalTermsPerEdge: 4,
    learningMaxEdgesPerTrustedStudy: 32,
    learningGraphMaxEdges: 16384,
    learningGraphEvidenceScale: 4,
    learningGraphConsolidationThreshold: 0.82,
    learningGraphContestedThreshold: 0.26,
    learningTrustedSourceThreshold: 0.85,
    learningPredictionPrior: 0.18,
    learningPredictionEWMAAlpha: 0.12,
    learningHypothesisMax: 8192,
    learningHypothesisPromotionThreshold: 0.82,
    learningHypothesisRejectionThreshold: 0.26,
    learningReplayMaxItems: 2048,
    learningReplayBaseIntervalMs: 1200,
    learningReplayMaxIntervalMs: 300000,
    learningGeneralizationMinSubjects: 3,
    learningGeneralizationMinConfidence: 0.72,
    learningGeneralizationMaxPatterns: 4096,
    learningContradictionScanLimit: 96,
    learningContradictionMinConfidence: 0.55,
    learningEvaluatorEWMAAlpha: 0.10,
    learningEvaluatorPromptHistoryMax: 512,
    learningRegressionEvery: 2,
    learningRegressionRollbackEnabled: true,
    learningRegressionCases: 512,
    learningRegressionTolerance: 0.035,
    learningRegressionMinCases: 8,
    learningPersistenceEvery: 8,
    entityMaxMatches: 6,
    entityMaxConcepts: 24,
    entityGraphSeedTrust: 0.97,
    entityGraphSeedStrength: 0.92,
    selfImprovementEnabled: true,
    selfImprovementProposalEvery: 8,
    selfImprovementEvaluationWindow: 8,
    selfImprovementRollbackTolerance: 0.025,
    selfImprovementStep: 0.015,
    selfImprovementEWMAAlpha: 0.16,
    memoryWorkingCapacity: 32,
    memoryEpisodicCapacity: 384,
    memoryRetrievalLimit: 8,
    memoryRecencyHalfLifeMs: 604800000,
    temporalKnowledgeStaleAfterMs: 604800000,
    temporalKnowledgeStaleFloor: 0.45,
    temporalKnowledgeUnknownFreshness: 0.88,
    graphReasonerMaxDepth: 3,
    graphReasonerMaxFrontier: 32,
    graphReasonerMaxPaths: 16,
    graphReasonerScanLimit: 256,
    graphReasonerMinEdgeConfidence: 0.72,
    graphReasonerMinDerivedConfidence: 0.70,
    learnedKnowledgeResultLimit: 12,
    learnedKnowledgeMinConfidence: 0.78,
    learnedKnowledgeMinRelevance: 0.10,
    learnedKnowledgeScanLimit: 256,
    learnedKnowledgeReasoning: true,
    claimVerifierMinScore: 0.58,
    claimVerifierTrustedThreshold: 0.78,
    claimVerifierDirectionMismatch: 0.25,
    uncertaintyMinConfidenceCeiling: 0.55,
    uncertaintyContradictionPenalty: 0.22,
    uncertaintyReasoningPenalty: 0.08,
    counterfactualMaxAssumptionChars: 512,
    counterfactualMaxDisabledConcepts: 16,
    fastInitialBatch: 4,
    fastRefineBatch: 6,
    fastSearchBatch: 6,
    fastFocusedBatch: 8,
    fastFocusedBatchMin: 4,
    fastFocusedBatchMax: 12,
    fastFocusedEvidenceLimit: 384,
    fastFocusedDirectLimit: 192,
    fastFocusedRSLLimit: 64,
    fastYieldEveryBatches: 6,
    wordMatrixDim: 64,
    phraseMaxN: 4,
    phraseMaxEntries: 32768,
    curiosityEnabled: true,
    curiosityWeight: 0.14,
    curiosityMaxConfidenceDrop: 3.25,
    curiosityQueueMax: 256,
    interferenceBranchCount: 4,
    interferenceWordConfidenceWindowPct: 2,
    interferencePositionTolerance: 1,
    interferenceNormalizedPositionTolerance: 0.11,
    interferenceMinAgreement: 2,
    trainingInfoResultLimit: 24,
    trainingEvidenceLimit: 256,
    trainingBranchWordLimit: 64,
    trainingQueryCacheMax: 256,
    cognitiveEnabled: true,
    cognitiveActiveBudgetMs: 1.5,
    cognitiveWarmBudgetMs: 4,
    cognitiveIdleBudgetMs: 10,
    cognitiveActiveConcepts: 1,
    cognitiveWarmConcepts: 2,
    cognitiveIdleConcepts: 5,
    cognitiveActiveStudyEvery: 8,
    cognitiveWarmStudyEvery: 3,
    cognitiveIdleStudyEvery: 1,
    cognitiveConceptLookahead: 16,
    cognitiveMaxConceptChars: 1600,
    cognitiveMaxRelationTargets: 10,
    cognitiveBaseStudyGain: 0.18,
    cognitiveRelationGain: 0.12,
    cognitiveAttentionDecay: 0.992,
    cognitiveAttentionCapacity: 512,
    cognitiveThoughtHistoryCapacity: 48,
    cognitivePersistenceEvery: 16,
    cognitiveGoalCapacity: 256,
    cognitiveGoalHarvestLimit: 20,
    cognitiveActiveGoalLimit: 0,
    cognitiveWarmGoalLimit: 2,
    cognitiveIdleGoalLimit: 4,
    cognitiveDeepIdleGoalLimit: 6,
    cognitiveDeepIdleAfterCycles: 8,
    cognitiveDeepIdleBudgetMs: 22,
    cognitiveGoalMinPriority: 0.18,
    cognitiveCuriosityMinUncertainty: 0.42,
    cognitiveStaleReviewFreshness: 0.68,
    cognitiveWorkspaceHistoryCapacity: 32,
    conversationCognitionEnabled: true,
    conversationSoftInfluence: 0.16,
    conversationHistoryTurns: 8,
    conversationMaxAnchors: 28,
    conversationMaxPriorAnchors: 16,
    reasoningCoordinatorEnabled: true,
    reasoningCoordinatorTaskGuidanceStrength: 0.14,
    reasoningCoordinatorMaxSeeds: 28,
    reasoningCoordinatorMaxPriorityConcepts: 48,
    reasoningCoordinatorGraphDepth: 3,
    reasoningCoordinatorGraphPathLimit: 12,
    reasoningCoordinatorHypothesisLimit: 12,
    reasoningCoordinatorHistory: 128,
    architectureKnowledgeMaxMatches: 8,
    architectureGraphSeedTrust: 0.995,
    architectureGraphSeedStrength: 0.97,
    explanationPlannerMaxPriorityTerms: 48,
    explanationPlannerMaxEvidence: 28,
    explanationPlannerGraphPerSeed: 10,
    explanationPlannerCorrelationLimit: 14,
    stateProcessorEnabled: true,
    stateMaxStates: 8,
    stateReinforcementStrength: 0.44,
    stateCancellationStrength: 0.24,
    stateSharpenPower: 1.18,
    statePropagationPasses: 4,
    stateMaxPropagationPasses: 6,
    stateEntropyPassBoost: 1,
    stateConsensusWeight: 0.12,
    stateTransitionWeight: 0.035,
    stateKnowledgeWeight: 0.035,
    stateFocusWeight: 0.025,
    stateStabilityWeight: 0.05,
    stateReliabilityWeight: 0.055,
    stateContextWeight: 0.045,
    stateSemanticWeight: 0.055,
    stateCertaintyWeight: 0.07,
    stateCentralityWeight: 0.045,
    stateRobustnessWeight: 0.05,
    stateHigherOrderWeight: 0.075,
    stateContradictionWeight: 0.06,
    stateNoveltyWeight: 0.01,
    stateMomentum: 0.32,
    stateDamping: 0.70,
    stateAdaptiveStepMin: 0.18,
    stateAdaptiveStepMax: 0.56,
    stateUncertaintyDecay: 0.18,
    stateSupportCertaintyGain: 0.22,
    stateConflictUncertaintyGain: 0.26,
    stateCentralityIterations: 3,
    stateEnergyEpsilon: 0.00002,
    statePerturbationRadius: 0.006,
    stateRollbackOnEnergyIncrease: true,
    stateLiftCap: 0.075,
    stateDropCap: 0.12,
    stateConvergenceEpsilon: 0.00035,
    stateEntropyTemperature: 0.025,
    stateAmplitudeAmplificationEnabled: true,
    stateAmplitudeWeight: 0.085,
    stateAmplitudeSeedGain: 0.035,
    stateAmplitudeCoupling: 0.11,
    stateAmplitudeMarkThreshold: 0.56,
    stateAmplitudeRoundsMin: 1,
    stateAmplitudeRoundsMax: 4,
    semanticMaxKnowledgeRows: 6,
    semanticMaxPropositionsPerBranch: 14,
    semanticBranchConfidencePenaltyPct: 1.4,
    semanticMinimumKnowledgeTrust: 0.58,
    semanticConfidenceWindowPct: 2,
    semanticMinAgreement: 2,
    semanticMaxClauses: 4,
    semanticSimilarityThreshold: 0.28,
    ideaFusionEnabled: true,
    ideaFusionMaxIdeasPerSentence: 3,
    ideaFusionSimpleMaxIdeasPerSentence: 2,
    ideaFusionMinSupport: 2,
    ideaFusionConfidenceWindowPct: 6,
    ideaFusionMaxSentenceWords: 46,
    ideaFusionAllowCrossSubject: true,
    ideaFusionCrossSubjectMinAssociation: 0.18,
    ideaFusionCrossSubjectMinSharedTokenRatio: 0.14,
    ideaFusionMaxSubjectChangesPerSentence: 2,
    keywordSelectionEnabled: true,
    keywordSelectionMaxPerClause: 18,
    keywordSelectionPreserveNumbers: true,
    syntacticBridgeEnabled: true,
    syntacticBridgePreferKeywordSurface: true,
    syntaxSmoothingEnabled: true,
    syntaxSmoothingMaxPasses: 2,
    surfaceMaxWords: 96,
    surfaceMinLexicalCoverage: 0.90,
    surfaceMinGrammarScore: 0.72,
    surfaceMaxRejectedRatio: 0.08,
    mixedSlotBranchStayReward: 0.10,
    mixedSlotBranchSwitchPenalty: 0.075,
    mixedSlotFunctionSwitchPenalty: 0.12,
    mixedSlotTransitionWeight: 0.16,
    mixedSlotCompatibilityWeight: 0.14,
    mixedSlotPreferredWeight: 0.10,
    mixedSlotKnowledgeWeight: 0.05,
    mixedSlotFocusWeight: 0.05,
    mixedSlotMaxSwitchDensity: 0.34,
    mixedSlotIslandRepair: true,
    outputCoherenceMinimumScore: 0.72,
    outputCoherenceSemanticBonus: 0.035,
    outputCoherenceMaxDuplicateRatio: 0.22,
    outputCoherenceMaxUnknownRatio: 0.10
  
}
,
  lastHotPathMs: 0,
  localBenchmark: null
}
;


function send(type, payload = {
}
, requestId = null) {

  postMessage({
 type, requestId, ...payload 
}
);

}


function setPhase(next, reason = '') {

  if (runtime.phase === next) return;

  runtime.phase = next;

  runtime.trace?.push(
    VilotTraceEvent.PHASE_CHANGE,
    0,
    next === Phase.ACTIVE ? 1 : next === Phase.WARM ? 2 : next === Phase.IDLE ? 3 : next === Phase.SUSPENDED ? 4 : 0
  );

  send('PHASE', {
 phase: next, reason 
}
);

}


function currentInterval() {

  switch (runtime.phase) {

    case Phase.ACTIVE: return runtime.options.activeTickMs;

    case Phase.WARM: return runtime.options.warmTickMs;

    case Phase.IDLE: return runtime.options.idleTickMs;

    default: return 1000;

  
}

}


function scheduleBackground() {

  clearTimeout(runtime.timer);

  if (!runtime.ready || runtime.phase === Phase.SUSPENDED) return;

  runtime.timer = setTimeout(backgroundTick, currentInterval());

}


function chooseAutomaticPhase(now) {

  if (runtime.phase === Phase.SUSPENDED) return;

  if (!runtime.visible && runtime.options.suspendWhenHidden) {

    setPhase(Phase.SUSPENDED, 'document-hidden');

    return;

  
}


  const age = now - runtime.lastForegroundAt;

  if (age < runtime.options.warmAfterMs) setPhase(Phase.ACTIVE, 'recent-foreground');

  else if (age < runtime.options.idleAfterMs) setPhase(Phase.WARM, 'cooling');

  else setPhase(Phase.IDLE, 'idle');

}


function homeostasisInterval() {

  if (runtime.phase === Phase.ACTIVE) return Math.max(80, runtime.options.activeHomeostasisMs | 0 || 240);

  if (runtime.phase === Phase.WARM) return Math.max(120, runtime.options.warmHomeostasisMs | 0 || 520);

  return Math.max(200, runtime.options.idleHomeostasisMs | 0 || 900);

}


function exportLearningBundle() {

  return {

    schemaVersion: 16,
    architecture: 'vilotni2.5-integrated-multistate-cognitive-amplitude-reasoning-self-improving',
    rsl: runtime.rsl?.exportLearningState?.() || null,
    wordState: runtime.wordState?.exportState?.() || null,
    phraseState: runtime.phraseState?.exportState?.() || null,
    wordMatrix: runtime.wordMatrix?.exportState?.() || null,
    languageState: runtime.languageState?.exportState?.() || null,
    correlationState: runtime.correlation?.exportState?.() || null,
    memoryState: runtime.memory?.exportState?.() || null,
    cognitiveState: runtime.cognitive?.exportState?.() || null,
    learningControllerState: runtime.learningController?.exportState?.() || null,
    selfImprovementState: runtime.selfImprovement?.exportState?.() || null,
    reasoningMemoryState: runtime.reasoningMemory?.exportState?.() || null
  
}
;

}


function pushDialogueTurn(role, text, metadata = {
}
) {

  const clean = String(text || '').trim();

  if (!clean) return null;

  const turn = {

    role: String(role || 'unknown').toLowerCase(),
    text: clean.slice(0, 4096),
    at: Date.now(),
    confidence: Number.isFinite(metadata.confidence) ? Number(metadata.confidence) : null
  
}
;

  runtime.dialogueContext.push(turn);

  const cap = Math.max(4, runtime.dialogueContextCapacity | 0 || 16);

  if (runtime.dialogueContext.length > cap) runtime.dialogueContext.splice(0, runtime.dialogueContext.length - cap);

  return turn;

}


function dialogueContextSnapshot(limit = 12) {

  const n = Math.max(1, Math.min(24, limit | 0 || 12));

  return runtime.dialogueContext.slice(-n).map(turn => ({
 ...turn 
}
));

}


function deadlineClarification(promptText, analysis, search) {

  const subject = String(
    search?.planSummary?.subjectWords?.[0] ||
    analysis?.contentWords?.[0] ||
    ''
  ).trim();

  const target = subject ? ` about ${subject}` : '';

  const role = String(analysis?.requestedSlot || search?.planSummary?.requestedRole || 'content');


  if (role === 'mechanism') {

    return `I identified the topic${target}, but I could not verify a >80% mechanism before the response deadline. Which part of how it works should I focus on?`;

  
}

  if (role === 'cause') {

    return `I identified the topic${target}, but I could not verify a >80% causal answer before the response deadline. Are you asking for the cause, the effect, or one specific link?`;

  
}

  if (role === 'selection') {

    return `I identified the comparison${target}, but I could not verify a >80% choice before the response deadline. Which criteria should I compare?`;

  
}

  if (role === 'definition' || role === 'identity') {

    return `I identified the topic${target}, but I could not verify a >80% definition before the response deadline. Which aspect should I define?`;

  
}

  return `I identified the topic${target}, but I could not verify a >80% answer before the response deadline. Please narrow the relationship or detail you want.`;

}


function outputLearningQuality(generated) {

  const diagnostics = generated?.diagnostics || {
}
;

  const meaning = diagnostics?.meaningFirst || {
}
;

  const surface = meaning?.surface || {
}
;

  const coherence = meaning?.coherence || {
}
;


  const rows = [];


  const add = (value, weight) => {

    const number = Number(value);

    if (!Number.isFinite(number)) return;

    rows.push({

      value: Math.max(0, Math.min(1, number)),
      weight
    
}
);

  
}
;


  add(coherence?.selectedScore, 0.34);

  add(surface?.grammarScore, 0.20);

  add(surface?.lexicalCoverage, 0.18);

  add(meaning?.agreementRatio, 0.16);

  add((Number(generated?.confidence) || 0) / 100, 0.12);


  if (!rows.length) {

    return Math.max(
      0,
      Math.min(
        1,
        (Number(generated?.confidence) || 0) / 100
      )
    );

  
}


  let total = 0;

  let weight = 0;


  for (const row of rows) {

    total += row.value * row.weight;

    weight += row.weight;

  
}


  return weight
    ? total / weight
    : 0.5;

}


function enqueueLearning(item) {

  if (!item || !item.type) return;

  const max = Math.max(64, runtime.options.learningQueueMax | 0 || 512);

  if (runtime.learningQueue.length >= max) {

    // Preserve newest foreground experiences. Drop the oldest queued replay item,
    // not model capacity or RSL state.
    runtime.learningQueue.shift();

    runtime.learningDropped++;

  
}

  runtime.learningQueue.push(item);

}


async function processLearningQueue(epochAtStart, phase) {

  if (!runtime.learningQueue.length) return {
 processed: 0, remaining: 0, ms: 0 
}
;

  const started = performance.now();

  const maxItems = phase === Phase.IDLE ? 12 : phase === Phase.WARM ? 5 : 2;

  const maxMs = phase === Phase.IDLE ? 18 : phase === Phase.WARM ? 10 : 5;

  let processed = 0;


  while (runtime.learningQueue.length && processed < maxItems && performance.now() - started < maxMs) {

    if (runtime.foregroundBusy || runtime.foregroundEpoch !== epochAtStart) break;

    const item = runtime.learningQueue.shift();

    if (!item) break;


    if (item.type === 'user') {

      runtime.correlation?.observeText?.(item.text, {

        source: 'user',
        accepted: true,
        episodic: true,
        countCycle: true,
        counterfactual: item.counterfactual === true,
        analysisSummary: item.analysisSummary || null
      
}
);

    
}
 else if (item.type === 'failure') {

      const ch = item.channels || {
}
;

      const quality = Math.max(0, Math.min(1,
        (Number(ch.learnedPOSContext) || 0) * 0.24 +
        (Number(ch.grammarLegality) || 0) * 0.24 +
        (Number(ch.dynamicPOSCoherence) || 0) * 0.18 +
        (Number(ch.scaffoldFit) || 0) * 0.16 +
        (Number(ch.sentenceCompleteness) || 0) * 0.12 +
        (Number(ch.correlationCoherence) || 0) * 0.06
      ));

      const deficit = Math.max(0, Math.max(80, runtime.options.minResponseConfidence || 80) - (Number(item.confidence) || 0));

      const gain = Math.min(3.4, 0.95 + deficit / 28);

      runtime.wordState?.observeTrace?.(item.trace || [], {

        confidence: Number(item.confidence) || 0, accepted: false, quality, gain: Math.min(2.0, gain * 0.65)
      
}
);

      runtime.correlation?.learnFailure?.(item.promptText || '', item.text, {

        confidence: Number(item.confidence) || 0,
        quality,
        gain,
        channels: ch
      
}
);

    
}
 else if (item.type === 'response') {

      // Accepted output updates recurrent context after it is already visible to
      // the user. This keeps response latency separate from consolidation.
      await runtime.rsl.observeText(item.text, {

        countTurn: false,
        syncState: false,
        source: 'model_response',
        learnTransitions: item.counterfactual !== true,
        learningGain: item.counterfactual === true ? 0.35 : 1.0,
        maxSteps: Math.min(192, runtime.options.maxTokens || 192)
      
}
);

      runtime.wordState?.observeTrace?.(item.trace || [], {

        confidence: item.confidence, accepted: true, quality: Math.max(0.8, (Number(item.confidence) || 80) / 100), gain: 0.75
      
}
);

      runtime.correlation?.observeText?.(item.text, {

        source: 'model_response',
        confidence: item.confidence,
        accepted: true,
        episodic: true,
        countCycle: true,
        counterfactual: item.counterfactual === true,
        analysisSummary: item.analysisSummary || null
      
}
);

      if (item.counterfactual !== true) {

        runtime.correlation?.observePair?.(item.promptText || '', item.text, {

          confidence: item.confidence
        
}
);

        runtime.languageState?.observePair?.(item.promptText || '', item.text, {

          confidence: item.confidence, gain: Math.max(0.45, (Number(item.confidence) || 80) / 100)
        
}
);

      
}

    
}


    runtime.learningController?.observeExperience?.({

      ...item,
      responseText:
        item.responseText ||
        item.text ||
        '',
      analysisSummary:
        item.analysisSummary ||
        null,
      route:
        item.route ||
        ''
    
}
);


    processed++;

    runtime.learningProcessed++;

  
}


  if (processed) runtime.storage?.scheduleSave?.(() => exportLearningBundle());

  return {
 processed, remaining: runtime.learningQueue.length, ms: performance.now() - started 
}
;

}


async function backgroundTick() {


  if (!runtime.ready || runtime.phase === Phase.SUSPENDED) return;


  const start = performance.now();

  const epochAtStart = runtime.foregroundEpoch;

  chooseAutomaticPhase(start);


  // Foreground generation has priority. Prompt/response/failure consolidation
  // lives in the learning queue, so Correlation learning never competes with
  // the blocking response path for the same worker/GPU.
  if (!runtime.foregroundBusy && epochAtStart === runtime.foregroundEpoch && runtime.phase !== Phase.SUSPENDED) {

    await processLearningQueue(epochAtStart, runtime.phase);

    if (runtime.foregroundBusy || epochAtStart !== runtime.foregroundEpoch) {

      scheduleBackground();

      return;

    
}

    const phaseDecay = runtime.phase === Phase.ACTIVE ? 0.9992 : runtime.phase === Phase.WARM ? 0.997 : 0.990;

    const phaseInhibition = runtime.phase === Phase.ACTIVE ? 0.002 : runtime.phase === Phase.WARM ? 0.006 : 0.012;

    const now = performance.now();


    // A full homeostasis sweep touches all six million neurons. Structural, lexical, correlation and curiosity learning still run every background tick, while the dense
    // recurrent sweep is cadence-controlled so it cannot monopolize the GPU.
    if (now - runtime.lastHomeostasisAt >= homeostasisInterval()) {

      runtime.compute.homeostasis({

        decay: phaseDecay,
        inhibition: phaseInhibition,
        baselinePull: 0.0015,
        tick: runtime.tickCount
      
}
);

      runtime.lastHomeostasisAt = now;

    
}


    runtime.tickCount++;


    if (runtime.cognitive) {

      await runtime.cognitive.backgroundTick(
        runtime.phase,
        epochAtStart,
        () => runtime.foregroundEpoch
      );

    
}
 else {

      runtime.rsl?.backgroundTick?.(runtime.phase);

      runtime.correlation?.backgroundTick?.(
        runtime.phase,
        epochAtStart,
        () => runtime.foregroundEpoch
      );

    
}


    if (
      !runtime.foregroundBusy &&
      epochAtStart === runtime.foregroundEpoch &&
      runtime.learningController
    ) {

      await runtime.learningController.backgroundTick(
        runtime.phase,
        epochAtStart,
        () => runtime.foregroundEpoch
      );

    
}

    if (
      !runtime.foregroundBusy &&
      epochAtStart === runtime.foregroundEpoch &&
      runtime.selfImprovement
    ) {
      await runtime.selfImprovement.backgroundTick(runtime.phase);
    }


    if (
      runtime.cognitive?.dirty &&
      runtime.tickCount %
        Math.max(
          4,
          runtime.options.cognitivePersistenceEvery | 0 ||
          16
        ) === 0
    ) {

      runtime.storage?.scheduleSave?.(
        () => exportLearningBundle()
      );

      runtime.cognitive.dirty = false;

    
}


    if (
      runtime.learningController?.dirty &&
      runtime.tickCount %
        Math.max(
          4,
          runtime.options.learningPersistenceEvery | 0 ||
          8
        ) === 0
    ) {

      runtime.storage?.scheduleSave?.(
        () => exportLearningBundle()
      );

      runtime.learningController.dirty = false;

    
}

    if (runtime.selfImprovement?.dirty) {
      runtime.storage?.scheduleSave?.(() => exportLearningBundle());
      runtime.selfImprovement.dirty = false;
    }

  
}


  runtime.trace.push(
    VilotTraceEvent.BACKGROUND_TICK,
    performance.now() - start,
    runtime.tickCount,
    runtime.foregroundEpoch
  );

  scheduleBackground();

}


async function boot(options = {
}
) {

  if (runtime.ready) return statusPayload();

  runtime.trace = new VilotTrace(options.traceCapacity || 512);

  runtime.trace.push(VilotTraceEvent.BOOT_BEGIN);


  Object.assign(runtime.options, options || {
}
);

  runtime.arena = new VilotArena({

    stateDim: runtime.options.stateDim,
    maxTokens: runtime.options.maxTokens,
    maxCandidates: runtime.options.maxCandidates,
    topK: runtime.options.topK
  
}
);


  runtime.compute = new VilotCompute(runtime.arena, runtime.trace, runtime.options);

  await runtime.compute.init('./Shaders.wgsl?v=25-system-37-hotfix-1');

  runtime.rsl = new VilotRSL(runtime.arena, runtime.compute, runtime.trace, runtime.options);

  runtime.storage = new VilotStorage({
 key: 'vilotni25-learning-schema-10-6m-four-branch-interference' 
}
);

  runtime.ruleStore = new VilotRuleStore(runtime.options);

  await runtime.ruleStore.load('./', 'v=25-system-37-hotfix-1');

  runtime.processor = new VilotProcessor(runtime.arena, runtime.rsl, runtime.trace, runtime.options, runtime.ruleStore);

  await runtime.processor.load('./Words.json?v=25-system-37-hotfix-1');

  runtime.grammarEngine = new VilotGrammarEngine(runtime.processor, runtime.ruleStore, runtime.options);

  runtime.processor.grammarEngine = runtime.grammarEngine;

  runtime.lexicon = new VilotLexicon(runtime.processor, runtime.options);

  runtime.wordState = new VilotWordState(runtime.lexicon, runtime.rsl, runtime.options);

  runtime.phraseState = new VilotPhraseState(runtime.lexicon, runtime.options);

  runtime.wordMatrix = new VilotWordMatrix(runtime.lexicon, runtime.options);

  runtime.languageState = new VilotLanguageState(runtime.processor, runtime.rsl, runtime.lexicon, runtime.wordState, runtime.phraseState, runtime.options);

  let learned = null;

  try {

    await runtime.storage.init();

    learned = await runtime.storage.load();

    if (learned?.rsl) {

      runtime.rsl.importLearningState(learned.rsl);

      runtime.wordState.importState(learned.wordState);

      runtime.phraseState.importState(learned.phraseState);

      runtime.wordMatrix.importState(learned.wordMatrix);

      runtime.languageState.importState(learned.languageState);

    
}
 else if (learned) {

      // One-way compatibility with the previous RSL-only production state.
      runtime.rsl.importLearningState(learned);

    
}

  
}
 catch (_) {
}

  runtime.memory = new VilotMemory(runtime.options);

  if (learned?.memoryState) runtime.memory.importState?.(learned.memoryState);

  const lexicalLearning = {
 wordState: runtime.wordState, phraseState: runtime.phraseState, wordMatrix: runtime.wordMatrix, languageState: runtime.languageState 
}
;

  runtime.correlation = new VilotCorrelation(runtime.rsl, runtime.processor, runtime.memory, runtime.trace, runtime.options, lexicalLearning);

  if (learned?.correlationState) runtime.correlation.importState?.(learned.correlationState);

  runtime.matrix = new VilotMatrixEngine(runtime.options);

  runtime.queryPlanner = new VilotQueryPlanner(runtime.processor, runtime.rsl, runtime.languageState, runtime.options, runtime.ruleStore);

  runtime.evidenceSearch = new VilotEvidenceSearch(runtime.processor, runtime.rsl, lexicalLearning, runtime.correlation, runtime.options, runtime.ruleStore);

  runtime.candidateRanker = new VilotCandidateRanker(runtime.processor, runtime.options);

  runtime.latencyProfiler = new VilotLatencyProfiler(runtime.options);

  runtime.latencyGovernor = new VilotLatencyGovernor(runtime.options, runtime.latencyProfiler);

  runtime.baseDecoder = new VilotDecoder(runtime.arena, runtime.rsl, runtime.processor, runtime.trace, runtime.options, runtime.matrix, lexicalLearning);

  runtime.trainingKnowledge = new VilotTrainingKnowledge(runtime.processor, runtime.options);
  try { await runtime.trainingKnowledge.load('./TrainingInfo.json?v=25-system-37-hotfix-1'); }
  catch (error) { runtime.initWarnings.push({ stage:'training-knowledge', message:String(error?.message || error) }); }

  runtime.entityResolver = new VilotEntityResolver(runtime.processor, runtime.options);
  try { await runtime.entityResolver.load('./EntityKnowledge.json?v=25-system-37-hotfix-1'); }
  catch (error) { runtime.initWarnings.push({ stage:'entity-knowledge', message:String(error?.message || error) }); }

  runtime.architectureKnowledge = new VilotArchitectureKnowledge(runtime.options);
  try { await runtime.architectureKnowledge.load('./ArchitectureKnowledge.json?v=25-system-37-hotfix-1'); }
  catch (error) { runtime.initWarnings.push({ stage:'architecture-knowledge', message:String(error?.message || error) }); }

  runtime.explanationPlanner = new VilotExplanationPlanner({
    processor: runtime.processor,
    knowledgeGraph: runtime.knowledgeGraph,
    correlation: runtime.correlation,
    architectureKnowledge: runtime.architectureKnowledge
  }, runtime.options);


  runtime.knowledgePolicy =
    new VilotKnowledgePolicy(
      runtime.options
    );


  runtime.knowledgeGraph =
    new VilotKnowledgeGraph(
      runtime.options,
      runtime.knowledgePolicy
    );


  runtime.temporalKnowledge =
    new VilotTemporalKnowledge(
      runtime.options,
      runtime.ruleStore
    );


  runtime.predictionError =
    new VilotPredictionError(
      runtime.knowledgeGraph,
      runtime.rsl,
      runtime.processor,
      runtime.options
    );


  runtime.hypothesisStore =
    new VilotHypothesisStore(
      runtime.options
    );


  runtime.replayScheduler =
    new VilotReplayScheduler(
      runtime.options,
      runtime.ruleStore
    );


  runtime.generalization =
    new VilotGeneralization(
      runtime.options
    );


  runtime.contradictionResolver =
    new VilotContradictionResolver(
      runtime.knowledgeGraph,
      runtime.hypothesisStore,
      runtime.options
    );


  runtime.learningEvaluator =
    new VilotLearningEvaluator(
      runtime.processor,
      runtime.options
    );


  runtime.graphReasoner =
    new VilotGraphReasoner(
      runtime.knowledgeGraph,
      runtime.temporalKnowledge,
      runtime.options,
      runtime.knowledgePolicy
    );


  runtime.propositionGraph = new VilotPropositionGraph(
    runtime.ruleStore,
    runtime.correlation,
    runtime.knowledgeGraph,
    runtime.options
  );

  runtime.mechanismGraph = new VilotMechanismGraph(
    runtime.ruleStore,
    runtime.options
  );

  runtime.reasoningMemory = new VilotReasoningMemory(runtime.options);
  if (learned?.reasoningMemoryState) runtime.reasoningMemory.importState?.(learned.reasoningMemoryState);

  runtime.reasoningQuality = new VilotReasoningQuality(
    runtime.ruleStore,
    runtime.options
  );


  runtime.learnedKnowledgeBridge =
    new VilotLearnedKnowledgeBridge(
      runtime.knowledgeGraph,
      runtime.hypothesisStore,
      runtime.temporalKnowledge,
      runtime.processor,
      runtime.graphReasoner,
      runtime.options,
      runtime.knowledgePolicy
    );


  runtime.uncertaintyState =
    new VilotUncertaintyState(
      runtime.learningEvaluator,
      runtime.options,
      runtime.ruleStore,
      runtime.temporalKnowledge
    );


  runtime.counterfactualWorkspace =
    new VilotCounterfactualWorkspace(
      runtime.processor,
      runtime.graphReasoner,
      runtime.options
    );


  runtime.claimVerifier =
    new VilotClaimVerifier(
      runtime.knowledgeGraph,
      runtime.temporalKnowledge,
      runtime.graphReasoner,
      runtime.uncertaintyState,
      runtime.options,
      runtime.knowledgePolicy
    );


  runtime.evidenceSearch.attachLearnedKnowledgeBridge?.(
    runtime.learnedKnowledgeBridge
  );

  runtime.evidenceSearch.attachMemory?.(
    runtime.memory
  );


  runtime.learningController =
    new VilotLearningController(
      {

        rsl: runtime.rsl,
        processor: runtime.processor,
        correlation: runtime.correlation,
        trainingKnowledge: runtime.trainingKnowledge,
        graph: runtime.knowledgeGraph,
        predictionError: runtime.predictionError,
        hypotheses: runtime.hypothesisStore,
        replay: runtime.replayScheduler,
        generalization: runtime.generalization,
        contradictions: runtime.contradictionResolver,
        evaluator: runtime.learningEvaluator,
        policy: runtime.knowledgePolicy
      
}
,
      runtime.options
    );


  if (learned?.learningControllerState) {

    runtime.learningController.importState?.(
      learned.learningControllerState
    );

  
}

  // Seed the real-world entity layer after persisted graph import so trusted
  // entity relations are always present without deleting learned edges.
  runtime.entityResolver?.seedKnowledgeGraph?.(runtime.knowledgeGraph);
  runtime.architectureKnowledge?.seedKnowledgeGraph?.(runtime.knowledgeGraph);
  if (runtime.explanationPlanner) runtime.explanationPlanner.graph = runtime.knowledgeGraph;

  runtime.selfImprovement = new VilotSelfImprovement({
    evaluator: runtime.learningEvaluator,
    predictionError: runtime.predictionError,
    hypotheses: runtime.hypothesisStore,
    replay: runtime.replayScheduler,
    graph: runtime.knowledgeGraph,
    ruleStore: runtime.ruleStore,
    runtimeOptions: runtime.options
  }, runtime.options);

  if (learned?.selfImprovementState) {
    runtime.selfImprovement.importState?.(learned.selfImprovementState);
  }


  runtime.cognitive = new VilotCognitive({

    rsl: runtime.rsl,
    processor: runtime.processor,
    memory: runtime.memory,
    correlation: runtime.correlation,
    languageState: runtime.languageState,
    wordState: runtime.wordState,
    phraseState: runtime.phraseState,
    wordMatrix: runtime.wordMatrix,
    trainingKnowledge: runtime.trainingKnowledge,
    entityResolver: runtime.entityResolver,
    selfImprovement: runtime.selfImprovement,
    learningController: runtime.learningController,
    knowledgeGraph: runtime.knowledgeGraph,
    predictionError: runtime.predictionError,
    hypothesisStore: runtime.hypothesisStore,
    replayScheduler: runtime.replayScheduler,
    contradictionResolver: runtime.contradictionResolver,
    learningEvaluator: runtime.learningEvaluator,
    graphReasoner: runtime.graphReasoner,
    learnedKnowledgeBridge: runtime.learnedKnowledgeBridge,
    uncertaintyState: runtime.uncertaintyState,
    temporalKnowledge: runtime.temporalKnowledge,
    trace: runtime.trace
  
}
, runtime.options);


  if (learned?.cognitiveState) {

    runtime.cognitive.importState?.(
      learned.cognitiveState
    );

  
}


  runtime.stateProcessor = new VilotStateProcessor({

    confidenceWindow: runtime.options.interferenceWordConfidenceWindowPct / 100,
    reinforcementStrength: runtime.options.stateReinforcementStrength,
    cancellationStrength: runtime.options.stateCancellationStrength,
    sharpenPower: runtime.options.stateSharpenPower,
    propagationPasses: runtime.options.statePropagationPasses,
    maxPropagationPasses: runtime.options.stateMaxPropagationPasses,
    entropyPassBoost: runtime.options.stateEntropyPassBoost,
    consensusWeight: runtime.options.stateConsensusWeight,
    transitionWeight: runtime.options.stateTransitionWeight,
    knowledgeWeight: runtime.options.stateKnowledgeWeight,
    focusWeight: runtime.options.stateFocusWeight,
    stabilityWeight: runtime.options.stateStabilityWeight,
    reliabilityWeight: runtime.options.stateReliabilityWeight,
    contextWeight: runtime.options.stateContextWeight,
    semanticWeight: runtime.options.stateSemanticWeight,
    certaintyWeight: runtime.options.stateCertaintyWeight,
    centralityWeight: runtime.options.stateCentralityWeight,
    robustnessWeight: runtime.options.stateRobustnessWeight,
    higherOrderWeight: runtime.options.stateHigherOrderWeight,
    contradictionWeight: runtime.options.stateContradictionWeight,
    noveltyWeight: runtime.options.stateNoveltyWeight,
    momentum: runtime.options.stateMomentum,
    damping: runtime.options.stateDamping,
    adaptiveStepMin: runtime.options.stateAdaptiveStepMin,
    adaptiveStepMax: runtime.options.stateAdaptiveStepMax,
    uncertaintyDecay: runtime.options.stateUncertaintyDecay,
    supportCertaintyGain: runtime.options.stateSupportCertaintyGain,
    conflictUncertaintyGain: runtime.options.stateConflictUncertaintyGain,
    centralityIterations: runtime.options.stateCentralityIterations,
    energyEpsilon: runtime.options.stateEnergyEpsilon,
    perturbationRadius: runtime.options.statePerturbationRadius,
    rollbackOnEnergyIncrease: runtime.options.stateRollbackOnEnergyIncrease,
    stateLiftCap: runtime.options.stateLiftCap,
    stateDropCap: runtime.options.stateDropCap,
    convergenceEpsilon: runtime.options.stateConvergenceEpsilon,
    entropyTemperature: runtime.options.stateEntropyTemperature,
    maxStates: runtime.options.stateMaxStates,
    amplitudeAmplificationEnabled: runtime.options.stateAmplitudeAmplificationEnabled,
    amplitudeWeight: runtime.options.stateAmplitudeWeight,
    amplitudeSeedGain: runtime.options.stateAmplitudeSeedGain,
    amplitudeCoupling: runtime.options.stateAmplitudeCoupling,
    amplitudeMarkThreshold: runtime.options.stateAmplitudeMarkThreshold,
    amplitudeRoundsMin: runtime.options.stateAmplitudeRoundsMin,
    amplitudeRoundsMax: runtime.options.stateAmplitudeRoundsMax
  
}
);

  runtime.conversationalCognition = new VilotConversationalCognition(
    runtime.processor,
    runtime.options
  );

  runtime.reasoningCoordinator = new VilotReasoningCoordinator({
    cognitive: runtime.cognitive,
    graphReasoner: runtime.graphReasoner,
    knowledgeGraph: runtime.knowledgeGraph,
    hypothesisStore: runtime.hypothesisStore,
    predictionError: runtime.predictionError,
    learningEvaluator: runtime.learningEvaluator,
    stateProcessor: runtime.stateProcessor,
    matrix: runtime.matrix,
    entityResolver: runtime.entityResolver,
    architectureKnowledge: runtime.architectureKnowledge,
    conversationalCognition: runtime.conversationalCognition,
    temporalKnowledge: runtime.temporalKnowledge,
    uncertaintyState: runtime.uncertaintyState,
    selfImprovement: runtime.selfImprovement,
    reasoningMemory: runtime.reasoningMemory,
    reasoningQuality: runtime.reasoningQuality,
    runtimeOptions: runtime.options
  }, runtime.options);


  runtime.semanticComposer =
    new VilotSemanticComposer(
      runtime.processor,
      runtime.trainingKnowledge,
      runtime.options
    );


  runtime.clausePlanner =
    new VilotClausePlanner(
      runtime.processor,
      runtime.stateProcessor,
      runtime.options
    );


  runtime.ideaFusion =
    new VilotIdeaFusion(
      runtime.processor,
      runtime.options
    );


  runtime.keywordSelector =
    new VilotKeywordSelector(
      runtime.processor,
      runtime.options
    );


  runtime.syntacticBridge =
    new VilotSyntacticBridge(
      runtime.processor,
      runtime.options,
      runtime.ruleStore
    );


  runtime.syntaxSmoother =
    new VilotSyntaxSmoother(
      runtime.processor,
      runtime.options,
      runtime.ruleStore
    );


  runtime.surfaceRealizer =
    new VilotSurfaceRealizer(
      runtime.processor,
      runtime.options,
      runtime.keywordSelector,
      runtime.syntacticBridge,
      runtime.syntaxSmoother
    );


  runtime.mixedSlotPlanner =
    new VilotMixedSlotPlanner(
      runtime.options
    );


  runtime.outputCoherence =
    new VilotOutputCoherence(
      runtime.processor,
      runtime.options
    );


  runtime.sentenceMixer =
    new VilotSentenceMixer(
      runtime.processor,
      runtime.options,
      runtime.matrix,
      runtime.ruleStore
    );


  runtime.decoder = new VilotInterferenceDecoder(
    runtime.baseDecoder,
    runtime.processor,
    runtime.options,
    runtime.trainingKnowledge,
    runtime.stateProcessor,
    runtime.semanticComposer,
    runtime.clausePlanner,
    runtime.ideaFusion,
    runtime.surfaceRealizer,
    runtime.mixedSlotPlanner,
    runtime.outputCoherence,
    runtime.learnedKnowledgeBridge,
    runtime.claimVerifier,
    runtime.uncertaintyState,
    runtime.counterfactualWorkspace,
    runtime.sentenceMixer
  );

  runtime.fastLane = new VilotFastLane(
    runtime.decoder,
    runtime.correlation,
    runtime.options,
    runtime.queryPlanner,
    runtime.evidenceSearch,
    runtime.candidateRanker
  );

  runtime.fastLane.counterfactualWorkspace = runtime.counterfactualWorkspace;


  runtime.coldStartWarmup =
    new VilotColdStartWarmup(
      {

        compute:
          runtime.compute,
        processor:
          runtime.processor,
        languageState:
          runtime.languageState,
        queryPlanner:
          runtime.queryPlanner,
        evidenceSearch:
          runtime.evidenceSearch,
        candidateRanker:
          runtime.candidateRanker,
        decoder:
          runtime.decoder,
        trainingKnowledge:
          runtime.trainingKnowledge
      
}
,
      runtime.options
    );


  // Complete first-use compilation/JIT work while the runtime still reports
  // BOOTING so the first real user request does not pay the cold-start bill.
  await runtime.coldStartWarmup.run();


  runtime.ready = true;

  runtime.lastForegroundAt = performance.now();

  runtime.lastHomeostasisAt = runtime.lastForegroundAt;

  setPhase(Phase.ACTIVE, 'boot');

  runtime.trace.push(VilotTraceEvent.BOOT_READY, 0, runtime.arena.stateDim);

  scheduleBackground();

  return statusPayload();

}


function foregroundUpdate(vector, options = {
}
) {

  const start = performance.now();

  runtime.foregroundEpoch++;

  runtime.lastForegroundAt = start;

  setPhase(Phase.ACTIVE, 'foreground-command');


  runtime.arena.setInput(vector || [], Number.isFinite(options.scale) ? options.scale : 1);

  runtime.compute.recurrent({

    decay: Number.isFinite(options.decay) ? options.decay : 0.985,
    inputGain: Number.isFinite(options.inputGain) ? options.inputGain : 0.36,
    inhibition: Number.isFinite(options.inhibition) ? options.inhibition : 0.018,
    baselinePull: Number.isFinite(options.baselinePull) ? options.baselinePull : 0.002,
    tick: runtime.tickCount,
    denseInput: runtime.arena.input
  
}
);

  runtime.arena.clearInput();


  runtime.lastHotPathMs = performance.now() - start;

  runtime.trace.push(
    VilotTraceEvent.FOREGROUND_UPDATE,
    runtime.lastHotPathMs,
    runtime.foregroundEpoch,
    runtime.arena.revision
  );

  scheduleBackground();


  return {

    hotPathMs: runtime.lastHotPathMs,
    stateRevision: runtime.arena.revision,
    foregroundEpoch: runtime.foregroundEpoch
  
}
;

}


function runLocalBenchmark(iterations = 2000) {

  iterations = Math.max(50, Math.min(50000, iterations | 0));

  const arena = runtime.arena;


  // Preserve the real persistent state. Benchmark on the alternate preallocated
  // buffers and restore references afterward without allocating tensors.
  const savedRead = arena.readState;

  const savedWrite = arena.writeState;

  const savedRevision = arena.revision;

  const savedSwapCount = arena.swapCount;


  const hadDenseInput = Boolean(arena.input);

  const bench = arena.ensureBenchmarkBuffers();

  const denseInput = arena.ensureDenseInput();

  arena.readState = bench.a;

  arena.writeState = bench.b;

  bench.a.fill(0.0125);

  bench.b.fill(0);

  denseInput.fill(0.025);


  const samples = new Float64Array(iterations);

  for (let n = 0;
 n < iterations;
 n++) {

    const t0 = performance.now();

    runtime.compute.runCPU(true, 0.985, 0.36, 0.018, 0.002);

    samples[n] = performance.now() - t0;

  
}


  let sum = 0;

  let min = Infinity;

  let max = 0;

  for (let i = 0;
 i < samples.length;
 i++) {

    const v = samples[i];

    sum += v;

    if (v < min) min = v;

    if (v > max) max = v;

  
}


  // Sorting is diagnostic-only, never part of the hot path.
  const sorted = Array.from(samples).sort((a, b) => a - b);

  const p50 = sorted[Math.floor(sorted.length * 0.50)] || 0;

  const p95 = sorted[Math.floor(sorted.length * 0.95)] || 0;


  arena.readState = savedRead;

  arena.writeState = savedWrite;

  arena.revision = savedRevision;

  arena.swapCount = savedSwapCount;

  arena.clearInput();

  arena.releaseBenchmarkBuffers();

  if (!hadDenseInput) arena.input = null;


  runtime.localBenchmark = {

    iterations,
    meanMs: sum / iterations,
    minMs: min,
    p50Ms: p50,
    p95Ms: p95,
    maxMs: max,
    stateDim: arena.stateDim,
    note: 'CPU recurrent-kernel benchmark inside the worker; excludes message roundtrip and GPU completion.'
  
}
;

  runtime.trace.push(VilotTraceEvent.BENCHMARK, 0, runtime.localBenchmark.meanMs, runtime.localBenchmark.p95Ms);

  return runtime.localBenchmark;

}


async function statusPayload(sync = false) {

  let energy = null;

  if (sync && runtime.compute) {

    const start = performance.now();

    energy = await runtime.compute.syncStateToCPU();

    runtime.trace?.push(VilotTraceEvent.STATE_SYNC, performance.now() - start, energy.meanAbs, energy.peak);

  
}


  return {

    buildId: V25_BUILD_ID,
    ready: runtime.ready,
    phase: runtime.phase,
    visible: runtime.visible,
    tickCount: runtime.tickCount,
    foregroundEpoch: runtime.foregroundEpoch,
    lastHotPathMs: runtime.lastHotPathMs,
    arena: runtime.arena?.stats?.() || null,
    compute: runtime.compute?.info?.() || null,
    energy,
    localBenchmark: runtime.localBenchmark,
    rsl: runtime.rsl?.status?.() || null,
    processor: runtime.processor?.status?.() || null,
    lexicon: runtime.lexicon?.status?.() || null,
    wordState: runtime.wordState?.status?.() || null,
    phraseState: runtime.phraseState?.status?.() || null,
    wordMatrix: runtime.wordMatrix?.status?.() || null,
    languageState: runtime.languageState?.status?.() || null,
    decoder: runtime.decoder?.status?.() || null,
    memory: runtime.memory?.status?.() || null,
    dialogueContext: {
 turns: runtime.dialogueContext.length, capacity: runtime.dialogueContextCapacity 
}
,
    correlation: runtime.correlation?.status?.() || null,
    cognitive: runtime.cognitive?.status?.() || null,
    learningController: runtime.learningController?.status?.() || null,
    knowledgePolicy: runtime.knowledgePolicy?.status?.() || null,
    knowledgeGraph: runtime.knowledgeGraph?.status?.() || null,
    temporalKnowledge: runtime.temporalKnowledge?.status?.() || null,
    graphReasoner: runtime.graphReasoner?.status?.() || null,
    learnedKnowledgeBridge: runtime.learnedKnowledgeBridge?.status?.() || null,
    uncertaintyState: runtime.uncertaintyState?.status?.() || null,
    counterfactualWorkspace: runtime.counterfactualWorkspace?.status?.() || null,
    claimVerifier: runtime.claimVerifier?.status?.() || null,
    predictionError: runtime.predictionError?.status?.() || null,
    hypothesisStore: runtime.hypothesisStore?.status?.() || null,
    replayScheduler: runtime.replayScheduler?.status?.() || null,
    generalization: runtime.generalization?.status?.() || null,
    contradictionResolver: runtime.contradictionResolver?.status?.() || null,
    learningEvaluator: runtime.learningEvaluator?.status?.() || null,
    entityResolver: runtime.entityResolver?.status?.() || null,
    architectureKnowledge: runtime.architectureKnowledge?.status?.() || null,
    explanationPlanner: runtime.explanationPlanner?.status?.() || null,
    propositionGraph: runtime.propositionGraph?.status?.() || null,
    mechanismGraph: runtime.mechanismGraph?.status?.() || null,
    reasoningMemory: runtime.reasoningMemory?.status?.() || null,
    reasoningQuality: runtime.reasoningQuality?.status?.() || null,
    conversationalCognition: runtime.conversationalCognition?.status?.() || null,
    reasoningCoordinator: runtime.reasoningCoordinator?.status?.() || null,
    selfImprovement: runtime.selfImprovement?.status?.() || null,
    semanticComposer: runtime.semanticComposer?.status?.() || null,
    clausePlanner: runtime.clausePlanner?.status?.() || null,
    ideaFusion: runtime.ideaFusion?.status?.() || null,
    keywordSelector: runtime.keywordSelector?.status?.() || null,
    syntacticBridge: runtime.syntacticBridge?.status?.() || null,
    syntaxSmoother: runtime.syntaxSmoother?.status?.() || null,
    surfaceRealizer: runtime.surfaceRealizer?.status?.() || null,
    mixedSlotPlanner: runtime.mixedSlotPlanner?.status?.() || null,
    outputCoherence: runtime.outputCoherence?.status?.() || null,
    coldStartWarmup: runtime.coldStartWarmup?.status?.() || null,
    matrix: runtime.matrix?.status?.() || null,
    queryPlanner: runtime.queryPlanner?.status?.() || null,
    evidenceSearch: runtime.evidenceSearch?.status?.() || null,
    candidateRanker: runtime.candidateRanker?.status?.() || null,
    latencyProfiler: runtime.latencyProfiler?.status?.() || null,
    latencyGovernor: runtime.latencyGovernor?.status?.() || null,
    ruleStore: runtime.ruleStore?.status?.() || null,
    grammarEngine: runtime.grammarEngine?.status?.() || null,
    fastLane: runtime.fastLane?.status?.() || null,
    learningQueue: {

      pending: runtime.learningQueue.length,
      processed: runtime.learningProcessed,
      dropped: runtime.learningDropped,
      adaptivePending: runtime.learningController?.queue?.length || 0
    
}
,
    storage: runtime.storage?.status?.() || null,
    goals: {

      neuronCount: runtime.arena?.stateDim || 0,
      minResponseConfidence: Math.max(80, Number(runtime.options.minResponseConfidence) || 80),
      latencyTargetMs: Number(runtime.options.latencyTargetMs) || 40,
      highEffort: true,
      unifiedVilotNI2: true,
      sixMillionNeuronCore: true,
      trainingInfoRequired: true,
      trainingInfoPresentAsReference: false,
      trainingInfoLoadedForResponses: true,
      dynamicPOSPerOccurrence: true,
      continuousBackgroundLearning: true,
      predictionErrorLearning: true,
      provisionalHypothesesSeparatedFromFacts: true,
      persistentKnowledgeGraph: true,
      prioritizedSpacedReplay: true,
      crossExampleGeneralization: true,
      contradictionResolution: true,
      learningQualityEvaluation: true,
      autonomousHypothesisTesting: true,
      provenanceWeightedLearning: true,
      learnedKnowledgeFeedsResponses: true,
      relationSpecificGraphReasoning: true,
      semanticClaimVerification: true,
      multiDimensionalUncertainty: true,
      temporalKnowledgeValidity: true,
      semanticEpisodicRetrieval: true,
      counterfactualIsolation: true,
      learningRegressionRollback: true,
      cognitiveExecutiveGoals: true,
      foregroundReasoningWorkspace: true,
      alwaysOnConversationalCognition: true,
      priorContextNotRequiredForResponse: true,
      interactionClassificationSoftOnly: true,
      backgroundThoughtIntegration: true,
      amplitudeReasoningStateSelection: true,
      architectureSelfKnowledge: true,
      autonomousCognitiveSelfStudy: true,
      userIndependentTrainingInfoStudy: true,
      meaningFirstOutput: true,
      semanticCollapseBeforeSurfaceWords: true,
      multiIdeaSentenceFusion: true,
      globalFourPathMixedSlotPlanning: true,
      antiWordSaladCoherenceGate: true,
      continuousCorrelationLearning: true,
      unifiedCorrelationLearning: true,
      curiosityExploration: true,
      confidenceRemakesUntilAboveFloor: true,
      blockingLearningDuringGeneration: false,
      batchedCandidateSearch: true,
      focusedQueryPlanning: true,
      focusedEvidenceSearch: true,
      nonInflatingCandidateRanking: true,
      externalGrammarRules: true,
      externalPunctuationRules: true,
      externalQueryPatterns: true,
      externalContextRules: true,
      semanticRelationRouting: true,
      lexicalDefinitionIndex: true,
      residentCandidateMatrix: true,
      learnedWordState: true,
      learnedPhraseState: true,
      learnedWordMatrix: true,
      adaptiveLexicalConfidence: true,
      acceptedResponseLearningDeferred: true,
      productionRuntime: true,
      orderedRecurrentBatching: true,
      sparseActiveExecution: true,
      exactZeroSkip: true,
      foregroundPriorityScheduling: true
    
}
,
    backgroundIntervals: {

      activeMs: runtime.options.activeTickMs,
      warmMs: runtime.options.warmTickMs,
      idleMs: runtime.options.idleTickMs,
      activeHomeostasisMs: runtime.options.activeHomeostasisMs,
      warmHomeostasisMs: runtime.options.warmHomeostasisMs,
      idleHomeostasisMs: runtime.options.idleHomeostasisMs
    
}

  
}
;

}


onmessage = async event => {

  const message = event.data || {
}
;

  const type = message.type || '';

  const requestId = message.requestId ?? null;

  runtime.trace?.push?.(VilotTraceEvent.COMMAND, 0, requestId || 0);


  try {

    if (type !== 'BOOT' && !runtime.ready) {

      await boot({
}
);

    
}


    switch (type) {

      case 'BOOT': {

        const status = await boot(message.options || {
}
);

        send('READY', {
 status 
}
, requestId);

        break;

      
}


      case 'PING':
        send('PONG', {
 workerNow: performance.now(), stateRevision: runtime.arena.revision 
}
, requestId);

        break;


      case 'FOREGROUND_UPDATE': {

        const result = foregroundUpdate(message.vector, message.options || {
}
);

        send('FOREGROUND_DONE', result, requestId);

        break;

      
}


      case 'RSL_OBSERVE': {

        const result = await runtime.rsl.observeText(message.text || '', message.options || {
}
);

        runtime.lastForegroundAt = performance.now();

        runtime.foregroundEpoch++;

        setPhase(Phase.ACTIVE, 'rsl-observe');

        scheduleBackground();

        send('RSL_RESULT', {
 result 
}
, requestId);

        break;

      
}


      case 'RSL_STATUS':
        send('RSL_STATUS_RESULT', {
 rsl: runtime.rsl.status() 
}
, requestId);

        break;


      case 'GENERATE': {

        const totalStart = performance.now();

        runtime.foregroundBusy = true;

        runtime.lastForegroundAt = totalStart;

        runtime.foregroundEpoch++;

        setPhase(Phase.ACTIVE, 'generate-fast-lane');


        const promptText = String(message.text || '');


        const latencyBudget =
          runtime.latencyGovernor?.begin?.(totalStart) || {

            started: totalStart,
            workDeadlineAt:
              totalStart +
              runtime.options.latencyTargetMs -
              6,
            responseDeadlineAt:
              totalStart +
              runtime.options.latencyTargetMs
          
}
;


        const rslStarted = performance.now();


        const promptState = await runtime.rsl.observeText(promptText, {

          ...(message.options || {
}
),
          countTurn: true,
          syncState: false,
          source: 'user',
          learnTransitions: true,
          maxSteps: Math.min(
            runtime.options.latencyRSLMaxSteps || 48,
            runtime.options.maxTokens || 192
          )
        
}
);


        runtime.latencyGovernor?.observe?.(
          'rsl',
          performance.now() - rslStarted
        );


        const analysisStarted = performance.now();


        const baseAnalysis = runtime.languageState.enrichAnalysis(
          runtime.processor.analyzePrompt(promptText)
        );

        const entityAnalysis = runtime.entityResolver?.enrichAnalysis?.(
          baseAnalysis,
          promptText
        ) || baseAnalysis;

        const architectureAnalysis = runtime.architectureKnowledge?.enrichAnalysis?.(
          entityAnalysis,
          promptText
        ) || entityAnalysis;

        // Conversational cognition is always present, but only as a soft signal.
        // It does not select a mode or gate the normal reasoning architecture.
        const analysis = runtime.conversationalCognition?.enrichAnalysis?.(
          architectureAnalysis,
          promptText,
          dialogueContextSnapshot(12)
        ) || architectureAnalysis;

        analysis.propositionGraph = runtime.propositionGraph?.build?.(
          promptText,
          analysis
        ) || null;
        analysis.mechanismGraph = runtime.mechanismGraph?.build?.(
          promptText,
          analysis,
          analysis.propositionGraph
        ) || null;
        analysis.temporalAssessment = runtime.temporalKnowledge?.analyzePropositionGraph?.(
          analysis.propositionGraph
        ) || null;
        analysis.propositionUncertainty = runtime.uncertaintyState?.assessPropositions?.(
          analysis.propositionGraph,
          { analysis, mechanismGraph: analysis.mechanismGraph, temporalAssessment: analysis.temporalAssessment }
        ) || [];
        analysis.reasoningMemoryHints = runtime.reasoningMemory?.retrieve?.(analysis, 6) || [];
        analysis.reasoningQualityPreflight = runtime.reasoningQuality?.preflight?.(analysis) || null;

        runtime.processor?.enrichCognitivePacket?.(analysis, {
          correlation: runtime.correlation,
          convergenceEpsilon: runtime.options.cognitiveForegroundConvergenceEpsilon
        });
        runtime.correlation?.observePropositionGraph?.(analysis.propositionGraph || analysis.complexLanguage, {
          source: 'foreground-structured-reasoning',
          prompt: promptText
        });
        const foregroundCognitiveState = runtime.cognitive?.thinkForeground?.(
          promptText,
          analysis,
          {
            deadlineAt: latencyBudget.workDeadlineAt,
            dialogueContext: dialogueContextSnapshot(12)
          }
        ) || null;
        if (foregroundCognitiveState) analysis.cognitiveState = foregroundCognitiveState;

        const explanationPlan = runtime.explanationPlanner?.plan?.(
          promptText,
          analysis
        ) || null;
        if (explanationPlan) analysis.explanationPlan = explanationPlan;
        if (analysis.mechanismGraph?.active || explanationPlan?.active) {
          analysis.mechanismGraph = runtime.mechanismGraph?.build?.(
            promptText,
            analysis,
            analysis.propositionGraph
          ) || analysis.mechanismGraph;
          analysis.temporalAssessment = runtime.temporalKnowledge?.analyzePropositionGraph?.(analysis.propositionGraph) || analysis.temporalAssessment;
          analysis.propositionUncertainty = runtime.uncertaintyState?.assessPropositions?.(
            analysis.propositionGraph,
            { analysis, mechanismGraph: analysis.mechanismGraph, temporalAssessment: analysis.temporalAssessment }
          ) || analysis.propositionUncertainty || [];
          analysis.reasoningQualityPreflight = runtime.reasoningQuality?.preflight?.(analysis) || analysis.reasoningQualityPreflight;
          runtime.processor?.enrichCognitivePacket?.(analysis, {
            correlation: runtime.correlation,
            convergenceEpsilon: runtime.options.cognitiveForegroundConvergenceEpsilon
          });
        }


        runtime.latencyGovernor?.observe?.(
          'analysis',
          performance.now() - analysisStarted
        );


        const foregroundCounterfactual =
          runtime.counterfactualWorkspace?.analyze?.(
            promptText,
            analysis
          ) || {
 active: false 
}
;


        analysis.counterfactual = Boolean(foregroundCounterfactual.active);
        const reasoningStarted = performance.now();
        const reasoningWorkspace = runtime.reasoningCoordinator?.plan?.(
          promptText,
          analysis,
          {
            deadlineAt: latencyBudget.workDeadlineAt,
            counterfactualWorkspace: foregroundCounterfactual,
            dialogueContext: dialogueContextSnapshot(12)
          }
        ) || null;

        if (reasoningWorkspace) {
          analysis.reasoningWorkspace = reasoningWorkspace;
          analysis.reasoningPriorityConcepts = reasoningWorkspace.priorityTerms || [];
          analysis.reasoningBranchModes = reasoningWorkspace.branchModes || [];
        }

        runtime.latencyGovernor?.observe?.(
          'reasoning',
          performance.now() - reasoningStarted
        );


        // Full prompt learning remains queued for background consolidation.
        enqueueLearning({

          type: 'user',
          text: promptText,
          counterfactual: Boolean(foregroundCounterfactual.active),
          analysisSummary: {

            counterfactual: Boolean(foregroundCounterfactual.active),
            intent: analysis.intent,
            requestedSlot: analysis.requestedSlot,
            contentWords: (analysis.contentWords || []).slice(0, 24),
            entityNames: (analysis.entities || []).map(row => row.canonical).slice(0, 8),
            architectureComponents: (analysis.architectureComponents || []).map(row => row.canonical).slice(0, 8),
            reasoningState: reasoningWorkspace?.selectedState?.mode || null,
            reasoningPriority: (reasoningWorkspace?.priorityTerms || []).slice(0, 12),
            explanationKind: analysis?.explanationPlan?.kind || null,
            explanationEvidence: Number(analysis?.explanationPlan?.directEvidence || 0),
            conversationFreedom: Number(analysis?.conversationState?.responseFreedom || 0),
            conversationAnchors: (analysis?.conversationCurrentAnchors || []).slice(0, 12),
            propositionCoverage: Number(analysis?.propositionGraph?.metrics?.structuralCoverage || 0),
            mechanismActive: Boolean(analysis?.mechanismGraph?.active),
            mechanismCompleteness: Number(analysis?.mechanismGraph?.completeness || 0),
            reasoningMemoryHints: (analysis?.reasoningMemoryHints || []).slice(0, 4).map(row => row.mode)
          
}

        
}
);


        const minConfidence = Math.max(80, Number(runtime.options.minResponseConfidence) || 80);

        const priorDialogueContext = dialogueContextSnapshot(12);


        const search = await runtime.fastLane.run(
          promptText,
          analysis,
          {

            ...(message.options || {
}
),
            minConfidence,
            dialogueContext: priorDialogueContext,
            reasoningWorkspace,
            deadlineAt: latencyBudget.workDeadlineAt,
            responseDeadlineAt: latencyBudget.responseDeadlineAt,
            latencyGovernor: runtime.latencyGovernor,
            latencyStart: totalStart
          
}
,
          progress => send('RESPONSE_PROGRESS', progress)
        );


        if (search.contextRequired) {

          const clarificationText = String(search.clarificationText || 'What should I use as the context for that question?');

          runtime.foregroundBusy = false;

          scheduleBackground();

          const totalMs = performance.now() - totalStart;


          runtime.latencyGovernor?.finish?.(
            totalStart,
            {

              route: search.route || 'context-required',
              confidence: 0,
              candidateBatches: 0,
              candidateCount: 0,
              deadlineExceeded: Boolean(search.deadlineExceeded)
            
}

          );


          send('GENERATE_RESULT', {

            result: {

              text: clarificationText,
              confidence: null,
              contextRequired: true,
              contextResolved: false,
              contextReason: search.planSummary?.contextReason || 'context-dependent',
              attempts: 0,
              candidateBatches: 0,
              totalMs,
              latencyTargetMs: runtime.options.latencyTargetMs,
              withinLatencyTarget: totalMs < runtime.options.latencyTargetMs,
              latencyWarning: totalMs >= runtime.options.latencyTargetMs,
              fastLaneMs: search.fastLaneMs,
              fastLaneRoute: search.route,
              analysis: {

                intent: analysis.intent,
                requestedSlot: analysis.requestedSlot,
                isQuestion: analysis.isQuestion,
                lexicalCoverage: analysis.lexicalCoverage,
                contentWords: (analysis.contentWords || []).slice(0, 24),
            entityNames: (analysis.entities || []).map(row => row.canonical).slice(0, 8),
            architectureComponents: (analysis.architectureComponents || []).map(row => row.canonical).slice(0, 8),
            reasoningState: reasoningWorkspace?.selectedState?.mode || null,
            reasoningPriority: (reasoningWorkspace?.priorityTerms || []).slice(0, 12),
            conversationFreedom: Number(analysis?.conversationState?.responseFreedom || 0),
            conversationAnchors: (analysis?.conversationCurrentAnchors || []).slice(0, 12),
            propositionCoverage: Number(analysis?.propositionGraph?.metrics?.structuralCoverage || 0),
            mechanismActive: Boolean(analysis?.mechanismGraph?.active),
            mechanismCompleteness: Number(analysis?.mechanismGraph?.completeness || 0),
            reasoningMemoryHints: (analysis?.reasoningMemoryHints || []).slice(0, 4).map(row => row.mode)
              
}

            
}

          
}
, requestId);

          break;

        
}


        if (search.deadlineRequired) {

          const deadlineText = String(
            search.clarificationText ||
            deadlineClarification(promptText, analysis, search)
          );


          // This is an abstention/clarification, not a sub-80 factual answer.
          // The >80 factual-emission invariant remains intact.
          pushDialogueTurn('user', promptText);

          pushDialogueTurn('assistant', deadlineText, {

            confidence: 0,
            latencyAbstention: true
          
}
);


          runtime.foregroundBusy = false;

          const totalMs = performance.now() - totalStart;


          runtime.latencyGovernor?.finish?.(
            totalStart,
            {

              route: search.route || 'hard-deadline-abstention',
              confidence: 0,
              candidateBatches: search.batchCount || 0,
              candidateCount: search.candidateCount || 0,
              deadlineExceeded: true,
              latencyAbstention: true
            
}

          );


          send('GENERATE_RESULT', {

            result: {

              text: deadlineText,
              confidence: null,
              latencyAbstention: true,
              contextRequired: false,
              attempts: search.candidateCount || 0,
              candidateBatches: search.batchCount || 0,
              totalMs,
              latencyTargetMs: runtime.options.latencyTargetMs,
              withinLatencyTarget: totalMs < runtime.options.latencyTargetMs,
              latencyWarning: totalMs >= runtime.options.latencyTargetMs,
              protectedWorkDeadlineMs:
                runtime.options.latencyTargetMs -
                runtime.options.latencySafetyReserveMs,
              deadlineExceeded: true,
              searchStoppedBy:
                search.searchStoppedBy ||
                'latency-hard-deadline',
              bestObservedConfidence:
                Number(search.bestObservedConfidence) ||
                Number(search.bestCandidate?.confidence) ||
                0,
              fastLaneMs: search.fastLaneMs,
              fastLaneRoute: search.route,
              analysis: {

                intent: analysis.intent,
                requestedSlot: analysis.requestedSlot,
                isQuestion: analysis.isQuestion,
                lexicalCoverage: analysis.lexicalCoverage,
                contentWords: (analysis.contentWords || []).slice(0, 24),
            entityNames: (analysis.entities || []).map(row => row.canonical).slice(0, 8),
            architectureComponents: (analysis.architectureComponents || []).map(row => row.canonical).slice(0, 8),
            reasoningState: reasoningWorkspace?.selectedState?.mode || null,
            reasoningPriority: (reasoningWorkspace?.priorityTerms || []).slice(0, 12),
            conversationFreedom: Number(analysis?.conversationState?.responseFreedom || 0),
            conversationAnchors: (analysis?.conversationCurrentAnchors || []).slice(0, 12),
            propositionCoverage: Number(analysis?.propositionGraph?.metrics?.structuralCoverage || 0),
            mechanismActive: Boolean(analysis?.mechanismGraph?.active),
            mechanismCompleteness: Number(analysis?.mechanismGraph?.completeness || 0),
            reasoningMemoryHints: (analysis?.reasoningMemoryHints || []).slice(0, 4).map(row => row.mode)
              
}

            
}

          
}
, requestId);


          // Everything after the response commit is background-only.
          setTimeout(() => {

            enqueueLearning({

              type: 'deadline_miss',
              promptText,
              text: String(search.bestCandidate?.text || ''),
              responseText: '',
              confidence: Number(search.bestCandidate?.confidence) || 0,
              route: search.route || 'hard-deadline-abstention',
              counterfactual: Boolean(foregroundCounterfactual.active),
              analysisSummary: {

                counterfactual: Boolean(foregroundCounterfactual.active),
                intent: analysis.intent,
                requestedSlot: analysis.requestedSlot,
                contentWords: (analysis.contentWords || []).slice(0, 24),
            entityNames: (analysis.entities || []).map(row => row.canonical).slice(0, 8),
            architectureComponents: (analysis.architectureComponents || []).map(row => row.canonical).slice(0, 8),
            reasoningState: reasoningWorkspace?.selectedState?.mode || null,
            reasoningPriority: (reasoningWorkspace?.priorityTerms || []).slice(0, 12),
            conversationFreedom: Number(analysis?.conversationState?.responseFreedom || 0),
            conversationAnchors: (analysis?.conversationCurrentAnchors || []).slice(0, 12),
            propositionCoverage: Number(analysis?.propositionGraph?.metrics?.structuralCoverage || 0),
            mechanismActive: Boolean(analysis?.mechanismGraph?.active),
            mechanismCompleteness: Number(analysis?.mechanismGraph?.completeness || 0),
            reasoningMemoryHints: (analysis?.reasoningMemoryHints || []).slice(0, 4).map(row => row.mode)
              
}

            
}
);

            scheduleBackground();

          
}
, 0);

          break;

        
}


        let generated = search.selected;

        const bestObserved = generated || search.bestCandidate || null;

        const bestObservedConfidence = Number(bestObserved?.confidence) || 0;

        if (!generated || !(Number(generated?.confidence) > minConfidence)) {

          throw new Error(`FastLane confidence invariant failed: response did not clear >${minConfidence}%.`);

        
}

        const withheld = false;


        const rawEmittedText = String(generated?.text || '');

        // Final universal syntax pass. Every selected VilotNI 2.5 answer, no
        // matter which FastLane/coherence route produced it, receives the same
        // conservative grammar cleanup before it reaches the user. Semantic
        // bridge insertion happens earlier in Decoder/SurfaceRealizer; this pass
        // only normalizes the finished surface and is accepted only when the
        // audited grammar is not worse than the original candidate.
        const finalSyntaxPass = runtime.syntaxSmoother?.smooth?.(rawEmittedText, analysis) || {
          text: rawEmittedText, repairs: [], diagnostics: null
        };

        const rawSyntaxAudit = runtime.processor?.auditResponse?.(rawEmittedText) || null;
        const smoothSyntaxAudit = runtime.processor?.auditResponse?.(finalSyntaxPass.text) || null;
        const rawGrammar = Number(rawSyntaxAudit?.grammarScore ?? rawSyntaxAudit?.legalRatio ?? 0);
        const smoothGrammar = Number(smoothSyntaxAudit?.grammarScore ?? smoothSyntaxAudit?.legalRatio ?? 0);
        const acceptGlobalSmoothing = Boolean(finalSyntaxPass.text) &&
          (smoothSyntaxAudit?.illegal || 0) === 0 &&
          (!rawSyntaxAudit || smoothGrammar + 0.015 >= rawGrammar);

        const syntaxSelectedText = acceptGlobalSmoothing
          ? String(finalSyntaxPass.text || rawEmittedText)
          : rawEmittedText;

        const emittedText = runtime.entityResolver?.restoreCasing?.(
          syntaxSelectedText,
          analysis?.entities || []
        ) || syntaxSelectedText;

        const globalSyntaxDiagnostics = {
          applied: acceptGlobalSmoothing,
          repairCount: finalSyntaxPass?.repairs?.length || 0,
          rawGrammar: Number.isFinite(rawGrammar) ? Number(rawGrammar.toFixed(4)) : null,
          smoothedGrammar: Number.isFinite(smoothGrammar) ? Number(smoothGrammar.toFixed(4)) : null,
          details: finalSyntaxPass?.diagnostics || null
        };

        const reasoningEvaluation = runtime.reasoningCoordinator?.evaluateResponse?.(
          emittedText,
          reasoningWorkspace
        ) || null;

        const structureVerification = runtime.claimVerifier?.verifyPromptStructure?.(
          emittedText,
          analysis
        ) || null;
        const propositionContradictions = runtime.contradictionResolver?.assessPropositionGraph?.(
          analysis?.propositionGraph
        ) || { count: 0, conflicts: [], safe: true };
        const reasoningQuality = runtime.reasoningQuality?.evaluate?.(
          emittedText,
          analysis,
          reasoningWorkspace,
          propositionContradictions
        ) || null;
        runtime.reasoningMemory?.observe?.({
          analysis,
          plan: reasoningWorkspace,
          evaluation: reasoningEvaluation,
          quality: reasoningQuality
        });
        const languageDiagnosis = runtime.selfImprovement?.diagnoseLanguageStructure?.({
          analysis,
          structureVerification,
          responseText: emittedText
        }) || null;

        if (generated?.diagnostics) {
          generated.diagnostics.reasoning = {
            selectedState: reasoningWorkspace?.selectedState || null,
            branchModes: reasoningWorkspace?.branchModes || [],
            unresolved: reasoningWorkspace?.unresolved || [],
            priorityTerms: (reasoningWorkspace?.priorityTerms || []).slice(0, 20),
            evaluation: reasoningEvaluation,
            languageStructure: structureVerification,
            languageDiagnosis,
            propositionContradictions,
            reasoningQuality,
            mechanismGraph: analysis?.mechanismGraph ? { active:analysis.mechanismGraph.active, completeness:analysis.mechanismGraph.completeness, missing:analysis.mechanismGraph.missing, stages:(analysis.mechanismGraph.stages||[]).slice(0,12) } : null
          };
        }

        runtime.selfImprovement?.observeResponse?.({
          promptText,
          responseText: emittedText,
          confidence: Number(generated?.confidence) || 0,
          channels: generated?.diagnostics?.channels || {},
          grammar: Number.isFinite(smoothGrammar) ? smoothGrammar : rawGrammar,
          semantic: Number(search?.rankSummary?.top?.[0]?.focus ?? generated?.diagnostics?.channels?.knowledgeSupport ?? 0.5),
          reasoningAlignment: Number(reasoningEvaluation?.alignment ?? search?.rankSummary?.top?.[0]?.reasoningAlignment ?? 0.5),
          reasoning: reasoningEvaluation,
          cognitiveState: analysis?.cognitiveState || null,
          contextAlignment: Number(search?.rankSummary?.top?.[0]?.conversationAlignment ?? 0.5),
          contradictionSafety: Number(generated?.diagnostics?.claimVerification?.consistency ?? 0.5),
          totalMs: performance.now() - totalStart,
          latencyTargetMs: runtime.options.latencyTargetMs,
          syntax: globalSyntaxDiagnostics,
          entityCount: analysis?.entities?.length || 0,
          structuralComplexity: Number(analysis?.structuralComplexity || 0),
          clauseCount: analysis?.complexLanguage?.clauseCount || 0,
          propositionCount: analysis?.complexLanguage?.propositionCount || 0,
          structureVerification,
          languageDiagnosis,
          reasoningQuality,
          mechanismActive: Boolean(analysis?.mechanismGraph?.active),
          propositionContradictions,
          analysis,
          uncertaintyState: generated?.uncertaintyState || generated?.diagnostics?.meaningFirst?.uncertaintyState || null,
          temporalConsistency: generated?.uncertaintyState?.temporalConsistency ?? null
        });

        pushDialogueTurn('user', promptText);

        pushDialogueTurn('assistant', emittedText, {
 confidence: Number(generated?.confidence) || 0 
}
);


        // Failed alternatives and selected-response learning are deferred until
        // after the user-visible response is committed.
        const confidenceHistory = search.failed
          .slice(-24)
          .map(x => Number(x?.confidence) || 0);

        if (generated) confidenceHistory.push(Number(generated?.confidence) || 0);

        const gatedDiagnostics = generated?.diagnostics
          ? {

              ...generated.diagnostics,
              confidenceGateFailed: false,
              confidenceHistory: confidenceHistory.slice(-128),
              foregroundLearningCycles: 0,
              learningDeferredToBackground: true,
              adaptivePredictionErrorLearningQueued: true,
              failedCandidatesQueuedForStudy: Math.min(96, search.failed.length),
              unifiedSixMillionCore: true,
              correlationActive: true,
              unifiedCorrelationActive: true,
              curiosityActive: true,
              curiositySelection: search.curiosity || null,
              batchedCandidateSearch: true,
              residentCandidateMatrix: true,
              learnedWordState: true,
              learnedPhraseState: true,
              learnedWordMatrix: true,
              generalLanguageState: true,
              tenseAspectMoodVoiceAgreement: true,
              epistemicUnknownHandling: true,
              lexicalAdaptation: generated?.diagnostics?.channels?.lexicalAdaptation ?? 0.5,
              fastLaneMs: search.fastLaneMs,
              fastLaneRoute: search.route || null,
              focusedFastPath: Boolean(search.focusedFastPath),
              queryPlan: search.planSummary || null,
              evidenceSearch: search.evidenceSummary || null,
              fastRank: search.rankSummary || null,
              batchCount: search.batchCount,
              candidateCount: search.candidateCount,
              remadeUntilAboveConfidenceFloor: true,
              globalSyntaxSmoothing: globalSyntaxDiagnostics
            
}

          : {

              confidenceGateFailed: true,
              confidenceHistory: confidenceHistory.slice(-128),
              bestObservedConfidence,
              confidenceFloor: minConfidence,
              foregroundLearningCycles: 0,
              learningDeferredToBackground: true,
              adaptivePredictionErrorLearningQueued: true,
              failedCandidatesQueuedForStudy: Math.min(96, search.failed.length),
              batchedCandidateSearch: true,
              fastLaneMs: search.fastLaneMs,
              batchCount: search.batchCount,
              candidateCount: search.candidateCount,
              searchExhausted: Boolean(search.searchExhausted),
              searchStoppedBy: search.searchStoppedBy || null,
              remadeUntilAboveConfidenceFloor: false,
              globalSyntaxSmoothing: globalSyntaxDiagnostics
            
}
;


        generated = {

          ...(generated || {
}
),
          text: emittedText,
          confidence: generated?.confidence,
          diagnostics: gatedDiagnostics,
          withheld,
          bestObservedConfidence,
          confidenceGate: minConfidence,
          confidenceRule: `>${minConfidence}`,
          attempts: search.candidateCount,
          candidateBatches: search.batchCount,
          selfLearningCycles: 0,
          searchExhausted: Boolean(search.searchExhausted),
          searchStoppedBy: search.searchStoppedBy || null,
          backgroundLearningQueued: runtime.learningQueue.length
        
}
;


        runtime.foregroundBusy = false;

        const totalMs = performance.now() - totalStart;


        runtime.latencyGovernor?.finish?.(
          totalStart,
          {

            route: search.route || null,
            confidence: Number(generated?.confidence) || 0,
            candidateBatches: search.batchCount || 0,
            candidateCount: search.candidateCount || 0,
            deadlineExceeded: Boolean(search.deadlineExceeded)
          
}

        );


        send('GENERATE_RESULT', {

          result: {

            text: emittedText,
            confidence: generated?.confidence,
            withheld: false,
            bestObservedConfidence,
            confidenceGate: minConfidence,
            confidenceRule: `>${minConfidence}`,
            attempts: search.candidateCount,
            candidateBatches: search.batchCount,
            totalMs,
            latencyTargetMs: runtime.options.latencyTargetMs,
            withinLatencyTarget: totalMs < runtime.options.latencyTargetMs,
            latencyWarning: totalMs >= runtime.options.latencyTargetMs,
            protectedWorkDeadlineMs:
              runtime.options.latencyTargetMs -
              runtime.options.latencySafetyReserveMs,
            deadlineExceeded: Boolean(search.deadlineExceeded),
            counterfactual: Boolean(generated?.counterfactual),
            analysis: {

              intent: analysis.intent,
              requestedSlot: analysis.requestedSlot,
              isQuestion: analysis.isQuestion
            
}

          
}

        
}
, requestId);


        setTimeout(() => {

          send('RESPONSE_DIAGNOSTICS', {

            diagnostics: generated?.diagnostics || null,
            semanticPlan: generated?.semanticPlan || null,
            reasoningPlan: reasoningWorkspace ? {
              mode: reasoningWorkspace.mode,
              selectedState: reasoningWorkspace.selectedState,
              branchModes: reasoningWorkspace.branchModes,
              unresolved: reasoningWorkspace.unresolved,
              priorityTerms: (reasoningWorkspace.priorityTerms || []).slice(0, 20),
              matrixState: reasoningWorkspace.matrixState,
              evaluation: reasoningEvaluation
            } : null,
            uncertaintyState: generated?.uncertaintyState || null,
            counterfactual: Boolean(generated?.counterfactual),
            promptState,
            fastLane: {

              route: search.route || null,
              plan: search.planSummary || null,
              evidence: search.evidenceSummary || null,
              rank: search.rankSummary || null,
              batchCount: search.batchCount,
              candidateCount: search.candidateCount
            
}

          
}
, requestId);

        
}
, 0);


        // Response is already committed. Correlation updates, curiosity mining,
        // failure replay, and adaptive-learning queue construction cannot add to
        // the visible response latency anymore.
        setTimeout(() => {

          runtime.cognitive?.observeForeground?.({

            promptText,
            responseText: emittedText,
            subject: analysis.contentWords?.[0] || '',
            intent: analysis.intent,
            requestedSlot: analysis.requestedSlot,
            confidence: Number(generated?.confidence) || 0,
            uncertaintyState: generated?.uncertaintyState || null,
            counterfactual: Boolean(generated?.counterfactual),
            reasoningAlignment: Number(reasoningEvaluation?.alignment ?? 0.5),
            reasoningState: reasoningWorkspace?.selectedState || null,
            conversationFreedom: Number(analysis?.conversationState?.responseFreedom || 0),
            conversationAnchors: (analysis?.conversationCurrentAnchors || []).slice(0, 12),
            deadlineMiss: false
          
}
);


          runtime.correlation?.noteSelected?.(
            promptText,
            generated,
            {
 confidence: generated?.confidence 
}

          );


          runtime.correlation?.queueCuriosityCandidates?.(
            promptText,
            search.consideredCandidates || search.failed || [],
            generated,
            analysis
          );


          for (const failed of (search.failed || []).slice(-96)) {

            enqueueLearning({

              type: 'failure',
              promptText,
              text: String(failed?.text || ''),
              responseText: String(failed?.text || ''),
              confidence: Number(failed?.confidence) || 0,
              channels: failed?.diagnostics?.channels || null,
              trace: failed?.diagnostics?.generationTrace || [],
              route: search.route || '',
              counterfactual: Boolean(generated?.counterfactual || foregroundCounterfactual.active),
              analysisSummary: {

                counterfactual: Boolean(generated?.counterfactual || foregroundCounterfactual.active),
                intent: analysis.intent,
                requestedSlot: analysis.requestedSlot,
                contentWords: (analysis.contentWords || []).slice(0, 24),
            entityNames: (analysis.entities || []).map(row => row.canonical).slice(0, 8),
            architectureComponents: (analysis.architectureComponents || []).map(row => row.canonical).slice(0, 8),
            reasoningState: reasoningWorkspace?.selectedState?.mode || null,
            reasoningPriority: (reasoningWorkspace?.priorityTerms || []).slice(0, 12),
            conversationFreedom: Number(analysis?.conversationState?.responseFreedom || 0),
            conversationAnchors: (analysis?.conversationCurrentAnchors || []).slice(0, 12),
            propositionCoverage: Number(analysis?.propositionGraph?.metrics?.structuralCoverage || 0),
            mechanismActive: Boolean(analysis?.mechanismGraph?.active),
            mechanismCompleteness: Number(analysis?.mechanismGraph?.completeness || 0),
            reasoningMemoryHints: (analysis?.reasoningMemoryHints || []).slice(0, 4).map(row => row.mode)
              
}

            
}
);

          
}


          enqueueLearning({

            type: 'response',
            promptText,
            text: emittedText,
            responseText: emittedText,
            confidence: Number(generated?.confidence) || 0,
            quality: outputLearningQuality(generated),
            trace: generated?.diagnostics?.generationTrace || [],
            route: search.route || '',
            counterfactual: Boolean(generated?.counterfactual || foregroundCounterfactual.active),
            analysisSummary: {

              counterfactual: Boolean(generated?.counterfactual || foregroundCounterfactual.active),
              intent: analysis.intent,
              requestedSlot: analysis.requestedSlot,
              contentWords: (analysis.contentWords || []).slice(0, 24),
            entityNames: (analysis.entities || []).map(row => row.canonical).slice(0, 8),
            architectureComponents: (analysis.architectureComponents || []).map(row => row.canonical).slice(0, 8),
            reasoningState: reasoningWorkspace?.selectedState?.mode || null,
            reasoningPriority: (reasoningWorkspace?.priorityTerms || []).slice(0, 12),
            conversationFreedom: Number(analysis?.conversationState?.responseFreedom || 0),
            conversationAnchors: (analysis?.conversationCurrentAnchors || []).slice(0, 12),
            propositionCoverage: Number(analysis?.propositionGraph?.metrics?.structuralCoverage || 0),
            mechanismActive: Boolean(analysis?.mechanismGraph?.active),
            mechanismCompleteness: Number(analysis?.mechanismGraph?.completeness || 0),
            reasoningMemoryHints: (analysis?.reasoningMemoryHints || []).slice(0, 4).map(row => row.mode)
            
}

          
}
);


          scheduleBackground();

        
}
, 0);

        break;

      
}


      case 'PROCESSOR_INSPECT': {

        send('PROCESSOR_INSPECT_RESULT', {
 word: runtime.processor.inspectWord(message.word || '') 
}
, requestId);

        break;

      
}


      case 'DECODER_INSPECT': {

        const results = runtime.decoder.inspectLearnedContext(message.query || '', message.limit || 8);

        send('DECODER_INSPECT_RESULT', {
 results 
}
, requestId);

        break;

      
}


      case 'DECODER_STATUS': {

        send('DECODER_STATUS_RESULT', {
 decoder: runtime.decoder.status(), processor: runtime.processor.status() 
}
, requestId);

        break;

      
}


      case 'CHECKPOINT': {

        // A checkpoint must capture the real GPU-resident state, not a stale
        // CPU mirror. This readback is deliberately outside the hot path.
        await runtime.compute.syncStateToCPU();

        runtime.arena.checkpoint();

        runtime.rsl?.checkpoint?.();

        send('CHECKPOINTED', {
 stateRevision: runtime.arena.revision 
}
, requestId);

        break;

      
}


      case 'RESTORE_CHECKPOINT':
        runtime.arena.restoreCheckpoint();

        runtime.rsl?.restoreCheckpoint?.();

        runtime.compute.uploadCPUState();

        send('RESTORED', {
 stateRevision: runtime.arena.revision 
}
, requestId);

        break;


      case 'RESET_STATE':
        runtime.arena.resetState();

        runtime.rsl?.reset?.();

        runtime.memory?.reset?.();

        runtime.dialogueContext.length = 0;

        runtime.correlation?.reset?.();

        runtime.wordState?.reset?.();

        runtime.languageState?.reset?.();

        runtime.phraseState?.reset?.();

        runtime.wordMatrix?.reset?.();

        runtime.learningController?.reset?.();

        try {
 await runtime.storage?.clear?.();
 
}
 catch (_) {
}

        runtime.trace?.push?.(VilotTraceEvent.RSL_RESET, 0);

        runtime.compute.uploadCPUState();

        send('RESET_DONE', {
 stateRevision: runtime.arena.revision 
}
, requestId);

        break;


      case 'SET_DIALOGUE_CONTEXT': {

        const incoming = Array.isArray(message.turns) ? message.turns : [];

        runtime.dialogueContext.length = 0;

        for (const raw of incoming.slice(-runtime.dialogueContextCapacity)) {

          const text = String(raw?.text || '').trim();

          if (!text) continue;

          runtime.dialogueContext.push({

            role: String(raw?.role || 'unknown').toLowerCase(),
            text: text.slice(0, 4096),
            at: Number(raw?.at) || Date.now(),
            confidence: Number.isFinite(Number(raw?.confidence)) ? Number(raw.confidence) : null
          
}
);

        
}

        send('DIALOGUE_CONTEXT_SET', {
 turns: runtime.dialogueContext.length 
}
, requestId);

        break;

      
}


      case 'CLEAR_DIALOGUE_CONTEXT':
        runtime.dialogueContext.length = 0;

        send('DIALOGUE_CONTEXT_CLEARED', {
 turns: 0 
}
, requestId);

        break;


      case 'SET_VISIBILITY':
        runtime.visible = message.visible !== false;

        if (runtime.visible && runtime.phase === Phase.SUSPENDED) {

          setPhase(Phase.WARM, 'document-visible');

          runtime.lastForegroundAt = performance.now() - runtime.options.warmAfterMs;

          scheduleBackground();

        
}
 else {

          chooseAutomaticPhase(performance.now());

          scheduleBackground();

        
}

        send('VISIBILITY_SET', {
 visible: runtime.visible, phase: runtime.phase 
}
, requestId);

        break;


      case 'SUSPEND':
        clearTimeout(runtime.timer);

        setPhase(Phase.SUSPENDED, message.reason || 'manual');

        runtime.trace.push(VilotTraceEvent.SUSPEND);

        send('SUSPENDED', {
 phase: runtime.phase 
}
, requestId);

        break;


      case 'RESUME':
        runtime.lastForegroundAt = performance.now();

        setPhase(Phase.WARM, 'resume');

        runtime.trace.push(VilotTraceEvent.RESUME);

        scheduleBackground();

        send('RESUMED', {
 phase: runtime.phase 
}
, requestId);

        break;


      case 'GET_STATUS': {

        const status = await statusPayload(Boolean(message.sync));

        send('STATUS', {
 status 
}
, requestId);

        break;

      
}


      case 'GET_TRACE':
        send('TRACE', {
 trace: runtime.trace.snapshot(message.limit || 64) 
}
, requestId);

        break;


      case 'BENCHMARK_LOCAL':
        send('BENCHMARK_RESULT', {
 benchmark: runLocalBenchmark(message.iterations) 
}
, requestId);

        break;


      case 'DESTROY':
        clearTimeout(runtime.timer);

        try {
 await runtime.storage?.flush?.();
 
}
 catch (_) {
}

        runtime.storage?.close?.();

        runtime.compute?.destroy?.();

        runtime.ready = false;

        send('DESTROYED', {
}
, requestId);

        close();

        break;


      default:
        throw new Error(`Unknown VilotNI 2 worker command: ${type}`);

    
}

  
}
 catch (error) {

    runtime.foregroundBusy = false;

    runtime.trace?.push?.(VilotTraceEvent.ERROR);

    send('ERROR', {

      message: String(error?.message || error),
      stack: String(error?.stack || '')
    
}
, requestId);

  
}

}
;


/**
 * Worker production research extension.
 *
 * Executable, data-driven numerical/statistical/graph utilities used by
 * diagnostics, background learning, calibration, matrix experiments and
 * future compute paths. No prompt-specific phrases or benchmark answers live
 * here. Methods stay out of the foreground hot path unless explicitly called.
 */
(() => {

  'use strict';


  class VilotWorkerProductionLab {

    constructor(options = {
}
) {

      this.options = {
 ...options 
}
;

      this.calls = 0;

      this.createdAt = performance?.now?.() ?? Date.now();

    
}


    clamp(x, lo = 0, hi = 1) {

      const n = Number(x);

      if (!Number.isFinite(n)) return lo;

      return Math.max(lo, Math.min(hi, n));

    
}


    sum(values) {

      let total = 0;

      for (const value of values || []) {

        const n = Number(value);

        if (Number.isFinite(n)) total += n;

      
}

      return total;

    
}


    mean(values) {

      let total = 0;

      let count = 0;

      for (const value of values || []) {

        const n = Number(value);

        if (!Number.isFinite(n)) continue;

        total += n;

        count++;

      
}

      return count ? total / count : 0;

    
}


    variance(values) {

      const input = Array.from(values || [], Number).filter(Number.isFinite);

      if (input.length < 2) return 0;

      const m = this.mean(input);

      let total = 0;

      for (const value of input) {

        const d = value - m;

        total += d * d;

      
}

      return total / input.length;

    
}


    stddev(values) {

      return Math.sqrt(Math.max(0, this.variance(values)));

    
}


    normalize(values, epsilon = 1e-9) {

      const input = Array.from(values || [], Number);

      let norm2 = 0;

      for (const value of input) if (Number.isFinite(value)) norm2 += value * value;

      const norm = Math.sqrt(norm2) + Math.max(Number.EPSILON, epsilon);

      return input.map(value => Number.isFinite(value) ? value / norm : 0);

    
}


    softmax(values, temperature = 1) {

      const input = Array.from(values || [], Number);

      if (!input.length) return [];

      const t = Math.max(1e-6, Number(temperature) || 1);

      let max = -Infinity;

      for (const value of input) if (Number.isFinite(value) && value > max) max = value;

      if (!Number.isFinite(max)) return input.map(() => 1 / input.length);

      const exps = input.map(value => Math.exp((Number.isFinite(value) ? value : -1e9) / t - max / t));

      const z = exps.reduce((a,b) => a + b, 0) || 1;

      return exps.map(value => value / z);

    
}


    logSoftmax(values, temperature = 1) {

      const probs = this.softmax(values, temperature);

      return probs.map(value => Math.log(Math.max(1e-12, value)));

    
}


    entropy(probabilities) {

      let h = 0;

      for (const raw of probabilities || []) {

        const p = Number(raw);

        if (!Number.isFinite(p) || p <= 0) continue;

        h -= p * Math.log2(p);

      
}

      return h;

    
}


    dot(a, b) {

      const n = Math.min(a?.length || 0, b?.length || 0);

      let total = 0;

      for (let i = 0;
 i < n;
 i++) total += (Number(a[i]) || 0) * (Number(b[i]) || 0);

      return total;

    
}


    cosine(a, b, epsilon = 1e-9) {

      const n = Math.min(a?.length || 0, b?.length || 0);

      let dot = 0, aa = 0, bb = 0;

      for (let i = 0;
 i < n;
 i++) {

        const x = Number(a[i]) || 0;

        const y = Number(b[i]) || 0;

        dot += x * y;

        aa += x * x;

        bb += y * y;

      
}

      return dot / (Math.sqrt(aa * bb) + Math.max(epsilon, Number.EPSILON));

    
}


    l1Distance(a, b) {

      const n = Math.max(a?.length || 0, b?.length || 0);

      let total = 0;

      for (let i = 0;
 i < n;
 i++) total += Math.abs((Number(a?.[i]) || 0) - (Number(b?.[i]) || 0));

      return total;

    
}


    l2Distance(a, b) {

      const n = Math.max(a?.length || 0, b?.length || 0);

      let total = 0;

      for (let i = 0;
 i < n;
 i++) {

        const d = (Number(a?.[i]) || 0) - (Number(b?.[i]) || 0);

        total += d * d;

      
}

      return Math.sqrt(total);

    
}


    topK(items, k = 8, score = x => Number(x?.score ?? x) || 0) {

      const limit = Math.max(0, k | 0);

      if (!limit) return [];

      const heap = [];

      for (const item of items || []) {

        const value = score(item);

        if (!Number.isFinite(value)) continue;

        heap.push({
 item, value 
}
);

      
}

      heap.sort((a,b) => b.value - a.value);

      return heap.slice(0, limit).map(entry => entry.item);

    
}


    argmax(values) {

      let index = -1;

      let best = -Infinity;

      for (let i = 0;
 i < (values?.length || 0);
 i++) {

        const value = Number(values[i]);

        if (Number.isFinite(value) && value > best) {

          best = value;

          index = i;

        
}

      
}

      return {
 index, value: best 
}
;

    
}


    quantile(values, q = 0.5) {

      const input = Array.from(values || [], Number).filter(Number.isFinite).sort((a,b) => a-b);

      if (!input.length) return 0;

      const p = Math.max(0, Math.min(1, Number(q) || 0));

      const pos = (input.length - 1) * p;

      const lo = Math.floor(pos), hi = Math.ceil(pos);

      if (lo === hi) return input[lo];

      const t = pos - lo;

      return input[lo] * (1 - t) + input[hi] * t;

    
}


    median(values) {

      return this.quantile(values, 0.5);

    
}


    mad(values) {

      const input = Array.from(values || [], Number).filter(Number.isFinite);

      if (!input.length) return 0;

      const m = this.median(input);

      return this.median(input.map(value => Math.abs(value - m)));

    
}


    ema(previous, current, alpha = 0.1) {

      const a = Math.max(0, Math.min(1, Number(alpha) || 0));

      const p = Number(previous);

      const c = Number(current);

      if (!Number.isFinite(c)) return Number.isFinite(p) ? p : 0;

      if (!Number.isFinite(p)) return c;

      return p + (c - p) * a;

    
}


    rollingMean(values, window = 8) {

      const input = Array.from(values || [], Number);

      const w = Math.max(1, window | 0);

      const out = new Array(input.length).fill(0);

      let sum = 0;

      for (let i = 0;
 i < input.length;
 i++) {

        sum += Number.isFinite(input[i]) ? input[i] : 0;

        if (i >= w) sum -= Number.isFinite(input[i-w]) ? input[i-w] : 0;

        out[i] = sum / Math.min(w, i + 1);

      
}

      return out;

    
}


    rollingVariance(values, window = 8) {

      const input = Array.from(values || [], Number);

      const w = Math.max(2, window | 0);

      const out = new Array(input.length).fill(0);

      for (let i = 0;
 i < input.length;
 i++) {

        const start = Math.max(0, i - w + 1);

        out[i] = this.variance(input.slice(start, i + 1));

      
}

      return out;

    
}


    zScores(values) {

      const input = Array.from(values || [], Number);

      const m = this.mean(input);

      const sd = this.stddev(input) || 1;

      return input.map(value => Number.isFinite(value) ? (value - m) / sd : 0);

    
}


    pearson(a, b) {

      const n = Math.min(a?.length || 0, b?.length || 0);

      if (n < 2) return 0;

      let sx=0, sy=0, sxx=0, syy=0, sxy=0, count=0;

      for (let i=0;
i<n;
i++) {

        const x=Number(a[i]), y=Number(b[i]);

        if (!Number.isFinite(x)||!Number.isFinite(y)) continue;

        sx+=x;
 sy+=y;
 sxx+=x*x;
 syy+=y*y;
 sxy+=x*y;
 count++;

      
}

      if (count<2) return 0;

      const num=count*sxy-sx*sy;

      const den=Math.sqrt(Math.max(0,(count*sxx-sx*sx)*(count*syy-sy*sy)));

      return den ? num/den : 0;

    
}


    covariance(a, b) {

      const n = Math.min(a?.length || 0, b?.length || 0);

      if (n < 2) return 0;

      const ax = Array.from(a).slice(0,n).map(Number);

      const bx = Array.from(b).slice(0,n).map(Number);

      const ma = this.mean(ax), mb = this.mean(bx);

      let total = 0, count = 0;

      for (let i=0;
i<n;
i++) {

        if (!Number.isFinite(ax[i]) || !Number.isFinite(bx[i])) continue;

        total += (ax[i]-ma)*(bx[i]-mb);

        count++;

      
}

      return count ? total/count : 0;

    
}


    weightedMean(values, weights) {

      const n = Math.min(values?.length || 0, weights?.length || 0);

      let sum = 0, weight = 0;

      for (let i=0;
i<n;
i++) {

        const x=Number(values[i]), w=Number(weights[i]);

        if (!Number.isFinite(x)||!Number.isFinite(w)||w<=0) continue;

        sum += x*w;

        weight += w;

      
}

      return weight ? sum/weight : 0;

    
}


    weightedVariance(values, weights) {

      const n = Math.min(values?.length || 0, weights?.length || 0);

      const mean = this.weightedMean(values, weights);

      let sum = 0, weight = 0;

      for (let i=0;
i<n;
i++) {

        const x=Number(values[i]), w=Number(weights[i]);

        if (!Number.isFinite(x)||!Number.isFinite(w)||w<=0) continue;

        const d=x-mean;

        sum += w*d*d;

        weight += w;

      
}

      return weight ? sum/weight : 0;

    
}


    sigmoid(x) {

      const n = Math.max(-60, Math.min(60, Number(x) || 0));

      return 1 / (1 + Math.exp(-n));

    
}


    tanh(x) {

      const n = Math.max(-30, Math.min(30, Number(x) || 0));

      const e2 = Math.exp(2*n);

      return (e2 - 1) / (e2 + 1);

    
}


    gelu(x) {

      const n=Number(x)||0;

      return 0.5*n*(1+Math.tanh(Math.sqrt(2/Math.PI)*(n+0.044715*n*n*n)));

    
}


    swish(x, beta = 1) {

      const n=Number(x)||0;

      return n * this.sigmoid((Number(beta)||1)*n);

    
}


    stableHash(value, seed = 2166136261) {

      const text = String(value ?? '');

      let h = seed >>> 0;

      for (let i=0;
i<text.length;
i++) {

        h ^= text.charCodeAt(i);

        h = Math.imul(h, 16777619) >>> 0;

      
}

      return h >>> 0;

    
}


    hashVector(tokens, dim = 64) {

      const size=Math.max(4,dim|0);

      const out=new Float32Array(size);

      for (const token of tokens||[]) {

        const text=String(token??'');

        const h=this.stableHash(text);

        const i=h%size;

        const sign=(h&1)?1:-1;

        out[i]+=sign;

      
}

      return out;

    
}


    ngrams(tokens, minN = 1, maxN = 3) {

      const input=Array.from(tokens||[], x=>String(x));

      const out=[];

      const lo=Math.max(1,minN|0), hi=Math.max(lo,maxN|0);

      for (let n=lo;
n<=hi;
n++) {

        for (let i=0;
i+n<=input.length;
i++) out.push(input.slice(i,i+n));

      
}

      return out;

    
}


    ngramCounts(tokens, minN = 1, maxN = 3) {

      const counts=new Map();

      for (const gram of this.ngrams(tokens,minN,maxN)) {

        const key=gram.join('');

        counts.set(key,(counts.get(key)||0)+1);

      
}

      return counts;

    
}


    transitionCounts(tokens) {

      const input=Array.from(tokens||[],x=>String(x));

      const map=new Map();

      for (let i=1;
i<input.length;
i++) {

        const key=input[i-1]+''+input[i];

        map.set(key,(map.get(key)||0)+1);

      
}

      return map;

    
}


    mutualInformation(jointCount, leftCount, rightCount, total) {

      const j=Number(jointCount)||0, l=Number(leftCount)||0, r=Number(rightCount)||0, t=Number(total)||0;

      if (j<=0||l<=0||r<=0||t<=0) return 0;

      return Math.log2((j*t)/(l*r));

    
}


    decayMap(map, factor = 0.995, floor = 1e-6) {

      if (!(map instanceof Map)) return map;

      const f=Math.max(0,Math.min(1,Number(factor)||0));

      for (const [key,value] of map) {

        const next=(Number(value)||0)*f;

        if (Math.abs(next)<floor) map.delete(key);
 else map.set(key,next);

      
}

      return map;

    
}


    pruneMap(map, maxSize = 4096) {

      if (!(map instanceof Map) || map.size<=maxSize) return map;

      const entries=Array.from(map.entries()).sort((a,b)=>(Number(b[1])||0)-(Number(a[1])||0));

      map.clear();

      for (const [key,value] of entries.slice(0,Math.max(0,maxSize|0))) map.set(key,value);

      return map;

    
}


    sparseAdd(target, indices, values, gain = 1) {

      const g=Number(gain)||1;

      const n=Math.min(indices?.length||0, values?.length||0);

      for (let i=0;
i<n;
i++) {

        const idx=indices[i]|0;

        if (idx<0||idx>=target.length) continue;

        target[idx]+=(Number(values[i])||0)*g;

      
}

      return target;

    
}


    sparseDot(dense, indices, values) {

      const n=Math.min(indices?.length||0,values?.length||0);

      let total=0;

      for (let i=0;
i<n;
i++) {

        const idx=indices[i]|0;

        if (idx<0||idx>=dense.length) continue;

        total+=(Number(dense[idx])||0)*(Number(values[i])||0);

      
}

      return total;

    
}


    status() {

      return {

        module: 'Worker',
        role: 'production-research-extension',
        calls: this.calls,
        methodCount: Object.getOwnPropertyNames(Object.getPrototypeOf(this)).length - 1,
        ageMs: (performance?.now?.() ?? Date.now()) - this.createdAt
      
}
;

    
}

  
}


  globalThis.VilotWorkerProductionLab = VilotWorkerProductionLab;

}
)();

