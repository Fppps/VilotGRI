/**
 * VilotNI 1.5 - RSL v23 Type-Driven Query + Commitment / Memory RSI RLEngine
 *
 * Deep mode uses one neural forward, one-time latent reasoning, Event/Query frames, prompt-rooted provenance, a lightweight learned RL policy, a Human-Decodable Proposition grammar, pronoun/person/case agreement, an operator-driven semantic Answer Slot Contract, a grounded word-association graph, whole-vocab vocab.json association neighborhoods with corroborated +0.5 positive token evidence, feedback-learned canonical word associations, relation-aware semantic routing, anchor-normalized dual centroids, active continuation vetoes, a compact POS-balanced candidate pool, adaptive ambiguity branching, a tiny constrained beam, whole-answer verification, and adaptive generation length. RSL does not regenerate the answer in repeated passes.
 * It spends a tiny amount of CPU work selecting a better token from the SAME
 * neural forward pass, hard-blocks clause-local morphological-family repetition, then exits once a clean prompt-anchored clause
 * has been formed. Low mode keeps the cheaper stochastic sampler.
 *
 * Goal:
 *   Deep = one fast neural pass + one-time latent setup + constrained beam search + one final verifier.
 *   Low  = simple one-path sampling.
 */

class RLEngine {
  constructor(network) {
    this.network = network;
    this.trajectory = [];
    this.episodeCount = 0;
    this.cumulativeReward = 0;
    this.lastSelfReward = 0;
    this.mode = "rsl";
    this.isDeepLearning = true;
    this.turnCounter = 0;

    this.lexicalEngineInfo =
      this.inspectLexicalEngine();

    this.maxTokenCeiling = 360;
    this.normalizedEmbeddings = null;
    this.normalizedEmbeddingsRevision = -1;
    this.initNormalizedEmbeddings();

    // Lightweight RSL word-pattern memory. It learns token/POS bigrams and
    // trigrams from rewarded trajectories. This is generic sequence learning,
    // not prompt-specific or answer-template logic.
    this.tokenPatternMemory = new Map();
    this.posPatternMemory = new Map();

    // RSL v5 semantic association memory:
    // sparse canonical word<->word links learned from explicit feedback.
    // Token n-grams learn ordering; associations learn which concepts belong
    // together even when they are not adjacent.
    this.wordAssociationMemory = new Map();

    // Context-scoped RLHF memory. Global n-grams remain weak so feedback on
    // one answer style does not contaminate unrelated chatbot requests.
    this.scopedTokenPatternMemory = new Map();
    this.scopedPosPatternMemory = new Map();

    // Tiny learned preference model. It learns from thumbs-up/down over
    // quality features instead of treating every token in a liked response as
    // universally good.
    this.preferenceWeights = {
      instruction: 0.90,
      answer: 0.85,
      semantic: 0.80,
      coherence: 0.78,
      structure: 0.72,
      grounding: 0.88,
      critic: 1.05,
      repetition: 0.48,
      echo: 0.38
    };
    this.preferenceBias = -2.65;
    this.preferenceLearningRate = 0.055;
    this.preferenceExamples = 0;

    // RSL v23: one compact parameter block. These values describe generic
    // language/semantic structure only. They never contain subject/topic rules.
    this.reasoningConfig = {
      provenanceHardFloor: 0.18,
      provenanceSoftFloor: 0.34,
      // Topic/event relevance is not enough. A question is complete only
      // when the semantic slot it asks for has actually been filled.
      eventQueryResolveFloor: 0.70,
      eventQueryHighConfidenceFloor: 0.84,
      querySlotResolveFloor: 0.66,
      querySlotHighConfidenceFloor: 0.82,

      // A role-compatible word is not enough. The words must form the
      // requested relation: mechanism, cause, effect, location, quantity, etc.
      relationStructureResolveFloor: 0.67,
      relationStructureHighConfidenceFloor: 0.83,
      argumentCompletenessFloor: 0.68,
      mechanismTopologyFloor: 0.62,
      tenseContinuityFloor: 0.70,

      epistemicHighConfidenceFloor: 0.72,

      // V19 semantic-clause graph. RSL now routes the whole clause into a
      // generic semantic frame instead of forcing every prompt through one
      // event representation.
      semanticFrameResolveFloor: 0.64,
      semanticFrameHighConfidenceFloor: 0.84,
      semanticGraphResolveFloor: 0.66,
      semanticGraphHighConfidenceFloor: 0.84,
      semanticGraphCandidateWeight: 3.2,
      semanticGraphHardFloor: 0.16,
      semanticGraphSoftFloor: 0.36,

      // Small, mostly-independent confidence model used after verification.
      semanticCoreConfidenceWeights: {
        frame: 0.18,
        query: 0.24,
        relation: 0.24,
        grounding: 0.12,
        language: 0.10,
        epistemic: 0.08,
        neural: 0.04
      },

      semanticCoreConfidenceCaps: {
        frameVeryLow: 0.38,
        frameLow: 0.54,
        queryVeryLow: 0.36,
        queryLow: 0.52,
        relationVeryLow: 0.34,
        relationLow: 0.50,
        groundingLow: 0.58,
        languageLow: 0.62
      },

      // V20 final verifier. These channels are reconstructed from the finished
      // answer and prompt contract. Generator beam/RL/policy scores do not
      // participate in the final confidence estimate.
      independentVerifierWeights: {
        act: 0.24,
        proposition: 0.20,
        promptRelation: 0.20,
        grounding: 0.12,
        language: 0.10,
        epistemic: 0.07,
        contradiction: 0.04,
        neural: 0.03
      },

      independentVerifierCaps: {
        actMissing: 0.28,
        actWeak: 0.44,
        propositionMissing: 0.34,
        propositionWeak: 0.50,
        promptRelationMissing: 0.32,
        promptRelationWeak: 0.52,
        groundingWeak: 0.60,
        languageWeak: 0.62,
        contradictionWeak: 0.54
      },

      independentActResolveFloor: 0.64,
      independentPropositionResolveFloor: 0.60,
      independentPromptRelationResolveFloor: 0.60,
      independentLanguageResolveFloor: 0.56,
      independentGroundingResolveFloor: 0.44,
      independentHighConfidenceFloor: 0.84,

      // V20.1 adaptive answer recovery.
      // Confidence is a search trigger, never a score target. The independent
      // verifier stays unchanged; low confidence only grants more CPU search
      // over the SAME neural logits.
      answerAcceptanceConfidence: 0.70,
      professionalLanguageFloor: 0.64,
      refinementMaxRounds: 2,
      refinementMinGain: 0.020,
      refinementStrongGain: 0.045,
      refinementBeamGrowth: 1,
      refinementBranchGrowth: 1,
      refinementVerifyGrowth: 3,
      refinementMaxBeam: 5,
      refinementMaxBranch: 3,
      refinementMaxVerify: 10,
      refinementAmbiguityDrop: 0.18,
      refinementTokenSlack: 6,
      refinementGraphPressure: 2.6,
      refinementActPressure: 1.9,
      refinementRelationPressure: 2.4,
      refinementGroundingPressure: 1.2,
      refinementLanguagePressure: 0.9,

      // V20.2 candidate-pool recovery.
      // Round 0 remains small/fast. Only a failed independently-verified
      // answer widens the vocabulary pool, and it does so from the SAME logits.
      recoveryPoolFamilyBase: 14,
      recoveryPoolFamilyGrowth: 8,
      recoveryPoolFamilyMax: 30,

      recoveryPoolFirstHopBase: 7,
      recoveryPoolFirstHopGrowth: 3,
      recoveryPoolFirstHopMax: 13,

      recoveryPoolSecondHopBase: 3,
      recoveryPoolSecondHopGrowth: 1,
      recoveryPoolSecondHopMax: 5,

      recoveryPoolSecondHopSeedsBase: 12,
      recoveryPoolSecondHopSeedsGrowth: 8,
      recoveryPoolSecondHopSeedsMax: 28,

      // Each recovery round may admit slightly weaker DISCOVERY candidates,
      // but normal RSL/provenance/independent verification still decides if
      // they can be used or trusted.
      recoveryPoolNeighborThresholdDrop: 0.035,
      recoveryPoolPropositionThresholdDrop: 0.025,

      // Cheap whole-vocabulary reranking signals used only while rebuilding
      // a failed answer's candidate pool.
      recoveryPoolTypeWeight: 0.72,
      recoveryPoolRoleWeight: 0.92,
      recoveryPoolAnchorWeight: 0.38,

      // Hard latency guard. Required/seed/punctuation tokens are always kept;
      // ranked recovery additions are bounded by this value.
      recoveryPoolMaxSize: 420,

      // V20.3 adaptive neural retry + automatic mistake learning.
      // Fast/easy prompts remain one-forward. A second neural forward is only
      // permitted after independent verification rejects the first answer and
      // one cheap same-logit recovery has failed.
      maxNeuralForwards: 2,
      cpuRecoveryRoundsBeforeNeuralRetry: 1,
      neuralRetryEnabled: true,
      neuralRetryContextTokens: 12,
      neuralRetryFailedTokenPenalty: 2.8,
      neuralRetryFailedBigramPenalty: 4.2,
      neuralRetryTokenSlack: 8,

      // Automatic RSI is deliberately weaker than explicit human feedback.
      // It updates behavior/search policy only; it never promotes generated
      // factual statements into the knowledge graph.
      automaticRSIEnabled: true,
      automaticRSINegativeMax: 0.34,
      automaticRSIPositiveMax: 0.18,
      automaticRSIMinFailureReward: 0.08,
      automaticRSIMinContrastGain: 0.035,
      automaticRSIPairLearningRate: 0.032,
      automaticRSIReplayLearningRate: 0.012,
      automaticRSIReplayLimit: 384,
      automaticRSIReplayBatch: 10,

      // V21: user input is training data for both response construction and
      // next-user-input prediction. These signals never become factual truth.
      userInputLearningEnabled: true,
      userStateDim: 12,
      userStateLearningRate: 0.075,
      userStateWeightLimit: 2.25,
      userStateErrorEMA: 0.92,
      userTokenTransitionLimit: 3072,
      userPOSTransitionLimit: 256,
      userCrossTurnTransitionLimit: 2048,
      userTransitionDecay: 0.997,
      userTokenTransitionWeight: 0.42,
      userPOSTransitionWeight: 0.66,
      userCarryoverWeight: 0.95,
      userFutureContinuityWeight: 0.34,
      userPredictedTokenWeight: 0.12,
      userPredictionMaxBonus: 1.65,
      userCarryoverFloor: 0.38,
      userCarryoverAnchorLimit: 8,
      userNextTokenLimit: 8,

      // V21.1 Self-Why reasoning.
      // Generation asks WHY each candidate belongs before committing to it.
      // This critic never contributes directly to final confidence.
      selfWhyEnabled: true,
      selfWhyCandidateWeight: 2.35,
      selfWhyCandidateSoftFloor: 0.43,
      selfWhyCandidateHardFloor: 0.19,
      selfWhyMaxCandidateBonus: 1.45,

      selfWhyClauseStopFloor: 0.60,
      selfWhyRelationStopFloor: 0.52,
      selfWhyEvidenceStopFloor: 0.28,

      selfWhyMismatchGap: 0.22,
      selfWhyMismatchPenalty: 0.20,

      selfWhyWeights: {
        promptFit: 0.16,
        relationFit: 0.24,
        evidenceSupport: 0.17,
        dependencySupport: 0.10,
        grounding: 0.09,
        novelty: 0.08,
        answerProgress: 0.16
      },

      // V22 four-layer memory.
      memoryEnabled: true,
      memoryCandidateWeight: 1.15,
      memoryContextGroundingWeight: 0.34,
      memoryTrustedTruthWeight: 0.72,
      memoryTruthFloor: 0.42,
      memoryCandidateLimit: 72,
      memoryEpisodePromptLimit: 32,
      memoryEpisodeResponseLimit: 48,
      memoryProcedureStrongHint: 0.72,
      memoryProcedureStrongSuccess: 0.66,

      // V23 type-driven phrasing + answer commitment.
      phraseParserRequired: true,
      requestLikelihoodFloor: 0.68,
      answerCommitmentEnabled: true,
      answerCommitmentResolveFloor: 0.66,
      answerCommitmentCandidateWeight: 2.20,
      answerCommitmentStopFloor: 0.66,
      answerCommitmentPropositionFloor: 0.60,
      answerCommitmentPredicateFloor: 0.58,
      answerCommitmentTargetFloor: 0.42,
      answerCommitmentLanguageFloor: 0.58,

      // Confidence is fulfillment-heavy. Fluency or neural certainty cannot
      // compensate for an incomplete semantic relationship.
      confidenceWeights: {
        meaning: 0.30,
        structure: 0.19,
        fulfillment: 0.46,
        neural: 0.05
      },

      fulfillmentWeights: {
        eventQuery: 0.23,
        querySlot: 0.25,
        relationStructure: 0.25,
        epistemic: 0.07,
        answer: 0.06,
        relation: 0.04,
        contract: 0.03,
        proposition: 0.03,
        shape: 0.02,
        semanticRole: 0.01,
        instruction: 0.01
      },

      confidenceCaps: {
        slotVeryLow: 0.38,
        slotLow: 0.52,
        slotPartial: 0.66,
        slotNear: 0.79,

        relationVeryLow: 0.36,
        relationLow: 0.50,
        relationPartial: 0.64,
        relationNear: 0.78,

        eventLow: 0.50,
        eventPartial: 0.65,
        eventNear: 0.80
      },

      rlPolicyWeight: 0.95,
      rlLearningRate: 0.16,
      rlReplayLearningRate: 0.045,
      rlDiscount: 0.93,
      rlWeightLimit: 2.75,
      rlReplayLimit: 768,
      rlReplayBatch: 20,

      // V18.1 continuation control. Target length is a ceiling, never a quota.
      prefixCheckpointMinLength: 4,
      prefixCheckpointMinQuality: 0.48,
      prefixCheckpointMinDelta: 0.008,
      semanticStopMinLength: 5,
      semanticStopResolutionFloor: 0.70,
      semanticStopQualityFloor: 0.60,
      completionLatchFloor: 0.70,
      driftStepDrop: 0.026,
      driftBestDrop: 0.070,
      driftHardDrop: 0.115,
      driftPatience: 2,
      hardDriftPatience: 3,
      rollbackMargin: 0.045,
      tailFraction: 0.34,
      tailHighConfidenceFloor: 0.72,
      discardedTailReward: -0.18,

      // V18.3 lexical trust + anti-monopoly controls.
      // Vocab-neighbor edges may discover candidates, but never become
      // semantic/typed factual evidence simply because they were copied into
      // a relation table.
      discoveryRelationTrust: 0.0,
      unknownRelationTrust: 0.42,

      // Generic modifiers must contribute to the requested semantic slot.
      modifierSoftFloor: 0.46,
      modifierHardFloor: 0.18,
      modifierPenaltyWeight: 6.2,

      // Cross-turn token fatigue is intentionally short-lived. It discourages
      // any ordinary content token from becoming the model's favorite word.
      lexicalFatigueDecay: 0.68,
      lexicalFatigueStart: 0.92,
      lexicalFatiguePenalty: 2.15,
      lexicalFatigueModifierMultiplier: 1.45,
      lexicalFatigueLimit: 512,
      monopolyRewardMaxDamp: 0.48,
      monopolyRewardScale: 0.24
    };

    // Real RL policy: generalized state/action features, not token/topic IDs.
    // It starts neutral and learns from explicit human feedback.
    this.rlFeatureCount = 27;
    this.rlPolicyWeights = new Float32Array(this.rlFeatureCount);
    this.rlReplay = [];
    this.rlUpdates = 0;

    // Contrastive RSI replay. These records contain only generalized policy
    // feature vectors + improvement margins, never prompt/topic strings.
    this.rsiReplay = [];
    this.rsiUpdates = 0;
    this.automaticMistakesLearned = 0;

    // Calibration diagnostic: when Self-Why likes a clause that the independent
    // verifier rejects, RSI learns that the internal reasoning was overconfident.
    this.selfWhyDisagreementEMA = 0.0;
    this.selfWhyDisagreementCount = 0;

    this.lastDiscardedTrajectory = [];

    // Short-lived cross-turn lexical usage. This is token-generic and decays
    // every completed response; it is not a blacklist.
    this.recentLexicalUsage = new Map();
    this.lexicalUsageTurns = 0;

    // V19 gets a fresh namespace. We can import compatible preference/policy
    // weights from V18 once, but do not import old replay/association pollution.
    // V21 dual-predictive input learner.
    this.userTokenTransitions = new Map();
    this.userPOSTransitions = new Map();
    this.userCrossTurnTransitions = new Map();
    this.userStateDim = this.reasoningConfig.userStateDim;
    this.userStateWeights = new Float32Array(this.userStateDim * this.userStateDim);
    this.userStateBias = new Float32Array(this.userStateDim);
    for (let i = 0; i < this.userStateDim; i++) {
      this.userStateWeights[i * this.userStateDim + i] = 0.35;
    }
    this.pendingUserState = null;
    this.pendingUserPrediction = null;
    this.pendingNextUserTokens = [];
    this.previousUserAnchors = [];
    this.previousUserLastContent = null;
    this.userStateErrorEMA = 0.50;
    this.userNextTokenHitEMA = 0.0;
    this.userInputTurns = 0;
    this.predictivePersistenceKey = "vilotni15_user_predictive_v21";

    this.persistenceKey = "vilotni15_rsl_type_phrase_v23";
    this.legacyPersistenceKey = "vilotni15_rsl_memory_v22";

    this.maxPatternEntries = 4096;
    this.maxScopedPatternEntries = 6144;
    this.maxAssociationEntries = 8192;
    this.loadPatternMemory();
    this.loadUserPredictiveMemory();

    // Dedicated V22 memory modules. Missing modules disable memory gracefully.
    this.memoryStore = null;
    this.memoryRetrieval = null;
    this.memoryReady = Promise.resolve(false);
    this.initMemorySystem();

    // Conservative English-ish POS matrix. The matrix is a preference system,
    // not a language model replacement. Prompt anchoring + network logits still
    // choose the actual words.
    this.transitionMatrix = {
      "START->Det":        { legal: true, weight: 2.4 },
      "START->Adj":        { legal: true, weight: 3.2 },
      "START->Noun":       { legal: true, weight: 3.8 },
      "START->ProperNoun": { legal: true, weight: 4.0 },
      "START->Pronoun":    { legal: true, weight: 3.2 },
      "START->Num":        { legal: true, weight: 2.0 },

      "Det->Adj":          { legal: true, weight: 3.2 },
      "Det->Noun":         { legal: true, weight: 4.1 },
      "Det->ProperNoun":   { legal: true, weight: 3.6 },
      "Det->Num":          { legal: true, weight: 2.8 },

      "Adj->Adj":          { legal: true, weight: 1.4 },
      "Adj->Noun":         { legal: true, weight: 4.4 },
      "Adj->ProperNoun":   { legal: true, weight: 3.7 },
      "Adj->Prep":         { legal: true, weight: 1.8 },
      "Adj->Conj":         { legal: true, weight: 1.5 },

      // Noun compounds remain legal where grammar and semantics support them.
      "Noun->Noun":        { legal: true, weight: 2.8 },
      "Noun->ProperNoun":  { legal: true, weight: 2.0 },
      "Noun->Modal":       { legal: true, weight: 3.8 },
      "Noun->Aux":         { legal: true, weight: 3.5 },
      "Noun->Verb":        { legal: true, weight: 4.2 },
      "Noun->Adv":         { legal: true, weight: 1.8 },
      "Noun->Prep":        { legal: true, weight: 2.2 },
      "Noun->Conj":        { legal: true, weight: 1.6 },

      "ProperNoun->Noun":  { legal: true, weight: 2.8 },
      "ProperNoun->Modal": { legal: true, weight: 3.8 },
      "ProperNoun->Aux":   { legal: true, weight: 3.5 },
      "ProperNoun->Verb":  { legal: true, weight: 4.2 },
      "ProperNoun->Prep":  { legal: true, weight: 2.2 },
      "ProperNoun->Adv":   { legal: true, weight: 1.7 },
      "ProperNoun->Conj":  { legal: true, weight: 1.6 },

      "Pronoun->Modal":    { legal: true, weight: 4.0 },
      "Pronoun->Aux":      { legal: true, weight: 3.8 },
      "Pronoun->Verb":     { legal: true, weight: 4.0 },
      "Pronoun->Adv":      { legal: true, weight: 1.5 },
      "Pronoun->Prep":     { legal: true, weight: 1.6 },
      "Pronoun->Conj":     { legal: true, weight: 1.4 },

      "Modal->Verb":       { legal: true, weight: 4.8 },
      "Modal->Aux":        { legal: true, weight: 2.8 },
      "Modal->Adv":        { legal: true, weight: 1.8 },
      "Aux->Verb":         { legal: true, weight: 3.4 },
      "Aux->Adj":          { legal: true, weight: 3.4 },
      "Aux->Noun":         { legal: true, weight: 2.8 },
      "Aux->ProperNoun":   { legal: true, weight: 2.8 },
      "Aux->Pronoun":      { legal: true, weight: 2.4 },
      "Aux->Det":          { legal: true, weight: 3.5 },
      "Aux->Prep":         { legal: true, weight: 3.0 },
      "Aux->Num":          { legal: true, weight: 2.2 },
      "Aux->Adv":          { legal: true, weight: 1.8 },
      "Aux->Conj":         { legal: true, weight: 1.2 },

      "Verb->Det":         { legal: true, weight: 3.0 },
      "Verb->Adj":         { legal: true, weight: 2.6 },
      "Verb->Noun":        { legal: true, weight: 4.0 },
      "Verb->ProperNoun":  { legal: true, weight: 3.6 },
      "Verb->Pronoun":     { legal: true, weight: 2.5 },
      "Verb->Adv":         { legal: true, weight: 2.2 },
      "Verb->Prep":        { legal: true, weight: 2.8 },
      "Verb->Conj":        { legal: true, weight: 1.2 },
      "Verb->Num":         { legal: true, weight: 2.2 },

      "Adv->Verb":         { legal: true, weight: 2.6 },
      "Adv->Adj":          { legal: true, weight: 2.0 },
      "Adv->Adv":          { legal: true, weight: 0.8 },
      "Adv->Prep":         { legal: true, weight: 1.3 },
      "Adv->Modal":        { legal: true, weight: 1.3 },
      "Adv->Aux":          { legal: true, weight: 1.3 },
      "Adv->Noun":         { legal: true, weight: 1.2 },
      "Adv->ProperNoun":   { legal: true, weight: 1.2 },
      "Adv->Conj":         { legal: true, weight: 1.0 },

      "Prep->Det":         { legal: true, weight: 3.6 },
      "Prep->Adj":         { legal: true, weight: 2.7 },
      "Prep->Noun":        { legal: true, weight: 4.1 },
      "Prep->ProperNoun":  { legal: true, weight: 4.1 },
      "Prep->Pronoun":     { legal: true, weight: 3.0 },
      "Prep->Num":         { legal: true, weight: 3.1 },

      "Conj->Det":         { legal: true, weight: 2.8 },
      "Conj->Adj":         { legal: true, weight: 2.0 },
      "Conj->Noun":        { legal: true, weight: 3.2 },
      "Conj->ProperNoun":  { legal: true, weight: 3.2 },
      "Conj->Pronoun":     { legal: true, weight: 3.0 },
      "Conj->Adv":         { legal: true, weight: 1.6 },
      "Conj->Verb":        { legal: true, weight: 1.3 },

      "Num->Noun":         { legal: true, weight: 3.7 },
      "Num->ProperNoun":   { legal: true, weight: 2.6 },
      "Num->Prep":         { legal: true, weight: 2.0 },
      "Num->Conj":         { legal: true, weight: 1.3 }
    };
  }

  inspectLexicalEngine() {
    const vocab =
      this.network?.vocab;

    const capabilities =
      typeof vocab?.getCapabilities === "function"
        ? vocab.getCapabilities()
        : {};

    const apiVersion =
      Number(
        capabilities?.apiVersion ??
        vocab?.apiVersion ??
        0
      ) || 0;

    return {
      apiVersion,
      engineVersion:
        capabilities?.engineVersion ??
        vocab?.engineVersion ??
        "legacy",
      schemaVersion:
        capabilities?.schemaVersion ??
        vocab?.schemaVersion ??
        0,
      knowledgeLinks:
        typeof vocab?.getKnowledgeLink === "function",
      knowledgeNeighbors:
        typeof vocab?.getKnowledgeNeighbors === "function",
      typedRelations:
        typeof vocab?.getRelations === "function",
      semanticClasses:
        typeof vocab?.getSemanticClass === "function",
      collocations:
        typeof vocab?.getCollocations === "function",
      grammar:
        typeof vocab?.getGrammar === "function",
      semanticRoles:
        typeof vocab?.getRoleCompatibility === "function",
      propositionFit:
        typeof vocab?.getPropositionFit === "function",
      morphologyFamilies:
        typeof vocab?.sameMorphologicalFamily === "function" &&
        typeof vocab?.getMorphologyAffinity === "function",
      promptAnswerSeparation:
        typeof vocab?.getPromptReusePolicy === "function" &&
        typeof vocab?.getAnchorSupport === "function"
    };
  }

  getVocabKnowledgeLink(
    aToken,
    bToken
  ) {
    const vocab =
      this.network?.vocab;

    if (
      vocab &&
      typeof vocab.getKnowledgeLink === "function"
    ) {
      return vocab.getKnowledgeLink(
        aToken,
        bToken
      );
    }

    const details =
      this.getStaticKnowledgeDetails(
        aToken,
        bToken
      );

    const association =
      Math.max(
        0,
        Math.min(
          1,
          details.effectiveWeight || 0
        )
      );

    return {
      known:
        association > 0,
      score:
        association > 0
          ? 0.52 +
            association * 0.42
          : 0.55,
      positiveSupport:
        association * 0.70,
      association,
      typedRelation: 0.0,
      collocation: 0.0,
      lexicalFamily: 0.0,
      classFit: 0.55
    };
  }

  initNormalizedEmbeddings() {
    if (!this.network || !this.network.embeddings) return;
    const vocabSize = this.network.vocabSize;
    const embDim = this.network.embDim;
    this.normalizedEmbeddings = new Float32Array(vocabSize * embDim);

    for (let i = 0; i < vocabSize; i++) {
      const off = i * embDim;
      let normSq = 0;
      for (let d = 0; d < embDim; d++) {
        const x = this.network.embeddings[off + d];
        normSq += x * x;
      }
      const invNorm = normSq > 0 ? 1.0 / Math.sqrt(normSq) : 1.0;
      for (let d = 0; d < embDim; d++) {
        this.normalizedEmbeddings[off + d] = this.network.embeddings[off + d] * invNorm;
      }
    }

    this.normalizedEmbeddingsRevision =
      Number.isFinite(this.network.embeddingRevision)
        ? this.network.embeddingRevision
        : 0;
  }

  ensureNormalizedEmbeddingsCurrent() {
    const revision =
      Number.isFinite(this.network?.embeddingRevision)
        ? this.network.embeddingRevision
        : 0;

    if (
      !this.normalizedEmbeddings ||
      this.normalizedEmbeddingsRevision !== revision
    ) {
      this.initNormalizedEmbeddings();
    }
  }

  setMode(newMode) {
    if (newMode === "rsl" || newMode === "user" || newMode === "rl") {
      this.mode = newMode;
    }
  }

  shouldUpdateDeepSynapses() {
    this.turnCounter++;
    return this.isDeepLearning || (this.turnCounter % 5 === 0);
  }

  getCanonicalType(tokenId) {
    const rawType = this.network.vocab?.idToType?.[tokenId];
    if (!rawType) return "Other";
    if (rawType.includes("Modal")) return "Modal";
    if (rawType.includes("Auxiliary")) return "Aux";
    if (rawType.includes("Verb")) return "Verb";
    if (rawType.includes("Proper Noun")) return "ProperNoun";
    if (rawType.includes("Noun")) return "Noun";
    if (rawType.includes("Pronoun")) return "Pronoun";
    if (rawType.includes("Adjective")) return "Adj";
    if (rawType.includes("Adverb")) return "Adv";
    if (rawType.includes("Determiner")) return "Det";
    if (rawType.includes("Preposition")) return "Prep";
    if (rawType.includes("Conjunction")) return "Conj";
    if (rawType.includes("Punctuation")) return "Punct";
    if (rawType.includes("Number")) return "Num";
    return rawType;
  }

  stemWord(word) {
    return String(word || "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "")
      .replace(/(ing|edly|edly|ed|es|s)$/i, "")
      .slice(0, 8);
  }

  isContentType(type) {
    return ["Noun", "ProperNoun", "Verb", "Adj", "Adv", "Num"].includes(type);
  }


  canonicalTokenId(tokenId) {
    const vocab =
      this.network?.vocab;

    if (
      vocab &&
      typeof vocab.canonicalId === "function"
    ) {
      const canonical =
        vocab.canonicalId(tokenId);

      if (Number.isInteger(canonical)) {
        return canonical;
      }
    }

    return Number.isInteger(tokenId)
      ? tokenId
      : Number(tokenId);
  }

  associationKey(aToken, bToken) {
    const a =
      this.canonicalTokenId(aToken);
    const b =
      this.canonicalTokenId(bToken);

    if (
      !Number.isInteger(a) ||
      !Number.isInteger(b)
    ) {
      return "";
    }

    // Associations are semantic/co-occurrence links, so they are symmetric.
    // Directional ordering is still handled by tokenPatternMemory.
    return a <= b
      ? `${a}|${b}`
      : `${b}|${a}`;
  }

  getLearnedAssociation(aToken, bToken) {
    const key =
      this.associationKey(
        aToken,
        bToken
      );

    if (!key) return 0.0;

    return Math.max(
      -2.5,
      Math.min(
        4.0,
        this.wordAssociationMemory.get(key) || 0.0
      )
    );
  }

  getCachedPairSimilarity(
    aToken,
    bToken,
    contextModel = null
  ) {
    const a =
      this.canonicalTokenId(aToken);
    const b =
      this.canonicalTokenId(bToken);

    if (
      !Number.isInteger(a) ||
      !Number.isInteger(b)
    ) {
      return 0.0;
    }

    if (a === b) return 1.0;

    let cache =
      contextModel?.wordAssociationCache;

    if (
      contextModel &&
      !(cache instanceof Map)
    ) {
      cache = new Map();
      contextModel.wordAssociationCache =
        cache;
    }

    const key =
      a <= b
        ? `${a}|${b}`
        : `${b}|${a}`;

    if (
      cache &&
      cache.has(key)
    ) {
      return cache.get(key);
    }

    const similarity =
      this.dotTokenPair(a, b);

    if (cache) {
      // Tiny prompt-local cache: the beam repeatedly tests the same candidate
      // against the same subject/action anchors.
      if (cache.size > 4096) {
        cache.clear();
      }
      cache.set(key, similarity);
    }

    return similarity;
  }

  getStaticKnowledgeDetails(
    aToken,
    bToken
  ) {
    const vocab =
      this.network?.vocab;

    if (
      vocab &&
      typeof vocab.getAssociationDetails === "function"
    ) {
      return vocab.getAssociationDetails(
        aToken,
        bToken
      );
    }

    const weight =
      vocab &&
      typeof vocab.getAssociationWeight === "function"
        ? vocab.getAssociationWeight(
            aToken,
            bToken
          )
        : 0.0;

    return {
      rawWeight: weight,
      effectiveWeight: weight,
      trust: weight > 0 ? 0.70 : 0.0,
      source:
        weight > 0
          ? "legacy"
          : "none"
    };
  }

  getStaticKnowledgeAssociation(
    aToken,
    bToken
  ) {
    const vocab =
      this.network?.vocab;

    if (
      !vocab ||
      typeof vocab.getAssociationWeight !== "function"
    ) {
      return 0.0;
    }

    return Math.max(
      0,
      Math.min(
        1,
        vocab.getAssociationWeight(
          aToken,
          bToken
        ) || 0.0
      )
    );
  }

  isDiscoveryOnlyLexicalSource(source) {
    const normalized =
      String(source || "")
        .trim()
        .toLowerCase()
        .replace(/[_\s]+/g, "-");

    return (
      normalized === "vocab-neighbor" ||
      normalized === "neighbor" ||
      normalized === "vocabulary-neighbor" ||
      normalized === "id-neighbor"
    );
  }

  getTrustedTypedRelationEvidence(
    lexicalLink
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const details =
      lexicalLink?.relationDetails;

    const relations = [
      ...(details?.forward || []),
      ...(details?.reverse || [])
    ];

    if (!relations.length) {
      // Legacy vocab APIs may expose only a scalar typed relation. Treat that
      // as weak/unknown rather than fully trusted.
      return (
        clamp(
          lexicalLink?.typedRelation
        ) *
        this.reasoningConfig
          .unknownRelationTrust
      );
    }

    let trusted = 0.0;

    for (const item of relations) {
      const source =
        item?.source || "unknown";

      if (
        this.isDiscoveryOnlyLexicalSource(
          source
        )
      ) {
        continue;
      }

      const sourceTrust =
        String(source)
          .toLowerCase() === "unknown"
          ? this.reasoningConfig
              .unknownRelationTrust
          : 1.0;

      trusted =
        Math.max(
          trusted,
          clamp(item?.weight) *
          sourceTrust
        );
    }

    return clamp(trusted);
  }

  getTrustedPropositionEvidence(
    subjectToken,
    predicateToken,
    candidateToken,
    role = "content"
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const vocab =
      this.network?.vocab;

    const roleFitRaw =
      vocab?.getRoleCompatibility?.(
        candidateToken,
        role
      );

    const roleFit =
      Number.isFinite(roleFitRaw)
        ? clamp(roleFitRaw)
        : (
            this.isContentType(
              this.getCanonicalType(
                candidateToken
              )
            )
              ? 0.50
              : 0.20
          );

    const predicateEvidence =
      Number.isInteger(predicateToken)
        ? this.getWordAssociationEvidence(
            candidateToken,
            predicateToken,
            null
          )
        : null;

    const subjectEvidence =
      Number.isInteger(subjectToken)
        ? this.getWordAssociationEvidence(
            candidateToken,
            subjectToken,
            null
          )
        : null;

    const predicateTrusted =
      Math.max(
        predicateEvidence
          ?.typedKnowledge || 0,
        predicateEvidence
          ?.staticKnowledge || 0,
        (
          predicateEvidence
            ?.collocationKnowledge || 0
        ) * 0.72
      );

    const subjectTrusted =
      Math.max(
        subjectEvidence
          ?.typedKnowledge || 0,
        subjectEvidence
          ?.staticKnowledge || 0,
        (
          subjectEvidence
            ?.collocationKnowledge || 0
        ) * 0.58
      );

    // Semantic-role compatibility is useful, but it cannot by itself turn a
    // discovery-only relation into proposition knowledge.
    const score =
      clamp(
        roleFit * 0.62 +
        predicateTrusted * 0.28 +
        subjectTrusted * 0.10
      );

    return {
      score,
      roleFit,
      predicateTrusted,
      subjectTrusted,
      typedRelation:
        Math.max(
          predicateEvidence
            ?.typedKnowledge || 0,
          subjectEvidence
            ?.typedKnowledge || 0
        ),
      collocation:
        Math.max(
          predicateEvidence
            ?.collocationKnowledge || 0,
          subjectEvidence
            ?.collocationKnowledge || 0
        ),
      classFit:
        Math.max(
          predicateEvidence
            ?.classFit || 0.55,
          subjectEvidence
            ?.classFit || 0.55
        )
    };
  }

  getWordAssociationEvidence(
    aToken,
    bToken,
    contextModel = null
  ) {
    const clamp = value =>
      Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

    const similarity =
      this.getCachedPairSimilarity(
        aToken,
        bToken,
        contextModel
      );

    // Embeddings are relevance evidence only. They never prove a fact.
    const embeddingEvidence =
      clamp((similarity + 0.08) / 0.70);

    const learnedRaw =
      this.getLearnedAssociation(aToken, bToken);

    // Feedback associations are policy/relevance memory, not knowledge.
    const learnedEvidence =
      clamp(0.50 + learnedRaw / 5.0);

    const staticDetails =
      this.getStaticKnowledgeDetails(aToken, bToken);

    const lexicalLink =
      this.getVocabKnowledgeLink(aToken, bToken);

    const source =
      String(staticDetails?.source || "none").toLowerCase();

    // Automatic vocabulary adjacency is discovery-only. It may help a word
    // enter the candidate pool, but cannot raise factual/semantic confidence.
    const discoveryOnly =
      this.isDiscoveryOnlyLexicalSource(
        source
      );

    const typedKnowledge =
      this.getTrustedTypedRelationEvidence(
        lexicalLink
      );

    const collocationKnowledge =
      clamp(lexicalLink?.collocation || 0);

    const classFit =
      clamp(
        Number.isFinite(lexicalLink?.classFit)
          ? lexicalLink.classFit
          : 0.55
      );

    const staticWeight =
      clamp(staticDetails?.effectiveWeight || 0);

    const staticTrust =
      clamp(staticDetails?.trust || 0);

    const trustedStatic =
      discoveryOnly
        ? 0.0
        : staticWeight * Math.max(0.35, staticTrust);

    const hasTrustedExplicit =
      typedKnowledge >= 0.04 ||
      collocationKnowledge >= 0.06 ||
      trustedStatic >= 0.04;

    const explicitLink =
      !discoveryOnly &&
      lexicalLink?.known &&
      hasTrustedExplicit
        ? clamp(
            Math.max(
              0,
              (lexicalLink.score || 0.50) -
              0.50
            ) * 2.0
          )
        : 0.0;

    const explicitKnowledge =
      clamp(
        Math.max(
          typedKnowledge,
          collocationKnowledge * 0.72,
          trustedStatic,
          explicitLink
        )
      );

    const relevanceEvidence =
      clamp(
        embeddingEvidence * 0.78 +
        learnedEvidence * 0.22
      );

    // Association evidence remains useful for continuation ranking, but only
    // explicit/typed knowledge can make it look factual.
    const evidence =
      clamp(
        relevanceEvidence * 0.76 +
        explicitKnowledge * 0.24
      );

    return {
      evidence,
      similarity,
      embeddingEvidence,
      learnedEvidence,
      learnedRaw,
      relevanceEvidence,
      staticKnowledge: trustedStatic,
      corroboratedKnowledge: explicitKnowledge,
      knowledgeSupport:
        explicitKnowledge > 0
          ? 0.50 + explicitKnowledge * 0.50
          : 0.50,
      knowledgeKnown:
        explicitKnowledge >= 0.18,
      typedKnowledge,
      collocationKnowledge,
      classFit,
      lexicalFamily:
        clamp(lexicalLink?.lexicalFamily || 0),
      staticSource: source,
      staticTrust,
      discoveryOnly
    };
  }

  getGroundedWordAssociation(
    candidateToken,
    generated,
    contextModel,
    sentenceState,
    patternContext = null
  ) {
    const type =
      this.getCanonicalType(
        candidateToken
      );

    if (!this.isContentType(type)) {
      return {
        evidence: 0.62,
        local: 0.62,
        anchor: 0.62,
        triangle: 0.62,
        learned: 0.50,
        staticKnowledge: 0.0,
        knowledgeSupport: 0.55,
        typedKnowledge: 0.0,
        collocationKnowledge: 0.0,
        classFit: 0.55,
        knowledgeKnown: false,
        hasLocal: false,
        hasAnchor: false
      };
    }

    const recentContent =
      patternContext?.recentContentTokens ||
      [];

    let local = 0.0;
    let localLearned = 0.50;
    let localStatic = 0.0;
    let localKnowledge = 0.55;
    let localTyped = 0.0;
    let localCollocation = 0.0;
    let localClassFit = 0.55;
    let localKnown = false;
    let hasLocal = false;

    // Last three content words form the local semantic thread.
    const localStart =
      Math.max(
        0,
        recentContent.length - 3
      );

    for (
      let i = localStart;
      i < recentContent.length;
      i++
    ) {
      const pair =
        this.getWordAssociationEvidence(
          candidateToken,
          recentContent[i],
          contextModel
        );

      const distance =
        recentContent.length - i;

      const recencyWeight =
        distance === 1
          ? 1.0
          : distance === 2
            ? 0.88
            : 0.72;

      local = Math.max(
        local,
        pair.evidence *
        recencyWeight
      );

      localLearned = Math.max(
        localLearned,
        pair.learnedEvidence
      );

      localStatic = Math.max(
        localStatic,
        pair.corroboratedKnowledge ||
        pair.staticKnowledge ||
        0
      );

      localKnowledge = Math.max(
        localKnowledge,
        pair.knowledgeSupport || 0.55
      );

      localTyped = Math.max(
        localTyped,
        pair.typedKnowledge || 0
      );

      localCollocation = Math.max(
        localCollocation,
        pair.collocationKnowledge || 0
      );

      localClassFit = Math.max(
        localClassFit,
        pair.classFit || 0.55
      );

      localKnown =
        localKnown ||
        Boolean(pair.knowledgeKnown);

      hasLocal = true;
    }

    // Stable anchors cannot drift because they come from the original prompt
    // or the clause's identified subject/predicate.
    const anchors = [];

    const pushAnchor = token => {
      const canonical =
        this.canonicalTokenId(token);

      if (
        Number.isInteger(canonical) &&
        !anchors.includes(canonical)
      ) {
        anchors.push(canonical);
      }
    };

    pushAnchor(
      sentenceState?.subjectToken
    );

    pushAnchor(
      sentenceState?.predicateToken
    );

    pushAnchor(
      contextModel?.answerContract
        ?.relationVerbToken
    );

    for (
      const token of
        contextModel?.answerContract
          ?.subjectTokens || []
    ) {
      pushAnchor(token);
      if (anchors.length >= 8) break;
    }

    for (
      const token of
        contextModel?.promptContentTokens ||
        []
    ) {
      pushAnchor(token);
      if (anchors.length >= 10) break;
    }

    let anchor = 0.0;
    let anchorLearned = 0.50;
    let anchorStatic = 0.0;
    let anchorKnowledge = 0.55;
    let anchorTyped = 0.0;
    let anchorCollocation = 0.0;
    let anchorClassFit = 0.55;
    let anchorKnown = false;
    let hasAnchor = false;

    for (const token of anchors) {
      if (
        token ===
        this.canonicalTokenId(candidateToken)
      ) {
        anchor = 1.0;
        hasAnchor = true;
        break;
      }

      const pair =
        this.getWordAssociationEvidence(
          candidateToken,
          token,
          contextModel
        );

      anchor = Math.max(
        anchor,
        pair.evidence
      );

      anchorLearned = Math.max(
        anchorLearned,
        pair.learnedEvidence
      );

      anchorStatic = Math.max(
        anchorStatic,
        pair.corroboratedKnowledge ||
        pair.staticKnowledge ||
        0
      );

      anchorKnowledge = Math.max(
        anchorKnowledge,
        pair.knowledgeSupport || 0.55
      );

      anchorTyped = Math.max(
        anchorTyped,
        pair.typedKnowledge || 0
      );

      anchorCollocation = Math.max(
        anchorCollocation,
        pair.collocationKnowledge || 0
      );

      anchorClassFit = Math.max(
        anchorClassFit,
        pair.classFit || 0.55
      );

      anchorKnown =
        anchorKnown ||
        Boolean(pair.knowledgeKnown);

      hasAnchor = true;
    }

    if (!hasLocal) {
      local =
        generated?.length
          ? 0.46
          : 0.62;
    }

    if (!hasAnchor) {
      anchor = 0.55;
    }

    // Dual grounding is the anti-drift signal:
    // "locally associated" is not enough if the chain has wandered away from
    // the original question. "prompt-related" alone is not enough if the word
    // does not continue the sentence's current thought.
    const triangle =
      Math.sqrt(
        Math.max(
          0,
          local * anchor
        )
      );

    const learned =
      Math.max(
        localLearned,
        anchorLearned
      );

    const staticKnowledge =
      Math.max(
        localStatic,
        anchorStatic
      );

    const knowledgeSupport =
      Math.max(
        localKnowledge,
        anchorKnowledge
      );

    const typedKnowledge =
      Math.max(
        localTyped,
        anchorTyped
      );

    const collocationKnowledge =
      Math.max(
        localCollocation,
        anchorCollocation
      );

    const classFit =
      Math.max(
        localClassFit,
        anchorClassFit
      );

    const knowledgeKnown =
      localKnown ||
      anchorKnown;

    const evidence =
      Math.max(
        0,
        Math.min(
          1,
          local * 0.32 +
          anchor * 0.42 +
          triangle * 0.20 +
          learned * 0.06
        )
      );

    return {
      evidence,
      local,
      anchor,
      triangle,
      learned,
      staticKnowledge,
      knowledgeSupport,
      typedKnowledge,
      collocationKnowledge,
      classFit,
      knowledgeKnown,
      hasLocal,
      hasAnchor
    };
  }

  learnWordAssociations(
    reward,
    promptTokens = null
  ) {
    if (
      !this.trajectory?.length ||
      !Number.isFinite(reward)
    ) {
      return;
    }

    const signed =
      Math.max(
        -1,
        Math.min(1, reward)
      );

    if (Math.abs(signed) < 0.05) {
      return;
    }

    const update =
      (a, b, delta) => {
        const key =
          this.associationKey(a, b);

        if (!key) return;

        const next =
          Math.max(
            -2.5,
            Math.min(
              4.0,
              (this.wordAssociationMemory.get(key) || 0) +
              delta
            )
          );

        this.wordAssociationMemory.set(
          key,
          next
        );
      };

    const responseContent = [];

    for (const step of this.trajectory) {
      const raw =
        step?.token;

      if (!Number.isFinite(raw)) {
        continue;
      }

      const token =
        this.canonicalTokenId(raw);

      const type =
        this.getCanonicalType(token);

      if (
        this.isContentType(type)
      ) {
        responseContent.push(token);
      }
    }

    const baseStep =
      signed * 0.18;

    // Learn local semantic neighborhoods, not only exact word order.
    for (
      let i = 0;
      i < responseContent.length;
      i++
    ) {
      for (
        let distance = 1;
        distance <= 3;
        distance++
      ) {
        const j =
          i + distance;

        if (
          j >= responseContent.length
        ) {
          break;
        }

        const weight =
          distance === 1
            ? 1.0
            : distance === 2
              ? 0.62
              : 0.38;

        update(
          responseContent[i],
          responseContent[j],
          baseStep * weight
        );
      }
    }

    // Feedback can also teach which answer words are grounded in the prompt.
    // On a Good response, strengthen each response word toward its two closest
    // prompt concepts. On a Bad response, weaken those same connections.
    const promptContent = [];

    for (const raw of promptTokens || []) {
      if (!Number.isFinite(raw)) continue;

      const token =
        this.canonicalTokenId(raw);

      if (
        this.isContentType(
          this.getCanonicalType(token)
        ) &&
        !promptContent.includes(token)
      ) {
        promptContent.push(token);
      }

      if (promptContent.length >= 12) {
        break;
      }
    }

    if (
      promptContent.length &&
      responseContent.length
    ) {
      for (const responseToken of responseContent) {
        let bestA = null;
        let bestB = null;

        for (const promptToken of promptContent) {
          const sim =
            this.getCachedPairSimilarity(
              responseToken,
              promptToken,
              null
            );

          const entry = {
            token: promptToken,
            sim
          };

          if (
            !bestA ||
            sim > bestA.sim
          ) {
            bestB = bestA;
            bestA = entry;
          } else if (
            !bestB ||
            sim > bestB.sim
          ) {
            bestB = entry;
          }
        }

        if (bestA) {
          update(
            responseToken,
            bestA.token,
            baseStep * 0.92
          );
        }

        if (bestB) {
          update(
            responseToken,
            bestB.token,
            baseStep * 0.52
          );
        }
      }
    }

    this.trimPatternMap(
      this.wordAssociationMemory,
      this.maxAssociationEntries
    );
  }


  getPronounProfile(word) {
    const w =
      String(word || "")
        .toLowerCase();

    const profiles = {
      i: {
        person: 1,
        number: "singular",
        subject: true,
        object: false,
        possessive: false,
        reflexive: false
      },
      me: {
        person: 1,
        number: "singular",
        subject: false,
        object: true,
        possessive: false,
        reflexive: false
      },
      my: {
        person: 1,
        number: "singular",
        subject: false,
        object: false,
        possessive: true,
        reflexive: false
      },
      mine: {
        person: 1,
        number: "singular",
        subject: false,
        object: true,
        possessive: true,
        reflexive: false
      },
      myself: {
        person: 1,
        number: "singular",
        subject: false,
        object: true,
        possessive: false,
        reflexive: true
      },

      you: {
        person: 2,
        number: "either",
        subject: true,
        object: true,
        possessive: false,
        reflexive: false
      },
      your: {
        person: 2,
        number: "either",
        subject: false,
        object: false,
        possessive: true,
        reflexive: false
      },
      yours: {
        person: 2,
        number: "either",
        subject: false,
        object: true,
        possessive: true,
        reflexive: false
      },
      yourself: {
        person: 2,
        number: "singular",
        subject: false,
        object: true,
        possessive: false,
        reflexive: true
      },
      yourselves: {
        person: 2,
        number: "plural",
        subject: false,
        object: true,
        possessive: false,
        reflexive: true
      },

      he: {
        person: 3,
        number: "singular",
        subject: true,
        object: false,
        possessive: false,
        reflexive: false
      },
      him: {
        person: 3,
        number: "singular",
        subject: false,
        object: true,
        possessive: false,
        reflexive: false
      },
      his: {
        person: 3,
        number: "singular",
        subject: false,
        object: true,
        possessive: true,
        reflexive: false
      },
      himself: {
        person: 3,
        number: "singular",
        subject: false,
        object: true,
        possessive: false,
        reflexive: true
      },

      she: {
        person: 3,
        number: "singular",
        subject: true,
        object: false,
        possessive: false,
        reflexive: false
      },
      her: {
        person: 3,
        number: "singular",
        subject: false,
        object: true,
        possessive: true,
        reflexive: false
      },
      hers: {
        person: 3,
        number: "singular",
        subject: false,
        object: true,
        possessive: true,
        reflexive: false
      },
      herself: {
        person: 3,
        number: "singular",
        subject: false,
        object: true,
        possessive: false,
        reflexive: true
      },

      it: {
        person: 3,
        number: "singular",
        subject: true,
        object: true,
        possessive: false,
        reflexive: false
      },
      its: {
        person: 3,
        number: "singular",
        subject: false,
        object: false,
        possessive: true,
        reflexive: false
      },
      itself: {
        person: 3,
        number: "singular",
        subject: false,
        object: true,
        possessive: false,
        reflexive: true
      },

      we: {
        person: 1,
        number: "plural",
        subject: true,
        object: false,
        possessive: false,
        reflexive: false
      },
      us: {
        person: 1,
        number: "plural",
        subject: false,
        object: true,
        possessive: false,
        reflexive: false
      },
      our: {
        person: 1,
        number: "plural",
        subject: false,
        object: false,
        possessive: true,
        reflexive: false
      },
      ours: {
        person: 1,
        number: "plural",
        subject: false,
        object: true,
        possessive: true,
        reflexive: false
      },
      ourselves: {
        person: 1,
        number: "plural",
        subject: false,
        object: true,
        possessive: false,
        reflexive: true
      },

      they: {
        person: 3,
        number: "plural",
        subject: true,
        object: false,
        possessive: false,
        reflexive: false
      },
      them: {
        person: 3,
        number: "plural",
        subject: false,
        object: true,
        possessive: false,
        reflexive: false
      },
      their: {
        person: 3,
        number: "plural",
        subject: false,
        object: false,
        possessive: true,
        reflexive: false
      },
      theirs: {
        person: 3,
        number: "plural",
        subject: false,
        object: true,
        possessive: true,
        reflexive: false
      },
      themselves: {
        person: 3,
        number: "plural",
        subject: false,
        object: true,
        possessive: false,
        reflexive: true
      }
    };

    return profiles[w] || null;
  }

  pronounAuxAgreement(
    subjectWord,
    auxWord
  ) {
    const subject =
      String(subjectWord || "")
        .toLowerCase();

    const aux =
      String(auxWord || "")
        .toLowerCase();

    if (!subject || !aux) {
      return 0.70;
    }

    const modalWords =
      new Set([
        "can", "could", "will", "would",
        "shall", "should", "may", "might",
        "must"
      ]);

    if (modalWords.has(aux)) {
      return 1.0;
    }

    const allowed = {
      i: new Set([
        "am", "was", "have", "had",
        "do", "did", "be", "been"
      ]),
      you: new Set([
        "are", "were", "have", "had",
        "do", "did", "be", "been"
      ]),
      he: new Set([
        "is", "was", "has", "had",
        "does", "did", "be", "been"
      ]),
      she: new Set([
        "is", "was", "has", "had",
        "does", "did", "be", "been"
      ]),
      it: new Set([
        "is", "was", "has", "had",
        "does", "did", "be", "been"
      ]),
      we: new Set([
        "are", "were", "have", "had",
        "do", "did", "be", "been"
      ]),
      they: new Set([
        "are", "were", "have", "had",
        "do", "did", "be", "been"
      ])
    };

    const set =
      allowed[subject];

    if (!set) {
      return 0.70;
    }

    return set.has(aux)
      ? 1.0
      : 0.0;
  }

  loadPatternMemory() {
    if (typeof localStorage === "undefined") return;

    try {
      let raw =
        localStorage.getItem(
          this.persistenceKey
        );

      let legacyImport = false;

      if (!raw) {
        raw =
          localStorage.getItem(
            this.legacyPersistenceKey
          );

        legacyImport =
          Boolean(raw);
      }

      if (!raw) return;

      const parsed =
        JSON.parse(raw);

      // Old token/association memories were trained under the pre-graph
      // semantics. Keep them isolated during migration so they cannot define
      // V19's semantic graph.
      if (!legacyImport) {
        for (const [k, v] of parsed.token || []) {
          if (Number.isFinite(v)) {
            this.tokenPatternMemory.set(k, v);
          }
        }

        for (const [k, v] of parsed.pos || []) {
          if (Number.isFinite(v)) {
            this.posPatternMemory.set(k, v);
          }
        }

        for (const [k, v] of parsed.association || []) {
          if (Number.isFinite(v)) {
            this.wordAssociationMemory.set(k, v);
          }
        }

        for (const [k, v] of parsed.scopedToken || []) {
          if (Number.isFinite(v)) {
            this.scopedTokenPatternMemory.set(k, v);
          }
        }

        for (const [k, v] of parsed.scopedPos || []) {
          if (Number.isFinite(v)) {
            this.scopedPosPatternMemory.set(k, v);
          }
        }
      }

      const preference = parsed.preference;

      if (preference && typeof preference === "object") {
        for (const key of Object.keys(this.preferenceWeights)) {
          const value = preference.weights?.[key];
          if (Number.isFinite(value)) {
            this.preferenceWeights[key] =
              Math.max(-3.0, Math.min(3.0, value));
          }
        }

        if (Number.isFinite(preference.bias)) {
          this.preferenceBias =
            Math.max(-5.0, Math.min(5.0, preference.bias));
        }

        if (Number.isFinite(preference.examples)) {
          this.preferenceExamples =
            Math.max(0, preference.examples);
        }
      }

      const rl = parsed.rlPolicy;
      if (rl && Array.isArray(rl.weights)) {
        const n = Math.min(this.rlFeatureCount, rl.weights.length);
        for (let i = 0; i < n; i++) {
          if (Number.isFinite(rl.weights[i])) {
            this.rlPolicyWeights[i] = Math.max(
              -this.reasoningConfig.rlWeightLimit,
              Math.min(this.reasoningConfig.rlWeightLimit, rl.weights[i])
            );
          }
        }
        this.rlUpdates = Number.isFinite(rl.updates) ? Math.max(0, rl.updates) : 0;
      }

      if (
        !legacyImport &&
        Array.isArray(parsed.rlReplay)
      ) {
        this.rlReplay =
          parsed.rlReplay
            .filter(
              item =>
                item &&
                Array.isArray(item.features) &&
                item.features.length ===
                  this.rlFeatureCount &&
                Number.isFinite(item.target)
            )
            .slice(
              -this.reasoningConfig
                .rlReplayLimit
            );
      }

      if (
        !legacyImport &&
        Array.isArray(parsed.rsiReplay)
      ) {
        this.rsiReplay =
          parsed.rsiReplay
            .filter(
              item =>
                item &&
                Array.isArray(item.bad) &&
                item.bad.length ===
                  this.rlFeatureCount &&
                Array.isArray(item.good) &&
                item.good.length ===
                  this.rlFeatureCount &&
                Number.isFinite(item.margin)
            )
            .slice(
              -this.reasoningConfig
                .automaticRSIReplayLimit
            );
      }

      if (
        !legacyImport &&
        Number.isFinite(
          parsed.rsiUpdates
        )
      ) {
        this.rsiUpdates =
          Math.max(
            0,
            parsed.rsiUpdates
          );
      }

      if (
        !legacyImport &&
        Number.isFinite(
          parsed.automaticMistakesLearned
        )
      ) {
        this.automaticMistakesLearned =
          Math.max(
            0,
            parsed.automaticMistakesLearned
          );
      }

      if (
        !legacyImport &&
        parsed.selfWhyStats &&
        typeof parsed.selfWhyStats === "object"
      ) {
        if (
          Number.isFinite(
            parsed.selfWhyStats.disagreementEMA
          )
        ) {
          this.selfWhyDisagreementEMA =
            Math.max(
              0,
              Math.min(
                1,
                parsed.selfWhyStats.disagreementEMA
              )
            );
        }

        if (
          Number.isFinite(
            parsed.selfWhyStats.disagreementCount
          )
        ) {
          this.selfWhyDisagreementCount =
            Math.max(
              0,
              parsed.selfWhyStats.disagreementCount
            );
        }
      }

      if (Array.isArray(parsed.lexicalUsage)) {
        for (const [key, value] of parsed.lexicalUsage) {
          const token = Number(key);
          if (
            Number.isInteger(token) &&
            Number.isFinite(value) &&
            value > 0
          ) {
            this.recentLexicalUsage.set(
              token,
              Math.min(4.0, value)
            );
          }
        }
      }

      if (Number.isFinite(parsed.lexicalUsageTurns)) {
        this.lexicalUsageTurns =
          Math.max(0, parsed.lexicalUsageTurns);
      }
    } catch (_) {
      // Persistence is optional. Generation must never depend on storage.
    }
  }

  savePatternMemory() {
    if (typeof localStorage === "undefined") return;

    try {
      localStorage.setItem(
        this.persistenceKey,
        JSON.stringify({
          token: Array.from(this.tokenPatternMemory.entries()),
          pos: Array.from(this.posPatternMemory.entries()),
          association: Array.from(this.wordAssociationMemory.entries()),
          scopedToken: Array.from(this.scopedTokenPatternMemory.entries()),
          scopedPos: Array.from(this.scopedPosPatternMemory.entries()),
          preference: {
            weights: { ...this.preferenceWeights },
            bias: this.preferenceBias,
            examples: this.preferenceExamples
          },
          rlPolicy: {
            version: 1,
            weights: Array.from(this.rlPolicyWeights),
            updates: this.rlUpdates
          },
          rlReplay: this.rlReplay.slice(-256),
          rsiReplay:
            this.rsiReplay.slice(
              -this.reasoningConfig
                .automaticRSIReplayLimit
            ),
          rsiUpdates:
            this.rsiUpdates,
          automaticMistakesLearned:
            this.automaticMistakesLearned,
          selfWhyStats: {
            disagreementEMA:
              this.selfWhyDisagreementEMA,
            disagreementCount:
              this.selfWhyDisagreementCount
          },
          lexicalUsage:
            Array.from(
              this.recentLexicalUsage.entries()
            )
              .sort((a, b) => b[1] - a[1])
              .slice(
                0,
                this.reasoningConfig
                  .lexicalFatigueLimit
              ),
          lexicalUsageTurns:
            this.lexicalUsageTurns
        })
      );
    } catch (_) {
      // Ignore quota/privacy-mode failures.
    }
  }


  loadUserPredictiveMemory() {
    if (typeof localStorage === "undefined") return;
    try {
      const raw = localStorage.getItem(this.predictivePersistenceKey);
      if (!raw) return;
      const p = JSON.parse(raw);
      const restore = (map, arr, limit) => {
        if (!Array.isArray(arr)) return;
        for (const [k, v] of arr.slice(0, limit)) {
          if (k != null && Number.isFinite(Number(v)) && Number(v) > 0) map.set(String(k), Math.min(12, Number(v)));
        }
      };
      restore(this.userTokenTransitions, p.token, this.reasoningConfig.userTokenTransitionLimit);
      restore(this.userPOSTransitions, p.pos, this.reasoningConfig.userPOSTransitionLimit);
      restore(this.userCrossTurnTransitions, p.cross, this.reasoningConfig.userCrossTurnTransitionLimit);
      if (Array.isArray(p.weights)) for (let i = 0; i < Math.min(p.weights.length, this.userStateWeights.length); i++) if (Number.isFinite(p.weights[i])) this.userStateWeights[i] = p.weights[i];
      if (Array.isArray(p.bias)) for (let i = 0; i < Math.min(p.bias.length, this.userStateBias.length); i++) if (Number.isFinite(p.bias[i])) this.userStateBias[i] = p.bias[i];
      const vec = a => Array.isArray(a) && a.length === this.userStateDim ? Float32Array.from(a) : null;
      this.pendingUserState = vec(p.pendingState);
      this.pendingUserPrediction = vec(p.pendingPrediction);
      this.pendingNextUserTokens = Array.isArray(p.pendingTokens) ? p.pendingTokens.filter(Number.isInteger).slice(0, this.reasoningConfig.userNextTokenLimit) : [];
      this.previousUserAnchors = Array.isArray(p.previousAnchors) ? p.previousAnchors.filter(Number.isInteger).slice(0, this.reasoningConfig.userCarryoverAnchorLimit) : [];
      this.previousUserLastContent = Number.isInteger(p.previousLastContent) ? p.previousLastContent : null;
      if (Number.isFinite(p.errorEMA)) this.userStateErrorEMA = Math.max(0, Math.min(1, p.errorEMA));
      if (Number.isFinite(p.hitEMA)) this.userNextTokenHitEMA = Math.max(0, Math.min(1, p.hitEMA));
      if (Number.isFinite(p.turns)) this.userInputTurns = Math.max(0, p.turns);
    } catch (_) {}
  }

  saveUserPredictiveMemory() {
    if (typeof localStorage === "undefined") return;
    try {
      const top = (map, limit) => Array.from(map.entries()).sort((a,b) => b[1]-a[1]).slice(0, limit);
      localStorage.setItem(this.predictivePersistenceKey, JSON.stringify({
        version: 1,
        token: top(this.userTokenTransitions, this.reasoningConfig.userTokenTransitionLimit),
        pos: top(this.userPOSTransitions, this.reasoningConfig.userPOSTransitionLimit),
        cross: top(this.userCrossTurnTransitions, this.reasoningConfig.userCrossTurnTransitionLimit),
        weights: Array.from(this.userStateWeights),
        bias: Array.from(this.userStateBias),
        pendingState: this.pendingUserState ? Array.from(this.pendingUserState) : null,
        pendingPrediction: this.pendingUserPrediction ? Array.from(this.pendingUserPrediction) : null,
        pendingTokens: this.pendingNextUserTokens,
        previousAnchors: this.previousUserAnchors,
        previousLastContent: this.previousUserLastContent,
        errorEMA: this.userStateErrorEMA,
        hitEMA: this.userNextTokenHitEMA,
        turns: this.userInputTurns
      }));
    } catch (_) {}
  }

  trimUserPredictiveMap(map, limit) {
    if (map.size <= limit) return;
    const keep = Array.from(map.entries()).sort((a,b) => b[1]-a[1]).slice(0, limit);
    map.clear();
    for (const [k,v] of keep) map.set(k,v);
  }

  trimPatternMap(map, limit = this.maxPatternEntries) {
    if (map.size <= limit) return;

    const entries = Array.from(map.entries())
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, limit);

    map.clear();
    for (const [k, v] of entries) map.set(k, v);
  }

  getPreferenceScope(contextModel) {
    const intent =
      contextModel?.instructionContract?.intent ||
      "answer";
    const shape =
      contextModel?.answerShape?.kind ||
      "prose";
    const role =
      contextModel?.instructionContract?.requiredRole ||
      "content";
    return `${intent}|${shape}|${role}`;
  }

  preferenceFeatureVector(metrics) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value) ? value : 0
        )
      );

    return {
      instruction: clamp(
        metrics?.instructionAdherenceScore ??
        metrics?.instructionAdherence
      ),
      answer: clamp(
        metrics?.answerCompletionScore ??
        metrics?.answerCompletion
      ),
      semantic: clamp(
        metrics?.semanticAnswerIntegrity ??
        metrics?.semantic
      ),
      coherence: clamp(
        metrics?.globalCoherenceScore ??
        metrics?.globalCoherence ??
        metrics?.coherence
      ),
      structure: clamp(
        metrics?.shapeDecodabilityScore ??
        metrics?.shapeDecodability ??
        metrics?.structureGroup
      ),
      grounding: clamp(
        metrics?.groundingIntegrityScore ??
        metrics?.groundingIntegrity ??
        metrics?.groundedWordRatio
      ),
      critic: clamp(
        metrics?.universalCriticScore ??
        metrics?.universalCritic ??
        metrics?.criticalFloor
      ),
      repetition: clamp(
        metrics?.repetitionScore ??
        metrics?.repetition
      ),
      echo: clamp(
        metrics?.echoIntegrity ??
        1.0
      )
    };
  }

  predictPreference(metrics) {
    const x = this.preferenceFeatureVector(metrics);
    let z = this.preferenceBias;

    for (const [key, weight] of Object.entries(this.preferenceWeights)) {
      z += weight * (x[key] ?? 0);
    }

    z = Math.max(-12, Math.min(12, z));
    return 1.0 / (1.0 + Math.exp(-z));
  }

  updatePreferenceModel(reward, metrics) {
    if (!Number.isFinite(reward) || !metrics) return;

    const target = reward > 0 ? 1.0 : 0.0;
    const x = this.preferenceFeatureVector(metrics);
    const prediction = this.predictPreference(metrics);
    const error = target - prediction;
    const lr =
      this.preferenceLearningRate *
      (0.60 + Math.min(1, Math.abs(error)) * 0.80);

    for (const key of Object.keys(this.preferenceWeights)) {
      this.preferenceWeights[key] =
        Math.max(
          -3.0,
          Math.min(
            3.0,
            this.preferenceWeights[key] +
            lr * error * (x[key] ?? 0)
          )
        );
    }

    this.preferenceBias =
      Math.max(
        -5.0,
        Math.min(
          5.0,
          this.preferenceBias + lr * error * 0.35
        )
      );

    this.preferenceExamples++;
  }

  getFeedbackPolicyUpdate(reward, diagnostics = null) {
    const signed =
      Math.max(-1, Math.min(1, Number(reward) || 0));

    const critic =
      Math.max(
        0,
        Math.min(
          1,
          diagnostics?.eventQueryResolution ??
          diagnostics?.universalCritic ??
          diagnostics?.criticalFloor ??
          0.50
        )
      );

    const preference =
      diagnostics
        ? this.predictPreference(diagnostics)
        : 0.50;

    const qualityScale =
      signed > 0
        ? (0.30 + critic * 0.55 + preference * 0.15)
        : (0.82 + (1.0 - critic) * 0.18);

    const surprise =
      diagnostics
        ? Math.abs((signed > 0 ? 1.0 : 0.0) - preference)
        : 0.50;

    return {
      reward:
        signed *
        Math.max(0.25, Math.min(1.0, qualityScale)),
      lr: 0.0045 + surprise * 0.0045
    };
  }

  patternMemoryBonus(
    generated,
    candidateToken,
    contextModel = null
  ) {
    let bonus = 0.0;
    const n = generated.length;
    const candidate = this.canonicalTokenId(candidateToken);
    const scope =
      contextModel
        ? this.getPreferenceScope(contextModel)
        : null;

    if (n >= 1) {
      const prev = this.canonicalTokenId(generated[n - 1]);
      const tokenBigram = `${prev}>${candidate}`;
      const a = this.getCanonicalType(prev);
      const b = this.getCanonicalType(candidate);

      bonus += (this.tokenPatternMemory.get(tokenBigram) || 0) * 0.25;
      bonus += (this.posPatternMemory.get(`${a}>${b}`) || 0) * 0.18;

      if (scope) {
        bonus +=
          (this.scopedTokenPatternMemory.get(`${scope}|${tokenBigram}`) || 0) * 0.82;
        bonus +=
          (this.scopedPosPatternMemory.get(`${scope}|${a}>${b}`) || 0) * 0.58;
      }
    }

    if (n >= 2) {
      const prev2 = this.canonicalTokenId(generated[n - 2]);
      const prev1 = this.canonicalTokenId(generated[n - 1]);
      const tokenTrigram = `${prev2}>${prev1}>${candidate}`;
      const a = this.getCanonicalType(prev2);
      const b = this.getCanonicalType(prev1);
      const c = this.getCanonicalType(candidate);

      bonus += (this.tokenPatternMemory.get(tokenTrigram) || 0) * 0.34;
      bonus += (this.posPatternMemory.get(`${a}>${b}>${c}`) || 0) * 0.26;

      if (scope) {
        bonus +=
          (this.scopedTokenPatternMemory.get(`${scope}|${tokenTrigram}`) || 0) * 1.05;
        bonus +=
          (this.scopedPosPatternMemory.get(`${scope}|${a}>${b}>${c}`) || 0) * 0.76;
      }
    }

    return Math.max(-4.0, Math.min(4.0, bonus));
  }

  learnTrajectoryPatterns(
    reward,
    contextModel = null
  ) {
    if (!this.trajectory?.length || !Number.isFinite(reward)) return;

    const signed = Math.max(-1, Math.min(1, reward));
    if (Math.abs(signed) < 0.05) return;

    const tokens = this.trajectory
      .map(item => item?.token)
      .filter(Number.isFinite)
      .map(token => this.canonicalTokenId(token));

    if (tokens.length < 2) return;

    const scope =
      contextModel
        ? this.getPreferenceScope(contextModel)
        : "general|prose|content";

    const globalStep = signed * 0.045;
    const scopedStep = signed * 0.18;

    const update = (map, key, delta) => {
      const next = Math.max(
        -2.5,
        Math.min(4.0, (map.get(key) || 0) + delta)
      );
      map.set(key, next);
    };

    for (let i = 1; i < tokens.length; i++) {
      const bigram = `${tokens[i - 1]}>${tokens[i]}`;
      const p0 = this.getCanonicalType(tokens[i - 1]);
      const p1 = this.getCanonicalType(tokens[i]);

      update(this.tokenPatternMemory, bigram, globalStep * 0.50);
      update(this.posPatternMemory, `${p0}>${p1}`, globalStep * 0.40);
      update(this.scopedTokenPatternMemory, `${scope}|${bigram}`, scopedStep * 0.70);
      update(this.scopedPosPatternMemory, `${scope}|${p0}>${p1}`, scopedStep * 0.52);

      if (i >= 2) {
        const trigram = `${tokens[i - 2]}>${tokens[i - 1]}>${tokens[i]}`;
        const p2 = this.getCanonicalType(tokens[i - 2]);

        update(this.tokenPatternMemory, trigram, globalStep * 0.62);
        update(this.posPatternMemory, `${p2}>${p0}>${p1}`, globalStep * 0.52);
        update(this.scopedTokenPatternMemory, `${scope}|${trigram}`, scopedStep);
        update(this.scopedPosPatternMemory, `${scope}|${p2}>${p0}>${p1}`, scopedStep * 0.82);
      }
    }

    this.trimPatternMap(this.tokenPatternMemory);
    this.trimPatternMap(this.posPatternMemory);
    this.trimPatternMap(this.scopedTokenPatternMemory, this.maxScopedPatternEntries);
    this.trimPatternMap(this.scopedPosPatternMemory, this.maxScopedPatternEntries);
  }

  buildLocalPatternContext(generated) {
    const recent = generated.slice(-32);
    const seenBigrams = new Set();
    const seenTrigrams = new Set();
    const contentTokens = [];

    let maxFunctionRun = 0;
    let functionRun = 0;

    for (let i = 0; i < recent.length; i++) {
      const tok = recent[i];
      const type = this.getCanonicalType(tok);

      if (this.isContentType(type)) {
        contentTokens.push(tok);
      }

      const isFunction =
        ["Det", "Prep", "Conj", "Modal", "Aux"].includes(type);

      if (isFunction) {
        functionRun++;
        maxFunctionRun = Math.max(maxFunctionRun, functionRun);
      } else if (type !== "Punct") {
        functionRun = 0;
      }

      if (i >= 1) {
        seenBigrams.add(`${recent[i - 1]}>${tok}`);
      }

      if (i >= 2) {
        seenTrigrams.add(
          `${recent[i - 2]}>${recent[i - 1]}>${tok}`
        );
      }
    }

    let recentContentCentroid = null;

    if (contentTokens.length && this.normalizedEmbeddings) {
      const embDim = this.network.embDim;
      recentContentCentroid = new Float32Array(embDim);

      const tail = contentTokens.slice(-5);

      for (const tok of tail) {
        const off = (tok % this.network.vocabSize) * embDim;
        for (let d = 0; d < embDim; d++) {
          recentContentCentroid[d] += this.normalizedEmbeddings[off + d];
        }
      }

      for (let d = 0; d < embDim; d++) {
        recentContentCentroid[d] /= tail.length;
      }

      this.normalizeVector(recentContentCentroid);
    }

    return {
      seenBigrams,
      seenTrigrams,
      recentContentCentroid,
      recentContentTokens:
        contentTokens.slice(-3),
      maxFunctionRun
    };
  }

  getClauseSemanticEvidence(candidateToken, sentenceState) {
    const type = this.getCanonicalType(candidateToken);

    if (!this.isContentType(type)) {
      return {
        evidence: 0.62,
        similarity: 0.0,
        locked: false
      };
    }

    const subjectToken =
      Number.isFinite(sentenceState?.subjectToken)
        ? sentenceState.subjectToken
        : null;

    const predicateToken =
      Number.isFinite(sentenceState?.predicateToken)
        ? sentenceState.predicateToken
        : null;

    const lastContentToken =
      Number.isFinite(sentenceState?.lastContentToken)
        ? sentenceState.lastContentToken
        : null;

    const locked =
      predicateToken !== null &&
      (sentenceState?.clauseContentCount || 0) >= 2;

    const similarities = [];
    const weights = [];

    if (subjectToken !== null) {
      similarities.push(
        this.dotTokenPair(candidateToken, subjectToken)
      );
      weights.push(locked ? 0.30 : 0.48);
    }

    if (predicateToken !== null) {
      similarities.push(
        this.dotTokenPair(candidateToken, predicateToken)
      );
      weights.push(locked ? 0.46 : 0.30);
    }

    if (lastContentToken !== null) {
      similarities.push(
        this.dotTokenPair(candidateToken, lastContentToken)
      );
      weights.push(locked ? 0.24 : 0.22);
    }

    if (!similarities.length) {
      return {
        evidence: 0.62,
        similarity: 0.0,
        locked
      };
    }

    let weightedSimilarity = 0.0;
    let weightSum = 0.0;

    for (let i = 0; i < similarities.length; i++) {
      weightedSimilarity += similarities[i] * weights[i];
      weightSum += weights[i];
    }

    const similarity =
      weightSum > 0
        ? weightedSimilarity / weightSum
        : 0.0;

    const evidence = Math.max(
      0,
      Math.min(
        1,
        (similarity + 0.10) / 0.62
      )
    );

    return {
      evidence,
      similarity,
      locked
    };
  }

  updateClauseSemanticState(sentenceState, token, curType) {
    if (!sentenceState || !this.isContentType(curType)) return;

    const previous =
      Number.isFinite(sentenceState.lastContentToken)
        ? sentenceState.lastContentToken
        : null;

    let continuity = 0.72;

    if (previous !== null) {
      const sim = this.dotTokenPair(previous, token);

      continuity = Math.max(
        0,
        Math.min(
          1,
          (sim + 0.10) / 0.62
        )
      );
    }

    const oldEMA =
      Number.isFinite(sentenceState.clauseSemanticEMA)
        ? sentenceState.clauseSemanticEMA
        : continuity;

    sentenceState.clauseSemanticEMA =
      oldEMA * 0.72 +
      continuity * 0.28;

    if (continuity < 0.25) {
      sentenceState.clauseDriftCount =
        (sentenceState.clauseDriftCount || 0) + 1;
    } else if (continuity > 0.55) {
      sentenceState.clauseDriftCount =
        Math.max(
          0,
          (sentenceState.clauseDriftCount || 0) - 1
        );
    }

    sentenceState.lastContentToken = token;
    sentenceState.clauseContentCount =
      (sentenceState.clauseContentCount || 0) + 1;
  }

  getContinuationEvidence(
    candidateToken,
    generated,
    contextModel,
    sentenceState,
    patternContext,
    routedSemantic
  ) {
    const type = this.getCanonicalType(candidateToken);
    const prevToken =
      generated.length
        ? generated[generated.length - 1]
        : null;

    const prevType =
      prevToken !== null
        ? this.getCanonicalType(prevToken)
        : null;

    const transition = this.getTransitionRule(
      prevType,
      type
    );

    const transitionEvidence = transition.legal
      ? Math.max(
          0,
          Math.min(1, transition.weight / 5.0)
        )
      : 0.0;

    let localSimilarity = 0.0;
    let hasLocalSemanticContext = false;

    if (
      this.isContentType(type) &&
      patternContext?.recentContentCentroid
    ) {
      localSimilarity = this.dotTokenVector(
        candidateToken,
        patternContext.recentContentCentroid
      );
      hasLocalSemanticContext = true;
    }

    const localEvidence = hasLocalSemanticContext
      ? Math.max(
          0,
          Math.min(
            1,
            (localSimilarity + 0.12) / 0.72
          )
        )
      : 0.62;

    const routeEvidence = Math.max(
      0,
      Math.min(
        1,
        (routedSemantic + 0.08) / 0.72
      )
    );

    const memoryRaw =
      this.patternMemoryBonus(
        generated,
        candidateToken,
        contextModel
      );

    const learnedPatternEvidence = Math.max(
      0,
      Math.min(
        1,
        0.50 + memoryRaw / 8.0
      )
    );

    const clause =
      this.getClauseSemanticEvidence(
        candidateToken,
        sentenceState
      );

    const association =
      this.getGroundedWordAssociation(
        candidateToken,
        generated,
        contextModel,
        sentenceState,
        patternContext
      );

    let evidence;

    if (clause.locked) {
      evidence =
        clause.evidence * 0.28 +
        localEvidence * 0.16 +
        routeEvidence * 0.16 +
        association.evidence * 0.28 +
        transitionEvidence * 0.07 +
        learnedPatternEvidence * 0.05;
    } else {
      evidence =
        clause.evidence * 0.12 +
        localEvidence * 0.17 +
        routeEvidence * 0.24 +
        association.evidence * 0.28 +
        transitionEvidence * 0.12 +
        learnedPatternEvidence * 0.07;
    }

    return {
      evidence: Math.max(0, Math.min(1, evidence)),
      localSimilarity,
      localEvidence,
      routeEvidence,
      transitionEvidence,
      learnedPatternEvidence,
      associationEvidence:
        association.evidence,
      associationLocal:
        association.local,
      associationAnchor:
        association.anchor,
      associationTriangle:
        association.triangle,
      associationLearned:
        association.learned,
      associationStatic:
        association.staticKnowledge || 0.0,
      associationKnowledge:
        association.knowledgeSupport || 0.55,
      associationTyped:
        association.typedKnowledge || 0.0,
      associationCollocation:
        association.collocationKnowledge || 0.0,
      associationClassFit:
        association.classFit || 0.55,
      associationKnowledgeKnown:
        Boolean(association.knowledgeKnown),
      clauseEvidence: clause.evidence,
      clauseSimilarity: clause.similarity,
      clauseLocked: clause.locked
    };
  }

  dynamicWordPatternScore(
    candidateToken,
    generated,
    contextModel,
    sentenceState,
    patternContext,
    continuation = null
  ) {
    const type = this.getCanonicalType(candidateToken);
    const n = generated.length;
    let score =
      this.patternMemoryBonus(
        generated,
        candidateToken,
        contextModel
      );

    // Prevent exact local phrase loops before they are emitted.
    if (n >= 1) {
      const bigram =
        `${generated[n - 1]}>${candidateToken}`;

      if (patternContext.seenBigrams.has(bigram)) {
        score -= 7.5;
      }
    }

    if (n >= 2) {
      const trigram =
        `${generated[n - 2]}>${generated[n - 1]}>${candidateToken}`;

      if (patternContext.seenTrigrams.has(trigram)) {
        score -= 15.0;
      }
    }

    // Local semantic continuation now has enough weight to matter.
    if (
      this.isContentType(type) &&
      patternContext.recentContentCentroid
    ) {
      const localSimilarity =
        continuation?.localSimilarity ??
        this.dotTokenVector(
          candidateToken,
          patternContext.recentContentCentroid
        );

      if (localSimilarity > 0.08) {
        score += Math.min(
          4.0,
          localSimilarity * 5.2
        );
      } else if (
        localSimilarity < -0.05 &&
        sentenceState.predicateReady
      ) {
        score -= Math.min(
          5.5,
          Math.abs(localSimilarity) * 6.5 + 1.0
        );
      }
    }

    // Word-to-word grounded association is a first-class meaning signal.
    if (
      continuation &&
      this.isContentType(type)
    ) {
      score +=
        (continuation.associationEvidence - 0.50) *
        7.0;

      // Strong local association with weak prompt anchoring is semantic drift:
      // each word may "make sense" after the last one while the sentence walks
      // away from the user's question.
      if (
        continuation.associationLocal >= 0.52 &&
        continuation.associationAnchor < 0.25 &&
        n >= 3
      ) {
        score -= 5.0;
      }

      // A new content word that connects to neither its local thread nor the
      // stable prompt anchors should almost never survive Deep search.
      if (
        continuation.associationLocal < 0.18 &&
        continuation.associationAnchor < 0.18 &&
        sentenceState.predicateReady &&
        n >= 4
      ) {
        score -= 8.0;
      }

      if (
        continuation.associationTriangle >= 0.62
      ) {
        score += 1.8;
      }

      // Positive association rule:
      // when vocab knowledge and live semantic evidence agree, the candidate
      // receives up to +0.5 "good". This now applies across the whole content
      // vocabulary instead of a hand-picked concept list.
      score +=
        Math.min(
          0.5,
          Math.max(
            0,
            (
              continuation.associationKnowledge || 0.55
            ) - 0.50
          )
        );

      score +=
        Math.min(
          0.8,
          (continuation.associationTyped || 0) * 0.45 +
          (continuation.associationCollocation || 0) * 0.35
        );
    }

    // The combined continuation evidence acts like a cheap local verifier.
    if (continuation) {
      score +=
        (continuation.evidence - 0.50) * 4.3;

      if (continuation.clauseLocked) {
        score +=
          (continuation.clauseEvidence - 0.50) * 5.8;

        if (
          continuation.clauseEvidence < 0.22 &&
          continuation.routeEvidence < 0.34
        ) {
          score -= 4.5;
        }
      }
    }

    // Semantic drift recovery: re-anchor or stop instead of extending garbage.
    const driftEMA =
      Number.isFinite(sentenceState?.clauseSemanticEMA)
        ? sentenceState.clauseSemanticEMA
        : 0.72;

    if (
      continuation?.clauseLocked &&
      driftEMA < 0.42
    ) {
      if (
        continuation.clauseEvidence >= 0.58 ||
        continuation.routeEvidence >= 0.62
      ) {
        score += 2.4;
      } else if (this.isContentType(type)) {
        score -= 2.8;
      }
    }

    if (
      this.clauseComplete(sentenceState) &&
      ["Conj", "Prep", "Adv"].includes(type)
    ) {
      score -= Math.max(
        0,
        patternContext.maxFunctionRun - 1
      ) * 1.0;
    }

    return score;
  }

  normalizeVector(vec) {
    let normSq = 0;
    for (let i = 0; i < vec.length; i++) normSq += vec[i] * vec[i];
    if (normSq <= 1e-12) return vec;

    const inv = 1.0 / Math.sqrt(normSq);
    for (let i = 0; i < vec.length; i++) vec[i] *= inv;
    return vec;
  }

  blendVectors(a, b, wa = 0.5, wb = 0.5) {
    const out = new Float32Array(a.length);
    for (let i = 0; i < out.length; i++) {
      out[i] = a[i] * wa + b[i] * wb;
    }
    return this.normalizeVector(out);
  }

  dotTokenVector(tokenId, vector) {
    if (!vector || !this.normalizedEmbeddings) return 0.0;
    const embDim = this.network.embDim;
    const off = (tokenId % this.network.vocabSize) * embDim;

    let dot = 0;
    for (let d = 0; d < embDim; d++) {
      dot += this.normalizedEmbeddings[off + d] * vector[d];
    }
    return Math.max(-1.0, Math.min(1.0, dot));
  }

  dotTokenPair(aToken, bToken) {
    if (!this.normalizedEmbeddings) return 0.0;

    const embDim = this.network.embDim;
    const aOff = (aToken % this.network.vocabSize) * embDim;
    const bOff = (bToken % this.network.vocabSize) * embDim;

    let dot = 0.0;
    for (let d = 0; d < embDim; d++) {
      dot +=
        this.normalizedEmbeddings[aOff + d] *
        this.normalizedEmbeddings[bOff + d];
    }

    return Math.max(-1.0, Math.min(1.0, dot));
  }

  /**
   * Dual-Centroid Semantic Routing
   *
   * topicCentroid  = entities / concepts / descriptive content
   * intentCentroid = actions / relations / requested operation
   *
   * The routing weights change with clause state instead of collapsing the
   * entire prompt into one average semantic vector.
   */
  getRoutedSemanticScore(tokenId, contextModel, sentenceState = null) {
    const topic = contextModel?.topicScores?.[tokenId]
      ?? contextModel?.semanticScores?.[tokenId]
      ?? 0.0;

    const intent = contextModel?.intentScores?.[tokenId]
      ?? contextModel?.semanticScores?.[tokenId]
      ?? 0.0;

    let topicWeight = 0.60;
    let intentWeight = 0.40;

    if (sentenceState) {
      if (!sentenceState.subjectReady) {
        // Find the thing we are talking about.
        topicWeight = 0.72;
        intentWeight = 0.28;
      } else if (!sentenceState.predicateReady) {
        // Once a subject exists, aggressively route toward the action/relation.
        topicWeight = 0.25;
        intentWeight = 0.75;
      } else {
        // After the predicate, route back toward useful content/complements.
        topicWeight = 0.64;
        intentWeight = 0.36;
      }
    }

    const dual = topic * topicWeight + intent * intentWeight;

    // Latent reasoning is a third signal, not a replacement for the two
    // explicit centroids. This keeps routing interpretable and stable.
    const latent = contextModel?.reasoningScores?.[tokenId];
    if (Number.isFinite(latent)) {
      return Math.max(-1.0, Math.min(1.0, dual * 0.68 + latent * 0.32));
    }

    return Math.max(-1.0, Math.min(1.0, dual));
  }

  /**
   * Latent Chain-of-Thought Reasoning
   *
   * No textual chain is generated or exposed. Deep mode performs several tiny
   * 64-dimensional refinement hops over a compact candidate neighborhood from
   * the SAME neural forward pass. Each hop updates an internal reasoning
   * centroid, then generation uses that latent state for semantic routing.
   *
   * This adds reasoning depth without adding another 3M-neuron WebGPU pass.
   */
  runLatentReasoning(rawLogits, contextModel, maxHops = 4) {
    const vocab = this.network.vocab;
    const vocabSize = this.network.vocabSize;
    const embDim = this.network.embDim;

    if (!rawLogits || !contextModel || !this.normalizedEmbeddings) {
      return {
        centroid: contextModel?.centroid || null,
        scores: null,
        stability: 0.0,
        hopsUsed: 0
      };
    }

    const topicCentroid = contextModel.topicCentroid || contextModel.centroid;
    const intentCentroid = contextModel.intentCentroid || contextModel.centroid;

    let state = this.blendVectors(topicCentroid, intentCentroid, 0.58, 0.42);

    // Build a compact reasoning neighborhood once. Raw logits provide neural
    // evidence; the two centroids provide semantic evidence.
    const pool = [];
    let maxRaw = -Infinity;
    for (let i = 0; i < rawLogits.length; i++) {
      if (rawLogits[i] > maxRaw) maxRaw = rawLogits[i];
    }

    for (let i = 0; i < vocabSize; i++) {
      if (!vocab?.hasToken?.(i)) continue;

      const type = this.getCanonicalType(i);
      if (type === "Punct" || type === "Other") continue;

      const topic = contextModel.topicScores?.[i] ?? 0.0;
      const intent = contextModel.intentScores?.[i] ?? 0.0;
      const semantic = Math.max(topic, intent, (topic + intent) * 0.5);

      const neuralSupport = Math.exp(
        Math.max(-8.0, Math.min(0.0, rawLogits[i] - maxRaw))
      );

      const rankScore = semantic * 2.1 + neuralSupport * 1.4;

      if (pool.length < 48) {
        pool.push({ token: i, rankScore, neuralSupport, topic, intent });
        pool.sort((a, b) => b.rankScore - a.rankScore);
      } else if (rankScore > pool[pool.length - 1].rankScore) {
        pool[pool.length - 1] = {
          token: i,
          rankScore,
          neuralSupport,
          topic,
          intent
        };
        pool.sort((a, b) => b.rankScore - a.rankScore);
      }
    }

    if (!pool.length) {
      return {
        centroid: state,
        scores: null,
        stability: 1.0,
        hopsUsed: 0
      };
    }

    let stability = 0.0;
    let hopsUsed = 0;

    for (let hop = 0; hop < maxHops; hop++) {
      const accumulator = new Float32Array(embDim);
      let weightSum = 0.0;

      for (const item of pool) {
        const stateSim = this.dotTokenVector(item.token, state);
        const dualSupport = item.topic * 0.55 + item.intent * 0.45;

        // A bounded attention-like weight. It is intentionally cheap and uses
        // no textual intermediate reasoning.
        const logWeight =
          stateSim * 1.55 +
          dualSupport * 1.05 +
          Math.log(item.neuralSupport + 1e-5);

        const weight = Math.exp(Math.max(-8.0, Math.min(5.0, logWeight)));
        const off = item.token * embDim;

        for (let d = 0; d < embDim; d++) {
          accumulator[d] += this.normalizedEmbeddings[off + d] * weight;
        }
        weightSum += weight;
      }

      if (weightSum > 0) {
        for (let d = 0; d < embDim; d++) accumulator[d] /= weightSum;
        this.normalizeVector(accumulator);
      }

      const nextState = new Float32Array(embDim);
      for (let d = 0; d < embDim; d++) {
        nextState[d] =
          state[d] * 0.50 +
          accumulator[d] * 0.28 +
          topicCentroid[d] * 0.13 +
          intentCentroid[d] * 0.09;
      }
      this.normalizeVector(nextState);

      let dot = 0;
      for (let d = 0; d < embDim; d++) dot += state[d] * nextState[d];
      stability = Math.max(-1.0, Math.min(1.0, dot));

      state = nextState;
      hopsUsed = hop + 1;

      // The latent thought has converged. Do not burn CPU on redundant hops.
      if (hop >= 1 && stability >= 0.9985) break;
    }

    // One full vocabulary projection after the latent chain converges.
    const reasoningScores = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      if (!vocab?.hasToken?.(i)) continue;
      reasoningScores[i] = this.dotTokenVector(i, state);
    }

    return {
      centroid: state,
      scores: reasoningScores,
      stability,
      hopsUsed
    };
  }

  tokenWord(tokenId) {
    return this.network.vocab?.idToToken?.[tokenId] || "";
  }

  sentenceCaseToken(tokenId) {
    const word = this.tokenWord(tokenId);
    if (!word || !/^[a-z]/.test(word)) return tokenId;
    const title = word.charAt(0).toUpperCase() + word.slice(1);
    const alt = this.network.vocab?.tokenToId?.get?.(title);
    return Number.isFinite(alt) ? alt : tokenId;
  }

  /**
   * Structural question analysis.
   *
   * No hardcoded prompts and no "how -> by" / "why -> because" tables.
   * The parser only detects that the input is interrogative and identifies
   * which token behaves like the query operator. Its meaning is left to the
   * learned embeddings, dual centroids, latent reasoning, and RSL search.
   */
  parseTypePhrases(tokens) {
    const Parser = globalThis?.VilotTypePhraseParser;
    if (typeof Parser?.parse !== "function") {
      if (this.reasoningConfig?.phraseParserRequired) {
        throw new Error("VilotTypePhraseParser is missing. Load phrase_parser.js before rl_agent.js.");
      }
      return null;
    }
    return Parser.parse(tokens, {
      getType: token => this.getCanonicalType(token),
      getWord: token => this.tokenWord(token),
      getPronounProfile: token => this.getPronounProfile(this.tokenWord(token))
    });
  }

  analyzePromptStructure(cleanPromptTokens) {
    const phraseProfile = this.parseTypePhrases(cleanPromptTokens);
    const words = cleanPromptTokens.map(token=>String(this.tokenWord(token)||"").toLowerCase());
    const types = cleanPromptTokens.map(token=>this.getCanonicalType(token));
    const hasQuestionMark = Boolean(phraseProfile?.questionEnding);
    const operatorIndex = phraseProfile?.embeddedOperatorIndex >= 0 ? phraseProfile.embeddedOperatorIndex : (phraseProfile?.frontedOperatorIndex ?? -1);
    const helperIndex = phraseProfile?.helperIndex ?? -1;
    const grammaticalSubjectIndex = phraseProfile?.grammaticalSubjectIndex ?? -1;
    const requestVerbIndex = phraseProfile?.matrixPredicateIndex ?? -1;
    const addressedRequest = Boolean(phraseProfile?.requestLike);
    const isQuestion = hasQuestionMark || Boolean(phraseProfile?.helperFronted) || Boolean(phraseProfile?.openQuestionLike);
    return {
      isQuestion,
      hasQuestionMark,
      operatorIndex,
      operatorToken: operatorIndex >= 0 ? cleanPromptTokens[operatorIndex] : null,
      operatorWord: operatorIndex >= 0 ? words[operatorIndex] : "",
      operatorType: operatorIndex >= 0 ? types[operatorIndex] : null,
      helperIndex,
      helperToken: helperIndex >= 0 ? cleanPromptTokens[helperIndex] : null,
      grammaticalSubjectIndex,
      grammaticalSubjectToken: grammaticalSubjectIndex >= 0 ? cleanPromptTokens[grammaticalSubjectIndex] : null,
      grammaticalSubjectWord: grammaticalSubjectIndex >= 0 ? words[grammaticalSubjectIndex] : "",
      subjectPronounPerson: phraseProfile?.subjectPronounPerson ?? null,
      subjectPronounNumber: phraseProfile?.subjectPronounNumber ?? null,
      addressedRequest,
      requestLikelihood: phraseProfile?.requestLikelihood || 0.0,
      requestVerbIndex,
      requestVerbToken: requestVerbIndex >= 0 ? cleanPromptTokens[requestVerbIndex] : null,
      phraseProfile
    };
  }

  instructionRoleFromContract(answerContract) {
    switch (answerContract?.kind) {
      case "mechanism": return "mechanism";
      case "causal": return "cause";
      case "effect": return "content";
      case "temporal": return "time";
      case "locative": return "location";
      case "quantitative": return "quantity";
      case "definition": return "definition";
      case "evaluative": return "evaluation";
      case "boolean": return "boolean";
      case "imperative": return "content";
      case "factual_generation": return "content";
      default: return "content";
    }
  }

  compileInstructionContract(
    cleanPromptTokens,
    questionProfile,
    answerContract,
    answerShape,
    promptAnchorTokens
  ) {
    const vocab = this.network?.vocab;

    let intent = "answer";

    switch (answerContract?.kind) {
      case "mechanism": intent = "explain_mechanism"; break;
      case "causal": intent = "explain_cause"; break;
      case "effect": intent = "explain_effect"; break;
      case "temporal": intent = "answer_time"; break;
      case "locative": intent = "answer_location"; break;
      case "quantitative": intent = "answer_quantity"; break;
      case "definition": intent = "define"; break;
      case "evaluative": intent = "evaluate"; break;
      case "boolean": intent = "yes_no"; break;
      case "factual_generation": intent = "generate_facts"; break;
      case "clarification": intent = "clarify"; break;
      case "imperative": intent = "follow_instruction"; break;
      default:
        intent = questionProfile?.isQuestion
          ? "answer"
          : "continue";
        break;
    }

    if (answerShape?.kind === "list") {
      intent = "list";
    }

    const sourceAnchorTokens = [];
    const sourceAnchorSet = new Set();
    const sourceAnchorStems = new Set();
    const mustReuseTokens = new Set();
    const reusableTokens = new Set();

    const addAnchor = token => {
      if (!Number.isInteger(token) || !vocab?.hasToken?.(token)) return;
      if (sourceAnchorSet.has(token)) return;

      sourceAnchorSet.add(token);
      sourceAnchorTokens.push(token);

      const stem = this.stemWord(this.tokenWord(token));
      if (stem.length >= 3) sourceAnchorStems.add(stem);

      const type = this.getCanonicalType(token);
      const policy = typeof vocab?.getPromptReusePolicy === "function"
        ? vocab.getPromptReusePolicy(token)
        : null;

      if (type === "ProperNoun" || policy?.properName) {
        mustReuseTokens.add(token);
      }
    };

    for (const token of promptAnchorTokens || []) {
      addAnchor(token);
    }

    for (const token of answerContract?.subjectTokens || []) {
      addAnchor(token);
      reusableTokens.add(token);
    }

    if (Number.isInteger(answerContract?.relationVerbToken)) {
      addAnchor(answerContract.relationVerbToken);
      reusableTokens.add(answerContract.relationVerbToken);
    }

    for (const token of answerContract?.numericPromptTokens || []) {
      addAnchor(token);
      reusableTokens.add(token);
    }

    for (const token of answerContract?.quantityUnitTokens || []) {
      addAnchor(token);
      reusableTokens.add(token);
    }

    // Definition answers may repeat the defined term once, but they should
    // still expand beyond it rather than just conjugating/mirroring it.
    if (answerContract?.kind === "definition") {
      for (const token of answerContract?.subjectTokens || []) {
        reusableTokens.add(token);
      }
    }

    let surfaceReuseBudget = 2;
    let minNovelContent = 2;

    switch (answerShape?.kind) {
      case "short":
        surfaceReuseBudget = 1;
        minNovelContent = 1;
        break;
      case "temporal":
        surfaceReuseBudget = 1;
        minNovelContent = answerShape?.temporalMode === "predictive" ? 2 : 1;
        break;
      case "yes_no":
        surfaceReuseBudget = 1;
        minNovelContent = 1;
        break;
      case "definition":
        surfaceReuseBudget = 1;
        minNovelContent = 2;
        break;
      case "list":
        surfaceReuseBudget = Math.max(1, Math.min(3, answerShape?.targetItems || 3));
        minNovelContent = Math.max(2, (answerShape?.targetItems || 3) * 2);
        break;
      default:
        surfaceReuseBudget = 2;
        minNovelContent = answerContract?.requiresRelation ? 3 : 2;
        break;
    }

    const requiredRole = this.instructionRoleFromContract(answerContract);

    return {
      version: 1,
      intent,
      shape: answerShape?.kind || "prose",
      requiredRole,
      isQuestion: Boolean(questionProfile?.isQuestion),
      subjectTokens: [...(answerContract?.subjectTokens || [])],
      predicateToken: answerContract?.relationVerbToken ?? null,
      requiredSlots: [...(answerContract?.requiredSlots || [])],
      sourceAnchorTokens,
      sourceAnchorSet,
      sourceAnchorStems,
      mustReuseTokens,
      reusableTokens,
      surfaceReuseBudget,
      minNovelContent,
      numericPromptTokens:
        [...(answerContract?.numericPromptTokens || [])],
      quantityUnitTokens:
        [...(answerContract?.quantityUnitTokens || [])],
      requiresNumericEvidence:
        Boolean(answerContract?.requiresNumericEvidence),
      factualityRequired:
        Boolean(answerContract?.factualityRequired),
      underspecified:
        Boolean(answerContract?.underspecified),
      expansionRequired:
        Boolean(questionProfile?.isQuestion) ||
        answerContract?.kind === "imperative" ||
        answerContract?.kind === "factual_generation"
    };
  }

  buildAnswerPlan(instructionContract, answerShape) {
    const seedTokens = new Set();

    // Prompt meaning and prompt surface form are deliberately separate.
    // Only tokens whose surface form is genuinely identity-bearing (normally
    // proper names) are forced into the answer candidate pool. Subject nouns
    // and prompt predicates remain semantic anchors, not mandatory output.
    for (const token of instructionContract?.mustReuseTokens || []) {
      seedTokens.add(token);
    }

    return {
      intent: instructionContract?.intent || "answer",
      shape: answerShape?.kind || "prose",
      requiredRole: instructionContract?.requiredRole || "content",
      seedTokens,
      surfaceReuseBudget: instructionContract?.surfaceReuseBudget ?? 2,
      minNovelContent: instructionContract?.minNovelContent ?? 2,
      requiresNumericEvidence:
        Boolean(instructionContract?.requiresNumericEvidence),
      factualityRequired:
        Boolean(instructionContract?.factualityRequired),
      underspecified:
        Boolean(instructionContract?.underspecified),
      numericPromptTokens:
        [...(instructionContract?.numericPromptTokens || [])],
      quantityUnitTokens:
        [...(instructionContract?.quantityUnitTokens || [])],
      expansionRequired: Boolean(instructionContract?.expansionRequired)
    };
  }

  isSourceAnchorFamily(tokenId, contextModel) {
    const contract = contextModel?.instructionContract;
    if (!contract || !Number.isInteger(tokenId)) return false;

    if (contract.sourceAnchorSet?.has(tokenId)) return true;

    const vocab = this.network?.vocab;

    for (const anchor of contract.sourceAnchorTokens || []) {
      if (
        typeof vocab?.sameMorphologicalFamily === "function" &&
        vocab.sameMorphologicalFamily(anchor, tokenId)
      ) {
        return true;
      }
    }

    const stem = this.stemWord(this.tokenWord(tokenId));
    return stem.length >= 3 && contract.sourceAnchorStems?.has(stem);
  }

  getSemanticAnchorSupport(tokenId, contextModel) {
    const contract = contextModel?.instructionContract;
    const vocab = this.network?.vocab;

    if (!contract?.sourceAnchorTokens?.length || !Number.isInteger(tokenId)) {
      return {
        semanticSupport: 0.0,
        expansionSupport: 0.0,
        bestAnchor: null
      };
    }

    let semanticSupport = 0.0;
    let expansionSupport = 0.0;
    let bestAnchor = null;

    for (const anchor of contract.sourceAnchorTokens) {
      let semantic = 0.0;
      let expansion = 0.0;

      if (typeof vocab?.getAnchorSupport === "function") {
        const support = vocab.getAnchorSupport(anchor, tokenId);
        semantic = support?.semanticSupport || 0.0;
        expansion = support?.expansionSupport || 0.0;
      } else {
        const link = this.getVocabKnowledgeLink(anchor, tokenId);
        semantic = Math.max(0, Math.min(1, link?.score || 0.55));
        expansion = Math.max(0, (semantic - 0.50) * 1.7);
      }

      if (semantic > semanticSupport) {
        semanticSupport = semantic;
        bestAnchor = anchor;
      }

      expansionSupport = Math.max(expansionSupport, expansion);
    }

    return {
      semanticSupport: Math.max(0, Math.min(1, semanticSupport)),
      expansionSupport: Math.max(0, Math.min(1, expansionSupport)),
      bestAnchor
    };
  }

  getSourceEchoAssessment(
    tokenId,
    generated,
    sentenceState,
    contextModel
  ) {
    const contract = contextModel?.instructionContract;
    const vocab = this.network?.vocab;

    if (!contract || !this.isSourceAnchorFamily(tokenId, contextModel)) {
      return {
        isEcho: false,
        hardVeto: false,
        scoreAdjustment: 0.0,
        currentReuse: 0
      };
    }

    let currentReuse = 0;
    const usedFamilies = [];

    for (const used of generated) {
      if (!this.isContentType(this.getCanonicalType(used))) continue;
      if (!this.isSourceAnchorFamily(used, contextModel)) continue;

      let newFamily = true;
      for (const prior of usedFamilies) {
        if (
          typeof vocab?.sameMorphologicalFamily === "function" &&
          vocab.sameMorphologicalFamily(prior, used)
        ) {
          newFamily = false;
          break;
        }
      }

      if (newFamily) {
        usedFamilies.push(used);
        currentReuse++;
      }
    }

    const mustReuse = contract.mustReuseTokens?.has(tokenId);
    const reusable = contract.reusableTokens?.has(tokenId);
    const atSubjectStart =
      Boolean(sentenceState?.isStart) &&
      contract.subjectTokens?.includes(tokenId);
    const predicateNeeded =
      !sentenceState?.predicateReady &&
      tokenId === contract.predicateToken;

    if (mustReuse) {
      return {
        isEcho: true,
        hardVeto: false,
        scoreAdjustment: 0.15,
        currentReuse
      };
    }

    if (
      currentReuse >= contract.surfaceReuseBudget &&
      !atSubjectStart &&
      !predicateNeeded
    ) {
      return {
        isEcho: true,
        hardVeto: true,
        scoreAdjustment: -12.0,
        currentReuse
      };
    }

    let reuseCost = 0.70;
    if (typeof vocab?.getPromptReusePolicy === "function") {
      reuseCost = vocab.getPromptReusePolicy(tokenId)?.reuseCost ?? reuseCost;
    }

    let scoreAdjustment = -1.4 - reuseCost * 2.2;

    if (atSubjectStart || predicateNeeded || reusable) {
      scoreAdjustment *= 0.38;
    }

    return {
      isEcho: true,
      hardVeto: false,
      scoreAdjustment,
      currentReuse
    };
  }

  getPromptEchoDiagnostics(tokens, contextModel) {
    const contract = contextModel?.instructionContract;

    if (!contract) {
      return {
        surfaceEchoRatio: 0.0,
        gratuitousEchoRatio: 0.0,
        echoIntegrity: 1.0,
        sourceFamiliesUsed: 0,
        novelContentRatio: 1.0,
        novelExpansionScore: 1.0,
        contentCount: 0
      };
    }

    const vocab = this.network?.vocab;
    let contentCount = 0;
    let echoCount = 0;
    let sourceFamiliesUsed = 0;
    const echoFamilyReps = [];

    for (const token of tokens || []) {
      if (!this.isContentType(this.getCanonicalType(token))) continue;
      contentCount++;

      if (!this.isSourceAnchorFamily(token, contextModel)) continue;
      echoCount++;

      let newFamily = true;
      for (const prior of echoFamilyReps) {
        if (
          typeof vocab?.sameMorphologicalFamily === "function" &&
          vocab.sameMorphologicalFamily(prior, token)
        ) {
          newFamily = false;
          break;
        }
      }

      if (newFamily) {
        echoFamilyReps.push(token);
        sourceFamiliesUsed++;
      }
    }

    const surfaceEchoRatio = contentCount
      ? echoCount / contentCount
      : 0.0;

    const gratuitousFamilies = Math.max(
      0,
      sourceFamiliesUsed - (contract.surfaceReuseBudget || 0)
    );

    const gratuitousEchoRatio = contentCount
      ? gratuitousFamilies / contentCount
      : 0.0;

    const novelContent = Math.max(0, contentCount - echoCount);
    const novelContentRatio = contentCount
      ? novelContent / contentCount
      : 0.0;

    const novelExpansionScore = Math.max(
      0,
      Math.min(
        1,
        novelContent / Math.max(1, contract.minNovelContent || 1)
      )
    );

    const echoIntegrity = Math.max(
      0,
      Math.min(
        1,
        1.0 -
        gratuitousEchoRatio * 1.6 -
        Math.max(0, surfaceEchoRatio - 0.42) * 0.85
      )
    );

    return {
      surfaceEchoRatio,
      gratuitousEchoRatio,
      echoIntegrity,
      sourceFamiliesUsed,
      novelContentRatio,
      novelExpansionScore,
      contentCount
    };
  }

  measureInstructionAdherence(
    tokens,
    contextModel,
    metrics = {}
  ) {
    const echo = metrics.echo || this.getPromptEchoDiagnostics(tokens, contextModel);
    const semanticCoverage = Math.max(0, Math.min(1, metrics.semanticCoverage ?? this.promptCoverage(tokens, contextModel)));
    const shape = Math.max(0, Math.min(1, metrics.shape ?? 0.0));
    const slots = Math.max(0, Math.min(1, metrics.slots ?? 0.0));
    const role = Math.max(0, Math.min(1, metrics.role ?? 0.0));

    const expansion = contextModel?.instructionContract?.expansionRequired
      ? echo.novelExpansionScore
      : Math.max(0.70, echo.novelExpansionScore);

    const score = Math.max(
      0,
      Math.min(
        1,
        semanticCoverage * 0.24 +
        echo.echoIntegrity * 0.22 +
        expansion * 0.18 +
        shape * 0.14 +
        slots * 0.12 +
        role * 0.10
      )
    );

    return {
      score,
      semanticCoverage,
      echoIntegrity: echo.echoIntegrity,
      surfaceEchoRatio: echo.surfaceEchoRatio,
      gratuitousEchoRatio: echo.gratuitousEchoRatio,
      novelContentRatio: echo.novelContentRatio,
      novelExpansionScore: echo.novelExpansionScore,
      sourceFamiliesUsed: echo.sourceFamiliesUsed
    };
  }


  classifyTemporalQuestion(
    cleanPromptTokens,
    questionProfile,
    answerContract
  ) {
    const words =
      cleanPromptTokens.map(
        token =>
          String(
            this.tokenWord(token) || ""
          ).toLowerCase()
      );

    const has =
      word =>
        words.includes(word);

    const futureMarkers =
      [
        "will", "shall", "would",
        "could", "might", "may",
        "should", "can"
      ];

    const pastMarkers =
      [
        "was", "were", "did",
        "had", "happened",
        "occurred", "released",
        "born", "started", "ended"
      ];

    const presentMarkers =
      [
        "is", "are", "am",
        "does", "do",
        "has", "have"
      ];

    const predictive =
      futureMarkers.some(has) ||
      has("future");

    const relative =
      [
        "before", "after", "during",
        "until", "since", "within"
      ].some(has);

    const historical =
      pastMarkers.some(has);

    const present =
      !predictive &&
      !historical &&
      presentMarkers.some(has);

    const tense =
      predictive
        ? "future"
        : historical
          ? "past"
          : present
            ? "present"
            : "neutral";

    const mode =
      predictive
        ? "predictive"
        : relative
          ? "relative"
          : historical
            ? "absolute"
            : "direct";

    return {
      mode,
      tense,
      predictive,
      relative,
      historical,
      present,
      requiresClause:
        predictive ||
        relative,
      allowsFragment:
        !predictive &&
        !relative,
      minimumTemporalEvidence:
        predictive
          ? 0.72
          : 0.68
    };
  }

  temporalTokenEvidence(
    tokenId,
    contextModel = null
  ) {
    const vocab =
      this.network?.vocab;

    let temporal;

    if (
      vocab &&
      typeof vocab.getTemporalProfile ===
        "function"
    ) {
      temporal =
        vocab.getTemporalProfile(
          tokenId
        );
    } else {
      const type =
        this.getCanonicalType(
          tokenId
        );

      const word =
        String(
          this.tokenWord(tokenId) || ""
        ).toLowerCase();

      let score = 0.0;
      let kind = "none";

      if (type === "Num") {
        score = 0.92;
        kind = "number";
      } else if (
        [
          "before", "after", "during",
          "until", "since", "within",
          "around", "by"
        ].includes(word)
      ) {
        score = 0.78;
        kind = "boundary";
      } else if (
        [
          "today", "tomorrow",
          "yesterday", "soon",
          "later", "eventually",
          "now", "currently"
        ].includes(word)
      ) {
        score = 0.88;
        kind = "relative";
      }

      temporal = {
        score,
        kind,
        absolute: false,
        relative:
          kind === "relative",
        duration: false,
        boundary:
          kind === "boundary",
        frequency: false
      };
    }

    const targetTense =
      contextModel?.answerShape
        ?.temporalTense ||
      "neutral";

    let tense = "neutral";
    let tenseStrength = 0.0;
    let tenseCompatibility = 0.82;
    let tenseContradiction = false;

    if (
      vocab &&
      typeof vocab
        .getTemporalTenseCompatibility ===
        "function"
    ) {
      const info =
        vocab.getTemporalTenseCompatibility(
          tokenId,
          targetTense
        );

      tense =
        info.tense ||
        "neutral";

      tenseStrength =
        info.strength || 0.0;

      tenseCompatibility =
        Number.isFinite(
          info.compatibility
        )
          ? info.compatibility
          : 0.82;

      tenseContradiction =
        Boolean(
          info.contradiction
        );
    }

    return {
      ...temporal,
      tense,
      tenseStrength,
      tenseCompatibility,
      tenseContradiction
    };
  }

  getGeneratedTemporalTense(
    generated,
    contextModel
  ) {
    let strongest = null;

    for (const token of generated || []) {
      const info =
        this.temporalTokenEvidence(
          token,
          contextModel
        );

      if (
        info.tense === "neutral" ||
        info.tenseStrength < 0.60
      ) {
        continue;
      }

      if (
        !strongest ||
        info.tenseStrength >
          strongest.tenseStrength
      ) {
        strongest = info;
      }
    }

    return strongest;
  }

  temporalTenseAssessment(
    tokenId,
    generated,
    contextModel
  ) {
    const info =
      this.temporalTokenEvidence(
        tokenId,
        contextModel
      );

    let compatibility =
      info.tenseCompatibility;

    let contradiction =
      Boolean(
        info.tenseContradiction
      );

    const existing =
      this.getGeneratedTemporalTense(
        generated,
        contextModel
      );

    if (
      existing &&
      info.tense !== "neutral" &&
      info.tenseStrength >= 0.60 &&
      existing.tense !== info.tense
    ) {
      contradiction = true;
      compatibility = 0.0;
    }

    if (
      info.tense === "neutral" &&
      info.score >= 0.68
    ) {
      compatibility =
        Math.max(
          compatibility,
          0.86
        );
    }

    return {
      ...info,
      compatibility:
        Math.max(
          0,
          Math.min(
            1,
            compatibility
          )
        ),
      contradiction
    };
  }


  buildSemanticClauseGraph(
    cleanPromptTokens,
    questionProfile,
    answerContract
  ) {
    const nodes = [];
    const edges = [];

    const predicateToken =
      Number.isInteger(
        answerContract
          ?.relationVerbToken
      )
        ? answerContract
            .relationVerbToken
        : null;

    const predicateIndex =
      predicateToken === null
        ? -1
        : cleanPromptTokens
            .indexOf(predicateToken);

    const subjectSet =
      new Set(
        answerContract
          ?.subjectTokens || []
      );

    const targetSet =
      new Set(
        answerContract
          ?.targetTokens || []
      );

    let conditionOpen = false;
    let conditionMarkerToken = null;

    const structuralConditionWords =
      new Set([
        "if", "when", "unless",
        "while", "before", "after",
        "until", "since"
      ]);

    for (
      let index = 0;
      index < cleanPromptTokens.length;
      index++
    ) {
      const token =
        cleanPromptTokens[index];

      const type =
        this.getCanonicalType(
          token
        );

      const word =
        String(
          this.tokenWord(token) || ""
        ).toLowerCase();

      if (type === "Punct") {
        continue;
      }

      if (
        structuralConditionWords.has(
          word
        ) &&
        index !==
          questionProfile?.operatorIndex
      ) {
        conditionOpen = true;
        conditionMarkerToken = token;

        nodes.push({
          token,
          index,
          type,
          role: "condition_marker",
          condition: true
        });

        continue;
      }

      let role = "other";

      if (
        subjectSet.has(token)
      ) {
        role = "entity";
      } else if (
        token === predicateToken
      ) {
        role = "process";
      } else if (
        targetSet.has(token)
      ) {
        role = "target";
      } else if (type === "Verb") {
        role = "action";
      } else if (
        ["Noun", "ProperNoun", "Pronoun"]
          .includes(type)
      ) {
        role =
          predicateIndex >= 0 &&
          index > predicateIndex
            ? "argument"
            : "entity";
      } else if (type === "Adj") {
        role = "property";
      } else if (type === "Adv") {
        role = "modifier";
      } else if (type === "Num") {
        role = "quantity";
      } else if (
        ["Prep", "Conj"].includes(type)
      ) {
        role = "connector";
      } else if (
        ["Modal", "Aux"].includes(type)
      ) {
        role = "modality";
      }

      nodes.push({
        token,
        index,
        type,
        role,
        condition:
          conditionOpen
      });
    }

    const entityNodes =
      nodes.filter(
        node =>
          node.role === "entity"
      );

    const processNodes =
      nodes.filter(
        node =>
          ["process", "action"]
            .includes(node.role)
      );

    const targetNodes =
      nodes.filter(
        node =>
          ["target", "argument"]
            .includes(node.role)
      );

    const propertyNodes =
      nodes.filter(
        node =>
          node.role === "property"
      );

    const quantityNodes =
      nodes.filter(
        node =>
          node.role === "quantity"
      );

    const mainEntity =
      entityNodes[0] || null;

    const mainProcess =
      processNodes.find(
        node =>
          node.token ===
          predicateToken
      ) ||
      processNodes[0] ||
      null;

    for (const entity of entityNodes) {
      if (mainProcess) {
        edges.push({
          from: entity.token,
          to: mainProcess.token,
          type: "participant",
          structural: true
        });
      }
    }

    for (const target of targetNodes) {
      if (mainProcess) {
        edges.push({
          from: mainProcess.token,
          to: target.token,
          type: "argument",
          structural: true
        });
      }
    }

    for (const property of propertyNodes) {
      const parent =
        mainProcess?.token ??
        mainEntity?.token;

      if (Number.isInteger(parent)) {
        edges.push({
          from: parent,
          to: property.token,
          type: "property",
          structural: true
        });
      }
    }

    if (
      conditionMarkerToken !== null &&
      mainProcess
    ) {
      edges.push({
        from: conditionMarkerToken,
        to: mainProcess.token,
        type: "condition",
        structural: true
      });
    }

    const hasEntity =
      entityNodes.length > 0;

    const hasProcess =
      processNodes.length > 0;

    const hasTarget =
      targetNodes.length > 0;

    const hasState =
      propertyNodes.length > 0;

    const hasCondition =
      nodes.some(
        node => node.condition
      );

    const graphCompleteness =
      Math.max(
        0,
        Math.min(
          1,
          (hasEntity ? 0.28 : 0) +
          (hasProcess ? 0.30 : 0) +
          (
            hasTarget ||
            hasState
              ? 0.22
              : 0.08
          ) +
          (
            hasCondition
              ? 0.12
              : 0.08
          )
        )
      );

    return {
      version: 1,
      nodes,
      edges,
      mainEntityToken:
        mainEntity?.token ?? null,
      mainProcessToken:
        mainProcess?.token ?? null,
      entityTokens:
        entityNodes.map(
          node => node.token
        ),
      processTokens:
        processNodes.map(
          node => node.token
        ),
      targetTokens:
        targetNodes.map(
          node => node.token
        ),
      propertyTokens:
        propertyNodes.map(
          node => node.token
        ),
      quantityTokens:
        quantityNodes.map(
          node => node.token
        ),
      hasEntity,
      hasProcess,
      hasTarget,
      hasState,
      hasCondition,
      graphCompleteness
    };
  }

  routeSemanticFrame(
    cleanPromptTokens,
    questionProfile,
    answerContract,
    answerShape,
    eventFrame,
    queryFrame,
    clauseGraph
  ) {
    const kind =
      answerContract?.kind ||
      "general";

    const isRequest =
      Boolean(
        questionProfile
          ?.addressedRequest
      ) ||
      [
        "imperative",
        "factual_generation"
      ].includes(kind);

    let primaryFrame =
      "relation";

    if (isRequest) {
      primaryFrame = "request";
    } else if (
      kind === "definition"
    ) {
      primaryFrame =
        clauseGraph?.hasProcess
          ? "relation"
          : "definition";
    } else if (
      kind === "quantitative"
    ) {
      primaryFrame = "quantity";
    } else if (
      clauseGraph?.hasCondition &&
      kind === "effect"
    ) {
      primaryFrame = "condition";
    } else if (
      kind === "mechanism"
    ) {
      primaryFrame =
        clauseGraph?.hasProcess
          ? "process"
          : "entity";
    } else if (
      kind === "causal" ||
      kind === "temporal"
    ) {
      primaryFrame =
        clauseGraph?.hasProcess
          ? "event"
          : "state";
    } else if (
      kind === "boolean"
    ) {
      primaryFrame = "relation";
    } else if (
      clauseGraph?.hasProcess &&
      clauseGraph?.hasTarget
    ) {
      primaryFrame = "action";
    } else if (
      clauseGraph?.hasProcess
    ) {
      primaryFrame = "process";
    } else if (
      clauseGraph?.hasState
    ) {
      primaryFrame = "state";
    } else if (
      clauseGraph?.hasEntity
    ) {
      primaryFrame = "entity";
    } else {
      primaryFrame = "conversation";
    }

    let queryTarget =
      queryFrame?.requestedSlot ||
      "content";

    // HOW + a first-person action usually requests a procedure rather than a
    // description of how an external system operates. This is grammatical
    // perspective, not a topic rule.
    if (
      queryTarget === "mechanism" &&
      questionProfile
        ?.subjectPronounPerson === 1 &&
      clauseGraph?.hasProcess
    ) {
      queryTarget = "procedure";
    }

    const requiresProcessChain =
      [
        "mechanism",
        "procedure",
        "cause",
        "effect"
      ].includes(queryTarget);

    const requiresEntityAnchor =
      ![
        "clarification",
        "content",
        "execution"
      ].includes(queryTarget);

    const requiresCondition =
      primaryFrame === "condition";

    const requiresNumeric =
      queryTarget === "quantity";

    const frameConfidence =
      Math.max(
        0,
        Math.min(
          1,
          (
            clauseGraph
              ?.graphCompleteness ||
            0
          ) *
          0.72 +
          (
            queryFrame
              ?.eventCompleteness ||
            0
          ) *
          0.28
        )
      );

    return {
      version: 1,
      primaryFrame,
      queryTarget,
      missingRelation:
        queryTarget,
      requiredRole:
        queryFrame?.requiredRole ||
        "content",
      requiresProcessChain,
      requiresEntityAnchor,
      requiresCondition,
      requiresNumeric,
      frameConfidence,
      answerShape:
        answerShape?.kind ||
        "prose",
      tense:
        eventFrame?.tense ||
        "neutral"
    };
  }

  getSemanticFrameTypeCompatibility(
    candidateToken,
    contextModel
  ) {
    const type =
      this.getCanonicalType(
        candidateToken
      );

    const target =
      contextModel
        ?.semanticFrame
        ?.queryTarget ||
      "content";

    switch (target) {
      case "mechanism":
      case "procedure":
        if (type === "Verb") return 1.0;
        if (type === "Prep") return 0.84;
        if (["Noun", "ProperNoun"].includes(type)) return 0.68;
        if (type === "Adv") return 0.52;
        return 0.40;

      case "cause":
        if (["Prep", "Conj"].includes(type)) return 0.92;
        if (type === "Verb") return 0.82;
        if (["Noun", "ProperNoun"].includes(type)) return 0.70;
        return 0.42;

      case "effect":
        if (["Verb", "Adj"].includes(type)) return 0.92;
        if (["Noun", "ProperNoun"].includes(type)) return 0.72;
        return 0.44;

      case "time_condition":
        if (["Adv", "Prep", "Num"].includes(type)) return 0.90;
        if (["Noun", "ProperNoun"].includes(type)) return 0.68;
        return 0.40;

      case "location":
        if (["Prep", "Noun", "ProperNoun", "Adv"].includes(type)) return 0.88;
        return 0.40;

      case "quantity":
        if (type === "Num") return 1.0;
        if (["Noun", "Adj"].includes(type)) return 0.68;
        return 0.38;

      case "identity":
        if (["Noun", "ProperNoun", "Adj"].includes(type)) return 0.88;
        return 0.42;

      case "truth":
        if (["Aux", "Modal", "Adv", "Verb"].includes(type)) return 0.76;
        return 0.46;

      case "evaluation":
        if (["Adj", "Adv", "Modal"].includes(type)) return 0.82;
        return 0.48;

      default:
        return this.isContentType(type)
          ? 0.66
          : 0.52;
    }
  }

  getSemanticGraphCandidateFit(
    candidateToken,
    generated,
    contextModel,
    sentenceState,
    cachedProvenance = null,
    cachedQueryFit = null
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const provenance =
      cachedProvenance ||
      this.getPromptRootedProvenance(
        candidateToken,
        contextModel
      );

    const queryFit =
      Number.isFinite(
        cachedQueryFit
      )
        ? cachedQueryFit
        : this.getEventQueryCandidateFit(
            candidateToken,
            contextModel,
            sentenceState
          );

    const typeFit =
      this.getSemanticFrameTypeCompatibility(
        candidateToken,
        contextModel
      );

    let localEdgeSupport = 0.0;
    let seen = 0;

    for (
      let i =
        (generated?.length || 0) - 1;
      i >= 0 && seen < 3;
      i--
    ) {
      const prior =
        generated[i];

      if (
        !this.isContentType(
          this.getCanonicalType(
            prior
          )
        )
      ) {
        continue;
      }

      seen++;

      const pair =
        this.getWordAssociationEvidence(
          candidateToken,
          prior,
          contextModel
        );

      const trusted =
        Math.max(
          pair?.typedKnowledge || 0,
          pair?.staticKnowledge || 0,
          (
            pair?.collocationKnowledge ||
            0
          ) * 0.80,
          (
            pair?.embeddingEvidence ||
            0
          ) * 0.18
        );

      localEdgeSupport =
        Math.max(
          localEdgeSupport,
          clamp(trusted)
        );
    }

    const rootSupport =
      clamp(
        Math.max(
          provenance.grounding || 0,
          provenance.factual || 0,
          provenance.relevance || 0
        )
      );

    return clamp(
      queryFit * 0.30 +
      typeFit * 0.22 +
      rootSupport * 0.30 +
      localEdgeSupport * 0.18
    );
  }

  buildEventFrame(
    cleanPromptTokens,
    questionProfile,
    answerContract
  ) {
    const unique = values => {
      const out = [];
      for (const value of values || []) {
        if (Number.isInteger(value) && !out.includes(value)) out.push(value);
      }
      return out;
    };

    const predicateToken =
      Number.isInteger(answerContract?.relationVerbToken)
        ? answerContract.relationVerbToken
        : null;

    const subjectTokens =
      unique(answerContract?.subjectTokens || []);

    const objectTokens =
      unique(answerContract?.targetTokens || []);

    const temporal =
      this.classifyTemporalQuestion(
        cleanPromptTokens,
        questionProfile,
        answerContract
      );

    const modifiers = [];
    const conditionTokens = [];

    for (let i = 0; i < cleanPromptTokens.length; i++) {
      const token = cleanPromptTokens[i];
      if (token === questionProfile?.operatorToken) continue;
      if (token === questionProfile?.helperToken) continue;
      if (subjectTokens.includes(token) || objectTokens.includes(token) || token === predicateToken) continue;

      const type = this.getCanonicalType(token);
      if (["Adj", "Adv"].includes(type)) modifiers.push(token);
      if (["Prep", "Conj"].includes(type)) conditionTokens.push(token);
    }

    const rootTokens = unique([
      ...subjectTokens,
      ...(predicateToken !== null ? [predicateToken] : []),
      ...objectTokens
    ]);

    return {
      version: 1,
      subjectTokens,
      predicateToken,
      objectTokens,
      modifierTokens: unique(modifiers),
      conditionTokens: unique(conditionTokens),
      helperToken: questionProfile?.helperToken ?? null,
      tense: temporal?.tense || "neutral",
      modality:
        this.getCanonicalType(questionProfile?.helperToken) === "Modal"
          ? "modal"
          : "direct",
      rootTokens,
      hasPredicate: Number.isInteger(predicateToken),
      completeness:
        Math.max(
          0,
          Math.min(
            1,
            (subjectTokens.length ? 0.34 : 0) +
            (Number.isInteger(predicateToken) ? 0.36 : 0) +
            (objectTokens.length || modifiers.length ? 0.30 : 0)
          )
        )
    };
  }

  buildQueryFrame(
    answerContract,
    answerShape,
    eventFrame
  ) {
    const kind = answerContract?.kind || "general";
    const slotByKind = {
      temporal: "time_condition",
      causal: "cause",
      effect: "effect",
      mechanism: "mechanism",
      locative: "location",
      quantitative: "quantity",
      boolean: "truth",
      definition: "identity",
      evaluative: "evaluation",
      factual_generation: "content",
      imperative: "execution",
      clarification: "clarification",
      general: "content"
    };

    const requestedSlot = slotByKind[kind] || "content";

    return {
      version: 1,
      kind,
      requestedSlot,
      requiredRole:
        this.instructionRoleFromContract(answerContract),
      requiresEvent:
        !["clarification", "factual_generation"].includes(kind),
      requiresGrounding:
        Boolean(answerContract?.factualityRequired),
      requiresNumericEvidence:
        Boolean(answerContract?.requiresNumericEvidence),
      answerShape: answerShape?.kind || "prose",
      eventCompleteness: eventFrame?.completeness || 0
    };
  }

  getPromptRootedProvenance(
    candidateToken,
    contextModel
  ) {
    const clamp = value =>
      Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

    const type = this.getCanonicalType(candidateToken);

    // Function words are licensed by grammar, not by factual associations.
    if (!this.isContentType(type)) {
      return {
        grounding: 0.72,
        factual: 0.0,
        relevance: 0.58,
        policy: 0.50,
        typed: 0.0,
        explicit: 0.0,
        discovery: 0.0,
        exact: false,
        rootCount: 0
      };
    }

    const roots = [];
    const addRoot = token => {
      const canonical = this.canonicalTokenId(token);
      if (
        Number.isInteger(canonical) &&
        this.network?.vocab?.hasToken?.(canonical) &&
        !roots.includes(canonical)
      ) {
        roots.push(canonical);
      }
    };

    for (const token of contextModel?.eventFrame?.rootTokens || []) addRoot(token);
    for (const token of contextModel?.instructionContract?.sourceAnchorTokens || []) {
      addRoot(token);
      if (roots.length >= 12) break;
    }

    const candidate = this.canonicalTokenId(candidateToken);

    const cache = contextModel?.provenanceCache;
    if (cache instanceof Map && cache.has(candidate)) {
      return cache.get(candidate);
    }

    let exact = false;
    let typed = 0.0;
    let explicit = 0.0;
    let relevance = 0.0;
    let policy = 0.50;
    let discovery = 0.0;

    for (const root of roots) {
      if (root === candidate) {
        // Prompt identity is strong grounding, but it is not evidence that the
        // proposition in the prompt is factually true.
        exact = true;
        relevance = 1.0;
        continue;
      }

      const pair = this.getWordAssociationEvidence(candidate, root, contextModel);
      typed = Math.max(typed, pair.typedKnowledge || 0);
      explicit = Math.max(explicit, pair.corroboratedKnowledge || 0);
      relevance = Math.max(relevance, pair.embeddingEvidence || 0);
      policy = Math.max(policy, pair.learnedEvidence || 0.50);
      if (pair.discoveryOnly) {
        discovery = Math.max(discovery, pair.embeddingEvidence || 0);
      }
    }

    const role = contextModel?.queryFrame?.requiredRole || "content";
    const roleFit =
      this.network?.vocab?.getRoleCompatibility?.(candidateToken, role);

    const proposition =
      this.getTrustedPropositionEvidence(
        contextModel
          ?.eventFrame
          ?.subjectTokens?.[0],
        contextModel
          ?.eventFrame
          ?.predicateToken,
        candidateToken,
        role
      );

    const propositionEvidence =
      Number.isFinite(
        proposition?.score
      )
        ? clamp(
            Math.max(
              0,
              proposition.score - 0.50
            ) * 2.0
          )
        : 0.0;

    const roleEvidence =
      Number.isFinite(roleFit)
        ? clamp(Math.max(0, roleFit - 0.45) / 0.55)
        : 0.0;

    const memorySignal =
      this.getMemoryCandidateSignals(
        candidate,
        contextModel
      );

    relevance =
      Math.max(
        relevance,
        memorySignal.relevance *
          this.reasoningConfig.memoryContextGroundingWeight
      );

    const trustedMemoryTruth =
      memorySignal.truthSupport >=
        this.reasoningConfig.memoryTruthFloor
        ? memorySignal.truthSupport *
          memorySignal.authority *
          this.reasoningConfig.memoryTrustedTruthWeight
        : 0.0;

    const factual =
      clamp(
        Math.max(
          exact ? 1.0 : 0.0,
          typed,
          explicit,
          propositionEvidence * 0.88,
          trustedMemoryTruth
        )
      );

    const grounding =
      clamp(
        Math.max(
          exact ? 1.0 : 0.0,
          factual,
          roleEvidence * 0.72,
          memorySignal.relevance * 0.46
        )
      );

    const result = {
      grounding,
      factual,
      relevance,
      policy,
      typed,
      explicit,
      discovery,
      exact,
      memoryRelevance:
        memorySignal.relevance,
      memoryTruthSupport:
        memorySignal.truthSupport,
      memoryAuthority:
        memorySignal.authority,
      rootCount: roots.length
    };

    if (cache instanceof Map) {
      if (cache.size > 512) cache.clear();
      cache.set(candidate, result);
    }

    return result;
  }

  getEventQueryCandidateFit(
    candidateToken,
    contextModel,
    sentenceState = null
  ) {
    const clamp = value =>
      Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

    const frame = contextModel?.queryFrame;
    if (!frame) return 0.50;

    const type = this.getCanonicalType(candidateToken);
    const role = frame.requiredRole || "content";
    const roleFit =
      this.network?.vocab?.getRoleCompatibility?.(candidateToken, role);

    let fit = Number.isFinite(roleFit) ? clamp(roleFit) : 0.45;

    const markerIds = contextModel?.answerContract?.markerTokenIds || [];
    if (markerIds.includes(candidateToken)) fit = Math.max(fit, 0.90);

    switch (frame.requestedSlot) {
      case "time_condition": {
        const temporal = this.temporalTokenEvidence(candidateToken, contextModel);
        fit = Math.max(fit, temporal?.score || 0);
        break;
      }
      case "quantity":
        if (type === "Num") fit = 1.0;
        break;
      case "truth":
        if (["Aux", "Modal", "Adv"].includes(type)) fit = Math.max(fit, 0.62);
        break;
      case "cause":
      case "effect":
      case "mechanism":
      case "location":
      case "evaluation":
      case "identity":
      case "content":
      case "execution":
      case "clarification":
      default:
        break;
    }

    return clamp(fit);
  }

  isComparativeLike(
    tokenId,
    explicitType = null
  ) {
    const type =
      explicitType ||
      this.getCanonicalType(
        tokenId
      );

    if (
      !["Adj", "Adv"].includes(type)
    ) {
      return false;
    }

    const word =
      String(
        this.tokenWord(tokenId) || ""
      ).toLowerCase();

    const grammar =
      this.network
        ?.vocab
        ?.getGrammar?.(
          tokenId
        ) || {};

    const degree =
      String(
        grammar.degree || ""
      ).toLowerCase();

    if (
      degree === "comparative"
    ) {
      return true;
    }

    const lemma =
      String(
        grammar.lemma || ""
      ).toLowerCase();

    // Productive English comparative morphology. The lemma check avoids
    // treating ordinary -er adjectives/adverbs as comparative when metadata
    // says the surface and lemma are identical.
    if (
      word.length >= 4 &&
      word.endsWith("er") &&
      lemma &&
      lemma !== word
    ) {
      return true;
    }

    // Functional comparative determiners/adjectives may have no useful lemma.
    return (
      word === "more" ||
      word === "less"
    );
  }

  isPastLikeVerb(
    tokenId
  ) {
    if (
      this.getCanonicalType(
        tokenId
      ) !== "Verb"
    ) {
      return false;
    }

    const word =
      String(
        this.tokenWord(tokenId) || ""
      ).toLowerCase();

    const grammar =
      this.network
        ?.vocab
        ?.getGrammar?.(
          tokenId
        ) || {};

    const tense =
      String(
        grammar.tense || ""
      ).toLowerCase();

    if (
      tense === "past"
    ) {
      return true;
    }

    const lemma =
      String(
        grammar.lemma || ""
      ).toLowerCase();

    return Boolean(
      word.length >= 4 &&
      /(?:ied|ed)$/.test(word) &&
      lemma &&
      lemma !== word
    );
  }

  getRelationStructureCandidateBonus(
    candidateToken,
    contextModel,
    sentenceState,
    eventQueryFit = null
  ) {
    const query =
      contextModel?.queryFrame;

    if (
      !query ||
      !contextModel
        ?.questionProfile
        ?.isQuestion
    ) {
      return 0.0;
    }

    const type =
      this.getCanonicalType(
        candidateToken
      );

    const slot =
      query.requestedSlot ||
      "content";

    const marker =
      Boolean(
        contextModel
          ?.answerContract
          ?.markerTokenIds
          ?.includes(
            candidateToken
          )
      );

    const sourceFamily =
      this.isSourceAnchorFamily(
        candidateToken,
        contextModel
      );

    const fit =
      Number.isFinite(
        eventQueryFit
      )
        ? eventQueryFit
        : this.getEventQueryCandidateFit(
            candidateToken,
            contextModel,
            sentenceState
          );

    const relationOpen =
      Boolean(
        sentenceState
          ?.slotRelationOpened
      );

    let bonus = 0.0;

    switch (slot) {
      case "mechanism":
        if (marker) {
          bonus += 3.0;
        }

        if (
          type === "Verb" &&
          !sourceFamily
        ) {
          bonus +=
            2.4 +
            fit * 1.4;
        }

        if (
          relationOpen &&
          this.isContentType(type) &&
          !sourceFamily
        ) {
          bonus +=
            Math.max(
              0,
              fit - 0.34
            ) *
            2.7;
        }

        if (
          this.isComparativeLike(
            candidateToken,
            type
          ) &&
          !relationOpen &&
          fit < 0.72
        ) {
          bonus -= 1.8;
        }
        break;

      case "cause":
        if (marker) {
          bonus += 2.7;
        } else if (
          this.isContentType(type) &&
          !sourceFamily
        ) {
          bonus +=
            Math.max(
              0,
              fit - 0.36
            ) *
            2.2;
        }
        break;

      case "effect":
        if (
          ["Verb", "Adj", "Noun", "ProperNoun"]
            .includes(type) &&
          !sourceFamily
        ) {
          bonus +=
            Math.max(
              0,
              fit - 0.34
            ) *
            2.4;
        }
        break;

      case "location":
        if (marker) {
          bonus += 2.2;
        } else if (
          ["Noun", "ProperNoun", "Adv"]
            .includes(type) &&
          !sourceFamily
        ) {
          bonus +=
            Math.max(
              0,
              fit - 0.34
            ) *
            2.0;
        }
        break;

      case "time_condition":
        bonus +=
          Math.max(
            0,
            fit - 0.48
          ) *
          2.0;
        break;

      case "quantity":
        if (type === "Num") {
          bonus += 3.2;
        }
        break;

      default:
        break;
    }

    const targetTense =
      contextModel
        ?.eventFrame
        ?.tense ||
      "neutral";

    if (
      targetTense === "future" &&
      this.isPastLikeVerb(
        candidateToken
      )
    ) {
      // A future question may be answered in a timeless/general present, but
      // an unexplained past-tense predicate is usually a relation mismatch.
      bonus -=
        sentenceState
          ?.needsBaseVerb
          ? 4.2
          : 2.8;
    }

    return bonus;
  }

  computeEventQueryResolution(
    tokens,
    contextModel
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const query =
      contextModel?.queryFrame;

    const event =
      contextModel?.eventFrame;

    if (!query) {
      return {
        score: 0.50,
        slotFit: 0.50,
        slotCompleteness: 0.50,
        slotEvidenceCount: 0,
        markerSupport: 0.0,
        roleSupport: 0.50,
        grounding: 0.50,
        eventAttachment: 0.50,
        argumentCompleteness: 0.50,
        relationTopology: 0.50,
        relationStructureIntegrity: 0.50,
        mechanismTopology: 0.50,
        comparativeAttachment: 1.0,
        tenseContinuity: 1.0,
        contradictionIntegrity: 1.0,
        querySlotResolved: false,
        relationStructureResolved: false,
        resolved: false
      };
    }

    const roots =
      event?.rootTokens || [];

    const rootSet =
      new Set(
        roots.map(
          token =>
            this.canonicalTokenId(
              token
            )
        )
      );

    const markerIds =
      new Set(
        contextModel
          ?.answerContract
          ?.markerTokenIds || []
      );

    const requestedRole =
      query.requiredRole ||
      "content";

    const relationalSlots =
      new Set([
        "mechanism",
        "cause",
        "effect",
        "location",
        "time_condition",
        "evaluation"
      ]);

    let groundingSum = 0.0;
    let groundingCount = 0;
    let rootMatches = 0;
    let contradictions = 0;
    let temporalTense = null;

    let markerSupport = 0.0;
    let roleSupport = 0.0;
    let bestSlotEvidence = 0.0;
    let secondSlotEvidence = 0.0;
    let slotEvidenceCount = 0;
    let propositionSignals = 0;
    let truthMarkers = 0;
    let numericEvidence = 0.0;
    let temporalEvidence = 0.0;

    // Relation topology state.
    let generatedPredicateSeen = false;
    let generatedVerbCount = 0;
    let novelVerbEvidence = 0.0;
    let processRoleEvidence = 0.0;
    let markerSeen = false;
    let markerIndex = -1;
    let markerComplementEvidence = 0.0;
    let contentAfterPredicate = 0;
    let novelContentAfterPredicate = 0;
    let secondaryPredicateEvidence = 0.0;
    let nounRun = 0;
    let maxNounRun = 0;

    let comparativeCount = 0;
    let comparativeAttachmentSum = 0.0;

    let modalCount = 0;
    let pastLikeVerbCount = 0;

    let previousType = null;

    const registerSlotEvidence =
      evidence => {
        const value =
          clamp(evidence);

        if (value < 0.34) {
          return;
        }

        slotEvidenceCount++;

        if (
          value >
          bestSlotEvidence
        ) {
          secondSlotEvidence =
            bestSlotEvidence;

          bestSlotEvidence =
            value;
        } else if (
          value >
          secondSlotEvidence
        ) {
          secondSlotEvidence =
            value;
        }
      };

    const list =
      tokens || [];

    for (
      let tokenIndex = 0;
      tokenIndex < list.length;
      tokenIndex++
    ) {
      const token =
        list[tokenIndex];

      const type =
        this.getCanonicalType(
          token
        );

      if (type === "Punct") {
        previousType = type;
        continue;
      }

      const canonical =
        this.canonicalTokenId(
          token
        );

      const provenance =
        this.getPromptRootedProvenance(
          token,
          contextModel
        );

      const fit =
        this.getEventQueryCandidateFit(
          token,
          contextModel,
          null
        );

      const roleFitRaw =
        this.network
          ?.vocab
          ?.getRoleCompatibility?.(
            token,
            requestedRole
          );

      const roleFit =
        Number.isFinite(roleFitRaw)
          ? clamp(roleFitRaw)
          : 0.45;

      roleSupport =
        Math.max(
          roleSupport,
          roleFit
        );

      const isRoot =
        rootSet.has(
          canonical
        );

      const isPromptFamily =
        this.isSourceAnchorFamily(
          token,
          contextModel
        );

      const isMarker =
        markerIds.has(
          token
        );

      if (isMarker) {
        markerSeen = true;
        markerIndex = tokenIndex;

        markerSupport =
          Math.max(
            markerSupport,
            Math.max(
              0.72,
              fit
            )
          );
      }

      if (
        type === "Modal"
      ) {
        modalCount++;
      }

      if (
        type === "Verb"
      ) {
        generatedVerbCount++;

        if (
          this.isPastLikeVerb(
            token
          )
        ) {
          pastLikeVerbCount++;
        }

        if (
          !isPromptFamily
        ) {
          novelVerbEvidence =
            Math.max(
              novelVerbEvidence,
              fit *
              (
                0.50 +
                provenance.grounding *
                0.30 +
                roleFit * 0.20
              )
            );

          if (
            generatedPredicateSeen
          ) {
            secondaryPredicateEvidence =
              Math.max(
                secondaryPredicateEvidence,
                novelVerbEvidence
              );
          }
        }

        generatedPredicateSeen = true;
      }

      if (
        ["Noun", "ProperNoun"]
          .includes(type)
      ) {
        nounRun++;
        maxNounRun =
          Math.max(
            maxNounRun,
            nounRun
          );
      } else if (
        type !== "Det" &&
        type !== "Adj"
      ) {
        nounRun = 0;
      }

      if (
        this.isComparativeLike(
          token,
          type
        )
      ) {
        comparativeCount++;

        let attachment = 0.0;

        if (
          generatedPredicateSeen
        ) {
          if (
            [
              "Verb", "Aux",
              "Modal", "Adv", "Adj"
            ].includes(
              previousType
            )
          ) {
            attachment = 1.0;
          } else if (
            [
              "Noun",
              "ProperNoun",
              "Pronoun",
              "Num"
            ].includes(
              previousType
            )
          ) {
            // Object + comparative adverb can be valid ("runs X faster"), but
            // it is weaker than direct predicate/modifier attachment.
            attachment = 0.62;
          } else {
            attachment = 0.45;
          }
        }

        comparativeAttachmentSum +=
          attachment;
      }

      if (
        this.isContentType(type)
      ) {
        groundingSum +=
          provenance.grounding;

        groundingCount++;

        const eventRelevant =
          Math.max(
            provenance.grounding || 0,
            provenance.relevance || 0
          );

        if (
          generatedPredicateSeen &&
          type !== "Verb"
        ) {
          contentAfterPredicate++;
        }

        if (
          generatedPredicateSeen &&
          !isPromptFamily
        ) {
          novelContentAfterPredicate++;
        }

        if (
          markerSeen &&
          tokenIndex >
            markerIndex &&
          !isPromptFamily
        ) {
          markerComplementEvidence =
            Math.max(
              markerComplementEvidence,
              fit *
              (
                0.40 +
                eventRelevant *
                  0.35 +
                roleFit *
                  0.25
              )
            );
        }

        if (
          !isRoot &&
          !isPromptFamily
        ) {
          const evidence =
            fit *
            (
              0.42 +
              eventRelevant *
                0.36 +
              roleFit *
                0.22
            );

          registerSlotEvidence(
            evidence
          );

          processRoleEvidence =
            Math.max(
              processRoleEvidence,
              roleFit *
              (
                0.46 +
                fit * 0.34 +
                eventRelevant * 0.20
              )
            );
        }
      }

      if (
        isRoot ||
        provenance.factual >= 0.58 ||
        provenance.grounding >= 0.72
      ) {
        rootMatches++;
      }

      if (
        [
          "Verb",
          "Aux",
          "Modal"
        ].includes(type)
      ) {
        propositionSignals++;
      }

      if (isMarker) {
        truthMarkers++;
      }

      if (
        query.requestedSlot ===
          "quantity"
      ) {
        const word =
          String(
            this.tokenWord(token) ||
            ""
          );

        if (
          type === "Num" ||
          /^\d+(?:\.\d+)?$/
            .test(word)
        ) {
          numericEvidence = 1.0;

          registerSlotEvidence(
            Math.max(
              0.88,
              fit
            )
          );
        }
      }

      if (
        query.requestedSlot ===
          "time_condition"
      ) {
        const info =
          this.temporalTokenEvidence(
            token,
            contextModel
          );

        temporalEvidence =
          Math.max(
            temporalEvidence,
            info?.score || 0
          );

        if (
          (info?.score || 0) >=
            0.62
        ) {
          registerSlotEvidence(
            Math.max(
              fit,
              info.score
            )
          );
        }

        if (
          info?.tense &&
          info.tense !==
            "neutral" &&
          info.tenseStrength >=
            0.60
        ) {
          if (
            temporalTense &&
            temporalTense !==
              info.tense
          ) {
            contradictions++;
          }

          temporalTense =
            temporalTense ||
            info.tense;
        }

        if (
          info?.tenseContradiction
        ) {
          contradictions++;
        }
      }

      previousType = type;
    }

    const grounding =
      groundingCount
        ? clamp(
            groundingSum /
            groundingCount
          )
        : 0.42;

    const eventAttachment =
      roots.length
        ? clamp(
            rootMatches /
            Math.max(
              1,
              Math.min(
                roots.length,
                3
              )
            )
          )
        : 0.52;

    const comparativeAttachment =
      comparativeCount
        ? clamp(
            comparativeAttachmentSum /
            comparativeCount
          )
        : 1.0;

    const nounStackIntegrity =
      maxNounRun <= 3
        ? 1.0
        : clamp(
            1.0 -
            (
              maxNounRun -
              3
            ) *
            0.18
          );

    const dependencySupport =
      markerSeen
        ? clamp(
            markerComplementEvidence *
            0.82 +
            (
              markerComplementEvidence >
              0
                ? 0.18
                : 0
            )
          )
        : 1.0;

    const argumentCompleteness =
      clamp(
        (
          generatedPredicateSeen
            ? 0.28
            : 0.0
        ) +
        eventAttachment * 0.23 +
        (
          contentAfterPredicate > 0
            ? 0.19
            : 0.04
        ) +
        dependencySupport * 0.12 +
        comparativeAttachment * 0.08 +
        nounStackIntegrity * 0.10
      );

    const targetTense =
      event?.tense ||
      "neutral";

    let tenseContinuity = 1.0;

    if (
      targetTense === "future"
    ) {
      if (
        pastLikeVerbCount > 0 &&
        modalCount === 0
      ) {
        tenseContinuity = 0.36;
      } else if (
        modalCount > 0
      ) {
        tenseContinuity = 1.0;
      } else {
        // Timeless/general-present explanations remain valid answers to many
        // future mechanism/effect questions.
        tenseContinuity = 0.82;
      }
    } else if (
      targetTense === "past"
    ) {
      tenseContinuity =
        pastLikeVerbCount > 0
          ? 1.0
          : 0.72;
    } else if (
      targetTense === "present"
    ) {
      tenseContinuity =
        pastLikeVerbCount > 0
          ? 0.48
          : 0.92;
    }

    let relationTopology = 0.55;
    let mechanismTopology = 0.55;

    switch (
      query.requestedSlot
    ) {
      case "mechanism": {
        const markerTopology =
          markerSeen
            ? markerComplementEvidence
            : 0.0;

        const actionTopology =
          novelVerbEvidence *
          (
            novelContentAfterPredicate >
            0
              ? 1.0
              : 0.78
          );

        const roleTopology =
          processRoleEvidence *
          (
            markerSeen
              ? 0.78
              : novelVerbEvidence > 0.40
                ? 0.70
                : 0.28
          );

        mechanismTopology =
          clamp(
            Math.max(
              markerTopology,
              actionTopology,
              roleTopology
            )
          );

        relationTopology =
          mechanismTopology;
        break;
      }

      case "cause":
        relationTopology =
          clamp(
            Math.max(
              markerSeen
                ? markerComplementEvidence
                : 0.0,
              bestSlotEvidence *
              (
                secondaryPredicateEvidence >
                0.30
                  ? 0.92
                  : 0.62
              )
            )
          );
        break;

      case "effect":
        relationTopology =
          clamp(
            Math.max(
              secondaryPredicateEvidence,
              bestSlotEvidence *
              (
                novelContentAfterPredicate >
                0
                  ? 0.82
                  : 0.52
              )
            )
          );
        break;

      case "location":
        relationTopology =
          clamp(
            Math.max(
              markerSeen
                ? markerComplementEvidence
                : 0.0,
              bestSlotEvidence *
              0.82
            )
          );
        break;

      case "time_condition":
        relationTopology =
          clamp(
            Math.max(
              temporalEvidence,
              bestSlotEvidence *
              0.74
            )
          );
        break;

      case "quantity":
        relationTopology =
          clamp(
            numericEvidence *
            0.72 +
            bestSlotEvidence *
            0.28
          );
        break;

      case "truth":
        relationTopology =
          clamp(
            (
              propositionSignals > 0
                ? 0.58
                : 0.18
            ) +
            Math.min(
              0.30,
              truthMarkers *
              0.30
            )
          );
        break;

      default:
        relationTopology =
          clamp(
            bestSlotEvidence *
            0.56 +
            argumentCompleteness *
            0.44
          );
        break;
    }

    const relationStructureIntegrity =
      clamp(
        argumentCompleteness *
          0.28 +
        relationTopology *
          0.40 +
        comparativeAttachment *
          0.10 +
        tenseContinuity *
          0.16 +
        nounStackIntegrity *
          0.06
      );

    let slotCompleteness = 0.0;

    if (
      query.requestedSlot ===
        "truth"
    ) {
      const stance =
        truthMarkers > 0
          ? 1.0
          : propositionSignals > 0
            ? 0.58
            : 0.16;

      slotCompleteness =
        clamp(
          stance * 0.42 +
          bestSlotEvidence * 0.18 +
          eventAttachment * 0.14 +
          relationStructureIntegrity *
            0.26
        );
    } else if (
      query.requestedSlot ===
        "quantity"
    ) {
      slotCompleteness =
        clamp(
          numericEvidence * 0.50 +
          bestSlotEvidence * 0.18 +
          roleSupport * 0.07 +
          relationStructureIntegrity *
            0.25
        );
    } else if (
      query.requestedSlot ===
        "time_condition"
    ) {
      slotCompleteness =
        clamp(
          temporalEvidence * 0.44 +
          bestSlotEvidence * 0.15 +
          markerSupport * 0.08 +
          roleSupport * 0.06 +
          relationStructureIntegrity *
            0.27
        );
    } else if (
      query.requestedSlot ===
        "mechanism"
    ) {
      slotCompleteness =
        clamp(
          bestSlotEvidence * 0.18 +
          secondSlotEvidence * 0.05 +
          markerSupport * 0.05 +
          roleSupport * 0.06 +
          eventAttachment * 0.08 +
          argumentCompleteness * 0.16 +
          mechanismTopology * 0.42
        );
    } else if (
      relationalSlots.has(
        query.requestedSlot
      )
    ) {
      const evidenceBreadth =
        slotEvidenceCount >= 2
          ? 1.0
          : slotEvidenceCount === 1
            ? 0.62
            : 0.0;

      slotCompleteness =
        clamp(
          bestSlotEvidence * 0.27 +
          secondSlotEvidence * 0.07 +
          markerSupport * 0.08 +
          roleSupport * 0.07 +
          evidenceBreadth * 0.08 +
          argumentCompleteness * 0.15 +
          relationTopology * 0.28
        );
    } else {
      slotCompleteness =
        clamp(
          bestSlotEvidence * 0.47 +
          secondSlotEvidence * 0.09 +
          markerSupport * 0.06 +
          roleSupport * 0.08 +
          eventAttachment * 0.08 +
          relationStructureIntegrity *
            0.22
        );
    }

    const contradictionIntegrity =
      clamp(
        Math.pow(
          0.48,
          Math.min(
            3,
            contradictions
          )
        ) *
        (
          0.72 +
          tenseContinuity *
          0.28
        )
      );

    const eventNeed =
      query.requiresEvent
        ? eventAttachment
        : 0.72;

    let score =
      clamp(
        slotCompleteness * 0.44 +
        relationStructureIntegrity *
          0.24 +
        grounding * 0.12 +
        eventNeed * 0.12 +
        (event?.completeness || 0.55) *
          0.08
      );

    score *=
      contradictionIntegrity;

    if (
      contextModel
        ?.answerContract
        ?.underspecified
    ) {
      score =
        Math.max(
          score,
          query.requestedSlot ===
            "clarification"
            ? 0.70
            : 0.28
        );
    }

    const structureRequired =
      relationalSlots.has(
        query.requestedSlot
      ) ||
      [
        "quantity",
        "truth"
      ].includes(
        query.requestedSlot
      );

    const relationStructureResolved =
      !structureRequired ||
      relationStructureIntegrity >=
        this.reasoningConfig
          .relationStructureResolveFloor;

    const topologySatisfied =
      query.requestedSlot !==
        "mechanism" ||
      mechanismTopology >=
        this.reasoningConfig
          .mechanismTopologyFloor;

    const argumentsSatisfied =
      argumentCompleteness >=
      this.reasoningConfig
        .argumentCompletenessFloor;

    const tenseSatisfied =
      tenseContinuity >=
      this.reasoningConfig
        .tenseContinuityFloor;

    const querySlotResolved =
      slotCompleteness >=
        this.reasoningConfig
          .querySlotResolveFloor &&
      relationStructureResolved &&
      topologySatisfied &&
      argumentsSatisfied &&
      tenseSatisfied;

    return {
      score:
        clamp(score),
      slotFit:
        bestSlotEvidence,
      slotCompleteness,
      slotEvidenceCount,
      markerSupport,
      roleSupport,
      grounding,
      eventAttachment,
      argumentCompleteness,
      relationTopology,
      relationStructureIntegrity,
      mechanismTopology,
      comparativeAttachment,
      nounStackIntegrity,
      tenseContinuity,
      contradictionIntegrity,
      contradictions,
      querySlotResolved,
      relationStructureResolved,
      resolved:
        querySlotResolved &&
        score >=
          this.reasoningConfig
            .eventQueryResolveFloor
    };
  }

  computeSemanticCoreResolution(
    tokens,
    contextModel,
    eventQueryResolution = null
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const eventQuery =
      eventQueryResolution ||
      this.computeEventQueryResolution(
        tokens,
        contextModel
      );

    const frame =
      contextModel?.semanticFrame;

    const graph =
      contextModel?.clauseGraph;

    const content =
      (tokens || []).filter(
        token =>
          this.isContentType(
            this.getCanonicalType(
              token
            )
          )
      );

    if (!content.length) {
      return {
        frameFit: 0.0,
        queryResolution: 0.0,
        relationCompleteness: 0.0,
        grounding: 0.0,
        edgeContinuity: 0.0,
        rootedness: 0.0,
        processChain: 0.0,
        score: 0.0,
        criticalFloor: 0.0,
        resolved: false
      };
    }

    let groundingSum = 0.0;
    let rooted = 0;
    let novelProcessEvidence = 0.0;
    let trustedEdgeSum = 0.0;
    let trustedEdgeCount = 0;

    let previousContent = null;

    for (const token of content) {
      const provenance =
        this.getPromptRootedProvenance(
          token,
          contextModel
        );

      groundingSum +=
        provenance.grounding || 0;

      if (
        this.isSourceAnchorFamily(
          token,
          contextModel
        ) ||
        provenance.grounding >= 0.56 ||
        provenance.factual >= 0.50
      ) {
        rooted++;
      }

      const type =
        this.getCanonicalType(
          token
        );

      if (
        type === "Verb" &&
        !this.isSourceAnchorFamily(
          token,
          contextModel
        )
      ) {
        const fit =
          this.getSemanticGraphCandidateFit(
            token,
            [],
            contextModel,
            null,
            provenance,
            this.getEventQueryCandidateFit(
              token,
              contextModel,
              null
            )
          );

        novelProcessEvidence =
          Math.max(
            novelProcessEvidence,
            fit
          );
      }

      if (
        Number.isInteger(
          previousContent
        )
      ) {
        const pair =
          this.getWordAssociationEvidence(
            previousContent,
            token,
            contextModel
          );

        const trustedEdge =
          clamp(
            Math.max(
              pair?.typedKnowledge || 0,
              pair?.staticKnowledge || 0,
              (
                pair
                  ?.collocationKnowledge ||
                0
              ) * 0.82,
              (
                pair
                  ?.embeddingEvidence ||
                0
              ) * 0.16
            )
          );

        trustedEdgeSum +=
          trustedEdge;

        trustedEdgeCount++;
      }

      previousContent =
        token;
    }

    const grounding =
      clamp(
        groundingSum /
        Math.max(
          1,
          content.length
        )
      );

    const rootedness =
      clamp(
        rooted /
        Math.max(
          1,
          content.length
        )
      );

    const edgeContinuity =
      trustedEdgeCount
        ? clamp(
            trustedEdgeSum /
            trustedEdgeCount
          )
        : (
            content.length <= 1
              ? 0.72
              : 0.44
          );

    const queryResolution =
      clamp(
        eventQuery
          ?.slotCompleteness
      );

    const oldRelation =
      clamp(
        eventQuery
          ?.relationStructureIntegrity
      );

    const processChain =
      frame
        ?.requiresProcessChain
        ? clamp(
            Math.max(
              novelProcessEvidence,
              eventQuery
                ?.mechanismTopology ||
              0,
              eventQuery
                ?.relationTopology ||
              0
            ) *
            (
              0.62 +
              edgeContinuity *
                0.20 +
              rootedness *
                0.18
            )
          )
        : 1.0;

    const expectedEntity =
      frame
        ?.requiresEntityAnchor
        ? (
            graph?.hasEntity
              ? 1.0
              : 0.44
          )
        : 1.0;

    const expectedProcess =
      [
        "process",
        "action",
        "event",
        "condition"
      ].includes(
        frame?.primaryFrame
      )
        ? (
            graph?.hasProcess
              ? 1.0
              : 0.42
          )
        : 1.0;

    const frameFit =
      clamp(
        (
          frame?.frameConfidence ||
          0.50
        ) * 0.34 +
        expectedEntity * 0.18 +
        expectedProcess * 0.16 +
        rootedness * 0.18 +
        (
          eventQuery
            ?.eventAttachment ||
          0.50
        ) * 0.14
      );

    const relationCompleteness =
      clamp(
        oldRelation * 0.32 +
        edgeContinuity * 0.18 +
        rootedness * 0.18 +
        (
          eventQuery
            ?.argumentCompleteness ||
          0
        ) * 0.14 +
        processChain * 0.18
      );

    const semanticGraphScore =
      clamp(
        frameFit * 0.25 +
        queryResolution * 0.31 +
        relationCompleteness * 0.28 +
        grounding * 0.16
      );

    const criticalFloor =
      Math.min(
        frameFit,
        queryResolution,
        relationCompleteness,
        frame?.requiresEntityAnchor
          ? grounding
          : 1.0
      );

    return {
      frameKind:
        frame?.primaryFrame ||
        "relation",
      queryTarget:
        frame?.queryTarget ||
        contextModel
          ?.queryFrame
          ?.requestedSlot ||
        "content",
      frameFit,
      queryResolution,
      relationCompleteness,
      grounding,
      edgeContinuity,
      rootedness,
      processChain,
      score:
        semanticGraphScore,
      criticalFloor,
      resolved:
        frameFit >=
          this.reasoningConfig
            .semanticFrameResolveFloor &&
        queryResolution >=
          this.reasoningConfig
            .querySlotResolveFloor &&
        relationCompleteness >=
          this.reasoningConfig
            .semanticGraphResolveFloor
    };
  }

  computeEpistemicCoverage(
    tokens,
    contextModel,
    eventQuery = null
  ) {
    const clamp = value =>
      Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

    let factualSum = 0.0;
    let relevanceSum = 0.0;
    let count = 0;

    for (const token of tokens || []) {
      const type = this.getCanonicalType(token);
      if (!this.isContentType(type)) continue;
      const p = this.getPromptRootedProvenance(token, contextModel);
      factualSum += p.factual;
      relevanceSum += Math.max(p.relevance, p.grounding);
      count++;
    }

    const factual = count ? factualSum / count : 0.45;
    const relevance = count ? relevanceSum / count : 0.45;
    const resolution = eventQuery?.score ?? 0.50;

    const score = contextModel?.queryFrame?.requiresGrounding
      ? clamp(factual * 0.48 + relevance * 0.18 + resolution * 0.34)
      : clamp(relevance * 0.42 + resolution * 0.42 + 0.16);

    return {
      score,
      factual: clamp(factual),
      relevance: clamp(relevance),
      state:
        score >= 0.72 ? "supported" :
        score >= 0.50 ? "partial" :
        contextModel?.answerContract?.underspecified ? "ambiguous" :
        "unsupported"
    };
  }

  getCrossTurnLexicalFatigue(
    tokenId
  ) {
    const canonical =
      this.canonicalTokenId(
        tokenId
      );

    if (
      !Number.isInteger(canonical)
    ) {
      return 0.0;
    }

    return Math.max(
      0,
      Number(
        this.recentLexicalUsage
          .get(canonical)
      ) || 0
    );
  }

  getLexicalFatiguePenalty(
    tokenId,
    type,
    contextModel = null
  ) {
    if (
      !this.isContentType(type) ||
      this.isSourceAnchorFamily(
        tokenId,
        contextModel
      )
    ) {
      return 0.0;
    }

    const fatigue =
      this.getCrossTurnLexicalFatigue(
        tokenId
      );

    const excess =
      Math.max(
        0,
        fatigue -
        this.reasoningConfig
          .lexicalFatigueStart
      );

    if (excess <= 0) {
      return 0.0;
    }

    const modifierMultiplier =
      type === "Adv"
        ? this.reasoningConfig
            .lexicalFatigueModifierMultiplier
        : 1.0;

    return Math.min(
      5.5,
      excess *
      this.reasoningConfig
        .lexicalFatiguePenalty *
      modifierMultiplier
    );
  }

  recordResponseLexicalUsage(
    tokens
  ) {
    const decay =
      this.reasoningConfig
        .lexicalFatigueDecay;

    for (
      const [
        token,
        value
      ] of
        Array.from(
          this.recentLexicalUsage
            .entries()
        )
    ) {
      const next =
        value * decay;

      if (next < 0.08) {
        this.recentLexicalUsage
          .delete(token);
      } else {
        this.recentLexicalUsage
          .set(token, next);
      }
    }

    const unique =
      new Set();

    for (const raw of tokens || []) {
      const token =
        this.canonicalTokenId(raw);

      if (
        !Number.isInteger(token)
      ) {
        continue;
      }

      const type =
        this.getCanonicalType(
          token
        );

      if (
        !this.isContentType(type)
      ) {
        continue;
      }

      unique.add(token);
    }

    for (const token of unique) {
      this.recentLexicalUsage.set(
        token,
        Math.min(
          4.0,
          (
            this.recentLexicalUsage
              .get(token) ||
            0
          ) +
          1.0
        )
      );
    }

    if (
      this.recentLexicalUsage.size >
      this.reasoningConfig
        .lexicalFatigueLimit
    ) {
      const keep =
        Array.from(
          this.recentLexicalUsage
            .entries()
        )
          .sort(
            (a, b) =>
              b[1] - a[1]
          )
          .slice(
            0,
            this.reasoningConfig
              .lexicalFatigueLimit
          );

      this.recentLexicalUsage
        .clear();

      for (
        const [
          token,
          value
        ] of keep
      ) {
        this.recentLexicalUsage
          .set(token, value);
      }
    }

    this.lexicalUsageTurns++;
  }

  getModifierInformativeness(
    candidateToken,
    contextModel,
    sentenceState,
    provenance = null,
    queryFit = null
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const type =
      this.getCanonicalType(
        candidateToken
      );

    if (type !== "Adv") {
      return 1.0;
    }

    if (
      this.isSourceAnchorFamily(
        candidateToken,
        contextModel
      )
    ) {
      return 0.92;
    }

    const slot =
      contextModel
        ?.queryFrame
        ?.requestedSlot ||
      "content";

    const role =
      contextModel
        ?.queryFrame
        ?.requiredRole ||
      "content";

    const p =
      provenance ||
      this.getPromptRootedProvenance(
        candidateToken,
        contextModel
      );

    const fit =
      Number.isFinite(queryFit)
        ? queryFit
        : this.getEventQueryCandidateFit(
            candidateToken,
            contextModel,
            sentenceState
          );

    const roleFitRaw =
      this.network
        ?.vocab
        ?.getRoleCompatibility?.(
          candidateToken,
          role
        );

    const roleFit =
      Number.isFinite(roleFitRaw)
        ? clamp(roleFitRaw)
        : 0.35;

    let temporal =
      0.0;

    if (
      slot === "time_condition"
    ) {
      temporal =
        clamp(
          this.temporalTokenEvidence(
            candidateToken,
            contextModel
          )?.score
        );
    }

    // Adverbs are useful when they actually fill the requested role, are
    // prompt-rooted, or carry strong temporal evidence. Generic modifier
    // flexibility alone is not semantic information.
    let informativeness =
      clamp(
        roleFit * 0.34 +
        fit * 0.24 +
        p.grounding * 0.14 +
        p.factual * 0.18 +
        temporal * 0.10
      );

    const relationalSlot =
      new Set([
        "mechanism",
        "cause",
        "effect",
        "location",
        "quantity",
        "time_condition"
      ]).has(slot);

    if (
      relationalSlot &&
      roleFit < 0.50 &&
      p.factual < 0.28 &&
      temporal < 0.58
    ) {
      informativeness =
        Math.min(
          informativeness,
          0.34
        );
    }

    return informativeness;
  }

  initMemorySystem() {
    if (!this.reasoningConfig.memoryEnabled) return false;

    const Store = globalThis?.VilotMemoryStore;
    const Retrieval = globalThis?.VilotMemoryRetrieval;

    if (
      typeof Store !== "function" ||
      typeof Retrieval !== "function"
    ) {
      return false;
    }

    try {
      this.memoryStore =
        new Store({
          dbName: "vilotni15_memory_v1"
        });

      this.memoryRetrieval =
        new Retrieval(
          this.memoryStore,
          {
            topWorking: 4,
            topEpisodes: 4,
            topSemantic: 6,
            topProcedural: 3,
            maxCandidateTokens:
              this.reasoningConfig.memoryCandidateLimit
          }
        );

      this.memoryReady =
        Promise.resolve(
          this.memoryStore.ready
        )
          .then(() => true)
          .catch(() => false);

      return true;
    } catch (_) {
      this.memoryStore = null;
      this.memoryRetrieval = null;
      this.memoryReady = Promise.resolve(false);
      return false;
    }
  }

  async ensureMemoryReady() {
    if (
      !this.memoryStore ||
      !this.memoryRetrieval
    ) {
      return false;
    }

    try {
      return Boolean(await this.memoryReady);
    } catch (_) {
      return false;
    }
  }

  async prepareMemoryContext(
    cleanPromptTokens,
    contextModel
  ) {
    contextModel.memoryBundle = null;
    contextModel.memoryRecoveryHint = null;

    if (!(await this.ensureMemoryReady())) {
      return null;
    }

    try {
      const bundle =
        this.memoryRetrieval.retrieve(
          contextModel,
          cleanPromptTokens
        );

      contextModel.memoryBundle = bundle;
      contextModel.memoryRecoveryHint =
        bundle?.recoveryHint || null;

      contextModel?.provenanceCache?.clear?.();
      contextModel?.semanticGraphCandidateCache?.clear?.();

      return bundle;
    } catch (_) {
      contextModel.memoryBundle = null;
      return null;
    }
  }

  getMemoryCandidateSignals(
    candidateToken,
    contextModel
  ) {
    if (
      !this.memoryRetrieval ||
      !contextModel?.memoryBundle
    ) {
      return {
        relevance: 0.0,
        truthSupport: 0.0,
        authority: 0.0,
        bonus: 0.0
      };
    }

    const signal =
      this.memoryRetrieval.getCandidateSignals(
        candidateToken,
        contextModel.memoryBundle
      );

    const relevance =
      Math.max(
        0,
        Math.min(
          1,
          signal?.relevance || 0
        )
      );

    const truthSupport =
      Math.max(
        0,
        Math.min(
          1,
          signal?.truthSupport || 0
        )
      );

    const authority =
      Math.max(
        0,
        Math.min(
          1,
          signal?.authority || 0
        )
      );

    const bonus =
      relevance *
        this.reasoningConfig.memoryCandidateWeight +
      truthSupport *
        authority *
        0.52;

    return {
      relevance,
      truthSupport,
      authority,
      bonus:
        Math.max(
          0,
          Math.min(
            1.75,
            bonus
          )
        )
    };
  }

  getMemoryRecoveryHint(
    contextModel
  ) {
    const hint =
      contextModel?.memoryRecoveryHint;

    if (
      !hint ||
      hint.score <
        this.reasoningConfig.memoryProcedureStrongHint ||
      hint.successRate <
        this.reasoningConfig.memoryProcedureStrongSuccess
    ) {
      return null;
    }

    return hint;
  }

  memoryMistakeTags(
    diagnostics
  ) {
    const tags = [];
    const add = tag => {
      if (!tags.includes(tag)) {
        tags.push(tag);
      }
    };

    if ((diagnostics?.independentActFulfillment || 0) < 0.50) {
      add("answer_act");
    }

    if (diagnostics?.answerCommitmentResolved === false) {
      add("commitment");
    }

    if ((diagnostics?.independentPropositionCompleteness || 0) < 0.50) {
      add("proposition");
    }

    if ((diagnostics?.independentPromptRelationMatch || 0) < 0.50) {
      add("relation");
    }

    if ((diagnostics?.independentGrounding || 0) < 0.44) {
      add("grounding");
    }

    if ((diagnostics?.independentLanguageIntegrity || 0) < 0.58) {
      add("language");
    }

    if (
      (diagnostics?.selfWhyVerifierGap || 0) >=
      this.reasoningConfig.selfWhyMismatchGap
    ) {
      add("self_why_overconfidence");
    }

    return tags;
  }

  commitMemoryTurn(
    cleanPromptTokens,
    winner,
    diagnostics,
    contextModel,
    metadata = null
  ) {
    if (!this.memoryStore) return;

    try {
      const userTokens =
        (
          contextModel?.promptContentTokens ||
          cleanPromptTokens ||
          []
        )
          .filter(Number.isInteger)
          .slice(
            0,
            this.reasoningConfig.memoryEpisodePromptLimit
          );

      const assistantTokens =
        (winner?.tokens || [])
          .filter(Number.isInteger)
          .slice(
            0,
            this.reasoningConfig.memoryEpisodeResponseLimit
          );

      const frame =
        contextModel?.semanticFrame?.primaryFrame ||
        "relation";

      const queryTarget =
        contextModel?.semanticFrame?.queryTarget ||
        contextModel?.queryFrame?.requestedSlot ||
        "content";

      const answerAct =
        diagnostics?.independentRequiredAct ||
        this.getIndependentRequiredAnswerAct(
          contextModel
        );

      const resolved =
        Boolean(diagnostics?.independentResolved);

      const confidence =
        Number(diagnostics?.final) || 0;

      const strategy =
        metadata?.neuralRetryUsed
          ? "neural_retry"
          : (
              (metadata?.refinementRounds || 0) > 0
                ? "cpu_expand"
                : "direct"
            );

      this.memoryStore.addWorkingTurn({
        userTokens,
        assistantTokens,
        frame,
        queryTarget,
        answerAct,
        resolved
      });

      this.memoryStore.addEpisode({
        promptTokens: userTokens,
        responseTokens: assistantTokens,
        frame,
        queryTarget,
        answerAct,
        strategy,
        confidence,
        verifier:
          diagnostics?.independentVerifier || 0,
        actFulfillment:
          diagnostics?.independentActFulfillment || 0,
        propositionCompleteness:
          diagnostics?.independentPropositionCompleteness || 0,
        relationMatch:
          diagnostics?.independentPromptRelationMatch || 0,
        grounding:
          diagnostics?.independentGrounding || 0,
        language:
          diagnostics?.independentLanguageIntegrity || 0,
        resolved,
        mistakeTags:
          this.memoryMistakeTags(diagnostics),
        usefulness:
          Math.max(
            0,
            Math.min(
              1,
              (diagnostics?.independentVerifier || 0) * 0.72 +
              confidence * 0.28
            )
          )
      });

      // Only user assertions become semantic memory, and they remain
      // low-authority until separately verified.
      if (
        !contextModel?.questionProfile?.isQuestion &&
        !contextModel?.questionProfile?.addressedRequest &&
        !contextModel?.answerContract?.underspecified &&
        contextModel?.eventFrame?.subjectTokens?.length &&
        Number.isInteger(
          contextModel?.eventFrame?.predicateToken
        )
      ) {
        this.memoryStore.upsertSemantic({
          subjectTokens:
            contextModel.eventFrame.subjectTokens,
          predicateToken:
            contextModel.eventFrame.predicateToken,
          objectTokens:
            contextModel.eventFrame.objectTokens,
          modifierTokens:
            contextModel.eventFrame.modifierTokens,
          frame,
          source: "user_stated",
          trust: 0.22,
          relevance: 0.88
        });
      }

      const success =
        resolved &&
        confidence >=
          this.reasoningConfig.answerAcceptanceConfidence;

      const reward =
        Math.max(
          -1,
          Math.min(
            1,
            (diagnostics?.independentVerifier || 0) * 2.0 -
            1.0
          )
        );

      // Procedural memory stores HOW a similar reasoning state was repaired,
      // never generated factual content.
      this.memoryStore.reinforceProcedure({
        frame,
        queryTarget,
        strategy,
        success,
        reward,
        notes:
          this.memoryMistakeTags(diagnostics)
      });
    } catch (_) {
      // Memory must never block response generation.
    }
  }

  getSelfWhyCandidate(
    candidateToken,
    generated,
    contextModel,
    sentenceState,
    cachedProvenance = null,
    cachedQueryFit = null,
    cachedSemanticGraphFit = null,
    humanTransition = null
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    if (!this.reasoningConfig.selfWhyEnabled) {
      return {
        score: 0.50,
        promptFit: 0.50,
        relationFit: 0.50,
        evidenceSupport: 0.50,
        dependencySupport: 0.50,
        grounding: 0.50,
        novelty: 0.50,
        contradictionRisk: 0.0,
        answerProgress: 0.50
      };
    }

    const type =
      this.getCanonicalType(candidateToken);

    const provenance =
      cachedProvenance ||
      this.getPromptRootedProvenance(
        candidateToken,
        contextModel
      );

    const queryFit =
      Number.isFinite(cachedQueryFit)
        ? cachedQueryFit
        : this.getEventQueryCandidateFit(
            candidateToken,
            contextModel,
            sentenceState
          );

    const semanticGraphFit =
      Number.isFinite(cachedSemanticGraphFit)
        ? cachedSemanticGraphFit
        : this.getSemanticGraphCandidateFit(
            candidateToken,
            generated,
            contextModel,
            sentenceState,
            provenance,
            queryFit
          );

    const role =
      contextModel?.queryFrame?.requiredRole ||
      contextModel?.semanticFrame?.requiredRole ||
      "content";

    const roleFitRaw =
      this.network?.vocab?.getRoleCompatibility?.(
        candidateToken,
        role
      );

    const roleFit =
      Number.isFinite(roleFitRaw)
        ? clamp(roleFitRaw)
        : 0.45;

    const exact =
      this.isSourceAnchorFamily(
        candidateToken,
        contextModel
      );

    const promptFit =
      clamp(
        Math.max(
          exact ? 0.92 : 0.0,
          provenance?.relevance || 0,
          (provenance?.grounding || 0) * 0.82
        )
      );

    const relationFit =
      clamp(
        queryFit * 0.38 +
        semanticGraphFit * 0.40 +
        roleFit * 0.22
      );

    const trusted =
      this.getTrustedPropositionEvidence(
        contextModel?.eventFrame?.subjectTokens?.[0] ??
          contextModel?.answerContract?.subjectTokens?.[0],
        contextModel?.eventFrame?.predicateToken ??
          contextModel?.answerContract?.relationVerbToken,
        candidateToken,
        role
      );

    const factualRequired =
      Boolean(
        contextModel?.answerPlan?.factualityRequired
      );

    const memorySignal =
      this.getMemoryCandidateSignals(
        candidateToken,
        contextModel
      );

    const memoryEvidence =
      memorySignal.truthSupport >=
        this.reasoningConfig.memoryTruthFloor
        ? memorySignal.truthSupport *
          memorySignal.authority
        : 0.0;

    const evidenceSupport =
      factualRequired
        ? clamp(
            (provenance?.factual || 0) * 0.40 +
            (trusted?.score || 0) * 0.34 +
            (trusted?.typedRelation || 0) * 0.10 +
            memoryEvidence * 0.16
          )
        : clamp(
            0.58 +
            (provenance?.grounding || 0) * 0.18 +
            semanticGraphFit * 0.16 +
            roleFit * 0.08
          );

    let dependencySupport =
      Number.isFinite(humanTransition?.quality)
        ? clamp(humanTransition.quality)
        : 0.62;

    if (sentenceState?.needsBaseVerb) {
      dependencySupport =
        ["Verb", "Aux", "Modal"].includes(type)
          ? Math.max(dependencySupport, 0.90)
          : Math.min(dependencySupport, 0.34);
    }

    if (sentenceState?.needsObjectAfterPrep) {
      dependencySupport =
        ["Det", "Noun", "ProperNoun", "Pronoun", "Num", "Adj"].includes(type)
          ? Math.max(dependencySupport, 0.86)
          : Math.min(dependencySupport, 0.36);
    }

    if (sentenceState?.needsNominalAfterDet) {
      dependencySupport =
        ["Adj", "Noun", "ProperNoun", "Num"].includes(type)
          ? Math.max(dependencySupport, 0.90)
          : Math.min(dependencySupport, 0.32);
    }

    const grounding =
      clamp(provenance?.grounding);

    const alreadyUsed =
      generated?.includes(candidateToken);

    const familyUsed =
      this.isContentType(type)
        ? this.countMorphologicalFamilyInCurrentClause(
            candidateToken,
            generated || []
          ) > 0
        : false;

    const novelty =
      alreadyUsed
        ? 0.02
        : familyUsed
          ? 0.18
          : 1.0;

    const slotProgress =
      clamp(
        this.getLiveAnswerSlotProgress(
          sentenceState,
          contextModel
        )?.relation
      );

    const slotNeed =
      contextModel?.questionProfile?.isQuestion
        ? 1.0 - slotProgress
        : 0.45;

    const answerProgress =
      clamp(
        relationFit * 0.54 +
        semanticGraphFit * 0.30 +
        queryFit * 0.16
      ) *
      (
        0.62 +
        slotNeed * 0.38
      );

    let contradictionRisk = 0.0;

    if (
      contextModel?.eventFrame?.tense === "future" &&
      this.isPastLikeVerb(candidateToken)
    ) {
      contradictionRisk =
        Math.max(
          contradictionRisk,
          0.78
        );
    }

    if (
      this.isContentType(type) &&
      relationFit < 0.28 &&
      promptFit < 0.34
    ) {
      contradictionRisk =
        Math.max(
          contradictionRisk,
          0.62
        );
    }

    const failedPathPenalty =
      this.getFailedTrajectorySuppression(
        candidateToken,
        generated?.length
          ? generated[generated.length - 1]
          : null,
        contextModel
      );

    if (failedPathPenalty > 0) {
      contradictionRisk =
        Math.max(
          contradictionRisk,
          clamp(failedPathPenalty / 6.0)
        );
    }

    const w =
      this.reasoningConfig.selfWhyWeights;

    const positive =
      promptFit * w.promptFit +
      relationFit * w.relationFit +
      evidenceSupport * w.evidenceSupport +
      dependencySupport * w.dependencySupport +
      grounding * w.grounding +
      novelty * w.novelty +
      answerProgress * w.answerProgress;

    return {
      score:
        clamp(
          positive -
          contradictionRisk * 0.18
        ),
      promptFit,
      relationFit,
      evidenceSupport,
      dependencySupport,
      grounding,
      novelty,
      contradictionRisk,
      answerProgress
    };
  }

  getSelfWhyClause(
    tokens,
    contextModel,
    sentenceState = null,
    eventQueryResolution = null,
    semanticCoreResolution = null
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const eventQuery =
      eventQueryResolution ||
      this.computeEventQueryResolution(
        tokens,
        contextModel
      );

    const semanticCore =
      semanticCoreResolution ||
      this.computeSemanticCoreResolution(
        tokens,
        contextModel,
        eventQuery
      );

    const content =
      (tokens || []).filter(
        token =>
          this.isContentType(
            this.getCanonicalType(token)
          )
      );

    if (!content.length) {
      return {
        score: 0.0,
        promptFit: 0.0,
        relationFit: 0.0,
        evidenceSupport: 0.0,
        dependencySupport: 0.0,
        grounding: 0.0,
        novelty: 0.0,
        contradictionRisk: 1.0,
        answerProgress: 0.0
      };
    }

    let promptSum = 0.0;
    let evidenceSum = 0.0;
    let groundingSum = 0.0;
    const stems = new Set();

    const role =
      contextModel?.queryFrame?.requiredRole ||
      "content";

    const factualRequired =
      Boolean(
        contextModel?.answerPlan?.factualityRequired
      );

    for (const token of content) {
      const provenance =
        this.getPromptRootedProvenance(
          token,
          contextModel
        );

      promptSum +=
        Math.max(
          provenance?.relevance || 0,
          (provenance?.grounding || 0) * 0.82
        );

      const trusted =
        this.getTrustedPropositionEvidence(
          contextModel?.eventFrame?.subjectTokens?.[0] ??
            contextModel?.answerContract?.subjectTokens?.[0],
          contextModel?.eventFrame?.predicateToken ??
            contextModel?.answerContract?.relationVerbToken,
          token,
          role
        );

      evidenceSum +=
        factualRequired
          ? clamp(
              (provenance?.factual || 0) * 0.52 +
              (trusted?.score || 0) * 0.48
            )
          : clamp(
              0.62 +
              (provenance?.grounding || 0) * 0.20 +
              (trusted?.score || 0) * 0.18
            );

      groundingSum +=
        provenance?.grounding || 0;

      const word =
        String(
          this.tokenWord(token) || ""
        ).toLowerCase();

      stems.add(
        this.stemWord(word) || word
      );
    }

    const promptFit =
      clamp(promptSum / content.length);

    const relationFit =
      clamp(
        semanticCore?.relationCompleteness
      );

    const evidenceSupport =
      clamp(evidenceSum / content.length);

    const dependencySupport =
      (
        sentenceState?.needsBaseVerb ||
        sentenceState?.needsObjectAfterPrep ||
        sentenceState?.needsNominalAfterDet
      )
        ? 0.30
        : clamp(
            eventQuery?.argumentCompleteness ??
            0.72
          );

    const grounding =
      clamp(
        groundingSum / content.length
      );

    const novelty =
      clamp(
        stems.size /
        Math.max(1, content.length)
      );

    const contradictionRisk =
      clamp(
        1.0 -
        (
          eventQuery?.contradictionIntegrity ??
          0.50
        )
      );

    const answerProgress =
      clamp(
        semanticCore?.queryResolution
      );

    const w =
      this.reasoningConfig.selfWhyWeights;

    const positive =
      promptFit * w.promptFit +
      relationFit * w.relationFit +
      evidenceSupport * w.evidenceSupport +
      dependencySupport * w.dependencySupport +
      grounding * w.grounding +
      novelty * w.novelty +
      answerProgress * w.answerProgress;

    return {
      score:
        clamp(
          positive -
          contradictionRisk * 0.18
        ),
      promptFit,
      relationFit,
      evidenceSupport,
      dependencySupport,
      grounding,
      novelty,
      contradictionRisk,
      answerProgress
    };
  }

  getSelfWhyVerifierDisagreement(
    diagnostics
  ) {
    const selfWhy =
      Number(
        diagnostics?.selfWhyClauseScore
      );

    const independent =
      Number(
        diagnostics?.independentVerifier
      );

    if (
      !Number.isFinite(selfWhy) ||
      !Number.isFinite(independent)
    ) {
      return 0.0;
    }

    return Math.max(
      0,
      Math.min(
        1,
        selfWhy - independent
      )
    );
  }

  getRLPolicyFeatures(
    candidateToken,
    generated,
    contextModel,
    sentenceState,
    rawLogit = 0,
    cachedProvenance = null,
    cachedQueryFit = null
  ) {
    const clamp = value =>
      Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

    const type = this.getCanonicalType(candidateToken);
    const provenance = cachedProvenance || this.getPromptRootedProvenance(candidateToken, contextModel);
    const queryFit = Number.isFinite(cachedQueryFit)
      ? cachedQueryFit
      : this.getEventQueryCandidateFit(candidateToken, contextModel, sentenceState);
    const routed =
      this.getRoutedSemanticScore(
        candidateToken,
        contextModel,
        sentenceState
      );

    const semanticGraphFit =
      this.getSemanticGraphCandidateFit(
        candidateToken,
        generated,
        contextModel,
        sentenceState,
        provenance,
        queryFit
      );

    const role =
      contextModel
        ?.queryFrame
        ?.requiredRole ||
      "content";
    const roleFit = this.network?.vocab?.getRoleCompatibility?.(candidateToken, role);
    const exact = this.isSourceAnchorFamily(candidateToken, contextModel) ? 1.0 : 0.0;
    const progress =
      Math.min(
        1,
        (generated?.length || 0) /
        18.0
      );

    const logit =
      1.0 /
      (
        1.0 +
        Math.exp(
          -Math.max(
            -8,
            Math.min(8, rawLogit)
          ) /
          2.5
        )
      );

    const crossTurnFatigue =
      this.getCrossTurnLexicalFatigue(
        candidateToken
      );

    const crossTurnFreshness =
      clamp(
        1.0 -
        Math.max(
          0,
          crossTurnFatigue -
          this.reasoningConfig
            .lexicalFatigueStart
        ) /
        2.2
      );

    const inputSignals = this.getUserPredictionSignals(
      candidateToken,
      generated?.length ? generated[generated.length - 1] : null,
      contextModel
    );

    const selfWhy =
      this.getSelfWhyCandidate(
        candidateToken,
        generated,
        contextModel,
        sentenceState,
        provenance,
        queryFit,
        semanticGraphFit,
        null
      );
    const userSequenceFit = clamp(
      inputSignals.tokenTransition * 0.42 +
      inputSignals.posTransition * 0.58
    );

    return new Float32Array([
      1.0,
      logit,
      this.isContentType(type) ? 1.0 : 0.0,
      ["Aux", "Modal", "Prep", "Conj", "Det"].includes(type) ? 1.0 : 0.0,
      exact,
      provenance.grounding,
      provenance.factual,
      provenance.relevance,
      provenance.policy,
      queryFit,
      Number.isFinite(roleFit) ? clamp(roleFit) : 0.45,
      clamp((routed + 1.0) * 0.5),
      sentenceState?.subjectReady ? 1.0 : 0.0,
      sentenceState?.predicateReady ? 1.0 : 0.0,
      progress,
      (
        generated?.includes(
          candidateToken
        )
          ? 0.0
          : 1.0
      ) *
      crossTurnFreshness,
      semanticGraphFit,
      this.getSemanticFrameTypeCompatibility(
        candidateToken,
        contextModel
      ),
      contextModel
        ?.semanticFrame
        ?.requiresProcessChain
        ? 1.0
        : 0.0,
      contextModel
        ?.semanticFrame
        ?.frameConfidence || 0.50,
      userSequenceFit,
      inputSignals.carryover,
      inputSignals.futureContinuity,
      inputSignals.predictorConfidence,
      selfWhy.score,
      selfWhy.evidenceSupport,
      selfWhy.answerProgress
    ]);
  }

  getRLPolicyValue(features) {
    if (!features || !this.rlPolicyWeights) return 0.0;
    let value = 0.0;
    const n = Math.min(features.length, this.rlPolicyWeights.length);
    for (let i = 0; i < n; i++) value += features[i] * this.rlPolicyWeights[i];
    return Math.max(-3.0, Math.min(3.0, value));
  }

  trainRLPolicy(
    reward,
    trajectory,
    diagnostics = null
  ) {
    if (!Array.isArray(trajectory) || !trajectory.length || !Number.isFinite(reward)) return;

    const signed = Math.max(-1, Math.min(1, reward));
    const critic = Math.max(0, Math.min(1,
      diagnostics?.independentVerifier ??
      diagnostics?.independentActFulfillment ??
      diagnostics?.independentPromptRelationMatch ??
      diagnostics?.semanticRelationCompleteness ??
      diagnostics?.semanticGraphQueryResolution ??
      diagnostics?.relationStructureIntegrity ??
      diagnostics?.querySlotResolution ??
      diagnostics?.eventQueryResolution ??
      diagnostics?.universalCritic ??
      diagnostics?.criticalFloor ?? 0.50
    ));

    // Positive feedback is quality-gated to avoid teaching the policy to love
    // answers that only fooled a proxy. Negative feedback remains strong.
    const terminal = signed > 0
      ? signed * (0.35 + critic * 0.65)
      : signed;

    let G = terminal;
    const alpha = this.reasoningConfig.rlLearningRate;
    const limit = this.reasoningConfig.rlWeightLimit;

    for (let i = trajectory.length - 1; i >= 0; i--) {
      const features = trajectory[i]?.rlFeatures;
      if (!Array.isArray(features) || features.length !== this.rlFeatureCount) {
        G *= this.reasoningConfig.rlDiscount;
        continue;
      }

      const prediction =
        this.getRLPolicyValue(
          features
        );

      const token =
        trajectory[i]?.token;

      const type =
        Number.isInteger(token)
          ? this.getCanonicalType(token)
          : "Other";

      const fatigue =
        Number.isInteger(token) &&
        this.isContentType(type)
          ? this.getCrossTurnLexicalFatigue(
              token
            )
          : 0.0;

      const fatigueExcess =
        Math.max(
          0,
          fatigue -
          this.reasoningConfig
            .lexicalFatigueStart
        );

      const monopolyDamp =
        signed > 0
          ? Math.min(
              this.reasoningConfig
                .monopolyRewardMaxDamp,
              fatigueExcess *
              this.reasoningConfig
                .monopolyRewardScale
            )
          : 0.0;

      const stepTarget =
        G *
        (
          1.0 -
          monopolyDamp
        );

      const error =
        Math.max(
          -2.0,
          Math.min(
            2.0,
            stepTarget -
            prediction
          )
        );

      const stepScale =
        alpha /
        Math.sqrt(
          1 +
          (
            trajectory.length -
            1 -
            i
          ) *
          0.18
        );

      for (let j = 0; j < this.rlFeatureCount; j++) {
        const feature = Number(features[j]) || 0;
        this.rlPolicyWeights[j] = Math.max(
          -limit,
          Math.min(limit, this.rlPolicyWeights[j] + stepScale * error * feature)
        );
      }

      this.rlReplay.push({
        features:
          Array.from(features),
        target:
          stepTarget
      });

      G *=
        this.reasoningConfig
          .rlDiscount;
    }

    if (this.rlReplay.length > this.reasoningConfig.rlReplayLimit) {
      this.rlReplay.splice(0, this.rlReplay.length - this.reasoningConfig.rlReplayLimit);
    }

    // Tiny deterministic replay batch, only on explicit feedback.
    const batch = Math.min(this.reasoningConfig.rlReplayBatch, this.rlReplay.length);
    const replayAlpha = this.reasoningConfig.rlReplayLearningRate;
    if (batch > 0) {
      const stride = Math.max(1, Math.floor(this.rlReplay.length / batch));
      let used = 0;
      for (let index = this.rlReplay.length - 1; index >= 0 && used < batch; index -= stride, used++) {
        const item = this.rlReplay[index];
        const prediction = this.getRLPolicyValue(item.features);
        const error = Math.max(-2.0, Math.min(2.0, item.target - prediction));
        for (let j = 0; j < this.rlFeatureCount; j++) {
          const feature = Number(item.features[j]) || 0;
          this.rlPolicyWeights[j] = Math.max(
            -limit,
            Math.min(limit, this.rlPolicyWeights[j] + replayAlpha * error * feature)
          );
        }
      }
    }

    this.rlUpdates++;
  }


  userPredictiveSigmoid(x) {
    x = Math.max(-8, Math.min(8, Number(x) || 0));
    return 1 / (1 + Math.exp(-x));
  }

  getUserContentTokens(tokens) {
    return (tokens || []).filter(t => Number.isInteger(t) && this.isContentType(this.getCanonicalType(t)));
  }

  encodeUserState(tokens, contextModel) {
    const q = contextModel?.questionProfile || {};
    const f = contextModel?.semanticFrame || {};
    const g = contextModel?.clauseGraph || {};
    const c = contextModel?.answerContract || {};
    const target = f.queryTarget || contextModel?.queryFrame?.requestedSlot || "content";
    let pronouns = 0, nonPunct = 0;
    for (const t of tokens || []) {
      const type = this.getCanonicalType(t);
      if (type === "Punct") continue;
      nonPunct++;
      if (type === "Pronoun") pronouns++;
    }
    const pd = Math.max(0, Math.min(1, pronouns / Math.max(1, nonPunct) * 3));
    const continuity = Math.max(0, Math.min(1, pd * 0.65 + (c.underspecified ? 0.50 : 0) + ((contextModel?.promptContentTokens?.length || 0) <= 1 ? 0.22 : 0)));
    return new Float32Array([
      q.isQuestion ? 1 : 0,
      q.addressedRequest ? 1 : 0,
      c.underspecified ? 1 : 0,
      ["process","action","event"].includes(f.primaryFrame) ? 1 : 0,
      f.primaryFrame === "condition" || g.hasCondition ? 1 : 0,
      target === "truth" ? 1 : 0,
      ["mechanism","procedure"].includes(target) ? 1 : 0,
      ["cause","effect"].includes(target) ? 1 : 0,
      ["time_condition","location","quantity"].includes(target) ? 1 : 0,
      pd,
      (g.hasEntity ? 0.5 : 0) + (g.hasProcess ? 0.5 : 0),
      continuity
    ]);
  }

  predictUserState(source) {
    const out = new Float32Array(this.userStateDim);
    if (!source || source.length !== this.userStateDim) { out.fill(0.5); return out; }
    for (let o = 0; o < this.userStateDim; o++) {
      let z = this.userStateBias[o];
      const base = o * this.userStateDim;
      for (let i = 0; i < this.userStateDim; i++) z += this.userStateWeights[base+i] * source[i];
      out[o] = this.userPredictiveSigmoid(z);
    }
    return out;
  }

  updateUserStatePredictor(actual) {
    if (!this.pendingUserState || !this.pendingUserPrediction || actual.length !== this.userStateDim) return null;
    const lr = this.reasoningConfig.userStateLearningRate, lim = this.reasoningConfig.userStateWeightLimit;
    let mae = 0;
    for (let o = 0; o < this.userStateDim; o++) {
      const err = actual[o] - this.pendingUserPrediction[o];
      mae += Math.abs(err);
      const base = o * this.userStateDim;
      for (let i = 0; i < this.userStateDim; i++) {
        this.userStateWeights[base+i] = Math.max(-lim, Math.min(lim, this.userStateWeights[base+i] + lr * err * this.pendingUserState[i]));
      }
      this.userStateBias[o] = Math.max(-lim, Math.min(lim, this.userStateBias[o] + lr * err));
    }
    mae /= this.userStateDim;
    const a = this.reasoningConfig.userStateErrorEMA;
    this.userStateErrorEMA = this.userStateErrorEMA * a + mae * (1-a);
    return mae;
  }

  reinforceUserTransition(map, key, amount=1) {
    if (!key) return;
    map.set(key, Math.min(12, (map.get(key)||0) * this.reasoningConfig.userTransitionDecay + Math.max(0,amount)));
  }

  predictiveStrength(map, key) {
    return Math.max(0, Math.min(1, 1 - Math.exp(-(map.get(key)||0) * 0.48)));
  }

  learnUserSequence(tokens) {
    let pt = null, pp = null;
    for (const t of tokens || []) {
      if (!Number.isInteger(t)) continue;
      const pos = this.getCanonicalType(t);
      if (pos === "Punct") continue;
      if (Number.isInteger(pt)) this.reinforceUserTransition(this.userTokenTransitions, `${pt}>${t}`, 1);
      if (pp) this.reinforceUserTransition(this.userPOSTransitions, `${pp}>${pos}`, 1);
      pt = t; pp = pos;
    }
    this.trimUserPredictiveMap(this.userTokenTransitions, this.reasoningConfig.userTokenTransitionLimit);
    this.trimUserPredictiveMap(this.userPOSTransitions, this.reasoningConfig.userPOSTransitionLimit);
  }

  predictNextUserTokens(tokens) {
    const content = this.getUserContentTokens(tokens);
    const last = content.length ? content[content.length-1] : null;
    if (!Number.isInteger(last)) return [];
    const prefix = `${last}>`, items = [];
    const scan = (map, scale) => {
      for (const [k,v] of map.entries()) if (k.startsWith(prefix)) {
        const t = Number(k.slice(prefix.length));
        if (Number.isInteger(t) && this.network?.vocab?.hasToken?.(t)) items.push([t,v*scale]);
      }
    };
    scan(this.userCrossTurnTransitions, 1);
    if (!items.length) scan(this.userTokenTransitions, 0.45);
    items.sort((a,b)=>b[1]-a[1]);
    return items.slice(0,this.reasoningConfig.userNextTokenLimit).map(x=>x[0]);
  }

  prepareUserInputLearning(tokens, contextModel) {
    if (!this.reasoningConfig.userInputLearningEnabled) return null;
    const actual = this.encodeUserState(tokens, contextModel);
    const stateError = this.updateUserStatePredictor(actual);
    const content = this.getUserContentTokens(tokens);

    if (Number.isInteger(this.previousUserLastContent) && content.length) {
      for (let i=0;i<Math.min(3,content.length);i++) {
        this.reinforceUserTransition(this.userCrossTurnTransitions, `${this.previousUserLastContent}>${content[i]}`, i===0?1:(i===1?0.7:0.5));
      }
      this.trimUserPredictiveMap(this.userCrossTurnTransitions, this.reasoningConfig.userCrossTurnTransitionLimit);
    }

    if (this.pendingNextUserTokens.length && content.length) {
      const predicted = new Set(this.pendingNextUserTokens);
      const hit = content.slice(0,3).some(t=>predicted.has(t)) ? 1 : 0;
      this.userNextTokenHitEMA = this.userNextTokenHitEMA * 0.9 + hit * 0.1;
    }

    const profile = {
      currentState: actual,
      predictedNextState: this.predictUserState(actual),
      predictedNextTokenIds: this.predictNextUserTokens(tokens),
      carryoverNeed: actual[this.userStateDim-1],
      carryoverAnchorTokens: this.previousUserAnchors.slice(0,this.reasoningConfig.userCarryoverAnchorLimit),
      statePredictionError: Number.isFinite(stateError) ? stateError : null,
      statePredictionAccuracy: Math.max(0,Math.min(1,1-this.userStateErrorEMA)),
      nextTokenHitEMA: this.userNextTokenHitEMA
    };
    contextModel.userInputLearningProfile = profile;
    return profile;
  }

  commitUserInputLearning(tokens, contextModel) {
    const p = contextModel?.userInputLearningProfile;
    if (!p || !this.reasoningConfig.userInputLearningEnabled) return;
    this.learnUserSequence(tokens);
    this.pendingUserState = new Float32Array(p.currentState);
    this.pendingUserPrediction = new Float32Array(p.predictedNextState);
    this.pendingNextUserTokens = p.predictedNextTokenIds.slice(0,this.reasoningConfig.userNextTokenLimit);
    this.previousUserAnchors = (contextModel?.promptAnchorTokens || contextModel?.promptContentTokens || []).filter(Number.isInteger).slice(0,this.reasoningConfig.userCarryoverAnchorLimit);
    const content = this.getUserContentTokens(tokens);
    this.previousUserLastContent = content.length ? content[content.length-1] : null;
    this.userInputTurns++;
    this.saveUserPredictiveMemory();
  }

  getUserPredictionSignals(candidateToken, prevToken, contextModel) {
    const p = contextModel?.userInputLearningProfile;
    if (!p) return {tokenTransition:0,posTransition:0,carryover:0,predictedNextToken:0,futureContinuity:0,predictorConfidence:0.5,bonus:0};
    const curType = this.getCanonicalType(candidateToken);
    const prevType = Number.isInteger(prevToken) ? this.getCanonicalType(prevToken) : "START";
    const tokenTransition = Number.isInteger(prevToken) ? this.predictiveStrength(this.userTokenTransitions,`${prevToken}>${candidateToken}`) : 0;
    const posTransition = this.predictiveStrength(this.userPOSTransitions,`${prevType}>${curType}`);
    const carryover = p.carryoverNeed >= this.reasoningConfig.userCarryoverFloor && p.carryoverAnchorTokens.includes(candidateToken) ? p.carryoverNeed : 0;
    const predictedNextToken = p.predictedNextTokenIds.includes(candidateToken) ? 1 : 0;
    const futureContinuity = p.predictedNextState?.length === this.userStateDim
      ? Math.max(0,Math.min(1,p.predictedNextState[0]*0.25 + p.predictedNextState[this.userStateDim-1]*0.75)) : 0.5;
    const anchor = this.isSourceAnchorFamily(candidateToken,contextModel) ? 1 : 0;
    const predictorConfidence = p.statePredictionAccuracy;
    let bonus = tokenTransition*this.reasoningConfig.userTokenTransitionWeight +
      posTransition*this.reasoningConfig.userPOSTransitionWeight +
      carryover*this.reasoningConfig.userCarryoverWeight +
      predictedNextToken*this.reasoningConfig.userPredictedTokenWeight +
      futureContinuity*anchor*this.reasoningConfig.userFutureContinuityWeight;
    bonus *= 0.72 + predictorConfidence*0.28;
    bonus = Math.max(0,Math.min(this.reasoningConfig.userPredictionMaxBonus,bonus));
    return {tokenTransition,posTransition,carryover,predictedNextToken,futureContinuity,predictorConfidence,bonus};
  }

  getAutomaticVerifierReward(
    diagnostics
  ) {
    if (!diagnostics) {
      return 0.0;
    }

    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const act =
      clamp(
        diagnostics
          .independentActFulfillment
      );

    const proposition =
      clamp(
        diagnostics
          .independentPropositionCompleteness
      );

    const relation =
      clamp(
        diagnostics
          .independentPromptRelationMatch
      );

    const grounding =
      clamp(
        diagnostics
          .independentGrounding
      );

    const language =
      clamp(
        diagnostics
          .independentLanguageIntegrity
      );

    const contradiction =
      clamp(
        diagnostics
          .independentContradictionIntegrity
      );

    const quality =
      act * 0.22 +
      proposition * 0.20 +
      relation * 0.24 +
      grounding * 0.12 +
      language * 0.14 +
      contradiction * 0.08;

    const selfWhyGap =
      this.getSelfWhyVerifierDisagreement(
        diagnostics
      );

    if (
      selfWhyGap >=
      this.reasoningConfig.selfWhyMismatchGap
    ) {
      this.selfWhyDisagreementEMA =
        this.selfWhyDisagreementEMA *
          0.92 +
        selfWhyGap *
          0.08;

      this.selfWhyDisagreementCount++;
    }

    if (
      !diagnostics
        .independentResolved
    ) {
      const failure =
        Math.max(
          this.reasoningConfig
            .automaticRSIMinFailureReward,
          (
            1.0 -
            quality
          ) *
          this.reasoningConfig
            .automaticRSINegativeMax +
          Math.max(
            0,
            selfWhyGap -
            this.reasoningConfig.selfWhyMismatchGap
          ) *
          this.reasoningConfig.selfWhyMismatchPenalty
        );

      return -Math.min(
        this.reasoningConfig
          .automaticRSINegativeMax,
        failure
      );
    }

    if (
      diagnostics.final >=
      this.reasoningConfig
        .answerAcceptanceConfidence
    ) {
      const strength =
        clamp(
          (
            diagnostics.final -
            this.reasoningConfig
              .answerAcceptanceConfidence
          ) /
          Math.max(
            0.01,
            1.0 -
            this.reasoningConfig
              .answerAcceptanceConfidence
          )
        );

      return (
        0.06 +
        strength *
        (
          this.reasoningConfig
            .automaticRSIPositiveMax -
          0.06
        )
      );
    }

    return -Math.min(
      0.08,
      (
        this.reasoningConfig
          .answerAcceptanceConfidence -
        diagnostics.final
      ) *
      0.18
    );
  }

  getTrajectoryMeanFeatures(
    trajectory
  ) {
    const mean =
      new Float32Array(
        this.rlFeatureCount
      );

    if (
      !Array.isArray(trajectory) ||
      !trajectory.length
    ) {
      return mean;
    }

    let count = 0;

    for (const step of trajectory) {
      const features =
        step?.rlFeatures;

      if (
        !Array.isArray(features) ||
        features.length !==
          this.rlFeatureCount
      ) {
        continue;
      }

      count++;

      for (
        let j = 0;
        j < this.rlFeatureCount;
        j++
      ) {
        mean[j] +=
          Number(features[j]) || 0;
      }
    }

    if (count > 0) {
      for (
        let j = 0;
        j < this.rlFeatureCount;
        j++
      ) {
        mean[j] /= count;
      }
    }

    return mean;
  }

  replayRSIContrasts() {
    if (
      !this.rsiReplay.length
    ) {
      return;
    }

    const batch =
      Math.min(
        this.reasoningConfig
          .automaticRSIReplayBatch,
        this.rsiReplay.length
      );

    const stride =
      Math.max(
        1,
        Math.floor(
          this.rsiReplay.length /
          batch
        )
      );

    const lr =
      this.reasoningConfig
        .automaticRSIReplayLearningRate;

    const limit =
      this.reasoningConfig
        .rlWeightLimit;

    let used = 0;

    for (
      let index =
        this.rsiReplay.length - 1;
      index >= 0 &&
      used < batch;
      index -= stride, used++
    ) {
      const item =
        this.rsiReplay[index];

      const margin =
        Math.max(
          0,
          Math.min(
            1,
            item.margin
          )
        );

      for (
        let j = 0;
        j < this.rlFeatureCount;
        j++
      ) {
        const direction =
          (
            Number(item.good[j]) ||
            0
          ) -
          (
            Number(item.bad[j]) ||
            0
          );

        this.rlPolicyWeights[j] =
          Math.max(
            -limit,
            Math.min(
              limit,
              this.rlPolicyWeights[j] +
              lr *
              margin *
              direction
            )
          );
      }
    }
  }

  trainRSIContrast(
    badTrajectory,
    badDiagnostics,
    goodTrajectory,
    goodDiagnostics,
    strategy = "recovery"
  ) {
    if (
      !this.reasoningConfig
        .automaticRSIEnabled
    ) {
      return 0.0;
    }

    const badUtility =
      this.getRecoveryCandidateUtility(
        badDiagnostics
      );

    const goodUtility =
      this.getRecoveryCandidateUtility(
        goodDiagnostics
      );

    const gain =
      goodUtility -
      badUtility;

    if (
      !Number.isFinite(gain) ||
      gain <
        this.reasoningConfig
          .automaticRSIMinContrastGain
    ) {
      return 0.0;
    }

    const bad =
      this.getTrajectoryMeanFeatures(
        badTrajectory
      );

    const good =
      this.getTrajectoryMeanFeatures(
        goodTrajectory
      );

    const margin =
      Math.max(
        0,
        Math.min(
          1,
          gain
        )
      );

    const lr =
      this.reasoningConfig
        .automaticRSIPairLearningRate;

    const limit =
      this.reasoningConfig
        .rlWeightLimit;

    for (
      let j = 0;
      j < this.rlFeatureCount;
      j++
    ) {
      const direction =
        good[j] -
        bad[j];

      this.rlPolicyWeights[j] =
        Math.max(
          -limit,
          Math.min(
            limit,
            this.rlPolicyWeights[j] +
            lr *
            margin *
            direction
          )
        );
    }

    this.rsiReplay.push({
      bad:
        Array.from(bad),
      good:
        Array.from(good),
      margin,
      strategy
    });

    if (
      this.rsiReplay.length >
      this.reasoningConfig
        .automaticRSIReplayLimit
    ) {
      this.rsiReplay.splice(
        0,
        this.rsiReplay.length -
        this.reasoningConfig
          .automaticRSIReplayLimit
      );
    }

    this.replayRSIContrasts();

    this.rsiUpdates++;

    return gain;
  }

  learnFromAutomaticRSIAttempts(
    attempts
  ) {
    if (
      !this.reasoningConfig
        .automaticRSIEnabled ||
      !Array.isArray(attempts) ||
      !attempts.length
    ) {
      return {
        learned: false,
        contrastGain: 0.0,
        updates: 0
      };
    }

    const initial =
      attempts[0];

    let best =
      initial;

    let bestUtility =
      this.getRecoveryCandidateUtility(
        initial.diagnostics
      );

    for (
      const attempt of attempts
    ) {
      const utility =
        this.getRecoveryCandidateUtility(
          attempt.diagnostics
        );

      if (
        utility >
        bestUtility
      ) {
        best =
          attempt;
        bestUtility =
          utility;
      }
    }

    let updates = 0;

    // Failed attempts become weak negative policy training. This changes
    // behavior/search preference only; no generated claim is stored as fact.
    if (
      !this.isRecoveredAnswerAcceptable(
        initial.diagnostics
      )
    ) {
      const negative =
        this.getAutomaticVerifierReward(
          initial.diagnostics
        );

      if (negative < 0) {
        this.trainRLPolicy(
          negative,
          initial.winner
            ?.trajectory || [],
          initial.diagnostics
        );

        updates++;
        this.automaticMistakesLearned++;
      }
    }

    let contrastGain = 0.0;

    if (
      best !== initial
    ) {
      contrastGain =
        this.trainRSIContrast(
          initial.winner
            ?.trajectory || [],
          initial.diagnostics,
          best.winner
            ?.trajectory || [],
          best.diagnostics,
          best.strategy ||
            "recovery"
        );

      if (
        contrastGain >
        0
      ) {
        updates++;

        const positive =
          this.getAutomaticVerifierReward(
            best.diagnostics
          );

        if (positive > 0) {
          this.trainRLPolicy(
            positive,
            best.winner
              ?.trajectory || [],
            best.diagnostics
          );

          updates++;
        }
      }
    }

    return {
      learned:
        updates > 0,
      contrastGain,
      updates
    };
  }

  buildFailedTrajectorySuppression(
    tokens,
    contextModel
  ) {
    const tokenCounts =
      new Map();

    const bigrams =
      new Set();

    let previous = null;

    for (const token of tokens || []) {
      if (
        !Number.isInteger(token)
      ) {
        continue;
      }

      const type =
        this.getCanonicalType(
          token
        );

      if (
        type === "Punct"
      ) {
        continue;
      }

      if (
        !this.isSourceAnchorFamily(
          token,
          contextModel
        )
      ) {
        tokenCounts.set(
          token,
          (
            tokenCounts.get(token) ||
            0
          ) +
          1
        );
      }

      if (
        Number.isInteger(previous)
      ) {
        bigrams.add(
          `${previous}>${token}`
        );
      }

      previous = token;
    }

    return {
      tokenCounts,
      bigrams
    };
  }

  getFailedTrajectorySuppression(
    candidateToken,
    prevToken,
    contextModel
  ) {
    const suppression =
      contextModel
        ?.failedTrajectorySuppression;

    if (
      !suppression ||
      contextModel
        ?.neuralRetryActive !==
        true
    ) {
      return 0.0;
    }

    if (
      this.isSourceAnchorFamily(
        candidateToken,
        contextModel
      )
    ) {
      return 0.0;
    }

    const count =
      suppression.tokenCounts
        ?.get(candidateToken) ||
      0;

    let penalty =
      count *
      this.reasoningConfig
        .neuralRetryFailedTokenPenalty;

    if (
      Number.isInteger(prevToken) &&
      suppression.bigrams
        ?.has(
          `${prevToken}>${candidateToken}`
        )
    ) {
      penalty +=
        this.reasoningConfig
          .neuralRetryFailedBigramPenalty;
    }

    return penalty;
  }

  buildNeuralRetryInput(
    cleanPromptTokens,
    failedTokens
  ) {
    const suffix = [];

    for (
      const token of
        failedTokens || []
    ) {
      if (
        !Number.isInteger(token)
      ) {
        continue;
      }

      const type =
        this.getCanonicalType(
          token
        );

      if (type === "Punct") {
        continue;
      }

      suffix.push(token);
    }

    const boundedSuffix =
      suffix.slice(
        -this.reasoningConfig
          .neuralRetryContextTokens
      );

    return [
      ...cleanPromptTokens,
      ...boundedSuffix
    ];
  }

  resetReasoningForNeuralRetry(
    contextModel
  ) {
    contextModel.reasoningInitialized =
      false;

    contextModel.reasoningCentroid =
      null;

    contextModel.reasoningScores =
      null;

    contextModel.reasoningStability =
      0.0;

    contextModel.reasoningHops =
      0;

    contextModel.deepCandidatePool =
      null;

    contextModel.deepCandidatePoolRound =
      -1;

    contextModel.deepCandidatePoolStats =
      null;

    contextModel.semanticGraphCandidateCache =
      new Map();
  }

  detectAnswerShape(
    cleanPromptTokens,
    questionProfile,
    answerContract
  ) {
    const words =
      cleanPromptTokens.map(
        token =>
          String(
            this.tokenWord(token) || ""
          ).toLowerCase()
      );

    const types =
      cleanPromptTokens.map(
        token =>
          this.getCanonicalType(token)
      );

    const has =
      word =>
        words.includes(word);

    const first =
      words[0] || "";

    const listWords =
      new Set([
        "list", "lists", "name",
        "names", "examples", "example",
        "options", "option", "ways",
        "way", "types", "type",
        "kinds", "kind", "items",
        "steps", "reasons", "features", "fact", "facts"
      ]);

    const definitionWords =
      new Set([
        "define", "definition",
        "meaning", "means"
      ]);

    const listRequested =
      words.some(
        word =>
          listWords.has(word)
      ) ||
      (
        has("some") &&
        (
          has("what") ||
          has("which")
        )
      ) ||
      (
        questionProfile?.addressedRequest &&
        (
          has("list") ||
          has("name")
        )
      );

    if (listRequested) {
      let targetItems = 3;

      for (
        let i = 0;
        i < words.length;
        i++
      ) {
        if (types[i] === "Num") {
          const parsed =
            Number(words[i]);

          if (
            Number.isFinite(parsed) &&
            parsed >= 1
          ) {
            targetItems =
              Math.max(
                1,
                Math.min(
                  8,
                  Math.round(parsed)
                )
              );
            break;
          }
        }
      }

      return {
        kind: "list",
        targetItems,
        allowsFragments: true,
        requiresPredicate: false,
        directness: 0.72
      };
    }

    if (
      definitionWords.has(first) ||
      (
        has("what") &&
        (
          has("is") ||
          has("are") ||
          has("means")
        ) &&
        answerContract?.kind ===
          "definition"
      )
    ) {
      return {
        kind: "definition",
        targetItems: 1,
        allowsFragments: false,
        requiresPredicate: true,
        directness: 0.78
      };
    }

    if (answerContract?.kind === "boolean") {
      return {
        kind: "yes_no",
        targetItems: 1,
        allowsFragments: false,
        requiresPredicate: true,
        directness: 0.94
      };
    }

    if (
      questionProfile?.operatorWord === "when" ||
      answerContract?.kind === "temporal"
    ) {
      const temporal=this.classifyTemporalQuestion(cleanPromptTokens,questionProfile,answerContract);
      return {kind:"temporal",targetItems:1,temporalMode:temporal.mode,temporalTense:temporal.tense,allowsFragments:temporal.allowsFragment,requiresPredicate:temporal.requiresClause,directness:temporal.predictive?0.78:0.94,minimumTemporalEvidence:temporal.minimumTemporalEvidence};
    }

    if (
      ["who","where","which"].includes(questionProfile?.operatorWord) ||
      answerContract?.kind === "quantitative"
    ) {
      return {kind:"short",targetItems:1,allowsFragments:true,requiresPredicate:false,directness:0.92};
    }

    if (
      answerContract?.kind ===
        "clarification"
    ) {
      return {
        kind: "short",
        targetItems: 1,
        allowsFragments: true,
        requiresPredicate: false,
        directness: 1.0
      };
    }

    return {
      kind: "prose",
      targetItems: 1,
      allowsFragments: false,
      requiresPredicate: true,
      directness: 0.58
    };
  }

  getCurrentClauseTokens(tokens) {
    if (!tokens?.length) {
      return [];
    }

    let start = 0;

    for (
      let i = tokens.length - 1;
      i >= 0;
      i--
    ) {
      const word =
        this.tokenWord(tokens[i]);

      if (
        [
          ".",
          "!",
          "?",
          ";"
        ].includes(word)
      ) {
        start = i + 1;
        break;
      }
    }

    return tokens.slice(start);
  }

  countMorphologicalFamilyInCurrentClause(
    candidateToken,
    generated
  ) {
    const vocab =
      this.network?.vocab;

    const candidateType =
      this.getCanonicalType(
        candidateToken
      );

    if (
      !this.isContentType(
        candidateType
      )
    ) {
      return 0;
    }

    const clause =
      this.getCurrentClauseTokens(
        generated
      );

    let count = 0;

    for (const usedToken of clause) {
      const usedType =
        this.getCanonicalType(
          usedToken
        );

      if (
        !this.isContentType(
          usedType
        )
      ) {
        continue;
      }

      let sameFamily = false;

      if (
        typeof vocab
          ?.sameMorphologicalFamily ===
          "function"
      ) {
        sameFamily =
          vocab.sameMorphologicalFamily(
            usedToken,
            candidateToken
          );
      } else {
        const usedStem =
          this.stemWord(
            this.tokenWord(
              usedToken
            )
          );

        const candidateStem =
          this.stemWord(
            this.tokenWord(
              candidateToken
            )
          );

        sameFamily =
          usedStem.length >= 3 &&
          candidateStem.length >= 3 &&
          usedStem === candidateStem;
      }

      if (sameFamily) {
        count++;
      }
    }

    return count;
  }

  hasMorphologicalFamilyRepeat(
    candidateToken,
    generated
  ) {
    return (
      this.countMorphologicalFamilyInCurrentClause(
        candidateToken,
        generated
      ) > 0
    );
  }

  measureMorphologicalDiversity(tokens) {
    if (!tokens?.length) {
      return {
        score: 0.0,
        repeatedFamilies: 0,
        contentTokens: 0
      };
    }

    const vocab =
      this.network?.vocab;

    let repeatedFamilies = 0;
    let contentTokens = 0;

    let clauseRepresentatives = [];

    const resetClause = () => {
      clauseRepresentatives = [];
    };

    for (const token of tokens) {
      const word =
        this.tokenWord(token);

      if (
        [
          ".",
          "!",
          "?",
          ";"
        ].includes(word)
      ) {
        resetClause();
        continue;
      }

      const type =
        this.getCanonicalType(token);

      if (
        !this.isContentType(type)
      ) {
        continue;
      }

      contentTokens++;

      let repeated = false;

      for (
        const existing of
          clauseRepresentatives
      ) {
        if (
          typeof vocab
            ?.sameMorphologicalFamily ===
            "function"
        ) {
          if (
            vocab.sameMorphologicalFamily(
              existing,
              token
            )
          ) {
            repeated = true;
            break;
          }
        } else {
          const a =
            this.stemWord(
              this.tokenWord(
                existing
              )
            );

          const b =
            this.stemWord(word);

          if (
            a.length >= 3 &&
            b.length >= 3 &&
            a === b
          ) {
            repeated = true;
            break;
          }
        }
      }

      if (repeated) {
        repeatedFamilies++;
      } else {
        clauseRepresentatives.push(
          token
        );
      }
    }

    const score =
      contentTokens
        ? Math.max(
            0,
            Math.min(
              1,
              1.0 -
              repeatedFamilies /
              contentTokens *
              1.65
            )
          )
        : 1.0;

    return {
      score,
      repeatedFamilies,
      contentTokens
    };
  }

  answerShapeTokenBias(
    tokenId,
    curType,
    sentenceState,
    contextModel
  ) {
    const shape =
      contextModel?.answerShape?.kind ||
      "prose";

    if (shape === "temporal") {
      const temporal =
        this.temporalTenseAssessment(
          tokenId,
          [],
          contextModel
        );

      let bias =
        (temporal.score - 0.35) *
        5.2;

      bias +=
        (
          temporal.compatibility -
          0.50
        ) *
        4.4;

      if (
        temporal.score >= 0.80 &&
        temporal.compatibility >= 0.80
      ) {
        bias += 1.8;
      }

      if (
        temporal.contradiction
      ) {
        bias -= 6.0;
      }

      if (
        temporal.score < 0.20 &&
        sentenceState.predicateReady &&
        this.isContentType(curType)
      ) {
        bias -= 1.4;
      }

      return bias;
    }

    if (shape === "list") {
      if (sentenceState.isStart) {
        if (
          [
            "Noun",
            "ProperNoun",
            "Adj",
            "Num"
          ].includes(curType)
        ) {
          return 2.8;
        }

        if (
          [
            "Det",
            "Pronoun"
          ].includes(curType)
        ) {
          return 0.8;
        }
      }

      if (
        sentenceState.subjectReady &&
        !sentenceState.predicateReady
      ) {
        if (
          [
            "Noun",
            "ProperNoun",
            "Adj",
            "Prep"
          ].includes(curType)
        ) {
          return 1.6;
        }

        if (curType === "Punct") {
          return 4.0;
        }
      }
    }

    if (
      shape === "short" ||
      shape === "yes_no"
    ) {
      if (
        sentenceState.isStart &&
        [
          "Noun",
          "ProperNoun",
          "Pronoun",
          "Num",
          "Adj"
        ].includes(curType)
      ) {
        return 2.2;
      }

      if (
        sentenceState.subjectReady &&
        !sentenceState.predicateReady &&
        curType === "Punct"
      ) {
        return 3.5;
      }
    }

    return 0.0;
  }

  getVocabTokenId(word) {
    const vocab = this.network.vocab;
    if (!vocab || typeof word !== "string") return null;

    const exact = vocab.tokenToId?.get?.(word);
    if (Number.isFinite(exact)) return exact;

    const lower = word.toLowerCase();

    const folded =
      vocab.lowerTokenToId?.get?.(lower);

    if (Number.isFinite(folded)) return folded;

    const directLower =
      vocab.tokenToId?.get?.(lower);

    return Number.isFinite(directLower)
      ? directLower
      : null;
  }

  buildMarkerCentroid(words) {
    const embDim = this.network.embDim;
    const out = new Float32Array(embDim);
    let count = 0;

    for (const word of words || []) {
      const tokenId = this.getVocabTokenId(word);
      if (!Number.isFinite(tokenId)) continue;

      const off =
        (tokenId % this.network.vocabSize) *
        embDim;

      for (let d = 0; d < embDim; d++) {
        out[d] +=
          this.normalizedEmbeddings[off + d];
      }

      count++;
    }

    if (!count) return null;

    for (let d = 0; d < embDim; d++) {
      out[d] /= count;
    }

    this.normalizeVector(out);
    return out;
  }

  inferQueryKindFromOperator(operatorToken, operatorType, cleanPromptTokens, phraseProfile) {
    if (!Number.isInteger(operatorToken)) return null;
    const vocab=this.network?.vocab;
    const roleScores={mechanism:0, causal:0, temporal:0, locative:0, quantitative:0, definition:0, evaluative:0};
    const roleMap={mechanism:"mechanism", causal:"cause", temporal:"time", locative:"location", quantitative:"quantity", definition:"definition", evaluative:"evaluation"};
    for (const [kind,role] of Object.entries(roleMap)) {
      const value=vocab?.getRoleCompatibility?.(operatorToken,role);
      if (Number.isFinite(value)) roleScores[kind]=Math.max(0,Math.min(1,value));
    }
    const temporal=vocab?.getTemporalProfile?.(operatorToken);
    if (Number.isFinite(temporal?.score)) roleScores.temporal=Math.max(roleScores.temporal,Math.min(1,temporal.score));
    const semanticClass=vocab?.getSemanticClass?.(operatorToken);
    if (semanticClass==="quantity") roleScores.quantitative=Math.max(roleScores.quantitative,0.94);
    if (semanticClass==="location") roleScores.locative=Math.max(roleScores.locative,0.94);
    if (semanticClass==="time") roleScores.temporal=Math.max(roleScores.temporal,0.94);
    if (operatorType==="Adv") roleScores.mechanism=Math.max(roleScores.mechanism,0.40);
    if (["Pronoun","Det"].includes(operatorType)) roleScores.definition=Math.max(roleScores.definition,0.42);
    if (["Pronoun","Det"].includes(operatorType) && phraseProfile?.hasEmbeddedClause) roleScores.definition*=0.72;
    let bestKind=null,bestScore=-Infinity;
    for (const [kind,score] of Object.entries(roleScores)) if (score>bestScore) {bestKind=kind;bestScore=score;}
    return bestScore>=0.34 ? {kind:bestKind,score:bestScore,scores:roleScores} : null;
  }

  buildAnswerContract(cleanPromptTokens, questionProfile) {
    const words=cleanPromptTokens.map(token=>String(this.tokenWord(token)||"").toLowerCase());
    const types=cleanPromptTokens.map(token=>this.getCanonicalType(token));
    const phraseProfile=questionProfile?.phraseProfile || this.parseTypePhrases(cleanPromptTokens);
    const operatorToken=questionProfile?.operatorToken ?? null;
    const operatorType=questionProfile?.operatorType ?? null;
    const operatorInference=this.inferQueryKindFromOperator(operatorToken,operatorType,cleanPromptTokens,phraseProfile);
    const isClosedQuestion=Boolean(phraseProfile?.closedQuestionLike) && !questionProfile?.addressedRequest && (phraseProfile?.frontedOperatorIndex ?? -1)<0;
    const isRequest=Boolean(questionProfile?.addressedRequest || phraseProfile?.imperativeLike);
    let kind="general";
    if (isClosedQuestion) kind="boolean";
    else if (questionProfile?.addressedRequest && phraseProfile?.embeddedOperatorIndex>=0) {
      const embeddedToken=cleanPromptTokens[phraseProfile.embeddedOperatorIndex];
      const embeddedType=this.getCanonicalType(embeddedToken);
      kind=this.inferQueryKindFromOperator(embeddedToken,embeddedType,cleanPromptTokens,phraseProfile)?.kind || "imperative";
    } else if (phraseProfile?.openQuestionLike && operatorInference?.kind) kind=operatorInference.kind;
    else if (isRequest) kind="imperative";
    if (kind==="general" && phraseProfile?.openQuestionLike) kind=operatorType==="Adv" ? "mechanism" : "definition";

    const relationVerbToken=phraseProfile?.answerPredicateToken ?? null;
    const relationVerbIndex=Number.isInteger(phraseProfile?.answerPredicateIndex) ? phraseProfile.answerPredicateIndex : -1;
    const subjectTokens=(phraseProfile?.subjectTokenIds || []).filter(token=>Number.isInteger(token)&&token!==operatorToken).slice(0,6);
    const targetTokens=(phraseProfile?.complementTokenIds || []).filter(token=>Number.isInteger(token)&&token!==operatorToken&&token!==relationVerbToken).slice(0,8);

    if (!subjectTokens.length) {
      for (let i=0;i<cleanPromptTokens.length;i++) {
        const token=cleanPromptTokens[i],type=types[i];
        if (token===operatorToken || token===relationVerbToken) continue;
        if (["Noun","ProperNoun","Pronoun","Num"].includes(type)) {subjectTokens.push(token); if(subjectTokens.length>=3) break;}
      }
    }
    if (!targetTokens.length && relationVerbIndex>=0) {
      for (let i=relationVerbIndex+1;i<cleanPromptTokens.length;i++) {
        const token=cleanPromptTokens[i],type=types[i];
        if (token===operatorToken || type==="Punct") continue;
        if (this.isContentType(type)||type==="Pronoun") {targetTokens.push(token); if(targetTokens.length>=6) break;}
      }
    }

    const numericPromptTokens=[];
    const quantityUnitTokens=[];
    for (let i=0;i<cleanPromptTokens.length;i++) {
      const token=cleanPromptTokens[i],type=types[i];
      if (type==="Num") numericPromptTokens.push(token);
      const semanticClass=this.network?.vocab?.getSemanticClass?.(token);
      const q=this.network?.vocab?.getRoleCompatibility?.(token,"quantity");
      if (semanticClass==="quantity" || (Number.isFinite(q)&&q>=0.68)) quantityUnitTokens.push(token);
    }
    if (kind==="mechanism" && (numericPromptTokens.length||quantityUnitTokens.length) && operatorType==="Adv") kind="quantitative";

    const underspecified=Boolean(questionProfile?.isQuestion) && [...subjectTokens,...targetTokens].length===0 && numericPromptTokens.length===0;
    if (underspecified) kind="clarification";

    // These are semantic seed atoms only; they do not define phrase structure.
    const markerWordsByKind={
      temporal:["time","date","future","later"],
      quantitative:["number","amount","quantity","total"],
      causal:["cause","reason","because","result"],
      effect:["result","effect","outcome","change"],
      locative:["location","place","region","area"],
      mechanism:["process","method","step","through"],
      boolean:["yes","no","not","true","false","maybe"],
      evaluative:["better","depends","recommended"],
      definition:["means","definition","property"],
      imperative:["answer","result","output"],
      clarification:["details","question","more"],
      general:[]
    };
    const markerWords=markerWordsByKind[kind]||[];
    const markerTokenIds=[];
    for (const marker of markerWords) {
      const id=this.getVocabTokenId(marker);
      if (Number.isFinite(id)&&!markerTokenIds.includes(id)) markerTokenIds.push(id);
    }
    const requiredSlotsByKind={
      mechanism:["subject","predicate","mechanism"],
      causal:["subject","predicate","cause"],
      effect:["subject","predicate","content"],
      temporal:["subject","predicate","time"],
      locative:["subject","predicate","location"],
      quantitative:["subject","predicate","quantity"],
      boolean:["subject","predicate","commitment"],
      evaluative:["subject","predicate","evaluation"],
      definition:["subject","predicate","definition"],
      factual_generation:["subject","content"],
      imperative:["predicate","content"],
      clarification:["content"],
      general:questionProfile?.isQuestion?["subject","predicate"]:[]
    };
    return {
      kind,
      isQuestion:Boolean(questionProfile?.isQuestion),
      operatorToken,
      operatorWord:questionProfile?.operatorWord||"",
      operatorType,
      operatorDemandsPredicate:Boolean(questionProfile?.isQuestion)&&operatorType==="Adv",
      relationVerbToken,
      subjectTokens,
      targetTokens,
      markerWords,
      markerTokenIds,
      requiredSlots:requiredSlotsByKind[kind]||[],
      numericPromptTokens,
      quantityUnitTokens,
      requiresNumericEvidence:kind==="quantitative",
      factualityRequired:Boolean(questionProfile?.isQuestion),
      underspecified,
      requiresRelation:!["general","clarification"].includes(kind),
      phraseProfile,
      requiresCommitment:kind==="boolean",
      commitmentType:kind==="boolean"?"proposition":"none",
      allowedCommitments:kind==="boolean"?["affirm","deny","uncertain","conditional"]:[]
    };
  }

  getRequiredSemanticRole(
    sentenceState,
    contextModel
  ) {
    const contract =
      contextModel?.answerContract;

    if (!sentenceState?.subjectReady) {
      return "subject";
    }

    if (!sentenceState?.predicateReady) {
      return "predicate";
    }

    if (
      sentenceState?.needsObjectAfterPrep ||
      sentenceState?.needsNominalAfterDet
    ) {
      return "object";
    }

    const relationRequired =
      contract?.requiredSlots?.some(
        slot =>
          ![
            "subject",
            "predicate"
          ].includes(slot)
      );

    if (
      relationRequired &&
      (
        sentenceState?.slotRelationEvidence ||
        0
      ) < 0.72
    ) {
      switch (contract?.kind) {
        case "mechanism":
          return "mechanism";
        case "causal":
          return "cause";
        case "temporal":
          return "time";
        case "locative":
          return "location";
        case "quantitative":
          return "quantity";
        case "definition":
          return "definition";
        case "evaluative":
          return "evaluation";
        default:
          return "content";
      }
    }

    if (
      sentenceState?.objectCount === 0
    ) {
      return "object";
    }

    return "content";
  }

  getRoleSensitiveKnowledgeEvidence(
    tokenId,
    sentenceState,
    contextModel,
    roleOverride = null
  ) {
    const vocab =
      this.network?.vocab;

    const role =
      roleOverride ||
      this.getRequiredSemanticRole(
        sentenceState,
        contextModel
      );

    const type =
      this.getCanonicalType(
        tokenId
      );

    let roleFit = 0.50;

    if (
      vocab &&
      typeof vocab.getRoleCompatibility === "function"
    ) {
      roleFit =
        vocab.getRoleCompatibility(
          tokenId,
          role
        );
    } else {
      if (role === "predicate") {
        roleFit =
          type === "Verb"
            ? 1.0
            : (
                [
                  "Aux",
                  "Modal",
                  "Adj"
                ].includes(type)
                  ? 0.55
                  : 0.10
              );
      } else if (role === "subject") {
        roleFit =
          [
            "Noun",
            "ProperNoun",
            "Pronoun"
          ].includes(type)
            ? 1.0
            : 0.10;
      } else {
        roleFit =
          this.isContentType(type)
            ? 0.62
            : 0.20;
      }
    }

    const subjectToken =
      Number.isInteger(
        sentenceState?.subjectToken
      )
        ? sentenceState.subjectToken
        : contextModel?.answerContract
            ?.subjectTokens?.[0];

    const predicateToken =
      Number.isInteger(
        sentenceState?.predicateToken
      )
        ? sentenceState.predicateToken
        : contextModel?.answerContract
            ?.relationVerbToken;

    let propositionFit = roleFit;
    let predicateFit = roleFit;
    let subjectKnowledge = 0.55;
    let typedRelation = 0.0;
    let collocation = 0.0;
    let classFit = 0.55;

    const trusted =
      this.getTrustedPropositionEvidence(
        subjectToken,
        predicateToken,
        tokenId,
        role
      );

    propositionFit =
      Math.max(
        roleFit * 0.82,
        trusted.score
      );

    predicateFit =
      Math.max(
        roleFit * 0.78,
        trusted.predicateTrusted
      );

    subjectKnowledge =
      Math.max(
        0.50,
        trusted.subjectTrusted
      );

    typedRelation =
      trusted.typedRelation;

    collocation =
      trusted.collocation;

    classFit =
      trusted.classFit;

    const overall =
      Math.max(
        0,
        Math.min(
          1,
          roleFit * 0.38 +
          propositionFit * 0.42 +
          predicateFit * 0.10 +
          subjectKnowledge * 0.05 +
          typedRelation * 0.03 +
          collocation * 0.02
        )
      );

    return {
      role,
      overall,
      roleFit:
        Math.max(
          0,
          Math.min(1, roleFit)
        ),
      propositionFit:
        Math.max(
          0,
          Math.min(
            1,
            propositionFit
          )
        ),
      predicateFit:
        Math.max(
          0,
          Math.min(
            1,
            predicateFit
          )
        ),
      subjectKnowledge:
        Math.max(
          0,
          Math.min(
            1,
            subjectKnowledge
          )
        ),
      typedRelation:
        Math.max(
          0,
          Math.min(
            1,
            typedRelation
          )
        ),
      collocation:
        Math.max(
          0,
          Math.min(
            1,
            collocation
          )
        ),
      classFit:
        Math.max(
          0,
          Math.min(
            1,
            classFit
          )
        )
    };
  }

  roleSensitiveTokenBonus(
    tokenId,
    generated,
    sentenceState,
    contextModel
  ) {
    const type =
      this.getCanonicalType(
        tokenId
      );

    if (
      !this.isContentType(type) &&
      ![
        "Pronoun",
        "Num"
      ].includes(type)
    ) {
      return 0.0;
    }

    const fit =
      this.getRoleSensitiveKnowledgeEvidence(
        tokenId,
        sentenceState,
        contextModel
      );

    let bonus =
      (fit.overall - 0.50) *
      7.0;

    if (
      fit.overall >= 0.78
    ) {
      bonus += 1.8;
    }

    if (
      fit.typedRelation >= 0.70
    ) {
      bonus += 0.7;
    }

    if (
      fit.collocation >= 0.75
    ) {
      bonus += 0.8;
    }

    // Once a question is waiting for its requested semantic role, generic
    // topic-related words do not get to coast on Know/Assoc alone.
    const role =
      fit.role;

    const roleIsSpecific =
      [
        "mechanism",
        "cause",
        "time",
        "location",
        "quantity",
        "definition",
        "evaluation"
      ].includes(role);

    if (
      roleIsSpecific &&
      fit.overall < 0.30 &&
      generated.length >= 3
    ) {
      bonus -= 5.5;
    }

    return bonus;
  }

  getContractRelationTokenEvidence(
    tokenId,
    type,
    contextModel,
    sentenceState = null
  ) {
    const contract =
      contextModel?.answerContract;

    if (!contract) return 0.0;

    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    if (
      contract.markerTokenIds?.includes(tokenId)
    ) {
      // A relation marker ("by", "because", "in", etc.) only OPENS the
      // requested-information phrase. It is not the information itself.
      return 0.38;
    }

    const contractScore =
      clamp(
        contextModel?.contractScores?.[tokenId]
      );

    const routed =
      clamp(
        this.getRoutedSemanticScore(
          tokenId,
          contextModel,
          sentenceState
        )
      );

    const support =
      Math.max(
        contractScore,
        routed * 0.85
      );

    switch (contract.kind) {
      case "mechanism":
        if (type === "Prep") {
          return clamp(0.18 + support * 0.42);
        }
        if (["Verb", "Adv"].includes(type)) {
          return clamp(support * 0.92);
        }
        if (["Noun", "ProperNoun", "Adj"].includes(type)) {
          return clamp(support * 0.88);
        }
        break;

      case "causal":
        if (["Conj", "Prep"].includes(type)) {
          return clamp(0.18 + support * 0.42);
        }
        if (["Adv", "Verb", "Noun", "ProperNoun", "Adj"].includes(type)) {
          return clamp(support * 0.90);
        }
        break;

      case "temporal": {
        const temporal=this.temporalTokenEvidence(tokenId,contextModel);
        if(temporal.score>=0.68)return clamp(temporal.score*0.78+support*0.22);
        return clamp(temporal.score*0.40+support*0.18);
      }

      case "locative":
        if (type === "Prep") {
          return clamp(0.16 + support * 0.38);
        }
        if (
          ["Noun", "ProperNoun", "Adv", "Adj"].includes(type)
        ) {
          return clamp(support * 0.92);
        }
        break;

      case "quantitative":
        if (type === "Num") return 1.0;
        if (["Noun", "Adj"].includes(type)) {
          return clamp(0.20 + support * 0.80);
        }
        break;

      case "boolean":
        if (["Adv", "Modal", "Aux"].includes(type)) {
          return clamp(0.30 + support * 0.70);
        }
        break;

      case "evaluative":
        if (
          ["Modal", "Aux", "Adj", "Adv"].includes(type)
        ) {
          return clamp(0.32 + support * 0.68);
        }
        break;

      case "definition":
        if (
          ["Noun", "ProperNoun", "Adj"].includes(type)
        ) {
          return clamp(0.30 + support * 0.70);
        }
        break;

      case "factual_generation":
      case "imperative":
        if (this.isContentType(type)) {
          return clamp(support);
        }
        break;
    }

    return clamp(support * 0.40);
  }

  updateAnswerSlotState(
    state,
    tokenId,
    type,
    contextModel
  ) {
    const contract =
      contextModel?.answerContract;

    if (!state || !contract) return;

    const predicateBefore =
      state.slotPredicateEvidence || 0.0;

    const subjectTokens =
      contract.subjectTokens || [];

    for (
      let i = 0;
      i < subjectTokens.length && i < 30;
      i++
    ) {
      if (tokenId === subjectTokens[i]) {
        state.slotSubjectMask |= (1 << i);
      }
    }

    const requiredSubjectCount =
      Math.min(
        3,
        Math.max(1, subjectTokens.length)
      );

    let subjectHits = 0;
    let mask = state.slotSubjectMask || 0;

    while (mask) {
      subjectHits += mask & 1;
      mask >>>= 1;
    }

    state.slotSubjectEvidence =
      subjectTokens.length
        ? Math.max(
            state.slotSubjectEvidence || 0,
            Math.min(
              1,
              subjectHits / requiredSubjectCount
            )
          )
        : 1.0;

    const hadSubject =
      (state.slotSubjectEvidence || 0) >= 0.30;

    let predicateEvidence = 0.0;

    if (
      Number.isFinite(contract.relationVerbToken) &&
      tokenId === contract.relationVerbToken
    ) {
      predicateEvidence = 1.0;
    } else if (type === "Verb") {
      const intentSupport =
        Math.max(
          0,
          contextModel?.intentScores?.[tokenId] || 0
        );

      const semanticSupport =
        Math.max(
          0,
          contextModel?.semanticScores?.[tokenId] || 0
        );

      predicateEvidence =
        Math.min(
          0.90,
          0.58 +
          Math.max(
            intentSupport,
            semanticSupport
          ) * 0.32
        );
    } else if (
      ["Aux", "Modal"].includes(type)
    ) {
      predicateEvidence = 0.42;
    }

    if (predicateEvidence > 0) {
      state.slotPredicateEvidence =
        Math.max(
          state.slotPredicateEvidence || 0,
          predicateEvidence
        );

      if (hadSubject) {
        state.slotPredicateAfterSubject = true;
      }
    }

    // The first predicate token cannot simultaneously satisfy the requested
    // HOW/WHY/WHEN/WHERE information. The relation slot must come after it.
    if (predicateBefore >= 0.34) {
      const isMarker =
        Boolean(
          contract.markerTokenIds?.includes(tokenId)
        );

      let relationEvidence =
        this.getContractRelationTokenEvidence(
          tokenId,
          type,
          contextModel,
          state
        );

      if (!isMarker) {
        const roleFit =
          this.getRoleSensitiveKnowledgeEvidence(
            tokenId,
            state,
            contextModel,
            contract.kind === "causal"
              ? "cause"
              : contract.kind === "temporal"
                ? "time"
                : contract.kind === "locative"
                  ? "location"
                  : contract.kind === "quantitative"
                    ? "quantity"
                    : contract.kind === "definition"
                      ? "definition"
                      : contract.kind === "evaluative"
                        ? "evaluation"
                        : contract.kind === "mechanism"
                          ? "mechanism"
                          : "content"
          );

        relationEvidence *=
          (
            0.18 +
            roleFit.overall * 0.82
          );
      }

      const markerContentKinds =
        new Set([
          "mechanism",
          "causal",
          "temporal",
          "locative"
        ]);

      if (isMarker) {
        state.slotRelationOpened = true;
        state.slotRelationMarkerEvidence =
          Math.max(
            state.slotRelationMarkerEvidence || 0,
            relationEvidence
          );
      } else if (
        this.isContentType(type) ||
        type === "Num" ||
        type === "Adv"
      ) {
        state.slotRelationContentEvidence =
          Math.max(
            state.slotRelationContentEvidence || 0,
            relationEvidence
          );
      }

      if (
        markerContentKinds.has(contract.kind)
      ) {
        const marker =
          Math.max(
            0,
            Math.min(
              1,
              state.slotRelationMarkerEvidence || 0
            )
          );

        const content =
          Math.max(
            0,
            Math.min(
              1,
              state.slotRelationContentEvidence || 0
            )
          );

        // Marker alone cannot complete the slot. Relevant content can partially
        // answer without a marker, but marker + content can reach full credit.
        const opened =
          state.slotRelationOpened
            ? 1.0
            : 0.0;

        state.slotRelationEvidence =
          Math.max(
            state.slotRelationEvidence || 0,
            Math.min(
              1,
              content *
              (
                0.68 +
                opened * 0.32
              ) *
              (
                0.88 +
                marker * 0.12
              )
            )
          );
      } else if (relationEvidence > 0) {
        // Quantity/boolean/definition/evaluation slots can be fulfilled by
        // direct content such as a number, yes/no, or defining property.
        state.slotRelationEvidence =
          Math.max(
            state.slotRelationEvidence || 0,
            relationEvidence
          );
      }

      if (
        (state.slotRelationEvidence || 0) >= 0.18
      ) {
        state.slotRelationAfterPredicate = true;
      }
    }
  }

  getLiveAnswerSlotProgress(
    state,
    contextModel
  ) {
    const contract =
      contextModel?.answerContract;

    if (!contract?.isQuestion) {
      return {
        subject: 1.0,
        predicate: 1.0,
        relation: 1.0,
        order: 1.0,
        overall: 1.0
      };
    }

    const required =
      contract.requiredSlots || [];

    const subjectRequired =
      required.includes("subject");

    const predicateRequired =
      required.includes("predicate");

    const relationRequired =
      required.some(
        slot =>
          !["subject", "predicate"].includes(slot)
      );

    const subject =
      subjectRequired
        ? Math.max(
            0,
            Math.min(
              1,
              state?.slotSubjectEvidence || 0
            )
          )
        : 1.0;

    const predicate =
      predicateRequired
        ? Math.max(
            0,
            Math.min(
              1,
              state?.slotPredicateEvidence || 0
            )
          )
        : 1.0;

    const relation =
      relationRequired
        ? Math.max(
            0,
            Math.min(
              1,
              state?.slotRelationEvidence || 0
            )
          )
        : 1.0;

    let order = 1.0;

    if (
      subjectRequired &&
      predicateRequired &&
      !state?.slotPredicateAfterSubject
    ) {
      order *= 0.72;
    }

    if (
      relationRequired &&
      !state?.slotRelationAfterPredicate
    ) {
      order *= 0.68;
    }

    const values = [];

    if (subjectRequired) values.push(subject);
    if (predicateRequired) values.push(predicate);
    if (relationRequired) values.push(relation);

    const average =
      values.length
        ? values.reduce(
            (sum, value) => sum + value,
            0
          ) / values.length
        : 1.0;

    const weakest =
      values.length
        ? Math.min(...values)
        : 1.0;

    const overall =
      Math.max(
        0,
        Math.min(
          1,
          (
            average * 0.68 +
            weakest * 0.32
          ) *
          (
            0.76 +
            order * 0.24
          )
        )
      );

    return {
      subject,
      predicate,
      relation,
      order,
      overall
    };
  }

  answerSlotBonus(
    tokenId,
    generated,
    sentenceState,
    contextModel
  ) {
    const contract =
      contextModel?.answerContract;

    if (!contract?.isQuestion) return 0.0;

    const type =
      this.getCanonicalType(tokenId);

    const progress =
      this.getLiveAnswerSlotProgress(
        sentenceState,
        contextModel
      );

    let bonus = 0.0;

    const subjectIndex =
      contract.subjectTokens?.indexOf(tokenId) ?? -1;

    // SLOT 1: establish the subject.
    if (
      progress.subject < 0.70 &&
      subjectIndex >= 0
    ) {
      bonus +=
        Math.max(
          2.2,
          6.8 - subjectIndex * 0.9
        );
    }

    // SLOT 2: interrogative adverbs strongly expect an action/predicate.
    if (
      progress.subject >= 0.30 &&
      progress.predicate < 0.70
    ) {
      if (
        Number.isFinite(contract.relationVerbToken) &&
        tokenId === contract.relationVerbToken
      ) {
        bonus +=
          contract.operatorDemandsPredicate
            ? 10.0
            : 8.0;
      } else if (type === "Verb") {
        bonus +=
          contract.operatorDemandsPredicate
            ? 6.2
            : 4.2;
      } else if (
        ["Aux", "Modal"].includes(type)
      ) {
        bonus +=
          contract.operatorDemandsPredicate
            ? 2.0
            : 1.3;
      } else if (
        contract.operatorDemandsPredicate &&
        this.isContentType(type) &&
        subjectIndex < 0
      ) {
        bonus -= 2.4;
      }
    }

    // SLOT 3: supply the missing information requested by the operator.
    const relationRequired =
      contract.requiredSlots?.some(
        slot =>
          !["subject", "predicate"].includes(slot)
      );

    if (
      relationRequired &&
      progress.predicate >= 0.55 &&
      progress.relation < 0.72
    ) {
      const isMarker =
        Boolean(
          contract.markerTokenIds?.includes(tokenId)
        );

      const relationEvidence =
        this.getContractRelationTokenEvidence(
          tokenId,
          type,
          contextModel,
          sentenceState
        );

      if (isMarker) {
        // Useful structural opener, but intentionally much weaker than actual
        // mechanism/cause/time/location content.
        bonus += 1.8;
      } else {
        bonus +=
          relationEvidence * 7.4;
      }

      if (
        !isMarker &&
        relationEvidence < 0.14 &&
        this.isContentType(type) &&
        !contextModel?.promptContentSet?.has(tokenId)
      ) {
        bonus -= 2.8;
      }
    }

    return bonus;
  }

  getAnswerSlotDiagnostics(
    tokens,
    contextModel
  ) {
    const state =
      this.createRSLDecodeState();

    for (const token of tokens || []) {
      const word =
        this.tokenWord(token);

      const type =
        this.getCanonicalType(token);

      this.updateSentenceState(
        state,
        token,
        word,
        type
      );

      this.updateAnswerSlotState(
        state,
        token,
        type,
        contextModel
      );

      if (
        [".", "!", "?"].includes(word)
      ) {
        this.resetSentenceState(state);
      }
    }

    return this.getLiveAnswerSlotProgress(
      state,
      contextModel
    );
  }

  answerContractBonus(
    tokenId,
    generated,
    sentenceState,
    contextModel
  ) {
    const contract =
      contextModel?.answerContract;

    if (!contract?.requiresRelation) return 0.0;

    const type =
      this.getCanonicalType(tokenId);

    const exactPrompt =
      contextModel.promptTokenSet?.has(tokenId);

    const contractScore =
      contextModel.contractScores?.[tokenId] ?? 0.0;

    let bonus = 0.0;

    // Before the predicate, the contract stays weak so topic/subject formation
    // remains natural. After the predicate it becomes much more influential.
    if (sentenceState?.predicateReady) {
      bonus +=
        Math.max(0, contractScore) * 3.4;
    } else {
      bonus +=
        Math.max(0, contractScore) * 0.8;
    }

    switch (contract.kind) {
      case "quantitative":
        if (type === "Num") bonus += 5.0;
        break;

      case "temporal": {
        const temporal=this.temporalTokenEvidence(tokenId,contextModel);
        if(temporal.score>=0.90)bonus+=3.4; else if(temporal.score>=0.72)bonus+=2.2; else if(temporal.score>=0.55)bonus+=0.8; else if(this.isContentType(type))bonus-=0.8;
        break;
      }

      case "causal":
        if (
          sentenceState?.predicateReady &&
          ["Conj", "Prep"].includes(type)
        ) {
          bonus += 1.8;
        }
        break;

      case "mechanism":
        if (
          sentenceState?.predicateReady &&
          ["Verb", "Prep", "Adv"].includes(type)
        ) {
          bonus += 1.3;
        }
        break;

      case "boolean":
        // A yes/no question can be answered by a normal declarative clause.
        // Do not force literal "yes"/"no", but prefer a concise proposition.
        if (
          sentenceState?.predicateReady &&
          this.isContentType(type)
        ) {
          bonus += 0.6;
        }
        break;

      case "evaluative":
        if (["Modal", "Aux"].includes(type)) {
          bonus += 1.2;
        }
        break;

      case "factual_generation":
      case "imperative":
        // Requests for content should produce NEW content rather than echo the
        // user's request vocabulary.
        if (
          this.isContentType(type) &&
          !exactPrompt
        ) {
          bonus += 2.0;
        }
        break;

      case "definition":
        if (
          sentenceState?.predicateReady &&
          this.isContentType(type) &&
          !exactPrompt
        ) {
          bonus += 1.5;
        }
        break;
    }

    return bonus;
  }

  estimateContractFulfillment(
    tokens,
    contextModel,
    sentenceState = null
  ) {
    const contract =
      contextModel?.answerContract;

    if (!contract?.requiresRelation) {
      return 0.70;
    }

    let markerMax = 0.0;
    let markerSum = 0.0;
    let markerCount = 0;
    let numCount = 0;
    let verbCount = 0;
    let prepConjCount = 0;
    let modalAuxCount = 0;
    let contentCount = 0;
    let temporalEvidenceMax=0.0,temporalEvidenceSum=0.0,temporalEvidenceCount=0;

    const outputSet = new Set(tokens);

    for (const token of tokens) {
      const type =
        this.getCanonicalType(token);

      if (type === "Num") numCount++;
      if (type === "Verb") verbCount++;
      if (["Prep", "Conj"].includes(type)) {
        prepConjCount++;
      }
      if (["Modal", "Aux"].includes(type)) {
        modalAuxCount++;
      }

      if(contract.kind==="temporal"){
        const temporal=this.temporalTokenEvidence(token,contextModel);
        temporalEvidenceMax=Math.max(temporalEvidenceMax,temporal.score);
        if(temporal.score>=0.40){temporalEvidenceSum+=temporal.score;temporalEvidenceCount++;}
      }
      if (this.isContentType(type)) {
        contentCount++;

        const word = this.tokenWord(token);
        const stem = this.stemWord(word);

        const exact =
          contextModel.promptContentSet?.has(token);

        const stemMatch =
          stem.length >= 3 &&
          contextModel.promptStems?.has(stem);

        // Novelty quality is evaluated after this pass using the already
        // computed semantic routing tables.
      }

      const marker =
        contextModel.contractScores?.[token];

      if (Number.isFinite(marker)) {
        const support =
          Math.max(0, marker);

        markerMax =
          Math.max(markerMax, support);

        markerSum += support;
        markerCount++;
      }
    }

    const markerMean =
      markerCount
        ? markerSum / markerCount
        : 0.0;

    let subjectHit = 0;

    for (const token of contract.subjectTokens || []) {
      if (outputSet.has(token)) subjectHit++;
    }

    const subjectCoverage =
      contract.subjectTokens?.length
        ? subjectHit /
          contract.subjectTokens.length
        : this.promptCoverage(
            tokens,
            contextModel
          );

    const novelStats =
      this.getNovelContentStats(
        tokens,
        contextModel
      );

    const novelty =
      contentCount
        ? Math.max(
            0,
            Math.min(
              1,
              novelStats.aligned /
              Math.max(1, contentCount * 0.35)
            )
          )
        : 0.0;

    // Irrelevant extra content now hurts contract fulfillment instead of
    // accidentally helping it. The penalty compounds as drift accumulates.
    const alignmentIntegrity =
      contentCount
        ? Math.max(
            0,
            Math.min(
              1,
              1.0 -
              (novelStats.unaligned /
                Math.max(1, contentCount)) * 1.25
            )
          )
        : 0.0;

    const clauseReady =
      sentenceState
        ? (this.clauseComplete(sentenceState) ? 1.0 : 0.0)
        : 0.65;

    let shape = 0.55;

    switch (contract.kind) {
      case "temporal": {
        const temporalMean=temporalEvidenceCount?temporalEvidenceSum/temporalEvidenceCount:0.0;
        const mode=contextModel?.answerShape?.temporalMode||"direct";
        const modeSupport=mode==="predictive"?(modalAuxCount>0?0.86:0.54):0.78;
        shape=Math.max(0,Math.min(1,temporalEvidenceMax*0.62+temporalMean*0.20+modeSupport*0.12+markerMax*0.06));
        if(temporalEvidenceMax<0.55)shape*=0.42;
        break;
      }

      case "quantitative":
        shape = Math.max(
          markerMax,
          numCount > 0 ? 1.0 : 0.0
        );
        break;

      case "causal":
        shape = Math.max(
          markerMax,
          prepConjCount > 0 ? 0.72 : 0.0
        );
        break;

      case "mechanism":
        shape = Math.max(
          markerMax,
          Math.min(
            1,
            verbCount * 0.24 +
            prepConjCount * 0.20
          )
        );
        break;

      case "boolean":
        shape = Math.max(
          markerMax,
          clauseReady * 0.86
        );
        break;

      case "evaluative":
        shape = Math.max(
          markerMax,
          modalAuxCount > 0 ? 0.82 : 0.0
        );
        break;

      case "factual_generation":
        shape =
          novelty * 0.72 +
          Math.min(
            1,
            contentCount / 8
          ) * 0.28;
        break;

      case "definition":
        shape =
          clauseReady * 0.45 +
          novelty * 0.35 +
          markerMax * 0.20;
        break;

      case "imperative":
        shape =
          novelty * 0.60 +
          clauseReady * 0.25 +
          markerMax * 0.15;
        break;
    }

    const baseFulfillment = Math.max(
      0,
      Math.min(
        1,
        shape * 0.40 +
        subjectCoverage * 0.22 +
        novelty * 0.16 +
        markerMean * 0.08 +
        alignmentIntegrity * 0.14
      )
    );

    // A response cannot get a high contract score merely by having the right
    // answer "shape." It must stay connected to both the subject and the
    // question. These gates are cheap scalar arithmetic.
    const subjectGate =
      0.55 + 0.45 * subjectCoverage;

    const alignmentGate =
      0.42 + 0.58 * alignmentIntegrity;

    return Math.max(
      0,
      Math.min(
        1,
        baseFulfillment *
        subjectGate *
        alignmentGate
      )
    );
  }

  /**
   * Generic answer pressure for interrogative prompts.
   *
   * The query operator's learned embedding contributes through the intent
   * centroid. Generation is simply discouraged from echoing that operator and
   * rewarded for adding novel, semantically routed content after a predicate.
   */
  questionRelationBonus(tokenId, sentenceState, contextModel) {
    const profile = contextModel?.questionProfile;
    if (!profile?.isQuestion) return 0.0;

    if (tokenId === profile.operatorToken) {
      return sentenceState.isStart ? -16.0 : -4.0;
    }

    const type = this.getCanonicalType(tokenId);
    const exactPrompt = contextModel?.promptTokenSet?.has(tokenId);

    if (sentenceState.predicateReady && this.isContentType(type) && !exactPrompt) {
      const routed = this.getRoutedSemanticScore(tokenId, contextModel, sentenceState);
      return Math.max(0.0, routed) * 2.6;
    }

    return 0.0;
  }

  getNovelContentStats(tokens, contextModel) {
    let total = 0;
    let aligned = 0;
    let supportSum = 0.0;

    for (const tok of tokens) {
      const type = this.getCanonicalType(tok);
      if (!this.isContentType(type)) continue;

      const word = this.tokenWord(tok);
      const stem = this.stemWord(word);
      const exact = contextModel?.promptContentSet?.has(tok);
      const stemMatch =
        stem.length >= 3 &&
        contextModel?.promptStems?.has(stem);

      if (exact || stemMatch) continue;

      total++;

      // Reuse semantic tables already built once for the prompt. Contract
      // support only helps when it also connects back to the topic, preventing
      // random "time-like" or "cause-like" words from satisfying a question.
      const topic = Math.max(
        0,
        contextModel?.topicScores?.[tok] || 0
      );
      const intent = Math.max(
        0,
        contextModel?.intentScores?.[tok] || 0
      );
      const semantic = Math.max(
        0,
        contextModel?.semanticScores?.[tok] || 0
      );
      const contract = Math.max(
        0,
        contextModel?.contractScores?.[tok] || 0
      );

      const dual = Math.sqrt(topic * intent);
      const contractTopic = Math.sqrt(topic * contract);

      const support = Math.max(
        0,
        Math.min(
          1,
          Math.max(
            semantic,
            dual * 0.95,
            contractTopic * 0.85
          )
        )
      );

      supportSum += support;

      // Relevant novelty, not merely "a word that wasn't in the prompt."
      if (support >= 0.16) {
        aligned++;
      }
    }

    return {
      total,
      aligned,
      unaligned: Math.max(0, total - aligned),
      meanSupport:
        total > 0
          ? supportSum / total
          : 1.0
    };
  }

  countNovelContent(tokens, contextModel) {
    return this.getNovelContentStats(
      tokens,
      contextModel
    ).aligned;
  }

  answerRelationSatisfied(tokens, sentenceState, contextModel) {
    if (!this.clauseComplete(sentenceState)) return false;

    const profile = contextModel?.questionProfile;
    if (!profile?.isQuestion) return true;

    // A question answer should contribute something beyond simply parroting
    // the prompt. The learned semantic router decides WHAT that contribution
    // is; this function only checks that one exists.
    const novelContent = this.countNovelContent(tokens, contextModel);
    const contentCount = tokens.reduce((count, tok) => {
      return count + (this.isContentType(this.getCanonicalType(tok)) ? 1 : 0);
    }, 0);

    const requiredNovel = Math.max(
      1,
      Math.min(3, Math.ceil(contentCount * 0.18))
    );

    if (novelContent < requiredNovel) {
      return false;
    }

    const contractFulfillment =
      this.estimateContractFulfillment(
        tokens,
        contextModel,
        sentenceState
      );

    return contractFulfillment >= 0.40;
  }

  resetSentenceState(sentenceState) {
    sentenceState.isStart = true;
    sentenceState.subjectReady = false;
    sentenceState.predicateReady = false;
    sentenceState.subjectToken = null;
    sentenceState.predicateToken = null;
    sentenceState.lastContentToken = null;
    sentenceState.clauseContentCount = 0;
    sentenceState.clauseSemanticEMA = 0.72;
    sentenceState.clauseDriftCount = 0;
    sentenceState.verbCount = 0;
    sentenceState.nounCount = 0;
    sentenceState.objectCount = 0;
    sentenceState.prepCount = 0;
    sentenceState.isPluralSubject = false;

    sentenceState.subjectPronounWord = "";
    sentenceState.subjectPerson = null;
    sentenceState.subjectNumber = null;
    sentenceState.lexicalPredicateCount = 0;
    sentenceState.needsBaseVerb = false;
    sentenceState.needsObjectAfterPrep = false;
    sentenceState.needsNominalAfterDet = false;
    sentenceState.copulaOpen = false;
    sentenceState.consecutiveAdverbs = 0;
    sentenceState.trailingModifierCount = 0;
    sentenceState.postPredicateContentCount = 0;
  }

  inferRequestedLength(cleanPromptTokens) {
    const words = cleanPromptTokens.map(t => this.tokenWord(t).toLowerCase());

    for (let i = 0; i < words.length; i++) {
      if (!/^\d+$/.test(words[i])) continue;

      const n = Number(words[i]);
      if (!Number.isFinite(n) || n <= 0) continue;

      const nearby = words.slice(Math.max(0, i - 2), Math.min(words.length, i + 3));
      if (nearby.some(w => ["word", "words", "token", "tokens"].includes(w))) {
        return Math.max(1, Math.min(this.maxTokenCeiling, Math.round(n)));
      }
    }

    return null;
  }

  estimateAnswerLength(cleanPromptTokens, contextModel) {
    const explicit = this.inferRequestedLength(cleanPromptTokens);
    if (explicit !== null) return explicit;

    const promptLen = Math.max(1, cleanPromptTokens.length);
    const contentCount = contextModel?.promptContentSet?.size || 0;

    let structuralSignals = 0;

    for (const tok of cleanPromptTokens) {
      const type = this.getCanonicalType(tok);
      if (["Conj", "Prep", "Aux", "Modal"].includes(type)) {
        structuralSignals++;
      }
    }

    // Compact by default. Complexity grows the target continuously.
    let target =
      6 +
      Math.sqrt(promptLen) * 1.9 +
      contentCount * 1.15 +
      Math.sqrt(structuralSignals + 1) * 1.8;

    if (contextModel?.questionProfile?.isQuestion) {
      target += Math.sqrt(contentCount + 1) * 1.5;
    }

    // Large prompts still scale all the way toward the 360-token ceiling.
    target += Math.max(0, promptLen - 28) * 0.90;
    target += Math.max(0, promptLen - 80) * 1.10;
    target += Math.max(0, promptLen - 160) * 1.30;

    return Math.max(
      8,
      Math.min(this.maxTokenCeiling, Math.round(target))
    );
  }

  getAdaptiveLatentHops(inputTokenCount, targetLength) {
    const n = Math.max(1, inputTokenCount);
    const out = Math.max(1, targetLength);

    // Cheap prompts get one latent refinement. Complexity can grow this to 4.
    const pressure =
      (1.0 - Math.exp(-n / 34.0)) * 0.62 +
      (1.0 - Math.exp(-out / 150.0)) * 0.38;

    return Math.max(1, Math.min(4, 1 + Math.floor(pressure * 4.0)));
  }

  getAdaptiveGenerationBudget(inputTokenCount, targetLength, contextModel) {
    const explicit = contextModel?.explicitRequestedLength;

    if (Number.isFinite(explicit)) {
      return Math.max(
        1,
        Math.min(
          this.maxTokenCeiling,
          Math.ceil(explicit + Math.min(12, Math.sqrt(explicit) + 2))
        )
      );
    }

    // 360 is AVAILABLE, but a 5-token question should not silently receive a
    // 360-token CPU budget. The budget grows continuously with requested work.
    const slack =
      5 +
      Math.sqrt(Math.max(1, inputTokenCount)) * 1.8 +
      Math.sqrt(Math.max(1, targetLength)) * 1.7;

    return Math.max(
      10,
      Math.min(
        this.maxTokenCeiling,
        Math.ceil(targetLength + slack)
      )
    );
  }

  getLivePrefixDiagnostics(
    tokens,
    contextModel,
    sentenceState
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const eventQuery =
      this.computeEventQueryResolution(
        tokens,
        contextModel
      );

    const semanticCore =
      this.computeSemanticCoreResolution(
        tokens,
        contextModel,
        eventQuery
      );

    const selfWhy =
      this.getSelfWhyClause(
        tokens,
        contextModel,
        sentenceState,
        eventQuery,
        semanticCore
      );

    let lastContentToken = null;

    for (
      let i = (tokens?.length || 0) - 1;
      i >= 0;
      i--
    ) {
      if (
        this.isContentType(
          this.getCanonicalType(tokens[i])
        )
      ) {
        lastContentToken = tokens[i];
        break;
      }
    }

    let provenanceGrounding = 0.55;
    let provenanceRelevance = 0.55;
    let queryFit = 0.50;

    if (Number.isInteger(lastContentToken)) {
      const provenance =
        this.getPromptRootedProvenance(
          lastContentToken,
          contextModel
        );

      provenanceGrounding =
        clamp(provenance?.grounding);

      provenanceRelevance =
        clamp(
          Math.max(
            provenance?.relevance || 0,
            provenance?.grounding || 0
          )
        );

      queryFit =
        clamp(
          this.getEventQueryCandidateFit(
            lastContentToken,
            contextModel,
            sentenceState
          )
        );
    }

    const slotProgress =
      clamp(
        this.getLiveAnswerSlotProgress(
          sentenceState,
          contextModel
        )?.overall
      );

    const semanticThread =
      clamp(
        Number.isFinite(
          sentenceState?.clauseSemanticEMA
        )
          ? sentenceState.clauseSemanticEMA
          : 0.60
      );

    const dependencyIntegrity =
      (
        sentenceState?.needsBaseVerb ||
        sentenceState?.needsObjectAfterPrep ||
        sentenceState?.needsNominalAfterDet
      )
        ? 0.28
        : 1.0;

    const querySlotResolution =
      clamp(
        eventQuery
          ?.slotCompleteness
      );

    const quality =
      clamp(
        semanticCore.frameFit * 0.15 +
        semanticCore.queryResolution * 0.20 +
        semanticCore.relationCompleteness * 0.20 +
        semanticCore.grounding * 0.09 +
        eventQuery.contradictionIntegrity * 0.06 +
        queryFit * 0.05 +
        provenanceGrounding * 0.03 +
        provenanceRelevance * 0.02 +
        slotProgress * 0.02 +
        semanticThread * 0.01 +
        dependencyIntegrity * 0.02 +
        selfWhy.score * 0.15
      );

    return {
      quality,
      eventQuery,
      semanticCore,
      selfWhy,
      querySlotResolution,
      querySlotResolved:
        Boolean(
          eventQuery
            ?.querySlotResolved
        ),
      provenanceGrounding,
      provenanceRelevance,
      queryFit,
      slotProgress,
      semanticThread,
      dependencyIntegrity
    };
  }

  isPrefixCheckpointEligible(
    tokens,
    sentenceState,
    contextModel,
    live = null
  ) {
    if (
      !tokens ||
      tokens.length <
        this.reasoningConfig
          .prefixCheckpointMinLength
    ) {
      return false;
    }

    if (
      sentenceState?.needsBaseVerb ||
      sentenceState?.needsObjectAfterPrep ||
      sentenceState?.needsNominalAfterDet
    ) {
      return false;
    }

    const shape =
      contextModel?.answerShape?.kind ||
      "prose";

    if (shape === "list") {
      return false;
    }

    const diagnostics =
      live ||
      this.getLivePrefixDiagnostics(
        tokens,
        contextModel,
        sentenceState
      );

    if (
      diagnostics.quality <
        this.reasoningConfig
          .prefixCheckpointMinQuality
    ) {
      return false;
    }

    if (
      ["short", "yes_no"].includes(shape)
    ) {
      return true;
    }

    return Boolean(
      sentenceState?.predicateReady ||
      diagnostics.eventQuery?.score >= 0.54
    );
  }

  materializeBestPrefix(
    beam,
    force = false
  ) {
    const bestTokens =
      beam?.bestPrefixTokens;

    if (
      !Array.isArray(bestTokens) ||
      !bestTokens.length ||
      bestTokens.length >=
        (beam?.tokens?.length || 0)
    ) {
      return beam;
    }

    const bestQuality =
      Number.isFinite(beam?.bestPrefixQuality)
        ? beam.bestPrefixQuality
        : -Infinity;

    const liveQuality =
      Number.isFinite(beam?.liveQuality)
        ? beam.liveQuality
        : bestQuality;

    if (
      !force &&
      bestQuality - liveQuality <
        this.reasoningConfig.rollbackMargin
    ) {
      return beam;
    }

    const discarded =
      (beam.trajectory || []).slice(
        bestTokens.length
      );

    return {
      ...beam,
      tokens: bestTokens.slice(),
      trajectory:
        Array.isArray(
          beam.bestPrefixTrajectory
        )
          ? beam.bestPrefixTrajectory.slice()
          : (beam.trajectory || []).slice(
              0,
              bestTokens.length
            ),
      state:
        beam.bestPrefixState
          ? this.cloneRSLDecodeState(
              beam.bestPrefixState
            )
          : beam.state,
      sentenceCount:
        Number.isFinite(
          beam.bestPrefixSentenceCount
        )
          ? beam.bestPrefixSentenceCount
          : beam.sentenceCount,
      pathScore:
        Number.isFinite(
          beam.bestPrefixPathScore
        )
          ? beam.bestPrefixPathScore
          : beam.pathScore,
      liveQuality: bestQuality,
      bestPrefixResolved:
        Boolean(
          beam.bestPrefixResolved
        ),
      discardedTrajectory: [
        ...(beam.discardedTrajectory || []),
        ...discarded
      ],
      rolledBackTokens:
        (beam.rolledBackTokens || 0) +
        discarded.length
    };
  }

  shouldStopDeepAnswer(
    generated,
    sentenceState,
    contextModel,
    targetLength,
    sentenceCount,
    liveDiagnostics = null
  ) {
    const explicitLength =
      contextModel?.explicitRequestedLength;

    if (
      explicitLength !== null &&
      explicitLength !== undefined
    ) {
      return (
        generated.length >=
        Math.max(1, explicitLength)
      );
    }

    const ceilingReached =
      generated.length >=
      Math.max(1, targetLength);

    const shape =
      contextModel?.answerShape?.kind ||
      "prose";

    const eventQueryResolution =
      this.computeEventQueryResolution(
        generated,
        contextModel
      );

    const semanticCore =
      this.computeSemanticCoreResolution(
        generated,
        contextModel,
        eventQueryResolution
      );

    const live =
      liveDiagnostics ||
      this.getLivePrefixDiagnostics(
        generated,
        contextModel,
        sentenceState
      );

    const dependenciesClosed =
      !sentenceState?.needsBaseVerb &&
      !sentenceState?.needsObjectAfterPrep &&
      !sentenceState?.needsNominalAfterDet;

    const semanticLengthReady =
      generated.length >=
        this.reasoningConfig
          .semanticStopMinLength;

    const querySlotReady =
      Boolean(
        eventQueryResolution
          .querySlotResolved
      ) &&
      eventQueryResolution
        .slotCompleteness >=
        this.reasoningConfig
          .querySlotResolveFloor;

    const selfWhy =
      live.selfWhy ||
      this.getSelfWhyClause(
        generated,
        contextModel,
        sentenceState,
        eventQueryResolution,
        semanticCore
      );

    const selfWhyReady =
      selfWhy.score >=
        this.reasoningConfig.selfWhyClauseStopFloor &&
      selfWhy.relationFit >=
        this.reasoningConfig.selfWhyRelationStopFloor &&
      (
        !contextModel?.answerPlan?.factualityRequired ||
        selfWhy.evidenceSupport >=
          this.reasoningConfig.selfWhyEvidenceStopFloor
      );

    const answerCommitment = this.getLiveAnswerCommitment(generated, contextModel);
    const commitmentReady = !contextModel?.answerContract?.requiresCommitment || (answerCommitment.resolved && answerCommitment.strength >= this.reasoningConfig.answerCommitmentStopFloor);

    const resolutionReady =
      querySlotReady &&
      selfWhyReady &&
      commitmentReady &&
      semanticCore.resolved &&
      semanticCore.queryResolution >=
        this.reasoningConfig
          .querySlotResolveFloor &&
      semanticCore.relationCompleteness >=
        this.reasoningConfig
          .semanticGraphResolveFloor &&
      eventQueryResolution.score >=
        this.reasoningConfig
          .semanticStopResolutionFloor;

    const qualityReady =
      live.quality >=
        this.reasoningConfig
          .semanticStopQualityFloor;

    const answered =
      this.answerRelationSatisfied(
        generated,
        sentenceState,
        contextModel
      );

    if (shape === "list") {
      const targetItems =
        Math.max(
          2,
          contextModel
            ?.answerShape
            ?.targetItems || 3
        );

      if (
        sentenceCount >= targetItems &&
        dependenciesClosed
      ) {
        return true;
      }

      return ceilingReached;
    }

    if (
      shape === "short" ||
      shape === "yes_no"
    ) {
      if (
        generated.length >= 2 &&
        dependenciesClosed &&
        querySlotReady &&
        resolutionReady &&
        qualityReady
      ) {
        return true;
      }

      return ceilingReached;
    }

    if (shape === "temporal") {
      let temporalMax = 0.0;

      for (const token of generated) {
        temporalMax =
          Math.max(
            temporalMax,
            this.temporalTokenEvidence(
              token,
              contextModel
            ).score
          );
      }

      const minimum =
        contextModel?.answerShape
          ?.minimumTemporalEvidence ??
        0.68;

      if (
        temporalMax >= minimum &&
        semanticLengthReady &&
        dependenciesClosed &&
        querySlotReady &&
        resolutionReady &&
        qualityReady &&
        (
          answered ||
          eventQueryResolution.resolved
        )
      ) {
        return true;
      }

      return ceilingReached;
    }

    const coverage =
      this.promptCoverage(
        generated,
        contextModel
      );

    if (
      semanticLengthReady &&
      dependenciesClosed &&
      querySlotReady &&
      qualityReady &&
      resolutionReady &&
      (
        answered ||
        eventQueryResolution.resolved
      ) &&
      coverage >= 0.30
    ) {
      return true;
    }

    // Target length is only a hard ceiling. It is never a reason to keep
    // generating when the event/query has already been resolved.
    return ceilingReached;
  }

  buildUniversalPromptFrame(cleanPromptTokens, rawLogits, contextModel) {
    const vocab = this.network.vocab;
    const vocabSize = this.network.vocabSize;

    const promptSeq = cleanPromptTokens
      .filter(tok => this.getCanonicalType(tok) !== "Punct");

    if (!rawLogits || !vocab || vocabSize <= 0) return null;

    const candidateMap = new Map();

    // Prompt tokens retain their original source positions. For questions,
    // suppress the leading interrogative and do-support helper so Deep forms a
    // declarative answer rather than simply rebuilding the question.
    const qProfile = contextModel?.questionProfile;

    for (let i = 0; i < promptSeq.length; i++) {
      const tok = promptSeq[i];
      if (!vocab.hasToken(tok)) continue;

      if (qProfile?.isQuestion && tok === qProfile.operatorToken) {
        continue;
      }

      // An inverted helper before the subject belongs to the question form.
      // We let the neural candidate pool reintroduce it only if the answer
      // actually needs it.
      if (
        qProfile?.isQuestion &&
        qProfile.helperIndex >= 0 &&
        i === qProfile.helperIndex
      ) {
        continue;
      }

      if (!candidateMap.has(tok)) {
        candidateMap.set(tok, {
          token: tok,
          source: "prompt",
          promptIndex: i
        });
      }
    }

    // Add a compact neural candidate set from the already-pruned Deep pool.
    // The full vocabulary was scanned once when the pool was built.
    const neuralTop = [];
    const sourceIds = contextModel?.deepCandidatePool?.length
      ? contextModel.deepCandidatePool
      : Int32Array.from({ length: vocabSize }, (_, i) => i);

    for (const i of sourceIds) {
      if (!vocab.hasToken(i)) continue;

      const type = this.getCanonicalType(i);
      if (type === "Punct" || type === "Other") continue;

      let score = rawLogits[i];
      score += this.getRoutedSemanticScore(i, contextModel, null) * 1.15;

      if (neuralTop.length < 28) {
        neuralTop.push({ token: i, score });
        neuralTop.sort((a, b) => b.score - a.score);
      } else if (score > neuralTop[neuralTop.length - 1].score) {
        neuralTop[neuralTop.length - 1] = { token: i, score };
        neuralTop.sort((a, b) => b.score - a.score);
      }
    }

    for (const entry of neuralTop) {
      if (!candidateMap.has(entry.token)) {
        candidateMap.set(entry.token, {
          token: entry.token,
          source: "neural",
          promptIndex: -1
        });
      }
    }

    const candidates = Array.from(candidateMap.values());
    if (!candidates.length) return null;

    const freshState = () => ({
      isStart: true,
      subjectReady: false,
      predicateReady: false,
      subjectToken: null,
      predicateToken: null,
      lastContentToken: null,
      clauseContentCount: 0,
      clauseSemanticEMA: 0.72,
      clauseDriftCount: 0,
      verbCount: 0,
      nounCount: 0,
      prepCount: 0,
      objectCount: 0,
      isPluralSubject: false
    });

    const cloneState = s => ({
      isStart: s.isStart,
      subjectReady: s.subjectReady,
      predicateReady: s.predicateReady,
      subjectToken: s.subjectToken,
      predicateToken: s.predicateToken,
      lastContentToken: s.lastContentToken,
      clauseContentCount: s.clauseContentCount,
      clauseSemanticEMA: s.clauseSemanticEMA,
      clauseDriftCount: s.clauseDriftCount,
      verbCount: s.verbCount,
      nounCount: s.nounCount,
      prepCount: s.prepCount,
      objectCount: s.objectCount,
      isPluralSubject: s.isPluralSubject
    });

    const startTypeBonus = type => {
      if (type === "ProperNoun" || type === "Pronoun") return 3.4;
      if (type === "Noun") return 3.1;
      if (type === "Det") return 2.7;
      if (type === "Adj") return 2.3;
      if (type === "Adv") return 0.6;
      if (type === "Verb") return 0.3;
      return -3.0;
    };

    const roleBonus = (type, state) => {
      if (!state.subjectReady) {
        if (["Noun", "ProperNoun", "Pronoun"].includes(type)) return 4.2;
        if (["Det", "Adj"].includes(type)) return 2.1;
        return -1.2;
      }

      if (!state.predicateReady) {
        if (["Verb", "Modal", "Aux"].includes(type)) return 4.8;
        if (["Noun", "ProperNoun", "Adj"].includes(type) && state.nounCount < 3) return 1.2;
        return -1.0;
      }

      if (state.objectCount === 0) {
        if (["Noun", "ProperNoun", "Pronoun"].includes(type)) return 3.3;
        if (["Det", "Adj", "Prep", "Adv"].includes(type)) return 1.6;
        if (["Verb", "Modal", "Aux"].includes(type)) return -4.5;
      }

      if (state.objectCount >= 1) {
        if (type === "Prep") return 1.3;
        if (["Adj", "Adv"].includes(type)) return 0.6;
        if (["Verb", "Modal", "Aux"].includes(type)) return -5.5;
      }

      return 0.0;
    };

    // General transition logic for the beam. The only extension over the base
    // DFA is allowing short nominal compounds before the predicate.
    const beamTransitionLegal = (beam, word, type) => {
      const state = beam.state;
      const prevToken = beam.tokens.length
        ? beam.tokens[beam.tokens.length - 1]
        : null;
      const prevType = prevToken !== null
        ? this.getCanonicalType(prevToken)
        : null;

      if (state.isStart) {
        return ["Det", "Adj", "Noun", "ProperNoun", "Pronoun", "Adv", "Verb"].includes(type);
      }

      if (
        !state.predicateReady &&
        state.nounCount < 3 &&
        ["Noun", "ProperNoun", "Adj"].includes(prevType) &&
        ["Noun", "ProperNoun"].includes(type)
      ) {
        return true;
      }

      if (
        state.predicateReady &&
        prevType !== "Conj" &&
        ["Verb", "Modal", "Aux"].includes(type)
      ) {
        return false;
      }

      return this.isTransitionLegal(beam.tokens, word, type, state, true);
    };

    const maxWords = Math.max(
      4,
      Math.min(14, Math.max(7, Math.round(promptSeq.length * 0.80)))
    );
    const beamWidth = 10;

    let beams = [{
      tokens: [],
      state: freshState(),
      score: 0,
      used: new Set(),
      lastPromptIndex: -1,
      promptUsed: 0
    }];

    const completed = [];

    for (let depth = 0; depth < maxWords; depth++) {
      const expanded = [];

      for (const beam of beams) {
        for (const cand of candidates) {
          if (beam.used.has(cand.token)) continue;

          const word = this.tokenWord(cand.token);
          const type = this.getCanonicalType(cand.token);
          if (!word || type === "Punct" || type === "Other") continue;
          if (!beamTransitionLegal(beam, word, type)) continue;

          const nextState = cloneState(beam.state);
          const nextTokens = [...beam.tokens, cand.token];

          let score = beam.score;

          // Neural support from the current deep pass.
          score += Math.max(-6, Math.min(6, rawLogits[cand.token])) * 0.28;

          // Broad route plus a cheap clause-semantic lane.
          const beamRouted =
            this.getRoutedSemanticScore(
              cand.token,
              contextModel,
              beam.state
            );

          score += beamRouted * 1.35;

          const beamClause =
            this.getClauseSemanticEvidence(
              cand.token,
              beam.state
            );

          if (beamClause.locked) {
            score +=
              (beamClause.evidence - 0.50) * 4.2;

            if (
              this.isContentType(type) &&
              beamClause.evidence < 0.15 &&
              beamRouted < 0.28
            ) {
              continue;
            }
          } else {
            score +=
              (beamClause.evidence - 0.50) * 1.5;
          }

          // Universal grammatical-role pressure.
          score += beam.tokens.length === 0
            ? startTypeBonus(type)
            : roleBonus(type, beam.state);

          // A question changes what constitutes a useful answer. "How" routes
          // toward mechanisms, "why" toward causes, "where" toward locations,
          // etc., without hardcoding the factual content.
          score += this.questionRelationBonus(
            cand.token,
            beam.state,
            contextModel
          );

          // Existing POS transition matrix still matters.
          if (beam.tokens.length) {
            const prevType = this.getCanonicalType(
              beam.tokens[beam.tokens.length - 1]
            );
            score += this.getTransitionRule(prevType, type).weight * 0.75;
          }

          let nextPromptUsed = beam.promptUsed;
          let nextLastPromptIndex = beam.lastPromptIndex;

          // Prefer prompt evidence over invented neural filler.
          if (cand.source === "prompt") {
            score += 3.0;
            nextPromptUsed++;

            if (beam.lastPromptIndex >= 0) {
              const jump = cand.promptIndex - beam.lastPromptIndex;

              // Prompt order helps, but reordering is allowed whenever grammar
              // makes the original order unsuitable for an answer clause.
              if (jump >= 0 && jump <= 3) score += 1.0;
              else if (jump < 0) {
                score -= Math.min(1.2, Math.abs(jump) * 0.15);
              } else {
                score -= Math.min(0.8, jump * 0.08);
              }
            }

            nextLastPromptIndex = cand.promptIndex;
          } else {
            score -= 0.8;
          }

          // Hard clause-local morphological family veto.
          // Do not permit constructions like "computer computing" or
          // "reasons reasoned" to survive beam expansion.
          if (
            this.hasMorphologicalFamilyRepeat(
              cand.token,
              beam.tokens
            )
          ) {
            continue;
          }

          this.updateSentenceState(
            nextState,
            cand.token,
            word,
            type
          );

          const nextUsed = new Set(beam.used);
          nextUsed.add(cand.token);

          const nextBeam = {
            tokens: nextTokens,
            state: nextState,
            score,
            used: nextUsed,
            lastPromptIndex: nextLastPromptIndex,
            promptUsed: nextPromptUsed
          };

          if (
            this.answerRelationSatisfied(nextTokens, nextState, contextModel) &&
            nextTokens.length >= 3
          ) {
            const promptRatio =
              nextPromptUsed / Math.max(1, nextTokens.length);

            const completeScore =
              score +
              promptRatio * 4.0 +
              (nextState.objectCount > 0 ? 1.2 : 0.0) -
              Math.max(0, nextTokens.length - 9) * 0.35;

            completed.push({
              ...nextBeam,
              score: completeScore
            });
          }

          expanded.push(nextBeam);
        }
      }

      if (!expanded.length) break;

      expanded.sort((a, b) => b.score - a.score);
      beams = expanded.slice(0, beamWidth);

      // Anytime stop when a short prompt-heavy clause is decisively ahead.
      if (completed.length) {
        completed.sort((a, b) => b.score - a.score);

        const bestComplete = completed[0];
        const bestActive = beams[0];

        if (
          bestComplete.tokens.length >= 4 &&
          bestComplete.promptUsed >= 2 &&
          bestComplete.score >= bestActive.score + 2.0
        ) {
          break;
        }
      }
    }

    if (!completed.length) return null;

    completed.sort((a, b) => b.score - a.score);
    const winner = completed[0];

    const tokens = [...winner.tokens];
    if (tokens.length) {
      tokens[0] = this.sentenceCaseToken(tokens[0]);
    }

    return {
      kind: "universal-pos-beam",
      tokens,
      complete: true,
      sourceTokenCount: winner.promptUsed,
      plannerScore: winner.score
    };
  }

  buildFrameTrajectory(frameTokens, rawLogits) {
    if (!frameTokens?.length) return [];
    let maxLogit = -Infinity;
    if (rawLogits) {
      for (let i = 0; i < rawLogits.length; i++) {
        if (rawLogits[i] > maxLogit) maxLogit = rawLogits[i];
      }
    }

    const trajectory = [];
    for (const tok of frameTokens) {
      let prob = 0.70;
      if (rawLogits && Number.isFinite(rawLogits[tok]) && Number.isFinite(maxLogit)) {
        // Prompt-frame support: a token that is close to the strongest neural
        // logit gets high support, while weakly supported copied tokens remain
        // modest. This is evidence, not an automatic confidence bonus.
        const delta = Math.max(-8.0, Math.min(0.0, rawLogits[tok] - maxLogit));
        prob = Math.max(0.50, Math.min(0.88, 0.58 + 0.30 * Math.exp(delta)));
      }
      trajectory.push({ token: tok, prob, margin: 0.0, scaffold: true });
    }
    return trajectory;
  }

  extractContextModel(cleanPromptTokens) {
    const embDim = this.network.embDim;
    const vocabSize = this.network.vocabSize;
    const vocab = this.network.vocab;

    if (!this.normalizedEmbeddings) {
      this.initNormalizedEmbeddings();
    }

    const broadCentroid = new Float32Array(embDim);
    const centroid = new Float32Array(embDim);
    const topicCentroid = new Float32Array(embDim);
    const intentCentroid = new Float32Array(embDim);

    const questionProfile =
      this.analyzePromptStructure(cleanPromptTokens);

    const promptTokenSet = new Set(cleanPromptTokens);
    const promptContentSet = new Set();
    const promptContentOrder = new Map();
    const promptContentTokens = [];
    const promptStems = new Set();
    const promptTransitions = new Map();

    // --------------------------------------------------------
    // Count token frequency first. Repeated words should not
    // gain importance simply because the prompt repeats them.
    // --------------------------------------------------------
    const tokenFreq = new Map();

    for (const tok of cleanPromptTokens) {
      tokenFreq.set(tok, (tokenFreq.get(tok) || 0) + 1);
    }

    const contentEntries = [];
    const topicEntries = [];
    const intentEntries = [];

    const typeImportance = type => {
      switch (type) {
        case "ProperNoun": return 1.45;
        case "Verb":       return 1.38;
        case "Noun":       return 1.30;
        case "Num":        return 1.05;
        case "Adj":        return 0.90;
        case "Adv":        return 0.78;
        case "Modal":      return 0.72;
        case "Aux":        return 0.62;
        case "Prep":       return 0.34;
        case "Pronoun":    return 0.28;
        default:           return 0.12;
      }
    };

    // Keep the strongest occurrence of each unique token as an
    // anchor candidate. This makes the anchor count grow
    // sublinearly with input size instead of one-for-one.
    const bestEntryByToken = new Map();

    for (let i = 0; i < cleanPromptTokens.length; i++) {
      const tok = cleanPromptTokens[i];
      const off = (tok % vocabSize) * embDim;
      const type = this.getCanonicalType(tok);
      const word = vocab?.idToToken?.[tok] || "";

      const position =
        i / Math.max(1, cleanPromptTokens.length - 1);

      // Broad signal preserves sentence-wide context, but only weakly.
      const broadWeight = 0.90 + 0.10 * position;

      for (let d = 0; d < embDim; d++) {
        broadCentroid[d] +=
          this.normalizedEmbeddings[off + d] * broadWeight;
      }

      const freq = tokenFreq.get(tok) || 1;
      const frequencyPenalty = 1.0 / Math.sqrt(freq);

      let importance =
        typeImportance(type) *
        frequencyPenalty *
        (0.94 + 0.12 * position);

      // A detected query operator describes what information is missing,
      // but usually should not dominate the semantic subject of the answer.
      if (
        questionProfile?.isQuestion &&
        tok === questionProfile.operatorToken
      ) {
        importance *= 0.30;
      }

      const entry = {
        tok,
        type,
        word,
        index: i,
        importance
      };

      const existing = bestEntryByToken.get(tok);
      if (!existing || importance > existing.importance) {
        bestEntryByToken.set(tok, entry);
      }

      if (this.isContentType(type)) {
        promptContentSet.add(tok);

        if (!promptContentOrder.has(tok)) {
          promptContentOrder.set(
            tok,
            promptContentTokens.length
          );
          promptContentTokens.push(tok);
        }

        const stem = this.stemWord(word);
        if (stem.length >= 3) promptStems.add(stem);
      }

      if (i + 1 < cleanPromptTokens.length) {
        const next = cleanPromptTokens[i + 1];
        let set = promptTransitions.get(tok);

        if (!set) {
          set = new Set();
          promptTransitions.set(tok, set);
        }

        set.add(next);
      }
    }

    this.normalizeVector(broadCentroid);

    for (const entry of bestEntryByToken.values()) {
      if (this.isContentType(entry.type)) {
        contentEntries.push(entry);
      }

      if (
        ["Noun", "ProperNoun", "Adj", "Num"].includes(entry.type)
      ) {
        topicEntries.push(entry);
      }

      if (
        ["Verb", "Modal", "Aux", "Adv", "Prep"].includes(entry.type)
      ) {
        intentEntries.push(entry);
      }
    }

    const rankEntries = entries =>
      entries.sort((a, b) => {
        if (b.importance !== a.importance) {
          return b.importance - a.importance;
        }
        return a.index - b.index;
      });

    rankEntries(contentEntries);
    rankEntries(topicEntries);
    rankEntries(intentEntries);

    // Anchor count scales with sqrt(unique-content-count), not linearly.
    // A 5-token and a 30-token prompt can therefore receive comparable
    // confidence when both answers capture their meaningful concepts.
    const uniqueContentCount = contentEntries.length;

    const anchorBudget = Math.max(
      2,
      Math.min(
        12,
        Math.ceil(
          1.5 + 2.15 * Math.sqrt(Math.max(1, uniqueContentCount))
        )
      )
    );

    const topicBudget = Math.max(
      1,
      Math.min(
        8,
        Math.ceil(
          1.0 + 1.75 * Math.sqrt(Math.max(1, topicEntries.length))
        )
      )
    );

    const intentBudget = Math.max(
      1,
      Math.min(
        7,
        Math.ceil(
          1.0 + 1.65 * Math.sqrt(Math.max(1, intentEntries.length))
        )
      )
    );

    const promptAnchorEntries =
      contentEntries.slice(0, anchorBudget);

    const topicAnchorEntries =
      topicEntries.slice(0, topicBudget);

    const intentAnchorEntries =
      intentEntries.slice(0, intentBudget);

    // Weighted anchor maps are used by promptCoverage().
    const promptAnchorWeights = new Map();
    const promptAnchorTokens = [];
    const promptAnchorStems = new Map();

    for (const entry of promptAnchorEntries) {
      // Normalize each anchor's contribution so no single proper noun can
      // monopolize coverage while still preserving relative importance.
      const weight = Math.max(
        0.45,
        Math.min(1.65, entry.importance)
      );

      promptAnchorWeights.set(entry.tok, weight);
      promptAnchorTokens.push(entry.tok);

      const stem = this.stemWord(entry.word);
      if (stem.length >= 3) {
        const oldWeight = promptAnchorStems.get(stem) || 0;
        promptAnchorStems.set(
          stem,
          Math.max(oldWeight, weight)
        );
      }
    }

    const accumulateEntries = (
      target,
      entries,
      fallback = broadCentroid
    ) => {
      let weightSum = 0.0;

      for (const entry of entries) {
        const off = (entry.tok % vocabSize) * embDim;
        const weight = Math.max(
          0.10,
          Math.min(1.75, entry.importance)
        );

        for (let d = 0; d < embDim; d++) {
          target[d] +=
            this.normalizedEmbeddings[off + d] * weight;
        }

        weightSum += weight;
      }

      if (weightSum > 0) {
        for (let d = 0; d < embDim; d++) {
          target[d] /= weightSum;
        }
        this.normalizeVector(target);
      } else {
        target.set(fallback);
      }
    };

    accumulateEntries(
      topicCentroid,
      topicAnchorEntries
    );

    accumulateEntries(
      intentCentroid,
      intentAnchorEntries
    );

    // General centroid = mostly important anchors + a small amount of broad
    // context. This prevents long prompts from smearing the latent direction.
    const anchorCentroid = new Float32Array(embDim);
    accumulateEntries(
      anchorCentroid,
      promptAnchorEntries
    );

    for (let d = 0; d < embDim; d++) {
      centroid[d] =
        anchorCentroid[d] * 0.76 +
        broadCentroid[d] * 0.24;
    }
    this.normalizeVector(centroid);

    // If one semantic route had no evidence, use the sharpened general route.
    if (!topicAnchorEntries.length) {
      topicCentroid.set(centroid);
    }

    if (!intentAnchorEntries.length) {
      intentCentroid.set(centroid);
    }

    const answerContract =
      this.buildAnswerContract(
        cleanPromptTokens,
        questionProfile
      );

    const answerShape =
      this.detectAnswerShape(
        cleanPromptTokens,
        questionProfile,
        answerContract
      );

    const eventFrame =
      this.buildEventFrame(
        cleanPromptTokens,
        questionProfile,
        answerContract
      );

    const queryFrame =
      this.buildQueryFrame(
        answerContract,
        answerShape,
        eventFrame
      );

    const clauseGraph =
      this.buildSemanticClauseGraph(
        cleanPromptTokens,
        questionProfile,
        answerContract
      );

    const semanticFrame =
      this.routeSemanticFrame(
        cleanPromptTokens,
        questionProfile,
        answerContract,
        answerShape,
        eventFrame,
        queryFrame,
        clauseGraph
      );

    // Existing APIs can keep using queryFrame, but V19's semantic router owns
    // the actual missing relation.
    queryFrame.semanticTarget =
      semanticFrame.queryTarget;

    const instructionContract =
      this.compileInstructionContract(
        cleanPromptTokens,
        questionProfile,
        answerContract,
        answerShape,
        promptAnchorTokens
      );

    const answerPlan =
      this.buildAnswerPlan(
        instructionContract,
        answerShape
      );

    const markerCentroid =
      this.buildMarkerCentroid(
        answerContract.markerWords
      );

    const contractCentroid =
      new Float32Array(embDim);

    if (markerCentroid) {
      for (let d = 0; d < embDim; d++) {
        contractCentroid[d] =
          markerCentroid[d] * 0.58 +
          intentCentroid[d] * 0.27 +
          topicCentroid[d] * 0.15;
      }
      this.normalizeVector(contractCentroid);
    } else {
      contractCentroid.set(intentCentroid);
    }

    const topicScores = new Float32Array(vocabSize);
    const intentScores = new Float32Array(vocabSize);
    const semanticScores = new Float32Array(vocabSize);
    const contractScores = new Float32Array(vocabSize);

    for (let i = 0; i < vocabSize; i++) {
      if (!vocab?.hasToken?.(i)) continue;

      const topic =
        this.dotTokenVector(i, topicCentroid);

      const intent =
        this.dotTokenVector(i, intentCentroid);

      topicScores[i] = topic;
      intentScores[i] = intent;

      contractScores[i] =
        this.dotTokenVector(
          i,
          contractCentroid
        );

      semanticScores[i] = Math.max(
        -1.0,
        Math.min(
          1.0,
          topic * 0.58 + intent * 0.42
        )
      );
    }

    return {
      centroid,
      broadCentroid,
      topicCentroid,
      intentCentroid,
      topicScores,
      intentScores,
      semanticScores,
      answerContract,
      answerShape,
      eventFrame,
      queryFrame,
      clauseGraph,
      semanticFrame,
      instructionContract,
      answerPlan,
      contractCentroid,
      contractScores,

      reasoningCentroid: null,
      reasoningScores: null,
      reasoningStability: 0.0,
      reasoningHops: 0,

      questionProfile,
      explicitRequestedLength:
        this.inferRequestedLength(cleanPromptTokens),

      promptTokenSet,
      promptContentSet,
      promptContentOrder,
      promptContentTokens,
      promptStems,
      promptTransitions,

      // Complexity-normalized anchor data.
      promptAnchorWeights,
      promptAnchorTokens,
      promptAnchorStems,
      anchorBudget,
      uniqueContentCount,

      // Prompt-local caches. They are discarded after the answer and avoid
      // repeating provenance work across beam candidates and final verification.
      provenanceCache: new Map(),
      semanticGraphCandidateCache:
        new Map()
    };
  }

  getTransitionRule(prevType, curType) {
    if (!prevType) return { legal: true, weight: 0.0 };
    const key = `${prevType}->${curType}`;
    return this.transitionMatrix[key] || { legal: false, weight: -100.0 };
  }

  clauseComplete(sentenceState) {
    return Boolean(
      sentenceState?.subjectReady &&
      sentenceState?.predicateReady &&
      !sentenceState?.needsBaseVerb &&
      !sentenceState?.needsObjectAfterPrep &&
      !sentenceState?.needsNominalAfterDet &&
      !sentenceState?.copulaOpen
    );
  }

  promptCoverage(generated, contextModel) {
    const anchorWeights =
      contextModel?.promptAnchorWeights;

    const anchors =
      contextModel?.instructionContract
        ?.sourceAnchorTokens ||
      contextModel?.promptAnchorTokens ||
      [];

    if (!anchors.length) {
      return 0.0;
    }

    const generatedContent =
      (generated || []).filter(
        token =>
          this.isContentType(
            this.getCanonicalType(token)
          )
      );

    if (!generatedContent.length) {
      return 0.0;
    }

    const vocab = this.network?.vocab;
    let matchedWeight = 0.0;
    let totalWeight = 0.0;

    for (const anchor of anchors) {
      const weight =
        anchorWeights?.get(anchor) ||
        1.0;

      totalWeight += weight;
      let bestSupport = 0.0;

      for (const token of generatedContent) {
        let tokenSupport = 0.0;

        if (
          typeof vocab?.getAnchorSupport === "function"
        ) {
          const direct =
            vocab.getAnchorSupport(
              anchor,
              token
            );

          tokenSupport =
            direct?.semanticSupport || 0.0;
        } else {
          const link =
            this.getVocabKnowledgeLink(
              anchor,
              token
            );

          tokenSupport =
            Math.max(
              0,
              Math.min(
                1,
                link?.score || 0.55
              )
            );
        }

        // Surface copying demonstrates some grounding but never earns full
        // semantic coverage. A novel, supported concept can score higher.
        if (
          token === anchor ||
          vocab?.sameMorphologicalFamily?.(
            anchor,
            token
          )
        ) {
          tokenSupport = Math.min(
            tokenSupport,
            token === anchor
              ? 0.64
              : 0.58
          );
        }

        bestSupport = Math.max(
          bestSupport,
          tokenSupport
        );
      }

      matchedWeight +=
        weight *
        Math.max(
          0,
          Math.min(1, bestSupport)
        );
    }

    if (totalWeight <= 0) return 0.0;

    return Math.max(
      0,
      Math.min(
        1,
        matchedWeight / totalWeight
      )
    );
  }

  getHumanTransitionAssessment(
    generatedTokens,
    candidateToken,
    curWord,
    curType,
    sentenceState,
    contextModel = null
  ) {
    const lowerCur =
      String(curWord || "")
        .toLowerCase();

    const prevToken =
      generatedTokens.length
        ? generatedTokens[
            generatedTokens.length - 1
          ]
        : null;

    const prevWord =
      prevToken !== null
        ? this.tokenWord(
            prevToken
          ).toLowerCase()
        : "";

    const prevType =
      prevToken !== null
        ? this.getCanonicalType(
            prevToken
          )
        : null;

    const pronoun =
      curType === "Pronoun"
        ? this.getPronounProfile(
            lowerCur
          )
        : null;

    let legal = true;
    let score = 0.0;
    let quality = 0.62;
    let good = 0.0;

    const markGood =
      (amount = 1.0) => {
        good += amount;
      };

    const intensifiers =
      new Set([
        "very", "really", "quite",
        "rather", "extremely", "highly",
        "more", "less", "most",
        "almost", "nearly"
      ]);

    // --------------------------------------------------------
    // Pronoun case.
    // --------------------------------------------------------
    if (
      sentenceState.isStart &&
      curType === "Pronoun"
    ) {
      if (
        pronoun &&
        !pronoun.subject
      ) {
        legal = false;
        quality = 0.0;
      } else {
        quality = 0.92;
        markGood();
      }
    }

    if (
      prevType === "Prep" &&
      curType === "Pronoun"
    ) {
      if (
        pronoun &&
        !pronoun.object
      ) {
        legal = false;
        quality = 0.0;
      } else {
        quality = 0.96;
        markGood();
      }
    }

    if (
      sentenceState.predicateReady &&
      curType === "Pronoun" &&
      prevType !== "Conj"
    ) {
      if (
        pronoun &&
        pronoun.subject &&
        !pronoun.object
      ) {
        legal = false;
        quality = 0.0;
      } else {
        quality = Math.max(
          quality,
          0.88
        );
        markGood();
      }
    }

    // Possessive determiners behave like determiners even if vocab metadata
    // labels them as pronouns.
    if (
      curType === "Pronoun" &&
      pronoun?.possessive &&
      !pronoun.object
    ) {
      if (
        ![
          "Noun", "ProperNoun",
          "Adj"
        ].includes(prevType) &&
        !sentenceState.isStart &&
        prevType !== "Prep"
      ) {
        score -= 1.0;
      }
    }

    // --------------------------------------------------------
    // Open grammatical dependencies.
    // --------------------------------------------------------
    if (
      sentenceState.needsBaseVerb
    ) {
      if (curType === "Verb") {
        quality = 0.98;
        score += 2.4;
        markGood();
      } else if (
        curType === "Adv" &&
        (sentenceState.consecutiveAdverbs || 0) < 1
      ) {
        quality = 0.68;
      } else {
        legal = false;
        quality = 0.0;
      }
    }

    if (
      sentenceState.needsObjectAfterPrep
    ) {
      const validObject =
        [
          "Det", "Adj", "Noun",
          "ProperNoun", "Pronoun",
          "Num"
        ].includes(curType);

      if (!validObject) {
        legal = false;
        quality = 0.0;
      } else if (
        curType === "Pronoun" &&
        pronoun &&
        !pronoun.object
      ) {
        legal = false;
        quality = 0.0;
      } else {
        quality = Math.max(
          quality,
          0.88
        );

        if (
          [
            "Noun", "ProperNoun",
            "Pronoun", "Num"
          ].includes(curType)
        ) {
          markGood();
        }
      }
    }

    if (
      sentenceState.needsNominalAfterDet
    ) {
      if (
        ![
          "Adj", "Noun",
          "ProperNoun", "Num"
        ].includes(curType)
      ) {
        legal = false;
        quality = 0.0;
      } else {
        quality = Math.max(
          quality,
          curType === "Adj"
            ? 0.78
            : 0.96
        );

        if (
          [
            "Noun", "ProperNoun",
            "Num"
          ].includes(curType)
        ) {
          markGood();
        }
      }
    }

    // --------------------------------------------------------
    // Subject/auxiliary agreement.
    // --------------------------------------------------------
    if (
      curType === "Aux" &&
      sentenceState.subjectPronounWord
    ) {
      const agreement =
        this.pronounAuxAgreement(
          sentenceState.subjectPronounWord,
          lowerCur
        );

      if (agreement <= 0.01) {
        legal = false;
        quality = 0.0;
      } else if (agreement >= 0.95) {
        score += 1.2;
        quality = Math.max(
          quality,
          0.98
        );
        markGood();
      }
    }

    // --------------------------------------------------------
    // Human-readable adjacency rules.
    // --------------------------------------------------------
    if (
      prevType === "Adv" &&
      curType === "Adv"
    ) {
      if (
        intensifiers.has(prevWord) &&
        !intensifiers.has(lowerCur)
      ) {
        quality = Math.min(
          quality,
          0.72
        );
      } else {
        legal = false;
        quality = 0.05;
      }
    }

    if (
      prevType === "Adv" &&
      ["Noun", "ProperNoun"].includes(
        curType
      )
    ) {
      // "usually probabilities" / "perhaps percentage" is exactly the sort of
      // locally legal but human-undecodable transition RSL used to accept.
      score -= 5.0;
      quality = Math.min(
        quality,
        0.20
      );

      if (
        sentenceState.predicateReady &&
        sentenceState.objectCount >= 1
      ) {
        legal = false;
      }
    }

    if (
      sentenceState.predicateReady &&
      (sentenceState.trailingModifierCount || 0) >= 2 &&
      ["Adv", "Adj"].includes(curType)
    ) {
      legal = false;
      quality = 0.10;
    }

    if (
      this.clauseComplete(
        sentenceState
      ) &&
      (sentenceState.postPredicateContentCount || 0) >= 5 &&
      ![
        "Punct", "Prep", "Conj"
      ].includes(curType)
    ) {
      legal = false;
      quality = 0.10;
    }

    // Strong ordinary proposition transitions earn the user's requested
    // positive +0.5 "good" signal later in buildScores().
    if (
      sentenceState.subjectReady &&
      !sentenceState.predicateReady &&
      ["Verb", "Modal", "Aux"].includes(
        curType
      )
    ) {
      quality = Math.max(
        quality,
        0.94
      );
      markGood();
    }

    if (
      sentenceState.predicateReady &&
      sentenceState.objectCount === 0 &&
      [
        "Det", "Noun",
        "ProperNoun", "Pronoun",
        "Prep"
      ].includes(curType)
    ) {
      quality = Math.max(
        quality,
        0.84
      );
      markGood(0.7);
    }

    if (
      prevType === "Det" &&
      ["Noun", "ProperNoun"].includes(
        curType
      )
    ) {
      quality = Math.max(
        quality,
        0.97
      );
      markGood();
    }

    if (
      prevType === "Prep" &&
      [
        "Noun", "ProperNoun",
        "Pronoun", "Num"
      ].includes(curType)
    ) {
      quality = Math.max(
        quality,
        0.96
      );
      markGood();
    }

    // --------------------------------------------------------
    // Conversational perspective.
    // --------------------------------------------------------
    const promptSubject =
      contextModel?.questionProfile
        ?.grammaticalSubjectWord || "";

    if (
      sentenceState.isStart &&
      promptSubject === "you" &&
      lowerCur === "i"
    ) {
      // "Can you...?" / "Do you...?" often maps naturally to an "I..." answer.
      score += 1.2;
      markGood(0.6);
    }

    if (
      contextModel?.questionProfile
        ?.addressedRequest &&
      lowerCur === "you" &&
      !sentenceState.isStart
    ) {
      score -= 2.0;
    }

    return {
      legal,
      score,
      quality:
        Math.max(
          0,
          Math.min(1, quality)
        ),
      good:
        Math.max(
          0,
          good
        )
    };
  }

  isTransitionLegal(
    generatedTokens,
    curWord,
    curType,
    sentenceState,
    deepMode,
    contextModel = null
  ) {
    const lowerCur = curWord.toLowerCase();
    const prevToken = generatedTokens.length > 0 ? generatedTokens[generatedTokens.length - 1] : null;
    const prevWord = prevToken !== null ? (this.network.vocab?.idToToken[prevToken] || "").toLowerCase() : "";
    const prevType = prevToken !== null ? this.getCanonicalType(prevToken) : null;

    if (curType === "Punct") {
      if (![".", "!", "?"].includes(curWord)) return false;

      const allowsFragments =
        Boolean(
          contextModel?.answerShape
            ?.allowsFragments
        );

      const fragmentComplete =
        allowsFragments &&
        sentenceState.subjectReady &&
        !sentenceState.needsObjectAfterPrep &&
        !sentenceState.needsNominalAfterDet &&
        !sentenceState.needsBaseVerb;

      if (
        !this.clauseComplete(
          sentenceState
        ) &&
        !fragmentComplete
      ) {
        return false;
      }
      if (["Prep", "Det", "Modal", "Conj"].includes(prevType)) return false;
      if (prevType === "Aux" && ["be", "is", "am", "are", "been"].includes(prevWord)) return false;
      return true;
    }

    if (sentenceState.isStart) {
      // Answers should normally start declaratively, not with the question's
      // interrogative token.
      return this.getTransitionRule("START", curType).legal;
    }

    // "to" can be an infinitive marker as well as a preposition.
    if (prevWord === "to" && curType === "Verb") {
      return true;
    }

    if (curType === "Prep" && (sentenceState.prepCount >= 3 || prevType === "Prep")) return false;
    // Article phonetics and basic number agreement.
    if (prevWord === "a" && (/^[aeiou]/i.test(curWord) || /^(ai|apu|npu|cpu|gpu)/i.test(curWord))) {
      return false;
    }
    if (prevWord === "an" && !/^[aeiou]/i.test(curWord) && !/^(ai|apu|npu|cpu|gpu)/i.test(curWord)) {
      return false;
    }

    if (prevType === "Det" && ["Noun", "ProperNoun"].includes(curType)) {
      const singularDets = new Set(["a", "an", "this", "each", "every", "another", "one"]);
      const pluralDets = new Set(["these", "those", "many", "several", "both", "few"]);
      const looksPlural = lowerCur.endsWith("s") && !lowerCur.endsWith("ss");

      if (singularDets.has(prevWord) && looksPlural) return false;
      if (pluralDets.has(prevWord) && !looksPlural) return false;
    }

    // Modal and do-support auxiliaries require a base-like verb.
    if (prevType === "Modal") {
      if (curType !== "Verb" && curType !== "Aux" && curType !== "Adv") return false;
      if (curType === "Verb" &&
          (lowerCur.endsWith("ing") || lowerCur.endsWith("ed") ||
           (lowerCur.endsWith("s") && !lowerCur.endsWith("ss")))) {
        return false;
      }
    }

    if (prevType === "Aux" && ["do", "does", "did"].includes(prevWord)) {
      if (curType !== "Verb" && curType !== "Adv") return false;
      if (curType === "Verb" &&
          (lowerCur.endsWith("ing") || lowerCur.endsWith("ed") ||
           (lowerCur.endsWith("s") && !lowerCur.endsWith("ss")))) {
        return false;
      }
    }

    if (prevType === "Aux" && curType === "Aux") return false;
    if (prevType === "Det" && curType === "Det") return false;
    if (prevType === "Conj" && curType === "Conj") return false;

    // Avoid chains of grammatical glue with no content between them.
    const functionTypes = new Set(["Det", "Prep", "Conj", "Modal", "Aux"]);
    if (functionTypes.has(prevType) && functionTypes.has(curType)) {
      const permitted =
        (prevType === "Modal" && curType === "Aux") ||
        (prevType === "Conj" && ["Det", "Aux", "Modal"].includes(curType));

      if (!permitted) return false;
    }

    // Prepositions need an object-like continuation, except "to + Verb".
    if (
      prevType === "Prep" &&
      prevWord !== "to" &&
      !["Det", "Adj", "Noun", "ProperNoun", "Pronoun", "Num"].includes(curType)
    ) {
      return false;
    }

    // Strong subject/verb agreement checks for simple present forms. These are
    // deliberately conservative; ambiguous verbs are left to neural scoring.
    if (
      sentenceState.subjectReady &&
      !sentenceState.predicateReady &&
      curType === "Verb"
    ) {
      const verbHasS = lowerCur.endsWith("s") && !lowerCur.endsWith("ss");
      const subjectWord = sentenceState.subjectToken !== null
        ? this.tokenWord(sentenceState.subjectToken).toLowerCase()
        : "";

      const explicitlyNonThird = new Set(["i", "you", "we", "they"]);
      const explicitlyThird = new Set(["he", "she", "it"]);

      if ((sentenceState.isPluralSubject || explicitlyNonThird.has(subjectWord)) && verbHasS) {
        return false;
      }
      if (explicitlyThird.has(subjectWord) && !verbHasS) {
        return false;
      }
    }

    // Conjunctions should not be generic escape hatches in unfinished clauses.
    if (
      deepMode &&
      curType === "Conj" &&
      !this.clauseComplete(sentenceState)
    ) {
      return false;
    }

    // Prevent three-deep modifier chains and immediate Prep-object-Prep loops.
    if (deepMode && generatedTokens.length >= 2) {
      const t1 = this.getCanonicalType(
        generatedTokens[generatedTokens.length - 1]
      );
      const t2 = this.getCanonicalType(
        generatedTokens[generatedTokens.length - 2]
      );

      const modifier = t => t === "Adj" || t === "Adv";

      if (
        modifier(t1) &&
        modifier(t2) &&
        modifier(curType)
      ) {
        return false;
      }

      if (
        t2 === "Prep" &&
        ["Noun", "ProperNoun", "Pronoun", "Num"].includes(t1) &&
        curType === "Prep"
      ) {
        return false;
      }
    }

    // Before the predicate, allow compact noun compounds but prevent the model
    // from stacking an arbitrary river of nouns before ever reaching a verb.
    if (
      deepMode &&
      !sentenceState.predicateReady &&
      sentenceState.nounCount >= 4 &&
      ["Noun", "ProperNoun"].includes(curType)
    ) {
      return false;
    }

    // Deep mode is deliberately simple-clause biased. Once a predicate exists,
    // do not start another finite verb/modal/aux chain unless a conjunction has
    // explicitly opened a new clause. This kills outputs like
    // "chips accelerate model accelerates accelerating accelerated".
    if (
      deepMode &&
      sentenceState.predicateReady &&
      prevType !== "Conj" &&
      ["Verb", "Modal", "Aux"].includes(curType)
    ) {
      return false;
    }

    // Avoid endless noun piles after the predicate. Two object/complement nouns
    // are enough for short declarative answers; after that prefer a modifier,
    // prepositional phrase, or punctuation.
    if (
      deepMode &&
      sentenceState.predicateReady &&
      sentenceState.objectCount >= 2 &&
      ["Noun", "ProperNoun"].includes(prevType) &&
      ["Noun", "ProperNoun"].includes(curType)
    ) {
      return false;
    }

    return this.getTransitionRule(prevType, curType).legal;
  }

  roleBias(
    curType,
    sentenceState,
    contextModel = null
  ) {
    const shape =
      contextModel?.answerShape?.kind ||
      "prose";

    if (shape === "list") {
      if (!sentenceState.subjectReady) {
        if (
          [
            "Noun",
            "ProperNoun",
            "Adj",
            "Num"
          ].includes(curType)
        ) {
          return 5.2;
        }

        if (curType === "Det") {
          return 1.4;
        }

        return -1.3;
      }

      if (!sentenceState.predicateReady) {
        if (
          [
            "Noun",
            "ProperNoun",
            "Adj",
            "Prep"
          ].includes(curType)
        ) {
          return 2.4;
        }

        if (curType === "Verb") {
          return 2.2;
        }

        if (curType === "Punct") {
          return 5.0;
        }

        return -0.4;
      }
    }

    if (
      shape === "short" ||
      shape === "yes_no"
    ) {
      if (!sentenceState.subjectReady) {
        if (
          [
            "Noun",
            "ProperNoun",
            "Pronoun",
            "Num",
            "Adj"
          ].includes(curType)
        ) {
          return 4.8;
        }
      }

      if (
        sentenceState.subjectReady &&
        !sentenceState.predicateReady &&
        curType === "Punct"
      ) {
        return 4.6;
      }
    }
    if (!sentenceState.subjectReady) {
      if (curType === "Noun" || curType === "ProperNoun" || curType === "Pronoun") return 4.5;
      if (curType === "Adj" || curType === "Det") return 1.8;
      return -2.0;
    }

    if (!sentenceState.predicateReady) {
      if (curType === "Verb") return 5.0;
      if (curType === "Modal" || curType === "Aux") return 3.8;
      if (curType === "Adv") return 0.5;
      if (curType === "Noun") return 0.4; // noun compound such as "model chips"
      return -1.0;
    }

    if (sentenceState.objectCount === 0) {
      if (curType === "Noun" || curType === "ProperNoun") return 3.4;
      if (curType === "Det" || curType === "Adj") return 1.8;
      if (curType === "Prep" || curType === "Adv") return 0.9;
      if (["Verb", "Modal", "Aux"].includes(curType)) return -4.0;
    }

    if (sentenceState.objectCount >= 1) {
      if (curType === "Punct") return 4.2;
      if (curType === "Prep") return 1.8;
      if (curType === "Adv" || curType === "Adj") return 0.8;
      if (["Verb", "Modal", "Aux"].includes(curType)) return -5.0;
    }

    if (curType === "Punct") return 3.0;
    return 0.0;
  }

  nextRoleLookahead(
    curType,
    sentenceState,
    contextModel = null
  ) {
    const shape =
      contextModel?.answerShape?.kind ||
      "prose";

    if (
      shape === "list" &&
      sentenceState.subjectReady &&
      !sentenceState.predicateReady &&
      curType === "Punct"
    ) {
      return 1.8;
    }

    if (
      (
        shape === "short" ||
        shape === "yes_no"
      ) &&
      sentenceState.subjectReady &&
      curType === "Punct"
    ) {
      return 1.5;
    }
    // Cheap one-step structural lookahead. No extra neural forward pass.
    if (!sentenceState.subjectReady) {
      return ["Noun", "ProperNoun", "Pronoun"].includes(curType) ? 1.2 : 0.0;
    }
    if (!sentenceState.predicateReady) {
      return ["Verb", "Modal", "Aux"].includes(curType) ? 1.5 : 0.0;
    }
    if (sentenceState.objectCount === 0) {
      return ["Noun", "ProperNoun", "Det", "Adj", "Prep"].includes(curType) ? 0.8 : 0.0;
    }
    return curType === "Punct" ? 1.0 : 0.0;
  }

  getStemCounts(tokens) {
    const counts = new Map();
    const vocab = this.network.vocab;
    for (const tok of tokens) {
      const type = this.getCanonicalType(tok);
      if (!this.isContentType(type)) continue;
      const word = vocab?.idToToken?.[tok] || "";
      const stem = this.stemWord(word);
      if (stem.length < 3) continue;
      counts.set(stem, (counts.get(stem) || 0) + 1);
    }
    return counts;
  }

  buildDeepCandidatePool(
    rawLogits,
    contextModel,
    cleanPromptTokens,
    options = null
  ) {
    const vocab =
      this.network.vocab;

    const vocabSize =
      this.network.vocabSize;

    const refinementRound =
      Math.max(
        0,
        Math.min(
          this.reasoningConfig
            .refinementMaxRounds,
          Number(
            options
              ?.refinementRound ??
            contextModel
              ?.refinementRound
          ) || 0
        )
      );

    const familyLimit =
      Math.min(
        this.reasoningConfig
          .recoveryPoolFamilyMax,
        this.reasoningConfig
          .recoveryPoolFamilyBase +
        refinementRound *
        this.reasoningConfig
          .recoveryPoolFamilyGrowth
      );

    const firstHopLimit =
      Math.min(
        this.reasoningConfig
          .recoveryPoolFirstHopMax,
        this.reasoningConfig
          .recoveryPoolFirstHopBase +
        refinementRound *
        this.reasoningConfig
          .recoveryPoolFirstHopGrowth
      );

    const secondHopLimit =
      Math.min(
        this.reasoningConfig
          .recoveryPoolSecondHopMax,
        this.reasoningConfig
          .recoveryPoolSecondHopBase +
        refinementRound *
        this.reasoningConfig
          .recoveryPoolSecondHopGrowth
      );

    const secondHopSeedLimit =
      Math.min(
        this.reasoningConfig
          .recoveryPoolSecondHopSeedsMax,
        this.reasoningConfig
          .recoveryPoolSecondHopSeedsBase +
        refinementRound *
        this.reasoningConfig
          .recoveryPoolSecondHopSeedsGrowth
      );

    const neighborThresholdDrop =
      refinementRound *
      this.reasoningConfig
        .recoveryPoolNeighborThresholdDrop;

    const propositionThresholdDrop =
      refinementRound *
      this.reasoningConfig
        .recoveryPoolPropositionThresholdDrop;

    const recovery =
      contextModel
        ?.recoveryProfile;

    const families =
      new Map();

    const familyOf =
      type => {
        if (
          type === "ProperNoun"
        ) {
          return "Noun";
        }

        if (
          type === "Modal"
        ) {
          return "Aux";
        }

        return type;
      };

    const insertTop =
      (
        family,
        token,
        score
      ) => {
        let arr =
          families.get(family);

        if (!arr) {
          arr = [];
          families.set(
            family,
            arr
          );
        }

        if (
          arr.length <
          familyLimit
        ) {
          arr.push({
            token,
            score
          });

          arr.sort(
            (a, b) =>
              b.score -
              a.score
          );
        } else if (
          score >
          arr[
            arr.length - 1
          ].score
        ) {
          arr[
            arr.length - 1
          ] = {
            token,
            score
          };

          arr.sort(
            (a, b) =>
              b.score -
              a.score
          );
        }
      };

    const requiredRole =
      contextModel
        ?.semanticFrame
        ?.requiredRole ||
      contextModel
        ?.queryFrame
        ?.requiredRole ||
      contextModel
        ?.answerPlan
        ?.requiredRole ||
      "content";

    // One bounded whole-vocabulary scan per pool expansion. Round 0 is the
    // original compact scan; later rounds reuse the SAME raw logits and change
    // only CPU-side ranking/capacity.
    for (
      let i = 0;
      i < vocabSize;
      i++
    ) {
      if (
        !vocab?.hasToken?.(i)
      ) {
        continue;
      }

      const type =
        this.getCanonicalType(i);

      if (type === "Other") {
        continue;
      }

      const semantic =
        this.getRoutedSemanticScore(
          i,
          contextModel,
          null
        );

      let score =
        rawLogits[i] +
        semantic * 1.25;

      if (
        refinementRound > 0 &&
        recovery?.enabled
      ) {
        const typeFit =
          this.getSemanticFrameTypeCompatibility(
            i,
            contextModel
          );

        const roleFitRaw =
          vocab
            ?.getRoleCompatibility?.(
              i,
              requiredRole
            );

        const roleFit =
          Number.isFinite(
            roleFitRaw
          )
            ? Math.max(
                0,
                Math.min(
                  1,
                  roleFitRaw
                )
              )
            : 0.45;

        const actPressure =
          recovery.actNeed * 0.42 +
          recovery.propositionNeed *
            0.58;

        const relationPressure =
          recovery.relationNeed *
            0.58 +
          recovery.propositionNeed *
            0.42;

        score +=
          typeFit *
          actPressure *
          this.reasoningConfig
            .recoveryPoolTypeWeight;

        score +=
          roleFit *
          relationPressure *
          this.reasoningConfig
            .recoveryPoolRoleWeight;

        if (
          recovery.propositionNeed >
            0.28 &&
          this.isSourceAnchorFamily(
            i,
            contextModel
          )
        ) {
          score +=
            recovery
              .propositionNeed *
            this.reasoningConfig
              .recoveryPoolAnchorWeight;
        }
      }

      insertTop(
        familyOf(type),
        i,
        score
      );
    }

    const requiredPool =
      new Set();

    const rankedPool =
      new Set();

    const addRequired =
      token => {
        if (
          Number.isInteger(token) &&
          vocab?.hasToken?.(token)
        ) {
          requiredPool.add(token);
        }
      };

    const addRanked =
      token => {
        if (
          Number.isInteger(token) &&
          vocab?.hasToken?.(token)
        ) {
          rankedPool.add(token);
        }
      };

    // Plan seeds are hard requirements.
    for (
      const tok of
        contextModel
          ?.answerPlan
          ?.seedTokens || []
    ) {
      addRequired(tok);
    }

    for (
      const tok of
        contextModel
          ?.memoryBundle
          ?.candidateTokenIds || []
    ) {
      addRanked(tok);
    }

    const inputProfile = contextModel?.userInputLearningProfile;
    if (inputProfile?.carryoverNeed >= this.reasoningConfig.userCarryoverFloor) {
      for (const tok of inputProfile.carryoverAnchorTokens || []) addRequired(tok);
    }

    // Recovery preserves every option from the prior compact pool. Widening
    // therefore cannot accidentally remove a candidate that was available in
    // the faster first search.
    for (
      const tok of
        options?.priorPool || []
    ) {
      addRequired(tok);
    }

    for (
      const arr of
        families.values()
    ) {
      for (
        const item of arr
      ) {
        addRanked(
          item.token
        );
      }
    }

    // Whole-vocabulary knowledge expansion. Recovery broadens discovery, not
    // trust: source/provenance verification remains unchanged.
    if (
      typeof vocab
        ?.getKnowledgeNeighbors ===
        "function" ||
      typeof vocab
        ?.getAssociations ===
        "function"
    ) {
      const anchors = [];

      const addAnchor =
        token => {
          const canonical =
            typeof vocab
              .canonicalId ===
              "function"
              ? vocab.canonicalId(
                  token
                )
              : token;

          if (
            Number.isInteger(
              canonical
            ) &&
            vocab.hasToken(
              canonical
            ) &&
            !anchors.includes(
              canonical
            )
          ) {
            anchors.push(
              canonical
            );
          }
        };

      for (
        const token of
          contextModel
            ?.instructionContract
            ?.sourceAnchorTokens ||
          contextModel
            ?.promptContentTokens ||
          []
      ) {
        addAnchor(token);

        if (
          anchors.length >=
          (
            refinementRound > 0
              ? 12
              : 8
          )
        ) {
          break;
        }
      }

      addAnchor(
        contextModel
          ?.answerContract
          ?.relationVerbToken
      );

      const firstHop = [];

      const firstHopNeighborFloor =
        Math.max(
          0.46,
          0.56 -
          neighborThresholdDrop
        );

      const firstHopPropositionFloor =
        Math.max(
          0.42,
          0.50 -
          propositionThresholdDrop
        );

      for (
        const anchor of anchors
      ) {
        const related =
          typeof vocab
            .getKnowledgeNeighbors ===
            "function"
            ? vocab
                .getKnowledgeNeighbors(
                  anchor,
                  firstHopLimit
                )
            : vocab
                .getAssociations(
                  anchor,
                  Math.max(
                    5,
                    firstHopLimit - 2
                  )
                );

        for (
          const item of related
        ) {
          if (
            !Number.isInteger(
              item?.id
            ) ||
            !vocab.hasToken(
              item.id
            )
          ) {
            continue;
          }

          const neighborScore =
            item.score ??
            item.weight ??
            0;

          const proposition =
            this.getTrustedPropositionEvidence(
              contextModel
                ?.answerContract
                ?.subjectTokens?.[0],
              contextModel
                ?.answerContract
                ?.relationVerbToken,
              item.id,
              requiredRole
            );

          const propositionFit =
            proposition?.score ??
            0.55;

          if (
            neighborScore >=
              firstHopNeighborFloor &&
            (
              propositionFit >=
                firstHopPropositionFloor ||
              neighborScore >=
                Math.max(
                  0.70,
                  0.78 -
                  neighborThresholdDrop
                )
            )
          ) {
            addRanked(
              item.id
            );

            firstHop.push(
              item.id
            );
          }
        }
      }

      const secondHopSeeds =
        firstHop.slice(
          0,
          secondHopSeedLimit
        );

      const secondHopNeighborFloor =
        Math.max(
          0.58,
          0.70 -
          neighborThresholdDrop
        );

      const secondHopPropositionFloor =
        Math.max(
          0.54,
          0.64 -
          propositionThresholdDrop
        );

      for (
        const seed of
          secondHopSeeds
      ) {
        const related =
          typeof vocab
            .getKnowledgeNeighbors ===
            "function"
            ? vocab
                .getKnowledgeNeighbors(
                  seed,
                  secondHopLimit
                )
            : vocab
                .getAssociations(
                  seed,
                  Math.max(
                    2,
                    secondHopLimit - 1
                  )
                );

        for (
          const item of related
        ) {
          if (
            !Number.isInteger(
              item?.id
            ) ||
            !vocab.hasToken(
              item.id
            )
          ) {
            continue;
          }

          const neighborScore =
            item.score ??
            item.weight ??
            0;

          const proposition =
            this.getTrustedPropositionEvidence(
              contextModel
                ?.answerContract
                ?.subjectTokens?.[0],
              contextModel
                ?.answerContract
                ?.relationVerbToken,
              item.id,
              requiredRole
            );

          if (
            neighborScore >=
              secondHopNeighborFloor &&
            (
              proposition?.score ??
              0.0
            ) >=
              secondHopPropositionFloor
          ) {
            addRanked(
              item.id
            );
          }
        }
      }
    }

    // Existing semantic scaffolds remain available. These are generic
    // language/answer-shape seeds, not subject-specific answer templates.
    if (
      contextModel
        ?.answerContract
        ?.kind ===
        "temporal"
    ) {
      const temporalSeedWords = [
        "today", "tomorrow",
        "yesterday", "now",
        "soon", "later",
        "eventually", "before",
        "after", "during",
        "until", "since",
        "within", "around",
        "year", "years",
        "month", "months",
        "week", "weeks",
        "day", "days"
      ];

      for (
        const word of
          temporalSeedWords
      ) {
        const id =
          this.getVocabTokenId(
            word
          );

        addRequired(id);
      }
    }

    if (
      contextModel
        ?.answerPlan
        ?.requiresNumericEvidence
    ) {
      for (
        const token of
          contextModel
            ?.answerPlan
            ?.numericPromptTokens ||
          []
      ) {
        addRequired(token);
      }

      for (
        const token of
          contextModel
            ?.answerPlan
            ?.quantityUnitTokens ||
          []
      ) {
        addRequired(token);
      }

      for (
        const word of [
          "number", "amount",
          "total", "bits",
          "bytes", "size",
          "rate"
        ]
      ) {
        addRequired(
          this.getVocabTokenId(
            word
          )
        );
      }
    }

    if (
      contextModel
        ?.answerPlan
        ?.underspecified
    ) {
      for (
        const word of [
          "topic", "question",
          "details", "more",
          "mean"
        ]
      ) {
        addRequired(
          this.getVocabTokenId(
            word
          )
        );
      }
    }

    // Punctuation is always retained.
    for (
      const p of [
        ".", ",", "!",
        "?", ";", ":"
      ]
    ) {
      addRequired(
        vocab
          ?.tokenToId
          ?.get?.(p)
      );
    }

    // Add ranked tokens up to the CPU latency guard. Required tokens and the
    // previous pool are never evicted.
    const result =
      new Set(
        requiredPool
      );

    const rankedEntries = [];

    for (
      const token of rankedPool
    ) {
      const type =
        this.getCanonicalType(
          token
        );

      const semantic =
        this.getRoutedSemanticScore(
          token,
          contextModel,
          null
        );

      let recoveryFit = 0.0;

      if (
        refinementRound > 0 &&
        recovery?.enabled
      ) {
        const typeFit =
          this.getSemanticFrameTypeCompatibility(
            token,
            contextModel
          );

        const roleFitRaw =
          vocab
            ?.getRoleCompatibility?.(
              token,
              requiredRole
            );

        const roleFit =
          Number.isFinite(
            roleFitRaw
          )
            ? Math.max(
                0,
                Math.min(
                  1,
                  roleFitRaw
                )
              )
            : 0.45;

        recoveryFit =
          typeFit *
            (
              recovery.actNeed *
                0.42 +
              recovery.propositionNeed *
                0.58
            ) +
          roleFit *
            (
              recovery.relationNeed *
                0.58 +
              recovery.propositionNeed *
                0.42
            );
      }

      rankedEntries.push({
        token,
        score:
          rawLogits[token] +
          semantic * 1.25 +
          recoveryFit * 0.85 +
          (
            type === "Punct"
              ? 0.2
              : 0.0
          )
      });
    }

    rankedEntries.sort(
      (a, b) =>
        b.score -
        a.score
    );

    const maxSize =
      Math.max(
        requiredPool.size,
        this.reasoningConfig
          .recoveryPoolMaxSize
      );

    for (
      const item of
        rankedEntries
    ) {
      if (
        result.size >=
        maxSize
      ) {
        break;
      }

      result.add(
        item.token
      );
    }

    contextModel.deepCandidatePoolStats = {
      refinementRound,
      familyLimit,
      firstHopLimit,
      secondHopLimit,
      secondHopSeedLimit,
      requiredCount:
        requiredPool.size,
      rankedCount:
        rankedPool.size,
      size:
        result.size
    };

    return Int32Array.from(
      result
    );
  }

  buildScores(
    rawLogits,
    generated,
    contextModel,
    sentenceState,
    deepMode,
    step,
    targetLength,
    reuseScores = null
  ) {
    const vocabSize = this.network.vocabSize;
    const vocab = this.network.vocab;
    const prevToken = generated.length ? generated[generated.length - 1] : null;
    const prevType = prevToken !== null ? this.getCanonicalType(prevToken) : null;

    const scores =
      reuseScores && reuseScores.length === vocabSize
        ? reuseScores
        : new Float32Array(vocabSize);

    scores.fill(-1e9);

    const recent = generated.slice(-10);
    const recentCounts = new Map();
    for (const t of recent) recentCounts.set(t, (recentCounts.get(t) || 0) + 1);

    const stemCounts = this.getStemCounts(generated.slice(-40));
    const generatedSet = new Set(generated.slice(-48));
    const coverageNow = this.promptCoverage(generated, contextModel);
    const patternContext = deepMode
      ? this.buildLocalPatternContext(generated)
      : null;

    const candidateIds =
      deepMode && contextModel?.deepCandidatePool?.length
        ? contextModel.deepCandidatePool
        : null;

    const scanCount = candidateIds ? candidateIds.length : vocabSize;

    for (let scanIndex = 0; scanIndex < scanCount; scanIndex++) {
      const i = candidateIds ? candidateIds[scanIndex] : scanIndex;

      if (!vocab?.hasToken?.(i)) continue;
      const word = vocab.idToToken[i] || "";
      const curType = this.getCanonicalType(i);

      // A declarative answer should not reinsert the question operator that
      // was deliberately removed by the prompt-frame planner. This is a cheap
      // generic structural veto, not prompt-specific wording.
      if (
        deepMode &&
        contextModel?.questionProfile?.isQuestion &&
        Number.isFinite(contextModel.questionProfile.operatorToken) &&
        i === contextModel.questionProfile.operatorToken
      ) {
        continue;
      }

      // Mid-sentence title-case variants of ordinary words are a common source
      // of scrambled output when the vocabulary contains both case variants.
      // Sentence starts are exempt, as are actual proper nouns.
      if (
        deepMode &&
        !sentenceState.isStart &&
        curType !== "ProperNoun" &&
        /^[A-Z][a-z]/.test(word)
      ) {
        continue;
      }

      if (
        !this.isTransitionLegal(
          generated,
          word,
          curType,
          sentenceState,
          deepMode,
          contextModel
        )
      ) {
        continue;
      }

      const humanTransition =
        deepMode
          ? this.getHumanTransitionAssessment(
              generated,
              i,
              word,
              curType,
              sentenceState,
              contextModel
            )
          : null;

      if (
        deepMode &&
        humanTransition &&
        !humanTransition.legal
      ) {
        continue;
      }

      let score =
        rawLogits[i];

      score -=
        this.getFailedTrajectorySuppression(
          i,
          prevToken,
          contextModel
        );

      score +=
        this.getTransitionRule(
          prevType,
          curType
        ).weight;

      const userPredictionSignals =
        this.getUserPredictionSignals(
          i,
          prevToken,
          contextModel
        );
      score += userPredictionSignals.bonus;

      if (humanTransition) {
        // Human-decodable word pairing is rewarded directly.
        score +=
          (humanTransition.quality - 0.50) *
          4.6;

        // Each strongly correct grammatical relationship contributes +0.5,
        // matching the positive-evidence rule used by semantic associations.
        score +=
          Math.min(
            1.5,
            humanTransition.good * 0.5
          );
      }

      const routedSemantic = this.getRoutedSemanticScore(
        i,
        contextModel,
        sentenceState
      );
      score += routedSemantic * (deepMode ? 1.55 : 0.70);

      const count = recentCounts.get(i) || 0;
      if (count) score -= count * count * (deepMode ? 8.0 : 3.0);
      if (i === prevToken) score -= 22.0;

      const stem = this.stemWord(word);
      const exactPrompt = contextModel.promptContentSet.has(i);
      const stemPrompt =
        stem.length >= 3 &&
        contextModel.promptStems.has(stem);
      const sameStemCount =
        this.countMorphologicalFamilyInCurrentClause(
          i,
          generated
        );

      // Hard lexical-family anti-repeat.
      // computer -> computing, reason -> reasoned -> reasons,
      // accelerate -> accelerated -> accelerating, etc.
      // One family member per clause is enough.
      if (
        sameStemCount > 0 &&
        this.isContentType(curType)
      ) {
        continue;
      }

      const sourceEcho =
        this.getSourceEchoAssessment(
          i,
          generated,
          sentenceState,
          contextModel
        );

      if (sourceEcho.hardVeto) {
        continue;
      }

      const provenance =
        this.getPromptRootedProvenance(
          i,
          contextModel
        );

      const eventQueryFit =
        this.getEventQueryCandidateFit(
          i,
          contextModel,
          sentenceState
        );

      score +=
        this.getRelationStructureCandidateBonus(
          i,
          contextModel,
          sentenceState,
          eventQueryFit
        );

      const semanticGraphFit =
        this.getSemanticGraphCandidateFit(
          i,
          generated,
          contextModel,
          sentenceState,
          provenance,
          eventQueryFit
        );

      score +=
        (
          semanticGraphFit -
          0.50
        ) *
        this.reasoningConfig
          .semanticGraphCandidateWeight;

      score +=
        this.getRecoveryCandidateBonus(
          i,
          contextModel,
          sentenceState,
          semanticGraphFit,
          provenance
        );

      const memorySignal =
        this.getMemoryCandidateSignals(
          i,
          contextModel
        );

      score += memorySignal.bonus;

      const selfWhy =
        this.getSelfWhyCandidate(
          i,
          generated,
          contextModel,
          sentenceState,
          provenance,
          eventQueryFit,
          semanticGraphFit,
          humanTransition
        );

      score +=
        Math.max(
          -this.reasoningConfig.selfWhyMaxCandidateBonus,
          Math.min(
            this.reasoningConfig.selfWhyMaxCandidateBonus,
            (
              selfWhy.score -
              0.50
            ) *
            this.reasoningConfig.selfWhyCandidateWeight
          )
        );

      score += this.answerCommitmentBonus(i, generated, contextModel);

      if (
        deepMode &&
        this.isContentType(curType) &&
        generated.length >= 2 &&
        !this.isSourceAnchorFamily(i, contextModel) &&
        selfWhy.score <
          this.reasoningConfig.selfWhyCandidateHardFloor &&
        selfWhy.relationFit < 0.32 &&
        selfWhy.promptFit < 0.40
      ) {
        continue;
      }

      if (
        deepMode &&
        this.isContentType(curType) &&
        selfWhy.score <
          this.reasoningConfig.selfWhyCandidateSoftFloor
      ) {
        score -=
          (
            this.reasoningConfig.selfWhyCandidateSoftFloor -
            selfWhy.score
          ) *
          4.2;
      }

      if (
        deepMode &&
        contextModel
          ?.questionProfile
          ?.isQuestion &&
        this.isContentType(curType) &&
        generated.length >= 3 &&
        !this.isSourceAnchorFamily(
          i,
          contextModel
        ) &&
        semanticGraphFit <
          this.reasoningConfig
            .semanticGraphHardFloor &&
        eventQueryFit < 0.54
      ) {
        continue;
      }

      if (deepMode && this.isContentType(curType)) {
        const requiresGrounding = Boolean(contextModel?.queryFrame?.requiresGrounding);

        // Novel content can be relevant without being factual, but a factual
        // answer cannot promote discovery-only neighborhood links into truth.
        if (
          requiresGrounding &&
          generated.length >= 3 &&
          !this.isSourceAnchorFamily(i, contextModel) &&
          provenance.grounding < this.reasoningConfig.provenanceHardFloor &&
          eventQueryFit < 0.68
        ) {
          continue;
        }

        if (provenance.grounding < this.reasoningConfig.provenanceSoftFloor) {
          score -= (this.reasoningConfig.provenanceSoftFloor - provenance.grounding) * 5.4;
        }

        if (
          semanticGraphFit <
          this.reasoningConfig
            .semanticGraphSoftFloor
        ) {
          score -=
            (
              this.reasoningConfig
                .semanticGraphSoftFloor -
              semanticGraphFit
            ) *
            4.8;
        }

        const liveSlot =
          this.getLiveAnswerSlotProgress(
            sentenceState,
            contextModel
          );

        const modifierInformativeness =
          this.getModifierInformativeness(
            i,
            contextModel,
            sentenceState,
            provenance,
            eventQueryFit
          );

        const lexicalFatiguePenalty =
          this.getLexicalFatiguePenalty(
            i,
            curType,
            contextModel
          );

        score -=
          lexicalFatiguePenalty;

        const slotNeed =
          contextModel
            ?.questionProfile
            ?.isQuestion
            ? Math.max(
                0,
                1.0 -
                (
                  liveSlot
                    ?.relation ??
                  0
                )
              )
            : 0.0;

        score +=
          (eventQueryFit - 0.50) *
          (
            3.2 +
            slotNeed * 3.4
          );

        if (
          slotNeed > 0.22 &&
          !this.isSourceAnchorFamily(
            i,
            contextModel
          )
        ) {
          score +=
            Math.max(
              0,
              eventQueryFit - 0.46
            ) *
            slotNeed *
            3.0;
        }

        score +=
          Math.max(
            0,
            provenance.grounding - 0.42
          ) *
          2.0;

        if (curType === "Adv") {
          const deficit =
            Math.max(
              0,
              this.reasoningConfig
                .modifierSoftFloor -
              modifierInformativeness
            );

          score -=
            deficit *
            this.reasoningConfig
              .modifierPenaltyWeight *
            (
              0.72 +
              slotNeed * 0.78
            );

          if (
            slotNeed > 0.46 &&
            modifierInformativeness <
              this.reasoningConfig
                .modifierHardFloor &&
            !this.isSourceAnchorFamily(
              i,
              contextModel
            )
          ) {
            continue;
          }
        }
      }

      const rlFeatures =
        this.getRLPolicyFeatures(
          i,
          generated,
          contextModel,
          sentenceState,
          rawLogits[i],
          provenance,
          eventQueryFit
        );

      score +=
        this.getRLPolicyValue(rlFeatures) *
        this.reasoningConfig.rlPolicyWeight;

      if (
        deepMode &&
        this.isContentType(curType) &&
        generated.length >= 2 &&
        !this.isSourceAnchorFamily(i, contextModel)
      ) {
        const anchorSupport =
          this.getSemanticAnchorSupport(i, contextModel);

        const roleSupport =
          this.getRoleSensitiveKnowledgeEvidence(
            i,
            sentenceState,
            contextModel
          );

        const routed =
          this.getRoutedSemanticScore(
            i,
            contextModel,
            sentenceState
          );

        const evidence = Math.max(
          anchorSupport?.semanticSupport || 0,
          anchorSupport?.expansionSupport || 0,
          roleSupport?.overall || 0,
          Math.max(0, (routed + 0.10) / 0.90)
        );

        const hasMeaningfulAnchors =
          (contextModel?.instructionContract?.sourceAnchorTokens?.length || 0) >= 2;

        if (hasMeaningfulAnchors && evidence < 0.29) {
          continue;
        }

        if (evidence < 0.42) {
          score -= (0.42 - evidence) * 7.0;
        }
      }

      if (
        contextModel?.answerPlan?.requiresNumericEvidence &&
        sentenceState?.predicateReady &&
        this.isContentType(curType)
      ) {
        const quantityFit =
          vocab?.getRoleCompatibility?.(i, "quantity") ??
          (curType === "Num" ? 1.0 : 0.35);

        if (quantityFit < 0.20 && generated.length >= 3) {
          score -= 2.8;
        }
      }

      if (
        contextModel?.answerShape?.kind ===
          "temporal"
      ) {
        const temporalTense =
          this.temporalTenseAssessment(
            i,
            generated,
            contextModel
          );

        if (
          temporalTense.contradiction &&
          temporalTense.score >= 0.55
        ) {
          continue;
        }

        score +=
          (
            temporalTense.compatibility -
            0.50
          ) *
          2.8;
      }

      score += sourceEcho.scoreAdjustment;

      if (
        this.isContentType(curType) &&
        !sourceEcho.isEcho
      ) {
        const anchorSupport =
          this.getSemanticAnchorSupport(
            i,
            contextModel
          );

        score +=
          Math.max(
            0,
            anchorSupport.expansionSupport - 0.30
          ) * 3.4;
      }

      let continuation = null;

      if (deepMode) {
        continuation = this.getContinuationEvidence(
          i,
          generated,
          contextModel,
          sentenceState,
          patternContext,
          routedSemantic
        );

        // Semantic/sequence veto:
        // a novel content token must be supported by either the current
        // semantic thread, the routed prompt meaning, or a strong transition.
        // Prompt anchors are exempt so the answer can always reconnect to the
        // user's actual concepts.
        if (
          this.isContentType(curType) &&
          generated.length >= 3
        ) {
          const clearlyBroken =
            continuation.evidence < 0.34 &&
            continuation.localSimilarity < 0.00 &&
            routedSemantic < 0.16;

          const extremelyBroken =
            continuation.evidence < 0.21;

          const unsupportedNovel =
            !exactPrompt &&
            !stemPrompt &&
            continuation.evidence < 0.42 &&
            continuation.localSimilarity < 0.10 &&
            routedSemantic < 0.22;

          const breaksLockedClause =
            continuation.clauseLocked &&
            continuation.clauseEvidence < 0.24 &&
            continuation.evidence < 0.40 &&
            routedSemantic < 0.32;

          const associationBroken =
            !exactPrompt &&
            !stemPrompt &&
            sentenceState.predicateReady &&
            generated.length >= 4 &&
            continuation.associationLocal < 0.16 &&
            continuation.associationAnchor < 0.20;

          const associationDrift =
            !exactPrompt &&
            !stemPrompt &&
            generated.length >= 5 &&
            continuation.associationLocal >= 0.48 &&
            continuation.associationAnchor < 0.16;

          const novelBroken =
            !exactPrompt &&
            !stemPrompt &&
            (
              clearlyBroken ||
              extremelyBroken ||
              unsupportedNovel ||
              associationBroken ||
              associationDrift
            );

          if (novelBroken || breaksLockedClause) {
            continue;
          }
        }

        // Strong continuation evidence gets a meaningful preference before
        // grammar and repetition reranking.
        score +=
          (continuation.evidence - 0.50) * 4.5;
        // Prompt anchors guide semantic grounding through the instruction
        // contract. Surface prompt tokens receive no automatic generation
        // bonus; reuse is handled by getSourceEchoAssessment().

        // Keep off-topic content from sneaking back in after prompt coverage.
        // This is only arithmetic over values already computed for the token.
        if (!exactPrompt && !stemPrompt && this.isContentType(curType)) {
          if (routedSemantic < 0.12) score -= 6.0;
          else if (routedSemantic < 0.22) score -= 3.0;
          else if (routedSemantic > 0.42) score += 1.2;

          if (
            continuation &&
            continuation.localSimilarity < 0.02 &&
            generated.length >= 3
          ) {
            score -= 2.2;
          }

          if (coverageNow >= 0.45 && routedSemantic < 0.26) {
            score -= 1.8;
          }

          // COMPOUNDING ALIGNMENT DEBT:
          // once a clause starts drifting, another weakly-related content word
          // becomes progressively more expensive. This uses values already in
          // memory and adds no neural passes or vocabulary-wide work.
          const clauseEMA =
            Number.isFinite(sentenceState?.clauseSemanticEMA)
              ? sentenceState.clauseSemanticEMA
              : 0.72;

          const routeDeficit = Math.max(
            0,
            Math.min(
              1,
              (0.38 - routedSemantic) / 0.38
            )
          );

          const clauseDeficit = Math.max(
            0,
            Math.min(
              1,
              (0.52 - clauseEMA) / 0.52
            )
          );

          const driftCount = Math.min(
            3,
            sentenceState?.clauseDriftCount || 0
          );

          const localDeficit =
            continuation
              ? Math.max(
                  0,
                  Math.min(
                    1,
                    (0.14 -
                      continuation.localSimilarity) /
                    0.34
                  )
                )
              : 0.0;

          const questionPressure =
            contextModel?.questionProfile?.isQuestion
              ? 1.18
              : 1.0;

          const compoundedDriftPenalty =
            questionPressure *
            routeDeficit *
            (
              4.5 +
              clauseDeficit * 4.0 +
              localDeficit * 2.5 +
              driftCount * 1.35
            );

          score -= compoundedDriftPenalty;

          // If the clause is already badly off course, do not let another
          // unsupported word make it even worse.
          if (
            generated.length >= 4 &&
            routeDeficit > 0.78 &&
            clauseDeficit > 0.52 &&
            localDeficit > 0.45
          ) {
            continue;
          }
        }

        if (prevToken !== null && contextModel.promptTransitions.get(prevToken)?.has(i) && !generatedSet.has(i)) {
          score += 4.6;
        }

        score +=
          this.roleBias(
            curType,
            sentenceState,
            contextModel
          );

        score +=
          this.nextRoleLookahead(
            curType,
            sentenceState,
            contextModel
          );

        score +=
          this.answerShapeTokenBias(
            i,
            curType,
            sentenceState,
            contextModel
          );

        score +=
          this.questionRelationBonus(
            i,
            sentenceState,
            contextModel
          );

        score +=
          this.answerContractBonus(
            i,
            generated,
            sentenceState,
            contextModel
          );

        score +=
          this.answerSlotBonus(
            i,
            generated,
            sentenceState,
            contextModel
          );

        score +=
          this.roleSensitiveTokenBonus(
            i,
            generated,
            sentenceState,
            contextModel
          );

        score += this.dynamicWordPatternScore(
          i,
          generated,
          contextModel,
          sentenceState,
          patternContext,
          continuation
        );

        if (
          !deepMode &&
          !sentenceState.isStart &&
          /^[A-Z][a-z]/.test(word) &&
          curType !== "ProperNoun"
        ) {
          score -= 2.0;
        }
      }

      if (curType === "Punct") {
        const coverage = this.promptCoverage(generated, contextModel);
        const answered = this.answerRelationSatisfied(
          generated,
          sentenceState,
          contextModel
        );
        const goodStop =
          answered &&
          step >= 3 &&
          coverage >= (deepMode ? 0.20 : 0.15);

        const semanticDrift =
          Number.isFinite(sentenceState?.clauseSemanticEMA)
            ? sentenceState.clauseSemanticEMA
            : 0.72;

        const driftStopBonus =
          semanticDrift < 0.42 &&
          this.clauseComplete(sentenceState)
            ? (0.42 - semanticDrift) * 14.0
            : 0.0;

        if (word === ".") {
          score +=
            (goodStop ? 8.5 : -10.0) +
            driftStopBonus;
        } else {
          score +=
            (goodStop ? 0.2 : -10.0) +
            driftStopBonus * 0.35;
        }

        // The user asked a question; the model is answering it, not echoing it.
        if (contextModel?.questionProfile?.isQuestion && word === "?") {
          score -= 12.0;
        }
      }

      // Once Deep has subject + predicate + at least one complement, ending the
      // sentence should beat inventing another clause unless a clearly relevant
      // prepositional/modifier continuation is much stronger.
      if (deepMode && this.clauseComplete(sentenceState) && sentenceState.objectCount >= 1 && step >= 4) {
        const clauseLengthPressure =
          Math.min(8.0, Math.max(0, step - 7) * 0.65);

        if (curType === "Punct" && word === ".") {
          score += 10.0 + clauseLengthPressure;
        } else if (["Verb", "Modal", "Aux"].includes(curType)) {
          score -= 9.0;
        } else if (
          this.isContentType(curType) &&
          !exactPrompt &&
          !stemPrompt
        ) {
          score -= 2.8 + Math.max(0, step - targetLength) * 0.35;
        } else if (
          ["Conj", "Prep"].includes(curType) &&
          step >= targetLength
        ) {
          score -= 3.0;
        }
      }

      scores[i] = score;
    }

    return scores;
  }

  selectLowToken(scores, temperature) {
    let maxLogit = -1e9;
    for (let i = 0; i < scores.length; i++) if (scores[i] > maxLogit) maxLogit = scores[i];

    let expSum = 0;
    const probs = new Float32Array(scores.length);
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] <= -1e8) continue;
      const p = Math.exp((scores[i] - maxLogit) / temperature);
      probs[i] = p;
      expSum += p;
    }

    if (expSum <= 0) return { token: -1, prob: 0.0 };
    const r = Math.random() * expSum;
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return { token: i, prob: probs[i] / expSum };
    }
    return { token: -1, prob: 0.0 };
  }

  selectDeepToken(scores, candidateIds = null) {
    // Deep only needs the winner and runner-up: the winner is emitted and the
    // runner-up defines the certainty margin. Keeping Top-8 was wasted CPU.
    // Scan the already-pruned candidate pool instead of the full vocabulary.
    let bestId = -1;
    let secondId = -1;
    let bestVal = -1e9;
    let secondVal = -1e9;

    const scanCount =
      candidateIds?.length
        ? candidateIds.length
        : scores.length;

    for (let scanIndex = 0; scanIndex < scanCount; scanIndex++) {
      const i = candidateIds?.length
        ? candidateIds[scanIndex]
        : scanIndex;

      const s = scores[i];
      if (s <= -1e8) continue;

      if (s > bestVal) {
        secondVal = bestVal;
        secondId = bestId;
        bestVal = s;
        bestId = i;
      } else if (s > secondVal) {
        secondVal = s;
        secondId = i;
      }
    }

    if (bestId < 0) {
      return { token: -1, prob: 0.0, margin: 0.0 };
    }

    const margin =
      secondId >= 0
        ? Math.max(0, bestVal - secondVal)
        : 8.0;

    const pseudoProb =
      1.0 / (1.0 + Math.exp(-margin));

    return {
      token: bestId,
      prob: pseudoProb,
      margin
    };
  }

  updateSentenceState(
    sentenceState,
    token,
    word,
    curType
  ) {
    const lower =
      String(word || "")
        .toLowerCase();

    const wasPredicateReady =
      Boolean(
        sentenceState.predicateReady
      );

    const wasNeedPrepObject =
      Boolean(
        sentenceState.needsObjectAfterPrep
      );

    const wasNeedNominal =
      Boolean(
        sentenceState.needsNominalAfterDet
      );

    const pronoun =
      curType === "Pronoun"
        ? this.getPronounProfile(
            lower
          )
        : null;

    // A conjunction after a complete proposition opens a fresh clause while
    // remaining in the same sentence.
    if (
      curType === "Conj" &&
      this.clauseComplete(
        sentenceState
      )
    ) {
      this.resetSentenceState(
        sentenceState
      );
      return;
    }

    // Resolve dependencies opened by earlier words.
    if (
      sentenceState.needsBaseVerb &&
      curType === "Verb"
    ) {
      sentenceState.needsBaseVerb =
        false;
    }

    if (
      wasNeedPrepObject &&
      [
        "Noun", "ProperNoun",
        "Pronoun", "Num"
      ].includes(curType)
    ) {
      sentenceState.needsObjectAfterPrep =
        false;
    }

    if (
      wasNeedNominal &&
      [
        "Noun", "ProperNoun",
        "Num"
      ].includes(curType)
    ) {
      sentenceState.needsNominalAfterDet =
        false;
    }

    if (curType === "Det") {
      sentenceState.needsNominalAfterDet =
        true;
    }

    if (curType === "Prep") {
      sentenceState.prepCount++;
      sentenceState.needsObjectAfterPrep =
        true;
    }

    // Subject establishment.
    if (
      !sentenceState.subjectReady &&
      [
        "Noun", "ProperNoun",
        "Pronoun"
      ].includes(curType)
    ) {
      sentenceState.subjectReady = true;
      sentenceState.subjectToken = token;

      if (
        curType === "Pronoun" &&
        pronoun
      ) {
        sentenceState.subjectPronounWord =
          lower;
        sentenceState.subjectPerson =
          pronoun.person;
        sentenceState.subjectNumber =
          pronoun.number;

        sentenceState.isPluralSubject =
          pronoun.number === "plural";
      } else {
        sentenceState.subjectPronounWord =
          "";
        sentenceState.subjectPerson = 3;
        sentenceState.subjectNumber =
          lower.endsWith("s") &&
          !lower.endsWith("ss")
            ? "plural"
            : "singular";

        sentenceState.isPluralSubject =
          sentenceState.subjectNumber ===
          "plural";
      }

      sentenceState.nounCount++;
    } else if (
      [
        "Noun", "ProperNoun",
        "Pronoun"
      ].includes(curType)
    ) {
      sentenceState.nounCount++;

      if (!sentenceState.predicateReady) {
        // Before the predicate, the newest nominal in a compact noun compound
        // is usually the grammatical head.
        sentenceState.subjectToken =
          token;

        if (
          curType === "Pronoun" &&
          pronoun
        ) {
          sentenceState.subjectPronounWord =
            lower;
          sentenceState.subjectPerson =
            pronoun.person;
          sentenceState.subjectNumber =
            pronoun.number;
          sentenceState.isPluralSubject =
            pronoun.number === "plural";
        } else {
          sentenceState.isPluralSubject =
            lower.endsWith("s") &&
            !lower.endsWith("ss");
        }
      } else {
        sentenceState.objectCount++;
      }
    }

    const doSupport =
      new Set([
        "do", "does", "did"
      ]);

    const copulas =
      new Set([
        "am", "is", "are",
        "was", "were", "be",
        "been", "being"
      ]);

    if (curType === "Modal") {
      sentenceState.needsBaseVerb =
        true;
      sentenceState.verbCount++;
    } else if (curType === "Verb") {
      sentenceState.verbCount++;

      if (
        sentenceState.subjectReady
      ) {
        sentenceState.predicateReady =
          true;

        sentenceState.lexicalPredicateCount =
          (sentenceState.lexicalPredicateCount || 0) +
          1;

        sentenceState.copulaOpen =
          false;

        if (
          !Number.isFinite(
            sentenceState.predicateToken
          )
        ) {
          sentenceState.predicateToken =
            token;
        }
      }
    } else if (curType === "Aux") {
      sentenceState.verbCount++;

      if (doSupport.has(lower)) {
        sentenceState.needsBaseVerb =
          true;
      }

      if (copulas.has(lower)) {
        sentenceState.copulaOpen =
          true;
      }
    }

    // Copular predicates: "X is useful", "X is a model".
    if (
      sentenceState.subjectReady &&
      sentenceState.verbCount > 0 &&
      [
        "Adj", "Noun",
        "ProperNoun"
      ].includes(curType)
    ) {
      sentenceState.predicateReady =
        true;

      if (
        sentenceState.copulaOpen
      ) {
        sentenceState.copulaOpen =
          false;
      }
    }

    if (curType === "Adv") {
      sentenceState.consecutiveAdverbs =
        (sentenceState.consecutiveAdverbs || 0) +
        1;
    } else if (curType !== "Punct") {
      sentenceState.consecutiveAdverbs =
        0;
    }

    if (
      wasPredicateReady &&
      this.isContentType(curType)
    ) {
      sentenceState.postPredicateContentCount =
        (sentenceState.postPredicateContentCount || 0) +
        1;

      if (
        ["Adv", "Adj"].includes(curType)
      ) {
        sentenceState.trailingModifierCount =
          (sentenceState.trailingModifierCount || 0) +
          1;
      } else {
        sentenceState.trailingModifierCount =
          Math.max(
            0,
            (sentenceState.trailingModifierCount || 0) - 1
          );
      }
    }

    this.updateClauseSemanticState(
      sentenceState,
      token,
      curType
    );

    sentenceState.isStart = false;
  }


  // ========================================================
  // RSL v2: SINGLE-FORWARD CONSTRAINED SEARCH
  // ========================================================
  //
  // The previous RSL repeatedly generated and rescored whole answers from the
  // same neural evidence. RSL v2 instead:
  //   1) runs ONE neural forward,
  //   2) builds ONE semantic/contract state,
  //   3) searches a tiny set of competing answer paths simultaneously,
  //   4) verifies only the finalists.
  //
  // This keeps "High" fundamentally different from Low while avoiding repeated
  // full-answer CPU passes and repeated context nudging.

  createRSLDecodeState() {
    return {
      isStart: true,
      subjectReady: false,
      predicateReady: false,
      subjectToken: null,
      predicateToken: null,
      lastContentToken: null,
      clauseContentCount: 0,
      clauseSemanticEMA: 0.72,
      clauseDriftCount: 0,
      verbCount: 0,
      nounCount: 0,
      objectCount: 0,
      prepCount: 0,
      isPluralSubject: false,

      // RSL v8 Human-Decodable Proposition state.
      subjectPronounWord: "",
      subjectPerson: null,
      subjectNumber: null,
      lexicalPredicateCount: 0,
      needsBaseVerb: false,
      needsObjectAfterPrep: false,
      needsNominalAfterDet: false,
      copulaOpen: false,
      consecutiveAdverbs: 0,
      trailingModifierCount: 0,
      postPredicateContentCount: 0,

      // Answer Slot Contract state. Primitive-only for cheap beam cloning.
      slotSubjectMask: 0,
      slotSubjectEvidence: 0.0,
      slotPredicateEvidence: 0.0,
      slotRelationEvidence: 0.0,
      slotRelationMarkerEvidence: 0.0,
      slotRelationContentEvidence: 0.0,
      slotRelationOpened: false,
      slotPredicateAfterSubject: false,
      slotRelationAfterPredicate: false
    };
  }

  cloneRSLDecodeState(state) {
    // All decode-state fields are primitives, so a shallow copy is both exact
    // and substantially cheaper than structuredClone().
    return { ...state };
  }

  getRSLTargetLength(baseTarget, promptLen, contextModel) {
    const explicit = contextModel?.explicitRequestedLength;

    const shape =
      contextModel?.answerShape?.kind ||
      "prose";

    if (Number.isFinite(explicit)) {
      return Math.max(
        1,
        Math.min(this.maxTokenCeiling, Math.round(explicit))
      );
    }

    if(shape==="temporal"){
      const mode=contextModel?.answerShape?.temporalMode||"direct";
      if(mode==="predictive")return Math.max(9,Math.min(baseTarget,20));
      if(mode==="relative")return Math.max(7,Math.min(baseTarget,16));
      return Math.max(3,Math.min(baseTarget,10));
    }

    if (shape === "short") {
      return Math.max(
        5,
        Math.min(
          baseTarget,
          12
        )
      );
    }

    if (shape === "yes_no") {
      return Math.max(
        7,
        Math.min(
          baseTarget,
          16
        )
      );
    }

    if (shape === "definition") {
      return Math.max(
        10,
        Math.min(
          baseTarget,
          22
        )
      );
    }

    if (shape === "list") {
      const items =
        Math.max(
          2,
          contextModel
            ?.answerShape
            ?.targetItems || 3
        );

      return Math.max(
        14,
        Math.min(
          baseTarget,
          8 + items * 7
        )
      );
    }

    // Short questions should produce compact answers. Let genuinely large or
    // complex prompts keep the full adaptive target.
    if (
      contextModel?.questionProfile?.isQuestion &&
      promptLen <= 18
    ) {
      const contentCount =
        contextModel?.promptContentTokens?.length || 1;

      const compactTarget =
        8 +
        contentCount * 1.55 +
        Math.sqrt(Math.max(1, promptLen)) * 0.90;

      return Math.max(
        10,
        Math.min(
          baseTarget,
          Math.round(compactTarget)
        )
      );
    }

    return baseTarget;
  }

  buildConfidenceRecoveryProfile(
    diagnostics,
    contextModel
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const act =
      clamp(
        diagnostics
          ?.independentActFulfillment
      );

    const proposition =
      clamp(
        diagnostics
          ?.independentPropositionCompleteness
      );

    const relation =
      clamp(
        diagnostics
          ?.independentPromptRelationMatch
      );

    const grounding =
      clamp(
        diagnostics
          ?.independentGrounding
      );

    const language =
      clamp(
        diagnostics
          ?.independentLanguageIntegrity
      );

    const epistemic =
      clamp(
        diagnostics
          ?.independentEpistemic
      );

    return {
      enabled: true,
      requiredAct:
        diagnostics
          ?.independentRequiredAct ||
        this.getIndependentRequiredAnswerAct(
          contextModel
        ),
      actNeed:
        1.0 - act,
      propositionNeed:
        1.0 - proposition,
      relationNeed:
        1.0 - relation,
      groundingNeed:
        1.0 - grounding,
      languageNeed:
        1.0 - language,
      epistemicNeed:
        1.0 - epistemic,
      previousConfidence:
        clamp(
          diagnostics?.final
        )
    };
  }

  getRecoveryCandidateBonus(
    candidateToken,
    contextModel,
    sentenceState,
    semanticGraphFit,
    provenance
  ) {
    const recovery =
      contextModel
        ?.recoveryProfile;

    if (!recovery?.enabled) {
      return 0.0;
    }

    const type =
      this.getCanonicalType(
        candidateToken
      );

    const typeFit =
      this.getSemanticFrameTypeCompatibility(
        candidateToken,
        contextModel
      );

    const graphFit =
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(
            semanticGraphFit
          )
            ? semanticGraphFit
            : 0
        )
      );

    const grounded =
      Math.max(
        0,
        Math.min(
          1,
          Math.max(
            provenance?.grounding || 0,
            provenance?.factual || 0
          )
        )
      );

    let bonus =
      graphFit *
      (
        recovery.propositionNeed *
          this.reasoningConfig
            .refinementGraphPressure +
        recovery.relationNeed *
          this.reasoningConfig
            .refinementRelationPressure
      );

    bonus +=
      typeFit *
      recovery.actNeed *
      this.reasoningConfig
        .refinementActPressure;

    bonus +=
      grounded *
      recovery.groundingNeed *
      this.reasoningConfig
        .refinementGroundingPressure;

    // When the verifier says language is weak, prefer grammatical bridge
    // categories instead of piling on more unrelated content words.
    if (
      recovery.languageNeed > 0.30
    ) {
      if (
        [
          "Aux", "Modal", "Prep",
          "Conj", "Det", "Verb"
        ].includes(type)
      ) {
        bonus +=
          recovery.languageNeed *
          this.reasoningConfig
            .refinementLanguagePressure;
      } else if (
        sentenceState
          ?.trailingModifierCount >= 2 &&
        ["Adj", "Adv"].includes(type)
      ) {
        bonus -=
          recovery.languageNeed *
          1.2;
      }
    }

    // Reusing a prompt anchor is useful when proposition identity was lost,
    // but it must not substitute for the missing semantic relation.
    if (
      recovery.propositionNeed > 0.35 &&
      this.isSourceAnchorFamily(
        candidateToken,
        contextModel
      )
    ) {
      bonus +=
        recovery.propositionNeed *
        0.65;
    }

    return Math.max(
      -2.5,
      Math.min(
        6.0,
        bonus
      )
    );
  }

  isRecoveredAnswerAcceptable(
    diagnostics
  ) {
    if (!diagnostics) {
      return false;
    }

    const confidence =
      Number.isFinite(
        diagnostics.final
      )
        ? diagnostics.final
        : 0.0;

    const language =
      Number.isFinite(
        diagnostics
          .independentLanguageIntegrity
      )
        ? diagnostics
            .independentLanguageIntegrity
        : 0.0;

    return Boolean(
      diagnostics.independentResolved
    ) &&
    confidence >=
      this.reasoningConfig
        .answerAcceptanceConfidence &&
    language >=
      this.reasoningConfig
        .professionalLanguageFloor;
  }

  getRecoveryCandidateUtility(
    diagnostics
  ) {
    if (!diagnostics) {
      return -Infinity;
    }

    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    return (
      clamp(
        diagnostics
          .independentActFulfillment
      ) * 0.22 +
      clamp(
        diagnostics
          .independentPropositionCompleteness
      ) * 0.20 +
      clamp(
        diagnostics
          .independentPromptRelationMatch
      ) * 0.23 +
      clamp(
        diagnostics
          .independentGrounding
      ) * 0.10 +
      clamp(
        diagnostics
          .independentLanguageIntegrity
      ) * 0.15 +
      clamp(
        diagnostics
          .independentContradictionIntegrity
      ) * 0.05 +
      clamp(
        diagnostics.final
      ) * 0.05
    );
  }

  getRSLSearchConfig(
    promptLen,
    targetLength,
    contextModel
  ) {
    const explicit =
      contextModel
        ?.explicitRequestedLength;

    const isQuestion =
      Boolean(
        contextModel
          ?.questionProfile
          ?.isQuestion
      );

    const factual =
      Boolean(
        contextModel
          ?.answerPlan
          ?.factualityRequired
      );

    const refinementRound =
      Math.max(
        0,
        Math.min(
          this.reasoningConfig
            .refinementMaxRounds,
          Number(
            contextModel
              ?.refinementRound
          ) || 0
        )
      );

    let beamWidth =
      (isQuestion || factual)
        ? 3
        : 2;

    if (
      promptLen > 26 ||
      targetLength > 34
    ) {
      beamWidth =
        Math.max(
          beamWidth,
          3
        );
    }

    if (
      promptLen > 90 ||
      targetLength > 100
    ) {
      beamWidth = 4;
    }

    if (
      Number.isFinite(explicit) &&
      explicit >= 120
    ) {
      beamWidth =
        Math.max(
          beamWidth,
          4
        );
    }

    beamWidth =
      Math.min(
        this.reasoningConfig
          .refinementMaxBeam,
        beamWidth +
        refinementRound *
        this.reasoningConfig
          .refinementBeamGrowth
      );

    let branchFactor =
      beamWidth >= 4
        ? 3
        : 2;

    branchFactor =
      Math.min(
        this.reasoningConfig
          .refinementMaxBranch,
        branchFactor +
        Math.max(
          0,
          refinementRound - 1
        ) *
        this.reasoningConfig
          .refinementBranchGrowth
      );

    const baseAmbiguity =
      promptLen <= 18
        ? 1.30
        : 1.18;

    const ambiguityMargin =
      Math.max(
        0.62,
        baseAmbiguity -
        refinementRound *
        this.reasoningConfig
          .refinementAmbiguityDrop
      );

    const verifyCount =
      Math.min(
        this.reasoningConfig
          .refinementMaxVerify,
        Math.min(
          5,
          beamWidth + 1
        ) +
        refinementRound *
        this.reasoningConfig
          .refinementVerifyGrowth
      );

    return {
      beamWidth,
      branchFactor,
      ambiguityMargin,
      verifyCount,
      refinementRound
    };
  }

  selectRSLChoices(
    scores,
    candidateIds,
    limit = 2,
    ambiguityMargin = 1.5
  ) {
    const top = [];

    const scanCount =
      candidateIds?.length
        ? candidateIds.length
        : scores.length;

    for (let scanIndex = 0; scanIndex < scanCount; scanIndex++) {
      const token =
        candidateIds?.length
          ? candidateIds[scanIndex]
          : scanIndex;

      const score = scores[token];
      if (score <= -1e8) continue;

      let insertAt = top.length;

      for (let j = 0; j < top.length; j++) {
        if (score > top[j].score) {
          insertAt = j;
          break;
        }
      }

      if (insertAt < limit + 1) {
        top.splice(insertAt, 0, {
          token,
          score
        });

        if (top.length > limit + 1) {
          top.length = limit + 1;
        }
      }
    }

    if (!top.length) return [];

    // High confidence token decisions stay greedy. Beam width is only spent
    // where the score landscape is genuinely ambiguous.
    let keep = Math.min(limit, top.length);

    if (
      top.length >= 2 &&
      top[0].score - top[1].score >= ambiguityMargin
    ) {
      keep = 1;
    }

    const selected = top.slice(0, keep);
    const maxScore = selected[0].score;

    let sumExp = 0.0;
    for (const item of selected) {
      item.exp = Math.exp(
        Math.max(-12, item.score - maxScore)
      );
      sumExp += item.exp;
    }

    for (let i = 0; i < selected.length; i++) {
      const nextScore =
        i + 1 < top.length
          ? top[i + 1].score
          : top[i].score - 8.0;

      selected[i].prob =
        sumExp > 0
          ? selected[i].exp / sumExp
          : (i === 0 ? 1.0 : 0.0);

      selected[i].margin = Math.max(
        0,
        selected[i].score - nextScore
      );

      delete selected[i].exp;
    }

    return selected;
  }

  getRSLBeamPriority(
    beam,
    targetLength,
    contextModel
  ) {
    const length =
      Math.max(1, beam.tokens.length);

    const averagePathScore =
      beam.pathScore / length;

    // Compress arbitrary token-score magnitude into a stable 0..1 channel.
    const pathEvidence =
      1.0 /
      (
        1.0 +
        Math.exp(
          -Math.max(-12, Math.min(12, averagePathScore / 5.0))
        )
      );

    const coverage =
      this.promptCoverage(
        beam.tokens,
        contextModel
      );

    const semanticThread = Math.max(
      0,
      Math.min(
        1,
        Number.isFinite(
          beam.state?.clauseSemanticEMA
        )
          ? beam.state.clauseSemanticEMA
          : 0.72
      )
    );

    const relationSatisfied =
      this.answerRelationSatisfied(
        beam.tokens,
        beam.state,
        contextModel
      );

    const slotProgress =
      this.getLiveAnswerSlotProgress(
        beam.state,
        contextModel
      );

    // Cheap whole-path grounded association. Only inspect the last content
    // token against the path's recent semantic thread and stable prompt anchors.
    const associationProgress =
      beam.tokens.length
        ? this.getGroundedWordAssociation(
            beam.tokens[beam.tokens.length - 1],
            beam.tokens.slice(0, -1),
            contextModel,
            beam.state,
            this.buildLocalPatternContext(
              beam.tokens.slice(0, -1)
            )
          ).evidence
        : 0.62;

    // Target length is a ceiling, not an attractor. Shorter resolved beams do
    // not lose priority merely because they stopped before the estimate.
    const lengthSafety =
      length <= targetLength
        ? 1.0
        : Math.max(
            0,
            1.0 -
            (length - targetLength) /
            Math.max(4, targetLength)
          );

    const instructionProgress =
      Math.max(
        0,
        Math.min(
          1,
          slotProgress.overall * 0.54 +
          semanticThread * 0.30 +
          (relationSatisfied ? 0.16 : 0.0)
        )
      );

    return (
      pathEvidence * 0.20 +
      coverage * 0.05 +
      semanticThread * 0.16 +
      associationProgress * 0.08 +
      slotProgress.overall * 0.25 +
      instructionProgress * 0.15 +
      (relationSatisfied ? 0.04 : 0.0) +
      lengthSafety * 0.01 +
      Math.max(0, Math.min(1, beam.liveQuality ?? 0.55)) * 0.09 +
      Math.max(0, Math.min(1, 0.50 + (beam.qualitySlope ?? 0.0) * 4.0)) * 0.04 +
      (beam.finished ? 0.01 : 0.0)
    );
  }

  async executeRSLSearch(
    cleanPromptTokens,
    contextModel,
    targetLength,
    maxTokens,
    preparedRawLogits = null
  ) {
    const reusingNeuralEvidence =
      preparedRawLogits &&
      preparedRawLogits.length ===
        this.network.vocabSize;

    // High/RSL gets exactly one neural forward. Recovery searches reuse the
    // same logits and only spend bounded CPU beam-search work.
    const rawLogits =
      reusingNeuralEvidence
        ? preparedRawLogits
        : await this.network
            .forwardSequence(
              cleanPromptTokens
            );

    if (
      !reusingNeuralEvidence ||
      !contextModel
        .reasoningInitialized
    ) {
      const latentReasoning =
        this.runLatentReasoning(
          rawLogits,
          contextModel,
          contextModel
            .latentHopBudget || 1
        );

      contextModel.reasoningCentroid =
        latentReasoning.centroid;

      contextModel.reasoningScores =
        latentReasoning.scores;

      contextModel.reasoningStability =
        latentReasoning.stability;

      contextModel.reasoningHops =
        latentReasoning.hopsUsed;

      contextModel.reasoningInitialized =
        true;
    }

    const refinementRound =
      Math.max(
        0,
        Number(
          contextModel
            ?.refinementRound
        ) || 0
      );

    const cachedPoolRound =
      Number.isFinite(
        contextModel
          ?.deepCandidatePoolRound
      )
        ? contextModel
            .deepCandidatePoolRound
        : -1;

    if (
      !contextModel
        .deepCandidatePool
        ?.length ||
      cachedPoolRound <
        refinementRound
    ) {
      const priorPool =
        contextModel
          .deepCandidatePool;

      contextModel.deepCandidatePool =
        this.buildDeepCandidatePool(
          rawLogits,
          contextModel,
          cleanPromptTokens,
          {
            refinementRound,
            priorPool
          }
        );

      contextModel.deepCandidatePoolRound =
        refinementRound;
    }

    const neuralForwardCount =
      reusingNeuralEvidence
        ? 0
        : 1;

    const config =
      this.getRSLSearchConfig(
        cleanPromptTokens.length,
        targetLength,
        contextModel
      );

    const scoreScratch =
      new Float32Array(
        this.network.vocabSize
      );

    let beams = [
      {
        tokens: [],
        trajectory: [],
        state: this.createRSLDecodeState(),
        pathScore: 0.0,
        sentenceCount: 0,
        finished: false,
        priority: 0.0,
        liveQuality: 0.0,
        qualitySlope: 0.0,
        driftStreak: 0,
        completionLatched: false,
        bestPrefixQuality: -Infinity,
        bestPrefixTokens: null,
        bestPrefixTrajectory: null,
        bestPrefixState: null,
        bestPrefixSentenceCount: 0,
        bestPrefixPathScore: 0.0,
        bestPrefixResolved: false,
        discardedTrajectory: [],
        rolledBackTokens: 0
      }
    ];

    const finalists = [];
    let branchEvaluations = 0;
    let peakBeams = 1;

    for (
      let step = 0;
      step < maxTokens;
      step++
    ) {
      const expanded = [];

      for (const beam of beams) {
        if (beam.finished) {
          expanded.push(beam);
          continue;
        }

        const scores =
          this.buildScores(
            rawLogits,
            beam.tokens,
            contextModel,
            beam.state,
            true,
            step,
            targetLength,
            scoreScratch
          );

        const choices =
          this.selectRSLChoices(
            scores,
            contextModel.deepCandidatePool,
            config.branchFactor,
            config.ambiguityMargin
          );

        if (!choices.length) continue;

        branchEvaluations += choices.length;

        for (const choice of choices) {
          const token = choice.token;
          const word =
            this.network.vocab?.idToToken?.[token] || "";
          const type =
            this.getCanonicalType(token);

          const state =
            this.cloneRSLDecodeState(
              beam.state
            );

          const tokens =
            beam.tokens.concat(token);

          this.updateSentenceState(
            state,
            token,
            word,
            type
          );

          this.updateAnswerSlotState(
            state,
            token,
            type,
            contextModel
          );

          let sentenceCount =
            beam.sentenceCount;

          const trajectory =
            beam.trajectory.concat({
              token,
              prob: choice.prob,
              margin: choice.margin,
              rlFeatures: Array.from(
                this.getRLPolicyFeatures(
                  token,
                  beam.tokens,
                  contextModel,
                  beam.state,
                  rawLogits[token]
                )
              )
            });

          const pathScore =
            beam.pathScore +
            choice.score;

          const punctuation =
            [".", "!", "?"].includes(word);

          if (punctuation) {
            sentenceCount++;
          }

          const live =
            this.getLivePrefixDiagnostics(
              tokens,
              contextModel,
              state
            );

          const previousQuality =
            Number.isFinite(beam.liveQuality)
              ? beam.liveQuality
              : live.quality;

          const qualitySlope =
            live.quality -
            previousQuality;

          let bestPrefixQuality =
            beam.bestPrefixQuality;

          let bestPrefixTokens =
            beam.bestPrefixTokens;

          let bestPrefixTrajectory =
            beam.bestPrefixTrajectory;

          let bestPrefixState =
            beam.bestPrefixState;

          let bestPrefixSentenceCount =
            beam.bestPrefixSentenceCount;

          let bestPrefixPathScore =
            beam.bestPrefixPathScore;

          let bestPrefixResolved =
            Boolean(
              beam.bestPrefixResolved
            );

          const checkpointEligible =
            this.isPrefixCheckpointEligible(
              tokens,
              state,
              contextModel,
              live
            );

          const checkpointResolved =
            Boolean(
              live
                .eventQuery
                .querySlotResolved
            ) &&
            live.eventQuery.resolved;

          if (
            checkpointEligible &&
            (
              (
                checkpointResolved &&
                !bestPrefixResolved
              ) ||
              (
                checkpointResolved ===
                  bestPrefixResolved &&
                (
                  !Number.isFinite(
                    bestPrefixQuality
                  ) ||
                  live.quality >=
                    bestPrefixQuality +
                    this.reasoningConfig
                      .prefixCheckpointMinDelta
                )
              )
            )
          ) {
            bestPrefixQuality =
              live.quality;

            bestPrefixTokens =
              tokens.slice();

            bestPrefixTrajectory =
              trajectory.slice();

            bestPrefixState =
              this.cloneRSLDecodeState(
                state
              );

            bestPrefixSentenceCount =
              sentenceCount;

            bestPrefixPathScore =
              pathScore;

            bestPrefixResolved =
              checkpointResolved;
          }

          const dropFromBest =
            Number.isFinite(bestPrefixQuality)
              ? bestPrefixQuality - live.quality
              : 0.0;

          const drifting =
            qualitySlope <=
              -this.reasoningConfig
                .driftStepDrop ||
            dropFromBest >=
              this.reasoningConfig
                .driftBestDrop;

          const driftStreak =
            drifting
              ? (beam.driftStreak || 0) + 1
              : Math.max(
                  0,
                  (beam.driftStreak || 0) - 1
                );

          const completionLatched =
            Boolean(
              beam.completionLatched
            ) ||
            (
              checkpointEligible &&
              live
                .eventQuery
                .querySlotResolved &&
              live.eventQuery.score >=
                this.reasoningConfig
                  .completionLatchFloor
            );

          let finished = false;

          if (
            punctuation ||
            this.isContentType(type)
          ) {
            finished =
              this.shouldStopDeepAnswer(
                tokens,
                state,
                contextModel,
                targetLength,
                sentenceCount,
                live
              );
          }

          const driftStop =
            Number.isFinite(bestPrefixQuality) &&
            bestPrefixTokens?.length >=
              this.reasoningConfig
                .prefixCheckpointMinLength &&
            (
              (
                completionLatched &&
                driftStreak >=
                  this.reasoningConfig
                    .driftPatience &&
                dropFromBest >=
                  this.reasoningConfig
                    .driftBestDrop
              ) ||
              (
                driftStreak >=
                  this.reasoningConfig
                    .hardDriftPatience &&
                dropFromBest >=
                  this.reasoningConfig
                    .driftHardDrop
              )
            );

          const ceilingStop =
            !Number.isFinite(
              contextModel?.explicitRequestedLength
            ) &&
            tokens.length >=
              Math.max(1, targetLength);

          finished =
            finished ||
            driftStop ||
            ceilingStop;

          if (
            Number.isFinite(
              contextModel?.explicitRequestedLength
            ) &&
            tokens.length >=
              contextModel.explicitRequestedLength
          ) {
            finished = true;
          }

          let next = {
            tokens,
            state,
            sentenceCount,
            finished,
            pathScore,
            trajectory,
            liveQuality:
              live.quality,
            qualitySlope,
            driftStreak,
            completionLatched,
            bestPrefixQuality,
            bestPrefixTokens,
            bestPrefixTrajectory,
            bestPrefixState,
            bestPrefixSentenceCount,
            bestPrefixPathScore,
            bestPrefixResolved,
            discardedTrajectory:
              beam.discardedTrajectory || [],
            rolledBackTokens:
              beam.rolledBackTokens || 0
          };

          if (
            finished &&
            Number.isFinite(bestPrefixQuality) &&
            bestPrefixTokens?.length &&
            (
              driftStop ||
              bestPrefixQuality -
                live.quality >=
                this.reasoningConfig
                  .rollbackMargin
            )
          ) {
            next =
              this.materializeBestPrefix(
                next,
                driftStop
              );
          }

          if (
            punctuation &&
            !next.finished
          ) {
            this.resetSentenceState(
              next.state
            );
          }

          next.priority =
            this.getRSLBeamPriority(
              next,
              targetLength,
              contextModel
            );

          expanded.push(next);

          if (next.finished) {
            finalists.push(next);
          }
        }
      }

      if (!expanded.length) break;

      // Collapse exact duplicate token paths before keeping the best beams.
      const deduped = new Map();

      for (const beam of expanded) {
        const key =
          beam.tokens.join(",");

        const previous =
          deduped.get(key);

        if (
          !previous ||
          beam.priority > previous.priority
        ) {
          deduped.set(key, beam);
        }
      }

      beams =
        Array.from(deduped.values())
          .sort(
            (a, b) =>
              b.priority - a.priority
          )
          .slice(
            0,
            config.beamWidth
          );

      peakBeams = Math.max(
        peakBeams,
        beams.length
      );

      if (!beams.length) break;

      const best = beams[0];
      const bestFinished =
        best.finished;

      const allFinished =
        beams.every(
          beam => beam.finished
        );

      if (
        bestFinished &&
        (
          allFinished ||
          (
            best.tokens.length >=
              Math.max(
                7,
                Math.round(
                  targetLength * 0.50
                )
              ) &&
            (
              beams.length === 1 ||
              best.priority -
                beams[1].priority >=
                0.055
            )
          )
        )
      ) {
        break;
      }
    }

    // If no beam reached a natural terminal state, the strongest live beams
    // still become finalists and are closed with a period before verification.
    const pool =
      finalists.length
        ? finalists
        : beams;

    const uniqueFinalists = [];
    const seen = new Set();

    for (
      const beam of
        [...pool, ...beams]
          .sort(
            (a, b) =>
              b.priority - a.priority
          )
    ) {
      const key =
        beam.tokens.join(",");

      if (seen.has(key)) continue;
      seen.add(key);

      uniqueFinalists.push(beam);

      if (
        uniqueFinalists.length >=
        config.verifyCount
      ) {
        break;
      }
    }

    const periodId =
      this.network.vocab?.tokenToId?.get?.(".") ??
      432;

    let winner = null;
    let winnerUtility = -Infinity;

    for (const sourceBeam of uniqueFinalists) {
      const beam =
        this.materializeBestPrefix(
          sourceBeam,
          false
        );

      const tokens =
        beam.tokens.slice();

      const trajectory =
        beam.trajectory.slice();

      const lastWord =
        this.network.vocab?.idToToken?.[
          tokens[tokens.length - 1]
        ] || "";

      if (
        ![".", "!", "?"].includes(lastWord)
      ) {
        tokens.push(periodId);

        trajectory.push({
          token: periodId,
          prob: 0.78,
          margin: 0.0,
          scaffold: true
        });
      }

      const candidate = {
        tokens,
        trajectory,
        rawLogits,
        livePrefixQuality:
          Number.isFinite(beam.liveQuality)
            ? beam.liveQuality
            : null,
        bestPrefixQuality:
          Number.isFinite(beam.bestPrefixQuality)
            ? beam.bestPrefixQuality
            : null,
        rolledBackTokens:
          beam.rolledBackTokens || 0
      };

      const evaluation =
        this.verifyCandidate(
          candidate,
          contextModel,
          targetLength
        );

      // RL already influenced token actions while the beams were built. Final
      // selection is deliberately dominated by an independent answer parser.
      const utility =
        evaluation.independentVerifierScore * 0.56 +
        evaluation.independentActFulfillmentScore * 0.10 +
        evaluation.independentPropositionCompletenessScore * 0.08 +
        evaluation.independentPromptRelationMatchScore * 0.10 +
        evaluation.independentGroundingScore * 0.05 +
        evaluation.independentLanguageIntegrityScore * 0.05 +
        evaluation.score * 0.04 +
        evaluation.tailIntegrityScore * 0.02;

      if (
        !winner ||
        utility > winnerUtility
      ) {
        winnerUtility = utility;

        winner = {
          ...candidate,
          ...evaluation,
          rslMeta: {
            beamWidth:
              config.beamWidth,
            peakBeams,
            branchFactor:
              config.branchFactor,
            branchEvaluations,
            finalistsVerified:
              uniqueFinalists.length,
            candidatePoolSize:
              contextModel
                ?.deepCandidatePool
                ?.length || 0,
            candidatePoolRound:
              contextModel
                ?.deepCandidatePoolStats
                ?.refinementRound || 0,
            candidatePoolFamilyLimit:
              contextModel
                ?.deepCandidatePoolStats
                ?.familyLimit || 0,
            neuralForwards: neuralForwardCount,
            rolledBackTokens:
              beam.rolledBackTokens || 0,
            discardedTrajectory:
              (beam.discardedTrajectory || []).slice(),
            bestPrefixQuality:
              Number.isFinite(beam.bestPrefixQuality)
                ? beam.bestPrefixQuality
                : null,
            finalPrefixQuality:
              Number.isFinite(beam.liveQuality)
                ? beam.liveQuality
                : null
          }
        };
      }
    }

    if (!winner) {
      return {
        tokens: [],
        trajectory: [],
        rawLogits,
        score: 0.0,
        rslMeta: {
          beamWidth:
            config.beamWidth,
          peakBeams,
          branchFactor:
            config.branchFactor,
          branchEvaluations,
          finalistsVerified: 0,
          candidatePoolSize:
            contextModel
              ?.deepCandidatePool
              ?.length || 0,
          candidatePoolRound:
            contextModel
              ?.deepCandidatePoolStats
              ?.refinementRound || 0,
          candidatePoolFamilyLimit:
            contextModel
              ?.deepCandidatePoolStats
              ?.familyLimit || 0,
          neuralForwards: neuralForwardCount
        }
      };
    }

    return winner;
  }

  async executeRollout(
    cleanPromptTokens,
    contextModel,
    targetLength,
    maxTokens,
    temperature,
    deepMode,
    sharedRawLogits = null
  ) {
    const trajectory = [];
    const generated = [];
    let step = 0;
    let rawLogits = null;

    const sentenceState = {
      isStart: true,
      subjectReady: false,
      predicateReady: false,
      subjectToken: null,
      predicateToken: null,
      lastContentToken: null,
      clauseContentCount: 0,
      clauseSemanticEMA: 0.72,
      clauseDriftCount: 0,
      verbCount: 0,
      nounCount: 0,
      objectCount: 0,
      prepCount: 0,
      isPluralSubject: false
    };

    // HIGH-PERFORMANCE DEEP PATH:
    // One real neural forward pass for the whole answer. No cache, no canned
    // answer, no second model. The neural logits are prompt-derived once, then
    // the POS/clause planner spends cheap CPU work assembling the response.
    // This removes the dominant latency source: a 15-layer WebGPU dispatch +
    // GPU->CPU readback for every generated token.
    if (deepMode) {
      rawLogits = sharedRawLogits ||
        await this.network.forwardSequence(cleanPromptTokens);

      // Initialize latent reasoning ONCE. Later RSL passes preserve and refine
      // this state instead of throwing it away and rebuilding it.
      if (!contextModel.reasoningInitialized) {
        const latentReasoning = this.runLatentReasoning(
          rawLogits,
          contextModel,
          contextModel.latentHopBudget || 1
        );

        contextModel.reasoningCentroid = latentReasoning.centroid;
        contextModel.reasoningScores = latentReasoning.scores;
        contextModel.reasoningStability = latentReasoning.stability;
        contextModel.reasoningHops = latentReasoning.hopsUsed;
        contextModel.reasoningInitialized = true;

        // Build the expensive vocabulary-pruning structure once per prompt.
        contextModel.deepCandidatePool =
          this.buildDeepCandidatePool(
            rawLogits,
            contextModel,
            cleanPromptTokens,
            {
              refinementRound: 0,
              priorPool: null
            }
          );

        contextModel.deepCandidatePoolRound =
          0;
      }

      // First try the universal grammatical planner. It now routes semantics
      // through topic + intent centroids plus the refined latent state.
      // nothing after the single neural pass and directly fixes the biggest
      // current failure: knowing the right words but placing them in nonsense
      // order.
      const frame = this.buildUniversalPromptFrame(
        cleanPromptTokens,
        rawLogits,
        contextModel
      );

      if (frame?.complete && frame.tokens.length >= 3) {
        for (const tok of frame.tokens) {
          if (step >= maxTokens) break;

          const word = this.tokenWord(tok);
          const curType = this.getCanonicalType(tok);
          generated.push(tok);
          this.updateSentenceState(sentenceState, tok, word, curType);
          step++;
        }

        trajectory.push(...this.buildFrameTrajectory(frame.tokens, rawLogits));
      }

      // Continue from the frame (or from an empty frame) using the SAME neural
      // logits. Deep can now produce multiple sentences and may consume any
      // amount up to the 360-token ceiling when the prompt warrants it.
      let sentenceCount = 0;

      while (step < maxTokens) {
        const scores = this.buildScores(
          rawLogits,
          generated,
          contextModel,
          sentenceState,
          true,
          step,
          targetLength
        );

        const picked = this.selectDeepToken(
          scores,
          contextModel.deepCandidatePool
        );
        if (picked.token < 0) break;

        const sampledToken = picked.token;
        const word = this.network.vocab?.idToToken?.[sampledToken] || "";
        const curType = this.getCanonicalType(sampledToken);

        trajectory.push({
          token: sampledToken,
          prob: picked.prob,
          margin: picked.margin || 0.0
        });

        generated.push(sampledToken);
        this.updateSentenceState(
          sentenceState,
          sampledToken,
          word,
          curType
        );
        step++;

        if ([".", "!", "?"].includes(word)) {
          sentenceCount++;

          const stop = this.shouldStopDeepAnswer(
            generated,
            sentenceState,
            contextModel,
            targetLength,
            sentenceCount
          );

          if (stop || step >= maxTokens) break;

          this.resetSentenceState(sentenceState);
          continue;
        }

        // If an explicit length request reaches its ceiling mid-sentence, let
        // the final punctuation append below close it cleanly.
        if (
          contextModel?.explicitRequestedLength !== null &&
          contextModel?.explicitRequestedLength !== undefined &&
          step >= contextModel.explicitRequestedLength
        ) {
          break;
        }
      }
    } else {
      // LOW PATH keeps the original autoregressive behavior. It asks the neural
      // network to refresh its logits after each generated token.
      let state = [...cleanPromptTokens];

      while (step < maxTokens) {
        const rawLogits = await this.network.forwardSequence(state);
        const scores = this.buildScores(rawLogits, generated, contextModel, sentenceState, false, step, targetLength);
        const picked = this.selectLowToken(scores, temperature);
        if (picked.token < 0) break;

        const sampledToken = picked.token;
        const word = this.network.vocab?.idToToken?.[sampledToken] || "";
        const curType = this.getCanonicalType(sampledToken);

        trajectory.push({
          token: sampledToken,
          prob: picked.prob,
          margin: 0.0,
          rlFeatures: Array.from(
            this.getRLPolicyFeatures(
              sampledToken,
              generated,
              contextModel,
              sentenceState,
              rawLogits[sampledToken]
            )
          )
        });
        generated.push(sampledToken);
        state.push(sampledToken);
        this.updateSentenceState(sentenceState, sampledToken, word, curType);
        step++;

        if ([".", "!", "?"].includes(word)) {
          if (step >= targetLength) break;

          this.resetSentenceState(sentenceState);
        }
      }
    }

    const lastWord = this.network.vocab?.idToToken?.[generated[generated.length - 1]] || "";
    if (![".", "!", "?"].includes(lastWord)) {
      const periodId = this.network.vocab?.tokenToId?.get?.(".") ?? 432;
      generated.push(periodId);
    }

    return {
      tokens: generated,
      trajectory,
      rawLogits: deepMode ? rawLogits : null
    };
  }


  splitAnswerItems(tokens) {
    const items = [];
    let current = [];

    for (const token of tokens || []) {
      const word =
        this.tokenWord(token);

      if (
        [
          ".",
          "!",
          "?",
          ";"
        ].includes(word)
      ) {
        if (current.length) {
          items.push(current);
          current = [];
        }
        continue;
      }

      current.push(token);
    }

    if (current.length) {
      items.push(current);
    }

    return items;
  }

  measureListDecodability(
    tokens,
    contextModel
  ) {
    const items =
      this.splitAnswerItems(tokens);

    if (!items.length) {
      return {
        score: 0.0,
        relevance: 0.0,
        readability: 0.0,
        parallelism: 0.0,
        diversity: 0.0,
        itemCoverage: 0.0,
        itemCount: 0
      };
    }

    let relevanceSum = 0.0;
    let readabilitySum = 0.0;

    const headFamilies = [];
    const allStems = [];

    for (const item of items) {
      const content =
        item.filter(
          token =>
            this.isContentType(
              this.getCanonicalType(
                token
              )
            )
        );

      let itemRel = 0.0;

      for (const token of content) {
        const routed =
          Math.max(
            0,
            Math.min(
              1,
              (
                this.getRoutedSemanticScore(
                  token,
                  contextModel,
                  null
                ) +
                1.0
              ) /
              2.0
            )
          );

        itemRel += routed;

        const stem =
          this.stemWord(
            this.tokenWord(token)
          );

        if (stem.length >= 3) {
          allStems.push(stem);
        }
      }

      relevanceSum +=
        content.length
          ? itemRel /
            content.length
          : 0.0;

      const firstContent =
        content[0];

      if (
        Number.isInteger(
          firstContent
        )
      ) {
        const type =
          this.getCanonicalType(
            firstContent
          );

        headFamilies.push(
          type === "ProperNoun"
            ? "Noun"
            : type
        );
      }

      const functionCount =
        item.length -
        content.length;

      const advCount =
        item.filter(
          token =>
            this.getCanonicalType(
              token
            ) === "Adv"
        ).length;

      const lastType =
        item.length
          ? this.getCanonicalType(
              item[
                item.length - 1
              ]
            )
          : null;

      const lengthFit =
        content.length >= 1 &&
        content.length <= 10
          ? 1.0
          : Math.max(
              0.2,
              1.0 -
              Math.abs(
                content.length - 5
              ) /
              12
            );

      const functionFit =
        item.length
          ? Math.max(
              0,
              1.0 -
              functionCount /
              Math.max(
                2,
                item.length
              )
          )
          : 0.0;

      const modifierFit =
        Math.max(
          0,
          1.0 -
          Math.max(
            0,
            advCount - 1
          ) * 0.28
        );

      const closureFit =
        [
          "Prep",
          "Det",
          "Modal",
          "Aux",
          "Conj"
        ].includes(lastType)
          ? 0.15
          : 1.0;

      readabilitySum +=
        Math.max(
          0,
          Math.min(
            1,
            lengthFit * 0.38 +
            functionFit * 0.24 +
            modifierFit * 0.18 +
            closureFit * 0.20
          )
        );
    }

    const itemCount =
      items.length;

    const relevance =
      relevanceSum /
      itemCount;

    const readability =
      readabilitySum /
      itemCount;

    let parallelism = 0.70;

    if (headFamilies.length >= 2) {
      const counts =
        new Map();

      for (
        const family of
          headFamilies
      ) {
        counts.set(
          family,
          (
            counts.get(family) || 0
          ) + 1
        );
      }

      const dominant =
        Math.max(
          ...counts.values()
        );

      parallelism =
        dominant /
        headFamilies.length;
    }

    const diversity =
      allStems.length
        ? Math.max(
            0,
            Math.min(
              1,
              new Set(allStems).size /
              allStems.length
            )
          )
        : 0.0;

    const targetItems =
      Math.max(
        1,
        contextModel
          ?.answerShape
          ?.targetItems || 3
      );

    const itemCoverage =
      Math.max(
        0,
        Math.min(
          1,
          itemCount /
          targetItems
        )
      );

    const score =
      Math.max(
        0,
        Math.min(
          1,
          relevance * 0.31 +
          readability * 0.29 +
          parallelism * 0.16 +
          diversity * 0.12 +
          itemCoverage * 0.12
        )
      );

    return {
      score,
      relevance,
      readability,
      parallelism,
      diversity,
      itemCoverage,
      itemCount
    };
  }

  measureShortAnswerDecodability(
    tokens,
    contextModel
  ) {
    const content =
      (tokens || []).filter(
        token =>
          this.isContentType(
            this.getCanonicalType(
              token
            )
          ) ||
          this.getCanonicalType(
            token
          ) === "Pronoun"
      );

    if (!content.length) {
      return {
        score: 0.0,
        relevance: 0.0,
        directness: 0.0,
        roleFit: 0.0
      };
    }

    let relevanceSum = 0.0;
    let roleFitSum = 0.0;

    const state =
      this.createRSLDecodeState();

    for (const token of content) {
      const routed =
        Math.max(
          0,
          Math.min(
            1,
            (
              this.getRoutedSemanticScore(
                token,
                contextModel,
                state
              ) +
              1.0
            ) /
            2.0
          )
        );

      relevanceSum += routed;

      const fit =
        this.getRoleSensitiveKnowledgeEvidence(
          token,
          state,
          contextModel
        );

      roleFitSum +=
        fit.overall;

      this.updateSentenceState(
        state,
        token,
        this.tokenWord(token),
        this.getCanonicalType(
          token
        )
      );
    }

    const relevance =
      relevanceSum /
      content.length;

    const roleFit =
      roleFitSum /
      content.length;

    const length =
      content.length;

    const directness =
      length <= 8
        ? 1.0
        : Math.max(
            0.25,
            1.0 -
            (length - 8) /
            18
          );

    const score =
      Math.max(
        0,
        Math.min(
          1,
          relevance * 0.38 +
          roleFit * 0.38 +
          directness * 0.24
        )
      );

    return {
      score,
      relevance,
      directness,
      roleFit
    };
  }


  measureTemporalAnswerDecodability(
    tokens,
    contextModel,
    grammarDecodability,
    propositionDecodability
  ) {
    if (!tokens?.length) {
      return {
        score: 0.0,
        temporalEvidence: 0.0,
        temporalCoverage: 0.0,
        temporalTenseAlignment: 0.0,
        temporalContradictions: 0,
        directness: 0.0,
        clauseFit: 0.0
      };
    }

    let evidenceMax = 0.0;
    let evidenceSum = 0.0;
    let evidenceCount = 0;
    let strongCount = 0;
    let contentCount = 0;

    let alignmentSum = 0.0;
    let alignmentCount = 0;
    let contradictions = 0;
    let seenStrongTense = null;

    for (const token of tokens) {
      const type =
        this.getCanonicalType(token);

      if (
        this.isContentType(type)
      ) {
        contentCount++;
      }

      const temporal =
        this.temporalTokenEvidence(
          token,
          contextModel
        );

      evidenceMax =
        Math.max(
          evidenceMax,
          temporal.score
        );

      if (
        temporal.score >= 0.40
      ) {
        evidenceSum +=
          temporal.score;
        evidenceCount++;
      }

      if (
        temporal.score >= 0.68
      ) {
        strongCount++;
      }

      if (
        temporal.score >= 0.55
      ) {
        alignmentSum +=
          temporal.tenseCompatibility ?? 0.82;

        alignmentCount++;

        if (
          temporal.tenseContradiction
        ) {
          contradictions++;
        }

        if (
          temporal.tense !== "neutral" &&
          temporal.tenseStrength >= 0.60
        ) {
          if (
            seenStrongTense &&
            seenStrongTense !==
              temporal.tense
          ) {
            contradictions++;
          } else {
            seenStrongTense =
              temporal.tense;
          }
        }
      }
    }

    const temporalEvidence =
      evidenceCount
        ? Math.max(
            evidenceMax,
            evidenceSum /
            evidenceCount
          )
        : evidenceMax;

    const temporalCoverage =
      Math.max(
        0,
        Math.min(
          1,
          strongCount /
          Math.max(
            1,
            Math.min(
              2,
              contentCount
            )
          )
        )
      );

    const temporalTenseAlignment =
      alignmentCount
        ? Math.max(
            0,
            Math.min(
              1,
              alignmentSum /
              alignmentCount
            )
          )
        : (
            evidenceMax >= 0.68
              ? 0.84
              : 0.45
          );

    const mode =
      contextModel?.answerShape
        ?.temporalMode || "direct";

    const length =
      tokens.filter(
        token =>
          this.getCanonicalType(token)
            !== "Punct"
      ).length;

    const preferredMax =
      mode === "predictive"
        ? 18
        : mode === "relative"
          ? 14
          : 9;

    const directness =
      length <= preferredMax
        ? 1.0
        : Math.max(
            0.25,
            1.0 -
            (
              length -
              preferredMax
            ) /
            18
          );

    const grammar =
      grammarDecodability?.score || 0;

    const proposition =
      propositionDecodability?.score || 0;

    const clauseFit =
      mode === "predictive"
        ? (
            grammar * 0.42 +
            proposition * 0.58
          )
        : (
            grammar * 0.28 +
            proposition * 0.22 +
            0.50
          );

    let score =
      Math.max(
        0,
        Math.min(
          1,
          temporalEvidence * 0.34 +
          temporalCoverage * 0.12 +
          temporalTenseAlignment * 0.28 +
          directness * 0.10 +
          clauseFit * 0.16
        )
      );

    const minimum =
      contextModel?.answerShape
        ?.minimumTemporalEvidence ??
      0.68;

    if (
      evidenceMax < minimum
    ) {
      score *=
        0.42 +
        evidenceMax * 0.35;
    }

    if (contradictions > 0) {
      score *=
        Math.pow(
          0.42,
          Math.min(
            3,
            contradictions
          )
        );
    }

    return {
      score,
      temporalEvidence,
      temporalCoverage,
      temporalTenseAlignment,
      temporalContradictions:
        contradictions,
      directness,
      clauseFit
    };
  }

  measureAnswerShapeDecodability(
    tokens,
    contextModel,
    grammarDecodability,
    propositionDecodability
  ) {
    const shape =
      contextModel?.answerShape?.kind ||
      "prose";

    const grammar =
      grammarDecodability?.score || 0;

    const proposition =
      propositionDecodability?.score || 0;

    if(shape==="temporal"){
      const temporal=this.measureTemporalAnswerDecodability(tokens,contextModel,grammarDecodability,propositionDecodability);
      return {kind:shape,score:temporal.score,list:null,short:null,temporal};
    }

    if (shape === "list") {
      const list =
        this.measureListDecodability(
          tokens,
          contextModel
        );

      return {
        kind: shape,
        score:
          Math.max(
            0,
            Math.min(
              1,
              list.score * 0.82 +
              grammar * 0.18
            )
          ),
        list,
        short: null
      };
    }

    if (
      shape === "short" ||
      shape === "yes_no"
    ) {
      const short =
        this.measureShortAnswerDecodability(
          tokens,
          contextModel
        );

      const propositionWeight =
        shape === "yes_no"
          ? 0.34
          : 0.18;

      return {
        kind: shape,
        score:
          Math.max(
            0,
            Math.min(
              1,
              short.score *
              (
                1.0 -
                propositionWeight
              ) +
              proposition *
              propositionWeight
            )
          ),
        list: null,
        short
      };
    }

    if (shape === "definition") {
      const short =
        this.measureShortAnswerDecodability(
          tokens,
          contextModel
        );

      return {
        kind: shape,
        score:
          Math.max(
            0,
            Math.min(
              1,
              proposition * 0.62 +
              short.score * 0.38
            )
          ),
        list: null,
        short
      };
    }

    return {
      kind: "prose",
      score:
        Math.sqrt(
          Math.max(
            0,
            grammar *
            proposition
          )
        ),
      list: null,
      short: null
    };
  }

  measurePropositionDecodability(
    tokens,
    contextModel
  ) {
    if (!tokens?.length) {
      return {
        score: 0.0,
        roleFit: 0.0,
        roleCoverage: 0.0,
        focus: 0.0,
        subjectPredicate: 0.0
      };
    }

    let state =
      this.createRSLDecodeState();

    let roleFitSum = 0.0;
    let roleFitCount = 0;
    let strongRoleCount = 0;
    let weakRoleCount = 0;

    let clauseCount = 0;
    let clauseScoreSum = 0.0;

    let clauseRoleBest = 0.0;
    let clauseRoleSum = 0.0;
    let clauseRoleCount = 0;
    let clauseStrong = 0;
    let clauseWeak = 0;

    const contract =
      contextModel?.answerContract;

    const requestedRole =
      contract?.kind === "mechanism"
        ? "mechanism"
        : contract?.kind === "causal"
          ? "cause"
          : contract?.kind === "temporal"
            ? "time"
            : contract?.kind === "locative"
              ? "location"
              : contract?.kind === "quantitative"
                ? "quantity"
                : contract?.kind === "definition"
                  ? "definition"
                  : contract?.kind === "evaluative"
                    ? "evaluation"
                    : "content";

    const finishClause = () => {
      if (
        !state.subjectReady &&
        !state.predicateReady &&
        clauseRoleCount === 0
      ) {
        return;
      }

      clauseCount++;

      const subjectScore =
        state.subjectReady
          ? 1.0
          : 0.0;

      const predicateScore =
        state.predicateReady
          ? 1.0
          : 0.0;

      const relationRequired =
        contract?.requiredSlots?.some(
          slot =>
            ![
              "subject",
              "predicate"
            ].includes(slot)
        );

      const relationScore =
        relationRequired
          ? clauseRoleBest
          : (
              clauseRoleCount
                ? Math.max(
                    0.50,
                    clauseRoleBest
                  )
                : 0.72
            );

      const focus =
        clauseRoleCount
          ? Math.max(
              0,
              Math.min(
                1,
                (
                  clauseStrong +
                  Math.max(
                    0,
                    clauseRoleCount -
                    clauseStrong -
                    clauseWeak
                  ) *
                  0.45
                ) /
                clauseRoleCount
              )
            )
          : (
              relationRequired
                ? 0.25
                : 0.72
            );

      const averageRole =
        clauseRoleCount
          ? clauseRoleSum /
            clauseRoleCount
          : (
              relationRequired
                ? 0.20
                : 0.65
            );

      const clauseScore =
        Math.max(
          0,
          Math.min(
            1,
            subjectScore * 0.20 +
            predicateScore * 0.27 +
            relationScore * 0.28 +
            averageRole * 0.15 +
            focus * 0.10
          )
        );

      clauseScoreSum +=
        clauseScore;
    };

    for (const token of tokens) {
      const word =
        this.tokenWord(token);

      const type =
        this.getCanonicalType(token);

      if (
        [
          ".",
          "!",
          "?"
        ].includes(word)
      ) {
        finishClause();

        state =
          this.createRSLDecodeState();

        clauseRoleBest = 0.0;
        clauseRoleSum = 0.0;
        clauseRoleCount = 0;
        clauseStrong = 0;
        clauseWeak = 0;
        continue;
      }

      const predicateWasReady =
        Boolean(
          state.predicateReady
        );

      if (
        predicateWasReady &&
        this.isContentType(type) &&
        !contract?.subjectTokens?.includes(
          token
        ) &&
        token !==
          contract?.relationVerbToken
      ) {
        const fit =
          this.getRoleSensitiveKnowledgeEvidence(
            token,
            state,
            contextModel,
            requestedRole
          );

        roleFitSum +=
          fit.overall;

        roleFitCount++;

        clauseRoleSum +=
          fit.overall;

        clauseRoleCount++;

        clauseRoleBest =
          Math.max(
            clauseRoleBest,
            fit.overall
          );

        if (
          fit.overall >= 0.64
        ) {
          strongRoleCount++;
          clauseStrong++;
        } else if (
          fit.overall < 0.34
        ) {
          weakRoleCount++;
          clauseWeak++;
        }
      }

      this.updateSentenceState(
        state,
        token,
        word,
        type
      );
    }

    finishClause();

    const roleFit =
      roleFitCount
        ? Math.max(
            0,
            Math.min(
              1,
              roleFitSum /
              roleFitCount
            )
          )
        : 0.0;

    const roleCoverage =
      roleFitCount
        ? Math.max(
            0,
            Math.min(
              1,
              strongRoleCount /
              Math.max(
                1,
                Math.min(
                  roleFitCount,
                  4
                )
              )
            )
          )
        : 0.0;

    const focus =
      roleFitCount
        ? Math.max(
            0,
            Math.min(
              1,
              1.0 -
              weakRoleCount /
              roleFitCount
            )
          )
        : 0.0;

    const clauseScore =
      clauseCount
        ? clauseScoreSum /
          clauseCount
        : 0.0;

    const subjectPredicate =
      clauseCount
        ? Math.max(
            0,
            Math.min(
              1,
              clauseScore
            )
          )
        : 0.0;

    const score =
      Math.max(
        0,
        Math.min(
          1,
          clauseScore * 0.48 +
          roleFit * 0.24 +
          roleCoverage * 0.18 +
          focus * 0.10
        )
      );

    return {
      score,
      roleFit,
      roleCoverage,
      focus,
      subjectPredicate
    };
  }

  measureHumanDecodability(
    tokens,
    contextModel
  ) {
    if (!tokens?.length) {
      return {
        score: 0.0,
        propositionIntegrity: 0.0,
        dependencyIntegrity: 0.0,
        pronounIntegrity: 0.0,
        transitionIntegrity: 0.0,
        modifierIntegrity: 0.0
      };
    }

    let state =
      this.createRSLDecodeState();

    const generated = [];

    let transitionQualitySum = 0.0;
    let transitionCount = 0;
    let invalidTransitions = 0;

    let pronounChecks = 0;
    let pronounCorrect = 0;

    let dependencyChecks = 0;
    let dependencyResolved = 0;

    let maxAdvRun = 0;
    let advRun = 0;

    let clauseWords = 0;
    let clauseCount = 0;
    let propositionScoreSum = 0.0;

    const finishHumanClause = () => {
      if (clauseWords <= 0) {
        return;
      }

      clauseCount++;

      const complete =
        this.clauseComplete(
          state
        );

      const roleScore =
        (
          (state.subjectReady ? 0.34 : 0.0) +
          (state.predicateReady ? 0.42 : 0.0) +
          (
            state.subjectReady &&
            state.predicateReady
              ? 0.24
              : 0.0
          )
        );

      const unresolved =
        Number(
          Boolean(
            state.needsBaseVerb
          )
        ) +
        Number(
          Boolean(
            state.needsObjectAfterPrep
          )
        ) +
        Number(
          Boolean(
            state.needsNominalAfterDet
          )
        ) +
        Number(
          Boolean(
            state.copulaOpen
          )
        );

      const dependencyScore =
        Math.max(
          0,
          1.0 -
          unresolved * 0.25
        );

      // One finite proposition should not drag on for 30+ mostly unattached
      // content words. This directly models whether a human can retain and
      // decode the grammatical frame.
      const lengthScore =
        clauseWords <= 16
          ? 1.0
          : Math.max(
              0.18,
              Math.exp(
                -(clauseWords - 16) /
                15.0
              )
            );

      const predicateDensity =
        state.lexicalPredicateCount > 0
          ? Math.min(
              1,
              (
                state.lexicalPredicateCount *
                13
              ) /
              Math.max(
                7,
                clauseWords
              )
            )
          : (
              state.predicateReady
                ? 0.62
                : 0.0
            );

      const modifierScore =
        Math.max(
          0,
          Math.min(
            1,
            1.0 -
            Math.max(
              0,
              maxAdvRun - 1
            ) * 0.25 -
            Math.max(
              0,
              (state.trailingModifierCount || 0) - 1
            ) * 0.12
          )
        );

      const clauseScore =
        Math.max(
          0,
          Math.min(
            1,
            roleScore * 0.34 +
            dependencyScore * 0.20 +
            lengthScore * 0.19 +
            predicateDensity * 0.15 +
            modifierScore * 0.12
          )
        );

      propositionScoreSum +=
        complete
          ? clauseScore
          : clauseScore * 0.58;
    };

    for (const token of tokens) {
      const word =
        this.tokenWord(token);

      const type =
        this.getCanonicalType(token);

      if (type === "Punct") {
        finishHumanClause();

        generated.push(token);
        state =
          this.createRSLDecodeState();

        clauseWords = 0;
        advRun = 0;
        maxAdvRun = 0;
        continue;
      }

      const beforeNeedsBase =
        Boolean(
          state.needsBaseVerb
        );

      const beforeNeedsPrep =
        Boolean(
          state.needsObjectAfterPrep
        );

      const beforeNeedsNominal =
        Boolean(
          state.needsNominalAfterDet
        );

      const assessment =
        this.getHumanTransitionAssessment(
          generated,
          token,
          word,
          type,
          state,
          contextModel
        );

      transitionQualitySum +=
        assessment.quality;

      transitionCount++;

      if (!assessment.legal) {
        invalidTransitions++;
      }

      if (type === "Pronoun") {
        pronounChecks++;

        if (assessment.legal) {
          pronounCorrect++;
        }
      }

      this.updateSentenceState(
        state,
        token,
        word,
        type
      );

      if (beforeNeedsBase) {
        dependencyChecks++;

        if (!state.needsBaseVerb) {
          dependencyResolved++;
        }
      }

      if (beforeNeedsPrep) {
        dependencyChecks++;

        if (
          !state.needsObjectAfterPrep
        ) {
          dependencyResolved++;
        }
      }

      if (beforeNeedsNominal) {
        dependencyChecks++;

        if (
          !state.needsNominalAfterDet
        ) {
          dependencyResolved++;
        }
      }

      if (type === "Adv") {
        advRun++;
        maxAdvRun =
          Math.max(
            maxAdvRun,
            advRun
          );
      } else {
        advRun = 0;
      }

      generated.push(token);
      clauseWords++;
    }

    finishHumanClause();

    const transitionIntegrity =
      transitionCount
        ? Math.max(
            0,
            Math.min(
              1,
              (
                transitionQualitySum /
                transitionCount
              ) *
              (
                1.0 -
                Math.min(
                  0.75,
                  invalidTransitions /
                  transitionCount
                )
              )
            )
          )
        : 0.0;

    const propositionIntegrity =
      clauseCount
        ? Math.max(
            0,
            Math.min(
              1,
              propositionScoreSum /
              clauseCount
            )
          )
        : 0.0;

    const dependencyIntegrity =
      dependencyChecks
        ? dependencyResolved /
          dependencyChecks
        : 1.0;

    const pronounIntegrity =
      pronounChecks
        ? pronounCorrect /
          pronounChecks
        : 1.0;

    const modifierIntegrity =
      Math.max(
        0,
        Math.min(
          1,
          1.0 -
          Math.max(
            0,
            maxAdvRun - 1
          ) * 0.22
        )
      );

    const score =
      Math.max(
        0,
        Math.min(
          1,
          propositionIntegrity * 0.38 +
          transitionIntegrity * 0.27 +
          dependencyIntegrity * 0.15 +
          pronounIntegrity * 0.12 +
          modifierIntegrity * 0.08
        )
      );

    return {
      score,
      propositionIntegrity,
      dependencyIntegrity,
      pronounIntegrity,
      transitionIntegrity,
      modifierIntegrity
    };
  }

  computeGroundingIntegrity(tokens, contextModel) {
    const content =
      (tokens || []).filter(
        token =>
          this.isContentType(
            this.getCanonicalType(token)
          )
      );

    if (!content.length) return 0.0;

    const meaningfulAnchors =
      contextModel?.instructionContract?.sourceAnchorTokens || [];

    let supported = 0.0;
    let counted = 0;
    let previousContent = null;

    for (const token of content) {
      if (this.isSourceAnchorFamily(token, contextModel)) {
        previousContent = token;
        continue;
      }

      counted++;

      const anchor =
        this.getSemanticAnchorSupport(token, contextModel);

      const role =
        this.getRoleSensitiveKnowledgeEvidence(
          token,
          null,
          contextModel
        );

      let pairSupport = 0.0;

      if (Number.isInteger(previousContent)) {
        const pair =
          this.getVocabKnowledgeLink(previousContent, token);
        pairSupport =
          Math.max(0, Math.min(1, pair?.score || 0.0));
      }

      const best = Math.max(
        anchor?.semanticSupport || 0,
        anchor?.expansionSupport || 0,
        role?.overall || 0,
        pairSupport
      );

      supported += Math.max(0, Math.min(1, best));
      previousContent = token;
    }

    if (!counted) {
      return meaningfulAnchors.length ? 0.82 : 0.62;
    }

    return Math.max(0, Math.min(1, supported / counted));
  }

  computeQuantitativeIntegrity(tokens, contextModel) {
    if (!contextModel?.answerPlan?.requiresNumericEvidence) {
      return 1.0;
    }

    const vocab = this.network?.vocab;
    let numericCount = 0;
    let quantityCount = 0;
    let promptNumberReuse = 0;

    const promptNumbers =
      new Set(contextModel?.answerPlan?.numericPromptTokens || []);

    for (const token of tokens || []) {
      const type = this.getCanonicalType(token);
      const word = String(this.tokenWord(token) || "");

      if (type === "Num" || /^\d+(?:\.\d+)?$/.test(word)) {
        numericCount++;
        if (promptNumbers.has(token)) promptNumberReuse++;
      }

      const semanticClass = vocab?.getSemanticClass?.(token);
      const quantityFit = vocab?.getRoleCompatibility?.(token, "quantity") ?? 0;

      if (semanticClass === "quantity" || quantityFit >= 0.72) {
        quantityCount++;
      }
    }

    const numericEvidence = numericCount > 0 ? 1.0 : 0.12;
    const quantityEvidence = quantityCount > 0 ? 1.0 : 0.28;
    const promptReuseEvidence =
      promptNumbers.size
        ? (promptNumberReuse > 0 ? 0.78 : 0.58)
        : 0.72;

    return Math.max(
      0,
      Math.min(
        1,
        numericEvidence * 0.52 +
        quantityEvidence * 0.34 +
        promptReuseEvidence * 0.14
      )
    );
  }

  computeTailIntegrity(
    tokens,
    contextModel
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const sequence =
      (tokens || []).filter(
        token =>
          this.getCanonicalType(token) !==
          "Punct"
      );

    if (!sequence.length) {
      return {
        score: 0.0,
        grounding: 0.0,
        relevance: 0.0,
        queryFit: 0.0,
        resolutionRetention: 0.0,
        contradictionIntegrity: 1.0,
        tailLength: 0
      };
    }

    const tailLength =
      Math.max(
        2,
        Math.min(
          8,
          Math.ceil(
            sequence.length *
            this.reasoningConfig
              .tailFraction
          )
        )
      );

    const split =
      Math.max(
        0,
        sequence.length -
        tailLength
      );

    const prefix =
      sequence.slice(0, split);

    const tail =
      sequence.slice(split);

    let groundingSum = 0.0;
    let relevanceSum = 0.0;
    let queryFitSum = 0.0;
    let contentCount = 0;
    let functionCount = 0;

    for (const token of tail) {
      const type =
        this.getCanonicalType(token);

      if (this.isContentType(type)) {
        const provenance =
          this.getPromptRootedProvenance(
            token,
            contextModel
          );

        groundingSum +=
          clamp(provenance?.grounding);

        relevanceSum +=
          clamp(
            Math.max(
              provenance?.relevance || 0,
              provenance?.grounding || 0
            )
          );

        queryFitSum +=
          clamp(
            this.getEventQueryCandidateFit(
              token,
              contextModel,
              null
            )
          );

        contentCount++;
      } else {
        functionCount++;
      }
    }

    const grounding =
      contentCount
        ? groundingSum /
          contentCount
        : 0.45;

    const relevance =
      contentCount
        ? relevanceSum /
          contentCount
        : 0.45;

    const queryFit =
      contentCount
        ? queryFitSum /
          contentCount
        : 0.45;

    const fullResolution =
      this.computeEventQueryResolution(
        sequence,
        contextModel
      );

    const prefixResolution =
      prefix.length >= 3
        ? this.computeEventQueryResolution(
            prefix,
            contextModel
          )
        : fullResolution;

    const resolutionRetention =
      prefixResolution.score > 0.20
        ? clamp(
            (fullResolution.score + 0.10) /
            (prefixResolution.score + 0.10)
          )
        : 0.80;

    const balance =
      clamp(
        1.0 -
        Math.max(
          0,
          functionCount -
          contentCount * 1.7
        ) /
        Math.max(
          2,
          tail.length
        )
      );

    // Very short answers do not have a meaningful "tail" distinct from the
    // whole answer, so avoid manufacturing a penalty from a 2-token suffix.
    const shortAnswerNeutral =
      sequence.length <= 6
        ? 0.10
        : 0.0;

    const score =
      clamp(
        grounding * 0.26 +
        relevance * 0.20 +
        queryFit * 0.20 +
        fullResolution.contradictionIntegrity * 0.12 +
        resolutionRetention * 0.16 +
        balance * 0.06 +
        shortAnswerNeutral
      );

    return {
      score,
      grounding: clamp(grounding),
      relevance: clamp(relevance),
      queryFit: clamp(queryFit),
      resolutionRetention,
      contradictionIntegrity:
        fullResolution
          .contradictionIntegrity,
      tailLength,
      contentCount,
      functionCount
    };
  }

  getAnswerCommitmentState(answerGraph, contextModel, matches = null) {
    const clamp=value=>Math.max(0,Math.min(1,Number.isFinite(value)?value:0));
    const contract=contextModel?.answerContract;
    if (!contract?.requiresCommitment) return {required:false,commitment:"none",explicitness:1,propositionPreservation:1,strength:1,resolved:true};
    const subjectMatch=clamp(matches?.subjectMatch), predicateMatch=clamp(matches?.predicateMatch), targetMatch=clamp(matches?.targetMatch);
    const propositionPreservation=clamp(subjectMatch*0.34+predicateMatch*0.36+targetMatch*0.30);
    const main=answerGraph?.mainClause||{completeness:0,hasPredicate:false};
    const explicitAffirm=(answerGraph?.affirmative||0)>0;
    const explicitDeny=(answerGraph?.negative||0)>0;
    const explicitUncertain=(answerGraph?.uncertain||0)>0;
    const conditional=(answerGraph?.conditionConnectors||0)>0 && propositionPreservation>=0.42;
    const implicitAffirm=!explicitDeny&&!explicitUncertain&&!conditional&&main.hasPredicate&&main.completeness>=this.reasoningConfig.answerCommitmentLanguageFloor&&subjectMatch>=0.50&&predicateMatch>=this.reasoningConfig.answerCommitmentPredicateFloor&&(!contract?.targetTokens?.length||targetMatch>=this.reasoningConfig.answerCommitmentTargetFloor)&&propositionPreservation>=this.reasoningConfig.answerCommitmentPropositionFloor;
    let commitment="none", explicitness=0;
    if (explicitUncertain) {commitment="uncertain";explicitness=0.92;}
    else if (explicitDeny) {commitment="deny";explicitness=1;}
    else if (conditional) {commitment="conditional";explicitness=0.76;}
    else if (explicitAffirm) {commitment="affirm";explicitness=1;}
    else if (implicitAffirm) {commitment="affirm";explicitness=0.70;}
    const strength=commitment==="none"?clamp(propositionPreservation*0.22):clamp(explicitness*0.48+propositionPreservation*0.40+main.completeness*0.12);
    return {required:true,commitment,explicitness,propositionPreservation,strength,resolved:commitment!=="none"&&strength>=this.reasoningConfig.answerCommitmentResolveFloor};
  }

  getLiveAnswerCommitment(tokens, contextModel) {
    if (!contextModel?.answerContract?.requiresCommitment) return {required:false,commitment:"none",explicitness:1,propositionPreservation:1,strength:1,resolved:true};
    const graph=this.parseIndependentAnswerGraph(tokens||[]), contract=contextModel.answerContract;
    const main=graph.mainClause||{subjects:[],predicates:[],complements:[],hasPredicate:false};
    const overlap=(a,b)=>{if(!a?.length||!b?.length)return 0;const set=new Set(a);let hit=0;for(const token of b)if(set.has(token))hit++;return Math.max(0,Math.min(1,hit/Math.max(1,Math.min(a.length,b.length))));};
    const subjectMatch=overlap(contract.subjectTokens||[],main.subjects);
    const predicateMatch=Number.isInteger(contract.relationVerbToken)?(main.predicates.includes(contract.relationVerbToken)?1:0):(main.hasPredicate?0.64:0);
    const targetMatch=contract.targetTokens?.length?overlap(contract.targetTokens,main.complements):1;
    return this.getAnswerCommitmentState(graph,contextModel,{subjectMatch,predicateMatch,targetMatch});
  }

  answerCommitmentBonus(tokenId, generated, contextModel) {
    const contract=contextModel?.answerContract;
    if (!this.reasoningConfig.answerCommitmentEnabled||!contract?.requiresCommitment)return 0;
    const current=this.getLiveAnswerCommitment(generated||[],contextModel);
    if(current.commitment!=="none")return 0;
    const word=String(this.tokenWord(tokenId)||"").toLowerCase(), type=this.getCanonicalType(tokenId);
    let bonus=0;
    if(["yes","no","not","true","false","maybe","perhaps","uncertain","unknown","depends"].includes(word))bonus+=1.20;
    if(contract.subjectTokens?.includes(tokenId))bonus+=0.74;
    if(Number.isInteger(contract.relationVerbToken)&&tokenId===contract.relationVerbToken)bonus+=1.18;
    if(contract.targetTokens?.includes(tokenId))bonus+=0.86;
    if(["Aux","Modal"].includes(type))bonus+=0.26;
    return Math.max(-1,Math.min(2.4,bonus*this.reasoningConfig.answerCommitmentCandidateWeight));
  }

  getIndependentRequiredAnswerAct(
    contextModel
  ) {
    const kind =
      contextModel
        ?.answerContract
        ?.kind ||
      "general";

    const map = {
      temporal: "time_condition",
      causal: "cause",
      effect: "effect",
      mechanism: "mechanism",
      locative: "location",
      quantitative: "quantity",
      boolean: "truth",
      definition: "definition",
      evaluative: "evaluation",
      factual_generation: "generation",
      imperative: "execution",
      clarification: "clarification",
      general:
        contextModel
          ?.questionProfile
          ?.isQuestion
          ? "answer"
          : "conversation"
    };

    return map[kind] || "answer";
  }

  parseIndependentAnswerGraph(
    tokens
  ) {
    const phraseProfile = this.parseTypePhrases(tokens || []);
    const clauses = [];
    const allTokens = [];

    const negationWords =
      new Set([
        "not", "no", "never",
        "neither", "nor",
        "cannot", "can't"
      ]);

    const affirmativeWords =
      new Set([
        "yes", "true", "correct",
        "indeed", "certainly"
      ]);

    const uncertaintyWords =
      new Set([
        "maybe", "perhaps",
        "possibly", "uncertain",
        "unsure", "unknown"
      ]);

    const mechanismWords =
      new Set([
        "by", "through", "via",
        "using", "with"
      ]);

    const causeWords =
      new Set([
        "because", "since", "due",
        "from"
      ]);

    const resultWords =
      new Set([
        "therefore", "thus", "hence",
        "so", "resulting", "causing"
      ]);

    const locationWords =
      new Set([
        "in", "on", "at", "inside",
        "within", "near", "under",
        "above", "between", "around"
      ]);

    const conditionWords =
      new Set([
        "if", "when", "unless", "once",
        "before", "after", "until", "while"
      ]);

    const definitionWords =
      new Set([
        "means", "mean", "refers",
        "called", "defined"
      ]);

    const closeClause = clause => {
      if (!clause.tokens.length) {
        return;
      }

      let predicateIndex = -1;
      let lexicalPredicateIndex = -1;

      for (
        let i = 0;
        i < clause.tokens.length;
        i++
      ) {
        const type =
          clause.tokens[i].type;

        if (
          lexicalPredicateIndex < 0 &&
          type === "Verb"
        ) {
          lexicalPredicateIndex = i;
        }

        if (
          predicateIndex < 0 &&
          ["Verb", "Aux", "Modal"]
            .includes(type)
        ) {
          predicateIndex = i;
        }
      }

      if (
        lexicalPredicateIndex >= 0
      ) {
        predicateIndex =
          lexicalPredicateIndex;
      }

      const subjects = [];
      const predicates = [];
      const auxiliaries = [];
      const complements = [];
      const quantities = [];
      const connectors = [];
      const content = [];

      for (
        let i = 0;
        i < clause.tokens.length;
        i++
      ) {
        const item = clause.tokens[i];
        const type = item.type;

        if (
          this.isContentType(type)
        ) {
          content.push(item.token);
        }

        if (
          ["Aux", "Modal"].includes(type)
        ) {
          auxiliaries.push(item.token);
        }

        if (type === "Verb") {
          predicates.push(item.token);
        }

        if (type === "Num") {
          quantities.push(item.token);
        }

        if (
          ["Prep", "Conj"].includes(type)
        ) {
          connectors.push(item.token);
        }

        if (
          ["Noun", "ProperNoun", "Pronoun"]
            .includes(type) &&
          (
            predicateIndex < 0 ||
            i < predicateIndex
          )
        ) {
          subjects.push(item.token);
        }

        if (
          predicateIndex >= 0 &&
          i > predicateIndex &&
          (
            this.isContentType(type) ||
            ["Pronoun"].includes(type)
          )
        ) {
          complements.push(item.token);
        }
      }

      if (
        !subjects.length
      ) {
        const nominal =
          clause.tokens.find(
            item =>
              [
                "Noun",
                "ProperNoun",
                "Pronoun"
              ].includes(item.type)
          );

        if (nominal) {
          subjects.push(
            nominal.token
          );
        }
      }

      const finalType =
        clause.tokens[
          clause.tokens.length - 1
        ]?.type;

      const hangingDependency =
        [
          "Prep", "Det", "Conj",
          "Modal"
        ].includes(finalType);

      const hasPredicate =
        predicates.length > 0 ||
        auxiliaries.length > 0;

      const hasNominal =
        subjects.length > 0 ||
        complements.some(
          token =>
            [
              "Noun",
              "ProperNoun",
              "Pronoun"
            ].includes(
              this.getCanonicalType(token)
            )
        );

      const hasComplement =
        complements.length > 0;

      let legalTransitions = 0;
      let transitions = 0;

      let previousType = null;

      for (const item of clause.tokens) {
        if (previousType) {
          transitions++;

          const transition =
            this.getTransitionRule(
              previousType,
              item.type
            );

          if (transition.legal) {
            legalTransitions++;
          }
        }

        previousType = item.type;
      }

      const transitionIntegrity =
        transitions
          ? legalTransitions /
            transitions
          : 0.76;

      const clauseCompleteness =
        Math.max(
          0,
          Math.min(
            1,
            (hasPredicate ? 0.36 : 0.0) +
            (hasNominal ? 0.23 : 0.0) +
            (hasComplement ? 0.18 : 0.05) +
            transitionIntegrity * 0.15 +
            (hangingDependency ? 0.0 : 0.08)
          )
        );

      clauses.push({
        tokens:
          clause.tokens.map(
            item => item.token
          ),
        subjects,
        predicates,
        auxiliaries,
        complements,
        quantities,
        connectors,
        content,
        predicateIndex,
        hasPredicate,
        hasNominal,
        hasComplement,
        hangingDependency,
        transitionIntegrity,
        completeness:
          clauseCompleteness
      });
    };

    let current = {
      tokens: []
    };

    let affirmative = 0;
    let negative = 0;
    let uncertain = 0;
    let mechanismConnectors = 0;
    let causeConnectors = 0;
    let resultConnectors = 0;
    let locationConnectors = 0;
    let conditionConnectors = 0;
    let definitionMarkers = 0;
    let temporalEvidence = 0.0;
    let numericCount = 0;
    let questionEnding = false;

    for (
      let index = 0;
      index < (tokens || []).length;
      index++
    ) {
      const token = tokens[index];
      const type =
        this.getCanonicalType(token);
      const word =
        String(
          this.tokenWord(token) || ""
        ).toLowerCase();

      allTokens.push(token);

      if (type === "Punct") {
        if (word === "?") {
          questionEnding = true;
        }

        closeClause(current);
        current = { tokens: [] };
        continue;
      }

      const item = {
        token,
        type,
        word,
        index
      };

      current.tokens.push(item);

      if (
        negationWords.has(word)
      ) {
        negative++;
      }

      if (
        affirmativeWords.has(word)
      ) {
        affirmative++;
      }

      if (
        uncertaintyWords.has(word)
      ) {
        uncertain++;
      }

      if (
        mechanismWords.has(word)
      ) {
        mechanismConnectors++;
      }

      if (causeWords.has(word)) {
        causeConnectors++;
      }

      if (resultWords.has(word)) {
        resultConnectors++;
      }

      if (locationWords.has(word)) {
        locationConnectors++;
      }

      if (conditionWords.has(word)) {
        conditionConnectors++;
      }

      if (definitionWords.has(word)) {
        definitionMarkers++;
      }

      if (type === "Num") {
        numericCount++;
      }

      const temporal =
        this.network
          ?.vocab
          ?.getTemporalProfile?.(
            token
          );

      if (
        Number.isFinite(
          temporal?.score
        )
      ) {
        temporalEvidence =
          Math.max(
            temporalEvidence,
            temporal.score
          );
      }
    }

    closeClause(current);

    const mainClause =
      clauses
        .slice()
        .sort(
          (a, b) =>
            (
              b.completeness +
              b.content.length * 0.025
            ) -
            (
              a.completeness +
              a.content.length * 0.025
            )
        )[0] || null;

    const allContent =
      clauses.flatMap(
        clause => clause.content
      );

    const allPredicates =
      clauses.flatMap(
        clause => clause.predicates
      );

    const allSubjects =
      clauses.flatMap(
        clause => clause.subjects
      );

    const allComplements =
      clauses.flatMap(
        clause => clause.complements
      );

    const averageClauseCompleteness =
      clauses.length
        ? clauses.reduce(
            (sum, clause) =>
              sum + clause.completeness,
            0
          ) / clauses.length
        : 0.0;

    return {
      version: 1,
      clauses,
      mainClause,
      allTokens,
      content: allContent,
      subjects: allSubjects,
      predicates: allPredicates,
      complements: allComplements,
      affirmative,
      negative,
      uncertain,
      mechanismConnectors,
      causeConnectors,
      resultConnectors,
      locationConnectors,
      conditionConnectors,
      definitionMarkers,
      temporalEvidence,
      numericCount,
      questionEnding,
      phraseProfile,
      averageClauseCompleteness,
      hasPredicate:
        clauses.some(
          clause => clause.hasPredicate
        ),
      hasNominal:
        clauses.some(
          clause => clause.hasNominal
        ),
      hasComplement:
        clauses.some(
          clause => clause.hasComplement
        )
    };
  }

  getIndependentPromptTokenMatch(
    candidateToken,
    promptTokens
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    if (
      !Number.isInteger(candidateToken) ||
      !(promptTokens?.length)
    ) {
      return 0.0;
    }

    const candidate =
      this.canonicalTokenId(
        candidateToken
      );

    const candidateWord =
      String(
        this.tokenWord(candidate) || ""
      ).toLowerCase();

    const candidateStem =
      this.stemWord(
        candidateWord
      );

    const candidateType =
      this.getCanonicalType(
        candidate
      );

    let best = 0.0;

    for (const rawRoot of promptTokens) {
      const root =
        this.canonicalTokenId(
          rawRoot
        );

      if (!Number.isInteger(root)) {
        continue;
      }

      if (candidate === root) {
        return 1.0;
      }

      const rootWord =
        String(
          this.tokenWord(root) || ""
        ).toLowerCase();

      const rootStem =
        this.stemWord(rootWord);

      if (
        candidateStem.length >= 3 &&
        rootStem.length >= 3 &&
        candidateStem === rootStem
      ) {
        best =
          Math.max(
            best,
            0.96
          );
      }

      const rootType =
        this.getCanonicalType(root);

      if (
        candidateType === "Pronoun" &&
        [
          "Noun", "ProperNoun",
          "Pronoun"
        ].includes(rootType)
      ) {
        best =
          Math.max(
            best,
            0.70
          );
      }

      const pair =
        this.getWordAssociationEvidence(
          candidate,
          root,
          null
        );

      const trusted =
        Math.max(
          pair?.typedKnowledge || 0,
          (pair?.staticKnowledge || 0) * 0.88,
          (pair?.collocationKnowledge || 0) * 0.72
        );

      best =
        Math.max(
          best,
          clamp(trusted)
        );
    }

    return clamp(best);
  }

  getIndependentPromptSetMatch(
    answerTokens,
    promptTokens
  ) {
    if (!(promptTokens?.length)) {
      return 0.74;
    }

    if (!(answerTokens?.length)) {
      return 0.0;
    }

    let best = 0.0;

    for (const token of answerTokens) {
      best =
        Math.max(
          best,
          this.getIndependentPromptTokenMatch(
            token,
            promptTokens
          )
        );
    }

    return best;
  }

  computeIndependentGrounding(
    answerGraph,
    contextModel
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const roots = [];

    const addRoot = token => {
      const canonical =
        this.canonicalTokenId(token);

      if (
        Number.isInteger(canonical) &&
        !roots.includes(canonical)
      ) {
        roots.push(canonical);
      }
    };

    for (
      const token of
        contextModel
          ?.answerContract
          ?.subjectTokens || []
    ) {
      addRoot(token);
    }

    addRoot(
      contextModel
        ?.answerContract
        ?.relationVerbToken
    );

    for (
      const token of
        contextModel
          ?.answerContract
          ?.targetTokens || []
    ) {
      addRoot(token);
    }

    for (
      const token of
        contextModel
          ?.instructionContract
          ?.sourceAnchorTokens || []
    ) {
      if (roots.length >= 12) break;
      addRoot(token);
    }

    if (!answerGraph.content.length) {
      return {
        grounding: 0.0,
        epistemic: 0.0,
        rootedRatio: 0.0
      };
    }

    const supportByToken =
      new Map();

    let groundingSum = 0.0;
    let epistemicSum = 0.0;
    let rootedCount = 0;

    let previous = null;

    for (const token of answerGraph.content) {
      const direct =
        this.getIndependentPromptTokenMatch(
          token,
          roots
        );

      let inherited = 0.0;
      let edgeKnowledge = 0.0;

      if (
        Number.isInteger(previous)
      ) {
        const previousSupport =
          supportByToken.get(previous) ||
          0.0;

        const pair =
          this.getWordAssociationEvidence(
            previous,
            token,
            null
          );

        edgeKnowledge =
          clamp(
            Math.max(
              pair?.typedKnowledge || 0,
              (pair?.staticKnowledge || 0) * 0.88,
              (pair?.collocationKnowledge || 0) * 0.74
            )
          );

        if (
          previousSupport >= 0.48 &&
          edgeKnowledge >= 0.42
        ) {
          inherited =
            previousSupport *
            edgeKnowledge *
            0.64;
        }
      }

      const support =
        clamp(
          Math.max(
            direct,
            inherited
          )
        );

      supportByToken.set(
        token,
        support
      );

      groundingSum += support;

      epistemicSum +=
        clamp(
          Math.max(
            direct,
            edgeKnowledge * 0.78,
            support * 0.72
          )
        );

      if (support >= 0.50) {
        rootedCount++;
      }

      previous = token;
    }

    return {
      grounding:
        clamp(
          groundingSum /
          answerGraph.content.length
        ),
      epistemic:
        clamp(
          epistemicSum /
          answerGraph.content.length
        ),
      rootedRatio:
        clamp(
          rootedCount /
          answerGraph.content.length
        )
    };
  }

  verifyIndependentAnswerAct(
    answerGraph,
    contextModel
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const requiredAct =
      this.getIndependentRequiredAnswerAct(
        contextModel
      );

    const contract =
      contextModel?.answerContract || {};

    const subjectRoots =
      contract.subjectTokens || [];

    const predicateRoots =
      Number.isInteger(
        contract.relationVerbToken
      )
        ? [contract.relationVerbToken]
        : [];

    const targetRoots =
      contract.targetTokens || [];

    const main =
      answerGraph.mainClause || {
        subjects: [],
        predicates: [],
        auxiliaries: [],
        complements: [],
        content: [],
        completeness: 0.0,
        hasPredicate: false,
        hasComplement: false
      };

    const subjectMatch =
      this.getIndependentPromptSetMatch(
        main.subjects,
        subjectRoots
      );

    let predicateMatch =
      this.getIndependentPromptSetMatch(
        main.predicates,
        predicateRoots
      );

    let targetMatch =
      this.getIndependentPromptSetMatch(
        main.complements,
        targetRoots
      );

    const hasAuxiliaryEllipsis =
      requiredAct === "truth" &&
      main.auxiliaries?.length > 0 &&
      main.predicates.length === 0 &&
      subjectMatch >= 0.60;

    if (hasAuxiliaryEllipsis) {
      predicateMatch =
        Math.max(
          predicateMatch,
          0.80
        );

      if (targetRoots.length) {
        targetMatch =
          Math.max(
            targetMatch,
            0.72
          );
      }
    }

    const grounding =
      this.computeIndependentGrounding(
        answerGraph,
        contextModel
      );

    let novelPredicate = 0.0;
    let predicateBridge = 0.0;

    for (const token of answerGraph.predicates) {
      const promptMatch =
        this.getIndependentPromptTokenMatch(
          token,
          predicateRoots
        );

      if (promptMatch < 0.62) {
        novelPredicate =
          Math.max(
            novelPredicate,
            1.0 - promptMatch
          );
      }

      for (const root of predicateRoots) {
        const pair =
          this.getWordAssociationEvidence(
            token,
            root,
            null
          );

        predicateBridge =
          Math.max(
            predicateBridge,
            clamp(
              Math.max(
                pair?.typedKnowledge || 0,
                (pair?.staticKnowledge || 0) * 0.88,
                (pair?.collocationKnowledge || 0) * 0.72
              )
            )
          );
      }
    }

    const complementCompleteness =
      main.complements.length
        ? Math.min(
            1,
            0.58 +
            Math.min(
              0.42,
              main.complements.length *
              0.14
            )
          )
        : 0.0;

    const propositionCompleteness =
      clamp(
        main.completeness * 0.54 +
        (main.hasPredicate ? 0.20 : 0.0) +
        (
          main.hasComplement ||
          requiredAct === "truth"
            ? 0.16
            : 0.0
        ) +
        (
          main.hangingDependency
            ? 0.0
            : 0.10
        )
      );

    const promptPropositionMatch =
      clamp(
        subjectMatch * 0.34 +
        predicateMatch * 0.36 +
        targetMatch * 0.30
      );

    const commitment = this.getAnswerCommitmentState(answerGraph, contextModel, {subjectMatch, predicateMatch, targetMatch});
    const stanceEvidence = commitment.required ? commitment.strength : (answerGraph.uncertain > 0 ? 0.88 : ((answerGraph.affirmative > 0 || answerGraph.negative > 0) ? 1.0 : 0.50));

    const mechanismStructure =
      clamp(
        Math.max(
          answerGraph.mechanismConnectors > 0 &&
          main.complements.length > 0
            ? 0.82
            : 0.0,
          novelPredicate *
          complementCompleteness
        )
      );

    const mechanismBridge =
      clamp(
        predicateBridge * 0.54 +
        grounding.grounding * 0.24 +
        subjectMatch * 0.22
      );

    const causeStructure =
      clamp(
        (
          answerGraph.causeConnectors > 0
            ? 0.66
            : 0.0
        ) +
        Math.min(
          0.34,
          Math.max(
            0,
            answerGraph.clauses.length - 1
          ) * 0.17
        )
      );

    const effectStructure =
      clamp(
        (
          answerGraph.resultConnectors > 0
            ? 0.62
            : 0.0
        ) +
        (
          answerGraph.clauses.length > 1
            ? 0.26
            : 0.0
        ) +
        novelPredicate * 0.22
      );

    const locationStructure =
      clamp(
        answerGraph.locationConnectors > 0 &&
        main.complements.length > 0
          ? 0.90
          : 0.12
      );

    const quantityStructure =
      clamp(
        answerGraph.numericCount > 0
          ? (
              main.complements.length > 0
                ? 1.0
                : 0.82
            )
          : 0.08
      );

    const temporalStructure =
      clamp(
        answerGraph.temporalEvidence * 0.78 +
        (
          answerGraph.conditionConnectors > 0
            ? 0.22
            : 0.0
        )
      );

    const definitionStructure =
      clamp(
        (
          answerGraph.definitionMarkers > 0
            ? 0.72
            : 0.0
        ) +
        (
          main.auxiliaries.length > 0 &&
          main.complements.length > 0
            ? 0.50
            : 0.0
        )
      );

    let actFulfillment = 0.40;
    let promptRelationMatch =
      promptPropositionMatch;
    let observedAct = "statement";

    switch (requiredAct) {
      case "truth":
        observedAct = commitment.commitment;
        promptRelationMatch = clamp(commitment.propositionPreservation * 0.58 + promptPropositionMatch * 0.42);
        actFulfillment = commitment.commitment === "none" ? clamp(commitment.strength * 0.28) : clamp(commitment.strength * 0.66 + promptRelationMatch * 0.34);
        break;

      case "mechanism":
        observedAct = "explain_process";
        promptRelationMatch =
          clamp(
            subjectMatch * 0.28 +
            mechanismBridge * 0.46 +
            predicateMatch * 0.12 +
            grounding.grounding * 0.14
          );

        actFulfillment =
          clamp(
            mechanismStructure * 0.52 +
            mechanismBridge * 0.32 +
            subjectMatch * 0.16
          );
        break;

      case "cause":
        observedAct = "explain_cause";
        promptRelationMatch =
          clamp(
            promptPropositionMatch * 0.50 +
            grounding.grounding * 0.24 +
            causeStructure * 0.26
          );
        actFulfillment =
          clamp(
            causeStructure * 0.62 +
            promptRelationMatch * 0.38
          );
        break;

      case "effect":
        observedAct = "explain_effect";
        promptRelationMatch =
          clamp(
            subjectMatch * 0.24 +
            predicateMatch * 0.22 +
            grounding.grounding * 0.22 +
            effectStructure * 0.32
          );
        actFulfillment =
          clamp(
            effectStructure * 0.60 +
            promptRelationMatch * 0.40
          );
        break;

      case "time_condition":
        observedAct = "locate_time_condition";
        promptRelationMatch =
          clamp(
            promptPropositionMatch * 0.46 +
            temporalStructure * 0.36 +
            grounding.grounding * 0.18
          );
        actFulfillment =
          clamp(
            temporalStructure * 0.68 +
            promptRelationMatch * 0.32
          );
        break;

      case "location":
        observedAct = "locate";
        promptRelationMatch =
          clamp(
            promptPropositionMatch * 0.48 +
            locationStructure * 0.34 +
            grounding.grounding * 0.18
          );
        actFulfillment =
          clamp(
            locationStructure * 0.68 +
            promptRelationMatch * 0.32
          );
        break;

      case "quantity":
        observedAct = "quantify";
        promptRelationMatch =
          clamp(
            promptPropositionMatch * 0.44 +
            quantityStructure * 0.36 +
            grounding.grounding * 0.20
          );
        actFulfillment =
          clamp(
            quantityStructure * 0.70 +
            promptRelationMatch * 0.30
          );
        break;

      case "definition":
        observedAct = "define";
        promptRelationMatch =
          clamp(
            subjectMatch * 0.42 +
            definitionStructure * 0.34 +
            grounding.grounding * 0.24
          );
        actFulfillment =
          clamp(
            definitionStructure * 0.58 +
            promptRelationMatch * 0.42
          );
        break;

      case "evaluation":
        observedAct = "evaluate";
        actFulfillment =
          clamp(
            propositionCompleteness * 0.36 +
            promptPropositionMatch * 0.34 +
            grounding.grounding * 0.30
          );
        break;

      case "clarification":
        observedAct = "clarify";
        actFulfillment =
          clamp(
            (
              answerGraph.questionEnding
                ? 0.72
                : 0.24
            ) +
            Math.min(
              0.28,
              answerGraph.content.length *
              0.07
            )
          );
        promptRelationMatch =
          Math.max(
            promptRelationMatch,
            0.62
          );
        break;

      case "generation":
      case "execution":
        observedAct = requiredAct;
        actFulfillment =
          clamp(
            answerGraph.content.length > 0
              ? 0.70 +
                Math.min(
                  0.30,
                  answerGraph.content.length *
                  0.03
                )
              : 0.0
          );
        promptRelationMatch =
          Math.max(
            promptRelationMatch,
            grounding.grounding
          );
        break;

      default:
        observedAct = "answer";
        actFulfillment =
          clamp(
            propositionCompleteness * 0.48 +
            promptRelationMatch * 0.34 +
            grounding.grounding * 0.18
          );
        break;
    }

    const polarityConflict =
      answerGraph.affirmative > 0 &&
      answerGraph.negative > 0;

    const uncertaintyConflict =
      answerGraph.uncertain > 0 &&
      (
        answerGraph.affirmative > 0 ||
        answerGraph.negative > 0
      );

    const contradictionIntegrity =
      clamp(
        1.0 -
        (polarityConflict ? 0.56 : 0.0) -
        (uncertaintyConflict ? 0.28 : 0.0)
      );

    const languageIntegrity =
      clamp(
        answerGraph.averageClauseCompleteness *
          0.72 +
        (
          answerGraph.hasPredicate
            ? 0.18
            : requiredAct === "clarification"
              ? 0.12
              : 0.0
        ) +
        (
          answerGraph.clauses.some(
            clause =>
              clause.hangingDependency
          )
            ? 0.0
            : 0.10
        )
      );

    const factualityRequired =
      Boolean(
        contextModel
          ?.answerPlan
          ?.factualityRequired
      );

    const score =
      clamp(
        actFulfillment * 0.24 +
        propositionCompleteness * 0.20 +
        promptRelationMatch * 0.20 +
        grounding.grounding * 0.12 +
        languageIntegrity * 0.10 +
        grounding.epistemic * 0.08 +
        contradictionIntegrity * 0.06
      );

    const criticalFloor =
      Math.min(
        actFulfillment,
        propositionCompleteness,
        promptRelationMatch,
        languageIntegrity,
        contradictionIntegrity,
        factualityRequired
          ? grounding.grounding
          : 1.0
      );

    const commitmentResolved = !contextModel?.answerContract?.requiresCommitment || commitment.resolved;
    const resolved =
      commitmentResolved &&
      actFulfillment >=
        this.reasoningConfig
          .independentActResolveFloor &&
      propositionCompleteness >=
        this.reasoningConfig
          .independentPropositionResolveFloor &&
      promptRelationMatch >=
        this.reasoningConfig
          .independentPromptRelationResolveFloor &&
      languageIntegrity >=
        this.reasoningConfig
          .independentLanguageResolveFloor &&
      (
        !factualityRequired ||
        grounding.grounding >=
          this.reasoningConfig
            .independentGroundingResolveFloor
      );

    return {
      version: 1,
      requiredAct,
      observedAct,
      actFulfillment,
      propositionCompleteness,
      promptRelationMatch,
      subjectMatch,
      predicateMatch,
      targetMatch,
      grounding:
        grounding.grounding,
      epistemic:
        grounding.epistemic,
      rootedRatio:
        grounding.rootedRatio,
      languageIntegrity,
      contradictionIntegrity,
      mechanismStructure,
      mechanismBridge,
      causeStructure,
      effectStructure,
      temporalStructure,
      locationStructure,
      quantityStructure,
      definitionStructure,
      commitment: commitment.commitment,
      commitmentStrength: commitment.strength,
      commitmentExplicitness: commitment.explicitness,
      propositionPreservation: commitment.propositionPreservation,
      commitmentResolved: commitment.resolved,
      score,
      criticalFloor,
      resolved
    };
  }

  computeIndependentAnswerVerification(
    tokens,
    contextModel
  ) {
    const answerGraph =
      this.parseIndependentAnswerGraph(
        tokens
      );

    const verifier =
      this.verifyIndependentAnswerAct(
        answerGraph,
        contextModel
      );

    return {
      ...verifier,
      graph: answerGraph
    };
  }

  computeIndependentConfidence(
    verifier,
    neuralSupport = 0.0,
    factualityRequired = false
  ) {
    const clamp = value =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(value)
            ? value
            : 0
        )
      );

    const weights =
      this.reasoningConfig
        .independentVerifierWeights;

    const act =
      clamp(verifier?.actFulfillment);
    const proposition =
      clamp(verifier?.propositionCompleteness);
    const relation =
      clamp(verifier?.promptRelationMatch);
    const grounding =
      clamp(verifier?.grounding);
    const language =
      clamp(verifier?.languageIntegrity);
    const epistemic =
      clamp(verifier?.epistemic);
    const contradiction =
      clamp(verifier?.contradictionIntegrity);
    const neural =
      clamp(neuralSupport);

    const base =
      clamp(
        act * weights.act +
        proposition * weights.proposition +
        relation * weights.promptRelation +
        grounding * weights.grounding +
        language * weights.language +
        epistemic * weights.epistemic +
        contradiction * weights.contradiction +
        neural * weights.neural
      );

    const critical =
      Math.min(
        act,
        proposition,
        relation,
        language,
        contradiction,
        factualityRequired
          ? grounding
          : 1.0
      );

    let score =
      clamp(
        base * 0.74 +
        Math.sqrt(
          Math.max(
            0,
            base * critical
          )
        ) * 0.26
      );

    let cap = 1.0;
    const caps =
      this.reasoningConfig
        .independentVerifierCaps;

    if (act < 0.26) {
      cap = Math.min(
        cap,
        caps.actMissing
      );
    } else if (act < 0.52) {
      cap = Math.min(
        cap,
        caps.actWeak
      );
    }

    if (proposition < 0.28) {
      cap = Math.min(
        cap,
        caps.propositionMissing
      );
    } else if (
      proposition <
        this.reasoningConfig
          .independentPropositionResolveFloor
    ) {
      cap = Math.min(
        cap,
        caps.propositionWeak
      );
    }

    if (relation < 0.28) {
      cap = Math.min(
        cap,
        caps.promptRelationMissing
      );
    } else if (
      relation <
        this.reasoningConfig
          .independentPromptRelationResolveFloor
    ) {
      cap = Math.min(
        cap,
        caps.promptRelationWeak
      );
    }

    if (
      factualityRequired &&
      grounding <
        this.reasoningConfig
          .independentGroundingResolveFloor
    ) {
      cap = Math.min(
        cap,
        caps.groundingWeak
      );
    }

    if (
      language <
        this.reasoningConfig
          .independentLanguageResolveFloor
    ) {
      cap = Math.min(
        cap,
        caps.languageWeak
      );
    }

    if (contradiction < 0.62) {
      cap = Math.min(
        cap,
        caps.contradictionWeak
      );
    }

    score = Math.min(
      score,
      cap
    );

    const highFloor =
      Math.min(
        act,
        proposition,
        relation,
        language,
        contradiction,
        factualityRequired
          ? grounding
          : 1.0
      );

    if (
      highFloor >=
        this.reasoningConfig
          .independentHighConfidenceFloor
    ) {
      const strength =
        clamp(
          (
            highFloor -
            this.reasoningConfig
              .independentHighConfidenceFloor
          ) /
          (
            1.0 -
            this.reasoningConfig
              .independentHighConfidenceFloor
          )
        );

      score =
        clamp(
          score +
          (1.0 - score) *
          0.52 *
          strength
        );
    }

    return {
      score,
      cap,
      base,
      critical,
      highFloor,
      neuralSupport: neural
    };
  }

  computeUniversalCritic(metrics, contextModel) {
    const clamp = value =>
      Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));

    const grounding = clamp(metrics?.groundingIntegrityScore);
    const quantitative = clamp(metrics?.quantitativeIntegrityScore);

    const independentAct =
      clamp(
        metrics
          ?.independentActFulfillmentScore
      );

    const independentProposition =
      clamp(
        metrics
          ?.independentPropositionCompletenessScore
      );

    const independentRelation =
      clamp(
        metrics
          ?.independentPromptRelationMatchScore
      );

    const independentGrounding =
      clamp(
        metrics
          ?.independentGroundingScore
      );
    const eventQuery =
      clamp(
        metrics
          ?.eventQueryResolutionScore
      );

    const querySlot =
      clamp(
        metrics
          ?.querySlotResolutionScore
      );

    const relationStructure =
      clamp(
        metrics
          ?.relationStructureIntegrityScore
      );

    const semanticFrameFit =
      clamp(
        metrics
          ?.semanticFrameFitScore
      );

    const semanticGraphQuery =
      clamp(
        metrics
          ?.semanticGraphQueryResolutionScore
      );

    const semanticRelation =
      clamp(
        metrics
          ?.semanticRelationCompletenessScore
      );

    const semanticGraphGrounding =
      clamp(
        metrics
          ?.semanticGraphGroundingScore
      );

    const semanticGraphIntegrity =
      clamp(
        metrics
          ?.semanticGraphIntegrityScore
      );

    const epistemic =
      clamp(
        metrics
          ?.epistemicCoverageScore
      );
    const contradiction = Number.isFinite(metrics?.contradictionIntegrityScore)
      ? clamp(metrics.contradictionIntegrityScore)
      : 1.0;
    const tailIntegrity = Number.isFinite(metrics?.tailIntegrityScore)
      ? clamp(metrics.tailIntegrityScore)
      : 1.0;

    const channels = [
      {
        value: independentAct,
        weight: 2.30
      },
      {
        value: independentProposition,
        weight: 2.00
      },
      {
        value: independentRelation,
        weight: 2.20
      },
      {
        value: independentGrounding,
        weight:
          contextModel
            ?.answerPlan
            ?.factualityRequired
            ? 1.55
            : 0.72
      },
      {
        value: semanticFrameFit,
        weight: 1.85
      },
      {
        value: semanticGraphQuery,
        weight: 2.15
      },
      {
        value: semanticRelation,
        weight: 2.10
      },
      {
        value: semanticGraphGrounding,
        weight:
          contextModel
            ?.answerPlan
            ?.factualityRequired
            ? 1.35
            : 0.82
      },
      {
        value: semanticGraphIntegrity,
        weight: 1.55
      },
      {
        value: querySlot,
        weight:
          contextModel
            ?.questionProfile
            ?.isQuestion
            ? 1.85
            : 0.70
      },
      {
        value: relationStructure,
        weight:
          contextModel
            ?.questionProfile
            ?.isQuestion
            ? 1.95
            : 0.70
      },
      {
        value: eventQuery,
        weight:
          contextModel
            ?.questionProfile
            ?.isQuestion
            ? 1.55
            : 0.85
      },
      {
        value: epistemic,
        weight:
          contextModel
            ?.answerPlan
            ?.factualityRequired
            ? 1.25
            : 0.65
      },
      { value: contradiction, weight: 1.10 },
      { value: tailIntegrity, weight: 1.05 },
      { value: clamp(metrics?.instructionAdherenceScore), weight: 1.00 },
      { value: clamp(metrics?.answerCompletionScore), weight: 0.95 },
      { value: clamp(metrics?.semanticAnswerIntegrity), weight: 0.80 },
      { value: clamp(metrics?.shapeDecodabilityScore), weight: 0.80 },
      { value: clamp(metrics?.globalCoherenceScore), weight: 0.70 },
      { value: clamp(metrics?.sequenceIntegrityScore), weight: 0.60 },
      { value: grounding, weight: contextModel?.answerPlan?.factualityRequired ? 0.95 : 0.45 },
      { value: quantitative, weight: contextModel?.answerPlan?.requiresNumericEvidence ? 1.20 : 0.20 }
    ];

    let logSum = 0.0;
    let weightSum = 0.0;
    for (const channel of channels) {
      const value = Math.max(0.04, channel.value);
      logSum += Math.log(value) * channel.weight;
      weightSum += channel.weight;
    }

    const geometric = Math.exp(logSum / Math.max(1e-6, weightSum));

    const critical = Math.min(
      contextModel
        ?.questionProfile
        ?.isQuestion
        ? independentAct
        : 1.0,
      contextModel
        ?.questionProfile
        ?.isQuestion
        ? independentProposition
        : 1.0,
      contextModel
        ?.questionProfile
        ?.isQuestion
        ? independentRelation
        : 1.0,
      contextModel
        ?.answerPlan
        ?.factualityRequired
        ? independentGrounding
        : 1.0,
      semanticFrameFit,
      contextModel
        ?.questionProfile
        ?.isQuestion
        ? semanticGraphQuery
        : 1.0,
      contextModel
        ?.questionProfile
        ?.isQuestion
        ? semanticRelation
        : 1.0,
      contextModel
        ?.answerPlan
        ?.factualityRequired
        ? semanticGraphGrounding
        : 1.0,
      contextModel
        ?.questionProfile
        ?.isQuestion
        ? querySlot
        : 1.0,
      contextModel
        ?.questionProfile
        ?.isQuestion
        ? relationStructure
        : 1.0,
      eventQuery,
      contradiction,
      tailIntegrity,
      contextModel?.answerPlan?.factualityRequired ? epistemic : 1.0,
      clamp(metrics?.instructionAdherenceScore),
      clamp(metrics?.answerCompletionScore),
      clamp(metrics?.shapeDecodabilityScore),
      contextModel?.answerPlan?.requiresNumericEvidence ? quantitative : 1.0
    );

    const score = clamp(geometric * 0.72 + critical * 0.28);

    return {
      score,
      geometric,
      critical,
      grounding,
      quantitative,
      independentAct,
      independentProposition,
      independentRelation,
      independentGrounding,
      semanticFrameFit,
      semanticGraphQuery,
      semanticRelation,
      semanticGraphGrounding,
      semanticGraphIntegrity,
      eventQuery,
      querySlot,
      relationStructure,
      epistemic,
      contradiction,
      tailIntegrity
    };
  }

  verifyCandidate(candidate, contextModel, targetLength) {
    const tokens = candidate.tokens || [];
    if (!tokens.length) {
      return {
        score: -1.0,
        semanticSim: 0.0,
        syntaxScore: 0.0,
        clauseOrderScore: 0.0,
        fluencyScore: 0.0,
        globalCoherenceScore: 0.0,
        sequenceIntegrityScore: 0.0,
        localSemanticContinuity: 0.0,
        semanticProgressionScore: 0.0,
        clauseIntegrityScore: 0.0,
        answerCompletionScore: 0.0,
        semanticAnswerIntegrity: 0.0,
        tailIntegrityScore: 0.0,
        wordAssociationIntegrity: 0.0,
        groundedWordRatio: 0.0,
        knowledgeAssociationEvidence: 0.60,
        knowledgeConsistencyScore: 0.60,
        knowledgeMatchDensity: 0.0,
        knowledgeCoverage: 0.0,
        typedKnowledgeEvidence: 0.50,
        collocationKnowledgeEvidence: 0.50,
        semanticClassFitEvidence: 0.55,
        humanDecodabilityScore: 0.0,
        shapeDecodabilityScore: 0.0,
        answerShapeKind: "prose",
        listDecodabilityScore: 0.0,
        listParallelismScore: 0.0,
        listItemCoverageScore: 0.0,
        shortAnswerDecodabilityScore: 0.0,
        temporalDecodabilityScore: 0.0,
        temporalEvidenceScore: 0.0,
        temporalTenseAlignmentScore: 0.0,
        temporalContradictionCount: 0,
        morphologyDiversityScore: 1.0,
        repeatedMorphologicalFamilies: 0,
        grammarDecodabilityScore: 0.0,
        propositionDecodabilityScore: 0.0,
        semanticRoleFitScore: 0.0,
        semanticRoleCoverageScore: 0.0,
        propositionFocusScore: 0.0,
        propositionIntegrityScore: 0.0,
        dependencyIntegrityScore: 0.0,
        pronounIntegrityScore: 0.0,
        humanTransitionIntegrityScore: 0.0,
        modifierIntegrityScore: 0.0,
        relationResolutionScore: 0.0,
        contractFulfillment: 0.0,
        answerSlotFulfillment: 0.0,
        answerSlotSubject: 0.0,
        answerSlotPredicate: 0.0,
        answerSlotRelation: 0.0,
        answerSlotOrder: 0.0,
        promptCoverage: 0.0,
        contentPrecision: 0.0,
        dualRouteCoherence: 0.0,
        topicRouteSupport: 0.0,
        intentRouteSupport: 0.0,
        latentStability: 0.0,
        repetitionScore: 0.0,
        stemDiversityScore: 0.0,
        rolloutCertainty: 0.0,
        contentRunScore: 0.0,
        casingIntegrityScore: 0.0,
        questionAlignmentScore: 0.0,
        independentVerifierScore: 0.0,
        independentActFulfillmentScore: 0.0,
        independentPropositionCompletenessScore: 0.0,
        independentPromptRelationMatchScore: 0.0,
        independentGroundingScore: 0.0,
        independentLanguageIntegrityScore: 0.0,
        independentEpistemicScore: 0.0,
        independentContradictionIntegrityScore: 1.0,
        independentRequiredAct: "answer",
        independentObservedAct: "statement",
        independentResolved: false,
        eventQueryResolutionScore: 0.0,
        querySlotResolutionScore: 0.0,
        querySlotResolved: false,
        relationStructureIntegrityScore: 0.0,
        argumentCompletenessScore: 0.0,
        relationTopologyScore: 0.0,
        mechanismTopologyScore: 0.0,
        comparativeAttachmentScore: 1.0,
        tenseContinuityScore: 1.0,
        promptRootedGroundingScore: 0.0,
        epistemicCoverageScore: 0.0,
        contradictionIntegrityScore: 1.0,
        epistemicState: "unsupported",
        compoundPenalty: 0.0,
        rawScore: 0.0
      };
    }

    const grammarDecodability =
      this.measureHumanDecodability(
        tokens,
        contextModel
      );

    const propositionDecodability =
      this.measurePropositionDecodability(
        tokens,
        contextModel
      );

    const grammarDecodabilityScore =
      grammarDecodability.score;

    const propositionDecodabilityScore =
      propositionDecodability.score;

    const answerShapeDecodability =
      this.measureAnswerShapeDecodability(
        tokens,
        contextModel,
        grammarDecodability,
        propositionDecodability
      );

    const shapeDecodabilityScore =
      answerShapeDecodability.score;

    const morphologyDiversity =
      this.measureMorphologicalDiversity(
        tokens
      );

    const morphologyDiversityScore =
      morphologyDiversity.score;

    const humanDecodabilityScore =
      shapeDecodabilityScore;

    let semantic = 0;
    let contentCount = 0;
    let relevantContentCount = 0;
    let subjects = 0;
    let verbs = 0;
    let prevType = null;
    let illegal = 0;
    let firstSubjectIndex = -1;
    let firstPredicateIndex = -1;
    let legalTransitions = 0;
    let transitionCount = 0;
    let transitionStrength = 0.0;
    let maxFunctionRun = 0;
    let functionRun = 0;
    let nonPunctCount = 0;
    let functionTokenCount = 0;
    let contentRun = 0;
    let maxContentRun = 0;
    let midSentenceCaseAnomalies = 0;

    let previousContentToken = null;
    let semanticTransitionSum = 0.0;
    let semanticTransitionCount = 0;

    let associationIntegritySum = 0.0;
    let associationIntegrityCount = 0;
    let groundedAssociationCount = 0;

    let explicitKnowledgeSum = 0.0;
    let explicitKnowledgeMatches = 0;

    let typedKnowledgeSum = 0.0;
    let typedKnowledgeMatches = 0;
    let collocationKnowledgeSum = 0.0;
    let collocationKnowledgeMatches = 0;
    let semanticClassFitSum = 0.0;
    let semanticClassFitCount = 0;
    let knowledgeSupportSum = 0.0;
    let knowledgeSupportCount = 0;

    const verifierRecentContent = [];

    // Global-coherence bookkeeping. These are linear in output length and do
    // not touch the expensive neural inference path.
    const lexicalSequence = [];
    const posSequence = [];
    const clauseSignatures = [];

    let clauseTypes = [];
    let clauseHasNominal = false;
    let clauseHasPredicate = false;
    let clauseContentCount = 0;
    let validClauses = 0;
    let totalClauses = 0;

    const finishClause = () => {
      if (!clauseTypes.length) return;

      const meaningful = clauseContentCount > 0 || clauseHasPredicate;
      if (meaningful) {
        totalClauses++;

        // A normal declarative clause needs a predicate plus an argument-like
        // element. A leading verb is allowed as a generic imperative shape.
        const leadingVerb =
          clauseTypes[0] === "Verb" ||
          clauseTypes[0] === "Aux" ||
          clauseTypes[0] === "Modal";

        if (
          clauseHasPredicate &&
          (clauseHasNominal || leadingVerb)
        ) {
          validClauses++;
        }

        // Keep only structure, never prompt-specific words.
        clauseSignatures.push(clauseTypes.slice(0, 12).join(">"));
      }

      clauseTypes = [];
      clauseHasNominal = false;
      clauseHasPredicate = false;
      clauseContentCount = 0;
    };

    const contentStems = [];
    for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
      const tok = tokens[tokenIndex];
      const type = this.getCanonicalType(tok);
      const word = this.network.vocab?.idToToken?.[tok] || "";
      const lowerWord = word.toLowerCase();

      if (
        tokenIndex > 0 &&
        type !== "Punct" &&
        type !== "ProperNoun" &&
        /^[A-Z][a-z]/.test(word)
      ) {
        const prevWord =
          this.network.vocab?.idToToken?.[tokens[tokenIndex - 1]] || "";
        if (![".", "!", "?"].includes(prevWord)) {
          midSentenceCaseAnomalies++;
        }
      }

      if (type !== "Punct") {
        nonPunctCount++;
        posSequence.push(type);

        const lexicalStem = this.stemWord(lowerWord);
        lexicalSequence.push(
          lexicalStem.length >= 2 ? lexicalStem : lowerWord
        );

        const isFunctionToken =
          ["Det", "Prep", "Conj", "Modal", "Aux"].includes(type);
        if (isFunctionToken) functionTokenCount++;

        if (this.isContentType(type)) {
          contentRun++;
          maxContentRun = Math.max(maxContentRun, contentRun);
        } else {
          contentRun = 0;
        }

        // A conjunction closes the preceding proposition and opens another.
        if (type === "Conj" && clauseTypes.length) {
          finishClause();
        }

        clauseTypes.push(type);

        if (["Noun", "ProperNoun", "Pronoun"].includes(type)) {
          clauseHasNominal = true;
        }
        if (["Verb", "Aux", "Modal"].includes(type)) {
          clauseHasPredicate = true;
        }
        if (this.isContentType(type)) {
          clauseContentCount++;
        }
      } else {
        contentRun = 0;
        finishClause();
      }

      if (this.isContentType(type)) {
        const routedSupport = this.getRoutedSemanticScore(
          tok,
          contextModel,
          null
        );
        const semanticSupport = Math.max(0, routedSupport);
        semantic += semanticSupport;
        contentCount++;

        if (previousContentToken !== null) {
          const pairSimilarity =
            this.getCachedPairSimilarity(
              previousContentToken,
              tok,
              contextModel
            );

          // Convert token-to-token embedding continuity into 0..1 evidence.
          // A small amount of novelty is healthy, a hard semantic jump is not.
          const pairEvidence = Math.max(
            0,
            Math.min(
              1,
              (pairSimilarity + 0.10) / 0.70
            )
          );

          semanticTransitionSum += pairEvidence;
          semanticTransitionCount++;
        }

        const verifierPatternContext = {
          recentContentTokens:
            verifierRecentContent.slice(-3),
          recentContentCentroid: null
        };

        const association =
          this.getGroundedWordAssociation(
            tok,
            [],
            contextModel,
            {
              subjectToken:
                contextModel?.answerContract
                  ?.subjectTokens?.[0] ?? null,
              predicateToken:
                contextModel?.answerContract
                  ?.relationVerbToken ?? null,
              predicateReady: true
            },
            verifierPatternContext
          );

        associationIntegritySum +=
          association.evidence;

        associationIntegrityCount++;

        if (
          association.evidence >= 0.48 &&
          association.anchor >= 0.30
        ) {
          groundedAssociationCount++;
        }

        const knowledgeSupport =
          Math.max(
            0,
            Math.min(
              1,
              association.knowledgeSupport || 0.55
            )
          );

        knowledgeSupportSum +=
          knowledgeSupport;

        knowledgeSupportCount++;

        semanticClassFitSum +=
          Math.max(
            0,
            Math.min(
              1,
              association.classFit || 0.55
            )
          );

        semanticClassFitCount++;

        if (
          association.knowledgeKnown
        ) {
          explicitKnowledgeSum +=
            knowledgeSupport;

          explicitKnowledgeMatches++;
        }

        if (
          (association.typedKnowledge || 0) > 0
        ) {
          typedKnowledgeSum +=
            association.typedKnowledge;

          typedKnowledgeMatches++;
        }

        if (
          (association.collocationKnowledge || 0) > 0
        ) {
          collocationKnowledgeSum +=
            association.collocationKnowledge;

          collocationKnowledgeMatches++;
        }

        verifierRecentContent.push(tok);
        if (verifierRecentContent.length > 4) {
          verifierRecentContent.shift();
        }

        previousContentToken = tok;

        const stem = this.stemWord(word);
        if (stem.length >= 3) contentStems.push(stem);

        const exactPrompt = contextModel.promptContentSet?.has(tok);
        const stemPrompt = stem.length >= 3 && contextModel.promptStems?.has(stem);
        if (exactPrompt || stemPrompt || semanticSupport >= 0.20) {
          relevantContentCount++;
        }
      }

      if (["Noun", "ProperNoun", "Pronoun"].includes(type)) {
        subjects++;
        if (firstSubjectIndex < 0) firstSubjectIndex = tokenIndex;
      }
      if (type === "Verb" || type === "Aux" || type === "Modal") {
        if (type !== "Modal") verbs++;
        if (firstPredicateIndex < 0) firstPredicateIndex = tokenIndex;
      }

      if (type !== "Punct") {
        if (prevType) {
          const transition = this.getTransitionRule(prevType, type);
          transitionCount++;

          if (transition.legal) {
            legalTransitions++;
            // Most valid matrix weights live roughly in the 0-5 range.
            transitionStrength += Math.max(
              0,
              Math.min(1, transition.weight / 5.0)
            );
          } else {
            illegal++;
          }
        }

        const isFunction = ["Det", "Prep", "Conj", "Modal", "Aux"].includes(type);
        if (isFunction) {
          functionRun++;
          maxFunctionRun = Math.max(maxFunctionRun, functionRun);
        } else {
          functionRun = 0;
        }

        prevType = type;
      } else {
        functionRun = 0;
      }
    }

    // Close a final unterminated structural segment for analysis.
    finishClause();

    const repeatedNgramPenalty = (sequence, n) => {
      if (sequence.length < n * 2) return 0.0;

      const counts = new Map();
      const total = sequence.length - n + 1;

      for (let i = 0; i <= sequence.length - n; i++) {
        const key = sequence.slice(i, i + n).join("|");
        counts.set(key, (counts.get(key) || 0) + 1);
      }

      let repeatedOccurrences = 0;
      let maxExcess = 0;

      for (const count of counts.values()) {
        if (count > 1) {
          const excess = count - 1;
          repeatedOccurrences += excess;
          maxExcess = Math.max(maxExcess, excess);
        }
      }

      const repeatRatio = repeatedOccurrences / Math.max(1, total);

      return Math.max(
        0,
        Math.min(
          1,
          repeatRatio * 1.65 +
          Math.min(0.45, maxExcess * 0.12)
        )
      );
    };

    const lexical3Penalty = repeatedNgramPenalty(lexicalSequence, 3);
    const lexical4Penalty = repeatedNgramPenalty(lexicalSequence, 4);
    const pos4Penalty = repeatedNgramPenalty(posSequence, 4);

    // Repeating an entire clause shape is a strong degeneration signal even
    // when the words are slightly different.
    let repeatedClauseShapes = 0;
    if (clauseSignatures.length > 1) {
      const shapeCounts = new Map();
      for (const signature of clauseSignatures) {
        shapeCounts.set(signature, (shapeCounts.get(signature) || 0) + 1);
      }
      for (const count of shapeCounts.values()) {
        if (count > 1) repeatedClauseShapes += count - 1;
      }
    }

    const clauseShapePenalty = clauseSignatures.length
      ? Math.min(
          1,
          repeatedClauseShapes / Math.max(1, clauseSignatures.length - 1)
        )
      : 0.0;

    const degenerationPenalty = Math.max(
      lexical3Penalty,
      lexical4Penalty,
      pos4Penalty * 0.72,
      clauseShapePenalty * 0.90
    );

    const repetitionIntegrityScore = Math.max(
      0,
      Math.min(1, 1.0 - degenerationPenalty)
    );

    const clauseIntegrityScore = totalClauses
      ? Math.max(0, Math.min(1, validClauses / totalClauses))
      : 0.0;

    // Extremely glue-heavy output is often syntactically legal word salad.
    // This is deliberately a broad linguistic check, not a prompt rule.
    const functionRatio = nonPunctCount
      ? functionTokenCount / nonPunctCount
      : 0.0;

    const functionBalanceScore = Math.max(
      0,
      Math.min(
        1,
        1.0 - Math.max(0, functionRatio - 0.48) / 0.32
      )
    );

    // Long uninterrupted content-word chains are a cheap and strong signal of
    // word salad. Natural noun compounds are tolerated up to four tokens.
    const contentRunScore = Math.max(
      0,
      Math.min(
        1,
        1.0 - Math.max(0, maxContentRun - 4) * 0.16
      )
    );

    const casingIntegrityScore = Math.max(
      0,
      Math.min(
        1,
        1.0 -
        midSentenceCaseAnomalies /
          Math.max(1, nonPunctCount * 0.18)
      )
    );

    const semanticSim =
      contentCount
        ? semantic / contentCount
        : 0.0;

    const contentPrecision = contentCount
      ? Math.max(
          0,
          Math.min(
            1,
            relevantContentCount / contentCount
          )
        )
      : 0.0;

    const wordAssociationIntegrity =
      associationIntegrityCount
        ? Math.max(
            0,
            Math.min(
              1,
              associationIntegritySum /
              associationIntegrityCount
            )
          )
        : 0.0;

    const groundedWordRatio =
      associationIntegrityCount
        ? Math.max(
            0,
            Math.min(
              1,
              groundedAssociationCount /
              associationIntegrityCount
            )
          )
        : 0.0;

    const knowledgeCoverage =
      contentCount
        ? Math.max(
            0,
            Math.min(
              1,
              explicitKnowledgeMatches /
              Math.max(
                1,
                Math.min(contentCount, 6)
              )
            )
          )
        : 0.0;

    const knownKnowledgeQuality =
      explicitKnowledgeMatches
        ? Math.max(
            0,
            Math.min(
              1,
              explicitKnowledgeSum /
              explicitKnowledgeMatches
            )
          )
        : 0.55;

    const typedKnowledgeEvidence =
      typedKnowledgeMatches
        ? Math.max(
            0,
            Math.min(
              1,
              typedKnowledgeSum /
              typedKnowledgeMatches
            )
          )
        : 0.50;

    const collocationKnowledgeEvidence =
      collocationKnowledgeMatches
        ? Math.max(
            0,
            Math.min(
              1,
              collocationKnowledgeSum /
              collocationKnowledgeMatches
            )
          )
        : 0.50;

    const semanticClassFitEvidence =
      semanticClassFitCount
        ? Math.max(
            0,
            Math.min(
              1,
              semanticClassFitSum /
              semanticClassFitCount
            )
          )
        : 0.55;

    const generalKnowledgeSupport =
      knowledgeSupportCount
        ? Math.max(
            0,
            Math.min(
              1,
              knowledgeSupportSum /
              knowledgeSupportCount
            )
          )
        : 0.55;

    const knowledgeConsistencyScore =
      Math.max(
        0,
        Math.min(
          1,
          0.60 +
          Math.max(
            0,
            knownKnowledgeQuality - 0.50
          ) * 0.28 +
          knowledgeCoverage * 0.12 +
          Math.max(
            0,
            typedKnowledgeEvidence - 0.50
          ) * 0.18 +
          Math.max(
            0,
            collocationKnowledgeEvidence - 0.50
          ) * 0.14 +
          Math.max(
            0,
            semanticClassFitEvidence - 0.55
          ) * 0.16 +
          Math.max(
            0,
            generalKnowledgeSupport - 0.55
          ) * 0.12
        )
      );

    const knowledgeAssociationEvidence =
      knowledgeConsistencyScore;

    const knowledgeMatchDensity =
      knowledgeCoverage;

    const localSemanticContinuity =
      semanticTransitionCount > 0
        ? Math.max(
            0,
            Math.min(
              1,
              semanticTransitionSum /
              semanticTransitionCount
            )
          )
        : (contentCount <= 1 ? 1.0 : 0.55);

    // Compute transition evidence BEFORE sequenceIntegrityScore uses it.
    // The previous patch declared this later in verifyCandidate(), which put
    // legalRatio in JavaScript's temporal dead zone at runtime.
    const legalRatio = transitionCount
      ? legalTransitions / transitionCount
      : 1.0;

    const meanTransitionStrength = legalTransitions
      ? transitionStrength / legalTransitions
      : 0.0;

    const semanticProgressionScore = Math.max(
      0,
      Math.min(
        1,
        localSemanticContinuity * 0.32 +
        semanticSim * 0.22 +
        contentPrecision * 0.16 +
        wordAssociationIntegrity * 0.20 +
        groundedWordRatio * 0.10
      )
    );

    // Sequence integrity now measures BOTH non-repetition and whether each
    // content step actually continues the same thought.
    const sequenceIntegrityScore = Math.max(
      0,
      Math.min(
        1,
        repetitionIntegrityScore * 0.46 +
        localSemanticContinuity * 0.34 +
        legalRatio * 0.12 +
        functionBalanceScore * 0.08
      )
    );

    const promptCoverage =
      this.promptCoverage(tokens, contextModel);

    const promptEcho =
      this.getPromptEchoDiagnostics(
        tokens,
        contextModel
      );

    const questionProfile = contextModel?.questionProfile;
    let operatorEchoCount = 0;

    if (
      questionProfile?.isQuestion &&
      Number.isFinite(questionProfile.operatorToken)
    ) {
      for (const tok of tokens) {
        if (tok === questionProfile.operatorToken) operatorEchoCount++;
      }
    }

    const operatorEchoScore = questionProfile?.isQuestion
      ? 1.0 / (1.0 + operatorEchoCount)
      : 1.0;

    const novelContentCount = this.countNovelContent(tokens, contextModel);
    const novelContentScore = contentCount
      ? Math.max(
          0,
          Math.min(
            1,
            novelContentCount / Math.max(1, contentCount * 0.22)
          )
        )
      : 0.0;

    const contractFulfillment =
      this.estimateContractFulfillment(
        tokens,
        contextModel,
        null
      );

    const answerSlots =
      this.getAnswerSlotDiagnostics(
        tokens,
        contextModel
      );

    const answerSlotFulfillment =
      answerSlots.overall;

    const instructionAdherence =
      this.measureInstructionAdherence(
        tokens,
        contextModel,
        {
          echo: promptEcho,
          semanticCoverage: promptCoverage,
          shape: shapeDecodabilityScore,
          slots: answerSlotFulfillment,
          role: propositionDecodability.roleFit
        }
      );

    const instructionAdherenceScore =
      instructionAdherence.score;

    const requiresRelation =
      Boolean(contextModel?.answerContract?.requiresRelation);

    const computedRelationResolutionScore = Math.max(
      0,
      Math.min(
        1,
        contractFulfillment * 0.34 +
        answerSlotFulfillment * 0.22 +
        clauseIntegrityScore * 0.09 +
        semanticProgressionScore * 0.11 +
        sequenceIntegrityScore * 0.07 +
        Math.min(1, promptCoverage * 1.15) * 0.10 +
        instructionAdherenceScore * 0.07
      )
    );

    // A relation channel is only evidence when the prompt contract actually
    // requires a relationship to be resolved. Otherwise it is neutral rather
    // than an artificial confidence penalty.
    const relationResolutionScore =
      requiresRelation
        ? computedRelationResolutionScore
        : 1.0;

    // "Answer completion" is grounded in meaning, not merely answer shape.
    const semanticAnswerIntegrity = Math.max(
      0,
      Math.min(
        1,
        semanticSim * 0.22 +
        contentPrecision * 0.20 +
        localSemanticContinuity * 0.14 +
        semanticProgressionScore * 0.14 +
        wordAssociationIntegrity * 0.20 +
        groundedWordRatio * 0.10
      )
    );

    const answerCompletionShape = questionProfile?.isQuestion
      ? Math.max(
          0,
          Math.min(
            1,
            relationResolutionScore * 0.19 +
            answerSlotFulfillment * 0.23 +
            clauseIntegrityScore * 0.08 +
            sequenceIntegrityScore * 0.10 +
            semanticProgressionScore * 0.10 +
            novelContentScore * 0.05 +
            Math.min(1, promptCoverage * 1.15) * 0.05 +
            instructionAdherenceScore * 0.14 +
            instructionAdherence.echoIntegrity * 0.04 +
            operatorEchoScore * 0.02
          )
        )
      : Math.max(
          0,
          Math.min(
            1,
            relationResolutionScore * 0.22 +
            clauseIntegrityScore * 0.23 +
            sequenceIntegrityScore * 0.25 +
            semanticProgressionScore * 0.30
          )
        );

    let answerCompletionScore =
      questionProfile?.isQuestion
        ? Math.max(
            0,
            Math.min(
              1,
              answerCompletionShape *
              (
                0.50 +
                semanticAnswerIntegrity * 0.50
              )
            )
          )
        : answerCompletionShape;

    if (
      contextModel?.answerShape?.kind &&
      contextModel.answerShape.kind !==
        "prose"
    ) {
      answerCompletionScore =
        Math.max(
          0,
          Math.min(
            1,
            answerCompletionScore * 0.58 +
            shapeDecodabilityScore * 0.42
          )
        );
    }

    // Global coherence operates above adjacent POS legality. This is what
    // catches output that is locally legal but globally scrambled.
    const globalCoherenceScore = Math.max(
      0,
      Math.min(
        1,
        clauseIntegrityScore * 0.18 +
        sequenceIntegrityScore * 0.20 +
        semanticProgressionScore * 0.17 +
        humanDecodabilityScore * 0.22 +
        functionBalanceScore * 0.05 +
        operatorEchoScore * 0.04 +
        instructionAdherence.echoIntegrity * 0.07 +
        contentRunScore * 0.05 +
        casingIntegrityScore * 0.03
      )
    );

    // Dual-centroid coherence asks whether the response remained connected to
    // BOTH what the prompt is about and what it is trying to do.
    let topicSupportSum = 0.0;
    let intentSupportSum = 0.0;
    let routeCount = 0;

    for (const tok of tokens) {
      const type = this.getCanonicalType(tok);
      if (!this.isContentType(type)) continue;

      topicSupportSum += Math.max(0, contextModel.topicScores?.[tok] || 0);
      intentSupportSum += Math.max(0, contextModel.intentScores?.[tok] || 0);
      routeCount++;
    }

    const topicRouteSupport = routeCount
      ? topicSupportSum / routeCount
      : 0.0;

    const intentRouteSupport = routeCount
      ? intentSupportSum / routeCount
      : 0.0;

    const dualRouteCoherence = Math.sqrt(
      Math.max(0, topicRouteSupport) *
      Math.max(0, intentRouteSupport)
    );

    const latentStability = Math.max(
      0,
      Math.min(1, contextModel.reasoningStability || 0)
    );

    const terminal = this.network.vocab?.idToToken?.[tokens[tokens.length - 1]] || "";
    const closed = [".", "!", "?"].includes(terminal);
    const syntaxScore = Math.max(0, Math.min(1,
      (closed ? 0.25 : 0) +
      (subjects > 0 ? 0.30 : 0) +
      (verbs > 0 ? 0.35 : 0) +
      (illegal === 0 ? 0.10 : Math.max(0, 0.10 - illegal * 0.04))
    ));

    // Sentence-shape quality is intentionally stricter than basic POS legality.
    // It rewards a subject before the predicate and penalizes predicate pileups.
    let clauseOrderScore = 0.0;
    if (firstSubjectIndex >= 0) clauseOrderScore += 0.25;
    if (firstPredicateIndex >= 0) clauseOrderScore += 0.25;
    if (
      firstSubjectIndex >= 0 &&
      firstPredicateIndex > firstSubjectIndex
    ) clauseOrderScore += 0.30;
    if (closed) clauseOrderScore += 0.10;
    if (verbs === 1) clauseOrderScore += 0.10;
    else if (verbs > 2) clauseOrderScore -= Math.min(0.25, (verbs - 2) * 0.08);
    clauseOrderScore = Math.max(0, Math.min(1, clauseOrderScore));


    // Fluency is intentionally local. A response can have nouns + a verb + a
    // period and still be word salad if its adjacent grammatical transitions
    // are poor. Long runs of function words are also penalized.
    const functionRunPenalty = Math.max(
      0,
      Math.min(0.35, (maxFunctionRun - 2) * 0.10)
    );

    const localFluency = Math.max(
      0,
      Math.min(
        1,
        legalRatio * 0.52 +
        meanTransitionStrength * 0.28 +
        clauseOrderScore * 0.20 -
        functionRunPenalty
      )
    );

    const fluencyScore = Math.max(
      0,
      Math.min(
        1,
        localFluency *
        (0.55 + 0.45 * clauseIntegrityScore) *
        (0.66 + 0.34 * contentRunScore) *
        (0.72 + 0.28 * casingIntegrityScore) *
        operatorEchoScore *
        (
          0.58 +
          0.42 *
          humanDecodabilityScore
        )
      )
    );

    const unique = new Set(tokens).size;
    const repetitionScore = Math.min(1, unique / Math.max(1, tokens.length) * 1.15);

    const uniqueStems = new Set(contentStems).size;
    const stemDiversityScore = contentStems.length
      ? Math.max(0, Math.min(1, uniqueStems / contentStems.length))
      : 1.0;

    const probs = candidate.trajectory?.map(x => x.prob || 0) || [];
    const rolloutCertainty = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : 0.0;

    const scaffoldSteps = candidate.trajectory?.filter(x => x.scaffold).length || 0;
    const frameUsage = tokens.length
      ? Math.max(0, Math.min(1, scaffoldSteps / Math.max(1, tokens.length - 1)))
      : 0.0;

    // Question alignment is deliberately multiplicative: a response that is
    // grammatically plausible but semantically unrelated must not keep a high
    // quality score. Geometric consensus makes one weak alignment channel
    // matter without requiring another neural pass.
    const coverageAlignment = Math.max(
      0,
      Math.min(1, promptCoverage * 1.35)
    );

    const alignmentChannels = [
      Math.max(0.02, contentPrecision),
      Math.max(0.02, semanticSim),
      Math.max(0.02, dualRouteCoherence),
      Math.max(0.02, wordAssociationIntegrity),
      Math.max(0.02, groundedWordRatio),
      Math.max(0.02, contractFulfillment),
      Math.max(0.02, answerSlotFulfillment),
      Math.max(0.02, instructionAdherenceScore),
      Math.max(0.02, instructionAdherence.echoIntegrity)
    ];

    let alignmentLog = 0.0;
    for (const value of alignmentChannels) {
      alignmentLog += Math.log(value);
    }

    const geometricAlignment = Math.exp(
      alignmentLog / alignmentChannels.length
    );

    const questionAlignmentScore = Math.max(
      0,
      Math.min(
        1,
        geometricAlignment * 0.78 +
        coverageAlignment * 0.12 +
        localSemanticContinuity * 0.10
      )
    );

    const alignmentDeficit =
      questionProfile?.isQuestion
        ? 1.0 - questionAlignmentScore
        : Math.max(
            0,
            1.0 -
            (
              contentPrecision * 0.42 +
              semanticSim * 0.30 +
              dualRouteCoherence * 0.28
            )
          );

    const structuralDeficit =
      1.0 -
      Math.min(
        fluencyScore,
        sequenceIntegrityScore,
        globalCoherenceScore
      );

    const relevanceDeficit =
      1.0 -
      Math.min(
        contentPrecision,
        Math.max(
          semanticSim,
          dualRouteCoherence
        )
      );

    const roughnessDeficit =
      1.0 -
      Math.min(
        repetitionScore,
        stemDiversityScore,
        contentRunScore,
        casingIntegrityScore,
        instructionAdherence.echoIntegrity
      );

    // "Bad + off-topic" is substantially worse than either problem alone.
    // The exponent keeps tiny alignment misses cheap while making large misses
    // increasingly severe.
    const compoundPenalty = Math.min(
      0.58,
      Math.pow(
        Math.max(0, alignmentDeficit),
        1.30
      ) *
      (
        0.10 +
        structuralDeficit * 0.18 +
        relevanceDeficit * 0.22 +
        roughnessDeficit * 0.10 +
        (1.0 - answerCompletionScore) * 0.08
      )
    );

    const groundingIntegrityScore =
      this.computeGroundingIntegrity(tokens, contextModel);

    const quantitativeIntegrityScore =
      this.computeQuantitativeIntegrity(tokens, contextModel);

    // V20 verifier is reconstructed from the finished answer. It does not use
    // beam priority, RL policy values, token-selection scores, or generator
    // slot-progress metrics.
    const independentVerification =
      this.computeIndependentAnswerVerification(
        tokens,
        contextModel
      );

    // Self-Why is generation-side reasoning only. It is recorded so RSI can
    // learn when internal reasoning and the independent verifier disagree.
    const selfWhyClause =
      this.getSelfWhyClause(
        tokens,
        contextModel,
        null,
        null,
        null
      );

    const independentVerifierScore =
      independentVerification.score;

    const independentActFulfillmentScore =
      independentVerification.actFulfillment;

    const independentPropositionCompletenessScore =
      independentVerification.propositionCompleteness;

    const independentPromptRelationMatchScore =
      independentVerification.promptRelationMatch;

    const independentGroundingScore =
      independentVerification.grounding;

    const independentLanguageIntegrityScore =
      independentVerification.languageIntegrity;

    const independentEpistemicScore =
      independentVerification.epistemic;

    const independentContradictionIntegrityScore =
      independentVerification.contradictionIntegrity;

    const eventQueryResolution =
      this.computeEventQueryResolution(tokens, contextModel);

    const semanticCore =
      this.computeSemanticCoreResolution(
        tokens,
        contextModel,
        eventQueryResolution
      );

    const epistemicCoverage =
      this.computeEpistemicCoverage(
        tokens,
        contextModel,
        eventQueryResolution
      );

    const semanticFrameFitScore =
      semanticCore.frameFit;

    const semanticGraphQueryResolutionScore =
      semanticCore.queryResolution;

    const semanticRelationCompletenessScore =
      semanticCore.relationCompleteness;

    const semanticGraphGroundingScore =
      semanticCore.grounding;

    const semanticGraphIntegrityScore =
      semanticCore.score;

    const eventQueryResolutionScore =
      eventQueryResolution.score;

    const querySlotResolutionScore =
      Math.max(
        0,
        Math.min(
          1,
          eventQueryResolution
            .slotCompleteness || 0
        )
      );

    const querySlotResolved =
      Boolean(
        eventQueryResolution
          .querySlotResolved
      );

    const relationStructureIntegrityScore =
      Math.max(
        0,
        Math.min(
          1,
          eventQueryResolution
            .relationStructureIntegrity || 0
        )
      );

    const argumentCompletenessScore =
      Math.max(
        0,
        Math.min(
          1,
          eventQueryResolution
            .argumentCompleteness || 0
        )
      );

    const relationTopologyScore =
      Math.max(
        0,
        Math.min(
          1,
          eventQueryResolution
            .relationTopology || 0
        )
      );

    const mechanismTopologyScore =
      Math.max(
        0,
        Math.min(
          1,
          eventQueryResolution
            .mechanismTopology || 0
        )
      );

    const comparativeAttachmentScore =
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(
            eventQueryResolution
              .comparativeAttachment
          )
            ? eventQueryResolution
                .comparativeAttachment
            : 1.0
        )
      );

    const tenseContinuityScore =
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(
            eventQueryResolution
              .tenseContinuity
          )
            ? eventQueryResolution
                .tenseContinuity
            : 1.0
        )
      );

    const promptRootedGroundingScore =
      eventQueryResolution.grounding;
    const epistemicCoverageScore = epistemicCoverage.score;
    const contradictionIntegrityScore = eventQueryResolution.contradictionIntegrity;

    const tailIntegrity =
      this.computeTailIntegrity(
        tokens,
        contextModel
      );

    const tailIntegrityScore =
      tailIntegrity.score;

    const universalCritic =
      this.computeUniversalCritic(
        {
          instructionAdherenceScore,
          answerCompletionScore,
          semanticAnswerIntegrity,
          shapeDecodabilityScore,
          globalCoherenceScore,
          sequenceIntegrityScore,
          contentPrecision,
          groundingIntegrityScore,
          quantitativeIntegrityScore,
          eventQueryResolutionScore,
          querySlotResolutionScore,
          relationStructureIntegrityScore,
          semanticFrameFitScore,
          semanticGraphQueryResolutionScore,
          semanticRelationCompletenessScore,
          semanticGraphGroundingScore,
          semanticGraphIntegrityScore,
          epistemicCoverageScore,
          contradictionIntegrityScore,
          tailIntegrityScore,
          independentVerifierScore,
          independentActFulfillmentScore,
          independentPropositionCompletenessScore,
          independentPromptRelationMatchScore,
          independentGroundingScore,
          independentLanguageIntegrityScore,
          independentEpistemicScore,
          independentContradictionIntegrityScore
        },
        contextModel
      );

    const universalCriticScore = universalCritic.score;

    const rawScore =
      independentVerifierScore * 0.18 +
      independentActFulfillmentScore * 0.07 +
      independentPropositionCompletenessScore * 0.06 +
      independentPromptRelationMatchScore * 0.07 +
      independentGroundingScore * 0.04 +
      independentLanguageIntegrityScore * 0.04 +
      promptCoverage * 0.02 +
      instructionAdherenceScore * 0.05 +
      instructionAdherence.echoIntegrity * 0.06 +
      instructionAdherence.novelExpansionScore * 0.05 +
      contentPrecision * 0.04 +
      semanticSim * 0.04 +
      dualRouteCoherence * 0.03 +
      wordAssociationIntegrity * 0.010 +
      groundedWordRatio * 0.010 +
      semanticFrameFitScore * 0.08 +
      semanticGraphQueryResolutionScore * 0.10 +
      semanticRelationCompletenessScore * 0.11 +
      semanticGraphGroundingScore * 0.05 +
      semanticGraphIntegrityScore * 0.07 +
      eventQueryResolutionScore * 0.025 +
      querySlotResolutionScore * 0.025 +
      relationStructureIntegrityScore * 0.035 +
      epistemicCoverageScore * 0.04 +
      contradictionIntegrityScore * 0.04 +
      tailIntegrityScore * 0.06 +
      grammarDecodabilityScore * 0.03 +
      propositionDecodabilityScore * 0.06 +
      shapeDecodabilityScore * 0.08 +
      morphologyDiversityScore * 0.03 +
      propositionDecodability.roleFit * 0.04 +
      relationResolutionScore * 0.05 +
      answerSlotFulfillment * 0.07 +
      syntaxScore * 0.02 +
      clauseOrderScore * 0.02 +
      fluencyScore * 0.04 +
      sequenceIntegrityScore * 0.05 +
      semanticProgressionScore * 0.05 +
      repetitionScore * 0.02 +
      stemDiversityScore * 0.02 +
      rolloutCertainty * 0.01 +
      latentStability * 0.01;

    const preCriticScore =
      Math.max(
        0,
        Math.min(1, rawScore - compoundPenalty)
      );

    const score =
      Math.max(
        0,
        Math.min(
          1,
          preCriticScore *
          (
            0.24 +
            independentVerifierScore * 0.34 +
            independentActFulfillmentScore * 0.09 +
            independentPropositionCompletenessScore * 0.08 +
            independentPromptRelationMatchScore * 0.09 +
            independentGroundingScore * 0.05 +
            independentLanguageIntegrityScore * 0.05 +
            universalCriticScore * 0.04 +
            tailIntegrityScore * 0.02
          )
        )
      );

    return {
      score,
      rawScore,
      preCriticScore,
      universalCriticScore,
      universalCriticCritical:
        universalCritic.critical,
      independentVerifierScore,
      independentActFulfillmentScore,
      independentPropositionCompletenessScore,
      independentPromptRelationMatchScore,
      independentGroundingScore,
      independentLanguageIntegrityScore,
      independentEpistemicScore,
      independentContradictionIntegrityScore,
      independentRequiredAct:
        independentVerification.requiredAct,
      independentObservedAct:
        independentVerification.observedAct,
      independentResolved:
        Boolean(
          independentVerification.resolved
        ),
      answerCommitment: independentVerification.commitment || "none",
      answerCommitmentStrength: independentVerification.commitmentStrength || 0.0,
      answerCommitmentResolved: Boolean(independentVerification.commitmentResolved),
      selfWhyClauseScore:
        selfWhyClause.score,
      selfWhyPromptFit:
        selfWhyClause.promptFit,
      selfWhyRelationFit:
        selfWhyClause.relationFit,
      selfWhyEvidenceSupport:
        selfWhyClause.evidenceSupport,
      selfWhyDependencySupport:
        selfWhyClause.dependencySupport,
      selfWhyGrounding:
        selfWhyClause.grounding,
      selfWhyNovelty:
        selfWhyClause.novelty,
      selfWhyContradictionRisk:
        selfWhyClause.contradictionRisk,
      selfWhyAnswerProgress:
        selfWhyClause.answerProgress,
      selfWhyVerifierGap:
        Math.max(
          0,
          selfWhyClause.score -
          independentVerifierScore
        ),
      groundingIntegrityScore,
      quantitativeIntegrityScore,
      semanticFrameFitScore,
      semanticGraphQueryResolutionScore,
      semanticRelationCompletenessScore,
      semanticGraphGroundingScore,
      semanticGraphIntegrityScore,
      semanticFrameKind:
        semanticCore.frameKind,
      semanticQueryTarget:
        semanticCore.queryTarget,
      eventQueryResolutionScore,
      querySlotResolutionScore,
      querySlotResolved,
      relationStructureIntegrityScore,
      argumentCompletenessScore,
      relationTopologyScore,
      mechanismTopologyScore,
      comparativeAttachmentScore,
      tenseContinuityScore,
      querySlotEvidenceCount:
        eventQueryResolution
          .slotEvidenceCount || 0,
      querySlotMarkerSupport:
        eventQueryResolution
          .markerSupport || 0,
      promptRootedGroundingScore,
      epistemicCoverageScore,
      contradictionIntegrityScore,
      tailIntegrityScore,
      tailGroundingScore:
        tailIntegrity.grounding,
      tailRelevanceScore:
        tailIntegrity.relevance,
      tailQueryFitScore:
        tailIntegrity.queryFit,
      tailResolutionRetention:
        tailIntegrity.resolutionRetention,
      tailLength:
        tailIntegrity.tailLength,
      epistemicState: epistemicCoverage.state,
      factualityRequired:
        Boolean(contextModel?.answerPlan?.factualityRequired),
      requiresNumericEvidence:
        Boolean(contextModel?.answerPlan?.requiresNumericEvidence),
      underspecifiedPrompt:
        Boolean(contextModel?.answerPlan?.underspecified),
      isQuestion:
        Boolean(questionProfile?.isQuestion),
      questionAlignmentScore,
      compoundPenalty,
      semanticSim,
      syntaxScore,
      clauseOrderScore,
      fluencyScore,
      globalCoherenceScore,
      sequenceIntegrityScore,
      localSemanticContinuity,
      semanticProgressionScore,
      clauseIntegrityScore,
      answerCompletionScore,
      semanticAnswerIntegrity,
      wordAssociationIntegrity,
      groundedWordRatio,
      knowledgeAssociationEvidence,
      knowledgeConsistencyScore,
      knowledgeMatchDensity,
      knowledgeCoverage,
      typedKnowledgeEvidence,
      collocationKnowledgeEvidence,
      semanticClassFitEvidence,
      lexicalApiVersion:
        this.lexicalEngineInfo?.apiVersion || 0,
      morphologyDiversityScore,
      repeatedMorphologicalFamilies:
        morphologyDiversity.repeatedFamilies,
      humanDecodabilityScore,
      shapeDecodabilityScore,
      answerShapeKind:
        answerShapeDecodability.kind,
      listDecodabilityScore:
        answerShapeDecodability
          .list?.score || 0.0,
      listParallelismScore:
        answerShapeDecodability
          .list?.parallelism || 0.0,
      listItemCoverageScore:
        answerShapeDecodability
          .list?.itemCoverage || 0.0,
      shortAnswerDecodabilityScore:
        answerShapeDecodability
          .short?.score || 0.0,
      temporalDecodabilityScore:
        answerShapeDecodability
          .temporal?.score || 0.0,
      temporalEvidenceScore:
        answerShapeDecodability
          .temporal?.temporalEvidence || 0.0,
      temporalTenseAlignmentScore:
        answerShapeDecodability
          .temporal?.temporalTenseAlignment || 0.0,
      temporalContradictionCount:
        answerShapeDecodability
          .temporal?.temporalContradictions || 0,
      grammarDecodabilityScore,
      propositionDecodabilityScore,
      semanticRoleFitScore:
        propositionDecodability.roleFit,
      semanticRoleCoverageScore:
        propositionDecodability.roleCoverage,
      propositionFocusScore:
        propositionDecodability.focus,
      propositionIntegrityScore:
        grammarDecodability.propositionIntegrity,
      dependencyIntegrityScore:
        grammarDecodability.dependencyIntegrity,
      pronounIntegrityScore:
        grammarDecodability.pronounIntegrity,
      humanTransitionIntegrityScore:
        grammarDecodability.transitionIntegrity,
      modifierIntegrityScore:
        grammarDecodability.modifierIntegrity,
      relationResolutionScore,
      requiresRelation,
      contractFulfillment,
      answerSlotFulfillment,
      answerSlotSubject: answerSlots.subject,
      answerSlotPredicate: answerSlots.predicate,
      answerSlotRelation: answerSlots.relation,
      answerSlotOrder: answerSlots.order,
      promptCoverage,
      instructionAdherenceScore,
      instructionIntent:
        contextModel?.instructionContract?.intent || "answer",
      instructionRole:
        contextModel?.instructionContract?.requiredRole || "content",
      surfaceEchoRatio:
        instructionAdherence.surfaceEchoRatio,
      gratuitousEchoRatio:
        instructionAdherence.gratuitousEchoRatio,
      echoIntegrity:
        instructionAdherence.echoIntegrity,
      novelContentRatio:
        instructionAdherence.novelContentRatio,
      novelExpansionScore:
        instructionAdherence.novelExpansionScore,
      sourceFamiliesUsed:
        instructionAdherence.sourceFamiliesUsed,
      surfaceReuseBudget:
        contextModel?.instructionContract?.surfaceReuseBudget || 0,
      anchorBudget: contextModel?.anchorBudget || 0,
      uniquePromptContent:
        contextModel?.uniqueContentCount || 0,
      contentPrecision,
      dualRouteCoherence,
      topicRouteSupport,
      intentRouteSupport,
      latentStability,
      repetitionScore,
      stemDiversityScore,
      rolloutCertainty,
      frameUsage,
      contentRunScore,
      casingIntegrityScore
    };
  }

  candidateQualityVector(candidate) {
    return {
      score: candidate?.score ?? -1.0,
      syntax: candidate?.syntaxScore ?? 0.0,
      order: candidate?.clauseOrderScore ?? 0.0,
      fluency: candidate?.fluencyScore ?? 0.0,
      precision: candidate?.contentPrecision ?? 0.0,
      coverage: candidate?.promptCoverage ?? 0.0,
      semantic: candidate?.semanticSim ?? 0.0,
      sequence: candidate?.sequenceIntegrityScore ?? 0.0,
      progression:
        candidate?.semanticProgressionScore ?? 0.0,
      relation:
        candidate?.relationResolutionScore ?? 0.0,
      alignment:
        candidate?.questionAlignmentScore ?? 0.0,
      instruction:
        candidate?.instructionAdherenceScore ?? 0.0,
      echo:
        candidate?.echoIntegrity ?? 0.0,
      slots:
        candidate?.answerSlotFulfillment ?? 0.0,
      repetition: candidate?.repetitionScore ?? 0.0
    };
  }

  isSafeRSLUpgrade(candidate, currentBest) {
    if (!candidate) return false;
    if (!currentBest) return true;

    const next = this.candidateQualityVector(candidate);
    const base = this.candidateQualityVector(currentBest);

    // Never trade a meaningful amount of readability for a tiny composite gain.
    if (next.fluency < base.fluency - 0.025) return false;
    if (next.syntax < base.syntax - 0.025) return false;
    if (next.order < base.order - 0.035) return false;
    if (next.precision < base.precision - 0.045) return false;
    if (next.sequence < base.sequence - 0.030) return false;
    if (next.progression < base.progression - 0.030) return false;
    if (next.relation < base.relation - 0.030) return false;
    if (next.alignment < base.alignment - 0.025) return false;
    if (next.instruction < base.instruction - 0.025) return false;
    if (next.echo < base.echo - 0.040) return false;
    if (next.slots < base.slots - 0.025) return false;
    if (next.repetition < base.repetition - 0.050) return false;

    // A candidate that is visibly cleaner may replace the baseline even when
    // its scalar score is nearly tied. Otherwise demand a real improvement.
    const languageGain =
      (next.fluency - base.fluency) * 0.32 +
      (next.syntax - base.syntax) * 0.17 +
      (next.order - base.order) * 0.16 +
      (next.sequence - base.sequence) * 0.15 +
      (next.progression - base.progression) * 0.14 +
      (next.relation - base.relation) * 0.06;

    const relevanceGain =
      (next.precision - base.precision) * 0.27 +
      (next.coverage - base.coverage) * 0.16 +
      (next.semantic - base.semantic) * 0.18 +
      (next.relation - base.relation) * 0.10 +
      (next.alignment - base.alignment) * 0.20 +
      (next.instruction - base.instruction) * 0.18 +
      (next.echo - base.echo) * 0.08;

    return (
      next.score >= base.score + 0.003 ||
      languageGain >= 0.025 ||
      relevanceGain >= 0.030
    );
  }

  getConfidenceDiagnostics(candidate) {
    const empty = {
      relevance: 0,
      language: 0,
      semantic: 0,
      answer: 0,
      consistency: 0,
      neural: 0,
      reasoning: 0,
      coherence: 0,
      sequence: 0,
      semanticProgression: 0,
      localSemanticContinuity: 0,
      relation: 0,
      requiresRelation: false,
      contractFulfillment: 0,
      integrity: 0,
      disagreement: 0,
      qualityFactor: 0,
      raw: 0,
      highQualityGate: 0,
      qualityConsensus: 0,
      final: 0,

      // RSL v4 hierarchical confidence groups.
      meaningGroup: 0,
      structureGroup: 0,
      fulfillmentGroup: 0,
      neuralGroup: 0,
      criticalFloor: 0,
      groupAgreement: 0,

      promptCoverage: 0,
      contentPrecision: 0,
      dualRouteCoherence: 0,
      syntax: 0,
      clauseOrder: 0,
      fluency: 0,
      globalCoherence: 0,
      answerCompletion: 0,
      semanticAnswerIntegrity: 0,
      wordAssociationIntegrity: 0,
      groundedWordRatio: 0,
      knowledgeAssociationEvidence: 0.60,
      knowledgeConsistency: 0.60,
      knowledgeMatchDensity: 0,
      knowledgeCoverage: 0,
      typedKnowledge: 0.50,
      collocationKnowledge: 0.50,
      semanticClassFit: 0.55,
      lexicalApiVersion: 0,
      knowledgeLift: 0,
      humanDecodability: 0,
      answerShape: "prose",
      shapeDecodability: 0,
      listDecodability: 0,
      listParallelism: 0,
      listItemCoverage: 0,
      shortAnswerDecodability: 0,
      temporalTenseAlignment: 0,
      temporalContradictions: 0,
      morphologyDiversity: 1,
      repeatedMorphologicalFamilies: 0,
      grammarDecodability: 0,
      propositionDecodability: 0,
      semanticRoleFit: 0,
      semanticRoleCoverage: 0,
      propositionFocus: 0,
      propositionIntegrity: 0,
      dependencyIntegrity: 0,
      pronounIntegrity: 0,
      humanTransitionIntegrity: 0,
      modifierIntegrity: 0,
      readabilityLift: 0,
      repetition: 0,
      stemDiversity: 0,
      contentRun: 0,
      casingIntegrity: 0,
      alignment: 0,
      instructionAdherence: 0,
      instructionIntent: "answer",
      instructionRole: "content",
      surfaceEchoRatio: 0,
      gratuitousEchoRatio: 0,
      echoIntegrity: 1,
      novelContentRatio: 0,
      novelExpansion: 0,
      surfaceReuseBudget: 0,

      // Kept for UI/backward compatibility. RSL v4 no longer multiplies
      // confidence by separate relation/alignment/slot gates.
      relationGate: 1,
      alignmentGate: 1,
      slots: 0,
      slotSubject: 0,
      slotPredicate: 0,
      slotRelation: 0,
      slotOrder: 0,
      slotGate: 1,

      compoundPenalty: 0,
      rawCandidateScore: 0,
      universalCritic: 0,
      independentVerifier: 0,
      independentActFulfillment: 0,
      independentPropositionCompleteness: 0,
      independentPromptRelationMatch: 0,
      independentGrounding: 0,
      independentLanguageIntegrity: 0,
      independentEpistemic: 0,
      independentContradictionIntegrity: 1,
      independentRequiredAct: "answer",
      independentObservedAct: "statement",
      independentResolved: false,
      groundingIntegrity: 0,
      quantitativeIntegrity: 1,
      semanticFrameFit: 0,
      semanticGraphQueryResolution: 0,
      semanticRelationCompleteness: 0,
      semanticGraphGrounding: 0,
      semanticGraphIntegrity: 0,
      semanticFrameKind: "relation",
      semanticQueryTarget: "content",
      eventQueryResolution: 0,
      querySlotResolution: 0,
      querySlotResolved: false,
      relationStructureIntegrity: 0,
      argumentCompleteness: 0,
      relationTopology: 0,
      mechanismTopology: 0,
      comparativeAttachment: 1,
      tenseContinuity: 1,
      tailIntegrity: 0,
      tailGrounding: 0,
      tailRelevance: 0,
      tailQueryFit: 0,
      tailResolutionRetention: 0,
      promptRootedGrounding: 0,
      epistemicCoverage: 0,
      contradictionIntegrity: 1,
      epistemicState: "unsupported",
      preferenceScore: 0.5,
      confidenceCap: 1,
      anchorBudget: 0,
      uniquePromptContent: 0
    };

    if (!candidate || !candidate.tokens?.length) {
      return empty;
    }

    const clamp = x =>
      Math.max(
        0,
        Math.min(
          1,
          Number.isFinite(x)
            ? x
            : 0
        )
      );

    const relevanceEvidence = clamp(
      clamp(candidate.promptCoverage) * 0.40 +
      clamp(candidate.contentPrecision) * 0.36 +
      clamp(candidate.dualRouteCoherence) * 0.24
    );

    const sequenceEvidence =
      clamp(candidate.sequenceIntegrityScore);

    const progressionEvidence =
      clamp(candidate.semanticProgressionScore);

    const coherenceEvidence =
      clamp(candidate.globalCoherenceScore);

    const languageEvidence = clamp(
      clamp(candidate.syntaxScore) * 0.16 +
      clamp(candidate.clauseOrderScore) * 0.17 +
      clamp(candidate.fluencyScore) * 0.25 +
      coherenceEvidence * 0.20 +
      sequenceEvidence * 0.22
    );

    const semanticEvidence = clamp(
      clamp(candidate.semanticSim) * 0.31 +
      clamp(candidate.dualRouteCoherence) * 0.25 +
      clamp(candidate.contentPrecision) * 0.22 +
      progressionEvidence * 0.22
    );

    const consistencyEvidence = clamp(
      clamp(candidate.repetitionScore) * 0.22 +
      clamp(candidate.stemDiversityScore) * 0.18 +
      sequenceEvidence * 0.30 +
      clamp(candidate.contentRunScore) * 0.18 +
      clamp(candidate.casingIntegrityScore) * 0.12
    );

    const answerEvidence =
      clamp(candidate.answerCompletionScore);

    const answerSemanticEvidence =
      clamp(candidate.semanticAnswerIntegrity);

    const requiresRelation =
      Boolean(candidate.requiresRelation);

    const relationEvidence =
      requiresRelation
        ? clamp(candidate.relationResolutionScore)
        : 1.0;

    const alignmentEvidence =
      clamp(candidate.questionAlignmentScore);

    const instructionEvidence =
      clamp(candidate.instructionAdherenceScore);

    const echoIntegrityEvidence =
      Number.isFinite(candidate.echoIntegrity)
        ? clamp(candidate.echoIntegrity)
        : 1.0;

    const novelExpansionEvidence =
      clamp(candidate.novelExpansionScore);

    const isQuestion =
      Boolean(candidate.isQuestion);

    const semanticFrameFit =
      clamp(
        candidate
          .semanticFrameFitScore
      );

    const semanticGraphQueryResolution =
      clamp(
        candidate
          .semanticGraphQueryResolutionScore
      );

    const semanticRelationCompleteness =
      clamp(
        candidate
          .semanticRelationCompletenessScore
      );

    const semanticGraphGrounding =
      clamp(
        candidate
          .semanticGraphGroundingScore
      );

    const semanticGraphIntegrity =
      clamp(
        candidate
          .semanticGraphIntegrityScore
      );

    const eventQueryResolution =
      clamp(
        candidate
          .eventQueryResolutionScore
      );

    const querySlotResolution =
      clamp(
        candidate
          .querySlotResolutionScore
      );

    const querySlotResolved =
      Boolean(
        candidate
          .querySlotResolved
      );

    const relationStructureIntegrity =
      clamp(
        candidate
          .relationStructureIntegrityScore
      );

    const argumentCompleteness =
      clamp(
        candidate
          .argumentCompletenessScore
      );

    const relationTopology =
      clamp(
        candidate
          .relationTopologyScore
      );

    const mechanismTopology =
      clamp(
        candidate
          .mechanismTopologyScore
      );

    const comparativeAttachment =
      Number.isFinite(
        candidate
          .comparativeAttachmentScore
      )
        ? clamp(
            candidate
              .comparativeAttachmentScore
          )
        : 1.0;

    const tenseContinuity =
      Number.isFinite(
        candidate
          .tenseContinuityScore
      )
        ? clamp(
            candidate
              .tenseContinuityScore
          )
        : 1.0;

    const tailIntegrity =
      Number.isFinite(candidate.tailIntegrityScore)
        ? clamp(candidate.tailIntegrityScore)
        : 1.0;

    const tailGrounding =
      Number.isFinite(candidate.tailGroundingScore)
        ? clamp(candidate.tailGroundingScore)
        : tailIntegrity;

    const tailRelevance =
      Number.isFinite(candidate.tailRelevanceScore)
        ? clamp(candidate.tailRelevanceScore)
        : tailIntegrity;

    const tailQueryFit =
      Number.isFinite(candidate.tailQueryFitScore)
        ? clamp(candidate.tailQueryFitScore)
        : tailIntegrity;

    const tailResolutionRetention =
      Number.isFinite(candidate.tailResolutionRetention)
        ? clamp(candidate.tailResolutionRetention)
        : tailIntegrity;

    const promptRootedGrounding =
      clamp(candidate.promptRootedGroundingScore);

    const epistemicCoverage =
      clamp(candidate.epistemicCoverageScore);

    const contradictionIntegrity =
      Number.isFinite(candidate.contradictionIntegrityScore)
        ? clamp(candidate.contradictionIntegrityScore)
        : 1.0;

    const slotEvidence =
      isQuestion
        ? clamp(candidate.answerSlotFulfillment)
        : answerEvidence;

    const neuralEvidence =
      clamp(candidate.rolloutCertainty);

    const reasoningEvidence =
      clamp(candidate.latentStability);

    const associationEvidence =
      clamp(
        candidate.wordAssociationIntegrity
      );

    const groundedWordEvidence =
      clamp(
        candidate.groundedWordRatio
      );

    const knowledgeEvidence =
      Number.isFinite(
        candidate.knowledgeConsistencyScore
      )
        ? clamp(
            candidate.knowledgeConsistencyScore
          )
        : Number.isFinite(
            candidate.knowledgeAssociationEvidence
          )
          ? clamp(
              candidate.knowledgeAssociationEvidence
            )
          : 0.60;

    const knowledgeDensity =
      clamp(
        candidate.knowledgeCoverage ??
        candidate.knowledgeMatchDensity
      );

    const typedKnowledgeEvidence =
      Number.isFinite(
        candidate.typedKnowledgeEvidence
      )
        ? clamp(
            candidate.typedKnowledgeEvidence
          )
        : 0.50;

    const collocationKnowledgeEvidence =
      Number.isFinite(
        candidate.collocationKnowledgeEvidence
      )
        ? clamp(
            candidate.collocationKnowledgeEvidence
          )
        : 0.50;

    const semanticClassFitEvidence =
      Number.isFinite(
        candidate.semanticClassFitEvidence
      )
        ? clamp(
            candidate.semanticClassFitEvidence
          )
        : 0.55;

    const humanDecodability =
      clamp(
        candidate.humanDecodabilityScore
      );

    const answerShape =
      String(
        candidate.answerShapeKind ||
        "prose"
      );

    let shapeDecodability =
      clamp(
        candidate.shapeDecodabilityScore ??
        candidate.humanDecodabilityScore
      );

    const temporalTenseAlignment =
      clamp(
        candidate.temporalTenseAlignmentScore
      );

    const temporalContradictions =
      Number.isFinite(
        candidate.temporalContradictionCount
      )
        ? Math.max(
            0,
            candidate.temporalContradictionCount
          )
        : 0;

    if (
      answerShape === "temporal"
    ) {
      shapeDecodability =
        clamp(
          shapeDecodability * 0.72 +
          temporalTenseAlignment * 0.28
        );

      if (
        temporalContradictions > 0
      ) {
        shapeDecodability *=
          Math.pow(
            0.55,
            Math.min(
              3,
              temporalContradictions
            )
          );
      }
    }

    const grammarDecodability =
      clamp(
        candidate.grammarDecodabilityScore
      );

    const propositionDecodability =
      clamp(
        candidate.propositionDecodabilityScore
      );

    const semanticRoleFit =
      clamp(
        candidate.semanticRoleFitScore
      );

    const semanticRoleCoverage =
      clamp(
        candidate.semanticRoleCoverageScore
      );

    const propositionFocus =
      clamp(
        candidate.propositionFocusScore
      );

    const propositionIntegrity =
      clamp(
        candidate.propositionIntegrityScore
      );

    const dependencyIntegrity =
      clamp(
        candidate.dependencyIntegrityScore
      );

    const pronounIntegrity =
      clamp(
        candidate.pronounIntegrityScore
      );

    const humanTransitionIntegrity =
      clamp(
        candidate.humanTransitionIntegrityScore
      );

    const modifierIntegrity =
      clamp(
        candidate.modifierIntegrityScore
      );

    // ------------------------------------------------------------
    // 1) MEANING: Does the answer stay about the user's question?
    // ------------------------------------------------------------
    const baseMeaningSupport = clamp(
      eventQueryResolution * 0.18 +
      querySlotResolution * 0.18 +
      relationStructureIntegrity * 0.20 +
      promptRootedGrounding * 0.09 +
      epistemicCoverage * 0.07 +
      contradictionIntegrity * 0.06 +
      propositionDecodability * 0.06 +
      semanticRoleFit * 0.05 +
      instructionEvidence * 0.04 +
      tailIntegrity * 0.03 +
      relevanceEvidence * 0.02 +
      semanticEvidence * 0.01 +
      alignmentEvidence * 0.005 +
      answerSemanticEvidence * 0.005
    );

    // No additive knowledge bonus. Knowledge must improve provenance/query
    // resolution upstream; it cannot award confidence a second time here.
    const knowledgeMeaningBonus = 0.0;
    const meaningSupport = baseMeaningSupport;

    const meaningFloor =
      Math.min(
        eventQueryResolution,
        querySlotResolution,
        relationStructureIntegrity,
        promptRootedGrounding,
        contradictionIntegrity,
        tailIntegrity,
        alignmentEvidence,
        answerSemanticEvidence,
        propositionDecodability,
        semanticRoleFit,
        instructionEvidence
      );

    const meaningGroup = clamp(
      meaningSupport * 0.82 +
      Math.sqrt(
        Math.max(
          0,
          meaningSupport * meaningFloor
        )
      ) * 0.18
    );

    // ------------------------------------------------------------
    // 2) STRUCTURE: Is it readable, ordered, and non-degenerate?
    // ------------------------------------------------------------
    const morphologyDiversity =
      clamp(
        candidate.morphologyDiversityScore
      );

    const structureSupport = clamp(
      relationStructureIntegrity * 0.18 +
      argumentCompleteness * 0.10 +
      shapeDecodability * 0.23 +
      grammarDecodability * 0.14 +
      morphologyDiversity * 0.08 +
      languageEvidence * 0.11 +
      coherenceEvidence * 0.07 +
      sequenceEvidence * 0.04 +
      tailIntegrity * 0.04 +
      consistencyEvidence * 0.01
    );

    const structureFloor =
      Math.min(
        relationStructureIntegrity,
        argumentCompleteness,
        shapeDecodability,
        languageEvidence,
        coherenceEvidence,
        sequenceEvidence,
        tailIntegrity,
        morphologyDiversity
      );

    const structureGroup = clamp(
      structureSupport * 0.86 +
      Math.sqrt(
        Math.max(
          0,
          structureSupport * structureFloor
        )
      ) * 0.14
    );

    // ------------------------------------------------------------
    // 3) ANSWER FULFILLMENT: Did it actually answer the requested kind?
    // ------------------------------------------------------------
    const contractEvidence =
      clamp(candidate.contractFulfillment);

    const fulfillmentWeights =
      this.reasoningConfig
        .fulfillmentWeights;

    const fulfillmentSupport =
      isQuestion
        ? clamp(
            eventQueryResolution *
              fulfillmentWeights.eventQuery +
            querySlotResolution *
              fulfillmentWeights.querySlot +
            relationStructureIntegrity *
              fulfillmentWeights.relationStructure +
            epistemicCoverage *
              fulfillmentWeights.epistemic +
            answerEvidence *
              fulfillmentWeights.answer +
            relationEvidence *
              fulfillmentWeights.relation +
            contractEvidence *
              fulfillmentWeights.contract +
            propositionDecodability *
              fulfillmentWeights.proposition +
            shapeDecodability *
              fulfillmentWeights.shape +
            semanticRoleFit *
              fulfillmentWeights.semanticRole +
            instructionEvidence *
              fulfillmentWeights.instruction
          )
        : clamp(
            answerEvidence * 0.48 +
            contractEvidence * 0.27 +
            relationEvidence * 0.25
          );

    const fulfillmentFloor =
      isQuestion
        ? Math.min(
            eventQueryResolution,
            querySlotResolution,
            relationStructureIntegrity,
            epistemicCoverage,
            contradictionIntegrity,
            slotEvidence,
            answerEvidence,
            relationEvidence,
            shapeDecodability,
            instructionEvidence
          )
        : Math.min(
            answerEvidence,
            contractEvidence,
            relationEvidence
          );

    const fulfillmentGroup = clamp(
      fulfillmentSupport * 0.82 +
      Math.sqrt(
        Math.max(
          0,
          fulfillmentSupport * fulfillmentFloor
        )
      ) * 0.18
    );

    // ------------------------------------------------------------
    // 4) NEURAL EVIDENCE: intentionally small; confidence should describe
    // answer quality, not merely decoder decisiveness.
    // ------------------------------------------------------------
    const neuralGroup = clamp(
      neuralEvidence * 0.62 +
      reasoningEvidence * 0.38
    );

    // ONE confidence model. No relationGate * alignmentGate * slotGate chain.
    const confidenceWeights =
      this.reasoningConfig
        .confidenceWeights;

    const base = clamp(
      meaningGroup *
        confidenceWeights.meaning +
      structureGroup *
        confidenceWeights.structure +
      fulfillmentGroup *
        confidenceWeights.fulfillment +
      neuralGroup *
        confidenceWeights.neural
    );

    const criticalFloor =
      Math.min(
        meaningGroup,
        structureGroup,
        fulfillmentGroup
      );

    const groups = [
      meaningGroup,
      structureGroup,
      fulfillmentGroup
    ];

    const groupMean =
      groups.reduce(
        (sum, value) =>
          sum + value,
        0
      ) / groups.length;

    let variance = 0.0;
    for (const value of groups) {
      const diff =
        value - groupMean;
      variance +=
        diff * diff;
    }
    variance /=
      groups.length;

    const disagreement =
      clamp(
        Math.sqrt(variance) * 1.55
      );

    const groupAgreement =
      clamp(
        1.0 - disagreement
      );

    // One mild balance term. A single weak critical group matters once,
    // without the same failure being multiplied through several gates.
    const balancedSupport =
      Math.sqrt(
        Math.max(
          0,
          base * criticalFloor
        )
      );

    const rawConfidence =
      clamp(
        base * 0.79 +
        balancedSupport * 0.21
      );

    const qualityFactor =
      clamp(
        0.94 +
        groupAgreement * 0.06
      );

    const preKnowledgeConfidence =
      clamp(
        rawConfidence *
        qualityFactor
      );

    // Knowledge is already represented through prompt-rooted provenance.
    // Do not grant a second additive confidence reward for association density.
    const knowledgeLift = 0.0;
    const knowledgeAdjusted = preKnowledgeConfidence;

    // RSL v8 readability accelerator:
    // Once Meaning + Structure + Fulfillment + Human Decodability all agree
    // that the answer is strong, confidence is allowed to climb into the 90s
    // quickly. Garbage gets zero benefit because the minimum channel is low.
    const readableConsensus =
      Math.min(
        meaningGroup,
        structureGroup,
        fulfillmentGroup,
        eventQueryResolution,
        querySlotResolution,
        relationStructureIntegrity,
        epistemicCoverage,
        contradictionIntegrity,
        tailIntegrity,
        shapeDecodability,
        instructionEvidence
      );

    const readableStrength =
      readableConsensus > 0.76
        ? Math.max(
            0,
            Math.min(
              1,
              (readableConsensus - 0.76) /
              0.24
            )
          )
        : 0.0;

    const readabilityLift =
      (
        1.0 -
        knowledgeAdjusted
      ) *
      0.80 *
      Math.pow(
        readableStrength,
        1.18
      );

    let calibrated =
      clamp(
        knowledgeAdjusted +
        readabilityLift
      );

    const universalCritic =
      clamp(candidate.universalCriticScore);

    const independentVerifier =
      clamp(
        candidate
          .independentVerifierScore
      );

    const independentActFulfillment =
      clamp(
        candidate
          .independentActFulfillmentScore
      );

    const independentPropositionCompleteness =
      clamp(
        candidate
          .independentPropositionCompletenessScore
      );

    const independentPromptRelationMatch =
      clamp(
        candidate
          .independentPromptRelationMatchScore
      );

    const independentGrounding =
      clamp(
        candidate
          .independentGroundingScore
      );

    const independentLanguageIntegrity =
      clamp(
        candidate
          .independentLanguageIntegrityScore
      );

    const independentEpistemic =
      clamp(
        candidate
          .independentEpistemicScore
      );

    const independentContradictionIntegrity =
      Number.isFinite(
        candidate
          .independentContradictionIntegrityScore
      )
        ? clamp(
            candidate
              .independentContradictionIntegrityScore
          )
        : 1.0;

    const groundingIntegrity =
      clamp(candidate.groundingIntegrityScore);

    const quantitativeIntegrity =
      Number.isFinite(candidate.quantitativeIntegrityScore)
        ? clamp(candidate.quantitativeIntegrityScore)
        : 1.0;

    const factualityRequired =
      Boolean(candidate.factualityRequired);

    const requiresNumericEvidence =
      Boolean(candidate.requiresNumericEvidence);

    let confidenceCap = 1.0;

    const confidenceCaps =
      this.reasoningConfig
        .confidenceCaps;

    if (isQuestion) {
      if (querySlotResolution < 0.34) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps.slotVeryLow
          );
      } else if (
        querySlotResolution < 0.50
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps.slotLow
          );
      } else if (
        querySlotResolution <
          this.reasoningConfig
            .querySlotResolveFloor
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps.slotPartial
          );
      } else if (
        querySlotResolution <
          this.reasoningConfig
            .querySlotHighConfidenceFloor
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps.slotNear
          );
      }

      if (
        relationStructureIntegrity < 0.34
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps
              .relationVeryLow
          );
      } else if (
        relationStructureIntegrity < 0.50
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps
              .relationLow
          );
      } else if (
        relationStructureIntegrity <
          this.reasoningConfig
            .relationStructureResolveFloor
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps
              .relationPartial
          );
      } else if (
        relationStructureIntegrity <
          this.reasoningConfig
            .relationStructureHighConfidenceFloor
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps
              .relationNear
          );
      }

      if (
        eventQueryResolution < 0.42
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps.eventLow
          );
      } else if (
        eventQueryResolution < 0.58
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps.eventPartial
          );
      } else if (
        eventQueryResolution <
          this.reasoningConfig
            .eventQueryHighConfidenceFloor
      ) {
        confidenceCap =
          Math.min(
            confidenceCap,
            confidenceCaps.eventNear
          );
      }
    }

    if (factualityRequired && epistemicCoverage < 0.40) {
      confidenceCap = Math.min(confidenceCap, 0.52);
    } else if (factualityRequired && epistemicCoverage < this.reasoningConfig.epistemicHighConfidenceFloor) {
      confidenceCap = Math.min(confidenceCap, 0.78);
    }

    if (contradictionIntegrity < 0.70) {
      confidenceCap = Math.min(confidenceCap, 0.56);
    }

    if (tailIntegrity < 0.34) {
      confidenceCap = Math.min(confidenceCap, 0.44);
    } else if (tailIntegrity < 0.48) {
      confidenceCap = Math.min(confidenceCap, 0.60);
    } else if (tailIntegrity < this.reasoningConfig.tailHighConfidenceFloor) {
      confidenceCap = Math.min(confidenceCap, 0.78);
    }

    if (universalCritic < 0.42) {
      confidenceCap = 0.48;
    } else if (universalCritic < 0.55) {
      confidenceCap = 0.62;
    } else if (universalCritic < 0.68) {
      confidenceCap = 0.76;
    } else if (universalCritic < 0.80) {
      confidenceCap = 0.88;
    }

    if (factualityRequired && groundingIntegrity < 0.48) {
      confidenceCap = Math.min(confidenceCap, 0.64);
    }

    if (requiresNumericEvidence && quantitativeIntegrity < 0.58) {
      confidenceCap = Math.min(confidenceCap, 0.62);
    }

    calibrated = Math.min(calibrated, confidenceCap);

    if (
      universalCritic > 0.86 &&
      criticalFloor > 0.78 &&
      tailIntegrity >=
        this.reasoningConfig
          .tailHighConfidenceFloor &&
      (
        !isQuestion ||
        (
          querySlotResolved &&
          querySlotResolution >=
            this.reasoningConfig
              .querySlotHighConfidenceFloor &&
          relationStructureIntegrity >=
            this.reasoningConfig
              .relationStructureHighConfidenceFloor &&
          eventQueryResolution >=
            this.reasoningConfig
              .eventQueryHighConfidenceFloor
        )
      ) &&
      (
        !factualityRequired ||
        epistemicCoverage >=
          this.reasoningConfig
            .epistemicHighConfidenceFloor
      )
    ) {
      const criticLift =
        Math.max(
          0,
          Math.min(1, (universalCritic - 0.86) / 0.14)
        );

      calibrated =
        clamp(
          calibrated +
          (1.0 - calibrated) * 0.42 * criticLift
        );
    }

    // ------------------------------------------------------------
    // V20 final confidence: independently reconstructed answer-act verifier.
    // Generation/RL scores are intentionally excluded. Neural confidence is
    // retained as a tiny 3% channel only.
    // ------------------------------------------------------------
    const independentConfidence =
      this.computeIndependentConfidence(
        {
          actFulfillment:
            independentActFulfillment,
          propositionCompleteness:
            independentPropositionCompleteness,
          promptRelationMatch:
            independentPromptRelationMatch,
          grounding:
            independentGrounding,
          languageIntegrity:
            independentLanguageIntegrity,
          epistemic:
            independentEpistemic,
          contradictionIntegrity:
            independentContradictionIntegrity
        },
        neuralGroup,
        factualityRequired
      );

    calibrated =
      independentConfidence.score;

    confidenceCap =
      independentConfidence.cap;

    const preferenceScore =
      this.preferenceExamples >= 3
        ? this.predictPreference(candidate)
        : 0.50;

    // Preference/RL affects generation policy only. It never modifies V20
    // confidence because that would let a learned habit validate itself.

    const integrity =
      structureGroup;

    // Compatibility diagnostics now summarize the grouped model rather than
    // controlling confidence independently.
    const highQualityGate =
      criticalFloor;

    const qualityConsensus =
      Math.pow(
        Math.max(
          1e-6,
          meaningGroup *
          structureGroup *
          fulfillmentGroup
        ),
        1 / 3
      );

    return {
      relevance: relevanceEvidence,
      language: languageEvidence,
      semantic: semanticEvidence,
      relation: relationEvidence,
      requiresRelation,
      answer: answerEvidence,
      consistency: consistencyEvidence,
      neural: neuralEvidence,
      reasoning: reasoningEvidence,
      coherence: coherenceEvidence,
      sequence: sequenceEvidence,
      semanticProgression:
        progressionEvidence,
      localSemanticContinuity:
        clamp(candidate.localSemanticContinuity),
      contractFulfillment:
        contractEvidence,

      meaningGroup,
      structureGroup,
      fulfillmentGroup,
      neuralGroup,
      criticalFloor,
      groupAgreement,

      relationGate: 1.0,
      alignmentGate: 1.0,
      slotGate: 1.0,

      integrity,
      disagreement,
      qualityFactor,
      raw: rawConfidence,
      highQualityGate,
      qualityConsensus,
      final: calibrated,

      promptCoverage:
        clamp(candidate.promptCoverage),
      contentPrecision:
        clamp(candidate.contentPrecision),
      dualRouteCoherence:
        clamp(candidate.dualRouteCoherence),
      syntax:
        clamp(candidate.syntaxScore),
      clauseOrder:
        clamp(candidate.clauseOrderScore),
      fluency:
        clamp(candidate.fluencyScore),
      globalCoherence:
        coherenceEvidence,
      answerCompletion:
        answerEvidence,
      semanticAnswerIntegrity:
        answerSemanticEvidence,
      wordAssociationIntegrity:
        associationEvidence,
      groundedWordRatio:
        groundedWordEvidence,
      knowledgeAssociationEvidence:
        knowledgeEvidence,
      knowledgeConsistency:
        knowledgeEvidence,
      knowledgeMatchDensity:
        knowledgeDensity,
      knowledgeCoverage:
        knowledgeDensity,
      typedKnowledge:
        typedKnowledgeEvidence,
      collocationKnowledge:
        collocationKnowledgeEvidence,
      semanticClassFit:
        semanticClassFitEvidence,
      lexicalApiVersion:
        this.lexicalEngineInfo?.apiVersion || 0,
      knowledgeLift,
      humanDecodability,
      answerShape,
      shapeDecodability,
      listDecodability:
        clamp(
          candidate.listDecodabilityScore
        ),
      listParallelism:
        clamp(
          candidate.listParallelismScore
        ),
      listItemCoverage:
        clamp(
          candidate.listItemCoverageScore
        ),
      shortAnswerDecodability:
        clamp(
          candidate.shortAnswerDecodabilityScore
        ),
      temporalTenseAlignment,
      temporalContradictions,
      morphologyDiversity,
      repeatedMorphologicalFamilies:
        Number.isFinite(
          candidate.repeatedMorphologicalFamilies
        )
          ? candidate.repeatedMorphologicalFamilies
          : 0,
      grammarDecodability,
      propositionDecodability,
      semanticRoleFit,
      semanticRoleCoverage,
      propositionFocus,
      propositionIntegrity,
      dependencyIntegrity,
      pronounIntegrity,
      humanTransitionIntegrity,
      modifierIntegrity,
      readabilityLift,
      repetition:
        clamp(candidate.repetitionScore),
      stemDiversity:
        clamp(candidate.stemDiversityScore),
      contentRun:
        clamp(candidate.contentRunScore),
      casingIntegrity:
        clamp(candidate.casingIntegrityScore),
      alignment:
        alignmentEvidence,
      instructionAdherence:
        instructionEvidence,
      instructionIntent:
        candidate.instructionIntent || "answer",
      instructionRole:
        candidate.instructionRole || "content",
      surfaceEchoRatio:
        clamp(candidate.surfaceEchoRatio),
      gratuitousEchoRatio:
        clamp(candidate.gratuitousEchoRatio),
      echoIntegrity:
        echoIntegrityEvidence,
      novelContentRatio:
        clamp(candidate.novelContentRatio),
      novelExpansion:
        novelExpansionEvidence,
      surfaceReuseBudget:
        Number.isFinite(candidate.surfaceReuseBudget)
          ? candidate.surfaceReuseBudget
          : 0,

      slots:
        slotEvidence,
      slotSubject:
        clamp(candidate.answerSlotSubject),
      slotPredicate:
        clamp(candidate.answerSlotPredicate),
      slotRelation:
        clamp(candidate.answerSlotRelation),
      slotOrder:
        clamp(candidate.answerSlotOrder),

      compoundPenalty:
        Number.isFinite(candidate.compoundPenalty)
          ? candidate.compoundPenalty
          : 0.0,
      rawCandidateScore:
        Number.isFinite(candidate.rawScore)
          ? candidate.rawScore
          : candidate.score,
      universalCritic,
      independentVerifier,
      independentActFulfillment,
      independentPropositionCompleteness,
      independentPromptRelationMatch,
      independentGrounding,
      independentLanguageIntegrity,
      independentEpistemic,
      independentContradictionIntegrity,
      independentRequiredAct:
        candidate
          .independentRequiredAct ||
        "answer",
      independentObservedAct:
        candidate
          .independentObservedAct ||
        "statement",
      independentResolved:
        Boolean(
          candidate
            .independentResolved
        ),
      answerCommitment: candidate.answerCommitment || candidate.commitment || "none",
      answerCommitmentStrength: clamp(candidate.answerCommitmentStrength ?? candidate.commitmentStrength),
      answerCommitmentResolved: Boolean(candidate.answerCommitmentResolved ?? candidate.commitmentResolved),

      // Diagnostic only: final confidence remains independent-verifier-owned.
      selfWhyClauseScore:
        clamp(candidate.selfWhyClauseScore),
      selfWhyPromptFit:
        clamp(candidate.selfWhyPromptFit),
      selfWhyRelationFit:
        clamp(candidate.selfWhyRelationFit),
      selfWhyEvidenceSupport:
        clamp(candidate.selfWhyEvidenceSupport),
      selfWhyDependencySupport:
        clamp(candidate.selfWhyDependencySupport),
      selfWhyGrounding:
        clamp(candidate.selfWhyGrounding),
      selfWhyNovelty:
        clamp(candidate.selfWhyNovelty),
      selfWhyContradictionRisk:
        clamp(candidate.selfWhyContradictionRisk),
      selfWhyAnswerProgress:
        clamp(candidate.selfWhyAnswerProgress),
      selfWhyVerifierGap:
        clamp(candidate.selfWhyVerifierGap),

      groundingIntegrity,
      quantitativeIntegrity,
      semanticFrameFit,
      semanticGraphQueryResolution,
      semanticRelationCompleteness,
      semanticGraphGrounding,
      semanticGraphIntegrity,
      semanticFrameKind:
        candidate
          .semanticFrameKind ||
        "relation",
      semanticQueryTarget:
        candidate
          .semanticQueryTarget ||
        "content",
      eventQueryResolution,
      querySlotResolution,
      querySlotResolved,
      relationStructureIntegrity,
      argumentCompleteness,
      relationTopology,
      mechanismTopology,
      comparativeAttachment,
      tenseContinuity,
      tailIntegrity,
      tailGrounding,
      tailRelevance,
      tailQueryFit,
      tailResolutionRetention,
      promptRootedGrounding,
      epistemicCoverage,
      contradictionIntegrity,
      epistemicState: candidate.epistemicState || "unsupported",
      preferenceScore,
      confidenceCap,
      anchorBudget:
        Number.isFinite(candidate.anchorBudget)
          ? candidate.anchorBudget
          : 0,
      uniquePromptContent:
        Number.isFinite(candidate.uniquePromptContent)
          ? candidate.uniquePromptContent
          : 0
    };
  }

  computeConfidence(candidate) {
    return this.getConfidenceDiagnostics(candidate).final;
  }

  // ========================================================
  // ANYTIME PIPELINE
  // Deep = ONE neural rollout with stronger per-token planning.
  // Low  = ONE simpler stochastic rollout.
  // ========================================================
  async generate(rawPromptTokens, onProgress = null) {
    const cleanPromptTokens = rawPromptTokens
      .map(t => (typeof t === "object" && t !== null ? t.id : t))
      .filter(Number.isFinite);

    if (!cleanPromptTokens.length) {
      this.lastDiscardedTrajectory = [];
      return { tokens: [], confidence: "0.0" };
    }

    this.lastDiscardedTrajectory = [];

    // The network can load/adapt embeddings independently of the RSL
    // engine. Refresh semantic vectors before building this prompt's context.
    this.ensureNormalizedEmbeddingsCurrent();

    const deepMode = this.mode === "rsl" && this.isDeepLearning;
    const promptLen = cleanPromptTokens.length;
    const contextModel = this.extractContextModel(cleanPromptTokens);
    this.prepareUserInputLearning(cleanPromptTokens, contextModel);

    await this.prepareMemoryContext(
      cleanPromptTokens,
      contextModel
    );

    const targetLength = deepMode
      ? this.estimateAnswerLength(cleanPromptTokens, contextModel)
      : Math.max(
          8,
          Math.min(
            this.maxTokenCeiling,
            Math.round(8 + Math.sqrt(promptLen) * 5 + promptLen * 0.55)
          )
        );

    const maxTokens = deepMode
      ? this.getAdaptiveGenerationBudget(
          promptLen,
          targetLength,
          contextModel
        )
      : Math.min(
          this.maxTokenCeiling,
          Math.max(targetLength, Math.round(targetLength * 1.20))
        );

    contextModel.latentHopBudget = deepMode
      ? this.getAdaptiveLatentHops(promptLen, targetLength)
      : 1;

    if (!deepMode) {
      const candidate = await this.executeRollout(
        cleanPromptTokens,
        contextModel,
        targetLength,
        maxTokens,
        0.42,
        false,
        null
      );

      const evaluation = this.verifyCandidate(
        candidate,
        contextModel,
        targetLength
      );
      const winner = { ...candidate, ...evaluation };

      if (onProgress) onProgress(1, 1, winner.score);

      this.trajectory = winner.trajectory || [];

      const confidenceDiagnostics =
        this.getConfidenceDiagnostics(winner);

      this.recordResponseLexicalUsage(
        winner.tokens
      );

      this.commitUserInputLearning(
        cleanPromptTokens,
        contextModel
      );

      this.commitMemoryTurn(
        cleanPromptTokens,
        winner,
        confidenceDiagnostics,
        contextModel,
        {
          refinementRounds: 0,
          neuralRetryUsed: false
        }
      );

      return {
        tokens: winner.tokens,
        confidence:
          (confidenceDiagnostics.final * 100).toFixed(1),
        confidenceDiagnostics: {
          ...confidenceDiagnostics,
          mode: "low",
          passesUsed: 1,
          preferredPasses: 1,
          generationBudget: maxTokens,
          targetLength,
          userStatePredictionAccuracy:
            contextModel?.userInputLearningProfile?.statePredictionAccuracy ?? null,
          nextUserTokenHitEMA:
            contextModel?.userInputLearningProfile?.nextTokenHitEMA ?? null,
          predictedNextUserTokens:
            contextModel?.userInputLearningProfile?.predictedNextTokenIds?.length || 0,
          memoryEnabled:
            Boolean(
              this.memoryStore &&
              this.memoryRetrieval
            ),
          memoryOverallRelevance:
            contextModel?.memoryBundle?.overallRelevance || 0.0
        }
      };
    }

    // ================================================================
    // RSL v2 DEEP: ONE FORWARD + ONE CONSTRAINED SEARCH
    // ================================================================
    const rslTargetLength =
      this.getRSLTargetLength(
        targetLength,
        promptLen,
        contextModel
      );

    const rslMaxTokens =
      this.getAdaptiveGenerationBudget(
        promptLen,
        rslTargetLength,
        contextModel
      );

    let winner =
      await this.executeRSLSearch(
        cleanPromptTokens,
        contextModel,
        rslTargetLength,
        rslMaxTokens
      );

    let confidenceDiagnostics =
      this.getConfidenceDiagnostics(
        winner
      );

    const initialWinner =
      winner;

    const initialDiagnostics =
      confidenceDiagnostics;

    const rsiAttempts = [
      {
        winner,
        diagnostics:
          confidenceDiagnostics,
        strategy: "initial"
      }
    ];

    let bestUtility =
      this.getRecoveryCandidateUtility(
        confidenceDiagnostics
      );

    let refinementRounds = 0;

    let previousConfidence =
      confidenceDiagnostics.final;

    let neuralForwardsUsed = 1;
    let neuralRetryUsed = false;
    let neuralRetryInputLength = 0;

    const sharedRawLogits =
      winner.rawLogits;

    const memoryRecoveryHint =
      this.getMemoryRecoveryHint(
        contextModel
      );

    let cpuRecoveryLimit =
      Math.min(
        this.reasoningConfig
          .refinementMaxRounds,
        this.reasoningConfig
          .cpuRecoveryRoundsBeforeNeuralRetry
      );

    if (
      memoryRecoveryHint?.strategy ===
      "neural_retry"
    ) {
      cpuRecoveryLimit = 0;
    } else if (
      memoryRecoveryHint?.strategy ===
      "cpu_expand"
    ) {
      cpuRecoveryLimit =
        Math.min(
          this.reasoningConfig
            .refinementMaxRounds,
          cpuRecoveryLimit + 1
        );
    }

    // First repair stage: cheap CPU-only search over the same logits.
    while (
      !this.isRecoveredAnswerAcceptable(
        confidenceDiagnostics
      ) &&
      refinementRounds <
        cpuRecoveryLimit &&
      sharedRawLogits
    ) {
      refinementRounds++;

      contextModel.refinementRound =
        refinementRounds;

      contextModel.recoveryProfile =
        this.buildConfidenceRecoveryProfile(
          confidenceDiagnostics,
          contextModel
        );

      const refinedMaxTokens =
        Math.min(
          this.maxTokenCeiling,
          rslMaxTokens +
          refinementRounds *
          this.reasoningConfig
            .refinementTokenSlack
        );

      const refined =
        await this.executeRSLSearch(
          cleanPromptTokens,
          contextModel,
          rslTargetLength,
          refinedMaxTokens,
          sharedRawLogits
        );

      const refinedDiagnostics =
        this.getConfidenceDiagnostics(
          refined
        );

      rsiAttempts.push({
        winner:
          refined,
        diagnostics:
          refinedDiagnostics,
        strategy:
          "cpu_expand"
      });

      const refinedUtility =
        this.getRecoveryCandidateUtility(
          refinedDiagnostics
        );

      const confidenceGain =
        refinedDiagnostics.final -
        confidenceDiagnostics.final;

      if (
        refinedUtility >
        bestUtility +
          this.reasoningConfig
            .refinementMinGain
      ) {
        winner =
          refined;

        confidenceDiagnostics =
          refinedDiagnostics;

        bestUtility =
          refinedUtility;
      }

      if (
        this.isRecoveredAnswerAcceptable(
          confidenceDiagnostics
        )
      ) {
        break;
      }

      const usefulGain =
        Math.max(
          confidenceGain,
          confidenceDiagnostics.final -
          previousConfidence
        );

      previousConfidence =
        confidenceDiagnostics.final;

      if (
        usefulGain <
          this.reasoningConfig
            .refinementMinGain
      ) {
        break;
      }
    }

    // Second repair stage: if the independent verifier still rejects the
    // answer, permit one fresh neural interpretation. This is self-conditioned
    // by the failed answer, but the failed trajectory is explicitly suppressed
    // during decoding so the model does not simply repeat it.
    if (
      !this.isRecoveredAnswerAcceptable(
        confidenceDiagnostics
      ) &&
      this.reasoningConfig
        .neuralRetryEnabled &&
      neuralForwardsUsed <
        this.reasoningConfig
          .maxNeuralForwards
    ) {
      neuralRetryUsed = true;

      const failedForRetry =
        winner?.tokens?.length
          ? winner
          : initialWinner;

      const failedDiagnostics =
        confidenceDiagnostics ||
        initialDiagnostics;

      contextModel.recoveryProfile =
        this.buildConfidenceRecoveryProfile(
          failedDiagnostics,
          contextModel
        );

      contextModel.failedTrajectorySuppression =
        this.buildFailedTrajectorySuppression(
          failedForRetry
            ?.tokens || [],
          contextModel
        );

      contextModel.neuralRetryActive =
        true;

      const retryInput =
        this.buildNeuralRetryInput(
          cleanPromptTokens,
          failedForRetry
            ?.tokens || []
        );

      neuralRetryInputLength =
        retryInput.length;

      const retryRawLogits =
        await this.network
          .forwardSequence(
            retryInput
          );

      neuralForwardsUsed++;

      this.resetReasoningForNeuralRetry(
        contextModel
      );

      contextModel.refinementRound =
        Math.min(
          this.reasoningConfig
            .refinementMaxRounds,
          Math.max(
            1,
            refinementRounds + 1
          )
        );

      const retryMaxTokens =
        Math.min(
          this.maxTokenCeiling,
          rslMaxTokens +
          this.reasoningConfig
            .neuralRetryTokenSlack
        );

      const neuralRetryWinner =
        await this.executeRSLSearch(
          cleanPromptTokens,
          contextModel,
          rslTargetLength,
          retryMaxTokens,
          retryRawLogits
        );

      const neuralRetryDiagnostics =
        this.getConfidenceDiagnostics(
          neuralRetryWinner
        );

      rsiAttempts.push({
        winner:
          neuralRetryWinner,
        diagnostics:
          neuralRetryDiagnostics,
        strategy:
          "neural_retry"
      });

      const retryUtility =
        this.getRecoveryCandidateUtility(
          neuralRetryDiagnostics
        );

      if (
        retryUtility >
        bestUtility +
          this.reasoningConfig
            .refinementMinGain
      ) {
        winner =
          neuralRetryWinner;

        confidenceDiagnostics =
          neuralRetryDiagnostics;

        bestUtility =
          retryUtility;
      }
    }

    // RSI learns only AFTER all attempts are evaluated, so it cannot change
    // the current answer mid-search. The next prompt benefits from the mistake.
    const automaticRSIResult =
      this.learnFromAutomaticRSIAttempts(
        rsiAttempts
      );

    contextModel.refinementRound = 0;
    contextModel.recoveryProfile = null;
    contextModel.neuralRetryActive = false;
    contextModel.failedTrajectorySuppression = null;

    if (onProgress) {
      onProgress(
        1,
        1,
        winner.score ?? 0
      );
    }

    this.trajectory =
      winner.trajectory || [];

    this.lastDiscardedTrajectory =
      Array.isArray(
        winner.rslMeta
          ?.discardedTrajectory
      )
        ? winner.rslMeta
            .discardedTrajectory
            .map(step => ({ ...step }))
        : [];

    const confidenceValue =
      confidenceDiagnostics.final;

    this.recordResponseLexicalUsage(
      winner.tokens || []
    );

    this.commitUserInputLearning(
      cleanPromptTokens,
      contextModel
    );

    this.commitMemoryTurn(
      cleanPromptTokens,
      winner,
      confidenceDiagnostics,
      contextModel,
      {
        refinementRounds,
        neuralRetryUsed
      }
    );

    return {
      tokens: winner.tokens || [],
      confidence:
        (confidenceValue * 100).toFixed(1),
      passesUsed: 1,
      preferredPasses: 1,
      generationBudget:
        rslMaxTokens,
      confidenceDiagnostics: {
        ...confidenceDiagnostics,
        mode: "deep",
        rslVersion: 23,
        passesUsed: 1,
        preferredPasses: 1,
        refinementRounds,
        neuralRetryUsed,
        neuralRetryInputLength,
        automaticRSILearned:
          automaticRSIResult
            ?.learned || false,
        automaticRSIUpdates:
          automaticRSIResult
            ?.updates || 0,
        automaticRSIContrastGain:
          automaticRSIResult
            ?.contrastGain || 0.0,
        totalRSIUpdates:
          this.rsiUpdates,
        automaticMistakesLearned:
          this.automaticMistakesLearned,
        selfWhyClauseScore:
          confidenceDiagnostics?.selfWhyClauseScore ?? null,
        selfWhyVerifierGap:
          confidenceDiagnostics?.selfWhyVerifierGap ?? null,
        selfWhyDisagreementEMA:
          this.selfWhyDisagreementEMA,
        selfWhyDisagreementCount:
          this.selfWhyDisagreementCount,
        memoryEnabled:
          Boolean(
            this.memoryStore &&
            this.memoryRetrieval
          ),
        memoryOverallRelevance:
          contextModel?.memoryBundle?.overallRelevance || 0.0,
        memoryWorkingRetrieved:
          contextModel?.memoryBundle?.working?.length || 0,
        memoryEpisodesRetrieved:
          contextModel?.memoryBundle?.episodic?.length || 0,
        memorySemanticRetrieved:
          contextModel?.memoryBundle?.semantic?.length || 0,
        memoryProceduralRetrieved:
          contextModel?.memoryBundle?.procedural?.length || 0,
        memoryRecoveryHint:
          contextModel?.memoryRecoveryHint?.strategy || null,
        userInputTurns: this.userInputTurns,
        userStatePredictionAccuracy:
          contextModel?.userInputLearningProfile?.statePredictionAccuracy ?? null,
        nextUserTokenHitEMA:
          contextModel?.userInputLearningProfile?.nextTokenHitEMA ?? null,
        predictedNextUserTokens:
          contextModel?.userInputLearningProfile?.predictedNextTokenIds?.length || 0,
        conversationCarryoverNeed:
          contextModel?.userInputLearningProfile?.carryoverNeed ?? 0.0,
        confidenceAcceptanceFloor:
          this.reasoningConfig
            .answerAcceptanceConfidence,
        recoveryAccepted:
          this.isRecoveredAnswerAcceptable(
            confidenceDiagnostics
          ),
        generationBudget:
          rslMaxTokens,
        targetLength:
          rslTargetLength,
        beamWidth:
          winner.rslMeta?.beamWidth || 1,
        peakBeams:
          winner.rslMeta?.peakBeams || 1,
        branchFactor:
          winner.rslMeta?.branchFactor || 1,
        branchEvaluations:
          winner.rslMeta?.branchEvaluations || 0,
        finalistsVerified:
          winner.rslMeta?.finalistsVerified || 1,
        candidatePoolSize:
          winner.rslMeta?.candidatePoolSize ||
          contextModel
            ?.deepCandidatePool
            ?.length ||
          0,
        candidatePoolRound:
          winner.rslMeta
            ?.candidatePoolRound || 0,
        candidatePoolFamilyLimit:
          winner.rslMeta
            ?.candidatePoolFamilyLimit || 0,
        neuralForwards:
          neuralForwardsUsed,
        rolledBackTokens:
          winner.rslMeta?.rolledBackTokens || 0,
        discardedTailSteps:
          winner.rslMeta?.discardedTrajectory?.length || 0,
        bestPrefixQuality:
          winner.rslMeta?.bestPrefixQuality ?? null,
        finalPrefixQuality:
          winner.rslMeta?.finalPrefixQuality ?? null,
        rawModelConfidence:
          confidenceDiagnostics.final,
        confidenceAfterMomentum:
          confidenceDiagnostics.final,
        confidenceMomentum: 0.0,
        firstPassPenalty: 0.0,
        unpenalizedSearchScore:
          winner.rawScore ??
          winner.score ??
          0.0
      }
    };
  }

  train(
    userRewardSignal,
    trajectoryOverride = null,
    promptTokensOverride = null,
    responseTokensOverride = null,
    diagnosticsOverride = null
  ) {
    this.episodeCount++;

    const signed =
      Math.max(-1, Math.min(1, Number(userRewardSignal) || 0));

    this.cumulativeReward += signed;

    const liveTrajectory = this.trajectory;

    if (Array.isArray(trajectoryOverride)) {
      this.trajectory = trajectoryOverride;
    }

    let contextModel = null;
    let criticDiagnostics = diagnosticsOverride;

    try {
      const prompt =
        (promptTokensOverride || [])
          .map(token =>
            typeof token === "object" && token !== null
              ? token.id
              : token
          )
          .filter(Number.isFinite);

      if (prompt.length) {
        this.ensureNormalizedEmbeddingsCurrent();
        contextModel = this.extractContextModel(prompt);

        if (
          !criticDiagnostics &&
          Array.isArray(responseTokensOverride) &&
          responseTokensOverride.length
        ) {
          const tokens =
            responseTokensOverride
              .map(token =>
                typeof token === "object" && token !== null
                  ? token.id
                  : token
              )
              .filter(Number.isFinite);

          const evaluation =
            this.verifyCandidate(
              {
                tokens,
                trajectory: this.trajectory || [],
                rawLogits: null
              },
              contextModel,
              Math.max(1, tokens.length)
            );

          criticDiagnostics =
            this.getConfidenceDiagnostics(evaluation);
        }
      }
    } catch (_) {
      // Feedback must still work if reconstruction fails.
    }

    const critic =
      Math.max(
        0,
        Math.min(
          1,
          criticDiagnostics?.universalCritic ??
          criticDiagnostics?.criticalFloor ??
          0.50
        )
      );

    const effectivePatternReward =
      signed > 0
        ? signed * (0.32 + critic * 0.68)
        : signed;

    this.learnTrajectoryPatterns(
      effectivePatternReward,
      contextModel
    );

    this.learnWordAssociations(
      effectivePatternReward,
      promptTokensOverride
    );

    if (criticDiagnostics) {
      this.updatePreferenceModel(signed, criticDiagnostics);
    }

    this.trainRLPolicy(
      signed,
      this.trajectory,
      criticDiagnostics
    );

    if (
      Array.isArray(this.lastDiscardedTrajectory) &&
      this.lastDiscardedTrajectory.length
    ) {
      const discardedReward =
        this.reasoningConfig
          .discardedTailReward *
        Math.max(
          0.35,
          Math.abs(signed)
        );

      this.trainRLPolicy(
        discardedReward,
        this.lastDiscardedTrajectory,
        {
          ...(criticDiagnostics || {}),
          eventQueryResolution:
            Math.min(
              0.48,
              criticDiagnostics
                ?.eventQueryResolution ??
              0.48
            ),
          universalCritic:
            Math.min(
              0.50,
              criticDiagnostics
                ?.universalCritic ??
              0.50
            )
        }
      );
    }

    this.trimPatternMap(
      this.wordAssociationMemory,
      this.maxAssociationEntries
    );

    this.trimPatternMap(
      this.scopedTokenPatternMemory,
      this.maxScopedPatternEntries
    );

    this.trimPatternMap(
      this.scopedPosPatternMemory,
      this.maxScopedPatternEntries
    );

    this.savePatternMemory();
    this.saveUserPredictiveMemory();
    this.trajectory = liveTrajectory;

    return {
      reward: signed,
      effectivePatternReward,
      critic,
      preference:
        criticDiagnostics
          ? this.predictPreference(criticDiagnostics)
          : 0.50,
      rlUpdates: this.rlUpdates,
      rlReplaySize:
        this.rlReplay.length,
      rsiUpdates:
        this.rsiUpdates,
      rsiReplaySize:
        this.rsiReplay.length,
      automaticMistakesLearned:
        this.automaticMistakesLearned,
      selfWhyDisagreementEMA:
        this.selfWhyDisagreementEMA,
      selfWhyDisagreementCount:
        this.selfWhyDisagreementCount,
      userInputTurns: this.userInputTurns,
      userStatePredictionAccuracy:
        Math.max(0, Math.min(1, 1.0 - this.userStateErrorEMA)),
      nextUserTokenHitEMA: this.userNextTokenHitEMA,
      discardedTailSteps:
        this.lastDiscardedTrajectory.length
    };
  }

}