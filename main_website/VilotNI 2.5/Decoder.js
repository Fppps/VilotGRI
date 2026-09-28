/**
 * VilotNI 2 - Decoder.js
 *
 * Production decoder for the unified VilotNI 2 model. TrainingInfo.json may be kept
 * beside the model as a reference dataset, but it is not loaded by this decoder.
 * Words.json supplies lexical POS possibilities while RSL resolves the active POS
 * per occurrence from learned context. Processor.js remains the hard legality gate.
 */
(() => {

  'use strict';


  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(x) ? x : 0));


  // Generic grammatical bridge inventory. These are not topic answers or
  // sentence templates. They are ordinary function/relational words resolved
  // through Words.json and scored by the same learned context as every other
  // token. The decoder only exposes them as candidates when content words begin
  // to stack without an expressed relation.
  const TRANSITION_POS = new Set(['Det','Prep','Conj','Aux','Modal','Pronoun','Punct']);

  const TRANSITION_WORDS = Object.freeze([
    'a','an','the','and','or','but','because','if','while','although','though',
    'of','to','in','on','at','for','from','with','without','by','through','using',
    'across','into','onto','over','under','between','among','during','before','after',
    'as','than','is','am','are','was','were','be','been','being','do','does','did',
    'have','has','had','can','could','will','would','shall','should','may','might','must'
  ]);

  const TRANSITION_WORD_SET = new Set(TRANSITION_WORDS);

  // Only relational connectors are injected as a rescue when content words
  // start stacking. Determiners and auxiliaries still exist in the ordinary
  // candidate pool/scaffold, but blindly injecting them between nouns produced
  // phrases such as "model the training" or repeated copulas.
  const BRIDGE_WORDS = Object.freeze([
    'and','or','but','because','if','while','although','though',
    'of','to','in','on','at','for','from','with','without','by','through','using',
    'across','into','onto','over','under','between','among','during','before','after',
    'as','than'
  ]);
  const BRIDGE_WORD_SET = new Set(BRIDGE_WORDS);

  const TERMINAL_WORD_SET = new Set(['.','!','?']);


  class VilotDecoder {

    constructor(arena, rsl, processor, trace = null, options = {
}
, matrixEngine = null, lexical = null) {

      this.arena = arena;

      this.rsl = rsl;

      this.processor = processor;

      this.runtimeOptions = options;

      this.trace = trace;

      this.matrix = matrixEngine || (globalThis.VilotMatrixEngine ? new VilotMatrixEngine(options) : null);

      this.lexical = lexical || {
}
;

      this.batchRankWeights = new Float32Array([0.27, 0.11, 0.08, 0.07, 0.05, 0.05, 0.05, 0.03, 0.03, 0.07, 0.05, 0.04, 0.02, 0.06, 0.09, 0.07, 0.06]);

      this.options = {

        maxResponseChars: Math.max(256, options.decoderMaxResponseChars | 0 || 2400),
        targetMin: Math.max(6, options.decoderPureTargetMin | 0 || 9),
        targetMax: Math.max(12, options.decoderPureTargetMax | 0 || 24),
        candidateLimit: Math.max(32, options.decoderPureCandidateLimit | 0 || 128),
        focusedStaticPoolLimit: Math.max(24, options.latencyFocusedStaticPoolLimit | 0 || 64),
        focusedDynamicRSLLimit: Math.max(6, options.latencyFocusedDynamicRSLLimit | 0 || 12),
        focusedCorrelationLimit: Math.max(8, options.latencyFocusedCorrelationLimit | 0 || 16),
        focusedTargetMax: Math.max(8, options.latencyFocusedTargetMax | 0 || 12),
        focusedTargetMin: Math.max(4, options.latencyFocusedTargetMin | 0 || 6),
        deadlineCheckStride: Math.max(4, options.latencyDeadlineCheckStride | 0 || 12),
        deadlineFinishReserveMs: Math.max(1, Number(options.latencyDecoderFinishReserveMs) || 3.5),
        syntaxBridgeAfterContentRun: Math.max(2, Math.min(4, options.syntaxBridgeAfterContentRun | 0 || 2)),
        syntaxTransitionCandidateScore: clamp(options.syntaxTransitionCandidateScore ?? 0.24, 0.05, 0.8),
        syntaxStackPenaltyBase: clamp(options.syntaxStackPenaltyBase ?? 0.045, 0.01, 0.2),
        syntaxTransitionBoost: clamp(options.syntaxTransitionBoost ?? 0.13, 0.02, 0.3),
        scaffoldCoreLock: options.scaffoldCoreLock !== false,
        repeatedBridgePenalty: clamp(options.repeatedBridgePenalty ?? 0.14, 0.04, 0.30),
        bridgeDensityPenalty: clamp(options.bridgeDensityPenalty ?? 0.06, 0.02, 0.18),
        prepositionDepthDecay: clamp(options.prepositionDepthDecay ?? 0.56, 0.25, 0.9),
        maxSamePrepositionHops: Math.max(1, Math.min(4, options.maxSamePrepositionHops | 0 || 2)),
        terminalCandidateScore: clamp(options.terminalCandidateScore ?? 0.32, 0.08, 0.8),
        terminalCompletionBoost: clamp(options.terminalCompletionBoost ?? 0.48, 0.12, 0.9),
        terminalCompletionAllowance: Math.max(2, Math.min(8, options.terminalCompletionAllowance | 0 || 4)),
        preserveFullModelBreadth: options.preserveFullModelBreadth !== false
      
}
;

      this.ready = true;

      this.lastGenerationMs = 0;

      this.lastDiagnostics = null;

      this.generationCount = 0;

    
}


    _capitalize(text) {

      return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;

    
}


    _staticCandidatePool(analysis, explorationLevel = 0) {

      const pool = new Map();

      const add = (entry, sourceScore = 0, source = 'unknown', focusScore = 0.5) => {

        if (!entry || entry.active === false || !entry.lower) return;

        const old = pool.get(entry.lower);

        const row = {
 entry, sourceScore: clamp(sourceScore, 0, 1.5), source, focusScore: clamp(focusScore, 0, 1) 
}
;

        if (!old || row.sourceScore > old.sourceScore || row.focusScore > (old.focusScore ?? 0.5)) pool.set(entry.lower, row);

      
}
;


      const seeds = (analysis?.contentWords || []).slice(0, 12);

      const direct = this.processor.associationCandidates(seeds, this.options.candidateLimit, {
 includeSeeds: true 
}
);

      for (const row of direct) add(row.entry, row.weight, row.source);


      // Widen the lexical graph as search effort increases. This is derived from
      // Words.json associations and learned vocabulary state, not from a canned
      // subject-answer table. Two-hop/three-hop discovery gives open-ended
      // questions enough vocabulary to actually construct an explanation.
      if (explorationLevel > 0) {

        let frontier = direct.filter(row => row.weight >= 0.42).slice(0, 48).map(row => row.entry.lower);

        const depth = Math.min(3, 1 + Math.floor(explorationLevel / 2));

        const seen = new Set(seeds.map(x => String(x).toLowerCase()));

        for (let hop = 0;
 hop < depth && frontier.length;
 hop++) {

          const rows = this.processor.associationCandidates(frontier, this.options.candidateLimit * 2, {
 includeSeeds: false 
}
);

          const next = [];

          const discount = Math.max(0.24, 0.58 - hop * 0.12);

          for (const row of rows) {

            const key = row.entry.lower;

            if (!seen.has(key)) next.push(key);

            seen.add(key);

            add(row.entry, row.weight * discount, `lexical-hop-${hop + 2}`);

          
}

          frontier = next.slice(0, 64);

        
}

      
}

      for (const word of analysis?.contentWords || []) {

        const entry = this.processor.resolveSayableWord(word, {
 allowRootFallback: true 
}
);

        if (entry) add(entry, 0.72, 'prompt');

      
}

      for (const word of analysis?.responseScaffold?.words || []) {

        const entry = this.processor.resolveSayableWord(word, {
 allowRootFallback: true 
}
);

        if (entry) add(entry, 0.78, 'pos-scaffold');

      
}

      const keywordWeights = analysis?.trainingKnowledge?.keywordWeights || analysis?.responseScaffold?.keywordWeights || {};
      for (const word of analysis?.trainingKnowledge?.keywordTargets || analysis?.branchKeywords || analysis?.responseScaffold?.keywordTargets || []) {
        const entry = this.processor.resolveSayableWord(word, { allowRootFallback: true });
        if (!entry) continue;
        const weight = clamp(Number(keywordWeights[word]) || 0.75);
        add(entry, 0.70 + weight * 0.52, 'training-keyword-vector', 0.62 + weight * 0.36);
      }

      // ExplanationPlanner supplies structured evidence vocabulary from the
      // graph/correlation/self-model systems. This remains ordinary candidate
      // material: no sentence is prewritten and the normal four-branch decoder
      // still decides how the concepts are expressed.
      const explanationPriority = [
        ...(analysis?.explanationPlan?.priorityTerms || []),
        ...(analysis?.explanationPlan?.evidenceTerms || [])
      ];
      for (const word of explanationPriority.slice(0, 96)) {
        const entry = this.processor.resolveSayableWord(word, { allowRootFallback: true });
        if (entry) add(entry, 0.92, 'explanation-plan-evidence', 0.88);
      }

      const structuredPriority = [
        ...(analysis?.propositionGraph?.terms || []),
        ...(analysis?.mechanismGraph?.priorityTerms || [])
      ];
      for (const word of structuredPriority.slice(0, 112)) {
        const entry = this.processor.resolveSayableWord(word, { allowRootFallback: true });
        if (entry) add(entry, 0.86, analysis?.mechanismGraph?.active ? 'mechanism-graph-structure' : 'proposition-graph-structure', 0.84);
      }

      for (const word of ['a','an','the','is','are','was','were','can','could','will','would','may','might','to','of','in','with','for','by','through','using','via','because','and','while','as','it','this','that']) {

        const entry = this.processor.resolveSayableWord(word, {
 allowRootFallback: false 
}
);

        if (entry) add(entry, 0.12, 'function-seed');

      
}

      const role = analysis?.requestedSlot || analysis?.questionState?.requestedRole || 'content';

      const roleLexemes = role === 'explanation'
        ? ['because','reason','cause','by','through','using','with','process','method','result','effect','change','lead','from','therefore']
        : role === 'mechanism'
          ? ['by','through','using','with','enable','allow','increase','reduce','improve','faster','speed','performance','efficiency','processing','compute','throughput','parallel']
        : role === 'identity'
          ? ['a','an','the','data','object','system','process','concept','type','kind','file','folder','memory','information']
        : role === 'cause'
          ? ['because','through','from','result','cause','increase','reduce','change','effect']
          : role === 'location'
            ? ['in','on','at','inside','within','through','near']
            : role === 'time'
              ? ['during','before','after','when','while','until','later','earlier']
              : [];

      for (const word of roleLexemes) {

        const entry = this.processor.resolveSayableWord(word, {
 allowRootFallback: true 
}
);

        if (entry) add(entry, explorationLevel > 0 ? 0.34 : 0.20, 'language-role-bridge');

      
}

      return Array.from(pool.values());

    
}


    _focusedStaticPool(
      staticPool,
      analysis,
      limit = this.options.focusedStaticPoolLimit
    ) {

      const rows = Array.isArray(staticPool) ? staticPool : [];

      if (rows.length <= limit) return rows;


      const mandatory = new Set([
        ...(analysis?.contentWords || []),
        ...(analysis?.responseScaffold?.words || []),
        ...(analysis?.trainingKnowledge?.keywordTargets || analysis?.branchKeywords || []),
        'a','an','the','is','are','was','were',
        'can','could','will','would','may','might',
        'to','of','in','with','for','by','through',
        'using','via','because','and','while','as',
        'it','this','that'
      ].map(value => String(value || '').toLowerCase()));


      const kept = new Map();


      const add = row => {

        const key = row?.entry?.lower;

        if (!key) return;


        const quality =
          (Number(row?.sourceScore) || 0) * 0.62 +
          (Number(row?.focusScore) || 0.5) * 0.38;


        const old = kept.get(key);

        if (!old || quality > old.quality) {

          kept.set(key, {
 row, quality 
}
);

        
}

      
}
;


      for (const row of rows) {

        if (mandatory.has(row?.entry?.lower)) add(row);

      
}


      const ranked = rows
        .map(row => ({

          row,
          quality:
            (Number(row?.sourceScore) || 0) * 0.62 +
            (Number(row?.focusScore) || 0.5) * 0.38
        
}
))
        .sort((a, b) => b.quality - a.quality);


      for (const item of ranked) {

        if (kept.size >= limit) break;

        add(item.row);

      
}


      return Array.from(kept.values(), item => item.row);

    
}


    _isTransitionEntry(entry, activePOS = '') {

      const lower = String(entry?.lower || entry?.surface || '').toLowerCase();
      const pos = String(activePOS || entry?.pos || entry?.activePOS || '');

      return TRANSITION_POS.has(pos) || TRANSITION_WORD_SET.has(lower);

    }


    _isTerminalEntry(entry, activePOS = '') {

      const lower = String(entry?.lower || entry?.surface || '').toLowerCase();
      const pos = String(activePOS || entry?.pos || entry?.activePOS || '');

      if (pos !== 'Punct' || !TERMINAL_WORD_SET.has(lower)) return false;

      const profile = this.processor?._punctuationProfile?.(entry);
      return !profile || profile.className === 'terminal';

    }


    _hasOpenDependency(state) {

      return Boolean(
        state?.openDeterminer ||
        state?.openPreposition ||
        state?.openModal ||
        state?.openCopulaComplement ||
        state?.continuationRequired
      );

    }


    _dependencyCompletionPOSAllowed(state, pos) {

      if (!this._hasOpenDependency(state)) return true;

      if (state?.openPreposition) {
        return ['Det','Adj','Noun','ProperNoun','Pronoun','Num'].includes(pos);
      }

      if (state?.openDeterminer) {
        return ['Adj','Noun','ProperNoun','Pronoun','Num'].includes(pos);
      }

      if (state?.openModal) {
        return ['Adv','Verb'].includes(pos);
      }

      if (state?.openCopulaComplement) {
        return ['Adv','Det','Adj','Noun','ProperNoun','Pronoun','Num','Prep','Verb'].includes(pos);
      }

      return pos !== 'Punct';

    }


    _contentRun(emitted) {

      const words = Array.isArray(emitted) ? emitted : [];
      let run = 0;

      for (let i = words.length - 1; i >= 0 && run < 8; i--) {

        const entry = this.processor.resolveSayableWord?.(words[i], { allowRootFallback: true });
        if (!entry || this._isTransitionEntry(entry, entry.pos) || !entry.contentWord) break;
        run++;

      }

      return run;

    }


    _candidatePool(analysis, state, emitted, staticPool, options = {
}
) {

      const pool = new Map();

      const add = (entry, sourceScore = 0, source = 'unknown', focusScore = 0.5) => {

        if (!entry || entry.active === false || !entry.lower) return;

        const old = pool.get(entry.lower);

        const row = {
 entry, sourceScore: clamp(sourceScore, 0, 1.5), source, focusScore: clamp(focusScore, 0, 1) 
}
;

        if (!old || row.sourceScore > old.sourceScore || row.focusScore > (old.focusScore ?? 0.5)) pool.set(entry.lower, row);

      
}
;


      for (const row of staticPool || []) add(row.entry, row.sourceScore, row.source, row.focusScore ?? 0.5);

      const previousWord = emitted[emitted.length - 1] || '';

      const dynamicRSLLimit =
        options.focusedFastPath
          ? this.options.focusedDynamicRSLLimit
          : 40;


      const dynamicCorrelationLimit =
        options.focusedFastPath
          ? this.options.focusedCorrelationLimit
          : 48;


      for (const learned of this.rsl?.learnedCandidates?.(previousWord, 'vilotni2', dynamicRSLLimit) || []) {

        const entry = this.processor.resolveSayableWord(learned.word, {
 allowRootFallback: true 
}
);

        if (entry) add(entry, learned.score, 'rsl-token');

      
}

      for (const learned of this.rsl?.contextCandidates?.(previousWord, state.lastPOS, dynamicRSLLimit) || []) {

        const entry = this.processor.resolveSayableWord(learned.word, {
 allowRootFallback: true 
}
);

        if (entry) add(entry, learned.score, 'rsl-pos-context');

      
}

      for (const learned of this.rsl?.contextWindowCandidates?.(
        state.previousWord, state.previousPOS, previousWord, state.lastPOS, dynamicRSLLimit
      ) || []) {

        const entry = this.processor.resolveSayableWord(learned.word, {
 allowRootFallback: true 
}
);

        if (entry) add(entry, learned.score, 'rsl-pos-window');

      
}

      const correlationSeeds = (analysis?.contentWords || []).slice(0, 12).concat(emitted.slice(-3));

      const conversationState = analysis?.conversationState || null;
      if (conversationState && Number(conversationState.responseFreedom || 0) > 0.20) {
        for (const hint of (conversationState.lexicalHints || []).slice(0, 8)) {
          const entry = this.processor.resolveSayableWord?.(hint.word, { allowRootFallback: true });
          if (!entry) continue;
          const weight = clamp(Number(hint.weight) || 0);
          add(entry, 0.16 + weight * 0.22, `conversation-${hint.source || 'soft'}`, 0.48 + weight * 0.20);
        }
      }

      for (const learned of this.rsl?.correlationCandidates?.(correlationSeeds, dynamicCorrelationLimit) || []) {

        const entry = this.processor.resolveSayableWord(learned.word, {
 allowRootFallback: true 
}
);

        if (entry) add(entry, learned.score, 'rsl-correlation');

      
}

      // Universal grammar rescue: after multiple consecutive content words,
      // expose ordinary relational/function words as candidates. They still
      // have to win the learned transition/context scoring below. This prevents
      // high-semantic-weight keyword stacks from starving grammar tokens out of
      // the search pool while remaining completely topic-independent.
      if (this._contentRun(emitted) >= this.options.syntaxBridgeAfterContentRun) {

        for (const word of BRIDGE_WORDS) {

          const entry = this.processor.resolveSayableWord?.(word, { allowRootFallback: true });
          if (entry) add(entry, this.options.syntaxTransitionCandidateScore, 'grammar-transition', 0.72);

        }

      }

      // Terminal punctuation is a real candidate, not a post-hoc patch. It is
      // only exposed when the clause has a subject/predicate and no unresolved
      // grammatical dependency. The scoring loop still decides whether the
      // sentence is semantically ready to collapse.
      if (state?.hasSubject && state?.hasPredicate && !this._hasOpenDependency(state)) {
        const terminal = this.processor.resolveSayableWord?.('.', { allowRootFallback: true });
        if (terminal) add(terminal, this.options.terminalCandidateScore, 'grammar-terminal', 0.82);
      }

      return Array.from(pool.values());

    
}


    _generate(promptText, analysis, options = {
}
) {

      const state = this.processor.createClauseState();

      const emitted = [];

      const trace = [];

      const effortPass = Math.max(0, Number(options.effortPass) || 0);

      const explorationLevel = Math.max(0, Number(options.explorationLevel) || 0);

      const selfLearningRound = Math.max(0, Number(options.selfLearningRound) || effortPass);

      const recoveryMode = Boolean(options.recoveryMode);

      const previousCandidateWords = (this.processor?.tokenize?.(String(options.previousCandidateText || ''), 192) || [])
        .filter(token => token.pos !== 'Punct')
        .map(token => String(token.lower || '').toLowerCase());

      const scaffoldWords = Array.isArray(analysis?.responseScaffold?.words) ? analysis.responseScaffold.words.slice() : [];

      const scaffoldPOS = Array.isArray(analysis?.responseScaffold?.pos) ? analysis.responseScaffold.pos.slice(0, scaffoldWords.length) : [];

      // A recovered scaffold is a constraint only when it can legally begin a
      // clause. Offline/learning branches used to create single-token scaffolds
      // such as ["by"], which made scaffoldCoreLock reject every legal first
      // word forever. Strip only structurally impossible leading bridge tokens;
      // their relation is still preserved by explanation/mechanism metadata.
      const unsafeLeadingBridge = new Set(['by','through','using','with','of','from','to','for']);
      while (scaffoldWords.length && unsafeLeadingBridge.has(String(scaffoldWords[0] || '').toLowerCase())) {
        scaffoldWords.shift();
        if (scaffoldPOS.length) scaffoldPOS.shift();
      }

      const keywordTargets = Array.from(new Set(
        analysis?.trainingKnowledge?.keywordTargets ||
        analysis?.branchKeywords ||
        analysis?.responseScaffold?.keywordTargets ||
        []
      )).slice(0, 96);

      const keywordWeights = analysis?.trainingKnowledge?.keywordWeights || analysis?.responseScaffold?.keywordWeights || {};
      const keywordTargetSet = new Set(keywordTargets.map(word => String(word || '').toLowerCase()));
      const keywordAssociationCache = new Map();
      const keywordExpansion = keywordTargets.length
        ? Math.min(12, 3 + Math.ceil(Math.sqrt(keywordTargets.length) * 2))
        : 0;

      const scaffoldNeedsExpansion = Boolean(
        scaffoldWords.length &&
        (['mechanism','causal','question','list'].includes(analysis?.intent) || keywordTargets.length)
      );

      const scaffoldSpan = Math.max(scaffoldWords.length, scaffoldPOS.length);

      const hasExtensionPOS = Boolean(analysis?.responseScaffold?.extensionPOS?.length);
      const requestedRole = String(
        analysis?.requestedSlot ||
        analysis?.questionState?.requestedRole ||
        analysis?.responseScaffold?.requestedRole ||
        'content'
      ).toLowerCase();

      // A grammatical extension scaffold already describes the intended answer
      // shape. Adding the full semantic keyword expansion on top of that was
      // forcing the decoder to keep talking after the sentence was complete,
      // which created topical association chains. Keep a small amount of room
      // for content, but do not turn every related keyword into another slot.
      const structuredKeywordExpansion = hasExtensionPOS
        ? Math.min(3, Math.ceil(keywordExpansion * 0.25))
        : keywordExpansion;

      let target = scaffoldSpan
        ? Math.max(
            4,
            Math.min(
              this.options.targetMax,
              scaffoldSpan +
                (scaffoldNeedsExpansion && !hasExtensionPOS ? 1 + Math.min(3, explorationLevel) : 0) +
                structuredKeywordExpansion
            )
          )
        : Math.max(
            this.options.targetMin,
            Math.min(
              this.options.targetMax,
              9 + Math.round(Math.sqrt(Math.max(1, analysis?.contentWords?.length || 1)) * 3) +
                Math.min(6, explorationLevel * 2) + structuredKeywordExpansion
            )
          );

      // Short-answer roles get a structural cap. This is role-level grammar,
      // not topic knowledge: identity/truth/location/time questions should not
      // drift through a long chain of merely associated concepts.
      if (hasExtensionPOS && requestedRole === 'identity') {
        target = Math.max(4, scaffoldSpan);
      } else if (hasExtensionPOS && ['truth','location','time'].includes(requestedRole)) {
        target = Math.min(target, Math.max(6, scaffoldSpan + 1));
      }

      const focusedScheduling = options.focusedFastPath === true;

      const focusedFastPath = focusedScheduling && !this.options.preserveFullModelBreadth;

      const deadlineAt = Number.isFinite(Number(options.deadlineAt))
        ? Number(options.deadlineAt)
        : Infinity;


      const effectiveTarget = focusedFastPath
        ? Math.max(
            this.options.focusedTargetMin,
            Math.min(target, this.options.focusedTargetMax)
          )
        : target;


      const usedContent = new Set();

      let contentRun = 0;
      let maxContentRun = 0;
      let syntaxTransitionsSelected = 0;
      let syntaxStackPenaltyTotal = 0;
      let clausePrepositionDepth = 0;
      const clausePrepositionCounts = new Map();
      let generatedContentAfterScaffold = 0;
      let terminalSelected = false;
      let terminalCompletionSteps = 0;

      const rawStaticPool =
        options.staticPool ||
        this._staticCandidatePool(analysis, explorationLevel);


      const staticPool = focusedFastPath
        ? this._focusedStaticPool(rawStaticPool, analysis)
        : rawStaticPool;


      let deadlineTruncated = false;

      const variantIndex = Math.max(0, Number(options.variantIndex) || 0);

      const predicateWord = String(analysis?.responseScaffold?.predicate || analysis?.clauseState?.predicateToken?.lower || '').toLowerCase();

      const subjectWords = analysis?.responseScaffold?.subject || [];

      const semanticAnchor = String((analysis?.requestedSlot === 'identity' ? subjectWords[subjectWords.length - 1] : predicateWord) || predicateWord || '').toLowerCase();

      const variantPivot = variantIndex > 0 ? ((variantIndex - 1) % Math.max(1, effectiveTarget)) : -1;

      // Keep the full candidate/model breadth. The only extra room is reserved
      // for completing an already-open grammar dependency, never for extending
      // the semantic search. In normal cases terminal punctuation wins before
      // this allowance is reached, reducing total decoder work.
      const hardSearchTarget = effectiveTarget + this.options.terminalCompletionAllowance;


      for (let step = 0;
 step < hardSearchTarget;
 step++) {

        if (
          Number.isFinite(deadlineAt) &&
          emitted.length >= Math.min(4, this.options.focusedTargetMin) &&
          performance.now() >= deadlineAt - this.options.deadlineFinishReserveMs
        ) {

          deadlineTruncated = true;

          // A deadline may stop an already-complete clause, but it may not
          // collapse a branch on a dangling determiner/preposition/modal. Give
          // the existing branch only the tiny structural completion allowance.
          if (!this._hasOpenDependency(state) || step >= hardSearchTarget - 1) break;

        
}

        // The allowance above is completion-only. Once the nominal target is
        // reached, do not spend extra semantic-search steps unless a grammar
        // dependency is still open.
        if (step >= effectiveTarget && !this._hasOpenDependency(state)) break;


        const pool = this._candidatePool(
          analysis,
          state,
          emitted,
          staticPool,
          options
        );

        const previousWord = emitted[emitted.length - 1] || '';

        const correlationSeeds = (analysis?.contentWords || []).slice(0, 10).concat(keywordTargets.slice(0, 10), emitted.slice(-3));

        this.lexical.wordMatrix?.prepareContext?.(correlationSeeds);

        let best = null;

        let second = null;

        let third = null;

        let fourth = null;

        let poolIndex = 0;


        for (const row of pool) {

          if (
            Number.isFinite(deadlineAt) &&
            (poolIndex % this.options.deadlineCheckStride) === 0 &&
            performance.now() >= deadlineAt - this.options.deadlineFinishReserveMs
          ) {

            deadlineTruncated = true;

            if (!this._hasOpenDependency(state)) break;

          
}


          poolIndex++;

          const lexicalEntry = row.entry;

          if (lexicalEntry.contentWord && usedContent.has(lexicalEntry.lower)) continue;

          const expectedPOS = scaffoldPOS[step] || '';
          const expectedWord = scaffoldWords[step] || '';

          const learnedPOS = this.rsl?.resolvePOS?.(lexicalEntry, state) || {

            pos: lexicalEntry.pos, certainty: 0.5, score: 0.5, distribution: [{
 pos: lexicalEntry.pos, score: 0.5 
}
]
          
}
;

          const canOccupyExpected = Boolean(expectedPOS && (lexicalEntry.possiblePOS || [lexicalEntry.pos]).includes(expectedPOS));

          const posResolution = canOccupyExpected
            ? {
 ...learnedPOS, pos: expectedPOS, certainty: Math.max(0.78, learnedPOS.certainty || 0), score: Math.max(0.78, learnedPOS.score || 0) 
}

            : learnedPOS;

          const entry = this.processor.materializePOS(lexicalEntry, posResolution.pos) || lexicalEntry;

          // The declarative core is recovered directly from the user's prompt
          // (for example "What is an AI?" -> "an AI is"). Do not let a
          // semantically related neighbor replace those core words. The open
          // generation begins after the scaffold, where keyword search belongs.
          if (
            this.options.scaffoldCoreLock &&
            expectedWord &&
            step < scaffoldWords.length &&
            String(entry.lower || '').toLowerCase() !== String(expectedWord).toLowerCase()
          ) continue;

          // Once the recovered question core is complete, content slots must be
          // filled by content words. This prevents auxiliaries/modals such as
          // "will" from being recycled as fake mechanism content merely because
          // dynamic POS is still young.
          const isTransitionCandidate = this._isTransitionEntry(entry, posResolution.pos);
          const isTerminalCandidate = this._isTerminalEntry(entry, posResolution.pos);
          const bridgeNeeded = contentRun >= this.options.syntaxBridgeAfterContentRun;
          const dependencyCompletionMode = Boolean(
            this._hasOpenDependency(state) &&
            (
              step >= effectiveTarget ||
              (
                Number.isFinite(deadlineAt) &&
                performance.now() >= deadlineAt - this.options.deadlineFinishReserveMs
              )
            )
          );

          if (dependencyCompletionMode && !this._dependencyCompletionPOSAllowed(state, posResolution.pos)) {
            continue;
          }

          const shortAnswerRole = ['identity','truth','location','time'].includes(requestedRole);
          const terminalContentFloor = shortAnswerRole ? 1 : 2;
          const terminalSemanticReady = Boolean(
            state.hasSubject &&
            state.hasPredicate &&
            !this._hasOpenDependency(state) &&
            generatedContentAfterScaffold >= terminalContentFloor
          );

          if (isTerminalCandidate && !terminalSemanticReady) continue;

          if (
            step >= scaffoldWords.length &&
            ['Verb','Noun','ProperNoun','Adj','Adv','Num'].includes(expectedPOS) &&
            !entry.contentWord &&
            !(bridgeNeeded && isTransitionCandidate) &&
            !(isTerminalCandidate && terminalSemanticReady)
          ) continue;

          const legality = this.processor.evaluateCandidate(entry, state, {
 activePOS: posResolution.pos 
}
);

          if (!legality.legal) continue;


          const neural = clamp(this.rsl?.activationForWord?.(entry.surface) ?? 0.5);

          const tokenTransition = clamp(this.rsl?.transitionScore?.(previousWord, entry.surface, 'vilotni2') ?? 0);

          const posTransition = clamp(this.rsl?.posTransitionScore?.(state.lastPOS, posResolution.pos) ?? 0);

          const posTrigram = clamp(this.rsl?.posTrigramScore?.(state.previousPOS, state.lastPOS, posResolution.pos) ?? 0);

          const contextTransition = clamp(this.rsl?.contextTransitionScore?.(
            previousWord, state.lastPOS, entry.surface, posResolution.pos
          ) ?? 0);

          const contextWindow = clamp(this.rsl?.contextWindowScore?.(
            state.previousWord, state.previousPOS, previousWord, state.lastPOS, entry.surface, posResolution.pos
          ) ?? 0);

          const wordPOS = clamp(this.rsl?.wordPOSScore?.(entry.lower, posResolution.pos) ?? 0);

          const morphology = clamp(this.rsl?.morphologyScore?.(entry.root, entry.surface, posResolution.pos) ?? 0);

          const lexicalDiscovery = clamp(row.sourceScore, 0, 1);

          const focusSupport = clamp(row.focusScore ?? 0.5);

          let keywordAlignment = 0;
          const lowerWord = String(entry.lower || '').toLowerCase();
          if (keywordTargetSet.has(lowerWord)) {
            keywordAlignment = clamp(Number(keywordWeights[lowerWord]) || 1);
          } else if (keywordTargets.length) {
            const cachedKeyword = keywordAssociationCache.get(lowerWord);
            if (cachedKeyword != null) {
              keywordAlignment = cachedKeyword;
            } else {
              let bestKeywordAssociation = 0;
              for (const keyword of keywordTargets.slice(0, 16)) {
                const association = clamp(this.processor.associationScore?.(keyword, lowerWord, { bidirectional: true }) ?? 0);
                const weightedAssociation = association * clamp(Number(keywordWeights[keyword]) || 0.72);
                if (weightedAssociation > bestKeywordAssociation) bestKeywordAssociation = weightedAssociation;
              }
              keywordAlignment = clamp(bestKeywordAssociation);
              keywordAssociationCache.set(lowerWord, keywordAlignment);
            }
          }

          const dynamicPOS = clamp(posResolution.score ?? 0);

          const predicateSupport = step >= scaffoldWords.length && semanticAnchor
            ? clamp(this.processor.associationScore?.(semanticAnchor, entry.lower, {
 bidirectional: true 
}
) ?? 0)
            : 0;

          const lexicalPairSupport = previousWord
            ? clamp(this.processor.associationScore?.(previousWord, entry.lower, {
 bidirectional: true 
}
) ?? 0)
            : 0;

          const correlationSupport = clamp(this.rsl?.correlationSupport?.(correlationSeeds, entry.lower) ?? 0);

          const wordStateSupport = clamp(this.lexical.wordState?.support?.(entry.lower, posResolution.pos) ?? 0.5);

          const phraseStateSupport = clamp(this.lexical.phraseState?.candidateSupport?.(emitted, entry.lower) ?? 0.5);

          const wordMatrixSupport = clamp(this.lexical.wordMatrix?.supportPrepared?.(entry.lower) ?? this.lexical.wordMatrix?.support?.(correlationSeeds, entry.lower) ?? 0.5);

          const expectedEntry = expectedWord
            ? this.processor.resolveSayableWord(expectedWord, {
 allowRootFallback: true 
}
)
            : null;

          const roleFit = expectedWord && entry.lower === expectedWord
            ? 1
            : (expectedEntry?.root && entry.root === expectedEntry.root ? 0.04 : 0);

          const posRoleFit = expectedPOS ? (posResolution.pos === expectedPOS ? 1 : 0) : 0.5;

          const languageSupport = clamp(this.lexical.languageState?.candidateSupport?.(
            analysis, state, entry, posResolution.pos, step, emitted
          ) ?? 0.5);


          // Function words carry little standalone semantic weight, so score
          // their value as transitions between nearby states. This lets learned
          // grammar/context support compete directly with keyword weight.
          const transitionSupport = isTransitionCandidate
            ? clamp(
                tokenTransition * 0.20 +
                posTransition * 0.18 +
                posTrigram * 0.13 +
                contextTransition * 0.18 +
                contextWindow * 0.13 +
                lexicalPairSupport * 0.08 +
                languageSupport * 0.10
              )
            : clamp(
                tokenTransition * 0.10 +
                contextTransition * 0.08 +
                contextWindow * 0.06
              );

          const transitionNeed = bridgeNeeded
            ? clamp((contentRun - this.options.syntaxBridgeAfterContentRun + 1) / 2)
            : 0;

          const transitionBoost = isTransitionCandidate
            ? transitionNeed * this.options.syntaxTransitionBoost * Math.max(0.88, Math.min(1.28, Number(this.runtimeOptions?.selfImprovementProfile?.transitionScale) || 1)) * (0.65 + transitionSupport * 0.75)
            : 0;

          const recentBridgeWords = emitted
            .slice(-7)
            .map(word => String(word || '').toLowerCase())
            .filter(word => BRIDGE_WORD_SET.has(word));
          const sameBridgeRecent = recentBridgeWords.filter(word => word === lowerWord).length;
          const prepLike = String(posResolution.pos || entry.pos || '') === 'Prep';
          const recentPrepCount = emitted
            .slice(-7)
            .map(word => this.processor.resolveSayableWord?.(word, { allowRootFallback: true }))
            .filter(item => String(item?.pos || '') === 'Prep').length;
          const samePrepDepth = prepLike
            ? (clausePrepositionCounts.get(lowerWord) || 0)
            : 0;

          // More than two identical preposition hops in one clause are not a
          // useful semantic expansion. Reject that branch position universally
          // instead of allowing chains such as "of ... of ... of" to consume
          // another full search step.
          if (prepLike && samePrepDepth >= this.options.maxSamePrepositionHops) continue;

          const prepositionDepthMultiplier = prepLike
            ? Math.pow(this.options.prepositionDepthDecay, clausePrepositionDepth)
            : 1;

          const bridgeHistoryPenalty = isTransitionCandidate
            ? Math.min(
                0.34,
                sameBridgeRecent * this.options.repeatedBridgePenalty +
                (prepLike && recentPrepCount >= 2
                  ? (recentPrepCount - 1) * this.options.bridgeDensityPenalty
                  : 0)
              )
            : 0;

          const terminalProgress = clamp((step + 1) / Math.max(1, effectiveTarget));
          const terminalCompletionBoost = isTerminalCandidate
            ? this.options.terminalCompletionBoost * Math.max(0.92, Math.min(1.16, Number(this.runtimeOptions?.selfImprovementProfile?.terminalScale) || 1)) *
              clamp(
                0.18 +
                terminalProgress * 0.72 +
                Math.min(0.30, clausePrepositionDepth * 0.10),
                0,
                1.25
              )
            : 0;

          // Penalize continued raw content stacking only when a grammatical
          // relation is already due. The first two or three content words can
          // still form normal compounds; progressively longer stacks become
          // increasingly expensive instead of automatically winning on keyword
          // support alone.
          const contentStackPenalty =
            (entry.contentWord &&
            !isTransitionCandidate &&
            bridgeNeeded
              ? Math.min(
                  0.32,
                  this.options.syntaxStackPenaltyBase *
                    (1 + Math.max(0, contentRun - this.options.syntaxBridgeAfterContentRun) * 1.55)
                )
              : 0) * Math.max(0.88, Math.min(1.28, Number(this.runtimeOptions?.selfImprovementProfile?.structurePenaltyScale) || 1));


          // Every structural preference here is learned by RSL. POS_Type only
          // supplies the set of roles a word may occupy; it does not prescribe
          // a fixed hand-written sequence.
          const rawScore =
            neural * 0.05 +
            tokenTransition * 0.08 +
            dynamicPOS * 0.08 +
            posTrigram * 0.07 +
            contextTransition * 0.06 +
            contextWindow * 0.06 +
            wordPOS * 0.03 +
            morphology * 0.02 +
            lexicalDiscovery * 0.03 +
            predicateSupport * 0.05 +
            lexicalPairSupport * 0.12 +
            posRoleFit * 0.05 +
            correlationSupport * 0.07 +
            roleFit * 0.09 +
            wordStateSupport * 0.06 +
            phraseStateSupport * 0.04 +
            wordMatrixSupport * 0.03 +
            languageSupport * 0.09 +
            focusSupport * 0.05 +
            keywordAlignment * 0.11 +
            transitionSupport * 0.08 +
            transitionBoost +
            terminalCompletionBoost;


          // A failed low-confidence response must be genuinely remade, not
          // regenerated word-for-word. This is an exploration penalty only; it
          // contains no linguistic or factual knowledge.
          const previousAtStep = previousCandidateWords[step] || '';

          const repeatPenalty = previousAtStep && previousAtStep === entry.lower && roleFit < 0.95
            ? Math.min(0.20, 0.045 + selfLearningRound * 0.007)
            : 0;

          const score =
            rawScore * prepositionDepthMultiplier -
            repeatPenalty -
            contentStackPenalty -
            bridgeHistoryPenalty;


          const candidate = {

            entry,
            score,
            rawScore,
            neural,
            tokenTransition,
            posTransition,
            posTrigram,
            contextTransition,
            contextWindow,
            wordPOS,
            dynamicPOS,
            posCertainty: clamp(posResolution.certainty ?? 0.5),
            posDistribution: posResolution.distribution?.slice?.(0, 6) || [],
            morphology,
            lexicalDiscovery,
            focusSupport,
            keywordAlignment,
            predicateSupport,
            lexicalPairSupport,
            correlationSupport,
            wordStateSupport,
            phraseStateSupport,
            wordMatrixSupport,
            repeatPenalty,
            roleFit,
            posRoleFit,
            languageSupport,
            transitionSupport,
            transitionNeed,
            transitionBoost,
            contentStackPenalty,
            bridgeHistoryPenalty,
            prepositionDepthMultiplier,
            samePrepDepth,
            terminalCompletionBoost,
            isTransitionCandidate,
            isTerminalCandidate,
            contentRunBefore: contentRun,
            source: row.source
          
}
;

          if (!best || score > best.score) {

            fourth = third;

            third = second;

            second = best;

            best = candidate;

          
}
 else if (!second || score > second.score) {

            fourth = third;

            third = second;

            second = candidate;

          
}
 else if (!third || score > third.score) {

            fourth = third;

            third = candidate;

          
}
 else if (!fourth || score > fourth.score) {

            fourth = candidate;

          
}

        
}


        if (!best) break;

        // Batched search diversifies trajectories at more than one deterministic
        // pivot as exploration grows. This changes the sentence plan without
        // mutating RSL or injecting factual knowledge.
        const pivotA = variantPivot;

        const pivotB = variantIndex > 2 ? ((variantIndex * 3 + explorationLevel) % Math.max(1, effectiveTarget)) : -1;

        const pivotC = recoveryMode && variantIndex > 4
          ? ((variantIndex * 5 + explorationLevel * 2 + 1) % Math.max(1, effectiveTarget))
          : -1;

        const pivotD = recoveryMode && variantIndex > 8
          ? ((variantIndex * 7 + explorationLevel + 3) % Math.max(1, effectiveTarget))
          : -1;

        let selected = best;

        if (second && (step === pivotA || step === pivotB || step === pivotD)) selected = second;

        if (third && step === pivotC) selected = third;

        const runnerUp = selected === best ? second : best;

        this.processor.acceptCandidate(selected.entry, state, {
 activePOS: selected.entry.pos 
}
);

        emitted.push(selected.entry.surface);

        if (selected.entry.contentWord) {
          usedContent.add(selected.entry.lower);
          if (step >= scaffoldWords.length) generatedContentAfterScaffold++;
        }

        if (String(selected.entry.pos || '') === 'Prep') {
          clausePrepositionDepth++;
          const prepKey = String(selected.entry.lower || selected.entry.surface || '').toLowerCase();
          clausePrepositionCounts.set(prepKey, (clausePrepositionCounts.get(prepKey) || 0) + 1);
        }

        if (selected.isTerminalCandidate) {
          terminalSelected = true;
          clausePrepositionDepth = 0;
          clausePrepositionCounts.clear();
        }

        if (selected.isTransitionCandidate || !selected.entry.contentWord) {
          if (selected.isTransitionCandidate && !selected.isTerminalCandidate && contentRun >= this.options.syntaxBridgeAfterContentRun) {
            syntaxTransitionsSelected++;
          }
          contentRun = 0;
        } else {
          contentRun++;
          if (contentRun > maxContentRun) maxContentRun = contentRun;
        }

        if (step >= effectiveTarget) terminalCompletionSteps++;

        syntaxStackPenaltyTotal += Number(selected.contentStackPenalty) || 0;

        trace.push({

          step,
          word: selected.entry.surface,
          pos: selected.entry.pos,
          total: selected.score,
          margin: Math.max(0, selected.score - (runnerUp?.score ?? 0)),
          liveNeural: selected.neural,
          tokenTransition: selected.tokenTransition,
          posTransition: selected.posTransition,
          posTrigram: selected.posTrigram,
          contextTransition: selected.contextTransition,
          contextWindow: selected.contextWindow,
          wordPOS: selected.wordPOS,
          dynamicPOS: selected.dynamicPOS,
          posCertainty: selected.posCertainty,
          posDistribution: selected.posDistribution,
          morphology: selected.morphology,
          lexicalDiscovery: selected.lexicalDiscovery,
          focusSupport: selected.focusSupport,
          keywordAlignment: selected.keywordAlignment,
          predicateSupport: selected.predicateSupport,
          lexicalPairSupport: selected.lexicalPairSupport,
          correlationSupport: selected.correlationSupport,
          wordStateSupport: selected.wordStateSupport,
          phraseStateSupport: selected.phraseStateSupport,
          wordMatrixSupport: selected.wordMatrixSupport,
          repeatPenalty: selected.repeatPenalty,
          roleFit: selected.roleFit,
          posRoleFit: selected.posRoleFit,
          languageSupport: selected.languageSupport,
          transitionSupport: selected.transitionSupport,
          transitionNeed: selected.transitionNeed,
          transitionBoost: selected.transitionBoost,
          contentStackPenalty: selected.contentStackPenalty,
          bridgeHistoryPenalty: selected.bridgeHistoryPenalty,
          prepositionDepthMultiplier: selected.prepositionDepthMultiplier,
          samePrepDepth: selected.samePrepDepth,
          terminalCompletionBoost: selected.terminalCompletionBoost,
          isTransitionCandidate: selected.isTransitionCandidate,
          isTerminalCandidate: selected.isTerminalCandidate,
          contentRunBefore: selected.contentRunBefore,
          contentRunAfter: contentRun,
          source: selected.source,
          batchVariant: variantIndex,

          // VilotNI 2.5 can reuse the already-scored local distribution as a
          // four-path superposition frame. This records alternatives only; it
          // does not alter V2's selected trajectory or run another decode.
          interferenceAlternatives: [best, second, third, fourth]
            .filter(Boolean)
            .map((alt, rank) => ({

              rank,
              word: alt.entry.surface,
              lower: alt.entry.lower,
              pos: alt.entry.pos,
              total: alt.score,
              rawScore: alt.rawScore,
              marginFromBest: Math.max(0, (best?.score ?? alt.score) - alt.score),
              posCertainty: alt.posCertainty,
              dynamicPOS: alt.dynamicPOS,
              roleFit: alt.roleFit,
              posRoleFit: alt.posRoleFit,
              focusSupport: alt.focusSupport,
              keywordAlignment: alt.keywordAlignment,
              lexicalDiscovery: alt.lexicalDiscovery,
              languageSupport: alt.languageSupport,
              transitionSupport: alt.transitionSupport,
              transitionNeed: alt.transitionNeed,
              transitionBoost: alt.transitionBoost,
              contentStackPenalty: alt.contentStackPenalty,
              bridgeHistoryPenalty: alt.bridgeHistoryPenalty,
              prepositionDepthMultiplier: alt.prepositionDepthMultiplier,
              samePrepDepth: alt.samePrepDepth,
              terminalCompletionBoost: alt.terminalCompletionBoost,
              isTransitionCandidate: alt.isTransitionCandidate,
              isTerminalCandidate: alt.isTerminalCandidate,
              contentRunBefore: alt.contentRunBefore,
              source: alt.source
            
}
))
        
}
);


        // Terminal punctuation is the normal successful collapse condition.
        // This avoids paying for extra candidate-search steps after the clause is
        // already complete. Never use a low-score early exit while a dependency
        // is still open, which is what produced endings such as "by.".
        if (selected.isTerminalCandidate) break;

        if (
          step >= this.options.targetMin - 1 &&
          state.hasSubject &&
          state.hasPredicate &&
          !this._hasOpenDependency(state) &&
          selected.score < 0.30
        ) break;

      
}


      // Final surface realization applies only generic lexical and punctuation
      // grammar. It can repair determiners/articles, realize sentence terminals,
      // and place commas at syntax-supported clause boundaries without injecting
      // subject facts or canned response text.
      const grammarRepair = this.processor.repairGeneratedSurface?.(emitted, analysis)
        || this.processor.repairGeneratedWords?.(emitted)
        || {
 words: emitted, repairs: [], text: '' 
}
;

      if (grammarRepair.words !== emitted) {

        emitted.splice(0, emitted.length, ...(grammarRepair.words || emitted));

      
}

      let text = String(grammarRepair.text || emitted.join(' ').replace(/\s+([.,!?;:])/g, '$1')).trim();

      text = this._capitalize(text);

      if (text && !/[.!?]$/.test(text)) text += this.processor._looksInterrogativeWords?.(emitted) ? '?' : '.';

      if (!text) {

        const seed = (analysis?.contentWords || []).slice(0, 4).join(' ');

        const realized = this.processor.realizePhrase(seed || 'response', {

          maxWords: 12,
          allowUnknown: false,
          allowRootFallback: true
        
}
);

        text = this._capitalize(realized.text || 'Response') + '.';

      
}


      let scaffoldFit = 0;

      if (scaffoldWords.length) {

        let matched = 0;

        for (let i = 0;
 i < Math.min(scaffoldWords.length, emitted.length);
 i++) {

          if (String(emitted[i] || '').toLowerCase() === scaffoldWords[i]) matched++;

          else {

            const expected = this.processor.lookup(scaffoldWords[i]);

            const actual = this.processor.lookup(emitted[i]);

            if (expected?.root && actual?.root === expected.root) matched += 0.55;

          
}

        
}

        scaffoldFit = clamp(matched / scaffoldWords.length);

      
}

      const conversationAlignment = this._conversationAlignment(text, analysis);
      const responseFreedom = clamp(analysis?.conversationState?.responseFreedom || 0);
      const conversationalScaffold = !scaffoldWords.length
        ? clamp(0.42 + conversationAlignment * 0.48)
        : scaffoldFit;

      return {

        text,
        contractFit: scaffoldWords.length
          ? clamp(0.40 + scaffoldFit * 0.58)
          : (state.hasSubject && state.hasPredicate
              ? clamp(0.72 + conversationAlignment * responseFreedom * 0.18)
              : clamp(0.38 + conversationAlignment * responseFreedom * 0.12)),
        scaffoldFit: conversationalScaffold,
        conversationAlignment,
        trace,
        lexicalCoverage: emitted.length ? 1 : 0,
        grammarRepairs: grammarRepair?.repairs || [],
        syntaxFlow: {
          transitionCandidatesEnabled: true,
          transitionsSelected: syntaxTransitionsSelected,
          maxContentRun,
          contentStackPenaltyTotal: Number(syntaxStackPenaltyTotal.toFixed(4)),
          terminalSelected,
          terminalCompletionSteps,
          generatedContentAfterScaffold,
          finalOpenDependency: this._hasOpenDependency(state),
          prepositionDepthDecay: this.options.prepositionDepthDecay
        },
        provenance: 'rsl-pos-context',
        deadlineTruncated,
        focusedFastPath,
        effectiveTarget,
        staticPoolSize: staticPool?.length || 0
      
}
;

    
}


    _neuralAgreement(text) {

      const tokens = this.processor.tokenize(text, 64).filter(token => token.pos !== 'Punct');

      if (!tokens.length) return 0.5;

      let sum = 0;

      let count = 0;

      for (const token of tokens) {

        const value = this.rsl?.activationForWord?.(token.surface);

        if (Number.isFinite(value)) {
 sum += value;
 count++;
 
}

      
}

      return count ? clamp(sum / count) : 0.5;

    
}


    _trajectoryCertainty(trace) {

      if (!trace?.length) return 0.5;

      return clamp(trace.reduce((sum, row) => sum + clamp((row.total || 0) * 0.72 + (row.margin || 0) * 1.8), 0) / trace.length);

    
}


    _dynamicPOSCoherence(trace) {

      if (!trace?.length) return 0;

      let sum = 0;

      for (const row of trace) {

        sum += clamp((row.dynamicPOS || 0) * 0.62 + (row.posCertainty || 0) * 0.38);

      
}

      return clamp(sum / trace.length);

    
}


    _correlationCoherence(trace) {

      if (!trace?.length) return 0;

      let sum = 0;

      for (const row of trace) sum += clamp(row.correlationSupport || 0);

      return clamp(sum / trace.length);

    
}


    _lexicalMargin(trace) {

      if (!trace?.length) return 0.5;

      let sum = 0;

      for (const row of trace) sum += clamp(0.5 + (Number(row.margin) || 0) * 2.5);

      return clamp(sum / trace.length);

    
}


    _transitionCoherence(trace) {

      if (!trace?.length) return 0.5;

      const rows = trace.filter(row => row?.isTransitionCandidate || Number(row?.transitionNeed) > 0);
      if (!rows.length) return 0.72;

      return clamp(
        rows.reduce((sum, row) => sum + clamp(Number(row?.transitionSupport) || 0), 0) / rows.length
      );

    }


    _contentFlowScore(trace) {

      if (!trace?.length) return 0.5;

      let maxRun = 0;
      let penalty = 0;

      for (const row of trace) {
        maxRun = Math.max(maxRun, Number(row?.contentRunAfter) || 0);
        penalty += Number(row?.contentStackPenalty) || 0;
      }

      const runScore = maxRun <= 2 ? 1 : maxRun === 3 ? 0.82 : maxRun === 4 ? 0.58 : 0.34;
      return clamp(runScore - Math.min(0.35, penalty * 0.45));

    }


    _conversationAlignment(text, analysis) {

      const state = analysis?.conversationState || null;
      if (!state) return 0.5;

      const words = new Set(
        (this.processor?.tokenize?.(String(text || ''), 192) || [])
          .filter(token => token?.pos !== 'Punct')
          .map(token => String(token.lower || token.surface || '').toLowerCase())
          .filter(Boolean)
      );

      if (!words.size) return 0;

      const coverage = anchors => {
        const rows = (anchors || []).flatMap(anchor => String(anchor || '').toLowerCase().split(/\s+/)).filter(Boolean);
        if (!rows.length) return 0.5;
        let hit = 0;
        for (const row of rows) if (words.has(row)) hit++;
        return hit / rows.length;
      };

      const current = coverage(state.currentAnchors);
      const prior = coverage(state.priorAnchors);
      const continuity = clamp(state.continuity || 0);
      const freedom = clamp(state.responseFreedom || 0);

      // Reward staying connected to what the user actually said. This never
      // penalizes long semantic-association chains; it only adds positive
      // evidence when a generated sentence remains anchored to the conversation.
      return clamp(current * 0.66 + prior * 0.16 + continuity * 0.10 + freedom * 0.08);

    }


    _buildDiagnostics(response, analysis, audit, started, options = {
}
) {

      const neural = this._neuralAgreement(response.text);

      const tokenSupport = clamp(this.rsl?.sequenceSupport?.(response.text) ?? 0);

      const structureSupport = clamp(this.rsl?.structureSupport?.(response.text, this.processor) ?? 0);

      const grammar = clamp(audit?.legalRatio ?? 0);

      const grammarScore = clamp(audit?.grammarScore ?? grammar);

      const contract = clamp(response.contractFit ?? 0.5);

      const trajectoryCertainty = this._trajectoryCertainty(response.trace);

      const dynamicPOSCoherence = this._dynamicPOSCoherence(response.trace);

      const correlationCoherence = this._correlationCoherence(response.trace);

      const lexical = clamp(response.lexicalCoverage ?? analysis?.lexicalCoverage ?? 0);

      const scaffoldFit = clamp(response.scaffoldFit ?? 0);

      const conversationAlignment = clamp(response.conversationAlignment ?? this._conversationAlignment(response.text, analysis));
      const responseFreedom = clamp(analysis?.conversationState?.responseFreedom || 0);

      const wordStateCoherence = clamp(this.lexical.wordState?.coherence?.(response.text, this.processor) ?? 0.5);

      const phraseStateCoherence = clamp(this.lexical.phraseState?.coherence?.(response.text, this.processor) ?? 0.5);

      const wordMatrixCoherence = clamp(this.lexical.wordMatrix?.coherence?.(response.text, this.processor) ?? 0.5);

      const lexicalMargin = this._lexicalMargin(response.trace);

      const keywordResponseAlignment = clamp(
        (response.trace || []).length
          ? (response.trace || []).reduce((sum, row) => sum + (Number(row?.keywordAlignment) || 0), 0) / (response.trace || []).length
          : 0
      );

      const transitionCoherence = this._transitionCoherence(response.trace);
      const contentFlow = this._contentFlowScore(response.trace);

      const surfaceGrammarRepairs = Array.isArray(response.grammarRepairs) ? response.grammarRepairs.length : 0;

      const languageStateSupport = clamp(this.lexical.languageState?.responseSupport?.(response.text, analysis) ?? 0.5);

      const lexicalAdaptation = clamp(
        wordStateCoherence * 0.40 + phraseStateCoherence * 0.30 +
        wordMatrixCoherence * 0.20 + lexicalMargin * 0.10
      );

      const finalTrace = response.trace?.[response.trace.length - 1];

      const sentenceComplete = clamp(
        response.contractFit >= 0.7 &&
        !audit?.issues?.some?.(x => (x.reasons || []).includes('OPEN_DEPENDENCY'))
          ? 1 : 0.35
      );

      const nonPunctuationTokens = (this.processor?.tokenize?.(String(response.text || ''), 64) || []).filter(token => token?.pos !== 'Punct');
      const generationTraceLength = Array.isArray(response.trace) ? response.trace.length : 0;
      const structuralDeadlock = Boolean(
        nonPunctuationTokens.length === 0 ||
        (generationTraceLength === 0 && grammar < 0.08 && scaffoldFit < 0.08)
      );


      // Production response confidence is calibrated around observable output
      // quality first: legality, completeness, prompt contract, and scaffold fit.
      // Learned RSL/POS support remains meaningful evidence, but a young model no
      // longer has to synchronously train itself just to make a grammatically
      // complete high-fit candidate cross the gate. This is not a confidence
      // clamp: malformed or low-fit candidates still score substantially lower.
      const baseConfidence = clamp(
        structureSupport * 0.08 +
        dynamicPOSCoherence * 0.10 +
        grammar * 0.19 +
        sentenceComplete * 0.15 +
        contract * 0.14 +
        scaffoldFit * 0.14 +
        transitionCoherence * 0.05 +
        contentFlow * 0.05 +
        tokenSupport * 0.04 +
        neural * 0.02 +
        trajectoryCertainty * 0.03 +
        lexical * 0.01 +
        conversationAlignment * 0.02 +
        languageStateSupport * 0.10
      );

      // Learned lexical state is centered at 0.5. Cold lexical memory therefore
      // does not artificially boost or suppress the established confidence model;
      // as WordState/PhraseState/WordMatrix learn, confidence can move naturally.
      let confidence = clamp(baseConfidence + (lexicalAdaptation - 0.5) * 0.20);

      if ((audit?.illegal || 0) > 0) confidence *= Math.max(0.35, audit.legalRatio * grammarScore);

      if (dynamicPOSCoherence < 0.42) confidence *= Math.max(0.45, dynamicPOSCoherence + 0.28);

      if (languageStateSupport < 0.60) confidence *= Math.max(0.42, languageStateSupport + 0.20);


      return {

        architecture: 'vilotni2-production-6m-general-language-fastlane-rsl-correlation-curiosity',
        trainingInfoUsed: false,
        channels: {

          liveNeural: neural,
          tokenSupport,
          learnedPOSContext: structureSupport,
          dynamicPOSCoherence,
          correlationCoherence,
          grammarLegality: grammar,
          sentenceCompleteness: sentenceComplete,
          answerContract: contract,
          scaffoldFit,
          conversationAlignment,
          responseFreedom,
          trajectoryCertainty,
          lexicalCoverage: lexical,
          wordStateCoherence,
          phraseStateCoherence,
          wordMatrixCoherence,
          lexicalMargin,
          keywordResponseAlignment,
          transitionCoherence,
          contentFlow,
          surfaceGrammarRepairs,
          languageStateSupport,
          lexicalAdaptation,
          baseConfidence,
          total: confidence
        
}
,
        confidence,
        syntaxFlow: response?.syntaxFlow || null,
        intent: analysis?.intent || 'statement',
        requestedSlot: analysis?.requestedSlot || 'content',
        audit,
        generationTrace: response.trace?.slice(0, 48) || [],
        effort: options.effort || 'high',
        effortPass: Number(options.effortPass) || 0,
        deadlineTruncated: Boolean(response.deadlineTruncated),
        structuralDeadlock,
        deadlockSignals: {
          nonPunctuationTokenCount: nonPunctuationTokens.length,
          generationTraceLength,
          scaffoldWordCount: Array.isArray(analysis?.responseScaffold?.words) ? analysis.responseScaffold.words.length : 0,
          scaffoldFirstWord: String(analysis?.responseScaffold?.words?.[0] || '')
        },
        focusedFastPath: Boolean(response.focusedFastPath),
        effectiveTarget: Number(response.effectiveTarget) || 0,
        staticPoolSize: Number(response.staticPoolSize) || 0,
        generationMs: performance.now() - started
      
}
;

    
}


    // No canned epistemic answer path exists in production. Unknown or weak
    // prompts remain in the same learned generation/search system as every
    // other prompt. KnowledgeState contributes uncertainty evidence only.

    _complexStructurePressure(analysis = {}) {
      const structure = analysis?.complexLanguage || {};
      const complexity = Math.max(0, Math.min(1, Number(structure.structuralComplexity || 0)));
      const unresolved = Math.min(1, (structure.unresolved || []).length / 4);
      return { complexity, unresolved, preserveRelations: (structure.relationTerms || []).slice(0,12), requiredRoles: (structure.requestedRoles || []).slice(0,8) };
    }

    async generate(promptText, analysis = null, options = {
}
) {

      const started = performance.now();

      analysis = analysis || this.processor.analyzePrompt(promptText);

      if (this.lexical.languageState && !analysis.languageState) analysis = this.lexical.languageState.enrichAnalysis(analysis);

      // Unknown terms no longer short-circuit generation. VilotNI first attempts
      // compositional reasoning from the surrounding known words and learned
      // grammar. FastLane may fall back to an epistemic boundary only after a
      // deep attempt fails repeatedly.
      const response = this._generate(promptText, analysis, options);

      let text = String(response.text || '').trim();

      if (text.length > this.options.maxResponseChars) text = text.slice(0, this.options.maxResponseChars).trimEnd() + '…';

      const audit = this.processor.auditResponse(text, {
 allowRootRepeat: false 
}
);

      const diagnostics = this._buildDiagnostics(response, analysis, audit, started, options);

      this.generationCount++;

      this.lastGenerationMs = diagnostics.generationMs;

      this.lastDiagnostics = diagnostics;

      this.trace?.push?.(
        globalThis.VilotTraceEvent?.DECODER_GENERATE || 22,
        this.lastGenerationMs,
        diagnostics.confidence,
        diagnostics.channels.total,
        0
      );

      return {

        text,
        confidence: Number((diagnostics.confidence * 100).toFixed(1)),
        diagnostics,
        trainingInfoUsed: false,
        generationMs: this.lastGenerationMs
      
}
;

    
}


    async generateBatch(promptText, analysis = null, options = {
}
) {

      const started = performance.now();

      analysis = analysis || this.processor.analyzePrompt(promptText);

      if (this.lexical.languageState && !analysis.languageState) analysis = this.lexical.languageState.enrichAnalysis(analysis);

      const batchSize = Math.max(1, Math.min(32, options.batchSize | 0 || 4));

      const variantBase = Math.max(0, options.variantBase | 0 || 0);

      const explorationLevel = Math.max(0, Number(options.explorationLevel) || 0);

      const staticPool = options.staticPool || this._staticCandidatePool(analysis, explorationLevel);

      const candidates = new Array(batchSize);


      // The batch shares prompt analysis and lexical discovery. Extra variants
      // are not started when the latency governor predicts that they cannot fit
      // inside the protected deadline.
      let produced = 0;


      for (let i = 0;
 i < batchSize;
 i++) {

        if (
          i > 0 &&
          options.latencyGovernor &&
          !options.latencyGovernor.canStart(
            'candidate',
            options.deadlineAt,
            options.latencyGovernor.options?.decoderFinishReserveMs || 3.5,
            10
          )
        ) {

          break;

        
}


        const candidateStarted = performance.now();


        const candidate = await this.generate(promptText, analysis, {

          ...options,
          staticPool,
          variantIndex: variantBase + i,
          effortPass: (options.batchIndex | 0) + i
        
}
);


        candidates[produced++] = candidate;


        options.latencyGovernor?.observe?.(
          'candidate',
          performance.now() - candidateStarted
        );

      
}


      candidates.length = produced;


      let ranked = candidates.map((candidate, index) => ({
 index, candidate, score: (Number(candidate.confidence) || 0) / 100 
}
));

      if (this.matrix && candidates.length) {

        this.matrix.ensure(candidates.length, this.batchRankWeights.length);

        for (let i = 0;
 i < candidates.length;
 i++) {

          const c = candidates[i];

          const ch = c?.diagnostics?.channels || {
}
;

          this.matrix.setRow(i, i, [
            (Number(c.confidence) || 0) / 100,
            Number(ch.total) || 0,
            Number(ch.grammarLegality) || 0,
            Number(ch.dynamicPOSCoherence) || 0,
            Number(ch.correlationCoherence) || 0,
            Number(ch.scaffoldFit) || 0,
            Number(ch.sentenceCompleteness) || 0,
            Number(ch.tokenSupport) || 0,
            Number(ch.trajectoryCertainty) || 0,
            Number(ch.wordStateCoherence) || 0.5,
            Number(ch.phraseStateCoherence) || 0.5,
            Number(ch.wordMatrixCoherence) || 0.5,
            Number(ch.lexicalMargin) || 0.5,
            Number(ch.languageStateSupport) || 0.5,
            Number(ch.keywordResponseAlignment) || 0,
            Number(ch.transitionCoherence) || 0.5,
            Number(ch.contentFlow) || 0.5
          ]);

        
}

        this.matrix.score(this.batchRankWeights, 0, candidates.length);

        ranked = candidates.map((candidate, index) => ({
 index, candidate, score: this.matrix.scores[index] 
}
))
          .sort((a, b) => b.score - a.score);

      
}
 else {

        ranked.sort((a, b) => b.score - a.score);

      
}


      return {

        candidates: ranked.map(row => row.candidate),
        best: ranked[0]?.candidate || null,
        batchSize: candidates.length,
        requestedBatchSize: batchSize,
        deadlineLimited: candidates.length < batchSize,
        batchMs: performance.now() - started,
        matrix: this.matrix?.status?.() || null
      
}
;

    
}


    inspectLearnedContext(query, limit = 12) {

      const analysis = this.processor.analyzePrompt(String(query || ''));

      const seed = analysis.contentWords?.[analysis.contentWords.length - 1] || analysis.words?.at?.(-1)?.lower || '';

      const seedEntry = seed ? this.processor.lookup(seed) : null;

      return {

        seed,
        pos: seedEntry?.pos || null,
        tokenCandidates: this.rsl?.learnedCandidates?.(seed, 'vilotni2', limit) || [],
        contextCandidates: this.rsl?.contextCandidates?.(seed, seedEntry?.pos || 'Other', limit) || []
      
}
;

    
}


    status() {

      return {

        ready: this.ready,
        generationCount: this.generationCount,
        lastGenerationMs: this.lastGenerationMs,
        adaptiveProfile: this.runtimeOptions?.selfImprovementProfile || null,
        lastConfidence: this.lastDiagnostics?.confidence ?? null,
        architecture: {

          unifiedVilotNI2: true,
          sixMillionCore: true,
          wordsJSONActive: true,
          trainingInfoPresent: true,
          trainingInfoLoadedForResponses: false,
          posRelationsLearnedByRSL: true,
          dynamicPOSPerOccurrence: true,
          posBigramAndTrigramLearning: true,
          contextualPOSWindowLearning: true,
          backgroundLearning: 'Correlation.js',
          batchedCandidateSearch: true,
          residentCandidateMatrix: Boolean(this.matrix),
          learnedWordState: Boolean(this.lexical.wordState),
          learnedPhraseState: Boolean(this.lexical.phraseState),
          learnedWordMatrix: Boolean(this.lexical.wordMatrix),
          generalLanguageState: Boolean(this.lexical.languageState),
          tenseAspectMoodVoiceAgreement: Boolean(this.lexical.languageState),
          unknownConceptBoundary: Boolean(this.lexical.languageState)
        
}

      
}
;

    
}

  
}


  globalThis.VilotDecoder = VilotDecoder;

}
)();


/**
 * Decoder production research extension.
 *
 * Executable, data-driven numerical/statistical/graph utilities used by
 * diagnostics, background learning, calibration, matrix experiments and
 * future compute paths. No prompt-specific phrases or benchmark answers live
 * here. Methods stay out of the foreground hot path unless explicitly called.
 */
(() => {

  'use strict';


  class VilotDecoderProductionLab {

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


    matVec(matrix, rows, cols, vector) {

      const r=Math.max(0,rows|0), c=Math.max(0,cols|0);

      const out=new Float32Array(r);

      for (let i=0;
i<r;
i++) {

        let sum=0;

        const base=i*c;

        for (let j=0;
j<c;
j++) sum+=(Number(matrix[base+j])||0)*(Number(vector[j])||0);

        out[i]=sum;

      
}

      return out;

    
}


    outerUpdate(matrix, rows, cols, a, b, rate = 0.01) {

      const r=Math.min(rows|0,a?.length||0), c=Math.min(cols|0,b?.length||0);

      const eta=Number(rate)||0;

      for (let i=0;
i<r;
i++) {

        const av=Number(a[i])||0;

        const base=i*cols;

        for (let j=0;
j<c;
j++) matrix[base+j]+=eta*av*(Number(b[j])||0);

      
}

      return matrix;

    
}


    rowNormalize(matrix, rows, cols, epsilon = 1e-9) {

      for (let i=0;
i<(rows|0);
i++) {

        const base=i*cols;

        let norm2=0;

        for (let j=0;
j<(cols|0);
j++) {
 const v=Number(matrix[base+j])||0;
 norm2+=v*v;
 
}

        const inv=1/(Math.sqrt(norm2)+epsilon);

        for (let j=0;
j<(cols|0);
j++) matrix[base+j]=(Number(matrix[base+j])||0)*inv;

      
}

      return matrix;

    
}


    vectorAdd(a, b, scale = 1) {

      const n=Math.max(a?.length||0,b?.length||0);

      const out=new Float32Array(n);

      const s=Number(scale)||0;

      for (let i=0;
i<n;
i++) out[i]=(Number(a?.[i])||0)+(Number(b?.[i])||0)*s;

      return out;

    
}


    vectorLerp(a, b, t = 0.5) {

      const n=Math.max(a?.length||0,b?.length||0);

      const out=new Float32Array(n);

      const k=Math.max(0,Math.min(1,Number(t)||0));

      for (let i=0;
i<n;
i++) out[i]=(Number(a?.[i])||0)*(1-k)+(Number(b?.[i])||0)*k;

      return out;

    
}


    orthogonalize(vector, basis) {

      const out=Float32Array.from(vector||[]);

      for (const b of basis||[]) {

        const denom=this.dot(b,b)||1;

        const scale=this.dot(out,b)/denom;

        for (let i=0;
i<Math.min(out.length,b.length);
i++) out[i]-=scale*(Number(b[i])||0);

      
}

      return Float32Array.from(this.normalize(out));

    
}


    powerIteration(matrix, n, iterations = 16) {

      const size=Math.max(1,n|0);

      let v=new Float32Array(size).fill(1/Math.sqrt(size));

      for (let step=0;
step<Math.max(1,iterations|0);
step++) {

        const next=this.matVec(matrix,size,size,v);

        v=Float32Array.from(this.normalize(next));

      
}

      return v;

    
}


    calibrationBins(pairs, bins = 10) {

      const count=Math.max(2,bins|0);

      const out=Array.from({
length:count
}
,(_,i)=>({
lo:i/count,hi:(i+1)/count,n:0,confidence:0,accuracy:0
}
));

      for (const pair of pairs||[]) {

        const c=Math.max(0,Math.min(1,Number(pair.confidence)||0));

        const y=Number(pair.correct)||0;

        const idx=Math.min(count-1,Math.floor(c*count));

        const bin=out[idx];

        bin.n++;
 bin.confidence+=c;
 bin.accuracy+=y;

      
}

      for (const bin of out) if (bin.n) {
 bin.confidence/=bin.n;
 bin.accuracy/=bin.n;
 
}

      return out;

    
}


    expectedCalibrationError(pairs, bins = 10) {

      const grouped=this.calibrationBins(pairs,bins);

      const total=grouped.reduce((s,b)=>s+b.n,0)||1;

      return grouped.reduce((s,b)=>s+(b.n/total)*Math.abs(b.accuracy-b.confidence),0);

    
}


    brierScore(pairs) {

      let total=0,count=0;

      for (const pair of pairs||[]) {

        const p=Math.max(0,Math.min(1,Number(pair.confidence)||0));

        const y=Number(pair.correct)?1:0;

        const d=p-y;

        total+=d*d;
 count++;

      
}

      return count?total/count:0;

    
}


    novelty(candidate, history, similarity = (a,b) => a === b ? 1 : 0) {

      let best=0;

      for (const prior of history||[]) best=Math.max(best,Number(similarity(candidate,prior))||0);

      return Math.max(0,1-best);

    
}


    diversity(items, similarity) {

      const input=Array.from(items||[]);

      if (input.length<2) return input.length?1:0;

      let total=0,count=0;

      for (let i=0;
i<input.length;
i++) for (let j=i+1;
j<input.length;
j++) {

        total+=1-Math.max(0,Math.min(1,Number(similarity(input[i],input[j]))||0));

        count++;

      
}

      return count?total/count:0;

    
}


    dedupeCandidates(items, key = x => String(x?.text ?? x)) {

      const best=new Map();

      for (const item of items||[]) {

        const k=key(item);

        const old=best.get(k);

        const score=Number(item?.confidence ?? item?.score)||0;

        const oldScore=Number(old?.confidence ?? old?.score)||-Infinity;

        if (!old||score>oldScore) best.set(k,item);

      
}

      return Array.from(best.values());

    
}


    status() {

      return {

        module: 'Decoder',
        role: 'production-research-extension',
        calls: this.calls,
        methodCount: Object.getOwnPropertyNames(Object.getPrototypeOf(this)).length - 1,
        ageMs: (performance?.now?.() ?? Date.now()) - this.createdAt
      
}
;

    
}

  
}


  globalThis.VilotDecoderProductionLab = VilotDecoderProductionLab;

}
)();

