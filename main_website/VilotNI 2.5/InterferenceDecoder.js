/**
 * VilotNI 2.5 - InterferenceDecoder.js
 *
 * FOUR FULL-SENTENCE INTERFERENCE
 *
 * VilotNI 2.5 generates four complete candidate sentences again. Each branch
 * receives a different evidence view, then the sentence batch is compared at
 * both sentence and word level.
 *
 * Sentence-level interference rule:
 *   - final confidence is within 2 percentage points, OR
 *   - the complete sentences are sufficiently alike.
 *
 * Word-level mixing still requires compatible sentence placement, grammar/POS,
 * roots/semantics, and the continuous multi-state processor. Confidence from
 * participating full sentences is blended but never exceeds the strongest real
 * branch confidence.
 *
 * This intentionally reverses the later one-pass projection optimization.
 * This is quantum-inspired superposition/interference, not quantum hardware.
 */
(() => {

  'use strict';


  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));


  class VilotInterferenceDecoder {

    constructor(baseDecoder, processor, options = {
}
, trainingKnowledge = null, stateProcessor = null, semanticComposer = null, clausePlanner = null, ideaFusion = null, surfaceRealizer = null, mixedSlotPlanner = null, outputCoherence = null, learnedKnowledgeBridge = null, claimVerifier = null, uncertaintyState = null, counterfactualWorkspace = null, sentenceMixer = null) {

      this.base = baseDecoder;

      this.processor = processor;

      this.rsl = baseDecoder?.rsl || null;

      this.trace = baseDecoder?.trace || null;

      this.trainingKnowledge = trainingKnowledge;

      this.stateProcessor = stateProcessor;

      this.semanticComposer = semanticComposer;

      this.clausePlanner = clausePlanner;

      this.ideaFusion = ideaFusion;

      this.surfaceRealizer = surfaceRealizer;

      this.mixedSlotPlanner = mixedSlotPlanner;

      this.outputCoherence = outputCoherence;

      this.learnedKnowledgeBridge = learnedKnowledgeBridge;

      this.claimVerifier = claimVerifier;

      this.uncertaintyState = uncertaintyState;

      this.counterfactualWorkspace = counterfactualWorkspace;

      this.sentenceMixer = sentenceMixer;

      this._activeSentenceMix = null;


      this.options = {

        branchCount: 4,
        fullSentenceBranches: options.interferenceFullSentenceBranches !== false,
        preserveFullModelBreadth: options.preserveFullModelBreadth !== false,
        sentenceSimilarityThreshold: Math.max(
          0.05,
          Math.min(1, Number(options.interferenceSentenceSimilarityThreshold) || 0.58)
        ),
        confidenceWindowPct: Math.max(
          0.1,
          Number(options.interferenceWordConfidenceWindowPct) || 2
        ),
        positionTolerance: Math.max(
          0,
          options.interferencePositionTolerance | 0 || 1
        ),
        normalizedPositionTolerance: Math.max(
          0.02,
          Number(options.interferenceNormalizedPositionTolerance) || 0.11
        ),
        minAgreement: Math.max(
          2,
          Math.min(4, options.interferenceMinAgreement | 0 || 2)
        ),
        maxMixedTrace: Math.max(
          32,
          options.interferenceMaxMixedTrace | 0 || 64
        ),

        focusedSuperpositionPoolLimit: Math.max(
          32,
          options.latencyFocusedSuperpositionPoolLimit | 0 || 72
        ),

        focusedKnowledgeLimit: Math.max(
          3,
          options.latencyFocusedKnowledgeLimit | 0 || 6
        ),

        semanticReserveMs: Math.max(
          0.5,
          Number(options.latencySemanticReserveMs) || 4
        ),

        coherenceReserveMs: Math.max(
          0.25,
          Number(options.latencyCoherenceReserveMs) || 1.5
        )
      
}
;


      this.generationCount = 0;

      this.lastDiagnostics = null;

      this.lastGenerationMs = 0;

      this.lastBaseDecodeMs = 0;

      this.lastCollapseMs = 0;

      this.lastSemanticMs = 0;

      this.lastSurfaceMs = 0;


      this._lexicalCompatibilityCache = new Map();

      this._transitionCache = new Map();

      this._cacheLimit = 4096;

    
}


    _entry(word) {

      return this.processor?.lookup?.(String(word || '').toLowerCase())
        || this.processor?.resolveSayableWord?.(
          String(word || ''),
          {
 allowRootFallback: true 
}

        )
        || null;

    
}


    _selectedLocalConfidence(candidate, row) {

      const candidateConfidence = clamp((Number(candidate?.confidence) || 0) / 100);

      if (!row) return candidateConfidence * 100;


      const total = clamp(row.total ?? 0.5);

      const pos = clamp(row.posCertainty ?? row.dynamicPOS ?? 0.5);

      const margin = clamp(0.5 + (Number(row.margin) || 0) * 2.2);

      const role = clamp(row.roleFit ?? 0.5);
      const transition = clamp(row.transitionSupport ?? 0.5);
      const flow = 1 - clamp((Number(row.contentStackPenalty) || 0) * 2.5);

      const local =
        total * 0.40 +
        pos * 0.18 +
        margin * 0.16 +
        role * 0.14 +
        transition * 0.07 +
        flow * 0.05;


      return clamp(candidateConfidence * 0.58 + local * 0.42) * 100;

    
}


    _alternativeConfidence(candidate, row, alt) {

      const selectedConfidence = this._selectedLocalConfidence(candidate, row);

      const alternatives = Array.isArray(row?.interferenceAlternatives)
        ? row.interferenceAlternatives
        : [];


      const bestScore = Number(alternatives[0]?.total);

      const altScore = Number(alt?.total);


      if (!Number.isFinite(bestScore) || !Number.isFinite(altScore)) {

        return selectedConfidence;

      
}


      // Translate the already-computed local score gap into a local confidence
      // gap. Close alternatives remain close; weak alternatives separate fast.
      const scoreGap = Math.max(0, bestScore - altScore);

      const confidencePenaltyPct = Math.min(24, scoreGap * 42);


      return Math.max(0, selectedConfidence - confidencePenaltyPct);

    
}


    _posCompatible(a, b) {

      if (!a || !b) return false;

      if (a === b) return true;


      const nominal = new Set(['Noun', 'ProperNoun']);

      const verbal = new Set(['Verb', 'Aux', 'Modal']);


      if (nominal.has(a) && nominal.has(b)) return true;

      if (verbal.has(a) && verbal.has(b)) return true;

      return false;

    
}


    _lexicallyCompatible(a, b, knowledgeWords) {

      if (!a || !b) return false;

      if (a.lower === b.lower) return true;


      const ea = this._entry(a.lower);

      const eb = this._entry(b.lower);

      if (!ea || !eb) return false;


      const rootA = String(ea.root || ea.rootFamily || ea.lemma || '').toLowerCase();

      const rootB = String(eb.root || eb.rootFamily || eb.lemma || '').toLowerCase();

      if (rootA && rootB && rootA === rootB) return true;


      const bothKnowledge =
        knowledgeWords?.has?.(a.lower) &&
        knowledgeWords?.has?.(b.lower);


      if (
        bothKnowledge &&
        this._posCompatible(ea.pos || a.pos, eb.pos || b.pos)
      ) {

        return true;

      
}


      const association = Number(
        this.processor?.associationScore?.(
          a.lower,
          b.lower,
          {
 bidirectional: true 
}

        ) || 0
      );


      return association >= 0.46 &&
        this._posCompatible(ea.pos || a.pos, eb.pos || b.pos);

    
}


    _positionClose(a, b, lenA, lenB) {

      const stepClose =
        Math.abs((a?.step || 0) - (b?.step || 0)) <=
        this.options.positionTolerance;


      const posA = lenA > 1 ? a.index / (lenA - 1) : 0;

      const posB = lenB > 1 ? b.index / (lenB - 1) : 0;


      const normalizedClose =
        Math.abs(posA - posB) <=
        this.options.normalizedPositionTolerance;


      return stepClose || normalizedClose;

    
}


    _nativeAlternative(row, rank = 0) {

      const alternatives = Array.isArray(row?.interferenceAlternatives)
        ? row.interferenceAlternatives
        : [];


      if (alternatives.length) {

        return alternatives[Math.min(rank, alternatives.length - 1)] || alternatives[0];

      
}


      return {

        rank: 0,
        word: row?.word || '',
        lower: String(row?.word || '').toLowerCase(),
        pos: row?.pos || this._entry(row?.word)?.pos || 'Other',
        total: Number(row?.total) || 0,
        posCertainty: Number(row?.posCertainty) || 0.5,
        roleFit: Number(row?.roleFit) || 0.5,
        focusSupport: Number(row?.focusSupport) || 0.5,
        transitionSupport: Number(row?.transitionSupport) || 0.5,
        transitionNeed: Number(row?.transitionNeed) || 0,
        transitionBoost: Number(row?.transitionBoost) || 0,
        contentStackPenalty: Number(row?.contentStackPenalty) || 0,
        isTransitionCandidate: Boolean(row?.isTransitionCandidate),
        source: row?.source || 'decoder-selected'
      
}
;

    
}


    _eligibleAlternatives(candidate, row) {

      const alternatives = Array.isArray(row?.interferenceAlternatives)
        ? row.interferenceAlternatives
        : [this._nativeAlternative(row, 0)];


      const selected = alternatives[0] || this._nativeAlternative(row, 0);

      const selectedConfidence =
        this._alternativeConfidence(candidate, row, selected);


      return alternatives
        .slice(0, this.options.branchCount)
        .map((alt, rank) => ({

          ...alt,
          rank,
          lower: String(alt.lower || alt.word || '').toLowerCase(),
          pos: alt.pos || this._entry(alt.word)?.pos || row?.pos || 'Other',
          confidence: this._alternativeConfidence(candidate, row, alt),
          selectedConfidence
        
}
))
        .filter(alt => alt.word);

    
}


    _branchChoice(candidate, row, step, branchIndex) {

      const alternatives = this._eligibleAlternatives(candidate, row);

      if (!alternatives.length) return null;


      const selected = alternatives[0];

      if (branchIndex === 0 || alternatives.length === 1) return selected;


      // Each branch samples a different nearby part of the local distribution.
      // An alternative is allowed to replace the selected word only when the
      // local confidence remains within the user's 2 percentage-point window.
      const patterns = [
        0,
        1 + (step % 2),
        1 + ((step + 1) % 3),
        1 + ((step * 2 + 1) % 3)
      ];


      let requestedRank = patterns[branchIndex] || branchIndex;

      requestedRank = Math.min(requestedRank, alternatives.length - 1);


      for (let offset = 0;
 offset < alternatives.length;
 offset++) {

        const index = (requestedRank + offset) % alternatives.length;

        const alt = alternatives[index];


        if (index === 0) continue;

        const closeEnough = this.stateProcessor
          ? this.stateProcessor.withinWindow(
              this.stateProcessor.fromPercent(alt.confidence),
              this.stateProcessor.fromPercent(selected.confidence),
              this.options.confidenceWindowPct / 100
            )
          : Math.abs(alt.confidence - selected.confidence) <=
              this.options.confidenceWindowPct;


        if (closeEnough) {

          return alt;

        
}

      
}


      return selected;

    
}


    _repair(words, analysis) {

      const surfaces = words.map(item => item.word);


      const repaired =
        this.processor?.repairGeneratedSurface?.(surfaces, analysis)
        || this.processor?.repairGeneratedWords?.(surfaces)
        || {
 words: surfaces, repairs: [], text: '' 
}
;


      const text = String(
        repaired.text ||
        (repaired.words || surfaces)
          .join(' ')
          .replace(/\s+([.,!?;:])/g, '$1')
      ).trim();


      return {

        text,
        repairs: repaired.repairs || [],
        words: repaired.words || surfaces
      
}
;

    
}


    _audit(text) {

      return this.processor?.auditResponse?.(
        text,
        {
 allowRootRepeat: false 
}

      ) || {

        legalRatio: 1,
        grammarScore: 1,
        illegal: 0,
        issues: []
      
}
;

    
}


    _deriveBranches(baseCandidate, analysis, knowledge, variantSeed = 0) {

      const reasoningModes = analysis?.reasoningWorkspace?.branchModes || analysis?.reasoningBranchModes || [];

      const sourceTrace =
        baseCandidate?.diagnostics?.generationTrace || [];


      if (!sourceTrace.length) {

        return Array.from(
          {
 length: this.options.branchCount 
}
,
          (_, branch) => ({

            ...baseCandidate,
            diagnostics: {

              ...(baseCandidate?.diagnostics || {
}
),
              superpositionBranch: branch,
              reasoningMode: reasoningModes[branch] || 'lexical-alternative'
            
}

          
}
)
        );

      
}


      const branches = [];


      for (let branchIndex = 0;
 branchIndex < this.options.branchCount;
 branchIndex++) {

        const chosen = [];


        for (let step = 0;
 step < sourceTrace.length;
 step++) {

          const row = sourceTrace[step];

          const shiftedBranch =
            branchIndex === 0
              ? 0
              : 1 + ((branchIndex - 1 + variantSeed) % 3);


          const alt =
            this._branchChoice(
              baseCandidate,
              row,
              step + variantSeed,
              shiftedBranch
            ) || this._nativeAlternative(row, 0);


          chosen.push({

            branch: branchIndex,
            step,
            index: step,
            word: alt.word,
            lower: String(alt.lower || alt.word || '').toLowerCase(),
            pos: alt.pos || row.pos || 'Other',
            confidence: Number(alt.confidence) ||
              this._selectedLocalConfidence(baseCandidate, row),
            candidateConfidence: Number(baseCandidate?.confidence) || 0,
            row: {

              ...row,
              word: alt.word,
              pos: alt.pos || row.pos,
              total: Number.isFinite(Number(alt.total))
                ? Number(alt.total)
                : row.total,
              posCertainty: alt.posCertainty ?? row.posCertainty,
              roleFit: alt.roleFit ?? row.roleFit,
              focusSupport: alt.focusSupport ?? row.focusSupport,
              transitionSupport: alt.transitionSupport ?? row.transitionSupport,
              transitionNeed: alt.transitionNeed ?? row.transitionNeed,
              transitionBoost: alt.transitionBoost ?? row.transitionBoost,
              contentStackPenalty: alt.contentStackPenalty ?? row.contentStackPenalty,
              isTransitionCandidate: alt.isTransitionCandidate ?? row.isTransitionCandidate,
              source: alt.source || row.source,
              interferenceWordConfidence: Number(alt.confidence) ||
                this._selectedLocalConfidence(baseCandidate, row),
              superpositionRank: Number(alt.rank) || 0
            
}

          
}
);

        
}


        const compact = [];

        for (const item of chosen) {

          const previous = compact[compact.length - 1];

          if (
            previous &&
            previous.lower === item.lower &&
            item.pos !== 'Punct'
          ) {

            continue;

          
}

          compact.push(item);

        
}


        const repaired = this._repair(compact, analysis);

        const audit = this._audit(repaired.text);


        const avgLocal =
          compact.reduce((sum, item) => sum + item.confidence, 0) /
          Math.max(1, compact.length);


        const grammar =
          clamp(audit?.grammarScore ?? audit?.legalRatio ?? 0.5);


        // A projected branch may be slightly weaker than the native path, but
        // its confidence can never exceed the actual V2 base candidate.
        const branchConfidence = Math.min(
          Number(baseCandidate?.confidence) || 0,
          avgLocal * 0.86 + grammar * 100 * 0.14
        );


        branches.push({

          ...baseCandidate,
          text: repaired.text || String(baseCandidate?.text || ''),
          confidence: Number(branchConfidence.toFixed(1)),
          diagnostics: {

            ...(baseCandidate?.diagnostics || {
}
),
            generationTrace: compact.map(item => item.row),
            superpositionBranch: branchIndex,
            superpositionSinglePass: true,
            reasoningMode: reasoningModes[branchIndex] || 'lexical-alternative',
            reasoningSelectedState: analysis?.reasoningWorkspace?.selectedState || null,
            grammarAudit: audit
          
}
,
          trainingInfoUsed: Boolean(knowledge?.rows?.length)
        
}
);

      
}


      return branches;

    
}


    _traceRows(candidate, branchIndex) {

      const trace = candidate?.diagnostics?.generationTrace || [];


      if (trace.length) {

        return trace
          .map((row, index) => ({

            branch: branchIndex,
            step: Number.isFinite(Number(row.step))
              ? Number(row.step)
              : index,
            index,
            word: String(row.word || ''),
            lower: String(row.word || '').toLowerCase(),
            pos: row.pos || this._entry(row.word)?.pos || 'Other',
            confidence: Number.isFinite(Number(row.interferenceWordConfidence))
              ? Number(row.interferenceWordConfidence)
              : this._selectedLocalConfidence(candidate, row),
            candidateConfidence: Number(candidate?.confidence) || 0,
            row
          
}
))
          .filter(row => row.word);

      
}


      const tokens =
        this.processor?.tokenize?.(
          String(candidate?.text || ''),
          192
        ) || [];


      return tokens
        .filter(token => token?.pos !== 'Punct')
        .map((token, index) => ({

          branch: branchIndex,
          step: index,
          index,
          word: token.surface || token.lower || '',
          lower: String(token.lower || token.surface || '').toLowerCase(),
          pos: token.pos || 'Other',
          confidence: Number(candidate?.confidence) || 0,
          candidateConfidence: Number(candidate?.confidence) || 0,
          row: null
        
}
))
        .filter(row => row.word);

    
}


    _nearestForPosition(rows, targetNormalized) {

      if (!rows?.length) return null;

      if (rows.length === 1) return rows[0];


      let best = null;

      let bestDistance = Infinity;


      for (const row of rows) {

        const normalized =
          row.index / Math.max(1, rows.length - 1);

        const distance =
          Math.abs(normalized - targetNormalized);


        if (distance < bestDistance) {

          bestDistance = distance;

          best = row;

        
}

      
}


      return best;

    
}


    _boundedCacheSet(cache, key, value) {

      if (cache.size >= this._cacheLimit) {

        const first = cache.keys().next();

        if (!first.done) cache.delete(first.value);

      
}

      cache.set(key, value);

      return value;

    
}


    _staticLexicalCompatibilityScore(a, b) {

      if (!a || !b) return 0;


      const lowerA = String(a.lower || a.word || '').toLowerCase();

      const lowerB = String(b.lower || b.word || '').toLowerCase();


      if (!lowerA || !lowerB) return 0;

      if (lowerA === lowerB) return 1;


      const key =
        lowerA < lowerB
          ? `${lowerA}\u0000${lowerB}`
          : `${lowerB}\u0000${lowerA}`;


      if (this._lexicalCompatibilityCache.has(key)) {

        return this._lexicalCompatibilityCache.get(key);

      
}


      const ea = this._entry(lowerA);

      const eb = this._entry(lowerB);


      if (!ea || !eb) {

        return this._boundedCacheSet(
          this._lexicalCompatibilityCache,
          key,
          0
        );

      
}


      if (!this._posCompatible(ea.pos || a.pos, eb.pos || b.pos)) {

        return this._boundedCacheSet(
          this._lexicalCompatibilityCache,
          key,
          0
        );

      
}


      const rootA =
        String(
          ea.root ||
          ea.rootFamily ||
          ea.lemma ||
          ''
        ).toLowerCase();


      const rootB =
        String(
          eb.root ||
          eb.rootFamily ||
          eb.lemma ||
          ''
        ).toLowerCase();


      if (rootA && rootB && rootA === rootB) {

        return this._boundedCacheSet(
          this._lexicalCompatibilityCache,
          key,
          0.97
        );

      
}


      const association =
        clamp(
          Number(
            this.processor?.associationScore?.(
              lowerA,
              lowerB,
              {
 bidirectional: true 
}

            ) || 0
          )
        );


      const score =
        association >= 0.46
          ? clamp(
              0.58 +
              association * 0.38
            )
          : 0;


      return this._boundedCacheSet(
        this._lexicalCompatibilityCache,
        key,
        score
      );

    
}


    _pairCompatibilityScore(a, b, branchRows, knowledgeWords) {

      if (!a || !b || a.branch === b.branch) return 0;


      const sentencePair =
        this.sentenceMixer?.pair?.(
          this._activeSentenceMix,
          a.branch,
          b.branch
        ) || null;


      // Sentence-level interference now follows the original experiment again:
      // two full candidate sentences may interact when final confidence is
      // within two percentage points OR when the complete sentences are alike.
      if (sentencePair && sentencePair.eligible === false) {

        return 0;

      
}


      const localConfidenceClose =
        Math.abs(a.confidence - b.confidence) <=
        this.options.confidenceWindowPct;


      const sentenceAlike =
        Boolean(sentencePair?.sentenceAlike) ||
        Number(sentencePair?.similarity || 0) >=
          this.options.sentenceSimilarityThreshold;


      if (!localConfidenceClose && !sentenceAlike) {

        return 0;

      
}


      if (!this._posCompatible(a.pos, b.pos)) {

        return 0;

      
}


      const rowsA = branchRows[a.branch] || [];

      const rowsB = branchRows[b.branch] || [];


      if (
        !this._positionClose(
          a,
          b,
          rowsA.length,
          rowsB.length
        )
      ) {

        return 0;

      
}


      if (a.lower === b.lower) return 1;


      let score =
        this._staticLexicalCompatibilityScore(
          a,
          b
        );


      const bothKnowledge =
        knowledgeWords?.has?.(a.lower) &&
        knowledgeWords?.has?.(b.lower);


      if (bothKnowledge) {

        score = Math.max(score, 0.84);

      
}


      return clamp(score);

    
}


    _transitionPrior(previous, candidate) {

      if (!previous || !candidate) return 0.5;


      const a =
        String(
          previous.lower ||
          previous.word ||
          ''
        ).toLowerCase();


      const b =
        String(
          candidate.lower ||
          candidate.word ||
          ''
        ).toLowerCase();


      if (!a || !b) return 0.5;


      const key = `${a}\u0000${b}`;


      if (this._transitionCache.has(key)) {

        return this._transitionCache.get(key);

      
}


      let score = 0.5;


      if (a === b) {

        score = 0.2;

      
}
 else {

        const association =
          clamp(
            Number(
              this.processor?.associationScore?.(
                a,
                b,
                {
 bidirectional: false 
}

              ) || 0
            )
          );


        const prevEntry = this._entry(a);

        const nextEntry = this._entry(b);


        let grammar = 0.5;


        if (prevEntry && nextEntry) {

          const p = prevEntry.pos;

          const q = nextEntry.pos;


          if (p === 'Determiner' && ['Noun','Adjective','ProperNoun'].includes(q)) grammar = 0.9;

          else if (['Noun','ProperNoun'].includes(p) && ['Verb','Aux','Modal'].includes(q)) grammar = 0.86;

          else if (['Verb','Aux','Modal'].includes(p) && ['Noun','ProperNoun','Determiner','Adjective','Adverb'].includes(q)) grammar = 0.78;

          else if (p === 'Adjective' && ['Noun','ProperNoun'].includes(q)) grammar = 0.88;

          else if (p === 'Adverb' && ['Verb','Adjective','Adverb'].includes(q)) grammar = 0.8;

        
}


        score =
          clamp(
            association * 0.58 +
            grammar * 0.42
          );

      
}


      return this._boundedCacheSet(
        this._transitionCache,
        key,
        score
      );

    
}


    _pairAgreement(a, b, branchRows, knowledgeWords) {

      return (
        this._pairCompatibilityScore(
          a,
          b,
          branchRows,
          knowledgeWords
        ) > 0
      );

    
}


    _chooseSlot(candidates, branchRows, knowledgeWords, mixVariant, previousSelected = null) {

      if (!candidates.length) return null;


      if (
        this.stateProcessor &&
        this.options.stateProcessorEnabled !== false
      ) {

        const result =
          this.stateProcessor.collapseCandidates(
            candidates,
            {

              confidenceWindow:
                this.options.confidenceWindowPct / 100,
              minAgreement:
                this.options.minAgreement,
              mixVariant,

              getState: candidate =>
                this.stateProcessor.fromPercent(
                  candidate?.confidence || 0
                ),

              compatible: (a, b) =>
                this._pairCompatibilityScore(
                  a,
                  b,
                  branchRows,
                  knowledgeWords
                ),

              knowledgeScore: candidate =>
                knowledgeWords?.has?.(
                  candidate?.lower
                )
                  ? 1
                  : 0,

              focusScore: candidate =>
                clamp(
                  candidate?.row?.focusSupport ??
                  0.5
                ),

              transitionScore: candidate => {

                const row = candidate?.row || {};
                const learned = this._transitionPrior(previousSelected, candidate);
                const syntax = clamp(row.transitionSupport ?? 0.5);
                const flow = 1 - clamp((Number(row.contentStackPenalty) || 0) * 2.5);

                return clamp(learned * 0.70 + syntax * 0.24 + flow * 0.06);

              },

              reliabilityScore: candidate => {

                const row = candidate?.row || {
}
;

                return clamp(
                  (row.posCertainty ?? 0.5) * 0.18 +
                  (row.dynamicPOS ?? 0.5) * 0.13 +
                  (row.roleFit ?? 0) * 0.15 +
                  (row.posRoleFit ?? 0.5) * 0.13 +
                  (row.languageSupport ?? 0.5) * 0.14 +
                  clamp((candidate?.candidateConfidence || 0) / 100) * 0.13 +
                  (row.transitionSupport ?? 0.5) * 0.08 +
                  (1 - clamp((Number(row.contentStackPenalty) || 0) * 2.5)) * 0.06 +
                  (1 - clamp(row.repeatPenalty ?? 0)) * 0.08
                );

              
}
,

              contextScore: candidate => {

                const row = candidate?.row || {
}
;

                return clamp(
                  (row.contextTransition ?? 0.5) * 0.18 +
                  (row.contextWindow ?? 0.5) * 0.17 +
                  (row.tokenTransition ?? 0.5) * 0.13 +
                  (row.lexicalPairSupport ?? 0.5) * 0.17 +
                  (row.phraseStateSupport ?? 0.5) * 0.13 +
                  (row.wordStateSupport ?? 0.5) * 0.08 +
                  (row.transitionSupport ?? 0.5) * 0.14
                );

              
}
,

              semanticScore: candidate => {

                const row = candidate?.row || {
}
;

                return clamp(
                  (row.predicateSupport ?? 0.5) * 0.20 +
                  (row.correlationSupport ?? 0.5) * 0.18 +
                  (row.wordMatrixSupport ?? 0.5) * 0.14 +
                  (row.lexicalDiscovery ?? 0.5) * 0.14 +
                  (row.focusSupport ?? 0.5) * 0.12 +
                  (row.languageSupport ?? 0.5) * 0.14 +
                  (row.roleFit ?? 0) * 0.08 +
                  (row.keywordAlignment ?? 0) * 0.16
                );

              
}
,

              noveltyScore: candidate => {

                const row = candidate?.row || {
}
;

                return clamp(
                  (row.lexicalDiscovery ?? 0.5) * 0.46 +
                  (1 - clamp(row.repeatPenalty ?? 0)) * 0.34 +
                  (row.focusSupport ?? 0.5) * 0.20
                );

              
}
,

              certaintyScore: candidate => {

                const row = candidate?.row || {
}
;

                return clamp(
                  (row.posCertainty ?? 0.5) * 0.30 +
                  (row.dynamicPOS ?? 0.5) * 0.22 +
                  (row.languageSupport ?? 0.5) * 0.20 +
                  (row.posRoleFit ?? 0.5) * 0.16 +
                  clamp((candidate?.candidateConfidence || 0) / 100) * 0.12
                );

              
}

            
}

          );


        if (result?.selected) {

          return {

            candidate: result.selected,
            support: result.support || 1,
            state: result.state,
            stateScore: result.score,
            stateDiagnostics:
              result.diagnostics
          
}
;

        
}

      
}


      // Fallback is the pre-StateProcessor V2.5 interference selector.
      const agreed = [];


      for (const candidate of candidates) {

        let support = 0;


        for (const other of candidates) {

          if (candidate === other) continue;


          if (
            this._pairAgreement(
              candidate,
              other,
              branchRows,
              knowledgeWords
            )
          ) {

            support++;

          
}

        
}


        if (
          support + 1 >=
          this.options.minAgreement
        ) {

          agreed.push({

            candidate,
            support: support + 1
          
}
);

        
}

      
}


      if (!agreed.length) return null;


      agreed.sort((a, b) => {

        const ca = a.candidate;

        const cb = b.candidate;


        const knowledgeA =
          knowledgeWords?.has?.(ca.lower)
            ? 1
            : 0;


        const knowledgeB =
          knowledgeWords?.has?.(cb.lower)
            ? 1
            : 0;


        const scoreA =
          a.support * 0.34 +
          clamp(ca.confidence / 100) * 0.34 +
          clamp(ca.candidateConfidence / 100) * 0.16 +
          knowledgeA * 0.10 +
          clamp(ca.row?.focusSupport ?? 0.5) * 0.06;


        const scoreB =
          b.support * 0.34 +
          clamp(cb.confidence / 100) * 0.34 +
          clamp(cb.candidateConfidence / 100) * 0.16 +
          knowledgeB * 0.10 +
          clamp(cb.row?.focusSupport ?? 0.5) * 0.06;


        if (scoreB !== scoreA) return scoreB - scoreA;


        const branchB =
          Number.isFinite(Number(cb?.branch))
            ? Number(cb.branch)
            : 0;


        const branchA =
          Number.isFinite(Number(ca?.branch))
            ? Number(ca.branch)
            : 0;


        return (
          ((branchB + mixVariant) % this.options.branchCount) -
          ((branchA + mixVariant) % this.options.branchCount)
        );

      
}
);


      return agreed[0];

    
}


    _mixBranches(branches, analysis, knowledge, mixVariant = 0, sentenceMix = null) {

      const validBranches =
        (branches || [])
          .filter(candidate =>
            candidate &&
            typeof candidate === 'object' &&
            String(candidate.text || '').trim()
          );


      if (!validBranches.length) {

        return {

          text: '',
          confidence: 0,
          diagnostics: {

            architecture:
              'vilotni2.5-no-valid-branch',
            interference: {

              branchCount: 0,
              mixedPositions: 0,
              confidenceWindowPct:
                this.options.confidenceWindowPct,
              fallback:
                'no-valid-branch'
            
}

          
}

        
}
;

      
}


      branches = validBranches;


      const sentenceAnalysis =
        sentenceMix ||
        this.sentenceMixer?.analyze?.(branches) ||
        null;


      this._activeSentenceMix = sentenceAnalysis;


      const branchRows =
        branches.map((candidate, branchIndex) =>
          this._traceRows(candidate, branchIndex)
        );


      const rankedBases = branches
        .map((candidate, branch) => ({

          candidate,
          branch,
          confidence: Number(candidate?.confidence) || 0
        
}
))
        .sort((a,b) => b.confidence - a.confidence);


      const baseMeta =
        rankedBases[mixVariant % Math.max(1, rankedBases.length)] ||
        rankedBases[0];


      const baseBranch = baseMeta?.branch ?? 0;

      const baseCandidate =
        baseMeta?.candidate || branches[0];


      const baseRows =
        branchRows[baseBranch] || [];


      const knowledgeWords =
        this.trainingKnowledge?.wordSupportSet?.(knowledge) ||
        new Set();


      if (!baseRows.length) {

        return {

          ...baseCandidate,
          diagnostics: {

            ...(baseCandidate?.diagnostics || {
}
),
            trainingInfoUsed: Boolean(knowledge?.rows?.length),
            interference: {

              branchCount: branches.length,
              confidenceWindowPct: this.options.confidenceWindowPct,
              mixedPositions: 0,
              branchContributions: [baseBranch],
              superpositionSinglePass: false,
              fullSentenceBranches: true,
              baseDecoderPasses: branches.length,
              sentenceInterference: sentenceAnalysis,
              fallback: 'no-generation-trace'
            
}

          
}
,
          trainingInfoUsed: Boolean(knowledge?.rows?.length)
        
}
;

      
}


      const mixed = [];

      const decisions = [];

      const contributions =
        new Set([baseBranch]);


      // Build the four local possibilities for every position first. The old
      // version committed greedily one slot at a time, which could make a
      // locally-good word destroy the phrase around it.
      const slotCandidates = [];

      const preferredRows = [];

      let previousPreferred = null;


      for (
        let slot = 0;

        slot < baseRows.length;

        slot++
      ) {

        const base =
          baseRows[slot];


        const targetNormalized =
          baseRows.length > 1
            ? slot /
              (
                baseRows.length -
                1
              )
            : 0;


        const localCandidates = [];


        for (
          let branch = 0;

          branch <
            branchRows.length;

          branch++
        ) {

          const row =
            this._nearestForPosition(
              branchRows[branch],
              targetNormalized
            );


          if (!row) continue;


          if (
            !this._posCompatible(
              base.pos,
              row.pos
            )
          ) {

            continue;

          
}


          const rowsA =
            branchRows[
              base.branch
            ] || [];


          const rowsB =
            branchRows[
              row.branch
            ] || [];


          if (
            !this._positionClose(
              base,
              row,
              rowsA.length,
              rowsB.length
            )
          ) {

            continue;

          
}


          localCandidates.push(
            row
          );

        
}


        // Always keep the original base row available as a continuity anchor.
        if (
          !localCandidates.some(
            row =>
              row.branch ===
                base.branch &&
              row.index ===
                base.index &&
              row.lower ===
                base.lower
          )
        ) {

          localCandidates.push(
            base
          );

        
}


        slotCandidates.push(
          localCandidates
        );


        const preferred =
          this._chooseSlot(
            localCandidates,
            branchRows,
            knowledgeWords,
            mixVariant + slot,
            previousPreferred
          );


        const preferredCandidate =
          preferred?.candidate ||
          base;


        preferredRows.push(
          preferredCandidate
        );


        previousPreferred =
          preferredCandidate;

      
}


      const planned =
        this.mixedSlotPlanner
          ?.plan?.(
            slotCandidates,
            baseRows,
            preferredRows,
            knowledgeWords,
            {

              transitionScore:
                (previous, current) => {

                  const learned = this._transitionPrior(previous, current);
                  const syntax = clamp(current?.row?.transitionSupport ?? 0.5);
                  const flow = 1 - clamp((Number(current?.row?.contentStackPenalty) || 0) * 2.5);
                  return clamp(learned * 0.70 + syntax * 0.24 + flow * 0.06);

                },

              compatibilityScore:
                (previous, current) =>
                  this._pairCompatibilityScore(
                    previous,
                    current,
                    branchRows,
                    knowledgeWords
                  )
            
}

          );


      const selectedPath =
        planned?.path?.length ===
          baseRows.length
          ? planned.path
          : preferredRows;


      let previousSelected = null;


      for (
        let slot = 0;

        slot < baseRows.length;

        slot++
      ) {

        const base =
          baseRows[slot];


        const selected =
          selectedPath[slot] ||
          base;


        const localCandidates =
          slotCandidates[slot] ||
          [];


        const chosen =
          this._chooseSlot(
            localCandidates,
            branchRows,
            knowledgeWords,
            mixVariant + slot,
            previousSelected
          );


        const selectedState =
          chosen?.candidate &&
          chosen.candidate.branch ===
            selected.branch &&
          chosen.candidate.lower ===
            selected.lower
            ? chosen
            : null;


        mixed.push({

          ...selected,
          word:
            selected.word,
          row:
            selected.row ||
            base.row
        
}
);


        if (
          selected.branch !==
          baseBranch
        ) {

          contributions.add(
            selected.branch
          );

        
}


        previousSelected =
          selected;


        decisions.push({

          slot,
          baseWord:
            base.word,
          selectedWord:
            selected.word,
          selectedBranch:
            selected.branch,
          selectedConfidence:
            Number(
              (
                Number(
                  selected
                    .confidence
                ) || 0
              ).toFixed(2)
            ),
          support:
            selectedState
              ?.support ||
            1,
          interfered:
            selected.branch !==
              base.branch ||
            selected.lower !==
              base.lower,
          stateProcessed:
            Boolean(
              selectedState
                ?.stateDiagnostics
            ),
          collapsedState:
            Number.isFinite(
              Number(
                selectedState
                  ?.state
              )
            )
              ? Number(
                  Number(
                    selectedState
                      .state
                  ).toFixed(6)
                )
              : null,
          stateScore:
            Number.isFinite(
              Number(
                selectedState
                  ?.stateScore
              )
            )
              ? Number(
                  Number(
                    selectedState
                      .stateScore
                  ).toFixed(6)
                )
              : null,
          stateProcessorMs:
            Number(
              selectedState
                ?.stateDiagnostics
                ?.processorMs
            ) || 0,
          stateOperations:
            Number(
              selectedState
                ?.stateDiagnostics
                ?.operations
            ) || 0,
          constructiveState:
            Number(
              selectedState
                ?.stateDiagnostics
                ?.constructiveTotal
            ) || 0,
          destructiveState:
            Number(
              selectedState
                ?.stateDiagnostics
                ?.destructiveTotal
            ) || 0,
          stateEntropy:
            Number(
              selectedState
                ?.stateDiagnostics
                ?.finalEntropy
            ) || 0,
          statePasses:
            Number(
              selectedState
                ?.stateDiagnostics
                ?.propagationPasses
            ) || 0,
          stateMargin:
            Number(
              selectedState
                ?.stateDiagnostics
                ?.scoreMargin
            ) || 0,
          oracleMarkedRatio:
            Number(selectedState?.stateDiagnostics?.matrixVectorState?.markedRatio) || 0,
          oracleDiscrimination:
            Number(selectedState?.stateDiagnostics?.matrixVectorState?.oracleDiscrimination) || 0,
          oracleBestProbability:
            Number(selectedState?.stateDiagnostics?.matrixVectorState?.bestProbability) || 0
        
}
);

      
}


      const compact = [];


      for (const item of mixed) {

        const previous =
          compact[compact.length - 1];


        if (
          previous &&
          previous.lower === item.lower &&
          item.pos !== 'Punct'
        ) {

          continue;

        
}


        compact.push(item);

      
}


      const repaired =
        this._repair(compact, analysis);


      const mixedText = repaired.text;

      const mixedAudit =
        this._audit(mixedText);


      const baseAudit =
        this._audit(baseCandidate?.text || '');


      const mixedPositions =
        decisions.filter(
          row =>
            row.interfered &&
            row.selectedBranch !== baseBranch
        ).length;


      const grammarMixed =
        clamp(
          mixedAudit?.grammarScore ??
          mixedAudit?.legalRatio ??
          0
        );


      const grammarBase =
        clamp(
          baseAudit?.grammarScore ??
          baseAudit?.legalRatio ??
          0
        );


      const validMix =
        mixedText &&
        contributions.size >= 2 &&
        mixedPositions > 0 &&
        (mixedAudit?.illegal || 0) === 0 &&
        grammarMixed >=
          Math.max(0.72, grammarBase - 0.06);


      const finalText =
        validMix
          ? mixedText
          : String(
              baseCandidate?.text ||
              mixedText ||
              ''
            ).trim();


      const branchConfidences =
        branches.map(
          candidate =>
            Number(candidate?.confidence) || 0
        );


      const maxBranchConfidence =
        Math.max(0, ...branchConfidences);


      const meanBranchConfidence =
        branchConfidences.reduce(
          (a,b) => a + b,
          0
        ) /
        Math.max(1, branchConfidences.length);


      const sentenceBlendedConfidence =
        Number.isFinite(Number(sentenceAnalysis?.blendedConfidence))
          ? Number(sentenceAnalysis.blendedConfidence)
          : meanBranchConfidence;


      const agreementRatio =
        decisions.length
          ? decisions.filter(
              row => row.interfered
            ).length / decisions.length
          : 0;


      let mergedConfidence =
        sentenceBlendedConfidence * 0.78 +
        agreementRatio * 100 * 0.12 +
        (validMix ? grammarMixed : grammarBase) *
          100 * 0.10;


      mergedConfidence =
        Math.min(
          maxBranchConfidence,
          mergedConfidence
        );

      const oracleRows = decisions.filter(row => row.stateProcessed);
      const meanOracleDiscrimination = oracleRows.length
        ? oracleRows.reduce((sum, row) => sum + clamp(row.oracleDiscrimination || 0), 0) / oracleRows.length
        : 1;
      const oracleSaturation = 1 - meanOracleDiscrimination;
      // A saturated oracle means the query projection selected broadly, not that
      // the factual answer is literally certain. The final user-facing confidence
      // is calibrated once after coherence selection; candidate choice is unchanged.


      if (!validMix) {

        mergedConfidence =
          Number(baseCandidate?.confidence) ||
          mergedConfidence;

      
}


      const selectedTrace =
        (validMix
          ? compact
          : branchRows[baseBranch]
        )
        .slice(0, this.options.maxMixedTrace)
        .map((item, index) => ({

          ...(item.row || {
}
),
          step: index,
          word: item.word,
          pos: item.pos,
          interferenceBranch: item.branch,
          interferenceWordConfidence:
            Number(item.confidence.toFixed(3))
        
}
));


      const subjects =
        (knowledge?.rows || [])
        .slice(0, 8)
        .map(row => row.entry?.Subject || '')
        .filter(Boolean);


      const stateDecisions =
        decisions.filter(
          row => row.stateProcessed
        );


      const stateProcessorMs =
        stateDecisions.reduce(
          (sum, row) =>
            sum + row.stateProcessorMs,
          0
        );


      const stateOperations =
        stateDecisions.reduce(
          (sum, row) =>
            sum + row.stateOperations,
          0
        );


      const collapsedStates =
        stateDecisions
          .map(row => row.collapsedState)
          .filter(Number.isFinite);


      const meanCollapsedState =
        collapsedStates.length
          ? collapsedStates.reduce(
              (a,b) => a + b,
              0
            ) /
            collapsedStates.length
          : 0;


      const diagnostics = {

        ...(baseCandidate?.diagnostics || {
}
),
        architecture:
          'vilotni2.5-four-full-sentence-interference-resonance-graph',
        trainingInfoUsed:
          Boolean(knowledge?.rows?.length),
        generationTrace: selectedTrace,
        interference: {

          branchCount: branches.length,
          confidenceWindowPct:
            this.options.confidenceWindowPct,
          positionTolerance:
            this.options.positionTolerance,
          normalizedPositionTolerance:
            this.options.normalizedPositionTolerance,
          mixedPositions:
            validMix ? mixedPositions : 0,
          agreementRatio:
            Number(agreementRatio.toFixed(4)),
          oracleConfidenceCalibration: {
            discrimination: Number(meanOracleDiscrimination.toFixed(4)),
            saturation: Number(oracleSaturation.toFixed(4)),
            policy: 'oracle-selection-certainty-is-not-epistemic-confidence'
          },
          baseBranch,
          branchContributions:
            Array.from(contributions).sort(),
          branchConfidences:
            branchConfidences.map(
              value => Number(value.toFixed(1))
            ),
          knowledgeSubjects: subjects,
          trainingSearch:
            knowledge?.diagnostics || null,
          usedMixedText: validMix,
          superpositionSinglePass: false,
          fullSentenceBranches: true,
          baseDecoderPasses: branches.length,
          sentenceInterference: sentenceAnalysis,
          continuousStateProcessor: {

            enabled:
              Boolean(this.stateProcessor),
            mode:
              'continuous-soft-bit',
            range: [0, 1],
            representation:
              'Float32Array',
            confidenceWindow:
              this.options.confidenceWindowPct / 100,
            processedSlots:
              stateDecisions.length,
            meanCollapsedState:
              Number(
                meanCollapsedState.toFixed(6)
              ),
            operations:
              stateOperations,
            processorMs:
              stateProcessorMs,
            meanFinalEntropy:
              stateDecisions.length
                ? Number(
                    (
                      stateDecisions.reduce(
                        (sum, row) =>
                          sum + row.stateEntropy,
                        0
                      ) /
                      stateDecisions.length
                    ).toFixed(6)
                  )
                : 0,
            meanPropagationPasses:
              stateDecisions.length
                ? Number(
                    (
                      stateDecisions.reduce(
                        (sum, row) =>
                          sum + row.statePasses,
                        0
                      ) /
                      stateDecisions.length
                    ).toFixed(3)
                  )
                : 0,
            meanCollapseMargin:
              stateDecisions.length
                ? Number(
                    (
                      stateDecisions.reduce(
                        (sum, row) =>
                          sum + row.stateMargin,
                        0
                      ) /
                      stateDecisions.length
                    ).toFixed(6)
                  )
                : 0,
            meanOracleMarkedRatio:
              stateDecisions.length
                ? Number((stateDecisions.reduce((sum, row) => sum + row.oracleMarkedRatio, 0) / stateDecisions.length).toFixed(6))
                : 0,
            meanOracleDiscrimination:
              stateDecisions.length
                ? Number((stateDecisions.reduce((sum, row) => sum + row.oracleDiscrimination, 0) / stateDecisions.length).toFixed(6))
                : 1
          
}
,
          mixedSlotPlanner:
            planned?.diagnostics ||
            null,
          decisions:
            decisions.slice(0, 48)
        
}

      
}
;


      return {

        ...baseCandidate,
        text: finalText,
        confidence:
          Number(mergedConfidence.toFixed(1)),
        diagnostics,
        trainingInfoUsed:
          Boolean(knowledge?.rows?.length),
        interference:
          diagnostics.interference,
        sentenceInterference: sentenceAnalysis,
        branches:
          branches.map(
            (candidate, branch) => ({

              branch,
              text:
                String(candidate?.text || ''),
              confidence:
                Number(candidate?.confidence) || 0,
              trainingInfoUsed:
                Boolean(knowledge?.rows?.length),
              projectedFromSinglePass: false,
              fullSentenceCandidate: true
            
}
)
          )
      
}
;

    
}


    _meaningFirst(
      branches,
      analysis,
      knowledge,
      lexicalFallback,
      promptText = '',
      mixVariant = 0,
      options = {
}

    ) {

      const started =
        performance.now();


      if (
        !this.semanticComposer ||
        !this.clausePlanner ||
        !this.surfaceRealizer
      ) {

        return lexicalFallback;

      
}


      const deadlineAt = Number.isFinite(Number(lexicalFallback?.deadlineAt))
        ? Number(lexicalFallback.deadlineAt)
        : Infinity;


      const latencyGovernor =
        lexicalFallback?.latencyGovernor ||
        null;


      if (
        !this.options.preserveFullModelBreadth &&
        (
          latencyGovernor?.shouldSkipSemantic?.(deadlineAt) ||
          (
            Number.isFinite(deadlineAt) &&
            performance.now() >= deadlineAt - this.options.semanticReserveMs
          )
        )
      ) {

        return {

          ...lexicalFallback,
          diagnostics: {

            ...(lexicalFallback?.diagnostics || {
}
),
            meaningFirst: {

              enabled: true,
              usedSemanticSurface: false,
              fallback: 'latency-budget',
              semanticMs: 0,
              surfaceMs: 0
            
}

          
}

        
}
;

      
}


      const counterfactualWorkspace =
        options.counterfactualWorkspace ||
        lexicalFallback?.counterfactualWorkspace ||
        null;


      const composition =
        this.semanticComposer.compose(
          branches,
          analysis,
          knowledge,
          lexicalFallback,
          {

            deadlineAt: this.options.preserveFullModelBreadth ? Infinity : deadlineAt,
            deadlineReserveMs: 0.8
          
}

        );


      const rawPlan =
        this.clausePlanner.plan(
          composition,
          analysis,
          promptText,
          {

            deadlineAt: this.options.preserveFullModelBreadth ? Infinity : deadlineAt,
            deadlineReserveMs: 0.55
          
}

        );


      let verifiedPlan =
        this.claimVerifier?.verifyPlan?.(
          rawPlan,
          {

            counterfactualWorkspace,
            knowledge
          
}

        ) || rawPlan;


      if (counterfactualWorkspace?.active) {

        verifiedPlan =
          this.counterfactualWorkspace?.tagPlan?.(
            verifiedPlan,
            counterfactualWorkspace
          ) || verifiedPlan;

      
}


      const uncertainty =
        this.uncertaintyState?.assess?.({

          branches,
          composition,
          plan: verifiedPlan,
          verification: verifiedPlan?.verification || {
}
,
          knowledge,
          learnedKnowledge: knowledge?.learnedKnowledge || null,
          counterfactual: counterfactualWorkspace,
          analysis,
          propositionGraph: analysis?.propositionGraph || null,
          mechanismGraph: analysis?.mechanismGraph || null
        
}
) || null;


      const fusionStarted =
        performance.now();


      const sentencePlan =
        this.ideaFusion?.plan?.(
          verifiedPlan,
          analysis,
          promptText
        ) ||
        verifiedPlan;


      latencyGovernor?.observe?.(
        'ideaFusion',
        performance.now() -
          fusionStarted
      );


      const surfaceStarted =
        performance.now();


      const realized =
        this.surfaceRealizer.realize(
          sentencePlan,
          analysis,
          lexicalFallback
        );


      this.lastSurfaceMs =
        performance.now() -
        surfaceStarted;


      this.lastSemanticMs =
        performance.now() -
        started;


      if (
        !realized?.valid ||
        !String(
          realized?.text ||
          ''
        ).trim()
      ) {

        return {

          ...lexicalFallback,

          diagnostics: {

            ...(lexicalFallback
              ?.diagnostics ||
              {
}
),

            meaningFirst: {

              enabled: true,
              usedSemanticSurface:
                false,
              fallback:
                'surface-validation-failed',
              composition:
                composition
                  ?.diagnostics ||
                null,
              plan:
                verifiedPlan
                  ?.diagnostics ||
                null,
              verification:
                verifiedPlan?.verification || null,
              uncertainty,
              surface:
                realized
                  ?.diagnostics ||
                null,
              semanticMs:
                this.lastSemanticMs,
              surfaceMs:
                this.lastSurfaceMs
            
}

          
}

        
}
;

      
}


      const branchConfidences =
        (branches || [])
          .map(
            branch =>
              Number(
                branch
                  ?.confidence
              ) || 0
          );


      const maxBranchConfidence =
        Math.max(
          0,
          ...branchConfidences
        );


      const rawMeanBranchConfidence =
        branchConfidences.length
          ? branchConfidences
              .reduce(
                (a, b) =>
                  a + b,
                0
              ) /
            branchConfidences
              .length
          : Number(
              lexicalFallback
                ?.confidence
            ) || 0;


      const meanBranchConfidence =
        Number.isFinite(Number(lexicalFallback?.sentenceInterference?.blendedConfidence))
          ? Number(lexicalFallback.sentenceInterference.blendedConfidence)
          : rawMeanBranchConfidence;


      const grammar =
        clamp(
          realized
            ?.audit
            ?.grammarScore ??
          realized
            ?.audit
            ?.legalRatio ??
          0
        );


      const semanticAgreement =
        clamp(
          verifiedPlan
            ?.agreementRatio ||
          0
        );


      let confidence =
        meanBranchConfidence *
          0.86 +
        semanticAgreement *
          100 *
          0.07 +
        grammar *
          100 *
          0.07;


      confidence =
        Math.min(
          maxBranchConfidence ||
          confidence,
          confidence
        );


      // Meaning-first output must never manufacture confidence higher than the
      // actual V2 branch evidence or bypass the normal FastLane >80 gate.
      confidence =
        Math.min(
          confidence,
          Number(
            lexicalFallback
              ?.confidence
          ) ||
          maxBranchConfidence ||
          confidence
        );

      const slotDecisions = lexicalFallback?.diagnostics?.interference?.decisions || lexicalFallback?.interference?.decisions || [];
      const oracleDecisionRows = slotDecisions.filter(row => row?.stateProcessed);
      const finalOracleDiscrimination = oracleDecisionRows.length
        ? oracleDecisionRows.reduce((sum, row) => sum + clamp(row?.oracleDiscrimination || 0), 0) / oracleDecisionRows.length
        : 1;
      const finalOracleSaturation = 1 - finalOracleDiscrimination;


      if (uncertainty) {

        confidence = this.uncertaintyState.capConfidence(confidence, uncertainty);

      
}


      const oldInterference =
        lexicalFallback
          ?.diagnostics
          ?.interference ||
        lexicalFallback
          ?.interference ||
        {
}
;


      const semanticCandidate = {

        ...lexicalFallback,
        text:
          realized.text,
        confidence:
          Number(
            confidence.toFixed(1)
          ),
        counterfactual: Boolean(counterfactualWorkspace?.active),
        uncertaintyState: uncertainty,
        verifiedClaims: verifiedPlan?.verification || null
      
}
;


      const strongestBranch =
        (branches || [])
          .slice()
          .sort(
            (a, b) =>
              (
                Number(
                  b?.confidence
                ) || 0
              ) -
              (
                Number(
                  a?.confidence
                ) || 0
              )
          )[0] ||
        null;


      const coherenceAllowed =
        this.options.preserveFullModelBreadth ||
        (
          !latencyGovernor?.shouldSkipCoherence?.(deadlineAt) &&
          (
            !Number.isFinite(deadlineAt) ||
            performance.now() < deadlineAt - this.options.coherenceReserveMs
          )
        );


      const coherenceSelection =
        (
          coherenceAllowed
            ? this.outputCoherence
                ?.select?.(
            [
              {

                label:
                  'semantic-surface',
                candidate:
                  semanticCandidate,
                semanticSurface:
                  true
              
}
,
              {

                label:
                  'word-level-mix',
                candidate:
                  lexicalFallback
              
}
,
              ...(branches || [])
                .map(
                  (branch, index) => ({

                    label:
                      `branch-${index + 1}`,
                    candidate:
                      branch
                  
}
)
                )
            ],
            {

              analysis,

              semanticPlan: {

                subject:
                  verifiedPlan?.subject ||
                  composition?.subject ||
                  '',
                verification: verifiedPlan?.verification || null,
                uncertainty
              
}
,
              mixedSlot:
                lexicalFallback
                  ?.diagnostics
                  ?.interference
                  ?.mixedSlotPlanner ||
                null
            
}

          )
            : null
        ) ||
        {

          selected:
            semanticCandidate,
          selectedLabel:
            'semantic-surface',
          selectedScore:
            1,
          diagnostics:
            null
        
}
;


      const selectedOutput =
        coherenceSelection
          ?.selected ||
        semanticCandidate;

      const selectedRawConfidence = Number(selectedOutput?.confidence) || confidence || 0;
      const finalReportedConfidence = Math.max(0, Math.min(100,
        selectedRawConfidence * (1 - finalOracleSaturation * 0.035)
      ));


      const diagnostics = {

        ...(lexicalFallback
          ?.diagnostics ||
          {
}
),

        architecture:
          'vilotni2.5-meaning-first-semantic-collapse',

        meaningFirst: {

          enabled: true,
          usedSemanticSurface:
            true,
          branchCount:
            composition
              ?.frames
              ?.length ||
            branches?.length ||
            0,
          semanticSubject:
            verifiedPlan?.subject ||
            composition?.subject ||
            '',
          clauseCount:
            verifiedPlan?.clauses
              ?.length ||
            0,
          clauseTypes:
            (
              verifiedPlan?.clauses ||
              []
            ).map(
              clause =>
                clause.type
            ),
          agreementRatio:
            semanticAgreement,
          oracleConfidenceCalibration: {
            discrimination: Number(finalOracleDiscrimination.toFixed(4)),
            saturation: Number(finalOracleSaturation.toFixed(4)),
            policy: 'oracle-selection-certainty-is-not-epistemic-confidence'
          },
          composition:
            composition
              ?.diagnostics ||
            null,
          plan:
            verifiedPlan
              ?.diagnostics ||
            null,
          ideaFusion:
            sentencePlan
              ?.diagnostics
              ?.ideaFusion ||
            null,
          sentenceCount:
            sentencePlan
              ?.sentences
              ?.length ||
            verifiedPlan?.clauses
              ?.length ||
            0,
          multiIdeaSentenceCount:
            (
              sentencePlan
                ?.sentences ||
              []
            ).filter(
              sentence =>
                (
                  sentence?.ideas
                    ?.length ||
                  0
                ) >= 2
            ).length,
          surface:
            realized
              ?.diagnostics ||
            null,
          coherence:
            coherenceSelection
              ?.diagnostics ||
            null,
          selectedOutput:
            coherenceSelection
              ?.selectedLabel ||
            'semantic-surface',
          semanticMs:
            this.lastSemanticMs,
          surfaceMs:
            this.lastSurfaceMs,
          extraBaseDecoderPasses:
            0
        
}
,

        interference: {

          ...oldInterference,
          semanticMeaningFirst:
            true,
          semanticAgreementRatio:
            Number(
              semanticAgreement.toFixed(4)
            ),
          oracleConfidenceCalibration: oldInterference?.oracleConfidenceCalibration || null,
          semanticClauseCount:
            verifiedPlan?.clauses
              ?.length ||
            0
        
}

      
}
;


      return {

        ...selectedOutput,
        confidence: Number(finalReportedConfidence.toFixed(1)),

        diagnostics,

        interference:
          diagnostics
            .interference,

        semanticPlan:
          {

            subject:
              verifiedPlan?.subject ||
              '',
            clauses:
              (
                verifiedPlan?.clauses ||
                []
              ).map(
                clause => ({

                  type:
                    clause.type,
                  relation:
                    clause.relation,
                  subject:
                    clause.subject,
                  value:
                    clause.value,
                  support:
                    clause.support,
                  confidence:
                    clause.confidence
                
}
)
              ),
            sentences:
              (
                sentencePlan
                  ?.sentences ||
                []
              ).map(
                sentence => ({

                  ideaCount:
                    sentence?.ideas
                      ?.length ||
                    0,
                  connectors:
                    sentence?.connectors ||
                    [],
                  types:
                    (
                      sentence?.ideas ||
                      []
                    ).map(
                      idea =>
                        idea.type
                    )
                
}
)
              )
          
}

      
}
;

    
}


    _buildSuperpositionPool(basePool, knowledge, options = {
}
) {

      const map = new Map();


      const add = row => {

        if (!row?.entry?.lower) return;

        const key = row.entry.lower;

        const old = map.get(key);


        if (
          !old ||
          Number(row.sourceScore) >
          Number(old.sourceScore)
        ) {

          map.set(key, row);

        
}

      
}
;


      for (const row of basePool || []) add(row);


      // Fold all four TrainingInfo evidence views into one lexical pool so the
      // single decoder pass can score all four possibilities simultaneously.
      for (
        let branch = 0;

        branch < this.options.branchCount;

        branch++
      ) {

        const pool =
          this.trainingKnowledge?.buildStaticPool?.(
            basePool || [],
            knowledge,
            branch
          ) || [];


        for (const row of pool) add(row);

      
}


      const rows =
        Array.from(map.values());


      if (
        !options.focusedFastPath ||
        rows.length <=
          this.options.focusedSuperpositionPoolLimit
      ) {

        return rows;

      
}


      rows.sort(
        (a, b) =>
          (
            (Number(b?.sourceScore) || 0) * 0.62 +
            (Number(b?.focusScore) || 0.5) * 0.38
          ) -
          (
            (Number(a?.sourceScore) || 0) * 0.62 +
            (Number(a?.focusScore) || 0.5) * 0.38
          )
      );


      return rows.slice(
        0,
        this.options.focusedSuperpositionPoolLimit
      );

    
}


    async generate(promptText, analysis = null, options = {
}
) {

      const batch =
        await this.generateBatch(
          promptText,
          analysis,
          {

            ...options,
            batchSize: 1
          
}

        );


      return batch.best ||
        batch.candidates?.[0] ||
        null;

    
}


    async generateBatch(promptText, analysis = null, options = {
}
) {

      const started = performance.now();


      analysis =
        analysis ||
        this.processor.analyzePrompt(promptText);


      if (
        this.base?.lexical?.languageState &&
        !analysis.languageState
      ) {

        analysis =
          this.base.lexical.languageState
            .enrichAnalysis(analysis);

      
}


      const requested =
        Math.max(
          1,
          Math.min(
            32,
            options.batchSize | 0 || 4
          )
        );


      const variantBase =
        Math.max(
          0,
          options.variantBase | 0 || 0
        );


      const preserveBreadth =
        this.options.preserveFullModelBreadth !== false;


      // Quality-first restoration: the complete TrainingInfo search is available
      // to all four full sentence branches. The old latency-focused 4/6 row cap
      // is retained only when preserveFullModelBreadth is explicitly disabled.
      const knowledgeLimit =
        preserveBreadth
          ? 12
          : options.focusedFastPath
            ? this.options.focusedKnowledgeLimit
            : 12;


      const effectiveDeadline =
        preserveBreadth
          ? Infinity
          : options.deadlineAt;


      const trainingKnowledge =
        this.trainingKnowledge?.search?.(
          promptText,
          analysis,
          options.queryPlan || null,
          {

            limit: knowledgeLimit,
            deadlineAt: effectiveDeadline
          
}

        ) || {

          rows: [],
          diagnostics: {

            ready: false,
            rowCount: 0
          
}

        
}
;


      const counterfactualWorkspace =
        options.counterfactualWorkspace ||
        this.counterfactualWorkspace?.analyze?.(
          promptText,
          analysis
        ) || null;


      const learnedKnowledge =
        options.learnedKnowledge ||
        this.learnedKnowledgeBridge?.search?.(
          promptText,
          analysis,
          options.queryPlan || null,
          {

            deadlineAt: effectiveDeadline,
            limit: preserveBreadth ? 12 : (options.focusedFastPath ? 4 : 8),
            reasoningDepth: preserveBreadth ? 3 : (options.focusedFastPath ? 2 : 3),
            counterfactualWorkspace
          
}

        ) || null;


      const combinedLimit =
        preserveBreadth
          ? Math.max(24, knowledgeLimit + 12)
          : Math.max(
              knowledgeLimit,
              knowledgeLimit +
                (options.focusedFastPath ? 2 : 6)
            );


      const combinedRows = [
        ...(trainingKnowledge?.rows || []),
        ...(learnedKnowledge?.rows || [])
      ]
        .sort(
          (a, b) =>
            (Number(b?.score) || 0) -
            (Number(a?.score) || 0)
        )
        .slice(0, combinedLimit);


      const knowledge = {

        ...trainingKnowledge,
        rows: combinedRows,
        learnedKnowledge,
        counterfactualWorkspace,
        diagnostics: {

          ...(trainingKnowledge?.diagnostics || {
}
),
          learnedKnowledge:
            learnedKnowledge?.diagnostics || null,
          combinedRowCount:
            combinedRows.length,
          counterfactual:
            Boolean(counterfactualWorkspace?.active),
          preserveFullModelBreadth:
            preserveBreadth
        
}

      
}
;


      const baseStaticPool =
        options.staticPool ||
        this.base._staticCandidatePool?.(
          analysis,
          options.explorationLevel || 0
        ) ||
        [];


      // Restore the original four FULL candidate-sentence experiment.
      // Each branch receives its own TrainingInfo view and runs a complete V2
      // sentence generation. This intentionally replaces the later optimization
      // that projected four pseudo-branches from one decoder sentence.
      const decodeStarted = performance.now();

      const branches = [];


      for (
        let branch = 0;

        branch < this.options.branchCount;

        branch++
      ) {

        const branchAnalysis =
          this.trainingKnowledge?.buildBranchAnalysis?.(
            analysis,
            knowledge,
            branch,
            options.queryPlan || null
          ) || analysis;


        const branchStaticPool =
          this.trainingKnowledge?.buildStaticPool?.(
            baseStaticPool,
            knowledge,
            branch,
            analysis,
            options.queryPlan || null
          ) || baseStaticPool;


        const candidate =
          await this.base.generate(
            promptText,
            branchAnalysis,
            {

              ...options,
              // Do not shrink the decoder's lexical target/pool for the four
              // quality branches. Shared analysis/static evidence still avoids
              // redundant data loading without removing model possibilities.
              focusedFastPath:
                preserveBreadth
                  ? false
                  : options.focusedFastPath,
              deadlineAt:
                effectiveDeadline,
              staticPool:
                branchStaticPool,
              variantIndex:
                variantBase *
                  this.options.branchCount +
                branch,
              effortPass:
                (options.batchIndex | 0) *
                  this.options.branchCount +
                branch,
              interferenceBranch:
                branch,
              trainingInfoSearch:
                knowledge?.diagnostics || null,
              superpositionCapture:
                true,
              preserveFullModelBreadth:
                preserveBreadth
            
}

          );


        if (candidate) {

          branches.push(candidate);

        
}

      
}


      this.lastBaseDecodeMs =
        performance.now() -
        decodeStarted;


      // If an unexpected branch generation failure occurs, retain the existing
      // projection machinery only as an emergency fallback. Normal operation is
      // four independently generated full sentences.
      let effectiveBranches =
        branches.filter(
          candidate =>
            candidate &&
            typeof candidate === 'object' &&
            String(candidate.text || '').trim()
        );


      if (
        effectiveBranches.length < 2 &&
        effectiveBranches[0]
      ) {

        effectiveBranches =
          this._deriveBranches(
            effectiveBranches[0],
            analysis,
            knowledge,
            variantBase
          );

      
}


      const sentenceInterference =
        this.sentenceMixer?.analyze?.(
          effectiveBranches
        ) || null;


      const semanticBranches =
        sentenceInterference?.participantIndexes?.length
          ? sentenceInterference.participantIndexes
              .map(index => effectiveBranches[index])
              .filter(Boolean)
          : effectiveBranches;


      const collapseStarted =
        performance.now();


      const candidates = [];


      for (
        let i = 0;

        i < requested;

        i++
      ) {

        const lexicalMixed =
          this._mixBranches(
            effectiveBranches,
            analysis,
            knowledge,
            i,
            sentenceInterference
          );


        lexicalMixed.deadlineAt =
          options.deadlineAt;


        lexicalMixed.latencyGovernor =
          options.latencyGovernor ||
          null;


        lexicalMixed.counterfactualWorkspace =
          counterfactualWorkspace;


        lexicalMixed.sentenceInterference =
          sentenceInterference;


        const mixed =
          this._meaningFirst(
            semanticBranches,
            analysis,
            knowledge,
            lexicalMixed,
            promptText,
            i,
            {

              ...options,
              counterfactualWorkspace
            
}

          );


        candidates.push(mixed);

      
}


      candidates.sort((a,b) => {

        const confidenceDelta =
          (Number(b?.confidence) || 0) -
          (Number(a?.confidence) || 0);


        if (confidenceDelta) {

          return confidenceDelta;

        
}


        return (
          (Number(b?.interference?.mixedPositions) || 0) -
          (Number(a?.interference?.mixedPositions) || 0)
        );

      
}
);


      this.lastCollapseMs =
        performance.now() -
        collapseStarted;


      options.latencyGovernor?.observe?.(
        'interference',
        this.lastCollapseMs
      );


      this.generationCount++;

      this.lastGenerationMs =
        performance.now() -
        started;


      this.lastDiagnostics =
        candidates[0]?.diagnostics || null;


      return {

        candidates,
        best: candidates[0] || null,
        batchSize: requested,
        branchCount:
          effectiveBranches.length,
        baseDecoderPasses:
          effectiveBranches.length,
        fullSentenceBranches:
          true,
        branchCandidates:
          effectiveBranches.map(
            (branch, index) => ({

              branch: index,
              confidence:
                Number(branch?.confidence) || 0,
              text:
                String(branch?.text || ''),
              fullSentenceCandidate:
                true
            
}
)
          ),
        sentenceInterference,
        trainingInfo:
          knowledge?.diagnostics || null,
        batchMs:
          this.lastGenerationMs,
        phaseTiming: {

          trainingSearchMs:
            Number(
              knowledge?.diagnostics?.searchMs
            ) || 0,
          baseDecodeMs:
            this.lastBaseDecodeMs,
          stateCollapseMs:
            this.lastCollapseMs,
          semanticCompositionMs:
            this.lastSemanticMs,
          surfaceRealizationMs:
            this.lastSurfaceMs,
          totalMs:
            this.lastGenerationMs
        
}
,
        baseDecodeMs:
          this.lastBaseDecodeMs,
        matrix:
          this.base?.matrix?.status?.() || null
      
}
;

    
}


    inspectLearnedContext(query, limit = 12) {

      return (
        this.base?.inspectLearnedContext?.(
          query,
          limit
        ) || null
      );

    
}


    status() {

      return {

        ready:
          Boolean(this.base?.ready),
        role:
          'meaning-first-four-full-sentence-semantic-interference-decoder',
        generationCount:
          this.generationCount,
        lastGenerationMs:
          this.lastGenerationMs,
        lastBaseDecodeMs:
          this.lastBaseDecodeMs,
        lastCollapseMs:
          this.lastCollapseMs,
        lastSemanticMs:
          this.lastSemanticMs,
        lastSurfaceMs:
          this.lastSurfaceMs,
        branchCount:
          this.options.branchCount,
        wordConfidenceWindowPct:
          this.options.confidenceWindowPct,
        positionTolerance:
          this.options.positionTolerance,
        normalizedPositionTolerance:
          this.options.normalizedPositionTolerance,
        trainingKnowledge:
          this.trainingKnowledge?.status?.() ||
          null,
        learnedKnowledgeBridge: this.learnedKnowledgeBridge?.status?.() || null,
        claimVerifier: this.claimVerifier?.status?.() || null,
        uncertaintyState: this.uncertaintyState?.status?.() || null,
        counterfactualWorkspace: this.counterfactualWorkspace?.status?.() || null,
        stateProcessor:
          this.stateProcessor?.status?.() ||
          null,
        semanticComposer:
          this.semanticComposer?.status?.() ||
          null,
        clausePlanner:
          this.clausePlanner?.status?.() ||
          null,
        ideaFusion:
          this.ideaFusion?.status?.() ||
          null,
        surfaceRealizer:
          this.surfaceRealizer?.status?.() ||
          null,
        mixedSlotPlanner:
          this.mixedSlotPlanner?.status?.() ||
          null,
        outputCoherence:
          this.outputCoherence?.status?.() ||
          null,
        lastInterference:
          this.lastDiagnostics?.interference ||
          null,
        baseDecoder:
          this.base?.status?.() || null,
        architecture: {

          vilotni25: true,
          copiedVilotNI2Core: true,
          sixMillionCore: true,
          wordsJSONActive: true,
          trainingInfoLoadedForResponses: true,
          fourBranchInterference: true,
          meaningFirstOutput: true,
          semanticComposer: true,
          clausePlanner: true,
          multiIdeaSentenceFusion: true,
          ideaFusion: true,
          surfaceRealizer: true,
          mixedSlotGlobalPathPlanner: true,
          antiWordSaladCoherenceGate: true,
          wordLevelInterferenceRetainedAsLexicalLayer: true,
          continuousStateProcessor: true,
          stateRange: [0, 1],
          stateRepresentation: 'Float32Array',
          superpositionSinglePass: false,
          fullIndependentSentenceBranches: true,
          foregroundBaseDecoderPassesPerCandidate: this.options.branchCount,
          sentenceMixer: this.sentenceMixer?.status?.() || null,
          localWordConfidenceWindowPct:
            this.options.confidenceWindowPct,
          finalConfidenceNeverExceedsStrongestBranch:
            true
        
}

      
}
;

    
}

  
}


  globalThis.VilotInterferenceDecoder =
    VilotInterferenceDecoder;

}
)();

