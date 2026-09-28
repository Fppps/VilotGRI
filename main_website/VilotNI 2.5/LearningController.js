/**
 * VilotNI 2.5 - LearningController.js
 *
 * Background learning coordinator.
 *
 * Learning flow:
 *   observe -> predict -> compare -> update graph/hypothesis -> replay ->
 *   generalize -> resolve contradictions -> consolidate into RSL.
 *
 * Foreground work only queues compact experience records. Heavy learning stays
 * in background phases and is interruptible by a new foreground request.
 */
(() => {

  'use strict';


  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));


  const normalize = value =>
    String(value || '')
      .toLowerCase()
      .replace(/[’]/g, "'")
      .replace(/[^a-z0-9'+.:\- ]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 192);


  class VilotLearningController {

    constructor(deps = {
}
, options = {
}
) {

      this.rsl = deps.rsl || null;

      this.processor = deps.processor || null;

      this.correlation = deps.correlation || null;

      this.trainingKnowledge = deps.trainingKnowledge || null;


      this.graph = deps.graph || null;

      this.predictionError = deps.predictionError || null;

      this.hypotheses = deps.hypotheses || null;

      this.replay = deps.replay || null;

      this.generalization = deps.generalization || null;

      this.contradictions = deps.contradictions || null;

      this.evaluator = deps.evaluator || null;

      this.policy = deps.policy || null;


      this.options = {

        enabled: options.learningControllerEnabled !== false,
        queueMax: Math.max(128, options.learningControllerQueueMax | 0 || 1024),

        activeBudgetMs: Math.max(0.5, Number(options.learningControllerActiveBudgetMs) || 1.5),
        warmBudgetMs: Math.max(1, Number(options.learningControllerWarmBudgetMs) || 5),
        idleBudgetMs: Math.max(2, Number(options.learningControllerIdleBudgetMs) || 14),

        activeEvents: Math.max(1, options.learningControllerActiveEvents | 0 || 1),
        warmEvents: Math.max(2, options.learningControllerWarmEvents | 0 || 4),
        idleEvents: Math.max(4, options.learningControllerIdleEvents | 0 || 12),

        replayActive: Math.max(0, options.learningReplayActive | 0 || 0),
        replayWarm: Math.max(1, options.learningReplayWarm | 0 || 1),
        replayIdle: Math.max(1, options.learningReplayIdle | 0 || 4),

        generalizeEvery: Math.max(2, options.learningGeneralizeEvery | 0 || 4),
        contradictionEvery: Math.max(2, options.learningContradictionEvery | 0 || 3),
        hypothesisTestEvery: Math.max(2, options.learningHypothesisTestEvery | 0 || 3),
        hypothesisTestsActive: Math.max(0, options.learningHypothesisTestsActive | 0 || 0),
        hypothesisTestsWarm: Math.max(1, options.learningHypothesisTestsWarm | 0 || 2),
        hypothesisTestsIdle: Math.max(1, options.learningHypothesisTestsIdle | 0 || 6),
        promptGroundingStudies: Math.max(0, Math.min(8, options.learningPromptGroundingStudies | 0 || 3)),
        promptGroundingMinTrust: clamp(options.learningPromptGroundingMinTrust ?? 0.75),

        rslPromotionGain: Math.max(0.01, Math.min(0.6, Number(options.learningRSLPromotionGain) || 0.14)),
        replayGain: Math.max(0.03, Math.min(0.7, Number(options.learningReplayGain) || 0.20)),
        maxLexicalTermsPerEdge: Math.max(2, Math.min(8, options.learningMaxLexicalTermsPerEdge | 0 || 4)),
        maxEdgesPerTrustedStudy: Math.max(8, Math.min(64, options.learningMaxEdgesPerTrustedStudy | 0 || 32)),
        regressionEvery: Math.max(1, options.learningRegressionEvery | 0 || 2),
        regressionRollbackEnabled: options.learningRegressionRollbackEnabled !== false
      
}
;


      this.queue = [];

      this.cycles = 0;

      this.processedEvents = 0;

      this.trustedStudies = 0;

      this.promotions = 0;

      this.replayCycles = 0;

      this.generalizations = 0;

      this.hypothesisTests = 0;

      this.hypothesisConfirmations = 0;

      this.hypothesisContradictions = 0;

      this.promptGroundedStudies = 0;

      this.interruptions = 0;

      this.dropped = 0;

      this.lastCycleMs = 0;

      this.lastEvent = null;

      this.lastReplay = null;

      this.lastPromotion = null;

      this.lastRegression = null;

      this.regressionRollbacks = 0;

      this._cycleMutationJournal = [];

      this._cycleMutationKeys = new Set();

      this.dirty = false;

    
}


    _interrupted(epoch, getForegroundEpoch) {

      if (typeof getForegroundEpoch !== 'function') return false;


      const changed =
        getForegroundEpoch() !== epoch;


      if (changed) this.interruptions++;

      return changed;

    
}


    _phaseBudget(phase) {

      if (phase === 'IDLE') return this.options.idleBudgetMs;

      if (phase === 'WARM') return this.options.warmBudgetMs;

      return this.options.activeBudgetMs;

    
}


    _phaseEvents(phase) {

      if (phase === 'IDLE') return this.options.idleEvents;

      if (phase === 'WARM') return this.options.warmEvents;

      return this.options.activeEvents;

    
}


    _phaseReplay(phase) {

      if (phase === 'IDLE') return this.options.replayIdle;

      if (phase === 'WARM') return this.options.replayWarm;

      return this.options.replayActive;

    
}


    _phaseHypothesisTests(phase) {

      if (phase === 'IDLE') return this.options.hypothesisTestsIdle;

      if (phase === 'WARM') return this.options.hypothesisTestsWarm;

      return this.options.hypothesisTestsActive;

    
}


    _terms(text, max = 12) {

      const rows =
        this.processor?.tokenize?.(
          String(text || ''),
          Math.max(8, max * 2)
        ) || [];


      const terms =
        rows.length
          ? rows
              .filter(row => row?.pos !== 'Punct')
              .map(row => normalize(row?.lower || row?.surface))
              .filter(Boolean)
          : normalize(text)
              .split(/\s+/)
              .filter(Boolean);


      return Array.from(new Set(terms)).slice(0, max);

    
}


    _trust(provenance, explicit = null) {

      if (this.policy?.sourceTrust) {

        return this.policy.sourceTrust(provenance, explicit);

      
}

      if (Number.isFinite(Number(explicit))) return clamp(explicit);


      switch (String(provenance || 'unknown')) {

        case 'traininginfo': return 0.98;

        case 'verified_source': return 0.96;

        case 'user': return 0.56;

        case 'background_replay': return 0.52;

        case 'correlation': return 0.48;

        case 'generalization': return 0.34;

        case 'cognitive_inference': return 0.30;

        case 'model_response': return 0.20;

        case 'failed_response': return 0.12;

        default: return 0.40;

      
}

    
}


    _compactExperience(event) {

      return {

        type: String(event?.type || ''),
        text: String(event?.text || '').slice(0, 2048),
        promptText: String(event?.promptText || '').slice(0, 2048),
        responseText: String(event?.responseText || event?.text || '').slice(0, 2048),
        confidence: Number(event?.confidence) || 0,
        quality: Number.isFinite(Number(event?.quality))
          ? clamp(event.quality)
          : null,
        counterfactual: event?.counterfactual === true || event?.analysisSummary?.counterfactual === true,
        analysisSummary: event?.analysisSummary
          ? {

              intent: String(event.analysisSummary.intent || ''),
              requestedSlot: String(event.analysisSummary.requestedSlot || ''),
              contentWords: Array.from(event.analysisSummary.contentWords || []).slice(0, 24),
              counterfactual: event.analysisSummary.counterfactual === true
            
}

          : null,
        route: String(event?.route || ''),
        at: Number(event?.at) || Date.now()
      
}
;

    
}


    observeExperience(event) {

      if (!this.options.enabled || !event?.type) return false;


      if (this.queue.length >= this.options.queueMax) {

        this.queue.shift();

        this.dropped++;

      
}


      this.queue.push({

        kind: 'experience',
        payload: this._compactExperience(event)
      
}
);


      return true;

    
}


    _trainingEdges(entry) {

      if (!entry || typeof entry !== 'object') return [];


      const subject =
        normalize(
          entry.Subject ||
          entry.ID ||
          ''
        );


      if (!subject) return [];


      const rows = [];


      const add = (relation, value) => {

        const target = normalize(value);


        if (
          !target ||
          target === subject
        ) {

          return;

        
}


        rows.push({

          from: subject,
          relation,
          to: target,
          domain:
            normalize(
              entry.AI_Subdomain ||
              entry.Domain ||
              entry.Category ||
              ''
            ),
          entryId:
            entry.ID ||
            null,
          validFrom: entry.Valid_From ?? entry.ValidFrom ?? null,
          validUntil: entry.Valid_Until ?? entry.ValidUntil ?? null,
          lastVerified: entry.Last_Verified ?? entry.LastVerified ?? null,
          timeless: entry.Timeless === true
        
}
);

      
}
;


      const many = (relation, value) => {

        const arr =
          Array.isArray(value)
            ? value
            : value == null || value === ''
              ? []
              : [value];


        for (const item of arr) add(relation, item);

      
}
;


      many('alias_of', entry.Aliases);

      many('keyword', entry.Keywords);

      many('related_to', entry.Related_To);

      many('part_of', entry.Part_Of);

      many('has_part', entry.Has_Parts);

      many('causes', entry.Causes);

      many('effect', entry.Effects);

      many('contrasts_with', entry.Contrasts_With);

      many('capability', entry.Capabilities);

      many('limitation', entry.Limitations);

      many('property', entry.Properties);

      many('evaluated_by', entry.Evaluation_Criteria);

      many('failure_mode', entry.Failure_Modes);

      many('tradeoff', entry.Tradeoffs);

      many('example', entry.Examples);


      if (entry.Purpose) add('purpose', entry.Purpose);

      if (entry.Mechanism) add('mechanism', entry.Mechanism);

      if (entry.Domain) add('in_domain', entry.Domain);

      if (entry.AI_Subdomain) add('in_subdomain', entry.AI_Subdomain);


      return rows.slice(0, 96);

    
}


    observeTrustedKnowledge(entry, metadata = {
}
) {

      if (!this.options.enabled || !entry) return false;


      const trust =
        clamp(
          metadata.trust ??
          entry.Trust ??
          0.96
        );


      const payload = {

        kind: 'trusted-knowledge',
        subject: String(entry.Subject || ''),
        entryId: entry.ID || null,
        trust,
        edges: this._trainingEdges(entry)
      
}
;


      if (!payload.edges.length) return false;


      if (this.queue.length >= this.options.queueMax) {

        this.queue.shift();

        this.dropped++;

      
}


      this.queue.push(payload);

      this.trustedStudies++;

      return true;

    
}


    _learnEdge(edge, metadata = {
}
) {

      if (!edge?.from || !edge?.relation || !edge?.to) return null;


      const provenance =
        String(metadata.provenance || 'unknown');


      const trust =
        this._trust(
          provenance,
          metadata.trust
        );


      const observed =
        clamp(
          metadata.observed ??
          metadata.support ??
          1
        );


      const polarity =
        Number(metadata.polarity ?? 1) >= 0
          ? 1
          : -1;


      const edgeKey = this.graph?._key?.(edge.from, edge.relation, edge.to) || null;

      const trustedMutation = provenance === 'traininginfo' || provenance === 'verified_source';

      if (
        this.options.regressionRollbackEnabled &&
        !trustedMutation &&
        edgeKey &&
        !this._cycleMutationKeys.has(edgeKey)
      ) {

        const existing = this.graph?.getEdge?.(edge.from, edge.relation, edge.to) || null;

        this._cycleMutationKeys.add(edgeKey);

        this._cycleMutationJournal.push({

          key: edgeKey,
          existed: Boolean(existing),
          snapshot: existing ? this.graph?.snapshotEdge?.(existing) || null : null,
          provenance
        
}
);

      
}


      if (provenance === 'traininginfo' || provenance === 'verified_source') {

        this.evaluator?.registerExpectedRelation?.(
          edge.from,
          edge.relation,
          edge.to,
          {
 weight: trust 
}

        );

      
}


      const prediction =
        this.predictionError?.evaluate?.(
          edge.from,
          edge.relation,
          edge.to,
          {

            observed,
            trust,
            novelty:
              metadata.novelty ??
              0.5
          
}

        ) || null;


      this.evaluator?.observePrediction?.(
        prediction
      );


      const adaptiveStrength =
        0.55 +
        (prediction?.gain || 0.25) *
        0.45;


      // Surprise controls most learning, but a direct high-trust TrainingInfo
      // relation should not become weaker merely because the lexical system
      // already expected it. Trusted structured evidence therefore gets a
      // strong evidence floor while prediction error still controls all other
      // provenance types and later refinement.
      const evidenceStrength =
        provenance === 'traininginfo' &&
        trust >=
          (this.graph?.options?.trustedThreshold ?? 0.85)
          ? Math.max(0.94, adaptiveStrength)
          : adaptiveStrength;


      const learned =
        this.graph?.observeEdge?.(
          edge.from,
          edge.relation,
          edge.to,
          {

            provenance,
            trust,
            strength:
              observed *
              evidenceStrength,
            polarity,
            usefulness:
              metadata.usefulness ??
              0.5,
            domain:
              edge.domain ||
              metadata.domain ||
              '',
            entryId:
              edge.entryId ||
              metadata.entryId ||
              null,
            validFrom: edge.validFrom ?? metadata.validFrom ?? null,
            validUntil: edge.validUntil ?? metadata.validUntil ?? null,
            lastVerified: edge.lastVerified ?? metadata.lastVerified ?? null,
            timeless: edge.timeless ?? metadata.timeless ?? null
          
}

        ) || null;


      const hypothesis =
        this.hypotheses?.observe?.(
          edge.from,
          edge.relation,
          edge.to,
          {

            provenance,
            trust,
            support: observed,
            polarity,
            predictionError:
              prediction?.signedError ||
              0,
            inferred:
              metadata.inferred === true,
            domain:
              edge.domain ||
              metadata.domain ||
              '',
            entryId:
              edge.entryId ||
              metadata.entryId ||
              null
          
}

        ) || null;


      if (learned) {

        const generalized =
          this.generalization?.observe?.(
            learned,
            edge.domain ||
            metadata.domain ||
            learned?.metadata?.domain ||
            ''
          );


        if (generalized) {

          this.hypotheses?.observe?.(
            generalized.from,
            generalized.relation,
            generalized.to,
            {

              provenance: 'generalization',
              trust: 0.34,
              support:
                generalized.confidence,
              inferred: true,
              predictionError:
                0,
              domain:
                generalized.from
                  .replace(/^domain:/, '')
            
}

          );


          this.generalizations++;

        
}

      
}


      const uncertainty =
        learned
          ? 1 -
            Math.abs(
              learned.confidence -
              0.5
            ) *
            2
          : 1;


      this.replay?.schedule?.(
        `edge:${normalize(edge.from)}|${normalize(edge.relation)}|${normalize(edge.to)}`,
        {

          type: 'edge',
          from: edge.from,
          relation: edge.relation,
          to: edge.to,
          subject:
            edge.from,
          entryId:
            edge.entryId ||
            metadata.entryId ||
            null
        
}
,
        {

          uncertainty,
          usefulness:
            metadata.usefulness ??
            learned?.usefulness ??
            0.5,
          predictionError:
            prediction?.absError ||
            0,
          novelty:
            metadata.novelty ??
            0.5,
          retrievalFrequency:
            Math.min(
              1,
              (learned?.retrievals || 0) /
              12
            ),
          contradictionPressure:
            learned?.contradictionPressure ||
            0
        
}

      );


      if (hypothesis?.status === 'promotable') {

        this._promote(hypothesis);

      
}


      return {

        prediction,
        learned,
        hypothesis
      
}
;

    
}


    _promote(hypothesis) {

      if (!hypothesis || hypothesis.status !== 'promotable') return false;


      if (!this.hypotheses?.promote?.(hypothesis)) return false;


      const lexicalFrom =
        this._terms(
          hypothesis.from,
          this.options.maxLexicalTermsPerEdge
        );


      const lexicalTo =
        this._terms(
          hypothesis.to,
          this.options.maxLexicalTermsPerEdge
        );


      let updates = 0;


      for (const a of lexicalFrom) {

        for (const b of lexicalTo) {

          if (!a || !b || a === b) continue;


          this.rsl?.learnCorrelation?.(
            a,
            b,
            {

              gain:
                this.options
                  .rslPromotionGain *
                clamp(
                  hypothesis.confidence
                ),
              source:
                'learning_controller',
              symmetric:
                [
                  'related to',
                  'alias of',
                  'contrasts with'
                ].includes(
                  normalize(hypothesis.relation)
                ),
              reverseGain: 0.68
            
}

          );


          updates++;


          if (updates >= 12) break;

        
}


        if (updates >= 12) break;

      
}


      this.promotions++;


      this.lastPromotion = {

        at: Date.now(),
        from: hypothesis.from,
        relation: hypothesis.relation,
        to: hypothesis.to,
        confidence: hypothesis.confidence,
        rslUpdates: updates
      
}
;


      this.dirty = true;

      return true;

    
}


    _processTrusted(payload) {

      const trust =
        clamp(
          payload?.trust ??
          0.96
        );


      let learned = 0;


      for (const edge of (payload?.edges || []).slice(0, this.options.maxEdgesPerTrustedStudy)) {

        const result =
          this._learnEdge(
            edge,
            {

              provenance: 'traininginfo',
              trust,
              observed: 1,
              novelty: 0.55,
              usefulness: 0.76,
              domain: edge.domain,
              entryId: edge.entryId
            
}

          );


        if (result) learned++;

      
}


      return learned;

    
}


    _processUser(event) {

      if (event?.counterfactual === true || event?.analysisSummary?.counterfactual === true) {

        return 0;

      
}


      const terms =
        this._terms(
          event.text,
          16
        );


      let learned = 0;


      for (
        let i = 0;

        i < terms.length - 1;

        i++
      ) {

        const result =
          this._learnEdge(
            {

              from: terms[i],
              relation: 'cooccurs_with',
              to: terms[i + 1]
            
}
,
            {

              provenance: 'user',
              trust: 0.56,
              observed: 0.62,
              novelty: 0.45,
              usefulness: 0.45
            
}

          );


        if (result) learned++;

      
}


      if (
        this.options.promptGroundingStudies > 0 &&
        this.trainingKnowledge?.search
      ) {

        const search =
          this.trainingKnowledge.search(
            event.text,
            event.analysisSummary || {
}
,
            null,
            {

              limit:
                this.options
                  .promptGroundingStudies
            
}

          );


        for (const row of search?.rows || []) {

          const entry = row?.entry;

          const trust = clamp(
            entry?.Trust ??
            0
          );


          if (
            !entry ||
            trust <
              this.options
                .promptGroundingMinTrust
          ) {

            continue;

          
}


          learned +=
            this._processTrusted({

              trust,
              edges:
                this._trainingEdges(
                  entry
                )
            
}
);


          this.promptGroundedStudies++;

        
}

      
}


      return learned;

    
}


    _processResponse(event) {

      if (event?.counterfactual === true || event?.analysisSummary?.counterfactual === true) {

        this.evaluator?.observeOutcome?.({

          promptText: event.promptText,
          responseText: event.responseText,
          confidence: event.confidence,
          quality: event.quality,
          analysisSummary: event.analysisSummary || null
        
}
);

        return 0;

      
}


      const promptTerms =
        this._terms(
          event.promptText,
          10
        );


      const responseTerms =
        this._terms(
          event.responseText,
          12
        );


      const quality =
        clamp(
          event.quality ??
          (
            Number(event.confidence) ||
            0
          ) / 100
        );


      let learned = 0;


      for (const a of promptTerms.slice(0, 5)) {

        for (const b of responseTerms.slice(0, 5)) {

          if (a === b) continue;


          const result =
            this._learnEdge(
              {

                from: a,
                relation: 'responds_with',
                to: b
              
}
,
              {

                provenance: 'model_response',
                trust: 0.20,
                observed:
                  0.35 +
                  quality * 0.35,
                novelty: 0.35,
                usefulness: 0.35
              
}

            );


          if (result) learned++;

        
}

      
}


      this.evaluator?.observeOutcome?.({

        promptText: event.promptText,
        responseText: event.responseText,
        confidence: event.confidence,
        quality,
        analysisSummary: event.analysisSummary || null
      
}
);


      return learned;

    
}


    _processFailure(event) {

      if (event?.counterfactual === true || event?.analysisSummary?.counterfactual === true) {

        return 0;

      
}


      const promptTerms =
        this._terms(
          event.promptText,
          6
        );


      const responseTerms =
        this._terms(
          event.responseText,
          6
        );


      let learned = 0;


      for (const a of promptTerms.slice(0, 3)) {

        for (const b of responseTerms.slice(0, 3)) {

          const result =
            this._learnEdge(
              {

                from: a,
                relation: 'responds_with',
                to: b
              
}
,
              {

                provenance: 'failed_response',
                trust: 0.12,
                observed:
                  clamp(
                    1 -
                    (
                      Number(
                        event.confidence
                      ) || 0
                    ) /
                    100
                  ),
                polarity: -1,
                novelty: 0.30,
                usefulness: 0.50
              
}

            );


          if (result) learned++;

        
}

      
}


      return learned;

    
}


    _processExperience(event) {

      if (!event) return 0;


      if (event.type === 'user') {

        return this._processUser(event);

      
}


      if (event.type === 'response') {

        return this._processResponse(event);

      
}


      if (event.type === 'failure') {

        return this._processFailure(event);

      
}


      return 0;

    
}


    _verifyHypothesis(item) {

      if (!item?.from || !item?.relation || !item?.to) {

        return {

          tested: false,
          reason: 'invalid-hypothesis'
        
}
;

      
}


      if (String(item.from).startsWith('domain:')) {

        return {

          tested: false,
          reason: 'abstract-domain-hypothesis'
        
}
;

      
}


      const entry =
        this._entryForReplay({

          subject: item.from
        
}
);


      if (!entry) {

        return {

          tested: false,
          reason: 'no-trusted-entry'
        
}
;

      
}


      const trust =
        clamp(
          entry.Trust ??
          0.96
        );


      const edges =
        this._trainingEdges(entry);


      const relation =
        normalize(item.relation);


      const target =
        normalize(item.to);


      const exact =
        edges.find(edge =>
          normalize(edge.relation) === relation &&
          normalize(edge.to) === target
        );


      if (exact) {

        this._learnEdge(
          exact,
          {

            provenance: 'traininginfo',
            trust,
            observed: 1,
            novelty: 0.12,
            usefulness: 0.82,
            domain: exact.domain,
            entryId: exact.entryId
          
}

        );


        this.hypothesisTests++;

        this.hypothesisConfirmations++;


        return {

          tested: true,
          confirmed: true,
          contradicted: false,
          subject: item.from,
          relation,
          target,
          entryId: entry.ID || null
        
}
;

      
}


      const opposite =
        this.contradictions
          ?.opposites
          ?.get?.(
            relation
          );


      if (opposite) {

        const conflict =
          edges.find(edge =>
            normalize(edge.relation) ===
              normalize(opposite) &&
            normalize(edge.to) === target
          );


        if (conflict) {

          this._learnEdge(
            {

              from: item.from,
              relation,
              to: item.to,
              domain:
                conflict.domain,
              entryId:
                conflict.entryId
            
}
,
            {

              provenance: 'traininginfo',
              trust,
              observed: 1,
              polarity: -1,
              novelty: 0.12,
              usefulness: 0.82,
              domain: conflict.domain,
              entryId: conflict.entryId
            
}

          );


          this.hypotheses
            ?.markContested?.(
              item,
              `trusted-opposite:${opposite}`
            );


          this.hypothesisTests++;

          this.hypothesisContradictions++;


          return {

            tested: true,
            confirmed: false,
            contradicted: true,
            opposite,
            subject: item.from,
            relation,
            target,
            entryId: entry.ID || null
          
}
;

        
}

      
}


      this.hypothesisTests++;


      return {

        tested: true,
        confirmed: false,
        contradicted: false,
        inconclusive: true,
        subject: item.from,
        relation,
        target,
        entryId: entry.ID || null
      
}
;

    
}


    _testHypotheses(limit = 2) {

      const max =
        Math.max(
          0,
          limit | 0
        );


      if (!max) {

        return {

          tested: 0,
          confirmed: 0,
          contradicted: 0,
          inconclusive: 0
        
}
;

      
}


      const rows =
        this.hypotheses
          ?.candidates?.(
            Math.max(
              max * 3,
              max
            ),
            [
              'provisional',
              'promotable',
              'contested'
            ]
          ) || [];


      let tested = 0;

      let confirmed = 0;

      let contradicted = 0;

      let inconclusive = 0;


      for (const row of rows) {

        if (tested >= max) break;


        const result =
          this._verifyHypothesis(row);


        if (!result?.tested) {

          continue;

        
}


        tested++;


        if (result.confirmed) {

          confirmed++;

        
}
 else if (result.contradicted) {

          contradicted++;

        
}
 else {

          inconclusive++;

        
}

      
}


      if (tested) {

        this.dirty = true;

      
}


      return {

        tested,
        confirmed,
        contradicted,
        inconclusive
      
}
;

    
}


    _entryForReplay(payload) {

      const subject =
        normalize(
          payload?.subject ||
          payload?.from ||
          ''
        );


      if (!subject) return null;


      const direct =
        this.trainingKnowledge
          ?.bySubject
          ?.get?.(
            subject
          );


      if (Array.isArray(direct) && direct.length) {

        return direct[0];

      
}


      return (
        this.trainingKnowledge
          ?.entries
          ?.find?.(
            entry =>
              normalize(
                entry?.Subject
              ) === subject
          ) ||
        null
      );

    
}


    _conceptText(entry) {

      if (!entry) return '';


      const values = [
        entry.Subject,
        entry.Description,
        entry.Long_Definition,
        entry.Purpose,
        entry.Mechanism,
        ...(Array.isArray(entry.Keywords) ? entry.Keywords : []),
        ...(Array.isArray(entry.Related_To) ? entry.Related_To : []),
        ...(Array.isArray(entry.Properties) ? entry.Properties : []),
        ...(Array.isArray(entry.Evaluation_Criteria) ? entry.Evaluation_Criteria : []),
        ...(Array.isArray(entry.Failure_Modes) ? entry.Failure_Modes : []),
        ...(Array.isArray(entry.Tradeoffs) ? entry.Tradeoffs : [])
      ];


      return values
        .map(value => String(value || '').trim())
        .filter(Boolean)
        .join(' ')
        .slice(0, 2200);

    
}


    _meanTrustedPredictionError(edges, trust = 0.95) {

      let total = 0;

      let count = 0;


      for (const edge of (edges || []).slice(0, 24)) {

        const result =
          this.predictionError
            ?.evaluate?.(
              edge.from,
              edge.relation,
              edge.to,
              {

                observed: 1,
                trust,
                novelty: 0.08
              
}

            );


        if (!result) continue;


        total +=
          Number(
            result.absError
          ) || 0;


        count++;

      
}


      return count
        ? total / count
        : 1;

    
}


    _runReplay(item) {

      const entry =
        this._entryForReplay(
          item?.payload
        );


      if (!entry) {

        this.replay?.complete?.(
          item.key,
          {

            success: false,
            predictionError: 1
          
}

        );


        return false;

      
}


      const subject =
        normalize(
          entry.Subject
        );


      const edges =
        this._trainingEdges(
          entry
        );


      const trust =
        clamp(
          entry.Trust ??
          0.95
        );


      const before =
        this._meanTrustedPredictionError(
          edges,
          trust
        );


      const text =
        this._conceptText(
          entry
        );


      if (text) {

        this.correlation?.observeText?.(
          text,
          {

            source:
              'background_replay',
            confidence:
              clamp(
                entry.Trust ??
                0.95
              ) * 100,
            accepted: true,
            episodic: false,
            countCycle: false,
            gain:
              this.options
                .replayGain
          
}

        );

      
}


      for (const relation of edges.slice(0, 24)) {

        this._learnEdge(
          relation,
          {

            provenance:
              'background_replay',
            trust:
              trust * 0.72,
            observed: 0.72,
            novelty: 0.20,
            usefulness: 0.64,
            domain:
              relation.domain,
            entryId:
              relation.entryId
          
}

        );

      
}


      const after =
        this._meanTrustedPredictionError(
          edges,
          trust
        );


      this.evaluator?.observeReplay?.(
        before,
        after
      );


      this.replay?.complete?.(
        item.key,
        {

          success: true,
          predictionError: after
        
}

      );


      this.replayCycles++;


      this.lastReplay = {

        at: Date.now(),
        subject,
        beforeError: before,
        afterError: after,
        edgeCount: edges.length
      
}
;


      this.dirty = true;

      return true;

    
}


    _promoteAvailable(limit = 6) {

      const rows =
        this.hypotheses
          ?.promotable?.(
            limit
          ) || [];


      let promoted = 0;


      for (const item of rows) {

        if (this._promote(item)) promoted++;

      
}


      return promoted;

    
}


    testHypotheses(limit = 2) {

      return this._testHypotheses(limit);

    
}


    runContradictionCheck() {

      return this.contradictions?.run?.() || null;

    
}


    benchmarkLearning() {

      return this.evaluator?.benchmark?.(this.graph) || null;

    
}


    _rollbackCycleMutations() {

      let restored = 0;

      for (const item of this._cycleMutationJournal.slice().reverse()) {

        if (item.existed && item.snapshot) {

          if (this.graph?.restoreEdge?.(item.snapshot)) restored++;

        
}
 else if (item.key) {

          if (this.graph?.removeEdge?.(item.key)) restored++;

        
}

      
}

      if (restored) {

        this.regressionRollbacks += restored;

        this.evaluator?.noteRollback?.(restored);

      
}

      return restored;

    
}


    async backgroundTick(
      phase = 'WARM',
      epochAtStart = 0,
      getForegroundEpoch = null
    ) {

      const started =
        performance.now();


      if (!this.options.enabled) {

        return {

          enabled: false,
          phase,
          processed: 0,
          replayed: 0,
          ms: 0
        
}
;

      
}


      this._cycleMutationJournal = [];

      this._cycleMutationKeys = new Set();


      const regressionBefore =
        this.cycles % this.options.regressionEvery === 0
          ? this.evaluator?.benchmark?.(this.graph) || null
          : null;


      const budget =
        this._phaseBudget(
          phase
        );


      const eventLimit =
        this._phaseEvents(
          phase
        );


      let processed = 0;

      let replayed = 0;

      let edgeUpdates = 0;


      while (
        this.queue.length &&
        processed < eventLimit &&
        performance.now() - started < budget
      ) {

        if (
          this._interrupted(
            epochAtStart,
            getForegroundEpoch
          )
        ) {

          break;

        
}


        const item =
          this.queue.shift();


        if (!item) break;


        if (item.kind === 'trusted-knowledge') {

          edgeUpdates +=
            this._processTrusted(
              item
            );

        
}
 else if (item.kind === 'experience') {

          edgeUpdates +=
            this._processExperience(
              item.payload
            );

        
}


        processed++;

        this.processedEvents++;

      
}


      this._promoteAvailable(6);


      const replayLimit =
        this._phaseReplay(
          phase
        );


      if (
        replayLimit > 0 &&
        performance.now() - started < budget
      ) {

        const due =
          this.replay?.nextBatch?.(
            replayLimit
          ) || [];


        for (const item of due) {

          if (
            performance.now() - started >=
            budget ||
            this._interrupted(
              epochAtStart,
              getForegroundEpoch
            )
          ) {

            break;

          
}


          if (this._runReplay(item)) {

            replayed++;

          
}

        
}

      
}


      this.cycles++;


      let hypothesisTest = null;


      if (
        this.cycles %
          this.options
            .hypothesisTestEvery ===
          0 &&
        performance.now() - started < budget &&
        !this._interrupted(
          epochAtStart,
          getForegroundEpoch
        )
      ) {

        hypothesisTest =
          this._testHypotheses(
            this._phaseHypothesisTests(
              phase
            )
          );

      
}


      let contradiction = null;


      if (
        this.cycles %
          this.options
            .contradictionEvery ===
        0
      ) {

        contradiction =
          this.contradictions
            ?.run?.() ||
          null;

      
}


      let regression = null;

      if (regressionBefore) {

        const regressionAfter = this.evaluator?.benchmark?.(this.graph) || null;

        regression = this.evaluator?.compareRegression?.(
          regressionBefore,
          regressionAfter,
          {

            mutationCount: this._cycleMutationJournal.length,
            untrustedMutationCount: this._cycleMutationJournal.length
          
}

        ) || null;


        if (
          regression?.regression &&
          this.options.regressionRollbackEnabled &&
          this._cycleMutationJournal.length
        ) {

          const rollbackCount = this._rollbackCycleMutations();

          regression = {

            ...regression,
            rollbackCount,
            afterRollback: this.evaluator?.benchmark?.(this.graph) || null
          
}
;

        
}

        this.lastRegression = regression;

      
}


      this.lastCycleMs =
        performance.now() -
        started;


      if (
        processed ||
        replayed ||
        edgeUpdates ||
        contradiction?.contested ||
        hypothesisTest?.tested
      ) {

        this.dirty = true;

      
}


      this.lastEvent = {

        at: Date.now(),
        phase,
        processed,
        replayed,
        edgeUpdates,
        queueRemaining:
          this.queue.length,
        contradiction,
        hypothesisTest,
        regression,
        ms:
          this.lastCycleMs
      
}
;


      return {

        enabled: true,
        phase,
        processed,
        replayed,
        edgeUpdates,
        queueRemaining:
          this.queue.length,
        contradiction,
        hypothesisTest,
        regression,
        ms:
          this.lastCycleMs
      
}
;

    
}


    reset() {

      this.queue.length = 0;

      this.cycles = 0;

      this.processedEvents = 0;

      this.trustedStudies = 0;

      this.promotions = 0;

      this.replayCycles = 0;

      this.generalizations = 0;

      this.hypothesisTests = 0;

      this.hypothesisConfirmations = 0;

      this.hypothesisContradictions = 0;

      this.promptGroundedStudies = 0;

      this.interruptions = 0;

      this.dropped = 0;

      this.lastCycleMs = 0;

      this.lastEvent = null;

      this.lastReplay = null;

      this.lastPromotion = null;

      this.lastRegression = null;

      this.regressionRollbacks = 0;

      this._cycleMutationJournal = [];

      this._cycleMutationKeys = new Set();

      this.dirty = false;


      this.graph?.reset?.();

      this.predictionError?.reset?.();

      this.hypotheses?.reset?.();

      this.replay?.reset?.();

      this.generalization?.reset?.();

      this.contradictions?.reset?.();

      this.evaluator?.reset?.();


      return true;

    
}


    exportState() {

      return {

        schemaVersion: 2,
        cycles: this.cycles,
        processedEvents: this.processedEvents,
        trustedStudies: this.trustedStudies,
        promotions: this.promotions,
        replayCycles: this.replayCycles,
        generalizations: this.generalizations,
        hypothesisTests: this.hypothesisTests,
        hypothesisConfirmations: this.hypothesisConfirmations,
        hypothesisContradictions: this.hypothesisContradictions,
        promptGroundedStudies: this.promptGroundedStudies,
        regressionRollbacks: this.regressionRollbacks,
        lastRegression: this.lastRegression,
        interruptions: this.interruptions,
        dropped: this.dropped,
        graph:
          this.graph?.exportState?.() ||
          null,
        predictionError:
          this.predictionError?.exportState?.() ||
          null,
        hypotheses:
          this.hypotheses?.exportState?.() ||
          null,
        replay:
          this.replay?.exportState?.() ||
          null,
        generalization:
          this.generalization?.exportState?.() ||
          null,
        contradictions:
          this.contradictions?.exportState?.() ||
          null,
        evaluator:
          this.evaluator?.exportState?.() ||
          null,
        queue:
          this.queue.slice(-128),
        lastEvent:
          this.lastEvent,
        lastReplay:
          this.lastReplay,
        lastPromotion:
          this.lastPromotion,
        lastRegression: this.lastRegression,
        regressionRollbacks: this.regressionRollbacks
      
}
;

    
}


    importState(state) {

      if (!state || ![1, 2].includes(Number(state.schemaVersion))) return false;


      this.cycles = Math.max(0, Number(state.cycles) || 0);

      this.processedEvents = Math.max(0, Number(state.processedEvents) || 0);

      this.trustedStudies = Math.max(0, Number(state.trustedStudies) || 0);

      this.promotions = Math.max(0, Number(state.promotions) || 0);

      this.replayCycles = Math.max(0, Number(state.replayCycles) || 0);

      this.generalizations = Math.max(0, Number(state.generalizations) || 0);

      this.hypothesisTests = Math.max(0, Number(state.hypothesisTests) || 0);

      this.hypothesisConfirmations = Math.max(0, Number(state.hypothesisConfirmations) || 0);

      this.hypothesisContradictions = Math.max(0, Number(state.hypothesisContradictions) || 0);

      this.promptGroundedStudies = Math.max(0, Number(state.promptGroundedStudies) || 0);

      this.interruptions = Math.max(0, Number(state.interruptions) || 0);

      this.dropped = Math.max(0, Number(state.dropped) || 0);


      this.graph?.importState?.(
        state.graph
      );


      this.predictionError?.importState?.(
        state.predictionError
      );


      this.hypotheses?.importState?.(
        state.hypotheses
      );


      this.replay?.importState?.(
        state.replay
      );


      this.generalization?.importState?.(
        state.generalization
      );


      this.contradictions?.importState?.(
        state.contradictions
      );


      this.evaluator?.importState?.(
        state.evaluator
      );


      this.queue =
        Array.isArray(state.queue)
          ? state.queue.slice(-128)
          : [];


      this.lastEvent =
        state.lastEvent ||
        null;


      this.lastReplay =
        state.lastReplay ||
        null;


      this.lastPromotion =
        state.lastPromotion ||
        null;


      this.lastRegression = state.lastRegression || null;

      this.regressionRollbacks = Math.max(0, Number(state.regressionRollbacks) || 0);

      this._cycleMutationJournal = [];

      this._cycleMutationKeys = new Set();


      this.dirty = false;

      return true;

    
}


    status() {

      return {

        ready: true,
        role: 'prediction-error-hypothesis-replay-generalization-learning-controller',
        enabled: this.options.enabled,
        cycles: this.cycles,
        processedEvents: this.processedEvents,
        trustedStudies: this.trustedStudies,
        promotions: this.promotions,
        replayCycles: this.replayCycles,
        generalizations: this.generalizations,
        hypothesisTests: this.hypothesisTests,
        hypothesisConfirmations: this.hypothesisConfirmations,
        hypothesisContradictions: this.hypothesisContradictions,
        promptGroundedStudies: this.promptGroundedStudies,
        interruptions: this.interruptions,
        dropped: this.dropped,
        queue: this.queue.length,
        lastCycleMs: this.lastCycleMs,
        lastEvent: this.lastEvent,
        lastReplay: this.lastReplay,
        lastPromotion: this.lastPromotion,
        graph: this.graph?.status?.() || null,
        predictionError: this.predictionError?.status?.() || null,
        hypotheses: this.hypotheses?.status?.() || null,
        replay: this.replay?.status?.() || null,
        generalization: this.generalization?.status?.() || null,
        contradictions: this.contradictions?.status?.() || null,
        evaluator: this.evaluator?.status?.() || null,
        safeguards: {

          foregroundHeavyLearning: false,
          inferredClaimsBecomeFactsAutomatically: false,
          modelResponsesCountAsTrustedFacts: false,
          trustedTrainingInfoCanConsolidate: true,
          generalizationsStayHypotheses: true,
          provisionalHypothesesSelfTestAgainstTrainingInfo: true,
          replayMeasuresPredictionErrorBeforeAndAfter: true
        
}

      
}
;

    
}

  
}


  globalThis.VilotLearningController = VilotLearningController;

}
)();

