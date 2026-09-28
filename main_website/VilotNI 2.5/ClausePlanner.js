/**
 * VilotNI 2.5 - ClausePlanner.js
 *
 * Collapses semantic propositions from the four branches before surface wording.
 * The continuous StateProcessor is reused at proposition level.
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


  class VilotClausePlanner {

    constructor(processor, stateProcessor, options = {
}
) {

      this.processor = processor;

      this.stateProcessor = stateProcessor;


      this.options = {

        semanticConfidenceWindowPct:
          Math.max(
            0.1,
            Number(
              options.semanticConfidenceWindowPct
            ) || 2
          ),

        minSemanticAgreement:
          Math.max(
            2,
            Math.min(
              4,
              options.semanticMinAgreement | 0 ||
              2
            )
          ),

        maxClauses:
          Math.max(
            1,
            Math.min(
              6,
              options.semanticMaxClauses | 0 ||
              3
            )
          ),

        semanticSimilarityThreshold:
          clamp(
            options.semanticSimilarityThreshold ??
            0.28
          )
      
}
;


      this.plans = 0;

      this.lastDiagnostics = null;

    
}


    _tokens(text) {

      const tokens =
        this.processor?.tokenize?.(
          String(text || ''),
          96
        ) || [];


      if (tokens.length) {

        return new Set(
          tokens
            .filter(
              token =>
                token?.pos !==
                'Punct'
            )
            .map(
              token =>
                normalize(
                  token?.lower ||
                  token?.surface
                )
            )
            .filter(Boolean)
        );

      
}


      return new Set(
        normalize(text)
          .split(/\s+/)
          .filter(Boolean)
      );

    
}


    _jaccard(a, b) {

      if (!a.size || !b.size) {

        return 0;

      
}


      let intersection = 0;


      for (const token of a) {

        if (b.has(token)) {

          intersection++;

        
}

      
}


      const union =
        a.size +
        b.size -
        intersection;


      return union
        ? intersection / union
        : 0;

    
}


    _semanticSimilarity(a, b) {

      if (!a || !b) return 0;


      if (
        normalize(a.subject) !==
        normalize(b.subject)
      ) {

        return 0;

      
}


      if (a.type !== b.type) {

        const compatiblePairs =
          new Set([
            'definition|property',
            'property|definition',
            'purpose|capability',
            'capability|purpose',
            'mechanism|cause',
            'cause|mechanism',
            'relation|part_of',
            'part_of|relation',
            'relation|has_part',
            'has_part|relation'
          ]);


        if (
          !compatiblePairs.has(
            `${a.type}|${b.type}`
          )
        ) {

          return 0;

        
}

      
}


      const textA =
        this._tokens(
          a.value
        );


      const textB =
        this._tokens(
          b.value
        );


      const overlap =
        this._jaccard(
          textA,
          textB
        );


      if (
        normalize(a.value) ===
        normalize(b.value)
      ) {

        return 1;

      
}


      let association = 0;


      const firstA =
        Array.from(textA)[0];


      const firstB =
        Array.from(textB)[0];


      if (
        firstA &&
        firstB
      ) {

        association =
          clamp(
            this.processor
              ?.associationScore?.(
                firstA,
                firstB,
                {

                  bidirectional:
                    true
                
}

              ) || 0
          );

      
}


      return clamp(
        overlap * 0.72 +
        association * 0.28
      );

    
}


    _intentOrder(
      analysis,
      promptText = ''
    ) {

      const intent =
        String(
          analysis?.intent ||
          ''
        ).toLowerCase();


      const requested =
        String(
          analysis?.requestedSlot ||
          ''
        ).toLowerCase();


      const lower =
        String(
          promptText ||
          analysis?.text ||
          ''
        ).toLowerCase();


      const explanationMode =
        analysis?.explanationMode ||
        null;

      const complexRoles = analysis?.complexLanguage?.requestedRoles || [];
      const complexFeatures = analysis?.complexLanguage?.features || {};

      if (complexRoles.includes('cause') && complexRoles.includes('mechanism')) {
        return ['cause','mechanism','effect','purpose','relation','definition'];
      }
      if (complexFeatures.comparison || analysis?.intent === 'comparison') {
        return ['contrast','relation','definition','property','mechanism'];
      }


      if (explanationMode === 'both') {

        return [
          'cause',
          'mechanism',
          'effect',
          'purpose',
          'relation',
          'definition'
        ];

      
}


      if (explanationMode === 'why') {

        return [
          'cause',
          'mechanism',
          'effect',
          'relation',
          'purpose',
          'definition'
        ];

      
}


      if (explanationMode === 'how') {

        return [
          'mechanism',
          'composition',
          'has_part',
          'capability',
          'effect',
          'purpose',
          'definition'
        ];

      
}


      if (
        /\b(compare|difference|different|versus|vs\.?)\b/
          .test(lower)
      ) {

        return [
          'contrast',
          'definition',
          'relation',
          'property'
        ];

      
}


      if (
        intent === 'mechanism' ||
        requested === 'mechanism'
      ) {

        return [
          'mechanism',
          'definition',
          'purpose',
          'relation',
          'effect'
        ];

      
}


      if (
        intent === 'causal' ||
        requested === 'cause'
      ) {

        return [
          'cause',
          'mechanism',
          'effect',
          'definition',
          'relation'
        ];

      
}


      if (
        intent === 'list'
      ) {

        return [
          'example',
          'capability',
          'has_part',
          'relation',
          'property'
        ];

      
}


      if (
        intent === 'boolean' ||
        requested === 'truth'
      ) {

        return [
          'property',
          'definition',
          'capability',
          'limitation',
          'relation'
        ];

      
}


      if (
        intent === 'definition' ||
        requested === 'identity'
      ) {

        return [
          'definition',
          'purpose',
          'mechanism',
          'property',
          'capability',
          'relation'
        ];

      
}


      return [
        'definition',
        'purpose',
        'mechanism',
        'property',
        'capability',
        'relation'
      ];

    
}


    _targetClauseCount(
      analysis,
      promptText = '',
      composition = null
    ) {

      const lower =
        String(
          promptText ||
          analysis?.text ||
          ''
        ).toLowerCase();


      const wordCount =
        Array.isArray(
          analysis?.words
        )
          ? analysis.words.length
          : lower
              .split(/\s+/)
              .filter(Boolean)
              .length;


      const detailed =
        /\b(detail|detailed|deep|fully|thorough|explain|how exactly|why exactly|more about)\b/
          .test(lower);


      const availableSemanticSlots =
        Object.values(
          composition?.slots ||
          {
}

        ).filter(
          rows =>
            Array.isArray(rows) &&
            rows.length
        ).length;


      const availableCount =
        Math.max(
          1,
          Math.min(
            this.options.maxClauses,
            availableSemanticSlots ||
            1
          )
        );

      const requestedRoleCount = new Set(analysis?.complexLanguage?.requestedRoles || []).size;
      const structuralClauseCount = Math.min(this.options.maxClauses, Number(analysis?.complexLanguage?.clauseCount || 0));
      const structuralDemand = Math.min(this.options.maxClauses, Math.max(requestedRoleCount, structuralClauseCount > 1 ? Math.min(3, structuralClauseCount) : 0));


      const intent =
        String(
          analysis?.intent ||
          ''
        ).toLowerCase();


      const explanationMode =
        analysis?.explanationMode ||
        null;


      if (structuralDemand >= 2 && Number(analysis?.structuralComplexity || 0) >= 0.30) {
        return Math.min(this.options.maxClauses, Math.max(2, Math.min(Math.max(structuralDemand, requestedRoleCount), Math.max(2, availableCount))));
      }

      if (explanationMode === 'both') {

        return Math.min(
          this.options.maxClauses,
          Math.max(3, Math.min(4, availableCount))
        );

      
}


      if (explanationMode === 'why' || explanationMode === 'how') {

        return Math.min(
          this.options.maxClauses,
          Math.max(2, Math.min(detailed ? 4 : 3, availableCount))
        );

      
}


      if (detailed) {

        return Math.min(
          this.options.maxClauses,
          Math.max(
            2,
            Math.min(
              4,
              availableCount
            )
          )
        );

      
}


      if (
        intent === 'definition' &&
        wordCount <= 6
      ) {

        // A compact definition may now carry a second supported idea, such as
        // purpose, mechanism, property, or relationship, instead of forcing the
        // answer to stop after one semantic statement.
        return availableCount >= 2
          ? 2
          : 1;

      
}


      if (
        intent === 'mechanism' ||
        intent === 'causal'
      ) {

        return Math.min(
          this.options.maxClauses,
          Math.max(
            1,
            Math.min(
              2,
              availableCount
            )
          )
        );

      
}


      return Math.min(
        this.options.maxClauses,
        Math.max(
          1,
          Math.min(
            2,
            availableCount
          )
        )
      );

    
}


    _collapseSlot(
      slot,
      candidates,
      slotIndex
    ) {

      if (!candidates?.length) {

        return null;

      
}


      const rows =
        candidates
          .slice()
          .sort(
            (a, b) =>
              b.confidence -
              a.confidence
          );


      if (
        this.stateProcessor &&
        rows.length >=
          this.options
            .minSemanticAgreement
      ) {

        const result =
          this.stateProcessor
            .collapseCandidates(
              rows,
              {

                confidenceWindow:
                  this.options
                    .semanticConfidenceWindowPct /
                  100,

                minAgreement:
                  this.options
                    .minSemanticAgreement,

                mixVariant:
                  slotIndex,

                getState:
                  candidate =>
                    this.stateProcessor
                      .fromPercent(
                        candidate
                          ?.confidence ||
                        0
                      ),

                compatible:
                  (a, b) =>
                    this
                      ._semanticSimilarity(
                        a,
                        b
                      ),

                knowledgeScore:
                  candidate =>
                    clamp(
                      candidate
                        ?.trust ||
                      0
                    ),

                focusScore:
                  candidate =>
                    clamp(
                      candidate
                        ?.branchSupport ??
                      0.5
                    ),

                transitionScore:
                  candidate =>
                    clamp(
                      candidate
                        ?.rowRelevance ??
                      0.5
                    ),

                reliabilityScore:
                  candidate =>
                    clamp(
                      candidate?.trust ??
                      0.5
                    ),

                contextScore:
                  candidate =>
                    clamp(
                      candidate?.branchSupport ??
                      0.5
                    ),

                semanticScore:
                  candidate =>
                    clamp(
                      candidate?.rowRelevance ??
                      0.5
                    ),

                noveltyScore:
                  candidate =>
                    clamp(
                      0.35 +
                      (1 - (candidate?.trust ?? 0.5)) * 0.15 +
                      (candidate?.branchSupport ?? 0.5) * 0.25 +
                      (candidate?.rowRelevance ?? 0.5) * 0.25
                    ),

                certaintyScore:
                  candidate =>
                    clamp(
                      (candidate?.trust ?? 0.5) * 0.45 +
                      (candidate?.branchSupport ?? 0.5) * 0.30 +
                      (candidate?.rowRelevance ?? 0.5) * 0.25
                    )
              
}

            );


        if (result?.selected) {

          const support =
            rows.filter(
              other =>
                this._semanticSimilarity(
                  result.selected,
                  other
                ) >=
                this.options
                  .semanticSimilarityThreshold
            ).length;


          return {

            proposition:
              result.selected,

            support,

            state:
              result.state,

            score:
              result.score,

            diagnostics:
              result.diagnostics
          
}
;

        
}

      
}


      // If no semantic interference cluster survives, only allow a fallback
      // when at least two branches independently support a close proposition.
      const best =
        rows[0];


      const support =
        rows.filter(
          other =>
            this._semanticSimilarity(
              best,
              other
            ) >=
            this.options
              .semanticSimilarityThreshold
        ).length;


      if (
        support <
        this.options
          .minSemanticAgreement
      ) {

        return null;

      
}


      return {

        proposition:
          best,
        support,
        state:
          clamp(
            best.confidence /
            100
          ),
        score:
          clamp(
            best.confidence /
            100
          ),
        diagnostics:
          null
      
}
;

    
}


    plan(
      composition,
      analysis,
      promptText = '',
      options = {
}

    ) {

      const started =
        performance.now();


      const deadlineAt = Number.isFinite(Number(options.deadlineAt))
        ? Number(options.deadlineAt)
        : Infinity;

      const deadlineReserveMs = Math.max(0.20, Number(options.deadlineReserveMs) || 0.55);

      let deadlineTruncated = false;


      const order =
        this._intentOrder(
          analysis,
          promptText
        );


      const targetCount =
        this._targetClauseCount(
          analysis,
          promptText,
          composition
        );


      const clauses = [];

      const usedSignatures =
        new Set();


      let stateProcessorMs = 0;

      let stateOperations = 0;

      let totalSupport = 0;


      for (
        let orderIndex = 0;

        orderIndex <
          order.length;

        orderIndex++
      ) {

        if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt - deadlineReserveMs) {

          deadlineTruncated = true;

          break;

        
}

        if (
          clauses.length >=
          targetCount
        ) {

          break;

        
}


        const slot =
          order[orderIndex];


        const candidates =
          composition
            ?.slots?.[slot] ||
          [];


        const collapsed =
          this._collapseSlot(
            slot,
            candidates,
            orderIndex
          );


        if (!collapsed) {

          continue;

        
}


        const proposition =
          collapsed.proposition;


        const signature =
          [
            proposition.type,
            normalize(
              proposition.subject
            ),
            normalize(
              proposition.value
            )
          ].join('|');


        if (
          usedSignatures.has(
            signature
          )
        ) {

          continue;

        
}


        usedSignatures.add(
          signature
        );


        clauses.push({

          index:
            clauses.length,

          type:
            proposition.type,

          relation:
            proposition.relation,

          subject:
            proposition.subject ||
            composition?.subject ||
            'concept',

          value:
            proposition.value,

          aliases:
            proposition.aliases ||
            [],

          completeClause:
            proposition
              .completeClause ===
            true,

          confidence:
            proposition.confidence,

          branchConfidence:
            proposition
              .branchConfidence,

          support:
            collapsed.support,

          collapsedState:
            collapsed.state,

          collapseScore:
            collapsed.score,

          trust:
            proposition.trust,

          sourceEntryId:
            proposition
              .sourceEntryId,

          sourceSubject:
            proposition
              .sourceSubject,

          sourceType:
            proposition
              .sourceType || '',

          verification:
            proposition
              .verification || null
        
}
);


        totalSupport +=
          collapsed.support;


        stateProcessorMs +=
          Number(
            collapsed
              ?.diagnostics
              ?.processorMs
          ) || 0;


        stateOperations +=
          Number(
            collapsed
              ?.diagnostics
              ?.operations
          ) || 0;

      
}


      const availableBranches =
        Math.max(
          1,
          composition
            ?.frames
            ?.length ||
          1
        );


      const agreementRatio =
        clauses.length
          ? clamp(
              (
                totalSupport /
                clauses.length
              ) /
              availableBranches
            )
          : 0;


      const confidence =
        clauses.length
          ? clauses.reduce(
              (sum, clause) =>
                sum +
                Number(
                  clause.confidence
                ),
              0
            ) /
            clauses.length
          : 0;


      const plan = {

        subject:
          composition?.subject ||
          clauses[0]?.subject ||
          'concept',

        intent:
          analysis?.intent ||
          'content',

        requestedSlot:
          analysis?.requestedSlot ||
          'content',

        order,

        clauses,

        targetClauseCount:
          targetCount,

        agreementRatio:
          Number(
            agreementRatio.toFixed(4)
          ),

        confidence:
          Number(
            confidence.toFixed(4)
          ),

        diagnostics: {

          plannerMs:
            performance.now() -
            started,

          deadlineTruncated,

          clauseCount:
            clauses.length,

          targetClauseCount:
            targetCount,

          agreementRatio:
            Number(
              agreementRatio.toFixed(4)
            ),

          stateProcessorMs,

          stateOperations
        
}

      
}
;


      this.plans++;

      this.lastDiagnostics =
        plan.diagnostics;


      return plan;

    
}


    status() {

      return {

        ready: true,
        role:
          'semantic-clause-collapse-planner',
        plans:
          this.plans,
        lastDiagnostics:
          this.lastDiagnostics,
        architecture: {

          semanticInterferenceBeforeWording:
            true,
          continuousStateProcessor:
            Boolean(
              this.stateProcessor
            ),
          extraDecoderPasses:
            0
        
}

      
}
;

    
}

  
}


  globalThis.VilotClausePlanner =
    VilotClausePlanner;

}
)();

