/**
 * VilotNI 2.5 - IdeaFusion.js
 *
 * Universal multi-idea sentence planning.
 *
 * This is topic-agnostic. It does not know or special-case CPU, AI, weather,
 * biology, or any other subject.
 *
 * ClausePlanner still decides which meanings are supported. IdeaFusion only
 * decides whether two or more already-approved semantic ideas can safely share
 * one sentence.
 *
 * Same-subject ideas can fuse directly. Different-subject ideas can also fuse
 * when their approved semantic clauses are related by explicit mention,
 * semantic association, causal/contrast/relation structure, or shared concept
 * support. Unrelated clauses stay in separate sentences.
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


  class VilotIdeaFusion {

    constructor(processor, options = {
}
) {

      this.processor = processor;


      this.options = {

        enabled:
          options.ideaFusionEnabled !== false,

        maxIdeasPerSentence:
          Math.max(
            2,
            Math.min(
              5,
              options.ideaFusionMaxIdeasPerSentence | 0 ||
              3
            )
          ),

        simpleMaxIdeasPerSentence:
          Math.max(
            2,
            Math.min(
              4,
              options.ideaFusionSimpleMaxIdeasPerSentence | 0 ||
              2
            )
          ),

        minSupport:
          Math.max(
            1,
            options.ideaFusionMinSupport | 0 ||
            2
          ),

        confidenceWindowPct:
          Math.max(
            1,
            Number(options.ideaFusionConfidenceWindowPct) ||
            6
          ),

        maxEstimatedWords:
          Math.max(
            18,
            options.ideaFusionMaxSentenceWords | 0 ||
            46
          ),

        allowCrossSubject:
          options.ideaFusionAllowCrossSubject !== false,

        crossSubjectMinAssociation:
          clamp(
            options.ideaFusionCrossSubjectMinAssociation ??
            0.18
          ),

        crossSubjectMinSharedTokenRatio:
          clamp(
            options.ideaFusionCrossSubjectMinSharedTokenRatio ??
            0.14
          ),

        maxSubjectChangesPerSentence:
          Math.max(
            1,
            Math.min(
              3,
              options.ideaFusionMaxSubjectChangesPerSentence | 0 ||
              2
            )
          )
      
}
;


      this.plans = 0;

      this.lastDiagnostics = null;

    
}


    _tokens(text) {

      const lexical =
        this.processor?.tokenize?.(
          String(text || ''),
          96
        ) || [];


      if (lexical.length) {

        return lexical
          .filter(row => row?.pos !== 'Punct')
          .map(row => normalize(row?.lower || row?.surface))
          .filter(Boolean);

      
}


      return normalize(text)
        .split(/\s+/)
        .filter(Boolean);

    
}


    _tokenSet(text) {

      return new Set(this._tokens(text));

    
}


    _jaccard(a, b) {

      if (!a?.size || !b?.size) return 0;


      let intersection = 0;


      for (const token of a) {

        if (b.has(token)) intersection++;

      
}


      const union =
        a.size +
        b.size -
        intersection;


      return union
        ? intersection / union
        : 0;

    
}


    _estimatedWords(clause) {

      return String(
        clause?.value ||
        ''
      )
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .length + 4;

    
}


    _sameSubject(a, b) {

      const left = normalize(a?.subject);

      const right = normalize(b?.subject);


      return Boolean(
        left &&
        right &&
        left === right
      );

    
}


    _subjectMention(a, b) {

      const aSubject =
        normalize(a?.subject);


      const bSubject =
        normalize(b?.subject);


      const aValue =
        normalize(a?.value);


      const bValue =
        normalize(b?.value);


      if (!aSubject || !bSubject) {

        return 0;

      
}


      if (
        aValue.includes(bSubject) ||
        bValue.includes(aSubject)
      ) {

        return 1;

      
}


      return 0;

    
}


    _subjectAssociation(a, b) {

      const left =
        this._tokens(
          a?.subject
        );


      const right =
        this._tokens(
          b?.subject
        );


      let best = 0;


      for (const x of left) {

        for (const y of right) {

          if (x === y) {

            best = 1;

            continue;

          
}


          const score =
            Number(
              this.processor
                ?.associationScore?.(
                  x,
                  y,
                  {

                    bidirectional:
                      true
                  
}

                )
            ) || 0;


          if (score > best) {

            best = score;

          
}

        
}

      
}


      return clamp(best);

    
}


    _contentOverlap(a, b) {

      const left =
        this._tokenSet(
          [
            a?.subject,
            a?.value
          ]
            .filter(Boolean)
            .join(' ')
        );


      const right =
        this._tokenSet(
          [
            b?.subject,
            b?.value
          ]
            .filter(Boolean)
            .join(' ')
        );


      return this._jaccard(
        left,
        right
      );

    
}


    _typedRelationSupport(a, b) {

      const pair =
        new Set([
          String(a?.type || '').toLowerCase(),
          String(b?.type || '').toLowerCase()
        ]);


      // These are generic semantic structures, not subject-specific rules.
      if (
        pair.has('cause') ||
        pair.has('effect') ||
        pair.has('contrast') ||
        pair.has('relation') ||
        pair.has('part_of') ||
        pair.has('has_part') ||
        pair.has('mechanism')
      ) {

        return 1;

      
}


      return 0;

    
}


    _crossSubjectCompatibility(a, b) {

      if (
        !this.options.allowCrossSubject
      ) {

        return {

          compatible: false,
          score: 0,
          reason:
            'cross-subject-disabled'
        
}
;

      
}


      const mention =
        this._subjectMention(
          a,
          b
        );


      const association =
        this._subjectAssociation(
          a,
          b
        );


      const overlap =
        this._contentOverlap(
          a,
          b
        );


      const typed =
        this._typedRelationSupport(
          a,
          b
        );


      const score =
        clamp(
          mention * 0.42 +
          association * 0.26 +
          overlap * 0.20 +
          typed * 0.12
        );


      const compatible =
        mention >= 1 ||
        association >=
          this.options
            .crossSubjectMinAssociation ||
        overlap >=
          this.options
            .crossSubjectMinSharedTokenRatio ||
        (
          typed >= 1 &&
          (
            association > 0 ||
            overlap > 0
          )
        );


      return {

        compatible,
        score,
        mention,
        association,
        overlap,
        typed
      
}
;

    
}


    _detailed(analysis, promptText = '') {

      const text = String(
        promptText ||
        analysis?.text ||
        ''
      ).toLowerCase();


      return /\b(detail|detailed|deep|fully|thorough|explain|how exactly|why exactly|more about|compare|relationship|related)\b/
        .test(text);

    
}


    _connector(previous, current) {

      const currentType =
        String(current?.type || '')
          .toLowerCase();


      if (
        currentType === 'limitation' ||
        currentType === 'contrast'
      ) {

        return 'but';

      
}


      const previousType =
        String(previous?.type || '')
          .toLowerCase();


      if (
        previousType === 'cause' &&
        currentType === 'effect'
      ) {

        return 'and';

      
}


      return 'and';

    
}


    _subjectChanges(group, nextClause) {

      if (!group?.ideas?.length) {

        return 0;

      
}


      let changes = 0;


      for (
        let i = 1;

        i < group.ideas.length;

        i++
      ) {

        if (
          !this._sameSubject(
            group.ideas[i - 1],
            group.ideas[i]
          )
        ) {

          changes++;

        
}

      
}


      const previous =
        group.ideas[
          group.ideas.length - 1
        ];


      if (
        previous &&
        nextClause &&
        !this._sameSubject(
          previous,
          nextClause
        )
      ) {

        changes++;

      
}


      return changes;

    
}


    _compatible(previous, current, group) {

      if (!previous || !current) {

        return {

          compatible: false,
          reason: 'missing-clause'
        
}
;

      
}


      const sameSubject =
        this._sameSubject(
          previous,
          current
        );


      const cross =
        sameSubject
          ? {

              compatible: true,
              score: 1,
              reason:
                'same-subject'
            
}

          : this._crossSubjectCompatibility(
              previous,
              current
            );


      if (!cross.compatible) {

        return {

          compatible: false,
          reason:
            'unrelated-subjects',
          cross
        
}
;

      
}


      const confidenceGap =
        Math.abs(
          (Number(previous?.confidence) || 0) -
          (Number(current?.confidence) || 0)
        );


      const supported =
        Math.min(
          Number(previous?.support) || 0,
          Number(current?.support) || 0
        ) >=
        this.options.minSupport;


      if (
        confidenceGap >
          this.options.confidenceWindowPct &&
        !supported
      ) {

        return {

          compatible: false,
          reason:
            'confidence-gap',
          cross
        
}
;

      
}


      const estimatedWords =
        (group?.estimatedWords || 0) +
        this._estimatedWords(current);


      if (
        estimatedWords >
        this.options.maxEstimatedWords
      ) {

        return {

          compatible: false,
          reason:
            'sentence-length',
          cross
        
}
;

      
}


      const subjectChanges =
        this._subjectChanges(
          group,
          current
        );


      if (
        subjectChanges >
        this.options
          .maxSubjectChangesPerSentence
      ) {

        return {

          compatible: false,
          reason:
            'subject-change-limit',
          cross
        
}
;

      
}


      // Examples can become list-heavy and are kept separate after two existing
      // ideas. This is structural, not topic-specific.
      if (
        current?.type === 'example' &&
        (group?.ideas?.length || 0) >= 2
      ) {

        return {

          compatible: false,
          reason:
            'example-density',
          cross
        
}
;

      
}


      return {

        compatible: true,
        reason:
          sameSubject
            ? 'same-subject'
            : 'related-cross-subject',
        cross,
        subjectChanges
      
}
;

    
}


    plan(
      clausePlan,
      analysis,
      promptText = ''
    ) {

      const started =
        performance.now();


      const clauses =
        clausePlan?.clauses ||
        [];


      if (
        !this.options.enabled ||
        clauses.length < 2
      ) {

        const result = {

          ...clausePlan,
          sentences:
            clauses.map(
              clause => ({

                ideas: [clause],
                connectors: [],
                ideaCount: 1,
                subject:
                  clause?.subject ||
                  clausePlan?.subject ||
                  '',
                estimatedWords:
                  this._estimatedWords(
                    clause
                  ),
                crossSubjectLinks: []
              
}
)
            ),
          diagnostics: {

            ...(clausePlan?.diagnostics || {
}
),
            ideaFusion: {

              enabled:
                this.options.enabled,
              universal:
                true,
              sentenceCount:
                clauses.length,
              ideaCount:
                clauses.length,
              fusedIdeaCount: 0,
              multiIdeaSentenceCount: 0,
              crossSubjectFusionCount: 0,
              fusionMs:
                performance.now() -
                started
            
}

          
}

        
}
;


        this.plans++;

        this.lastDiagnostics =
          result.diagnostics.ideaFusion;


        return result;

      
}


      const maxIdeas =
        this._detailed(
          analysis,
          promptText
        )
          ? this.options
              .maxIdeasPerSentence
          : this.options
              .simpleMaxIdeasPerSentence;


      const sentences = [];

      let current = null;


      const begin = clause => ({

        ideas: [clause],
        connectors: [],
        ideaCount: 1,
        subject:
          clause?.subject ||
          clausePlan?.subject ||
          '',
        estimatedWords:
          this._estimatedWords(
            clause
          ),
        crossSubjectLinks: []
      
}
);


      for (const clause of clauses) {

        if (!current) {

          current =
            begin(clause);

          continue;

        
}


        const previous =
          current.ideas[
            current.ideas.length - 1
          ];


        const compatibility =
          this._compatible(
            previous,
            clause,
            current
          );


        if (
          current.ideas.length <
            maxIdeas &&
          compatibility.compatible
        ) {

          current.connectors.push(
            this._connector(
              previous,
              clause
            )
          );


          if (
            !this._sameSubject(
              previous,
              clause
            )
          ) {

            current
              .crossSubjectLinks
              .push({

                from:
                  previous?.subject ||
                  '',
                to:
                  clause?.subject ||
                  '',
                score:
                  Number(
                    (
                      compatibility
                        ?.cross
                        ?.score ||
                      0
                    ).toFixed(4)
                  ),
                reason:
                  compatibility.reason
              
}
);

          
}


          current.ideas.push(
            clause
          );


          current.ideaCount =
            current.ideas.length;


          current.estimatedWords +=
            this._estimatedWords(
              clause
            );

        
}
 else {

          sentences.push(
            current
          );

          current =
            begin(clause);

        
}

      
}


      if (current) {

        sentences.push(
          current
        );

      
}


      const multiIdeaSentenceCount =
        sentences.filter(
          sentence =>
            sentence.ideaCount >= 2
        ).length;


      const fusedIdeaCount =
        sentences.reduce(
          (sum, sentence) =>
            sum +
            Math.max(
              0,
              sentence.ideaCount - 1
            ),
          0
        );


      const crossSubjectFusionCount =
        sentences.reduce(
          (sum, sentence) =>
            sum +
            (
              sentence
                .crossSubjectLinks
                ?.length ||
              0
            ),
          0
        );


      const diagnostics = {

        enabled: true,
        universal: true,
        topicSpecificRules: false,
        sentenceCount:
          sentences.length,
        ideaCount:
          clauses.length,
        fusedIdeaCount,
        multiIdeaSentenceCount,
        crossSubjectFusionCount,
        maxIdeasPerSentence:
          maxIdeas,
        fusionRatio:
          clauses.length
            ? clamp(
                fusedIdeaCount /
                clauses.length
              )
            : 0,
        fusionMs:
          performance.now() -
          started
      
}
;


      const result = {

        ...clausePlan,
        sentences,
        diagnostics: {

          ...(clausePlan?.diagnostics || {
}
),
          ideaFusion:
            diagnostics
        
}

      
}
;


      this.plans++;

      this.lastDiagnostics =
        diagnostics;


      return result;

    
}


    status() {

      return {

        ready: true,
        role:
          'universal-multi-idea-semantic-sentence-fusion',
        plans:
          this.plans,
        lastDiagnostics:
          this.lastDiagnostics,
        architecture: {

          topicAgnostic:
            true,
          fusesApprovedSemanticClauses:
            true,
          sameSubjectFusion:
            true,
          relatedCrossSubjectFusion:
            true,
          unrelatedIdeasStaySeparate:
            true,
          inventsNewFacts:
            false,
          supportsTwoOrMoreIdeasPerSentence:
            true,
          maxIdeasPerSentence:
            this.options.maxIdeasPerSentence,
          extraDecoderPasses:
            0
        
}

      
}
;

    
}

  
}


  globalThis.VilotIdeaFusion =
    VilotIdeaFusion;

}
)();

