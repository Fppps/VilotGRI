/**
 * VilotNI 2.5 - SemanticComposer.js
 *
 * Meaning-first branch composer.
 *
 * Converts the four projected decoder paths plus TrainingInfo knowledge into
 * structured semantic propositions before any final wording is selected.
 *
 * The composer does not emit a response. It only builds meaning candidates.
 * Words.json remains the authority for what the final system can say.
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


  class VilotSemanticComposer {

    constructor(processor, trainingKnowledge, options = {
}
) {

      this.processor = processor;

      this.trainingKnowledge = trainingKnowledge;


      this.options = {

        maxKnowledgeRows:
          Math.max(2, Math.min(12, options.semanticMaxKnowledgeRows | 0 || 6)),

        maxPropositionsPerBranch:
          Math.max(4, Math.min(32, options.semanticMaxPropositionsPerBranch | 0 || 14)),

        branchConfidencePenaltyPct:
          Math.max(0, Number(options.semanticBranchConfidencePenaltyPct) || 1.4),

        minimumKnowledgeTrust:
          clamp(options.semanticMinimumKnowledgeTrust ?? 0.58),

        tokenLimit:
          Math.max(16, Math.min(128, options.semanticTokenLimit | 0 || 64)),

        keywordResponseMode:
          options.keywordResponseMode !== false,

        allowDefinitionDirectOutput:
          options.keywordDefinitionDirectOutput === true
      
}
;


      this.compositions = 0;

      this.lastDiagnostics = null;

    
}


    _array(value) {

      if (Array.isArray(value)) return value;

      if (value == null || value === '') return [];

      return [value];

    
}


    _tokens(text) {

      const rows =
        this.processor?.tokenize?.(
          String(text || ''),
          this.options.tokenLimit
        ) || [];


      if (rows.length) {

        return rows
          .filter(row => row?.pos !== 'Punct')
          .map(row => normalize(row?.lower || row?.surface))
          .filter(Boolean);

      
}


      return normalize(text)
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, this.options.tokenLimit);

    
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


    _entryTrust(entry) {

      return clamp(entry?.Trust ?? 0.72);

    
}


    _canonicalSubject(entry, analysis, knowledge) {

      // Structured request context outranks generic TrainingInfo subjects. A
      // mechanism question about an architecture component must not become an
      // explanation of a retrieved meta-entry such as "embedded question".
      const planned = String(analysis?.explanationPlan?.subjects?.[0] || '').trim();
      if (planned) return planned;

      const architecture = String(analysis?.architectureComponents?.[0]?.canonical || '').trim();
      if (architecture) return architecture;

      const embeddedClause = analysis?.complexLanguage?.clauses?.find?.(clause => String(clause?.relation || '').startsWith('embedded_'));
      const embeddedSubject = String(embeddedClause?.subject || '').trim();
      if (embeddedSubject) return embeddedSubject;

      const entity = String(analysis?.entities?.[0]?.canonical || '').trim();
      if (entity) return entity;

      const fromEntry = String(entry?.Subject || '').trim();
      if (fromEntry) return fromEntry;

      const top = String(knowledge?.rows?.[0]?.entry?.Subject || '').trim();
      if (top) return top;

      const content = analysis?.contentWords || [];
      return content.length ? content.slice(0, 4).join(' ') : 'concept';

    }


    _subjectForms(entry) {

      return [
        String(entry?.Subject || '').trim(),
        ...this._array(entry?.Aliases)
      ]
        .map(value => normalize(value))
        .filter(Boolean)
        .sort((a, b) => b.length - a.length);

    
}


    _definitionComplement(entry) {

      const source =
        String(
          entry?.Long_Definition ||
          entry?.Description ||
          ''
        )
          .replace(/\s+/g, ' ')
          .trim();


      if (!source) return '';


      const normalizedSource =
        normalize(source);


      for (const subject of this._subjectForms(entry)) {

        if (!subject) continue;


        const rawSubject =
          String(
            entry?.Subject ||
            subject
          ).trim();


        const candidates = [
          rawSubject,
          ...this._array(entry?.Aliases)
        ]
          .map(value =>
            String(value || '').trim()
          )
          .filter(Boolean)
          .sort((a, b) => b.length - a.length);


        for (const candidate of candidates) {

          const escaped =
            candidate.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');


          const match =
            source.match(
              new RegExp(
                `^\\s*(?:an?\\s+|the\\s+)?${escaped}\\s+(?:is|are)\\s+(.+?)[.!?]?$`,
                'i'
              )
            );


          if (match?.[1]) {

            return match[1]
              .trim()
              .replace(/[.!?]+$/, '');

          
}

        
}

      
}


      const generic =
        source.match(
          /^\s*(?:an?|the)\s+.+?\s+(?:is|are)\s+(.+?)[.!?]?$/i
        );


      if (generic?.[1]) {

        return generic[1]
          .trim()
          .replace(/[.!?]+$/, '');

      
}


      // If it cannot be safely decomposed, preserve the semantic text as a
      // complete proposition rather than guessing a new fact.
      return source
        .replace(/[.!?]+$/, '')
        .trim();

    
}


    _branchProfile(branchIndex, analysis) {

      const requested =
        String(
          analysis?.requestedSlot ||
          analysis?.intent ||
          ''
        ).toLowerCase();


      const explanationMode =
        analysis?.explanationMode ||
        null;


      const explanationProfiles = {

        why: [
          ['cause', 'mechanism', 'effect', 'relation', 'definition'],
          ['cause', 'mechanism', 'purpose', 'effect', 'relation'],
          ['effect', 'cause', 'mechanism', 'property', 'definition'],
          ['cause', 'relation', 'limitation', 'mechanism', 'definition']
        ],
        how: [
          ['mechanism', 'composition', 'has_part', 'capability', 'definition'],
          ['mechanism', 'purpose', 'part_of', 'relation', 'capability'],
          ['mechanism', 'effect', 'capability', 'property', 'definition'],
          ['mechanism', 'capability', 'limitation', 'relation', 'definition']
        ],
        both: [
          ['cause', 'mechanism', 'effect', 'definition', 'relation'],
          ['mechanism', 'cause', 'purpose', 'capability', 'relation'],
          ['effect', 'mechanism', 'cause', 'property', 'definition'],
          ['cause', 'mechanism', 'limitation', 'relation', 'definition']
        ]
      
}
;


      const profiles = explanationProfiles[explanationMode] || [
        ['definition', 'purpose', 'property', 'relation'],
        ['mechanism', 'composition', 'capability', 'definition'],
        ['relation', 'part_of', 'has_part', 'cause', 'effect', 'definition'],
        ['purpose', 'capability', 'limitation', 'contrast', 'example', 'definition']
      ];


      const profile =
        profiles[
          branchIndex %
          profiles.length
        ].slice();

      if (this.options.keywordResponseMode && !this.options.allowDefinitionDirectOutput) {
        for (let i = profile.length - 1; i >= 0; i--) {
          if (profile[i] === 'definition') profile.splice(i, 1);
        }
      }


      const requestedMap = {

        identity: 'definition',
        definition: 'definition',
        explanation: explanationMode === 'why' ? 'cause' : 'mechanism',
        mechanism: 'mechanism',
        cause: 'cause',
        causal: 'cause',
        truth: 'property',
        quantity: 'property',
        selection: 'contrast',
        content: 'definition'
      
}
;


      const preferred =
        requestedMap[requested];


      if (preferred) {

        const index =
          profile.indexOf(preferred);


        if (index >= 0) {

          profile.splice(index, 1);

        
}


        profile.unshift(preferred);

      
}


      return profile;

    
}


    _proposition(
      type,
      entry,
      value,
      branch,
      branchConfidence,
      rowScore,
      branchText,
      extra = {
}

    ) {

      const subject =
        this._canonicalSubject(
          entry,
          extra.analysis,
          extra.knowledge
        );


      const cleanValue =
        String(value || '')
          .replace(/\s+/g, ' ')
          .trim()
          .replace(/[.!?]+$/, '');


      if (!cleanValue) return null;


      const valueTokens =
        this._tokenSet(cleanValue);


      const branchTokens =
        this._tokenSet(branchText);


      const overlap =
        this._jaccard(
          valueTokens,
          branchTokens
        );


      const trust =
        this._entryTrust(entry);


      const rowRelevance =
        clamp(
          (Number(rowScore) || 0) /
          8
        );


      // Branch confidence stays dominant. Semantic evidence can only subtract a
      // small amount, so near-equivalent branch meanings remain inside the
      // existing 2 percentage-point interference window.
      const supportPenalty =
        (1 - overlap) *
        this.options
          .branchConfidencePenaltyPct;


      const evidencePenalty =
        (1 - (
          trust * 0.58 +
          rowRelevance * 0.42
        )) *
        0.9;


      const confidence =
        Math.max(
          0,
          Math.min(
            Number(branchConfidence) || 0,
            (Number(branchConfidence) || 0) -
            supportPenalty -
            evidencePenalty
          )
        );


      return {

        id:
          [
            branch,
            type,
            normalize(subject),
            normalize(cleanValue)
              .slice(0, 96)
          ].join('|'),

        branch,
        type,
        subject,

        relation:
          extra.relation ||
          type,

        value:
          cleanValue,

        confidence:
          Number(
            confidence.toFixed(4)
          ),

        branchConfidence:
          Number(branchConfidence) || 0,

        trust:
          Number(trust.toFixed(4)),

        rowRelevance:
          Number(
            rowRelevance.toFixed(4)
          ),

        branchSupport:
          Number(
            overlap.toFixed(4)
          ),

        tokens:
          Array.from(
            valueTokens
          ).slice(
            0,
            this.options.tokenLimit
          ),

        sourceEntryId:
          entry?.ID ||
          null,

        sourceSubject:
          entry?.Subject ||
          subject,

        sourceType:
          entry?.Source_Type ||
          '',

        verification:
          entry?._verification ||
          null,

        completeClause:
          extra.completeClause === true,

        aliases:
          this._array(
            entry?.Aliases
          )
      
}
;

    
}


    _entryPropositions(
      row,
      branch,
      branchConfidence,
      branchText,
      analysis,
      knowledge
    ) {

      const entry =
        row?.entry ||
        {
}
;


      if (
        this._entryTrust(entry) <
        this.options.minimumKnowledgeTrust
      ) {

        return [];

      
}


      const propositions = [];


      const push = (
        type,
        value,
        extra = {
}

      ) => {

        const proposition =
          this._proposition(
            type,
            entry,
            value,
            branch,
            branchConfidence,
            row?.score,
            branchText,
            {

              ...extra,
              analysis,
              knowledge
            
}

          );


        if (proposition) {

          propositions.push(
            proposition
          );

        
}

      
}
;


      const definition =
        (!this.options.keywordResponseMode || this.options.allowDefinitionDirectOutput)
          ? this._definitionComplement(entry)
          : '';


      if (definition) {

        push(
          'definition',
          definition,
          {

            relation: 'is',
            completeClause:
              normalize(definition)
                .startsWith(
                  normalize(
                    entry?.Subject
                  )
                )
          
}

        );

      
}


      if (entry?.Purpose) {

        push(
          'purpose',
          entry.Purpose,
          {

            relation: 'purpose'
          
}

        );

      
}


      if (entry?.Mechanism) {

        push(
          'mechanism',
          entry.Mechanism,
          {

            relation: 'mechanism',
            completeClause: true
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Properties
        )
      ) {

        push(
          'property',
          value,
          {

            relation: 'property'
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Capabilities
        )
      ) {

        push(
          'capability',
          value,
          {

            relation: 'capability'
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Limitations
        )
      ) {

        push(
          'limitation',
          value,
          {

            relation: 'limitation'
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Examples
        )
      ) {

        push(
          'example',
          value,
          {

            relation: 'example'
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Contrasts_With
        )
      ) {

        push(
          'contrast',
          value,
          {

            relation: 'contrast'
          
}

        );

      
}


      for (const value of this._array(entry?.Evaluation_Criteria)) {

        push('evaluation', value, {
 relation: 'evaluated_by' 
}
);

      
}


      for (const value of this._array(entry?.Failure_Modes)) {

        push('failure_mode', value, {
 relation: 'failure_mode' 
}
);

      
}


      for (const value of this._array(entry?.Tradeoffs)) {

        push('tradeoff', value, {
 relation: 'tradeoff' 
}
);

      
}


      for (
        const value of
        this._array(
          entry?.Part_Of
        )
      ) {

        push(
          'part_of',
          value,
          {

            relation: 'part_of'
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Has_Parts
        )
      ) {

        push(
          'has_part',
          value,
          {

            relation: 'has_part'
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Causes
        )
      ) {

        push(
          'cause',
          value,
          {

            relation: 'cause'
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Effects
        )
      ) {

        push(
          'effect',
          value,
          {

            relation: 'effect'
          
}

        );

      
}


      for (
        const value of
        this._array(
          entry?.Related_To
        )
          .slice(0, 8)
      ) {

        push(
          'relation',
          value,
          {

            relation: 'related_to'
          
}

        );

      
}


      return propositions;

    
}


    compose(
      branches,
      analysis,
      knowledge,
      lexicalFallback = null,
      options = {
}

    ) {

      const started =
        performance.now();


      const deadlineAt = Number.isFinite(Number(options.deadlineAt))
        ? Number(options.deadlineAt)
        : Infinity;

      const deadlineReserveMs = Math.max(0.25, Number(options.deadlineReserveMs) || 0.8);

      let deadlineTruncated = false;


      const validBranches =
        (branches || [])
          .filter(branch =>
            branch &&
            typeof branch ===
              'object'
          );


      const knowledgeRows =
        (knowledge?.rows || [])
          .slice(
            0,
            this.options
              .maxKnowledgeRows
          );


      const lexicalText =
        String(
          lexicalFallback?.text ||
          ''
        );


      const frames = [];

      const slotMap =
        Object.create(null);


      const topSubject =
        this._canonicalSubject(
          knowledgeRows[0]?.entry,
          analysis,
          knowledge
        );


      for (
        let branchIndex = 0;

        branchIndex <
          validBranches.length;

        branchIndex++
      ) {

        if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt - deadlineReserveMs) {

          deadlineTruncated = true;

          break;

        
}

        const branch =
          validBranches[branchIndex];


        const branchConfidence =
          Number(
            branch?.confidence
          ) || 0;


        const branchText =
          [
            String(
              branch?.text ||
              ''
            ),
            lexicalText
          ]
            .filter(Boolean)
            .join(' ');


        const profile =
          this._branchProfile(
            branchIndex,
            analysis
          );


        const profileRank =
          new Map(
            profile.map(
              (type, index) => [
                type,
                index
              ]
            )
          );


        const props = [];


        for (
          const row of
          knowledgeRows
        ) {

          if (Number.isFinite(deadlineAt) && performance.now() >= deadlineAt - deadlineReserveMs) {

            deadlineTruncated = true;

            break;

          
}

          for (
            const proposition of
            this._entryPropositions(
              row,
              branchIndex,
              branchConfidence,
              branchText,
              analysis,
              knowledge
            )
          ) {

            props.push(
              proposition
            );

          
}

        
}


        props.sort(
          (a, b) => {

            const rankA =
              profileRank.has(a.type)
                ? profileRank.get(a.type)
                : profile.length + 2;


            const rankB =
              profileRank.has(b.type)
                ? profileRank.get(b.type)
                : profile.length + 2;


            if (
              rankA !== rankB
            ) {

              return rankA - rankB;

            
}


            return (
              b.confidence -
              a.confidence
            );

          
}

        );


        const selected =
          props.slice(
            0,
            this.options
              .maxPropositionsPerBranch
          );


        frames.push({

          branch:
            branchIndex,

          confidence:
            branchConfidence,

          subject:
            topSubject,

          profile,

          propositions:
            selected
        
}
);


        for (
          const proposition of
          selected
        ) {

          (
            slotMap[
              proposition.type
            ] ||= []
          ).push(
            proposition
          );

        
}

      
}


      const slotCounts =
        Object.fromEntries(
          Object.entries(
            slotMap
          ).map(
            ([slot, rows]) => [
              slot,
              rows.length
            ]
          )
        );


      const result = {

        promptStructure: { clauses: analysis?.clauseGraph || [], propositions: analysis?.propositions || [], references: analysis?.referenceLinks || [], relations: analysis?.propositionRelations || [], structuralComplexity: Number(analysis?.structuralComplexity || 0) },

        subject:
          topSubject,

        frames,

        slots:
          slotMap,

        lexicalSupportText:
          lexicalText,

        knowledgeSubjects:
          knowledgeRows
            .map(
              row =>
                row.entry
                  ?.Subject ||
                ''
            )
            .filter(Boolean),

        diagnostics: {

          composerMs:
            performance.now() -
            started,

          branchCount:
            frames.length,

          deadlineTruncated,

          propositionCount:
            frames.reduce(
              (sum, frame) =>
                sum +
                frame
                  .propositions
                  .length,
              0
            ),

          slotCounts,
          promptClauseCount: analysis?.clauseGraph?.length || 0,
          promptPropositionCount: analysis?.propositions?.length || 0,
          promptStructuralComplexity: Number(analysis?.structuralComplexity || 0)
        
}

      
}
;


      this.compositions++;

      this.lastDiagnostics =
        result.diagnostics;


      return result;

    
}


    status() {

      return {

        ready: true,
        role:
          'four-branch-semantic-composer',
        compositions:
          this.compositions,
        lastDiagnostics:
          this.lastDiagnostics,
        architecture: {

          meaningBeforeWords: true,
          usesTrainingInfoSemanticFields: true,
          emitsUserText: false
        
}

      
}
;

    
}

  
}


  globalThis.VilotSemanticComposer =
    VilotSemanticComposer;

}
)();

