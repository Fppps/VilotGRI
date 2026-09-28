/**
 * VilotNI 2.5 - SurfaceRealizer.js
 *
 * Final lexical layer for meaning-first output.
 *
 * Receives an already-collapsed semantic clause plan and converts it into
 * grammatical text using generic grammatical relations plus Words.json-backed
 * Processor realization/repair. It contains no subject-specific answer table.
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


  class VilotSurfaceRealizer {

    constructor(processor, options = {
}
, keywordSelector = null, syntacticBridge = null, syntaxSmoother = null) {

      this.processor = processor;

      this.keywordSelector = keywordSelector;

      this.syntacticBridge = syntacticBridge;

      this.syntaxSmoother = syntaxSmoother;


      this.options = {

        maxWords:
          Math.max(
            24,
            Math.min(
              192,
              options.surfaceMaxWords | 0 ||
              96
            )
          ),

        minLexicalCoverage:
          clamp(
            options.surfaceMinLexicalCoverage ??
            0.90
          ),

        maxRejectedRatio:
          clamp(
            options.surfaceMaxRejectedRatio ??
            0.08
          ),

        minGrammarScore:
          clamp(
            options.surfaceMinGrammarScore ??
            0.72
          )
      
}
;


      this.realizations = 0;

      this.lastDiagnostics = null;

    
}


    _lowerFirst(text) {

      const value =
        String(text || '').trim();


      if (!value) return value;


      return (
        value.charAt(0).toLowerCase() +
        value.slice(1)
      );

    
}


    _stripTerminal(text) {

      return String(text || '')
        .trim()
        .replace(/[.!?]+$/, '');

    
}


    _startsWithSubject(
      text,
      subject,
      aliases = []
    ) {

      const source =
        normalize(text);


      const subjects =
        [
          subject,
          ...aliases
        ]
          .map(normalize)
          .filter(Boolean)
          .sort(
            (a, b) =>
              b.length -
              a.length
          );


      return subjects.some(
        candidate =>
          source === candidate ||
          source.startsWith(
            `${candidate} `
          )
      );

    
}


    _stripLeadingSubject(
      text,
      subject,
      aliases = []
    ) {

      let source =
        this._stripTerminal(
          text
        );


      const candidates =
        [
          subject,
          ...aliases
        ]
          .map(value =>
            String(value || '')
              .trim()
          )
          .filter(Boolean)
          .sort(
            (a, b) =>
              b.length -
              a.length
          );


      for (
        const candidate of
        candidates
      ) {

        const escaped =
          candidate.replace(
            /[.*+?^${}()|[\]\\]/g,
            '\\$&'
          );


        const match =
          source.match(
            new RegExp(
              `^\\s*${escaped}\\s+(.+)$`,
              'i'
            )
          );


        if (match?.[1]) {

          return match[1]
            .trim();

        
}

      
}


      return source;

    
}


    _questionCore(analysis) {

      const scaffold = analysis?.responseScaffold || null;

      if (!scaffold) return '';


      const source = Array.isArray(scaffold.surfaces) && scaffold.surfaces.length
        ? scaffold.surfaces.slice()
        : Array.isArray(scaffold.words)
          ? scaffold.words.slice()
          : [];


      if (!source.length) return '';


      const bridges = new Set(['because','by','through','using','with','during','in']);

      while (source.length && bridges.has(String(source[source.length - 1] || '').toLowerCase())) {

        source.pop();

      
}


      return source
        .map(value => String(value || '').trim())
        .filter(Boolean)
        .join(' ')
        .replace(/\s+/g, ' ')
        .trim();

    
}


    _clauseText(
      clause,
      usePronoun = false,
      analysis = null,
      keywordGroup = null
    ) {

      const subject =
        usePronoun
          ? 'It'
          : String(
              clause?.subject ||
              'The concept'
            ).trim();


      let value =
        this._stripTerminal(
          clause?.value
        );


      if (!value) {

        return '';

      
}


      const aliases =
        clause?.aliases ||
        [];


      const explanationMode = analysis?.explanationMode || null;

      const questionCore = explanationMode ? this._questionCore(analysis) : '';


      // For Why/How questions, preserve the grammatical proposition from the
      // user's question and attach the supported semantic explanation to it.
      // This is a generic grammatical transformation, not a topic answer.
      if (questionCore && clause?.type === 'cause') {

        return `${questionCore} because ${this._lowerFirst(value)}`;

      
}


      if (questionCore && clause?.type === 'mechanism') {

        return `${questionCore} by ${this._lowerFirst(value)}`;

      
}


      if (
        clause
          ?.completeClause &&
        this._startsWithSubject(
          value,
          clause?.subject,
          aliases
        )
      ) {

        if (usePronoun) {

          const tail =
            this._stripLeadingSubject(
              value,
              clause?.subject,
              aliases
            );


          return tail
            ? `It ${this._lowerFirst(tail)}`
            : '';

        
}


        return value;

      
}


      const bridged =
        this.syntacticBridge
          ?.bridgeClause?.(
            clause,
            keywordGroup,
            analysis,
            { usePronoun }
          );


      if (bridged?.text) {

        return bridged.text;

      }


      switch (
        clause?.type
      ) {

        case 'definition':
          return (
            `${subject} is ${this._lowerFirst(value)}`
          );


        case 'purpose':
          return (
            `${subject} is used to ${this._lowerFirst(value)}`
          );


        case 'mechanism': {

          if (
            this._startsWithSubject(
              value,
              clause?.subject,
              aliases
            )
          ) {

            if (usePronoun) {

              const tail =
                this._stripLeadingSubject(
                  value,
                  clause?.subject,
                  aliases
                );


              return tail
                ? `It ${this._lowerFirst(tail)}`
                : '';

            
}


            return value;

          
}


          return (
            `${subject} works through ${this._lowerFirst(value)}`
          );

        
}


        case 'capability':
          return (
            `${subject} can ${this._lowerFirst(value)}`
          );


        case 'limitation':
          return (
            `${subject} is limited by ${this._lowerFirst(value)}`
          );


        case 'part_of':
          return (
            `${subject} is part of ${this._lowerFirst(value)}`
          );


        case 'has_part':
          return (
            `${subject} includes ${this._lowerFirst(value)}`
          );


        case 'cause':
          return (
            `${subject} can be caused by ${this._lowerFirst(value)}`
          );


        case 'effect':
          return (
            `${subject} can lead to ${this._lowerFirst(value)}`
          );


        case 'contrast':
          return (
            `${subject} differs from ${this._lowerFirst(value)}`
          );


        case 'example':
          return (
            `Examples include ${this._lowerFirst(value)}`
          );


        case 'property':
          return (
            `${subject} has ${this._lowerFirst(value)}`
          );


        case 'relation':
        default:
          return (
            `${subject} is related to ${this._lowerFirst(value)}`
          );

      
}

    
}


    _sentenceText(sentence, analysis = null, keywordSelection = null) {

      const ideas =
        sentence?.ideas ||
        [];


      if (!ideas.length) {

        return '';

      
}


      const firstSubject =
        normalize(
          ideas[0]?.subject
        );


      const rendered = [];


      for (
        let i = 0;

        i < ideas.length;

        i++
      ) {

        const idea =
          ideas[i];


        const usePronoun =
          i > 0 &&
          firstSubject &&
          normalize(
            idea?.subject
          ) ===
            firstSubject;


        const text =
          this._clauseText(
            idea,
            usePronoun,
            analysis,
            keywordSelection
              ?.byClauseIndex
              ?.[Number.isFinite(Number(idea?.index)) ? Number(idea.index) : i] ||
              null
          );


        if (text) {

          rendered.push(
            this._stripTerminal(
              text
            )
          );

        
}

      
}


      if (!rendered.length) {

        return '';

      
}


      if (rendered.length === 1) {

        return rendered[0];

      
}


      const connectors =
        sentence?.connectors ||
        [];


      if (rendered.length === 2) {

        const connector =
          this.syntacticBridge
            ?.connectorFor?.(
              ideas[0],
              ideas[1],
              connectors[0]
            ) ||
          (connectors[0] === 'but' ? 'but' : 'and');


        return (
          `${rendered[0]}, ${connector} ${this._lowerFirst(rendered[1])}`
        );

      
}


      // Three or more independent semantic ideas stay in one sentence without
      // creating comma splices. Semicolons preserve each complete clause while
      // the final conjunction marks the last coordinated idea.
      let text =
        rendered[0];


      for (
        let i = 1;

        i < rendered.length;

        i++
      ) {

        const connector =
          this.syntacticBridge
            ?.connectorFor?.(
              ideas[i - 1],
              ideas[i],
              connectors[i - 1]
            ) ||
          (connectors[i - 1] === 'but' ? 'but' : 'and');


        const final =
          i ===
          rendered.length - 1;


        if (final) {

          text +=
            `; ${connector} ${this._lowerFirst(rendered[i])}`;

        
}
 else {

          text +=
            `; ${this._lowerFirst(rendered[i])}`;

        
}

      
}


      return text;

    
}


    _lexicalize(text) {

      const realized =
        this.processor
          ?.realizePhrase?.(
            text,
            {

              maxWords:
                this.options
                  .maxWords,

              allowUnknown:
                false,

              allowRootFallback:
                true
            
}

          );


      if (!realized) {

        return {

          text:
            String(
              text ||
              ''
            ),

          lexicalCoverage:
            0,

          rejected: []
        
}
;

      
}


      return realized;

    
}


    _repair(
      text,
      analysis
    ) {

      const tokens =
        this.processor
          ?.tokenize?.(
            String(text || ''),
            this.options
              .maxWords * 2
          ) || [];


      const surfaces =
        tokens
          .map(
            token =>
              token?.surface ||
              token?.lower ||
              ''
          )
          .filter(Boolean);


      const repaired =
        this.processor
          ?.repairGeneratedSurface?.(
            surfaces,
            analysis
          );


      return {

        text:
          String(
            repaired?.text ||
            text ||
            ''
          ).trim(),

        repairs:
          repaired?.repairs ||
          []
      
}
;

    
}


    realize(
      plan,
      analysis,
      lexicalFallback = null
    ) {

      const started =
        performance.now();


      const clauses =
        plan?.clauses ||
        [];

      const promptStructure = analysis?.complexLanguage || null;


      if (!clauses.length) {

        return {

          valid: false,
          text: '',
          confidence: 0,
          diagnostics: {

            realizerMs:
              performance.now() -
              started,
            reason:
              'no-semantic-clauses'
          
}

        
}
;

      
}


      const keywordSelection =
        this.keywordSelector
          ?.select?.(
            plan,
            analysis
          ) ||
        null;


      const rawClauses = [];


      for (
        let i = 0;

        i < clauses.length;

        i++
      ) {

        const clause =
          this._clauseText(
            clauses[i],
            i > 0 &&
            normalize(
              clauses[i]
                ?.subject
            ) ===
            normalize(
              clauses[0]
                ?.subject
            ),
            analysis,
            keywordSelection
              ?.byClauseIndex
              ?.[Number.isFinite(Number(clauses[i]?.index)) ? Number(clauses[i].index) : i] ||
              null
          );


        if (clause) {

          rawClauses.push(
            this._stripTerminal(
              clause
            )
          );

        
}

      
}


      const plannedSentences =
        Array.isArray(
          plan?.sentences
        ) &&
        plan.sentences.length
          ? plan.sentences
          : clauses.map(
              clause => ({

                ideas: [clause],
                connectors: [],
                ideaCount: 1
              
}
)
            );


      const rawSentences =
        plannedSentences
          .map(
            sentence =>
              this._sentenceText(
                sentence,
                analysis,
                keywordSelection
              )
          )
          .map(
            sentence =>
              this._stripTerminal(
                sentence
              )
          )
          .filter(Boolean);


      if (!rawSentences.length) {

        return {

          valid: false,
          text: '',
          confidence: 0,
          diagnostics: {

            realizerMs:
              performance.now() -
              started,
            reason:
              'empty-surface-plan'
          
}

        
}
;

      
}


      const rawText =
        rawSentences
          .map(
            sentence =>
              `${sentence}.`
          )
          .join(' ');


      const lexical =
        this._lexicalize(
          rawText
        );


      const repaired =
        this._repair(
          lexical.text ||
          rawText,
          analysis
        );


      const smoothed =
        this.syntaxSmoother
          ?.smooth?.(
            repaired.text,
            analysis
          ) ||
        {
          text: repaired.text,
          repairs: [],
          diagnostics: null
        };


      const text =
        smoothed.text;


      const audit =
        this.processor
          ?.auditResponse?.(
            text,
            {

              allowRootRepeat:
                false,
              analysis
            
}

          ) || {

            legalRatio: 1,
            grammarScore: 1,
            illegal: 0,
            issues: []
          
}
;


      const grammarScore =
        clamp(
          audit
            ?.grammarScore ??
          audit
            ?.legalRatio ??
          0
        );


      const lexicalCoverage =
        Number(
          lexical
            ?.lexicalCoverage
        ) || 0;


      const fallbackWords =
        new Set(
          (
            this.processor
              ?.tokenize?.(
                String(
                  lexicalFallback
                    ?.text ||
                  ''
                ),
                128
              ) || []
          )
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


      const outputWords =
        (
          this.processor
            ?.tokenize?.(
              text,
              128
            ) || []
        )
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
          .filter(Boolean);


      const lexicalAgreement =
        outputWords.length
          ? outputWords.filter(
              word =>
                fallbackWords.has(
                  word
                )
            ).length /
            outputWords.length
          : 0;


      const rawWordCount =
        (
          this.processor
            ?.tokenize?.(
              rawText,
              256
            ) || []
        )
          .filter(
            token =>
              token?.pos !==
              'Punct'
          )
          .length;


      const rejectedRatio =
        rawWordCount
          ? (
              lexical
                ?.rejected
                ?.length ||
              0
            ) /
            rawWordCount
          : 0;


      const trailingWord =
        (
          this.processor
            ?.tokenize?.(
              text,
              256
            ) || []
        )
          .filter(
            token =>
              token?.pos !==
              'Punct'
          )
          .at(-1);


      const trailingLower =
        normalize(
          trailingWord?.lower ||
          trailingWord?.surface ||
          ''
        );


      const orphanTail =
        new Set([
          'and',
          'or',
          'but',
          'because',
          'the',
          'a',
          'an',
          'of',
          'to',
          'with',
          'for',
          'from',
          'by',
          'in',
          'on'
        ]).has(
          trailingLower
        );


      const valid =
        Boolean(text) &&
        (
          audit?.illegal ||
          0
        ) === 0 &&
        grammarScore >=
          this.options
            .minGrammarScore &&
        lexicalCoverage >=
          this.options
            .minLexicalCoverage &&
        rejectedRatio <=
          this.options
            .maxRejectedRatio &&
        !orphanTail;


      const result = {

        valid,

        text,

        confidence:
          Number(
            plan?.confidence ||
            0
          ),

        audit,

        lexicalCoverage,

        lexicalAgreement:
          Number(
            lexicalAgreement.toFixed(4)
          ),

        rawClauses,

        rawSentences,

        diagnostics: {

          realizerMs:
            performance.now() -
            started,

          clauseCount:
            rawClauses.length,

          sentenceCount:
            rawSentences.length,

          ideaCount:
            clauses.length,

          keywordCount:
            keywordSelection
              ?.keywords
              ?.length ||
            0,

          keywordSelection:
            keywordSelection
              ?.diagnostics ||
            null,

          syntacticBridge:
            this.syntacticBridge
              ?.lastDiagnostics ||
            null,

          syntaxSmoothing:
            smoothed.diagnostics ||
            null,

          multiIdeaSentenceCount:
            plannedSentences.filter(
              sentence =>
                (
                  sentence?.ideas
                    ?.length ||
                  0
                ) >= 2
            ).length,

          lexicalCoverage:
            Number(
              lexicalCoverage.toFixed(4)
            ),

          rejectedRatio:
            Number(
              rejectedRatio.toFixed(4)
            ),

          orphanTail,

          lexicalAgreement:
            Number(
              lexicalAgreement.toFixed(4)
            ),

          grammarScore:
            Number(
              grammarScore.toFixed(4)
            ),
          grammarAudit: audit?.advancedGrammar || null,

          illegal:
            Number(
              audit?.illegal
            ) || 0,

          rejectedWords:
            (
              lexical?.rejected ||
              []
            ).slice(0, 24),

          repairs:
            [
              ...(repaired.repairs || []),
              ...(smoothed.repairs || [])
            ]
              .slice(0, 32)
        
}

      
}
;


      this.realizations++;

      this.lastDiagnostics =
        result.diagnostics;


      return result;

    
}


    status() {

      return {

        ready: true,
        role:
          'meaning-first-surface-realizer',
        realizations:
          this.realizations,
        lastDiagnostics:
          this.lastDiagnostics,
        architecture: {

          usesWordsJSONThroughProcessor:
            true,
          semanticPlanAlreadyCollapsed:
            true,
          supportsMultiIdeaSentences:
            true,
          keywordSelection:
            Boolean(this.keywordSelector),
          syntacticBridging:
            Boolean(this.syntacticBridge),
          syntaxSmoothing:
            Boolean(this.syntaxSmoother),
          subjectSpecificTemplates:
            false,
          extraDecoderPasses:
            0
        
}

      
}
;

    
}

  
}


  globalThis.VilotSurfaceRealizer =
    VilotSurfaceRealizer;

}
)();

