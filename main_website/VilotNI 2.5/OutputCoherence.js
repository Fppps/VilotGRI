/**
 * VilotNI 2.5 - OutputCoherence.js
 *
 * Final anti-word-salad quality gate.
 *
 * Scores meaning-first output, word-level mixed output, and projected branch
 * fallbacks using general linguistic/structural signals. It does not contain
 * topic-specific answers.
 *
 * If a fancy mixed output is less coherent than an existing branch, the more
 * coherent output wins. No new decoder pass is required.
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


  class VilotOutputCoherence {

    constructor(processor, options = {
}
) {

      this.processor = processor;


      this.options = {

        minimumScore:
          clamp(
            options.outputCoherenceMinimumScore ??
            0.72
          ),

        semanticSurfaceBonus:
          Math.max(
            0,
            Number(
              options.outputCoherenceSemanticBonus
            ) || 0.035
          ),

        maxDuplicateRatio:
          clamp(
            options.outputCoherenceMaxDuplicateRatio ??
            0.22
          ),

        maxUnknownRatio:
          clamp(
            options.outputCoherenceMaxUnknownRatio ??
            0.10
          ),

        maxSwitchDensity:
          clamp(
            options.mixedSlotMaxSwitchDensity ??
            0.34
          )
      
}
;


      this.evaluations = 0;

      this.lastDiagnostics = null;

    
}


    _tokens(text) {

      return (
        this.processor
          ?.tokenize?.(
            String(text || ''),
            256
          ) || []
      );

    
}


    _content(tokens) {

      return (
        tokens || []
      ).filter(
        token =>
          token?.pos !==
          'Punct'
      );

    
}


    _duplicateRatio(tokens) {

      const words =
        this._content(tokens)
          .map(
            token =>
              normalize(
                token?.lower ||
                token?.surface
              )
          )
          .filter(Boolean);


      if (
        words.length < 2
      ) {

        return 0;

      
}


      let duplicateSignals = 0;


      for (
        let i = 1;

        i < words.length;

        i++
      ) {

        if (
          words[i] ===
          words[i - 1]
        ) {

          duplicateSignals++;

        
}


        if (
          i >= 3 &&
          words[i] ===
            words[i - 2] &&
          words[i - 1] ===
            words[i - 3]
        ) {

          duplicateSignals++;

        
}

      
}


      return clamp(
        duplicateSignals /
        Math.max(
          1,
          words.length
        )
      );

    
}


    _unknownRatio(tokens) {

      const words =
        this._content(tokens);


      if (!words.length) {

        return 1;

      
}


      let unknown = 0;


      for (const token of words) {

        const surface =
          token?.lower ||
          token?.surface ||
          '';


        const entry =
          this.processor
            ?.resolveSayableWord?.(
              surface,
              {

                allowRootFallback:
                  true
              
}

            );


        if (
          !entry ||
          entry.active === false
        ) {

          unknown++;

        
}

      
}


      return unknown /
        words.length;

    
}


    _functionOrphanPenalty(tokens) {

      const rows =
        this._content(tokens);


      if (!rows.length) {

        return 1;

      
}


      const first =
        rows[0];


      const last =
        rows[
          rows.length - 1
        ];


      const badStart =
        new Set([
          'and',
          'or',
          'but',
          'because',
          'than',
          'of',
          'to'
        ]);


      const badEnd =
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
        ]);


      let penalty = 0;


      if (
        badStart.has(
          normalize(
            first?.lower ||
            first?.surface
          )
        )
      ) {

        penalty += 0.28;

      
}


      if (
        badEnd.has(
          normalize(
            last?.lower ||
            last?.surface
          )
        )
      ) {

        penalty += 0.40;

      
}


      return clamp(
        penalty
      );

    
}


    _punctuationPenalty(text) {

      const source =
        String(text || '');


      const pairs = [
        ['(', ')'],
        ['[', ']'],
        ['{', '}']
      ];


      let penalty = 0;


      for (
        const [open, close]
        of pairs
      ) {

        const opens =
          source
            .split(open)
            .length - 1;


        const closes =
          source
            .split(close)
            .length - 1;


        if (opens !== closes) {

          penalty += 0.18;

        
}

      
}


      if (
        /[!?.,;:]{3,}/
          .test(source)
      ) {

        penalty += 0.12;

      
}


      return clamp(
        penalty
      );

    
}


    _sentenceFragmentPenalty(text) {

      const sentences =
        String(text || '')
          .split(/[.!?]+/)
          .map(
            sentence =>
              sentence.trim()
          )
          .filter(Boolean);


      if (!sentences.length) {

        return 1;

      
}


      let fragments = 0;


      for (
        const sentence of
        sentences
      ) {

        const tokens =
          this._content(
            this._tokens(
              sentence
            )
          );


        const hasNominal =
          tokens.some(
            token =>
              ['Noun','ProperNoun','Pronoun']
                .includes(
                  String(
                    token?.pos ||
                    ''
                  )
                )
          );


        const hasVerbal =
          tokens.some(
            token =>
              ['Verb','Aux','Modal']
                .includes(
                  String(
                    token?.pos ||
                    ''
                  )
                )
          );


        if (
          tokens.length < 3 ||
          (
            !hasNominal &&
            !hasVerbal
          )
        ) {

          fragments++;

        
}

      
}


      return fragments /
        sentences.length;

    
}


    _structuralFlowPenalty(tokens) {

      const rows = (tokens || []).filter(token => token?.pos !== 'Punct');
      if (!rows.length) return { total: 1, maxContentRun: 0, prepCount: 0, repeatedRoots: 0 };

      let run = 0;
      let maxContentRun = 0;
      const lowers = [];
      const roots = [];
      const prepWords = new Set([
        'of','to','in','on','at','for','from','with','without','by','through','using',
        'across','into','onto','over','under','between','among','during','before','after','as','than'
      ]);

      const canonical = word => {
        let w = normalize(word).replace(/\s+/g, '');
        if (w.length > 5 && w.endsWith('ies')) w = `${w.slice(0, -3)}y`;
        else if (w.length > 4 && w.endsWith('es')) w = w.slice(0, -2);
        else if (w.length > 4 && w.endsWith('s') && !w.endsWith('ss')) w = w.slice(0, -1);
        return w;
      };

      for (const token of rows) {
        const lower = normalize(token?.lower || token?.surface);
        lowers.push(lower);
        const entry = this.processor?.resolveSayableWord?.(lower, { allowRootFallback: true });
        if (entry?.contentWord) {
          run++;
          maxContentRun = Math.max(maxContentRun, run);
          roots.push(canonical(entry.root || entry.lower || lower));
        } else {
          run = 0;
          roots.push('');
        }
      }

      let penalty = 0;
      if (maxContentRun > 3) {
        penalty += Math.min(0.42, 0.07 * Math.pow(maxContentRun - 3, 1.42));
      }

      const preps = lowers.filter(word => prepWords.has(word));
      if (preps.length >= 3) penalty += Math.min(0.26, (preps.length - 2) * 0.08);

      const prepCounts = new Map();
      for (const word of preps) prepCounts.set(word, (prepCounts.get(word) || 0) + 1);
      for (const count of prepCounts.values()) {
        if (count > 1) penalty += Math.min(0.16, (count - 1) * 0.065);
      }

      let repeatedRoots = 0;
      for (let i = 0; i < roots.length; i++) {
        if (!roots[i]) continue;
        for (let j = Math.max(0, i - 5); j < i; j++) {
          if (roots[j] && roots[j] === roots[i]) {
            repeatedRoots++;
            penalty += 0.09;
            break;
          }
        }
      }

      // A copular identity sentence should not turn into a long unstructured
      // catalog after the copula. This check is generic across all subjects.
      const copulaIndex = lowers.findIndex(word => ['is','are','was','were'].includes(word));
      if (copulaIndex >= 0) {
        let postCopulaRun = 0;
        let postCopulaMax = 0;
        for (let i = copulaIndex + 1; i < rows.length; i++) {
          const entry = this.processor?.resolveSayableWord?.(lowers[i], { allowRootFallback: true });
          if (entry?.contentWord) {
            postCopulaRun++;
            postCopulaMax = Math.max(postCopulaMax, postCopulaRun);
          } else {
            postCopulaRun = 0;
          }
        }
        if (postCopulaMax > 3) penalty += Math.min(0.24, (postCopulaMax - 3) * 0.07);
      }

      return {
        total: clamp(penalty),
        maxContentRun,
        prepCount: preps.length,
        repeatedRoots
      };

    }


    _danglingTerminalPenalty(tokens) {

      const rows = Array.isArray(tokens) ? tokens : [];
      let lastLexical = null;

      for (let i = rows.length - 1; i >= 0; i--) {
        if (rows[i]?.pos !== 'Punct') {
          lastLexical = rows[i];
          break;
        }
      }

      if (!lastLexical) return 1;

      const pos = String(lastLexical.pos || '');
      if (pos === 'Prep') return 1;
      if (pos === 'Conj' || pos === 'Det' || pos === 'Modal') return 0.72;

      return 0;

    }


    _semanticSubjectCoverage(
      text,
      semanticPlan
    ) {

      const subject =
        normalize(
          semanticPlan
            ?.subject ||
          ''
        );


      if (!subject) {

        return 0.5;

      
}


      const source =
        normalize(text);


      if (
        source.includes(
          subject
        )
      ) {

        return 1;

      
}


      const parts =
        subject
          .split(/\s+/)
          .filter(Boolean);


      if (!parts.length) {

        return 0.5;

      
}


      const hits =
        parts.filter(
          part =>
            source.includes(
              part
            )
        ).length;


      return hits /
        parts.length;

    
}


    complexStructureScore(text, analysis = {}) {
      const structure = analysis?.complexLanguage || {};
      if (!(structure.clauses || []).length) return 0.5;
      const response = String(text || '').toLowerCase();
      const relations = structure.relationTerms || [];
      const cues = {cause:['because','since','due','from'],embedded_cause:['because','cause','reason','since'],embedded_mechanism:['by','through','using','process','works','via'],condition:['if','unless','when'],contrast:['but','although','whereas','however'],comparison:['than','compared','similar','different'],temporal_before:['before','earlier'],temporal_after:['after','later'],temporal_progression:['over time','later','gradually','eventually'],temporal_overlap:['while','during'],purpose:['so','to','for']};
      let total=0, hit=0;
      for(const rel of relations){const family=Object.keys(cues).find(k=>rel===k||rel.startsWith(k));if(!family)continue;total++;if(cues[family].some(c=>response.includes(c)))hit++;}
      return total?hit/total:0.72;
    }

    evaluate(
      candidate,
      context = {
}

    ) {

      const text =
        String(
          candidate?.text ||
          ''
        ).trim();


      const tokens =
        this._tokens(
          text
        );


      const audit =
        this.processor
          ?.auditResponse?.(
            text,
            {

              allowRootRepeat:
                false,
              analysis: context?.analysis || null
            
}

          ) || {
}
;


      const grammar =
        clamp(
          audit
            ?.grammarScore ??
          audit
            ?.legalRatio ??
          0
        );


      const duplicateRatio =
        this._duplicateRatio(
          tokens
        );


      const unknownRatio =
        this._unknownRatio(
          tokens
        );


      const orphanPenalty =
        this._functionOrphanPenalty(
          tokens
        );


      const punctuationPenalty =
        this._punctuationPenalty(
          text
        );


      const fragmentPenalty =
        this._sentenceFragmentPenalty(
          text
        );


      const structuralFlow =
        this._structuralFlowPenalty(
          tokens
        );


      const danglingTerminalPenalty =
        this._danglingTerminalPenalty(
          tokens
        );


      const subjectCoverage =
        this._semanticSubjectCoverage(
          text,
          context?.semanticPlan
        );


      const switchDensity =
        Number(
          context
            ?.mixedSlot
            ?.switchDensity
        ) || 0;


      const switchPenalty =
        switchDensity >
        this.options
          .maxSwitchDensity
          ? Math.min(
              0.35,
              (
                switchDensity -
                this.options
                  .maxSwitchDensity
              ) *
              0.85
            )
          : 0;


      const complexStructure = context?.analysis?.complexLanguage ? this.complexStructureScore(text, context.analysis) : 0.5;
      const complexWeight = Number(context?.analysis?.structuralComplexity || 0) >= 0.15 ? 0.08 : 0;

      const confidence =
        clamp(
          (
            Number(
              candidate
                ?.confidence
            ) || 0
          ) /
          100
        );


      const illegalPenalty =
        Math.min(
          0.5,
          (
            Number(
              audit
                ?.illegal
            ) || 0
          ) *
          0.08
        );


      let score =
        grammar * 0.34 +
        confidence * 0.16 +
        subjectCoverage * 0.10 +
        (1 - duplicateRatio) *
          0.10 +
        (1 - unknownRatio) *
          0.10 +
        (1 - fragmentPenalty) *
          0.08 +
        (1 - orphanPenalty) *
          0.06 +
        (1 - punctuationPenalty) *
          0.06 -
        switchPenalty -
        illegalPenalty -
        structuralFlow.total * 0.72 -
        danglingTerminalPenalty * 0.44 +
        complexStructure * complexWeight;


      if (
        context
          ?.semanticSurface ===
        true
      ) {

        score +=
          this.options
            .semanticSurfaceBonus;

      
}


      score =
        clamp(score);


      return {

        score,

        valid:
          Boolean(text) &&
          score >=
            this.options
              .minimumScore &&
          duplicateRatio <=
            this.options
              .maxDuplicateRatio &&
          unknownRatio <=
            this.options
              .maxUnknownRatio &&
          structuralFlow.total < 0.42 &&
          structuralFlow.maxContentRun <= 5 &&
          danglingTerminalPenalty === 0 &&
          (complexWeight === 0 || complexStructure >= 0.22) &&
          (
            Number(
              audit?.illegal
            ) || 0
          ) === 0,

        diagnostics: {

          score:
            Number(
              score.toFixed(4)
            ),

          grammar:
            Number(
              grammar.toFixed(4)
            ),
          grammarAudit: audit?.advancedGrammar || null,

          duplicateRatio:
            Number(
              duplicateRatio
                .toFixed(4)
            ),

          unknownRatio:
            Number(
              unknownRatio
                .toFixed(4)
            ),

          fragmentPenalty:
            Number(
              fragmentPenalty
                .toFixed(4)
            ),

          structuralFlowPenalty:
            Number(
              structuralFlow.total
                .toFixed(4)
            ),

          danglingTerminalPenalty:
            Number(
              danglingTerminalPenalty
                .toFixed(4)
            ),

          maxContentRun:
            structuralFlow.maxContentRun,

          prepCount:
            structuralFlow.prepCount,

          repeatedRoots:
            structuralFlow.repeatedRoots,

          orphanPenalty:
            Number(
              orphanPenalty
                .toFixed(4)
            ),

          punctuationPenalty:
            Number(
              punctuationPenalty
                .toFixed(4)
            ),

          subjectCoverage:
            Number(
              subjectCoverage
                .toFixed(4)
            ),

          complexStructure:
            Number(complexStructure.toFixed(4)),

          switchDensity:
            Number(
              switchDensity
                .toFixed(4)
            ),

          illegal:
            Number(
              audit
                ?.illegal
            ) || 0
        
}

      
}
;

    
}


    select(
      candidates,
      context = {
}

    ) {

      const started =
        performance.now();


      const seen =
        new Set();


      const rows = [];


      for (
        const item of
        candidates || []
      ) {

        const candidate =
          item?.candidate ||
          item;


        if (
          !candidate ||
          !String(
            candidate?.text ||
            ''
          ).trim()
        ) {

          continue;

        
}


        const key =
          normalize(
            candidate.text
          );


        if (
          !key ||
          seen.has(key)
        ) {

          continue;

        
}


        seen.add(key);


        const result =
          this.evaluate(
            candidate,
            {

              ...context,
              semanticSurface:
                item
                  ?.semanticSurface ===
                true
            
}

          );


        rows.push({

          candidate,
          label:
            item?.label ||
            'candidate',
          semanticSurface:
            item
              ?.semanticSurface ===
            true,
          ...result
        
}
);

      
}


      rows.sort(
        (a, b) => {

          if (
            b.score !== a.score
          ) {

            return (
              b.score -
              a.score
            );

          
}


          return (
            (
              Number(
                b.candidate
                  ?.confidence
              ) || 0
            ) -
            (
              Number(
                a.candidate
                  ?.confidence
              ) || 0
            )
          );

        
}

      );


      const selected =
        rows.find(
          row =>
            row.valid
        ) ||
        rows[0] ||
        null;


      const diagnostics = {

        selectorMs:
          performance.now() -
          started,

        selectedLabel:
          selected?.label ||
          null,

        selectedScore:
          selected
            ? Number(
                selected
                  .score
                  .toFixed(4)
              )
            : 0,

        candidateCount:
          rows.length,

        candidates:
          rows
            .slice(0, 8)
            .map(
              row => ({

                label:
                  row.label,

                score:
                  Number(
                    row.score
                      .toFixed(4)
                  ),

                valid:
                  row.valid,

                confidence:
                  Number(
                    row
                      .candidate
                      ?.confidence
                  ) || 0,

                diagnostics:
                  row.diagnostics
              
}
)
            )
      
}
;


      this.evaluations++;

      this.lastDiagnostics =
        diagnostics;


      return {

        selected:
          selected
            ?.candidate ||
          null,

        selectedLabel:
          selected?.label ||
          null,

        selectedScore:
          selected?.score ||
          0,

        diagnostics
      
}
;

    
}


    status() {

      return {

        ready: true,
        role:
          'anti-word-salad-output-coherence-gate',
        evaluations:
          this.evaluations,
        lastDiagnostics:
          this.lastDiagnostics,
        architecture: {

          topicSpecificRules:
            false,
          extraDecoderPasses:
            0,
          canRejectBrokenMix:
            true,
          comparesExistingOutputs:
            true
        
}

      
}
;

    
}

  
}


  globalThis.VilotOutputCoherence =
    VilotOutputCoherence;

}
)();

