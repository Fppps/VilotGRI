/**
 * VilotNI 2.5 - TrainingKnowledge.js
 *
 * Active TrainingInfo retrieval for the 2.5 response path.
 *
 * Design:
 * - Words.json / Processor.js remain the authority for sayable words.
 * - TrainingInfo.json supplies concept keywords and structured relations.
 * - Description/Long_Definition may remain in the dataset for reference and
 *   learning, but they are not copied into 2.5 response scaffolds.
 * - Normalized entry metadata is precomputed once at load.
 * - Query retrieval uses the inverted term index plus subject n-grams instead
 *   of scanning every TrainingInfo subject on each request.
 * - A bounded query cache avoids repeating the same semantic search.
 */
(() => {

  'use strict';


  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(
      lo,
      Math.min(
        hi,
        Number.isFinite(Number(x))
          ? Number(x)
          : 0
      )
    );


  const normalize = value =>
    String(value || '')
      .toLowerCase()
      .replace(/[’]/g, "'")
      .replace(/[^a-z0-9'+.-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();


  class VilotTrainingKnowledge {

    constructor(processor, options = {
}
) {

      this.processor = processor;


      this.options = {

        resultLimit:
          Math.max(
            4,
            options.trainingInfoResultLimit | 0 ||
            12
          ),

        evidenceLimit:
          Math.max(
            48,
            options.trainingEvidenceLimit | 0 ||
            128
          ),

        branchWordLimit:
          Math.max(
            16,
            options.trainingBranchWordLimit | 0 ||
            34
          ),

        maxIndexTermsPerEntry:
          Math.max(
            12,
            options.trainingMaxIndexTermsPerEntry | 0 ||
            48
          ),

        queryCacheMax:
          Math.max(
            32,
            options.trainingQueryCacheMax | 0 ||
            256
          ),

        keywordResponseMode:
          options.keywordResponseMode !== false,

        keywordDefinitionDirectOutput:
          options.keywordDefinitionDirectOutput === true,

        keywordVectorMaxTerms:
          Math.max(
            24,
            Math.min(192, options.keywordVectorMaxTerms | 0 || 96)
          ),

        keywordExactWeight:
          clamp(options.keywordExactWeight ?? 1),

        keywordRelationWeight:
          clamp(options.keywordRelationWeight ?? 0.78)
      
}
;


      this.ready = false;

      this.entries = [];

      this.bySubject = new Map();

      this.byTerm = new Map();

      this.entryMeta = new WeakMap();

      this.queryCache = new Map();


      this.cacheHits = 0;

      this.cacheMisses = 0;


      this.lastSearch = null;

      this.loadMs = 0;

      this.sourceMeta = null;

    
}


    _tokens(text, max = 128) {

      const source =
        String(text || '');


      const fromProcessor =
        this.processor?.tokenize?.(
          source,
          max
        ) || [];


      if (fromProcessor.length) {

        return fromProcessor
          .filter(token =>
            token?.pos !== 'Punct'
          )
          .map(token =>
            normalize(
              token?.lower ||
              token?.surface
            )
          )
          .filter(Boolean)
          .slice(0, max);

      
}


      return normalize(source)
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, max);

    
}


    _array(value) {

      if (Array.isArray(value)) {

        return value;

      
}


      if (
        value == null ||
        value === ''
      ) {

        return [];

      
}


      return [value];

    
}


    _addIndex(map, key, entry) {

      const norm =
        normalize(key);


      if (!norm) return;


      let bucket =
        map.get(norm);


      if (!bucket) {

        map.set(
          norm,
          bucket = []
        );

      
}


      if (!bucket.includes(entry)) {

        bucket.push(entry);

      
}

    
}


    _cacheSet(key, value) {

      if (
        this.queryCache.size >=
        this.options.queryCacheMax
      ) {

        const first =
          this.queryCache
            .keys()
            .next();


        if (!first.done) {

          this.queryCache.delete(
            first.value
          );

        
}

      
}


      this.queryCache.set(
        key,
        value
      );


      return value;

    
}


    _metadata(entry) {

      let meta =
        this.entryMeta.get(entry);


      if (meta) return meta;


      const subject =
        normalize(entry?.Subject);


      const aliases =
        this._array(entry?.Aliases)
          .map(normalize)
          .filter(Boolean);


      const keywords =
        this._array(entry?.Keywords)
          .map(normalize)
          .filter(Boolean);


      const related =
        this._array(entry?.Related_To)
          .map(normalize)
          .filter(Boolean);


      const subjectTokens =
        new Set(
          this._tokens(
            subject,
            32
          )
        );


      const aliasTokens =
        new Set(
          aliases.flatMap(value =>
            this._tokens(
              value,
              16
            )
          )
        );


      const keywordTokens =
        new Set(
          keywords.flatMap(value =>
            this._tokens(
              value,
              8
            )
          )
        );


      const relatedTokens =
        new Set(
          related.flatMap(value =>
            this._tokens(
              value,
              12
            )
          )
        );


      // Response vocabulary is intentionally built from explicit keyword and
      // structured-relation fields. Description and Long_Definition are not
      // response-word reservoirs in keyword-response mode.
      const lexicalValues = [
        entry?.Subject,
        ...this._array(entry?.Keywords),
        ...this._array(entry?.Search_Terms),
        ...this._array(entry?.Semantic_Slots),
        ...this._array(entry?.Aliases),
        ...this._array(entry?.Related_To),
        entry?.Purpose,
        entry?.Mechanism,
        ...this._array(entry?.Capabilities),
        ...this._array(entry?.Limitations),
        ...this._array(entry?.Examples),
        ...this._array(entry?.Contrasts_With),
        ...this._array(entry?.Part_Of),
        ...this._array(entry?.Has_Parts),
        ...this._array(entry?.Causes),
        ...this._array(entry?.Effects),
        ...this._array(entry?.Properties),
        entry?.AI_Subdomain,
        ...this._array(entry?.Evaluation_Criteria),
        ...this._array(entry?.Failure_Modes),
        ...this._array(entry?.Tradeoffs)
      ];

      const definitionTokens = new Set([
        ...this._tokens(entry?.Description, 96),
        ...this._tokens(entry?.Long_Definition, 160)
      ]);


      const lexicalTokens = [];

      const lexicalSeen =
        new Set();


      for (
        const value of
        lexicalValues
      ) {

        for (
          const token of
          this._tokens(
            value,
            128
          )
        ) {

          if (
            !token ||
            lexicalSeen.has(token)
          ) {

            continue;

          
}


          lexicalSeen.add(token);

          lexicalTokens.push(token);


          if (
            lexicalTokens.length >=
            256
          ) {

            break;

          
}

        
}


        if (
          lexicalTokens.length >=
          256
        ) {

          break;

        
}

      
}


      meta = {

        subject,
        aliases,
        keywords,
        related,

        subjectTokens,
        aliasTokens,
        keywordTokens,
        relatedTokens,

        aliasSet:
          new Set(aliases),

        keywordSet:
          new Set(keywords),

        relatedSet:
          new Set(related),

        trust:
          clamp(
            entry?.Trust ?? 0.72
          ),

        sourceType:
          String(
            entry?.Source_Type || ''
          ).toLowerCase(),

        lexicalTokens,
        responseKeywordTokens: new Set(lexicalTokens),
        definitionTokens
      
}
;


      this.entryMeta.set(
        entry,
        meta
      );


      return meta;

    
}


    _indexEntry(entry) {

      const meta =
        this._metadata(entry);


      if (meta.subject) {

        this._addIndex(
          this.bySubject,
          meta.subject,
          entry
        );

      
}


      const terms =
        new Set([
          meta.subject,
          ...meta.aliases,
          ...meta.keywords,
          ...meta.related,
          ...meta.subjectTokens,
          ...meta.aliasTokens,
          ...meta.keywordTokens,
          ...meta.relatedTokens
        ]);


      let count = 0;


      for (
        const term of
        terms
      ) {

        if (!term) continue;


        this._addIndex(
          this.byTerm,
          term,
          entry
        );


        if (
          ++count >=
          this.options
            .maxIndexTermsPerEntry
        ) {

          break;

        
}

      
}

    
}


    loadObject(data) {

      const started =
        performance?.now?.() ??
        Date.now();


      const rawEntries =
        Object.values(
          data?.Training_Entries ||
          {
}

        );


      this.entries =
        rawEntries.filter(
          entry =>
            entry &&
            typeof entry === 'object'
        );


      this.bySubject.clear();

      this.byTerm.clear();


      this.entryMeta =
        new WeakMap();


      this.queryCache.clear();

      this.cacheHits = 0;

      this.cacheMisses = 0;


      for (
        const entry of
        this.entries
      ) {

        this._indexEntry(entry);

      
}


      this.sourceMeta = {

        schemaVersion:
          data?.Schema_Version ??
          null,

        format:
          data?.Format ||
          null,

        role:
          data?.Role ||
          null,

        entryCount:
          this.entries.length
      
}
;


      this.ready = true;


      this.loadMs =
        (
          performance?.now?.() ??
          Date.now()
        ) - started;


      return this.status();

    
}


    async load(
      url = './TrainingInfo.json'
    ) {

      const started =
        performance.now();


      const response =
        await fetch(url);


      if (!response.ok) {

        throw new Error(
          `TrainingInfo load failed: ${response.status}`
        );

      
}


      const data =
        await response.json();


      this.loadObject(data);


      this.loadMs =
        performance.now() -
        started;


      return this.status();

    
}


    _responseUseAllowed(
      entry,
      plan,
      analysis
    ) {

      if (
        entry?.Decoder_Knowledge_Eligible ===
        false
      ) {

        return false;

      
}


      const use =
        entry?.Response_Use;


      if (
        !use ||
        typeof use !== 'object'
      ) {

        return true;

      
}


      const role =
        String(
          plan?.requestedRole ||
          analysis?.requestedSlot ||
          'content'
        ).toLowerCase();


      const intent =
        String(
          plan?.intent ||
          analysis?.intent ||
          ''
        ).toLowerCase();


      if (
        role === 'identity' ||
        intent === 'definition'
      ) {

        return (
          use.Definition !== false ||
          use.General === true
        );

      
}


      if (
        role === 'mechanism' ||
        role === 'cause' ||
        role === 'explanation' ||
        intent === 'mechanism' ||
        intent === 'causal' ||
        intent === 'explanation'
      ) {

        return (
          use.How === true ||
          use.General === true ||
          use.Definition === true
        );

      
}


      if (role === 'truth') {

        return (
          use.Yes_No === true ||
          use.General === true
        );

      
}


      return (
        use.General !== false ||
        use.Definition === true ||
        use.How === true
      );

    
}


    _score(
      entry,
      queryPhrase,
      queryTokens,
      focusSeeds,
      plan,
      analysis
    ) {

      if (
        !this._responseUseAllowed(
          entry,
          plan,
          analysis
        )
      ) {

        return 0;

      
}


      const meta =
        this._metadata(entry);


      let score = 0;


      if (
        meta.subject &&
        (
          queryPhrase.includes(
            meta.subject
          ) ||
          meta.subject.includes(
            queryPhrase
          )
        )
      ) {

        score += 4.2;

      
}


      for (
        const alias of
        meta.aliases
      ) {

        if (
          alias &&
          (
            queryPhrase.includes(
              alias
            ) ||
            alias.includes(
              queryPhrase
            )
          )
        ) {

          score += 3.5;

        
}

      
}


      let subjectHits = 0;

      let aliasHits = 0;

      let keywordHits = 0;

      let relatedHits = 0;


      for (
        const token of
        queryTokens
      ) {

        if (
          meta.subjectTokens.has(
            token
          )
        ) {

          subjectHits++;

        
}


        if (
          meta.aliasTokens.has(
            token
          )
        ) {

          aliasHits++;

        
}


        if (
          meta.keywordTokens.has(
            token
          )
        ) {

          keywordHits++;

        
}


        if (
          meta.relatedTokens.has(
            token
          )
        ) {

          relatedHits++;

        
}

      
}


      score +=
        subjectHits * 1.35;


      score +=
        aliasHits * 1.10;


      score +=
        keywordHits * 1.25;


      score +=
        relatedHits * 0.42;


      for (
        const seed of
        focusSeeds
      ) {

        if (
          meta.subject === seed
        ) {

          score += 2.6;

        
}


        if (
          meta.keywordSet.has(seed)
        ) {

          score += 1.25;

        
}


        if (
          meta.aliasSet.has(seed)
        ) {

          score += 1.4;

        
}

      
}


      const explanationMode =
        analysis?.explanationMode ||
        plan?.explanationMode ||
        null;


      if (explanationMode) {

        const mechanismCount = this._array(entry?.Mechanism).length + (entry?.Mechanism ? 1 : 0);

        const causeCount = this._array(entry?.Causes).length;

        const effectCount = this._array(entry?.Effects).length;

        const purposeCount = entry?.Purpose ? 1 : 0;


        if (explanationMode === 'how') {

          score += Math.min(2.4, mechanismCount * 1.25 + effectCount * 0.35 + purposeCount * 0.25);

        
}
 else if (explanationMode === 'why') {

          score += Math.min(2.4, causeCount * 0.95 + mechanismCount * 0.65 + effectCount * 0.35);

        
}
 else if (explanationMode === 'both') {

          score += Math.min(3.0, causeCount * 0.75 + mechanismCount * 0.85 + effectCount * 0.35 + purposeCount * 0.2);

        
}

      
}


      score *=
        0.68 +
        meta.trust * 0.32;


      if (
        meta.sourceType ===
        'curated'
      ) {

        score += 0.45;

      
}


      if (
        entry
          ?.Decoder_Knowledge_Eligible ===
        true
      ) {

        score += 0.25;

      
}


      return Math.max(
        0,
        score
      );

    
}


    _subjectNgrams(
      sequence,
      maxN = 4
    ) {

      const rows = [];

      const seen = new Set();

      const n = sequence.length;


      for (
        let size =
          Math.min(
            maxN,
            n
          );

        size >= 2;

        size--
      ) {

        for (
          let i = 0;

          i + size <= n;

          i++
        ) {

          const phrase =
            sequence
              .slice(
                i,
                i + size
              )
              .join(' ');


          if (
            !phrase ||
            seen.has(phrase)
          ) {

            continue;

          
}


          seen.add(phrase);

          rows.push(phrase);

        
}

      
}


      return rows;

    
}


    search(
      promptText,
      analysis = null,
      plan = null,
      options = {
}

    ) {

      if (!this.ready) {

        return {

          rows: [],
          diagnostics: {

            ready: false,
            rowCount: 0
          
}
,
          supportWords:
            new Set()
        
}
;

      
}


      const started =
        performance.now();


      const deadlineAt = Number(options.deadlineAt);

      const deadlineReached = (reserveMs = 0.45) =>
        Number.isFinite(deadlineAt) &&
        performance.now() >= deadlineAt - Math.max(0, reserveMs);

      let deadlineTruncated = false;


      const queryPhrase =
        normalize(promptText);


      const querySequence =
        this._tokens(
          promptText,
          64
        );


      const queryTokens =
        Array.from(
          new Set([
            ...querySequence,
            ...(
              analysis
                ?.contentWords ||
              []
            ).map(normalize),
            ...(
              analysis?.roots ||
              []
            ).map(normalize),
            ...(
              plan?.focusSeeds ||
              []
            ).map(normalize),
            ...(
              plan?.subjectWords ||
              []
            ).map(normalize)
          ].filter(Boolean))
        );


      const focusSeeds =
        new Set([
          ...(
            plan?.focusSeeds ||
            []
          ),
          ...(
            plan?.subjectWords ||
            []
          ),
          ...(
            analysis
              ?.contentWords ||
            []
          )
        ]
          .map(normalize)
          .filter(Boolean)
        );


      const limit =
        Math.max(
          4,
          options.limit | 0 ||
          this.options.resultLimit
        );


      const cacheKey = [
        queryPhrase,
        String(
          plan?.intent ||
          analysis?.intent ||
          ''
        ),
        String(
          plan?.requestedRole ||
          analysis?.requestedSlot ||
          ''
        ),
        queryTokens.join(','),
        limit
      ].join('|');


      const cached =
        this.queryCache.get(
          cacheKey
        );


      if (cached) {

        this.cacheHits++;


        const diagnostics = {

          ...cached.diagnostics,
          searchMs:
            performance.now() -
            started,
          cacheHit: true,
          cacheSize:
            this.queryCache.size
        
}
;


        this.lastSearch =
          diagnostics;


        return {

          rows:
            cached.rows,
          diagnostics,
          supportWords:
            cached.supportWords
        
}
;

      
}


      this.cacheMisses++;


      const candidates =
        new Set();


      for (
        const token of
        queryTokens
      ) {

        if (deadlineReached()) {

          deadlineTruncated = true;

          break;

        
}

        for (
          const entry of
          this.byTerm.get(token) ||
          []
        ) {

          candidates.add(entry);

        
}


        for (
          const entry of
          this.bySubject.get(token) ||
          []
        ) {

          candidates.add(entry);

        
}

      
}


      for (
        const phrase of
        this._subjectNgrams(
          querySequence,
          4
        )
      ) {

        if (deadlineReached()) {

          deadlineTruncated = true;

          break;

        
}

        for (
          const entry of
          this.bySubject.get(phrase) ||
          []
        ) {

          candidates.add(entry);

        
}


        for (
          const entry of
          this.byTerm.get(phrase) ||
          []
        ) {

          candidates.add(entry);

        
}

      
}


      let ranked = [];

      for (const entry of candidates) {

        if (deadlineReached()) {

          deadlineTruncated = true;

          break;

        
}

        const score = this._score(
          entry,
          queryPhrase,
          queryTokens,
          focusSeeds,
          plan,
          analysis
        );

        if (score > 0) ranked.push({
 entry, score, expanded: false 
}
);

      
}

      ranked.sort((a, b) => b.score - a.score);


      // One relation hop keeps nearby concepts reachable without converting
      // TrainingInfo into a canned response table.
      const expanded =
        new Map();


      for (
        const row of
        ranked.slice(0, 5)
      ) {

        if (deadlineReached()) {

          deadlineTruncated = true;

          break;

        
}

        const meta =
          this._metadata(
            row.entry
          );


        for (
          const relation of
          meta.related
        ) {

          for (
            const linked of
            this.bySubject.get(
              relation
            ) || []
          ) {

            if (
              candidates.has(
                linked
              )
            ) {

              continue;

            
}


            const score =
              this._score(
                linked,
                queryPhrase,
                queryTokens,
                focusSeeds,
                plan,
                analysis
              ) +
              row.score * 0.28;


            const old =
              expanded.get(
                linked
              );


            if (
              !old ||
              score > old.score
            ) {

              expanded.set(
                linked,
                {

                  entry: linked,
                  score,
                  expanded: true,
                  relation
                
}

              );

            
}

          
}

        
}

      
}


      ranked =
        ranked
          .concat(
            Array.from(
              expanded.values()
            )
          )
          .sort(
            (a, b) =>
              b.score -
              a.score
          )
          .slice(
            0,
            limit
          );


      const supportWords =
        new Set();


      for (
        const row of
        ranked
      ) {

        for (
          const token of
          this._metadata(
            row.entry
          ).lexicalTokens
        ) {

          supportWords.add(
            token
          );

        
}

      
}


      const diagnostics = {

        searchMs:
          performance.now() -
          started,

        cacheHit: false,

        cacheSize:
          Math.min(
            this.options.queryCacheMax,
            this.queryCache.size + 1
          ),

        queryTokens:
          queryTokens.slice(
            0,
            24
          ),

        rowCount:
          ranked.length,

        candidateEntryCount:
          candidates.size,

        supportWordCount:
          supportWords.size,

        deadlineTruncated,

        top:
          ranked
            .slice(0, 8)
            .map(row => ({

              id:
                row.entry?.ID ||
                null,

              subject:
                row.entry
                  ?.Subject ||
                '',

              score:
                Number(
                  row.score.toFixed(4)
                ),

              trust:
                Number(
                  row.entry?.Trust ??
                  0
                ),

              sourceType:
                row.entry
                  ?.Source_Type ||
                '',

              expanded:
                Boolean(
                  row.expanded
                )
            
}
))
      
}
;


      this.lastSearch =
        diagnostics;


      const result = {

        rows: ranked,
        diagnostics,
        supportWords
      
}
;


      if (!deadlineTruncated) {

        this._cacheSet(
          cacheKey,
          result
        );

      
}


      return result;

    
}


    _branchTexts(
      result,
      branchIndex,
      analysis = null,
      plan = null
    ) {
      const rows = result?.rows || [];
      if (!rows.length) return [];

      const first = rows[0]?.entry || {};
      const second = rows[1]?.entry || {};
      const third = rows[2]?.entry || {};
      const mode = analysis?.explanationMode || plan?.explanationMode || null;
      const role = String(
        plan?.requestedRole ||
        analysis?.requestedSlot ||
        analysis?.questionState?.requestedRole ||
        'content'
      ).toLowerCase();
      const values = [];
      const add = (...items) => values.push(...items.flatMap(value => Array.isArray(value) ? value : [value]));

      // Branches select different structured keyword neighborhoods. Raw
      // Description/Long_Definition text is deliberately excluded so the decoder
      // has to construct a response from Words + explicit structured metadata.
      // Identity questions are intentionally narrower than general topical
      // questions: "what is X?" should stay on X, its aliases, class and
      // properties instead of walking outward through Related_To/Search_Terms.
      if (mode === 'why') {
        if (branchIndex === 0) add(first.Causes, first.Mechanism, first.Effects, first.Keywords);
        else if (branchIndex === 1) add(first.Causes, first.Related_To, first.Purpose, first.Properties);
        else if (branchIndex === 2) add(second.Causes, first.Effects, second.Mechanism, first.Keywords);
        else add(first.Causes, first.Limitations, first.Related_To, first.Search_Terms);
      } else if (mode === 'how') {
        if (branchIndex === 0) add(first.Mechanism, first.Has_Parts, first.Capabilities, first.Keywords);
        else if (branchIndex === 1) add(first.Mechanism, first.Part_Of, first.Related_To, first.Purpose);
        else if (branchIndex === 2) add(first.Mechanism, first.Effects, second.Capabilities, first.Search_Terms);
        else add(first.Mechanism, first.Capabilities, first.Properties, first.Limitations);
      } else if (mode === 'both') {
        if (branchIndex === 0) add(first.Causes, first.Mechanism, first.Effects, first.Keywords);
        else if (branchIndex === 1) add(first.Mechanism, first.Has_Parts, first.Capabilities, first.Purpose);
        else if (branchIndex === 2) add(first.Causes, first.Effects, second.Mechanism, second.Causes);
        else add(first.Mechanism, first.Related_To, first.Properties, first.Limitations);
      } else if (role === 'identity') {
        if (branchIndex === 0) {
          add(first.Subject, first.Aliases, first.Keywords, first.Knowledge_Kind, first.Properties);
        } else if (branchIndex === 1) {
          add(first.Aliases, first.Knowledge_Kind, first.Category, first.Domain, first.Keywords);
        } else if (branchIndex === 2) {
          add(first.Properties, first.Part_Of, first.Has_Parts, first.Keywords, second.Keywords);
        } else {
          add(first.Subject, first.Aliases, first.Keywords, first.Properties, first.Contrasts_With);
        }
      } else if (branchIndex === 0) {
        add(first.Keywords, first.Search_Terms, first.Properties, first.Capabilities, first.Semantic_Slots);
      } else if (branchIndex === 1) {
        add(first.Aliases, first.Related_To, first.Part_Of, first.Has_Parts, first.Keywords);
      } else if (branchIndex === 2) {
        add(first.Mechanism, first.Capabilities, first.Causes, first.Effects, first.Keywords, second.Keywords);
      } else {
        add(first.Keywords, second.Keywords, third.Keywords, first.Related_To, second.Related_To,
            first.Contrasts_With, first.Limitations, first.Evaluation_Criteria, first.Failure_Modes, first.Tradeoffs);
      }

      return values
        .map(value => String(value || '').trim())
        .filter(Boolean);
    }



    _branchKeywordVector(result, branchIndex = 0, analysis = null, plan = null) {
      const rows = result?.rows || [];
      if (!rows.length) return [];

      const role = String(
        plan?.requestedRole ||
        analysis?.requestedSlot ||
        analysis?.questionState?.requestedRole ||
        'content'
      ).toLowerCase();

      const weighted = new Map();
      const addToken = (token, weight, source) => {
        const word = normalize(token);
        if (!word) return;
        const lexical = this.processor?.resolveSayableWord?.(word, { allowRootFallback: true }) || null;
        if (!lexical || lexical.active === false || !lexical.contentWord) return;
        const key = lexical.lower || word;
        const previous = weighted.get(key);
        const row = { word: key, surface: lexical.surface || key, pos: lexical.pos || 'Other', weight: clamp(weight), source };
        if (!previous || row.weight > previous.weight) weighted.set(key, row);
      };
      const addValue = (value, weight, source) => {
        for (const token of this._tokens(value, 96)) addToken(token, weight, source);
      };

      const first = rows[0]?.entry || {};
      addValue(first.Subject, 1, 'subject');
      for (const value of this._array(first.Keywords)) addValue(value, this.options.keywordExactWeight, 'keyword');
      for (const value of this._array(first.Aliases)) addValue(value, 0.94, 'alias');

      if (role === 'identity') {
        // Identity answers use near-subject descriptors only. Search_Terms and
        // Related_To are retrieval fields, not definition content, and were the
        // main source of "AI -> model -> training -> network -> MoE" drift.
        addValue(first.Knowledge_Kind, 0.88, 'knowledge-kind');
        addValue(first.Category, 0.72, 'category');
        addValue(first.Domain, 0.70, 'domain');
        for (const value of this._array(first.Properties)) addValue(value, 0.82, 'property');
        for (const value of this._array(first.Part_Of)) addValue(value, 0.66, 'part-of');
        for (const value of this._array(first.Has_Parts)) addValue(value, 0.62, 'has-parts');
      } else {
        for (const value of this._array(first.Search_Terms)) addValue(value, 0.92, 'search-term');
        for (const value of this._array(first.Related_To)) addValue(value, this.options.keywordRelationWeight, 'relation');
      }

      const branchTexts = this._branchTexts(result, branchIndex, analysis, plan);
      for (const value of branchTexts) addValue(value, role === 'identity' ? 0.88 : 0.84, `branch-${branchIndex + 1}`);

      // Query words are anchors rather than copied answers. They keep each
      // branch tied to the user's requested concept even when the lexical graph
      // contains many strongly connected neighbors.
      for (const value of analysis?.contentWords || []) addValue(value, 0.98, 'prompt-keyword');
      const identityRoleWords = new Set(['type','kind','system','process','concept','object']);
      for (const value of plan?.focusSeeds || []) {
        const normalizedFocus = normalize(value);
        const weight = role === 'identity' && identityRoleWords.has(normalizedFocus) ? 0.58 : (role === 'identity' ? 0.90 : 0.94);
        addValue(value, weight, identityRoleWords.has(normalizedFocus) ? 'identity-role-word' : 'focus-seed');
      }

      return Array.from(weighted.values())
        .sort((a, b) => b.weight - a.weight || a.word.localeCompare(b.word))
        .slice(0, this.options.keywordVectorMaxTerms);
    }

    buildStaticPool(
      basePool,
      result,
      branchIndex = 0,
      analysis = null,
      plan = null
    ) {
      const map = new Map();
      const add = (entry, sourceScore, focusScore, source) => {
        if (!entry || entry.active === false || !entry.lower) return;
        const row = {
          entry,
          sourceScore: clamp(sourceScore, 0, 1.5),
          focusScore: clamp(focusScore, 0, 1),
          source
        };
        const old = map.get(entry.lower);
        if (!old || row.sourceScore > old.sourceScore || row.focusScore > old.focusScore) map.set(entry.lower, row);
      };

      for (const row of basePool || []) add(row.entry, row.sourceScore, row.focusScore ?? 0.5, row.source || 'base-evidence');

      const keywordVector = this._branchKeywordVector(result, branchIndex, analysis, plan);
      for (const item of keywordVector) {
        const entry = this.processor?.resolveSayableWord?.(item.word, { allowRootFallback: true });
        if (!entry) continue;
        add(
          entry,
          0.70 + item.weight * 0.48,
          0.58 + item.weight * 0.40,
          `training-keyword-${item.source}`
        );
      }

      return Array.from(map.values());
    }



    buildBranchAnalysis(
      analysis,
      result,
      branchIndex = 0,
      plan = null
    ) {
      if (!analysis || !result?.rows?.length) return analysis;

      const vector = this._branchKeywordVector(result, branchIndex, analysis, plan);
      if (!vector.length) return analysis;

      const original = analysis.responseScaffold || {};
      const explanationMode = analysis?.explanationMode || plan?.explanationMode || null;
      const words = Array.isArray(original.words) ? original.words.slice() : [];
      const surfaces = Array.isArray(original.surfaces) ? original.surfaces.slice() : words.slice();
      const pos = Array.isArray(original.pos) ? original.pos.slice(0, words.length) : [];

      // Why/How may append a relation operator only after a real declarative
      // scaffold already exists. An empty scaffold must never become ["by"]
      // or ["because"], because scaffoldCoreLock would then require a
      // preposition/subordinator as the first generated word and make the
      // candidate space impossible. MechanismGraph/ExplanationPlanner carry
      // the relation when no lexical scaffold exists yet.
      if (explanationMode) {
        const bridgeSet = new Set(['because','by','through','using','with']);
        while (words.length && bridgeSet.has(String(words[words.length - 1] || '').toLowerCase())) {
          words.pop(); surfaces.pop(); if (pos.length > words.length) pos.pop();
        }
        const hasPredicate = Boolean(original?.predicate) || pos.some(tag => ['Verb','Aux','Modal','Auxiliary Verb','Modal Verb'].includes(String(tag || '')));
        const hasLexicalCore = words.length >= 2 && hasPredicate;
        if (hasLexicalCore) {
          const bridgeWord = explanationMode === 'why'
            ? 'because'
            : explanationMode === 'how'
              ? 'by'
              : (branchIndex % 2 === 0 ? 'because' : 'by');
          const bridge = this.processor?.resolveSayableWord?.(bridgeWord, { allowRootFallback: false }) || null;
          if (bridge) { words.push(bridge.lower); surfaces.push(bridge.surface); pos.push(bridge.pos); }
        }
      }

      const keywordTargets = vector.map(row => row.word);
      const keywordWeights = Object.create(null);
      const keywordSources = Object.create(null);
      for (const row of vector) {
        keywordWeights[row.word] = row.weight;
        keywordSources[row.word] = row.source;
      }

      const scaffold = {
        ...original,
        source: `training-keyword-branch-${branchIndex + 1}`,
        kind: explanationMode
          ? `training-keyword-explanation-${explanationMode}`
          : 'training-keyword-guided-scaffold',
        words,
        surfaces,
        pos,
        subject: original.subject || (analysis.contentWords || []).slice(0, 4),
        predicate: original.predicate || 'is',
        requestedRole: explanationMode === 'both'
          ? (branchIndex % 2 === 0 ? 'cause' : 'mechanism')
          : original.requestedRole || analysis.requestedSlot || null,
        requestedRoles: analysis.requestedSlots || original.requestedRoles || [],
        explanationMode,
        extensionPOS: original.extensionPOS || [],
        keywordTargets,
        keywordWeights,
        keywordResponseMode: true
      };

      return {
        ...analysis,
        responseScaffold: scaffold,
        branchKeywords: keywordTargets,
        trainingKnowledge: {
          branch: branchIndex + 1,
          subjects: result.rows.slice(0, 4).map(row => row.entry?.Subject || '').filter(Boolean),
          keywordResponseMode: true,
          keywordTargets,
          keywordWeights,
          keywordSources,
          copiedDefinitionText: false
        }
      };
    }



    wordSupportSet(result) {

      if (
        result
          ?.supportWords
        instanceof Set
      ) {

        return result.supportWords;

      
}


      const set =
        new Set();


      for (
        const row of
        result?.rows ||
        []
      ) {

        for (
          const token of
          this._metadata(
            row.entry
          ).lexicalTokens
        ) {

          set.add(token);

        
}

      
}


      return set;

    
}


    status() {

      return {

        ready:
          this.ready,

        role:
          'training-info-keyword-vector-evidence',

        source:
          this.sourceMeta,

        loadMs:
          this.loadMs,

        indexedSubjects:
          this.bySubject.size,

        indexedTerms:
          this.byTerm.size,

        queryCacheSize:
          this.queryCache.size,

        queryCacheMax:
          this.options.queryCacheMax,

        cacheHits:
          this.cacheHits,

        cacheMisses:
          this.cacheMisses,

        lastSearch:
          this.lastSearch
      
}
;

    
}

  
}


  globalThis.VilotTrainingKnowledge =
    VilotTrainingKnowledge;

}
)();

