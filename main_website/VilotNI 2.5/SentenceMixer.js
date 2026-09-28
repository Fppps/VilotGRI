/**
 * VilotNI 2.5 - SentenceMixer.js
 * System 34 foundation capacity expansion.
 *
 * Four full candidate sentences still interfere under the original rule:
 * confidence-close OR sufficiently alike. System 33 makes "alike" structural,
 * not merely lexical. Proposition roles, relation families, subject continuity,
 * mechanism stages, polarity and clause connectors now participate without
 * reducing candidate breadth or replacing matrix/amplitude interference.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));
  const normalize = value => String(value || '').toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9'+.\-]+/g, ' ').replace(/\s+/g, ' ').trim();
  const uniq = rows => Array.from(new Set((rows || []).filter(Boolean)));

  class VilotSentenceMixer {
    constructor(processor, options = {}, matrixEngine = null, ruleStore = null) {
      this.processor = processor;
      this.ruleStore = ruleStore;
      this.matrix = matrixEngine || (globalThis.VilotMatrixEngine ? new globalThis.VilotMatrixEngine({ matrixMaxRows: 64, matrixMaxCols: 12, matrixTopK: 8 }) : null);
      this.options = {
        confidenceWindowPct: Math.max(0.1, Number(options.interferenceSentenceConfidenceWindowPct) || Number(options.interferenceWordConfidenceWindowPct) || 2),
        similarityThreshold: clamp(options.interferenceSentenceSimilarityThreshold ?? 0.58),
        rootWeight: clamp(options.interferenceSentenceRootWeight ?? 0.18),
        lexicalWeight: clamp(options.interferenceSentenceLexicalWeight ?? 0.32),
        orderWeight: clamp(options.interferenceSentenceOrderWeight ?? 0.14),
        lengthWeight: clamp(options.interferenceSentenceLengthWeight ?? 0.06),
        structureWeight: clamp(options.interferenceSentenceStructureWeight ?? 0.30),
        minParticipants: Math.max(2, Math.min(4, options.interferenceSentenceMinParticipants | 0 || 2)),
        amplitudeEnabled: options.interferenceSentenceAmplitudeEnabled !== false,
        amplitudeCoupling: clamp(options.interferenceSentenceAmplitudeCoupling ?? 0.12),
        amplitudeThreshold: clamp(options.interferenceSentenceAmplitudeThreshold ?? 0.54)
      };
      this.analyses = 0;
      this.structuralComparisons = 0;
      this.last = null;
    }

    _rules() { return this.ruleStore?.sentenceCompositionRules?.() || this.ruleStore?.get?.('sentenceComposition') || {}; }
    _reasoningRules() { return this.ruleStore?.structuredReasoningRules?.() || {}; }

    _tokens(text) {
      const source = String(text || '');
      const parsed = this.processor?.tokenize?.(source, 256) || [];
      if (parsed.length) return parsed.filter(row => row && row.pos !== 'Punct').map(row => normalize(row.lower || row.surface)).filter(Boolean);
      return normalize(source).split(/\s+/).filter(Boolean);
    }

    _roots(tokens) {
      return (tokens || []).map(word => {
        const entry = this.processor?.lookup?.(word) || this.processor?.resolveSayableWord?.(word, { allowRootFallback: true }) || null;
        return normalize(entry?.root || entry?.lemma || word);
      }).filter(Boolean);
    }

    _setJaccard(a, b) {
      const left = new Set(a || []), right = new Set(b || []);
      if (!left.size || !right.size) return 0;
      let intersection = 0;
      for (const value of left) if (right.has(value)) intersection++;
      const union = left.size + right.size - intersection;
      return union ? intersection / union : 0;
    }

    _bigramSet(tokens) {
      const set = new Set();
      for (let i = 1; i < (tokens?.length || 0); i++) set.add(`${tokens[i - 1]}\u0000${tokens[i]}`);
      return set;
    }

    _bigramOverlap(a, b) {
      const left = this._bigramSet(a || []), right = this._bigramSet(b || []);
      if (!left.size || !right.size) return 0;
      let intersection = 0;
      for (const value of left) if (right.has(value)) intersection++;
      return intersection / Math.max(left.size, right.size, 1);
    }

    _lengthSimilarity(a, b) {
      const x = Math.max(1, a?.length || 0), y = Math.max(1, b?.length || 0);
      return Math.min(x, y) / Math.max(x, y);
    }

    _relationFamily(relation) {
      const key = normalize(relation).replace(/\s+/g, '-');
      for (const [family, members] of Object.entries(this._reasoningRules()?.Relation_Families || {})) {
        if ((members || []).map(v => normalize(v).replace(/\s+/g, '-')).includes(key)) return family;
      }
      return key || 'relation';
    }

    _textRelations(text) {
      const clean = ` ${normalize(text)} `;
      const rows = [];
      const cues = {
        cause: ['because','causes','caused','due to'], result: ['so','therefore','results in','leads to','produces'],
        condition: ['if','unless','provided'], contrast: ['but','although','though','whereas','however'],
        temporal: ['before','after','while','during','then','later','earlier'], purpose: ['so that','in order to'],
        mechanism: ['through','using','by','works','processes','evaluates','selects','updates'],
        comparison: ['than','similar','different','compared'], feedback: ['feedback','returns','replays','reinforces','rolls back']
      };
      for (const [type, phrases] of Object.entries(cues)) {
        if (phrases.some(phrase => clean.includes(` ${normalize(phrase)} `))) rows.push(type);
      }
      return uniq(rows);
    }

    _extractPlanClauses(candidate) {
      const diag = candidate?.diagnostics || {};
      const sources = [
        diag?.meaningFirst?.plan?.clauses,
        diag?.meaningFirst?.plan?.items,
        diag?.reasoning?.mechanismGraph?.stages,
        diag?.reasoning?.plan?.clauses,
        candidate?.clauses,
        candidate?.propositions
      ];
      for (const source of sources) if (Array.isArray(source) && source.length) return source.slice(0, 24);
      return [];
    }

    _structureOf(candidate, index = 0) {
      const text = String(candidate?.text || '');
      const tokens = this._tokens(text);
      const roots = this._roots(tokens);
      const clauses = this._extractPlanClauses(candidate);
      const relations = [];
      const subjects = [];
      const stages = [];
      const dependencies = [];

      for (const row of clauses) {
        const relation = normalize(row?.relation || row?.type || row?.requestedRole || row?.role || row?.slot || '');
        if (relation) relations.push(this._relationFamily(relation));
        const subject = normalize(row?.subject || row?.agent || row?.from || '');
        if (subject) subjects.push(...this._roots(subject.split(/\s+/)));
        const stage = normalize(row?.slot || row?.mechanismSlot || row?.role || '');
        if (/input|precondition|operation|transition|output|feedback|condition|constraint|termination/.test(stage)) stages.push(stage);
        const dep = normalize(row?.relationToParent || row?.dependency || '');
        if (dep) dependencies.push(dep);
      }

      relations.push(...this._textRelations(text));
      const connectors = tokens.filter(word => ['because','if','unless','although','though','while','after','before','when','whereas','since','until','and','but','or','so','then'].includes(word));
      const polarity = /\b(?:not|never|no|cannot|can't|without|neither|nor)\b/i.test(text) ? -1 : 1;
      const firstContent = roots.find(root => !['the','a','an','it','this','that','they','he','she','we','you','i'].includes(root)) || '';

      return {
        index,
        tokens,
        roots,
        clauseCount: Math.max(1, String(text).split(/[.;!?]+/).filter(Boolean).length, clauses.length),
        relations: uniq(relations),
        subjects: uniq(subjects.length ? subjects : firstContent ? [firstContent] : []),
        stages: uniq(stages),
        dependencies: uniq(dependencies),
        connectors: uniq(connectors),
        polarity,
        hasQuestion: /\?\s*$/.test(text),
        mechanismLike: stages.length > 0 || relations.includes('mechanism') || /\b(?:input|output|feedback|process|step|stage)\b/i.test(text)
      };
    }

    _structureSimilarity(a, b) {
      this.structuralComparisons++;
      if (!a || !b) return { score: 0, components: {} };
      if (a.polarity !== b.polarity) return { score: 0.08, components: { polarity: 0 } };
      const relation = this._setJaccard(a.relations, b.relations);
      const subject = this._setJaccard(a.subjects, b.subjects);
      const dependency = this._setJaccard(a.dependencies, b.dependencies);
      const stage = a.mechanismLike || b.mechanismLike ? this._setJaccard(a.stages, b.stages) : 0.72;
      const connector = this._setJaccard(a.connectors, b.connectors);
      const clause = this._lengthSimilarity(new Array(a.clauseCount), new Array(b.clauseCount));
      const components = { relation, subject, dependency, stage, connector, clause, polarity: 1 };
      const score = clamp(relation * 0.28 + subject * 0.20 + dependency * 0.14 + stage * 0.14 + connector * 0.10 + clause * 0.14);
      return { score, components };
    }

    similarity(aText, bText) {
      const a = this._tokens(aText), b = this._tokens(bText);
      if (!a.length || !b.length) return 0;
      if (normalize(aText) === normalize(bText)) return 1;
      const lexical = this._setJaccard(a, b);
      const roots = this._setJaccard(this._roots(a), this._roots(b));
      const order = this._bigramOverlap(a, b);
      const length = this._lengthSimilarity(a, b);
      const total = this.options.lexicalWeight + this.options.rootWeight + this.options.orderWeight + this.options.lengthWeight || 1;
      return clamp((lexical * this.options.lexicalWeight + roots * this.options.rootWeight + order * this.options.orderWeight + length * this.options.lengthWeight) / total);
    }

    _pairSimilarity(left, right) {
      const lexicalScore = this.similarity(left.text, right.text);
      const structural = this._structureSimilarity(left.structure, right.structure);
      const baseWeight = this.options.lexicalWeight + this.options.rootWeight + this.options.orderWeight + this.options.lengthWeight;
      const total = Math.max(0.001, baseWeight + this.options.structureWeight);
      const combined = clamp((lexicalScore * baseWeight + structural.score * this.options.structureWeight) / total);
      return { lexicalScore, structuralScore: structural.score, structuralComponents: structural.components, combined };
    }

    analyze(candidates = []) {
      const rows = (candidates || []).slice(0, 4).map((candidate, index) => ({
        index,
        candidate,
        text: String(candidate?.text || '').trim(),
        confidence: Math.max(0, Math.min(100, Number(candidate?.confidence) || 0)),
        keywordAlignment: clamp(candidate?.diagnostics?.channels?.keywordResponseAlignment ?? 0),
        propositionCoverage: clamp(candidate?.diagnostics?.reasoning?.reasoningQuality?.propositionCoverage ?? candidate?.diagnostics?.channels?.structureCoverage ?? 0.5),
        mechanismCoverage: clamp(candidate?.diagnostics?.reasoning?.reasoningQuality?.mechanismCompleteness ?? candidate?.diagnostics?.channels?.mechanismCoverage ?? 0.5)
      })).filter(row => row.text);
      for (const row of rows) row.structure = this._structureOf(row.candidate, row.index);

      const pairs = [];
      const adjacency = new Map(rows.map(row => [row.index, new Set()]));
      for (let i = 0; i < rows.length; i++) {
        for (let j = i + 1; j < rows.length; j++) {
          const left = rows[i], right = rows[j];
          const confidenceGap = Math.abs(left.confidence - right.confidence);
          const sim = this._pairSimilarity(left, right);
          const confidenceClose = confidenceGap <= this.options.confidenceWindowPct;
          const sentenceAlike = sim.combined >= this.options.similarityThreshold;
          const eligible = confidenceClose || sentenceAlike;
          const relationCompatible = left.structure.polarity === right.structure.polarity || sim.structuralScore >= 0.76;
          const reason = confidenceClose && sentenceAlike ? 'confidence-and-structural-similarity' : confidenceClose ? 'confidence-window' : sentenceAlike ? 'sentence-structural-similarity' : 'not-eligible';
          pairs.push({
            a: left.index, b: right.index,
            confidenceGap: Number(confidenceGap.toFixed(3)),
            similarity: Number(sim.combined.toFixed(6)),
            lexicalSimilarity: Number(sim.lexicalScore.toFixed(6)),
            structuralSimilarity: Number(sim.structuralScore.toFixed(6)),
            structuralComponents: sim.structuralComponents,
            confidenceClose, sentenceAlike, relationCompatible,
            eligible: eligible && relationCompatible,
            rawEligible: eligible,
            reason: eligible && !relationCompatible ? 'polarity-or-structure-conflict' : reason
          });
          if (eligible && relationCompatible) {
            adjacency.get(left.index)?.add(right.index);
            adjacency.get(right.index)?.add(left.index);
          }
        }
      }

      const strongest = rows.slice().sort((a, b) => b.confidence - a.confidence)[0] || null;
      const participantSet = new Set();
      if (strongest) {
        const queue = [strongest.index];
        while (queue.length) {
          const current = queue.shift();
          if (participantSet.has(current)) continue;
          participantSet.add(current);
          for (const next of adjacency.get(current) || []) if (!participantSet.has(next)) queue.push(next);
        }
      }
      if (participantSet.size < this.options.minParticipants && strongest) {
        participantSet.clear();
        participantSet.add(strongest.index);
      }
      const participants = rows.filter(row => participantSet.has(row.index));
      const maxConfidence = participants.length ? Math.max(...participants.map(row => row.confidence)) : 0;

      let amplitude = null;
      const amplitudeWeights = new Map();
      if (this.options.amplitudeEnabled && this.matrix && participants.length >= 2) {
        const n = participants.length;
        this.matrix.ensure(n, 8);
        const coupling = new Float32Array(n * n);
        for (let i = 0; i < n; i++) {
          const row = participants[i];
          const related = pairs.filter(pair => pair.eligible && (pair.a === row.index || pair.b === row.index));
          const meanSimilarity = related.length ? related.reduce((sum, pair) => sum + pair.similarity, 0) / related.length : 0;
          const meanStructural = related.length ? related.reduce((sum, pair) => sum + pair.structuralSimilarity, 0) / related.length : 0;
          const degree = related.length / Math.max(1, rows.length - 1);
          this.matrix.setRow(i, i, [row.confidence / 100, meanSimilarity, meanStructural, degree, row.keywordAlignment, row.propositionCoverage, row.mechanismCoverage, row.structure.polarity > 0 ? 1 : 0]);
          for (let j = 0; j < n; j++) {
            if (i === j) { coupling[i * n + j] = 1; continue; }
            const pair = this.pair({ pairs }, participants[i].index, participants[j].index);
            coupling[i * n + j] = pair?.eligible ? clamp(0.35 + pair.similarity * 0.40 + pair.structuralSimilarity * 0.25) : 0;
          }
        }
        amplitude = this.matrix.amplitudeAmplify({
          rows: n,
          queryVector: [0.34, 0.14, 0.14, 0.06, 0.08, 0.10, 0.10, 0.04],
          prior: participants.map(row => row.confidence / 100),
          couplingMatrix: coupling, coherenceMatrix: coupling, couplingStride: n,
          coupling: this.options.amplitudeCoupling, markThreshold: this.options.amplitudeThreshold,
          roundsMin: 1, roundsMax: 3
        });
        for (let i = 0; i < n; i++) amplitudeWeights.set(participants[i].index, Number(amplitude.probabilities?.[i]) || 0);
      }

      let weighted = 0, weightTotal = 0;
      for (const row of participants) {
        const related = pairs.filter(pair => pair.eligible && (pair.a === row.index || pair.b === row.index));
        const meanSimilarity = related.length ? related.reduce((sum, pair) => sum + pair.similarity, 0) / related.length : 0;
        const meanStructural = related.length ? related.reduce((sum, pair) => sum + pair.structuralSimilarity, 0) / related.length : 0;
        const amplitudeWeight = amplitudeWeights.get(row.index) || 0;
        const qualitySupport = row.propositionCoverage * 0.55 + row.mechanismCoverage * 0.45;
        const weight = 1 + meanSimilarity * 0.7 + meanStructural * 0.7 + qualitySupport * 0.35 + amplitudeWeight * Math.max(1, participants.length);
        weighted += row.confidence * weight;
        weightTotal += weight;
      }
      const meanConfidence = weightTotal ? weighted / weightTotal : maxConfidence;
      const eligiblePairs = pairs.filter(pair => pair.eligible);
      const agreement = pairs.length ? eligiblePairs.length / pairs.length : 0;
      const blendedConfidence = Math.min(maxConfidence, meanConfidence * 0.94 + agreement * 100 * 0.06);

      const result = {
        branchCount: rows.length,
        confidenceWindowPct: this.options.confidenceWindowPct,
        similarityThreshold: this.options.similarityThreshold,
        pairs,
        structures: rows.map(row => ({ index: row.index, clauseCount: row.structure.clauseCount, relations: row.structure.relations, subjects: row.structure.subjects, stages: row.structure.stages, polarity: row.structure.polarity })),
        participantIndexes: participants.map(row => row.index),
        participantCount: participants.length,
        agreementRatio: Number(agreement.toFixed(6)),
        maxConfidence: Number(maxConfidence.toFixed(3)),
        meanConfidence: Number(meanConfidence.toFixed(3)),
        blendedConfidence: Number(blendedConfidence.toFixed(3)),
        canMix: participants.length >= this.options.minParticipants,
        amplitudeInterference: amplitude ? {
          enabled: true, rounds: amplitude.rounds, markedCount: amplitude.markedCount,
          oracleThreshold: Number((amplitude.oracleThreshold || 0).toFixed(6)),
          probabilities: participants.map(row => ({ branch: row.index, probability: Number((amplitudeWeights.get(row.index) || 0).toFixed(6)) })),
          featureMatrixShape: [participants.length, 8]
        } : { enabled: false },
        policy: 'confidence-within-window-OR-structure-aware-sentence-similarity',
        structureAware: true
      };
      this.analyses++;
      this.last = result;
      return result;
    }

    pair(analysis, a, b) {
      const left = Math.min(Number(a) || 0, Number(b) || 0), right = Math.max(Number(a) || 0, Number(b) || 0);
      return analysis?.pairs?.find(pair => pair.a === left && pair.b === right) || null;
    }

    compositionDecision(leftCandidate, rightCandidate, relation = '') {
      const left = this._structureOf(leftCandidate, 0), right = this._structureOf(rightCandidate, 1);
      const similarity = this._structureSimilarity(left, right);
      const key = normalize(relation || right.relations?.[0] || 'independent').replace(/\s+/g, '_');
      const spec = this._rules()?.Relations?.[key] || this._rules()?.Relations?.independent || { action: 'separate', merge_threshold: 1 };
      const maxClauses = Number(this._rules()?.Merge_Constraints?.max_clauses_per_sentence) || 3;
      const clauseSafe = left.clauseCount + right.clauseCount <= maxClauses;
      const polaritySafe = left.polarity === right.polarity || !this._rules()?.Merge_Constraints?.do_not_merge_conflicting_polarity;
      const allowed = clauseSafe && polaritySafe && similarity.score >= Number(spec.merge_threshold ?? 0.5);
      const grammarConnector = this.processor?.grammarEngine?.connectorForRelation?.(key) || '';
      return { relation: key, action: allowed ? spec.action : 'separate', connector: allowed ? (spec.connector || grammarConnector) : '', allowed, structuralSimilarity: similarity.score, clauseSafe, polaritySafe, grammarConnector };
    }

    status() {
      return {
        ready: true,
        role: 'four-full-sentence-structure-aware-interference-policy',
        analyses: this.analyses,
        structuralComparisons: this.structuralComparisons,
        confidenceWindowPct: this.options.confidenceWindowPct,
        similarityThreshold: this.options.similarityThreshold,
        policy: 'mix when confidence is close OR complete sentence/proposition structure is alike',
        amplitudePolicy: 'classical Grover-inspired weighting after eligibility; never replaces the OR rule',
        ruleDriven: Boolean(this._rules()?.Schema_Version),
        last: this.last
      };
    }
  }


  const V34_COMPOSITION_PLANS=Object.freeze([
    { relation:'same_claim', strategy:'merge', threshold:0.75 },
    { relation:'elaboration', strategy:'subordinate_or_follow', threshold:0.64 },
    { relation:'cause', strategy:'causal_link', threshold:0.70 },
    { relation:'result', strategy:'causal_link', threshold:0.70 },
    { relation:'contrast', strategy:'coordinate_or_split', threshold:0.66 },
    { relation:'concession', strategy:'subordinate', threshold:0.68 },
    { relation:'condition', strategy:'subordinate', threshold:0.72 },
    { relation:'sequence', strategy:'ordered_merge', threshold:0.64 },
    { relation:'temporal_before', strategy:'ordered_merge', threshold:0.70 },
    { relation:'temporal_after', strategy:'ordered_merge', threshold:0.70 },
    { relation:'comparison', strategy:'parallel', threshold:0.62 },
    { relation:'alternative', strategy:'coordinate', threshold:0.62 },
    { relation:'independent', strategy:'split', threshold:1.00 },
    { relation:'mechanism_stage', strategy:'ordered_merge', threshold:0.64 },
    { relation:'feedback', strategy:'causal_link', threshold:0.68 },
    { relation:'example', strategy:'follow', threshold:0.58 },
    { relation:'clarification', strategy:'follow', threshold:0.58 },
    { relation:'evidence', strategy:'follow', threshold:0.66 },
    { relation:'exception', strategy:'subordinate', threshold:0.72 },
    { relation:'purpose', strategy:'subordinate', threshold:0.66 },
  ]);
  const v34MNorm=v=>String(v??'').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const v34MArr=v=>Array.isArray(v)?v:[];
  class V34SentencePlanner {
    constructor(owner){this.owner=owner;this.last=null;this.history=[];}
    planSpec(relation){const key=v34MNorm(relation).replace(/\s+/g,'_');return V34_COMPOSITION_PLANS.find(x=>x.relation===key)||{relation:key,strategy:'split',threshold:0.72};}
    propositionKeys(candidate={}){
      const rows=v34MArr(candidate.propositions||candidate.analysis?.propositionGraph?.propositions||candidate.propositionGraph?.propositions);return new Set(rows.map(p=>`${v34MNorm(p.subject)}|${v34MNorm(p.relation)}|${v34MNorm(p.object)}`).filter(k=>k!=='||'));
    }
    mechanismKeys(candidate={}){return new Set(v34MArr(candidate.mechanismStages||candidate.mechanismGraph?.stages).map((s,i)=>v34MNorm(s.id||s.label||s.operation||`stage_${i}`)).filter(Boolean));}
    referenceSet(candidate={}){return new Set(v34MArr(candidate.references||candidate.referenceLinks).map(r=>v34MNorm(r.antecedent||r.target||r.surface)).filter(Boolean));}
    polarity(candidate={}){const rows=v34MArr(candidate.propositions||candidate.propositionGraph?.propositions);if(!rows.length)return'unknown';const neg=rows.filter(p=>p.polarity===false||p.negated).length;return neg>rows.length/2?'negative':'positive';}
    overlap(a,b){let common=0;for(const x of a)if(b.has(x))common++;return common/Math.max(1,a.size,b.size);}
    structuralScore(left,right){
      const prop=this.overlap(this.propositionKeys(left),this.propositionKeys(right));const mech=this.overlap(this.mechanismKeys(left),this.mechanismKeys(right));const refs=this.overlap(this.referenceSet(left),this.referenceSet(right));
      const subjectA=new Set(v34MArr(left.propositions||left.propositionGraph?.propositions).map(p=>v34MNorm(p.subject)).filter(Boolean));const subjectB=new Set(v34MArr(right.propositions||right.propositionGraph?.propositions).map(p=>v34MNorm(p.subject)).filter(Boolean));const subjects=this.overlap(subjectA,subjectB);
      const polarityConflict=this.polarity(left)!=='unknown'&&this.polarity(right)!=='unknown'&&this.polarity(left)!==this.polarity(right);
      return{score:Math.min(1,prop*0.38+mech*0.22+refs*0.14+subjects*0.26),propositionOverlap:prop,mechanismOverlap:mech,referenceOverlap:refs,subjectOverlap:subjects,polarityConflict};
    }
    inferRelation(left,right,meta={}){
      if(meta.relation)return v34MNorm(meta.relation).replace(/\s+/g,'_');
      const lr=v34MArr(left.relations||left.propositionRelations), rr=v34MArr(right.relations||right.propositionRelations);const all=[...lr,...rr].map(x=>v34MNorm(x.relation||x.type||x));
      for(const key of ['condition','cause','result','contrast','concession','temporal_before','temporal_after','comparison','sequence','feedback'])if(all.includes(key))return key;
      const structure=this.structuralScore(left,right);if(structure.propositionOverlap>0.72)return'same_claim';if(structure.subjectOverlap>0.6)return'elaboration';return'independent';
    }
    complexity(candidate={}){const text=String(candidate.text||candidate.output||candidate.sentence||'');const tokens=text.split(/\s+/).filter(Boolean).length;const clauses=(text.match(/\b(?:because|if|unless|although|while|after|before|whereas|but|and|or)\b/gi)||[]).length+1;const props=this.propositionKeys(candidate).size;return{tokens,clauses,props,score:Math.min(1,tokens/55+clauses/8+props/10)};}
    decision(left,right,meta={}){
      const relation=this.inferRelation(left,right,meta);const spec=this.planSpec(relation);const structural=this.structuralScore(left,right);const lc=this.complexity(left),rc=this.complexity(right);const totalTokens=lc.tokens+rc.tokens;let action=spec.strategy;
      const unresolved=v34MArr(left.references).filter(r=>r.resolved===false).length+v34MArr(right.references).filter(r=>r.resolved===false).length;
      if(structural.polarityConflict&&!['contrast','concession'].includes(relation))action='split';if(totalTokens>52)action='split';if(unresolved>0&&structural.subjectOverlap<0.5)action='split';if(relation==='independent')action='split';
      return{relation,spec,structural,leftComplexity:lc,rightComplexity:rc,totalTokens,unresolvedReferences:unresolved,action};
    }
    order(candidates=[]){
      const priority={definition:10,input:20,precondition:25,operation:30,transition:40,output:50,feedback:60,limitation:70,example:80};
      return candidates.map((c,i)=>({c,i,p:Number(c.order??priority[v34MNorm(c.role||c.stage||c.type)]??(100+i))})).sort((a,b)=>a.p-b.p||a.i-b.i).map(x=>x.c);
    }
    coverage(candidates=[],required=[]){
      const all=new Set();for(const c of candidates)for(const k of this.propositionKeys(c))all.add(k);const req=new Set(required.map(r=>typeof r==='string'?v34MNorm(r):`${v34MNorm(r.subject)}|${v34MNorm(r.relation)}|${v34MNorm(r.object)}`));let hit=0;for(const k of req)if(all.has(k)||Array.from(all).some(x=>x.includes(k)))hit++;return req.size?hit/req.size:1;
    }
    plan(candidates=[],meta={}){
      const ordered=this.order(candidates);const steps=[];for(let i=0;i<ordered.length-1;i++)steps.push(this.decision(ordered[i],ordered[i+1],meta));const groups=[];let current=[];
      ordered.forEach((candidate,i)=>{current.push(candidate);const decision=steps[i];if(!decision||decision.action==='split'){groups.push(current);current=[];}});if(current.length)groups.push(current);
      const result={ordered,steps,groups,coverage:this.coverage(ordered,meta.requiredPropositions||[])};this.last=result;this.history.push({at:Date.now(),candidates:ordered.length,groups:groups.length,coverage:result.coverage});if(this.history.length>96)this.history.shift();return result;
    }
    composeText(group=[],bridge=null){
      if(!group.length)return'';let text=String(group[0].text||group[0].output||group[0].sentence||'');for(let i=1;i<group.length;i++){const right=String(group[i].text||group[i].output||group[i].sentence||'');const relation=this.inferRelation(group[i-1],group[i],{});if(bridge?.composeIdeas)text=bridge.composeIdeas(text,right,relation,{});else text=`${text.replace(/[.!?]+$/,'')}, ${right.replace(/^[A-Z]/,c=>c.toLowerCase())}`;}return text;}
  }
  VilotSentenceMixer.prototype.planComposition=function(candidates=[],meta={}){this.v34Planner||=new V34SentencePlanner(this);return this.v34Planner.plan(candidates,meta);};
  VilotSentenceMixer.prototype.composeCandidateSet=function(candidates=[],meta={}){const plan=this.planComposition(candidates,meta);let text=plan.groups.map(g=>this.v34Planner.composeText(g,meta.bridge)).filter(Boolean).join(' ');const grammar=this.processor?.grammarEngine?.audit?.(text,meta.analysis||null)||null;if(grammar&&grammar.score<0.58&&plan.groups.length>1){text=plan.groups.map(g=>g.map(c=>String(c.text||c.output||c.sentence||'').trim()).filter(Boolean).join(' ')).filter(Boolean).join(' ');}return{...plan,text,grammarAudit:grammar};};
  const v34OriginalMixerAnalyze=VilotSentenceMixer.prototype.analyze;
  VilotSentenceMixer.prototype.analyze=function(candidates=[]){const base=v34OriginalMixerAnalyze.call(this,candidates)||{};this.v34Planner||=new V34SentencePlanner(this);const structured=this.v34Planner.plan(candidates,{});return{...base,structuredPlan:structured,system34:true};};



  const V34_MIX_SCENARIOS = Object.freeze([
    {
      id: "merge-scenario-1",
      family: "merge",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "split-scenario-2",
      family: "split",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "subordinate-scenario-3",
      family: "subordinate",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "coordinate-scenario-4",
      family: "coordinate",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "sequence-scenario-5",
      family: "sequence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "contrast-scenario-6",
      family: "contrast",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "mechanism-scenario-7",
      family: "mechanism",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "comparison-scenario-8",
      family: "comparison",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "reference-risk-scenario-9",
      family: "reference-risk",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "coverage-scenario-10",
      family: "coverage",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "merge-scenario-11",
      family: "merge",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "split-scenario-12",
      family: "split",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "subordinate-scenario-13",
      family: "subordinate",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "coordinate-scenario-14",
      family: "coordinate",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "sequence-scenario-15",
      family: "sequence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "contrast-scenario-16",
      family: "contrast",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "mechanism-scenario-17",
      family: "mechanism",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "comparison-scenario-18",
      family: "comparison",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "reference-risk-scenario-19",
      family: "reference-risk",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "coverage-scenario-20",
      family: "coverage",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "merge-scenario-21",
      family: "merge",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "split-scenario-22",
      family: "split",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "subordinate-scenario-23",
      family: "subordinate",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "coordinate-scenario-24",
      family: "coordinate",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "sequence-scenario-25",
      family: "sequence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "contrast-scenario-26",
      family: "contrast",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "mechanism-scenario-27",
      family: "mechanism",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "comparison-scenario-28",
      family: "comparison",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "reference-risk-scenario-29",
      family: "reference-risk",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "coverage-scenario-30",
      family: "coverage",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "merge-scenario-31",
      family: "merge",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "split-scenario-32",
      family: "split",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "subordinate-scenario-33",
      family: "subordinate",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "coordinate-scenario-34",
      family: "coordinate",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "sequence-scenario-35",
      family: "sequence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "contrast-scenario-36",
      family: "contrast",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "mechanism-scenario-37",
      family: "mechanism",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "comparison-scenario-38",
      family: "comparison",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "reference-risk-scenario-39",
      family: "reference-risk",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "coverage-scenario-40",
      family: "coverage",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "merge-scenario-41",
      family: "merge",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "split-scenario-42",
      family: "split",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "subordinate-scenario-43",
      family: "subordinate",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "coordinate-scenario-44",
      family: "coordinate",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "sequence-scenario-45",
      family: "sequence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "contrast-scenario-46",
      family: "contrast",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "mechanism-scenario-47",
      family: "mechanism",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "comparison-scenario-48",
      family: "comparison",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "reference-risk-scenario-49",
      family: "reference-risk",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "coverage-scenario-50",
      family: "coverage",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "merge-scenario-51",
      family: "merge",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "split-scenario-52",
      family: "split",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "subordinate-scenario-53",
      family: "subordinate",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "coordinate-scenario-54",
      family: "coordinate",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "sequence-scenario-55",
      family: "sequence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "contrast-scenario-56",
      family: "contrast",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "mechanism-scenario-57",
      family: "mechanism",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "comparison-scenario-58",
      family: "comparison",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "reference-risk-scenario-59",
      family: "reference-risk",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "coverage-scenario-60",
      family: "coverage",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "merge-scenario-61",
      family: "merge",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "split-scenario-62",
      family: "split",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "subordinate-scenario-63",
      family: "subordinate",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "coordinate-scenario-64",
      family: "coordinate",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "sequence-scenario-65",
      family: "sequence",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "contrast-scenario-66",
      family: "contrast",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "mechanism-scenario-67",
      family: "mechanism",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "comparison-scenario-68",
      family: "comparison",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "reference-risk-scenario-69",
      family: "reference-risk",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
  ]);
  VilotSentenceMixer.prototype.compositionScenarios = function() { return V34_MIX_SCENARIOS.map(row => ({ ...row })); };

  globalThis.VilotSentenceMixer = VilotSentenceMixer;
})();
