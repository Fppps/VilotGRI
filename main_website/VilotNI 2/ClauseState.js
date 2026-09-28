/**
 * VilotNI 2 - ClauseState.js
 *
 * General clause-role parser for declaratives, interrogatives, commands,
 * fragments, and multi-clause text. It uses contextual POS roles and support
 * words but contains no subject-matter facts.
 */
(() => {
  'use strict';

  const NOMINAL = new Set(['Det','Adj','Noun','ProperNoun','Pronoun','Num']);
  const SUPPORT = new Set(['am','is','are','was','were','be','been','being','do','does','did','have','has','had','can','could','will','would','shall','should','may','might','must']);
  const COPULA = new Set(['am','is','are','was','were','be','been','being']);

  class VilotClauseState {
    constructor(processor, rsl, options = {}) {
      this.processor = processor;
      this.rsl = rsl;
      this.options = options;
      this.patternCounts = new Map();
      this.observations = 0;
      this.last = null;
    }

    resolveTokens(tokens) {
      const rows = (tokens || []).filter(t => t?.pos !== 'Punct');
      const state = { previousWord:'', previousPOS:'START2', lastWord:'', lastPOS:'START' };
      return rows.map(token => {
        const lexicalPOS = token.pos;
        const lexicalPossible = Array.from(new Set(token.possiblePOS || [token.pos]));
        const resolution = this.rsl?.resolvePOS?.(token.entry, state, { fallbackPOS: token.pos }) || { pos: token.pos, certainty: 0.5 };
        const entry = this.processor?.materializePOS?.(token.entry, resolution.pos) || token.entry;
        const out = {
          ...token,
          lexicalPOS,
          lexicalPossiblePOS: lexicalPossible,
          pos: resolution.pos || token.pos,
          posCertainty: resolution.certainty ?? 0.5,
          entry,
          grammarInferredPOS: false
        };
        state.previousWord = state.lastWord; state.previousPOS = state.lastPOS;
        state.lastWord = token.lower; state.lastPOS = out.pos;
        return out;
      });
    }

    _canBe(token, pos) {
      if (!token) return false;
      if (token.pos === pos || token.lexicalPOS === pos) return true;
      if ((token.possiblePOS || []).includes(pos) || (token.lexicalPossiblePOS || []).includes(pos)) return true;
      return false;
    }

    _setPOS(token, pos, reason = 'grammar-context') {
      if (!token) return token;
      token.pos = pos;
      token.grammarInferredPOS = true;
      token.grammarInferenceReason = reason;
      token.entry = this.processor?.materializePOS?.(token.entry, pos) || { ...(token.entry || {}), pos };
      return token;
    }

    _firstVerb(words, start = 0, allowSupport = false) {
      for (let i = Math.max(0,start); i < words.length; i++) {
        const w = words[i];
        if (this._canBe(w, 'Verb')) return i;
        if (allowSupport && (w.pos === 'Aux' || w.pos === 'Modal')) return i;
      }
      return -1;
    }


    _looksNominalPhrase(words, start) {
      if (start < 0 || start >= words.length) return false;
      let sawNominal = false;
      for (let i = start; i < words.length; i++) {
        const token = words[i];
        if (!token) return false;
        if (token.pos === 'Det' || token.pos === 'Adj' || token.pos === 'Num') continue;
        if (token.pos === 'Noun' || token.pos === 'ProperNoun' || token.pos === 'Pronoun') { sawNominal = true; continue; }
        // A noun can be lexically mislabeled as a verb in the source vocabulary.
        // After a determiner, nominalize only when Noun is a possible lexical role.
        const prev = words[i - 1];
        if (prev?.pos === 'Det' && this._canBe(token, 'Noun')) {
          this._setPOS(token, 'Noun', 'copular-determiner-nominal');
          sawNominal = true;
          continue;
        }
        return false;
      }
      return sawNominal;
    }

    _subjectAndPredicateAfterSupport(words, supportIndex) {
      const start = supportIndex + 1;
      if (start >= words.length) return { subjectStart:start, subjectEnd:start-1, predicateIndex:-1 };

      // First prefer a lexically plausible verb. A determiner begins a noun
      // phrase even if the raw vocabulary mislabeled its noun as a verb.
      let explicitVerb = -1;
      for (let i = start; i < words.length; i++) {
        const prev = words[i-1];
        if (prev?.pos === 'Det' && i === start + 1) continue;
        if (this._canBe(words[i], 'Verb')) { explicitVerb = i; break; }
      }
      if (explicitVerb > start) {
        for (let i = start; i < explicitVerb; i++) {
          if (i > start && words[i-1]?.pos === 'Det' && words[i].pos === 'Verb') this._setPOS(words[i], 'Noun', 'determiner-nominal');
        }
        if (words[explicitVerb].pos !== 'Verb') this._setPOS(words[explicitVerb], 'Verb', 'support-selects-verb-role');
        return { subjectStart:start, subjectEnd:explicitVerb-1, predicateIndex:explicitVerb };
      }

      // If the dictionary does not know the verb role yet, infer it from the
      // question grammar. This is exactly the kind of occurrence-level POS
      // evidence Correlation can later learn instead of requiring a hand edit.
      if (words[start]?.pos === 'Det' && start + 2 < words.length) {
        if (words[start+1].pos === 'Verb') this._setPOS(words[start+1], 'Noun', 'determiner-nominal');
        this._setPOS(words[start+2], 'Verb', 'support-requires-main-verb');
        return { subjectStart:start, subjectEnd:start+1, predicateIndex:start+2 };
      }
      if (start + 1 < words.length) {
        this._setPOS(words[start+1], 'Verb', 'support-requires-main-verb');
        return { subjectStart:start, subjectEnd:start, predicateIndex:start+1 };
      }
      return { subjectStart:start, subjectEnd:words.length-1, predicateIndex:-1 };
    }

    analyze(tokens, question = {}, baseAnalysis = {}) {
      const words = this.resolveTokens(tokens);
      const compoundConsumesSecond = Boolean(question.compound && (question.compound.startsWith('how_') || question.compound === 'what_kind' || question.compound === 'what_type'));
      const whOffset = question.wh ? (compoundConsumesSecond ? 2 : 1) : 0;
      const supportIndex = question.supportIndex >= 0 ? question.supportIndex : -1;
      let predicateIndex = -1;
      let subjectStart = 0;
      let subjectEnd = -1;
      let objectStart = -1;

      if (question.isQuestion && supportIndex >= 0) {
        const support = words[supportIndex];
        if (COPULA.has(support.lower)) {
          // Definition/copular questions such as "What is a file?" contain a
          // nominal phrase after the copula, not a hidden main verb. Recognize
          // that phrase before considering alternative verb parses.
          if (this._looksNominalPhrase(words, supportIndex + 1)) {
            predicateIndex = supportIndex;
            subjectStart = supportIndex + 1;
            subjectEnd = words.length - 1;
            objectStart = -1;
          } else {
            const lexicalVerb = this._firstVerb(words, supportIndex + 1, false);
            if (lexicalVerb >= 0 && lexicalVerb > supportIndex + 1) {
              const split = this._subjectAndPredicateAfterSupport(words, supportIndex);
              subjectStart=split.subjectStart; subjectEnd=split.subjectEnd; predicateIndex=split.predicateIndex;
              objectStart=predicateIndex>=0?predicateIndex+1:-1;
            } else {
              predicateIndex = supportIndex;
              subjectStart = supportIndex + 1;
              subjectEnd = words.length - 1;
              objectStart = -1;
            }
          }
        } else {
          const split = this._subjectAndPredicateAfterSupport(words, supportIndex);
          subjectStart=split.subjectStart; subjectEnd=split.subjectEnd; predicateIndex=split.predicateIndex;
          objectStart=predicateIndex>=0?predicateIndex+1:-1;
        }
      } else if (question.isQuestion && question.wh) {
        predicateIndex = this._firstVerb(words, whOffset, true);
        if (predicateIndex < 0 && whOffset + 1 < words.length) {
          // Subject WH questions often have no auxiliary inversion:
          // "Which processor handles graphics?"
          subjectStart = whOffset;
          subjectEnd = whOffset;
          predicateIndex = whOffset + 1;
          this._setPOS(words[predicateIndex], 'Verb', 'wh-subject-main-verb');
          objectStart = predicateIndex + 1;
        } else if (predicateIndex >= 0) {
          if (question.family === 'person' && predicateIndex === whOffset) {
            subjectStart = 0; subjectEnd = whOffset - 1;
          } else {
            subjectStart = whOffset;
            subjectEnd = predicateIndex - 1;
          }
          objectStart = predicateIndex + 1;
        }
      } else {
        predicateIndex = this._firstVerb(words, 0, true);
        if (predicateIndex >= 0) {
          subjectStart = 0;
          subjectEnd = predicateIndex - 1;
          objectStart = predicateIndex + 1;
        }
      }

      let subjectTokens = subjectEnd >= subjectStart ? words.slice(subjectStart, subjectEnd + 1) : [];
      if (question.isQuestion && supportIndex >= 0 && predicateIndex === supportIndex && COPULA.has(words[supportIndex]?.lower)) {
        subjectTokens = words.slice(supportIndex + 1);
      }
      const predicateToken = predicateIndex >= 0 ? words[predicateIndex] : null;
      const objectTokens = objectStart >= 0 ? words.slice(objectStart) : [];

      // Normalize nominal roles inside the subject phrase after determiners.
      for (let i = 0; i < subjectTokens.length; i++) {
        if (i > 0 && subjectTokens[i-1]?.pos === 'Det' && subjectTokens[i]?.pos === 'Verb') this._setPOS(subjectTokens[i], 'Noun', 'determiner-nominal');
      }

      let clauseType = 'fragment';
      if (question.isQuestion) clauseType = 'interrogative';
      else if (words[0]?.pos === 'Verb') clauseType = 'imperative';
      else if (subjectTokens.length && predicateToken) clauseType = 'declarative';

      const posSignature = words.map(t => t.pos).join('>');
      const result = {
        words,
        lowers: words.map(t => t.lower),
        clauseType,
        subjectTokens,
        predicateToken,
        predicateIndex,
        objectTokens,
        supportIndex,
        supportToken: supportIndex >= 0 ? words[supportIndex] : null,
        posSignature,
        hasSubject: subjectTokens.some(t => NOMINAL.has(t.pos)),
        hasPredicate: Boolean(predicateToken),
        complete: Boolean((subjectTokens.length || clauseType === 'imperative') && predicateToken),
        inferredPOS: words.filter(t => t.grammarInferredPOS).map(t => ({ word:t.lower, pos:t.pos, reason:t.grammarInferenceReason }))
      };
      this.last = result;
      return result;
    }

    observe(state, gain = 1) {
      if (!state?.posSignature) return;
      const key = `${state.clauseType}|${state.posSignature}`;
      this.patternCounts.set(key, (this.patternCounts.get(key) || 0) + Math.max(0.05, Number(gain) || 1));
      this.observations++;
    }

    support(state) {
      if (!state) return 0.5;
      const structural = (state.complete ? 0.72 : 0.34) + (state.hasSubject ? 0.12 : 0) + (state.hasPredicate ? 0.12 : 0);
      if (!this.observations || !state.posSignature) return Math.min(1, structural);
      const key = `${state.clauseType}|${state.posSignature}`;
      const learned = Math.min(1, Math.log1p(this.patternCounts.get(key) || 0) / 5);
      return Math.min(1, structural * 0.78 + learned * 0.22);
    }

    exportState() { return { observations: this.observations, patternCounts: Array.from(this.patternCounts.entries()) }; }
    importState(state) { if (!state) return false; this.observations=Number(state.observations)||0; this.patternCounts=new Map(state.patternCounts||[]); return true; }
    reset() { this.observations=0; this.patternCounts.clear(); this.last=null; }
    status() { return { role:'general-clause-role-state', observations:this.observations, learnedClausePatterns:this.patternCounts.size, last:this.last ? {clauseType:this.last.clauseType,posSignature:this.last.posSignature,complete:this.last.complete}:null }; }
  }

  globalThis.VilotClauseState = VilotClauseState;
})();

/**
 * ClauseState production research extension.
 *
 * Executable, data-driven numerical/statistical/graph utilities used by
 * diagnostics, background learning, calibration, matrix experiments and
 * future compute paths. No prompt-specific phrases or benchmark answers live
 * here. Methods stay out of the foreground hot path unless explicitly called.
 */
(() => {
  'use strict';

  class VilotClauseStateProductionLab {
    constructor(options = {}) {
      this.options = { ...options };
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
      for (let i = 0; i < n; i++) total += (Number(a[i]) || 0) * (Number(b[i]) || 0);
      return total;
    }

    cosine(a, b, epsilon = 1e-9) {
      const n = Math.min(a?.length || 0, b?.length || 0);
      let dot = 0, aa = 0, bb = 0;
      for (let i = 0; i < n; i++) {
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
      for (let i = 0; i < n; i++) total += Math.abs((Number(a?.[i]) || 0) - (Number(b?.[i]) || 0));
      return total;
    }

    l2Distance(a, b) {
      const n = Math.max(a?.length || 0, b?.length || 0);
      let total = 0;
      for (let i = 0; i < n; i++) {
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
        heap.push({ item, value });
      }
      heap.sort((a,b) => b.value - a.value);
      return heap.slice(0, limit).map(entry => entry.item);
    }

    argmax(values) {
      let index = -1;
      let best = -Infinity;
      for (let i = 0; i < (values?.length || 0); i++) {
        const value = Number(values[i]);
        if (Number.isFinite(value) && value > best) {
          best = value;
          index = i;
        }
      }
      return { index, value: best };
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
      for (let i = 0; i < input.length; i++) {
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
      for (let i = 0; i < input.length; i++) {
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
      for (let i=0;i<n;i++) {
        const x=Number(a[i]), y=Number(b[i]);
        if (!Number.isFinite(x)||!Number.isFinite(y)) continue;
        sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; count++;
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
      for (let i=0;i<n;i++) {
        if (!Number.isFinite(ax[i]) || !Number.isFinite(bx[i])) continue;
        total += (ax[i]-ma)*(bx[i]-mb);
        count++;
      }
      return count ? total/count : 0;
    }

    weightedMean(values, weights) {
      const n = Math.min(values?.length || 0, weights?.length || 0);
      let sum = 0, weight = 0;
      for (let i=0;i<n;i++) {
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
      for (let i=0;i<n;i++) {
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
      for (let i=0;i<text.length;i++) {
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
      for (let n=lo;n<=hi;n++) {
        for (let i=0;i+n<=input.length;i++) out.push(input.slice(i,i+n));
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
      for (let i=1;i<input.length;i++) {
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
        if (Math.abs(next)<floor) map.delete(key); else map.set(key,next);
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
      for (let i=0;i<n;i++) {
        const idx=indices[i]|0;
        if (idx<0||idx>=target.length) continue;
        target[idx]+=(Number(values[i])||0)*g;
      }
      return target;
    }

    sparseDot(dense, indices, values) {
      const n=Math.min(indices?.length||0,values?.length||0);
      let total=0;
      for (let i=0;i<n;i++) {
        const idx=indices[i]|0;
        if (idx<0||idx>=dense.length) continue;
        total+=(Number(dense[idx])||0)*(Number(values[i])||0);
      }
      return total;
    }

    matVec(matrix, rows, cols, vector) {
      const r=Math.max(0,rows|0), c=Math.max(0,cols|0);
      const out=new Float32Array(r);
      for (let i=0;i<r;i++) {
        let sum=0;
        const base=i*c;
        for (let j=0;j<c;j++) sum+=(Number(matrix[base+j])||0)*(Number(vector[j])||0);
        out[i]=sum;
      }
      return out;
    }

    outerUpdate(matrix, rows, cols, a, b, rate = 0.01) {
      const r=Math.min(rows|0,a?.length||0), c=Math.min(cols|0,b?.length||0);
      const eta=Number(rate)||0;
      for (let i=0;i<r;i++) {
        const av=Number(a[i])||0;
        const base=i*cols;
        for (let j=0;j<c;j++) matrix[base+j]+=eta*av*(Number(b[j])||0);
      }
      return matrix;
    }

    rowNormalize(matrix, rows, cols, epsilon = 1e-9) {
      for (let i=0;i<(rows|0);i++) {
        const base=i*cols;
        let norm2=0;
        for (let j=0;j<(cols|0);j++) { const v=Number(matrix[base+j])||0; norm2+=v*v; }
        const inv=1/(Math.sqrt(norm2)+epsilon);
        for (let j=0;j<(cols|0);j++) matrix[base+j]=(Number(matrix[base+j])||0)*inv;
      }
      return matrix;
    }

    vectorAdd(a, b, scale = 1) {
      const n=Math.max(a?.length||0,b?.length||0);
      const out=new Float32Array(n);
      const s=Number(scale)||0;
      for (let i=0;i<n;i++) out[i]=(Number(a?.[i])||0)+(Number(b?.[i])||0)*s;
      return out;
    }

    vectorLerp(a, b, t = 0.5) {
      const n=Math.max(a?.length||0,b?.length||0);
      const out=new Float32Array(n);
      const k=Math.max(0,Math.min(1,Number(t)||0));
      for (let i=0;i<n;i++) out[i]=(Number(a?.[i])||0)*(1-k)+(Number(b?.[i])||0)*k;
      return out;
    }

    orthogonalize(vector, basis) {
      const out=Float32Array.from(vector||[]);
      for (const b of basis||[]) {
        const denom=this.dot(b,b)||1;
        const scale=this.dot(out,b)/denom;
        for (let i=0;i<Math.min(out.length,b.length);i++) out[i]-=scale*(Number(b[i])||0);
      }
      return Float32Array.from(this.normalize(out));
    }

    powerIteration(matrix, n, iterations = 16) {
      const size=Math.max(1,n|0);
      let v=new Float32Array(size).fill(1/Math.sqrt(size));
      for (let step=0;step<Math.max(1,iterations|0);step++) {
        const next=this.matVec(matrix,size,size,v);
        v=Float32Array.from(this.normalize(next));
      }
      return v;
    }

    calibrationBins(pairs, bins = 10) {
      const count=Math.max(2,bins|0);
      const out=Array.from({length:count},(_,i)=>({lo:i/count,hi:(i+1)/count,n:0,confidence:0,accuracy:0}));
      for (const pair of pairs||[]) {
        const c=Math.max(0,Math.min(1,Number(pair.confidence)||0));
        const y=Number(pair.correct)||0;
        const idx=Math.min(count-1,Math.floor(c*count));
        const bin=out[idx];
        bin.n++; bin.confidence+=c; bin.accuracy+=y;
      }
      for (const bin of out) if (bin.n) { bin.confidence/=bin.n; bin.accuracy/=bin.n; }
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
        total+=d*d; count++;
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
      for (let i=0;i<input.length;i++) for (let j=i+1;j<input.length;j++) {
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

    agreement(scores) {
      const input=Array.from(scores||[],Number).filter(Number.isFinite);
      if (!input.length) return 0;
      const sd=this.stddev(input);
      return 1/(1+sd);
    }

    adaptiveRate(confidence, visits, base = 0.05) {
      const c=Math.max(0,Math.min(1,Number(confidence)||0));
      const v=Math.max(0,Number(visits)||0);
      return Math.max(1e-5,(Number(base)||0.05)*(0.25+0.75*(1-c))/Math.sqrt(1+v*0.05));
    }

    boundedUpdate(oldValue, observation, rate = 0.05, lo = -1, hi = 1) {
      const old=Number(oldValue)||0;
      const obs=Number(observation)||0;
      const eta=Math.max(0,Math.min(1,Number(rate)||0));
      return Math.max(lo,Math.min(hi,old+(obs-old)*eta));
    }

    temporalWeight(ageMs, halfLifeMs = 60000) {
      const age=Math.max(0,Number(ageMs)||0);
      const half=Math.max(1,Number(halfLifeMs)||1);
      return Math.pow(0.5,age/half);
    }

    reservoirPush(array, item, seen, capacity, random = Math.random) {
      const cap=Math.max(1,capacity|0);
      if (array.length<cap) { array.push(item); return true; }
      const j=Math.floor((random?.()??Math.random())*Math.max(1,seen));
      if (j<cap) { array[j]=item; return true; }
      return false;
    }

    chunk(items, size = 64) {
      const input=Array.from(items||[]), n=Math.max(1,size|0), out=[];
      for (let i=0;i<input.length;i+=n) out.push(input.slice(i,i+n));
      return out;
    }

    stableMerge(left, right, key = x => String(x)) {
      const out=[], seen=new Set();
      for (const item of [...(left||[]),...(right||[])]) {
        const k=key(item);
        if (seen.has(k)) continue;
        seen.add(k); out.push(item);
      }
      return out;
    }

    mapObject(object, fn) {
      const out={};
      for (const [key,value] of Object.entries(object||{})) out[key]=fn(value,key);
      return out;
    }

    serializeMap(map) {
      return Array.from(map instanceof Map ? map.entries() : []);
    }

    deserializeMap(entries) {
      return new Map(Array.isArray(entries)?entries:[]);
    }

    deepClone(value) {
      if (value == null || typeof value !== 'object') return value;
      if (ArrayBuffer.isView(value)) return value.slice ? value.slice() : new value.constructor(value);
      if (Array.isArray(value)) return value.map(item=>this.deepClone(item));
      if (value instanceof Map) return new Map(Array.from(value,([k,v])=>[k,this.deepClone(v)]));
      const out={};
      for (const [k,v] of Object.entries(value)) out[k]=this.deepClone(v);
      return out;
    }

    minMax(values) {
      let min=Infinity,max=-Infinity;
      for (const raw of values||[]) {
        const value=Number(raw);
        if (!Number.isFinite(value)) continue;
        if (value<min) min=value;
        if (value>max) max=value;
      }
      return {min:Number.isFinite(min)?min:0,max:Number.isFinite(max)?max:0};
    }

    histogram(values, bins = 16) {
      const input=Array.from(values||[],Number).filter(Number.isFinite);
      const count=Math.max(2,bins|0), out=new Uint32Array(count);
      if (!input.length) return {bins:out,min:0,max:0};
      const {min,max}=this.minMax(input), span=max-min||1;
      for (const value of input) out[Math.min(count-1,Math.floor(((value-min)/span)*count))]++;
      return {bins:out,min,max};
    }

    rank(values) {
      const pairs=Array.from(values||[],(value,index)=>({value:Number(value)||0,index})).sort((a,b)=>a.value-b.value);
      const out=new Float64Array(pairs.length);
      for (let i=0;i<pairs.length;i++) out[pairs[i].index]=i;
      return out;
    }

    spearman(a,b) {
      return this.pearson(this.rank(a),this.rank(b));
    }

    binarySearch(array, value, compare = (a,b) => a-b) {
      let lo=0,hi=array?.length||0;
      while (lo<hi) {
        const mid=(lo+hi)>>1;
        if (compare(array[mid],value)<0) lo=mid+1; else hi=mid;
      }
      return lo;
    }

    insertSorted(array, item, compare) {
      const idx=this.binarySearch(array,item,compare);
      array.splice(idx,0,item);
      return idx;
    }

    monotonicQueueMax(values, window = 8) {
      const input=Array.from(values||[],Number), w=Math.max(1,window|0), q=[], out=[];
      for (let i=0;i<input.length;i++) {
        while(q.length&&q[0]<=i-w) q.shift();
        while(q.length&&input[q[q.length-1]]<=input[i]) q.pop();
        q.push(i); out.push(input[q[0]]);
      }
      return out;
    }

    prefixSums(values) {
      const out=new Float64Array((values?.length||0)+1);
      for (let i=0;i<(values?.length||0);i++) out[i+1]=out[i]+(Number(values[i])||0);
      return out;
    }

    rangeSum(prefix, start, end) {
      const lo=Math.max(0,start|0), hi=Math.max(lo,Math.min((prefix?.length||1)-1,end|0));
      return (Number(prefix?.[hi])||0)-(Number(prefix?.[lo])||0);
    }

    runLengthEncode(items) {
      const input=Array.from(items||[]), out=[];
      for (const item of input) {
        const last=out[out.length-1];
        if (last&&Object.is(last.value,item)) last.count++;
        else out.push({value:item,count:1});
      }
      return out;
    }

    runLengthDecode(runs) {
      const out=[];
      for (const run of runs||[]) for(let i=0;i<Math.max(0,run.count|0);i++) out.push(run.value);
      return out;
    }

    pairwise(items) {
      const input=Array.from(items||[]), out=[];
      for(let i=1;i<input.length;i++) out.push([input[i-1],input[i]]);
      return out;
    }

    windowed(items, size = 3) {
      const input=Array.from(items||[]), n=Math.max(1,size|0), out=[];
      for(let i=0;i+n<=input.length;i++) out.push(input.slice(i,i+n));
      return out;
    }

    flatten(nested) {
      const out=[];
      for(const group of nested||[]) for(const item of group||[]) out.push(item);
      return out;
    }

    groupBy(items, keyFn) {
      const map=new Map();
      for(const item of items||[]) {
        const key=keyFn(item);
        let group=map.get(key);
        if(!group) map.set(key,group=[]);
        group.push(item);
      }
      return map;
    }

    countBy(items, keyFn = x => x) {
      const map=new Map();
      for(const item of items||[]) {
        const key=keyFn(item);
        map.set(key,(map.get(key)||0)+1);
      }
      return map;
    }

    weightedChoice(items, weights, random = Math.random) {
      const n=Math.min(items?.length||0,weights?.length||0);
      let total=0;
      for(let i=0;i<n;i++) total+=Math.max(0,Number(weights[i])||0);
      if(total<=0) return items?.[0];
      let r=(random?.()??Math.random())*total;
      for(let i=0;i<n;i++) { r-=Math.max(0,Number(weights[i])||0); if(r<=0) return items[i]; }
      return items[n-1];
    }

    gini(values) {
      const input=Array.from(values||[],Number).filter(v=>Number.isFinite(v)&&v>=0).sort((a,b)=>a-b);
      if(!input.length) return 0;
      let sum=0, weighted=0;
      for(let i=0;i<input.length;i++){ sum+=input[i]; weighted+=(i+1)*input[i]; }
      return sum? (2*weighted)/(input.length*sum)-(input.length+1)/input.length : 0;
    }

    klDivergence(p,q,epsilon=1e-12) {
      const n=Math.min(p?.length||0,q?.length||0);
      let total=0;
      for(let i=0;i<n;i++) {
        const a=Math.max(epsilon,Number(p[i])||0), b=Math.max(epsilon,Number(q[i])||0);
        total+=a*Math.log(a/b);
      }
      return total;
    }

    jsDivergence(p,q) {
      const n=Math.min(p?.length||0,q?.length||0), m=new Float64Array(n);
      for(let i=0;i<n;i++) m[i]=0.5*((Number(p[i])||0)+(Number(q[i])||0));
      return 0.5*this.klDivergence(p,m)+0.5*this.klDivergence(q,m);
    }

    logisticCalibrate(score, slope = 1, bias = 0) {
      return this.sigmoid((Number(score)||0)*(Number(slope)||1)+(Number(bias)||0));
    }

    margin(scores) {
      const input=Array.from(scores||[],Number).filter(Number.isFinite).sort((a,b)=>b-a);
      return input.length>1?input[0]-input[1]:(input[0]||0);
    }

    effectiveSampleSize(weights) {
      let s=0,s2=0;
      for(const raw of weights||[]){const w=Math.max(0,Number(raw)||0);s+=w;s2+=w*w;}
      return s2? (s*s)/s2 : 0;
    }

    winsorize(values, lowQ = 0.01, highQ = 0.99) {
      const input=Array.from(values||[],Number);
      const lo=this.quantile(input,lowQ), hi=this.quantile(input,highQ);
      return input.map(v=>Number.isFinite(v)?Math.max(lo,Math.min(hi,v)):0);
    }

    robustScale(values) {
      const input=Array.from(values||[],Number), med=this.median(input), mad=this.mad(input)||1;
      return input.map(v=>Number.isFinite(v)?(v-med)/(1.4826*mad):0);
    }

    stateChecksum(arrays) {
      let h=2166136261>>>0;
      for(const arr of arrays||[]) for(let i=0;i<(arr?.length||0);i++) {
        const x=Math.fround(Number(arr[i])||0);
        const view=new DataView(new ArrayBuffer(4));
        view.setFloat32(0,x,true);
        h^=view.getUint32(0,true); h=Math.imul(h,16777619)>>>0;
      }
      return h>>>0;
    }

    finiteRatio(values) {
      let finite=0,total=0;
      for(const v of values||[]){total++;if(Number.isFinite(Number(v)))finite++;}
      return total?finite/total:1;
    }

    clipNorm(vector, maxNorm = 1) {
      const input=Array.from(vector||[],Number), norm=Math.sqrt(this.dot(input,input));
      if(!Number.isFinite(norm)||norm<=maxNorm) return Float32Array.from(input.map(v=>Number.isFinite(v)?v:0));
      const scale=Math.max(0,Number(maxNorm)||0)/Math.max(norm,1e-12);
      return Float32Array.from(input.map(v=>(Number.isFinite(v)?v:0)*scale));
    }

    rollingMean2(values) {
      return this.rollingMean(values, 2);
    }

    rollingMean3(values) {
      return this.rollingMean(values, 3);
    }

    rollingMean4(values) {
      return this.rollingMean(values, 4);
    }

    rollingMean5(values) {
      return this.rollingMean(values, 5);
    }

    rollingMean6(values) {
      return this.rollingMean(values, 6);
    }

    rollingMean7(values) {
      return this.rollingMean(values, 7);
    }

    rollingMean8(values) {
      return this.rollingMean(values, 8);
    }

    rollingMean9(values) {
      return this.rollingMean(values, 9);
    }

    rollingMean10(values) {
      return this.rollingMean(values, 10);
    }

    rollingMean11(values) {
      return this.rollingMean(values, 11);
    }

    rollingMean12(values) {
      return this.rollingMean(values, 12);
    }

    rollingMean13(values) {
      return this.rollingMean(values, 13);
    }

    rollingMean14(values) {
      return this.rollingMean(values, 14);
    }

    rollingMean15(values) {
      return this.rollingMean(values, 15);
    }

    rollingMean16(values) {
      return this.rollingMean(values, 16);
    }

    rollingMean17(values) {
      return this.rollingMean(values, 17);
    }

    rollingMean18(values) {
      return this.rollingMean(values, 18);
    }

    rollingMean19(values) {
      return this.rollingMean(values, 19);
    }

    rollingMean20(values) {
      return this.rollingMean(values, 20);
    }

    rollingMean21(values) {
      return this.rollingMean(values, 21);
    }

    rollingMean22(values) {
      return this.rollingMean(values, 22);
    }

    rollingMean23(values) {
      return this.rollingMean(values, 23);
    }

    rollingMean24(values) {
      return this.rollingMean(values, 24);
    }

    rollingMean25(values) {
      return this.rollingMean(values, 25);
    }

    rollingMean26(values) {
      return this.rollingMean(values, 26);
    }

    rollingMean27(values) {
      return this.rollingMean(values, 27);
    }

    rollingMean28(values) {
      return this.rollingMean(values, 28);
    }

    rollingMean29(values) {
      return this.rollingMean(values, 29);
    }

    rollingMean30(values) {
      return this.rollingMean(values, 30);
    }

    status() {
      return {
        module: 'ClauseState',
        role: 'production-research-extension',
        calls: this.calls,
        methodCount: Object.getOwnPropertyNames(Object.getPrototypeOf(this)).length - 1,
        ageMs: (performance?.now?.() ?? Date.now()) - this.createdAt
      };
    }
  }

  globalThis.VilotClauseStateProductionLab = VilotClauseStateProductionLab;
})();
