/**
 * VilotNI 2.5 - SyntaxSmoother.js
 * System 34 foundation capacity expansion.
 *
 * Clause-aware last-mile realization cleanup. It is deliberately constrained:
 * the smoother may repair punctuation, agreement, duplicate function words and
 * connector collisions, but it may not invent semantic content or reorder
 * content words.
 */
(() => {
  'use strict';

  const norm = value => String(value || '').toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9'+.\- ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const escapeRE = value => String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  class VilotSyntaxSmoother {
    constructor(processor, options = {}, ruleStore = null) {
      this.processor = processor;
      this.ruleStore = ruleStore;
      this.options = {
        enabled: options.syntaxSmoothingEnabled !== false,
        maxPasses: Math.max(1, Math.min(6, options.syntaxSmoothingMaxPasses | 0 || 3)),
        maxRepairs: Math.max(8, Math.min(96, options.syntaxSmoothingMaxRepairs | 0 || 48))
      };
      this.smoothCount = 0;
      this.lastDiagnostics = null;
    }

    _rules() {
      return this.ruleStore?.syntaxPatterns?.() || this.ruleStore?.get?.('syntaxPatterns') || {};
    }

    _record(repairs, rule, before, after, meta = null) {
      if (before !== after && repairs.length < this.options.maxRepairs) repairs.push({ rule, before, after, ...(meta ? { meta } : {}) });
      return after;
    }

    _capitalizeSentences(text) {
      return String(text || '').replace(/(^|[.!?]\s+)([a-z])/g, (m, lead, ch) => `${lead}${ch.toUpperCase()}`);
    }

    _functionSets() {
      const data = this._rules()?.Function_Words || {};
      const flatten = names => new Set(names.flatMap(name => data?.[name] || []));
      return {
        all: flatten(['determiners','auxiliaries','modals','coordinators','subordinators','prepositions']),
        coordinators: flatten(['coordinators']),
        subordinators: flatten(['subordinators']),
        prepositions: flatten(['prepositions']),
        auxiliaries: flatten(['auxiliaries']),
        modals: flatten(['modals']),
        determiners: flatten(['determiners'])
      };
    }

    preserveComplexConnectors(tokens = []) {
      const connectors = this._functionSets().subordinators;
      return (tokens || []).filter((token, index, arr) => {
        if (!index) return true;
        const here = norm(token);
        const prev = norm(arr[index - 1]);
        return !(connectors.has(here) && here === prev);
      });
    }

    _repairWhitespaceAndPunctuation(text, repairs) {
      let next = text.replace(/\s+/g, ' ').trim();
      text = this._record(repairs, 'collapse-whitespace', text, next);
      next = text
        .replace(/\s+([,.;:!?])/g, '$1')
        .replace(/([,.;:!?])(?=[A-Za-z0-9])/g, '$1 ')
        .replace(/\s+([)\]])/g, '$1')
        .replace(/([([])\s+/g, '$1');
      text = this._record(repairs, 'punctuation-spacing', text, next);
      next = text.replace(/([!?.,])\1{1,}/g, '$1');
      return this._record(repairs, 'duplicate-punctuation', text, next);
    }

    _repairDuplicateFunctionWords(text, repairs) {
      const sets = this._functionSets();
      const words = Array.from(sets.all).filter(Boolean).sort((a, b) => b.length - a.length);
      if (!words.length) return text;
      const pattern = words.map(escapeRE).join('|');
      const re = new RegExp(`\\b(${pattern})\\s+\\1\\b`, 'gi');
      let next = text.replace(re, '$1');
      text = this._record(repairs, 'duplicate-function-word', text, next);
      next = text
        .replace(/\band\s+but\b/gi, 'but')
        .replace(/\bbut\s+and\b/gi, 'but')
        .replace(/\band\s+or\b/gi, 'or')
        .replace(/\bor\s+and\b/gi, 'or')
        .replace(/\bto\s+to\b/gi, 'to')
        .replace(/\bof\s+of\b/gi, 'of')
        .replace(/\bby\s+by\b/gi, 'by')
        .replace(/\bif\s+if\b/gi, 'if')
        .replace(/\bbecause\s+because\b/gi, 'because');
      return this._record(repairs, 'bridge-collision', text, next);
    }

    _repairArticles(text, repairs) {
      let next = text.replace(/\ba\s+([aeiou][a-z0-9'-]*)/gi, 'an $1');
      text = this._record(repairs, 'indefinite-article-vowel', text, next);
      next = text.replace(/\ban\s+([^aeiou\W][a-z0-9'-]*)/gi, 'a $1');
      text = this._record(repairs, 'indefinite-article-consonant', text, next);

      // Remove singular indefinite articles before nouns that lexical morphology
      // explicitly identifies as plural. This fixes "an inputs"/"a outputs"
      // without treating every word ending in -s as plural.
      const tokens = this.processor?.tokenize?.(text, 256) || [];
      for (let i = 0; i < tokens.length - 1; i++) {
        const article = String(tokens[i]?.lower || tokens[i]?.surface || '').toLowerCase();
        if (article !== 'a' && article !== 'an') continue;
        const nounToken = tokens.slice(i + 1).find(token => token?.pos !== 'Punct');
        if (!nounToken) continue;
        const entry = nounToken.entry || this.processor?.lookup?.(String(nounToken.lower || nounToken.surface || '').toLowerCase()) || null;
        if (!entry || this.processor?._inferNominalNumber?.(entry) !== 'plural') continue;
        const nounSurface = String(nounToken.surface || nounToken.lower || '').trim();
        if (!nounSurface) continue;
        const re = new RegExp(`\b(?:a|an)\s+${escapeRE(nounSurface)}\b`, 'i');
        text = this._record(repairs, 'remove-indefinite-article-before-plural', text, text.replace(re, nounSurface), { noun:nounSurface });
      }
      return text;
    }

    _repairAgreement(text, repairs) {
      // Keep this intentionally conservative. Only high-confidence local
      // pronoun/copula and demonstrative-number collisions are repaired.
      let next = text
        .replace(/\bI\s+is\b/g, 'I am')
        .replace(/\bI\s+are\b/g, 'I am')
        .replace(/\b(he|she|it|this|that)\s+are\b/gi, '$1 is')
        .replace(/\b(we|they|these|those)\s+is\b/gi, '$1 are')
        .replace(/\b(he|she|it|this|that)\s+were\b/gi, '$1 was')
        .replace(/\b(we|they|these|those)\s+was\b/gi, '$1 were');
      text = this._record(repairs, 'local-copula-agreement', text, next);
      next = text
        .replace(/\bthese\s+([A-Za-z][A-Za-z'-]*)\s+is\b/g, 'these $1 are')
        .replace(/\bthose\s+([A-Za-z][A-Za-z'-]*)\s+is\b/g, 'those $1 are');
      return this._record(repairs, 'demonstrative-number-agreement', text, next);
    }

    _repairConnectorPunctuation(text, repairs, analysis) {
      let next = text;
      // Remove a comma before restrictive that/which only when there is no
      // clear parenthetical comma pair. This avoids creating "..., that ...".
      next = next.replace(/,\s+that\b/gi, ' that');
      text = this._record(repairs, 'restrictive-that-punctuation', text, next);

      next = text
        .replace(/,\s*(because|although|though|while)\b/gi, ' $1')
        .replace(/\b(because|if|unless|although|though|while|after|before|when|whereas|since|until)\s*,\s*/gi, '$1 ');
      text = this._record(repairs, 'subordinate-bridge-punctuation', text, next);

      // For obvious independent clause coordination, insert a comma before
      // but/yet/so. "and" is excluded because noun-phrase coordination is too
      // ambiguous without a full parse.
      next = text.replace(/([A-Za-z0-9])\s+(but|yet|so)\s+((?:I|you|he|she|it|we|they|this|that|these|those|[A-Z][A-Za-z0-9'-]+)\s+(?:is|are|was|were|can|could|will|would|should|may|might|must|has|have|had|does|do|did)\b)/g, '$1, $2 $3');
      text = this._record(repairs, 'independent-coordinator-comma', text, next);

      const complex = analysis?.complexLanguage || {};
      const hasConditional = (complex.dependencies || []).some(d => /condition/.test(String(d?.relation || ''))) || /^\s*(if|unless)\b/i.test(text);
      if (hasConditional) {
        // Initial dependent clauses commonly need a comma before the main
        // clause. Only add one when an explicit comma is absent and the parser
        // exposes a clause boundary.
        const clauses = complex.clauses || [];
        if (clauses.length >= 2 && /^\s*(if|unless|although|though|when|after|before|while)\b/i.test(text) && !/,/.test(text.slice(0, Math.max(0, text.length * 0.6)))) {
          const firstText = String(clauses[0]?.text || '').trim().replace(/[,.!?]+$/, '');
          if (firstText && norm(text).startsWith(norm(firstText))) {
            const idx = text.toLowerCase().indexOf(firstText.toLowerCase());
            const end = idx >= 0 ? idx + firstText.length : -1;
            if (end > 0 && text[end] !== ',') {
              next = `${text.slice(0, end)},${text.slice(end)}`;
              text = this._record(repairs, 'initial-dependent-clause-comma', text, next);
            }
          }
        }
      }
      return text;
    }

    _repairRepeatedConnectors(text, repairs) {
      const sets = this._functionSets();
      const connectors = Array.from(sets.subordinators).filter(Boolean).sort((a, b) => b.length - a.length);
      let next = text;
      for (const connector of connectors) {
        const re = new RegExp(`\\b${escapeRE(connector)}\\s+${escapeRE(connector)}\\b`, 'gi');
        next = next.replace(re, connector);
      }
      text = this._record(repairs, 'repeated-complex-connector', text, next);
      next = text.replace(/\b(because|although|though|while|if|unless)\s+(and|but)\s+\1\b/gi, '$1');
      return this._record(repairs, 'nested-connector-collision', text, next);
    }

    _repairDanglingTerminal(text, repairs) {
      const invalid = new Set(this._rules()?.Terminal_Invalid || []);
      const parts = text.trim().split(/\s+/);
      if (!parts.length) return text;
      let changed = false;
      while (parts.length > 2) {
        const terminal = norm(parts[parts.length - 1]).replace(/[.!?]+$/g, '');
        if (!invalid.has(terminal)) break;
        parts.pop();
        changed = true;
      }
      if (!changed) return text;
      return this._record(repairs, 'dangling-terminal-function-word', text, parts.join(' '));
    }

    _repairSentenceBoundary(text, repairs) {
      let next = text.replace(/([.!?])\s*([a-z])/g, (m, p, ch) => `${p} ${ch.toUpperCase()}`);
      text = this._record(repairs, 'sentence-boundary-capitalization', text, next);
      next = this._capitalizeSentences(text);
      return this._record(repairs, 'sentence-capitalization', text, next);
    }

    _semanticSafety(before, after) {
      if (before === after) return true;
      const funcs = this._functionSets().all;
      const content = value => {
        const parsed = this.processor?.tokenize?.(value, 256) || [];
        if (parsed.length) return parsed.filter(t => t?.pos !== 'Punct').map(t => norm(t?.lower || t?.surface)).filter(w => w && !funcs.has(w));
        return norm(value).split(/\s+/).filter(w => w && !funcs.has(w));
      };
      const a = content(before);
      const b = content(after);
      // Do not accept a smoother result that loses or invents multiple content
      // tokens. Inflectional token differences are tolerated by root lookup.
      const roots = rows => rows.map(word => {
        const entry = this.processor?.lookup?.(word) || this.processor?.resolveSayableWord?.(word, { allowRootFallback: true }) || null;
        return norm(entry?.root || entry?.lemma || word);
      });
      const ra = roots(a), rb = roots(b);
      const counts = rows => rows.reduce((m, v) => (m.set(v, (m.get(v) || 0) + 1), m), new Map());
      const ca = counts(ra), cb = counts(rb);
      let delta = 0;
      for (const key of new Set([...ca.keys(), ...cb.keys()])) delta += Math.abs((ca.get(key) || 0) - (cb.get(key) || 0));
      return delta <= 2;
    }

    smooth(input, analysis = null) {
      const started = performance.now();
      const repairs = [];
      const original = String(input || '').trim();
      let text = original;
      if (!this.options.enabled || !text) {
        return { text, repairs, diagnostics: { smootherMs: performance.now() - started, repairCount: 0, ruleDriven: Boolean(this._rules()?.Schema_Version) } };
      }

      for (let pass = 0; pass < this.options.maxPasses; pass++) {
        const beforePass = text;
        text = this._repairWhitespaceAndPunctuation(text, repairs);
        text = this._repairDuplicateFunctionWords(text, repairs);
        text = this._repairArticles(text, repairs);
        text = this._repairAgreement(text, repairs);
        text = this._repairRepeatedConnectors(text, repairs);
        text = this._repairConnectorPunctuation(text, repairs, analysis);
        text = this._repairDanglingTerminal(text, repairs);
        text = this._repairSentenceBoundary(text, repairs);
        if (text === beforePass) break;
      }

      if (text && !/[.!?]$/.test(text)) {
        text = this._record(repairs, 'terminal-punctuation', text, `${text}.`);
      }

      // System 36 shared grammar pass. Only high-confidence surface repairs are
      // accepted, and this class still performs its independent semantic-root
      // safety check afterward.
      const grammarRepair = this.processor?.grammarEngine?.repair?.(text, analysis) || null;
      if (grammarRepair?.changed && grammarRepair.text) {
        const beforeGrammar = text;
        text = grammarRepair.text;
        for (const row of grammarRepair.repairs || []) {
          if (repairs.length >= this.options.maxRepairs) break;
          repairs.push({ rule: `grammar-engine:${row.rule || 'repair'}`, before: row.before, after: row.after });
        }
        if (beforeGrammar !== text && !grammarRepair.repairs?.length) repairs.push({rule:'grammar-engine:surface-repair',before:beforeGrammar,after:text});
      }

      const safe = this._semanticSafety(original, text);
      if (!safe) {
        repairs.push({ rule: 'semantic-safety-rollback', before: text, after: original });
        text = original;
      }

      const complex = analysis?.complexLanguage || {};
      const result = {
        text,
        repairs,
        diagnostics: {
          smootherMs: performance.now() - started,
          repairCount: repairs.length,
          explanationMode: analysis?.explanationMode || null,
          clauseCount: complex.clauseCount || complex.clauses?.length || 0,
          dependencyCount: complex.dependencies?.length || 0,
          semanticSafetyPassed: safe,
          ruleDriven: Boolean(this._rules()?.Schema_Version),
          grammarSchema: Number(this.processor?.grammarEngine?.rules?.()?.Schema_Version || 0),
          grammarScore: Number(this.processor?.grammarEngine?.lastAudit?.score || 0),
          grammarRepairCount: Number(grammarRepair?.repairs?.length || 0)
        }
      };

      this.smoothCount++;
      this.lastDiagnostics = result.diagnostics;
      return result;
    }

    status() {
      return {
        ready: true,
        role: 'clause-aware-semantically-conservative-final-syntax-smoother',
        smoothCount: this.smoothCount,
        lastDiagnostics: this.lastDiagnostics,
        architecture: {
          semanticChangesAllowed: false,
          clauseAware: true,
          ruleDriven: Boolean(this._rules()?.Schema_Version),
          bridgeCollisionRepair: true,
          agreementRepair: true,
          articleRepair: true,
          connectorRepair: true,
          punctuationRepair: true,
          semanticSafetyRollback: true
        }
      };
    }
  }


  const V34_SMOOTHING_RULES=Object.freeze([
    { id:'double_space', family:'whitespace', safety:'safe', enabled:true },
    { id:'space_before_punctuation', family:'punctuation', safety:'safe', enabled:true },
    { id:'missing_space_after_punctuation', family:'punctuation', safety:'safe', enabled:true },
    { id:'duplicate_comma', family:'punctuation', safety:'safe', enabled:true },
    { id:'duplicate_period', family:'punctuation', safety:'safe', enabled:true },
    { id:'mixed_terminal', family:'punctuation', safety:'safe', enabled:true },
    { id:'lowercase_sentence_start', family:'capitalization', safety:'safe', enabled:true },
    { id:'lowercase_i', family:'capitalization', safety:'safe', enabled:true },
    { id:'duplicate_article', family:'function_word', safety:'safe', enabled:true },
    { id:'duplicate_auxiliary', family:'function_word', safety:'safe', enabled:true },
    { id:'duplicate_modal', family:'function_word', safety:'safe', enabled:true },
    { id:'duplicate_coordinator', family:'function_word', safety:'safe', enabled:true },
    { id:'repeated_subordinator', family:'connector', safety:'safe', enabled:true },
    { id:'connector_collision', family:'connector', safety:'review', enabled:true },
    { id:'dangling_subordinator', family:'connector', safety:'review', enabled:true },
    { id:'dangling_preposition', family:'terminal', safety:'review', enabled:true },
    { id:'dangling_modal', family:'terminal', safety:'review', enabled:true },
    { id:'dangling_determiner', family:'terminal', safety:'review', enabled:true },
    { id:'agreement_singular_plural', family:'agreement', safety:'review', enabled:true },
    { id:'article_a_an', family:'agreement', safety:'safe', enabled:true },
    { id:'independent_clause_comma', family:'clause', safety:'review', enabled:true },
    { id:'restrictive_that_comma', family:'clause', safety:'review', enabled:true },
    { id:'condition_comma', family:'clause', safety:'review', enabled:true },
    { id:'introductory_clause_comma', family:'clause', safety:'review', enabled:true },
    { id:'contrast_connector', family:'clause', safety:'review', enabled:true },
    { id:'causal_connector', family:'clause', safety:'review', enabled:true },
    { id:'temporal_connector', family:'clause', safety:'review', enabled:true },
    { id:'comparison_connector', family:'clause', safety:'review', enabled:true },
    { id:'negation_scope', family:'semantic', safety:'protected', enabled:true },
    { id:'condition_scope', family:'semantic', safety:'protected', enabled:true },
    { id:'reference_chain', family:'semantic', safety:'protected', enabled:true },
    { id:'temporal_order', family:'semantic', safety:'protected', enabled:true },
    { id:'mechanism_order', family:'semantic', safety:'protected', enabled:true },
    { id:'quantifier_scope', family:'semantic', safety:'protected', enabled:true },
    { id:'proper_name', family:'semantic', safety:'protected', enabled:true },
    { id:'number_unit', family:'semantic', safety:'protected', enabled:true },
    { id:'quote_balance', family:'punctuation', safety:'review', enabled:true },
    { id:'parenthesis_balance', family:'punctuation', safety:'review', enabled:true },
    { id:'colon_list', family:'punctuation', safety:'review', enabled:true },
    { id:'semicolon_clause', family:'punctuation', safety:'review', enabled:true },
    { id:'pronoun_case', family:'reference', safety:'review', enabled:true },
    { id:'reflexive_binding', family:'reference', safety:'protected', enabled:true },
    { id:'relative_clause_attachment', family:'reference', safety:'protected', enabled:true },
    { id:'demonstrative_reference', family:'reference', safety:'protected', enabled:true },
    { id:'sentence_fragment', family:'clause', safety:'review', enabled:true },
    { id:'run_on_sentence', family:'clause', safety:'review', enabled:true },
    { id:'overmerged_clause', family:'clause', safety:'review', enabled:true },
    { id:'underconnected_clause', family:'clause', safety:'review', enabled:true },
    { id:'repeated_subject', family:'style', safety:'review', enabled:true },
    { id:'repeated_predicate', family:'style', safety:'review', enabled:true },
    { id:'redundant_bridge', family:'style', safety:'safe', enabled:true },
    { id:'empty_parenthetical', family:'style', safety:'safe', enabled:true },
    { id:'list_parallelism', family:'style', safety:'review', enabled:true },
    { id:'comparison_parallelism', family:'style', safety:'review', enabled:true },
    { id:'mechanism_parallelism', family:'style', safety:'review', enabled:true },
    { id:'multi_part_parallelism', family:'style', safety:'review', enabled:true },
    { id:'terminal_capitalization', family:'terminal', safety:'safe', enabled:true },
    { id:'terminal_mark', family:'terminal', safety:'safe', enabled:true },
    { id:'question_terminal', family:'terminal', safety:'safe', enabled:true },
    { id:'exclamation_preservation', family:'terminal', safety:'safe', enabled:true },
  ]);
  const v34SNorm=v=>String(v??'').replace(/\s+/g,' ').trim();
  const v34STokens=v=>String(v??'').match(/[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[^\sA-Za-z0-9]/g)||[];
  class V34SyntaxEngine {
    constructor(owner){this.owner=owner;this.last=null;this.history=[];}
    protectedFacts(analysis={}){
      const set=new Set();
      const add=v=>{const s=String(v??'').trim();if(s)set.add(s.toLowerCase());};
      for(const p of analysis?.propositionGraph?.propositions||analysis?.propositions||[]){add(p.subject);add(p.relation);add(p.object);add(p.condition);}
      for(const r of analysis?.referenceLinks||[]){add(r.surface);add(r.antecedent);add(r.target);}
      for(const e of analysis?.temporalAssessment?.relations||[]){add(e.from);add(e.to);add(e.relation);}
      for(const s of analysis?.mechanismGraph?.stages||[]){add(s.label);add(s.operation);add(s.input);add(s.output);}
      return set;
    }
    sentenceSplit(text=''){return String(text||'').match(/[^.!?]+[.!?]?/g)?.map(x=>x.trim()).filter(Boolean)||[];}
    clauseSplit(sentence=''){
      const connectors=/\b(?:because|although|though|if|unless|when|while|after|before|since|until|so that|whereas|but|and|or|yet)\b/gi;
      const parts=[];let last=0;let m;while((m=connectors.exec(sentence))){if(m.index>last)parts.push({text:sentence.slice(last,m.index).trim(),connector:null});parts.push({text:m[0],connector:m[0].toLowerCase()});last=connectors.lastIndex;}if(last<sentence.length)parts.push({text:sentence.slice(last).trim(),connector:null});return parts.filter(x=>x.text);
    }
    fingerprint(text=''){
      const tokens=v34STokens(text).map(t=>t.toLowerCase());
      const content=tokens.filter(t=>/^[a-z0-9]/.test(t)&&!new Set(['the','a','an','and','or','but','to','of','in','on','at','for','with','by','is','are','was','were','be','been','being']).has(t));
      return {tokens,content:new Set(content),negations:tokens.filter(t=>['not','no','never','without'].includes(t)).length,numbers:tokens.filter(t=>/^\d/.test(t)),connectors:tokens.filter(t=>['because','if','unless','although','while','after','before','whereas','but'].includes(t))};
    }
    semanticSafety(before,after,analysis={}){
      const a=this.fingerprint(before),b=this.fingerprint(after); let missing=0;
      for(const word of a.content)if(!b.content.has(word))missing++;
      const contentLoss=missing/Math.max(1,a.content.size); const negationChanged=a.negations!==b.negations;
      const numberChanged=JSON.stringify(a.numbers)!==JSON.stringify(b.numbers);
      const protectedFacts=this.protectedFacts(analysis); const missingProtected=[];
      for(const fact of protectedFacts) if(fact.length>2 && before.toLowerCase().includes(fact) && !after.toLowerCase().includes(fact)) missingProtected.push(fact);
      const safe=contentLoss<=0.08&&!negationChanged&&!numberChanged&&!missingProtected.length;
      return {safe,contentLoss,negationChanged,numberChanged,missingProtected:missingProtected.slice(0,12)};
    }
    punctuationPass(text,repairs){
      let out=String(text||''); const apply=(id,rx,repl)=>{const before=out;out=out.replace(rx,repl);if(out!==before)repairs.push({rule:id,before,after:out});};
      apply('double_space',/[ 	]{2,}/g,' '); apply('space_before_punctuation',/\s+([,.;:!?])/g,'$1'); apply('missing_space_after_punctuation',/([,;:!?])(?=[A-Za-z])/g,'$1 ');
      apply('duplicate_comma',/,+/g,','); apply('duplicate_period',/\.2(?!\.)/g,'.'); apply('empty_parenthetical',/\(\s*\)/g,'');
      return out.trim();
    }
    articlePass(text,repairs){
      let out=String(text||''); const before=out;
      out=out.replace(/\b(a)\s+([aeiou][a-z]*)\b/gi,(m,a,w)=>`an ${w}`).replace(/\b(an)\s+([^aeiou\W][a-z]*)\b/gi,(m,a,w)=>`a ${w}`);
      out=out.replace(/\b(the|a|an)\s+\1\b/gi,'$1');

      // The deep system-34/35 pass must preserve the same nominal-number
      // guarantee as the base smoother. Never leave a/an in front of a noun
      // that the lexicon explicitly marks plural.
      const processor=this.owner?.processor;
      if(processor?.tokenize){
        const tokens=processor.tokenize(out,256)||[];
        for(let i=0;i<tokens.length-1;i++){
          const article=String(tokens[i]?.lower||tokens[i]?.surface||'').toLowerCase();
          if(article!=='a'&&article!=='an')continue;
          const noun=tokens.slice(i+1).find(token=>token?.pos!=='Punct');
          if(!noun)continue;
          const entry=noun.entry||processor.lookup?.(String(noun.lower||noun.surface||'').toLowerCase())||null;
          if(!entry||processor._inferNominalNumber?.(entry)!=='plural')continue;
          const surface=String(noun.surface||noun.lower||'').trim();
          if(!surface)continue;
          const rx=new RegExp(`\\b(?:a|an)\\s+${escapeRE(surface)}\\b`,'i');
          out=out.replace(rx,surface);
        }
      }
      if(out!==before)repairs.push({rule:'article_a_an_or_plural',before,after:out});return out;
    }
    duplicateFunctionPass(text,repairs){
      const functions=['and','or','but','because','although','if','when','while','after','before','to','of','for','with','is','are','was','were','can','could','will','would','should'];let out=text;
      for(const word of functions){const rx=new RegExp(`\\b(${word})\\s+\\1\\b`,'gi');const before=out;out=out.replace(rx,'$1');if(out!==before)repairs.push({rule:'duplicate_function_word',word,before,after:out});}return out;
    }
    connectorPass(text,repairs){
      let out=text; const collisions=[[/\bbecause\s+so\b/gi,'because'],[/\balthough\s+but\b/gi,'although'],[/\bif\s+then\s+then\b/gi,'if then'],[/\bbut\s+however\b/gi,'however'],[/\band\s+also\s+also\b/gi,'and also']];
      for(const [rx,repl] of collisions){const before=out;out=out.replace(rx,repl);if(out!==before)repairs.push({rule:'connector_collision',before,after:out});}return out;
    }
    terminalPass(text,analysis,repairs){
      let out=text.trim(); if(!out)return out; const question=Boolean(analysis?.isQuestion||analysis?.questionState?.isQuestion||/^\s*(who|what|when|where|why|how|which|can|could|would|should|is|are|do|does|did)\b/i.test(out));
      if(!/[.!?]$/.test(out)){const before=out;out+=question?'?':'.';repairs.push({rule:'terminal_mark',before,after:out});}return out;
    }
    capitalizationPass(text,repairs){
      let out=text; const before=out;out=out.replace(/(^|[.!?]\s+)([a-z])/g,(m,p,c)=>p+c.toUpperCase()).replace(/\bi\b/g,'I');if(out!==before)repairs.push({rule:'capitalization',before,after:out});return out;
    }
    clauseIntegrity(text,analysis={}){
      const sentences=this.sentenceSplit(text);const issues=[];
      sentences.forEach((sentence,si)=>{const clauses=this.clauseSplit(sentence);clauses.forEach((part,ci)=>{if(part.connector&&ci===clauses.length-1)issues.push({type:'dangling-connector',sentence:si,connector:part.connector});});if(sentence.split(/\s+/).length>55)issues.push({type:'overlong-sentence',sentence:si});});
      const required=(analysis?.clauseGraph||[]).filter(c=>c?.relation||c?.type).length;return{sentences:sentences.length,issues,requiredClauses:required};
    }
    run(text,analysis={}){
      const before=String(text||'');let out=before;const repairs=[];out=this.punctuationPass(out,repairs);out=this.duplicateFunctionPass(out,repairs);out=this.articlePass(out,repairs);out=this.connectorPass(out,repairs);out=this.capitalizationPass(out,repairs);out=this.terminalPass(out,analysis,repairs);
      const safety=this.semanticSafety(before,out,analysis);if(!safety.safe)out=before;const integrity=this.clauseIntegrity(out,analysis);const result={text:v34SNorm(out),repairs:safety.safe?repairs:[],rolledBack:!safety.safe,safety,integrity};this.last=result;this.history.push({at:Date.now(),repairs:result.repairs.length,rolledBack:result.rolledBack,issues:integrity.issues.length});if(this.history.length>128)this.history.shift();return result;
    }
  }
  const v34OriginalSmooth=VilotSyntaxSmoother.prototype.smooth;
  VilotSyntaxSmoother.prototype.smooth=function(input,analysis=null){
    const base=v34OriginalSmooth.call(this,input,analysis)||{text:String(input||'')};this.v34Engine||=new V34SyntaxEngine(this);const deep=this.v34Engine.run(base.text??base.output??String(input||''),analysis||{});
    return {...base,text:deep.text,output:deep.text,system34:true,deepRepairs:deep.repairs,deepSafety:deep.safety,clauseIntegrity:deep.integrity,deepRollback:deep.rolledBack};
  };
  VilotSyntaxSmoother.prototype.deepRules=function(){return V34_SMOOTHING_RULES.map(x=>({...x}));};
  VilotSyntaxSmoother.prototype.analyzeClauseIntegrity=function(text,analysis={}){this.v34Engine||=new V34SyntaxEngine(this);return this.v34Engine.clauseIntegrity(text,analysis);};



  const V34_SYNTAX_SCENARIOS = Object.freeze([
    {
      id: "punctuation-scenario-1",
      family: "punctuation",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "agreement-scenario-2",
      family: "agreement",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "connector-scenario-3",
      family: "connector",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "clause-boundary-scenario-4",
      family: "clause-boundary",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "terminal-scenario-5",
      family: "terminal",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "reference-scenario-6",
      family: "reference",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "negation-scenario-7",
      family: "negation",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "condition-scenario-8",
      family: "condition",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "parallelism-scenario-9",
      family: "parallelism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "semantic-safety-scenario-10",
      family: "semantic-safety",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "punctuation-scenario-11",
      family: "punctuation",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "agreement-scenario-12",
      family: "agreement",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "connector-scenario-13",
      family: "connector",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "clause-boundary-scenario-14",
      family: "clause-boundary",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "terminal-scenario-15",
      family: "terminal",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "reference-scenario-16",
      family: "reference",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "negation-scenario-17",
      family: "negation",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "condition-scenario-18",
      family: "condition",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "parallelism-scenario-19",
      family: "parallelism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "semantic-safety-scenario-20",
      family: "semantic-safety",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "punctuation-scenario-21",
      family: "punctuation",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "agreement-scenario-22",
      family: "agreement",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "connector-scenario-23",
      family: "connector",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "clause-boundary-scenario-24",
      family: "clause-boundary",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "terminal-scenario-25",
      family: "terminal",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "reference-scenario-26",
      family: "reference",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "negation-scenario-27",
      family: "negation",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "condition-scenario-28",
      family: "condition",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "parallelism-scenario-29",
      family: "parallelism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "semantic-safety-scenario-30",
      family: "semantic-safety",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "punctuation-scenario-31",
      family: "punctuation",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "agreement-scenario-32",
      family: "agreement",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "connector-scenario-33",
      family: "connector",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "clause-boundary-scenario-34",
      family: "clause-boundary",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "terminal-scenario-35",
      family: "terminal",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "reference-scenario-36",
      family: "reference",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "negation-scenario-37",
      family: "negation",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "condition-scenario-38",
      family: "condition",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "parallelism-scenario-39",
      family: "parallelism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "semantic-safety-scenario-40",
      family: "semantic-safety",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "punctuation-scenario-41",
      family: "punctuation",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "agreement-scenario-42",
      family: "agreement",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "connector-scenario-43",
      family: "connector",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "clause-boundary-scenario-44",
      family: "clause-boundary",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "terminal-scenario-45",
      family: "terminal",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "reference-scenario-46",
      family: "reference",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: true,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "negation-scenario-47",
      family: "negation",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "condition-scenario-48",
      family: "condition",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
  ]);
  VilotSyntaxSmoother.prototype.syntaxScenarios = function() { return V34_SYNTAX_SCENARIOS.map(row => ({ ...row })); };

  globalThis.VilotSyntaxSmoother = VilotSyntaxSmoother;
})();
