/**
 * VilotNI 2 - Processor.js
 * Production structural syntax, lexical contracts, morphology, and hard legality.
 *
 * Ownership rule:
 *   Processor decides whether a candidate is structurally legal.
 *   Processor does NOT decide which legal candidate should win.
 *
 * Dataset:
 *   Words.json follows the user's Vocab -> <Category>-Vocab -> word structure.
 */
(() => {
  'use strict';

  const CONTENT_POS = new Set(['Noun','ProperNoun','Verb','Adj','Adv','Num']);
  const SUBJECT_POS = new Set(['Noun','ProperNoun','Pronoun','Num']);
  const NOMINAL_POS = new Set(['Det','Adj','Noun','ProperNoun','Pronoun','Num']);
  const STOP_WORDS = new Set([
    'a','an','the','this','that','these','those','my','your','our','their','its',
    'is','are','am','was','were','be','been','being','do','does','did','have','has','had',
    'can','could','will','would','should','may','might','must','shall',
    'of','to','in','on','at','for','from','with','by','as','and','or','but',
    'what','why','who','whom','whose','which','when','where','how','please'
  ]);

  const QUESTION_WORDS = new Set(['what','why','who','whom','whose','which','when','where','how']);
  const YES_NO_OPENERS = new Set([
    'is','are','am','was','were','do','does','did','can','could','will','would',
    'should','may','might','must','shall','has','have','had'
  ]);

  const DERIVATIONAL_PREFIXES = [
    ['explan','explain'],['explain','explain'],
    ['accelerat','accelerate'],['comput','compute'],['calculat','calculate'],
    ['inspect','inspect'],['predict','predict'],['generat','generate'],
    ['activat','activate'],['inhibit','inhibit'],['consolidat','consolidate'],
    ['persist','persist'],['reason','reason'],['learn','learn'],['train','train'],
    ['decod','decode'],['encod','encode'],['process','process'],['associat','associate'],
    ['represent','represent'],['optimiz','optimize'],['normaliz','normalize'],
    ['stabil','stable'],['probab','probability'],['semantic','semantic'],
    ['grammat','grammar'],['morpholog','morphology'],['lexic','lexical']
  ];

  const SUFFIX_RULES = [
    ['abilities','able'],['ability','able'],['ableness','able'],['ably',''],['able',''],
    ['ations','ate'],['ation','ate'],['ators','ate'],['ator','ate'],
    ['ications','icate'],['ication','icate'],['izers','ize'],['izer','ize'],
    ['ments',''],['ment',''],['nesses',''],['ness',''],['ities','y'],['ity',''],
    ['ingly',''],['edly',''],['ing',''],['ied','y'],['ies','y'],['ed',''],['es',''],['s','']
  ];

  const normalizePOS = pos => {
    const p = String(pos || 'Other');
    switch (p) {
      case 'Adjective': return 'Adj';
      case 'Adverb': return 'Adv';
      case 'Determiner': return 'Det';
      case 'Preposition': return 'Prep';
      case 'Conjunction': return 'Conj';
      case 'Auxiliary Verb': return 'Aux';
      case 'Modal Verb': return 'Modal';
      case 'Proper Noun': return 'ProperNoun';
      case 'Number': return 'Num';
      case 'Punctuation': return 'Punct';
      default: return p;
    }
  };

  class VilotProcessor {
    constructor(arena, rsl, trace = null, options = {}, ruleStore = null) {
      this.arena = arena;
      this.rsl = rsl;
      this.trace = trace;
      this.ruleStore = ruleStore;
      this.options = {
        strictRootFamilies: options.strictRootFamilies !== false,
        rootRepeatWindow: Math.max(6, options.rootRepeatWindow | 0 || 32),
        maxPromptTokens: Math.max(32, options.maxPromptTokens | 0 || 256),
        maxResponseTokens: Math.max(32, options.maxResponseTokens | 0 || 512)
      };

      this.ready = false;
      this.schemaVersion = 0;
      this.entries = new Map();
      // Every spelling may legitimately occupy more than one POS role. The
      // canonical entries map remains the lexical lookup fast path while this
      // map preserves all observed POS_Type alternatives from Words.json.
      this.posAlternatives = new Map();
      this.parentEntries = new Map();
      this.rootMembers = new Map();
      this.categoryCounts = new Map();
      this.associationIndex = new Map();
      this.ambiguousPOSWords = 0;
      this.posTypeRelations = {};
      this.posTypeTrigrams = [];
      this.loadedWords = 0;
      this.loadedParents = 0;
      this.lastAnalysis = null;
      this.lastAudit = null;
      this.lastLoadMs = 0;
    }

    async load(url = './Words.json') {
      const started = performance.now();
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Processor.js could not load ${url}: HTTP ${response.status}`);
      const data = await response.json();
      const vocab = data?.Vocab || {};
      this.posTypeRelations = data?.POS_Type_Relations || {};
      this.posTypeTrigrams = Array.isArray(data?.POS_Type_Trigrams) ? data.POS_Type_Trigrams : [];

      this.schemaVersion = Number(data?.Schema_Version || 0);
      this.entries.clear();
      this.posAlternatives.clear();
      this.parentEntries.clear();
      this.rootMembers.clear();
      this.categoryCounts.clear();
      this.associationIndex.clear();
      this.ambiguousPOSWords = 0;

      for (const [categoryName, bucket] of Object.entries(vocab)) {
        if (!bucket || typeof bucket !== 'object') continue;
        let count = 0;
        for (const [surface, raw] of Object.entries(bucket)) {
          if (!raw || typeof raw !== 'object') continue;
          const parent = this._normalizeEntry(surface, raw, categoryName, null);
          this._registerEntry(parent);
          this.parentEntries.set(parent.lower, parent);
          count++;

          const variants = raw.Word_Types || {};
          for (const [variantSurface, variantRaw] of Object.entries(variants)) {
            const variant = this._normalizeEntry(
              variantSurface,
              variantRaw,
              categoryName,
              parent
            );
            this._registerEntry(variant);
          }
        }
        this.categoryCounts.set(categoryName, count);
      }

      this._finalizePOSAlternatives();
      this._buildAssociationIndex();
      this.loadedWords = this.entries.size;
      this.loadedParents = this.parentEntries.size;
      this.ready = true;
      this.lastLoadMs = performance.now() - started;
      this.trace?.push?.(globalThis.VilotTraceEvent?.PROCESSOR_LOAD || 20, this.lastLoadMs, this.loadedWords, this.loadedParents);
      return this.status();
    }

    _normalizeEntry(surface, raw, categoryName, parent) {
      const lower = String(surface || '').toLowerCase();
      const pos = normalizePOS(raw?.POS_Type || parent?.pos || 'Other');
      const lemma = String(raw?.Lemma || parent?.lemma || lower).toLowerCase();
      const root = String(raw?.Root_Family || parent?.root || this.deriveRoot(lower, lemma)).toLowerCase();
      const semanticClass = String(raw?.Semantic_Class || parent?.semanticClass || 'unclassified');
      const associations = Array.isArray(raw?.Associations)
        ? raw.Associations
        : (parent?.associations || []);
      return {
        surface: String(surface || ''),
        lower,
        id: raw?.ID || null,
        category: String(raw?.Word_Vocab || parent?.category || categoryName.replace(/-Vocab$/,'')),
        categoryName,
        pos,
        // possiblePOS is finalized after the whole vocabulary is loaded. It is
        // intentionally lexical metadata, not a hard decision about this token
        // occurrence. RSL resolves the active POS from context at runtime.
        possiblePOS: [pos],
        lemma,
        root,
        semanticClass,
        contentWord: raw?.Content_Word !== undefined
          ? Boolean(raw.Content_Word)
          : (parent?.contentWord ?? CONTENT_POS.has(pos)),
        grammar: raw?.Grammar || parent?.grammar || {},
        associations,
        parent: parent?.surface || null,
        active: raw?.Active !== false,
        originalTokenId: raw?.Original_Token_ID ?? parent?.originalTokenId ?? null
      };
    }

    _registerEntry(entry) {
      if (!entry.lower) return;

      let alternatives = this.posAlternatives.get(entry.lower);
      if (!alternatives) {
        alternatives = new Map();
        this.posAlternatives.set(entry.lower, alternatives);
      }
      const oldAlt = alternatives.get(entry.pos);
      if (!oldAlt || (!oldAlt.active && entry.active)) alternatives.set(entry.pos, entry);

      const existing = this.entries.get(entry.lower);
      if (!existing || (!existing.active && entry.active)) this.entries.set(entry.lower, entry);
      if (entry.root) {
        let members = this.rootMembers.get(entry.root);
        if (!members) {
          members = new Set();
          this.rootMembers.set(entry.root, members);
        }
        members.add(entry.lower);
      }
    }

    _finalizePOSAlternatives() {
      this.ambiguousPOSWords = 0;
      for (const [lower, alternatives] of this.posAlternatives) {
        const canonical = this.entries.get(lower);
        const possible = new Set(alternatives.keys());

        // A spelling does not need to be duplicated across vocabulary buckets
        // to acquire another possible role. If members of its derivational root
        // family are observed in other POS classes, those classes become
        // contextual possibilities that RSL may select. This is a lexical
        // generalization, not a hand-written per-word rule.
        const root = canonical?.root || '';
        if (root) {
          for (const memberLower of this.rootMembers.get(root) || []) {
            const memberAlternatives = this.posAlternatives.get(memberLower);
            if (memberAlternatives) {
              for (const pos of memberAlternatives.keys()) {
                if (pos !== 'Punct' && this._surfaceCanTakePOS(lower, root, pos, canonical?.pos)) possible.add(pos);
              }
            } else {
              const member = this.entries.get(memberLower);
              if (member?.pos && member.pos !== 'Punct' && this._surfaceCanTakePOS(lower, root, member.pos, canonical?.pos)) possible.add(member.pos);
            }
          }
        }

        const possiblePOS = Array.from(possible);
        if (possiblePOS.length > 1) this.ambiguousPOSWords++;
        if (canonical) canonical.possiblePOS = possiblePOS.slice();
        for (const entry of alternatives.values()) entry.possiblePOS = possiblePOS.slice();
      }
    }

    _surfaceCanTakePOS(lower, root, candidatePOS, canonicalPOS) {
      if (candidatePOS === canonicalPOS) return true;
      const word = String(lower || '').toLowerCase();
      const r = String(root || '').toLowerCase();
      if (candidatePOS === 'Verb') {
        if (word === r || /(ed|ing|ize|ise|ate|fy)$/.test(word)) return true;
        if (word.endsWith('es')) {
          const stem = word.slice(0, -2);
          return stem === r || `${stem}e` === r || `${stem}s` === r;
        }
        if (word.endsWith('s')) {
          const stem = word.slice(0, -1);
          return stem === r || `${stem}e` === r;
        }
        return false;
      }
      if (candidatePOS === 'Noun' && word === r) return true;
      if (candidatePOS === 'Adj') {
        return /(able|ible|al|ary|ful|ic|ical|ish|ive|less|ous)$/.test(word);
      }
      if (candidatePOS === 'Adv') return /ly$/.test(word);
      // Do not infer noun/proper-noun/function-word roles from a derivational
      // neighbor. Those require a direct lexical observation.
      return false;
    }

    posCandidates(wordOrEntry) {
      const entry = typeof wordOrEntry === 'string' ? this.lookup(wordOrEntry) : wordOrEntry;
      if (!entry) return [];
      const alternatives = this.posAlternatives.get(entry.lower);
      if (!alternatives?.size) return [entry];
      return Array.from(alternatives.values()).filter(x => x?.active !== false);
    }

    materializePOS(wordOrEntry, pos) {
      const entry = typeof wordOrEntry === 'string' ? this.lookup(wordOrEntry) : wordOrEntry;
      if (!entry) return null;
      const normalized = normalizePOS(pos || entry.pos);
      const exact = this.posAlternatives.get(entry.lower)?.get(normalized);
      const source = exact || entry;
      if (source.pos === normalized && source.possiblePOS) return source;
      return {
        ...source,
        pos: normalized,
        possiblePOS: Array.from(new Set([...(entry.possiblePOS || [entry.pos]), normalized]))
      };
    }

    deriveRoot(word, lemma = '') {
      let w = String(lemma || word || '').toLowerCase().trim();
      w = w.replace(/[\s.,!?;:()[\]{}"`]+/g, '');
      w = w.replace(/’/g, "'");
      if (w.endsWith("'s")) w = w.slice(0, -2);
      else if (w.endsWith("s'")) w = w.slice(0, -1);
      if (!w) return '';

      for (const [prefix, root] of DERIVATIONAL_PREFIXES) {
        if (w.startsWith(prefix) && w.length >= prefix.length) return root;
      }

      let base = w;
      for (const [suffix, replacement] of SUFFIX_RULES) {
        if (base.endsWith(suffix) && base.length - suffix.length >= 3) {
          const stem = base.slice(0, -suffix.length) + replacement;
          if (stem.length >= 3) {
            base = stem;
            break;
          }
        }
      }
      if (base.length > 4 && base.at(-1) === base.at(-2) && !/[aeiou]/.test(base.at(-1))) {
        base = base.slice(0, -1);
      }
      return base;
    }

    lookup(word) {
      const lower = String(word || '').toLowerCase();
      const found = this.entries.get(lower);
      if (found) return found;
      return {
        surface: String(word || ''),
        lower,
        id: null,
        category: 'Unknown',
        categoryName: 'Unknown-Vocab',
        pos: this._guessPOS(lower),
        possiblePOS: [this._guessPOS(lower)],
        lemma: lower,
        root: this.deriveRoot(lower),
        semanticClass: 'unknown',
        contentWord: true,
        grammar: {},
        associations: [],
        parent: null,
        active: true,
        originalTokenId: null,
        guessed: true
      };
    }

    inspectWord(word) {
      const entry = this.lookup(word);
      return {
        ...entry,
        rootFamilyMembers: entry.root
          ? Array.from(this.rootMembers.get(entry.root) || []).slice(0, 64)
          : []
      };
    }

    _guessPOS(lower) {
      if (!lower) return 'Other';
      if (/^[.!?,;:()\-]+$/.test(lower)) return 'Punct';
      if (/^\d+(?:\.\d+)?$/.test(lower)) return 'Num';
      if (QUESTION_WORDS.has(lower)) return 'Pronoun';
      if (YES_NO_OPENERS.has(lower)) return lower === 'can' || lower === 'could' || lower === 'will' || lower === 'would' || lower === 'should' || lower === 'may' || lower === 'might' || lower === 'must' || lower === 'shall' ? 'Modal' : 'Aux';
      if (lower.endsWith('ly')) return 'Adv';
      if (/(ing|ed|ize|ise|ate|fy)$/.test(lower)) return 'Verb';
      if (/(ous|ful|less|able|ible|ive|al|ic|ish|ary)$/.test(lower)) return 'Adj';
      return 'Noun';
    }

    _expandContractions(text) {
      let source = String(text || '').replace(/\u2019/g, "'");

      // Expand grammatical contractions before lexical lookup. This keeps the
      // parser from treating forms such as "what's" or "doesn't" as one
      // unknown content token. The expansion is language structure only, not
      // subject knowledge, so every topic gets the same behavior.
      const exact = new Map([
        ["can't", 'can not'], ["cannot", 'can not'], ["won't", 'will not'],
        ["don't", 'do not'], ["doesn't", 'does not'], ["didn't", 'did not'],
        ["isn't", 'is not'], ["aren't", 'are not'], ["wasn't", 'was not'], ["weren't", 'were not'],
        ["hasn't", 'has not'], ["haven't", 'have not'], ["hadn't", 'had not'],
        ["couldn't", 'could not'], ["wouldn't", 'would not'], ["shouldn't", 'should not'],
        ["mustn't", 'must not'], ["mightn't", 'might not'], ["needn't", 'need not'],
        ["i'm", 'i am'], ["you're", 'you are'], ["we're", 'we are'], ["they're", 'they are'],
        ["i've", 'i have'], ["you've", 'you have'], ["we've", 'we have'], ["they've", 'they have'],
        ["i'll", 'i will'], ["you'll", 'you will'], ["we'll", 'we will'], ["they'll", 'they will'],
        ["he'll", 'he will'], ["she'll", 'she will'], ["it'll", 'it will'], ["that'll", 'that will']
      ]);

      source = source.replace(/\b[A-Za-z]+(?:'[A-Za-z]+)\b/g, token => {
        const lower = token.toLowerCase();
        const expanded = exact.get(lower);
        if (!expanded) return token;
        // Preserve only the initial capitalization. Token identity remains
        // case-insensitive elsewhere in Processor.js.
        return /^[A-Z]/.test(token) ? expanded.charAt(0).toUpperCase() + expanded.slice(1) : expanded;
      });

      // WH + 's is overwhelmingly a support-verb contraction in an
      // interrogative position. Splitting it restores the existing generic
      // definition/copular route: "What's an X?" -> "What is an X?".
      // This is deliberately topic-agnostic and does not inject an answer.
      source = source.replace(/\b(what|who|where|when|why|how)'s\b/gi, (match, wh) => {
        const expansion = `${wh} is`;
        return /^[A-Z]/.test(match) ? expansion.charAt(0).toUpperCase() + expansion.slice(1) : expansion;
      });

      return source;
    }

    tokenize(text, maxTokens = this.options.maxPromptTokens) {
      const source = this._expandContractions(text);
      const matches = source.match(/[A-Za-z0-9_]+(?:['’\-][A-Za-z0-9_]+)*|[.!?,;:()~\-]/g) || [];
      const out = [];
      const n = Math.min(matches.length, Math.max(1, maxTokens | 0));
      for (let i = 0; i < n; i++) {
        const surface = matches[i];
        const entry = this.lookup(surface);
        out.push({
          index: i,
          surface,
          lower: surface.toLowerCase(),
          pos: entry.pos,
          possiblePOS: entry.possiblePOS || [entry.pos],
          root: entry.root,
          semanticClass: entry.semanticClass,
          contentWord: entry.contentWord,
          entry
        });
      }
      return out;
    }

    findVerbForm(entryOrWord, mode = 'base') {
      const entry = typeof entryOrWord === 'string' ? this.lookup(entryOrWord) : entryOrWord;
      if (!entry) return null;
      const root = entry.root || this.deriveRoot(entry.lower);
      const members = Array.from(this.rootMembers.get(root) || []);
      const candidates = members
        .map(lower => this.entries.get(lower))
        .filter(x => x?.active !== false && (x.pos === 'Verb' || (x.possiblePOS || []).includes('Verb')));
      const choose = predicate => candidates.find(x => predicate(x.lower));
      if (mode === 'present3sg') {
        return choose(w => /(?:s|es)$/.test(w) && !/(ss|us)$/.test(w)) || entry;
      }
      if (mode === 'past') return choose(w => /ed$/.test(w)) || entry;
      if (mode === 'progressive') return choose(w => /ing$/.test(w)) || entry;
      return choose(w => w === root || w === entry.lemma) || entry;
    }

    buildResponseScaffold(analysis) {
      const words = analysis?.words || [];
      if (!words.length) return null;
      const first = words[0];
      if (analysis.intent !== 'boolean' || !['Aux','Modal'].includes(first.pos)) return null;

      // Generic English do-support / modal inversion recovery based only on POS
      // roles. This does not assert a fact or consult a subject-definition table.
      let predicateIndex = -1;
      for (let i = 1; i < words.length; i++) {
        if (words[i].pos === 'Verb') { predicateIndex = i; break; }
      }
      if (predicateIndex <= 1) return null;

      const subject = words.slice(1, predicateIndex);
      const predicate = words[predicateIndex];
      const object = words.slice(predicateIndex + 1);
      let predicateEntry = predicate.entry;
      const opener = first.lower;
      const output = subject.slice();

      if (first.pos === 'Modal') {
        output.push(first, predicate);
      } else if (opener === 'does') {
        predicateEntry = this.findVerbForm(predicate.entry, 'present3sg') || predicate.entry;
        output.push({ ...predicate, surface: predicateEntry.surface, lower: predicateEntry.lower, pos: 'Verb', entry: predicateEntry });
      } else if (opener === 'did') {
        predicateEntry = this.findVerbForm(predicate.entry, 'past') || predicate.entry;
        output.push({ ...predicate, surface: predicateEntry.surface, lower: predicateEntry.lower, pos: 'Verb', entry: predicateEntry });
      } else if (opener === 'do') {
        output.push(predicate);
      } else {
        // Copular/other auxiliaries are kept between the recovered subject and
        // the remaining predicate material when no lexical main-verb rewrite applies.
        output.push(first, predicate);
      }
      output.push(...object);

      return {
        source: 'pos-inversion-recovery',
        words: output.map(t => t.lower),
        surfaces: output.map(t => t.surface),
        pos: output.map(t => t.pos),
        subject: subject.map(t => t.lower),
        predicate: predicateEntry?.lower || predicate.lower,
        object: object.map(t => t.lower),
        opener
      };
    }

    analyzePrompt(text) {
      const tokens = this.tokenize(text, this.options.maxPromptTokens);
      const words = tokens.filter(t => t.pos !== 'Punct');
      const lowers = words.map(t => t.lower);
      const first = lowers[0] || '';
      const rawLower = String(text || '').toLowerCase();
      const isQuestion = /\?\s*$/.test(String(text || '')) || QUESTION_WORDS.has(first) || YES_NO_OPENERS.has(first);
      const operator = QUESTION_WORDS.has(first) ? first : (YES_NO_OPENERS.has(first) ? first : '');

      let intent = 'statement';
      if (lowers.includes('define') || lowers.includes('definition') || lowers.includes('meaning') || lowers.includes('means') || (first === 'what' && (lowers.includes('is') || lowers.includes('are')))) {
        intent = 'definition';
      } else if ((first === 'what' && (lowers[1] === 'does' || lowers[1] === 'do')) || first === 'how' || rawLower.includes('how does') || rawLower.includes('how do ')) {
        intent = 'mechanism';
      } else if (first === 'why' || lowers.includes('cause') || lowers.includes('causes')) {
        intent = 'causal';
      } else if (lowers.some(w => ['list','examples','example','ways','types','kinds'].includes(w))) {
        intent = 'list';
      } else if (YES_NO_OPENERS.has(first)) {
        intent = 'boolean';
      } else if (isQuestion) {
        intent = 'question';
      }

      const content = words.filter(t => t.contentWord && !STOP_WORDS.has(t.lower));
      const roots = [];
      const seenRoots = new Set();
      for (const t of content) {
        if (t.root && !seenRoots.has(t.root)) {
          seenRoots.add(t.root);
          roots.push(t.root);
        }
      }

      // Candidate subject phrases are contiguous 1-4 word n-grams stripped of
      // leading question/support words. Decoder.js uses these for exact concept lookup.
      const meaningful = words.filter(t => !STOP_WORDS.has(t.lower));
      const phrases = [];
      const phraseSeen = new Set();
      for (let width = Math.min(4, meaningful.length); width >= 1; width--) {
        for (let i = 0; i + width <= meaningful.length; i++) {
          const phrase = meaningful.slice(i, i + width).map(t => t.lower).join(' ');
          if (!phrase || phraseSeen.has(phrase)) continue;
          phraseSeen.add(phrase);
          phrases.push(phrase);
        }
      }

      const analysis = {
        text: String(text || ''),
        tokens,
        words,
        lowers,
        content,
        contentWords: content.map(t => t.lower),
        roots,
        phrases,
        isQuestion,
        operator,
        intent,
        hasNegation: lowers.includes('not') || lowers.includes("n't") || lowers.includes('never'),
        requestedSlot: this._requestedSlot(intent, operator, lowers),
        lexicalCoverage: words.length ? words.filter(t => !t.entry?.guessed).length / words.length : 0,
        tokenCount: tokens.length,
        wordCount: words.length
      };
      analysis.responseScaffold = this.buildResponseScaffold(analysis);
      this.lastAnalysis = analysis;
      return analysis;
    }

    _requestedSlot(intent, operator, lowers) {
      if (intent === 'definition') return 'identity';
      if (intent === 'mechanism') return 'mechanism';
      if (intent === 'causal') return 'cause';
      if (intent === 'boolean') return 'truth';
      if (operator === 'when') return 'time';
      if (operator === 'where') return 'location';
      if (operator === 'who' || operator === 'whom' || operator === 'whose') return 'person';
      if (operator === 'which') return 'selection';
      if (lowers.includes('many') || lowers.includes('much')) return 'quantity';
      return 'content';
    }

    createClauseState() {
      return {
        words: [],
        roots: new Set(),
        recentRoots: [],
        previousWord: '',
        previousPOS: 'START2',
        lastWord: '',
        lastPOS: 'START',
        wordHistory: [],
        posHistory: [],
        wordCount: 0,
        contentCount: 0,
        hasSubject: false,
        hasPredicate: false,
        openDeterminer: false,
        openDeterminerEntry: null,
        openDeterminerFirstModifierSeen: false,
        openPreposition: false,
        openModal: false,
        clauseCount: 1,
        punctuationHistory: [],
        lastPunctuation: '',
        wordsSinceBoundary: 0,
        parenthesisDepth: 0,
        continuationRequired: false
      };
    }

    _cloneClauseState(state) {
      return {
        ...state,
        words: state.words.slice(),
        wordHistory: state.wordHistory.slice(),
        posHistory: state.posHistory.slice(),
        punctuationHistory: Array.from(state.punctuationHistory || []),
        roots: new Set(state.roots),
        recentRoots: state.recentRoots.slice()
      };
    }

    _punctuationProfile(entryOrWord) {
      const entry = typeof entryOrWord === 'string' ? this.lookup(entryOrWord) : entryOrWord;
      const surface = String(entry?.surface || entryOrWord || '');
      const external = this.ruleStore?.punctuation?.(surface) || {};
      // Central punctuation data is allowed to refine/override lexical metadata.
      // Words.json still supplies the token identity; PunctuationRules.json owns
      // the generic placement/force policy.
      const grammar = { ...(entry?.grammar || {}), ...external };
      const fallbackClass = /[.!?]/.test(surface) ? 'terminal'
        : surface === ',' ? 'separator'
        : surface === ';' ? 'clause_separator'
        : surface === ':' ? 'introducer'
        : surface === '(' ? 'paired_opener'
        : surface === ')' ? 'paired_closer'
        : surface === '-' ? 'connector'
        : 'stylistic';
      return {
        surface,
        className: String(grammar.punctuation_class || fallbackClass),
        sentenceForce: String(grammar.sentence_force || ''),
        separatorRole: String(grammar.separator_role || ''),
        pair: String(grammar.pair || ''),
        canStartClause: grammar.can_start_clause === true,
        canEndClause: grammar.can_end_clause === true,
        closesSentence: grammar.closes_sentence === true || /[.!?]/.test(surface),
        requiresLeftClause: grammar.requires_left_clause === true,
        requiresRightClause: grammar.requires_right_clause === true,
        requiresLeftContent: grammar.requires_left_content === true,
        requiresRightContent: grammar.requires_right_content === true,
        requiresInterrogativeForm: grammar.requires_interrogative_form === true,
        generatedDefault: grammar.generated_default !== false
      };
    }

    _tokenPOS(entryOrWord) {
      const entry = typeof entryOrWord === 'string' ? this.lookup(entryOrWord) : entryOrWord;
      return normalizePOS(entry?.pos || 'Other');
    }

    _spanHasClause(words, start = 0, end = null) {
      const list = Array.from(words || []);
      const stop = end == null ? list.length : Math.min(list.length, Math.max(start, end));
      let subjectSeen = false;
      for (let i = Math.max(0, start | 0); i < stop; i++) {
        const entry = this.lookup(list[i]);
        const possible = new Set((entry?.possiblePOS || [entry?.pos]).map(normalizePOS));
        if (possible.has('Punct')) continue;
        const canSubject = Array.from(SUBJECT_POS).some(pos => possible.has(pos));
        const canPredicate = ['Verb','Aux','Modal'].some(pos => possible.has(pos));
        if (!subjectSeen && canSubject) {
          subjectSeen = true;
          continue;
        }
        if (subjectSeen && canPredicate) return true;
      }
      return false;
    }

    _looksInterrogativeWords(words) {
      const lexical = Array.from(words || []).filter(word => this._tokenPOS(word) !== 'Punct');
      if (!lexical.length) return false;
      const first = String(lexical[0] || '').toLowerCase();
      if (YES_NO_OPENERS.has(first)) return true;
      if (!QUESTION_WORDS.has(first)) return false;
      const second = this.lookup(lexical[1] || '');
      const secondPOS = new Set((second?.possiblePOS || [second?.pos]).map(normalizePOS));
      if (['Aux','Modal'].some(pos => secondPOS.has(pos))) return true;
      // Subject-WH questions such as "Who won?" or "What changed?" do not
      // require auxiliary inversion. Restrict this exception to WH forms that
      // can themselves occupy the subject slot, so embedded phrases such as
      // "what AI does" remain declarative when used as an answer.
      if (['who','what','which'].includes(first) && secondPOS.has('Verb')) return true;
      return false;
    }

    _commaInsertionPoints(words) {
      const list = Array.from(words || [], word => String(word || '')).filter(Boolean);
      const points = new Set();
      const externalComma = this.ruleStore?.commaRules?.() || {};
      const externalGrammar = this.ruleStore?.coordination?.() || {};
      const externalClause = this.ruleStore?.clauseRules?.() || {};
      const coordinators = new Set(
        Array.isArray(externalComma.coordinators) ? externalComma.coordinators :
        Array.isArray(externalGrammar.coordinators) ? externalGrammar.coordinators :
        ['and','but','or','nor','for','so','yet']
      );
      const subordinateOpeners = new Set(
        Array.isArray(externalComma.introductory_subordinate_openers) ? externalComma.introductory_subordinate_openers :
        Array.isArray(externalClause.subordinate_openers) ? externalClause.subordinate_openers :
        ['although','because','when','while','if','since','unless','after','before','once','though','whereas']
      );

      // Independent clause + coordinator + independent clause.
      for (let i = 1; i < list.length - 1; i++) {
        const lower = list[i].toLowerCase();
        if (!coordinators.has(lower)) continue;
        if (this._spanHasClause(list, 0, i) && this._spanHasClause(list, i + 1, list.length)) {
          if (list[i - 1] !== ',') points.add(i);
        }
      }

      // Introductory subordinate clause. Pick the earliest split where both the
      // introductory span and the remaining span are clause-complete. This is
      // syntax-driven and does not depend on the topic being discussed.
      if (subordinateOpeners.has(String(list[0] || '').toLowerCase())) {
        const candidates = [];
        for (let split = 3; split <= list.length - 2; split++) {
          if (!this._spanHasClause(list, 0, split) || !this._spanHasClause(list, split, list.length)) continue;
          const firstRight = this.lookup(list[split]);
          const rightPOS = new Set((firstRight?.possiblePOS || [firstRight?.pos]).map(normalizePOS));
          let boundaryStrength = 0;
          if (rightPOS.has('Pronoun')) boundaryStrength = 4;
          else if (rightPOS.has('ProperNoun')) boundaryStrength = 3;
          else if (rightPOS.has('Det')) boundaryStrength = 2;
          else if (rightPOS.has('Noun') || rightPOS.has('Num')) boundaryStrength = 1;
          if (boundaryStrength) candidates.push({ split, boundaryStrength });
        }
        candidates.sort((a,b) => b.boundaryStrength - a.boundaryStrength || a.split - b.split);
        const best = candidates[0];
        if (best && list[best.split - 1] !== ',') points.add(best.split);
      }
      return points;
    }

    repairGeneratedSurface(words, analysis = {}) {
      const lexicalRepair = this.repairGeneratedWords(words);
      const base = Array.from(lexicalRepair.words || [], word => String(word || '')).filter(Boolean);
      const repairs = Array.from(lexicalRepair.repairs || []);
      const insertions = this._commaInsertionPoints(base);
      const punctuated = [];
      for (let i = 0; i < base.length; i++) {
        if (insertions.has(i) && punctuated[punctuated.length - 1] !== ',') {
          punctuated.push(',');
          repairs.push({ index: i, from: '', to: ',', reason: 'COMMA_CLAUSE_BOUNDARY' });
        }
        punctuated.push(base[i]);
      }

      // Remove illegal punctuation stacks while preserving paired punctuation.
      const clean = [];
      let parenDepth = 0;
      for (let i = 0; i < punctuated.length; i++) {
        const token = punctuated[i];
        const pos = this._tokenPOS(token);
        if (pos !== 'Punct') {
          clean.push(token);
          continue;
        }
        const profile = this._punctuationProfile(token);
        const previous = clean[clean.length - 1] || '';
        const previousIsPunct = previous && this._tokenPOS(previous) === 'Punct';
        if (profile.className === 'paired_opener') {
          parenDepth++;
          clean.push(token);
          continue;
        }
        if (profile.className === 'paired_closer') {
          if (parenDepth <= 0 || previousIsPunct) {
            repairs.push({ index: i, from: token, to: '', reason: 'UNBALANCED_CLOSER_REMOVED' });
            continue;
          }
          parenDepth--;
          clean.push(token);
          continue;
        }
        if (previousIsPunct && !['paired_closer'].includes(profile.className)) {
          repairs.push({ index: i, from: token, to: '', reason: 'PUNCTUATION_STACK_REMOVED' });
          continue;
        }
        clean.push(token);
      }

      // Generated answers default to a declarative terminal. A question mark is
      // selected only when the generated sentence itself has interrogative form,
      // not merely because the user asked a question.
      const terminalIndex = clean.length - 1;
      const terminal = terminalIndex >= 0 && /[.!?]/.test(clean[terminalIndex]) ? clean[terminalIndex] : '';
      const interrogative = this._looksInterrogativeWords(clean);
      const terminalRules = this.ruleStore?.terminalRules?.() || {};
      const desiredTerminal = interrogative
        ? String(terminalRules.default_interrogative || '?')
        : String(terminalRules.default_declarative || '.');
      if (!terminal) {
        clean.push(desiredTerminal);
        repairs.push({ index: clean.length - 1, from: '', to: desiredTerminal, reason: interrogative ? 'QUESTION_MARK_REALIZATION' : 'PERIOD_REALIZATION' });
      } else if (terminal === '?' && !interrogative) {
        clean[terminalIndex] = '.';
        repairs.push({ index: terminalIndex, from: '?', to: '.', reason: 'QUESTION_MARK_FORM_MISMATCH' });
      } else if (terminal === '.' && interrogative) {
        clean[terminalIndex] = '?';
        repairs.push({ index: terminalIndex, from: '.', to: '?', reason: 'INTERROGATIVE_TERMINAL_REPAIR' });
      }

      let text = clean.join(' ')
        .replace(/\s+([.,!?;:])/g, '$1')
        .replace(/\(\s+/g, '(')
        .replace(/\s+\)/g, ')')
        .replace(/\s*-\s*/g, '-')
        .replace(/([,;:])([^\s)])/g, '$1 $2')
        .replace(/([.!?])([^\s)])/g, '$1 $2')
        .trim();
      return { words: clean, text, repairs };
    }

    _determinerProfile(entry) {
      const external = this.ruleStore?.determiner?.(entry?.lower || entry?.surface || '') || {};
      const grammar = entry?.grammar || {};
      return {
        className: String(external.class || grammar.determiner_class || ''),
        numberRequirement: String(external.number_requirement || grammar.number_requirement || 'any'),
        articleOnset: String(external.article_onset || grammar.article_onset || ''),
        mismatchFallback: String(external.mismatch_fallback || grammar.mismatch_fallback || ''),
        requiresCoordinationOrPlural: external.requires_coordination_or_plural === true
      };
    }

    _inferNominalNumber(entry) {
      if (!entry) return 'unknown';
      const grammar = entry.grammar || {};
      if (grammar.number === 'singular' || grammar.grammatical_number === 'singular') return 'singular';
      if (grammar.number === 'plural' || grammar.grammatical_number === 'plural') return 'plural';
      if (grammar.likely_plural === true) return 'plural';
      if (grammar.likely_plural === false && ['Noun','ProperNoun'].includes(entry.pos) &&
          (!entry.parent || String(entry.lower || '') === String(entry.lemma || ''))) return 'singular';

      const lower = String(entry.lower || '').toLowerCase();
      const pos = normalizePOS(entry.pos);
      if (pos === 'Num') {
        const numeric = Number(lower.replace(/,/g, ''));
        if (Number.isFinite(numeric)) return numeric === 1 ? 'singular' : 'plural';
        return 'unknown';
      }
      if (pos !== 'Noun') return 'unknown';

      // Conservative lexical morphology fallback. A noun is treated as plural
      // only when its spelling looks plural AND its root family contains a
      // plausible singular member. This avoids turning every final-s noun into
      // a hard plural assertion.
      if (/s$/.test(lower) && !/(ss|us|is)$/.test(lower)) {
        const lexicalRoot = String(entry.root || '');
        const derivedRoot = String(this.deriveRoot(lower, ''));
        const roots = Array.from(new Set([lexicalRoot, derivedRoot].filter(Boolean)));
        for (const root of roots) {
          const members = Array.from(this.rootMembers.get(root) || []);
          if (members.some(member => member !== lower && !/s$/.test(member))) return 'plural';
          const singular = this.entries.get(root);
          if (singular && normalizePOS(singular.pos) === 'Noun' && singular.lower !== lower) return 'plural';
        }
      }
      if (!/s$/.test(lower)) return 'singular';
      return 'unknown';
    }

    _onsetClass(wordOrEntry) {
      const entry = typeof wordOrEntry === 'string' ? this.lookup(wordOrEntry) : wordOrEntry;
      const explicit = String(entry?.grammar?.onset_class || '');
      if (explicit === 'vowel' || explicit === 'consonant') return explicit;
      const surface = String(entry?.surface || entry?.lower || wordOrEntry || '').trim().toLowerCase();
      if (!surface) return 'unknown';
      return /^[aeiou]/.test(surface) ? 'vowel' : 'consonant';
    }

    _findDeterminerVariant(className, onset = '') {
      const wantedClass = String(className || '');
      const wantedOnset = String(onset || '');
      if (!wantedClass) return null;
      for (const entry of this.entries.values()) {
        if (!entry || entry.active === false || normalizePOS(entry.pos) !== 'Det') continue;
        const profile = this._determinerProfile(entry);
        if (profile.className !== wantedClass) continue;
        if (wantedOnset && profile.articleOnset && profile.articleOnset !== wantedOnset) continue;
        if (wantedOnset && wantedClass === 'indefinite_article' && !profile.articleOnset) continue;
        return entry;
      }
      return null;
    }

    _determinerHead(words, determinerIndex) {
      const firstIndex = determinerIndex + 1;
      if (firstIndex >= words.length) return null;
      const firstEntry = this.lookup(words[firstIndex]);
      let head = null;
      for (let index = firstIndex; index < words.length; index++) {
        const entry = this.lookup(words[index]);
        const pos = normalizePOS(entry.pos);
        if (['Adj','Adv'].includes(pos)) continue;
        if (['Noun','ProperNoun','Num'].includes(pos)) {
          // English noun phrases can contain nominal modifiers (for example
          // "AI systems"). Keep walking through the contiguous nominal span
          // and use its final nominal as the agreement head.
          head = { firstIndex, firstEntry, headIndex: index, headEntry: entry };
          continue;
        }
        break;
      }
      return head;
    }

    repairGeneratedWords(words) {
      const out = Array.from(words || [], word => String(word || '')).filter(Boolean);
      const repairs = [];
      for (let i = 0; i < out.length; i++) {
        const detEntry = this.lookup(out[i]);
        if (normalizePOS(detEntry.pos) !== 'Det') continue;
        const profile = this._determinerProfile(detEntry);
        if (!profile.className) continue;
        const phrase = this._determinerHead(out, i);
        if (!phrase) continue;

        const onset = this._onsetClass(phrase.firstEntry);
        let replacement = null;
        let reason = '';

        // Indefinite articles are selected by the following pronounced word,
        // so adjective+noun phrases such as "an artificial system" work without
        // any subject-specific response rule.
        if (profile.className === 'indefinite_article' && onset !== 'unknown' && profile.articleOnset !== onset) {
          replacement = this._findDeterminerVariant('indefinite_article', onset);
          reason = 'ARTICLE_ONSET_AGREEMENT';
        }

        const headNumber = this._inferNominalNumber(phrase.headEntry);
        const numberMismatch = profile.numberRequirement !== 'any' &&
          headNumber !== 'unknown' && profile.numberRequirement !== headNumber;
        if (numberMismatch && profile.mismatchFallback) {
          if (profile.mismatchFallback === 'indefinite_article' && headNumber === 'singular') {
            replacement = this._findDeterminerVariant('indefinite_article', onset);
            reason = 'DETERMINER_NUMBER_REPAIR';
          } else {
            replacement = this._findDeterminerVariant(profile.mismatchFallback, onset);
            reason = 'DETERMINER_NUMBER_REPAIR';
          }
        }

        if (replacement && replacement.lower !== detEntry.lower) {
          repairs.push({ index: i, from: out[i], to: replacement.surface, reason });
          out[i] = replacement.surface;
        }
      }
      return { words: out, repairs };
    }

    evaluateCandidate(candidate, state, options = {}) {
      const entry = typeof candidate === 'string' ? this.lookup(candidate) : candidate;
      const word = String(entry?.surface || candidate || '');
      const lower = String(entry?.lower || word.toLowerCase());
      const pos = normalizePOS(options.activePOS || entry?.activePOS || entry?.pos || 'Other');
      const root = String(entry?.root || this.deriveRoot(lower));
      const reasons = [];
      let legal = true;
      let grammar = 0;

      if (!word) return { legal: false, grammar: -1, reasons: ['EMPTY_TOKEN'], entry };

      if (pos === 'Punct') {
        const profile = this._punctuationProfile(entry);
        const previousWasPunct = state.lastPOS === 'Punct';
        const openDependency = state.openDeterminer || state.openPreposition || state.openModal;

        if (profile.className === 'terminal') {
          if (state.wordCount < 2 || !state.hasSubject || !state.hasPredicate) {
            legal = false; reasons.push('INCOMPLETE_SENTENCE_TERMINAL');
          }
          if (openDependency) { legal = false; reasons.push('OPEN_DEPENDENCY'); }
          if ((state.parenthesisDepth || 0) > 0) { legal = false; reasons.push('UNBALANCED_PARENTHESIS'); }
          if (previousWasPunct) { legal = false; reasons.push('PUNCTUATION_STACK'); }
          if (profile.requiresInterrogativeForm && options.interrogativeForm === false) {
            legal = false; reasons.push('QUESTION_MARK_WITHOUT_INTERROGATIVE_FORM');
          }
          grammar += legal ? 1 : -1;
        } else if (profile.className === 'separator') {
          if ((state.wordsSinceBoundary || state.wordCount) < 2 || previousWasPunct) {
            legal = false; reasons.push('ILLEGAL_COMMA_POSITION');
          }
          if (openDependency) { legal = false; reasons.push('OPEN_DEPENDENCY'); }
          grammar += legal ? 0.9 : -1;
        } else if (profile.className === 'clause_separator') {
          if (!state.hasSubject || !state.hasPredicate || previousWasPunct || openDependency) {
            legal = false; reasons.push('SEMICOLON_NEEDS_COMPLETE_LEFT_CLAUSE');
          }
          grammar += legal ? 1 : -1;
        } else if (profile.className === 'introducer') {
          if (state.wordCount < 2 || previousWasPunct || openDependency) {
            legal = false; reasons.push('COLON_NEEDS_LEFT_CONTENT');
          }
          grammar += legal ? 0.85 : -1;
        } else if (profile.className === 'paired_opener') {
          if (previousWasPunct && state.lastPunctuation !== ',') {
            legal = false; reasons.push('ILLEGAL_PARENTHESIS_OPENER');
          }
          grammar += legal ? 0.75 : -1;
        } else if (profile.className === 'paired_closer') {
          if ((state.parenthesisDepth || 0) <= 0 || previousWasPunct) {
            legal = false; reasons.push('UNMATCHED_PARENTHESIS_CLOSER');
          }
          grammar += legal ? 0.8 : -1;
        } else if (profile.className === 'connector') {
          if (state.wordCount < 1 || previousWasPunct) {
            legal = false; reasons.push('ILLEGAL_CONNECTOR_POSITION');
          }
          grammar += legal ? 0.65 : -1;
        } else if (profile.generatedDefault === false && options.allowStylisticPunctuation !== true) {
          legal = false;
          reasons.push('STYLISTIC_PUNCTUATION_DISABLED');
          grammar = -1;
        }
        return { legal, grammar: Math.max(-1, Math.min(1, grammar)), reasons, entry, activePOS: 'Punct' };
      }

      if (state.wordCount === 0 && ['Conj','Prep','Punct'].includes(pos)) {
        legal = false;
        reasons.push('ILLEGAL_CLAUSE_START');
      }

      if (state.lastWord === lower && state.wordCount > 0) {
        legal = false;
        reasons.push('EXACT_REPEAT');
      }

      if (this.options.strictRootFamilies && CONTENT_POS.has(pos) && root) {
        const rootSeen = state.roots.has(root);
        if (rootSeen && !options.allowRootRepeat) {
          legal = false;
          reasons.push('ROOT_FAMILY_COLLISION');
        }
      }

      if (state.openDeterminer && !['Adj','Noun','ProperNoun','Num'].includes(pos)) {
        legal = false;
        reasons.push('DETERMINER_NEEDS_NOMINAL');
      }

      if (options.strictDeterminerAgreement === true && state.openDeterminer && state.openDeterminerEntry) {
        const detProfile = this._determinerProfile(state.openDeterminerEntry);
        if (!state.openDeterminerFirstModifierSeen && ['Adj','Noun','ProperNoun','Num'].includes(pos)) {
          const onset = this._onsetClass(entry);
          if (detProfile.className === 'indefinite_article' && detProfile.articleOnset &&
              onset !== 'unknown' && detProfile.articleOnset !== onset) {
            legal = false;
            reasons.push('ARTICLE_ONSET_MISMATCH');
          }
        }
        if (['Noun','ProperNoun','Num'].includes(pos) && detProfile.numberRequirement !== 'any') {
          const nominalNumber = this._inferNominalNumber(entry);
          if (nominalNumber !== 'unknown' && nominalNumber !== detProfile.numberRequirement) {
            legal = false;
            reasons.push('DETERMINER_NUMBER_MISMATCH');
          }
        }
      }

      if (state.openPreposition && !NOMINAL_POS.has(pos)) {
        legal = false;
        reasons.push('PREPOSITION_NEEDS_OBJECT');
      }

      if (state.openModal && !['Verb','Adv'].includes(pos)) {
        legal = false;
        reasons.push('MODAL_NEEDS_VERB');
      }

      // Processor is a legality arbiter, not a language model. POS preference
      // is learned by RSL/Correlation rather than awarded here.
      grammar = legal ? 1 : -1;

      if (pos === 'Det' && state.lastPOS === 'Det') { legal = false; reasons.push('DETERMINER_STACK'); }
      if (pos === 'Prep' && state.lastPOS === 'Prep') { legal = false; reasons.push('PREPOSITION_STACK'); }
      if (pos === 'Conj' && state.lastPOS === 'Conj') { legal = false; reasons.push('CONJUNCTION_STACK'); }

      return {
        legal,
        grammar: Math.max(-1, Math.min(1, grammar)),
        reasons,
        entry,
        root,
        activePOS: pos
      };
    }

    acceptCandidate(candidate, state, options = {}) {
      const assessment = this.evaluateCandidate(candidate, state, options);
      if (!assessment.legal) return assessment;
      const entry = assessment.entry;
      const pos = assessment.activePOS || normalizePOS(entry.pos);
      const lower = entry.lower;
      const root = assessment.root;

      const wasDeterminerOpen = state.openDeterminer;
      const wasPrepositionOpen = state.openPreposition;
      const wasModalOpen = state.openModal;

      if (pos === 'Punct') {
        const profile = this._punctuationProfile(entry);
        state.words.push(entry.surface);
        state.wordHistory.push(lower);
        state.posHistory.push('Punct');
        state.punctuationHistory.push(entry.surface);
        state.previousWord = state.lastWord;
        state.previousPOS = state.lastPOS;
        state.lastWord = lower;
        state.lastPOS = 'Punct';
        state.lastPunctuation = entry.surface;
        if (profile.className === 'paired_opener') state.parenthesisDepth = (state.parenthesisDepth || 0) + 1;
        if (profile.className === 'paired_closer') state.parenthesisDepth = Math.max(0, (state.parenthesisDepth || 0) - 1);
        if (profile.className === 'separator') state.continuationRequired = true;
        if (profile.className === 'introducer') state.continuationRequired = true;
        if (profile.className === 'clause_separator') {
          state.clauseCount++;
          state.hasSubject = false;
          state.hasPredicate = false;
          state.wordsSinceBoundary = 0;
          state.continuationRequired = true;
        }
        if (profile.className === 'terminal') state.continuationRequired = false;
        return assessment;
      }

      state.words.push(entry.surface);
      state.wordHistory.push(lower);
      state.posHistory.push(pos);
      state.previousWord = state.lastWord;
      state.previousPOS = state.lastPOS;
      state.lastWord = lower;
      state.lastPOS = pos;
      state.wordCount++;
      state.wordsSinceBoundary = (state.wordsSinceBoundary || 0) + 1;
      state.lastPunctuation = '';
      if (state.continuationRequired) state.continuationRequired = false;
      if (entry.contentWord || CONTENT_POS.has(pos)) state.contentCount++;

      if (CONTENT_POS.has(pos) && root) {
        state.roots.add(root);
        state.recentRoots.push(root);
        if (state.recentRoots.length > this.options.rootRepeatWindow) state.recentRoots.shift();
      }

      if (!state.hasSubject && SUBJECT_POS.has(pos)) state.hasSubject = true;
      if (['Verb','Aux'].includes(pos) && state.hasSubject) state.hasPredicate = true;

      // Open dependencies are hard structural state, not language-model scores.
      // Adjectives may extend a determiner phrase; determiners/adjectives may
      // extend a prepositional object; adverbs may intervene before a modal's verb.
      if (pos === 'Det') {
        state.openDeterminer = true;
        state.openDeterminerEntry = entry;
        state.openDeterminerFirstModifierSeen = false;
      } else if (wasDeterminerOpen && pos === 'Adj') {
        state.openDeterminer = true;
        state.openDeterminerEntry = state.openDeterminerEntry || null;
        state.openDeterminerFirstModifierSeen = true;
      } else {
        state.openDeterminer = false;
        state.openDeterminerEntry = null;
        state.openDeterminerFirstModifierSeen = false;
      }

      if (pos === 'Prep') state.openPreposition = true;
      else if (wasPrepositionOpen && ['Det','Adj'].includes(pos)) state.openPreposition = true;
      else state.openPreposition = false;

      if (pos === 'Modal') state.openModal = true;
      else if (wasModalOpen && pos === 'Adv') state.openModal = true;
      else state.openModal = false;

      return assessment;
    }

    _auditPunctuation(tokens) {
      const issues = [];
      const list = Array.from(tokens || []);
      let sentenceStart = 0;
      let parenDepth = 0;

      const segmentWords = (a, b) => list.slice(a, b).filter(t => t.pos !== 'Punct').map(t => t.surface);
      const nextLexical = index => {
        for (let i = index + 1; i < list.length; i++) if (list[i].pos !== 'Punct') return list[i];
        return null;
      };
      const prevLexical = index => {
        for (let i = index - 1; i >= 0; i--) if (list[i].pos !== 'Punct') return list[i];
        return null;
      };

      for (let i = 0; i < list.length; i++) {
        const token = list[i];
        if (token.pos !== 'Punct') continue;
        const p = this._punctuationProfile(token.entry || token.surface);
        const prev = list[i - 1] || null;
        const next = list[i + 1] || null;
        if (p.className === 'paired_opener') { parenDepth++; continue; }
        if (p.className === 'paired_closer') {
          if (parenDepth <= 0) issues.push({ index: i, word: token.surface, reasons: ['UNMATCHED_PARENTHESIS_CLOSER'] });
          else parenDepth--;
          continue;
        }
        if (p.className === 'separator') {
          if (!prevLexical(i) || !nextLexical(i) || (prev?.pos === 'Punct' && prev.surface !== ')') || next?.pos === 'Punct') {
            issues.push({ index: i, word: token.surface, reasons: ['ILLEGAL_COMMA_POSITION'] });
          }
        }
        if (p.className === 'clause_separator') {
          const left = segmentWords(sentenceStart, i);
          let rightStop = list.length;
          for (let j = i + 1; j < list.length; j++) if (/[.!?]/.test(list[j].surface)) { rightStop = j; break; }
          const right = segmentWords(i + 1, rightStop);
          if (!this._spanHasClause(left) || !this._spanHasClause(right)) {
            issues.push({ index: i, word: token.surface, reasons: ['SEMICOLON_REQUIRES_TWO_CLAUSES'] });
          }
        }
        if (p.className === 'introducer') {
          if (!prevLexical(i) || !nextLexical(i) || next?.pos === 'Punct') {
            issues.push({ index: i, word: token.surface, reasons: ['COLON_REQUIRES_CONTENT_ON_BOTH_SIDES'] });
          }
        }
        if (p.className === 'terminal') {
          const sentenceWords = segmentWords(sentenceStart, i);
          const interrogative = this._looksInterrogativeWords(sentenceWords);
          if (token.surface === '.' && interrogative) {
            issues.push({ index: i, word: token.surface, reasons: ['INTERROGATIVE_NEEDS_QUESTION_MARK'] });
          }
          sentenceStart = i + 1;
        }
      }
      if (parenDepth !== 0) issues.push({ index: list.length - 1, word: '(', reasons: ['UNBALANCED_PARENTHESIS'] });
      const last = list[list.length - 1];
      if (list.length && !(last?.pos === 'Punct' && /[.!?]/.test(last.surface))) {
        issues.push({ index: list.length - 1, word: last?.surface || '', reasons: ['MISSING_TERMINAL_PUNCTUATION'] });
      }
      return issues;
    }

    auditResponse(text, options = {}) {
      const tokens = this.tokenize(text, this.options.maxResponseTokens);
      const punctuationIssues = this._auditPunctuation(tokens);
      let state = this.createClauseState();
      let illegal = punctuationIssues.length;
      let rootCollisions = 0;
      let exactRepeats = 0;
      let grammarSum = 0;
      let scored = 0;
      const issues = punctuationIssues.slice();
      let sentenceStartIndex = 0;

      for (const token of tokens) {
        const posResolution = token.pos === 'Punct'
          ? { pos: 'Punct', certainty: 1 }
          : (this.rsl?.resolvePOS?.(token.entry, state) || { pos: token.pos, certainty: 0.5 });
        const activeEntry = token.pos === 'Punct'
          ? token.entry
          : this.materializePOS(token.entry, posResolution.pos);
        if ((activeEntry?.pos || token.pos) === 'Punct') {
          const currentSentenceWords = tokens.slice(sentenceStartIndex, token.index)
            .filter(t => t.pos !== 'Punct').map(t => t.surface);
          const punctuationAssessment = this.evaluateCandidate(activeEntry, state, {
            allowRootRepeat: true,
            activePOS: 'Punct',
            interrogativeForm: this._looksInterrogativeWords(currentSentenceWords),
            allowStylisticPunctuation: options.allowStylisticPunctuation === true
          });
          if (!punctuationAssessment.legal) {
            illegal++;
            issues.push({ index: token.index, word: token.surface, reasons: punctuationAssessment.reasons });
          } else {
            this.acceptCandidate(activeEntry, state, {
              allowRootRepeat: true,
              activePOS: 'Punct',
              interrogativeForm: this._looksInterrogativeWords(currentSentenceWords),
              allowStylisticPunctuation: options.allowStylisticPunctuation === true
            });
          }
          if (/[.!?]/.test(token.surface)) {
            state = this.createClauseState();
            sentenceStartIndex = token.index + 1;
          }
          continue;
        }
        const assessment = this.evaluateCandidate(activeEntry, state, {
          allowRootRepeat: options.allowRootRepeat === true,
          activePOS: posResolution.pos,
          strictDeterminerAgreement: true
        });
        grammarSum += assessment.grammar;
        scored++;
        if (!assessment.legal) {
          illegal++;
          if (assessment.reasons.includes('ROOT_FAMILY_COLLISION')) rootCollisions++;
          if (assessment.reasons.includes('EXACT_REPEAT')) exactRepeats++;
          issues.push({ index: token.index, word: token.surface, root: token.root, reasons: assessment.reasons });
          // Continue state tracking even for an audited rejection, except exact duplicate.
          if (!assessment.reasons.includes('EXACT_REPEAT')) {
            this.acceptCandidate(activeEntry, state, { allowRootRepeat: true, activePOS: posResolution.pos, strictDeterminerAgreement: true });
          }
        } else {
          this.acceptCandidate(activeEntry, state, { allowRootRepeat: true, activePOS: posResolution.pos, strictDeterminerAgreement: true });
        }
      }

      const legalRatio = tokens.length ? Math.max(0, 1 - illegal / tokens.length) : 0;
      const grammarScore = Math.max(0, Math.min(1, 0.5 + (scored ? grammarSum / scored : 0) * 0.5));
      const result = {
        tokenCount: tokens.length,
        illegal,
        rootCollisions,
        exactRepeats,
        legalRatio,
        grammarScore,
        issues: issues.slice(0, 32)
      };
      this.lastAudit = result;
      return result;
    }

    // For dynamically assembled fallback output, enforce the hard root-family
    // policy. Curated factual descriptions are audited but are not destructively
    // rewritten, because removing a necessary word from trusted source text can
    // create a worse grammatical claim than the repetition itself.
    filterGeneratedWords(words) {
      const state = this.createClauseState();
      const out = [];
      const rejected = [];
      for (const raw of words || []) {
        const entry = this.lookup(raw);
        if (entry.pos === 'Punct' && /[.!?]/.test(entry.surface)) {
          if (out.length) out.push(entry.surface);
          Object.assign(state, this.createClauseState());
          continue;
        }
        const assessment = this.evaluateCandidate(entry, state);
        if (!assessment.legal) {
          rejected.push({ word: raw, root: entry.root, reasons: assessment.reasons });
          continue;
        }
        this.acceptCandidate(entry, state);
        out.push(entry.surface);
      }
      return { words: out, rejected };
    }



    isKnownWord(word) {
      const key = String(word || '').toLowerCase();
      const entry = this.entries.get(key);
      return Boolean(entry && entry.active !== false);
    }

    resolveSayableWord(word, options = {}) {
      const raw = String(word || '');
      const key = raw.toLowerCase();
      const exact = this.entries.get(key);
      if (exact && exact.active !== false) return exact;

      if (options.allowRootFallback !== false) {
        const root = this.deriveRoot(key);
        const members = root ? this.rootMembers.get(root) : null;
        if (members) {
          for (const member of members) {
            const candidate = this.entries.get(member);
            if (candidate?.active !== false) return candidate;
          }
        }
      }
      return null;
    }

    _buildAssociationIndex() {
      this.associationIndex.clear();
      for (const [lower, base] of this.entries) {
        if (!base || base.active === false || !Array.isArray(base.associations) || !base.associations.length) continue;
        const resolved = [];
        for (const association of base.associations) {
          const target = typeof association === 'string'
            ? association
            : (association?.Word || association?.word || association?.Token || association?.token || '');
          if (!target) continue;
          const entry = this.resolveSayableWord(target, { allowRootFallback: true });
          if (!entry || entry.active === false) continue;
          const weight = typeof association === 'object'
            ? Number(association.Weight ?? association.weight ?? 0.45)
            : 0.45;
          const source = typeof association === 'object'
            ? (association.Source || association.source || 'association')
            : 'association';
          resolved.push({ entry, weight: Math.max(0, Math.min(1.5, Number(weight) || 0)), source: String(source || 'association') });
        }
        if (resolved.length) this.associationIndex.set(lower, resolved);
      }
    }

    associationScore(fromWord, toWord, options = {}) {
      const from = String(fromWord || '').toLowerCase();
      const to = String(toWord || '').toLowerCase();
      if (!from || !to) return 0;
      const direct = (this.associationIndex.get(from) || []).find(row => row.entry?.lower === to);
      let score = direct ? Math.max(0, Math.min(1, Number(direct.weight) || 0)) : 0;
      if (options.bidirectional !== false) {
        const reverse = (this.associationIndex.get(to) || []).find(row => row.entry?.lower === from);
        if (reverse) score = Math.max(score, Math.max(0, Math.min(1, Number(reverse.weight) || 0)) * 0.92);
      }
      return score;
    }

    associationCandidates(seedWords, limit = 64, options = {}) {
      const out = new Map();
      const seeds = Array.isArray(seedWords) ? seedWords : [seedWords];
      const add = (entry, weight, source, seed) => {
        if (!entry || entry.active === false || !entry.lower) return;
        const existing = out.get(entry.lower);
        const row = {
          entry,
          weight: Math.max(0, Math.min(1.5, Number(weight) || 0)),
          source: String(source || 'association'),
          seed: String(seed || '')
        };
        if (!existing || row.weight > existing.weight) out.set(entry.lower, row);
      };

      for (const seed of seeds) {
        const base = this.resolveSayableWord(seed, { allowRootFallback: true });
        if (!base) continue;
        if (options.includeSeeds !== false) add(base, 0.72, 'seed', seed);
        const indexed = this.associationIndex.get(base.lower) || [];
        for (const row of indexed) add(row.entry, row.weight, row.source, seed);
      }
      return Array.from(out.values())
        .sort((a,b) => b.weight - a.weight)
        .slice(0, Math.max(1, limit | 0));
    }

    realizePhrase(text, options = {}) {
      const maxWords = Math.max(1, Math.min(256, options.maxWords | 0 || 64));
      const allowRootFallback = options.allowRootFallback !== false;
      const allowUnknown = options.allowUnknown === true;
      const tokens = this.tokenize(String(text || ''), Math.max(maxWords * 3, 32));
      const out = [];
      const rejected = [];
      let wordCount = 0;

      for (const token of tokens) {
        if (token.pos === 'Punct') {
          if (/^[.,!?;:]$/.test(token.surface) && out.length) out.push(token.surface);
          continue;
        }
        if (wordCount >= maxWords) break;
        const sayable = this.resolveSayableWord(token.surface, { allowRootFallback });
        if (sayable) {
          // Preserve the requested surface form when that exact spelling exists.
          // This avoids parent entries such as "Whose" forcing capitalization
          // into the middle of a sentence while still keeping Words.json in control.
          const exactKnown = this.entries.has(String(token.surface || '').toLowerCase());
          out.push(exactKnown ? token.surface : sayable.surface);
          wordCount++;
        } else if (allowUnknown) {
          out.push(token.surface);
          wordCount++;
        } else {
          rejected.push(token.surface);
        }
      }

      let textOut = '';
      for (const piece of out) {
        if (/^[.,!?;:]$/.test(piece)) {
          textOut = textOut.trimEnd() + piece + ' ';
        } else {
          textOut += piece + ' ';
        }
      }
      textOut = textOut.trim();
      return {
        text: textOut,
        words: out.filter(x => !/^[.,!?;:]$/.test(x)),
        rejected,
        requestedWords: tokens.filter(t => t.pos !== 'Punct').length,
        realizedWords: wordCount,
        lexicalCoverage: tokens.length ? wordCount / Math.max(1, tokens.filter(t => t.pos !== 'Punct').length) : 0
      };
    }

    status() {
      return {
        ready: this.ready,
        schemaVersion: this.schemaVersion,
        loadedWords: this.loadedWords,
        loadedParents: this.loadedParents,
        rootFamilies: this.rootMembers.size,
        ambiguousPOSWords: this.ambiguousPOSWords,
        associationSourcesIndexed: this.associationIndex.size,
        dynamicPOSCandidates: true,
        posRelationTypes: Object.keys(this.posTypeRelations || {}).length,
        posTrigramSeeds: this.posTypeTrigrams.length,
        categories: Object.fromEntries(this.categoryCounts),
        loadMs: this.lastLoadMs,
        externalRuleData: this.ruleStore?.status?.() || null,
        strictRootFamilies: this.options.strictRootFamilies
      };
    }
  }

  globalThis.VilotProcessor = VilotProcessor;
})();

/**
 * Processor production research extension.
 *
 * Executable, data-driven numerical/statistical/graph utilities used by
 * diagnostics, background learning, calibration, matrix experiments and
 * future compute paths. No prompt-specific phrases or benchmark answers live
 * here. Methods stay out of the foreground hot path unless explicitly called.
 */
(() => {
  'use strict';

  class VilotProcessorProductionLab {
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

    status() {
      return {
        module: 'Processor',
        role: 'production-research-extension',
        calls: this.calls,
        methodCount: Object.getOwnPropertyNames(Object.getPrototypeOf(this)).length - 1,
        ageMs: (performance?.now?.() ?? Date.now()) - this.createdAt
      };
    }
  }

  globalThis.VilotProcessorProductionLab = VilotProcessorProductionLab;
})();
