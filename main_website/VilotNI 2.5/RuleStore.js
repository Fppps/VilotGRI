/**
 * VilotNI 2 - RuleStore.js
 *
 * Loads compact structural/routing datasets that are intentionally separate
 * from Words.json and TrainingInfo.json. These files contain grammar,
 * punctuation, query-routing, context-dependency and relation metadata. They do
 * not contain prompt-specific answers and do not bypass the Decoder confidence
 * gate.
 */
(() => {
  'use strict';

  const normalize = value => String(value || '').trim().toLowerCase();

  class VilotRuleStore {
    constructor(options = {}) {
      this.options = { ...options };
      this.ready = false;
      this.loadedAt = 0;
      this.lastLoadMs = 0;
      this.errors = [];
      this.data = Object.create(null);
      // Seed typed fallbacks immediately so offline/partial-load callers never
      // observe an unshaped policy object during startup. The prototype method
      // is installed later in this module before any constructor is invoked.
      this._v34NormalizeAll?.();
    }

    async _loadJSON(url, key) {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) throw new Error(`${key} failed to load: HTTP ${response.status}`);
      const value = await response.json();
      this.data[key] = value && typeof value === 'object' ? value : {};
      return this.data[key];
    }

    async load(baseURL = './', version = 'v=25-system-37') {
      const started = performance.now();
      const root = String(baseURL || './').replace(/\/?$/, '/');
      const suffix = version ? `?${String(version).replace(/^\?/, '')}` : '';
      const files = {
        grammar: 'GrammarRules.json',
        punctuation: 'PunctuationRules.json',
        query: 'QueryPatterns.json',
        context: 'ContextRules.json',
        semantic: 'SemanticRelations.json',
        definitionIndex: 'DefinitionIndex.json',
        complexSyntax: 'ComplexSyntaxPatterns.json',
        propositionRelations: 'PropositionRelations.json',
        discourse: 'DiscourseRules.json',
        references: 'ReferenceRules.json',
        structuredReasoning: 'StructuredReasoningRules.json',
        mechanismRelations: 'MechanismRelations.json',
        uncertaintyRules: 'UncertaintyRules.json',
        temporalRelations: 'TemporalRelations.json',
        syntaxPatterns: 'SyntaxPatterns.json',
        sentenceComposition: 'SentenceCompositionRules.json',
        replayPolicy: 'ReplayPolicy.json',
        selfImprovementPolicy: 'SelfImprovementPolicy.json'
      };

      this.errors.length = 0;
      const tasks = Object.entries(files).map(async ([key, file]) => {
        try {
          await this._loadJSON(`${root}${file}${suffix}`, key);
        } catch (error) {
          this.errors.push({ key, file, message: String(error?.message || error) });
          this.data[key] = {};
        }
      });
      await Promise.all(tasks);
      this.ready = true;
      this.loadedAt = Date.now();
      this.lastLoadMs = performance.now() - started;
      return this.status();
    }

    get(name) {
      return this.data[String(name || '')] || null;
    }

    punctuation(symbol) {
      const key = String(symbol || '');
      return this.data.punctuation?.Symbols?.[key] || null;
    }

    commaRules() {
      return this.data.punctuation?.Comma || null;
    }

    terminalRules() {
      return this.data.punctuation?.Terminal || null;
    }

    internalPunctuationRules() {
      return this.data.punctuation?.Internal || null;
    }

    listPunctuationRules() {
      return this.data.punctuation?.Lists || null;
    }

    quoteRules() {
      return this.data.punctuation?.Quotes || null;
    }

    determiner(word) {
      return this.data.grammar?.Determiners?.[normalize(word)] || null;
    }

    agreement() {
      return this.data.grammar?.Agreement || null;
    }

    coordination() {
      return this.data.grammar?.Coordination || null;
    }

    clauseRules() {
      return this.data.grammar?.Clause || null;
    }

    roleBridges(role) {
      const key = normalize(role);
      const direct = this.data.query?.Role_Bridges?.[key];
      if (Array.isArray(direct)) return direct.slice();
      const relation = this.data.semantic?.Relations?.[key]?.cues;
      return Array.isArray(relation) ? relation.slice() : [];
    }

    operatorRule(operator) {
      return this.data.query?.Question_Operators?.[normalize(operator)] || null;
    }

    selfContainedRules() {
      return this.data.query?.Self_Contained || null;
    }

    contextRules() {
      return this.data.context || null;
    }

    exactEllipsis(text) {
      const clean = normalize(text)
        .replace(/[?!.,;:]+$/g, '')
        .replace(/\s+/g, ' ')
        .trim();
      return this.data.context?.Exact_Ellipsis?.[clean] || null;
    }

    relation(role) {
      return this.data.semantic?.Relations?.[normalize(role)] || null;
    }

    definitionEntry(word) {
      return this.data.definitionIndex?.Entries?.[normalize(word)] || null;
    }

    definitionEntries(words, limit = 8) {
      const out = [];
      const seen = new Set();
      const max = Math.max(1, Math.min(64, limit | 0 || 8));
      for (const word of words || []) {
        const key = normalize(word);
        if (!key || seen.has(key)) continue;
        const entry = this.definitionEntry(key);
        if (!entry) continue;
        seen.add(key);
        out.push({ word: key, ...entry });
        if (out.length >= max) break;
      }
      return out;
    }

    complexSyntaxRules() { return this.data.complexSyntax || null; }

    propositionRelationRules() { return this.data.propositionRelations || null; }

    discourseRules() { return this.data.discourse || null; }

    referenceRules() { return this.data.references || null; }

    structuredReasoningRules() { return this.data.structuredReasoning || null; }

    mechanismRelationRules() { return this.data.mechanismRelations || null; }

    uncertaintyRules() { return this.data.uncertaintyRules || null; }

    temporalRelationRules() { return this.data.temporalRelations || null; }

    syntaxPatterns() { return this.data.syntaxPatterns || null; }

    sentenceCompositionRules() { return this.data.sentenceComposition || null; }

    replayPolicy() { return this.data.replayPolicy || null; }

    selfImprovementPolicy() { return this.data.selfImprovementPolicy || null; }

    mechanismSlot(relation) {
      return this.data.mechanismRelations?.Relations?.[normalize(relation)] || null;
    }

    complexRelation(name) { return this.data.propositionRelations?.Relations?.[normalize(name)] || null; }

    status() {
      const definitionCount = Number(this.data.definitionIndex?.Generated_Entry_Count || 0);
      return {
        role: 'structural-rule-and-routing-data',
        ready: this.ready,
        lastLoadMs: this.lastLoadMs,
        loadedAt: this.loadedAt,
        errors: this.errors.slice(),
        datasets: {
          grammar: Boolean(this.data.grammar?.Schema_Version),
          punctuation: Boolean(this.data.punctuation?.Schema_Version),
          query: Boolean(this.data.query?.Schema_Version),
          context: Boolean(this.data.context?.Schema_Version),
          semantic: Boolean(this.data.semantic?.Schema_Version),
          definitionIndex: Boolean(this.data.definitionIndex?.Schema_Version),
          complexSyntax: Boolean(this.data.complexSyntax?.Schema_Version),
          propositionRelations: Boolean(this.data.propositionRelations?.Schema_Version),
          discourse: Boolean(this.data.discourse?.Schema_Version),
          references: Boolean(this.data.references?.Schema_Version),
          structuredReasoning: Boolean(this.data.structuredReasoning?.Schema_Version),
          mechanismRelations: Boolean(this.data.mechanismRelations?.Schema_Version),
          uncertaintyRules: Boolean(this.data.uncertaintyRules?.Schema_Version),
          temporalRelations: Boolean(this.data.temporalRelations?.Schema_Version),
          syntaxPatterns: Boolean(this.data.syntaxPatterns?.Schema_Version),
          sentenceComposition: Boolean(this.data.sentenceComposition?.Schema_Version),
          replayPolicy: Boolean(this.data.replayPolicy?.Schema_Version),
          selfImprovementPolicy: Boolean(this.data.selfImprovementPolicy?.Schema_Version)
        },
        definitionIndexEntries: definitionCount
      };
    }
  }



  // System 34 defensive dataset shaping. Offline or partially loaded JSON must
  // degrade to empty, typed structures instead of producing undefined-key faults.
  const V34_RULE_SHAPES = Object.freeze({
    grammar: { Schema_Version:0, Build:'', Determiners:{}, Agreement:{}, Pronouns:{}, Pronoun_Case_Rules:{}, Auxiliaries:{primary:{be:[],have:[],do:[]},modal:[]}, Modals:{}, Tense_Aspect_Voice:{}, Coordination:{coordinators:[]}, Clause:{subordinate_openers:[]}, Clause_Types:{}, Clause_Linkers:{}, Complementizers:{}, Noun_Phrase:{}, Verb_Phrase:{}, Adjective_Phrase:{}, Adverb_Phrase:{}, Prepositions:{}, Prepositional_Phrase:{}, Relative_Clause:{markers:[]}, Question_Form:{wh_words:[],yes_no_openers:[]}, Negation:{markers:[]}, Comparison:{}, Participles:{}, Compound_Grammar:{}, Punctuation_Interface:{}, Information_Structure:{}, Sentence_Completeness:{}, Explanation_Grammar:{}, Verb_Classes:{}, Verb_Frames:{}, Verb_Frame_Policies:{}, Morphosyntax_Heuristics:{}, Generation_Constraints:{avoid_terminal_function_words:[],no_surface_may_start_with_orphan_bridge:[]}, Repair_Policy:{}, Scoring:{components:{},severity_penalties:{}}, Structural_Scenarios:[], Complex_Clause:{}, Diagnostics:{issue_codes:{}}, Evolution:{} },
    references: { Schema_Version: 0, Reference_Classes:{}, Reflexives:{}, Personal_Pronouns:{}, Possessives:{}, Demonstratives:{}, Relative_References:{}, Anaphoric_Nominals:{}, Event_And_Proposition_References:{}, Resolution_Priority:[], Candidate_Scoring:{}, Candidate_Features:{}, Ambiguity:{}, Ambiguity_Protocol:{}, Split_Antecedents:{}, Constraints:{}, Pronouns:{}, Nominal_Reference_Patterns:[], Resolution_Programs:{} },
    structuredReasoning: { Schema_Version:0, Proposition_Schema:{}, Relation_Families:{}, Relation_Ontology:{}, Mechanism_Slots:{}, Reasoning_Families:{}, Reasoning_Programs:{}, Failure_Taxonomy:{}, Decomposition:{}, Evidence_Policy:{}, Uncertainty_Propagation:{}, Inference_Contracts:{}, Counterfactual:{}, Contradiction:{}, Termination_Rules:{}, Response_Verification:{}, Verification_Checks:[], Quality_Thresholds:{}, Reasoning_Strategies:{} },
    replayPolicy: { Schema_Version:0, Curricula:{ general:{ families:['generalization','control'], variant_count:4, control_ratio:0.34 } }, Priority_Weights:{}, Spacing:{}, Anti_Regression:{} },
    selfImprovementPolicy: { Schema_Version:0, Diagnosis_Targets:{}, Promotion:{}, Bounds:{}, Safeguards:{}, Fallback_Diagnosis:{ metrics:['reasoning'], profiles:{reasoningScale:0.1}, replay:'general' } },
    uncertaintyRules: { Schema_Version:0, Dimensions:{}, Epistemic_Status:{}, Proposition_Penalties:{}, Propagation:{}, States:[] },
    temporalRelations: { Schema_Version:0, Relations:{}, Cues:{}, Aspect:{}, Frequency:{}, Inference:{}, Freshness:{} },
    syntaxPatterns: { Schema_Version:0, Function_Words:{}, Bridge_Map:{}, Connector_Map:{}, Clause_Order:{}, Repairs:{}, Terminal_Invalid:[], Semantic_Safety:{} },
    sentenceComposition: { Schema_Version:0, Relations:{ independent:{action:'separate',merge_threshold:1} }, Structure_Features:{}, Merge_Constraints:{}, Sentence_Order:[] },
    complexSyntax: { Schema_Version:0, Patterns:{}, Clause_Types:{}, Relations:{} },
    propositionRelations: { Schema_Version:0, Relations:{} },
    mechanismRelations: { Schema_Version:0, Relations:{} }
  });

  const v34MergeShape = (shape, value) => {
    const out = Array.isArray(shape) ? [] : {};
    if (shape && typeof shape === 'object' && !Array.isArray(shape)) {
      for (const [key, child] of Object.entries(shape)) {
        const incoming = value && typeof value === 'object' ? value[key] : undefined;
        if (Array.isArray(child)) out[key] = Array.isArray(incoming) ? incoming : child.slice();
        else if (child && typeof child === 'object') out[key] = v34MergeShape(child, incoming);
        else out[key] = incoming === undefined ? child : incoming;
      }
    }
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const [key, incoming] of Object.entries(value)) {
        if (!(key in out)) out[key] = incoming;
      }
    }
    return out;
  };

  VilotRuleStore.prototype._v34NormalizeDataset = function(key, value) {
    const shape = V34_RULE_SHAPES[key];
    if (!shape) return value && typeof value === 'object' ? value : {};
    return v34MergeShape(shape, value && typeof value === 'object' ? value : {});
  };

  VilotRuleStore.prototype._v34NormalizeAll = function() {
    for (const [key, shape] of Object.entries(V34_RULE_SHAPES)) {
      this.data[key] = this._v34NormalizeDataset(key, this.data[key] || shape);
    }
    const self = this.data.selfImprovementPolicy = this._v34NormalizeDataset('selfImprovementPolicy', this.data.selfImprovementPolicy || {});
    const replay = this.data.replayPolicy = this._v34NormalizeDataset('replayPolicy', this.data.replayPolicy || {});
    if (!self.Diagnosis_Targets || typeof self.Diagnosis_Targets !== 'object' || Array.isArray(self.Diagnosis_Targets)) self.Diagnosis_Targets = {};
    if (!replay.Curricula || typeof replay.Curricula !== 'object' || Array.isArray(replay.Curricula)) replay.Curricula = {};
    const diagnosisNames = [
      'surface-grammar','semantic-retrieval','reasoning-selection','conversation-focus',
      'complex-language-coverage','reference-or-dependency-resolution','proposition-coverage',
      'mechanism-completeness','temporal-reasoning','confidence-calibration','semantic-drift',
      'claim-consistency','cognitive-convergence','multi-part-request-coverage','uncertainty-localization',
      'causal-reasoning','comparison-reasoning','counterfactual-reasoning','replay-curriculum','control-regression','general'
    ];
    for (const name of diagnosisNames) {
      self.Diagnosis_Targets[name] ||= { metrics:['reasoning'], profiles:{ reasoningScale:0.1 }, replay:name };
      replay.Curricula[name] ||= { families:['generalization','control'], variant_count:4, control_ratio:0.34 };
    }
    return this.data;
  };

  const v34OriginalLoad = VilotRuleStore.prototype.load;
  VilotRuleStore.prototype.load = async function(baseURL = './', version = 'v=25-system-37') {
    const result = await v34OriginalLoad.call(this, baseURL, version);
    this._v34NormalizeAll();
    return { ...result, system34DefensiveShapes:true, errors:this.errors.slice() };
  };

  const v34OriginalGet = VilotRuleStore.prototype.get;
  VilotRuleStore.prototype.get = function(name) {
    const key = String(name || '');
    const value = v34OriginalGet.call(this, key);
    if (value) return value;
    if (V34_RULE_SHAPES[key]) {
      this.data[key] = this._v34NormalizeDataset(key, {});
      return this.data[key];
    }
    return null;
  };


  // System 35 policy guards. These are the only supported entry points for
  // dynamic diagnosis/curriculum lookups. They always return typed objects,
  // even if a JSON file is missing, malformed, or only partially loaded.
  VilotRuleStore.prototype.safeDataset = function(name) {
    const key = String(name || '');
    if (V34_RULE_SHAPES[key]) {
      this.data[key] = this._v34NormalizeDataset(key, this.data[key] || {});
      if (key === 'selfImprovementPolicy' || key === 'replayPolicy') this._v34NormalizeAll();
      return this.data[key];
    }
    const value = this.data[key];
    return value && typeof value === 'object' ? value : {};
  };

  VilotRuleStore.prototype.diagnosisTarget = function(name) {
    this._v34NormalizeAll();
    const key = String(name || 'general');
    const policy = this.data.selfImprovementPolicy || {};
    const table = policy.Diagnosis_Targets && typeof policy.Diagnosis_Targets === 'object' ? policy.Diagnosis_Targets : {};
    return table[key] || table.general || policy.Fallback_Diagnosis || { metrics:['reasoning'], profiles:{ reasoningScale:0.1 }, replay:key };
  };

  VilotRuleStore.prototype.replayCurriculum = function(name) {
    this._v34NormalizeAll();
    const key = String(name || 'general');
    const policy = this.data.replayPolicy || {};
    const table = policy.Curricula && typeof policy.Curricula === 'object' ? policy.Curricula : {};
    return table[key] || table.general || { families:['generalization','control'], variant_count:4, control_ratio:0.34 };
  };

  VilotRuleStore.prototype.replayPolicy = function() {
    this._v34NormalizeAll();
    return this.data.replayPolicy;
  };

  VilotRuleStore.prototype.selfImprovementPolicy = function() {
    this._v34NormalizeAll();
    return this.data.selfImprovementPolicy;
  };

  VilotRuleStore.prototype.referenceRules = function() { return this.safeDataset('references'); };
  VilotRuleStore.prototype.structuredReasoningRules = function() { return this.safeDataset('structuredReasoning'); };
  VilotRuleStore.prototype.temporalRelationRules = function() { return this.safeDataset('temporalRelations'); };
  VilotRuleStore.prototype.uncertaintyRules = function() { return this.safeDataset('uncertaintyRules'); };
  VilotRuleStore.prototype.syntaxPatterns = function() { return this.safeDataset('syntaxPatterns'); };
  VilotRuleStore.prototype.sentenceCompositionRules = function() { return this.safeDataset('sentenceComposition'); };


  // System 36 grammar ontology accessors. All return shaped objects so grammar
  // consumers remain available during partial/offline dataset loading.
  VilotRuleStore.prototype.grammarRules = function() { return this.safeDataset('grammar'); };
  VilotRuleStore.prototype.grammarSection = function(name) {
    const grammar = this.grammarRules();
    const key = String(name || '');
    const value = grammar?.[key];
    return value && typeof value === 'object' ? value : {};
  };
  VilotRuleStore.prototype.verbFrame = function(word) {
    const key = normalize(word);
    return this.grammarSection('Verb_Frames')?.[key] || null;
  };
  VilotRuleStore.prototype.grammarScoring = function() { return this.grammarSection('Scoring'); };
  VilotRuleStore.prototype.explanationGrammar = function(kind = '') {
    const table = this.grammarSection('Explanation_Grammar');
    const key = normalize(kind).replace(/\s+/g, '_');
    return key ? (table?.[key] || null) : table;
  };



  globalThis.VilotRuleStore = VilotRuleStore;
})();
