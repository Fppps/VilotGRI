/**
 * VilotNI 2.5 - SyntacticBridge.js
 * System 34 foundation capacity expansion.
 *
 * Converts semantic clause/proposition structure into grammatical relation
 * bridges. The content anchors remain model-selected; this layer supplies only
 * grammatical function words and relation-compatible scaffolding.
 */
(() => {
  'use strict';

  const normalize = value => String(value || '')
    .toLowerCase()
    .replace(/[’]/g, "'")
    .replace(/[^a-z0-9'+.\-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  class VilotSyntacticBridge {
    constructor(processor, options = {}, ruleStore = null) {
      this.processor = processor;
      this.ruleStore = ruleStore;
      this.options = {
        enabled: options.syntacticBridgeEnabled !== false,
        preferKeywordSurface: options.syntacticBridgePreferKeywordSurface !== false,
        maxBridgeWords: Math.max(2, Math.min(16, options.syntacticBridgeMaxBridgeWords | 0 || 8))
      };
      this.bridges = 0;
      this.compositions = 0;
      this.lastDiagnostics = null;
    }

    _rules() {
      return this.ruleStore?.syntaxPatterns?.() || this.ruleStore?.get?.('syntaxPatterns') || {};
    }

    _compositionRules() {
      return this.ruleStore?.sentenceCompositionRules?.() || this.ruleStore?.get?.('sentenceComposition') || {};
    }

    _lowerFirst(text) {
      const value = String(text || '').trim();
      return value ? value.charAt(0).toLowerCase() + value.slice(1) : value;
    }

    _stripTerminal(text) {
      return String(text || '').trim().replace(/[.!?]+$/, '');
    }

    _cleanBridgeWords(words = []) {
      const out = [];
      for (const word of words) {
        const clean = String(word || '').trim();
        if (!clean) continue;
        if (out.length && normalize(out[out.length - 1]) === normalize(clean)) continue;
        out.push(clean);
        if (out.length >= this.options.maxBridgeWords) break;
      }
      return out;
    }

    _phrase(keywordItems, fallback = '') {
      const items = Array.isArray(keywordItems) ? keywordItems : [];
      const words = items.map(item => String(item?.surface || item?.word || '').trim()).filter(Boolean);
      const keywordText = words.join(' ').replace(/\s+/g, ' ').trim();
      const original = this._stripTerminal(fallback);

      if (this.options.preferKeywordSurface && keywordText && original) {
        const selected = new Set(items.map(item => normalize(item?.word || item?.surface)).filter(Boolean));
        const tokens = this.processor?.tokenize?.(original, 128) || [];
        const selectedPositions = [];
        for (let i = 0; i < tokens.length; i++) {
          const lower = normalize(tokens[i]?.lower || tokens[i]?.surface);
          if (selected.has(lower)) selectedPositions.push(i);
        }

        if (selectedPositions.length) {
          const first = selectedPositions[0];
          const last = selectedPositions[selectedPositions.length - 1];
          const functionPOS = new Set(['Det','Prep','Conj','Aux','Modal','Pronoun','Punct']);
          const configured = this._rules()?.Function_Words || {};
          const functionWords = new Set(Object.values(configured).flat().map(normalize));
          if (!functionWords.size) {
            ['a','an','the','and','or','but','because','if','while','although','though','of','to','in','on','at','for','from','with','without','by','through','into','onto','over','under','between','among','during','before','after','as','than','is','am','are','was','were','be','been','being','do','does','did','have','has','had','can','could','will','would','shall','should','may','might','must'].forEach(w => functionWords.add(w));
          }
          const isFunctionToken = token => {
            const lower = normalize(token?.lower || token?.surface);
            const pos = String(token?.pos || token?.activePOS || token?.entry?.pos || 'Other');
            return functionPOS.has(pos) || functionWords.has(lower);
          };

          let start = first;
          while (start > 0 && isFunctionToken(tokens[start - 1])) {
            const prior = tokens[start - 1] || {};
            const priorSurface = String(prior.surface || prior.lower || '').trim();
            const priorPOS = String(prior.pos || prior.activePOS || prior.entry?.pos || 'Other');
            if (priorPOS === 'Punct' || /[.!?;:]$/.test(priorSurface)) break;
            start--;
          }

          const kept = [];
          for (let i = start; i <= last; i++) {
            const token = tokens[i] || {};
            const lower = normalize(token.lower || token.surface);
            const pos = String(token.pos || token.activePOS || token.entry?.pos || 'Other');
            const surface = String(token.surface || token.lower || '').trim();
            if (!surface) continue;
            if (selected.has(lower)) kept.push(surface);
            else if (isFunctionToken(token)) {
              if (pos === 'Punct' && !/^[,;:]$/.test(surface)) continue;
              kept.push(surface);
            }
          }
          const bridged = kept.join(' ')
            .replace(/\s+([,;:])/g, '$1')
            .replace(/([,;:])(?=[A-Za-z0-9])/g, '$1 ')
            .replace(/\s+/g, ' ')
            .trim();
          if (bridged) return bridged;
        }
      }

      if (this.options.preferKeywordSurface && keywordText) return keywordText;
      return original || keywordText;
    }

    _subject(clause, group, usePronoun) {
      if (usePronoun) return 'It';
      const original = this._stripTerminal(clause?.subject || clause?.agent || '');
      return original || this._phrase(group?.subject, 'The concept') || 'The concept';
    }

    _value(clause, group) {
      const source = clause?.value || clause?.object || clause?.patient || clause?.result || '';
      const keywordText = this._phrase(group?.value, source);
      return keywordText || this._stripTerminal(source);
    }

    _questionCore(analysis) {
      const scaffold = analysis?.responseScaffold || null;
      const source = Array.isArray(scaffold?.surfaces) && scaffold.surfaces.length
        ? scaffold.surfaces.slice()
        : Array.isArray(scaffold?.words) ? scaffold.words.slice() : [];
      const terminalBridges = new Set(['because','by','through','using','with','during','in','if','when','after','before']);
      while (source.length && terminalBridges.has(normalize(source[source.length - 1]))) source.pop();
      return source.map(value => String(value || '').trim()).filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
    }

    _bridgeOptions(type) {
      const map = this._rules()?.Bridge_Map || {};
      const key = normalize(type).replace(/\s+/g, '_');
      const direct = map[key] || map[normalize(type)] || null;
      const rows = Array.isArray(direct) ? direct : direct ? [direct] : [];
      const grammarRows = this.processor?.grammarEngine?.relationBridge?.(key) || [];
      return Array.from(new Set([...rows.map(String), ...grammarRows.map(String)].filter(Boolean)));
    }

    _templateFor(type, subject, value, analysis, clause = {}) {
      const key = normalize(type).replace(/\s+/g, '_');
      const bridgeOptions = this._bridgeOptions(key);
      const questionCore = analysis?.explanationMode ? this._questionCore(analysis) : '';
      const primary = bridgeOptions[0] || '';
      let text = '';
      let bridgeWords = [];

      if (questionCore && ['cause','mechanism','purpose'].includes(key)) {
        const connector = key === 'cause' ? 'because' : key === 'purpose' ? 'so that' : 'by';
        text = `${questionCore} ${connector} ${this._lowerFirst(value)}`;
        bridgeWords = connector.split(/\s+/);
        return { text, bridgeWords, template: 'question-core' };
      }

      switch (key) {
        case 'definition':
          text = `${subject} ${primary || 'is'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'is').split(/\s+/);
          break;
        case 'purpose':
          text = `${subject} ${primary || 'is used to'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'is used to').split(/\s+/);
          break;
        case 'mechanism':
        case 'operation':
          text = `${subject} ${primary || 'works through'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'works through').split(/\s+/);
          break;
        case 'capability':
          text = `${subject} ${primary || 'can'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'can').split(/\s+/);
          break;
        case 'limitation':
        case 'constraint':
          text = `${subject} ${primary || 'is limited by'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'is limited by').split(/\s+/);
          break;
        case 'part_of':
          text = `${subject} ${primary || 'is part of'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'is part of').split(/\s+/);
          break;
        case 'has_part':
          text = `${subject} ${primary || 'includes'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'includes').split(/\s+/);
          break;
        case 'cause':
          text = `${subject} ${primary || 'is caused by'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'is caused by').split(/\s+/);
          break;
        case 'effect':
        case 'result':
          text = `${subject} ${primary || 'can lead to'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'can lead to').split(/\s+/);
          break;
        case 'feedback':
          text = `${subject} ${primary || 'feeds back into'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'feeds back into').split(/\s+/);
          break;
        case 'condition': {
          const main = this._stripTerminal(clause?.main || subject);
          text = `If ${this._lowerFirst(value)}, ${main}`;
          bridgeWords = ['if'];
          break;
        }
        case 'contrast':
          text = `${subject} ${primary || 'differs from'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'differs from').split(/\s+/);
          break;
        case 'example':
          text = `${primary || 'Examples include'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'Examples include').split(/\s+/);
          break;
        case 'property':
          text = `${subject} ${primary || 'has'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'has').split(/\s+/);
          break;
        case 'sequence':
          text = `${subject}, then ${this._lowerFirst(value)}`;
          bridgeWords = ['then'];
          break;
        case 'relation':
        default:
          text = `${subject} ${primary || 'is related to'} ${this._lowerFirst(value)}`;
          bridgeWords = (primary || 'is related to').split(/\s+/);
          break;
      }
      return { text, bridgeWords: this._cleanBridgeWords(bridgeWords), template: key };
    }

    bridgeClause(clause, group = null, analysis = null, options = {}) {
      const started = performance.now();
      if (!this.options.enabled || !clause) return null;

      const usePronoun = options.usePronoun === true;
      const subject = this._subject(clause, group, usePronoun);
      const value = this._value(clause, group);
      if (!value) return null;

      const type = normalize(clause.type || clause.relation || clause.requestedRole || 'relation').replace(/\s+/g, '_');
      const built = this._templateFor(type, subject, value, analysis, clause);
      const result = {
        text: this._stripTerminal(built.text),
        type,
        bridgeWords: built.bridgeWords,
        insertedCount: built.bridgeWords.length,
        template: built.template,
        diagnostics: {
          bridgeMs: performance.now() - started,
          keywordSubjectCount: group?.subject?.length || 0,
          keywordValueCount: group?.value?.length || 0,
          bridgeWords: built.bridgeWords.slice(),
          ruleDriven: Boolean(this._rules()?.Schema_Version),
          clauseRelation: clause?.relation || null,
          requestedRole: clause?.requestedRole || null
        }
      };
      this.bridges++;
      this.lastDiagnostics = result.diagnostics;
      return result;
    }

    connectorFor(leftIdea, rightIdea, requested = '') {
      const explicit = normalize(requested).replace(/\s+/g, '_');
      const relation = normalize(rightIdea?.relation || rightIdea?.type || explicit || '').replace(/\s+/g, '_');
      const configured = this._rules()?.Connector_Map || {};
      const composition = this._compositionRules()?.Relations || {};
      if (configured[explicit]) return configured[explicit];
      if (configured[relation]) return configured[relation];
      if (composition[explicit]?.connector) return composition[explicit].connector;
      if (composition[relation]?.connector) return composition[relation].connector;
      if (['but','and','or','because','if','unless','after','before','when','although','whereas','then'].includes(explicit)) return explicit;
      if (normalize(rightIdea?.type) === 'contrast') return 'but';
      return 'and';
    }

    composeIdeas(leftText, rightText, relation = 'addition', meta = {}) {
      const left = this._stripTerminal(leftText);
      const right = this._stripTerminal(rightText);
      if (!left) return right;
      if (!right) return left;
      const key = normalize(relation).replace(/\s+/g, '_');
      const spec = this._compositionRules()?.Relations?.[key] || {};
      const connector = spec.connector || this.connectorFor({ text: left }, { text: right, relation: key }, key);
      let text;
      if (spec.action === 'subordinate' && ['because','although','if','unless','after','before','when','while'].includes(connector)) {
        if (['if','unless','after','before','when','although'].includes(connector) && meta.preferFronted) text = `${connector.charAt(0).toUpperCase() + connector.slice(1)} ${this._lowerFirst(right)}, ${this._lowerFirst(left)}`;
        else text = `${left} ${connector} ${this._lowerFirst(right)}`;
      } else if (spec.action === 'sequence' || connector === 'then') {
        text = `${left}, then ${this._lowerFirst(right)}`;
      } else if (spec.action === 'separate') {
        text = `${left}. ${right.charAt(0).toUpperCase() + right.slice(1)}`;
      } else {
        text = `${left}${['but','so','yet'].includes(connector) ? ',' : ''} ${connector} ${this._lowerFirst(right)}`;
      }
      this.compositions++;
      return text.replace(/\s+/g, ' ').trim();
    }

    status() {
      return {
        ready: true,
        role: 'data-driven-clause-and-proposition-syntactic-function-word-bridge',
        bridges: this.bridges,
        compositions: this.compositions,
        lastDiagnostics: this.lastDiagnostics,
        architecture: {
          relationDriven: true,
          clauseAware: true,
          propositionCompatible: true,
          insertsPrepositions: true,
          insertsConjunctions: true,
          insertsSubordinators: true,
          ruleDriven: Boolean(this._rules()?.Schema_Version),
          topicSpecificAnswers: false
        }
      };
    }
  }


  const V34_BRIDGE_RELATIONS=Object.freeze([
    { relation:'identity', bridge:'is', family:'copular', preservesDirection:true },
    { relation:'definition', bridge:'is', family:'copular', preservesDirection:true },
    { relation:'property', bridge:'has', family:'copular', preservesDirection:true },
    { relation:'part_of', bridge:'is part of', family:'relational', preservesDirection:true },
    { relation:'has_part', bridge:'includes', family:'relational', preservesDirection:true },
    { relation:'purpose', bridge:'is used to', family:'purpose', preservesDirection:true },
    { relation:'capability', bridge:'can', family:'modal', preservesDirection:true },
    { relation:'limitation', bridge:'cannot always', family:'modal', preservesDirection:true },
    { relation:'cause', bridge:'because', family:'causal', preservesDirection:true },
    { relation:'effect', bridge:'therefore', family:'causal', preservesDirection:true },
    { relation:'condition', bridge:'if', family:'conditional', preservesDirection:true },
    { relation:'negative_condition', bridge:'unless', family:'conditional', preservesDirection:true },
    { relation:'contrast', bridge:'but', family:'contrast', preservesDirection:true },
    { relation:'concession', bridge:'although', family:'contrast', preservesDirection:true },
    { relation:'comparison', bridge:'compared with', family:'comparison', preservesDirection:true },
    { relation:'similarity', bridge:'similarly', family:'comparison', preservesDirection:true },
    { relation:'difference', bridge:'whereas', family:'comparison', preservesDirection:true },
    { relation:'temporal_before', bridge:'before', family:'temporal', preservesDirection:true },
    { relation:'temporal_after', bridge:'after', family:'temporal', preservesDirection:true },
    { relation:'simultaneous', bridge:'while', family:'temporal', preservesDirection:true },
    { relation:'sequence', bridge:'then', family:'sequence', preservesDirection:true },
    { relation:'example', bridge:'for example', family:'discourse', preservesDirection:true },
    { relation:'elaboration', bridge:'specifically', family:'discourse', preservesDirection:true },
    { relation:'addition', bridge:'and', family:'coordination', preservesDirection:true },
    { relation:'alternative', bridge:'or', family:'coordination', preservesDirection:true },
    { relation:'result', bridge:'so', family:'causal', preservesDirection:true },
    { relation:'reason', bridge:'because', family:'causal', preservesDirection:true },
    { relation:'evidence', bridge:'which is supported by', family:'epistemic', preservesDirection:true },
    { relation:'uncertainty', bridge:'may', family:'epistemic', preservesDirection:true },
    { relation:'hypothesis', bridge:'could', family:'epistemic', preservesDirection:true },
    { relation:'mechanism', bridge:'by', family:'mechanism', preservesDirection:true },
    { relation:'input', bridge:'takes', family:'mechanism', preservesDirection:true },
    { relation:'output', bridge:'produces', family:'mechanism', preservesDirection:true },
    { relation:'feedback', bridge:'feeds back into', family:'mechanism', preservesDirection:true },
    { relation:'update', bridge:'updates', family:'mechanism', preservesDirection:true },
    { relation:'evaluation', bridge:'evaluates', family:'mechanism', preservesDirection:true },
    { relation:'selection', bridge:'selects', family:'mechanism', preservesDirection:true },
    { relation:'dependency', bridge:'depends on', family:'dependency', preservesDirection:true },
    { relation:'requirement', bridge:'requires', family:'dependency', preservesDirection:true },
    { relation:'exception', bridge:'except when', family:'dependency', preservesDirection:true },
  ]);
  const v34BNorm=v=>String(v??'').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g,' ').replace(/\s+/g,' ').trim();
  class V34BridgeEngine {
    constructor(owner){this.owner=owner;this.last=null;this.history=[];}
    spec(relation){const key=v34BNorm(relation).replace(/\s+/g,'_');return V34_BRIDGE_RELATIONS.find(x=>x.relation===key)||{relation:key,bridge:'and',family:'coordination',preservesDirection:true};}
    classifyRelation(meta={}){
      const explicit=v34BNorm(meta.relation||meta.type||meta.dependency||'').replace(/\s+/g,'_');if(explicit)return explicit;
      if(meta.condition)return 'condition';if(meta.cause)return 'cause';if(meta.contrast)return 'contrast';if(meta.time==='before')return 'temporal_before';if(meta.time==='after')return 'temporal_after';return 'addition';
    }
    referenceRisk(left='',right='',meta={}){
      const pronouns=(String(right).toLowerCase().match(/\b(it|they|them|this|that|these|those|he|she|him|her)\b/g)||[]).length;
      const subjects=[...(meta.leftSubjects||[]),...(meta.rightSubjects||[])].filter(Boolean);const distinct=new Set(subjects.map(v34BNorm)).size;
      return Math.min(1,pronouns*0.18+Math.max(0,distinct-1)*0.16+(meta.referenceAmbiguity||0));
    }
    clauseLength(text=''){return String(text||'').trim().split(/\s+/).filter(Boolean).length;}
    punctuationFor(relation,left,right){
      const family=this.spec(relation).family;const independentLeft=/\b(?:is|are|was|were|can|will|does|do|did|has|have|had|produces?|uses?|requires?)\b/i.test(left);const independentRight=/\b(?:is|are|was|were|can|will|does|do|did|has|have|had|produces?|uses?|requires?)\b/i.test(right);
      if(['contrast','causal','coordination'].includes(family)&&independentLeft&&independentRight)return ', ';if(family==='discourse')return '. ';return ' ';
    }
    bridgeText(relation,left,right,meta={}){
      const spec=this.spec(relation);let bridge=spec.bridge;const punctuation=this.punctuationFor(relation,left,right);if(spec.family==='conditional'&&relation==='condition')return `If ${right.replace(/[.!?]+$/,'')}, ${left.replace(/^[A-Z]/,c=>c.toLowerCase())}`;
      if(spec.family==='temporal'&&relation==='temporal_before')return `${left.replace(/[.!?]+$/,'')} before ${right.replace(/[.!?]+$/,'')}`;
      if(spec.family==='temporal'&&relation==='temporal_after')return `${left.replace(/[.!?]+$/,'')} after ${right.replace(/[.!?]+$/,'')}`;
      if(spec.family==='causal'&&relation==='cause')return `${left.replace(/[.!?]+$/,'')} because ${right.replace(/[.!?]+$/,'')}`;
      if(spec.family==='causal'&&relation==='result')return `${left.replace(/[.!?]+$/,'')}, so ${right.replace(/^[A-Z]/,c=>c.toLowerCase()).replace(/[.!?]+$/,'')}`;
      return `${left.replace(/[.!?]+$/,'')}${punctuation}${bridge} ${right.replace(/^[A-Z]/,c=>c.toLowerCase()).replace(/[.!?]+$/,'')}`;
    }
    decide(left,right,relation,meta={}){
      const rel=this.classifyRelation({...meta,relation});const risk=this.referenceRisk(left,right,meta);const total=this.clauseLength(left)+this.clauseLength(right);const spec=this.spec(rel);
      const split=risk>0.64||total>44||meta.conflictingPolarity||meta.unresolvedReference;
      return {relation:rel,spec,referenceRisk:risk,totalTokens:total,action:split?'split':'bridge',reason:split?(risk>0.64?'reference-risk':'complexity'):'compatible'};
    }
    compose(left,right,relation,meta={}){
      const decision=this.decide(left,right,relation,meta);let text;if(decision.action==='split')text=`${String(left).replace(/[.!?]*$/,'.')} ${String(right).replace(/^[a-z]/,c=>c.toUpperCase())}`;else text=this.bridgeText(decision.relation,String(left),String(right),meta);
      text=text.replace(/\s+/g,' ').replace(/\s+([,.;:!?])/g,'$1').trim();const result={text,decision};this.last=result;this.history.push({at:Date.now(),...decision});if(this.history.length>128)this.history.shift();return result;
    }
    composeSequence(ideas=[],relations=[],meta={}){
      const rows=(ideas||[]).map(String).filter(Boolean);if(!rows.length)return{text:'',steps:[]};let text=rows[0],steps=[];
      for(let i=1;i<rows.length;i++){const relation=relations[i-1]||'addition';const result=this.compose(text,rows[i],relation,meta);steps.push(result.decision);text=result.text;}return{text,steps};
    }
  }
  const v34OriginalComposeIdeas=VilotSyntacticBridge.prototype.composeIdeas;
  VilotSyntacticBridge.prototype.composeIdeas=function(leftText,rightText,relation='addition',meta={}){
    const base=v34OriginalComposeIdeas.call(this,leftText,rightText,relation,meta);this.v34Engine||=new V34BridgeEngine(this);const deep=this.v34Engine.compose(leftText,rightText,relation,meta);
    if(deep.decision.referenceRisk>0.64||deep.decision.action==='split')return deep.text;return typeof base==='string'&&base.trim()?base:deep.text;
  };
  VilotSyntacticBridge.prototype.composeSequence=function(ideas,relations,meta={}){this.v34Engine||=new V34BridgeEngine(this);return this.v34Engine.composeSequence(ideas,relations,meta);};
  VilotSyntacticBridge.prototype.bridgeDecision=function(left,right,relation,meta={}){this.v34Engine||=new V34BridgeEngine(this);return this.v34Engine.decide(left,right,relation,meta);};
  VilotSyntacticBridge.prototype.bridgeCatalog=function(){return V34_BRIDGE_RELATIONS.map(x=>({...x}));};



  const V34_BRIDGE_SCENARIOS = Object.freeze([
    {
      id: "cause-scenario-1",
      family: "cause",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "result-scenario-2",
      family: "result",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "condition-scenario-3",
      family: "condition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "contrast-scenario-4",
      family: "contrast",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "concession-scenario-5",
      family: "concession",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "comparison-scenario-6",
      family: "comparison",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "sequence-scenario-7",
      family: "sequence",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "temporal-scenario-8",
      family: "temporal",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "mechanism-scenario-9",
      family: "mechanism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "discourse-scenario-10",
      family: "discourse",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "cause-scenario-11",
      family: "cause",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "result-scenario-12",
      family: "result",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "condition-scenario-13",
      family: "condition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "contrast-scenario-14",
      family: "contrast",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "concession-scenario-15",
      family: "concession",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "comparison-scenario-16",
      family: "comparison",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "sequence-scenario-17",
      family: "sequence",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "temporal-scenario-18",
      family: "temporal",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "mechanism-scenario-19",
      family: "mechanism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "discourse-scenario-20",
      family: "discourse",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "cause-scenario-21",
      family: "cause",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "result-scenario-22",
      family: "result",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "condition-scenario-23",
      family: "condition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "contrast-scenario-24",
      family: "contrast",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "concession-scenario-25",
      family: "concession",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "comparison-scenario-26",
      family: "comparison",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "sequence-scenario-27",
      family: "sequence",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "temporal-scenario-28",
      family: "temporal",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "mechanism-scenario-29",
      family: "mechanism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "discourse-scenario-30",
      family: "discourse",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "cause-scenario-31",
      family: "cause",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "result-scenario-32",
      family: "result",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "condition-scenario-33",
      family: "condition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "contrast-scenario-34",
      family: "contrast",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "concession-scenario-35",
      family: "concession",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "comparison-scenario-36",
      family: "comparison",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "sequence-scenario-37",
      family: "sequence",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "temporal-scenario-38",
      family: "temporal",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "mechanism-scenario-39",
      family: "mechanism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "discourse-scenario-40",
      family: "discourse",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "cause-scenario-41",
      family: "cause",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "result-scenario-42",
      family: "result",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "condition-scenario-43",
      family: "condition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "contrast-scenario-44",
      family: "contrast",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "concession-scenario-45",
      family: "concession",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "comparison-scenario-46",
      family: "comparison",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "sequence-scenario-47",
      family: "sequence",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "temporal-scenario-48",
      family: "temporal",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "mechanism-scenario-49",
      family: "mechanism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "discourse-scenario-50",
      family: "discourse",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "cause-scenario-51",
      family: "cause",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "result-scenario-52",
      family: "result",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "condition-scenario-53",
      family: "condition",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "contrast-scenario-54",
      family: "contrast",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "concession-scenario-55",
      family: "concession",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "comparison-scenario-56",
      family: "comparison",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "sequence-scenario-57",
      family: "sequence",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "temporal-scenario-58",
      family: "temporal",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "mechanism-scenario-59",
      family: "mechanism",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "discourse-scenario-60",
      family: "discourse",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "cause-scenario-61",
      family: "cause",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
  ]);
  VilotSyntacticBridge.prototype.bridgeScenarios = function() { return V34_BRIDGE_SCENARIOS.map(row => ({ ...row })); };

  globalThis.VilotSyntacticBridge = VilotSyntacticBridge;
})();
