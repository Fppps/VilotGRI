/**
 * VilotNI 2.5 - ConversationalCognition.js
 *
 * Always-on conversational continuity layer.
 *
 * This is deliberately NOT a mode switch, permission toggle, or hard routing
 * classifier. Every prompt still enters the same VilotNI 2.5 reasoning and
 * generation architecture. This layer contributes small, soft signals about
 * current-topic anchors, recent dialogue continuity, and how much freedom the
 * response can take when the user is making an observation, fragment, joke,
 * or casual statement rather than requesting a rigid answer shape.
 *
 * It never contains canned answers and never substitutes for the decoder.
 */
(() => {
  'use strict';

  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const normalize=v=>String(v||'').toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'+.\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const uniq=values=>Array.from(new Set((values||[]).map(v=>normalize(v)).filter(Boolean)));
  const QUESTION_WORDS=new Set(['what','who','where','when','why','how','which','whom','whose']);
  const SUPPORT=new Set(['am','is','are','was','were','be','been','being','do','does','did','have','has','had','can','could','will','would','shall','should','may','might','must']);
  const LIGHT_WORDS=new Set(['the','a','an','and','or','but','so','just','really','very','too','also','still','then','well','like','lol','lmao','bro','man','yeah','yep','nah','nope','oi']);
  const DEICTIC=new Set(['it','this','that','these','those','he','she','they','them','him','her','there','here','one','ones','something','someone']);

  class VilotConversationalCognition {
    constructor(processor=null, options={}) {
      this.processor=processor;
      this.options={
        historyTurns:Math.max(2,Math.min(16,options.conversationHistoryTurns|0||8)),
        maxAnchors:Math.max(8,Math.min(48,options.conversationMaxAnchors|0||28)),
        maxPriorAnchors:Math.max(4,Math.min(32,options.conversationMaxPriorAnchors|0||16)),
        softInfluence:clamp(options.conversationSoftInfluence??0.16,0.04,0.30)
      };
      this.turns=0;
      this.last=null;
    }

    _tokens(text) {
      const parsed=this.processor?.tokenize?.(String(text||''),192)||[];
      if(parsed.length) return parsed.filter(t=>t?.pos!=='Punct').map(t=>({
        lower:normalize(t.lower||t.surface),
        pos:String(t.pos||''),
        content:Boolean(this.processor?.resolveSayableWord?.(t.lower||t.surface,{allowRootFallback:true})?.contentWord)
      })).filter(t=>t.lower);
      return normalize(text).split(/\s+/).filter(Boolean).map(lower=>({lower,pos:'',content:!LIGHT_WORDS.has(lower)&&!QUESTION_WORDS.has(lower)&&!SUPPORT.has(lower)}));
    }

    _content(text) {
      return uniq(this._tokens(text).filter(t=>t.content||(!LIGHT_WORDS.has(t.lower)&&!QUESTION_WORDS.has(t.lower)&&!SUPPORT.has(t.lower))).map(t=>t.lower));
    }

    _overlap(a,b) {
      const A=new Set(a||[]),B=new Set(b||[]);
      if(!A.size||!B.size)return 0;
      let hit=0; for(const x of A) if(B.has(x)) hit++;
      return hit/Math.max(1,Math.min(A.size,B.size));
    }

    analyze(promptText, analysis={}, dialogueContext=[]) {
      const raw=String(promptText||'').trim();
      const normalized=normalize(raw);
      const tokens=this._tokens(raw);
      const lowers=tokens.map(t=>t.lower);
      const currentContent=uniq([
        ...(analysis?.contentWords||[]),
        ...(analysis?.roots||[]),
        ...(analysis?.entityCanonicalTokens||[]),
        ...(analysis?.entityConcepts||[]),
        ...(analysis?.architectureConcepts||[]),
        ...(analysis?.entities||[]).flatMap(row=>[row.canonical,...(row.roles||[]),...(row.fields||[])]),
        ...this._content(raw)
      ]).slice(0,this.options.maxAnchors);

      const hasQuestionMark=/\?\s*$/.test(raw);
      const startsQuestion=QUESTION_WORDS.has(lowers[0]||'')||(SUPPORT.has(lowers[0]||'')&&lowers.length>1);
      const questionPressure=clamp((hasQuestionMark?0.55:0)+(startsQuestion?0.34:0)+(analysis?.isQuestion?0.20:0));
      const explicitRequest=clamp(
        (['explain','describe','compare','define','tell','show','list','calculate','write','create','make'].includes(lowers[0]||'')?0.62:0)+
        (analysis?.requestedSlot&&analysis.requestedSlot!=='content'?0.22:0)
      );

      const recent=(dialogueContext||[]).slice(-this.options.historyTurns).filter(turn=>String(turn?.text||'').trim());
      const priorRows=[];
      for(let i=recent.length-1,age=0;i>=0&&priorRows.length<this.options.historyTurns;i--,age++) {
        const turn=recent[i];
        const content=this._content(turn.text);
        if(!content.length)continue;
        const currentOverlap=this._overlap(currentContent,content);
        const deictic=lowers.some(w=>DEICTIC.has(w));
        const recency=Math.pow(0.82,age);
        const relevance=clamp(currentOverlap*0.70+(deictic?0.18:0)+recency*0.12);
        priorRows.push({role:String(turn.role||'unknown'),content,relevance,recency,text:String(turn.text||'').slice(0,240)});
      }
      priorRows.sort((a,b)=>b.relevance-a.relevance);
      const priorAnchors=uniq(priorRows.slice(0,4).flatMap(row=>row.content)).slice(0,this.options.maxPriorAnchors);
      const continuity=this._overlap(currentContent,priorAnchors);

      const lexicalDensity=lowers.length?currentContent.length/Math.max(1,lowers.length):0;
      const shortCasual=lowers.length<=5&&!hasQuestionMark&&questionPressure<0.4;
      const fragmentary=Boolean(raw)&&!hasQuestionMark&&lowers.length<=7&&!(analysis?.clauseState?.predicateToken);
      const responseFreedom=clamp(
        0.34+
        (shortCasual?0.28:0)+
        (fragmentary?0.12:0)+
        (continuity*0.12)-
        (questionPressure*0.34)-
        (explicitRequest*0.26)
      );

      // These are influence weights, not a selected interaction mode. Multiple
      // dispositions remain active simultaneously and only nudge later scoring.
      const dispositions={
        acknowledge:clamp(0.34+responseFreedom*0.42+(shortCasual?0.16:0)),
        extend:clamp(0.42+lexicalDensity*0.22+continuity*0.20),
        reflect:clamp(0.30+responseFreedom*0.34+continuity*0.16),
        explain:clamp(0.24+questionPressure*0.36+explicitRequest*0.30),
        compare:clamp(0.18+((lowers.includes('vs')||lowers.includes('versus')||lowers.includes('compare'))?0.56:0)+continuity*0.08),
        askBack:clamp(0.10+(currentContent.length===0?0.28:0)+(fragmentary?0.08:0))
      };

      const currentAnchors=currentContent.slice(0,this.options.maxAnchors);
      const anchors=uniq([...currentAnchors,...priorAnchors]).slice(0,this.options.maxAnchors);

      // Lexical hints are ordinary discourse/function words already expected to
      // exist in Words.json. They are only exposed as weak candidates. They are
      // not phrases, templates, or required openings, and multiple competing
      // dispositions contribute at the same time.
      const lexicalHints=[];
      const hint=(word,weight,source)=>{ if(weight>0.24) lexicalHints.push({word,weight:clamp(weight),source}); };
      hint('yeah',dispositions.acknowledge*0.78,'acknowledge');
      hint('yes',dispositions.acknowledge*0.52,'acknowledge');
      hint('also',dispositions.extend*0.74,'extend');
      hint('still',dispositions.extend*0.54,'extend');
      hint('though',Math.max(dispositions.extend,dispositions.compare)*0.62,'relation');
      hint('maybe',dispositions.reflect*0.66,'reflect');
      hint('really',Math.max(dispositions.acknowledge,dispositions.reflect)*0.48,'reflect');
      hint('because',dispositions.explain*0.72,'explain');
      hint('while',dispositions.compare*0.72,'compare');
      hint('why',dispositions.askBack*0.58,'ask-back');
      hint('how',dispositions.askBack*0.54,'ask-back');
      hint('what',dispositions.askBack*0.50,'ask-back');
      lexicalHints.sort((a,b)=>b.weight-a.weight);

      const standaloneCapable=true;
      const result={
        version:1,
        policy:'always-on-soft-conversational-cognition',
        currentAnchors,
        priorAnchors,
        anchors,
        continuity,
        responseFreedom,
        questionPressure,
        explicitRequest,
        lexicalDensity,
        shortCasual,
        fragmentary,
        dispositions,
        lexicalHints:lexicalHints.slice(0,8),
        softInfluence:this.options.softInfluence,
        standaloneCapable,
        priorTurnsConsidered:priorRows.length,
        relevantPriorTurns:priorRows.slice(0,4).map(row=>({role:row.role,relevance:Number(row.relevance.toFixed(4)),anchors:(row.content||[]).slice(0,8)}))
      };
      this.turns++;
      this.last=result;
      return result;
    }

    enrichAnalysis(analysis,promptText,dialogueContext=[]) {
      const state=this.analyze(promptText,analysis,dialogueContext);
      return {
        ...analysis,
        conversationState:state,
        conversationAnchors:state.anchors,
        conversationCurrentAnchors:state.currentAnchors,
        conversationPriorAnchors:state.priorAnchors,
        conversationPriority:(state.currentAnchors||[]).slice(0,12)
      };
    }

    status(){return {ready:true,role:'always-on-soft-conversational-cognition',turns:this.turns,last:this.last};}
  }

  globalThis.VilotConversationalCognition=VilotConversationalCognition;
})();
