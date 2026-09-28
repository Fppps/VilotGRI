/**
 * VilotNI 2.5 - MechanismGraph.js
 *
 * Generic HOW/process graph builder. It converts proposition and learned
 * relations into ordered input, operation, transition, output, feedback,
 * condition and constraint slots without storing topic-specific answers.
 */
(() => {
  'use strict';
  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const norm=v=>String(v||'').toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'+.\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const uniq=values=>Array.from(new Set((values||[]).map(norm).filter(Boolean)));

  class VilotMechanismGraph {
    constructor(ruleStore=null, options={}) {
      this.ruleStore=ruleStore;
      this.options={maxStages:Math.max(6,Math.min(64,options.mechanismGraphMaxStages|0||28))};
      this.builds=0; this.last=null;
    }

    _relations(){return this.ruleStore?.get?.('mechanismRelations')?.Relations||{};}
    _rules(){return this.ruleStore?.get?.('structuredReasoning')||{};}
    _slot(relation){
      const r=norm(relation); const map=this._relations();
      if(map[r])return map[r];
      if(/input|receive|source|from|require|depend/.test(r))return 'input';
      if(/process|transform|evaluate|select|compute|compare|use|learn|search/.test(r))return 'operation';
      if(/feed|trigger|cause|enable|pass/.test(r))return 'transition';
      if(/produce|generate|result|output|change/.test(r))return 'output';
      if(/update|return|replay|reinforce|correct|rollback|learn-from/.test(r))return 'feedback';
      if(/condition|if|when|unless/.test(r))return 'condition';
      if(/threshold|limit|safeguard|constraint|deadline/.test(r))return 'constraint';
      return 'operation';
    }

    build(promptText='', analysis={}, propositionGraph=null) {
      const started=performance.now();
      const requested=new Set([...(analysis?.requestedSlots||[]),analysis?.requestedSlot,analysis?.intent].map(norm));
      const lower=norm(promptText);
      const active=requested.has('mechanism')||requested.has('explanation')||/\bhow\b|\bwork(?:s|ing)?\b|\bprocess\b|\bmechanism\b|\bfunction(?:s|ing)?\b/.test(lower);
      const graph=propositionGraph||analysis?.propositionGraph||{};
      const evidence=[...(graph.propositions||[]).map(p=>({from:p.subject,relation:p.relation,to:p.object,confidence:p.certainty,source:p.source||'proposition'})),...(graph.supportEdges||[])];
      const explanation=analysis?.explanationPlan||null;
      for(const row of [...(explanation?.relations||[]),...(explanation?.relationEdges||[]),...(explanation?.evidence||[])]) evidence.push(row);
      const slots={input:[],operation:[],transition:[],output:[],feedback:[],condition:[],constraint:[]};
      const seen=new Set();
      for(const row of evidence){
        const from=norm(row?.from||row?.subject), relation=norm(row?.relation||row?.type), to=norm(row?.to||row?.object);
        if(!from&&!to)continue;
        const slot=this._slot(relation); const key=`${slot}|${from}|${relation}|${to}`; if(seen.has(key))continue; seen.add(key);
        slots[slot].push({from,relation:relation||'processes',to,confidence:clamp(row?.confidence??row?.strength??row?.score??.55),source:String(row?.source||'mechanism-evidence')});
      }
      for(const key of Object.keys(slots)) slots[key]=slots[key].sort((a,b)=>b.confidence-a.confidence).slice(0,this.options.maxStages);
      const order=this.ruleStore?.get?.('mechanismRelations')?.Ordering||['input','condition','operation','transition','output','feedback','constraint'];
      const stages=[];
      for(const slot of order) for(const row of slots[slot]||[]) stages.push({index:stages.length,slot,...row});
      const preferred=['input','operation','output'];
      const preferredHits=preferred.filter(k=>slots[k]?.length).length;
      const minimum=['operation','output'];
      const minimumHits=minimum.filter(k=>slots[k]?.length).length;
      const completeness=active?clamp(preferredHits/preferred.length*.65+minimumHits/minimum.length*.35):clamp(.5+Math.min(.5,stages.length*.04));
      const missing=active?preferred.filter(k=>!slots[k]?.length):[];
      const priorityTerms=uniq(stages.flatMap(s=>[s.from,s.relation,s.to])).slice(0,48);
      const result={
        version:1,mode:'generic-mechanism-graph',active,slots,stages:stages.slice(0,this.options.maxStages),missing,priorityTerms,
        subject:norm((analysis?.entities||[])[0]?.canonical||(analysis?.architectureComponents||[])[0]?.canonical||(graph.propositions||[])[0]?.subject||(analysis?.contentWords||[])[0]||''),
        requestedRoles:uniq([...(analysis?.requestedSlots||[]),analysis?.requestedSlot]),
        completeness,
        feedbackLoop:Boolean(slots.feedback.length),
        conditionsPresent:Boolean(slots.condition.length),
        elapsedMs:performance.now()-started
      };
      this.builds++;this.last=result;return result;
    }

    status(){return {ready:true,role:'generic-mechanism-process-graph',builds:this.builds,last:this.last?{active:this.last.active,stages:this.last.stages.length,completeness:this.last.completeness,missing:this.last.missing}:null};}
  }
  globalThis.VilotMechanismGraph=VilotMechanismGraph;
})();
