/**
 * VilotNI 2.5 - ReasoningQuality.js
 *
 * Monitors whether a generated answer still covers the structured request. It
 * scores proposition coverage, mechanism completeness, multi-part coverage,
 * contradiction safety and semantic drift before the result is fed into
 * self-improvement.
 */
(() => {
  'use strict';
  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const norm=v=>String(v||'').toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'+.\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const STOP=new Set(['the','a','an','is','are','was','were','be','been','being','to','of','in','on','at','for','from','with','by','and','or','but','as','it','this','that','these','those']);
  const toks=v=>Array.from(new Set(norm(v).split(/\s+/).filter(x=>x&&x.length>1&&!STOP.has(x))));

  class VilotReasoningQuality {
    constructor(ruleStore=null, options={}){this.ruleStore=ruleStore;this.options={maxTerms:Math.max(16,Math.min(128,options.reasoningQualityMaxTerms|0||72))};this.evaluations=0;this.last=null;}
    _coverage(responseSet, values=[]){const terms=Array.from(new Set(values.flatMap(toks))).slice(0,this.options.maxTerms);if(!terms.length)return 1;let hit=0;for(const t of terms)if(responseSet.has(t))hit++;return hit/terms.length;}
    preflight(analysis={}){
      const p=analysis.propositionGraph||{},m=analysis.mechanismGraph||{};
      const unresolved=[...(p.unresolved||[]),...(m.missing||[])];
      const readiness=clamp((p.metrics?.structuralCoverage??.5)*.55+(m.active?(m.completeness??.4):.75)*.30+(1-Math.min(1,unresolved.length/6))*.15);
      return {readiness,unresolved,propositionCoverage:p.metrics?.structuralCoverage??.5,mechanismCompleteness:m.active?(m.completeness??.4):1};
    }
    evaluate(responseText='',analysis={},plan=null,contradictions=null){
      const responseSet=new Set(toks(responseText)); const p=analysis.propositionGraph||{},m=analysis.mechanismGraph||{};
      const promptCoverage=this._coverage(responseSet,[...(analysis.contentWords||[]),...(analysis.entityConcepts||[]),...(analysis.architectureConcepts||[])]);
      const propositionCoverage=this._coverage(responseSet,(p.propositions||[]).flatMap(x=>[x.subject,x.relation,x.object]));
      const requestedParts=[...(analysis.requestedSlots||[]),...(analysis.complexLanguage?.requestedRoles||[])];
      const multiPartCoverage=requestedParts.length<=1?1:clamp(new Set(requestedParts.filter(r=>[...responseSet].some(t=>norm(r).includes(t)||t.includes(norm(r))))).size/requestedParts.length+.35);
      const mechanismTerms=(m.stages||[]).flatMap(x=>[x.from,x.relation,x.to]);
      const mechanismTermCoverage=m.active?this._coverage(responseSet,mechanismTerms):1;
      const mechanismCompleteness=m.active?clamp((m.completeness??.4)*.46+mechanismTermCoverage*.54):1;
      const contradictionCount=Number(contradictions?.count??contradictions?.conflicts?.length??0); const contradictionSafety=clamp(1-Math.min(.85,contradictionCount*.22));
      const priority=(plan?.priorityTerms||[]).slice(0,40); const priorityCoverage=this._coverage(responseSet,priority);
      const anchor=new Set(toks((analysis.contentWords||[]).join(' '))); const responseTerms=[...responseSet];
      const unrelated=responseTerms.filter(t=>!anchor.has(t)&&!priority.some(pv=>toks(pv).includes(t))).length;
      const driftSafety=responseTerms.length?clamp(1-unrelated/responseTerms.length*.45):.5;
      const overall=clamp(promptCoverage*.18+propositionCoverage*.22+mechanismCompleteness*.22+multiPartCoverage*.10+contradictionSafety*.12+priorityCoverage*.10+driftSafety*.06);
      const result={overall,promptCoverage,propositionCoverage,mechanismCompleteness,multiPartCoverage,contradictionSafety,priorityCoverage,driftSafety,contradictionCount,missingMechanismSlots:m.missing||[]};
      this.evaluations++;this.last=result;return result;
    }
    status(){return {ready:true,role:'reasoning-quality-monitor',evaluations:this.evaluations,last:this.last};}
  }
  globalThis.VilotReasoningQuality=VilotReasoningQuality;
})();
