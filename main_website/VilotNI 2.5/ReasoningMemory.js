/**
 * VilotNI 2.5 - ReasoningMemory.js
 *
 * Stores reusable reasoning-strategy outcomes, not answers. Successful plans
 * can bias later planning for structurally similar tasks while failed plans are
 * down-weighted. No source files are rewritten.
 */
(() => {
  'use strict';
  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const norm=v=>String(v||'').toLowerCase().replace(/[^a-z0-9_+\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const uniq=v=>Array.from(new Set((v||[]).map(norm).filter(Boolean)));

  class VilotReasoningMemory {
    constructor(options={}) {
      this.options={capacity:Math.max(64,Math.min(4096,options.reasoningMemoryCapacity|0||768)),minUtility:clamp(options.reasoningMemoryMinUtility??.48)};
      this.entries=new Map();this.observations=0;this.last=null;
    }
    signature(analysis={}) {
      const f=analysis?.complexLanguage?.features||{};
      const roles=uniq([...(analysis?.requestedSlots||[]),analysis?.requestedSlot,analysis?.intent]).sort();
      const flags=[f.conditional?'conditional':'',f.temporal?'temporal':'',f.comparison?'comparison':'',f.negation?'negation':'',f.multiClause?'multi-clause':'',f.embeddedQuestion?'embedded':'',analysis?.mechanismGraph?.active?'mechanism-active':'',analysis?.counterfactual?'counterfactual':'',analysis?.entities?.length?'entity':'',analysis?.architectureComponents?.length?'self-model':''].filter(Boolean).sort();
      return [...roles,...flags].join('|')||'general';
    }
    retrieve(analysis={},limit=6){
      const sig=this.signature(analysis); const exact=[]; const related=[];
      const sigParts=new Set(sig.split('|'));
      for(const row of this.entries.values()){
        const parts=new Set(row.signature.split('|')); let hit=0; for(const x of sigParts)if(parts.has(x))hit++;
        const similarity=hit/Math.max(1,new Set([...sigParts,...parts]).size);
        const score=clamp(row.utility*.60+row.successRate*.24+similarity*.16);
        if(row.signature===sig) exact.push({...row,similarity:1,score}); else if(similarity>=.34) related.push({...row,similarity,score});
      }
      return [...exact,...related].sort((a,b)=>b.score-a.score).slice(0,Math.max(1,limit|0||6));
    }
    observe({analysis={},plan=null,evaluation=null,quality=null}={}){
      const signature=this.signature(analysis); const mode=norm(plan?.selectedState?.mode||evaluation?.selectedMode||'unknown');
      const key=`${signature}|${mode}`; const utility=clamp(quality?.overall??evaluation?.alignment??.5);
      const old=this.entries.get(key)||{key,signature,mode,observations:0,wins:0,utility:.5,successRate:.5,priorityTerms:[],updatedAt:0};
      const observations=old.observations+1; const wins=old.wins+(utility>=this.options.minUtility?1:0);
      const next={...old,observations,wins,utility:old.utility*.82+utility*.18,successRate:wins/observations,priorityTerms:uniq([...(plan?.priorityTerms||[]),...(old.priorityTerms||[])]).slice(0,24),updatedAt:Date.now()};
      this.entries.set(key,next);this.observations++;this.last=next;
      if(this.entries.size>this.options.capacity){const rows=[...this.entries.values()].sort((a,b)=>(a.utility*a.successRate)-(b.utility*b.successRate));const removeCount=this.entries.size-this.options.capacity;for(let i=0;i<removeCount;i++)this.entries.delete(rows[i].key);}
      return next;
    }
    exportState(){return {schemaVersion:1,observations:this.observations,entries:[...this.entries.values()].slice(-this.options.capacity),last:this.last};}
    importState(state){if(!state||Number(state.schemaVersion)!==1)return false;this.entries.clear();for(const row of state.entries||[])if(row?.key)this.entries.set(row.key,row);this.observations=Math.max(0,Number(state.observations)||0);this.last=state.last||null;return true;}
    reset(){this.entries.clear();this.observations=0;this.last=null;return true;}
    status(){return {ready:true,role:'reusable-reasoning-strategy-memory',entries:this.entries.size,observations:this.observations,last:this.last};}
  }
  globalThis.VilotReasoningMemory=VilotReasoningMemory;
})();
