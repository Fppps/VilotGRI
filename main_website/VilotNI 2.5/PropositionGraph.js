/**
 * VilotNI 2.5 - PropositionGraph.js
 *
 * Canonical proposition layer above tokens and keywords. It keeps subject,
 * relation, object, conditions, time, polarity, certainty and provenance in one
 * shared representation that other reasoning systems can consume.
 */
(() => {
  'use strict';

  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const norm=v=>String(v||'').toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'+.\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const uniq=values=>Array.from(new Set((values||[]).map(norm).filter(Boolean)));
  const split=v=>uniq(norm(v).split(/\s+/));

  class VilotPropositionGraph {
    constructor(ruleStore=null, correlation=null, knowledgeGraph=null, options={}) {
      this.ruleStore=ruleStore;
      this.correlation=correlation;
      this.knowledgeGraph=knowledgeGraph;
      this.options={
        maxPropositions:Math.max(8,Math.min(96,options.propositionGraphMaxPropositions|0||48)),
        maxSupportEdges:Math.max(12,Math.min(192,options.propositionGraphMaxSupportEdges|0||72)),
        graphRelationLimit:Math.max(8,Math.min(96,options.propositionGraphRelationLimit|0||40))
      };
      this.builds=0; this.last=null;
    }

    _resolveTerm(value, analysis={}) {
      const clean=norm(value); if(!clean)return '';
      const refs=analysis?.complexLanguage?.references||analysis?.referenceLinks||[];
      const direct=refs.find(r=>norm(r.token)===clean&&r.resolved&&r.antecedent);
      return direct?norm(direct.antecedent):clean;
    }

    _relationOf(p={}) {
      return norm(p.predicateRoot||p.predicate||p.relationToParent||p.type||'related-to')||'related-to';
    }

    _qualifiers(p={}, clause={}, analysis={}) {
      const relation=norm(p.relationToParent||clause.relation||'');
      const parent=(analysis?.complexLanguage?.propositions||[]).find(x=>x.clauseId===p.parentClauseId)||null;
      const condition=/condition|if|unless/.test(relation)?norm(clause.text||p.object||''):'';
      const time=(analysis?.complexLanguage?.temporalModifiers||[]).map(x=>norm(x.marker||x.relation)).filter(Boolean);
      return {
        condition,
        time:time.slice(0,6),
        polarity:p.negated?-1:1,
        modality:norm(p.modal||clause.modal||''),
        parentProposition:parent?.id||null
      };
    }

    _supportEdges(seeds=[]) {
      const out=[]; const seen=new Set();
      const push=row=>{
        const from=norm(row?.from), relation=norm(row?.relation||row?.type||'related-to'), to=norm(row?.to);
        if(!from||!to||from===to)return;
        const key=`${from}|${relation}|${to}`; if(seen.has(key))return; seen.add(key);
        out.push({from,relation,to,strength:clamp(row?.strength??row?.score??.5),confidence:clamp(row?.confidence??.5),source:String(row?.source||row?.provenance||'support')});
      };
      for(const row of this.correlation?.relationCandidates?.(seeds,this.options.graphRelationLimit)||[]) push(row);
      for(const seed of seeds.slice(0,16)) {
        for(const edge of this.knowledgeGraph?.related?.(seed,8)||[]) push({...edge,source:'knowledge-graph'});
      }
      return out.slice(0,this.options.maxSupportEdges);
    }

    build(promptText='', analysis={}) {
      const started=performance.now();
      const complex=analysis?.complexLanguage||{};
      const clauses=complex.clauses||analysis?.clauseGraph||[];
      const sourceProps=complex.propositions||analysis?.propositions||[];
      const propositions=[]; const dependencies=[];

      for(const p of sourceProps.slice(0,this.options.maxPropositions)) {
        const clause=clauses.find(c=>c.id===p.clauseId)||{};
        const subject=this._resolveTerm(p.subject||clause.subject,analysis);
        const relation=this._relationOf(p);
        const object=this._resolveTerm(p.object||clause.object,analysis);
        const q=this._qualifiers(p,clause,analysis);
        if(!subject&&!relation&&!object)continue;
        const id=`pg-${propositions.length+1}`;
        propositions.push({
          id,
          sourceId:p.id||null,
          clauseId:p.clauseId||null,
          subject,
          relation,
          object,
          condition:q.condition,
          time:q.time,
          polarity:q.polarity,
          modality:q.modality,
          certainty:clamp(p.confidence??.62),
          requestedRole:norm(p.requestedRole||p.reasoningFamily||''),
          type:norm(p.type||p.relationToParent||'assertion'),
          source:'prompt-structure'
        });
        if(p.parentClauseId||q.parentProposition) dependencies.push({from:id,to:q.parentProposition||p.parentClauseId,relation:norm(p.relationToParent||'depends-on')});
      }

      // If a prompt is structurally simple, preserve a proposition rather than
      // forcing downstream systems to fall back to a flat keyword bag.
      if(!propositions.length) {
        const subject=norm((analysis?.entities||[])[0]?.canonical||(analysis?.contentWords||[])[0]||'');
        const relation=norm(analysis?.requestedSlot||analysis?.intent||'about');
        const object=norm((analysis?.contentWords||[]).slice(subject?1:0,6).join(' '));
        if(subject||object) propositions.push({id:'pg-1',sourceId:null,clauseId:null,subject,relation,object,condition:'',time:[],polarity:analysis?.hasNegation?-1:1,modality:'',certainty:clamp(analysis?.lexicalCoverage??.55),requestedRole:norm(analysis?.requestedSlot||''),type:'prompt-frame',source:'prompt-analysis'});
      }

      const seeds=uniq([
        ...(analysis?.contentWords||[]),...(analysis?.roots||[]),...(analysis?.entityConcepts||[]),
        ...propositions.flatMap(p=>[p.subject,p.relation,p.object])
      ]).slice(0,48);
      const supportEdges=this._supportEdges(seeds);
      const terms=uniq([...seeds,...supportEdges.flatMap(e=>[e.from,e.relation,e.to])]);
      const requestedRoles=uniq([...(analysis?.requestedSlots||[]),analysis?.requestedSlot,...propositions.map(p=>p.requestedRole)]);
      const unresolved=uniq([...(complex.unresolved||[])]);
      const referenceCoverage=(complex.references||[]).length?((complex.references||[]).filter(r=>r.resolved).length/(complex.references||[]).length):1;
      const propositionCompleteness=propositions.length?propositions.reduce((sum,p)=>sum+([p.subject,p.relation,p.object].filter(Boolean).length/3),0)/propositions.length:0;
      const dependencyCoverage=dependencies.length||propositions.length<=1?1:clamp(dependencies.length/Math.max(1,propositions.length-1));
      const result={
        version:1,
        mode:'canonical-proposition-graph',
        prompt:String(promptText||''),
        propositions,
        dependencies,
        supportEdges,
        terms,
        requestedRoles,
        unresolved,
        metrics:{
          propositionCompleteness:clamp(propositionCompleteness),
          referenceCoverage:clamp(referenceCoverage),
          dependencyCoverage:clamp(dependencyCoverage),
          structuralCoverage:clamp(propositionCompleteness*.52+referenceCoverage*.24+dependencyCoverage*.24)
        },
        elapsedMs:performance.now()-started
      };
      this.builds++; this.last=result; return result;
    }

    status(){return {ready:true,role:'canonical-proposition-graph',builds:this.builds,last:this.last?{propositions:this.last.propositions.length,supportEdges:this.last.supportEdges.length,structuralCoverage:this.last.metrics.structuralCoverage,elapsedMs:this.last.elapsedMs}:null};}
  }

  globalThis.VilotPropositionGraph=VilotPropositionGraph;
})();
