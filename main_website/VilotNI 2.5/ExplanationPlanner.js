/**
 * VilotNI 2.5 - ExplanationPlanner.js
 *
 * Generic explanation/mechanism planning. This is not a prompt-answer table.
 * It turns "explain how/why X..." style requests into a structured evidence
 * contract that the existing cognitive, graph, candidate and decoder systems
 * can consume without narrowing the architecture to a canned response.
 */
(() => {
  'use strict';

  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const norm=v=>String(v||'').toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'+.\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const uniq=values=>Array.from(new Set((values||[]).map(v=>norm(v)).filter(Boolean)));
  const STOP=new Set(['the','a','an','of','to','for','and','or','in','on','with','about','please','explain','describe','tell','show','me','how','why','what','which','who','does','do','did','is','are','was','were','can','could','would','should','will','work','works','working','over','time','it','itself','this','that']);
  const stem=w=>{
    let s=norm(w);
    if(!s)return '';
    if(s.length>6&&s.endsWith('ments'))s=s.slice(0,-5);
    if(s.length>5&&s.endsWith('ment'))s=s.slice(0,-4);
    if(s.length>5&&s.endsWith('ing'))s=s.slice(0,-3);
    if(s==='matrices')s='matrix';
    else if(s.length>4&&s.endsWith('ies'))s=s.slice(0,-3)+'y';
    else if(s.length>4&&s.endsWith('ed'))s=s.slice(0,-2);
    else if(s.length>4&&s.endsWith('es')){ if(/(?:ches|shes|sses|xes|zes)$/.test(s))s=s.slice(0,-2); else s=s.slice(0,-1); }
    else if(s.length>3&&s.endsWith('s')&&!s.endsWith('ss'))s=s.slice(0,-1);
    return s;
  };
  const words=v=>norm(v).split(/\s+/).filter(Boolean);
  const contentWords=v=>words(v).filter(w=>!STOP.has(w));

  class VilotExplanationPlanner {
    constructor(deps={}, options={}) {
      this.processor=deps.processor||null;
      this.graph=deps.knowledgeGraph||null;
      this.correlation=deps.correlation||null;
      this.architectureKnowledge=deps.architectureKnowledge||null;
      this.options={
        maxPriorityTerms:Math.max(16,Math.min(96,options.explanationPlannerMaxPriorityTerms|0||48)),
        maxEvidence:Math.max(8,Math.min(64,options.explanationPlannerMaxEvidence|0||28)),
        graphPerSeed:Math.max(4,Math.min(24,options.explanationPlannerGraphPerSeed|0||10)),
        correlationLimit:Math.max(4,Math.min(32,options.explanationPlannerCorrelationLimit|0||14))
      };
      this.plans=0; this.last=null;
    }

    _kind(promptText, analysis={}) {
      const text=norm(promptText);
      const first=words(text)[0]||'';
      const role=String(analysis?.requestedSlot||analysis?.questionState?.requestedRole||'').toLowerCase();
      const explanationMode=analysis?.explanationMode||analysis?.questionState?.explanationMode||null;
      if(explanationMode==='both'||role==='explanation')return 'explanation';
      if(role==='cause'||first==='why'||/\b(reason|cause|causes|caused)\b/.test(text))return 'cause';
      if(role==='mechanism'||first==='how'||/^explain\s+how\b/.test(text)||/\b(work|works|working|process|mechanism|improve|improves|learn|learns)\b/.test(text))return 'mechanism';
      if(first==='explain'||first==='describe')return 'explanation';
      return null;
    }

    _subjects(promptText, analysis={}) {
      const out=[];
      for(const row of analysis?.architectureComponents||[]) out.push(row.canonical);
      for(const row of analysis?.entities||[]) out.push(row.canonical);
      for(const v of analysis?.responseScaffold?.subject||[]) out.push(v);
      for(const v of analysis?.subjectWords||[]) out.push(v);
      const clause=analysis?.clauseState;
      for(const t of clause?.subjectTokens||[]) out.push(t.lower||t.surface);
      const raw=(analysis?.contentWords||[]).filter(Boolean);
      if(!out.length) out.push(...raw.slice(0,6));
      return uniq(out).slice(0,12);
    }

    _architectureEvidence(analysis={}) {
      const rows=[];
      for(const component of analysis?.architectureComponents||[]) {
        const subject=component.canonical;
        if(component.purpose) rows.push({from:subject,relation:'purpose',to:component.purpose,score:0.995,source:'architecture-purpose'});
        for(const input of component.inputs||[]) rows.push({from:subject,relation:'receives',to:input,score:0.96,source:'architecture-input'});
        for(const output of component.outputs||[]) rows.push({from:subject,relation:'produces',to:output,score:0.96,source:'architecture-output'});
        for(const [relation,targets] of Object.entries(component.relations||{})) {
          for(const target of Array.isArray(targets)?targets:[targets]) rows.push({from:subject,relation,to:target,score:0.97,source:'architecture-relation'});
        }
      }
      for(const row of analysis?.architectureEvidence||[]) rows.push(row);
      return rows;
    }

    _graphEvidence(seeds) {
      const rows=[];
      if(!this.graph?.related)return rows;
      for(const seed of seeds.slice(0,8)) {
        for(const edge of this.graph.related(seed,this.options.graphPerSeed)||[]) {
          rows.push({
            from:edge.from,
            relation:edge.relation,
            to:edge.to,
            score:clamp((edge.strength||0)*0.44+(edge.trust||0)*0.34+(edge.usefulness||0)*0.22),
            source:'knowledge-graph'
          });
        }
      }
      return rows;
    }

    _correlationEvidence(seeds) {
      if(!this.correlation?.relationCandidates)return [];
      return (this.correlation.relationCandidates(seeds,this.options.correlationLimit)||[]).map(row=>({
        from:row.from,
        relation:row.relation,
        to:row.to,
        score:clamp(row.score??((row.strength||0)*0.6+(row.confidence||0)*0.4)),
        source:'typed-correlation'
      }));
    }

    _cognitiveEvidence(analysis={}) {
      return (analysis?.cognitivePacket?.relations||analysis?.cognitiveState?.packet?.relations||[]).slice(0,24).map(row=>({
        from:row.from||row.a||row.source,
        relation:row.relation||row.type||'associated_with',
        to:row.to||row.b||row.target,
        score:clamp(row.score??row.strength??row.confidence??0.62),
        source:'cognitive-relation'
      })).filter(row=>row.from&&row.to);
    }

    _relationPreference(kind) {
      if(kind==='cause') return new Set(['cause','causes','caused_by','affects','produces','leads_to','results_in','depends_on','influences','changes']);
      if(kind==='mechanism') return new Set(['uses','used_by','implemented_by','implements','feeds','receives','produces','influences','depends_on','part_of','process','purpose','associated_with']);
      return new Set(['uses','feeds','receives','produces','influences','depends_on','part_of','purpose','cause','causes','affects','implemented_by','implements','associated_with']);
    }

    _rankEvidence(rows, kind, subjectStems) {
      const prefs=this._relationPreference(kind);
      const ranked=[]; const seen=new Set();
      for(const row of rows||[]) {
        if(!row?.from||!row?.to)continue;
        const key=`${norm(row.from)}|${norm(row.relation)}|${norm(row.to)}`;
        if(seen.has(key))continue; seen.add(key);
        const endpointStems=new Set([...contentWords(row.from),...contentWords(row.to)].map(stem));
        let subjectHit=0; for(const s of subjectStems) if(endpointStems.has(s)) subjectHit++;
        const relation=norm(row.relation).replace(/\s+/g,'_');
        const roleBoost=prefs.has(relation)?0.18:0;
        const score=clamp((Number(row.score)||0.55)*0.74+Math.min(0.18,subjectHit*0.09)+roleBoost);
        ranked.push({...row,score});
      }
      return ranked.sort((a,b)=>b.score-a.score).slice(0,this.options.maxEvidence);
    }

    plan(promptText, analysis={}) {
      const kind=this._kind(promptText,analysis);
      if(!kind)return null;
      const subjects=this._subjects(promptText,analysis);
      const seedTerms=uniq([...subjects,...(analysis?.contentWords||[]),...(analysis?.architectureConcepts||[]),...(analysis?.entityConcepts||[])]).slice(0,32);
      const subjectStems=new Set(subjects.flatMap(contentWords).map(stem).filter(Boolean));
      const evidence=this._rankEvidence([
        ...this._architectureEvidence(analysis),
        ...this._graphEvidence(seedTerms),
        ...this._correlationEvidence(seedTerms),
        ...this._cognitiveEvidence(analysis)
      ],kind,subjectStems);

      const relationTerms=[]; const evidenceTerms=[];
      for(const row of evidence) {
        relationTerms.push(row.relation);
        evidenceTerms.push(...contentWords(row.from),...contentWords(row.to));
      }
      const actionTerms=uniq([...(analysis?.responseScaffold?.predicate?[analysis.responseScaffold.predicate]:[]),...(analysis?.predicateWords||[]),...(analysis?.objectWords||[])]);
      const requiredSlots=kind==='cause'
        ? ['subject','cause','effect']
        : kind==='mechanism'
          ? ['subject','mechanism','result']
          : ['subject','mechanism','result'];
      const priorityTerms=uniq([...subjects,...actionTerms,...relationTerms,...evidenceTerms,...(analysis?.reasoningPriorityConcepts||[])]).slice(0,this.options.maxPriorityTerms);
      const directEvidence=evidence.filter(row=>row.score>=0.72).length;
      const confidence=clamp(0.46+Math.min(0.22,subjects.length*0.05)+Math.min(0.26,directEvidence*0.045));

      const plan={
        active:true,
        version:1,
        kind,
        subjects,
        actionTerms,
        requiredSlots,
        priorityTerms,
        relationTerms:uniq(relationTerms).slice(0,20),
        evidence:evidence.map(row=>({from:row.from,relation:row.relation,to:row.to,score:row.score,source:row.source})),
        evidenceTerms:uniq(evidenceTerms).slice(0,64),
        directEvidence,
        confidence,
        completeEnough:subjects.length>0&&directEvidence>=2
      };

      analysis.explanationPlan=plan;
      analysis.explanationEvidence=plan.evidence;
      analysis.reasoningPriorityConcepts=uniq([...(analysis.reasoningPriorityConcepts||[]),...priorityTerms]).slice(0,64);
      if(!analysis.requestedSlot||analysis.requestedSlot==='content') analysis.requestedSlot=kind==='cause'?'cause':'mechanism';
      this.plans++; this.last={at:Date.now(),kind,subjects:subjects.slice(0,6),evidence:directEvidence,confidence};
      return plan;
    }

    status(){return {role:'generic-structured-explanation-planner',plans:this.plans,last:this.last};}
  }

  globalThis.VilotExplanationPlanner=VilotExplanationPlanner;
})();
