/**
 * VilotNI 2.5 - ComplexLanguage.js
 *
 * Clause/proposition parser for long, nested prompts. It preserves structural
 * relationships above the keyword graph: embedded questions, subordinate and
 * coordinate clauses, conditionals, temporal order, comparison, negation and
 * local coreference. It provides structure only; it does not contain answers.
 */
(() => {
  'use strict';

  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const norm=v=>String(v||'').toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'+.\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const uniq=values=>Array.from(new Set((values||[]).map(v=>norm(v)).filter(Boolean)));
  const VERB_POS=new Set(['Verb','Aux','Modal','Auxiliary Verb','Modal Verb']);
  const NOMINAL_POS=new Set(['Noun','ProperNoun','Pronoun','Num','Adj','Det']);
  const FUNCTION_WORDS=new Set(['the','a','an','this','that','these','those','of','to','in','on','at','for','from','with','by','as','and','or','but','if','because','while','after','before','when','although','though','unless','that','which','who','whom','whose']);

  class VilotComplexLanguage {
    constructor(ruleStore=null, processor=null, options={}) {
      this.ruleStore=ruleStore;
      this.processor=processor;
      this.options={maxClauses:Math.max(4,Math.min(24,options.complexLanguageMaxClauses|0||16)),maxDepth:Math.max(2,Math.min(8,options.complexLanguageMaxDepth|0||6)),maxPropositions:Math.max(8,Math.min(48,options.complexLanguageMaxPropositions|0||32))};
      this.parses=0; this.last=null;
    }

    _data(name){return this.ruleStore?.get?.(name)||{};}
    _tokens(text,analysis={}){
      const existing=(analysis.tokens||[]).filter(t=>t&&t.pos!=='Punct').map((t,i)=>({index:Number.isFinite(t.index)?t.index:i,surface:String(t.surface||t.lower||''),lower:norm(t.lower||t.surface),pos:String(t.pos||'Other'),root:norm(t.root||''),entry:t.entry||null}));
      if(existing.length)return existing;
      return String(text||'').split(/\s+/).filter(Boolean).map((w,i)=>({index:i,surface:w,lower:norm(w),pos:'Other',root:norm(w),entry:null}));
    }
    _allTokens(text,analysis={}){
      if((analysis.tokens||[]).length)return analysis.tokens.map((t,i)=>({index:Number.isFinite(t.index)?t.index:i,surface:String(t.surface||t.lower||''),lower:norm(t.lower||t.surface),pos:String(t.pos||'Other'),root:norm(t.root||''),entry:t.entry||null}));
      const raw=String(text||'').match(/[A-Za-z0-9'+.\-]+|[.,!?;:()]/g)||[];
      return raw.map((w,i)=>({index:i,surface:w,lower:norm(w),pos:/^[.,!?;:()]$/.test(w)?'Punct':'Other',root:norm(w),entry:null}));
    }
    _phraseMap(object={}){
      return Object.entries(object||{}).map(([phrase,meta])=>({phrase:norm(phrase),parts:norm(phrase).split(/\s+/),meta:meta||{}})).sort((a,b)=>b.parts.length-a.parts.length);
    }
    _matchAt(words,index,rows){
      for(const row of rows){let ok=true;for(let j=0;j<row.parts.length;j++){if(words[index+j]?.lower!==row.parts[j]){ok=false;break;}}if(ok)return row;}
      return null;
    }
    _verbIndex(words,start=0,end=words.length){
      for(let i=start;i<end;i++){if(VERB_POS.has(words[i]?.pos))return i;}
      return -1;
    }
    _nominalText(words,start,end){
      const slice=words.slice(Math.max(0,start),Math.max(start,end));
      return slice.filter((w,index,arr)=>{
        if(NOMINAL_POS.has(w.pos)||(!FUNCTION_WORDS.has(w.lower)&&w.pos!=='Punct')) return true;
        // Preserve coordination inside a noun phrase, e.g. "people and entities".
        if(['and','or','nor'].includes(w.lower)) {
          const prev=arr[index-1], next=arr[index+1];
          return Boolean(prev&&next&&(NOMINAL_POS.has(prev.pos)||prev.pos==='Adj')&&(NOMINAL_POS.has(next.pos)||next.pos==='Adj'));
        }
        return false;
      }).map(w=>w.surface).join(' ').trim();
    }
    _predicateFrame(words,start,end){
      const vi=this._verbIndex(words,start,end);
      if(vi<0)return {subject:this._nominalText(words,start,end),predicate:'',object:'',predicateIndex:-1};
      const subject=this._nominalText(words,start,vi);
      let objectStart=vi+1;
      while(objectStart<end&&['Aux','Modal'].includes(words[objectStart]?.pos))objectStart++;
      const object=this._nominalText(words,objectStart,end);
      return {subject,predicate:words[vi]?.surface||'',predicateRoot:words[vi]?.root||words[vi]?.lower||'',object,predicateIndex:vi};
    }
    _relationForMarker(marker,syntax){
      const m=norm(marker);
      const sub=syntax?.Subordinators?.[m]; if(sub)return sub.relation||'subordinate';
      const coord=syntax?.Coordinators?.[m]; if(coord)return coord.relation||'coordination';
      const comp=syntax?.Comparison_Markers?.[m]; if(comp)return comp.relation||'comparison';
      return 'sequence';
    }
    _requestedRole(marker,syntax){return syntax?.Embedded_Question_Markers?.[norm(marker)]||null;}

    parse(text,analysis={}){
      const started=performance.now();
      const syntax=this._data('complexSyntax');
      const propRules=this._data('propositionRelations');
      const discourse=this._data('discourse');
      const references=this._data('references');
      const all=this._allTokens(text,analysis);
      const words=all.filter(t=>t.pos!=='Punct'||![',','.','?','!',';',':'].includes(t.surface));
      const cleanWords=all.filter(t=>t.pos!=='Punct');
      const subRows=this._phraseMap(syntax.Subordinators);
      const coordRows=this._phraseMap(syntax.Coordinators);
      const compRows=this._phraseMap(syntax.Comparison_Markers);
      const temporalModifierRows=this._phraseMap(syntax.Temporal_Modifiers);
      const directiveRows=this._phraseMap(syntax.Directive_Verbs);
      const markers=[];
      for(let i=0;i<cleanWords.length;i++){
        let match=this._matchAt(cleanWords,i,subRows)||this._matchAt(cleanWords,i,compRows)||this._matchAt(cleanWords,i,coordRows);
        if(match){
          const relation=this._relationForMarker(match.phrase,syntax);
          const isCoordinator=Boolean(syntax?.Coordinators?.[match.phrase]);
          // Do not split noun/adjective coordination into fake clauses.
          // "people and entities" is one object; "stores people and resolves
          // entities" is clause/predicate coordination because both sides carry
          // verbal structure.
          if(isCoordinator && ['addition','alternative','negative_addition'].includes(relation)) {
            const left=cleanWords.slice(Math.max(0,i-6),i);
            const right=cleanWords.slice(i+match.parts.length,Math.min(cleanWords.length,i+match.parts.length+6));
            const leftHasVerb=left.some(w=>VERB_POS.has(w?.pos));
            const rightHasVerb=right.some(w=>VERB_POS.has(w?.pos));
            const prev=cleanWords[i-1], next=cleanWords[i+match.parts.length];
            const nominalPair=Boolean(prev&&next&&(NOMINAL_POS.has(prev.pos)||prev.pos==='Adj')&&(NOMINAL_POS.has(next.pos)||next.pos==='Adj'));
            if(nominalPair && !(leftHasVerb && rightHasVerb)) continue;
          }
          markers.push({index:i,end:i+match.parts.length,marker:match.phrase,relation,meta:match.meta});i+=match.parts.length-1;
        }
      }
      const temporalModifiers=[];
      for(let i=0;i<cleanWords.length;i++){
        const tm=this._matchAt(cleanWords,i,temporalModifierRows);
        if(tm){temporalModifiers.push({index:i,end:i+tm.parts.length,marker:tm.phrase,relation:tm.meta?.relation||'temporal_progression'});i+=tm.parts.length-1;}
      }
      const embedded=[];
      for(let i=0;i<cleanWords.length;i++){
        const role=this._requestedRole(cleanWords[i].lower,syntax); if(!role)continue;
        const prior=cleanWords.slice(Math.max(0,i-4),i).map(w=>w.lower);
        const directive=prior.reverse().find(w=>syntax?.Directive_Verbs?.[w]?.allows_embedded_question);
        if(directive||i>0)embedded.push({index:i,marker:cleanWords[i].lower,role,directive:directive||null});
      }

      // Build spans from relation markers while preserving the marker as metadata.
      const cuts=new Set([0,cleanWords.length]); markers.forEach(m=>{cuts.add(m.index);cuts.add(m.end);}); embedded.forEach(e=>{cuts.add(e.index);cuts.add(e.index+1);});
      const sortedCuts=[...cuts].sort((a,b)=>a-b);
      const rawSpans=[];
      for(let i=0;i<sortedCuts.length-1;i++){
        const start=sortedCuts[i],end=sortedCuts[i+1]; if(end<=start)continue;
        const spanWords=cleanWords.slice(start,end); if(!spanWords.length)continue;
        if(spanWords.length===1 && (markers.some(m=>m.index===start&&m.end===end)||embedded.some(e=>e.index===start&&e.index+1===end)))continue;
        rawSpans.push({start,end,words:spanWords});
      }
      if(!rawSpans.length&&cleanWords.length)rawSpans.push({start:0,end:cleanWords.length,words:cleanWords});

      const clauses=[];
      for(let i=0;i<Math.min(this.options.maxClauses,rawSpans.length);i++){
        const span=rawSpans[i];
        const preceding=[...markers].filter(m=>m.end<=span.start).sort((a,b)=>b.end-a.end)[0]||null;
        const exactPre=markers.find(m=>m.end===span.start)||preceding;
        const exactEmbedded=embedded.find(e=>e.index+1===span.start)||null;
        const frame=this._predicateFrame(cleanWords,span.start,span.end);
        const negated=span.words.some(w=>(syntax.Negation_Scope_Markers||[]).includes(w.lower));
        const modal=span.words.find(w=>(syntax.Modal_Markers||[]).includes(w.lower))?.lower||null;
        clauses.push({
          id:`clause-${i+1}`,index:i,start:span.start,end:span.end,text:span.words.map(w=>w.surface).join(' '),
          words:span.words.map(w=>w.lower),subject:frame.subject,predicate:frame.predicate,predicateRoot:frame.predicateRoot,
          object:frame.object,negated,modal,marker:exactEmbedded?.marker||exactPre?.marker||null,relation:exactEmbedded?`embedded_${exactEmbedded.role}`:(exactPre?.relation||(i===0?'main':'sequence')),
          parentId:i===0?null:'clause-1',depth:i===0?0:1
        });
      }

      // Embedded WH clauses under directives are explicit clauses, not just keyword hints.
      // Example: "Explain how VilotNI 2.5 improves itself over time" gets a
      // directive clause plus an embedded mechanism clause whose subject is VilotNI 2.5.
      for (const emb of embedded.slice(0, 8)) {
        const start = emb.index + 1;
        if (start >= cleanWords.length) continue;
        const nextMarker = markers.filter(m => m.index > emb.index).sort((a,b)=>a.index-b.index)[0];
        const nextEmbedded = embedded.filter(e => e.index > emb.index).sort((a,b)=>a.index-b.index)[0];
        const ends=[nextMarker?.index,nextEmbedded?.index,cleanWords.length].filter(Number.isFinite);
        const end=Math.min(...ends);
        if (end <= start) continue;
        const alreadyCovered = clauses.some(c => c.start === start && c.end === end && String(c.relation).startsWith('embedded_'));
        if (alreadyCovered) continue;
        const frame = this._predicateFrame(cleanWords, start, end);
        clauses.push({
          id:`clause-${clauses.length+1}`, index:clauses.length, start, end,
          text:cleanWords.slice(start,end).map(w=>w.surface).join(' '),
          words:cleanWords.slice(start,end).map(w=>w.lower),
          subject:frame.subject, predicate:frame.predicate, predicateRoot:frame.predicateRoot, object:frame.object,
          negated:cleanWords.slice(start,end).some(w=>(syntax.Negation_Scope_Markers||[]).includes(w.lower)),
          modal:cleanWords.slice(start,end).find(w=>(syntax.Modal_Markers||[]).includes(w.lower))?.lower||null,
          marker:emb.marker, relation:`embedded_${emb.role}`, parentId:clauses[0]?.id||null, depth:1
        });
      }

      // If a leading conditional/subordinate clause precedes the embedded request,
      // attach it to that request instead of treating the condition as the main assertion.
      const primaryEmbedded = clauses.find(c => String(c.relation || '').startsWith('embedded_')) || null;
      if (primaryEmbedded) {
        for (const clause of clauses) {
          if (clause === primaryEmbedded) continue;
          if (/condition|temporal|concession|contrast/.test(String(clause.relation || '')) && !clause.parentId) clause.parentId = primaryEmbedded.id;
        }
        if (clauses[0] && /condition|temporal|concession|contrast/.test(String(clauses[0].relation || ''))) primaryEmbedded.parentId = null;
      }

      // Directive and embedded request structure.
      const firstDirective=cleanWords.map(w=>w.lower).find(w=>syntax?.Directive_Verbs?.[w])||null;
      const embeddedRoles=uniq(embedded.map(x=>x.role));
      const requestedRoles=uniq([...(analysis.requestedSlots||[]),...embeddedRoles]);
      const mainAct=firstDirective?syntax.Directive_Verbs[firstDirective]?.act:(analysis.intent||analysis.requestedSlot||'statement');

      const propositions=[];
      for(const clause of clauses){
        const relationRule=propRules?.Relations?.[clause.relation]||{};
        propositions.push({
          id:`prop-${propositions.length+1}`,clauseId:clause.id,type:clause.relation==='main'?'assertion':clause.relation,
          subject:clause.subject,predicate:clause.predicate,object:clause.object,negated:clause.negated,modal:clause.modal,
          relationToParent:clause.relation,parentClauseId:clause.parentId,reasoningFamily:relationRule.reasoning_family||null,
          requestedRole:String(clause.relation||'').startsWith('embedded_')?String(clause.relation).slice('embedded_'.length):null,
          confidence:clamp(clause.predicate?0.82:0.58)
        });
      }
      for(const emb of embedded.slice(0,8)){
        const embeddedClause=clauses.find(c=>String(c.relation||'')===`embedded_${emb.role}`&&c.start===emb.index+1) || clauses.find(c=>String(c.relation||'').startsWith('embedded_')) || null;
        const existing=propositions.find(p=>p.clauseId===embeddedClause?.id&&p.type===`embedded_${emb.role}`);
        if(existing){ existing.requestedRole=emb.role; existing.confidence=Math.max(existing.confidence||0,0.94); continue; }
        propositions.push({
          id:`prop-${propositions.length+1}`,
          clauseId:embeddedClause?.id||clauses[0]?.id||null,
          type:`embedded_${emb.role}`,
          subject:embeddedClause?.subject||'',
          predicate:embeddedClause?.predicate||emb.directive||mainAct||'',
          object:embeddedClause?.object||'',
          negated:Boolean(embeddedClause?.negated),
          modal:embeddedClause?.modal||null,
          relationToParent:`embedded_${emb.role}`,
          parentClauseId:embeddedClause?.parentId||clauses[0]?.id||null,
          reasoningFamily:emb.role,requestedRole:emb.role,confidence:0.94
        });
      }

      // System 33 reference resolution keeps competing antecedents rather than
      // collapsing every pronoun into the nearest noun. The JSON ontology may
      // target entities, events, processes, propositions or discourse spans.
      const refRows=[];
      const reflexives=references?.Reflexives||{};
      const pronouns={...(references?.Pronouns||{}),...(references?.Personal_Pronouns||{}),...(references?.Demonstratives||{}),...(references?.Possessives||{})};
      const candidateCfg=references?.Candidate_Scoring||{};
      const ambiguityCfg=references?.Ambiguity||{};
      const scoreCandidate=(cand,meta={})=>clamp(
        (cand.recency??.5)*(candidateCfg.recency??.26)+
        (cand.roleScore??.5)*(candidateCfg.grammatical_role??.17)+
        (cand.numberScore??.8)*(candidateCfg.number_agreement??.12)+
        (cand.personScore??.8)*(candidateCfg.person_agreement??.08)+
        (cand.classScore??.7)*(candidateCfg.semantic_class??.18)+
        (cand.focusScore??.5)*(candidateCfg.discourse_focus??.12)+
        (cand.parallelScore??.5)*(candidateCfg.parallel_structure??.07)+
        (meta.reflexiveSameSubject?(candidateCfg.same_clause_reflexive_bonus??.34):0)+
        (meta.explicitName?(candidateCfg.explicit_name_bonus??.22):0)-
        (meta.crossSubject?(candidateCfg.cross_subject_penalty??.20):0)-
        (meta.semanticMismatch?(candidateCfg.semantic_mismatch_penalty??.30):0)
      );
      const candidateFor=(value,strategy,recency,roleScore=.7,classScore=.7,meta={})=>({antecedent:String(value||'').trim(),strategy,recency,roleScore,classScore,score:scoreCandidate({recency,roleScore,classScore},meta)});
      for(let i=0;i<cleanWords.length;i++){
        const token=cleanWords[i];
        const lower=token.lower;
        if(!reflexives[lower]&&!pronouns[lower])continue;
        const prev=cleanWords[i-1], next=cleanWords[i+1];
        if (lower === 'that' && (next?.pos === 'Det' || next?.pos === 'Noun' || next?.pos === 'ProperNoun') && (prev?.pos === 'Verb' || prev?.pos === 'Aux')) continue;
        const clause=clauses.find(c=>i>=c.start&&i<c.end)||null;
        const candidates=[];
        if(reflexives[lower]&&clause?.subject)candidates.push(candidateFor(clause.subject,'same-clause-subject',1,1,1,{reflexiveSameSubject:true}));
        if(!reflexives[lower]){
          if(clause?.subject)candidates.push(candidateFor(clause.subject,'same-clause-subject',.96,.96,.85));
          if(clause?.object)candidates.push(candidateFor(clause.object,'same-clause-object',.92,.78,.78));
        }
        const priorClauses=clauses.filter(c=>c.end<=i).sort((a,b)=>b.end-a.end).slice(0,4);
        priorClauses.forEach((row,idx)=>{
          if(row.subject)candidates.push(candidateFor(row.subject,'previous-clause-subject',Math.max(.52,.90-idx*.10),.88,.82));
          if(row.object)candidates.push(candidateFor(row.object,'previous-clause-object',Math.max(.46,.80-idx*.10),.68,.76));
          if((references?.Demonstratives?.[lower]?.classes||[]).some(x=>['event','proposition','process','discourse'].includes(x))&&row.text){
            candidates.push(candidateFor(row.text,'previous-clause-event-or-proposition',Math.max(.56,.92-idx*.10),.82,.92));
          }
        });
        for(const [idx,entity] of (analysis?.entities||[]).slice(0,4).entries()){
          const value=entity?.canonical||entity?.surface||'';
          if(value)candidates.push(candidateFor(value,'recent-entity',Math.max(.48,.78-idx*.08),.70,.88,{explicitName:true}));
        }
        // De-duplicate candidates and keep the strongest score for each span.
        const byAntecedent=new Map();
        for(const row of candidates){
          const key=norm(row.antecedent); if(!key)continue;
          if(!byAntecedent.has(key)||row.score>byAntecedent.get(key).score)byAntecedent.set(key,row);
        }
        const ranked=Array.from(byAntecedent.values()).sort((a,b)=>b.score-a.score).slice(0,Math.max(2,Number(ambiguityCfg.keep_top_candidates)||4));
        const top=ranked[0]||null, second=ranked[1]||null;
        const minScore=Number(ambiguityCfg.resolve_min_score??.48), margin=Number(ambiguityCfg.resolve_margin??.10);
        const decisive=Boolean(top&&top.score>=minScore&&(!second||(top.score-second.score)>=margin||reflexives[lower]));
        refRows.push({
          token:lower,index:i,clauseId:clause?.id||null,
          antecedent:decisive?top.antecedent:null,
          strategy:decisive?top.strategy:'ambiguous-or-unresolved',
          resolved:decisive,
          confidence:decisive?clamp(.45+top.score*.55):clamp(top?.score??.25),
          candidates:ranked.map(row=>({antecedent:row.antecedent,strategy:row.strategy,score:Number(row.score.toFixed(4))})),
          ambiguity:ranked.length>1?clamp(1-Math.max(0,(top?.score||0)-(second?.score||0))*2.5):(decisive?0:.65)
        });
      }
      // Multi-word event/proposition/discourse references such as "this process"
      // or "what happened" receive clause-level antecedents as candidates.
      const phraseRefs={...(references?.Event_And_Proposition_References||{}),...(references?.Anaphoric_Nominals||{})};
      const lowerText=cleanWords.map(w=>w.lower).join(' ');
      for(const [phrase,rule] of Object.entries(phraseRefs)){
        const key=norm(phrase); if(!key||!lowerText.includes(key))continue;
        const prior=clauses.slice(0,-1).filter(c=>c.text).at(-1)||clauses[0]||null;
        if(!prior)continue;
        const existing=refRows.some(r=>r.token===key); if(existing)continue;
        refRows.push({token:key,index:Math.max(0,lowerText.indexOf(key)),clauseId:prior.id||null,antecedent:prior.text,strategy:`${rule?.class||rule?.selection||'discourse'}-reference`,resolved:true,confidence:.82,candidates:[{antecedent:prior.text,strategy:'previous-clause',score:.82}],ambiguity:.12});
      }

      const directiveWords=new Set(Object.keys(syntax?.Directive_Verbs||{}));
      const propositionTerms=uniq(propositions.flatMap(p=>[p.subject,p.predicate,p.object,p.type,p.reasoningFamily]).flatMap(v=>norm(v).split(/\s+/))).filter(w=>w.length>1&&!FUNCTION_WORDS.has(w)&&!directiveWords.has(w));
      const relationTerms=uniq([...propositions.map(p=>p.relationToParent).filter(Boolean),...temporalModifiers.map(m=>m.relation)]);
      const unresolved=[];
      for(const r of refRows)if(!r.resolved)unresolved.push(`reference:${r.token}`);
      if(requestedRoles.includes('cause')&&!relationTerms.some(r=>/cause|result/.test(r)))unresolved.push('cause-link');
      if(requestedRoles.includes('mechanism')&&!clauses.some(c=>c.predicate))unresolved.push('mechanism-predicate');
      const conditional=markers.some(m=>/condition/.test(m.relation));
      const temporal=markers.some(m=>/temporal|simultaneous/.test(m.relation)) || temporalModifiers.length>0;
      const comparison=markers.some(m=>/comparison|contrast|similarity|difference/.test(m.relation)) || cleanWords.some(w=>['compare','comparison','difference','different','similar','similarity','than','whereas'].includes(w.lower));
      const weights=syntax?.Complexity_Weights||{};
      const structuralComplexity=clamp(
        (clauses.length-1)*(weights.clause||.12)+markers.filter(m=>!['addition','alternative'].includes(m.relation)).length*(weights.subordination||.18)+embedded.length*(weights.embedded_question||.16)+refRows.length*(weights.reference||.08)+(cleanWords.some(w=>(syntax.Negation_Scope_Markers||[]).includes(w.lower))?(weights.negation||.08):0)+(comparison?(weights.comparison||.12):0)+(conditional?(weights.conditional||.16):0)+(temporal?(weights.temporal||.10):0)
      );
      const result={
        version:1,mode:'multi-clause-proposition-graph',mainAct,requestedRoles,embeddedQuestions:embedded,
        clauses,propositions:propositions.slice(0,this.options.maxPropositions),references:refRows,markers,temporalModifiers,
        propositionTerms,relationTerms,unresolved,features:{conditional,temporal,comparison,negation:cleanWords.some(w=>(syntax.Negation_Scope_Markers||[]).includes(w.lower)),coordination:markers.some(m=>['addition','alternative'].includes(m.relation)),embeddedQuestion:embedded.length>0,multiClause:clauses.length>1},
        structuralComplexity,clauseCount:clauses.length,propositionCount:propositions.length,dependencyDepth:clauses.reduce((m,c)=>Math.max(m,c.depth||0),0),elapsedMs:performance.now()-started
      };
      this.parses++; this.last=result; return result;
    }

    status(){return {ready:true,role:'complex-language-and-proposition-parser',parses:this.parses,last:this.last?{clauseCount:this.last.clauseCount,propositionCount:this.last.propositionCount,structuralComplexity:this.last.structuralComplexity,requestedRoles:this.last.requestedRoles}:null};}
  }
  globalThis.VilotComplexLanguage=VilotComplexLanguage;
})();
