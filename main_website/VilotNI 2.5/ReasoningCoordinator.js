/**
 * VilotNI 2.5 - ReasoningCoordinator.js
 *
 * Foreground cognitive workspace. It does not replace the existing decoder or
 * reduce branch/candidate breadth. Before response generation it turns the
 * parsed task into multiple reasoning states, folds in Cognitive.js background
 * thoughts, graph paths, hypotheses, entities and architecture self-knowledge,
 * then uses the existing StateProcessor/MatrixEngine amplitude machinery to
 * amplify useful reasoning directions. The resulting plan becomes additional
 * evidence for the unchanged four-branch language system.
 */
(() => {
  'use strict';

  const clamp=(x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const norm=v=>String(v||'').toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'+.\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const uniq=values=>Array.from(new Set((values||[]).map(v=>norm(v)).filter(Boolean)));
  const tokens=v=>uniq(norm(v).split(/\s+/));
  const QUERY_FILLERS=new Set(['what','who','where','when','why','how','which','whom','whose','is','are','was','were','do','does','did','can','could','would','should','will','explain','describe','tell','show','work','works','working','about','please','the','a','an','of','to','for','and','or','in','on','with']);
  const overlap=(a,b)=>{
    const A=new Set(uniq(a)), B=new Set(uniq(b));
    if(!A.size||!B.size)return 0;
    let hit=0; for(const x of A) if(B.has(x)) hit++;
    return hit/Math.max(1,A.size+B.size-hit);
  };

  class VilotReasoningCoordinator {
    constructor(deps={}, options={}) {
      this.cognitive=deps.cognitive||null;
      this.graphReasoner=deps.graphReasoner||null;
      this.graph=deps.knowledgeGraph||null;
      this.hypotheses=deps.hypothesisStore||null;
      this.predictionError=deps.predictionError||null;
      this.evaluator=deps.learningEvaluator||null;
      this.stateProcessor=deps.stateProcessor||null;
      this.matrix=deps.matrix||null;
      this.entityResolver=deps.entityResolver||null;
      this.architectureKnowledge=deps.architectureKnowledge||null;
      this.conversationalCognition=deps.conversationalCognition||null;
      this.temporal=deps.temporalKnowledge||null;
      this.uncertainty=deps.uncertaintyState||null;
      this.selfImprovement=deps.selfImprovement||null;
      this.reasoningMemory=deps.reasoningMemory||null;
      this.reasoningQuality=deps.reasoningQuality||null;
      this.runtimeOptions=deps.runtimeOptions||options;
      this.options={
        enabled:options.reasoningCoordinatorEnabled!==false,
        maxSeeds:Math.max(8,Math.min(48,options.reasoningCoordinatorMaxSeeds|0||28)),
        maxPriorityConcepts:Math.max(12,Math.min(96,options.reasoningCoordinatorMaxPriorityConcepts|0||48)),
        graphDepth:Math.max(1,Math.min(4,options.reasoningCoordinatorGraphDepth|0||3)),
        graphPathLimit:Math.max(4,Math.min(32,options.reasoningCoordinatorGraphPathLimit|0||12)),
        hypothesisLimit:Math.max(4,Math.min(32,options.reasoningCoordinatorHypothesisLimit|0||12)),
        taskGuidanceStrength:Math.max(0.04,Math.min(0.24,Number(options.reasoningCoordinatorTaskGuidanceStrength)||0.14)),
        historyMax:Math.max(32,Math.min(512,options.reasoningCoordinatorHistory|0||128))
      };
      this.plans=0;
      this.totalMs=0;
      this.history=[];
      this.last=null;
    }

    _taskProfile(promptText,analysis={}) {
      const p=norm(promptText);
      const words=new Set(p.split(/\s+/).filter(Boolean));
      const role=String(analysis?.requestedSlot||analysis?.requestedRole||'content').toLowerCase();
      const why=words.has('why')||role==='cause';
      const how=words.has('how')||role==='mechanism'||role==='explanation';
      const compare=words.has('compare')||words.has('difference')||words.has('different')||words.has('similar')||words.has('relationship')||words.has('connection')||words.has('connect');
      const temporal=['when','before','after','during','history','historical','first','later','timeline'].some(w=>words.has(w))||role==='time'||role==='duration';
      const complex=analysis?.complexLanguage||{};
      const counterfactual=['if','would','could','hypothetical','suppose','imagine'].some(w=>words.has(w));
      const selfModel=Boolean(analysis?.selfModelQuery||(analysis?.architectureComponents||[]).length);
      const entity=Boolean((analysis?.entities||[]).length);
      return {role,why,how,compare:compare||Boolean(complex?.features?.comparison),temporal:temporal||Boolean(complex?.features?.temporal),counterfactual:counterfactual||Boolean(complex?.features?.conditional),selfModel,entity,multiClause:Boolean(complex?.features?.multiClause),embeddedQuestion:Boolean(complex?.features?.embeddedQuestion),negation:Boolean(complex?.features?.negation),structuralComplexity:Number(complex?.structuralComplexity||0)};
    }

    _seedConcepts(promptText,analysis={}) {
      return uniq([
        ...(analysis?.contentWords||[]),
        ...(analysis?.roots||[]),
        ...(analysis?.entityCanonicalTokens||[]),
        ...(analysis?.entityConcepts||[]),
        ...(analysis?.architectureConcepts||[]),
        ...(analysis?.architectureComponents||[]).map(row=>row.canonical),
        ...(analysis?.conversationCurrentAnchors||[]),
        ...(analysis?.conversationPriorAnchors||[]),
        ...(analysis?.cognitivePacket?.activeConcepts||[]),
        ...(analysis?.cognitiveState?.activeConcepts||[]),
        ...(analysis?.explanationPlan?.priorityTerms||[]),
        ...(analysis?.explanationPlan?.evidenceTerms||[]),
        ...(analysis?.propositionTerms||[]),
        ...(analysis?.propositionRelations||[]),
        ...(analysis?.propositionGraph?.terms||[]),
        ...(analysis?.mechanismGraph?.priorityTerms||[]),
        ...(analysis?.reasoningMemoryHints||[]).flatMap(row=>row?.priorityTerms||[]),
        ...(analysis?.entities||[]).flatMap(row=>[row.canonical,...(row.roles||[]),...(row.fields||[]),...(row.concepts||[])]),
        ...tokens(promptText).filter(w=>w.length>2 && !QUERY_FILLERS.has(w) && !/^\d+(?:\.\d+)?$/.test(w))
      ]).slice(0,this.options.maxSeeds);
    }

    _graphContext(seeds,deadlineAt) {
      if(!this.graphReasoner?.reason||!seeds.length)return {paths:[],derived:[],concepts:[],relations:[],diagnostics:null};
      const result=this.graphReasoner.reason(seeds,{maxDepth:this.options.graphDepth,deadlineAt});
      const concepts=[]; const relations=[];
      for(const row of (result?.paths||[]).slice(0,this.options.graphPathLimit)) {
        const verified=this.graphReasoner.verifyPath?.(row.id);
        for(const edge of verified?.edges||[]) {
          concepts.push(edge.from,edge.to);
          relations.push(edge.relation);
        }
      }
      for(const row of result?.derived||[]) { concepts.push(row.from,row.to); relations.push(row.relation); }
      return {...result,concepts:uniq(concepts),relations:uniq(relations)};
    }

    _hypothesisContext(seeds) {
      const seedSet=new Set(seeds);
      const rows=(this.hypotheses?.candidates?.(this.options.hypothesisLimit*3)||[]).filter(row=>{
        const terms=uniq([row.from,row.to,row.relation,...tokens(row.from),...tokens(row.to)]);
        return terms.some(x=>seedSet.has(x))||overlap(terms,seeds)>=0.12;
      }).slice(0,this.options.hypothesisLimit);
      return rows;
    }

    _state(id,mode,base,concepts,relations,extras={}) {
      const uniqueConcepts=uniq(concepts).slice(0,48);
      const uniqueRelations=uniq(relations).slice(0,24);
      return {
        id,mode,
        state:clamp(base),
        baseScore:clamp(base),
        concepts:uniqueConcepts,
        relations:uniqueRelations,
        knowledge:clamp(extras.knowledge??0.5),
        focus:clamp(extras.focus??0.5),
        transition:clamp(extras.transition??0.5),
        reliability:clamp(extras.reliability??0.5),
        context:clamp(extras.context??0.5),
        semantic:clamp(extras.semantic??0.5),
        novelty:clamp(extras.novelty??0.5),
        certainty:clamp(extras.certainty??base),
        evidenceCount:Math.max(0,Number(extras.evidenceCount)||0),
        source:extras.source||'reasoning-coordinator'
      };
    }

    _buildStates(profile,seeds,graphCtx,hypotheses,background,analysis) {
      const adaptive=this.runtimeOptions?.selfImprovementProfile||{};
      const strategyScale=Math.max(0.88,Math.min(1.28,Number(adaptive.reasoningScale)||1));
      const relationScale=Math.max(0.86,Math.min(1.30,Number(adaptive.relationScale)||1));
      const causalScale=Math.max(0.86,Math.min(1.30,Number(adaptive.causalScale)||1));
      const contextScale=Math.max(0.86,Math.min(1.26,Number(adaptive.contextScale)||1));
      const reflectionScale=Math.max(0.86,Math.min(1.30,Number(adaptive.reflectionScale)||1));
      const structureReasoningScale=Math.max(0.86,Math.min(1.32,Number(adaptive.structureReasoningScale)||1));
      const referenceScale=Math.max(0.86,Math.min(1.30,Number(adaptive.referenceScale)||1));
      const propositionScale=Math.max(0.86,Math.min(1.32,Number(adaptive.propositionScale)||1));
      const mechanismScale=Math.max(0.86,Math.min(1.34,Number(adaptive.mechanismScale)||1));
      const bgConcepts=uniq([
        ...(background?.attention||[]).map(row=>row.concept),
        ...(background?.thoughts||[]).flatMap(row=>[row.subject,row.strongestRelation]),
        background?.workspace?.focus,
        background?.workspace?.currentGoal?.target,
        ...(analysis?.cognitiveState?.activeConcepts||[]),
        ...(analysis?.cognitivePacket?.activeConcepts||[])
      ]);
      const structuralConcepts=uniq([...(analysis?.propositionTerms||[]),...(analysis?.propositionRelations||[]),...(analysis?.propositionGraph?.terms||[]),...(analysis?.complexLanguage?.unresolved||[])]);
      const propositionRelations=uniq((analysis?.propositionGraph?.propositions||[]).map(row=>row.relation));
      const mechanismTerms=uniq(analysis?.mechanismGraph?.priorityTerms||[]);
      const mechanismStageRelations=uniq((analysis?.mechanismGraph?.stages||[]).map(row=>row.relation));
      const archConcepts=uniq([...(analysis?.architectureConcepts||[]),...(analysis?.architectureComponents||[]).map(x=>x.canonical),...(analysis?.explanationPlan?.priorityTerms||[])]);
      const memoryHints=analysis?.reasoningMemoryHints||[];
      const memoryBoost=mode=>{
        const row=memoryHints.find(x=>String(x?.mode||'')===mode);
        return row?Math.min(.10,Math.max(0,Number(row.score||row.utility||0)-.45)*.18):0;
      };
      const entityConcepts=uniq([...(analysis?.entityConcepts||[]),...(analysis?.entities||[]).map(x=>x.canonical)]);
      const graphConcepts=graphCtx.concepts||[];
      const graphRelations=graphCtx.relations||[];
      const hypothesisConcepts=uniq(hypotheses.flatMap(row=>[row.from,row.relation,row.to]));
      const conversation=analysis?.conversationState||{};
      const conversationalAnchors=uniq([...(analysis?.conversationCurrentAnchors||[]),...(analysis?.conversationPriorAnchors||[])]);
      const softConversation=Math.min(0.18,Math.max(0.04,Number(conversation.softInfluence)||0.16));
      const responseFreedom=clamp(conversation.responseFreedom||0);
      const continuity=clamp(conversation.continuity||0);

      const semanticBase=clamp((0.56+Math.min(.22,graphConcepts.length*.012)+Math.min(.12,seeds.length*.008)+softConversation*responseFreedom*.16)*strategyScale);
      const guidance=this.options.taskGuidanceStrength;
      const mechanismDemand=profile.why||profile.how||['cause','mechanism','explanation'].includes(profile.role);
      const mechanismRelations=uniq([...(analysis?.explanationPlan?.relationTerms||[]),...mechanismStageRelations,...propositionRelations, ...graphRelations].filter(r=>/cause|result|produce|use|feed|receive|process|method|through|from|by|influence|affect|depend|part|implement|purpose|evaluate|select|update|return|transform|trigger/i.test(r)));
      const mechanismBase=clamp((0.54+(mechanismDemand?guidance*.65:0)+Math.min(.14,mechanismRelations.length*.025)+Math.min(.10,Number(analysis?.mechanismGraph?.completeness||0)*.10)+memoryBoost('causal-mechanism'))*causalScale*structureReasoningScale*mechanismScale);
      const relationDemand=profile.compare||seeds.length>1||profile.entity||profile.selfModel;
      const relationalBase=clamp((0.54+(relationDemand?guidance*.55:0)+Math.min(.14,(entityConcepts.length+archConcepts.length)*.01)+Math.min(.08,structuralConcepts.length*.006)+memoryBoost(profile.selfModel?'structural-self-model':'relational-structural'))*relationScale*propositionScale);
      const temporalBoost=profile.temporal?guidance*.25:0;
      const hypothesisDemand=profile.counterfactual||hypotheses.length>0||(background?.thoughts||[]).length>0;
      const hypothesisBase=clamp((0.50+(hypothesisDemand?guidance*.50:0)+Math.min(.16,hypotheses.length*.025)+Math.min(.08,bgConcepts.length*.01)+memoryBoost('hypothesis-reflection'))*reflectionScale);

      return [
        this._state('reason-semantic','semantic-network',clamp(semanticBase+memoryBoost('semantic-network')),[...seeds,...conversationalAnchors,...structuralConcepts,...graphConcepts,...archConcepts],uniq([...graphRelations,...propositionRelations]),{
          knowledge:0.76+Math.min(.18,graphConcepts.length*.01), focus:0.82, semantic:0.94, context:clamp((0.74+continuity*softConversation)*contextScale),
          reliability:0.82, transition:0.62, novelty:0.44, evidenceCount:(graphCtx.paths||[]).length+(graphCtx.derived||[]).length
        }),
        this._state('reason-mechanism',profile.temporal&&!mechanismDemand?'temporal-mechanism':'causal-mechanism',clamp(mechanismBase+temporalBoost),[...seeds,...conversationalAnchors,...mechanismTerms,...graphConcepts,...mechanismRelations],mechanismRelations,{
          knowledge:0.71, focus:mechanismDemand?0.95:0.58, semantic:0.84, context:clamp(0.79+continuity*softConversation),
          reliability:0.78, transition:0.82, novelty:0.48, evidenceCount:mechanismRelations.length+Number(analysis?.explanationPlan?.directEvidence||0)
        }),
        this._state('reason-relational',profile.selfModel?'structural-self-model':'relational-structural',relationalBase,[...seeds,...conversationalAnchors,...entityConcepts,...archConcepts,...graphConcepts],graphRelations,{
          knowledge:profile.selfModel?0.96:0.80, focus:relationDemand?0.91:0.62, semantic:0.88, context:clamp((0.86+continuity*softConversation)*contextScale),
          reliability:profile.selfModel?0.94:0.82, transition:0.72, novelty:0.42, evidenceCount:entityConcepts.length+archConcepts.length
        }),
        this._state('reason-hypothesis','hypothesis-reflection',hypothesisBase,[...seeds,...conversationalAnchors,...hypothesisConcepts,...bgConcepts],uniq(hypotheses.map(row=>row.relation)),{
          knowledge:0.58, focus:hypothesisDemand?0.78:0.48, semantic:0.72, context:clamp((0.88+continuity*softConversation+responseFreedom*softConversation*.25)*contextScale),
          reliability:hypotheses.length?0.66:0.52, transition:0.68, novelty:0.84, certainty:hypotheses.length?0.61:0.48,
          evidenceCount:hypotheses.length+(background?.thoughts||[]).length
        })
      ];
    }

    _collapse(states,profile) {
      if(!this.stateProcessor?.collapseCandidates||states.length<2) {
        const selected=[...states].sort((a,b)=>b.state-a.state)[0]||null;
        return {selected,state:selected?.state||0,certainty:selected?.certainty||0.5,score:selected?.state||0,diagnostics:{mode:'reasoning-direct-selection',stateCount:states.length}};
      }
      const queryVector=[
        0.18, // raw
        profile.selfModel?0.13:0.115, // knowledge: task hint stays deliberately weak
        0.14, // focus
        (profile.how||profile.why)?0.075:0.06, // transition/mechanism: soft steering only
        0.09, // reliability
        0.09, // context
        0.15, // semantic
        0.06, // certainty
        profile.counterfactual?0.04:0.03, // novelty: soft steering only
        0.08, // support
        0.04, // centrality
        0.07 // non-conflict
      ];
      return this.stateProcessor.collapseCandidates(states,{
        getState:s=>s.state,
        knowledgeScore:s=>s.knowledge,
        focusScore:s=>s.focus,
        transitionScore:s=>s.transition,
        reliabilityScore:s=>s.reliability,
        contextScore:s=>s.context,
        semanticScore:s=>s.semantic,
        noveltyScore:s=>s.novelty,
        certaintyScore:s=>s.certainty,
        compatible:(a,b)=>clamp(0.30+overlap(a.concepts,b.concepts)*0.52+overlap(a.relations,b.relations)*0.18),
        minAgreement:2,
        queryVector
      });
    }

    _priorityConcepts(states,collapse,seeds) {
      const amplitudes=collapse?.diagnostics?.amplitudeVector||[];
      const projections=collapse?.diagnostics?.queryProjectionVector||[];
      const weighted=[];
      states.forEach((state,index)=>{
        const w=clamp((state.state*.42)+(Number(amplitudes[index])||0)*.34+(Number(projections[index])||0)*.24);
        state.concepts.forEach((concept,rank)=>weighted.push({concept,weight:w*(1-Math.min(.55,rank*.015)),mode:state.mode}));
      });
      seeds.forEach((concept,rank)=>weighted.push({concept,weight:.98-Math.min(.28,rank*.012),mode:'prompt-anchor'}));
      const best=new Map();
      for(const row of weighted){const old=best.get(row.concept);if(!old||row.weight>old.weight)best.set(row.concept,row);}
      return Array.from(best.values()).sort((a,b)=>b.weight-a.weight).slice(0,this.options.maxPriorityConcepts);
    }

    plan(promptText,analysis={},options={}) {
      const started=performance.now();
      if(!this.options.enabled) return null;
      const deadlineAt=Number.isFinite(Number(options.deadlineAt))?Number(options.deadlineAt):Infinity;
      const profile=this._taskProfile(promptText,analysis);
      if(!analysis.reasoningMemoryHints && this.reasoningMemory?.retrieve) analysis.reasoningMemoryHints=this.reasoningMemory.retrieve(analysis,6);
      const seeds=this._seedConcepts(promptText,analysis);
      const background=this.cognitive?.foregroundContext?.(promptText,analysis)||null;
      const foreground=analysis?.cognitiveState||null;
      const graphCtx=this._graphContext(seeds,deadlineAt);
      const hypotheses=this._hypothesisContext(seeds);
      const states=this._buildStates(profile,seeds,graphCtx,hypotheses,background,analysis);
      const collapse=this._collapse(states,profile);
      const priority=this._priorityConcepts(states,collapse,seeds);
      const selected=collapse?.selected||states[0]||null;
      const unresolved=[];
      if((profile.why||profile.how)&&!states[1].relations.length) unresolved.push('mechanism-evidence');
      for(const slot of (analysis?.mechanismGraph?.missing||[])) if(!unresolved.includes(`mechanism:${slot}`)) unresolved.push(`mechanism:${slot}`);
      if(profile.compare&&seeds.length<2) unresolved.push('comparison-second-concept');
      if(profile.selfModel&&!analysis?.architectureComponents?.length) unresolved.push('self-model-component-resolution');
      if((collapse?.uncertainty??0)>0.48) unresolved.push('reasoning-state-uncertainty');
      for(const item of (analysis?.complexLanguage?.unresolved||[])) if(!unresolved.includes(item)) unresolved.push(item);
      const result={
        version:2,
        mode:'integrated-cognitive-amplitude-reasoning',
        task:profile,
        taskGuidanceStrength:this.options.taskGuidanceStrength,
        languageStructure:{clauseCount:analysis?.complexLanguage?.clauseCount||0,propositionCount:analysis?.complexLanguage?.propositionCount||0,requestedRoles:analysis?.complexLanguage?.requestedRoles||[],relations:analysis?.propositionRelations||[],references:analysis?.referenceLinks||[],structuralComplexity:Number(analysis?.structuralComplexity||0)},
        propositionGraph:analysis?.propositionGraph?{propositions:(analysis.propositionGraph.propositions||[]).slice(0,24),dependencies:(analysis.propositionGraph.dependencies||[]).slice(0,24),metrics:analysis.propositionGraph.metrics||null}:null,
        propositionUncertainty:(analysis?.propositionUncertainty||[]).slice(0,24),
        temporalAssessment:analysis?.temporalAssessment?{consistency:Number(analysis.temporalAssessment.consistency??1),ambiguity:Number(analysis.temporalAssessment.ambiguity??0),relations:(analysis.temporalAssessment.relations||[]).slice(0,24),contradictions:(analysis.temporalAssessment.contradictions||[]).slice(0,12)}:null,
        mechanismGraph:analysis?.mechanismGraph?{active:Boolean(analysis.mechanismGraph.active),stages:(analysis.mechanismGraph.stages||[]).slice(0,24),missing:(analysis.mechanismGraph.missing||[]),completeness:Number(analysis.mechanismGraph.completeness||0)}:null,
        reasoningMemoryHints:(analysis?.reasoningMemoryHints||[]).slice(0,6).map(row=>({mode:row.mode,score:row.score,utility:row.utility,successRate:row.successRate})),
        qualityPreflight:this.reasoningQuality?.preflight?.(analysis)||null,
        seeds,
        priorityConcepts:priority,
        priorityTerms:priority.map(row=>row.concept),
        branchModes:states.map(row=>row.mode),
        states:states.map((row,index)=>({
          id:row.id, mode:row.mode, baseScore:Number(row.baseScore.toFixed(4)),
          state:Number(row.state.toFixed(4)), concepts:row.concepts.slice(0,24), relations:row.relations.slice(0,16),
          evidenceCount:row.evidenceCount,
          amplitudeProbability:Number(collapse?.diagnostics?.amplitudeVector?.[index]||0),
          queryProjection:Number(collapse?.diagnostics?.queryProjectionVector?.[index]||0)
        })),
        selectedState:selected?{id:selected.id,mode:selected.mode,score:Number(collapse?.score||selected.state||0),certainty:Number(collapse?.certainty||selected.certainty||0)}:null,
        unresolved,
        graph:{paths:(graphCtx.paths||[]).slice(0,this.options.graphPathLimit),derived:(graphCtx.derived||[]).slice(0,8),diagnostics:graphCtx.diagnostics||null},
        hypotheses:hypotheses.map(row=>({from:row.from,relation:row.relation,to:row.to,confidence:row.confidence,status:row.status})),
        background:background?{
          attention:(background.attention||[]).slice(0,8),
          thoughts:(background.thoughts||[]).slice(0,6),
          workspace:background.workspace||null
        }:null,
        cognitive:foreground?{
          cycleCount:foreground.cycleCount||0,
          progress:foreground.progress??null,
          converged:Boolean(foreground.converged),
          activeConcepts:(foreground.activeConcepts||[]).slice(0,24),
          relations:(foreground.relations||[]).slice(0,20),
          unresolved:(foreground.unresolved||[]).slice(0,12)
        }:null,
        architecture:(analysis?.architectureComponents||[]).map(row=>({canonical:row.canonical,type:row.type,confidence:row.confidence})),
        conversation:analysis?.conversationState?{
          responseFreedom:Number(analysis.conversationState.responseFreedom||0),
          continuity:Number(analysis.conversationState.continuity||0),
          currentAnchors:(analysis.conversationCurrentAnchors||[]).slice(0,12),
          priorAnchors:(analysis.conversationPriorAnchors||[]).slice(0,10),
          dispositions:analysis.conversationState.dispositions||null,
          policy:analysis.conversationState.policy||'soft'
        }:null,
        matrixState:{
          mode:collapse?.diagnostics?.mode||null,
          amplitude:collapse?.diagnostics?.matrixVectorState||null,
          stateVector:collapse?.diagnostics?.stateVector||[],
          amplitudeVector:collapse?.diagnostics?.amplitudeVector||[],
          queryProjectionVector:collapse?.diagnostics?.queryProjectionVector||[],
          entropy:collapse?.diagnostics?.finalEntropy??null,
          uncertainty:collapse?.uncertainty??null,
          scoreMargin:collapse?.diagnostics?.scoreMargin??null
        },
        elapsedMs:performance.now()-started
      };
      this.plans++; this.totalMs+=result.elapsedMs; this.last=result;
      this.history.push({at:Date.now(),task:profile,selectedState:result.selectedState,unresolved:[...unresolved],elapsedMs:result.elapsedMs,priorityTerms:(result.priorityTerms||[]).slice(0,12)});
      if(this.history.length>this.options.historyMax)this.history.splice(0,this.history.length-this.options.historyMax);
      this.cognitive?.observeReasoningPlan?.(result);
      return result;
    }

    evaluateResponse(text,plan) {
      if(!plan)return {alignment:.5,priorityCoverage:.5,selectedStateCoverage:.5,modeCoverage:.5};
      const responseTokens=new Set(tokens(text));
      const rows=(plan.priorityConcepts||[]).slice(0,32);
      let weightedHit=0,weightedTotal=0;
      for(const row of rows){
        const ct=tokens(row.concept); const weight=Math.max(.05,Number(row.weight)||.2); weightedTotal+=weight;
        if(ct.some(t=>responseTokens.has(t)))weightedHit+=weight;
      }
      const priorityCoverage=weightedTotal?weightedHit/weightedTotal:.5;
      const selected=(plan.states||[]).find(row=>row.id===plan.selectedState?.id);
      const selectedTerms=uniq(selected?.concepts||[]).flatMap(tokens);
      const selectedStateCoverage=selectedTerms.length?selectedTerms.filter(t=>responseTokens.has(t)).length/selectedTerms.length:.5;
      const requested=plan.task||{};
      const mechanismWords=new Set(['because','by','through','using','from','cause','causes','process','method','result','produces','works']);
      const relationWords=new Set(['and','while','whereas','between','relationship','connects','uses','feeds','part']);
      let modeCoverage=.5;
      if(requested.why||requested.how) modeCoverage=[...responseTokens].some(t=>mechanismWords.has(t))?1:.25;
      else if(requested.compare) modeCoverage=[...responseTokens].some(t=>relationWords.has(t))?1:.3;
      else if(requested.selfModel) modeCoverage=priorityCoverage;
      const propositionTerms=uniq((plan?.propositionGraph?.propositions||[]).flatMap(row=>[row.subject,row.relation,row.object])).flatMap(tokens);
      const propositionCoverage=propositionTerms.length?propositionTerms.filter(t=>responseTokens.has(t)).length/propositionTerms.length:.5;
      const mechanismTerms=uniq((plan?.mechanismGraph?.stages||[]).flatMap(row=>[row.from,row.relation,row.to])).flatMap(tokens);
      const mechanismCoverage=plan?.mechanismGraph?.active?(mechanismTerms.length?mechanismTerms.filter(t=>responseTokens.has(t)).length/mechanismTerms.length:.25):.5;
      const alignment=clamp(priorityCoverage*.42+selectedStateCoverage*.20+modeCoverage*.14+propositionCoverage*.12+mechanismCoverage*.12);
      return {alignment,priorityCoverage,selectedStateCoverage,modeCoverage,propositionCoverage,mechanismCoverage};
    }

    status(){return {ready:true,role:'foreground-multi-state-cognitive-reasoning-coordinator',plans:this.plans,averageMs:this.plans?this.totalMs/this.plans:0,last:this.last?{task:this.last.task,selectedState:this.last.selectedState,unresolved:this.last.unresolved,elapsedMs:this.last.elapsedMs,priorityTerms:(this.last.priorityTerms||[]).slice(0,12)}:null};}
  }

  globalThis.VilotReasoningCoordinator=VilotReasoningCoordinator;
})();
