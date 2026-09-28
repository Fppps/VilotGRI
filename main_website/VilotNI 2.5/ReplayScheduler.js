/**
 * VilotNI 2.5 - ReplayScheduler.js
 * System 34 foundation capacity expansion.
 *
 * Replay is now a reasoning curriculum, not only a spaced queue. Weak results
 * are classified into failure families, assigned structurally varied replay
 * cases, and paired with control cases so one fix cannot silently damage an
 * unrelated capability.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));
  const norm = value => String(value || '').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g, ' ').replace(/\s+/g, ' ').trim();

  class VilotReplayScheduler {
    constructor(options = {}, ruleStore = null) {
      this.ruleStore = ruleStore;
      this.options = {
        maxItems: Math.max(128, options.learningReplayMaxItems | 0 || 2048),
        baseIntervalMs: Math.max(250, options.learningReplayBaseIntervalMs | 0 || 1200),
        maxIntervalMs: Math.max(5000, options.learningReplayMaxIntervalMs | 0 || 300000),
        maxVariantsPerItem: Math.max(2, Math.min(16, options.learningReplayMaxVariantsPerItem | 0 || 10)),
        minControlRatio: clamp(options.learningReplayMinControlRatio ?? 0.25, 0.10, 0.60)
      };
      this.items = new Map();
      this.scheduled = 0;
      this.replayed = 0;
      this.failuresScheduled = 0;
      this.controlReplays = 0;
      this.curriculumCounts = {};
      this.outcomeByCurriculum = {};
      this.last = null;
    }

    _policy() {
      const value = this.ruleStore?.replayPolicy?.() || this.ruleStore?.safeDataset?.('replayPolicy') || this.ruleStore?.get?.('replayPolicy') || {};
      const policy = value && typeof value === 'object' ? value : {};
      if (!policy.Curricula || typeof policy.Curricula !== 'object' || Array.isArray(policy.Curricula)) policy.Curricula = {};
      return policy;
    }

    _ensureRuntimeState() {
      if (!(this.items instanceof Map)) this.items = new Map();
      if (!this.curriculumCounts || typeof this.curriculumCounts !== 'object' || Array.isArray(this.curriculumCounts)) this.curriculumCounts = {};
      if (!this.outcomeByCurriculum || typeof this.outcomeByCurriculum !== 'object' || Array.isArray(this.outcomeByCurriculum)) this.outcomeByCurriculum = {};
      this.scheduled = Math.max(0, Number(this.scheduled) || 0);
      this.replayed = Math.max(0, Number(this.replayed) || 0);
      this.failuresScheduled = Math.max(0, Number(this.failuresScheduled) || 0);
      this.controlReplays = Math.max(0, Number(this.controlReplays) || 0);
      return this;
    }

    _priority(meta = {}) {
      const weights = this._policy()?.Priority_Weights || {};
      const pairs = [
        ['uncertainty', meta.uncertainty ?? 0.5, weights.uncertainty ?? 0.18],
        ['usefulness', meta.usefulness ?? 0.5, weights.usefulness ?? 0.12],
        ['predictionError', meta.predictionError ?? 0, weights.predictionError ?? 0.22],
        ['novelty', meta.novelty ?? 0.5, weights.novelty ?? 0.10],
        ['retrievalFrequency', meta.retrievalFrequency ?? 0, weights.retrievalFrequency ?? 0.06],
        ['contradictionPressure', meta.contradictionPressure ?? 0, weights.contradictionPressure ?? 0.14],
        ['diagnosisPressure', meta.diagnosisPressure ?? meta.pressure ?? 0, weights.diagnosisPressure ?? 0.12],
        ['controlNeed', meta.controlNeed ?? 0, weights.controlNeed ?? 0.06]
      ];
      let sum = 0, total = 0;
      for (const [, value, weight] of pairs) { sum += clamp(value) * clamp(weight); total += clamp(weight); }
      return clamp(total ? sum / total : 0.5);
    }

    _curriculumFor(meta = {}, payload = {}) {
      const key = norm(meta.curriculum || meta.diagnosis || meta.failureType || payload.diagnosis || payload.failureType || '').replace(/\s+/g, '-');
      const curricula = this._policy()?.Curricula || {};
      if (key && curricula[key]) return key;
      const aliases = {
        'reference-resolution-deficit': 'reference-or-dependency-resolution',
        'reference-resolution': 'reference-or-dependency-resolution',
        'grammar-deficit': 'surface-grammar',
        'complex-language': 'complex-language-coverage',
        'mechanism': 'mechanism-completeness',
        'proposition': 'proposition-coverage',
        'temporal': 'temporal-reasoning',
        'calibration-gap': 'confidence-calibration',
        'drift': 'semantic-drift'
      };
      if (aliases[key] && curricula[aliases[key]]) return aliases[key];
      return key || 'general';
    }

    _curriculumSpec(name) {
      return this.ruleStore?.replayCurriculum?.(name) || this._policy()?.Curricula?.[name] || { families: ['generalization','control'], variant_count: 4, control_ratio: this.options.minControlRatio };
    }

    _variantDescriptors(curriculum, payload = {}) {
      const spec = this._curriculumSpec(curriculum);
      const families = Array.isArray(spec.families) && spec.families.length ? spec.families : ['general'];
      const count = Math.max(1, Math.min(this.options.maxVariantsPerItem, Number(spec.variant_count) || families.length));
      const out = [];
      for (let i = 0; i < count; i++) {
        out.push({
          family: families[i % families.length],
          variantIndex: i,
          preserveMeaning: true,
          mutateStructure: true,
          sourcePrompt: String(payload.promptText || payload.prompt || '').slice(0, 512)
        });
      }
      return out;
    }

    _evictIfNeeded() {
      if (this.items.size < this.options.maxItems) return;
      const rows = Array.from(this.items.values()).sort((a, b) => {
        const masteryA = a.strength * (1 + Math.min(6, a.replays) * 0.08);
        const masteryB = b.strength * (1 + Math.min(6, b.replays) * 0.08);
        if (a.priority !== b.priority) return a.priority - b.priority;
        return masteryB - masteryA;
      });
      const victim = rows[0];
      if (victim) this.items.delete(victim.key);
    }

    schedule(key, payload = {}, meta = {}) {
      this._ensureRuntimeState();
      const id = String(key || '').trim();
      if (!id) return null;
      if (!this.items.has(id)) this._evictIfNeeded();

      const curriculum = this._curriculumFor(meta, payload);
      const spec = this._curriculumSpec(curriculum);
      const old = this.items.get(id) || {
        key: id,
        payload: {},
        priority: 0,
        strength: 0.25,
        replays: 0,
        successes: 0,
        failures: 0,
        controlSuccesses: 0,
        controlFailures: 0,
        intervalMs: this.options.baseIntervalMs,
        dueAt: 0,
        createdAt: Date.now(),
        lastReplayAt: 0,
        curriculum,
        kind: norm(meta.replayKind || payload.replayKind || payload.kind || 'knowledge') || 'knowledge',
        variants: [],
        variantCursor: 0,
        controlRatio: clamp(spec.control_ratio ?? this.options.minControlRatio, this.options.minControlRatio, 0.65),
        history: []
      };

      old.payload = { ...old.payload, ...payload };
      old.curriculum = curriculum || old.curriculum;
      old.variants = this._variantDescriptors(old.curriculum, old.payload);
      old.controlRatio = clamp(spec.control_ratio ?? old.controlRatio ?? this.options.minControlRatio, this.options.minControlRatio, 0.65);
      old.priority = Math.max(old.priority * 0.78, this._priority(meta));
      old.meta = { ...(old.meta || {}), ...meta };
      old.kind = norm(meta.replayKind || payload.replayKind || old.kind || 'knowledge') || 'knowledge';

      const proposedDue = Date.now() + Math.max(50, Math.round(this.options.baseIntervalMs * (0.35 + (1 - old.priority) * 0.65)));
      old.dueAt = old.dueAt > 0 ? Math.min(old.dueAt, proposedDue) : proposedDue;
      this.items.set(id, old);
      this.scheduled++;
      this.curriculumCounts[old.curriculum] = (this.curriculumCounts[old.curriculum] || 0) + 1;
      this.last = { action: 'schedule', key: id, curriculum: old.curriculum, priority: old.priority, dueAt: old.dueAt, variants: old.variants.length };
      return old;
    }

    scheduleFailure(diagnosis = {}, payload = {}, meta = {}) {
      const type = norm(diagnosis.type || diagnosis.failureType || 'general').replace(/\s+/g, '-');
      const promptKey = norm(payload.promptText || payload.prompt || '').slice(0, 120).replace(/\s+/g, '_');
      const key = `failure:${type}:${promptKey || Date.now()}`;
      this.failuresScheduled++;
      return this.schedule(key, { ...payload, diagnosis: type, diagnosisDetails: diagnosis }, {
        ...meta,
        diagnosis: type,
        replayKind: 'diagnostic',
        diagnosisPressure: clamp(diagnosis.pressure ?? diagnosis.deficit ?? 0.65),
        predictionError: clamp(meta.predictionError ?? diagnosis.predictionError ?? 0.5),
        uncertainty: clamp(meta.uncertainty ?? diagnosis.uncertainty ?? 0.6),
        usefulness: clamp(meta.usefulness ?? 0.8)
      });
    }

    _makeReplayCase(item, asControl = false) {
      const variants = item.variants?.length ? item.variants : this._variantDescriptors(item.curriculum, item.payload);
      const variant = variants[item.variantCursor % Math.max(1, variants.length)] || { family: 'general', variantIndex: 0 };
      item.variantCursor = (item.variantCursor + 1) % Math.max(1, variants.length);
      return {
        ...item,
        replayCase: {
          curriculum: item.curriculum,
          family: asControl ? 'control' : variant.family,
          variantIndex: variant.variantIndex,
          isControl: asControl,
          preserveMeaning: true,
          mutateStructure: !asControl,
          sourcePrompt: variant.sourcePrompt || String(item.payload?.promptText || item.payload?.prompt || '')
        }
      };
    }

    nextBatch(limit = 2, now = Date.now()) {
      const max = Math.max(1, limit | 0 || 2);
      const due = Array.from(this.items.values())
        .filter(item => item.dueAt <= now && item.kind !== 'diagnostic')
        .sort((a, b) => b.priority !== a.priority ? b.priority - a.priority : a.dueAt - b.dueAt);
      if (!due.length) return [];

      const rows = [];
      const requiredControls = max >= 3 ? Math.max(1, Math.floor(max * this.options.minControlRatio)) : 0;
      let controls = 0;
      for (const item of due) {
        if (rows.length >= max) break;
        const wantsControl = controls < requiredControls && rows.length >= max - requiredControls;
        rows.push(this._makeReplayCase(item, wantsControl));
        if (wantsControl) controls++;
      }
      return rows;
    }


    nextDiagnosticBatch(limit = 2, now = Date.now()) {
      const max = Math.max(1, limit | 0 || 2);
      return Array.from(this.items.values())
        .filter(item => item.dueAt <= now && item.kind === 'diagnostic')
        .sort((a, b) => b.priority !== a.priority ? b.priority - a.priority : a.dueAt - b.dueAt)
        .slice(0, max)
        .map(item => this._makeReplayCase(item, false));
    }

    complete(key, result = {}) {
      this._ensureRuntimeState();
      const item = this.items.get(String(key || ''));
      if (!item) return false;
      const success = result.success !== false;
      const isControl = Boolean(result.isControl || result.replayCase?.isControl);
      const residualError = clamp(result.predictionError ?? 0);
      const targetScore = clamp(result.targetScore ?? (success ? 0.8 : 0.35));
      const controlScore = clamp(result.controlScore ?? (isControl ? targetScore : 0.8));

      item.replays++;
      item.lastReplayAt = Date.now();
      if (isControl) {
        this.controlReplays++;
        if (success) item.controlSuccesses++; else item.controlFailures++;
      } else if (success) item.successes++; else item.failures++;

      item.strength = clamp(item.strength + (success ? 0.10 : -0.08) + (1 - residualError) * 0.05 + (targetScore - 0.5) * 0.04);
      const spacingPolicy = this._policy()?.Spacing || {};
      const successMult = Number(spacingPolicy.success_multiplier) || 1.35;
      const failureMult = Number(spacingPolicy.failure_multiplier) || 0.55;
      const spacing = 1 + item.strength * 5 + Math.min(6, item.replays) * 0.45;
      const multiplier = success ? successMult : failureMult;
      item.intervalMs = Math.min(this.options.maxIntervalMs, Math.max(this.options.baseIntervalMs, Math.round(this.options.baseIntervalMs * spacing * multiplier)));
      item.dueAt = Date.now() + item.intervalMs;
      item.priority = clamp(item.priority * 0.58 + residualError * 0.32 + (success ? 0 : 0.10) + (isControl && !success ? 0.12 : 0));

      item.history ||= [];
      item.history.push({ at: Date.now(), success, isControl, residualError, targetScore, controlScore });
      if (item.history.length > 32) item.history.splice(0, item.history.length - 32);
      this.replayed++;

      const outcome = this.outcomeByCurriculum[item.curriculum] || { targetRuns: 0, targetSuccess: 0, controlRuns: 0, controlSuccess: 0, targetScore: 0, controlScore: 0 };
      if (isControl) {
        outcome.controlRuns++;
        if (success) outcome.controlSuccess++;
        outcome.controlScore = outcome.controlScore * 0.8 + controlScore * 0.2;
      } else {
        outcome.targetRuns++;
        if (success) outcome.targetSuccess++;
        outcome.targetScore = outcome.targetScore * 0.8 + targetScore * 0.2;
      }
      this.outcomeByCurriculum[item.curriculum] = outcome;

      this.last = { action: 'complete', key: item.key, curriculum: item.curriculum, success, isControl, strength: item.strength, nextInMs: item.intervalMs, targetScore, controlScore };
      return true;
    }

    curriculumOutcome(name) {
      this._ensureRuntimeState();
      const row = this.outcomeByCurriculum[String(name || '')] || null;
      if (!row) return null;
      return {
        ...row,
        targetSuccessRate: row.targetRuns ? row.targetSuccess / row.targetRuns : null,
        controlSuccessRate: row.controlRuns ? row.controlSuccess / row.controlRuns : null
      };
    }

    reset() {
      this.items.clear();
      this.scheduled = 0;
      this.replayed = 0;
      this.failuresScheduled = 0;
      this.controlReplays = 0;
      this.curriculumCounts = {};
      this.outcomeByCurriculum = {};
      this.last = null;
      return true;
    }

    exportState() {
      this._ensureRuntimeState();
      return {
        schemaVersion: 2,
        scheduled: this.scheduled,
        replayed: this.replayed,
        failuresScheduled: this.failuresScheduled,
        controlReplays: this.controlReplays,
        curriculumCounts: { ...this.curriculumCounts },
        outcomeByCurriculum: { ...this.outcomeByCurriculum },
        items: Array.from(this.items.values()).sort((a, b) => b.priority - a.priority).slice(0, this.options.maxItems)
      };
    }

    importState(state) {
      if (!state || ![1, 2].includes(Number(state.schemaVersion))) return false;
      this.items.clear();
      for (const item of Array.isArray(state.items) ? state.items : []) {
        if (!item?.key) continue;
        const curriculum = item.curriculum || this._curriculumFor(item.meta || {}, item.payload || {});
        this.items.set(String(item.key), {
          successes: 0, failures: 0, controlSuccesses: 0, controlFailures: 0,
          variants: this._variantDescriptors(curriculum, item.payload || {}), variantCursor: 0,
          controlRatio: this.options.minControlRatio, history: [], ...item, curriculum
        });
      }
      this.scheduled = Math.max(0, Number(state.scheduled) || 0);
      this.replayed = Math.max(0, Number(state.replayed) || 0);
      this.failuresScheduled = Math.max(0, Number(state.failuresScheduled) || 0);
      this.controlReplays = Math.max(0, Number(state.controlReplays) || 0);
      this.curriculumCounts = state.curriculumCounts && typeof state.curriculumCounts === 'object' ? { ...state.curriculumCounts } : {};
      this.outcomeByCurriculum = state.outcomeByCurriculum && typeof state.outcomeByCurriculum === 'object' ? { ...state.outcomeByCurriculum } : {};
      this._ensureRuntimeState();
      return true;
    }

    status() {
      this._ensureRuntimeState();
      const now = Date.now();
      const dueRows = Array.from(this.items.values()).filter(item => item.dueAt <= now);
      const due = dueRows.length;
      const dueKnowledge = dueRows.filter(item => item.kind !== 'diagnostic').length;
      const dueDiagnostic = dueRows.filter(item => item.kind === 'diagnostic').length;
      const byCurriculum = {};
      for (const item of this.items.values()) byCurriculum[item.curriculum] = (byCurriculum[item.curriculum] || 0) + 1;
      return {
        ready: true,
        role: 'failure-targeted-curriculum-replay-with-control-regression-cases',
        queued: this.items.size,
        due,
        dueKnowledge,
        dueDiagnostic,
        scheduled: this.scheduled,
        replayed: this.replayed,
        failuresScheduled: this.failuresScheduled,
        controlReplays: this.controlReplays,
        byCurriculum,
        outcomes: Object.fromEntries(Object.keys(this.outcomeByCurriculum).map(key => [key, this.curriculumOutcome(key)])),
        ruleDriven: Boolean(this._policy()?.Schema_Version),
        last: this.last
      };
    }
  }


  const V34_REPLAY_OPERATORS=Object.freeze([
    { id:'paraphrase', family:'surface', mode:'preserve', enabled:true },
    { id:'active_passive', family:'syntax', mode:'preserve', enabled:true },
    { id:'clause_reorder', family:'syntax', mode:'preserve', enabled:true },
    { id:'subordinate_to_coordinate', family:'syntax', mode:'preserve', enabled:true },
    { id:'coordinate_to_subordinate', family:'syntax', mode:'preserve', enabled:true },
    { id:'pronoun_substitution', family:'reference', mode:'preserve', enabled:true },
    { id:'demonstrative_reference', family:'reference', mode:'preserve', enabled:true },
    { id:'split_antecedent', family:'reference', mode:'preserve', enabled:true },
    { id:'event_reference', family:'reference', mode:'preserve', enabled:true },
    { id:'proposition_reference', family:'reference', mode:'preserve', enabled:true },
    { id:'temporal_before_after', family:'temporal', mode:'preserve', enabled:true },
    { id:'temporal_overlap', family:'temporal', mode:'preserve', enabled:true },
    { id:'duration_insertion', family:'temporal', mode:'preserve', enabled:true },
    { id:'frequency_insertion', family:'temporal', mode:'preserve', enabled:true },
    { id:'aspect_shift', family:'temporal', mode:'controlled', enabled:true },
    { id:'mechanism_stage_omission', family:'mechanism', mode:'challenge', enabled:true },
    { id:'mechanism_stage_reorder', family:'mechanism', mode:'challenge', enabled:true },
    { id:'feedback_loop', family:'mechanism', mode:'preserve', enabled:true },
    { id:'conditional_process', family:'mechanism', mode:'preserve', enabled:true },
    { id:'multi_component_process', family:'mechanism', mode:'preserve', enabled:true },
    { id:'negation_scope', family:'semantics', mode:'challenge', enabled:true },
    { id:'condition_scope', family:'semantics', mode:'challenge', enabled:true },
    { id:'distractor_insertion', family:'semantics', mode:'challenge', enabled:true },
    { id:'polysemy', family:'semantics', mode:'challenge', enabled:true },
    { id:'relation_direction', family:'semantics', mode:'challenge', enabled:true },
    { id:'two_part_request', family:'planning', mode:'preserve', enabled:true },
    { id:'three_part_request', family:'planning', mode:'preserve', enabled:true },
    { id:'dependent_parts', family:'planning', mode:'preserve', enabled:true },
    { id:'shared_context_parts', family:'planning', mode:'preserve', enabled:true },
    { id:'topic_shift_control', family:'planning', mode:'control', enabled:true },
    { id:'confidence_direct_inferred', family:'calibration', mode:'preserve', enabled:true },
    { id:'ambiguous_reference', family:'calibration', mode:'challenge', enabled:true },
    { id:'conflicting_evidence', family:'calibration', mode:'challenge', enabled:true },
    { id:'stale_evidence', family:'calibration', mode:'challenge', enabled:true },
    { id:'out_of_domain', family:'calibration', mode:'control', enabled:true },
    { id:'baseline_control', family:'control', mode:'control', enabled:true },
    { id:'cross_capability_control', family:'control', mode:'control', enabled:true },
    { id:'retention_control', family:'control', mode:'control', enabled:true },
    { id:'simple_control', family:'control', mode:'control', enabled:true },
    { id:'adversarial_control', family:'control', mode:'control', enabled:true },
  ]);
  const v34RClamp=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(v))?Number(v):0));
  const v34RNorm=v=>String(v??'').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g,' ').replace(/\s+/g,' ').trim();
  class V34ReplayEngine {
    constructor(owner){this.owner=owner;this.outcomes=new Map();this.generations=0;this.last=null;}
    safePolicy(){const p=this.owner?._policy?.()||{};return{...p,Curricula:p&&typeof p.Curricula==='object'?p.Curricula:{},Priority_Weights:p&&typeof p.Priority_Weights==='object'?p.Priority_Weights:{},Spacing:p&&typeof p.Spacing==='object'?p.Spacing:{},Anti_Regression:p&&typeof p.Anti_Regression==='object'?p.Anti_Regression:{}};}
    curriculum(name){const policy=this.safePolicy();const key=v34RNorm(name).replace(/\s+/g,'-')||'general';return policy.Curricula[key]||policy.Curricula.general||{families:['generalization','control'],variant_count:4,control_ratio:0.34};}
    operatorPool(curriculum){
      const families=new Set((curriculum?.families||[]).map(v34RNorm));const pool=V34_REPLAY_OPERATORS.filter(op=>families.has(op.id)||families.has(op.family)||op.mode==='control');return pool.length?pool:V34_REPLAY_OPERATORS.filter(op=>['surface','syntax','control'].includes(op.family));
    }
    difficulty(item,operator,index){const failures=Number(item.failures||0),successes=Number(item.successes||0),strength=v34RClamp(item.strength??0.25);const base=0.28+failures*0.08-successes*0.025+(1-strength)*0.26+index*0.025;const bonus=operator.mode==='challenge'?0.18:operator.mode==='control'?-0.08:0;return v34RClamp(base+bonus,0.08,0.96);}
    makeVariants(item,count){
      const spec=this.curriculum(item.curriculum);const pool=this.operatorPool(spec);const max=Math.max(1,Math.min(24,Number(count||spec.variant_count||8)));const out=[];
      for(let i=0;i<max;i++){const op=pool[i%pool.length];out.push({id:`${item.key}:v${i}`,operator:op.id,family:op.family,mode:op.mode,difficulty:this.difficulty(item,op,i),preserveMeaning:op.mode!=='challenge'||!['mechanism_stage_omission','mechanism_stage_reorder','negation_scope','relation_direction'].includes(op.id),sourcePrompt:String(item.payload?.promptText||item.payload?.prompt||'').slice(0,768),expectedTarget:item.meta?.diagnosis||item.curriculum,control:op.mode==='control'});}
      this.generations+=out.length;return out;
    }
    controlsFor(item,min=2){const pool=V34_REPLAY_OPERATORS.filter(op=>op.mode==='control');return pool.slice(0,Math.max(min,Math.ceil((item.variants?.length||4)*(item.controlRatio||0.3)))).map((op,i)=>({id:`${item.key}:control:${i}`,operator:op.id,family:op.family,mode:'control',difficulty:0.22+i*0.05,preserveMeaning:true,sourcePrompt:String(item.payload?.promptText||item.payload?.prompt||'').slice(0,768),expectedTarget:'control',control:true}));}
    scorePriority(item){const failurePressure=v34RClamp((item.failures||0)/Math.max(1,(item.replays||0)+1));const uncertainty=v34RClamp(item.meta?.uncertainty??0.5);const prediction=v34RClamp(item.meta?.predictionError??0.4);const novelty=v34RClamp(item.meta?.novelty??0.5);const controlNeed=v34RClamp(item.controlFailures?0.9:0.2);return v34RClamp(failurePressure*0.25+uncertainty*0.20+prediction*0.25+novelty*0.15+controlNeed*0.15);}
    diversify(items=[],limit=8){
      const selected=[];const usedCurricula=new Set(),usedFamilies=new Set();const rows=items.slice().sort((a,b)=>this.scorePriority(b)-this.scorePriority(a));
      for(const item of rows){if(selected.length>=limit)break;const curriculum=item.curriculum||'general';const variants=item.variants?.length?item.variants:this.makeVariants(item);const family=variants[item.variantCursor%Math.max(1,variants.length)]?.family||'general';const diversityBonus=(usedCurricula.has(curriculum)?0:0.18)+(usedFamilies.has(family)?0:0.12);selected.push({item,score:v34RClamp(this.scorePriority(item)+diversityBonus),family,curriculum});usedCurricula.add(curriculum);usedFamilies.add(family);}
      return selected.sort((a,b)=>b.score-a.score).map(x=>x.item);
    }
    record(item,result={}){const key=item.curriculum||'general';const row=this.outcomes.get(key)||{runs:0,successes:0,controls:0,controlSuccesses:0,difficulty:0};row.runs++;if(result.success!==false)row.successes++;if(result.isControl){row.controls++;if(result.success!==false)row.controlSuccesses++;}row.difficulty+=(Number(result.difficulty||0.5)-row.difficulty)/row.runs;this.outcomes.set(key,row);return{...row,successRate:row.successes/row.runs,controlSuccessRate:row.controls?row.controlSuccesses/row.controls:null};}
    curriculumHealth(name){const row=this.outcomes.get(name);if(!row)return{name,runs:0,health:'unknown'};const success=row.successes/Math.max(1,row.runs),control=row.controls?row.controlSuccesses/row.controls:1;return{name,...row,successRate:success,controlSuccessRate:control,health:control<0.75?'regressing':success<0.6?'weak':'stable'};}
    status(){return{generations:this.generations,outcomes:Object.fromEntries(Array.from(this.outcomes.keys()).map(k=>[k,this.curriculumHealth(k)])),last:this.last};}
  }
  const v34OriginalSchedule=VilotReplayScheduler.prototype.schedule;
  VilotReplayScheduler.prototype.schedule=function(key,payload={},meta={}){const item=v34OriginalSchedule.call(this,key,payload,meta);if(!item)return item;this.v34Engine||=new V34ReplayEngine(this);item.variants=this.v34Engine.makeVariants(item,Math.max(item.variants?.length||0,Number(this.v34Engine.curriculum(item.curriculum).variant_count||8)));item.controls=this.v34Engine.controlsFor(item,2);return item;};
  const v34OriginalNextBatch=VilotReplayScheduler.prototype.nextBatch;
  VilotReplayScheduler.prototype.nextBatch=function(limit=2,now=Date.now()){const base=v34OriginalNextBatch.call(this,Math.max(limit,limit*2),now)||[];this.v34Engine||=new V34ReplayEngine(this);const items=base.map(x=>this.items.get(x.key)||x);return this.v34Engine.diversify(items,Math.max(1,limit)).map(item=>this._makeReplayCase(item,false));};
  const v34OriginalComplete=VilotReplayScheduler.prototype.complete;
  VilotReplayScheduler.prototype.complete=function(key,result={}){const item=this.items.get(String(key||''));const ok=v34OriginalComplete.call(this,key,result);if(ok&&item){this.v34Engine||=new V34ReplayEngine(this);this.v34Engine.record(item,result);}return ok;};
  VilotReplayScheduler.prototype.curriculumHealth=function(name){this.v34Engine||=new V34ReplayEngine(this);return this.v34Engine.curriculumHealth(name);};
  const v34OriginalReplayStatus=VilotReplayScheduler.prototype.status;
  VilotReplayScheduler.prototype.status=function(){const base=v34OriginalReplayStatus.call(this);this.v34Engine||=new V34ReplayEngine(this);return{...base,system34:this.v34Engine.status()};};



  const V34_REPLAY_SCENARIOS = Object.freeze([
    {
      id: "curriculum-scenario-1",
      family: "curriculum",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "variant-scenario-2",
      family: "variant",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "control-scenario-3",
      family: "control",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "difficulty-scenario-4",
      family: "difficulty",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "priority-scenario-5",
      family: "priority",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "spacing-scenario-6",
      family: "spacing",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "retention-scenario-7",
      family: "retention",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "diversity-scenario-8",
      family: "diversity",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "regression-scenario-9",
      family: "regression",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "mastery-scenario-10",
      family: "mastery",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "curriculum-scenario-11",
      family: "curriculum",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "variant-scenario-12",
      family: "variant",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "control-scenario-13",
      family: "control",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "difficulty-scenario-14",
      family: "difficulty",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "priority-scenario-15",
      family: "priority",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "spacing-scenario-16",
      family: "spacing",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "retention-scenario-17",
      family: "retention",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "diversity-scenario-18",
      family: "diversity",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "regression-scenario-19",
      family: "regression",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "mastery-scenario-20",
      family: "mastery",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "curriculum-scenario-21",
      family: "curriculum",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "variant-scenario-22",
      family: "variant",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "control-scenario-23",
      family: "control",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "difficulty-scenario-24",
      family: "difficulty",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "priority-scenario-25",
      family: "priority",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "spacing-scenario-26",
      family: "spacing",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "retention-scenario-27",
      family: "retention",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "diversity-scenario-28",
      family: "diversity",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "regression-scenario-29",
      family: "regression",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "mastery-scenario-30",
      family: "mastery",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "curriculum-scenario-31",
      family: "curriculum",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "variant-scenario-32",
      family: "variant",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "control-scenario-33",
      family: "control",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "difficulty-scenario-34",
      family: "difficulty",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "priority-scenario-35",
      family: "priority",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "spacing-scenario-36",
      family: "spacing",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "retention-scenario-37",
      family: "retention",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "diversity-scenario-38",
      family: "diversity",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "regression-scenario-39",
      family: "regression",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "mastery-scenario-40",
      family: "mastery",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "curriculum-scenario-41",
      family: "curriculum",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "variant-scenario-42",
      family: "variant",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "control-scenario-43",
      family: "control",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "difficulty-scenario-44",
      family: "difficulty",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "priority-scenario-45",
      family: "priority",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "spacing-scenario-46",
      family: "spacing",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "retention-scenario-47",
      family: "retention",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "diversity-scenario-48",
      family: "diversity",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "regression-scenario-49",
      family: "regression",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "mastery-scenario-50",
      family: "mastery",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "curriculum-scenario-51",
      family: "curriculum",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "variant-scenario-52",
      family: "variant",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "control-scenario-53",
      family: "control",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "difficulty-scenario-54",
      family: "difficulty",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "priority-scenario-55",
      family: "priority",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "spacing-scenario-56",
      family: "spacing",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "retention-scenario-57",
      family: "retention",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "diversity-scenario-58",
      family: "diversity",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "regression-scenario-59",
      family: "regression",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "mastery-scenario-60",
      family: "mastery",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "curriculum-scenario-61",
      family: "curriculum",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "variant-scenario-62",
      family: "variant",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "control-scenario-63",
      family: "control",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "difficulty-scenario-64",
      family: "difficulty",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
  ]);
  VilotReplayScheduler.prototype.replayScenarios = function() { return V34_REPLAY_SCENARIOS.map(row => ({ ...row })); };

  globalThis.VilotReplayScheduler = VilotReplayScheduler;
})();
