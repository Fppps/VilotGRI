/**
 * VilotNI 2.5 - TemporalKnowledge.js
 * System 34 foundation capacity expansion.
 *
 * This layer now serves two jobs:
 *  1) knowledge validity/freshness for learned graph evidence;
 *  2) event-time reasoning over proposition graphs.
 *
 * Temporal inference is deliberately conservative. It records ambiguity instead
 * of inventing dates or event order when the prompt does not support one.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));
  const norm = value => String(value || '').toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9:+.\- ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const uniq = rows => Array.from(new Set((rows || []).filter(Boolean)));

  class VilotTemporalKnowledge {
    constructor(options = {}, ruleStore = null) {
      this.ruleStore = ruleStore;
      this.options = {
        staleAfterMs: Math.max(60_000, Number(options.temporalKnowledgeStaleAfterMs) || 7 * 24 * 60 * 60 * 1000),
        staleFloor: clamp(options.temporalKnowledgeStaleFloor ?? 0.45),
        unknownFreshness: clamp(options.temporalKnowledgeUnknownFreshness ?? 0.88),
        maxInferenceHops: Math.max(1, Math.min(8, options.temporalKnowledgeMaxInferenceHops | 0 || 4)),
        maxEvents: Math.max(8, Math.min(96, options.temporalKnowledgeMaxEvents | 0 || 48))
      };
      this.assessments = 0;
      this.expired = 0;
      this.graphAnalyses = 0;
      this.inferredRelations = 0;
      this.last = null;
      this.lastGraph = null;
    }

    _rules() {
      return this.ruleStore?.temporalRelationRules?.() || this.ruleStore?.get?.('temporalRelations') || {};
    }

    _time(value) {
      if (value == null || value === '') return null;
      if (Number.isFinite(Number(value))) return Number(value);
      const parsed = Date.parse(String(value));
      return Number.isFinite(parsed) ? parsed : null;
    }

    _relationInfo(relation) {
      const key = norm(relation).replace(/\s+/g, '-');
      return this._rules()?.Relations?.[key] || this._rules()?.Relations?.[norm(relation)] || null;
    }

    _canonicalRelation(value) {
      const raw = norm(value).replace(/_/g, '-');
      const direct = {
        'temporal-before': 'before', 'earlier': 'before', 'previously': 'before', 'prior-to': 'before',
        'temporal-after': 'after', 'later': 'after', 'then': 'after', 'subsequently': 'after',
        'temporal-overlap': 'overlaps', 'while': 'overlaps', 'during': 'during',
        'simultaneous': 'simultaneous', 'at-the-same-time': 'simultaneous',
        'temporal-limit': 'until', 'since': 'since', 'starts': 'starts', 'begins': 'starts',
        'ends': 'finishes', 'finishes': 'finishes', 'continues': 'continues', 'repeats': 'repeats'
      };
      if (direct[raw]) return direct[raw];
      if (this._relationInfo(raw)) return raw;
      return '';
    }

    _extractCueRelations(text = '') {
      const clean = ` ${norm(text)} `;
      const cues = this._rules()?.Cues || {};
      const out = [];
      for (const [relation, phrases] of Object.entries(cues)) {
        for (const phrase of phrases || []) {
          const token = ` ${norm(phrase)} `;
          if (token.trim() && clean.includes(token)) {
            out.push({ relation, cue: norm(phrase), confidence: 0.76 });
            break;
          }
        }
      }
      return out;
    }

    assess(edgeOrMetadata, now = Date.now()) {
      const edge = edgeOrMetadata || {};
      const metadata = edge.metadata || edge;
      const validFrom = this._time(metadata.validFrom ?? metadata.Valid_From ?? metadata.valid_from);
      const validUntil = this._time(metadata.validUntil ?? metadata.Valid_Until ?? metadata.valid_until);
      const lastVerified = this._time(metadata.lastVerified ?? metadata.last_verified ?? edge.lastSeen);
      const explicitlyTimeless = metadata.timeless === true || metadata.Timeless === true;

      const active = (validFrom == null || now >= validFrom) && (validUntil == null || now <= validUntil);

      let freshness;
      if (explicitlyTimeless) {
        freshness = 1;
      } else if (lastVerified != null) {
        const age = Math.max(0, now - lastVerified);
        const ratio = Math.exp(-Math.log(2) * age / this.options.staleAfterMs);
        freshness = Math.max(this.options.staleFloor, clamp(ratio));
      } else {
        freshness = this.options.unknownFreshness;
      }
      if (!active) freshness = 0;

      const result = {
        active,
        expired: !active,
        freshness,
        validFrom,
        validUntil,
        lastVerified,
        timeless: explicitlyTimeless,
        state: !active ? 'expired' : freshness < 0.62 ? 'stale' : freshness < 0.84 ? 'aging' : 'current'
      };
      this.assessments++;
      if (!active) this.expired++;
      this.last = result;
      return result;
    }

    applyMetadata(edge, metadata = {}) {
      if (!edge) return null;
      edge.metadata ||= {};
      for (const [source, target] of [
        ['validFrom', 'validFrom'], ['Valid_From', 'validFrom'],
        ['validUntil', 'validUntil'], ['Valid_Until', 'validUntil'],
        ['lastVerified', 'lastVerified'], ['Last_Verified', 'lastVerified'],
        ['timeless', 'timeless'], ['Timeless', 'timeless'],
        ['duration', 'duration'], ['frequency', 'frequency'], ['aspect', 'aspect']
      ]) {
        if (metadata[source] != null) edge.metadata[target] = metadata[source];
      }
      return edge;
    }

    staleEdges(graph, limit = 16) {
      const rows = graph?.topEdges?.(Math.max(32, limit * 4)) || [];
      return rows
        .map(edge => ({ edge, temporal: this.assess(edge) }))
        .filter(row => row.temporal.active && row.temporal.freshness < 0.72)
        .sort((a, b) => a.temporal.freshness - b.temporal.freshness)
        .slice(0, Math.max(1, limit | 0 || 16));
    }

    _eventFromProposition(p, index) {
      const relation = this._canonicalRelation(p?.relation || p?.type || '');
      const text = [p?.subject, p?.relation, p?.object, ...(Array.isArray(p?.time) ? p.time : [p?.time])].filter(Boolean).join(' ');
      const cues = this._extractCueRelations(text);
      const temporalRelation = relation || cues[0]?.relation || '';
      const id = String(p?.id || `event-${index + 1}`);
      const label = norm(p?.object || p?.subject || p?.relation || id);
      const explicitTime = (Array.isArray(p?.time) ? p.time : [p?.time]).map(this._time.bind(this)).find(Number.isFinite) ?? null;
      return {
        id,
        propositionId: id,
        label,
        subject: norm(p?.subject),
        predicate: norm(p?.relation),
        object: norm(p?.object),
        relation: temporalRelation,
        time: explicitTime,
        cues,
        confidence: clamp(p?.certainty ?? 0.62),
        polarity: Number(p?.polarity ?? 1) >= 0 ? 1 : -1,
        rawTime: Array.isArray(p?.time) ? p.time.slice() : p?.time ? [p.time] : []
      };
    }

    _pushRelation(list, seen, from, relation, to, confidence = 0.72, source = 'prompt') {
      const a = String(from || '');
      const b = String(to || '');
      const r = this._canonicalRelation(relation) || norm(relation);
      if (!a || !b || !r || a === b) return false;
      const key = `${a}|${r}|${b}`;
      if (seen.has(key)) return false;
      seen.add(key);
      list.push({ from: a, relation: r, to: b, confidence: clamp(confidence), source });
      return true;
    }

    _invertRelations(relations) {
      const out = [];
      const seen = new Set(relations.map(r => `${r.from}|${r.relation}|${r.to}`));
      for (const row of relations.slice()) {
        const info = this._relationInfo(row.relation);
        const inverse = info?.inverse;
        if (inverse) this._pushRelation(out, seen, row.to, inverse, row.from, row.confidence * 0.98, 'inverse');
        if (info?.symmetric) this._pushRelation(out, seen, row.to, row.relation, row.from, row.confidence * 0.98, 'symmetric');
      }
      return out;
    }

    _transitiveClosure(relations) {
      const all = relations.slice();
      const seen = new Set(all.map(r => `${r.from}|${r.relation}|${r.to}`));
      const transitive = new Set(Object.entries(this._rules()?.Relations || {}).filter(([, info]) => info?.transitive).map(([name]) => name));
      const maxHops = Math.max(1, Number(this._rules()?.Inference?.max_transitive_hops) || this.options.maxInferenceHops);
      for (let hop = 0; hop < maxHops; hop++) {
        let added = 0;
        const snapshot = all.slice();
        for (const left of snapshot) {
          if (!transitive.has(left.relation)) continue;
          for (const right of snapshot) {
            if (left.relation !== right.relation || left.to !== right.from || left.from === right.to) continue;
            const confidence = clamp(Math.min(left.confidence, right.confidence) * 0.90);
            if (this._pushRelation(all, seen, left.from, left.relation, right.to, confidence, 'transitive-inference')) {
              added++;
              this.inferredRelations++;
            }
          }
        }
        if (!added) break;
      }
      return all;
    }

    _contradictions(relations) {
      const index = new Set(relations.map(r => `${r.from}|${r.relation}|${r.to}`));
      const conflicts = [];
      const push = (a, b, kind) => {
        const key = [a.from, a.to, kind].sort().join('|');
        if (conflicts.some(x => x.key === key)) return;
        conflicts.push({ key, kind, a, b });
      };
      for (const row of relations) {
        if (row.relation === 'before') {
          const reverse = relations.find(r => r.from === row.from && r.to === row.to && r.relation === 'after') || relations.find(r => r.from === row.to && r.to === row.from && r.relation === 'before');
          if (reverse) push(row, reverse, 'order-conflict');
        }
        if (row.relation === 'after') {
          const reverse = relations.find(r => r.from === row.from && r.to === row.to && r.relation === 'before');
          if (reverse) push(row, reverse, 'order-conflict');
        }
      }
      return conflicts.map(({ key, ...rest }) => rest);
    }

    _explicitRelations(events, graph) {
      const relations = [];
      const seen = new Set();
      const deps = Array.isArray(graph?.dependencies) ? graph.dependencies : [];
      const eventIds = new Set(events.map(e => e.id));

      for (const dep of deps) {
        const relation = this._canonicalRelation(dep?.relation);
        if (!relation || !eventIds.has(String(dep?.from)) || !eventIds.has(String(dep?.to))) continue;
        this._pushRelation(relations, seen, dep.from, relation, dep.to, dep.confidence ?? 0.78, 'dependency');
      }

      for (let i = 0; i < events.length; i++) {
        const event = events[i];
        if (event.time != null) {
          for (let j = i + 1; j < events.length; j++) {
            const other = events[j];
            if (other.time == null || event.time === other.time) continue;
            if (event.time < other.time) this._pushRelation(relations, seen, event.id, 'before', other.id, Math.min(event.confidence, other.confidence), 'explicit-time');
            else this._pushRelation(relations, seen, event.id, 'after', other.id, Math.min(event.confidence, other.confidence), 'explicit-time');
          }
        }
      }

      // When a proposition explicitly names a temporal relation, connect it to
      // its dependency target when one exists. Without a target, retain the cue
      // as ambiguous instead of guessing an event.
      for (const event of events) {
        if (!event.relation) continue;
        const dep = deps.find(d => String(d?.from) === event.id || String(d?.to) === event.id);
        if (!dep) continue;
        const other = String(dep.from) === event.id ? dep.to : dep.from;
        this._pushRelation(relations, seen, event.id, event.relation, other, event.confidence * 0.86, 'proposition-cue');
      }
      return relations;
    }

    analyzePropositionGraph(graph) {
      if (!graph || !Array.isArray(graph.propositions)) {
        return { consistency: 1, ambiguity: 0, events: [], relations: [], inferred: [], contradictions: [], propositionStates: {} };
      }
      const events = graph.propositions.slice(0, this.options.maxEvents).map((p, index) => this._eventFromProposition(p, index));
      const base = this._explicitRelations(events, graph);
      const inverse = this._invertRelations(base);
      const withInverse = base.concat(inverse);
      const all = this._transitiveClosure(withInverse);
      const contradictions = this._contradictions(all);
      const inferred = all.filter(r => r.source === 'transitive-inference' || r.source === 'inverse' || r.source === 'symmetric');
      const propositionStates = {};
      let ambiguousCount = 0;

      for (const event of events) {
        const hasCue = Boolean(event.relation || event.cues.length || event.rawTime.length);
        const attached = all.filter(r => r.from === event.id || r.to === event.id);
        const contradicted = contradictions.some(c => c.a?.from === event.id || c.a?.to === event.id || c.b?.from === event.id || c.b?.to === event.id);
        const ambiguous = Boolean(hasCue && !attached.length && event.time == null);
        if (ambiguous) ambiguousCount++;
        propositionStates[event.id] = {
          ambiguous,
          contradicted,
          relationCount: attached.length,
          confidence: clamp(event.confidence * (ambiguous ? 0.82 : 1) * (contradicted ? 0.65 : 1))
        };
      }

      const temporalEvents = events.filter(e => e.relation || e.rawTime.length || e.cues.length);
      const ambiguity = temporalEvents.length ? clamp(ambiguousCount / temporalEvents.length) : 0;
      const contradictionPressure = events.length ? clamp(contradictions.length / Math.max(1, events.length)) : 0;
      const consistency = clamp(1 - contradictionPressure * 0.65 - ambiguity * 0.35);
      const result = {
        events,
        relations: all,
        inferred,
        contradictions,
        contradictionCount: contradictions.length,
        ambiguity,
        consistency,
        propositionStates,
        metrics: {
          eventCount: events.length,
          temporalEventCount: temporalEvents.length,
          explicitRelationCount: base.length,
          inferredRelationCount: inferred.length,
          ambiguity,
          consistency
        }
      };
      this.graphAnalyses++;
      this.lastGraph = result;
      return result;
    }

    compareEvents(graph, leftId, rightId) {
      const state = graph?.relations ? graph : this.analyzePropositionGraph(graph);
      const direct = (state?.relations || []).filter(r => (r.from === leftId && r.to === rightId) || (r.from === rightId && r.to === leftId));
      if (!direct.length) return { relation: 'unknown', confidence: 0, ambiguous: true };
      const ranked = direct.slice().sort((a, b) => b.confidence - a.confidence);
      const top = ranked[0];
      if (top.from === leftId) return { relation: top.relation, confidence: top.confidence, ambiguous: ranked.length > 1 && ranked[1].relation !== top.relation };
      const inverse = this._relationInfo(top.relation)?.inverse || top.relation;
      return { relation: inverse, confidence: top.confidence, ambiguous: ranked.length > 1 && ranked[1].relation !== top.relation };
    }

    status() {
      return {
        ready: true,
        role: 'knowledge-validity-plus-event-temporal-reasoning-layer',
        assessments: this.assessments,
        expiredObservations: this.expired,
        graphAnalyses: this.graphAnalyses,
        inferredRelations: this.inferredRelations,
        ruleDriven: Boolean(this._rules()?.Schema_Version),
        last: this.last,
        lastGraph: this.lastGraph ? {
          events: this.lastGraph.events.length,
          relations: this.lastGraph.relations.length,
          inferred: this.lastGraph.inferred.length,
          contradictions: this.lastGraph.contradictions.length,
          ambiguity: this.lastGraph.ambiguity,
          consistency: this.lastGraph.consistency
        } : null
      };
    }
  }


  // System 34 temporal engine: event intervals, aspect, frequency, relative anchors,
  // safe relation algebra, uncertainty-aware closure, and contradiction traces.
  const V34_TEMPORAL_RELATIONS = Object.freeze([
    { id:'before', inverse:'after', transitive:true, symmetric:false, confidenceDecay:0.04 },
    { id:'after', inverse:'before', transitive:true, symmetric:false, confidenceDecay:0.04 },
    { id:'meets', inverse:'met-by', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'met-by', inverse:'meets', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'overlaps', inverse:'overlapped-by', transitive:false, symmetric:true, confidenceDecay:0.08 },
    { id:'overlapped-by', inverse:'overlaps', transitive:false, symmetric:true, confidenceDecay:0.08 },
    { id:'during', inverse:'contains-time', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'contains-time', inverse:'during', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'starts', inverse:'started-by', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'started-by', inverse:'starts', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'finishes', inverse:'finished-by', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'finished-by', inverse:'finishes', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'equal-time', inverse:'equal-time', transitive:true, symmetric:true, confidenceDecay:0.04 },
    { id:'simultaneous', inverse:'simultaneous', transitive:true, symmetric:true, confidenceDecay:0.04 },
    { id:'continues-through', inverse:'contains-time', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'since', inverse:'before', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'until', inverse:'after', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'repeats-before', inverse:'after', transitive:false, symmetric:false, confidenceDecay:0.08 },
    { id:'precedes-eventually', inverse:'follows-eventually', transitive:true, symmetric:false, confidenceDecay:0.04 },
    { id:'follows-eventually', inverse:'precedes-eventually', transitive:true, symmetric:false, confidenceDecay:0.04 },
  ]);
  const v34TClamp=(v,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(v))?Number(v):0));
  const v34TNorm=v=>String(v??'').toLowerCase().replace(/[^a-z0-9_.+\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const v34TArr=v=>Array.isArray(v)?v:[];

  class V34TemporalEngine {
    constructor(owner) { this.owner=owner; this.history=[]; this.last=null; }
    spec(relation) { const key=v34TNorm(relation).replace(/\s+/g,'-'); return V34_TEMPORAL_RELATIONS.find(r=>r.id===key)||{id:key,inverse:null,transitive:false,symmetric:false,confidenceDecay:0.10}; }
    parseDuration(text='') {
      const source=String(text||'').toLowerCase();
      const match=source.match(/(?:for\s+)?(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)/i);
      if(!match) return null;
      const value=Number(match[1]); const unit=match[2].toLowerCase();
      const factors={ms:1,millisecond:1,milliseconds:1,second:1000,seconds:1000,sec:1000,secs:1000,minute:60000,minutes:60000,min:60000,mins:60000,hour:3600000,hours:3600000,hr:3600000,hrs:3600000,day:86400000,days:86400000,week:604800000,weeks:604800000,month:2629800000,months:2629800000,year:31557600000,years:31557600000};
      return {value,unit,milliseconds:value*(factors[unit]||0),surface:match[0]};
    }
    parseFrequency(text='') {
      const s=String(text||'').toLowerCase();
      const lexical=[['always',1],['usually',0.82],['often',0.68],['frequently',0.66],['sometimes',0.50],['occasionally',0.34],['rarely',0.16],['never',0]];
      for(const [cue,rate] of lexical) if(new RegExp(`\b${cue}\b`).test(s)) return {kind:'lexical',cue,rate,confidence:0.88};
      const every=s.match(/\bevery\s+(\d+)?\s*(second|minute|hour|day|week|month|year)s?\b/);
      if(every) return {kind:'periodic',every:Number(every[1]||1),unit:every[2],confidence:0.92};
      const times=s.match(/\b(\d+)\s+times?\s+(?:per|a|each)\s+(day|week|month|year)\b/);
      if(times) return {kind:'rate',count:Number(times[1]),unit:times[2],confidence:0.92};
      return null;
    }
    parseAspect(text='') {
      const s=String(text||'').toLowerCase();
      const tags=[];
      if(/\b(?:is|are|was|were|has been|have been)\s+\w+ing\b/.test(s)) tags.push('ongoing');
      if(/\b(?:has|have|had)\s+\w+(?:ed|en)\b/.test(s)) tags.push('completed_or_resultative');
      if(/\b(?:began|started|starts|begin|begins)\b/.test(s)) tags.push('inceptive');
      if(/\b(?:stopped|ended|finished|ceased|stops|ends)\b/.test(s)) tags.push('cessative');
      if(/\b(?:usually|often|every|regularly|habitually)\b/.test(s)) tags.push('habitual');
      return tags;
    }
    parseRelativeTime(text='') {
      const s=String(text||'').toLowerCase(); const out=[];
      const cues=[['before','before'],['after','after'],['while','simultaneous'],['during','during'],['since','since'],['until','until'],['when','simultaneous'],['later','after'],['earlier','before'],['then','after'],['meanwhile','simultaneous'],['eventually','precedes-eventually']];
      for(const [cue,relation] of cues) if(new RegExp(`\b${cue}\b`).test(s)) out.push({cue,relation,confidence:cue==='when'?0.62:0.86});
      return out;
    }
    normalizeEvent(event,index=0) {
      const id=String(event?.id||`event:${index}`); const text=String(event?.text||event?.surface||event?.label||'');
      const start=Number.isFinite(Number(event?.start))?Number(event.start):null; const end=Number.isFinite(Number(event?.end))?Number(event.end):null;
      return {...event,id,text,start,end,duration:event?.duration||this.parseDuration(text),frequency:event?.frequency||this.parseFrequency(text),aspect:v34TArr(event?.aspect).length?v34TArr(event.aspect):this.parseAspect(text),confidence:v34TClamp(event?.confidence??0.76)};
    }
    intervalRelation(a,b) {
      if(a.start==null||b.start==null) return null;
      const ae=a.end==null?a.start:a.end, be=b.end==null?b.start:b.end;
      if(ae < b.start) return 'before';
      if(a.start > be) return 'after';
      if(a.start===b.start && ae===be) return 'equal-time';
      if(a.start===b.start && ae<be) return 'starts';
      if(a.start===b.start && ae>be) return 'started-by';
      if(ae===be && a.start>b.start) return 'finishes';
      if(ae===be && a.start<b.start) return 'finished-by';
      if(a.start>b.start && ae<be) return 'during';
      if(a.start<b.start && ae>be) return 'contains-time';
      return 'overlaps';
    }
    buildGraph(events=[],relations=[]) {
      const nodes=new Map(v34TArr(events).map((e,i)=>{const n=this.normalizeEvent(e,i);return[n.id,n];}));
      const edges=[]; const seen=new Set();
      const add=(from,relation,to,confidence=0.75,source='explicit',depth=0)=>{
        if(!from||!to||from===to)return false; const rel=v34TNorm(relation).replace(/\s+/g,'-'); const key=`${from}|${rel}|${to}`;
        if(seen.has(key))return false; seen.add(key); edges.push({from,to,relation:rel,confidence:v34TClamp(confidence),source,depth,inferred:source!=='explicit'&&source!=='interval'}); return true;
      };
      for(const r of v34TArr(relations)) add(r.from||r.left,r.relation||r.type,r.to||r.right,r.confidence??0.78,r.source||'explicit',r.depth||0);
      const list=Array.from(nodes.values());
      for(let i=0;i<list.length;i++) for(let j=i+1;j<list.length;j++) { const rel=this.intervalRelation(list[i],list[j]); if(rel)add(list[i].id,rel,list[j].id,0.96,'interval',0); }
      for(const edge of edges.slice()) { const spec=this.spec(edge.relation); if(spec.inverse)add(edge.to,spec.inverse,edge.from,edge.confidence*0.98,'inverse',edge.depth); if(spec.symmetric)add(edge.to,edge.relation,edge.from,edge.confidence*0.98,'symmetric',edge.depth); }
      return {nodes,edges,add,seen};
    }
    safeClosure(graph,maxHops=5) {
      for(let hop=1;hop<=maxHops;hop++) {
        let added=0; const snapshot=graph.edges.slice();
        for(const a of snapshot) { const spec=this.spec(a.relation); if(!spec.transitive)continue;
          for(const b of snapshot) { if(a.to!==b.from||b.relation!==a.relation)continue; const confidence=Math.min(a.confidence,b.confidence)*(1-spec.confidenceDecay); if(confidence<0.46)continue; if(graph.add(a.from,a.relation,b.to,confidence,'transitive',Math.max(a.depth||0,b.depth||0)+1))added++; }
        }
        if(!added)break;
      }
      return graph;
    }
    contradictions(graph) {
      const out=[]; const opposite={before:'after',after:'before',during:'contains-time','contains-time':'during',starts:'started-by','started-by':'starts',finishes:'finished-by','finished-by':'finishes'};
      const keys=new Set(graph.edges.map(e=>`${e.from}|${e.relation}|${e.to}`));
      for(const edge of graph.edges) {
        const inverse=opposite[edge.relation]; if(inverse && keys.has(`${edge.from}|${inverse}|${edge.to}`)) out.push({type:'opposite-temporal-relations',from:edge.from,to:edge.to,a:edge.relation,b:inverse,severity:0.9});
        if(edge.relation==='before' && keys.has(`${edge.to}|before|${edge.from}`)) out.push({type:'temporal-cycle',from:edge.from,to:edge.to,severity:1});
      }
      return out.filter((x,i,a)=>a.findIndex(y=>JSON.stringify(y)===JSON.stringify(x))===i);
    }
    topologicalOrder(graph) {
      const before=graph.edges.filter(e=>e.relation==='before'&&e.confidence>=0.55); const indegree=new Map(Array.from(graph.nodes.keys()).map(k=>[k,0])); const adj=new Map(Array.from(graph.nodes.keys()).map(k=>[k,[]]));
      for(const e of before){ if(!adj.has(e.from)||!indegree.has(e.to))continue; adj.get(e.from).push(e.to); indegree.set(e.to,(indegree.get(e.to)||0)+1); }
      const q=Array.from(indegree).filter(([,d])=>d===0).map(([k])=>k); const order=[];
      while(q.length){ const id=q.shift(); order.push(id); for(const to of adj.get(id)||[]){indegree.set(to,indegree.get(to)-1);if(indegree.get(to)===0)q.push(to);} }
      return {order,cyclic:order.length!==graph.nodes.size};
    }
    reason(events=[],relations=[]) {
      const graph=this.safeClosure(this.buildGraph(events,relations)); const contradictions=this.contradictions(graph); const order=this.topologicalOrder(graph);
      const ambiguity=v34TClamp((graph.edges.filter(e=>e.confidence<0.62).length/Math.max(1,graph.edges.length))*0.7+(order.cyclic?0.3:0));
      const consistency=v34TClamp(1-Math.min(0.75,contradictions.length*0.18)-(order.cyclic?0.24:0));
      const result={events:Array.from(graph.nodes.values()),relations:graph.edges,contradictions,order:order.order,cyclic:order.cyclic,ambiguity,consistency};
      this.last=result;this.history.push({at:Date.now(),events:result.events.length,relations:result.relations.length,consistency});if(this.history.length>96)this.history.shift();return result;
    }
    compare(graph,leftId,rightId) {
      const rows=v34TArr(graph?.relations||graph?.edges).filter(e=>(e.from===leftId&&e.to===rightId)||(e.from===rightId&&e.to===leftId));
      return rows.sort((a,b)=>Number(b.confidence||0)-Number(a.confidence||0));
    }
  }

  const v34OriginalTemporalAnalyze=VilotTemporalKnowledge.prototype.analyzePropositionGraph;
  VilotTemporalKnowledge.prototype.analyzePropositionGraph=function(graph){
    const base=v34OriginalTemporalAnalyze.call(this,graph)||{}; this.v34Engine||=new V34TemporalEngine(this);
    const events=v34TArr(base.events).length?base.events:v34TArr(graph?.propositions).map((p,i)=>({id:p.id||`p:${i}`,text:[p.subject,p.relation,p.object,p.time].filter(Boolean).join(' '),confidence:p.confidence,time:p.time,start:p.start,end:p.end}));
    const deep=this.v34Engine.reason(events,[...v34TArr(base.relations),...v34TArr(graph?.temporalRelations)]);
    const result={...base,...deep,relations:deep.relations,contradictions:[...v34TArr(base.contradictions),...deep.contradictions],consistency:Math.min(Number(base.consistency??1),deep.consistency),ambiguity:Math.max(Number(base.ambiguity??0),deep.ambiguity),system34:true};
    this.lastGraph=result;return result;
  };
  VilotTemporalKnowledge.prototype.parseDuration=function(text){this.v34Engine||=new V34TemporalEngine(this);return this.v34Engine.parseDuration(text);};
  VilotTemporalKnowledge.prototype.parseFrequency=function(text){this.v34Engine||=new V34TemporalEngine(this);return this.v34Engine.parseFrequency(text);};
  VilotTemporalKnowledge.prototype.parseAspect=function(text){this.v34Engine||=new V34TemporalEngine(this);return this.v34Engine.parseAspect(text);};
  VilotTemporalKnowledge.prototype.reasonTimeline=function(events,relations){this.v34Engine||=new V34TemporalEngine(this);return this.v34Engine.reason(events,relations);};



  const V34_TEMPORAL_SCENARIOS = Object.freeze([
    {
      id: "ordering-scenario-1",
      family: "ordering",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "overlap-scenario-2",
      family: "overlap",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "duration-scenario-3",
      family: "duration",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "frequency-scenario-4",
      family: "frequency",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "aspect-scenario-5",
      family: "aspect",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "since-until-scenario-6",
      family: "since-until",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "interval-scenario-7",
      family: "interval",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "event-chain-scenario-8",
      family: "event-chain",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "relative-time-scenario-9",
      family: "relative-time",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "freshness-scenario-10",
      family: "freshness",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "ordering-scenario-11",
      family: "ordering",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "overlap-scenario-12",
      family: "overlap",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "duration-scenario-13",
      family: "duration",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "frequency-scenario-14",
      family: "frequency",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "aspect-scenario-15",
      family: "aspect",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "since-until-scenario-16",
      family: "since-until",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "interval-scenario-17",
      family: "interval",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "event-chain-scenario-18",
      family: "event-chain",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "relative-time-scenario-19",
      family: "relative-time",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "freshness-scenario-20",
      family: "freshness",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "ordering-scenario-21",
      family: "ordering",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "overlap-scenario-22",
      family: "overlap",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "duration-scenario-23",
      family: "duration",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "frequency-scenario-24",
      family: "frequency",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "aspect-scenario-25",
      family: "aspect",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "since-until-scenario-26",
      family: "since-until",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "interval-scenario-27",
      family: "interval",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "event-chain-scenario-28",
      family: "event-chain",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "relative-time-scenario-29",
      family: "relative-time",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "freshness-scenario-30",
      family: "freshness",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "ordering-scenario-31",
      family: "ordering",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "overlap-scenario-32",
      family: "overlap",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "duration-scenario-33",
      family: "duration",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "frequency-scenario-34",
      family: "frequency",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "aspect-scenario-35",
      family: "aspect",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "since-until-scenario-36",
      family: "since-until",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "interval-scenario-37",
      family: "interval",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "event-chain-scenario-38",
      family: "event-chain",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "relative-time-scenario-39",
      family: "relative-time",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "freshness-scenario-40",
      family: "freshness",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "ordering-scenario-41",
      family: "ordering",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "overlap-scenario-42",
      family: "overlap",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "duration-scenario-43",
      family: "duration",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "frequency-scenario-44",
      family: "frequency",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "aspect-scenario-45",
      family: "aspect",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "since-until-scenario-46",
      family: "since-until",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "interval-scenario-47",
      family: "interval",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "event-chain-scenario-48",
      family: "event-chain",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "relative-time-scenario-49",
      family: "relative-time",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "freshness-scenario-50",
      family: "freshness",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "ordering-scenario-51",
      family: "ordering",
      complexity: 1,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "overlap-scenario-52",
      family: "overlap",
      complexity: 2,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
    {
      id: "duration-scenario-53",
      family: "duration",
      complexity: 3,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.75,
    },
    {
      id: "frequency-scenario-54",
      family: "frequency",
      complexity: 4,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.8,
    },
    {
      id: "aspect-scenario-55",
      family: "aspect",
      complexity: 5,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.85,
    },
    {
      id: "since-until-scenario-56",
      family: "since-until",
      complexity: 1,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.9,
    },
    {
      id: "interval-scenario-57",
      family: "interval",
      complexity: 2,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.55,
    },
    {
      id: "event-chain-scenario-58",
      family: "event-chain",
      complexity: 3,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.6,
    },
    {
      id: "relative-time-scenario-59",
      family: "relative-time",
      complexity: 4,
      requires_context: false,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.65,
    },
    {
      id: "freshness-scenario-60",
      family: "freshness",
      complexity: 5,
      requires_context: true,
      preserve_structure: true,
      allow_ambiguity: false,
      verification: ["coverage", "consistency", "uncertainty"],
      recovery: ["localize", "retry", "verify"],
      weight: 0.7,
    },
  ]);
  VilotTemporalKnowledge.prototype.temporalScenarios = function() { return V34_TEMPORAL_SCENARIOS.map(row => ({ ...row })); };

  globalThis.VilotTemporalKnowledge = VilotTemporalKnowledge;
})();
