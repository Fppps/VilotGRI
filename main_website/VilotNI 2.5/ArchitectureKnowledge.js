/**
 * VilotNI 2.5 - ArchitectureKnowledge.js
 *
 * Structured self-model of the runtime. This is architecture data, not canned
 * answers. It lets ordinary graph reasoning and evidence search reason about the
 * model's own components the same way they reason about other structured facts.
 */
(() => {
  'use strict';
  const norm = value => String(value || '').toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9'+.\- ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const uniq = values => Array.from(new Set((values || []).map(v => String(v || '').trim()).filter(Boolean)));
  const clamp = (x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));
  const STOP = new Set(['the','a','an','of','to','for','and','or','in','on','with','about','please','explain','describe','tell','show','me','how','why','what','which','who','does','do','did','is','are','was','were','can','could','would','should','will','work','works','working','over','time','vilotni','2.5','2']);
  const stem = value => {
    let s = norm(value);
    if (!s) return '';
    if (s.length > 6 && s.endsWith('ments')) s = s.slice(0,-5);
    if (s.length > 5 && s.endsWith('ment')) s = s.slice(0,-4);
    if (s.length > 5 && s.endsWith('ing')) s = s.slice(0,-3);
    if (s === 'matrices') s = 'matrix';
    else if (s.length > 4 && s.endsWith('ies')) s = `${s.slice(0,-3)}y`;
    else if (s.length > 4 && s.endsWith('ed')) s = s.slice(0,-2);
    else if (s.length > 4 && s.endsWith('es')) {
      if (/(?:ches|shes|sses|xes|zes)$/.test(s)) s = s.slice(0,-2);
      else s = s.slice(0,-1);
    }
    else if (s.length > 3 && s.endsWith('s') && !s.endsWith('ss')) s = s.slice(0,-1);
    return s;
  };
  const meaningful = value => norm(value).split(/\s+/).filter(Boolean).filter(w => !STOP.has(w)).map(stem).filter(Boolean);

  class VilotArchitectureKnowledge {
    constructor(options = {}) {
      this.options = {
        maxMatches: Math.max(1, Math.min(16, options.architectureKnowledgeMaxMatches | 0 || 8)),
        graphSeedTrust: clamp(options.architectureGraphSeedTrust ?? 0.995, 0.85, 1),
        graphSeedStrength: clamp(options.architectureGraphSeedStrength ?? 0.97, 0.65, 1)
      };
      this.ready = false;
      this.model = 'VilotNI 2.5';
      this.components = [];
      this.aliases = new Map();
      this.resolutions = 0;
      this.graphSeeds = 0;
      this.last = null;
    }

    async load(url = './ArchitectureKnowledge.json') {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) throw new Error(`ArchitectureKnowledge could not load ${url}: HTTP ${response.status}`);
      const data = await response.json();
      this.model = String(data?.Model || 'VilotNI 2.5');
      this.components = Array.isArray(data?.Components) ? data.Components.filter(row => row?.canonical) : [];
      this.aliases.clear();
      for (const component of this.components) {
        for (const alias of uniq([component.canonical, component.id, ...(component.aliases || [])])) {
          const key = norm(alias);
          if (key) this.aliases.set(key, component);
        }
        const searchable = [
          component.canonical,
          ...(component.aliases || []),
          component.purpose || '',
          ...(component.inputs || []),
          ...(component.outputs || []),
          ...(component.concepts || []),
          ...Object.keys(component.relations || {}),
          ...Object.values(component.relations || {}).flat()
        ];
        component.__searchStems = Array.from(new Set(searchable.flatMap(meaningful)));
      }
      this.ready = true;
      return this.status();
    }

    resolve(text, limit = this.options.maxMatches) {
      const input = norm(text);
      if (!input || !this.ready) return [];
      const rows = [];
      for (const [alias, component] of this.aliases) {
        if (!(` ${input} `).includes(` ${alias} `)) continue;
        rows.push({
          id: component.id || null,
          canonical: component.canonical,
          alias,
          type: component.type || 'architecture-component',
          purpose: component.purpose || '',
          inputs: uniq(component.inputs),
          outputs: uniq(component.outputs),
          concepts: uniq(component.concepts),
          relations: component.relations || {},
          confidence: input === alias ? 0.995 : Math.min(0.99, 0.93 + Math.min(0.05, alias.split(/\s+/).length * 0.01))
        });
      }

      // If the user explicitly references VilotNI but describes a component by
      // behavior instead of its exact name (for example, "improves itself over
      // time"), resolve by structured descriptor overlap. This remains a data
      // lookup over architecture metadata, not a canned response route.
      const modelMention = /\bvilotni(?:\s*2(?:\.5)?)?\b/.test(input);
      if (modelMention) {
        const promptStems = meaningful(input);
        const promptSet = new Set(promptStems);
        for (const component of this.components) {
          const search = new Set(component.__searchStems || []);
          let hits = 0;
          for (const token of promptSet) if (search.has(token)) hits++;
          if (!hits) continue;
          const aliasStemSets = uniq([component.canonical, ...(component.aliases || [])]).map(v => new Set(meaningful(v)));
          let aliasHits = 0;
          for (const set of aliasStemSets) {
            let local = 0;
            for (const token of promptSet) if (set.has(token)) local++;
            aliasHits = Math.max(aliasHits, local);
          }
          if (hits < 2 && aliasHits < 1) continue;
          const denom = Math.max(2, Math.min(promptSet.size || 1, 8));
          const confidence = Math.min(0.91, 0.68 + Math.min(0.15, hits / denom * 0.18) + Math.min(0.08, aliasHits * 0.04));
          rows.push({
            id: component.id || null,
            canonical: component.canonical,
            alias: 'descriptor-overlap',
            type: component.type || 'architecture-component',
            purpose: component.purpose || '',
            inputs: uniq(component.inputs),
            outputs: uniq(component.outputs),
            concepts: uniq(component.concepts),
            relations: component.relations || {},
            confidence
          });
        }
      }
      rows.sort((a,b)=>b.confidence-a.confidence || b.alias.length-a.alias.length);
      const selected=[]; const seen=new Set();
      for (const row of rows) {
        const key=norm(row.canonical); if(seen.has(key)) continue;
        seen.add(key); selected.push(row); if(selected.length>=limit) break;
      }
      this.resolutions++;
      this.last={at:Date.now(),input:String(text||'').slice(0,256),matches:selected.map(x=>x.canonical)};
      return selected;
    }

    enrichAnalysis(analysis, promptText) {
      const matches = this.resolve(promptText);
      if (!matches.length) return analysis;
      const concepts = uniq(matches.flatMap(row => [row.canonical, ...row.inputs, ...row.outputs, ...row.concepts, ...Object.keys(row.relations || {}), ...Object.values(row.relations || {}).flat()]));
      analysis.architectureComponents = matches;
      analysis.architectureConcepts = concepts.slice(0, 64);
      analysis.architectureEvidence = matches.flatMap(row => {
        const out = [];
        if (row.purpose) out.push({ from: row.canonical, relation: 'purpose', to: row.purpose, score: row.confidence, source: 'architecture-purpose' });
        for (const value of row.inputs || []) out.push({ from: row.canonical, relation: 'receives', to: value, score: row.confidence * 0.97, source: 'architecture-input' });
        for (const value of row.outputs || []) out.push({ from: row.canonical, relation: 'produces', to: value, score: row.confidence * 0.97, source: 'architecture-output' });
        for (const [relation, targets] of Object.entries(row.relations || {})) {
          for (const value of Array.isArray(targets) ? targets : [targets]) out.push({ from: row.canonical, relation, to: value, score: row.confidence * 0.98, source: 'architecture-relation' });
        }
        return out;
      }).slice(0, 48);
      analysis.selfModelQuery = true;
      return analysis;
    }

    seedKnowledgeGraph(graph) {
      if (!graph?.observeEdge || !this.ready) return 0;
      let count = 0;
      const observe = (from, relation, to, metadata={}) => {
        if (!from || !relation || !to || norm(from)===norm(to)) return;
        const old = graph.getEdge?.(from, relation, to);
        if (old?.sourceKinds?.architecture_seed) return;
        const edge = graph.observeEdge(from, relation, to, {
          provenance: 'architecture_seed',
          trust: this.options.graphSeedTrust,
          strength: this.options.graphSeedStrength,
          usefulness: 0.94,
          domain: 'vilotni-architecture',
          timeless: true,
          ...metadata
        });
        if (edge) count++;
      };
      for (const component of this.components) {
        const name = component.canonical;
        observe(name, 'part_of', this.model, { entryId: component.id || null });
        observe(name, 'type', component.type || 'architecture-component', { entryId: component.id || null });
        for (const concept of uniq(component.concepts)) observe(name, 'associated_with', concept, { entryId: component.id || null });
        for (const input of uniq(component.inputs)) observe(name, 'receives', input, { entryId: component.id || null });
        for (const output of uniq(component.outputs)) observe(name, 'produces', output, { entryId: component.id || null });
        for (const [relation, targets] of Object.entries(component.relations || {})) {
          for (const target of uniq(Array.isArray(targets) ? targets : [targets])) observe(name, relation, target, { entryId: component.id || null });
        }
        for (const alias of uniq(component.aliases)) if (norm(alias)!==norm(name)) observe(name, 'alias', alias, { entryId: component.id || null });
      }
      this.graphSeeds += count;
      return count;
    }

    status() {
      return { ready:this.ready, role:'structured-runtime-self-model', model:this.model, components:this.components.length, aliases:this.aliases.size, resolutions:this.resolutions, graphSeeds:this.graphSeeds, last:this.last };
    }
  }

  globalThis.VilotArchitectureKnowledge = VilotArchitectureKnowledge;
})();
