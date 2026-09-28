/**
 * VilotNI 2.5 - EntityResolver.js
 *
 * Real-world entity layer. It recognizes multi-word names before ordinary
 * keyword splitting, exposes compact semantic seeds, restores canonical casing,
 * and can seed the persistent KnowledgeGraph with trusted structured relations.
 * It never contains response templates and does not suppress semantic expansion.
 */
(() => {
  'use strict';
  const norm = value => String(value || '').toLowerCase().replace(/[’]/g,"'").replace(/[^a-z0-9'+.\- ]+/g,' ').replace(/\s+/g,' ').trim();
  const uniq = values => Array.from(new Set((values || []).map(v=>String(v||'').trim()).filter(Boolean)));
  const clamp = (x,lo=0,hi=1)=>Math.max(lo,Math.min(hi,Number.isFinite(Number(x))?Number(x):0));

  class VilotEntityResolver {
    constructor(processor, options = {}) {
      this.processor = processor;
      this.options = {
        maxMatches: Math.max(1, Math.min(12, options.entityMaxMatches | 0 || 6)),
        maxConcepts: Math.max(4, Math.min(48, options.entityMaxConcepts | 0 || 24)),
        graphSeedTrust: clamp(options.entityGraphSeedTrust ?? 0.97, 0.75, 1),
        graphSeedStrength: clamp(options.entityGraphSeedStrength ?? 0.92, 0.5, 1)
      };
      this.ready = false;
      this.entities = [];
      this.aliases = new Map();
      this.resolutions = 0;
      this.graphSeeds = 0;
      this.last = null;
    }

    async load(url='./EntityKnowledge.json') {
      const response = await fetch(url,{cache:'no-store'});
      if (!response.ok) throw new Error(`EntityResolver could not load ${url}: HTTP ${response.status}`);
      const data = await response.json();
      this.entities = Array.isArray(data?.Entities) ? data.Entities.filter(x=>x?.canonical && x?.type) : [];
      this.aliases.clear();
      for (const entity of this.entities) {
        const aliases = uniq([entity.canonical, ...(entity.aliases || [])]);
        for (const alias of aliases) {
          const key = norm(alias);
          if (!key) continue;
          const old = this.aliases.get(key);
          if (!old || key.length > norm(old.alias).length) this.aliases.set(key,{entity,alias});
        }
      }
      this.ready = true;
      return this.status();
    }

    _matchAlias(textNorm, aliasNorm) {
      if (!textNorm || !aliasNorm) return false;
      return (` ${textNorm} `).includes(` ${aliasNorm} `);
    }

    resolve(text, limit=this.options.maxMatches) {
      const input = norm(text);
      if (!input || !this.ready) return [];
      const candidates=[];
      for (const [alias,row] of this.aliases) {
        if (!this._matchAlias(input,alias)) continue;
        const wordCount=alias.split(/\s+/).length;
        const exact=input===alias;
        candidates.push({
          canonical:row.entity.canonical,
          id:row.entity.id || null,
          type:row.entity.type,
          alias:row.alias,
          roles:uniq(row.entity.roles),
          fields:uniq(row.entity.fields),
          concepts:uniq(row.entity.concepts),
          relations:row.entity.relations || {},
          realWorld:row.entity.real_world !== false,
          confidence:clamp((exact?0.99:0.91) + Math.min(0.04,(wordCount-1)*0.02)),
          exact
        });
      }
      candidates.sort((a,b)=>(b.exact-a.exact)||norm(b.alias).length-norm(a.alias).length||b.confidence-a.confidence);
      const selected=[]; const seen=new Set();
      for (const row of candidates) {
        const key=norm(row.canonical); if(seen.has(key)) continue;
        seen.add(key); selected.push(row); if(selected.length>=limit) break;
      }
      this.resolutions++;
      this.last={at:Date.now(),input:String(text||'').slice(0,256),matches:selected.map(x=>({canonical:x.canonical,type:x.type,confidence:x.confidence}))};
      return selected;
    }

    enrichAnalysis(analysis, promptText) {
      const entities=this.resolve(promptText);
      if (!entities.length) return analysis;
      const concepts=uniq(entities.flatMap(e=>[...e.roles,...e.fields,...e.concepts])).slice(0,this.options.maxConcepts);
      const canonicalTokens=uniq(entities.flatMap(e=>norm(e.canonical).split(/\s+/)));
      analysis.entities=entities;
      analysis.entityConcepts=concepts;
      analysis.entityCanonicalTokens=canonicalTokens;
      analysis.entityResolved=true;
      analysis.entityResolutionConfidence=Math.max(...entities.map(e=>e.confidence));
      return analysis;
    }

    seedKnowledgeGraph(graph) {
      if (!graph?.observeEdge) return 0;
      let count=0;
      for (const entity of this.entities) {
        const from=entity.canonical;
        const rels=entity.relations || {};
        for (const [relation,targets] of Object.entries(rels)) {
          for (const target of uniq(Array.isArray(targets)?targets:[targets])) {
            if (!target || norm(target)===norm(from)) continue;
            const existing=graph.getEdge?.(from,relation,target);
            if (existing?.sourceKinds?.entity_seed) continue;
            const edge=graph.observeEdge(from,relation,target,{
              provenance:'entity_seed', trust:this.options.graphSeedTrust,
              strength:this.options.graphSeedStrength, usefulness:0.82,
              domain:(entity.fields || [])[0] || 'real-world-person',
              entryId:entity.id || null, timeless:true
            });
            if (edge) count++;
          }
        }
        for (const alias of uniq(entity.aliases)) {
          if (!alias || norm(alias)===norm(from)) continue;
          const existingAlias=graph.getEdge?.(from,'alias',alias);
          if (existingAlias?.sourceKinds?.entity_seed) continue;
          if (graph.observeEdge(from,'alias',alias,{provenance:'entity_seed',trust:this.options.graphSeedTrust,strength:0.98,usefulness:0.76,timeless:true})) count++;
        }
      }
      this.graphSeeds += count;
      return count;
    }

    restoreCasing(text, matches=[]) {
      let out=String(text||'');
      const rows = matches?.length ? matches : this.resolve(out);
      for (const row of rows) {
        const canonical=String(row.canonical||'').trim();
        if (!canonical) continue;
        const escaped=canonical.split(/\s+/).map(x=>x.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('\\s+');
        out=out.replace(new RegExp(`\\b${escaped}\\b`,'gi'),canonical);
        for (const alias of [row.alias,...(row.aliases||[])]) {
          const a=String(alias||'').trim(); if(!a || a.length<4) continue;
          if(norm(a)===norm(canonical)) continue;
        }
      }
      return out;
    }

    status(){return {ready:this.ready,role:'real-world-multiword-entity-resolver',entities:this.entities.length,aliases:this.aliases.size,resolutions:this.resolutions,graphSeeds:this.graphSeeds,last:this.last};}
  }
  globalThis.VilotEntityResolver=VilotEntityResolver;
})();
