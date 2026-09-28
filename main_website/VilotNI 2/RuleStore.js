/**
 * VilotNI 2 - RuleStore.js
 *
 * Loads compact structural/routing datasets that are intentionally separate
 * from Words.json and TrainingInfo.json. These files contain grammar,
 * punctuation, query-routing, context-dependency and relation metadata. They do
 * not contain prompt-specific answers and do not bypass the Decoder confidence
 * gate.
 */
(() => {
  'use strict';

  const normalize = value => String(value || '').trim().toLowerCase();

  class VilotRuleStore {
    constructor(options = {}) {
      this.options = { ...options };
      this.ready = false;
      this.loadedAt = 0;
      this.lastLoadMs = 0;
      this.errors = [];
      this.data = Object.create(null);
    }

    async _loadJSON(url, key) {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) throw new Error(`${key} failed to load: HTTP ${response.status}`);
      const value = await response.json();
      this.data[key] = value && typeof value === 'object' ? value : {};
      return this.data[key];
    }

    async load(baseURL = './', version = 'v=2-rule-data-1') {
      const started = performance.now();
      const root = String(baseURL || './').replace(/\/?$/, '/');
      const suffix = version ? `?${String(version).replace(/^\?/, '')}` : '';
      const files = {
        grammar: 'GrammarRules.json',
        punctuation: 'PunctuationRules.json',
        query: 'QueryPatterns.json',
        context: 'ContextRules.json',
        semantic: 'SemanticRelations.json',
        definitionIndex: 'DefinitionIndex.json'
      };

      this.errors.length = 0;
      const tasks = Object.entries(files).map(async ([key, file]) => {
        try {
          await this._loadJSON(`${root}${file}${suffix}`, key);
        } catch (error) {
          this.errors.push({ key, file, message: String(error?.message || error) });
          this.data[key] = {};
        }
      });
      await Promise.all(tasks);
      this.ready = true;
      this.loadedAt = Date.now();
      this.lastLoadMs = performance.now() - started;
      return this.status();
    }

    get(name) {
      return this.data[String(name || '')] || null;
    }

    punctuation(symbol) {
      const key = String(symbol || '');
      return this.data.punctuation?.Symbols?.[key] || null;
    }

    commaRules() {
      return this.data.punctuation?.Comma || null;
    }

    terminalRules() {
      return this.data.punctuation?.Terminal || null;
    }

    determiner(word) {
      return this.data.grammar?.Determiners?.[normalize(word)] || null;
    }

    agreement() {
      return this.data.grammar?.Agreement || null;
    }

    coordination() {
      return this.data.grammar?.Coordination || null;
    }

    clauseRules() {
      return this.data.grammar?.Clause || null;
    }

    roleBridges(role) {
      const key = normalize(role);
      const direct = this.data.query?.Role_Bridges?.[key];
      if (Array.isArray(direct)) return direct.slice();
      const relation = this.data.semantic?.Relations?.[key]?.cues;
      return Array.isArray(relation) ? relation.slice() : [];
    }

    operatorRule(operator) {
      return this.data.query?.Question_Operators?.[normalize(operator)] || null;
    }

    selfContainedRules() {
      return this.data.query?.Self_Contained || null;
    }

    contextRules() {
      return this.data.context || null;
    }

    exactEllipsis(text) {
      const clean = normalize(text)
        .replace(/[?!.,;:]+$/g, '')
        .replace(/\s+/g, ' ')
        .trim();
      return this.data.context?.Exact_Ellipsis?.[clean] || null;
    }

    relation(role) {
      return this.data.semantic?.Relations?.[normalize(role)] || null;
    }

    definitionEntry(word) {
      return this.data.definitionIndex?.Entries?.[normalize(word)] || null;
    }

    definitionEntries(words, limit = 8) {
      const out = [];
      const seen = new Set();
      const max = Math.max(1, Math.min(64, limit | 0 || 8));
      for (const word of words || []) {
        const key = normalize(word);
        if (!key || seen.has(key)) continue;
        const entry = this.definitionEntry(key);
        if (!entry) continue;
        seen.add(key);
        out.push({ word: key, ...entry });
        if (out.length >= max) break;
      }
      return out;
    }

    status() {
      const definitionCount = Number(this.data.definitionIndex?.Generated_Entry_Count || 0);
      return {
        role: 'structural-rule-and-routing-data',
        ready: this.ready,
        lastLoadMs: this.lastLoadMs,
        loadedAt: this.loadedAt,
        errors: this.errors.slice(),
        datasets: {
          grammar: Boolean(this.data.grammar?.Schema_Version),
          punctuation: Boolean(this.data.punctuation?.Schema_Version),
          query: Boolean(this.data.query?.Schema_Version),
          context: Boolean(this.data.context?.Schema_Version),
          semantic: Boolean(this.data.semantic?.Schema_Version),
          definitionIndex: Boolean(this.data.definitionIndex?.Schema_Version)
        },
        definitionIndexEntries: definitionCount
      };
    }
  }

  globalThis.VilotRuleStore = VilotRuleStore;
})();
