/**
 * VilotNI 2.5 - KeywordSelector.js
 *
 * Meaning-first keyword selection for the 2.5 response path.
 *
 * This stage intentionally keeps content words and semantic anchors separate
 * from function words. SyntacticBridge is responsible for placing grammatical
 * bridges around those anchors later. It does not emit definitions or answers.
 */
(() => {
  'use strict';

  const CONTENT_POS = new Set(['Noun', 'ProperNoun', 'Verb', 'Adj', 'Adv', 'Num']);
  const FUNCTION_WORDS = new Set([
    'a','an','the','and','or','but','because','if','while','although','though',
    'of','to','in','on','at','for','from','with','without','by','through','into',
    'onto','over','under','between','among','during','before','after','as','than',
    'is','am','are','was','were','be','been','being','do','does','did','have',
    'has','had','can','could','will','would','shall','should','may','might','must'
  ]);

  const normalize = value => String(value || '')
    .toLowerCase()
    .replace(/[’]/g, "'")
    .replace(/[^a-z0-9'+.-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  class VilotKeywordSelector {
    constructor(processor, options = {}) {
      this.processor = processor;
      this.options = {
        enabled: options.keywordSelectionEnabled !== false,
        maxKeywordsPerClause: Math.max(4, Math.min(32, options.keywordSelectionMaxPerClause | 0 || 18)),
        preserveNumbers: options.keywordSelectionPreserveNumbers !== false
      };
      this.selections = 0;
      this.lastDiagnostics = null;
    }

    _tokens(text) {
      const source = String(text || '').trim();
      if (!source) return [];
      const tokens = this.processor?.tokenize?.(source, 96) || [];
      if (tokens.length) return tokens;
      return normalize(source).split(/\s+/).filter(Boolean).map(word => ({
        surface: word,
        lower: word,
        pos: 'Other'
      }));
    }

    _isContent(token) {
      const lower = normalize(token?.lower || token?.surface);
      const pos = String(token?.pos || token?.activePOS || token?.entry?.pos || 'Other');
      if (!lower || FUNCTION_WORDS.has(lower) || pos === 'Punct') return false;
      if (CONTENT_POS.has(pos)) return true;
      if (this.options.preserveNumbers && /^[-+]?\d+(?:\.\d+)?$/.test(lower)) return true;
      // Unknown lexical items can still be useful anchors, but only when they
      // look like ordinary words rather than punctuation/function material.
      return pos === 'Other' && /^[a-z0-9][a-z0-9'+.-]*$/.test(lower);
    }

    _extract(text, role, baseScore = 1) {
      const out = [];
      const seen = new Set();
      for (const token of this._tokens(text)) {
        if (!this._isContent(token)) continue;
        const lower = normalize(token?.lower || token?.surface);
        if (!lower || seen.has(lower)) continue;
        seen.add(lower);
        out.push({
          word: lower,
          surface: String(token?.surface || token?.lower || lower),
          pos: String(token?.pos || token?.activePOS || token?.entry?.pos || 'Other'),
          role,
          score: Number(baseScore) || 1
        });
        if (out.length >= this.options.maxKeywordsPerClause) break;
      }
      return out;
    }

    select(plan, analysis = null) {
      const started = performance.now();
      const clauses = Array.isArray(plan?.clauses) ? plan.clauses : [];
      if (!this.options.enabled) {
        return { enabled: false, groups: [], byClauseIndex: Object.create(null), keywords: [], diagnostics: { selectorMs: performance.now() - started } };
      }

      const groups = [];
      const all = [];
      const byClauseIndex = Object.create(null);

      for (let i = 0; i < clauses.length; i++) {
        const clause = clauses[i] || {};
        const subject = this._extract(clause.subject, 'subject', 1.0);
        const value = this._extract(clause.value, 'value', 0.94);
        const relation = this._extract(clause.relation, 'relation', 0.88);

        const group = {
          index: Number.isFinite(Number(clause.index)) ? Number(clause.index) : i,
          type: String(clause.type || 'relation'),
          subject,
          value,
          relation,
          subjectText: subject.map(item => item.surface).join(' ').trim(),
          valueText: value.map(item => item.surface).join(' ').trim(),
          relationText: relation.map(item => item.surface).join(' ').trim()
        };

        groups.push(group);
        byClauseIndex[group.index] = group;
        all.push(...subject, ...relation, ...value);
      }

      const unique = [];
      const seen = new Set();
      for (const item of all) {
        const key = `${item.role}:${item.word}`;
        if (seen.has(key)) continue;
        seen.add(key);
        unique.push(item);
      }

      const result = {
        enabled: true,
        groups,
        byClauseIndex,
        keywords: unique,
        diagnostics: {
          selectorMs: performance.now() - started,
          clauseCount: groups.length,
          keywordCount: unique.length,
          explanationMode: analysis?.explanationMode || null
        }
      };

      this.selections++;
      this.lastDiagnostics = result.diagnostics;
      return result;
    }

    status() {
      return {
        ready: true,
        role: 'content-keyword-selector',
        selections: this.selections,
        lastDiagnostics: this.lastDiagnostics,
        architecture: {
          stripsFunctionWordsBeforeBridging: true,
          emitsDefinitions: false,
          preservesSemanticClauseOrder: true
        }
      };
    }
  }

  globalThis.VilotKeywordSelector = VilotKeywordSelector;
})();
