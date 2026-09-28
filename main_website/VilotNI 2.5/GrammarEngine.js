/**
 * VilotNI 2.5 - GrammarEngine.js
 * System 36 grammar foundation remaster.
 *
 * Shared structural grammar analyzer used by Processor, SurfaceRealizer,
 * SyntaxSmoother, SyntacticBridge, SentenceMixer and OutputCoherence.
 * Grammar rules come from GrammarRules.json; this engine does not contain
 * prompt-specific answers and never substitutes for semantic reasoning.
 */
(() => {
  'use strict';

  const clamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(Number(value)) ? Number(value) : 0));
  const norm = value => String(value ?? '').toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9'+.\- ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const uniq = rows => Array.from(new Set((rows || []).filter(Boolean)));
  const escapeRE = value => String(value ?? '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const wordsOf = value => String(value ?? '').trim().split(/\s+/).filter(Boolean);
  const first = rows => Array.isArray(rows) && rows.length ? rows[0] : null;

  class VilotGrammarEngine {
    constructor(processor, ruleStore, options = {}) {
      this.processor = processor;
      this.ruleStore = ruleStore;
      this.options = {
        enabled: options.grammarEngineEnabled !== false,
        maxIssues: Math.max(16, Math.min(256, Number(options.grammarEngineMaxIssues) || 96)),
        maxTokens: Math.max(32, Math.min(1024, Number(options.grammarEngineMaxTokens) || 512)),
        repairEnabled: options.grammarEngineRepairEnabled !== false,
        repairMax: Math.max(4, Math.min(64, Number(options.grammarEngineRepairMax) || 24))
      };
      this.audits = 0;
      this.repairs = 0;
      this.lastAudit = null;
      this.lastRepair = null;
      this.auditCache = new Map();
      this.auditCacheLimit = Math.max(64, Math.min(1024, Number(options.grammarEngineAuditCache) || 256));
    }

    rules() {
      return this.ruleStore?.grammarRules?.() || this.ruleStore?.safeDataset?.('grammar') || this.ruleStore?.get?.('grammar') || {};
    }

    section(name, fallback = {}) {
      const value = this.rules()?.[name];
      return value && typeof value === 'object' ? value : fallback;
    }

    tokenize(text) {
      const parsed = this.processor?.tokenize?.(String(text || ''), this.options.maxTokens) || [];
      if (parsed.length) return parsed.map((token, index) => ({
        ...token,
        index: Number.isFinite(Number(token?.index)) ? Number(token.index) : index,
        surface: String(token?.surface || token?.lower || ''),
        lower: norm(token?.lower || token?.surface || ''),
        pos: String(token?.activePOS || token?.pos || token?.entry?.pos || 'Other')
      }));
      const raw = String(text || '').match(/[A-Za-z0-9][A-Za-z0-9'.+-]*|[.,!?;:()[\]"-]/g) || [];
      return raw.map((surface, index) => ({ index, surface, lower: norm(surface), pos: /^[.,!?;:()[\]"-]$/.test(surface) ? 'Punct' : 'Other', entry: null }));
    }

    lexicalTokens(text) {
      return this.tokenize(text).filter(token => token.pos !== 'Punct');
    }

    _entry(token) {
      if (token?.entry) return token.entry;
      return this.processor?.lookup?.(token?.lower || token?.surface || '') || null;
    }

    _possiblePOS(token) {
      const entry = this._entry(token);
      const values = Array.isArray(entry?.possiblePOS) ? entry.possiblePOS : [token?.pos || entry?.pos || 'Other'];
      return new Set(values.map(value => String(value || '').replace(/[^A-Za-z]/g, '').toLowerCase()).filter(Boolean));
    }

    _isPOS(token, ...names) {
      const set = this._possiblePOS(token);
      return names.some(name => set.has(String(name || '').toLowerCase()));
    }

    _isVerbLike(token) {
      return this._isPOS(token, 'Verb', 'Aux', 'Modal') || this._isAuxSurface(token?.lower);
    }

    _isNominal(token) {
      return this._isPOS(token, 'Noun', 'Pronoun', 'ProperNoun', 'Name', 'Number') || /^(i|you|he|she|it|we|they|this|that|these|those|who|what|which|there)$/i.test(token?.surface || '');
    }

    _isAuxSurface(word) {
      const low = norm(word);
      const aux = this.section('Auxiliaries');
      const groups = [aux?.primary?.be, aux?.primary?.have, aux?.primary?.do, aux?.modal];
      return groups.some(group => Array.isArray(group) && group.includes(low));
    }

    _isModal(word) {
      const modals = this.section('Auxiliaries')?.modal || [];
      return Array.isArray(modals) && modals.includes(norm(word));
    }

    _isDeterminer(word) {
      return Boolean(this.section('Determiners')?.[norm(word)]);
    }

    _isPreposition(word) {
      return Boolean(this.section('Prepositions')?.[norm(word)]);
    }

    _isCoordinator(word) {
      return (this.section('Coordination')?.coordinators || []).includes(norm(word));
    }

    _isSubordinator(word) {
      const low = norm(word);
      const openers = this.section('Clause')?.subordinate_openers || [];
      return openers.includes(low) || Object.values(this.section('Clause_Types')).some(row => (row?.markers || []).includes(low));
    }

    _severityWeight(severity) {
      const table = this.section('Scoring')?.severity_penalties || {};
      return Number(table?.[severity]) || ({ fatal:0.42, major:0.18, moderate:0.09, minor:0.035 }[severity] || 0.05);
    }

    _issueSeverity(code, fallback = 'moderate') {
      return this.section('Diagnostics')?.issue_codes?.[code] || fallback;
    }

    _pushIssue(issues, code, detail = {}, severity = null) {
      if (issues.length >= this.options.maxIssues) return;
      issues.push({
        code,
        severity: severity || this._issueSeverity(code),
        ...detail
      });
    }

    _splitSentences(tokens) {
      const out = [];
      let current = [];
      for (const token of tokens) {
        current.push(token);
        if (token.pos === 'Punct' && /[.!?]/.test(token.surface)) {
          out.push(current);
          current = [];
        }
      }
      if (current.length) out.push(current);
      return out;
    }

    _content(tokens) {
      return (tokens || []).filter(token => token.pos !== 'Punct');
    }

    _lowerRows(tokens) {
      return this._content(tokens).map(token => token.lower);
    }

    _sentenceHasFinitePredicate(tokens) {
      const rows = this._content(tokens);
      if (!rows.length) return false;
      for (let i = 0; i < rows.length; i++) {
        const token = rows[i];
        const low = token.lower;
        if (this._isModal(low)) return Boolean(rows[i + 1]);
        if (/^(am|is|are|was|were|do|does|did|have|has|had|will|shall|can|could|may|might|must|should|would)$/.test(low)) return true;
        if (this._isPOS(token, 'Verb', 'Aux', 'Modal')) return true;
      }
      return false;
    }

    _sentenceHasSubject(tokens) {
      const rows = this._content(tokens);
      if (!rows.length) return false;
      const firstWord = rows[0]?.lower;
      if (this._isSubordinator(firstWord)) {
        const comma = tokens.findIndex(token => token.surface === ',');
        if (comma >= 0) return this._sentenceHasSubject(tokens.slice(0, comma)) || this._sentenceHasSubject(tokens.slice(comma + 1));
      }
      if (/^(please|let)$/i.test(firstWord)) return true;
      if (/^(what|who|which)$/i.test(firstWord) && this._sentenceHasFinitePredicate(rows.slice(1))) return true;
      if (/^(is|are|was|were|do|does|did|have|has|had|can|could|may|might|must|should|will|would)$/i.test(firstWord)) {
        return rows.slice(1, 5).some(token => this._isNominal(token));
      }
      return rows.slice(0, Math.min(rows.length, 8)).some(token => this._isNominal(token));
    }

    _looksImperative(tokens) {
      const rows = this._content(tokens);
      if (!rows.length) return false;
      const firstToken = rows[0];
      if (/^please$/i.test(firstToken.lower) && rows[1]) return this._isPOS(rows[1], 'Verb');
      return this._isPOS(firstToken, 'Verb') && !this._isNominal(firstToken);
    }

    _questionMode(tokens) {
      const rows = this._content(tokens);
      const firstWord = rows[0]?.lower || '';
      const wh = this.section('Question_Form')?.wh_words || [];
      if (wh.includes(firstWord)) return 'wh';
      if ((this.section('Question_Form')?.yes_no_openers || []).includes(firstWord)) return 'polar';
      return 'none';
    }

    _numberCue(token) {
      const low = norm(token?.lower || token?.surface);
      const cues = this.section('Agreement')?.subject_number_cues || {};
      if ((cues.first_singular || []).includes(low)) return 'first_singular';
      if ((cues.singular || []).includes(low)) return 'singular';
      if ((cues.plural || []).includes(low)) return 'plural';
      if ((cues.second || []).includes(low)) return 'second';
      const entry = this._entry(token);
      const grammar = entry?.grammar || entry?.morphology || {};
      const explicit = String(grammar.number || entry?.number || '').toLowerCase();
      if (explicit.includes('plural')) return 'plural';
      if (explicit.includes('singular')) return 'singular';
      if (this.processor?._inferNominalNumber && entry) {
        const inferred = this.processor._inferNominalNumber(entry);
        if (inferred === 'plural' || inferred === 'singular') return inferred;
      }
      if (/s$/i.test(low) && !/(ss|us|is|news|physics)$/i.test(low)) return 'plural';
      return 'unknown';
    }

    _findSubjectBefore(tokens, verbIndex) {
      for (let i = verbIndex - 1; i >= 0 && i >= verbIndex - 8; i--) {
        const token = tokens[i];
        if (token.pos === 'Punct' && /[.;:!?]/.test(token.surface)) break;
        if (this._isNominal(token)) return { token, index:i };
      }
      return null;
    }

    _verbForm(word) {
      const low = norm(word);
      if (!low) return 'unknown';
      if (/ing$/.test(low)) return 'present_participle';
      if (/(ed|en)$/.test(low) || /^(been|done|gone|known|made|built|shown|given|found|thought|taught|written|read|seen|said|had)$/.test(low)) return 'past_participle_or_past';
      if (/s$/.test(low) && !/ss$/.test(low)) return 'third_singular_present';
      return 'base_or_other';
    }

    _hasDanglingCopulaComplement(tokens) {
      const rows = this._content(tokens);
      if (!rows.length) return false;
      const last = rows[rows.length - 1];
      if (!/^(am|is|are|was|were)$/.test(last?.lower || '')) return false;
      // Preserve short conversational ellipsis such as "He is." while
      // rejecting generated definitional/explanatory fragments such as
      // "A quantum vector is."
      return rows.length >= 3;
    }

    _auditCompleteness(sentences, issues) {
      let total = 0;
      let good = 0;
      for (const sentence of sentences) {
        const rows = this._content(sentence);
        if (!rows.length) continue;
        total++;
        const imperative = this._looksImperative(sentence);
        const questionMode = this._questionMode(sentence);
        const hasSubject = imperative || this._sentenceHasSubject(sentence);
        const hasPredicate = this._sentenceHasFinitePredicate(sentence);
        const danglingCopula = this._hasDanglingCopulaComplement(sentence);
        if (!hasSubject && questionMode === 'none') this._pushIssue(issues, 'MISSING_SUBJECT', { span: rows.slice(0, 5).map(t => t.surface).join(' ') });
        if (!hasPredicate) this._pushIssue(issues, 'MISSING_FINITE_PREDICATE', { span: rows.map(t => t.surface).join(' ') });
        if (danglingCopula) this._pushIssue(issues, 'DANGLING_COPULA_COMPLEMENT', { word:rows.at(-1)?.surface, span:rows.map(t => t.surface).join(' ') }, 'major');
        if ((hasSubject || imperative || questionMode !== 'none') && hasPredicate && !danglingCopula) good++;
      }
      return total ? good / total : 0;
    }

    _auditDeterminers(tokens, issues) {
      let checks = 0, good = 0;
      for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];
        if (!this._isDeterminer(token.lower)) continue;
        checks++;
        const rule = this.section('Determiners')?.[token.lower] || {};
        const next = tokens.slice(i + 1, i + 5).find(row => row.pos !== 'Punct' && !this._isPOS(row, 'Adjective', 'Adverb'));
        if (!next) {
          this._pushIssue(issues, 'DANGLING_DETERMINER', { index:i, word:token.surface }, 'major');
          continue;
        }
        const number = this._numberCue(next);
        if (rule.number_requirement === 'singular' && number === 'plural') {
          this._pushIssue(issues, 'DETERMINER_NUMBER_MISMATCH', { index:i, determiner:token.surface, head:next.surface, expected:'singular', actual:number });
          continue;
        }
        if (rule.number_requirement === 'plural' && number === 'singular') {
          this._pushIssue(issues, 'DETERMINER_NUMBER_MISMATCH', { index:i, determiner:token.surface, head:next.surface, expected:'plural', actual:number });
          continue;
        }
        if ((token.lower === 'a' || token.lower === 'an') && next?.surface) {
          const onset = /^[aeiou]/i.test(next.surface) ? 'vowel' : 'consonant';
          if ((token.lower === 'a' && onset === 'vowel') || (token.lower === 'an' && onset === 'consonant')) {
            this._pushIssue(issues, 'ARTICLE_ONSET_MISMATCH', { index:i, article:token.surface, following:next.surface }, 'minor');
            continue;
          }
        }
        good++;
      }
      return checks ? good / checks : 1;
    }

    _auditAgreement(tokens, issues) {
      const rows = this._content(tokens);
      let checks = 0, good = 0;
      for (let i = 0; i < rows.length; i++) {
        const word = rows[i].lower;
        if (!/^(am|is|are|was|were|has|have|does|do)$/.test(word)) continue;
        const originalIndex = tokens.indexOf(rows[i]);
        const found = this._findSubjectBefore(tokens, originalIndex);
        if (!found) continue;
        checks++;
        const number = this._numberCue(found.token);
        let ok = true;
        if (number === 'first_singular') ok = ['am','was','have','do'].includes(word);
        else if (number === 'singular') ok = ['is','was','has','does'].includes(word);
        else if (number === 'plural') ok = ['are','were','have','do'].includes(word);
        else if (number === 'second') ok = ['are','were','have','do'].includes(word);
        if (!ok) this._pushIssue(issues, 'SUBJECT_VERB_AGREEMENT', { subject:found.token.surface, verb:rows[i].surface, number });
        else good++;
      }
      return checks ? good / checks : 1;
    }

    _auditAuxChains(tokens, issues) {
      const rows = this._content(tokens);
      let checks = 0, good = 0;
      for (let i = 0; i < rows.length; i++) {
        const low = rows[i].lower;
        if (!this._isModal(low) && !/^(have|has|had|am|is|are|was|were|be|been|being|do|does|did)$/.test(low)) continue;
        const next = rows[i + 1];
        if (!next) {
          if (this._isModal(low) || /^(do|does|did|have|has|had)$/.test(low)) this._pushIssue(issues, 'DANGLING_MODAL', { word:rows[i].surface }, 'major');
          continue;
        }
        checks++;
        const form = this._verbForm(next.lower);
        let ok = true;
        if (this._isModal(low) && form === 'third_singular_present') ok = false;
        if (/^(do|does|did)$/.test(low) && form === 'third_singular_present') ok = false;
        if (/^(have|has|had)$/.test(low) && this._isPOS(next, 'Verb') && !['past_participle_or_past','base_or_other'].includes(form)) ok = false;
        if (/^(am|is|are|was|were|be|been|being)$/.test(low) && this._isPOS(next, 'Verb') && form === 'third_singular_present') ok = false;
        if (!ok) this._pushIssue(issues, this._isModal(low) ? 'MODAL_INFLECTION' : 'AUX_CHAIN_ORDER', { auxiliary:rows[i].surface, following:next.surface, form });
        else good++;
      }
      return checks ? good / checks : 1;
    }

    _auditPronounCase(tokens, issues) {
      const pronouns = this.section('Pronouns');
      let checks = 0, good = 0;
      for (let i = 0; i < tokens.length; i++) {
        const low = tokens[i].lower;
        const info = pronouns?.[low];
        if (!info) continue;
        const prev = tokens[i - 1];
        const next = tokens[i + 1];
        let expected = null;
        if (prev && this._isPreposition(prev.lower)) expected = 'object';
        else if (i === 0 || (prev?.pos === 'Punct' && /[.!?;:]/.test(prev.surface)) || this._isSubordinator(prev?.lower)) expected = 'subject';
        else if (prev && this._isVerbLike(prev)) expected = 'object';
        if (!expected) continue;
        checks++;
        const c = String(info.case || '');
        const ok = expected === 'subject' ? /subject/.test(c) : /object/.test(c);
        if (!ok && !/reflexive|possessive/.test(c)) this._pushIssue(issues, 'PRONOUN_CASE', { pronoun:tokens[i].surface, expected, actual:c, index:i });
        else good++;
        if (/possessive_determiner/.test(c) && (!next || !this._isNominal(next))) this._pushIssue(issues, 'PRONOUN_CASE', { pronoun:tokens[i].surface, expected:'possessive_before_nominal', index:i }, 'minor');
      }
      return checks ? good / checks : 1;
    }

    _auditFunctionTerminals(tokens, issues) {
      const lexical = this._content(tokens);
      if (!lexical.length) return 0;
      const terminal = lexical[lexical.length - 1];
      if (/^(am|is|are|was|were)$/.test(terminal.lower) && lexical.length >= 3) {
        this._pushIssue(issues, 'DANGLING_COPULA_COMPLEMENT', { word:terminal.surface, index:terminal.index }, 'major');
        return 0;
      }
      const invalid = new Set(this.section('Generation_Constraints')?.avoid_terminal_function_words || []);
      if (invalid.has(terminal.lower)) {
        const code = this._isPreposition(terminal.lower) ? 'DANGLING_PREPOSITION' : this._isSubordinator(terminal.lower) ? 'DANGLING_SUBORDINATOR' : this._isModal(terminal.lower) ? 'DANGLING_MODAL' : 'DANGLING_DETERMINER';
        this._pushIssue(issues, code, { word:terminal.surface, index:terminal.index });
        return 0;
      }
      return 1;
    }

    _auditOrphanStart(tokens, issues) {
      const lexical = this._content(tokens);
      if (!lexical.length) return 0;
      const low = lexical[0].lower;
      const forbidden = this.section('Generation_Constraints')?.no_surface_may_start_with_orphan_bridge || [];
      if (forbidden.includes(low) && !this._isSubordinator(low)) {
        this._pushIssue(issues, 'ORPHAN_BRIDGE', { word:lexical[0].surface, index:lexical[0].index }, 'major');
        return 0;
      }
      return 1;
    }

    _auditCoordination(tokens, issues) {
      let checks = 0, good = 0;
      for (let i = 0; i < tokens.length; i++) {
        if (!this._isCoordinator(tokens[i].lower)) continue;
        checks++;
        const left = tokens.slice(0, i).reverse().find(token => token.pos !== 'Punct');
        const right = tokens.slice(i + 1).find(token => token.pos !== 'Punct');
        if (!left || !right || this._isCoordinator(left.lower) || this._isCoordinator(right.lower)) {
          this._pushIssue(issues, 'EMPTY_COORDINAND', { coordinator:tokens[i].surface, index:i }, 'major');
          continue;
        }
        const leftPos = this._possiblePOS(left), rightPos = this._possiblePOS(right);
        const overlap = Array.from(leftPos).some(pos => rightPos.has(pos));
        if (!overlap && ['and','or','nor'].includes(tokens[i].lower)) {
          this._pushIssue(issues, 'COORDINATION_PARALLELISM', { coordinator:tokens[i].surface, left:left.surface, right:right.surface }, 'minor');
        }
        good++;
      }
      return checks ? good / checks : 1;
    }

    _auditQuestions(sentences, issues) {
      let checks = 0, good = 0;
      for (const sentence of sentences) {
        const rows = this._content(sentence);
        if (!rows.length) continue;
        const terminal = sentence[sentence.length - 1]?.surface || '';
        const mode = this._questionMode(sentence);
        if (mode === 'none' && terminal !== '?') continue;
        checks++;
        if (terminal !== '?') {
          // Direct question forms in answer generation can sometimes be quoted or
          // embedded, so this remains a minor issue rather than a hard failure.
          this._pushIssue(issues, 'QUESTION_FORM', { issue:'direct_question_without_question_mark', span:rows.map(t=>t.surface).join(' ') }, 'minor');
          continue;
        }
        if (mode === 'none') {
          this._pushIssue(issues, 'QUESTION_FORM', { issue:'question_mark_without_interrogative_structure' });
          continue;
        }
        if (mode === 'wh') {
          const wh = rows[0]?.lower;
          const second = rows[1]?.lower;
          const subjectWh = ['who','what','which'].includes(wh) && rows[1] && this._isVerbLike(rows[1]);
          const auxInverted = this._isAuxSurface(second);
          if (!subjectWh && !auxInverted && rows.length > 2) {
            this._pushIssue(issues, 'QUESTION_FORM', { issue:'missing_auxiliary_inversion', wh, second });
            continue;
          }
        }
        good++;
      }
      return checks ? good / checks : 1;
    }

    _auditNegation(tokens, issues) {
      let checks = 0, good = 0;
      for (let i = 0; i < tokens.length; i++) {
        if (tokens[i].lower !== 'not') continue;
        checks++;
        const prev = tokens.slice(Math.max(0, i - 2), i).reverse().find(t => t.pos !== 'Punct');
        if (!prev || (!this._isAuxSurface(prev.lower) && prev.lower !== 'to')) {
          this._pushIssue(issues, 'NEGATION_SCOPE_RISK', { negation:tokens[i].surface, previous:prev?.surface || '' }, 'minor');
        } else good++;
      }
      return checks ? good / checks : 1;
    }

    _auditComparisons(tokens, issues) {
      const rows = this._content(tokens);
      let checks = 0, good = 0;
      for (let i = 0; i < rows.length; i++) {
        const low = rows[i].lower;
        if (low === 'than') {
          checks++;
          if (!rows[i - 1] || !rows[i + 1]) this._pushIssue(issues, 'COMPARISON_INCOMPLETE', { marker:'than' });
          else good++;
        }
        if (low === 'as' && rows[i + 2]?.lower === 'as') {
          checks++;
          if (!rows[i + 1]) this._pushIssue(issues, 'COMPARISON_INCOMPLETE', { marker:'as...as' });
          else good++;
        }
      }
      return checks ? good / checks : 1;
    }

    _auditRelativeClauses(tokens, issues) {
      const markers = new Set(this.section('Relative_Clause')?.markers || []);
      let checks = 0, good = 0;
      for (let i = 1; i < tokens.length; i++) {
        if (!markers.has(tokens[i].lower)) continue;
        if (['what','when','where'].includes(tokens[i].lower) && this._isVerbLike(tokens[i - 1])) continue;
        checks++;
        const antecedent = tokens.slice(Math.max(0, i - 5), i).reverse().find(token => this._isNominal(token));
        const rightPredicate = tokens.slice(i + 1, i + 8).some(token => this._isVerbLike(token));
        if (!antecedent || !rightPredicate) this._pushIssue(issues, 'RELATIVE_WITHOUT_ANTECEDENT', { marker:tokens[i].surface, antecedent:antecedent?.surface || null });
        else good++;
      }
      return checks ? good / checks : 1;
    }

    _auditVerbFrames(tokens, issues) {
      const rows = this._content(tokens);
      const frames = this.section('Verb_Frames');
      let checks = 0, good = 0;
      for (let i = 0; i < rows.length; i++) {
        const token = rows[i];
        if (!this._isPOS(token, 'Verb') && !frames?.[token.lower]) continue;
        const lemma = norm(this._entry(token)?.lemma || this._entry(token)?.root || token.lower).replace(/^(to )/, '');
        const spec = frames?.[lemma] || frames?.[token.lower];
        if (!spec) continue;
        checks++;
        const complements = Array.isArray(spec.complements) ? spec.complements : [];
        if (!complements.length) { good++; continue; }
        const after = rows.slice(i + 1);
        const hasNP = after.slice(0, 6).some(row => this._isNominal(row) || this._isDeterminer(row.lower));
        const hasClause = after.some(row => ['that','what','who','which','where','when','why','how','whether','if'].includes(row.lower));
        const hasToInf = after.some((row, j) => row.lower === 'to' && after[j + 1]);
        const canBeIntransitive = complements.some(frame => !String(frame.pattern || '').match(/\bNP\b.*\bNP\b|THAT_CLAUSE|WH_CLAUSE|TO_INF/));
        const expectsObject = complements.every(frame => String(frame.pattern || '').match(/\bNP\b.*\bNP\b|THAT_CLAUSE|WH_CLAUSE|TO_INF/));
        if (expectsObject && !hasNP && !hasClause && !hasToInf && !canBeIntransitive) {
          this._pushIssue(issues, 'VERB_FRAME_MISSING_ARGUMENT', { verb:token.surface, lemma });
          continue;
        }
        good++;
      }
      return checks ? good / checks : 1;
    }

    _auditPunctuation(text, tokens, issues) {
      let score = 1;
      const trimmed = String(text || '').trim();
      if (trimmed && !/[.!?]$/.test(trimmed)) {
        this._pushIssue(issues, 'TERMINAL_PUNCTUATION', { textEnd:trimmed.slice(-16) }, 'minor');
        score -= 0.15;
      }
      if (/\b[^.!?;]+,\s+(?:I|you|he|she|it|we|they|this|that|these|those|[A-Z][A-Za-z0-9'-]+)\s+(?:is|are|was|were|can|could|will|would|should|may|might|must|has|have|had|does|do|did)\b/.test(trimmed)) {
        this._pushIssue(issues, 'COMMA_SPLICE', {}, 'moderate');
        score -= 0.22;
      }
      if ((trimmed.match(/\(/g) || []).length !== (trimmed.match(/\)/g) || []).length) score -= 0.15;
      if ((trimmed.match(/"/g) || []).length % 2 !== 0) score -= 0.12;
      return clamp(score);
    }

    _componentScore(name, value) {
      return { name, score:clamp(value) };
    }

    audit(text, analysis = null, options = {}) {
      const started = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const source = String(text || '').trim();
      const cacheKey = source;
      if (cacheKey && options?.noCache !== true && this.auditCache.has(cacheKey)) {
        const cached = this.auditCache.get(cacheKey);
        this.auditCache.delete(cacheKey);
        this.auditCache.set(cacheKey, cached);
        this.lastAudit = cached;
        return { ...cached, cached:true, auditMs:0 };
      }
      const tokens = this.tokenize(source);
      const sentences = this._splitSentences(tokens);
      const issues = [];

      if (!this.options.enabled || !source) {
        const result = { score:0, grammarScore:0, issues:source ? [] : [{code:'EMPTY_SURFACE',severity:'fatal'}], components:{}, tokenCount:tokens.length, auditMs:0 };
        this.lastAudit = result;
        return result;
      }

      const components = {
        sentence_completeness: this._auditCompleteness(sentences, issues),
        agreement: this._auditAgreement(tokens, issues),
        determiners: this._auditDeterminers(tokens, issues),
        auxiliary_chain: this._auditAuxChains(tokens, issues),
        clause_structure: 1,
        question_structure: this._auditQuestions(sentences, issues),
        coordination: this._auditCoordination(tokens, issues),
        pronoun_case: this._auditPronounCase(tokens, issues),
        negation_scope: this._auditNegation(tokens, issues),
        verb_frame: this._auditVerbFrames(tokens, issues),
        punctuation: this._auditPunctuation(source, tokens, issues)
      };

      const terminalScore = this._auditFunctionTerminals(tokens, issues);
      const orphanScore = this._auditOrphanStart(tokens, issues);
      const relativeScore = this._auditRelativeClauses(tokens, issues);
      const comparisonScore = this._auditComparisons(tokens, issues);
      components.clause_structure = clamp((terminalScore + orphanScore + relativeScore + comparisonScore) / 4);

      const weights = this.section('Scoring')?.components || {};
      let weighted = 0;
      let weightSum = 0;
      for (const [name, value] of Object.entries(components)) {
        const weight = Number(weights?.[name]);
        const w = Number.isFinite(weight) && weight > 0 ? weight : 0.08;
        weighted += clamp(value) * w;
        weightSum += w;
      }
      let score = weightSum ? weighted / weightSum : 0;

      // Penalize unique structural faults without allowing repeated reports of
      // the same span to catastrophically zero an otherwise grammatical answer.
      const seenPenaltyKeys = new Set();
      let penalty = 0;
      for (const issue of issues) {
        const key = `${issue.code}|${issue.index ?? ''}|${issue.word ?? issue.verb ?? issue.marker ?? ''}`;
        if (seenPenaltyKeys.has(key)) continue;
        seenPenaltyKeys.add(key);
        penalty += this._severityWeight(issue.severity) * (issue.severity === 'minor' ? 0.35 : issue.severity === 'moderate' ? 0.28 : 0.22);
      }
      score = clamp(score - Math.min(0.42, penalty));
      // A surface with no complete clause must never receive a pass merely
      // because the other grammar dimensions had nothing to inspect.
      if (components.sentence_completeness < 0.5) score *= 0.45;
      if (components.clause_structure < 0.5) score *= 0.72;
      score = clamp(score);

      const fatal = issues.some(issue => issue.severity === 'fatal');
      const majorCount = issues.filter(issue => issue.severity === 'major').length;
      const strong = Number(this.section('Scoring')?.strong_surface ?? 0.86);
      const accept = !fatal && majorCount <= 1 && components.sentence_completeness >= 0.5 && score >= Number(this.section('Scoring')?.minimum_for_surface_acceptance ?? 0.70);
      const result = {
        score,
        grammarScore:score,
        accept,
        strong:score >= strong,
        tokenCount:tokens.length,
        sentenceCount:sentences.length,
        issueCount:issues.length,
        majorCount,
        issues:issues.slice(0, this.options.maxIssues),
        components,
        ruleSchema:Number(this.rules()?.Schema_Version || 0),
        build:String(this.rules()?.Build || ''),
        analysisSignals:{
          complexClauseCount:Number(analysis?.complexLanguage?.clauseCount || analysis?.complexLanguage?.clauses?.length || 0),
          propositionCount:Number(analysis?.propositionGraph?.propositions?.length || 0),
          mechanismStages:Number(analysis?.mechanismGraph?.stages?.length || 0)
        },
        auditMs:(typeof performance !== 'undefined' ? performance.now() : Date.now()) - started
      };
      this.audits++;
      this.lastAudit = result;
      if (cacheKey && options?.noCache !== true) {
        this.auditCache.set(cacheKey, result);
        while (this.auditCache.size > this.auditCacheLimit) this.auditCache.delete(this.auditCache.keys().next().value);
      }
      return result;
    }

    relationBridge(relation, options = {}) {
      const key = norm(relation).replace(/\s+/g, '_');
      const aliases = {
        causal:'cause', reason:'cause', why:'cause', effect:'result', consequence:'result',
        mechanism_stage:'sequence', operation:'mechanism', process:'mechanism', conditional:'condition',
        concession:'concession', temporal_before:'temporal_before', temporal_after:'temporal_after',
        simultaneous:'simultaneous', contrast:'contrast', comparison:'comparison', purpose:'purpose', feedback:'feedback',
        evidence:'evidence', example:'example', elaboration:'elaboration', alternative:'alternative', sequence:'sequence'
      };
      const target = aliases[key] || key;
      const row = this.section('Clause_Linkers')?.[target] || null;
      const choices = Array.isArray(row?.preferred) ? row.preferred.slice() : [];
      if (options.excludeInitialPrepositions) return choices.filter(value => !['by','through','using','with','of','to','from'].includes(norm(value)));
      return choices;
    }

    connectorForRelation(relation) {
      return first(this.relationBridge(relation)) || '';
    }

    explanationPlan(kind = 'mechanism') {
      const key = norm(kind).replace(/\s+/g, '_');
      const table = this.section('Explanation_Grammar');
      return table?.[key] || table?.mechanism || null;
    }

    verbFrame(word) {
      const low = norm(word);
      const entry = this.processor?.lookup?.(low);
      const lemma = norm(entry?.lemma || entry?.root || low);
      return this.section('Verb_Frames')?.[lemma] || this.section('Verb_Frames')?.[low] || null;
    }

    _safeReplace(text, regex, replacement, repairs, rule) {
      const next = String(text || '').replace(regex, replacement);
      if (next !== text && repairs.length < this.options.repairMax) repairs.push({rule,before:text,after:next});
      return next;
    }

    repair(text, analysis = null) {
      const original = String(text || '').trim();
      if (!this.options.repairEnabled || !original) return { text:original, repairs:[], before:null, after:null, changed:false };
      let out = original;
      const repairs = [];

      // Safe surface-only repairs.
      out = this._safeReplace(out, /\s+/g, ' ', repairs, 'grammar-collapse-whitespace');
      out = this._safeReplace(out, /\s+([,.;:!?])/g, '$1', repairs, 'grammar-punctuation-spacing');
      out = this._safeReplace(out, /([,;:!?])(?=[A-Za-z0-9])/g, '$1 ', repairs, 'grammar-punctuation-separation');
      out = this._safeReplace(out, /\b(a)\s+([aeiou][A-Za-z0-9'-]*)/gi, 'an $2', repairs, 'grammar-a-to-an');
      out = this._safeReplace(out, /\b(an)\s+([^aeiou\W][A-Za-z0-9'-]*)/gi, 'a $2', repairs, 'grammar-an-to-a');
      out = this._safeReplace(out, /\b(I)\s+(is|are)\b/g, 'I am', repairs, 'grammar-i-copula');
      out = this._safeReplace(out, /\b(he|she|it|this|that)\s+are\b/gi, '$1 is', repairs, 'grammar-singular-copula');
      out = this._safeReplace(out, /\b(we|they|these|those)\s+is\b/gi, '$1 are', repairs, 'grammar-plural-copula');
      out = this._safeReplace(out, /\b(can|could|may|might|must|shall|should|will|would)\s+([A-Za-z]+s)\b/gi, (match, modal, verb) => {
        // Only strip a simple third-person -s when the base form exists in the
        // lexicon, otherwise leave the unknown/domain word untouched.
        const base = verb.replace(/s$/i, '');
        const known = this.processor?.lookup?.(base);
        return known ? `${modal} ${base}` : match;
      }, repairs, 'grammar-modal-base-form');
      out = this._safeReplace(out, /\b(the|a|an|this|that|these|those)\s+\1\b/gi, '$1', repairs, 'grammar-duplicate-determiner');
      out = this._safeReplace(out, /\b(can|could|may|might|must|shall|should|will|would)\s+\1\b/gi, '$1', repairs, 'grammar-duplicate-modal');

      // Indefinite article before an explicit lexical plural.
      const tokenized = this.tokenize(out);
      for (let i = 0; i < tokenized.length - 1; i++) {
        const article = tokenized[i]?.lower;
        if (article !== 'a' && article !== 'an') continue;
        const next = tokenized.slice(i + 1).find(token => token.pos !== 'Punct');
        if (!next || this._numberCue(next) !== 'plural') continue;
        const re = new RegExp(`\\b(?:a|an)\\s+${escapeRE(next.surface)}\\b`, 'i');
        const replaced = out.replace(re, next.surface);
        if (replaced !== out && repairs.length < this.options.repairMax) {
          repairs.push({rule:'grammar-remove-article-before-plural',before:out,after:replaced,head:next.surface});
          out = replaced;
        }
      }

      if (out && !/[.!?]$/.test(out)) {
        const next = `${out}.`;
        repairs.push({rule:'grammar-terminal-punctuation',before:out,after:next});
        out = next;
      }

      const before = this.audit(original, analysis);
      const after = this.audit(out, analysis);
      // Do not keep a repair pass that makes grammar worse or loses visible
      // numeric/negation content. SyntaxSmoother performs an additional semantic
      // root-safety check after this layer.
      const protectedBefore = new Set((original.match(/\b(?:not|never|no|without|\d+(?:\.\d+)?)\b/gi) || []).map(norm));
      const protectedAfter = new Set((out.match(/\b(?:not|never|no|without|\d+(?:\.\d+)?)\b/gi) || []).map(norm));
      const protectedLost = Array.from(protectedBefore).some(value => !protectedAfter.has(value));
      if (protectedLost || after.score + 0.01 < before.score) {
        const result = { text:original, repairs:[], changed:false, rolledBack:true, before, after };
        this.lastRepair = result;
        return result;
      }
      const result = { text:out, repairs, changed:out !== original, rolledBack:false, before, after };
      this.repairs += repairs.length;
      this.lastRepair = result;
      return result;
    }

    status() {
      return {
        ready:Boolean(this.rules()?.Schema_Version),
        role:'shared-data-driven-grammar-engine',
        schema:Number(this.rules()?.Schema_Version || 0),
        build:String(this.rules()?.Build || ''),
        audits:this.audits,
        repairs:this.repairs,
        auditCacheSize:this.auditCache.size,
        verbFrames:Object.keys(this.section('Verb_Frames')).length,
        structuralScenarios:(this.section('Structural_Scenarios') || []).length,
        lastAudit:this.lastAudit ? {score:this.lastAudit.score,issueCount:this.lastAudit.issueCount,components:this.lastAudit.components} : null
      };
    }
  }

  globalThis.VilotGrammarEngine = VilotGrammarEngine;
})();
