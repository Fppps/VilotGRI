/*
 * VilotNI 1.5 - Type-Driven Phrase Parser v1
 * Phrase boundaries are derived from vocab JSON token TYPES, not fixed phrases.
 */
class VilotTypePhraseParser {
  static parse(tokens, api = {}) {
    const getType = typeof api.getType === "function" ? api.getType : (() => "Other");
    const getWord = typeof api.getWord === "function" ? api.getWord : (() => "");
    const getPronounProfile = typeof api.getPronounProfile === "function" ? api.getPronounProfile : (() => null);

    const entries = (tokens || []).map((token, index) => ({
      token,
      index,
      type: String(getType(token) || "Other"),
      word: String(getWord(token) || "")
    }));

    const isNominalType = type => ["Noun", "ProperNoun", "Pronoun", "Num"].includes(type);
    const isPredicateType = type => ["Verb", "Aux", "Modal"].includes(type);
    const isHelperType = type => ["Aux", "Modal"].includes(type);
    const isOperatorCandidateType = type => ["Adv", "Pronoun", "Det", "Adj", "Num"].includes(type);
    const isNPBody = type => ["Det", "Adj", "Noun", "ProperNoun", "Pronoun", "Num"].includes(type);

    const nonPunct = entries.filter(entry => entry.type !== "Punct");
    const first = nonPunct[0] || null;
    const last = entries.length ? entries[entries.length - 1] : null;
    const questionEnding = Boolean(last && last.type === "Punct" && last.word.trim() === "?");

    const phrases = [];
    let i = 0;
    while (i < entries.length) {
      const entry = entries[i];

      if (entry.type === "Punct") {
        phrases.push({kind:"PUNCT", start:i, end:i+1, tokenIds:[entry.token], types:[entry.type]});
        i++;
        continue;
      }

      if (entry.type === "Prep") {
        const start = i++;
        while (i < entries.length && isNPBody(entries[i].type)) i++;
        phrases.push({kind:"PP", start, end:i, tokenIds:entries.slice(start,i).map(x=>x.token), types:entries.slice(start,i).map(x=>x.type)});
        continue;
      }

      if (["Modal","Aux","Verb"].includes(entry.type)) {
        const start = i;
        let sawVerb = entry.type === "Verb";
        i++;
        while (i < entries.length) {
          const type = entries[i].type;
          if (type === "Adv") { i++; continue; }
          if (!sawVerb && ["Aux","Modal","Verb"].includes(type)) {
            if (type === "Verb") sawVerb = true;
            i++;
            continue;
          }
          break;
        }
        phrases.push({kind:"VP", start, end:i, tokenIds:entries.slice(start,i).map(x=>x.token), types:entries.slice(start,i).map(x=>x.type)});
        continue;
      }

      if (isNPBody(entry.type)) {
        const start = i;
        if (entry.type === "Pronoun") {
          i++;
        } else {
          i++;
          while (i < entries.length && isNPBody(entries[i].type) && entries[i].type !== "Pronoun") i++;
        }
        phrases.push({kind:"NP", start, end:i, tokenIds:entries.slice(start,i).map(x=>x.token), types:entries.slice(start,i).map(x=>x.type)});
        continue;
      }

      if (entry.type === "Adv") {
        const start = i;
        while (i < entries.length && entries[i].type === "Adv") i++;
        phrases.push({kind:"ADVP", start, end:i, tokenIds:entries.slice(start,i).map(x=>x.token), types:entries.slice(start,i).map(x=>x.type)});
        continue;
      }

      if (entry.type === "Conj") {
        phrases.push({kind:"CONJ", start:i, end:i+1, tokenIds:[entry.token], types:[entry.type]});
        i++;
        continue;
      }

      phrases.push({kind:"OTHER", start:i, end:i+1, tokenIds:[entry.token], types:[entry.type]});
      i++;
    }

    const clauses = [];
    let clauseStart = 0;
    const closeClause = end => {
      if (end <= clauseStart) { clauseStart = end; return; }
      const span = entries.slice(clauseStart, end).filter(item => item.type !== "Punct");
      if (!span.length) { clauseStart = end; return; }
      const lexicalVerb = span.find(item => item.type === "Verb") || null;
      const firstPredicate = lexicalVerb || span.find(item => isPredicateType(item.type)) || null;
      const predicateIndex = firstPredicate ? firstPredicate.index : -1;
      const subjectEntries = span.filter(item => isNominalType(item.type) && (predicateIndex < 0 || item.index < predicateIndex));
      const complementEntries = span.filter(item => predicateIndex >= 0 && item.index > predicateIndex && (isNominalType(item.type) || ["Adj","Adv"].includes(item.type)));
      clauses.push({
        start: clauseStart,
        end,
        tokenIds: span.map(item=>item.token),
        types: span.map(item=>item.type),
        predicateIndex,
        predicateToken: firstPredicate?.token ?? null,
        subjectTokenIds: subjectEntries.map(item=>item.token),
        complementTokenIds: complementEntries.map(item=>item.token),
        hasPredicate: predicateIndex >= 0,
        hasNominal: span.some(item=>isNominalType(item.type))
      });
      clauseStart = end;
    };

    for (let j=0; j<entries.length; j++) {
      if (entries[j].type === "Punct") {
        closeClause(j);
        clauseStart = j + 1;
      } else if (entries[j].type === "Conj" && j > clauseStart) {
        closeClause(j);
        clauseStart = j + 1;
      }
    }
    closeClause(entries.length);

    const firstIndex = first?.index ?? -1;
    const helperEntry = nonPunct.find(entry=>isHelperType(entry.type)) || null;
    const helperIndex = helperEntry?.index ?? -1;
    const helperFronted = firstIndex >= 0 && firstIndex === helperIndex;

    const lexicalVerbEntries = nonPunct.filter(entry=>entry.type === "Verb");
    const lexicalVerbIndices = lexicalVerbEntries.map(entry=>entry.index);
    const firstLexicalVerbIndex = lexicalVerbIndices[0] ?? -1;

    let grammaticalSubjectIndex = -1;
    const subjectSearchStart = helperFronted ? helperIndex + 1 : Math.max(0, firstIndex);
    const subjectSearchEnd = firstLexicalVerbIndex >= 0 ? firstLexicalVerbIndex : entries.length;
    for (let j=subjectSearchStart; j<subjectSearchEnd; j++) {
      if (isNominalType(entries[j]?.type)) { grammaticalSubjectIndex = j; break; }
    }
    if (grammaticalSubjectIndex < 0 && !helperFronted) {
      const fallback = nonPunct.find(entry=>isNominalType(entry.type));
      grammaticalSubjectIndex = fallback?.index ?? -1;
    }

    const grammaticalSubject = grammaticalSubjectIndex >= 0 ? entries[grammaticalSubjectIndex] : null;
    const subjectPronounProfile = grammaticalSubject?.type === "Pronoun" ? getPronounProfile(grammaticalSubject.token) : null;

    let matrixPredicateIndex = -1;
    for (const entry of lexicalVerbEntries) {
      if (grammaticalSubjectIndex < 0 || entry.index > grammaticalSubjectIndex) { matrixPredicateIndex = entry.index; break; }
    }
    if (matrixPredicateIndex < 0) matrixPredicateIndex = firstLexicalVerbIndex;

    let embeddedPredicateIndex = -1;
    for (const entry of lexicalVerbEntries) {
      if (matrixPredicateIndex >= 0 && entry.index > matrixPredicateIndex) { embeddedPredicateIndex = entry.index; break; }
    }
    const hasEmbeddedClause = embeddedPredicateIndex >= 0;

    let frontedOperatorIndex = -1;
    if (first && isOperatorCandidateType(first.type)) {
      let personalPronoun = false;
      if (first.type === "Pronoun") {
        const profile = getPronounProfile(first.token);
        personalPronoun = Boolean(profile && Number.isFinite(profile.person));
      }
      const hasPredicateAfter = lexicalVerbIndices.some(index=>index > first.index);
      const helperAfter = helperIndex > first.index;
      if (!personalPronoun && (helperAfter || (questionEnding && hasPredicateAfter))) frontedOperatorIndex = first.index;
    }

    let embeddedOperatorIndex = -1;
    if (matrixPredicateIndex >= 0 && embeddedPredicateIndex >= 0) {
      for (let j=matrixPredicateIndex+1; j<embeddedPredicateIndex; j++) {
        const item = entries[j];
        if (!item || !isOperatorCandidateType(item.type)) continue;
        if (item.type === "Pronoun") {
          const profile = getPronounProfile(item.token);
          if (profile && Number.isFinite(profile.person)) continue;
        }
        embeddedOperatorIndex = item.index;
        break;
      }
    }

    const openQuestionLike = frontedOperatorIndex >= 0;
    const closedQuestionLike = helperFronted && !openQuestionLike;
    const imperativeLike = first?.type === "Verb";

    let requestLikelihood = 0.0;
    if (helperFronted) requestLikelihood += 0.16;
    if (subjectPronounProfile?.person === 2) requestLikelihood += 0.28;
    if (matrixPredicateIndex >= 0) requestLikelihood += 0.12;
    const objectAfterMatrix = matrixPredicateIndex >= 0 && entries.some(item=>item.index > matrixPredicateIndex && isNominalType(item.type));
    if (objectAfterMatrix) requestLikelihood += 0.18;
    if (hasEmbeddedClause) requestLikelihood += 0.30;
    if (embeddedOperatorIndex >= 0) requestLikelihood += 0.12;
    if (questionEnding) requestLikelihood += 0.04;
    requestLikelihood = Math.max(0, Math.min(1, requestLikelihood));
    const requestLike = requestLikelihood >= 0.68;

    const answerPredicateIndex = requestLike && embeddedPredicateIndex >= 0 ? embeddedPredicateIndex : matrixPredicateIndex;
    const answerPredicateToken = answerPredicateIndex >= 0 ? entries[answerPredicateIndex]?.token ?? null : null;

    const subjectTokenIds = [];
    if (grammaticalSubjectIndex >= 0) {
      const subjectPhrase = phrases.find(phrase=>phrase.kind === "NP" && phrase.start <= grammaticalSubjectIndex && phrase.end > grammaticalSubjectIndex);
      for (const token of subjectPhrase?.tokenIds || [entries[grammaticalSubjectIndex]?.token]) {
        if (Number.isInteger(token)) subjectTokenIds.push(token);
      }
    }

    const complementTokenIds = [];
    if (answerPredicateIndex >= 0) {
      for (const item of entries) {
        if (item.index <= answerPredicateIndex) continue;
        if (item.type === "Punct") break;
        if (["Noun","ProperNoun","Pronoun","Num","Adj","Adv"].includes(item.type)) complementTokenIds.push(item.token);
      }
    }

    return {
      version:1,
      entries,
      phrases,
      clauses,
      typeSequence: entries.map(entry=>entry.type),
      phraseTypeSequence: phrases.map(phrase=>`${phrase.kind}(${phrase.types.join("+")})`),
      questionEnding,
      firstIndex,
      helperIndex,
      helperFronted,
      frontedOperatorIndex,
      embeddedOperatorIndex,
      grammaticalSubjectIndex,
      subjectPronounPerson: subjectPronounProfile?.person ?? null,
      subjectPronounNumber: subjectPronounProfile?.number ?? null,
      matrixPredicateIndex,
      embeddedPredicateIndex,
      hasEmbeddedClause,
      answerPredicateIndex,
      answerPredicateToken,
      subjectTokenIds,
      complementTokenIds,
      openQuestionLike,
      closedQuestionLike,
      imperativeLike,
      requestLikelihood,
      requestLike
    };
  }
}

globalThis.VilotTypePhraseParser = VilotTypePhraseParser;
