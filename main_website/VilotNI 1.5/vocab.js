/**
 * Dynamic Vocabulary Manager for VilotNI 1.5.
 * 100% data-driven from vocab.json. Zero hardcoded words or fallback dictionaries.
 */
class VocabularyManager {
  constructor(jsonPath = "vocab.json") {
    this.jsonPath = jsonPath;
    this.tokenToId = {};
    this.idToToken = {};
    this.vocabSize = 0;
    this.isLoaded = false;

    this.ready = this.loadFromJSON(this.jsonPath);
  }

  async loadFromJSON(path) {
    try {
      let response = await fetch(path);
      if (!response.ok && path === "vocab.json") {
        response = await fetch("Vocab.json");
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} - Could not load ${path}`);
      }

      const data = await response.json();
      const rawVocab = data.vocab || data;

      this.tokenToId = {};
      this.idToToken = {};
      let maxId = -1;

      // Ingest whatever words and IDs exist in the file
      for (const [token, id] of Object.entries(rawVocab)) {
        const numId = Number(id);
        this.tokenToId[token] = numId;
        this.idToToken[numId] = token;
        if (numId > maxId) maxId = numId;
      }

      // Exact size determined by the highest ID present in your JSON
      this.vocabSize = maxId >= 0 ? maxId + 1 : Object.keys(this.tokenToId).length;
      this.isLoaded = true;
      return this;
    } catch (err) {
      console.error(`[VocabularyManager] Error loading ${path}:`, err);
      return this;
    }
  }

  hasToken(id) {
    return this.idToToken[id] !== undefined;
  }

  tokenize(text) {
    // Splits words, numbers, and arbitrary punctuation symbols
    const rawTokens = text.match(/[\w']+|[^\s\w]/g) || [];
    return rawTokens.map(tok => {
      if (this.tokenToId[tok] !== undefined) {
        return this.tokenToId[tok];
      }
      // Dynamically registers any new word typed during the chat
      const newId = this.vocabSize++;
      this.tokenToId[tok] = newId;
      this.idToToken[newId] = tok;
      return newId;
    });
  }

  detokenize(tokenIds) {
    return tokenIds
      .map(id => this.idToToken[id])
      .filter(token => token !== undefined)
      .join(" ")
      .replace(/\s+([.,?!~)-])/g, "$1")
      .replace(/\(\s+/g, "(");
  }

  exportJSON() {
    return JSON.stringify({ vocab: this.tokenToId }, null, 4);
  }
}

if (typeof module !== "undefined") {
  module.exports = VocabularyManager;
}