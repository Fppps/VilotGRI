/**
 * VilotNI 1.5 Vocabulary & Syntactic Metadata Manager
 * - Resolves relative paths dynamically to avoid HTML 404 fallback crashes.
 * - Extracts token string, numeric ID, and Part-of-Speech category from vocab.json.
 * - Enforces primitive numeric indexing to avoid [object Object] collisions.
 */

class VocabularyManager {
  constructor(customPath = null) {
    this.customPath = customPath;
    this.tokenToId = new Map();
    this.caseFoldToId = new Map();
    this.idToToken = [];
    this.idToType = [];
    this.phraseEntries = [];
    this.vocabSize = 0;
    this.ready = this.load();
  }

  async load() {
    const candidatePaths = this.customPath
      ? [this.customPath]
      : [
          'vocab.json',
          './vocab.json',
          './VilotNI 1.5/vocab.json',
          '../vocab.json',
          '/vocab.json'
        ];

    let response = null;
    let resolvedUrl = '';

    for (const path of candidatePaths) {
      try {
        const res = await fetch(path);
        const contentType = res.headers.get('content-type') || '';
        if (res.ok && !contentType.includes('text/html')) {
          response = res;
          resolvedUrl = path;
          break;
        }
      } catch (_) {
        // Continue to the next fallback path
      }
    }

    if (!response) {
      throw new Error(
        `[VocabularyManager] Could not locate 'vocab.json'. Verified paths: ${candidatePaths.join(', ')}.`
      );
    }

    try {
      const data = await response.json();
      const vocabMap = data.vocab || data;

      const seenIds = new Map();
      for (const [word, meta] of Object.entries(vocabMap)) {
        const id = typeof meta === 'object' && meta !== null ? meta.id : meta;
        const type = typeof meta === 'object' && meta !== null ? meta.type : "Other";

        if (typeof id === 'number' && Number.isFinite(id) && id >= 0) {
          this.tokenToId.set(word, id);
          const folded = word.toLocaleLowerCase();
          if (!this.caseFoldToId.has(folded)) this.caseFoldToId.set(folded, id);

          if (seenIds.has(id) && seenIds.get(id) !== word) {
            console.warn(`[VocabularyManager] Duplicate token id ${id}: '${seenIds.get(id)}' and '${word}'. Keeping the first canonical output token.`);
          } else {
            seenIds.set(id, word);
            this.idToToken[id] = word;
            this.idToType[id] = type;
          }
        }
      }

      this.phraseEntries = Array.from(this.tokenToId.keys())
        .filter(word => /\s/.test(word))
        .sort((a, b) => b.length - a.length);

      this.vocabSize = this.idToToken.length;
    } catch (err) {
      console.error(`[VocabularyManager] Failed parsing JSON at ${resolvedUrl}:`, err);
      throw err;
    }
  }

  tokenize(text) {
    if (!text) return [];

    // Longest vocabulary phrases are matched first so entries such as
    // "VilotNI 1.5" survive as one token rather than being split apart.
    const escapedPhrases = this.phraseEntries.map(p =>
      p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    );
    const phrasePart = escapedPhrases.length ? `${escapedPhrases.join("|")}|` : "";
    const lexer = new RegExp(
      `${phrasePart}[A-Za-z]+(?:['’][A-Za-z]+)?|\\d+(?:\\.\\d+)?|[.,!?;:~–—()\-]`,
      "gi"
    );
    const rawWords = text.trim().match(lexer) || [];
    const tokens = [];

    for (const w of rawWords) {
      let id;
      if (this.tokenToId.has(w)) {
        id = this.tokenToId.get(w);
      } else if (this.tokenToId.has(w.toLowerCase())) {
        id = this.tokenToId.get(w.toLowerCase());
      } else {
        id = this.caseFoldToId.get(w.toLocaleLowerCase());
      }
      if (typeof id === "number") tokens.push(id);
    }
    return tokens;
  }

  detokenize(tokens) {
    if (!tokens || tokens.length === 0) return "";
    const words = [];
    for (const raw of tokens) {
      const id = typeof raw === 'object' && raw !== null ? raw.id : raw;
      const w = this.idToToken[id];
      if (w) words.push(w);
    }
    return words.join(" ").replace(/\s+([.,!?;:])/g, "$1");
  }

  hasToken(id) {
    return id >= 0 && id < this.vocabSize && Boolean(this.idToToken[id]);
  }
}

if (typeof window !== "undefined") {
  window.VocabularyManager = VocabularyManager;
}