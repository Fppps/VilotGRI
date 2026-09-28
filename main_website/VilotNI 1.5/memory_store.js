/*
 * VilotNI 1.5 - Memory Store v1
 *
 * Working memory: RAM-only recent turns.
 * Episodic memory: persisted conversation outcomes and mistakes.
 * Semantic memory: persisted propositions with source/trust/authority.
 * Procedural memory: persisted reasoning/recovery strategies.
 *
 * Relevance, truth support, and authority are independent signals.
 */

class VilotMemoryStore {
  constructor(options = {}) {
    this.dbName = options.dbName || "vilotni15_memory_v1";
    this.dbVersion = 1;

    this.maxWorking = options.maxWorking || 8;
    this.maxEpisodes = options.maxEpisodes || 512;
    this.maxSemantic = options.maxSemantic || 1024;
    this.maxProcedural = options.maxProcedural || 256;

    this.working = [];
    this.episodes = [];
    this.semantic = [];
    this.procedural = [];

    this.semanticByKey = new Map();
    this.proceduralByKey = new Map();

    this.db = null;
    this.persistent = typeof indexedDB !== "undefined";
    this.ready = this.init();
  }

  clamp01(value) {
    return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  }

  now() {
    return Date.now();
  }

  makeId(prefix) {
    return `${prefix}_${this.now()}_${Math.random().toString(36).slice(2, 10)}`;
  }

  normalizeTokens(tokens, limit = 64) {
    const out = [];
    const seen = new Set();

    for (const raw of tokens || []) {
      const token =
        typeof raw === "object" && raw !== null
          ? raw.id
          : raw;

      if (!Number.isInteger(token) || seen.has(token)) continue;

      seen.add(token);
      out.push(token);

      if (out.length >= limit) break;
    }

    return out;
  }

  sourceAuthority(source) {
    switch (String(source || "").toLowerCase()) {
      case "external_verified": return 0.98;
      case "human_confirmed": return 0.92;
      case "imported_trusted": return 0.88;
      case "trusted_training": return 0.86;
      case "user_stated": return 0.30;
      case "inferred": return 0.24;
      case "assistant_generated": return 0.16;
      default: return 0.18;
    }
  }

  semanticKey(entry) {
    const subject = this.normalizeTokens(entry?.subjectTokens, 6).join(",");
    const predicate = Number.isInteger(entry?.predicateToken)
      ? entry.predicateToken
      : "";
    const object = this.normalizeTokens(entry?.objectTokens, 10).join(",");
    return `${subject}|${predicate}|${object}`;
  }

  procedureKey(entry) {
    return [
      entry?.frame || "relation",
      entry?.queryTarget || "content",
      entry?.strategy || "direct"
    ].join("|");
  }

  async init() {
    if (!this.persistent) return this;

    try {
      this.db = await this.openDB();

      await Promise.all([
        this.loadStore("episodes", this.maxEpisodes),
        this.loadStore("semantic", this.maxSemantic),
        this.loadStore("procedural", this.maxProcedural)
      ]);

      this.reindex();
    } catch (_) {
      this.db = null;
      this.persistent = false;
    }

    return this;
  }

  openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onupgradeneeded = event => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains("episodes")) {
          const store = db.createObjectStore("episodes", { keyPath: "id" });
          store.createIndex("timestamp", "timestamp");
        }

        if (!db.objectStoreNames.contains("semantic")) {
          const store = db.createObjectStore("semantic", { keyPath: "id" });
          store.createIndex("key", "key", { unique: true });
          store.createIndex("updatedAt", "updatedAt");
        }

        if (!db.objectStoreNames.contains("procedural")) {
          const store = db.createObjectStore("procedural", { keyPath: "id" });
          store.createIndex("key", "key", { unique: true });
          store.createIndex("updatedAt", "updatedAt");
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () =>
        reject(request.error || new Error("Memory DB open failed."));
    });
  }

  loadStore(storeName, limit) {
    if (!this.db) return Promise.resolve();

    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(storeName, "readonly");
      const request = tx.objectStore(storeName).getAll();

      request.onsuccess = () => {
        const values = request.result || [];

        values.sort(
          (a, b) =>
            (b.updatedAt || b.timestamp || 0) -
            (a.updatedAt || a.timestamp || 0)
        );

        const bounded = values.slice(0, limit);

        if (storeName === "episodes") this.episodes = bounded;
        if (storeName === "semantic") this.semantic = bounded;
        if (storeName === "procedural") this.procedural = bounded;

        resolve();
      };

      request.onerror = () => reject(request.error);
    });
  }

  reindex() {
    this.semanticByKey.clear();
    for (const item of this.semantic) {
      if (item?.key) this.semanticByKey.set(item.key, item);
    }

    this.proceduralByKey.clear();
    for (const item of this.procedural) {
      if (item?.key) this.proceduralByKey.set(item.key, item);
    }
  }

  persistPut(storeName, value) {
    if (!this.persistent || !this.db) return Promise.resolve();

    return new Promise((resolve, reject) => {
      try {
        const tx = this.db.transaction(storeName, "readwrite");
        tx.objectStore(storeName).put(value);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      } catch (error) {
        reject(error);
      }
    });
  }

  persistDelete(storeName, id) {
    if (!this.persistent || !this.db) return Promise.resolve();

    return new Promise((resolve, reject) => {
      try {
        const tx = this.db.transaction(storeName, "readwrite");
        tx.objectStore(storeName).delete(id);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      } catch (error) {
        reject(error);
      }
    });
  }

  trimCollection(name, limit, storeName) {
    const list = this[name];

    if (list.length <= limit) return;

    list.sort(
      (a, b) =>
        (b.updatedAt || b.timestamp || 0) -
        (a.updatedAt || a.timestamp || 0)
    );

    const removed = list.splice(limit);

    for (const item of removed) {
      this.persistDelete(storeName, item.id).catch(() => {});
    }
  }

  addWorkingTurn(entry) {
    const item = {
      id: entry?.id || this.makeId("work"),
      timestamp: entry?.timestamp || this.now(),
      userTokens: this.normalizeTokens(entry?.userTokens, 64),
      assistantTokens: this.normalizeTokens(entry?.assistantTokens, 96),
      frame: entry?.frame || "conversation",
      queryTarget: entry?.queryTarget || "content",
      answerAct: entry?.answerAct || "respond",
      resolved: Boolean(entry?.resolved)
    };

    this.working.push(item);

    if (this.working.length > this.maxWorking) {
      this.working.splice(0, this.working.length - this.maxWorking);
    }

    return item;
  }

  addEpisode(entry) {
    const now = entry?.timestamp || this.now();

    const item = {
      id: entry?.id || this.makeId("ep"),
      timestamp: now,
      updatedAt: now,

      promptTokens: this.normalizeTokens(entry?.promptTokens, 48),
      responseTokens: this.normalizeTokens(entry?.responseTokens, 72),

      frame: entry?.frame || "relation",
      queryTarget: entry?.queryTarget || "content",
      answerAct: entry?.answerAct || "respond",
      strategy: entry?.strategy || "direct",

      confidence: this.clamp01(entry?.confidence),
      verifier: this.clamp01(entry?.verifier),
      actFulfillment: this.clamp01(entry?.actFulfillment),
      propositionCompleteness: this.clamp01(entry?.propositionCompleteness),
      relationMatch: this.clamp01(entry?.relationMatch),
      grounding: this.clamp01(entry?.grounding),
      language: this.clamp01(entry?.language),

      resolved: Boolean(entry?.resolved),
      mistakeTags: Array.from(new Set(entry?.mistakeTags || [])).slice(0, 12),
      usefulness: this.clamp01(entry?.usefulness),

      origin: "conversation"
    };

    this.episodes.unshift(item);
    this.trimCollection("episodes", this.maxEpisodes, "episodes");
    this.persistPut("episodes", item).catch(() => {});

    return item;
  }

  upsertSemantic(entry) {
    const key = entry?.key || this.semanticKey(entry);
    if (!key || key === "||") return null;

    const now = this.now();
    const source = entry?.source || "inferred";
    const authority = this.sourceAuthority(source);
    const existing = this.semanticByKey.get(key);

    if (existing) {
      existing.updatedAt = now;
      existing.lastSource = source;
      existing.sources = Array.from(
        new Set([...(existing.sources || []), source])
      ).slice(0, 8);

      existing.authority = Math.max(existing.authority || 0, authority);
      existing.relevance = Math.max(
        existing.relevance || 0,
        this.clamp01(entry?.relevance ?? 0.70)
      );

      // Low-authority repetition does not self-promote the proposition.
      const incomingTrust = Math.min(
        authority,
        this.clamp01(entry?.trust ?? authority * 0.80)
      );

      existing.trust = Math.max(existing.trust || 0, incomingTrust);

      existing.confirmations =
        (existing.confirmations || 0) + (entry?.confirmed ? 1 : 0);

      existing.contradictions =
        (existing.contradictions || 0) + (entry?.contradicted ? 1 : 0);

      existing.truthSupport = this.clamp01(
        existing.trust *
        existing.authority *
        Math.max(0.25, 1.0 - existing.contradictions * 0.12)
      );

      this.persistPut("semantic", existing).catch(() => {});
      return existing;
    }

    const trust = Math.min(
      authority,
      this.clamp01(entry?.trust ?? authority * 0.80)
    );

    const item = {
      id: entry?.id || this.makeId("sem"),
      key,
      createdAt: now,
      updatedAt: now,

      subjectTokens: this.normalizeTokens(entry?.subjectTokens, 6),
      predicateToken: Number.isInteger(entry?.predicateToken)
        ? entry.predicateToken
        : null,
      objectTokens: this.normalizeTokens(entry?.objectTokens, 10),
      modifierTokens: this.normalizeTokens(entry?.modifierTokens, 8),

      frame: entry?.frame || "relation",

      source,
      lastSource: source,
      sources: [source],

      authority,
      trust,
      relevance: this.clamp01(entry?.relevance ?? 0.70),

      confirmations: entry?.confirmed ? 1 : 0,
      contradictions: entry?.contradicted ? 1 : 0,
      truthSupport: this.clamp01(trust * authority)
    };

    this.semantic.unshift(item);
    this.semanticByKey.set(key, item);

    this.trimCollection("semantic", this.maxSemantic, "semantic");
    this.persistPut("semantic", item).catch(() => {});

    return item;
  }

  reinforceProcedure(entry) {
    const key = entry?.key || this.procedureKey(entry);
    const now = this.now();
    const reward = Math.max(-1, Math.min(1, Number(entry?.reward) || 0));
    const success = Boolean(entry?.success);

    const existing = this.proceduralByKey.get(key);

    if (existing) {
      existing.updatedAt = now;
      existing.count = (existing.count || 0) + 1;
      existing.successes = (existing.successes || 0) + (success ? 1 : 0);
      existing.rewardEMA = (existing.rewardEMA ?? 0) * 0.85 + reward * 0.15;
      existing.successRate =
        existing.successes / Math.max(1, existing.count);

      this.persistPut("procedural", existing).catch(() => {});
      return existing;
    }

    const item = {
      id: entry?.id || this.makeId("proc"),
      key,
      createdAt: now,
      updatedAt: now,

      frame: entry?.frame || "relation",
      queryTarget: entry?.queryTarget || "content",
      strategy: entry?.strategy || "direct",

      count: 1,
      successes: success ? 1 : 0,
      successRate: success ? 1.0 : 0.0,
      rewardEMA: reward,

      notes: Array.isArray(entry?.notes)
        ? entry.notes.slice(0, 8)
        : []
    };

    this.procedural.unshift(item);
    this.proceduralByKey.set(key, item);

    this.trimCollection("procedural", this.maxProcedural, "procedural");
    this.persistPut("procedural", item).catch(() => {});

    return item;
  }

  snapshot() {
    return {
      working: this.working.slice(),
      episodes: this.episodes.slice(),
      semantic: this.semantic.slice(),
      procedural: this.procedural.slice()
    };
  }
}

globalThis.VilotMemoryStore = VilotMemoryStore;
