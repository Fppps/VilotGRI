/*
 * VilotNI 1.5 - Memory Retrieval v1
 *
 * Synchronous after VilotMemoryStore.ready resolves.
 *
 * Relevance != truth support != authority.
 * Working/episodic memories can be very relevant while providing no factual
 * truth evidence whatsoever.
 */

class VilotMemoryRetrieval {
  constructor(store, options = {}) {
    this.store = store;

    this.topWorking = options.topWorking || 4;
    this.topEpisodes = options.topEpisodes || 4;
    this.topSemantic = options.topSemantic || 6;
    this.topProcedural = options.topProcedural || 3;

    this.maxCandidateTokens = options.maxCandidateTokens || 72;
  }

  clamp01(value) {
    return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  }

  uniqueTokens(values) {
    const out = [];
    const seen = new Set();

    for (const token of values || []) {
      if (!Number.isInteger(token) || seen.has(token)) continue;
      seen.add(token);
      out.push(token);
    }

    return out;
  }

  overlapScore(a, b) {
    const aa = new Set(this.uniqueTokens(a));
    const bb = new Set(this.uniqueTokens(b));

    if (!aa.size || !bb.size) return 0.0;

    let hits = 0;

    for (const token of aa) {
      if (bb.has(token)) hits++;
    }

    return this.clamp01(
      hits / Math.max(1, Math.min(aa.size, bb.size))
    );
  }

  recencyScore(timestamp, scaleMs) {
    if (!Number.isFinite(timestamp)) return 0.40;

    const age = Math.max(0, Date.now() - timestamp);

    return Math.exp(
      -age / Math.max(1, scaleMs)
    );
  }

  promptAnchors(contextModel, promptTokens) {
    return this.uniqueTokens([
      ...(contextModel?.eventFrame?.rootTokens || []),
      ...(contextModel?.instructionContract?.sourceAnchorTokens || []),
      ...(contextModel?.promptContentTokens || []),
      ...(promptTokens || [])
    ]).slice(0, 24);
  }

  entryTokens(entry) {
    return this.uniqueTokens([
      ...(entry?.promptTokens || []),
      ...(entry?.responseTokens || []),
      ...(entry?.userTokens || []),
      ...(entry?.assistantTokens || []),
      ...(entry?.subjectTokens || []),
      ...(Number.isInteger(entry?.predicateToken)
        ? [entry.predicateToken]
        : []),
      ...(entry?.objectTokens || []),
      ...(entry?.modifierTokens || [])
    ]);
  }

  frameMatch(entry, contextModel) {
    const current =
      contextModel?.semanticFrame?.primaryFrame || "relation";

    if (!entry?.frame) return 0.45;

    return entry.frame === current ? 1.0 : 0.30;
  }

  queryMatch(entry, contextModel) {
    const current =
      contextModel?.semanticFrame?.queryTarget ||
      contextModel?.queryFrame?.requestedSlot ||
      "content";

    if (!entry?.queryTarget) return 0.45;

    return entry.queryTarget === current ? 1.0 : 0.28;
  }

  retrieve(contextModel, promptTokens, options = {}) {
    const snapshot = this.store.snapshot();
    const anchors = this.promptAnchors(contextModel, promptTokens);

    const working = snapshot.working
      .map(item => ({
        ...item,
        retrievalScore: this.clamp01(
          this.overlapScore(anchors, this.entryTokens(item)) * 0.50 +
          this.frameMatch(item, contextModel) * 0.18 +
          this.queryMatch(item, contextModel) * 0.14 +
          this.recencyScore(item.timestamp, 1000 * 60 * 30) * 0.18
        ),
        truthSupport: 0.0,
        authority: 0.0,
        memoryType: "working"
      }))
      .sort((a, b) => b.retrievalScore - a.retrievalScore)
      .slice(0, options.topWorking || this.topWorking);

    const episodic = snapshot.episodes
      .map(item => ({
        ...item,
        retrievalScore: this.clamp01(
          this.overlapScore(anchors, this.entryTokens(item)) * 0.38 +
          this.frameMatch(item, contextModel) * 0.20 +
          this.queryMatch(item, contextModel) * 0.18 +
          (item.usefulness || 0) * 0.14 +
          this.recencyScore(
            item.updatedAt || item.timestamp,
            1000 * 60 * 60 * 24 * 5
          ) * 0.10
        ),
        truthSupport: 0.0,
        authority: 0.0,
        memoryType: "episodic"
      }))
      .filter(item => item.retrievalScore > 0.16)
      .sort((a, b) => b.retrievalScore - a.retrievalScore)
      .slice(0, options.topEpisodes || this.topEpisodes);

    const semantic = snapshot.semantic
      .map(item => {
        const overlap =
          this.overlapScore(anchors, this.entryTokens(item));

        const relationMatch =
          Number.isInteger(item.predicateToken) &&
          item.predicateToken === contextModel?.eventFrame?.predicateToken
            ? 1.0
            : 0.35;

        const relevance = this.clamp01(
          overlap * 0.48 +
          relationMatch * 0.18 +
          this.frameMatch(item, contextModel) * 0.14 +
          (item.relevance || 0) * 0.12 +
          this.recencyScore(
            item.updatedAt,
            1000 * 60 * 60 * 24 * 30
          ) * 0.08
        );

        const truthSupport = this.clamp01(
          item.truthSupport ||
          ((item.trust || 0) * (item.authority || 0))
        );

        return {
          ...item,
          retrievalScore: relevance,
          truthSupport,
          authority: this.clamp01(item.authority),
          memoryType: "semantic"
        };
      })
      .filter(item => item.retrievalScore > 0.18)
      .sort((a, b) => b.retrievalScore - a.retrievalScore)
      .slice(0, options.topSemantic || this.topSemantic);

    const procedural = snapshot.procedural
      .map(item => ({
        ...item,
        retrievalScore: this.clamp01(
          this.frameMatch(item, contextModel) * 0.34 +
          this.queryMatch(item, contextModel) * 0.34 +
          (item.successRate || 0) * 0.20 +
          this.clamp01((item.rewardEMA || 0) * 0.5 + 0.5) * 0.08 +
          this.recencyScore(
            item.updatedAt,
            1000 * 60 * 60 * 24 * 20
          ) * 0.04
        ),
        truthSupport: 0.0,
        authority: 0.0,
        memoryType: "procedural"
      }))
      .filter(item => item.retrievalScore > 0.30)
      .sort((a, b) => b.retrievalScore - a.retrievalScore)
      .slice(0, options.topProcedural || this.topProcedural);

    const relevanceByToken = new Map();
    const truthByToken = new Map();
    const authorityByToken = new Map();

    const addSignal = (token, relevance, truth, authority) => {
      if (!Number.isInteger(token)) return;

      relevanceByToken.set(
        token,
        Math.max(
          relevanceByToken.get(token) || 0,
          this.clamp01(relevance)
        )
      );

      truthByToken.set(
        token,
        Math.max(
          truthByToken.get(token) || 0,
          this.clamp01(truth)
        )
      );

      authorityByToken.set(
        token,
        Math.max(
          authorityByToken.get(token) || 0,
          this.clamp01(authority)
        )
      );
    };

    for (const item of working) {
      for (const token of this.entryTokens(item)) {
        addSignal(token, item.retrievalScore * 0.64, 0.0, 0.0);
      }
    }

    for (const item of episodic) {
      for (const token of this.entryTokens(item)) {
        addSignal(token, item.retrievalScore * 0.60, 0.0, 0.0);
      }
    }

    for (const item of semantic) {
      for (const token of this.entryTokens(item)) {
        addSignal(
          token,
          item.retrievalScore,
          item.truthSupport,
          item.authority
        );
      }
    }

    const candidateTokenIds = Array.from(
      relevanceByToken.entries()
    )
      .sort((a, b) => b[1] - a[1])
      .slice(0, this.maxCandidateTokens)
      .map(item => item[0]);

    const topProcedure = procedural[0] || null;

    return {
      anchors,
      working,
      episodic,
      semantic,
      procedural,

      candidateTokenIds,
      relevanceByToken,
      truthByToken,
      authorityByToken,

      recoveryHint: topProcedure
        ? {
            strategy: topProcedure.strategy,
            score: topProcedure.retrievalScore,
            successRate: topProcedure.successRate || 0,
            reward: topProcedure.rewardEMA || 0
          }
        : null,

      overallRelevance: this.clamp01(
        Math.max(
          working[0]?.retrievalScore || 0,
          episodic[0]?.retrievalScore || 0,
          semantic[0]?.retrievalScore || 0
        )
      )
    };
  }

  getCandidateSignals(token, bundle) {
    if (!bundle || !Number.isInteger(token)) {
      return {
        relevance: 0.0,
        truthSupport: 0.0,
        authority: 0.0
      };
    }

    return {
      relevance: bundle.relevanceByToken?.get(token) || 0.0,
      truthSupport: bundle.truthByToken?.get(token) || 0.0,
      authority: bundle.authorityByToken?.get(token) || 0.0
    };
  }
}

globalThis.VilotMemoryRetrieval = VilotMemoryRetrieval;
