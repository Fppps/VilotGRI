/**
 * VilotNI 1.5 Vocabulary & Syntactic Metadata Manager
 *
 * V7 INSTRUCTION-CONTRACT LEXICAL ENGINE
 * ------------------------
 * The physical vocabulary IDs remain unchanged so persisted neural weights stay
 * compatible. Orthographic aliases such as:
 *
 *   How / how
 *   Accelerate / accelerate
 *   Chips / chips
 *
 * are grouped into one canonical lexical identity for tokenization and
 * generation filtering.
 *
 * Important:
 * - IDs are NEVER renumbered.
 * - idToToken still preserves every physical token for compatibility/debugging.
 * - tokenToId resolves every spelling to its canonical physical ID.
 * - hasToken(NUMERIC_ID) returns true only for canonical active IDs by default,
 *   which automatically removes duplicate case aliases from RSL/Low candidate
 *   scans without changing the neural tensor dimensions.
 * - hasRawToken() is available when code needs to inspect every physical ID.
 */
class VocabularyManager {
  constructor(customPath = null) {
    this.customPath = customPath;

    // Canonical lookup maps used by normal VilotNI code.
    this.tokenToId = new Map();
    this.lowerTokenToId = new Map();

    // Raw physical-ID maps retained for compatibility and diagnostics.
    this.rawTokenToId = new Map();
    this.idToToken = [];
    this.idToType = [];

    // Canonical lexeme metadata.
    this.idToCanonicalId = [];
    this.idToCanonicalToken = [];
    this.isCanonicalId = [];
    this.canonicalIdToVariants = new Map();
    this.canonicalIds = [];

    // Sparse semantic knowledge graph loaded from vocab.json associations.
    // Map<canonicalId, Map<canonicalId, weight>>
    this.staticAssociations = new Map();
    this.staticAssociationMeta = new Map();
    this.typedRelations = new Map();
    this.collocationMap = new Map();
    this.grammarById = [];
    this.semanticClassById = [];
    this.lemmaById = [];
    this.formsById = new Map();
    this.schemaVersion = 1;

    this.apiVersion = 9;
    this.engineVersion = "9.0";
    this.capabilities = Object.freeze({
      canonicalLexemes: true,
      grammar: true,
      semanticClasses: true,
      typedRelations: true,
      collocations: true,
      morphology: true,
      scalarAssociations: true,
      knowledgeLinks: true,
      knowledgeNeighbors: true,
      semanticRoles: true,
      propositionFit: true,
      morphologyFamilies: true,
      promptAnswerSeparation: true,
      instructionContracts: true,
      temporalProfiles: true,
      temporalTenseProfiles: true
    });

    this.knowledgeAssociationCount = 0;

    // Physical tensor size must remain based on the largest physical ID.
    this.vocabSize = 0;

    // Number of lexemes actually exposed to generation/search.
    this.activeVocabSize = 0;
    this.aliasCount = 0;

    this.loaded = false;
    this.loadError = null;
    this.lastUnknownWords = [];
    this.ready = this.load();
  }

  canonicalKey(value) {
    return String(value ?? "")
      .normalize("NFKC")
      .replace(/[’‘]/g, "'")
      .replace(/[–—]/g, "-")
      .toLowerCase();
  }

  normalizeType(type) {
    return typeof type === "string" && type
      ? type
      : "Other";
  }

  async load() {
    const candidatePaths = this.customPath
      ? [this.customPath]
      : [
          "vocab.json",
          "./vocab.json",
          "./VilotNI 1.5/vocab.json",
          "../vocab.json",
          "/vocab.json"
        ];

    // Resolve vocab.json relative to vocab.js itself as well.
    if (
      !this.customPath &&
      typeof document !== "undefined" &&
      typeof window !== "undefined"
    ) {
      const script =
        document.currentScript ||
        Array.from(document.scripts || [])
          .find(s =>
            /(?:^|\/)vocab(?:\.min)?\.js(?:\?|$)/i.test(
              s.src || ""
            )
          );

      if (script?.src) {
        try {
          const scriptRelative =
            new URL("vocab.json", script.src).href;

          if (!candidatePaths.includes(scriptRelative)) {
            candidatePaths.unshift(scriptRelative);
          }
        } catch (_) {}
      }
    }

    let response = null;
    let resolvedUrl = "";

    for (const path of candidatePaths) {
      try {
        const res = await fetch(path);
        const contentType =
          res.headers.get("content-type") || "";

        if (
          res.ok &&
          !contentType.includes("text/html")
        ) {
          response = res;
          resolvedUrl = path;
          break;
        }
      } catch (_) {
        // Continue to the next fallback path.
      }
    }

    if (!response) {
      throw new Error(
        `[VocabularyManager] Could not locate 'vocab.json'. ` +
        `Verified paths: ${candidatePaths.join(", ")}.`
      );
    }

    try {
      const data = await response.json();
      this.schemaVersion = Number.isFinite(Number(data.schema_version))
        ? Number(data.schema_version)
        : 1;
      const vocabMap = data.vocab || data;

      // Reset in case load() is explicitly retried.
      this.tokenToId.clear();
      this.lowerTokenToId.clear();
      this.rawTokenToId.clear();
      this.idToToken = [];
      this.idToType = [];
      this.idToCanonicalId = [];
      this.idToCanonicalToken = [];
      this.isCanonicalId = [];
      this.canonicalIdToVariants.clear();
      this.canonicalIds = [];
      this.staticAssociations.clear();
      this.staticAssociationMeta.clear();
      this.typedRelations.clear();
      this.collocationMap.clear();
      this.grammarById = [];
      this.semanticClassById = [];
      this.lemmaById = [];
      this.formsById.clear();
      this.knowledgeAssociationCount = 0;

      const records = [];
      const idToRecord = new Map();
      const groups = new Map();

      // --------------------------------------------------------
      // PASS 1: preserve every physical ID exactly as stored.
      // --------------------------------------------------------
      for (const [word, meta] of Object.entries(vocabMap)) {
        const id =
          typeof meta === "object" && meta !== null
            ? meta.id
            : meta;

        const type =
          typeof meta === "object" && meta !== null
            ? this.normalizeType(meta.type)
            : "Other";

        if (!Number.isInteger(id) || id < 0) {
          continue;
        }

        const record = {
          word,
          id,
          type,
          meta:
            typeof meta === "object" && meta !== null
              ? meta
              : null
        };

        records.push(record);
        idToRecord.set(id, record);
        this.rawTokenToId.set(word, id);

        this.idToToken[id] = word;
        this.idToType[id] = type;

        const declaredCanonical =
          record.meta?.canonical;

        const key =
          this.canonicalKey(
            typeof declaredCanonical === "string"
              ? declaredCanonical
              : word
          );

        let group = groups.get(key);
        if (!group) {
          group = [];
          groups.set(key, group);
        }
        group.push(record);
      }

      // --------------------------------------------------------
      // PASS 2: resolve one stable active ID per lexical group.
      // Prefer valid canonical_id metadata from the repaired JSON.
      // Fall back to lowercase spelling, then the lowest ID.
      // --------------------------------------------------------
      for (const [key, group] of groups.entries()) {
        let canonicalRecord = null;

        const declaredIds = new Set();

        for (const record of group) {
          const declared =
            record.meta?.canonical_id ??
            record.meta?.canonicalId;

          if (Number.isInteger(declared)) {
            declaredIds.add(declared);
          }
        }

        if (declaredIds.size === 1) {
          const declaredId =
            declaredIds.values().next().value;

          const candidate =
            idToRecord.get(declaredId);

          if (
            candidate &&
            group.some(r => r.id === candidate.id)
          ) {
            canonicalRecord = candidate;
          }
        }

        if (!canonicalRecord) {
          canonicalRecord =
            group.find(
              r =>
                r.word.normalize("NFKC")
                  .replace(/[’‘]/g, "'")
                  .replace(/[–—]/g, "-") === key
            ) ||
            group.reduce(
              (best, r) =>
                !best || r.id < best.id
                  ? r
                  : best,
              null
            );
        }

        const canonicalId =
          canonicalRecord.id;

        const canonicalToken =
          canonicalRecord.word;

        // If metadata somehow disagrees on POS, use the canonical record for
        // every alias so RSL never sees capitalization as a POS change.
        const canonicalType =
          canonicalRecord.type;

        const variants = [];

        for (const record of group) {
          const id = record.id;

          this.idToCanonicalId[id] =
            canonicalId;

          this.idToCanonicalToken[id] =
            canonicalToken;

          this.idToType[id] =
            canonicalType;

          this.isCanonicalId[id] =
            id === canonicalId;

          variants.push(id);

          // Every exact surface spelling resolves to ONE lexical ID.
          this.tokenToId.set(
            record.word,
            canonicalId
          );
        }

        this.lowerTokenToId.set(
          key,
          canonicalId
        );

        this.canonicalIdToVariants.set(
          canonicalId,
          Int32Array.from(
            variants.sort((a, b) => a - b)
          )
        );

        this.canonicalIds.push(
          canonicalId
        );
      }

      this.canonicalIds.sort(
        (a, b) => a - b
      );

      // --------------------------------------------------------
      // PASS 3: resolve weighted semantic associations.
      //
      // JSON format:
      // "associations": [
      //   { "word": "entanglement", "weight": 1.0 },
      //   { "word": "physics", "weight": 0.9 }
      // ]
      //
      // Associations are made symmetric at runtime. This lets a sparse JSON
      // entry teach both "quantum -> entanglement" and the reverse connection.
      // --------------------------------------------------------
      const sourceTrust = source => {
        switch (source) {
          case "curated":
            return 1.00;
          case "lexical":
            return 0.90;
          case "vocab-neighbor":
            return 0.62;
          default:
            return 0.70;
        }
      };

      const setAssociation =
        (a, b, weight, source = "unknown") => {
          if (
            !Number.isInteger(a) ||
            !Number.isInteger(b) ||
            a === b
          ) {
            return;
          }

          const rawWeight = Math.max(
            0,
            Math.min(
              1,
              Number(weight)
            )
          );

          if (
            !Number.isFinite(rawWeight) ||
            rawWeight <= 0
          ) {
            return;
          }

          const trust =
            sourceTrust(source);

          // Effective weight is deliberately trust-adjusted. Automatically
          // inferred neighborhood links help RSL, but cannot pretend to be as
          // certain as curated or strong lexical relationships.
          const effectiveWeight =
            rawWeight * trust;

          let row =
            this.staticAssociations.get(a);

          if (!row) {
            row = new Map();
            this.staticAssociations.set(a, row);
          }

          const metaKey =
            `${a}|${b}`;

          const previous =
            row.get(b) || 0;

          if (effectiveWeight > previous) {
            row.set(
              b,
              effectiveWeight
            );

            this.staticAssociationMeta.set(
              metaKey,
              {
                rawWeight,
                effectiveWeight,
                trust,
                source
              }
            );
          }
        };

      for (const record of records) {
        const canonicalId =
          this.idToCanonicalId[record.id];

        if (
          !Number.isInteger(canonicalId) ||
          canonicalId !== record.id
        ) {
          continue;
        }

        const associations =
          record.meta?.associations;

        if (!Array.isArray(associations)) {
          continue;
        }

        for (const entry of associations) {
          const word =
            typeof entry === "string"
              ? entry
              : entry?.word;

          const weight =
            typeof entry === "string"
              ? 1.0
              : Number(entry?.weight ?? 1.0);

          const source =
            typeof entry === "string"
              ? "unknown"
              : String(
                  entry?.source || "unknown"
                );

          if (
            typeof word !== "string" ||
            !word
          ) {
            continue;
          }

          const targetId =
            this.tokenToId.get(word) ??
            this.lowerTokenToId.get(
              this.canonicalKey(word)
            );

          if (!Number.isInteger(targetId)) {
            continue;
          }

          setAssociation(
            canonicalId,
            targetId,
            weight,
            source
          );

          setAssociation(
            targetId,
            canonicalId,
            weight,
            source
          );
        }
      }

      // PASS 4: compile Vocab v2 lexical metadata into ID-indexed structures.
      for (const record of records) {
        const canonicalId = this.idToCanonicalId[record.id];
        if (!Number.isInteger(canonicalId) || canonicalId !== record.id) continue;
        const meta = record.meta || {};
        this.grammarById[canonicalId] = meta.grammar || null;
        this.semanticClassById[canonicalId] = meta.semantics?.class || "other";
        this.lemmaById[canonicalId] = meta.lemma || record.word;

        if (Array.isArray(meta.forms)) {
          const ids = meta.forms.map(word =>
            this.tokenToId.get(word) ?? this.lowerTokenToId.get(this.canonicalKey(word))
          ).filter(Number.isInteger);
          if (ids.length) this.formsById.set(canonicalId, Int32Array.from(Array.from(new Set(ids))));
        }

        const relations = meta.relations && typeof meta.relations === "object" ? meta.relations : {};
        for (const [relationName, entries] of Object.entries(relations)) {
          if (!Array.isArray(entries)) continue;
          let byRelation = this.typedRelations.get(canonicalId);
          if (!byRelation) { byRelation = new Map(); this.typedRelations.set(canonicalId, byRelation); }
          let row = byRelation.get(relationName);
          if (!row) { row = new Map(); byRelation.set(relationName, row); }
          for (const entry of entries) {
            const targetId = this.tokenToId.get(entry?.word) ?? this.lowerTokenToId.get(this.canonicalKey(entry?.word));
            if (!Number.isInteger(targetId)) continue;
            row.set(targetId, {
              weight: Math.max(0, Math.min(1, Number(entry?.weight ?? 1))),
              source: String(entry?.source || "unknown")
            });
          }
        }

        if (Array.isArray(meta.collocations)) {
          const row = new Map();
          for (const entry of meta.collocations) {
            const targetId = this.tokenToId.get(entry?.word) ?? this.lowerTokenToId.get(this.canonicalKey(entry?.word));
            if (!Number.isInteger(targetId)) continue;
            row.set(targetId, {
              weight: Math.max(0, Math.min(1, Number(entry?.weight ?? 1))),
              source: String(entry?.source || "unknown")
            });
          }
          if (row.size) this.collocationMap.set(canonicalId, row);
        }
      }

      let associationCount = 0;
      for (const row of this.staticAssociations.values()) {
        associationCount += row.size;
      }

      // Symmetric graph, so count unique undirected links.
      this.knowledgeAssociationCount =
        Math.floor(associationCount / 2);

      // IMPORTANT: keep physical tensor dimensions unchanged.
      this.vocabSize =
        this.idToToken.length;

      this.activeVocabSize =
        this.canonicalIds.length;

      this.aliasCount =
        records.length -
        this.activeVocabSize;

      if (this.vocabSize <= 0) {
        throw new Error(
          `[VocabularyManager] ${resolvedUrl} ` +
          `contained no valid token IDs.`
        );
      }

      this.loaded = true;
      this.loadError = null;

      console.info(
        `[VocabularyManager] schema v${this.schemaVersion}: loaded ${records.length} physical entries ` +
        `as ${this.activeVocabSize} canonical lexemes ` +
        `(${this.aliasCount} duplicate aliases inactive for generation), ` +
        `${this.knowledgeAssociationCount} static knowledge associations.`
      );

      return this;
    } catch (err) {
      this.loaded = false;
      this.loadError = err;

      console.error(
        `[VocabularyManager] Failed parsing/loading vocabulary at ` +
        `${resolvedUrl}:`,
        err
      );

      throw err;
    }
  }

  /**
   * Resolve a string OR physical numeric ID to its canonical physical ID.
   */
  canonicalId(value) {
    if (typeof value === "string") {
      const exact =
        this.tokenToId.get(value);

      if (Number.isInteger(exact)) {
        return exact;
      }

      const folded =
        this.lowerTokenToId.get(
          this.canonicalKey(value)
        );

      return Number.isInteger(folded)
        ? folded
        : null;
    }

    const rawId =
      typeof value === "object" &&
      value !== null
        ? Number(value.id)
        : Number(value);

    if (
      !Number.isInteger(rawId) ||
      rawId < 0 ||
      rawId >= this.vocabSize
    ) {
      return null;
    }

    const canonical =
      this.idToCanonicalId[rawId];

    return Number.isInteger(canonical)
      ? canonical
      : null;
  }

  canonicalToken(value) {
    const id =
      this.canonicalId(value);

    return Number.isInteger(id)
      ? this.idToToken[id] ?? null
      : null;
  }

  sameLexeme(a, b) {
    const aId = this.canonicalId(a);
    const bId = this.canonicalId(b);

    return (
      Number.isInteger(aId) &&
      Number.isInteger(bId) &&
      aId === bId
    );
  }

  getGrammar(value) {
    const id = this.canonicalId(value);
    return Number.isInteger(id) ? (this.grammarById[id] || null) : null;
  }

  getSemanticClass(value) {
    const id = this.canonicalId(value);
    return Number.isInteger(id) ? (this.semanticClassById[id] || "other") : "other";
  }

  getLemma(value) {
    const id = this.canonicalId(value);
    return Number.isInteger(id) ? (this.lemmaById[id] || this.idToToken[id] || "") : "";
  }

  getForms(value) {
    const id = this.canonicalId(value);
    if (!Number.isInteger(id)) return [];
    const forms = this.formsById.get(id);
    return forms ? Array.from(forms) : [];
  }

  getRelations(value, relation = null, limit = 16) {
    const id = this.canonicalId(value);
    if (!Number.isInteger(id)) return [];
    const byRelation = this.typedRelations.get(id);
    if (!byRelation) return [];
    const out = [];
    const collect = (name, row) => {
      for (const [targetId, info] of row.entries()) out.push({ relation:name, id:targetId, word:this.idToToken[targetId] || "", weight:info.weight, source:info.source });
    };
    if (relation) { const row = byRelation.get(relation); if (row) collect(relation,row); }
    else { for (const [name,row] of byRelation.entries()) collect(name,row); }
    return out.sort((a,b)=>b.weight-a.weight).slice(0,Math.max(1,Math.floor(limit)));
  }

  relationWeight(a, relation, b) {
    const aId=this.canonicalId(a), bId=this.canonicalId(b);
    if (!Number.isInteger(aId) || !Number.isInteger(bId)) return 0.0;
    return this.typedRelations.get(aId)?.get(relation)?.get(bId)?.weight || 0.0;
  }

  getCollocations(value, limit = 12) {
    const id=this.canonicalId(value);
    if (!Number.isInteger(id)) return [];
    const row=this.collocationMap.get(id);
    if (!row) return [];
    return Array.from(row.entries()).sort((a,b)=>b[1].weight-a[1].weight).slice(0,Math.max(1,Math.floor(limit))).map(([targetId,info])=>({id:targetId,word:this.idToToken[targetId] || "",weight:info.weight,source:info.source}));
  }


  getCapabilities() {
    return {
      apiVersion: this.apiVersion,
      engineVersion: this.engineVersion,
      schemaVersion: this.schemaVersion,
      ...this.capabilities
    };
  }

  getTypedRelationDetails(a, b) {
    const aId = this.canonicalId(a);
    const bId = this.canonicalId(b);

    if (
      !Number.isInteger(aId) ||
      !Number.isInteger(bId) ||
      aId === bId
    ) {
      return {
        weight: 0.0,
        forward: [],
        reverse: []
      };
    }

    const collect = (fromId, toId) => {
      const byRelation =
        this.typedRelations.get(fromId);

      if (!byRelation) return [];

      const out = [];

      for (
        const [relation, row]
        of byRelation.entries()
      ) {
        const info = row.get(toId);
        if (!info) continue;

        out.push({
          relation,
          weight:
            Math.max(
              0,
              Math.min(
                1,
                Number(info.weight) || 0
              )
            ),
          source:
            String(info.source || "unknown")
        });
      }

      return out.sort(
        (x, y) =>
          y.weight - x.weight
      );
    };

    const forward =
      collect(aId, bId);

    const reverse =
      collect(bId, aId);

    let weight = 0.0;

    for (
      const item of
        [...forward, ...reverse]
    ) {
      weight = Math.max(
        weight,
        item.weight
      );
    }

    return {
      weight,
      forward,
      reverse
    };
  }

  getCollocationWeight(a, b) {
    const aId = this.canonicalId(a);
    const bId = this.canonicalId(b);

    if (
      !Number.isInteger(aId) ||
      !Number.isInteger(bId) ||
      aId === bId
    ) {
      return 0.0;
    }

    const forward =
      this.collocationMap
        .get(aId)
        ?.get(bId)
        ?.weight || 0.0;

    const reverse =
      this.collocationMap
        .get(bId)
        ?.get(aId)
        ?.weight || 0.0;

    return Math.max(
      0,
      Math.min(
        1,
        Math.max(forward, reverse)
      )
    );
  }

  getSemanticCompatibility(a, b) {
    const aClass =
      this.getSemanticClass(a);

    const bClass =
      this.getSemanticClass(b);

    if (
      !aClass ||
      !bClass ||
      aClass === "other" ||
      bClass === "other"
    ) {
      return 0.55;
    }

    if (aClass === bClass) {
      return 0.90;
    }

    const generic =
      new Set([
        "entity_or_concept",
        "reference",
        "property",
        "modifier"
      ]);

    if (
      generic.has(aClass) ||
      generic.has(bClass)
    ) {
      return 0.64;
    }

    const key =
      [aClass, bClass]
        .sort()
        .join("|");

    const compatibility = {
      "ai_concept|representation": 0.90,
      "ai_concept|system": 0.88,
      "ai_concept|process": 0.84,
      "ai_concept|hardware": 0.82,
      "hardware|hardware_component": 0.94,
      "hardware|material": 0.86,
      "hardware|process": 0.82,
      "hardware|resource": 0.80,
      "hardware|system": 0.88,
      "computing_unit|process": 0.90,
      "computing_unit|physics_concept": 0.92,
      "field|physics_concept": 0.92,
      "physical_entity|physics_concept": 0.86,
      "information|process": 0.84,
      "procedure|process": 0.90,
      "action|process": 0.86,
      "action|quantity": 0.76,
      "process|quantity": 0.80,
      "resource|process": 0.80,
      "location|action": 0.64,
      "time|action": 0.68
    };

    return (
      compatibility[key] ??
      0.60
    );
  }


  getMorphologySignature(value) {
    const id = this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return null;
    }

    const lemma =
      String(
        this.getLemma(id) ||
        this.idToToken[id] ||
        ""
      ).toLowerCase();

    const forms =
      this.getForms(id);

    return {
      id,
      lemma,
      forms,
      familySize:
        1 + forms.length
    };
  }

  sameMorphologicalFamily(a, b) {
    const aId =
      this.canonicalId(a);

    const bId =
      this.canonicalId(b);

    if (
      !Number.isInteger(aId) ||
      !Number.isInteger(bId)
    ) {
      return false;
    }

    if (aId === bId) {
      return true;
    }

    const aLemma =
      String(
        this.getLemma(aId) || ""
      ).toLowerCase();

    const bLemma =
      String(
        this.getLemma(bId) || ""
      ).toLowerCase();

    if (
      aLemma &&
      bLemma &&
      aLemma === bLemma
    ) {
      return true;
    }

    const aForms =
      this.formsById.get(aId);

    if (
      aForms &&
      Array.from(aForms).includes(bId)
    ) {
      return true;
    }

    const bForms =
      this.formsById.get(bId);

    return Boolean(
      bForms &&
      Array.from(bForms).includes(aId)
    );
  }

  getMorphologyAffinity(a, b) {
    const aId =
      this.canonicalId(a);

    const bId =
      this.canonicalId(b);

    if (
      !Number.isInteger(aId) ||
      !Number.isInteger(bId)
    ) {
      return 0.0;
    }

    if (aId === bId) {
      return 1.0;
    }

    const aLemma =
      String(
        this.getLemma(aId) || ""
      ).toLowerCase();

    const bLemma =
      String(
        this.getLemma(bId) || ""
      ).toLowerCase();

    if (
      aLemma &&
      bLemma &&
      aLemma === bLemma
    ) {
      return 0.96;
    }

    if (
      this.sameMorphologicalFamily(
        aId,
        bId
      )
    ) {
      return 0.90;
    }

    return 0.0;
  }

  getMorphologyFamily(value, limit = 12) {
    const id =
      this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return [];
    }

    const out =
      new Set([id]);

    for (
      const formId of
        this.getForms(id)
    ) {
      if (
        Number.isInteger(formId) &&
        this.hasToken(formId)
      ) {
        out.add(formId);
      }
    }

    for (
      const item of
        this.getRelations(
          id,
          "lexical_family",
          limit
        )
    ) {
      if (
        Number.isInteger(item?.id) &&
        this.hasToken(item.id)
      ) {
        out.add(item.id);
      }
    }

    return Array.from(out)
      .slice(
        0,
        Math.max(
          1,
          Math.floor(limit)
        )
      );
  }


  getPromptReusePolicy(value) {
    const id = this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return {
        anchorStrength: 0.0,
        reuseCost: 1.0,
        surfaceReuse: false,
        properName: false,
        content: false
      };
    }

    const grammar = this.getGrammar(id) || {};
    const pos = grammar.pos || this.canonicalPOS(this.idToType[id]);
    const semanticClass = this.getSemanticClass(id);

    const properName = pos === "ProperNoun";
    const content = ["Noun", "ProperNoun", "Verb", "Adj", "Adv", "Num"].includes(pos);

    let anchorStrength = 0.35;
    let reuseCost = 0.85;

    switch (pos) {
      case "ProperNoun":
        anchorStrength = 1.0;
        reuseCost = 0.08;
        break;
      case "Noun":
        anchorStrength = 0.90;
        reuseCost = 0.52;
        break;
      case "Verb":
        anchorStrength = 0.92;
        reuseCost = 0.68;
        break;
      case "Num":
        anchorStrength = 0.82;
        reuseCost = 0.22;
        break;
      case "Adj":
        anchorStrength = 0.70;
        reuseCost = 0.74;
        break;
      case "Adv":
        anchorStrength = 0.54;
        reuseCost = 0.86;
        break;
      case "Pronoun":
        anchorStrength = 0.36;
        reuseCost = 0.58;
        break;
      default:
        anchorStrength = 0.22;
        reuseCost = 0.95;
        break;
    }

    if (["ai_concept", "physics_concept", "hardware", "system", "field"].includes(semanticClass)) {
      anchorStrength = Math.min(1.0, anchorStrength + 0.08);
    }

    return {
      anchorStrength,
      reuseCost,
      surfaceReuse: properName || pos === "Num",
      properName,
      content,
      pos,
      semanticClass
    };
  }

  getAnchorSupport(anchor, candidate) {
    const anchorId = this.canonicalId(anchor);
    const candidateId = this.canonicalId(candidate);

    if (!Number.isInteger(anchorId) || !Number.isInteger(candidateId)) {
      return {
        semanticSupport: 0.0,
        expansionSupport: 0.0,
        exact: false,
        morphological: false,
        known: false
      };
    }

    const exact = anchorId === candidateId;
    const morphological = !exact && this.sameMorphologicalFamily(anchorId, candidateId);

    if (exact) {
      return {
        semanticSupport: 1.0,
        expansionSupport: 0.18,
        exact: true,
        morphological: false,
        known: true
      };
    }

    if (morphological) {
      return {
        semanticSupport: 0.92,
        expansionSupport: 0.26,
        exact: false,
        morphological: true,
        known: true
      };
    }

    const link = this.getKnowledgeLink(anchorId, candidateId);
    const relation = Math.max(0, Math.min(1, link.typedRelation || 0));
    const collocation = Math.max(0, Math.min(1, link.collocation || 0));
    const classFit = Math.max(0, Math.min(1, link.classFit || 0.55));
    const knowledge = Math.max(0, Math.min(1, link.score || 0.55));

    const semanticSupport = Math.max(
      0,
      Math.min(
        1,
        knowledge * 0.54 +
        relation * 0.20 +
        collocation * 0.14 +
        classFit * 0.12
      )
    );

    const expansionSupport = Math.max(
      0,
      Math.min(
        1,
        Math.max(0, semanticSupport - 0.45) * 1.72
      )
    );

    return {
      semanticSupport,
      expansionSupport,
      exact: false,
      morphological: false,
      known: Boolean(link.known),
      relation,
      collocation,
      classFit
    };
  }


  getTemporalProfile(value) {
    const id = this.canonicalId(value);
    if (!Number.isInteger(id)) return {score:0.0,kind:"none",absolute:false,relative:false,duration:false,boundary:false,frequency:false};
    const word=String(this.idToToken[id]||"").toLowerCase();
    const grammar=this.getGrammar(id)||{};
    const pos=grammar.pos||this.canonicalPOS(this.idToType[id]);
    const semanticClass=this.getSemanticClass(id);
    const months=new Set(["january","february","march","april","may","june","july","august","september","october","november","december"]);
    const weekdays=new Set(["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]);
    const seasons=new Set(["spring","summer","autumn","fall","winter"]);
    const absoluteWords=new Set(["today","tomorrow","yesterday","tonight","noon","midnight","morning","afternoon","evening"]);
    const relativeWords=new Set(["now","soon","later","eventually","recently","currently","already","next","last","earlier","late","early","future","past","present"]);
    const units=new Set(["second","seconds","minute","minutes","hour","hours","day","days","week","weeks","month","months","year","years","decade","decades","century","centuries"]);
    const boundaryWords=new Set(["before","after","during","until","since","within","around","by"]);
    const frequencyWords=new Set(["daily","weekly","monthly","yearly","annually","often","sometimes","occasionally","rarely"]);
    let score=0.0,kind="none";
    const absolute=semanticClass==="time"||months.has(word)||weekdays.has(word)||seasons.has(word)||absoluteWords.has(word);
    const relative=relativeWords.has(word),duration=units.has(word),boundary=boundaryWords.has(word),frequency=frequencyWords.has(word);
    if(semanticClass==="time"){score=1.0;kind="time";}
    if(pos==="Num"||/^\\d{1,4}$/.test(word)){score=Math.max(score,0.92);kind="number";}
    if(months.has(word)||weekdays.has(word)||seasons.has(word)||absoluteWords.has(word)){score=Math.max(score,0.96);kind="absolute";}
    if(relative){score=Math.max(score,0.86);kind="relative";}
    if(duration){score=Math.max(score,0.90);kind="duration";}
    if(boundary){score=Math.max(score,0.78);kind="boundary";}
    if(frequency){score=Math.max(score,0.72);kind="frequency";}
    if(score===0&&pos==="Adv")score=0.22; else if(score===0&&(pos==="Noun"||pos==="ProperNoun"))score=0.14;
    return {score:Math.max(0,Math.min(1,score)),kind,absolute,relative,duration,boundary,frequency};
  }


  getTemporalTenseProfile(value) {
    const id =
      this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return {
        tense: "neutral",
        strength: 0.0,
        temporalScore: 0.0
      };
    }

    const word =
      String(
        this.idToToken[id] || ""
      ).toLowerCase();

    const temporal =
      this.getTemporalProfile(id);

    const futureWords =
      new Set([
        "tomorrow", "soon", "later",
        "eventually", "next", "future",
        "upcoming", "forthcoming",
        "afterward", "afterwards"
      ]);

    const pastWords =
      new Set([
        "yesterday", "earlier", "last",
        "past", "recently", "ago",
        "previously", "formerly"
      ]);

    const presentWords =
      new Set([
        "today", "now", "currently",
        "present", "presently",
        "tonight", "nowadays"
      ]);

    let tense = "neutral";
    let strength = 0.0;

    if (futureWords.has(word)) {
      tense = "future";
      strength = 1.0;
    } else if (pastWords.has(word)) {
      tense = "past";
      strength = 1.0;
    } else if (presentWords.has(word)) {
      tense = "present";
      strength = 0.92;
    }

    return {
      tense,
      strength,
      temporalScore:
        temporal.score || 0.0
    };
  }

  getTemporalTenseCompatibility(
    value,
    targetTense = "neutral"
  ) {
    const profile =
      this.getTemporalTenseProfile(
        value
      );

    const target =
      String(
        targetTense || "neutral"
      ).toLowerCase();

    if (
      target === "neutral" ||
      profile.tense === "neutral"
    ) {
      return {
        compatibility: 0.82,
        contradiction: false,
        ...profile
      };
    }

    if (profile.tense === target) {
      return {
        compatibility: 1.0,
        contradiction: false,
        ...profile
      };
    }

    if (
      target === "future" &&
      profile.tense === "present"
    ) {
      return {
        compatibility: 0.54,
        contradiction: false,
        ...profile
      };
    }

    if (
      target === "present" &&
      profile.tense === "future"
    ) {
      return {
        compatibility: 0.24,
        contradiction: true,
        ...profile
      };
    }

    if (
      target === "present" &&
      profile.tense === "past"
    ) {
      return {
        compatibility: 0.20,
        contradiction: true,
        ...profile
      };
    }

    return {
      compatibility: 0.0,
      contradiction: true,
      ...profile
    };
  }

  getRoleCompatibility(value, role = "content") {
    const id =
      this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return 0.0;
    }

    const grammar =
      this.getGrammar(id) || {};

    const pos =
      grammar.pos ||
      this.canonicalPOS(
        this.idToType[id]
      );

    const semanticClass =
      this.getSemanticClass(id);

    const roleName =
      String(role || "content")
        .toLowerCase();

    const classIn =
      (...items) =>
        items.includes(
          semanticClass
        );

    const posIn =
      (...items) =>
        items.includes(pos);

    switch (roleName) {
      case "subject":
        if (
          posIn(
            "Noun",
            "ProperNoun",
            "Pronoun"
          )
        ) {
          return 1.0;
        }
        return 0.10;

      case "predicate":
        if (pos === "Verb") return 1.0;
        if (pos === "Adj") return 0.76;
        if (
          pos === "Aux" ||
          pos === "Modal"
        ) {
          return 0.58;
        }
        return 0.08;

      case "mechanism":
        if (
          classIn(
            "process",
            "procedure",
            "hardware",
            "hardware_component",
            "system",
            "resource",
            "computing_unit",
            "action"
          )
        ) {
          return 0.94;
        }
        if (
          classIn(
            "information",
            "representation",
            "ai_concept",
            "physics_concept"
          )
        ) {
          return 0.62;
        }
        if (
          classIn(
            "quantity",
            "time",
            "location"
          )
        ) {
          return 0.12;
        }
        if (
          posIn(
            "Verb",
            "Noun",
            "ProperNoun"
          )
        ) {
          return 0.48;
        }
        if (pos === "Adv") {
          return 0.34;
        }
        return 0.16;

      case "cause":
        if (
          classIn(
            "action",
            "process",
            "procedure",
            "system",
            "event"
          )
        ) {
          return 0.90;
        }
        if (
          posIn(
            "Verb",
            "Noun",
            "ProperNoun"
          )
        ) {
          return 0.56;
        }
        return 0.22;

      case "time": {
        const temporal = this.getTemporalProfile(id);
        return temporal.score > 0 ? temporal.score : 0.08;
      }

      case "location":
        if (semanticClass === "location") {
          return 1.0;
        }
        if (
          posIn(
            "Noun",
            "ProperNoun"
          )
        ) {
          return 0.44;
        }
        return 0.10;

      case "quantity":
        if (
          semanticClass === "quantity" ||
          pos === "Num"
        ) {
          return 1.0;
        }
        if (pos === "Adj") {
          return 0.38;
        }
        return 0.10;

      case "definition":
        if (
          classIn(
            "property",
            "entity_or_concept",
            "representation",
            "system",
            "field",
            "physics_concept",
            "ai_concept",
            "hardware"
          )
        ) {
          return 0.86;
        }
        if (
          posIn(
            "Noun",
            "ProperNoun",
            "Adj"
          )
        ) {
          return 0.64;
        }
        return 0.18;

      case "evaluation":
        if (
          classIn(
            "property",
            "modifier",
            "quantity"
          )
        ) {
          return 0.82;
        }
        if (
          pos === "Adj" ||
          pos === "Adv"
        ) {
          return 0.74;
        }
        return 0.22;

      case "object":
      case "content":
      default:
        if (
          posIn(
            "Noun",
            "ProperNoun",
            "Pronoun",
            "Num"
          )
        ) {
          return 0.86;
        }
        if (pos === "Verb") {
          return 0.58;
        }
        if (pos === "Adj") {
          return 0.48;
        }
        if (pos === "Adv") {
          return 0.32;
        }
        return 0.18;
    }
  }

  getPredicateArgumentFit(
    predicate,
    candidate,
    role = "object"
  ) {
    const predicateId =
      this.canonicalId(predicate);

    const candidateId =
      this.canonicalId(candidate);

    if (
      !Number.isInteger(candidateId)
    ) {
      return {
        score: 0.0,
        roleFit: 0.0,
        knowledge: 0.50,
        typedRelation: 0.0,
        collocation: 0.0,
        classFit: 0.55
      };
    }

    const roleFit =
      this.getRoleCompatibility(
        candidateId,
        role
      );

    if (
      !Number.isInteger(predicateId)
    ) {
      return {
        score:
          Math.max(
            0,
            Math.min(
              1,
              roleFit * 0.76 +
              0.24 * 0.55
            )
          ),
        roleFit,
        knowledge: 0.55,
        typedRelation: 0.0,
        collocation: 0.0,
        classFit: 0.55
      };
    }

    const link =
      this.getKnowledgeLink(
        predicateId,
        candidateId
      );

    const typedRelation =
      Math.max(
        0,
        Math.min(
          1,
          link.typedRelation || 0
        )
      );

    const collocation =
      Math.max(
        0,
        Math.min(
          1,
          link.collocation || 0
        )
      );

    const classFit =
      Math.max(
        0,
        Math.min(
          1,
          link.classFit || 0.55
        )
      );

    const knowledge =
      Math.max(
        0,
        Math.min(
          1,
          link.score || 0.55
        )
      );

    const roleName =
      String(role || "object")
        .toLowerCase();

    const relationNames =
      new Set(
        [
          ...(link.relationDetails?.forward || []),
          ...(link.relationDetails?.reverse || [])
        ].map(
          item =>
            String(
              item?.relation || ""
            )
        )
      );

    let roleRelationBoost = 0.0;

    if (
      roleName === "mechanism" &&
      (
        relationNames.has("argument_related") ||
        relationNames.has("action_related") ||
        relationNames.has("related_to")
      )
    ) {
      roleRelationBoost = 0.12;
    } else if (
      roleName === "object" &&
      relationNames.has(
        "argument_related"
      )
    ) {
      roleRelationBoost = 0.14;
    } else if (
      roleName === "definition" &&
      (
        relationNames.has("related_to") ||
        relationNames.has("lexical_family")
      )
    ) {
      roleRelationBoost = 0.10;
    }

    const score =
      Math.max(
        0,
        Math.min(
          1,
          roleFit * 0.46 +
          knowledge * 0.20 +
          typedRelation * 0.17 +
          collocation * 0.10 +
          classFit * 0.07 +
          roleRelationBoost
        )
      );

    return {
      score,
      roleFit,
      knowledge,
      typedRelation,
      collocation,
      classFit,
      roleRelationBoost
    };
  }

  getPropositionFit(
    subject,
    predicate,
    candidate,
    role = "content"
  ) {
    const subjectId =
      this.canonicalId(subject);

    const candidateId =
      this.canonicalId(candidate);

    const predicateFit =
      this.getPredicateArgumentFit(
        predicate,
        candidateId,
        role
      );

    let subjectKnowledge = 0.55;

    if (
      Number.isInteger(subjectId) &&
      Number.isInteger(candidateId)
    ) {
      subjectKnowledge =
        this.getKnowledgeLink(
          subjectId,
          candidateId
        ).score || 0.55;
    }

    const score =
      Math.max(
        0,
        Math.min(
          1,
          predicateFit.score * 0.72 +
          subjectKnowledge * 0.18 +
          predicateFit.roleFit * 0.10
        )
      );

    return {
      score,
      subjectKnowledge,
      ...predicateFit
    };
  }

  getKnowledgeLink(a, b) {
    const aId = this.canonicalId(a);
    const bId = this.canonicalId(b);

    if (
      !Number.isInteger(aId) ||
      !Number.isInteger(bId)
    ) {
      return {
        known: false,
        score: 0.50,
        positiveSupport: 0.0,
        association: 0.0,
        typedRelation: 0.0,
        collocation: 0.0,
        lexicalFamily: 0.0,
        classFit: 0.55,
        relationDetails: {
          weight: 0.0,
          forward: [],
          reverse: []
        }
      };
    }

    if (aId === bId) {
      return {
        known: true,
        score: 1.0,
        positiveSupport: 1.0,
        association: 1.0,
        typedRelation: 1.0,
        collocation: 1.0,
        lexicalFamily: 1.0,
        classFit: 1.0,
        relationDetails: {
          weight: 1.0,
          forward: [],
          reverse: []
        }
      };
    }

    const association =
      this.getAssociationWeight(
        aId,
        bId
      );

    const relationDetails =
      this.getTypedRelationDetails(
        aId,
        bId
      );

    const typedRelation =
      relationDetails.weight;

    const collocation =
      this.getCollocationWeight(
        aId,
        bId
      );

    const lemmaA =
      this.getLemma(aId);

    const lemmaB =
      this.getLemma(bId);

    const lexicalFamily =
      lemmaA &&
      lemmaB &&
      lemmaA === lemmaB
        ? 0.96
        : 0.0;

    const classFit =
      this.getSemanticCompatibility(
        aId,
        bId
      );

    const positiveSupport =
      1.0 -
      (
        1.0 - collocation * 0.96
      ) *
      (
        1.0 - typedRelation * 0.90
      ) *
      (
        1.0 - lexicalFamily * 0.88
      ) *
      (
        1.0 - association * 0.70
      );

    const known =
      positiveSupport > 0.02;

    let score =
      known
        ? (
            0.52 +
            positiveSupport * 0.42 +
            Math.max(
              0,
              classFit - 0.55
            ) * 0.12
          )
        : (
            0.55 +
            Math.max(
              0,
              classFit - 0.55
            ) * 0.10
          );

    if (
      known &&
      classFit < 0.35 &&
      positiveSupport < 0.35
    ) {
      score -=
        (0.35 - classFit) * 0.20;
    }

    score =
      Math.max(
        0,
        Math.min(1, score)
      );

    return {
      known,
      score,
      positiveSupport:
        Math.max(
          0,
          Math.min(
            1,
            positiveSupport
          )
        ),
      association,
      typedRelation,
      collocation,
      lexicalFamily,
      classFit,
      relationDetails
    };
  }

  getKnowledgeNeighbors(value, limit = 24) {
    const id = this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return [];
    }

    const candidates =
      new Set();

    for (
      const item of
        this.getAssociations(id, 18)
    ) {
      candidates.add(item.id);
    }

    for (
      const item of
        this.getRelations(id, null, 18)
    ) {
      candidates.add(item.id);
    }

    for (
      const item of
        this.getCollocations(id, 12)
    ) {
      candidates.add(item.id);
    }

    for (
      const formId of
        this.getForms(id)
    ) {
      candidates.add(formId);
    }

    const ranked = [];

    for (const targetId of candidates) {
      if (
        !Number.isInteger(targetId) ||
        targetId === id ||
        !this.hasToken(targetId)
      ) {
        continue;
      }

      const link =
        this.getKnowledgeLink(
          id,
          targetId
        );

      ranked.push({
        id: targetId,
        word:
          this.idToToken[targetId] || "",
        score:
          link.score,
        positiveSupport:
          link.positiveSupport,
        typedRelation:
          link.typedRelation,
        collocation:
          link.collocation,
        association:
          link.association,
        classFit:
          link.classFit
      });
    }

    return ranked
      .sort(
        (a, b) =>
          b.score - a.score
      )
      .slice(
        0,
        Math.max(
          1,
          Math.floor(limit)
        )
      );
  }

  getLexicalSnapshot(value) {
    const id = this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return null;
    }

    return {
      id,
      token:
        this.idToToken[id] || "",
      type:
        this.idToType[id] || "Other",
      lemma:
        this.getLemma(id),
      grammar:
        this.getGrammar(id),
      semanticClass:
        this.getSemanticClass(id),
      forms:
        this.getForms(id),
      relations:
        this.getRelations(id, null, 12),
      collocations:
        this.getCollocations(id, 8),
      associations:
        this.getAssociations(id, 12)
    };
  }

  getAssociationWeight(a, b) {
    const aId =
      this.canonicalId(a);

    const bId =
      this.canonicalId(b);

    if (
      !Number.isInteger(aId) ||
      !Number.isInteger(bId) ||
      aId === bId
    ) {
      return 0.0;
    }

    return Math.max(
      0,
      Math.min(
        1,
        this.staticAssociations
          .get(aId)
          ?.get(bId) || 0.0
      )
    );
  }

  getAssociationDetails(a, b) {
    const aId =
      this.canonicalId(a);

    const bId =
      this.canonicalId(b);

    if (
      !Number.isInteger(aId) ||
      !Number.isInteger(bId) ||
      aId === bId
    ) {
      return {
        rawWeight: 0.0,
        effectiveWeight: 0.0,
        trust: 0.0,
        source: "none"
      };
    }

    const direct =
      this.staticAssociationMeta.get(
        `${aId}|${bId}`
      );

    const reverse =
      this.staticAssociationMeta.get(
        `${bId}|${aId}`
      );

    const best =
      !direct
        ? reverse
        : !reverse
          ? direct
          : (
              direct.effectiveWeight >=
              reverse.effectiveWeight
                ? direct
                : reverse
            );

    return best || {
      rawWeight: 0.0,
      effectiveWeight: 0.0,
      trust: 0.0,
      source: "none"
    };
  }

  getAssociations(value, limit = 16) {
    const id =
      this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return [];
    }

    const row =
      this.staticAssociations.get(id);

    if (!row) return [];

    return Array.from(row.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(
        0,
        Math.max(1, Math.floor(limit))
      )
      .map(([targetId, weight]) => {
        const details =
          this.getAssociationDetails(
            id,
            targetId
          );

        return {
          id: targetId,
          word:
            this.idToToken[targetId] || "",
          weight,
          rawWeight:
            details.rawWeight,
          trust:
            details.trust,
          source:
            details.source
        };
      });
  }

  hasAssociation(a, b, minimumWeight = 0.01) {
    return (
      this.getAssociationWeight(a, b) >=
      minimumWeight
    );
  }

  getVariants(value) {
    const id =
      this.canonicalId(value);

    if (!Number.isInteger(id)) {
      return [];
    }

    const variants =
      this.canonicalIdToVariants.get(id);

    return variants
      ? Array.from(variants)
      : [id];
  }

  isAlias(value) {
    const rawId =
      typeof value === "object" &&
      value !== null
        ? Number(value.id)
        : Number(value);

    if (!Number.isInteger(rawId)) {
      return false;
    }

    const canonical =
      this.canonicalId(rawId);

    return (
      Number.isInteger(canonical) &&
      canonical !== rawId
    );
  }

  tokenize(text) {
    if (!text) {
      this.lastUnknownWords = [];
      return [];
    }

    if (!this.loaded) {
      throw new Error(
        this.loadError
          ? `[VocabularyManager] Vocabulary unavailable: ` +
            `${this.loadError.message}`
          : "[VocabularyManager] Vocabulary is not ready yet."
      );
    }

    const rawWords =
      String(text)
        .trim()
        .match(
          /[A-Za-z]+(?:['’\-][A-Za-z]+)*|\d+(?:\.\d+)?|[.,!?;:~–—]/g
        ) || [];

    const tokens = [];
    const unknown = [];

    for (const word of rawWords) {
      // Canonical lookup is intentional. "How" and "how" now feed the same
      // embedding/head/token-pattern identity instead of splitting learning.
      const id =
        this.canonicalId(word);

      if (Number.isInteger(id)) {
        tokens.push(id);
      } else {
        unknown.push(word);
      }
    }

    this.lastUnknownWords = unknown;
    return tokens;
  }

  detokenize(tokens) {
    if (!tokens || tokens.length === 0) {
      return "";
    }

    const words = [];
    let sentenceStart = true;

    for (const raw of tokens) {
      const rawId =
        typeof raw === "object" &&
        raw !== null
          ? Number(raw.id)
          : Number(raw);

      const canonicalId =
        this.canonicalId(rawId);

      const id =
        Number.isInteger(canonicalId)
          ? canonicalId
          : rawId;

      let word =
        this.idToToken[id];

      if (!word) continue;

      const type =
        this.idToType[id] || "Other";

      // Canonical IDs usually prefer lowercase surface forms. Restore normal
      // display capitalization without creating a second neural token.
      if (
        sentenceStart &&
        type !== "Proper Noun" &&
        /^[a-z]/.test(word)
      ) {
        word =
          word.charAt(0).toUpperCase() +
          word.slice(1);
      }

      words.push(word);

      if (
        [".", "!", "?"].includes(word)
      ) {
        sentenceStart = true;
      } else if (
        ![",", ";", ":", "~", "–", "—"].includes(word)
      ) {
        sentenceStart = false;
      }
    }

    return words
      .join(" ")
      .replace(/\s+([.,!?;:])/g, "$1");
  }

  /**
   * Normal VilotNI generation semantics:
   * - strings are valid if they resolve to any canonical lexeme;
   * - numeric IDs are valid only if they are canonical ACTIVE IDs.
   *
   * This is what removes duplicate case variants from existing candidate scans
   * without requiring changes throughout RSL.
   */
  hasToken(value, includeAliases = false) {
    if (typeof value === "string") {
      return Number.isInteger(
        this.canonicalId(value)
      );
    }

    const id =
      typeof value === "object" &&
      value !== null
        ? Number(value.id)
        : Number(value);

    if (
      !Number.isInteger(id) ||
      id < 0 ||
      id >= this.vocabSize ||
      this.idToToken[id] === undefined
    ) {
      return false;
    }

    if (includeAliases) {
      return true;
    }

    return this.isCanonicalId[id] === true;
  }

  /**
   * Raw compatibility check for tools that deliberately need every physical ID.
   */
  hasRawToken(value) {
    if (typeof value === "string") {
      return this.rawTokenToId.has(value);
    }

    const id =
      typeof value === "object" &&
      value !== null
        ? Number(value.id)
        : Number(value);

    return (
      Number.isInteger(id) &&
      id >= 0 &&
      id < this.vocabSize &&
      this.idToToken[id] !== undefined
    );
  }
}

if (typeof window !== "undefined") {
  window.VocabularyManager =
    VocabularyManager;
}

if (typeof module !== "undefined") {
  module.exports =
    VocabularyManager;
}
