/**
 * VilotNI 1.5: 3M-Neuron WebGPU Architecture (Zero-Hardcoding Engine)
 * Features:
 * - Contrastive Online Prompt Adaptation: strengthens targets, suppresses strong wrong continuations, and reshapes semantic embeddings.
 * - Dynamic Positional Encoding: Sinusoidal sequence tracking across sequence steps.
 * - Latent Neighborhood Affinity: Cluster dispersion based strictly on vocabulary index topology.
 * - In-kernel layer-7 prompt-context reinjection.
 * - Isolated opt-in eligibility traces for reward learning.
 * - Double-Buffered Asynchronous Readback: Non-blocking GPU pipeline execution.
 * - Ephemeral Synaptic Memory: IndexedDB persistence for learned weights.
 */
class VilotNI15Network {
  constructor(backend, vocabSize) {
    if (!vocabSize || vocabSize <= 0) {
      throw new Error(`[Network 1.5] Invalid dynamic vocabSize: ${vocabSize}`);
    }

    this.backend = backend;
    this.device = backend.device;
    this.vocabSize = vocabSize;
    this.layerCount = 15;
    this.layerSize = 200000;
    this.fanIn = 4;
    this.poolDim = 256;
    this.embDim = 64;

    this.gpuLayers = [];
    this.db = null;
    this.activeReadbackIdx = 0;

    // Eligibility traces are opt-in. Normal inference cannot accumulate stale
    // reward gradients from unrelated prompts.
    this.traceFresh = false;

    this.initGPUBuffers();
    this.initSynapticDB();
  }

  getPositionalEncoding(pos, dim) {
    const pe = new Float32Array(dim);
    for (let i = 0; i < dim; i += 2) {
      const denom = Math.pow(10000, (2 * (i / 2)) / dim);
      pe[i] = Math.sin(pos / denom);
      if (i + 1 < dim) {
        pe[i + 1] = Math.cos(pos / denom);
      }
    }
    return pe;
  }

  initGPUBuffers() {
    const dev = this.device;

    this.bufferA = this.backend.createBuffer(
      this.layerSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    this.bufferB = this.backend.createBuffer(
      this.layerSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    this.contextBuffer = this.backend.createBuffer(
      this.layerSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.pooledBuffer = this.backend.createBuffer(
      this.poolDim * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );

    this.readbackBuffers = [
      dev.createBuffer({ size: this.poolDim * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      dev.createBuffer({ size: this.poolDim * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    ];

    const scale = Math.sqrt(2.0 / this.fanIn);
    for (let l = 0; l < this.layerCount; l++) {
      const wData = new Float32Array(this.layerSize * this.fanIn);
      for (let i = 0; i < wData.length; i++) {
        wData[i] = (Math.random() * 2 - 1) * scale;
      }

      const bData = new Float32Array(this.layerSize);
      const wBuffer = this.backend.createBuffer(wData.byteLength, GPUBufferUsage.STORAGE, wData);
      const bBuffer = this.backend.createBuffer(bData.byteLength, GPUBufferUsage.STORAGE, bData);
      const gBuffer = this.backend.createBuffer(wData.byteLength, GPUBufferUsage.STORAGE);

      const isResidual = l > 0 ? 1 : 0;
      const uData = new Uint32Array([l === 0 ? this.embDim : this.layerSize, this.layerSize, l, isResidual]);
      const uBuffer = this.backend.createBuffer(uData.byteLength, GPUBufferUsage.UNIFORM, uData);
      const rlUniformBuffer = this.backend.createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

      this.gpuLayers.push({
        wBuffer,
        bBuffer,
        gBuffer,
        uBuffer,
        rlUniformBuffer,
        count: wData.length
      });
    }

    this.headWeights = new Float32Array(this.poolDim * this.vocabSize);
    this.headGradients = new Float32Array(this.poolDim * this.vocabSize);
    const headScale = 1.35 * Math.sqrt(2.0 / this.poolDim);
    for (let i = 0; i < this.headWeights.length; i++) {
      this.headWeights[i] = (Math.random() * 2 - 1) * headScale;
    }

    // Embeddings initialize strictly via mathematical harmonic distribution (no hardcoded word categories)
    this.embeddings = new Float32Array(this.vocabSize * this.embDim);
    for (let i = 0; i < this.vocabSize; i++) {
      const freq = (i + 1) / this.vocabSize;
      for (let d = 0; d < this.embDim; d++) {
        this.embeddings[i * this.embDim + d] = Math.sin(freq * (d + 1) * Math.PI) * 0.20 + ((Math.random() * 2 - 1) * 0.04);
      }
    }

    this.currentLatent = new Float32Array(this.poolDim);
  }

  async initSynapticDB() {
    return new Promise((resolve) => {
      const request = indexedDB.open("VilotNI_SynapticMemory_1_5", 1);
      request.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains("weights")) {
          db.createObjectStore("weights");
        }
      };
      request.onsuccess = async (e) => {
        this.db = e.target.result;
        await this.loadPersistedWeights();
        resolve();
      };
      request.onerror = () => resolve();
    });
  }

  async savePersistedWeights() {
    if (!this.db) return;
    try {
      const tx = this.db.transaction("weights", "readwrite");
      const store = tx.objectStore("weights");
      store.put(this.headWeights.buffer, "headWeights");
      store.put(this.embeddings.buffer, "embeddings");
    } catch (err) {
      console.warn("Synaptic memory save failed:", err);
    }
  }

  async loadPersistedWeights() {
    if (!this.db) return false;
    return new Promise((resolve) => {
      const tx = this.db.transaction("weights", "readonly");
      const store = tx.objectStore("weights");

      const reqHead = store.get("headWeights");
      reqHead.onsuccess = () => {
        if (reqHead.result && reqHead.result.byteLength === this.headWeights.byteLength) {
          this.headWeights.set(new Float32Array(reqHead.result));

          const reqEmb = store.get("embeddings");
          reqEmb.onsuccess = () => {
            if (reqEmb.result && reqEmb.result.byteLength === this.embeddings.byteLength) {
              this.embeddings.set(new Float32Array(reqEmb.result));
              resolve(true);
              return;
            }
            resolve(false);
          };
          reqEmb.onerror = () => resolve(false);
        } else {
          resolve(false);
        }
      };
      reqHead.onerror = () => resolve(false);
    });
  }

  normalizeEmbeddingRow(tokenId, maxNorm = 1.75) {
    const off = tokenId * this.embDim;
    let normSq = 0.0;

    for (let d = 0; d < this.embDim; d++) {
      const v = this.embeddings[off + d];
      normSq += v * v;
    }

    const norm = Math.sqrt(normSq);
    if (norm <= maxNorm || norm <= 1e-8) return;

    const scale = maxNorm / norm;
    for (let d = 0; d < this.embDim; d++) {
      this.embeddings[off + d] *= scale;
    }
  }

  /**
   * Autonomous Contrastive Sequence Adaptation:
   * Strengthens the correct next token, suppresses the strongest wrong
   * continuations, and reshapes embeddings from real token co-occurrence.
   */
  async adaptToPromptTransitions(promptTokens, passes = 2, lr = 0.035) {
    if (!promptTokens || promptTokens.length < 2) return;

    const negativeCount = 3;

    for (let pass = 0; pass < passes; pass++) {
      for (let i = 0; i < promptTokens.length - 1; i++) {
        const prefix = promptTokens.slice(0, i + 1);
        const target = promptTokens[i + 1];

        const logits = await this.forwardSequence(
          prefix,
          { captureTrace: false }
        );

        let maxL = -Infinity;
        for (let v = 0; v < logits.length; v++) {
          if (logits[v] > maxL) maxL = logits[v];
        }

        let sumExp = 0.0;
        const probs = new Float32Array(logits.length);

        for (let v = 0; v < logits.length; v++) {
          const p = Math.exp(logits[v] - maxL);
          probs[v] = p;
          sumExp += p;
        }

        const invSum = 1.0 / (sumExp || 1.0);
        for (let v = 0; v < probs.length; v++) {
          probs[v] *= invSum;
        }

        const negatives = [];
        for (let v = 0; v < probs.length; v++) {
          if (v === target) continue;
          const score = probs[v];

          if (negatives.length < negativeCount) {
            negatives.push({ id: v, score });
            negatives.sort((a, b) => b.score - a.score);
          } else if (score > negatives[negatives.length - 1].score) {
            negatives[negatives.length - 1] = { id: v, score };
            negatives.sort((a, b) => b.score - a.score);
          }
        }

        const latent = this.currentLatent;
        const targetProb = probs[target] || 0.0;
        const targetError = (1.0 - targetProb) * lr;
        const targetOffset = target * this.poolDim;

        // Positive head update.
        for (let p = 0; p < this.poolDim; p++) {
          this.headWeights[targetOffset + p] +=
            targetError * latent[p];
        }

        // Contrastive negative head updates.
        for (const neg of negatives) {
          const negOffset = neg.id * this.poolDim;
          const negStrength = lr * neg.score * 0.55;

          for (let p = 0; p < this.poolDim; p++) {
            this.headWeights[negOffset + p] -=
              negStrength * latent[p];
          }
        }

        // Recent-prefix semantic centroid. This gradually replaces the initial
        // vocabulary-index topology with actual learned co-occurrence geometry.
        const context = new Float32Array(this.embDim);
        const contextStart = Math.max(0, prefix.length - 4);
        let contextCount = 0;

        for (let j = contextStart; j < prefix.length; j++) {
          const id = prefix[j];
          const off = id * this.embDim;

          for (let d = 0; d < this.embDim; d++) {
            context[d] += this.embeddings[off + d];
          }
          contextCount++;
        }

        if (contextCount > 0) {
          for (let d = 0; d < this.embDim; d++) {
            context[d] /= contextCount;
          }

          const targetEmbOffset = target * this.embDim;
          const embRate = targetError * 0.18;

          for (let d = 0; d < this.embDim; d++) {
            const current = this.embeddings[targetEmbOffset + d];
            this.embeddings[targetEmbOffset + d] +=
              embRate * (context[d] - current);
          }

          this.normalizeEmbeddingRow(target);

          // Push the strongest wrong continuations slightly away from this
          // local context, sharpening the semantic decision boundary.
          for (const neg of negatives) {
            const negEmbOffset = neg.id * this.embDim;
            const repel = lr * neg.score * 0.045;

            for (let d = 0; d < this.embDim; d++) {
              const direction =
                context[d] - this.embeddings[negEmbOffset + d];

              this.embeddings[negEmbOffset + d] -=
                repel * direction;
            }

            this.normalizeEmbeddingRow(neg.id);
          }
        }
      }
    }

    await this.savePersistedWeights();
  }

  async forwardSequence(tokenIds, options = null) {
    const dev = this.device;

    // Backward compatible:
    // forwardSequence(tokens) -> normal inference, no eligibility trace
    // forwardSequence(tokens, true) -> capture a reward trace
    // forwardSequence(tokens, { captureTrace: true }) -> same
    const captureTrace =
      options === true ||
      (options && options.captureTrace === true);
    const initialInput = new Float32Array(this.layerSize);
    const seqLen = tokenIds.length;

    for (let pos = 0; pos < seqLen; pos++) {
      const tokenId = tokenIds[pos];
      const offset = (tokenId % this.vocabSize) * this.embDim;
      const pe = this.getPositionalEncoding(pos, this.embDim);
      const recencyWeight = 0.85 + 0.15 * (pos / Math.max(1, seqLen));

      for (let d = 0; d < this.embDim; d++) {
        const tokenSignal = this.embeddings[offset + d] + pe[d];
        initialInput[d] += (tokenSignal * recencyWeight) / Math.sqrt(seqLen);
      }
    }
    dev.queue.writeBuffer(this.contextBuffer, 0, initialInput);

    const encoder = dev.createCommandEncoder();
    let currentIn = this.contextBuffer;
    let currentOut = this.bufferA;

    for (let l = 0; l < this.layerCount; l++) {
      const layer = this.gpuLayers[l];
      const bindGroup = dev.createBindGroup({
        layout: this.backend.forwardPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: layer.uBuffer } },
          { binding: 1, resource: { buffer: currentIn } },
          { binding: 2, resource: { buffer: layer.wBuffer } },
          { binding: 3, resource: { buffer: layer.bBuffer } },
          { binding: 4, resource: { buffer: currentOut } },
          { binding: 5, resource: { buffer: this.contextBuffer } }
        ]
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.backend.forwardPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.layerSize / 256));
      pass.end();

      if (captureTrace) {
        // This reward trace belongs only to THIS pass.
        encoder.clearBuffer(layer.gBuffer);

        const traceBindGroup = dev.createBindGroup({
          layout: this.backend.tracePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: layer.uBuffer } },
            { binding: 1, resource: { buffer: currentIn } },
            { binding: 2, resource: { buffer: currentOut } },
            { binding: 3, resource: { buffer: layer.gBuffer } }
          ]
        });

        const tracePass = encoder.beginComputePass();
        tracePass.setPipeline(this.backend.tracePipeline);
        tracePass.setBindGroup(0, traceBindGroup);
        tracePass.dispatchWorkgroups(Math.ceil(this.layerSize / 256));
        tracePass.end();
      }

      currentIn = currentOut;
      currentOut = (currentOut === this.bufferA) ? this.bufferB : this.bufferA;
    }

    const poolBindGroup = dev.createBindGroup({
      layout: this.backend.poolPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: currentIn } },
        { binding: 1, resource: { buffer: this.pooledBuffer } }
      ]
    });

    const poolPass = encoder.beginComputePass();
    poolPass.setPipeline(this.backend.poolPipeline);
    poolPass.setBindGroup(0, poolBindGroup);
    poolPass.dispatchWorkgroups(4);
    poolPass.end();

    const targetReadback = this.readbackBuffers[this.activeReadbackIdx];
    this.activeReadbackIdx = 1 - this.activeReadbackIdx;

    encoder.copyBufferToBuffer(this.pooledBuffer, 0, targetReadback, 0, this.poolDim * 4);
    dev.queue.submit([encoder.finish()]);

    await targetReadback.mapAsync(GPUMapMode.READ);
    const latent = new Float32Array(targetReadback.getMappedRange()).slice();
    targetReadback.unmap();
    this.currentLatent = latent;
    this.traceFresh = captureTrace;

    const logits = new Float32Array(this.vocabSize);
    for (let v = 0; v < this.vocabSize; v++) {
      let sum = 0;
      const off = v * this.poolDim;
      for (let p = 0; p < this.poolDim; p++) {
        sum += this.headWeights[off + p] * latent[p];
      }
      logits[v] = sum * 1.35;
    }

    return logits;
  }

  async captureRewardTrace(tokenIds) {
    return this.forwardSequence(
      tokenIds,
      { captureTrace: true }
    );
  }

  applyPolicyReward(reward, lr = 0.008, updateDeepLayers = true) {
    const dev = this.device;

    for (let i = 0; i < this.headWeights.length; i++) {
      this.headWeights[i] += lr * reward * this.headGradients[i];
      this.headWeights[i] *= 0.9999;
      this.headGradients[i] = 0;
    }

    if (reward > 0.1) {
      this.savePersistedWeights();
    }

    // Never reward stale traces. Deep-layer reward learning requires an
    // explicit captureRewardTrace(tokens) immediately beforehand.
    if (!updateDeepLayers || !this.traceFresh) return;

    const encoder = dev.createCommandEncoder();

    for (let l = 0; l < this.layerCount; l++) {
      const layer = this.gpuLayers[l];
      const rlUniformData = new Float32Array([reward, lr, 0, 0.9999]);
      new Uint32Array(rlUniformData.buffer)[2] = layer.count;

      dev.queue.writeBuffer(layer.rlUniformBuffer, 0, rlUniformData);

      const bindGroup = dev.createBindGroup({
        layout: this.backend.rewardPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: layer.rlUniformBuffer } },
          { binding: 1, resource: { buffer: layer.wBuffer } },
          { binding: 2, resource: { buffer: layer.gBuffer } }
        ]
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.backend.rewardPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(layer.count / 256));
      pass.end();
    }

    dev.queue.submit([encoder.finish()]);
    this.traceFresh = false;
  }
}