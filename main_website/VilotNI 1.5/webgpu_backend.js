/**
 * WebGPU Compute Backend for VilotNI 1.5.
 * Features:
 * - 15-step residual sparse propagation mesh across 3,000,000 active neurons.
 * - Synaptic eligibility trace kernel for deep weight updates.
 * - True full-coverage hybrid (Max + Mean) Step 15 pooling kernel for sharp latent clarity.
 * - In-kernel layer-7 prompt-context reinjection.
 */
class WebGPUComputeBackend {
  constructor() {
    this.device = null;
    this.forwardPipeline = null;
    this.tracePipeline = null;
    this.poolPipeline = null;
    this.rewardPipeline = null;

    this.deviceLost = false;
    this.deviceLostInfo = null;
    this.deviceGeneration = 0;
  }

  isLikelyDeviceLoss(error) {
    const message =
      String(
        error?.message ||
        error ||
        ""
      );

    return (
      this.deviceLost ||
      /external Instance reference/i.test(message) ||
      /GPUBuffer/i.test(message) ||
      /GPUDevice/i.test(message) ||
      /device\s+(?:was\s+)?lost/i.test(message) ||
      /mapAsync/i.test(message) ||
      /WebGPU.*(?:lost|invalid|destroyed)/i.test(message)
    );
  }

  markRuntimeFailure(error) {
    if (
      this.isLikelyDeviceLoss(error)
    ) {
      this.deviceLost = true;

      if (!this.deviceLostInfo) {
        this.deviceLostInfo = {
          reason: "runtime-error",
          message:
            String(
              error?.message ||
              error ||
              "Unknown WebGPU runtime failure"
            )
        };
      }
    }
  }

  assertUsable(stage = "WebGPU operation") {
    if (!this.device) {
      const err =
        new Error(
          `${stage}: WebGPU device is unavailable.`
        );

      err.code =
        "VILOTNI_WEBGPU_RUNTIME_FAILURE";

      throw err;
    }

    if (this.deviceLost) {
      const detail =
        this.deviceLostInfo?.message ||
        this.deviceLostInfo?.reason ||
        "device lost";

      const err =
        new Error(
          `${stage}: WebGPU device is no longer valid (${detail}).`
        );

      err.code =
        "VILOTNI_WEBGPU_RUNTIME_FAILURE";

      throw err;
    }
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported on this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU hardware adapter available.");
    this.device =
      await adapter.requestDevice();

    this.deviceLost = false;
    this.deviceLostInfo = null;

    const generation =
      ++this.deviceGeneration;

    this.device.lost
      .then(info => {
        if (
          generation !==
          this.deviceGeneration
        ) {
          return;
        }

        this.deviceLost = true;

        this.deviceLostInfo = {
          reason:
            info?.reason ||
            "unknown",
          message:
            info?.message ||
            "WebGPU device lost"
        };

        console.warn(
          "[WebGPU] Device lost:",
          this.deviceLostInfo
        );
      })
      .catch(() => {});

    await this.compilePipelines();
    return adapter.info || { description: "WebGPU Hardware Matrix Active" };
  }

  async compilePipelines() {
    this.assertUsable(
      "compilePipelines"
    );
    // 1. Forward Pass with Residual Synaptic Skip Connections
    const forwardWGSL = `
      struct LayerUniforms {
        prevSize: u32,
        currentSize: u32,
        layerIndex: u32,
        isResidual: u32,
      };

      @group(0) @binding(0) var<uniform> params: LayerUniforms;
      @group(0) @binding(1) var<storage, read> prevActivations: array<f32>;
      @group(0) @binding(2) var<storage, read> weights: array<f32>;
      @group(0) @binding(3) var<storage, read> biases: array<f32>;
      @group(0) @binding(4) var<storage, read_write> outputActivations: array<f32>;
      @group(0) @binding(5) var<storage, read> promptContext: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let neuronIdx = id.x;
        if (neuronIdx >= params.currentSize) { return; }

        let baseW = neuronIdx * 4u;
        let pSize = params.prevSize;
        let lIdx = params.layerIndex;

        let i0 = (neuronIdx * 7919u + lIdx * 101u) % pSize;
        let i1 = (neuronIdx * 7919u + 31u + lIdx * 101u) % pSize;
        let i2 = (neuronIdx * 7919u + 67u + lIdx * 101u) % pSize;
        let i3 = (neuronIdx * 7919u + 109u + lIdx * 101u) % pSize;

        var sum: f32 = biases[neuronIdx];
        sum += weights[baseW + 0u] * prevActivations[i0];
        sum += weights[baseW + 1u] * prevActivations[i1];
        sum += weights[baseW + 2u] * prevActivations[i2];
        sum += weights[baseW + 3u] * prevActivations[i3];

        // True mid-network prompt reinjection, inside the same GPU command
        // stream. Earlier layer work cannot overwrite this before layer 7 uses it.
        if (lIdx == 7u) {
          let contextIdx = neuronIdx % 64u;
          sum += 0.18 * promptContext[contextIdx];
        }

        var act: f32 = select(0.02 * sum, sum, sum > 0.0);

        // Residual skip preserves signal energy across 15 steps
        if (params.isResidual == 1u) {
          act += 0.5 * prevActivations[neuronIdx];
        }

        outputActivations[neuronIdx] = act;
      }
    `;

    // 2. Synaptic Eligibility Trace Kernel
    const traceWGSL = `
      struct LayerUniforms {
        prevSize: u32,
        currentSize: u32,
        layerIndex: u32,
        isResidual: u32,
      };

      @group(0) @binding(0) var<uniform> params: LayerUniforms;
      @group(0) @binding(1) var<storage, read> prevActivations: array<f32>;
      @group(0) @binding(2) var<storage, read> outputActivations: array<f32>;
      @group(0) @binding(3) var<storage, read_write> grads: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let neuronIdx = id.x;
        if (neuronIdx >= params.currentSize) { return; }

        let outAct = outputActivations[neuronIdx];
        if (abs(outAct) < 0.0001) { return; }

        let baseW = neuronIdx * 4u;
        let pSize = params.prevSize;
        let lIdx = params.layerIndex;

        let i0 = (neuronIdx * 7919u + lIdx * 101u) % pSize;
        let i1 = (neuronIdx * 7919u + 31u + lIdx * 101u) % pSize;
        let i2 = (neuronIdx * 7919u + 67u + lIdx * 101u) % pSize;
        let i3 = (neuronIdx * 7919u + 109u + lIdx * 101u) % pSize;

        grads[baseW + 0u] += outAct * prevActivations[i0];
        grads[baseW + 1u] += outAct * prevActivations[i1];
        grads[baseW + 2u] += outAct * prevActivations[i2];
        grads[baseW + 3u] += outAct * prevActivations[i3];
      }
    `;

    // 3. Full-Coverage Hybrid Pooling Kernel (Step 15 -> 256 Latent Dimensions)
    // Evaluates every single neuron in Step 15 to eliminate the 191,000-neuron blind spot
    const poolWGSL = `
      @group(0) @binding(0) var<storage, read> layer15Activations: array<f32>;
      @group(0) @binding(1) var<storage, read_write> pooledOutput: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let p = id.x;
        if (p >= 256u) { return; }

        // Proportional integer boundaries cover ALL 200,000 neurons.
        // Slices are 781/782 neurons and the final slice ends at 200000.
        let start = (p * 200000u) / 256u;
        let end = ((p + 1u) * 200000u) / 256u;

        var maxAct: f32 = -1e9;
        var sumAct: f32 = 0.0;
        var count: f32 = 0.0;

        for (var i = start; i < end; i = i + 1u) {
          let val = layer15Activations[i];
          if (val > maxAct) { maxAct = val; }
          sumAct += val;
          count += 1.0;
        }

        let meanAct = sumAct / max(1.0, count);
        // Blends salient peaks with distribution density
        pooledOutput[p] = tanh(0.75 * maxAct + 0.25 * meanAct);
      }
    `;

    // 4. Weight Update Kernel
    const rewardWGSL = `
      struct RLUniforms {
        reward: f32,
        lr: f32,
        count: u32,
        decay: f32,
      };

      @group(0) @binding(0) var<uniform> rlParams: RLUniforms;
      @group(0) @binding(1) var<storage, read_write> weights: array<f32>;
      @group(0) @binding(2) var<storage, read_write> grads: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let idx = id.x;
        if (idx >= rlParams.count) { return; }

        let g = grads[idx];
        weights[idx] = (weights[idx] + rlParams.lr * rlParams.reward * g) * rlParams.decay;
        grads[idx] = 0.0;
      }
    `;

    this.forwardPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: forwardWGSL }), entryPoint: "main" }
    });

    this.tracePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: traceWGSL }), entryPoint: "main" }
    });

    this.poolPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: poolWGSL }), entryPoint: "main" }
    });

    this.rewardPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: rewardWGSL }), entryPoint: "main" }
    });
  }

  createBuffer(size, usage, initialData = null) {
    this.assertUsable(
      "createBuffer"
    );

    const buffer = this.device.createBuffer({
      size: Math.ceil(size / 4) * 4,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: initialData !== null
    });
    if (initialData) {
      new Float32Array(buffer.getMappedRange()).set(initialData);
      buffer.unmap();
    }
    return buffer;
  }
}