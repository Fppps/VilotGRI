/**
 * WebGPU Compute Backend for VilotNI 1.5.
 * Features:
 * - Residual sparse forward propagation across 200,000-neuron steps.
 * - Synaptic eligibility trace kernel (solves the zero-gradient bug on the 3M units).
 * - Static weight-update reward kernel.
 */
class WebGPUComputeBackend {
  constructor() {
    this.device = null;
    this.forwardPipeline = null;
    this.tracePipeline = null;
    this.poolPipeline = null;
    this.rewardPipeline = null;
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported on this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU hardware adapter available.");
    this.device = await adapter.requestDevice();

    await this.compilePipelines();
    return adapter.info || { description: "WebGPU Active" };
  }

  async compilePipelines() {
    // 1. Forward Pass with Residual Skip Connections
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

        var act: f32 = select(0.01 * sum, sum, sum > 0.0);

        if (params.isResidual == 1u) {
          act += 0.5 * prevActivations[neuronIdx];
        }

        outputActivations[neuronIdx] = act;
      }
    `;

    // 2. Synaptic Eligibility Trace Kernel (Computes pre-post synaptic co-activations)
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

    // 3. Bottleneck Pooling Kernel (Layer 15 -> 256 dimensions)
    const poolWGSL = `
      @group(0) @binding(0) var<storage, read> layer15Activations: array<f32>;
      @group(0) @binding(1) var<storage, read_write> pooledOutput: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let p = id.x;
        if (p >= 256u) { return; }

        let stride = 200000u / 256u;
        let start = p * stride;
        var acc: f32 = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) {
          acc += layer15Activations[start + i];
        }
        pooledOutput[p] = tanh(acc / 32.0);
      }
    `;

    // 4. Weight Update Kernel (Applies reward to accumulated traces)
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