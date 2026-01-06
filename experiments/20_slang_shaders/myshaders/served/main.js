const canvas = document.getElementById("c");

if (!navigator.gpu) {
  throw new Error("WebGPU not supported");
}

const adapter = await navigator.gpu.requestAdapter();
const device  = await adapter.requestDevice();

const context = canvas.getContext("webgpu");
const format  = navigator.gpu.getPreferredCanvasFormat();

context.configure({
  device,
  format,
  alphaMode: "opaque"
});

// Load WGSL
// const wgsl = await fetch("build/toy1.wgsl").then(r => r.text());
const wgsl = await fetch("toy1.wgsl").then(r => r.text());


const module = device.createShaderModule({ code: wgsl });

// Uniform buffer
const uniformBufferSize = 16; // vec2 + float + padding
const uniformBuffer = device.createBuffer({
  size: uniformBufferSize,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});

const bindGroupLayout = device.createBindGroupLayout({
  entries: [{
    binding: 0,
    visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
    buffer: {}
  }]
});

const pipeline = device.createRenderPipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  }),
  vertex: {
    module,
    entryPoint: "vsMain"
  },
  fragment: {
    module,
    entryPoint: "psMain",
    targets: [{ format }]
  },
  primitive: {
    topology: "triangle-list"
  }
});

const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [{
    binding: 0,
    resource: { buffer: uniformBuffer }
  }]
});

function frame(t) {
  const width  = canvas.width  = canvas.clientWidth;
  const height = canvas.height = canvas.clientHeight;

  const data = new Float32Array([
    width, height,
    t * 0.001,
    0.0
  ]);
  device.queue.writeBuffer(uniformBuffer, 0, data);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store"
    }]
  });

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);


