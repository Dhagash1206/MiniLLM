// Runs the model in a Web Worker so generation never freezes the page.
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

// Always fetch from the Hub; never probe the Space's own host for model files.
env.allowLocalModels = false;

const stoppingCriteria = new InterruptableStoppingCriteria();
let tokenizer = null;
let model = null;
let maxContext = 1024;

// int8 ONNX on WebAssembly works in every modern browser. WebGPU runs the fp32 ONNX (same file the export
// verifies against PyTorch): a bigger download, but several times faster generation on most GPUs.
const WASM_BACKEND = { device: "wasm", dtype: "q8" };
const WEBGPU_BACKEND = { device: "webgpu", dtype: "fp32" };

async function webgpuAvailable() {
  try {
    return Boolean(self.navigator?.gpu && (await self.navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function load(modelRepo, preferredDevice) {
  const progress_callback = (data) => self.postMessage({ type: "progress", data });
  tokenizer = await AutoTokenizer.from_pretrained(modelRepo, { progress_callback });
  const tryWebgpu = preferredDevice !== "wasm" && (await webgpuAvailable());
  let backend = tryWebgpu ? WEBGPU_BACKEND : WASM_BACKEND;
  try {
    model = await AutoModelForCausalLM.from_pretrained(modelRepo, { ...backend, progress_callback });
  } catch (error) {
    if (backend === WASM_BACKEND) throw error;
    // Missing fp32 file, driver bug or out of GPU memory: the int8 WebAssembly build still works.
    backend = WASM_BACKEND;
    model = await AutoModelForCausalLM.from_pretrained(modelRepo, { ...backend, progress_callback });
  }
  maxContext = model.config.max_position_embeddings ?? maxContext;
  self.postMessage({ type: "ready", backend: backend.device });
}

function buildInputs(messages, systemPrompt, maxNewTokens) {
  const system = systemPrompt.trim() ? [{ role: "system", content: systemPrompt }] : [];
  let turns = messages;
  // Drop the oldest exchanges until prompt + answer fit in the model's context window.
  while (true) {
    const inputs = tokenizer.apply_chat_template([...system, ...turns], {
      add_generation_prompt: true,
      return_dict: true,
    });
    const length = inputs.input_ids.dims[1];
    if (length + maxNewTokens <= maxContext || turns.length === 1) {
      // A single over-long message still must not run past the position table.
      const allowedNewTokens = Math.min(maxNewTokens, maxContext - length);
      if (allowedNewTokens < 1) throw new Error("Message is too long for the model's context window.");
      return { inputs, allowedNewTokens };
    }
    turns = turns.length > 2 ? turns.slice(2) : turns.slice(-1);
  }
}

async function generate({ messages, settings }) {
  stoppingCriteria.reset();
  const { inputs, allowedNewTokens } = buildInputs(messages, settings.system_prompt, settings.max_new_tokens);
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => self.postMessage({ type: "token", text }),
  });
  await model.generate({
    ...inputs,
    max_new_tokens: allowedNewTokens,
    do_sample: settings.temperature > 0,
    temperature: Math.max(settings.temperature, 1e-5),
    top_p: settings.top_p,
    repetition_penalty: settings.repetition_penalty,
    streamer,
    stopping_criteria: stoppingCriteria,
  });
  self.postMessage({ type: "done" });
}

self.addEventListener("message", async (event) => {
  const { type } = event.data;
  try {
    if (type === "load") await load(event.data.modelRepo, event.data.device);
    else if (type === "generate") await generate(event.data);
    else if (type === "stop") stoppingCriteria.interrupt();
  } catch (error) {
    self.postMessage({ type: "error", message: String(error?.message ?? error) });
  }
});
