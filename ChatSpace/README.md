---
title: minillm 125M chat
colorFrom: indigo
colorTo: green
sdk: static
app_file: ChatPage.html
pinned: false
license: apache-2.0
---

Chat demo for a ~110M-parameter Llama-style model trained from scratch on FineWeb-Edu and
The Stack (Python, JavaScript, SQL), then fine-tuned on smol-smoltalk and Magicoder-OSS-Instruct.

The model runs entirely in the visitor's browser: Transformers.js loads the ONNX export in a Web
Worker (`ModelWorker.js`) and streams tokens to the chat page (`ChatPage.html`, `ChatClient.js`).
With WebGPU available it runs the fp32 export on the GPU; otherwise, or if that fails, the int8 export
(~135 MB) on WebAssembly. Both are cached after the first visit; add `?device=wasm` to force WebAssembly. Set the `MODEL_REPO` Space variable to the
public Hub repo holding the exported model (with its `onnx/` folder).
