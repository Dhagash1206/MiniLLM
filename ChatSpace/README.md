---
title: minillm 125M chat
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

Chat demo for a ~110M-parameter Llama-style model trained from scratch on FineWeb-Edu and
The Stack (Python, JavaScript, SQL), then fine-tuned on CodeAlpaca, Magicoder-OSS-Instruct and
an OpenHermes subset.

A FastAPI server (`App.py`) streams tokens over server-sent events to a plain HTML/JS chat page
(`Static/`). Set the `MODEL_REPO` Space variable to the Hub repo holding the exported model.
