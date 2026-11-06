"use strict";

// Static Spaces expose Space variables here; `?model=user/repo` overrides for local testing.
const MODEL_REPO = new URLSearchParams(location.search).get("model")
  || window.huggingface?.variables?.MODEL_REPO
  || "your-username/minillm-125m-chat";

const conversationElement = document.getElementById("conversation");
const emptyState = document.getElementById("empty-state");
const loadStatus = document.getElementById("load-status");
const composer = document.getElementById("composer");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const stopButton = document.getElementById("stop-button");
const clearButton = document.getElementById("clear-button");
const settingsToggle = document.getElementById("settings-toggle");
const settingsPanel = document.getElementById("settings-panel");

const sliderIds = ["max-new-tokens", "temperature", "top-p", "repetition-penalty"];
const history = [];
const fileProgress = new Map();
let modelReady = false;
let activeReply = null;  // { bubble, text } while a reply is streaming

const worker = new Worker("ModelWorker.js", { type: "module" });

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

// Renders ``` fenced blocks as <pre> and `inline` spans as <code>; everything else stays plain text.
function renderReply(text) {
  const parts = text.split("```");
  return parts.map((part, index) => {
    if (index % 2 === 1) {
      const code = part.replace(/^[\w+-]*\n/, "");
      return `<pre><code>${escapeHtml(code)}</code></pre>`;
    }
    return escapeHtml(part).replace(/`([^`\n]+)`/g, "<code>$1</code>");
  }).join("");
}

function addMessage(role, text) {
  emptyState.hidden = true;
  const row = document.createElement("div");
  row.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.innerHTML = renderReply(text);
  } else {
    bubble.textContent = text;
  }
  row.appendChild(bubble);
  conversationElement.appendChild(row);
  bubble.scrollIntoView({ block: "end" });
  return bubble;
}

function readSettings() {
  return {
    system_prompt: document.getElementById("system-prompt").value,
    max_new_tokens: Number(document.getElementById("max-new-tokens").value),
    temperature: Number(document.getElementById("temperature").value),
    top_p: Number(document.getElementById("top-p").value),
    repetition_penalty: Number(document.getElementById("repetition-penalty").value),
  };
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy || !modelReady;
  messageInput.disabled = isBusy || !modelReady;
  stopButton.hidden = !isBusy;
  clearButton.disabled = isBusy;
  if (!isBusy && modelReady) messageInput.focus();
}

function showLoadProgress(data) {
  if (data.status === "progress" && data.total) {
    fileProgress.set(data.file, { loaded: data.loaded, total: data.total });
    let loaded = 0;
    let total = 0;
    for (const entry of fileProgress.values()) { loaded += entry.loaded; total += entry.total; }
    loadStatus.textContent = `Downloading model: ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
  }
}

function finishReply() {
  if (!activeReply) return;
  const { bubble, text } = activeReply;
  bubble.classList.remove("typing");
  if (text) {
    history.push({ role: "assistant", content: text });
  } else {
    bubble.innerHTML = '<span class="status">(no reply)</span>';
    history.pop();  // Drop the unanswered user turn so roles keep alternating.
  }
  activeReply = null;
  setBusy(false);
}

worker.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "progress") {
    showLoadProgress(message.data);
  } else if (message.type === "ready") {
    modelReady = true;
    loadStatus.textContent = `Model loaded (${MODEL_REPO}). Replies are generated on your device.`;
    setBusy(false);
  } else if (message.type === "token" && activeReply) {
    activeReply.text += message.text;
    activeReply.bubble.innerHTML = renderReply(activeReply.text);
    activeReply.bubble.scrollIntoView({ block: "end" });
  } else if (message.type === "done") {
    finishReply();
  } else if (message.type === "error") {
    if (activeReply) {
      activeReply.bubble.parentElement.classList.add("error");
      activeReply.bubble.textContent = `Error: ${message.message}`;
      activeReply.text = "";
      history.pop();
      activeReply.bubble.classList.remove("typing");
      activeReply = null;
      setBusy(false);
    } else {
      loadStatus.textContent = `Could not load ${MODEL_REPO}: ${message.message}`;
      loadStatus.classList.add("error");
    }
  }
});

function sendMessage(text) {
  if (!modelReady || activeReply) return;
  history.push({ role: "user", content: text });
  addMessage("user", text);
  const bubble = addMessage("assistant", "");
  bubble.classList.add("typing");
  activeReply = { bubble, text: "" };
  setBusy(true);
  worker.postMessage({ type: "generate", messages: history.slice(), settings: readSettings() });
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendMessage(text);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = `${messageInput.scrollHeight}px`;
});

stopButton.addEventListener("click", () => worker.postMessage({ type: "stop" }));

clearButton.addEventListener("click", () => {
  history.length = 0;
  conversationElement.querySelectorAll(".message").forEach((element) => element.remove());
  emptyState.hidden = false;
});

settingsToggle.addEventListener("click", () => {
  settingsPanel.hidden = !settingsPanel.hidden;
  settingsToggle.setAttribute("aria-expanded", String(!settingsPanel.hidden));
});

for (const sliderId of sliderIds) {
  const slider = document.getElementById(sliderId);
  const output = document.getElementById(`${sliderId}-value`);
  slider.addEventListener("input", () => { output.textContent = slider.value; });
}

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.textContent));
});

setBusy(false);
worker.postMessage({ type: "load", modelRepo: MODEL_REPO });
