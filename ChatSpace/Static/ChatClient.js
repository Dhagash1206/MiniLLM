"use strict";

const conversationElement = document.getElementById("conversation");
const emptyState = document.getElementById("empty-state");
const composer = document.getElementById("composer");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const stopButton = document.getElementById("stop-button");
const clearButton = document.getElementById("clear-button");
const settingsToggle = document.getElementById("settings-toggle");
const settingsPanel = document.getElementById("settings-panel");

const sliderIds = ["max-new-tokens", "temperature", "top-p", "repetition-penalty"];
const history = [];
let activeController = null;

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
  sendButton.disabled = isBusy;
  messageInput.disabled = isBusy;
  stopButton.hidden = !isBusy;
  clearButton.disabled = isBusy;
  if (!isBusy) messageInput.focus();
}

// Parses a chunk of server-sent events; returns [events, leftover partial text].
function parseEvents(buffer) {
  const events = [];
  const blocks = buffer.split("\n\n");
  const leftover = blocks.pop();
  for (const block of blocks) {
    let eventName = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) eventName = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    events.push({ eventName, payload: data ? JSON.parse(data) : {} });
  }
  return [events, leftover];
}

async function sendMessage(text) {
  history.push({ role: "user", content: text });
  addMessage("user", text);
  const bubble = addMessage("assistant", "");
  bubble.classList.add("typing");
  setBusy(true);

  activeController = new AbortController();
  let reply = "";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history, ...readSettings() }),
      signal: activeController.signal,
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(typeof detail.detail === "string" ? detail.detail : `Request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const [events, leftover] = parseEvents(buffer);
      buffer = leftover;
      for (const { eventName, payload } of events) {
        if (eventName === "start" && payload.queued) {
          bubble.innerHTML = '<span class="status">Waiting for another reply to finish...</span>';
        } else if (eventName === "token") {
          reply += payload.text;
          bubble.innerHTML = renderReply(reply);
          bubble.scrollIntoView({ block: "end" });
        } else if (eventName === "error") {
          throw new Error(payload.message);
        }
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      bubble.parentElement.classList.add("error");
      bubble.textContent = `Error: ${error.message}`;
    }
  } finally {
    bubble.classList.remove("typing");
    if (reply) {
      history.push({ role: "assistant", content: reply });
    } else if (!bubble.parentElement.classList.contains("error")) {
      bubble.innerHTML = '<span class="status">(no reply)</span>';
      history.pop();  // Drop the unanswered user turn so roles keep alternating.
    } else {
      history.pop();
    }
    activeController = null;
    setBusy(false);
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text || activeController) return;
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

stopButton.addEventListener("click", () => activeController?.abort());

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
  button.addEventListener("click", () => {
    if (!activeController) sendMessage(button.textContent);
  });
});

messageInput.focus();
