const statusEl = document.querySelector("#service-status");
const outputEl = document.querySelector("#output");
const settingsForm = document.querySelector("#settings-form");
const vocabularyForm = document.querySelector("#vocabulary-form");
const transcriptionForm = document.querySelector("#transcription-form");
const wordTable = document.querySelector("#word-table");
const recordingStatusEl = document.querySelector("#recording-status");
const startRecordingButton = document.querySelector("#start-recording");
const stopRecordingButton = document.querySelector("#stop-recording");
const transcribeRecordingButton = document.querySelector("#transcribe-recording");
const recordedAudioEl = document.querySelector("#recorded-audio");
const logsEl = document.querySelector("#logs");
const logStreamStatusEl = document.querySelector("#log-stream-status");
const clearLogsButton = document.querySelector("#clear-logs");

let mediaRecorder = null;
let recordingStream = null;
let audioContext = null;
let recordingSource = null;
let recordingProcessor = null;
let recordingSilence = null;
let recordedSamples = [];
let recordedSampleRate = 44100;
let recordedBlob = null;
const seenLogIds = new Set();

function showOutput(value) {
  if (typeof value === "string") {
    outputEl.textContent = value || "无内容";
    return;
  }
  outputEl.textContent = JSON.stringify(value, null, 2);
}

function setStatus(text, className) {
  statusEl.textContent = text;
  statusEl.className = `status ${className || ""}`.trim();
}

function setLogStreamStatus(text, className) {
  logStreamStatusEl.textContent = text;
  logStreamStatusEl.className = className || "";
}

function parseHints(value) {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith("[")) {
    return JSON.parse(trimmed);
  }
  return trimmed
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function formatHints(value) {
  if (!value || value.length === 0) return "";
  return value.join(",");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail || body : body;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

async function loadHealth() {
  try {
    await requestJson("/health");
    setStatus("运行中", "ok");
  } catch (error) {
    setStatus("异常", "fail");
  }
}

async function loadSettings() {
  const settings = await requestJson("/api/settings");
  document.querySelector("#language-hints").value = formatHints(settings.language_hints);
  document.querySelector("#vocabulary-id").value = settings.vocabulary_id || "";
  showOutput({ settings });
}

async function loadLogs() {
  const result = await requestJson("/api/logs");
  logsEl.textContent = "";
  seenLogIds.clear();
  for (const event of result.data || []) {
    appendLogEvent(event);
  }
}

function connectLogStream() {
  if (!window.EventSource) {
    setLogStreamStatus("不支持实时连接", "fail");
    return;
  }

  const source = new EventSource("/api/logs/stream");
  source.onopen = () => setLogStreamStatus("实时连接", "ok");
  source.onerror = () => setLogStreamStatus("连接中断", "fail");
  source.onmessage = (message) => {
    try {
      appendLogEvent(JSON.parse(message.data));
    } catch (error) {
      appendLogEvent({
        id: `client-${Date.now()}`,
        timestamp: new Date().toISOString(),
        level: "warn",
        message: "日志解析失败",
        details: { error: error.message },
      });
    }
  };
}

function appendLogEvent(event) {
  if (seenLogIds.has(event.id)) return;
  seenLogIds.add(event.id);

  const row = document.createElement("div");
  row.className = `log-row ${escapeHtml(event.level || "info")}`;
  const details = event.details && Object.keys(event.details).length > 0
    ? JSON.stringify(event.details)
    : "";
  row.innerHTML = `
    <span class="log-time">${escapeHtml(formatLogTime(event.timestamp))}</span>
    <span class="log-level">${escapeHtml(event.level || "info")}</span>
    <span class="log-message">${escapeHtml(event.message || "")}</span>
    <span class="log-details">${escapeHtml(details)}</span>
  `;
  logsEl.append(row);

  while (logsEl.children.length > 300) {
    const first = logsEl.firstElementChild;
    if (first) first.remove();
  }
  logsEl.scrollTop = logsEl.scrollHeight;
}

function formatLogTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString("zh-CN", { hour12: false });
}

function addWordRow(word = { text: "", weight: 4, lang: "zh" }) {
  const row = document.createElement("div");
  row.className = "word-row";
  row.innerHTML = `
    <label>
      <span>热词</span>
      <input class="word-text" value="${escapeHtml(word.text)}" placeholder="佬友" />
    </label>
    <label>
      <span>权重</span>
      <input class="word-weight" value="${word.weight}" type="number" min="1" max="5" />
    </label>
    <label>
      <span>语言</span>
      <input class="word-lang" value="${escapeHtml(word.lang)}" placeholder="zh" />
    </label>
    <button class="icon-button remove-word" type="button" title="删除热词">×</button>
  `;
  row.querySelector(".remove-word").addEventListener("click", () => row.remove());
  wordTable.append(row);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function collectWords() {
  return Array.from(wordTable.querySelectorAll(".word-row"))
    .map((row) => ({
      text: row.querySelector(".word-text").value.trim(),
      weight: Number(row.querySelector(".word-weight").value || 4),
      lang: row.querySelector(".word-lang").value.trim() || "zh",
    }))
    .filter((word) => word.text);
}

async function saveSettings(event) {
  event.preventDefault();
  const button = settingsForm.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const payload = {};
    const hints = parseHints(document.querySelector("#language-hints").value);
    if (hints !== null) {
      payload.language_hints = hints;
    } else {
      payload.language_hints = null;
    }
    const vocabId = document.querySelector("#vocabulary-id").value.trim();
    // Only send vocabulary_id if the user explicitly set it; skip when empty
    // so the server preserves its current value.
    if (vocabId) {
      payload.vocabulary_id = vocabId;
    }
    await requestJson("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showOutput(`默认设置已保存。${vocabId ? `热词表: ${vocabId}` : "热词表未设置，使用已有默认值。"}`);
  } catch (error) {
    showOutput(`保存失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function createVocabulary(event) {
  event.preventDefault();
  const button = vocabularyForm.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const vocabulary = collectWords();
    if (vocabulary.length === 0) {
      throw new Error("请至少填写一个热词");
    }
    const result = await requestJson("/v1/audio/vocabularies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prefix: document.querySelector("#vocabulary-prefix").value.trim() || "custom",
        target_model: document.querySelector("#target-model").value,
        vocabulary,
      }),
    });
    const vocabId = result.vocabulary_id || "";
    document.querySelector("#vocabulary-id").value = vocabId;
    document.querySelector("#transcription-vocabulary-id").value = vocabId;
    // Auto-save as default setting
    if (vocabId) {
      try {
        await requestJson("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ vocabulary_id: vocabId }),
        });
      } catch { /* non-blocking */ }
    }
    showOutput({ vocabulary_id: vocabId, status: "created", message: "热词表已创建并设为默认" });
    // Refresh the vocabulary list and auto-check status
    await loadVocabularies();
    if (vocabId) {
      checkVocabularyStatus(vocabId);
    }
  } catch (error) {
    showOutput(`创建失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function loadVocabularies() {
  const listEl = document.querySelector("#vocabulary-list");
  try {
    const result = await requestJson("/api/vocabularies");
    const vocabularies = result.data || [];
    if (vocabularies.length === 0) {
      listEl.innerHTML = '<p class="hint">暂无热词表，创建一个吧。</p>';
      return;
    }
    listEl.innerHTML = "";
    for (const vocab of vocabularies) {
      const id = vocab.vocabulary_id || vocab.id || "";
      const status = vocab.status || "UNKNOWN";
      const statusClass = status === "OK" ? "vocab-status-ok" : "vocab-status-pending";
      const row = document.createElement("div");
      row.className = "vocab-row";
      row.innerHTML = `
        <div class="vocab-info">
          <span class="vocab-id" title="${escapeHtml(id)}">${escapeHtml(id)}</span>
          <span class="vocab-status ${statusClass}">${escapeHtml(status)}</span>
        </div>
        <div class="vocab-actions">
          <button class="icon-button vocab-use-btn" type="button" title="设为默认热词表">✓</button>
          <button class="icon-button vocab-delete-btn" type="button" title="删除热词表">×</button>
        </div>
      `;
      row.querySelector(".vocab-use-btn").addEventListener("click", async () => {
        document.querySelector("#vocabulary-id").value = id;
        document.querySelector("#transcription-vocabulary-id").value = id;
        // Auto-save as default setting
        try {
          await requestJson("/api/settings", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ vocabulary_id: id }),
          });
        } catch { /* non-blocking */ }
        showOutput(`已将 ${id} 设为默认热词表`);
      });
      row.querySelector(".vocab-delete-btn").addEventListener("click", async () => {
        if (!confirm(`确定删除热词表 ${id}？此操作不可恢复。`)) return;
        try {
          await requestJson(`/api/vocabularies/${encodeURIComponent(id)}`, { method: "DELETE" });
          await loadVocabularies();
        } catch (error) {
          showOutput(`删除失败：${error.message}`);
        }
      });
      listEl.append(row);
    }
  } catch (error) {
    listEl.innerHTML = `<p class="hint">加载失败：${escapeHtml(error.message)}</p>`;
  }
}

async function checkVocabularyStatus(vocabularyId) {
  try {
    const result = await requestJson(`/api/vocabularies/${encodeURIComponent(vocabularyId)}`);
    const status = result.status || "UNKNOWN";
    if (status === "OK") {
      showOutput({ vocabulary_id: vocabularyId, status: "OK", message: "热词表已就绪，可以使用" });
    } else {
      showOutput({ vocabulary_id: vocabularyId, status, message: "热词表尚未就绪，请稍后刷新重试" });
    }
    await loadVocabularies();
  } catch (error) {
    showOutput(`查询热词表状态失败：${error.message}`);
  }
}

async function transcribeAudio(event) {
  event.preventDefault();
  const button = transcriptionForm.querySelector("button[type=submit]");
  const file = document.querySelector("#audio-file").files[0];
  if (!file) {
    showOutput("请选择音频文件");
    return;
  }

  button.disabled = true;
  try {
    const formData = new FormData();
    formData.append("model", document.querySelector("#transcription-model").value);
    formData.append("file", file);

    const hints = document.querySelector("#transcription-language-hints").value.trim();
    const vocabularyId = document.querySelector("#transcription-vocabulary-id").value.trim();
    if (hints) formData.append("language_hints", hints);
    if (vocabularyId) formData.append("vocabulary_id", vocabularyId);

    const result = await requestJson("/v1/audio/transcriptions", {
      method: "POST",
      body: formData,
    });
    showOutput(result);
  } catch (error) {
    showOutput(`转写失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function startRecording() {
  const BrowserAudioContext = window.AudioContext || window.webkitAudioContext;
  if (!navigator.mediaDevices || !BrowserAudioContext) {
    showOutput("当前浏览器不支持录音，请使用 Chrome、Edge 或 Safari 的新版本。");
    return;
  }

  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new BrowserAudioContext();
    recordingSource = audioContext.createMediaStreamSource(recordingStream);
    recordingProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    recordingSilence = audioContext.createGain();
    recordingSilence.gain.value = 0;
    recordedSamples = [];
    recordedSampleRate = audioContext.sampleRate;
    recordedBlob = null;
    mediaRecorder = { state: "recording" };

    recordingProcessor.onaudioprocess = (event) => {
      recordedSamples.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    recordingSource.connect(recordingProcessor);
    recordingProcessor.connect(recordingSilence);
    recordingSilence.connect(audioContext.destination);

    startRecordingButton.disabled = true;
    stopRecordingButton.disabled = false;
    transcribeRecordingButton.disabled = true;
    recordingStatusEl.textContent = "录音中";
    showOutput("正在录音，完成后点击停止录音。");
  } catch (error) {
    stopRecordingTracks();
    recordingStatusEl.textContent = "录音失败";
    showOutput(`录音失败：${error.message}`);
  }
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state !== "recording") return;
  mediaRecorder.state = "inactive";
  recordedBlob = encodeWavBlob(mergeSamples(recordedSamples), recordedSampleRate);
  recordedAudioEl.src = URL.createObjectURL(recordedBlob);
  transcribeRecordingButton.disabled = recordedBlob.size === 0;
  stopRecordingTracks();
  stopRecordingAudioNodes();
  recordingStatusEl.textContent = "录音完成";
  showOutput(`录音完成，大小 ${Math.round(recordedBlob.size / 1024)} KB`);
  startRecordingButton.disabled = false;
  stopRecordingButton.disabled = true;
}

function stopRecordingTracks() {
  if (!recordingStream) return;
  for (const track of recordingStream.getTracks()) {
    track.stop();
  }
  recordingStream = null;
}

function stopRecordingAudioNodes() {
  if (recordingProcessor) {
    recordingProcessor.disconnect();
    recordingProcessor.onaudioprocess = null;
    recordingProcessor = null;
  }
  if (recordingSilence) {
    recordingSilence.disconnect();
    recordingSilence = null;
  }
  if (recordingSource) {
    recordingSource.disconnect();
    recordingSource = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
}

function mergeSamples(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const samples = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  return samples;
}

function encodeWavBlob(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

async function transcribeRecording() {
  if (!recordedBlob) {
    showOutput("请先完成一段录音");
    return;
  }

  transcribeRecordingButton.disabled = true;
  try {
    const formData = new FormData();
    formData.append("model", document.querySelector("#transcription-model").value);
    formData.append("file", recordedBlob, "recording.wav");

    const hints = document.querySelector("#transcription-language-hints").value.trim();
    const vocabularyId = document.querySelector("#transcription-vocabulary-id").value.trim();
    if (hints) formData.append("language_hints", hints);
    if (vocabularyId) formData.append("vocabulary_id", vocabularyId);

    const result = await requestJson("/v1/audio/transcriptions", {
      method: "POST",
      body: formData,
    });
    showOutput(result);
  } catch (error) {
    showOutput(`录音转写失败：${error.message}`);
  } finally {
    transcribeRecordingButton.disabled = false;
  }
}

document.querySelector("#reload-settings").addEventListener("click", () => {
  loadSettings().catch((error) => showOutput(`读取失败：${error.message}`));
});
document.querySelector("#refresh-vocabularies").addEventListener("click", () => {
  loadVocabularies().catch((error) => showOutput(`刷新失败：${error.message}`));
});
document.querySelector("#add-word").addEventListener("click", () => addWordRow());
document.querySelector("#clear-output").addEventListener("click", () => showOutput("等待操作"));
clearLogsButton.addEventListener("click", () => {
  logsEl.textContent = "";
  seenLogIds.clear();
});
startRecordingButton.addEventListener("click", startRecording);
stopRecordingButton.addEventListener("click", stopRecording);
transcribeRecordingButton.addEventListener("click", transcribeRecording);
settingsForm.addEventListener("submit", saveSettings);
vocabularyForm.addEventListener("submit", createVocabulary);
transcriptionForm.addEventListener("submit", transcribeAudio);

addWordRow({ text: "佬友", weight: 4, lang: "zh" });
loadHealth();
loadSettings().catch((error) => showOutput(`读取失败：${error.message}`));
loadVocabularies();
loadLogs().catch((error) => appendLogEvent({
  id: `client-${Date.now()}`,
  timestamp: new Date().toISOString(),
  level: "warn",
  message: "读取日志失败",
  details: { error: error.message },
}));
connectLogStream();
