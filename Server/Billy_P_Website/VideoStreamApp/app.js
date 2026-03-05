const preview = document.getElementById("preview");
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const serverUrlInput = document.getElementById("serverUrl");
const resolutionSelect = document.getElementById("resolution");
const fpsSelect = document.getElementById("fps");
const grayscaleToggle = document.getElementById("grayscale");
const qualityInput = document.getElementById("quality");
const viewerUrl = document.getElementById("viewerUrl");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const actualResolutionEl = document.getElementById("actualResolution");
const actualFpsEl = document.getElementById("actualFps");
const streamStatsEl = document.getElementById("streamStats");

if (!serverUrlInput.value || /192\.168\.|10\.|172\./.test(serverUrlInput.value)) {
  serverUrlInput.value = window.location.origin;
}

let mediaStream = null;
let captureTimer = null;
let sendInFlight = false;
let abortController = null;
let sentFrames = 0;
let statsWindowStart = 0;
let uploadLatencyMs = 0;

function setStatus(text) {
  statusEl.textContent = text;
}

function updateViewerUrl() {
  const base = serverUrlInput.value.replace(/\/$/, "");
  viewerUrl.textContent = `${base}/mjpeg`;
}

serverUrlInput.addEventListener("input", updateViewerUrl);
updateViewerUrl();

function parseResolution(value) {
  const [width, height] = value.split("x").map(Number);
  return { width, height };
}

function resetStats() {
  sentFrames = 0;
  statsWindowStart = performance.now();
  uploadLatencyMs = 0;
  streamStatsEl.textContent = "Sent FPS: - | Upload: - ms";
}

function updateActualSettings(track) {
  const settings = track.getSettings();
  const width = settings.width || canvas.width || "-";
  const height = settings.height || canvas.height || "-";
  const fps = settings.frameRate ? Number(settings.frameRate).toFixed(1) : "-";

  actualResolutionEl.textContent = `Resolution: ${width} x ${height}`;
  actualFpsEl.textContent = `Frame rate: ${fps} fps`;
}

function updateStreamStats() {
  if (!statsWindowStart) {
    return;
  }
  const elapsedSeconds = (performance.now() - statsWindowStart) / 1000;
  const measuredFps = elapsedSeconds > 0 ? (sentFrames / elapsedSeconds).toFixed(1) : "0.0";
  const latency = uploadLatencyMs ? uploadLatencyMs.toFixed(0) : "-";
  streamStatsEl.textContent = `Sent FPS: ${measuredFps} | Upload: ${latency} ms`;
}

async function startCapture() {
  const { width, height } = parseResolution(resolutionSelect.value);
  const fps = Number(fpsSelect.value);

  const constraints = {
    video: {
      width: { ideal: width },
      height: { ideal: height },
      frameRate: { ideal: fps }
    },
    audio: false
  };

  mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
  preview.srcObject = mediaStream;

  await new Promise((resolve) => {
    if (preview.readyState >= 1 && preview.videoWidth > 0 && preview.videoHeight > 0) {
      resolve();
      return;
    }
    preview.onloadedmetadata = () => resolve();
  });

  const track = mediaStream.getVideoTracks()[0];
  const settings = track.getSettings();
  canvas.width = preview.videoWidth || settings.width || width;
  canvas.height = preview.videoHeight || settings.height || height;
  updateActualSettings(track);
  resetStats();

  setStatus("Streaming");
  startButton.disabled = true;
  stopButton.disabled = false;

  abortController = new AbortController();

  const interval = Math.max(1000 / fps, 33);
  captureTimer = setInterval(() => {
    sendFrame().catch(() => {
      setStatus("Upload error");
    });
  }, interval);
}

function stopCapture() {
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (abortController) {
    abortController.abort();
    abortController = null;
  }

  sendInFlight = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Idle");
  actualResolutionEl.textContent = "Resolution: -";
  actualFpsEl.textContent = "Frame rate: -";
  streamStatsEl.textContent = "Sent FPS: - | Upload: - ms";
}

function applyGrayscale(ctx, width, height) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const y = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    data[i] = y;
    data[i + 1] = y;
    data[i + 2] = y;
  }
  ctx.putImageData(imageData, 0, 0);
}

async function sendFrame() {
  if (!mediaStream || sendInFlight) {
    return;
  }

  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.drawImage(preview, 0, 0, width, height);

  if (grayscaleToggle.checked) {
    applyGrayscale(ctx, width, height);
  }

  const quality = Number(qualityInput.value);
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", quality)
  );

  if (!blob) {
    return;
  }

  const base = serverUrlInput.value.replace(/\/$/, "");
  const uploadUrl = `${base}/upload`;

  sendInFlight = true;
  const startedAt = performance.now();
  try {
    const response = await fetch(uploadUrl, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
      signal: abortController ? abortController.signal : undefined
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    sentFrames += 1;
    uploadLatencyMs = performance.now() - startedAt;
    updateStreamStats();
  } finally {
    sendInFlight = false;
  }
}

startButton.addEventListener("click", async () => {
  try {
    await startCapture();
  } catch (error) {
    const reason = error && error.name ? error.name : "UnknownError";
    setStatus(`Camera error: ${reason}`);
    console.error(error);
  }
});

stopButton.addEventListener("click", stopCapture);

window.addEventListener("beforeunload", stopCapture);
