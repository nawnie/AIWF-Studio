const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  width: 1536,
  height: 1024,
  busy: false,
};

function toast(title, message, tone = "info") {
  const zone = $("#toastZone");
  const node = document.createElement("div");
  node.className = `toast ${tone === "warn" ? "warn" : ""}`;
  node.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span>`;
  zone.appendChild(node);
  window.setTimeout(() => node.remove(), 5200);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setText(id, value) {
  const node = $(id);
  if (node) node.textContent = value || "WIP";
}

function countTextarea(textarea) {
  const count = $(`[data-count-for="${textarea.id}"]`);
  if (count) count.textContent = `${textarea.value.length} / ${textarea.maxLength}`;
}

function wireTextCounts() {
  $$("textarea[maxlength]").forEach((textarea) => {
    countTextarea(textarea);
    textarea.addEventListener("input", () => countTextarea(textarea));
  });
}

function wireSliders() {
  $$("input[type='range']").forEach((range) => {
    const out = range.parentElement.querySelector("output");
    const render = () => {
      const value = range.step && range.step.includes(".") ? Number(range.value).toFixed(1) : range.value;
      if (out) out.value = value;
    };
    render();
    range.addEventListener("input", render);
  });
}

function wireRatios() {
  $$("#ratioRow button[data-width]").forEach((button) => {
    button.addEventListener("click", () => {
      $$("#ratioRow button").forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");
      state.width = Number(button.dataset.width);
      state.height = Number(button.dataset.height);
      toast("Aspect ratio", `${button.textContent.trim()} selected: ${state.width}×${state.height}`);
    });
  });
}

function wireWelcome() {
  const modal = $("#welcomeModal");
  const close = () => modal.classList.add("hidden");
  $("#closeWelcome")?.addEventListener("click", close);
  $("#getStarted")?.addEventListener("click", close);
  $("#showAdvanced")?.addEventListener("change", (event) => {
    document.body.classList.toggle("advanced-visible", event.target.checked);
  });
}

function wireAdvancedPanel() {
  $("#advancedToggle")?.addEventListener("click", () => {
    $("#advancedBox")?.classList.toggle("open");
  });
}

function wireTabs() {
  $$(".mode-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".mode-tab").forEach((node) => node.classList.remove("active"));
      tab.classList.add("active");
      if (tab.dataset.mode !== "image") {
        wip(tab.dataset.feature || tab.textContent.trim());
      }
    });
  });
}

function wireWipButtons() {
  $$('[data-feature]').forEach((button) => {
    if (button.id === "generateBtn") return;
    button.addEventListener("click", (event) => {
      const target = event.currentTarget;
      if (target.classList.contains("mode-tab") && target.dataset.mode === "image") return;
      if (target.matches("#showAdvanced")) return;
      wip(target.dataset.feature || target.textContent.trim());
    });
  });
}

async function wip(feature) {
  try {
    await fetch("/api/wip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature }),
    });
  } catch (_) {
    // The static page can still show WIP without the bridge.
  }
  toast("WIP", `${feature} is a placeholder until the backend route is wired.`, "warn");
}

async function loadStatus() {
  try {
    const response = await fetch("/api/runtime/status", { cache: "no-store" });
    const data = await response.json();
    setText("#engineState", data.engine_state);
    setText("#backendName", data.backend);
    setText("#deviceName", data.device);
    setText("#precisionName", data.precision);
    setText("#attentionName", data.attention);
    setText("#maxRes", data.max_resolution);
    setText("#vramText", data.vram);
    setText("#ramText", data.ram);
    setText("#storageText", data.storage);
    setText("#cpuText", data.cpu);
    setText("#loadedModel", data.loaded_model);
    setText("#topVram", data.vram);
    const status = data.backend_reachable
      ? `Backend reachable at ${data.backend_url}. Generate proxy ${data.proxy_enabled ? "enabled" : "off"}.`
      : `Second GUI shell is live. Main backend check: ${data.backend_note}.`;
    setText("#previewStatus", status);
  } catch (error) {
    setText("#engineState", "WIP bridge");
    setText("#previewStatus", "Preview shell is running without runtime status.");
  }
}

function generationPayload() {
  return {
    prompt: $("#prompt")?.value || "",
    negative_prompt: $("#negativePrompt")?.value || "",
    model: $("#modelSelect")?.value || "",
    sampler: $("#sampler")?.value || "DPM++ 2M Karras",
    steps: Number($("#steps")?.value || 30),
    cfg_scale: Number($("#cfg")?.value || 7),
    seed: Number($("#seed")?.value || -1),
    width: state.width,
    height: state.height,
  };
}

async function generate() {
  if (state.busy) return;
  state.busy = true;
  const button = $("#generateBtn");
  const original = button.textContent;
  button.textContent = "Generating / checking route…";
  $("#queueText").textContent = "1 task";
  setText("#previewStatus", "Sending request to Second GUI bridge…");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(generationPayload()),
    });
    const data = await response.json();
    if (data.ok && data.response?.images?.length) {
      const image = data.response.images[0];
      showImage(image);
      addRecentImage(image, `${state.width}×${state.height}`);
      toast("Generated", "Image returned from backend proxy.");
      setText("#previewStatus", "Image returned from AIWF backend proxy.");
    } else if (data.wip) {
      addWipThumb();
      toast(data.feature || "WIP", data.message || "This route is not wired yet.", "warn");
      setText("#previewStatus", data.message || "Generate is a WIP link in this shell.");
    } else {
      toast("Route response", "Backend responded, but no image payload was found.", "warn");
      setText("#previewStatus", "Backend responded without an image payload.");
    }
  } catch (error) {
    toast("Generate failed", `${error.name}: ${error.message}`, "warn");
    setText("#previewStatus", "Generate failed before reaching a backend route.");
  } finally {
    state.busy = false;
    button.textContent = original;
    $("#queueText").textContent = "0 tasks";
  }
}

function showImage(base64Image) {
  const stage = $("#previewStage");
  stage.innerHTML = "";
  const img = document.createElement("img");
  img.className = "generated-image";
  img.alt = "Generated output";
  img.src = base64Image.startsWith("data:") ? base64Image : `data:image/png;base64,${base64Image}`;
  img.style.width = "100%";
  img.style.height = "100%";
  img.style.objectFit = "contain";
  img.style.borderRadius = "12px";
  stage.appendChild(img);
}

function addRecentImage(base64Image, label) {
  const row = $("#thumbRow");
  const button = document.createElement("button");
  button.className = "thumb active";
  button.style.backgroundImage = `url(${base64Image.startsWith("data:") ? base64Image : `data:image/png;base64,${base64Image}`})`;
  button.style.backgroundSize = "cover";
  button.style.backgroundPosition = "center";
  button.innerHTML = `<span>${escapeHtml(label)}</span><small>now</small>`;
  row.prepend(button);
  $$(".thumb", row).slice(8).forEach((node) => node.remove());
}

function addWipThumb() {
  const row = $("#thumbRow");
  const button = document.createElement("button");
  button.className = "thumb city active";
  button.innerHTML = `<span>${state.width}×${state.height}</span><small>WIP</small>`;
  row.prepend(button);
  $$(".thumb", row).slice(8).forEach((node) => node.remove());
}

function wireGenerate() {
  $("#generateBtn")?.addEventListener("click", generate);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      generate();
    }
  });
}

function boot() {
  document.body.classList.remove("advanced-visible");
  wireTextCounts();
  wireSliders();
  wireRatios();
  wireWelcome();
  wireAdvancedPanel();
  wireTabs();
  wireWipButtons();
  wireGenerate();
  loadStatus();
  window.setInterval(loadStatus, 10000);
}

boot();
