const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  width: 1536,
  height: 1024,
  busy: false,
  catalogLoaded: false,
  lastCatalogNote: "",
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
    if (button.id === "refreshCatalog") return;
    if (button.id === "randomSeed") return;
    button.addEventListener("click", (event) => {
      const target = event.currentTarget;
      if (target.classList.contains("mode-tab")) return;
      if (target.matches("#showAdvanced")) return;
      wip(target.dataset.feature || target.textContent.trim());
    });
  });
}

function wireRealButtons() {
  $("#refreshCatalog")?.addEventListener("click", () => loadCatalog({ showToast: true }));
  $("#randomSeed")?.addEventListener("click", () => {
    const seed = Math.floor(Math.random() * 2147483647);
    $("#seed").value = String(seed);
    toast("Seed", `Random seed set to ${seed}`);
  });
  $("#modelSelect")?.addEventListener("change", updateSelectedModelCard);
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

function optionFromModel(model) {
  const option = document.createElement("option");
  option.value = model.id || "";
  option.textContent = model.title || model.id || "Untitled model";
  option.dataset.path = model.path || "";
  option.dataset.hash = model.hash || "";
  option.dataset.raw = JSON.stringify(model.raw || {});
  return option;
}

function optionFromSampler(sampler) {
  const option = document.createElement("option");
  option.value = sampler.id || sampler.title || "euler_a";
  option.textContent = sampler.title || sampler.id || "Sampler";
  return option;
}

function updateSelectedModelCard() {
  const selected = $("#modelSelect")?.selectedOptions?.[0];
  if (!selected) return;
  setText("#loadedModel", selected.textContent);
  const path = selected.dataset.path || "";
  const hash = selected.dataset.hash || "";
  setText("#modelSize", path ? "Runtime catalog" : "WIP");
  setText("#baseModel", inferBaseModel(selected.textContent));
  setText("#modelType", "Text-to-Image");
  if (hash) {
    $("#modelSize").title = `hash: ${hash}`;
  }
}

function inferBaseModel(label) {
  const lower = String(label || "").toLowerCase();
  if (lower.includes("sdxl")) return "SDXL";
  if (lower.includes("flux")) return "Flux";
  if (lower.includes("sd3")) return "SD3";
  if (lower.includes("1.5") || lower.includes("sd15") || lower.includes("sd 1")) return "SD 1.x";
  return "Runtime catalog";
}

async function loadCatalog({ showToast = false } = {}) {
  try {
    const response = await fetch("/api/catalog", { cache: "no-store" });
    const data = await response.json();

    if (Array.isArray(data.models) && data.models.length) {
      const select = $("#modelSelect");
      const current = select.value;
      select.innerHTML = "";
      data.models.forEach((model) => select.appendChild(optionFromModel(model)));
      if (current && Array.from(select.options).some((option) => option.value === current)) {
        select.value = current;
      }
      updateSelectedModelCard();
    }

    if (Array.isArray(data.samplers) && data.samplers.length) {
      const select = $("#sampler");
      const current = select.value;
      select.innerHTML = "";
      data.samplers.forEach((sampler) => select.appendChild(optionFromSampler(sampler)));
      if (current && Array.from(select.options).some((option) => option.value === current)) {
        select.value = current;
      }
    }

    state.catalogLoaded = Boolean(data.ok);
    const modelCount = data.models?.length || 0;
    const samplerCount = data.samplers?.length || 0;
    state.lastCatalogNote = `${modelCount} models, ${samplerCount} samplers`;
    if (showToast) {
      toast(data.ok ? "Catalog refreshed" : "Catalog WIP", data.ok ? state.lastCatalogNote : "Backend catalog route is not reachable yet.", data.ok ? "info" : "warn");
    }
  } catch (error) {
    if (showToast) {
      toast("Catalog failed", `${error.name}: ${error.message}`, "warn");
    }
  }
}

function progressLabel(progress) {
  if (!progress || !progress.ok) return "Idle";
  const stateText = String(progress.state || "idle");
  if (stateText === "idle") return "Idle";
  if (progress.progress > 0) return `${stateText} ${progress.progress}%`;
  return stateText;
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
    setText("#runtimeState", progressLabel(data.progress));
    setText("#queueText", data.queue_text || "0 tasks");
    const status = data.backend_reachable
      ? `Backend reachable at ${data.backend_url}. Proxy ${data.proxy_enabled ? `${data.proxy_mode} enabled` : "off"}.`
      : `Second GUI shell is live. Main backend check: ${data.backend_note}.`;
    if (!state.busy) {
      setText("#previewStatus", status);
    }
  } catch (error) {
    setText("#engineState", "WIP bridge");
    setText("#runtimeState", "Offline");
    if (!state.busy) {
      setText("#previewStatus", "Preview shell is running without runtime status.");
    }
  }
}

function generationPayload() {
  const modelSelect = $("#modelSelect");
  const selectedModel = modelSelect?.selectedOptions?.[0];
  const samplerSelect = $("#sampler");
  const selectedSampler = samplerSelect?.selectedOptions?.[0];
  return {
    prompt: $("#prompt")?.value || "",
    negative_prompt: $("#negativePrompt")?.value || "",
    model: selectedModel?.textContent || "",
    checkpoint_id: modelSelect?.value || null,
    sampler: samplerSelect?.value || "dpmpp_2m",
    sampler_label: selectedSampler?.textContent || samplerSelect?.value || "DPM++ 2M Karras",
    steps: Number($("#steps")?.value || 30),
    cfg_scale: Number($("#cfg")?.value || 7),
    seed: Number($("#seed")?.value || -1),
    width: state.width,
    height: state.height,
    batch_size: 1,
    batch_count: 1,
  };
}

async function generate() {
  if (state.busy) return;
  state.busy = true;
  const button = $("#generateBtn");
  const original = button.textContent;
  button.textContent = "Generating / checking route…";
  $("#queueText").textContent = "1 task";
  setText("#runtimeState", "sending");
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
      toast("Generated", `Image returned from ${data.source} route.`);
      setText("#previewStatus", `Image returned from AIWF backend ${data.source} route.`);
    } else if (data.wip) {
      addWipThumb();
      const details = Array.isArray(data.errors) && data.errors.length ? ` ${data.errors[0]}` : "";
      toast(data.feature || "WIP", `${data.message || "This route is not wired yet."}${details}`, "warn");
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
    loadStatus();
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
  wireRealButtons();
  wireWipButtons();
  wireGenerate();
  loadCatalog();
  loadStatus();
  window.setInterval(loadStatus, 4000);
  window.setInterval(() => loadCatalog(), 30000);
}

boot();
