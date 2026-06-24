(function () {
    const STUDIO = ".aiwf-studio";
    const APP = ".aiwf-app";
    const PROMPT_ID = "aiwf-prompt";
    const GENERATE_ID = "aiwf-generate";
    const SEND_TO_VIDEO_ID = "aiwf-send-to-video";
    const TOPBAR_STATUS_ID = "aiwf-topbar-status";
    const CLIENT_ERROR_TRAY_ID = "aiwf-client-error-tray";
    const CLIENT_ERROR_STORE_KEY = "aiwf-client-errors";
    const CLIENT_EVENT_STORE_KEY = "aiwf-client-events";
    const DEV_SESSION_KEY = "aiwf-dev-session";
    const CLIENT_ERROR_MAX = 20;
    const CLIENT_EVENT_MAX = 80;
    const ACTION_RING_MAX = 12;
    const HANDHELD_QUERY = "(max-width: 900px), (hover: none) and (pointer: coarse) and (max-width: 1100px)";
    const INITIAL_TITLE = document.title || "AIWF Studio";

    const BUSY_RE = /\*\*(generating|working|loading|running|stepping|step\s*\d|processing|saving|queued)\*\*/i;
    const DONE_RE = /\*\*(done|complete|saved|loaded|finished)\*\*/i;
    const ERROR_RE = /\*\*(error|failed|cancelled|canceled|stopped)\*\*/i;

    function promptTextarea() {
        const byId = document.getElementById(PROMPT_ID);
        if (byId) {
            if (byId.tagName === "TEXTAREA") {
                return byId;
            }
            const nested = byId.querySelector("textarea");
            if (nested) {
                return nested;
            }
        }
        return document.querySelector(`${STUDIO} .aiwf-prompt-input textarea`);
    }

    function generateButton() {
        const byId = document.getElementById(GENERATE_ID);
        if (byId) {
            if (byId.tagName === "BUTTON") {
                return byId;
            }
            const nested = byId.querySelector("button");
            if (nested) {
                return nested;
            }
        }
        return (
            document.querySelector(`${STUDIO} button.aiwf-generate-btn`) ||
            document.querySelector(`${STUDIO} .aiwf-generate-btn button`)
        );
    }

    function isPromptTarget(target) {
        const prompt = promptTextarea();
        if (!prompt) {
            return false;
        }
        return target === prompt || prompt.contains(target);
    }

    function triggerGenerate() {
        const button = generateButton();
        if (!button || button.disabled) {
            return false;
        }
        button.click();
        return true;
    }

    function onPromptKeydown(event) {
        const isEnter = event.key === "Enter" || event.code === "NumpadEnter";
        if (!isEnter || !(event.shiftKey || event.ctrlKey) || event.repeat) {
            return;
        }
        if (!isPromptTarget(event.target)) {
            return;
        }
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        triggerGenerate();
    }

    function markUiReady() {
        const app = document.querySelector(APP);
        if (app && !app.classList.contains("aiwf-ui-ready")) {
            app.classList.add("aiwf-ui-ready");
        }
    }

    function syncDeviceClasses() {
        const app = document.querySelector(APP);
        if (!app) {
            return;
        }
        const handheld = window.matchMedia(HANDHELD_QUERY).matches;
        const touch = window.matchMedia("(hover: none) and (pointer: coarse)").matches;
        app.classList.toggle("aiwf-handheld", handheld);
        app.classList.toggle("aiwf-touch", touch);
    }

    function watchStudioTab() {
        const app = document.querySelector(APP);
        if (!app) {
            return;
        }
        const selected = document.querySelector(".aiwf-nav-tabs .tab-nav button.selected");
        const onStudio = selected && /studio/i.test(selected.textContent || "");
        app.classList.toggle("aiwf-studio-tab", Boolean(onStudio));
    }

    let lastVideoTabSwitchAt = 0;

    function tabButtons() {
        return [
            ...document.querySelectorAll(
                '.aiwf-nav-tabs [role="tab"], .aiwf-nav-tabs .tab-nav button, .aiwf-nav-tabs button, [role="tab"]'
            ),
        ];
    }

    function switchToTab(label) {
        const wanted = String(label || "").trim().toLowerCase();
        const button = tabButtons().find((candidate) => {
            const text = (candidate.textContent || "").trim().toLowerCase();
            return text === wanted || text.startsWith(wanted);
        });
        if (!button) {
            reportClientEvent("tab_switch_missing", wanted, {
                tabs: tabButtons().map((candidate) => (candidate.textContent || "").trim()).filter(Boolean),
            });
            return false;
        }
        button.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
        requestAnimationFrame(() => {
            watchStudioTab();
            syncTopbarFromContext();
        });
        return true;
    }

    function maybeSwitchToVideoFromStatus(text) {
        if (!/\*\*sent to video\*\*|sent to video/i.test(text || "")) {
            return;
        }
        const now = Date.now();
        if (now - lastVideoTabSwitchAt < 1200) {
            return;
        }
        lastVideoTabSwitchAt = now;
        setTimeout(() => switchToTab("Video"), 50);
    }

    function switchToVideoSoon() {
        // Gradio may update tabs and callback outputs in separate DOM turns.
        // Try once for immediate feedback, then retry after the server callback
        // has had a moment to store the pending image for the Video tab.
        setTimeout(() => switchToTab("Video"), 80);
        setTimeout(() => switchToTab("Video"), 450);
        setTimeout(() => switchToTab("Video"), 900);
    }

    function activeTabPanel() {
        const tabs = document.querySelectorAll(".aiwf-nav-tabs > .tabitem");
        for (const panel of tabs) {
            if (panel.classList.contains("hidden") || panel.style.display === "none") {
                continue;
            }
            if (panel.offsetParent !== null) {
                return panel;
            }
        }
        return document.querySelector(".aiwf-nav-tabs > .tabitem.selected") || null;
    }

    function statusTextFromBar(bar) {
        if (!bar) {
            return "";
        }
        return (bar.textContent || "").trim();
    }

    function inferStatusState(text) {
        if (!text) {
            return "ready";
        }
        if (ERROR_RE.test(text)) {
            return "error";
        }
        if (BUSY_RE.test(text) || /step\s+\d+\s*\/\s*\d+/i.test(text)) {
            return "busy";
        }
        if (DONE_RE.test(text)) {
            return "done";
        }
        if (/^\*\*ready\*\*/i.test(text) || /^load an image/i.test(text)) {
            return "ready";
        }
        return "ready";
    }

    function shortLabelForState(state, text) {
        if (state === "busy") {
            const step = text.match(/step\s*(\d+)\s*\/\s*(\d+)/i);
            if (step) {
                return `Step ${step[1]}/${step[2]}`;
            }
            return "Working";
        }
        if (state === "done") {
            return "Done";
        }
        if (state === "error") {
            return "Error";
        }
        return "Ready";
    }

    function setTopbarStatus(state, label) {
        const pill = document.getElementById(TOPBAR_STATUS_ID);
        if (!pill) {
            return;
        }
        if (pill.dataset.state !== state) {
            pill.dataset.state = state;
        }
        const labelEl = pill.querySelector(".aiwf-status-label");
        if (labelEl && labelEl.textContent !== label) {
            labelEl.textContent = label;
        }
    }

    function titleProgressEnabled() {
        const settings = document.getElementById("aiwf-client-settings");
        if (!settings) {
            return true;
        }
        return settings.dataset.titleProgress !== "false";
    }

    function syncDocumentTitle(state, text) {
        if (!titleProgressEnabled()) {
            if (document.title !== INITIAL_TITLE) {
                document.title = INITIAL_TITLE;
            }
            return;
        }
        if (state !== "busy") {
            if (document.title !== INITIAL_TITLE) {
                document.title = INITIAL_TITLE;
            }
            return;
        }
        const progress = parseStepProgress(text);
        const label = progress
            ? `Step ${progress.step}/${progress.total}`
            : shortLabelForState(state, text);
        const wanted = `${label} - AIWF Studio`;
        if (document.title !== wanted) {
            document.title = wanted;
        }
    }

    function isAppBusy() {
        const app = document.querySelector(APP);
        if (!app) {
            return false;
        }
        if (app.querySelector(".progress-bar, .generating, [data-testid='block-progress']")) {
            return true;
        }
        const genBtn = generateButton();
        if (genBtn && genBtn.disabled && genBtn.classList.contains("pending")) {
            return true;
        }
        const primaryBusy = app.querySelector(
            ".aiwf-generate-btn button:disabled, button.primary:disabled, button[variant='primary']:disabled"
        );
        return Boolean(primaryBusy);
    }

    let stepProgressStartedAt = 0;

    function parseStepProgress(text) {
        const match = (text || "").match(/step\s*(\d+)\s*\/\s*(\d+)/i);
        if (!match) {
            return null;
        }
        const step = parseInt(match[1], 10);
        const total = parseInt(match[2], 10);
        if (!total || total < 1) {
            return null;
        }
        return { step, total, ratio: Math.min(1, Math.max(0, step / total)) };
    }

    function formatElapsed(ms) {
        const seconds = Math.max(0, Math.floor(ms / 1000));
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        if (mins > 0) {
            return `${mins}:${String(secs).padStart(2, "0")}`;
        }
        return `${secs}s`;
    }

    function syncStepProgressBar(text, state) {
        const wrap = document.getElementById("aiwf-progress-wrap");
        const fill = document.getElementById("aiwf-progress-fill");
        const stepEl = document.getElementById("aiwf-progress-step");
        const elapsedEl = document.getElementById("aiwf-progress-elapsed");
        if (!wrap || !fill || !stepEl || !elapsedEl) {
            return;
        }

        const progress = parseStepProgress(text);
        const busy = state === "busy" && progress;
        if (!busy) {
            wrap.hidden = true;
            stepProgressStartedAt = 0;
            fill.style.width = "0%";
            stepEl.textContent = "";
            elapsedEl.textContent = "";
            return;
        }

        if (!stepProgressStartedAt) {
            stepProgressStartedAt = Date.now();
        }
        wrap.hidden = false;
        fill.style.width = `${Math.round(progress.ratio * 100)}%`;
        stepEl.textContent = `Step ${progress.step}/${progress.total}`;
        const elapsed = Date.now() - stepProgressStartedAt;
        const etaMs =
            progress.step > 0
                ? (elapsed / progress.step) * (progress.total - progress.step)
                : 0;
        elapsedEl.textContent = `${formatElapsed(elapsed)} · ETA ${formatElapsed(etaMs)}`;
    }

    function syncStatusBars() {
        document.querySelectorAll(".aiwf-status-bar").forEach((bar) => {
            const text = statusTextFromBar(bar);
            const state = inferStatusState(text);
            bar.dataset.state = state;
            maybeSwitchToVideoFromStatus(text);
            syncStepProgressBar(text, state);
        });
    }

    let lastObservedState = "ready";

    function showToast(message, state) {
        let host = document.getElementById("aiwf-toasts");
        if (!host) {
            host = document.createElement("div");
            host.id = "aiwf-toasts";
            document.body.appendChild(host);
        }
        const toast = document.createElement("div");
        toast.className = "aiwf-toast";
        toast.dataset.state = state || "info";
        toast.textContent = message;
        host.appendChild(toast);
        requestAnimationFrame(() => toast.classList.add("aiwf-toast-show"));
        window.setTimeout(() => {
            toast.classList.remove("aiwf-toast-show");
            window.setTimeout(() => toast.remove(), 350);
        }, 4200);
        while (host.children.length > 4) {
            host.removeChild(host.firstChild);
        }
    }

    function syncGenerateButton(state) {
        const button = generateButton();
        if (!button) {
            return;
        }
        if (!button.dataset.aiwfLabel) {
            button.dataset.aiwfLabel = (button.textContent || "Generate").trim() || "Generate";
        }
        const busy = state === "busy";
        const wanted = busy ? "Generating…  Esc stops" : button.dataset.aiwfLabel;
        if ((button.textContent || "").trim() !== wanted) {
            button.textContent = wanted;
        }
        button.classList.toggle("aiwf-btn-busy", busy);
    }

    function toastForCompletion(state, text) {
        if (state === "error") {
            showToast("Generation failed — see status for details", "error");
            return;
        }
        const seconds = text.match(/in\s+([\d.]+)s/i);
        if (/stopp?ed|cancell?ed/i.test(text)) {
            showToast("Generation stopped", "warning");
        } else {
            showToast(seconds ? `Done in ${seconds[1]}s` : "Done", "done");
        }
    }

    function syncTopbarFromContext() {
        syncStatusBars();

        const panel = activeTabPanel();
        const statusBar = panel ? panel.querySelector(".aiwf-status-bar") : null;
        const text = statusTextFromBar(statusBar);
        let state = inferStatusState(text);

        if (isAppBusy()) {
            state = "busy";
        }

        const label = shortLabelForState(state, text);
        setTopbarStatus(state, label);
        syncDocumentTitle(state, text);
        syncGenerateButton(state);

        if (lastObservedState === "busy" && state !== "busy") {
            toastForCompletion(state, text);
        }
        lastObservedState = state;

        const app = document.querySelector(APP);
        if (app) {
            app.classList.toggle("aiwf-busy", state === "busy" || isAppBusy());
        }
    }

    function bindPromptHotkey() {
        const prompt = promptTextarea();
        if (!prompt || prompt.dataset.aiwfShiftEnterBound === "1") {
            return Boolean(prompt);
        }
        prompt.dataset.aiwfShiftEnterBound = "1";
        prompt.addEventListener("keydown", onPromptKeydown, true);
        return true;
    }

    function initDeviceWatchers() {
        syncDeviceClasses();
        watchStudioTab();
        window.addEventListener("resize", syncDeviceClasses, { passive: true });
        window.matchMedia(HANDHELD_QUERY).addEventListener("change", syncDeviceClasses);

        const tabs = document.querySelector(".aiwf-nav-tabs .tab-nav");
        if (tabs) {
            tabs.addEventListener(
                "click",
                () => {
                    requestAnimationFrame(() => {
                        watchStudioTab();
                        syncTopbarFromContext();
                    });
                },
                { passive: true }
            );
        }
    }

    function stopButton() {
        return document.querySelector(`${STUDIO} .aiwf-btn-stop button, ${STUDIO} button.aiwf-btn-stop`);
    }

    function triggerStop() {
        const button = stopButton();
        if (!button || button.disabled) {
            return false;
        }
        button.click();
        return true;
    }

    function onEscapeKeydown(event) {
        if (event.key !== "Escape" || event.repeat) {
            return;
        }
        if (!isAppBusy()) {
            return;
        }
        const target = event.target;
        if (target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT" || target.isContentEditable)) {
            return;
        }
        event.preventDefault();
        triggerStop();
    }

    function initHotkeys() {
        document.addEventListener("keydown", onPromptKeydown, true);
        document.addEventListener("keydown", onEscapeKeydown, true);
        bindPromptHotkey();

        const root = document.querySelector(APP) || document.body;
        let pendingSync = false;
        let lastTopbarSyncAt = 0;
        const observer = new MutationObserver(() => {
            if (pendingSync) {
                return;
            }
            pendingSync = true;
            requestAnimationFrame(() => {
                pendingSync = false;
                bindPromptHotkey();
                // Gradio mutates the DOM many times per second during a job.
                // Skip work while the tab is hidden and coalesce bursts so the
                // DOM-heavy topbar sync doesn't run on every animation frame.
                if (document.hidden) {
                    return;
                }
                const now = Date.now();
                if (now - lastTopbarSyncAt < 250) {
                    return;
                }
                lastTopbarSyncAt = now;
                syncTopbarFromContext();
            });
        });
        observer.observe(root, {
            childList: true,
            subtree: true,
        });
    }

    function devSessionId() {
        try {
            let id = sessionStorage.getItem(DEV_SESSION_KEY);
            if (!id) {
                id = `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
                sessionStorage.setItem(DEV_SESSION_KEY, id);
            }
            return id;
        } catch (_err) {
            return "sess-unknown";
        }
    }

    const actionRing = [];

    function recordAction(name, detail) {
        actionRing.push({
            time: new Date().toISOString(),
            name,
            detail: detail || "",
        });
        if (actionRing.length > ACTION_RING_MAX) {
            actionRing.shift();
        }
    }

    function studioContext() {
        const app = document.querySelector(APP);
        const panel = activeTabPanel();
        const modeBtn = document.querySelector(`${STUDIO} .aiwf-mode-toggle button.selected`);
        const statusBar = panel ? panel.querySelector(".aiwf-status-bar") : null;
        const genBtn = generateButton();
        return {
            session_id: devSessionId(),
            tab: panel ? (panel.id || "studio") : "unknown",
            mode: modeBtn ? (modeBtn.textContent || "").trim() : "",
            status: statusBar ? statusTextFromBar(statusBar) : "",
            busy: isAppBusy(),
            generate_disabled: Boolean(genBtn && genBtn.disabled),
            url: window.location.href,
            recent_actions: actionRing.slice(-ACTION_RING_MAX),
        };
    }

    function clientErrorTray() {
        return document.getElementById(CLIENT_ERROR_TRAY_ID);
    }

    function formatClientError(entry) {
        const lines = [
            `[${entry.time}] ${entry.kind}: ${entry.message}`,
            entry.source ? `source: ${entry.source}` : "",
            entry.url ? `page: ${entry.url}` : "",
            entry.stack || "",
        ].filter(Boolean);
        return lines.join("\n");
    }

    function loadClientErrors() {
        try {
            const raw = sessionStorage.getItem(CLIENT_ERROR_STORE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (_err) {
            return [];
        }
    }

    function saveClientErrors(entries) {
        try {
            sessionStorage.setItem(CLIENT_ERROR_STORE_KEY, JSON.stringify(entries.slice(-CLIENT_ERROR_MAX)));
        } catch (_err) {
            /* ignore quota */
        }
    }

    function showClientErrorTray(entry) {
        const tray = clientErrorTray();
        if (!tray) {
            return;
        }
        const text = tray.querySelector(".aiwf-client-error-text");
        if (text) {
            text.textContent = entry.message;
            text.title = formatClientError(entry);
        }
        tray.hidden = false;
        setTopbarStatus("error", "Browser error");
    }

    function loadClientEvents() {
        try {
            const raw = sessionStorage.getItem(CLIENT_EVENT_STORE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (_err) {
            return [];
        }
    }

    function saveClientEvents(entries) {
        try {
            sessionStorage.setItem(CLIENT_EVENT_STORE_KEY, JSON.stringify(entries.slice(-CLIENT_EVENT_MAX)));
        } catch (_err) {
            /* ignore quota */
        }
    }

    function postDevJson(path, payload) {
        // Diagnostics are best-effort: never let dev telemetry interrupt the
        // user's active Gradio job or page navigation.
        const body = JSON.stringify(payload);
        try {
            if (navigator.sendBeacon) {
                const blob = new Blob([body], { type: "application/json" });
                navigator.sendBeacon(path, blob);
                return;
            }
        } catch (_err) {
            /* fall through */
        }
        fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
            keepalive: true,
        }).catch(() => {});
    }

    function reportClientEvent(action, detail, extraContext) {
        const entry = {
            time: new Date().toISOString(),
            action,
            detail: detail || "",
            context: { ...studioContext(), ...(extraContext || {}) },
        };
        const entries = loadClientEvents();
        entries.push(entry);
        saveClientEvents(entries);
        postDevJson("/api/v1/client-events", {
            action: entry.action,
            detail: entry.detail,
            url: entry.context.url,
            session_id: entry.context.session_id,
            context: entry.context,
        });
    }

    function reportClientError(payload) {
        const context = { ...studioContext(), ...(payload.context || {}) };
        const entry = {
            time: new Date().toISOString(),
            kind: payload.kind || "error",
            message: String(payload.message || "Unknown browser error"),
            stack: payload.stack || "",
            source: payload.source || "",
            url: payload.url || window.location.href,
            context,
        };
        const entries = loadClientErrors();
        entries.push(entry);
        saveClientErrors(entries);
        showClientErrorTray(entry);
        reportClientEvent("client_error_reported", entry.message, { kind: entry.kind, source: entry.source });

        const body = JSON.stringify({
            kind: entry.kind,
            message: entry.message,
            stack: entry.stack,
            source: entry.source,
            url: entry.url,
            user_agent: navigator.userAgent,
            session_id: context.session_id,
            context,
        });

        postDevJson("/api/v1/client-errors", JSON.parse(body));
    }

    function initClientErrorHooks() {
        // Keep browser-side failures user-visible even when the Python callback
        // never receives the request.
        window.addEventListener(
            "error",
            (event) => {
                if (!event || !event.message) {
                    return;
                }
                reportClientError({
                    kind: "error",
                    message: event.message,
                    stack: event.error && event.error.stack ? event.error.stack : "",
                    source: event.filename ? `${event.filename}:${event.lineno || 0}` : "",
                });
            },
            true
        );

        window.addEventListener("unhandledrejection", (event) => {
            const reason = event && event.reason ? event.reason : "Unhandled promise rejection";
            const message = reason && reason.message ? reason.message : String(reason);
            const stack = reason && reason.stack ? reason.stack : "";
            reportClientError({
                kind: "unhandledrejection",
                message,
                stack,
                source: "promise",
            });
        });

        const nativeFetch = window.fetch.bind(window);
        window.fetch = async function patchedFetch(input, init) {
            const url = typeof input === "string" ? input : input && input.url ? input.url : "";
            const isGradio = /gradio_api|\/queue\/|\/call\//i.test(url);
            const method = (init && init.method) || "GET";
            if (isGradio) {
                recordAction("fetch_start", `${method} ${url}`);
            }
            let response;
            try {
                response = await nativeFetch(input, init);
            } catch (err) {
                if (isGradio) {
                    const message = err && err.message ? err.message : String(err);
                    reportClientError({
                        kind: "fetch_network",
                        message: `Gradio fetch failed: ${message}`,
                        source: url,
                    });
                }
                throw err;
            }
            if (isGradio) {
                recordAction("fetch_done", `${response.status} ${url}`);
                if (!response.ok) {
                    let bodySnippet = "";
                    try {
                        const clone = response.clone();
                        bodySnippet = (await clone.text()).slice(0, 600);
                    } catch (_err) {
                        bodySnippet = "";
                    }
                    reportClientError({
                        kind: "fetch",
                        message: `Gradio request failed (${response.status}) ${url}`,
                        source: url,
                        context: { method, body_snippet: bodySnippet },
                    });
                }
                // NOTE: successful /queue/ and /call/ requests are intentionally
                // NOT reported as telemetry. Gradio polls these many times per job,
                // and reporting each one ran DOM queries + a sessionStorage
                // re-serialize + an extra POST (server disk write) per request —
                // the bulk of client-event traffic for zero user value. Errors
                // above are still captured.
            }
            return response;
        };

        const nativeConsoleError = console.error.bind(console);
        // Gradio sometimes reports callback failures through console.error
        // after the fetch has already resolved; capture that path too.
        console.error = function patchedConsoleError(...args) {
            const text = args
                .map((arg) => {
                    if (arg && arg.message) {
                        return arg.message;
                    }
                    return String(arg);
                })
                .join(" ");
            if (/gradio|get_data|Blocks-/i.test(text)) {
                reportClientError({
                    kind: "console.error",
                    message: text.slice(0, 2000),
                    source: "console.error",
                });
            }
            nativeConsoleError(...args);
        };

        document.addEventListener("click", (event) => {
            const target = event.target;
            if (!(target instanceof HTMLElement)) {
                return;
            }
            const button = target.closest(".aiwf-client-error-copy");
            if (button) {
                const entries = loadClientErrors();
                const events = loadClientEvents();
                const text = [
                    entries.map(formatClientError).join("\n\n---\n\n"),
                    events.length ? "\n\n=== DEV EVENTS ===\n\n" : "",
                    events.map((e) => `[${e.time}] ${e.action}: ${e.detail}`).join("\n"),
                ].join("");
                if (navigator.clipboard && text) {
                    navigator.clipboard.writeText(text).catch(() => {});
                }
                return;
            }
            const generate = target.closest("#aiwf-generate button, .aiwf-generate-btn button");
            if (generate) {
                recordAction("generate_click", studioContext().mode);
                reportClientEvent("generate_click", studioContext().mode, studioContext());
                return;
            }
            const video = target.closest(`#${SEND_TO_VIDEO_ID} button, button#${SEND_TO_VIDEO_ID}`);
            if (video) {
                recordAction("send_to_video_click", "");
                reportClientEvent("send_to_video_click", "", studioContext());
                switchToVideoSoon();
                return;
            }
            const stop = target.closest(".aiwf-btn-stop button, button.aiwf-btn-stop");
            if (stop) {
                recordAction("stop_click", "");
                reportClientEvent("stop_click", "", studioContext());
                return;
            }
            const modeToggle = target.closest(".aiwf-mode-toggle button");
            if (modeToggle) {
                recordAction("mode_click", (modeToggle.textContent || "").trim());
                reportClientEvent("mode_change", (modeToggle.textContent || "").trim(), studioContext());
            }
        }, true);

        window.addEventListener("visibilitychange", () => {
            reportClientEvent("visibility", document.visibilityState, studioContext());
        });
        window.addEventListener("pagehide", () => {
            reportClientEvent("pagehide", "", studioContext());
        });
        window.addEventListener("online", () => reportClientEvent("network", "online", studioContext()));
        window.addEventListener("offline", () => reportClientEvent("network", "offline", studioContext()));
    }

    function initGenerateWatchdog() {
        let lastGenerateAt = 0;
        setInterval(() => {
            const genBtn = generateButton();
            if (!genBtn || !genBtn.disabled) {
                return;
            }
            const now = Date.now();
            if (now - lastGenerateAt < 120000) {
                return;
            }
            if (lastGenerateAt === 0) {
                lastGenerateAt = now;
                return;
            }
            if (isAppBusy()) {
                reportClientEvent("generate_stuck_watchdog", "Generate disabled >120s while busy", studioContext());
                lastGenerateAt = now;
            }
        }, 15000);

        const root = document.querySelector(APP) || document.body;
        let pendingGenCheck = false;
        const observer = new MutationObserver(() => {
            // class/disabled flip constantly across the Gradio tree; coalesce to
            // one check per frame instead of a querySelector per mutation.
            if (pendingGenCheck) {
                return;
            }
            pendingGenCheck = true;
            requestAnimationFrame(() => {
                pendingGenCheck = false;
                const genBtn = generateButton();
                if (genBtn && genBtn.disabled) {
                    lastGenerateAt = Date.now();
                }
            });
        });
        observer.observe(root, { subtree: true, attributes: true, attributeFilter: ["disabled", "class"] });
    }

    function boot() {
        markUiReady();
        initDeviceWatchers();
        initHotkeys();
        initClientErrorHooks();
        initGenerateWatchdog();
        reportClientEvent("studio_boot", "Studio JS initialized", studioContext());
        syncTopbarFromContext();
        // Pause the periodic topbar sync while the tab is hidden — no point doing
        // DOM work the user can't see, and it lets a backgrounded tab idle.
        setInterval(() => {
            if (!document.hidden) {
                syncTopbarFromContext();
            }
        }, 1000);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, { once: true });
    } else {
        requestAnimationFrame(boot);
    }
})();
