(function () {
    const STUDIO = ".aiwf-studio";
    const APP = ".aiwf-app";
    const PROMPT_ID = "aiwf-prompt";
    const GENERATE_ID = "aiwf-generate";
    const TOPBAR_STATUS_ID = "aiwf-topbar-status";
    const HANDHELD_QUERY = "(max-width: 900px), (hover: none) and (pointer: coarse) and (max-width: 1100px)";

    const BUSY_RE = /\*\*(generating|working|loading|running|stepping|step\s*\d|processing)\*\*/i;
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
        if (!isEnter || !event.shiftKey || event.repeat) {
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

    function syncStatusBars() {
        document.querySelectorAll(".aiwf-status-bar").forEach((bar) => {
            const text = statusTextFromBar(bar);
            const state = inferStatusState(text);
            bar.dataset.state = state;
        });
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
        const observer = new MutationObserver(() => {
            if (pendingSync) {
                return;
            }
            pendingSync = true;
            requestAnimationFrame(() => {
                pendingSync = false;
                bindPromptHotkey();
                syncTopbarFromContext();
            });
        });
        observer.observe(root, {
            childList: true,
            subtree: true,
        });
    }

    function boot() {
        markUiReady();
        initDeviceWatchers();
        initHotkeys();
        syncTopbarFromContext();
        setInterval(syncTopbarFromContext, 800);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot, { once: true });
    } else {
        requestAnimationFrame(boot);
    }
})();
