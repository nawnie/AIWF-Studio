# plugins/

Drop user extensions here — one folder per extension, each containing a
`plugin.py`. They load automatically at startup and can add REST API routes,
Gradio tabs, and event hooks.

- Start from the [`hello-extension/`](hello-extension/) template.
- Full guide: [`docs/EXTENSIONS.md`](../docs/EXTENSIONS.md).
- Manage (enable/disable, see errors) in **Settings → System → Extensions**
  in AIWF Studio Pro.

Only install extensions you trust: they run as Python code inside the app.
