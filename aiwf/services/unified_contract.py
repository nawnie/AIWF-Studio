"""Single source of truth for the unified workspace operations (/api/pro/unified/*).

Three surfaces are generated from this one list, so they cannot drift apart:
- the HTTP API's self-description, GET /api/pro/unified/describe (aiwf/web/unified_api.py);
- the agent CLI, integrations/agent_bridge/aiwf_studio_cli.py (one subcommand per operation);
- the agent MCP server, integrations/agent_bridge/aiwf_studio_mcp.py (one tool per operation).

This module is plain data with no imports, so the CLI and MCP server can load it
without pulling in the rest of AIWF Studio (torch, diffusers and so on).
A test checks that every operation here matches a real route and vice versa.
"""

SCHEMA_VERSION = "1"
BASE_PATH = "/api/pro/unified"
DEFAULT_BASE_URL = "http://127.0.0.1:7860"

# --- reusable parameter definitions ------------------------------------------------
# Each parameter says where it goes ("path", "query" or "body"), its JSON-schema
# type, whether it is required, and a description written for an agent.
_PROJECT_ID = {
    "name": "project_id", "in": "path", "required": True,
    "schema": {"type": "string", "pattern": "^aiwfp-[0-9a-f]{16}$"},
    "description": "Shared project ID (aiwfp-...) from list_projects or create_project.",
}
_REVISION = {
    "name": "manifest_sha256", "in": "body", "required": True,
    "schema": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "description": "The package's 64-hex manifest revision, exactly as list_packages returned it.",
}

# --- the operations ----------------------------------------------------------------
# "mutating" operations change state in Studio or a sibling app and are accepted
# only from the local machine. None of them can start training.
OPERATIONS = [
    {
        "name": "describe", "method": "GET", "path": "/describe", "mutating": False,
        "summary": "List every operation with its parameters, the recommended workflow and the safety rules.",
        "params": [],
    },
    {
        "name": "status", "method": "GET", "path": "/status", "mutating": False,
        "summary": "Check which engines are reachable (Dataset Studio, ReTrain, Qwen Chat, ComfyUI for images), why not if not, and which models each offers.",
        "params": [],
    },
    {
        "name": "get_setup", "method": "GET", "path": "/setup", "mutating": False,
        "summary": "Show where Studio saves images and finds models (saved value, effective folder, default, whether it exists, free space). Folders are changed only in the Studio setup wizard.",
        "params": [],
    },
    {
        "name": "list_projects", "method": "GET", "path": "/projects", "mutating": False,
        "summary": "List shared projects (ID, name, created time, counts of recorded actions).",
        "params": [],
    },
    {
        "name": "create_project", "method": "POST", "path": "/projects", "mutating": True,
        "summary": "Create a shared project. Its ID links Studio outputs, Dataset Studio assets, ReTrain imports and Qwen context.",
        "params": [
            {"name": "name", "in": "body", "required": True, "schema": {"type": "string", "minLength": 1, "maxLength": 120},
             "description": "Human-readable project name (1-120 printable characters)."},
        ],
    },
    {
        "name": "get_project", "method": "GET", "path": "/projects/{project_id}", "mutating": False,
        "summary": "Show one project with its recent ledger events (catalogs, imports, preflights, Qwen sends).",
        "params": [_PROJECT_ID],
    },
    {
        "name": "list_outputs", "method": "GET", "path": "/outputs", "mutating": False,
        "summary": "List recent AIWF Studio image outputs with paths relative to Studio's output folder, ready for catalog_outputs.",
        "params": [
            {"name": "limit", "in": "query", "required": False, "schema": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
             "description": "How many of the newest outputs to return (default 20)."},
        ],
    },
    {
        "name": "catalog_outputs", "method": "POST", "path": "/projects/{project_id}/catalog-outputs", "mutating": True,
        "summary": "Add Studio outputs to Dataset Studio's catalog, tagged with the project ID; the generation prompt becomes an import caption when none exists.",
        "params": [
            _PROJECT_ID,
            {"name": "output_paths", "in": "body", "required": True,
             "schema": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 200},
             "description": "Relative paths from list_outputs (or /api/pro/outputs/... URLs). Paths outside Studio's output folder are refused."},
        ],
    },
    {
        "name": "list_packages", "method": "GET", "path": "/datasets/packages", "mutating": False,
        "summary": "List Dataset Studio's published ReTrain text packages with their immutable revision hashes; invalid ones carry a reason.",
        "params": [],
    },
    {
        "name": "import_package", "method": "POST", "path": "/retrain/import", "mutating": True,
        "summary": "Import exactly the selected package revision into ReTrain. Refused with stale_revision if the package changed; re-importing reuses the same copy.",
        "params": [
            {**_PROJECT_ID, "in": "body"},
            {"name": "package_name", "in": "body", "required": True, "schema": {"type": "string", "minLength": 1, "maxLength": 180},
             "description": "Package name exactly as list_packages returned it."},
            _REVISION,
        ],
    },
    {
        "name": "preflight", "method": "POST", "path": "/retrain/preflight", "mutating": True,
        "summary": "Run a ReTrain dry-run plan against an imported revision. Never starts training; returns gates, VRAM estimate and dependency checks.",
        "params": [
            {**_PROJECT_ID, "in": "body"},
            {"name": "dataset_id", "in": "body", "required": True, "schema": {"type": "string", "pattern": "^sha256-[0-9a-f]{64}$"},
             "description": "dataset_id returned by import_package (sha256-<manifest_sha256>)."},
            _REVISION,
            {"name": "model_id", "in": "body", "required": True, "schema": {"type": "string", "minLength": 1, "maxLength": 120},
             "description": "A text-capable ReTrain model_id from status (capabilities.retrain.models)."},
            {"name": "settings", "in": "body", "required": False, "schema": {"type": "object"},
             "description": "Optional ReTrain settings, e.g. {\"method\": \"QLoRA\"}. Only method, tuneScope, lastNLayers, contextLength, microBatch, gradAccum, loraRank, precision, optimizer and scheduler are passed on."},
        ],
    },
    {
        "name": "list_models", "method": "GET", "path": "/models", "mutating": False,
        "summary": "List ReTrain's training models: on this PC or missing, and for missing ones whether Hugging Face lets anyone download it (open) or it needs the user's key (gated), with the download size.",
        "params": [
            {"name": "refresh", "in": "query", "required": False, "schema": {"type": "boolean", "default": False},
             "description": "Re-check Hugging Face instead of using the 10-minute cache."},
        ],
    },
    {
        "name": "download_model", "method": "POST", "path": "/models/download", "mutating": True,
        "summary": "Start downloading a missing model's weights from Hugging Face. Can be many GB and uses the network: only call it after the user agrees. Gated models are refused until the user has saved a key in the Studio UI.",
        "params": [
            {"name": "model_id", "in": "body", "required": True, "schema": {"type": "string", "minLength": 1, "maxLength": 120},
             "description": "A model_id from list_models whose status is missing and whose access state is open or gated_ok."},
        ],
    },
    {
        "name": "download_status", "method": "GET", "path": "/models/downloads/{job_id}", "mutating": False,
        "summary": "Progress of a model download (status, bytes done of total, files done).",
        "params": [
            {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "pattern": "^[0-9a-f]{12}$"},
             "description": "job_id returned by download_model."},
        ],
    },
    {
        "name": "cancel_download", "method": "POST", "path": "/models/downloads/{job_id}/cancel", "mutating": True,
        "summary": "Stop a running model download. Partial files are kept so a new download resumes.",
        "params": [
            {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "pattern": "^[0-9a-f]{12}$"},
             "description": "job_id returned by download_model."},
        ],
    },
    {
        "name": "qwen_context", "method": "GET", "path": "/projects/{project_id}/qwen-context", "mutating": False,
        "summary": "Show the exact project context card qwen_ask would send (names, IDs, counts, revision hashes; no files, paths or chat history).",
        "params": [_PROJECT_ID],
    },
    {
        "name": "qwen_ask", "method": "POST", "path": "/projects/{project_id}/qwen-ask", "mutating": True,
        "summary": "Ask Qwen Chat a question with the project context card attached. A model that is not loaded is loaded first (uses the GPU).",
        "params": [
            _PROJECT_ID,
            {"name": "model_id", "in": "body", "required": True, "schema": {"type": "string", "minLength": 1, "maxLength": 200},
             "description": "A Qwen Chat model_id from status (capabilities.qwen_chat.models); prefer one marked loaded."},
            {"name": "question", "in": "body", "required": True, "schema": {"type": "string", "minLength": 1, "maxLength": 4000},
             "description": "The question (1-4000 characters)."},
        ],
    },
    {
        "name": "model_families", "method": "GET", "path": "/model-families", "mutating": False,
        "summary": "Which trained artifacts (ReTrain adapters, image LoRAs, GGUF) work in which app; they are not interchangeable.",
        "params": [],
    },
    # --- images: Qwen Image 2.1 on the local ComfyUI engine (aiwf/services/image_jobs.py) ---
    {
        "name": "generate_image", "method": "POST", "path": "/images", "mutating": True,
        "summary": "Queue a Qwen Image 2.1 text-to-image job on this PC's GPU (ComfyUI). Returns a job to poll with image_status; the first job loads the model and takes longer.",
        "params": [
            {"name": "prompt", "in": "body", "required": True, "schema": {"type": "string", "minLength": 1, "maxLength": 2000},
             "description": "What the image should show (1-2000 characters)."},
            {"name": "aspect_ratio", "in": "body", "required": False,
             "schema": {"type": "string", "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"], "default": "1:1"},
             "description": "Image shape (default 1:1)."},
            {"name": "quality", "in": "body", "required": False, "schema": {"type": "string", "enum": ["draft", "standard"], "default": "draft"},
             "description": "draft is about 0.6 megapixels and faster; standard is about 1 megapixel."},
            {"name": "seed", "in": "body", "required": False, "schema": {"type": "integer", "minimum": 0, "maximum": 9007199254740991},
             "description": "Fixed seed to reproduce an image; omitted means a random seed (returned in the job)."},
            {"name": "project_id", "in": "body", "required": False, "schema": {"type": "string", "pattern": "^aiwfp-[0-9a-f]{16}$"},
             "description": "Optional shared project ID; the finished image is recorded in that project's ledger."},
        ],
    },
    {
        "name": "image_status", "method": "GET", "path": "/images/{job_id}", "mutating": False,
        "summary": "State of an image job (queued with position, running, done, failed or cancelled) and, when done, the saved images with paths relative to Studio's output folder.",
        "params": [
            {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "pattern": "^img-[0-9a-f]{12}$"},
             "description": "job_id returned by generate_image."},
        ],
    },
    {
        "name": "cancel_image", "method": "POST", "path": "/images/{job_id}/cancel", "mutating": True,
        "summary": "Cancel an image job: removed from ComfyUI's queue if waiting, interrupted if it is the job on the GPU.",
        "params": [
            {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "pattern": "^img-[0-9a-f]{12}$"},
             "description": "job_id returned by generate_image."},
        ],
    },
]

# Routes that are intentionally outside the agent operation contract. The key
# endpoints carry the user's Hugging Face key; chat-question is the Studio UI's
# local-only callback for recording a streamed Qwen exchange, not an agent tool;
# setup/folders changes where Studio saves and finds things, which is the person's
# choice in the setup wizard (agents can read it with get_setup).
UI_ONLY_ROUTES = [
    ("POST", "/models/hf-token"),
    ("POST", "/models/hf-token/clear"),
    ("POST", "/projects/{project_id}/chat-question"),
    ("POST", "/setup/folders"),
]

# Routes that return file bytes (images) for UIs to display. Agents get the same images as
# relative paths from image_status / list_outputs, so these are not offered as tools.
MEDIA_ROUTES = [("GET", "/images/{job_id}/files/{index}")]

# --- guidance shipped with describe and the MCP server's instructions ---------------
WORKFLOW = [
    "status: confirm Dataset Studio, ReTrain and Qwen Chat are reachable before acting.",
    "list_projects or create_project: every action below is recorded against one project ID.",
    "list_outputs then catalog_outputs: put Studio images into Dataset Studio's catalog with the project tag.",
    "list_packages: pick one ready package and keep its package_name and manifest_sha256 together.",
    "import_package with that exact pair: returns dataset_id = sha256-<manifest_sha256>.",
    "list_models: check the chosen model's weights are on this PC; if missing, tell the user and ask before download_model.",
    "preflight with dataset_id, manifest_sha256 and a text-capable model_id: a dry-run plan only.",
    "qwen_context, then qwen_ask: preview what will be sent, then ask Qwen Chat about the project.",
    "generate_image, then image_status every few seconds until done: new images land in Studio's output folder and can be cataloged like any output.",
]

RULES = [
    "Training is never started through this API; start runs in ReTrain itself.",
    "Never guess a package revision: always pass manifest_sha256 from list_packages. A changed package returns stale_revision (HTTP 409); refresh and choose again.",
    "Image outputs are cataloged only; Dataset Studio's ReTrain packages are text-only training data.",
    "Never start a model download without the user's go-ahead: weights can be many GB. Gated models need the user to save their Hugging Face key in the Studio UI; agents never handle the key.",
    "generate_image uses the GPU; check status (capabilities.comfyui) first and do not queue many jobs at once.",
    "Mutating operations are accepted only from the local machine.",
    "Errors return {\"detail\": {\"code\": ..., \"message\": ...}}; pro_not_running means AIWF Studio Pro is not running at the base URL.",
]


def find_operation(name: str) -> dict:
    """Return the operation with this name (dashes and underscores are interchangeable)."""
    wanted = name.replace("-", "_")
    for operation in OPERATIONS:
        if operation["name"] == wanted:
            return operation
    raise KeyError(f"Unknown operation: {name}")


def input_schema(operation: dict) -> dict:
    """JSON schema for one operation's arguments (used as an MCP tool inputSchema)."""
    properties = {}
    required = []
    # this loop merges each parameter's schema with its agent-facing description
    for param in operation["params"]:
        properties[param["name"]] = {**param["schema"], "description": param["description"]}
        if param["required"]:
            required.append(param["name"])
    schema = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def describe() -> dict:
    """The payload served at GET /api/pro/unified/describe."""
    return {
        "schema_version": SCHEMA_VERSION,
        "base_path": BASE_PATH,
        "default_base_url": DEFAULT_BASE_URL,
        "operations": OPERATIONS,
        "workflow": WORKFLOW,
        "rules": RULES,
        "clients": {
            "cli": r"C:\AI-Agent-Workspace\bin\aiwf-studio.cmd <operation> [--param value]  (or: call <operation> --args-json -  with JSON on stdin)",
            "mcp": r"F:\environments\aiwf-studio-agent-py312\Scripts\python.exe F:\AIWF_Studio\integrations\agent_bridge\aiwf_studio_mcp.py  (stdio; tools are named studio_<operation>)",
            "docs": r"F:\AIWF_Studio\docs\agent-workflows\STUDIO_FLOW_AGENT_ACCESS.md",
        },
    }
