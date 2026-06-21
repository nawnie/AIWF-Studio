# AI Bot Trainer Engine

Built-in optional worker for Causal LM post-training.

Supported methods:

- `qlora`: 4-bit base model load plus LoRA adapters.
- `lora`: unquantized base model plus LoRA adapters.
- `full`: full model fine-tuning.

This engine is opt-in. The base AIWF install does not install these training
dependencies. Click **Enable AI bot trainer** in the Training tab, then restart
through `launch.py`; only then will `engines/llm/.venv` be prepared from
`engines/llm/requirements.txt`.

The worker accepts chat-style `messages` JSONL, prompt/completion rows, or
plain text rows. Use the Training tab dataset builder to create a source-backed
`messages` JSONL pack without using the GPU.
