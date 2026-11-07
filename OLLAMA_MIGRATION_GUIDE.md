# Migrating a Python app from Anthropic to local Ollama

This guide captures the migration performed in this repository. It is intended as a repeatable checklist for similar Python apps that currently call Anthropic through LangChain. It uses `qwen2.5-coder:3b` as the example model.

## What changes in the architecture

The app's retrieval, prompts, validation, and response flow can stay in place. Replace the model provider at the shared LLM boundary:

```text
Before: application -> LangChain ChatAnthropic -> Anthropic cloud API
After:  application -> LangChain ChatOllama -> local Ollama server -> local model
```

Ollama's Anthropic-compatible endpoint settings in Claude Code setup guides are for Claude Code clients. A Python app using LangChain should use LangChain's Ollama integration (`ChatOllama`) directly.

## 1. Find every model call and provider dependency

Search the repository for:

- `ChatAnthropic`, `langchain_anthropic`, `langchain-anthropic`
- `anthropic`, `ANTHROPIC_API_KEY`, and `CLAUDE_*` settings
- Provider-specific call options such as `max_tokens`
- Input/output token prices, cost estimates, budget checks, and CLI confirmation flags
- Every caller of shared model factories and structured-output helpers

In this project, extraction, routing, and answer generation all went through `app/llm.py`. Changing that shared wrapper meant the higher-level workflow could keep its existing message and invocation patterns.

## 2. Install the LangChain Ollama integration

Replace the provider dependency in `requirements.txt`:

```text
# Remove
langchain-anthropic>=...

# Add
langchain-ollama>=0.2
```

Then update the active virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use the version range appropriate for the app's existing LangChain dependencies. Confirm that `langchain_ollama` imports in the same interpreter used to run the app.

## 3. Configure the local model and endpoint

Use simple settings with local defaults:

```python
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

Expose matching optional overrides in `.env.example`:

```dotenv
OLLAMA_MODEL=qwen2.5-coder:3b
OLLAMA_BASE_URL=http://localhost:11434
```

Do not put a real API key in the example file. This setup does not require `ANTHROPIC_API_KEY`. Keep `.env` ignored by Git and do not overwrite a developer's existing `.env` without checking it first.

Verify that the model is installed and the local service is up:

```powershell
ollama list
ollama run qwen2.5-coder:3b
```

If needed, download it with `ollama pull qwen2.5-coder:3b`. The Ollama server normally listens on port `11434`.

## 4. Replace the shared model factory

Create `ChatOllama` instances from the shared configuration. `num_predict` is Ollama's generation-length option; Anthropic's `max_tokens` is not the matching constructor argument.

```python
from langchain_ollama import ChatOllama

def local_llm(max_tokens: int = 1024) -> ChatOllama:
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
        num_predict=max_tokens,
    )
```

If an app has separate "large" and "small" model factories but only one local model available, both can use the same model with different output limits. These names then describe call settings, not separate model capabilities. Switch to different models only if they are installed and configured.

Keep application-facing helpers stable where possible. That limits the migration to the provider boundary and reduces changes across call sites.

## 5. Preserve structured outputs and retries

For code using LangChain's `with_structured_output`, test the selected Ollama model and integration version with a real schema. In this repository the working configuration was:

```python
runnable = llm.with_structured_output(
    schema,
    method="json_schema",
    include_raw=True,
)
```

`include_raw=True` preserves the parsed result, raw message, and parsing error so existing retry and validation logic can remain useful:

```python
result = runnable.invoke(messages)
usage.add(result["raw"])
if result.get("parsed") is not None and result.get("parsing_error") is None:
    return result["parsed"]
```

Keep the application's schema validation, retries, and final error reporting. A model/provider change is not a reason to accept malformed or incomplete output silently. Test extraction and routing separately because a model that answers normal chat prompts may still fail a constrained schema.

## 6. Remove cloud pricing assumptions

Local inference has no Anthropic per-token API charge. In this repository:

- Removed Anthropic input/output prices and router prices from config.
- Kept usage counters for calls and token metadata when the provider supplies it.
- Left `cost_usd` at zero so existing response and benchmark shapes still work.
- Removed the extraction-cost estimate, budget threshold, and the `--yes` override that existed only to bypass that cloud-cost threshold.
- Updated benchmark labels to distinguish LLM API cost from hardware/electricity costs.

For another app, retain cost fields only if downstream consumers require them, and label zero accurately: it means no metered LLM API cost, not zero electricity or hardware cost.

## 7. Make caches provider/model-aware

If model results are cached, a document hash alone is insufficient: changing the endpoint or model can produce different outputs for the same document. Include those settings in the cache key:

```python
cache_key = hashlib.sha256(
    f"{document_hash}:{OLLAMA_BASE_URL}:{OLLAMA_MODEL}".encode()
).hexdigest()
```

Include other generation inputs that affect cached results, such as prompt/schema versions, if the application changes them independently.

## 8. Keep the rest of the application unchanged unless required

Changing the chat-model provider does not by itself require changing embeddings, vector dimensions, database schemas, retrieval, citations, or API response formats. Keep those stable unless the new model causes a demonstrated incompatibility.

Update provider-specific descriptions and setup instructions in READMEs, architecture diagrams, case studies, agent instructions, and benchmark descriptions. Remove stale instructions to configure an Anthropic key or pass an Anthropic-only CLI flag.

## 9. Database and local-service pitfalls encountered here

These are separate from the LLM-provider migration, but they blocked the migrated app from running end to end:

### Start the services before running the app

This repository uses Docker Compose for Neo4j and PostgreSQL. On Windows, Docker Desktop must be running before `docker compose up -d` can start the containers.

```powershell
docker compose up -d
docker compose ps
```

Verify database connections using the same interpreter and configuration as the app, not just by checking that a port is open.

### Avoid collisions with a host PostgreSQL server

This machine already had a PostgreSQL service using host port `5432`. The Compose database was moved to host port `5433`, while it still listens on `5432` inside its container:

```yaml
ports: ["5433:5432"]
```

Update all three matching locations when changing that port:

1. Compose host-to-container port mapping.
2. The default application DSN in config.
3. `.env.example` and the developer's local `.env`.

The app's DSN in this repository uses `localhost:5433`. A port/password error naming `5432` is evidence that the failing process is not using that current DSN.

### Check inherited environment variables

`os.environ` values override values loaded from `.env`. A long-running VS Code terminal may keep an old `PG_DSN` after the file has been updated. Inspect only non-secret fields (host, port, user, database); do not print the password.

For a stale PowerShell session, remove the override or start a fresh terminal:

```powershell
Remove-Item Env:PG_DSN -ErrorAction SilentlyContinue
python -c "from app import config; print(config.PG_DSN.split('@', 1)[-1])"
```

Do not share or log the full DSN because it contains a password.

### Existing database volumes keep their initialized credentials

For PostgreSQL container images, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` are initialization settings. Changing them in Compose does not automatically change roles inside an already initialized persistent volume. Do not delete the volume as a shortcut; first verify the target server and preserve its data. If a password really needs changing, update the role on the correct database instance and verify a TCP connection using the app's DSN.

### Empty Neo4j graphs can produce property warnings

Neo4j may warn that statically referenced properties such as `e.type` do not exist when the graph is empty. In this project, using `properties(e)['type']` (and equivalent keys) in the initial entity-loading query avoided warnings while preserving the returned shape for populated nodes. Confirm whether a warning is harmless before changing query behavior.

## 10. Verification checklist

Run checks in this order so provider, database, and application failures are distinguishable:

1. **Ollama is reachable and the exact model tag exists.**
   - `ollama list`
   - `ollama run qwen2.5-coder:3b`
2. **The Python environment imports the selected integration.**
   - `python -c "from langchain_ollama import ChatOllama; print('import ok')"`
3. **A normal prompt works through the app's model factory.**
4. **Structured extraction returns the expected Pydantic schema.**
5. **Structured routing returns valid enum/field values.**
6. **Neo4j and PostgreSQL connections work using the app's effective config.**
7. **The ingestion CLI starts and completes; inspect graph/vector counts and logs.**
8. **Run focused compilation and configured tests.** If the repo has no test suite, use its documented syntax check, such as `python -m compileall app scripts`.
9. **Run the API and one representative query.** Confirm citations and error paths still behave correctly.
10. **Search again for stale Anthropic dependencies, environment variables, pricing, and docs.**

## Migration summary for the next repository

- Replace the provider package and shared chat-model factory.
- Configure the local Ollama endpoint and installed model.
- Test structured output explicitly; preserve validation and retries.
- Remove cloud-only pricing gates while preserving needed output fields.
- Include model/endpoint settings in cache keys.
- Update dependency, environment, CLI, README, and operational instructions together.
- Verify databases separately; do not mistake stale environment variables or host-port conflicts for model errors.
