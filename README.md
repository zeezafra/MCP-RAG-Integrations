# Lesson 3 project: scalable and modular RAG

The lesson is a conceptual reading rather than a filmed code-along. This
project makes each idea executable:

| Lesson concept | Project implementation |
| --- | --- |
| Separate orchestration from application code | `orchestration.py` owns catalog, routing, ranking, and tracing |
| Retrieval as a first-class capability | Discoverable MCP tool `retrieve_knowledge` |
| Multiple independent data sources | Text documents, an in-memory SQLite FAQ, and a runtime status feed behind four `rag://` resources |
| Dynamic, context-aware retrieval | Gemini selects source keys, depth, and keyword/broad ranking; offline mode has a deterministic fallback |
| Observability and control | Every route/read/retrieve decision is returned in `trace` |
| Modular and replaceable components | Each source is a catalog entry and independent text adapter |

`adapters.py` gives every source the same small `read()` contract. Products and
policies use document adapters, support is queried through SQLite, and live
status is generated at request time from environment configuration. Any one of
them can be replaced by an API adapter without changing the client,
orchestration code, resource URI, or model-facing contract.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Gemini API key to `GEMINI_API_KEY` in `.env`. The default model is
`gemini-3.6-flash`; set `GEMINI_MODEL` to another Gemini model ID if needed.

## Run

```powershell
python modular_rag_client.py --question "How do I return a damaged product?"
```

See discovery, deterministic routing, evidence, and the retrieval trace
without an API call:

```powershell
python modular_rag_client.py --question "Are there current shipping delays?" --no-llm
```

Try an unrelated question to observe the router's safe fallback to all sources:

```powershell
python modular_rag_client.py --question "Explain quantum entanglement." --no-llm
```

## Test

```powershell
python -m unittest discover -s tests -v
```

## Production extension seam

To replace the simulated status file with a live API:

1. Keep the `status` catalog key and `rag://status` URI stable.
2. Add an API-backed implementation of the `KnowledgeAdapter` contract and
   select it in `build_adapters()`.
3. Add timeout, caching, authentication, and retry behavior in that adapter.

The client and model-facing retrieval contract do not need to change.
