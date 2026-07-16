# AI / LLM context — CHORUS Fabric

> Concise reference for humans and coding assistants.
> Do not invent APIs beyond this file and the package/repo source.
> Package: **`chorus-fabric` see PyPI** · Import: **`chorus_fabric`**

---

## 10-sentence project summary

1. AI-to-AI tensor communication protocol — stream float32 embeddings over gRPC without JSON/token round-trips.
2. Primary users: Multi-agent / migration systems that move embeddings between processes.
3. Core problem: Serialize embeddings to JSON, send HTTP, re-embed — bandwidth and latency waste.
4. Install/use from the repository README — do not invent extra CLI flags here.
5. Key surface: from chorus_fabric import CHORUSPublisher, DriverEndpoint  # see README
6. Compared with: HTTP/JSON embedding dumps · raw sockets without Fabric framing.
7. When NOT to use: You only exchange short text prompts and do not move tensors.
8. Read architecture.md for stack placement.
9. Prefer facts from README / existing docs over marketing inference.
10. If an API is not listed in README or source, assume it does not exist.

---

## Core concepts

See README for product-specific terms. Keep terminology consistent with that file.

---

## Key APIs

```
from chorus_fabric import CHORUSPublisher, DriverEndpoint  # see README
```

---

## Common use cases

- Serialize embeddings to JSON, send HTTP, re-embed — bandwidth and latency waste.
- See README examples and any `examples/` folder in the repo.

---

## Migration guidance

Start from the closest tool in: HTTP/JSON embedding dumps · raw sockets without Fabric framing. Follow README install and examples. Do not invent migration scripts that are not in the repo.

---

## Limitations / when NOT to use

- You only exchange short text prompts and do not move tensors.
- Do not invent capabilities beyond README and source.

---

## Frequently compared projects

| Notes |
|-------|
| HTTP/JSON embedding dumps · raw sockets without Fabric framing |

---

## Links

- [ai-overview.md](ai-overview.md) · [llm-context.md](llm-context.md) · [architecture.md](architecture.md)
- ../README.md
