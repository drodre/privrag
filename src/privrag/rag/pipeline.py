from privrag.config import LLMBackend, get_settings
from privrag.embed import get_embedder
from privrag.llm import get_llm
from privrag.store import QdrantStore, SearchHit


def format_context(
    hits: list[SearchHit],
    max_chars: int = 12000,
    *,
    include_citations: bool = True,
) -> str:
    blocks: list[str] = []
    total = 0
    for i, h in enumerate(hits, start=1):
        if include_citations:
            src = h.payload.get("source_path", "?")
            block = f"[{i}] ({src})\n{h.text}"
        else:
            block = h.text.strip()
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block) + 2
    return "\n\n---\n\n".join(blocks)


def _system_prompt_for_citations(include_citations: bool) -> str:
    base = (
        "You answer questions about tabletop RPG documents. Use the provided context; "
        "if it is not enough, say so. Reply in the same language as the question when possible. "
    )
    if include_citations:
        return base + "Reference sources by their [n] index and file path when useful."
    return (
        base + "Do not cite sources, file paths, or bracketed references; answer concisely in plain prose."
    )


def retrieve(
    question: str,
    collection: str,
    limit: int = 5,
    filter_topic: str | None = None,
    source_path_prefix: str | None = None,
) -> list[SearchHit]:
    embedder = get_embedder()
    store = QdrantStore()
    store.ensure_collection(collection, embedder.vector_size)
    qv = embedder.encode([question])[0]
    return store.search(
        collection,
        qv,
        limit=limit,
        filter_topic=filter_topic,
        source_path_prefix=source_path_prefix,
    )


def answer(
    question: str,
    collection: str,
    limit: int = 5,
    use_llm: bool | None = None,
    filter_topic: str | None = None,
    source_path_prefix: str | None = None,
    llm_backend: LLMBackend | None = None,
    llm_model: str | None = None,
    max_tokens: int | None = None,
    include_citations: bool | None = None,
) -> tuple[list[SearchHit], str | None]:
    """Devuelve (hits, respuesta_llm o None si solo recuperación)."""
    hits = retrieve(
        question,
        collection,
        limit=limit,
        filter_topic=filter_topic,
        source_path_prefix=source_path_prefix,
    )
    s = get_settings()
    want_llm = use_llm if use_llm is not None else (s.llm_backend != LLMBackend.NONE)
    chosen_backend = llm_backend if llm_backend is not None else s.llm_backend
    if not want_llm or not hits or chosen_backend == LLMBackend.NONE:
        return hits, None
    llm = get_llm(backend=llm_backend, model=llm_model)
    cite = include_citations if include_citations is not None else s.llm_citations
    ctx = format_context(
        hits,
        max_chars=s.llm_max_context_chars,
        include_citations=cite,
    )
    system = _system_prompt_for_citations(cite)
    user = f"Context:\n\n{ctx}\n\nQuestion:\n{question}"
    text = llm.complete(system=system, user=user, max_tokens=max_tokens)
    return hits, text
