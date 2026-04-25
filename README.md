# privrag

RAG para documentos personales: **ingestión** → troceado → **embeddings** (locales o OpenAI) → almacén vectorial **Qdrant** → **consulta** recuperando fragmentos relevantes y, si lo configuras, respuesta con **LLM** (Ollama local, OpenAI o [OpenRouter](https://openrouter.ai)).

## Requisitos

- **Python 3.11+**
- **Docker** (solo para levantar Qdrant; el resto puede ir en tu máquina)
- Opcional: **Ollama** (u otro servidor compatible) si quieres respuestas generadas en local

## Instalación

```bash
git clone <tu-repo> privrag
cd privrag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Copia la plantilla de variables y edítala si hace falta:

```bash
cp .env.example .env
```

## Qdrant con Docker

La base vectorial se sirve con Compose (puerto **6333** por defecto):

```bash
docker compose up -d
```

Para parar:

```bash
docker compose down
```

Los datos persisten en el volumen Docker de Qdrant. Si cambias de versión de imagen, revisa la compatibilidad con `qdrant-client` del `pyproject.toml`.

## Variables de entorno

El archivo **`.env`** (junto al proyecto) controla Qdrant, embeddings, LLM y el tamaño de los chunks. Resumen:

| Variable | Descripción |
|----------|-------------|
| `QDRANT_URL` | URL del servidor Qdrant (por defecto `http://localhost:6333`) |
| `QDRANT_API_KEY` | Solo si Qdrant exige API key (p. ej. despliegue en la nube) |
| `EMBEDDING_BACKEND` | `local` (por defecto) u `openai` |
| `LOCAL_EMBEDDING_MODEL` | Modelo de sentence-transformers si `local` (multilingüe por defecto) |
| `OPENAI_API_KEY` | Necesaria si embeddings o LLM usan OpenAI |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings OpenAI (p. ej. `text-embedding-3-small`) |
| `LLM_BACKEND` | `ollama` (por defecto), `none`, `openai`, `openrouter` o `lmstudio` |
| `OLLAMA_BASE_URL` | Base URL de Ollama (por defecto `http://localhost:11434`) |
| `OLLAMA_MODEL` | Nombre del modelo en Ollama (p. ej. `llama3.2`) |
| `OPENAI_MODEL` | Modelo de chat si `LLM_BACKEND=openai` |
| `OPENROUTER_API_KEY` | Clave de [OpenRouter](https://openrouter.ai/keys) si `LLM_BACKEND=openrouter` |
| `OPENROUTER_BASE_URL` | Por defecto `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | Identificador del modelo en OpenRouter (p. ej. `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`) |
| `OPENROUTER_HTTP_REFERER` | Opcional; URL de referencia (OpenRouter lo recomienda en su documentación) |
| `OPENROUTER_APP_TITLE` | Opcional; nombre de la app (`X-Title`) |
| `LM_STUDIO_BASE_URL` | Host y puerto del servidor local; debe acabar en **`/v1`** (p. ej. `http://127.0.0.1:41343/v1`). Si omites `/v1`, privrag lo añade. La API es **POST** a `…/v1/chat/completions` (abrir la URL en el navegador muestra «Cannot GET», es normal) |
| `LM_STUDIO_API_KEY` | Cabecera `Bearer` (LM Studio suele aceptar cualquier valor; por defecto `lm-studio`) |
| `LM_STUDIO_MODEL` | Identificador del modelo cargado en LM Studio (vacío hasta que lo configures) |
| `CHUNK_SIZE` | Tamaño aproximado del trozo en caracteres |
| `CHUNK_OVERLAP` | Solapamiento entre trozos consecutivos |
| `LLM_MAX_CONTEXT_CHARS` | Máximo de caracteres de contexto enviados al LLM (por defecto 24000; bajar si Ollama devuelve error por tamaño) |
| `LLM_MAX_TOKENS` | Opcional; tope de tokens de **salida** del LLM. Si no se define, cada API/modelo usa su valor por defecto |
| `LLM_CITATIONS` | `true` (defecto): el modelo recibe índices/rutas en el contexto y se le pide citar; `false`: contexto más compacto y respuesta sin citas (menos tokens) |

La lista completa y comentarios están en **`.env.example`**.

## Uso: CLI `privrag`

Tras activar el entorno virtual, el ejecutable **`privrag`** expone dos comandos.

### Ayuda

```bash
privrag --help
privrag ingest --help
privrag query --help
```

### `ingest`: indexar documentos

Indexa un **archivo** o todos los documentos admitidos bajo una **carpeta** (recursivo).

**Formatos:** `.md`, `.markdown`, `.txt`, `.rst`, `.pdf`

```bash
privrag ingest RUTA [--collection NOMBRE] [--topic ETIQUETA]
```

| Opción | Corto | Descripción |
|--------|-------|-------------|
| `--collection` | `-c` | Nombre de la colección en Qdrant (por defecto `docs`) |
| `--topic` | `-g` | Metadato opcional guardado en cada chunk (p. ej. `dnd5e`) |

**Ejemplo:**

```bash
privrag ingest ./examples --collection docs
privrag ingest ./mis_reglas --collection dnd --topic dnd5e
```

La primera vez descargará el modelo de embeddings local (Hugging Face) si usas `EMBEDDING_BACKEND=local`.

### `query`: preguntar

Recupera los fragmentos más similares a tu pregunta y, según `LLM_BACKEND`, genera una respuesta con contexto o solo muestra pasajes.

```bash
privrag query "TU PREGUNTA" [--collection NOMBRE] [--limit K] [--no-llm] [--no-citations] [--topic ETIQUETA] [--source-prefix RUTA]
```

| Opción | Corto | Descripción |
|--------|-------|-------------|
| `--collection` | `-c` | Colección Qdrant (debe coincidir con la usada en `ingest`) |
| `--limit` | `-k` | Número de fragmentos a recuperar (por defecto 5) |
| `--no-llm` | — | Solo búsqueda vectorial: imprime puntuación, ruta y texto de cada hit (sin LLM) |
| `--no-citations` | — | Respuesta del LLM sin pedir citas; contexto sin rutas (menos tokens); no imprime el bloque «Fuentes» al final |
| `--topic` | `-g` | Solo chunks indexados con ese metadato (mismo valor que `ingest --topic`) |
| `--source-prefix` | — | Solo fuentes cuya ruta absoluta empieza por la indicada (archivo o carpeta) |

**Ejemplos:**

```bash
# Con LLM (Ollama u OpenAI según .env)
privrag query "¿Cómo funciona la iniciativa?" --collection docs

# Solo recuperación (útil sin GPU o para depurar)
privrag query "iniciativa" --collection docs --no-llm

# Filtrar por materia o por carpeta (útil con varios sistemas en la misma colección)
privrag query "ventaja" --collection docs --topic dnd5e --no-llm
privrag query "tabla" --collection docs --source-prefix ./examples --no-llm
```

Si `LLM_BACKEND=none` o el LLM no está disponible, verás el comportamiento de solo pasajes, similar a `--no-llm`.

## Interfaz web

Tras instalar el paquete, puedes levantar una UI mínima en el navegador (ingesta por subida de archivos y formulario de consulta):

```bash
privrag-web
```

Por defecto escucha en **http://127.0.0.1:8765**. Usa la misma configuración que la CLI (`.env`, Qdrant, embeddings y LLM). En la sección de consulta puedes elegir **proveedor** (Ollama, OpenAI, OpenRouter o **LM Studio**), **modelo**, **max_tokens** y si quieres **citas en la respuesta** por petición; si los dejas vacíos, se usan los valores del `.env`. La ingesta desde el navegador guarda los ficheros en un directorio temporal, los indexa y los borra; para carpetas grandes sigue siendo más práctico `privrag ingest` con una ruta en disco.

**Seguridad:** está pensado para uso local; no expongas el puerto a redes que no controles sin autenticación y HTTPS.

## Modos de operación

### Embeddings

- **`local`**: sin API externa; usa `sentence-transformers` y el modelo indicado en `LOCAL_EMBEDDING_MODEL` (por defecto multilingüe, adecuado para inglés y español).
- **`openai`**: requiere `OPENAI_API_KEY` y define `OPENAI_EMBEDDING_MODEL`. La dimensión del vector puede cambiar respecto al modelo local: usa **otra colección** o re-indexa si cambias de backend de embeddings.

### LLM

- **`ollama`**: asegúrate de tener Ollama en marcha y el modelo indicado en `OLLAMA_MODEL` descargado (`ollama pull ...`).
- **`openai`**: requiere `OPENAI_API_KEY` y opcionalmente ajusta `OPENAI_MODEL`.
- **`openrouter`**: usa la API compatible con OpenAI de [OpenRouter](https://openrouter.ai/docs) (`POST .../chat/completions`). Configura `OPENROUTER_API_KEY` y `OPENROUTER_MODEL` (lista de modelos en [openrouter.ai/models](https://openrouter.ai/models)). Opcionales: `OPENROUTER_HTTP_REFERER` y `OPENROUTER_APP_TITLE` (cabeceras que OpenRouter sugiere para estadísticas).
- **`lmstudio`**: [LM Studio](https://lmstudio.ai/) con el servidor local activo (pestaña **Developer** → servidor iniciado). Se usa `POST …/v1/chat/completions` (OpenAI-compatible); si devuelve 404, se reintenta con la API nativa `POST …/api/v1/chat`. Configura `LM_STUDIO_BASE_URL` (host y puerto que muestra LM Studio, con `/v1` al final), `LM_STUDIO_MODEL` y opcionalmente `LM_STUDIO_API_KEY` (Bearer).
- **`none`**: no se llama a ningún modelo generativo; solo tiene sentido junto a `--no-llm` o para inspeccionar recuperación cuando no quieres generación.

### Depurar LM Studio (endpoint correcto)

Si las consultas fallan, prueba qué responde el servidor en tu máquina:

```bash
privrag lmstudio-probe
privrag lmstudio-probe --json
privrag lmstudio-probe --base-url http://127.0.0.1:TU_PUERTO/v1 --model "id-del-modelo-en-lm-studio"
```

Con la interfaz web levantada: abre **http://127.0.0.1:8765/api/debug/lmstudio** (misma información en JSON).

Comprueba en LM Studio que el **servidor local esté iniciado** (Developer) y que el **puerto** coincida con `LM_STUDIO_BASE_URL`. Prueba manual con:

```bash
curl -sS "http://127.0.0.1:TU_PUERTO/v1/models" -H "Authorization: Bearer lm-studio"
```

Si `GET /v1/models` da 404, ese puerto no es el de la API o LM Studio no expone la ruta OpenAI en esa versión.

Si **todas** las pruebas devuelven **404** con HTML `Cannot GET` / `Cannot POST`, el puerto no es el del servidor LM Studio (a menudo es otra app). Vuelve a copiar la URL desde LM Studio → **Developer** → **Start Server**.

## Notas prácticas

- **Colección y modelo de embeddings:** cada colección está ligada a una dimensión de vector. Si cambias de modelo de embeddings, crea una colección nueva o borra la anterior y vuelve a ejecutar `ingest`.
- **Privacidad y coste:** modo local (embeddings + Ollama o LM Studio) evita enviar documentos a la nube; OpenAI u OpenRouter implican enviar el contexto y la pregunta a sus APIs según sus políticas y precios.
- **Hugging Face:** el primer uso del modelo local puede mostrar avisos de rate limit; opcionalmente configura `HF_TOKEN` si tienes cuenta.

## Estructura del paquete (referencia)

- `src/privrag/`: código (config, ingest, embeddings, Qdrant, LLM, pipeline RAG, CLI, interfaz web)
- `examples/`: documento de ejemplo para pruebas rápidas
- `docker-compose.yml`: servicio Qdrant
