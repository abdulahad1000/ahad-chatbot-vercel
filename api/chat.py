"""
api/chat.py — Vercel serverless function for the memory-augmented chatbot.

Same architecture as the local version, with three substitutions that make it
serverless-compatible:
  SQLite            -> Neon Postgres
  ChromaDB          -> embeddings stored as a JSONB column
  sentence-transf.  -> Gemini embedding API

The retrieval scoring is unchanged: recency decay + cosine relevance, both
min-max scaled, summed into a composite, ranked descending.
"""

import json
import math
import os
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler

import psycopg

DATABASE_URL = os.environ.get("DATABASE_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

EMBED_DIMS = 768
TOP_K = 5


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic_memory (
    id               SERIAL PRIMARY KEY,
    session_id       TEXT NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    importance_score REAL NOT NULL,
    embedding        JSONB
);
"""


def connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    conn = psycopg.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()
    return conn


def add_entry(conn, session_id, role, content, embedding):
    """Insert one conversation turn."""
    importance = round(min(1.0, max(0.1, len(content) / 100.0)), 2)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO episodic_memory
                (session_id, timestamp, role, content, importance_score, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id,
                datetime.now(timezone.utc),
                role,
                content,
                importance,
                json.dumps(embedding) if embedding else None,
            ),
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


def get_candidates(conn, exclude_session_id):
    """Fetch user-role entries from every session except the current one."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, session_id, timestamp, role, content, embedding
            FROM episodic_memory
            WHERE session_id != %s AND role = 'user' AND embedding IS NOT NULL
            """,
            (exclude_session_id,),
        )
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "session_id": r[1],
            "timestamp": r[2],
            "role": r[3],
            "content": r[4],
            "embedding": r[5],
        }
        for r in rows
    ]


# --------------------------------------------------------------------------
# Embeddings (Gemini)
# --------------------------------------------------------------------------


def embed(text):
    """Return an L2-normalized embedding vector for a single string."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    body = json.dumps(
        {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": EMBED_DIMS,
        }
    ).encode()

    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-001:embedContent",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
    )

    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read())

    vector = payload["embedding"]["values"]

    # Truncated Matryoshka vectors are not unit length, so normalize here.
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(a, b):
    """Dot product of two normalized vectors."""
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Retrieval scoring
# --------------------------------------------------------------------------


def min_max_scale(values, threshold=0.01):
    """Scale to [0, 1]; if the spread is negligible, clamp the raw values."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < threshold:
        return [max(0.0, min(1.0, v)) for v in values]
    return [(v - lo) / (hi - lo) for v in values]


def retrieve(conn, query_embedding, current_session_id):
    """Rank cross-session memories by recency + relevance."""
    candidates = get_candidates(conn, current_session_id)
    if not candidates:
        return []

    now = datetime.now(timezone.utc)
    recencies, relevances = [], []

    for cand in candidates:
        ts = cand["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hours = max(0.0, (now - ts).total_seconds() / 3600.0)
        recencies.append(0.995**hours)
        relevances.append(cosine(query_embedding, cand["embedding"]))

    norm_rec = min_max_scale(recencies)
    norm_rel = min_max_scale(relevances)

    results = []
    for i, cand in enumerate(candidates):
        results.append(
            {
                "session_id": cand["session_id"],
                "content": cand["content"],
                "recency": round(norm_rec[i], 4),
                "relevance": round(norm_rel[i], 4),
                "composite": round(norm_rec[i] + norm_rel[i], 4),
                "id": cand["id"],
            }
        )

    results.sort(key=lambda m: (m["composite"], m["recency"], m["id"]), reverse=True)
    return results[:TOP_K]


# --------------------------------------------------------------------------
# Reply generation (Groq)
# --------------------------------------------------------------------------


def generate_reply(user_message, memories):
    """Ask Groq for a reply, with retrieved memories injected into the prompt."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    if memories:
        context = "\n".join(
            f"[{i}] Session '{m['session_id']}': {m['content']}"
            for i, m in enumerate(memories, start=1)
        )
    else:
        context = "No prior relevant session memories found."

    prompt = (
        "You are a helpful AI assistant with access to long-term memory "
        "across user sessions.\n"
        "Below are relevant past memories retrieved from previous sessions:\n\n"
        f"{context}\n\n"
        "Answer the user's message accurately, using past session memory when "
        "relevant.\n"
        f"User message: {user_message}"
    )

    body = json.dumps(
        {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())

    return payload["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            message = (data.get("message") or "").strip()
            session_id = (data.get("session_id") or "").strip()

            if not message or not session_id:
                self._send(400, {"error": "message and session_id are required"})
                return

            conn = connect()
            try:
                query_vec = embed(message)
                add_entry(conn, session_id, "user", message, query_vec)

                memories = retrieve(conn, query_vec, session_id)
                reply = generate_reply(message, memories)

                add_entry(conn, session_id, "assistant", reply, None)
            finally:
                conn.close()

            self._send(200, {"reply": reply, "memories": memories})

        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
