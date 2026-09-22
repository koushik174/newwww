"use client";

/**
 * Legal RAG — single-page client (Next.js App Router stub).
 *
 * POSTs the question to the FastAPI backend's `POST /query` and renders the
 * grounded answer with clickable [S#] citations, the source list with
 * page/paragraph provenance and a verified flag, detected conflicts, and a
 * "needs human review" banner when the answer failed validation.
 */

import { useMemo, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Citation = {
  marker: string;
  source_id: string;
  source_type: string;
  title: string;
  page?: number | null;
  paragraph?: string | null;
  verified: boolean;
  reason?: string | null;
};

type Conflict = {
  kind: string;
  description: string;
  markers: string[];
};

type QueryResponse = {
  query: string;
  routed_to: string[];
  answer: string;
  citations: Citation[];
  conflicts: Conflict[];
  valid: boolean;
  abstained: boolean;
  regenerated: number;
  needs_human_review: boolean;
  warnings: string[];
};

export default function Page() {
  const [query, setQuery] = useState(
    "What is the deadline to file an EEOC charge for employment discrimination?"
  );
  const [resp, setResp] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sourceRefs = useRef<Record<string, HTMLLIElement | null>>({});

  async function ask() {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!r.ok) throw new Error(`API ${r.status}`);
      setResp(await r.json());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function scrollToSource(marker: string) {
    const el = sourceRefs.current[marker];
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.style.background = "#fff3bf";
      setTimeout(() => (el.style.background = ""), 1200);
    }
  }

  // Split the answer into text + clickable [S#] chips.
  const answerNodes = useMemo(() => {
    if (!resp) return null;
    const parts = resp.answer.split(/(\[S\d+\])/g);
    return parts.map((part, i) => {
      const m = part.match(/^\[S(\d+)\]$/);
      if (m) {
        const marker = `S${m[1]}`;
        return (
          <button
            key={i}
            onClick={() => scrollToSource(marker)}
            style={{
              color: "#1c7ed6",
              cursor: "pointer",
              border: "none",
              background: "none",
              padding: 0,
              font: "inherit",
            }}
            title="Jump to source"
          >
            {part}
          </button>
        );
      }
      return <span key={i}>{part}</span>;
    });
  }, [resp]);

  return (
    <main style={{ maxWidth: 820, margin: "40px auto", padding: 16, fontFamily: "system-ui" }}>
      <h1>Legal RAG</h1>
      <p style={{ color: "#666" }}>
        Grounded answers over US statutes, the CFR, and case law — every sentence cited.
      </p>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        rows={3}
        style={{ width: "100%", padding: 10, fontSize: 16 }}
      />
      <button onClick={ask} disabled={loading} style={{ marginTop: 8, padding: "8px 16px" }}>
        {loading ? "Thinking…" : "Ask"}
      </button>

      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}

      {resp && (
        <section style={{ marginTop: 24 }}>
          {!resp.valid && (
            <div style={{ background: "#ffe3e3", padding: 12, borderRadius: 6, marginBottom: 12 }}>
              ⚠️ This answer failed grounding validation and was routed to human review.
            </div>
          )}

          <div style={{ fontSize: 13, color: "#666" }}>
            routed to: {resp.routed_to.join(", ")} · valid: {String(resp.valid)} ·
            regenerated: {resp.regenerated}
          </div>

          <h2>Answer</h2>
          <p style={{ lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{answerNodes}</p>

          {resp.conflicts.length > 0 && (
            <>
              <h3>Detected conflicts</h3>
              <ul>
                {resp.conflicts.map((c, i) => (
                  <li key={i}>
                    <strong>[{c.kind}]</strong> {c.description}
                  </li>
                ))}
              </ul>
            </>
          )}

          <h3>Sources</h3>
          <ol>
            {resp.citations.map((c) => (
              <li key={c.marker} ref={(el) => (sourceRefs.current[c.marker] = el)}>
                <strong>[{c.marker}]</strong> {c.title}{" "}
                {c.page != null && <em>(p.{c.page}{c.paragraph ? `, ¶${c.paragraph}` : ""})</em>}{" "}
                {c.verified ? "✓ verified" : "✗ unverified"}
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
