"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const SOURCE_OPTIONS = [
  { id: "", label: "All sources" },
  { id: "markdown", label: "Markdown" },
  { id: "pdf", label: "PDF" },
  { id: "web", label: "Web" },
  { id: "code", label: "Code Docs" }
];

const SAMPLE_PROMPTS = [
  "What is the log retention policy?",
  "Summarize the incident response steps.",
  "Which modules mention rate limits?"
];

const buildFilter = (filters) => {
  const payload = {};
  if (filters.sourceType) {
    payload.source_type = filters.sourceType;
  }
  if (filters.owner) {
    payload.owner = filters.owner;
  }
  if (filters.accessScope) {
    payload.access_scope = filters.accessScope;
  }
  if (filters.origin) {
    payload.origin = filters.origin;
  }
  return Object.keys(payload).length ? payload : null;
};

export default function ChatPage() {
  const apiBase =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([
    {
      id: "intro",
      role: "assistant",
      content:
        "Ask a question about your ingested sources. I will only answer from retrieved context and cite evidence.",
      meta: {
        confidence: 0.84,
        mode: "grounded",
        refused: false
      }
    }
  ]);
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState({
    sourceType: "",
    owner: "",
    accessScope: "",
    origin: ""
  });
  const [citations, setCitations] = useState([]);
  const [claims, setClaims] = useState([]);
  const [selectedCitationId, setSelectedCitationId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [followUp, setFollowUp] = useState("");
  const activeRequestIdRef = useRef(0);

  useEffect(() => {
    const loadDocuments = async () => {
      try {
        const response = await fetch(`${apiBase}/documents`);
        const payload = await response.json();
        setDocuments(payload.documents || []);
      } catch (err) {
        setDocuments([]);
      }
    };
    loadDocuments();
  }, [apiBase]);

  const selectedCitation = useMemo(() => {
    return citations.find((item) => item.chunk_id === selectedCitationId) || null;
  }, [citations, selectedCitationId]);

  const sendMessage = async (text) => {
    if (!text.trim() || loading) {
      return;
    }
    setError("");
    setFollowUp("");
    setCitations([]);
    setClaims([]);
    setSelectedCitationId("");
    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    const requestId = Date.now();
    activeRequestIdRef.current = requestId;
    setLoading(true);

    try {
      const response = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          metadata_filter: buildFilter(filters)
        })
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Chat failed.");
      }
      const payload = await response.json();
      if (requestId !== activeRequestIdRef.current) {
        return;
      }
      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: payload.answer,
        meta: {
          confidence: payload.confidence,
          mode: payload.mode,
          refused: payload.refused
        }
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setCitations(payload.citations || []);
      setClaims(payload.claims || []);
      setSelectedCitationId(payload.citations?.[0]?.chunk_id || "");
      setFollowUp(payload.follow_up_question || "");
    } catch (err) {
      setError(err.message || "Chat failed. Check the API logs.");
    } finally {
      if (requestId === activeRequestIdRef.current) {
        setLoading(false);
      }
    }
  };

  return (
    <main className="page">
      <section className="hero">
        <span className="badge">Docs Chat</span>
        <h1>Ask grounded questions across every ingested source.</h1>
        <p>
          Filter retrieval by source metadata, review citations, and keep answers
          anchored to the exact evidence.
        </p>
      </section>

      <section className="chat-layout">
        <div className="card chat-panel">
          <div className="section-title">Conversation</div>
          <div className="chat-window">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message message-${message.role}`}
              >
                <div className="message-body">{message.content}</div>
                {message.role === "assistant" ? (
                  <div className="message-meta">
                    <span className="meta-pill">
                      Confidence: {(message.meta?.confidence || 0).toFixed(2)}
                    </span>
                    <span className="meta-pill">Mode: {message.meta?.mode}</span>
                    {message.meta?.refused ? (
                      <span className="meta-pill alert">Refused</span>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ))}
            {loading ? (
              <div className="message message-assistant">
                <div className="message-body">Thinking through the evidence...</div>
              </div>
            ) : null}
          </div>
          {error ? <div className="error">{error}</div> : null}
          {followUp ? (
            <div className="hint">Follow-up: {followUp}</div>
          ) : null}

          <div className="composer">
            <textarea
              value={query}
              placeholder="Ask about retention, incidents, or doc ownership..."
              onChange={(event) => setQuery(event.target.value)}
            />
            <div className="composer-actions">
              <div className="sample-prompts">
                {SAMPLE_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    className="pill"
                    type="button"
                    onClick={() => sendMessage(prompt)}
                    disabled={loading}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
              <button
                className="button"
                type="button"
                onClick={() => sendMessage(query)}
                disabled={loading}
              >
                {loading ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </div>

        <aside className="side-panel">
          <div className="card soft">
            <div className="section-title">Filters</div>
            <div className="field">
              <label htmlFor="sourceType">Source Type</label>
              <select
                id="sourceType"
                value={filters.sourceType}
                onChange={(event) =>
                  setFilters((prev) => ({
                    ...prev,
                    sourceType: event.target.value
                  }))
                }
              >
                {SOURCE_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="owner">Owner</label>
              <input
                id="owner"
                value={filters.owner}
                placeholder="docs-team"
                onChange={(event) =>
                  setFilters((prev) => ({
                    ...prev,
                    owner: event.target.value
                  }))
                }
              />
            </div>
            <div className="field">
              <label htmlFor="accessScope">Access Scope</label>
              <input
                id="accessScope"
                value={filters.accessScope}
                placeholder="internal"
                onChange={(event) =>
                  setFilters((prev) => ({
                    ...prev,
                    accessScope: event.target.value
                  }))
                }
              />
            </div>
            <div className="field">
              <label htmlFor="origin">Focus Document</label>
              <select
                id="origin"
                value={filters.origin}
                onChange={(event) =>
                  setFilters((prev) => ({
                    ...prev,
                    origin: event.target.value
                  }))
                }
              >
                <option value="">All documents</option>
                {documents.map((doc) => (
                  <option key={doc.doc_id} value={doc.origin}>
                    {doc.title}
                  </option>
                ))}
              </select>
            </div>
            <div className="hint">
              Filters are exact matches against metadata fields.
            </div>
          </div>

          <div className="card citations-panel">
            <div className="section-title">Citations</div>
            <div className="citation-list">
              {citations.length === 0 ? (
                <span className="hint">No citations yet. Send a query.</span>
              ) : (
                citations.map((citation) => (
                  <button
                    key={citation.chunk_id}
                    type="button"
                    className={`citation-item ${
                      citation.chunk_id === selectedCitationId ? "active" : ""
                    }`}
                    onClick={() => setSelectedCitationId(citation.chunk_id)}
                  >
                    <strong>{citation.title}</strong>
                    {citation.section_path ? (
                      <span>Section: {citation.section_path}</span>
                    ) : null}
                  </button>
                ))
              )}
            </div>
            <div className="evidence">
              <div className="section-title">Evidence Viewer</div>
              {selectedCitation ? (
                <div className="evidence-card">
                  <div className="evidence-header">
                    <strong>{selectedCitation.title}</strong>
                  </div>
                  <p>{selectedCitation.snippet}</p>
                  {selectedCitation.section_path ? (
                    <span className="hint">{selectedCitation.section_path}</span>
                  ) : null}
                </div>
              ) : (
                <span className="hint">Select a citation to preview evidence.</span>
              )}
            </div>
            {claims.length ? (
              <div className="claim-list">
                <div className="section-title">Claims</div>
                {claims.map((claim, index) => (
                  <div className="claim-item" key={`${claim.text}-${index}`}>
                    <span className="claim-index">Claim {index + 1}</span>
                    <p>{claim.text}</p>
                    <span className="hint">
                      {claim.citations.length} supporting citation(s)
                    </span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </aside>
      </section>
    </main>
  );
}
