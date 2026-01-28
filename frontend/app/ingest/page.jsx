"use client";

import { useEffect, useMemo, useState } from "react";

const DEFAULT_METADATA = `{
  "tags": ["policy"],
  "owner": "docs-team"
}`;

const SOURCE_TYPES = [
  { id: "markdown", label: "Markdown" },
  { id: "pdf", label: "PDF" },
  { id: "web", label: "Web" },
  { id: "code", label: "Code Docs" }
];

const SOURCE_CONFIG = {
  markdown: {
    title: "Retention Policy",
    origin: "/docs/retention.md",
    textLabel: "Markdown Content",
    textPlaceholder: "## Retention\nRetention policy for logs is 30 days.",
    metadataHint: "Add tags or ownership metadata."
  },
  pdf: {
    title: "Security Handbook",
    origin: "/docs/security-handbook.pdf",
    textLabel: "Extracted PDF Text",
    textPlaceholder: "Paste extracted PDF text here.",
    metadataHint: "Include page or section metadata if available."
  },
  web: {
    title: "Incident Runbook",
    origin: "https://docs.example.com/runbook",
    textLabel: "Web URL or Extracted Content",
    textPlaceholder: "Paste a URL or the extracted web page content here.",
    metadataHint: "Include canonical URL or crawl metadata."
  },
  code: {
    title: "API Reference",
    origin: "https://github.com/org/repo/docs/api.md",
    textLabel: "Extracted Code Docs Text",
    textPlaceholder: "Paste extracted README or docstring content here.",
    metadataHint: "Include repo path or module info."
  }
};

export default function IngestionPage() {
  const apiBase =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
  const [sourceType, setSourceType] = useState("markdown");
  const [title, setTitle] = useState(SOURCE_CONFIG.markdown.title);
  const [origin, setOrigin] = useState(SOURCE_CONFIG.markdown.origin);
  const [text, setText] = useState(
    "Retention policy for logs is 30 days. Audit reports are stored for 180 days."
  );
  const [metadata, setMetadata] = useState(DEFAULT_METADATA);
  const [showMetadata, setShowMetadata] = useState(false);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const config = SOURCE_CONFIG[sourceType];
    setTitle(config.title);
    setOrigin(config.origin);
    setFile(null);
  }, [sourceType]);

  const parsedMetadata = useMemo(() => {
    if (!metadata.trim()) {
      return { value: {}, error: "" };
    }
    try {
      return { value: JSON.parse(metadata), error: "" };
    } catch (err) {
      return { value: null, error: "Metadata must be valid JSON." };
    }
  }, [metadata]);

  const refreshStatus = async () => {
    setError("");
    try {
      const response = await fetch(`${apiBase}/ingest/status`);
      const data = await response.json();
      setStatus(data.jobs || []);
      const docResponse = await fetch(`${apiBase}/documents`);
      const docData = await docResponse.json();
      setDocuments(docData.documents || []);
    } catch (err) {
      setError("Could not reach the API. Is the backend running?");
    }
  };

  useEffect(() => {
    refreshStatus();
    const timer = setInterval(refreshStatus, 15000);
    return () => clearInterval(timer);
  }, []);

  const handleSubmit = async () => {
    setError("");
    setNotice("");
    if (parsedMetadata.error) {
      setError(parsedMetadata.error);
      return;
    }
    setBusy(true);
    try {
      let response;
      if (file && sourceType === "pdf") {
        const form = new FormData();
        form.append("title", title);
        form.append("origin", origin);
        form.append("metadata", JSON.stringify(parsedMetadata.value || {}));
        form.append("file", file);
        response = await fetch(`${apiBase}/ingest/pdf/file`, {
          method: "POST",
          body: form
        });
      } else {
        response = await fetch(`${apiBase}/ingest/${sourceType}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title,
            origin,
            text,
            metadata: parsedMetadata.value
          })
        });
      }
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Ingest failed");
      }
      const payload = await response.json();
      setNotice(`Indexed ${payload.chunks_indexed} chunks for ${payload.doc_id}.`);
      await refreshStatus();
    } catch (err) {
      setError(err.message || "Ingest failed. Check the API logs.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="page">
      <section className="hero">
        <span className="badge">Ingestion Console</span>
        <h1>Ingest sources in three clear steps.</h1>
        <p>
          Choose a source type, supply the content, and confirm the details.
          Job history and indexed sources update on the right.
        </p>
      </section>

      <section className="cards">
        <div className="card">
          <div className="section-title">Step 1 · Pick a Source</div>
          <div className="pill-group">
            {SOURCE_TYPES.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`pill ${sourceType === item.id ? "active" : ""}`}
                onClick={() => setSourceType(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div className="section-title">Step 2 · Provide Content</div>
          {sourceType === "pdf" ? (
            <div className="field">
              <label htmlFor="file">PDF File</label>
              <input
                id="file"
                type="file"
                onChange={(event) => {
                  const selected = event.target.files?.[0] || null;
                  setFile(selected);
                  if (selected) {
                    if (!title) {
                      setTitle(selected.name);
                    }
                    if (!origin) {
                      setOrigin(selected.name);
                    }
                  }
                }}
              />
              <span className="hint">Drop in a PDF to ingest as text.</span>
            </div>
          ) : (
            <div className="field">
              <label htmlFor="text">{SOURCE_CONFIG[sourceType].textLabel}</label>
              <textarea
                id="text"
                value={text}
                placeholder={SOURCE_CONFIG[sourceType].textPlaceholder}
                onChange={(event) => setText(event.target.value)}
              />
            </div>
          )}

          <div className="section-title">Step 3 · Confirm Details</div>
          <div className="row">
            <div className="field">
              <label htmlFor="title">Title</label>
              <input
                id="title"
                value={title}
                onChange={(event) => {
                  const nextTitle = event.target.value;
                  setTitle(nextTitle);
                  if (!origin) {
                    setOrigin(nextTitle.toLowerCase().replace(/\s+/g, "-"));
                  }
                }}
              />
            </div>
            <div className="field">
              <label htmlFor="origin">Origin</label>
              <input
                id="origin"
                value={origin}
                onChange={(event) => setOrigin(event.target.value)}
              />
            </div>
          </div>

          <button
            type="button"
            className="button secondary"
            onClick={() => setShowMetadata((prev) => !prev)}
          >
            {showMetadata ? "Hide Metadata" : "Add Metadata (Optional)"}
          </button>

          {showMetadata ? (
            <div className="field">
              <label htmlFor="metadata">Metadata (JSON)</label>
              <textarea
                id="metadata"
                value={metadata}
                onChange={(event) => setMetadata(event.target.value)}
              />
              <span className="hint">{SOURCE_CONFIG[sourceType].metadataHint}</span>
            </div>
          ) : null}

          {error ? <div className="error">{error}</div> : null}
          {notice ? <div className="hint">{notice}</div> : null}

          <div className="actions">
            <button className="button" onClick={handleSubmit} disabled={busy}>
              {busy ? "Ingesting..." : "Ingest Source"}
            </button>
            <button className="button secondary" onClick={refreshStatus} type="button">
              Refresh Status
            </button>
          </div>
        </div>

        <div className="card soft">
          <div className="section-title">Recent Jobs</div>
          <div className="status">
            {status.length === 0 ? (
              <span className="hint">No jobs yet. Submit your first source.</span>
            ) : (
              status.map((job) => (
                <div className="status-item" key={job.job_id}>
                  <strong>{job.title}</strong>
                  <span>{job.origin}</span>
                  <div className={`badge ${job.status === "failed" ? "failed" : ""}`}>
                    {job.status}
                  </div>
                  <span>
                    {job.source_type} - {new Date(job.created_at).toLocaleString()}
                  </span>
                  {job.error ? <span className="error">{job.error}</span> : null}
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="section-title">Ingested Sources</div>
        <div className="status">
          {documents.length === 0 ? (
            <span className="hint">No documents indexed yet.</span>
          ) : (
            documents.map((doc) => (
              <div className="status-item" key={doc.doc_id}>
                <strong>{doc.title}</strong>
                <span>{doc.origin}</span>
                <div className="badge">{doc.source_type}</div>
                {doc.tags?.length ? (
                  <span>Tags: {doc.tags.join(", ")}</span>
                ) : null}
              </div>
            ))
          )}
        </div>
      </section>
    </main>
  );
}
