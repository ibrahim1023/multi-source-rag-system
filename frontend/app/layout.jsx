import Link from "next/link";

import "./globals.css";

export const metadata = {
  title: "Multi-RAG Console",
  description: "Chat with sources, filter retrieval, and ingest new content."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div className="app-shell">
          <header className="app-header">
            <div className="brand">
              <span className="brand-mark">MR</span>
              <div>
                <div className="brand-title">Multi-RAG Console</div>
                <div className="brand-subtitle">Docs chat + ingestion</div>
              </div>
            </div>
            <nav className="app-nav">
              <Link className="nav-link" href="/">
                Chat
              </Link>
              <Link className="nav-link" href="/ingest">
                Ingest
              </Link>
            </nav>
            <div className="env-pill">Local API ready</div>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
