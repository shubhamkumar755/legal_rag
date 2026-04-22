import { useState } from "react";
import LoaderOverlay from "./components/LoaderOverlay";
import "./App.css";
import legalAi from "./legal-ai.svg";

export default function App() {

  const [tab, setTab] = useState("query");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleExplain = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      // QUERY MODE
      if (tab === "query") {
        const res = await fetch("http://127.0.0.1:8000/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ question: query }),
        });

        const data = await res.json();
        setResponse(data);
      }

      // UPLOAD MODE (basic version)
      if (tab === "upload" && file) {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("http://127.0.0.1:8000/analyze", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        setResponse(data);
      }

    } catch (err) {
      console.error(err);
      setError("Something went wrong. Backend probably isn't running.");
    }

    setLoading(false);
  };

  return (
    <div className="app">

      <LoaderOverlay
        show={loading}
        message="Analyzing your legal clause"
      />

      <div className="container">

        <div className="header">
          <h1 className="title">
            Legal<span className="animated-rag">RAG</span>
          </h1>

          <p className="subtitle">
            Plain-English explanations of Indian law
          </p>
        </div>

        <div className="main-layout">

          {/* LEFT PANEL */}
          <div className="left-panel">

            <img
              src={legalAi}
              alt="Legal AI illustration"
              className="overview-svg"
            />

            <h2 className="overview-title">
              Understand legal clauses instantly
            </h2>

            <p className="overview-text">
              LegalRAG explains complex legal clauses in plain English and supports
              explanations using verified references from Indian Acts to reduce
              hallucinations and improve trust.
            </p>

          </div>

          {/* RIGHT PANEL */}
          <div className="right-panel">

            <div className="tabs">

              <button
                className={tab === "query" ? "active" : ""}
                onClick={() => setTab("query")}
              >
                Type Query
              </button>

              <button
                className={tab === "upload" ? "active" : ""}
                onClick={() => setTab("upload")}
              >
                Upload Document
              </button>

            </div>

            {/* QUERY TAB */}
            {tab === "query" && (
              <div className="card">

                <label>
                  Enter your legal question or clause
                </label>

                <textarea
                  rows="6"
                  placeholder="Type clause here..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />

                <button
                  className="explain-btn"
                  onClick={handleExplain}
                >
                  Explain
                </button>

              </div>
            )}

            {/* UPLOAD TAB */}
            {tab === "upload" && (
              <div className="card">

                <label>
                  Upload a PDF or DOCX
                </label>

                <label className="upload-btn">
                  Choose File
                  <input
                    type="file"
                    onChange={(e) => setFile(e.target.files[0])}
                    hidden
                  />
                </label>

                <span className="file-name">
                  {file ? file.name : "No file chosen"}
                </span>

                <button
                  className="explain-btn"
                  onClick={handleExplain}
                >
                  Explain
                </button>

              </div>
            )}

            {/* RESPONSE DISPLAY */}
            {response && tab === "query" && (
              <div className="response-box">
                <h3>Answer</h3>
                <p>{response.answer}</p>

                {response.confidence && (
                  <p>
                    <strong>Confidence:</strong> {response.confidence.label}
                  </p>
                )}

                {response.warnings && response.warnings.length > 0 && (
                  <div>
                    <strong>Warnings:</strong>
                    <ul>
                      {response.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* DOCUMENT RESPONSE */}
            {response && tab === "upload" && (
              <div className="response-box">
                <h3>Document Analysis</h3>

                {response.map((clause, i) => (
                  <div key={i} className="clause-box">
                    <p><strong>Clause {clause.clause_id}</strong></p>
                    <p>{clause.explanation}</p>

                    {clause.warnings?.length > 0 && (
                      <ul>
                        {clause.warnings.map((w, j) => (
                          <li key={j}>{w}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}

              </div>
            )}

            {/* ERROR */}
            {error && (
              <div className="error-box">
                {error}
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}