import { useState } from "react";
import LoaderOverlay from "./components/LoaderOverlay";
import "./App.css";
import legalAi from "./legal-ai.svg";

export default function App() {

  const [tab, setTab] = useState("query");
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleExplain = async () => {

    setLoading(true);

    // simulate backend call (replace later)
    setTimeout(() => {
      setLoading(false);
    }, 3000);
  };

  return (
    <div className="app">

      <LoaderOverlay
        show={loading}
        message="Analyzing your legal clause"
      />



      <div className="container">

        <div className="container">

  {/* TITLE CENTERED */}
  <div className="header">

    <h1 className="title">
      Legal<span className="animated-rag">RAG</span>
    </h1>

    <p className="subtitle">
      Plain-English explanations of Indian law
    </p>

  </div>


  {/* TWO COLUMN SECTION */}

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


    {/* RIGHT PANEL (YOUR EXISTING UI) */}

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

    </div>

  </div>

</div>
        </div>

    </div>
  );
}