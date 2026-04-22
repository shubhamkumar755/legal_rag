import React from "react";

export default function LoaderOverlay({
  show = false,
  message = "Analyzing your legal clause",
}) {
  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">

      {/* background blur (light theme version) */}
      <div className="absolute inset-0 bg-[#f6f3ef]/80 backdrop-blur-sm"></div>

      <div
        className="relative flex gap-8 items-center rounded-xl shadow-lg border border-[#e6dfd8]"
        style={{
          width: 720,
          background: "transparent",
          padding: "40px",
        }}
      >

        {/* GOLD LEGAL LOADER */}
        <div className="relative w-40 h-40 flex items-center justify-center">

          {/* outer pulse */}
          <div className="absolute w-40 h-40 rounded-full bg-[#d4af37]/15 animate-ping"></div>

          {/* inner pulse */}
          <div className="absolute w-28 h-28 rounded-full bg-[#b8962e]/25 animate-ping delay-200"></div>

          {/* core orb */}
          <div
            className="w-16 h-16 rounded-full"
            style={{
              background:
                "linear-gradient(135deg, #8A6E2F, #d4af37, #b8962e)",
              boxShadow: "0 0 25px rgba(212,175,55,.4)",
            }}
          ></div>

        </div>

        {/* TEXT SECTION */}
        <div>

          <div
            className="mb-2"
            style={{
              fontSize: "20px",
              fontFamily: "Playfair Display",
              color: "#2d2622",
            }}
          >
            Hold tight — {message}
          </div>

          <div
            className="max-w-md"
            style={{
              fontSize: "14px",
              color: "#6b625c",
              lineHeight: 1.6,
            }}
          >
            LegalRAG is retrieving relevant legal provisions and verifying
            references from Indian Acts to generate a reliable explanation.
          </div>

          {/* GOLD PROGRESS BAR */}
          <div className="mt-6">

            <div
              style={{
                width: 420,
                height: 6,
                borderRadius: 999,
                background: "#eee7dd",
                overflow: "hidden",
                position: "relative",
              }}
            >

              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background:
                    "linear-gradient(90deg,#8A6E2F,#d4af37,#b8962e)",
                  animation: "progress 2.2s linear infinite",
                }}
              ></div>

            </div>

          </div>

        </div>

      </div>

      {/* animation styles */}
      <style>{`

        @keyframes progress {
          0% { transform: translateX(-100%); }
          50% { transform: translateX(0%); }
          100% { transform: translateX(100%); }
        }

        .delay-200 {
          animation-delay: 200ms;
        }

      `}</style>

    </div>
  );
}