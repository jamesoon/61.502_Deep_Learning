import { useEffect, useState, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { getSubmission } from "../api.js";
import Footer from "./Footer.jsx";

const LETTER_COLOR = {
  correct: "bg-green-100 text-green-800 border-green-300",
  wrong:   "bg-red-100 text-red-800 border-red-300",
  neutral: "bg-gray-100 text-gray-700 border-gray-200",
};

const POLL_INTERVAL_MS = 5000;

function ModelScoreboard({ studentScore, studentLabel = "You", modelScores }) {
  const rows = [
    { name: studentLabel, accuracy_pct: studentScore, isStudent: true },
    ...modelScores.map((m) => ({
      name: m.short_name || m.model,
      accuracy_pct: m.accuracy_pct,
      correct: m.correct,
      total: m.total,
    })),
  ].sort((a, b) => b.accuracy_pct - a.accuracy_pct);

  return (
    <div className="card mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-base font-semibold text-gray-800">Model Scoreboard</h2>
        <span className="text-xs text-gray-500">Accuracy vs ground truth on this exam</span>
      </div>
      <div className="space-y-2">
        {rows.map((row, i) => (
          <div key={row.name + i} className="flex items-center gap-3">
            <span className="w-6 text-xs font-mono text-gray-400">#{i + 1}</span>
            <span className={`flex-1 text-sm ${row.isStudent ? "font-semibold text-blue-700" : "text-gray-700"}`}>
              {row.name} {row.isStudent && <span className="text-[10px] uppercase tracking-wider text-blue-500 ml-1">student</span>}
            </span>
            <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden max-w-xs">
              <div
                className={`${row.isStudent ? "bg-blue-500" : "bg-emerald-500"} h-2`}
                style={{ width: `${Math.max(2, row.accuracy_pct)}%` }}
              />
            </div>
            <span className="w-20 text-right text-sm font-semibold text-gray-900">{row.accuracy_pct}%</span>
            {row.total != null && (
              <span className="w-14 text-right text-xs text-gray-500">{row.correct}/{row.total}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ScoreRing({ score }) {
  const color = score >= 70 ? "text-green-600" : score >= 50 ? "text-yellow-600" : "text-red-600";
  const label = score >= 70 ? "Excellent!" : score >= 50 ? "Good effort" : "Keep studying";
  return (
    <div className="flex flex-col items-center">
      <div className={`text-6xl font-bold ${color}`}>{score}%</div>
      <p className={`mt-1 font-medium ${color}`}>{label}</p>
    </div>
  );
}

function QuestionResult({ result, idx }) {
  const [expanded, setExpanded] = useState(false);
  const options = ["A", "B", "C", "D"];

  return (
    <div className={`card border-l-4 ${result.is_correct ? "border-l-green-500" : "border-l-red-500"}`}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-mono text-gray-400">Q{idx + 1}</span>
            {result.is_correct
              ? <span className="badge-correct">✓ Correct</span>
              : <span className="badge-wrong">✗ Incorrect</span>
            }
          </div>
          {result.image_url && (
            <img
              src={result.image_url}
              alt="Question figure"
              className="mb-3 max-h-56 w-auto rounded-lg border border-gray-200 bg-gray-50 object-contain"
            />
          )}
          <p className="text-gray-900 font-medium">{result.question}</p>

          {/* Options summary */}
          <div className="mt-3 grid grid-cols-2 gap-2">
            {options.map((letter) => {
              const optKey    = `option_${letter.toLowerCase()}`;
              const isCorrect = result.correct_answer === letter;
              const isStudent = result.student_answer === letter;
              const isWrong   = isStudent && !isCorrect;
              let cls = LETTER_COLOR.neutral, indicator = "";
              if (isCorrect) { cls = LETTER_COLOR.correct; indicator = " ✓"; }
              if (isWrong)   { cls = LETTER_COLOR.wrong;   indicator = " ✗"; }
              return (
                <div key={letter} className={`text-xs px-3 py-1.5 rounded-lg border flex items-center gap-1.5 ${cls}`}>
                  <span className="font-bold">{letter}.</span>
                  <span className="truncate">{result[optKey]}</span>
                  {indicator && <span className="ml-auto font-bold flex-shrink-0">{indicator}</span>}
                </div>
              );
            })}
          </div>

          {!result.is_correct && (
            <div className="mt-2 text-sm text-gray-600">
              Your answer: <span className="font-semibold text-red-600">{result.student_answer || "—"}</span>
              {" · "}
              Correct: <span className="font-semibold text-green-600">{result.correct_answer}</span>
            </div>
          )}
        </div>
      </div>

      {/* AI Explanation */}
      <div className="mt-3 pt-3 border-t border-gray-100">
        {result.explanation ? (
          <>
            <button
              className="text-sm text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1"
              onClick={() => setExpanded((e) => !e)}
            >
              🤖 AI Explanation {expanded ? "▲" : "▼"}
            </button>
            {expanded && (
              <div className="mt-2 text-sm text-gray-700 bg-blue-50 rounded-lg p-3 leading-relaxed border border-blue-100">
                {result.explanation}
              </div>
            )}
          </>
        ) : (
          <p className="text-xs text-gray-400 italic">AI explanation generating…</p>
        )}
      </div>
    </div>
  );
}

export default function Results() {
  const navigate  = useNavigate();
  const location  = useLocation();
  const [data, setData]     = useState(location.state?.submission ?? null);
  const [filter, setFilter] = useState("all");
  const [polling, setPolling] = useState(false);
  const pollRef = useRef(null);

  // Poll until status === COMPLETE (explanations filled in)
  useEffect(() => {
    if (!data) { navigate("/dashboard"); return; }
    if (data.status === "COMPLETE") return;

    setPolling(true);
    pollRef.current = setInterval(async () => {
      try {
        const { data: fresh } = await getSubmission(data.submissionId);
        setData(fresh);
        if (fresh.status === "COMPLETE") {
          clearInterval(pollRef.current);
          setPolling(false);
        }
      } catch {
        // Keep polling — transient errors are expected
      }
    }, POLL_INTERVAL_MS);

    return () => clearInterval(pollRef.current);
  }, [data?.submissionId]);

  if (!data) return null;

  const results  = data.results || [];
  const filtered = results.filter((r) => {
    if (filter === "correct") return r.is_correct;
    if (filter === "wrong")   return !r.is_correct;
    return true;
  });

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">📊 Exam Results</h1>
            <p className="text-sm text-gray-500">{data.student_name || "Student"}</p>
          </div>
          <div className="flex items-center gap-3">
            {polling && (
              <span className="text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded-full border border-amber-200">
                ⏳ AI explanations generating…
              </span>
            )}
            <button className="btn-secondary" onClick={() => navigate("/dashboard")}>
              ← Dashboard
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-4xl mx-auto px-6 py-8 w-full">
        {/* Score summary */}
        <div className="card mb-6">
          <div className="flex flex-col sm:flex-row items-center gap-8">
            <ScoreRing score={data.score_percent ?? 0} />
            <div className="flex-1 grid grid-cols-3 gap-4 text-center">
              <div className="p-4 bg-gray-50 rounded-xl">
                <div className="text-3xl font-bold text-gray-900">{data.total_questions}</div>
                <div className="text-sm text-gray-500 mt-1">Total</div>
              </div>
              <div className="p-4 bg-green-50 rounded-xl">
                <div className="text-3xl font-bold text-green-600">{data.correct}</div>
                <div className="text-sm text-gray-500 mt-1">Correct</div>
              </div>
              <div className="p-4 bg-red-50 rounded-xl">
                <div className="text-3xl font-bold text-red-600">
                  {(data.total_questions ?? 0) - (data.correct ?? 0)}
                </div>
                <div className="text-sm text-gray-500 mt-1">Incorrect</div>
              </div>
            </div>
          </div>
        </div>

        {/* Model scoreboard — student + HF models on the same questions */}
        {Array.isArray(data.model_scores) && data.model_scores.length > 0 && (
          <ModelScoreboard
            studentScore={data.score_percent ?? 0}
            studentLabel={data.studentName || "You"}
            modelScores={data.model_scores}
          />
        )}

        {/* Filter tabs */}
        <div className="flex gap-2 mb-4">
          {[
            { key: "all",     label: `All (${results.length})` },
            { key: "correct", label: `Correct (${results.filter((r) => r.is_correct).length})` },
            { key: "wrong",   label: `Incorrect (${results.filter((r) => !r.is_correct).length})` },
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                filter === key
                  ? "bg-blue-600 text-white"
                  : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Per-question results */}
        <div className="space-y-3">
          {filtered.map((result, idx) => (
            <QuestionResult
              key={result.questionId}
              result={result}
              idx={results.indexOf(result)}
            />
          ))}
        </div>
      </main>
      <Footer />
    </div>
  );
}
