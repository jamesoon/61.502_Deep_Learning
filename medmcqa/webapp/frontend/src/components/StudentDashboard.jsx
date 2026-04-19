import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getMyExam, getMySubmissions, getSubmission } from "../api.js";
import { signOut, getCurrentUserInfo } from "../auth.js";
import Footer from "./Footer.jsx";

function formatDate(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("en-SG", {
      day: "2-digit", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function ScoreBadge({ score, status }) {
  if (status !== "COMPLETE") {
    return (
      <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-amber-100 text-amber-700 border border-amber-200">
        Grading…
      </span>
    );
  }
  const pct = Math.round(Number(score) || 0);
  const cls = pct >= 70 ? "bg-green-100 text-green-700 border-green-200"
            : pct >= 50 ? "bg-yellow-100 text-yellow-700 border-yellow-200"
            :             "bg-red-100 text-red-700 border-red-200";
  return (
    <span className={`px-2.5 py-1 rounded-full text-xs font-bold border ${cls}`}>
      {pct}%
    </span>
  );
}

export default function StudentDashboard() {
  const navigate = useNavigate();

  const [user, setUser]               = useState(null);
  const [exam, setExam]               = useState(null);   // getMyExam response
  const [submissions, setSubmissions] = useState([]);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState("");

  useEffect(() => {
    (async () => {
      const info = await getCurrentUserInfo();
      setUser(info);
      try {
        const [examRes, subsRes] = await Promise.all([getMyExam(), getMySubmissions()]);
        setExam(examRes.data);
        setSubmissions(subsRes.data || []);
      } catch {
        setError("Failed to load dashboard. Please refresh.");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleSignOut = async () => {
    await signOut();
    navigate("/");
  };

  const handleViewResults = async (sub) => {
    // If complete, navigate straight away with lightweight object
    if (sub.status === "COMPLETE") {
      // Fetch full submission (with results array) before navigating
      try {
        const { data } = await getSubmission(sub.submissionId);
        navigate("/results", { state: { submission: data } });
      } catch {
        alert("Failed to load submission details.");
      }
      return;
    }
    // Still grading — navigate and let Results.jsx poll
    navigate("/results", { state: { submission: sub } });
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="text-4xl mb-3 animate-pulse">📚</div>
          <p className="text-gray-500">Loading your dashboard…</p>
        </div>
      </div>
    );
  }

  const examAssigned = exam?.assigned;

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">

      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🎓</span>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Student Dashboard</h1>
              <p className="text-sm text-gray-500">{user?.name || user?.email || "Student"}</p>
            </div>
          </div>
          <button className="btn-secondary" onClick={handleSignOut}>Sign Out</button>
        </div>
      </header>

      <main className="flex-1 max-w-4xl mx-auto px-6 py-8 w-full space-y-8">

        {error && (
          <div className="card border-l-4 border-l-red-500 text-red-700 text-sm">{error}</div>
        )}

        {/* ── Current Exam ─────────────────────────────────────────────── */}
        <section>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">Current Exam</h2>
          {examAssigned ? (
            <div className="card">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-semibold text-gray-900">{exam.name}</h3>
                  {exam.description && (
                    <p className="text-sm text-gray-500 mt-0.5">{exam.description}</p>
                  )}
                  <div className="flex flex-wrap gap-3 mt-3 text-sm text-gray-600">
                    <span className="flex items-center gap-1.5">
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {exam.duration_minutes || 60} minutes
                    </span>
                    <span className="flex items-center gap-1.5">
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                      {(exam.questions || []).length} questions
                    </span>
                  </div>
                  {(exam.topics || []).length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {exam.topics.map((t) => (
                        <span key={t} className="px-2 py-0.5 bg-blue-50 text-blue-600 text-xs rounded-full border border-blue-100">{t}</span>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  className="btn-primary flex-shrink-0"
                  onClick={() => navigate("/exam")}
                >
                  Start Exam →
                </button>
              </div>
            </div>
          ) : (
            <div className="card text-center py-8">
              <div className="text-4xl mb-3">📋</div>
              <p className="text-gray-600 font-medium">No exam has been assigned to you yet.</p>
              <p className="text-sm text-gray-400 mt-1">Please contact your administrator.</p>
            </div>
          )}
        </section>

        {/* ── Past Attempts ─────────────────────────────────────────────── */}
        <section>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Past Attempts
            {submissions.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full font-normal normal-case">
                {submissions.length}
              </span>
            )}
          </h2>

          {submissions.length === 0 ? (
            <div className="card text-center py-8">
              <div className="text-4xl mb-3">📂</div>
              <p className="text-gray-400 text-sm">No exam attempts yet.</p>
            </div>
          ) : (
            <div className="card p-0 overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="text-left px-5 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Date</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Score</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Result</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Status</th>
                    <th className="px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {submissions.map((sub) => {
                    const correct = Number(sub.correct) || 0;
                    const total   = Number(sub.total_questions) || 0;
                    return (
                      <tr key={sub.submissionId} className="hover:bg-gray-50 transition-colors">
                        <td className="px-5 py-3 text-gray-700">{formatDate(sub.submittedAt)}</td>
                        <td className="px-4 py-3">
                          <ScoreBadge score={sub.score_percent} status={sub.status} />
                        </td>
                        <td className="px-4 py-3 text-gray-600">
                          {sub.status === "COMPLETE"
                            ? <span>{correct} / {total} correct</span>
                            : <span className="text-gray-400">—</span>}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium
                            ${sub.status === "COMPLETE"
                              ? "bg-green-100 text-green-700"
                              : "bg-amber-100 text-amber-700"}`}>
                            {sub.status}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            className="text-xs px-3 py-1.5 rounded-lg border border-blue-300 text-blue-600 hover:bg-blue-50 transition-colors font-medium"
                            onClick={() => handleViewResults(sub)}
                          >
                            View Results →
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

      </main>

      <Footer />
    </div>
  );
}
