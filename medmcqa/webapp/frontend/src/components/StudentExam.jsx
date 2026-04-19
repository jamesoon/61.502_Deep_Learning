import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { getMyExam, submitExam } from "../api.js";
import { getCurrentUserInfo, signOut } from "../auth.js";
import Footer from "./Footer.jsx";

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export default function StudentExam() {
  const navigate = useNavigate();

  const [user, setUser]               = useState(null);
  const [exam, setExam]               = useState(null);     // group metadata
  const [questions, setQuestions]     = useState([]);
  const [answers, setAnswers]         = useState({});
  const [loading, setLoading]         = useState(true);
  const [submitting, setSubmitting]   = useState(false);
  const [currentIdx, setCurrentIdx]   = useState(0);
  const [error, setError]             = useState("");
  const [timeLeft, setTimeLeft]       = useState(null);     // seconds
  const [timeWarning, setTimeWarning] = useState(false);    // 5-min warning shown
  const [autoSubmitted, setAutoSubmitted] = useState(false);

  const timerRef = useRef(null);

  // Load exam on mount
  useEffect(() => {
    (async () => {
      const info = await getCurrentUserInfo();
      setUser(info);
      try {
        const { data } = await getMyExam();
        if (!data.assigned) {
          setError("No exam has been assigned to you yet. Please contact your administrator.");
          return;
        }
        const qs = data.questions || [];
        if (!qs.length) {
          setError("Your exam group has no questions. Please contact your administrator.");
          return;
        }
        setExam(data);
        setQuestions(qs);
        setTimeLeft((data.duration_minutes || 60) * 60);
      } catch {
        setError("Failed to load exam. Please refresh and try again.");
      } finally {
        setLoading(false);
      }
    })();
    return () => clearInterval(timerRef.current);
  }, []);

  // Countdown timer
  useEffect(() => {
    if (timeLeft === null || timeLeft <= 0) return;
    timerRef.current = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timerRef.current);
          return 0;
        }
        if (prev === 300) setTimeWarning(true);  // 5-min warning
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, [timeLeft !== null && timeLeft > 0 ? "started" : "stopped"]);

  // Auto-submit when timer hits 0
  useEffect(() => {
    if (timeLeft === 0 && !autoSubmitted && questions.length > 0) {
      setAutoSubmitted(true);
      doSubmit(true);
    }
  }, [timeLeft]);

  const selectAnswer = (qid, letter) =>
    setAnswers((a) => ({ ...a, [qid]: letter }));

  const doSubmit = useCallback(async (auto = false) => {
    if (!auto) {
      const unanswered = questions.filter((q) => !answers[q.questionId]);
      if (unanswered.length > 0) {
        if (!confirm(`You have ${unanswered.length} unanswered question(s). Submit anyway?`)) return;
      }
    }
    clearInterval(timerRef.current);
    setSubmitting(true);
    try {
      const { data } = await submitExam(answers, exam?.groupId);
      navigate("/results", { state: { submission: data } });
    } catch (err) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail;
      const msg = status === 400 && detail
        ? `Submission error: ${detail}`
        : status === 403
        ? "Access denied. Please sign in again."
        : err?.message || "Submission failed. Please try again.";
      alert(msg);
      setSubmitting(false);
    }
  }, [answers, exam, questions, navigate]);

  // ── Loading / Error states ────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="text-4xl mb-3 animate-pulse">📚</div>
          <p className="text-gray-500">Loading your exam…</p>
        </div>
      </div>
    );
  }

  if (error) {
    const handleSignOut = async () => { await signOut(); navigate("/"); };
    return (
      <div className="min-h-screen flex flex-col bg-gray-50">
        <div className="flex-1 flex items-center justify-center">
          <div className="card max-w-md text-center">
            <div className="text-4xl mb-3">📋</div>
            <p className="text-gray-700 font-medium">{error}</p>
            <div className="flex gap-3 justify-center mt-4">
              <button className="btn-secondary" onClick={() => navigate("/dashboard")}>
                ← Dashboard
              </button>
              <button className="btn-secondary" onClick={handleSignOut}>
                Sign Out
              </button>
            </div>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  const q            = questions[currentIdx];
  const answeredCount = Object.keys(answers).length;
  const progress     = (answeredCount / questions.length) * 100;
  const isAnswered   = (qid) => !!answers[qid];
  const isSelected   = (qid, letter) => answers[qid] === letter;

  const timerColor =
    timeLeft === null ? "text-gray-600"
    : timeLeft <= 60  ? "text-red-600 font-bold animate-pulse"
    : timeLeft <= 300 ? "text-amber-600 font-semibold"
    : "text-gray-700";

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">

      {/* 5-minute warning banner */}
      {timeWarning && timeLeft > 0 && (
        <div className="bg-amber-500 text-white text-center py-2 text-sm font-medium">
          ⚠️ 5 minutes remaining — please submit your exam soon!
          <button onClick={() => setTimeWarning(false)} className="ml-3 underline text-white/80 hover:text-white">Dismiss</button>
        </div>
      )}
      {timeLeft === 0 && (
        <div className="bg-red-600 text-white text-center py-2 text-sm font-medium">
          ⏰ Time's up! Your exam has been automatically submitted.
        </div>
      )}

      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm sticky top-0 z-10">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">{exam?.name || "Medical Exam"}</h1>
            <p className="text-sm text-gray-500">{user?.name || user?.email || "Student"}</p>
          </div>
          <div className="flex items-center gap-4">
            {/* Timer */}
            <div className={`flex items-center gap-1.5 ${timerColor}`}>
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="text-sm font-mono">{timeLeft !== null ? formatTime(timeLeft) : "--:--"}</span>
            </div>
            <div className="text-sm text-gray-600">
              <span className="font-semibold text-blue-600">{answeredCount}</span>
              <span className="text-gray-400"> / {questions.length} answered</span>
            </div>
            <button className="btn-primary" onClick={() => doSubmit(false)} disabled={submitting || timeLeft === 0}>
              {submitting ? "Submitting…" : "Submit Exam"}
            </button>
          </div>
        </div>
        {/* Progress bar */}
        <div className="max-w-5xl mx-auto mt-3">
          <div className="h-1.5 bg-gray-200 rounded-full">
            <div className="h-1.5 bg-blue-500 rounded-full transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </header>

      <div className="flex-1 max-w-5xl mx-auto px-6 py-6 flex gap-6 w-full">

        {/* Question navigator */}
        <aside className="w-48 flex-shrink-0">
          <div className="card sticky top-28">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Questions</h3>
            <div className="grid grid-cols-5 gap-1.5">
              {questions.map((question, idx) => (
                <button
                  key={question.questionId}
                  onClick={() => setCurrentIdx(idx)}
                  className={`w-8 h-8 text-xs font-medium rounded-lg transition-colors
                    ${idx === currentIdx ? "bg-blue-600 text-white" : ""}
                    ${isAnswered(question.questionId) && idx !== currentIdx ? "bg-green-100 text-green-700" : ""}
                    ${!isAnswered(question.questionId) && idx !== currentIdx ? "bg-gray-100 text-gray-600 hover:bg-gray-200" : ""}
                  `}
                >
                  {idx + 1}
                </button>
              ))}
            </div>
            <div className="mt-3 space-y-1 text-xs text-gray-500">
              <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded bg-green-100 border border-green-300" />Answered</div>
              <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded bg-gray-100 border border-gray-300" />Not answered</div>
            </div>
            {exam?.description && (
              <div className="mt-3 pt-3 border-t border-gray-100">
                <p className="text-xs text-gray-400 leading-relaxed">{exam.description}</p>
              </div>
            )}
          </div>
        </aside>

        {/* Current question */}
        <main className="flex-1">
          <div className="card">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-xs font-mono text-gray-400">Q{currentIdx + 1} of {questions.length}</span>
              {q.subject && (
                <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">{q.subject}</span>
              )}
              {q.topic && (
                <span className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">{q.topic}</span>
              )}
            </div>
            {q.image_url && (
              <div className="mb-5 flex justify-center">
                <img
                  src={q.image_url}
                  alt="Question figure"
                  className="max-h-80 w-auto rounded-lg border border-gray-200 bg-gray-50 object-contain"
                />
              </div>
            )}
            <p className="text-lg text-gray-900 font-medium leading-relaxed mb-6">{q.question}</p>

            <div className="space-y-3">
              {["A", "B", "C", "D"].map((letter) => {
                const optKey  = `option_${letter.toLowerCase()}`;
                const selected = isSelected(q.questionId, letter);
                return (
                  <button
                    key={letter}
                    onClick={() => selectAnswer(q.questionId, letter)}
                    disabled={timeLeft === 0}
                    className={`w-full text-left px-4 py-3 rounded-xl border-2 transition-all flex items-center gap-3
                      ${selected ? "border-blue-500 bg-blue-50 text-blue-900" : "border-gray-200 hover:border-gray-300 hover:bg-gray-50 text-gray-700"}
                      disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    <span className={`w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0
                      ${selected ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-600"}`}>
                      {letter}
                    </span>
                    <span className="text-sm">{q[optKey]}</span>
                  </button>
                );
              })}
            </div>

            <div className="flex justify-between mt-6 pt-4 border-t border-gray-100">
              <button className="btn-secondary" disabled={currentIdx === 0} onClick={() => setCurrentIdx((i) => i - 1)}>
                ← Previous
              </button>
              <button
                className={currentIdx === questions.length - 1 ? "btn-success" : "btn-primary"}
                onClick={() => {
                  if (currentIdx < questions.length - 1) setCurrentIdx((i) => i + 1);
                  else doSubmit(false);
                }}
                disabled={submitting || timeLeft === 0}
              >
                {currentIdx === questions.length - 1 ? (submitting ? "Submitting…" : "Submit Exam →") : "Next →"}
              </button>
            </div>
          </div>
        </main>
      </div>

      <Footer />
    </div>
  );
}
