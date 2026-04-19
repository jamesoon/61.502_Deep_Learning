import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { signUp, confirmSignUp, resendSignUpCode } from "../auth.js";
import Footer from "./Footer.jsx";

export default function SignUp() {
  const navigate = useNavigate();
  const [step, setStep]         = useState("form");   // "form" | "verify" | "done"
  const [name, setName]         = useState("");
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm]   = useState("");
  const [code, setCode]         = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [resent, setResent]     = useState(false);

  const handleSignUp = async (e) => {
    e.preventDefault();
    setError("");
    if (password !== confirm) { setError("Passwords do not match."); return; }
    if (password.length < 8)  { setError("Password must be at least 8 characters."); return; }
    setLoading(true);
    try {
      await signUp(email.trim(), password, name.trim());
      setStep("verify");
    } catch (err) {
      const msg = err?.message || "";
      if (msg.includes("UsernameExistsException") || msg.includes("already exists")) {
        setError("An account with this email already exists.");
      } else if (msg.includes("InvalidPassword") || msg.includes("password")) {
        setError("Password does not meet requirements (min 8 chars, uppercase, lowercase, number).");
      } else {
        setError(msg || "Sign-up failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await confirmSignUp(email.trim(), code.trim());
      setStep("done");
    } catch (err) {
      const msg = err?.message || "";
      if (msg.includes("CodeMismatch") || msg.includes("Invalid verification")) {
        setError("Incorrect code. Please check your email and try again.");
      } else if (msg.includes("ExpiredCode")) {
        setError("Code has expired. Please request a new one.");
      } else {
        setError(msg || "Verification failed.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    setError("");
    setResent(false);
    try {
      await resendSignUpCode(email.trim());
      setResent(true);
    } catch (err) {
      setError(err?.message || "Could not resend code.");
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md">

          {/* Header */}
          <div className="text-center mb-8">
            <div className="text-5xl mb-3">🏥</div>
            <h1 className="text-3xl font-bold text-gray-900">MedMCQA</h1>
            <p className="text-gray-500 mt-1">AI-Powered Medical Exam Platform</p>
            <p className="text-xs text-gray-400 mt-1">SUTD MSTR-DAIE Deep Learning Project</p>
          </div>

          {/* Step: Registration form */}
          {step === "form" && (
            <div className="card">
              <Link to="/" className="text-sm text-gray-500 hover:text-gray-700 mb-4 flex items-center gap-1">
                ← Back to sign in
              </Link>
              <h2 className="text-xl font-semibold mb-1">Create Student Account</h2>
              <p className="text-sm text-gray-500 mb-5">Register to take AI-powered medical exams.</p>

              <form onSubmit={handleSignUp} className="space-y-4">
                <div>
                  <label className="label">Full Name</label>
                  <input
                    type="text"
                    className="input"
                    placeholder="Dr. Jane Smith"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    autoFocus
                    required
                  />
                </div>
                <div>
                  <label className="label">Email</label>
                  <input
                    type="email"
                    className="input"
                    placeholder="your@email.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label className="label">Password</label>
                  <input
                    type="password"
                    className="input"
                    placeholder="Min 8 characters"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label className="label">Confirm Password</label>
                  <input
                    type="password"
                    className="input"
                    placeholder="Repeat password"
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                    required
                  />
                </div>

                {error && <p className="text-sm text-red-600">{error}</p>}

                <button type="submit" className="btn-success w-full" disabled={loading}>
                  {loading ? "Creating account…" : "Create Account"}
                </button>
              </form>

              <p className="text-xs text-center text-gray-400 mt-4">
                Already have an account?{" "}
                <Link to="/" className="text-blue-600 hover:underline">Sign in</Link>
              </p>
            </div>
          )}

          {/* Step: Email verification */}
          {step === "verify" && (
            <div className="card">
              <div className="text-center mb-6">
                <div className="text-4xl mb-3">📧</div>
                <h2 className="text-xl font-semibold">Check your email</h2>
                <p className="text-sm text-gray-500 mt-2">
                  We sent a 6-digit verification code to<br />
                  <span className="font-medium text-gray-700">{email}</span>
                </p>
              </div>

              <form onSubmit={handleVerify} className="space-y-4">
                <div>
                  <label className="label">Verification Code</label>
                  <input
                    type="text"
                    className="input text-center text-2xl tracking-widest font-mono"
                    placeholder="123456"
                    value={code}
                    onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                    maxLength={6}
                    autoFocus
                    required
                  />
                </div>

                {error && <p className="text-sm text-red-600">{error}</p>}
                {resent && <p className="text-sm text-green-600">Code resent — check your inbox.</p>}

                <button type="submit" className="btn-success w-full" disabled={loading || code.length < 6}>
                  {loading ? "Verifying…" : "Verify Email"}
                </button>
              </form>

              <button
                onClick={handleResend}
                className="w-full text-sm text-gray-500 hover:text-blue-600 mt-3 transition-colors"
              >
                Didn't receive a code? Resend
              </button>
            </div>
          )}

          {/* Step: Success */}
          {step === "done" && (
            <div className="card text-center">
              <div className="text-5xl mb-4">✅</div>
              <h2 className="text-xl font-semibold text-gray-900 mb-2">Account verified!</h2>
              <p className="text-sm text-gray-500 mb-6">
                Your student account is ready. Sign in to start your exam.
              </p>
              <button
                className="btn-success w-full"
                onClick={() => navigate("/", { state: { email } })}
              >
                Sign In Now →
              </button>
            </div>
          )}

        </div>
      </div>
      <Footer />
    </div>
  );
}
