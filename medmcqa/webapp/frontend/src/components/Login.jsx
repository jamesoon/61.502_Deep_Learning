import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { signIn, signInForce, signUp, confirmSignUp, resendSignUpCode, getCurrentUserInfo, isAuthenticated, ActiveSessionError } from "../auth.js";
import Footer from "./Footer.jsx";

function EyeIcon({ open }) {
  return open ? (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  ) : (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
    </svg>
  );
}

function FloatingInput({ label, type = "text", value, onChange, icon, required, autoFocus, rightSlot }) {
  return (
    <div className="relative border border-gray-300 rounded-lg focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent bg-white">
      {icon && (
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none">
          {icon}
        </div>
      )}
      <input
        type={type}
        className={`peer w-full bg-transparent rounded-lg px-4 pt-5 pb-2 text-sm focus:outline-none placeholder-transparent ${icon ? "pl-10" : "pl-4"} ${rightSlot ? "pr-10" : "pr-4"}`}
        placeholder={label}
        value={value}
        onChange={onChange}
        autoFocus={autoFocus}
        required={required}
      />
      <label className={`absolute top-1.5 text-xs text-gray-400 pointer-events-none transition-all
        peer-placeholder-shown:top-3.5 peer-placeholder-shown:text-sm
        peer-focus:top-1.5 peer-focus:text-xs peer-focus:text-blue-600
        ${icon ? "left-10" : "left-4"}`}>
        {label}{required && " *"}
      </label>
      {rightSlot && (
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          {rightSlot}
        </div>
      )}
    </div>
  );
}

function PersonIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
    </svg>
  );
}

function LockIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
    </svg>
  );
}

export default function Login() {
  const navigate  = useNavigate();
  const location  = useLocation();
  const [tab, setTab]           = useState("signin");   // "signin" | "signup"
  const [signupStep, setSignupStep] = useState("form"); // "form" | "verify" | "done"

  // Sign-in state
  const [email, setEmail]       = useState(location.state?.email || "");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw]     = useState(false);
  const sessionExpired = new URLSearchParams(location.search).get("reason") === "session_expired";
  const [signInError, setSignInError] = useState(
    sessionExpired ? "Your session has expired. Please sign in again." :
    location.state?.email ? "Account verified! You can now sign in." : ""
  );
  const [signInOk, setSignInOk] = useState(!sessionExpired && !!location.state?.email);
  const [signingIn, setSigningIn] = useState(false);
  const [sessionWarning, setSessionWarning] = useState(false);

  // Sign-up state
  const [suName, setSuName]     = useState("");
  const [suEmail, setSuEmail]   = useState("");
  const [suPw, setSuPw]         = useState("");
  const [suConfirm, setSuConfirm] = useState("");
  const [showSuPw, setShowSuPw] = useState(false);
  const [suCode, setSuCode]     = useState("");
  const [suError, setSuError]   = useState("");
  const [suLoading, setSuLoading] = useState(false);
  const [resent, setResent]     = useState(false);

  useEffect(() => {
    (async () => {
      if (await isAuthenticated()) {
        const info = await getCurrentUserInfo();
        navigate(info?.role === "Admin" ? "/admin" : "/dashboard", { replace: true });
      }
    })();
  }, []);

  const doNavigate = async () => {
    const info = await getCurrentUserInfo();
    navigate(info?.role === "Admin" ? "/admin" : "/dashboard", { replace: true });
  };

  const handleSignIn = async (e) => {
    e.preventDefault();
    setSigningIn(true);
    setSignInError("");
    setSignInOk(false);
    try {
      await signIn(email.trim(), password);
      await doNavigate();
    } catch (err) {
      if (err instanceof ActiveSessionError) {
        setSessionWarning(true);
      } else {
        const msg = err?.message || "";
        if (msg.includes("NotAuthorized") || msg.includes("Incorrect")) {
          setSignInError("Incorrect email or password.");
        } else if (msg.includes("UserNotFoundException")) {
          setSignInError("No account found with that email.");
        } else if (msg.includes("UserNotConfirmed")) {
          setSignInError("Please verify your email first. Check your inbox for the code.");
        } else {
          setSignInError(msg || "Sign-in failed. Please try again.");
        }
      }
    } finally {
      setSigningIn(false);
    }
  };

  const handleForceSignIn = async () => {
    setSessionWarning(false);
    setSigningIn(true);
    try {
      await signInForce(email.trim(), password);
      await doNavigate();
    } catch (err) {
      setSignInError(err?.message || "Sign-in failed. Please try again.");
    } finally {
      setSigningIn(false);
    }
  };

  const handleSignUp = async (e) => {
    e.preventDefault();
    setSuError("");
    if (suPw !== suConfirm) { setSuError("Passwords do not match."); return; }
    if (suPw.length < 8)    { setSuError("Password must be at least 8 characters."); return; }
    setSuLoading(true);
    try {
      await signUp(suEmail.trim(), suPw, suName.trim());
      setSignupStep("verify");
    } catch (err) {
      const msg = err?.message || "";
      if (msg.includes("UsernameExists") || msg.includes("already exists")) {
        setSuError("An account with this email already exists.");
      } else if (msg.includes("InvalidPassword") || msg.includes("password")) {
        setSuError("Password must have uppercase, lowercase, and a number.");
      } else {
        setSuError(msg || "Sign-up failed. Please try again.");
      }
    } finally {
      setSuLoading(false);
    }
  };

  const handleVerify = async (e) => {
    e.preventDefault();
    setSuError("");
    setSuLoading(true);
    try {
      await confirmSignUp(suEmail.trim(), suCode.trim());
      setSignupStep("done");
    } catch (err) {
      const msg = err?.message || "";
      if (msg.includes("CodeMismatch") || msg.includes("Invalid")) {
        setSuError("Incorrect code. Please check your email and try again.");
      } else if (msg.includes("Expired")) {
        setSuError("Code has expired. Please request a new one.");
      } else {
        setSuError(msg || "Verification failed.");
      }
    } finally {
      setSuLoading(false);
    }
  };

  const handleResend = async () => {
    setSuError(""); setResent(false);
    try { await resendSignUpCode(suEmail.trim()); setResent(true); }
    catch (err) { setSuError(err?.message || "Could not resend code."); }
  };

  const switchToSignIn = () => {
    setTab("signin");
    setEmail(suEmail);
    setSignInOk(true);
    setSignInError("Account verified! You can now sign in.");
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-100">
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-md">

          {/* Branding */}
          <div className="text-center mb-6">
            <div className="flex items-center justify-center gap-3 mb-4">
              <span className="text-5xl">🏥</span>
              <div className="text-left">
                <div className="text-xs font-bold tracking-widest text-gray-400 uppercase">SUTD</div>
                <div className="text-xs text-gray-400 leading-tight">Singapore University of<br/>Technology and Design</div>
              </div>
            </div>
            <h1 className="text-xl font-black tracking-wide text-gray-900 uppercase">
              MedMCQA — SUTD MSTR-DAIE
            </h1>
            <p className="text-sm text-gray-500 mt-1">Sign in or sign up to access the platform</p>
            <a
              href="/leaderboard"
              className="mt-2 inline-flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-800 hover:underline font-medium"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              View Model Leaderboard
            </a>
          </div>

          {/* Card */}
          <div className="bg-white rounded-2xl shadow-lg overflow-hidden">

            {/* Tabs */}
            <div className="flex border-b border-gray-200">
              {["signin", "signup"].map((t) => (
                <button
                  key={t}
                  onClick={() => { setTab(t); setSignupStep("form"); setSuError(""); setSignInError(""); setSignInOk(false); }}
                  className={`flex-1 py-4 text-sm font-medium transition-colors ${
                    tab === t
                      ? "text-blue-600 border-b-2 border-blue-600"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  {t === "signin" ? "Sign In" : "Sign Up"}
                </button>
              ))}
            </div>

            <div className="p-6">

              {/* ── SIGN IN ── */}
              {tab === "signin" && (
                <form onSubmit={handleSignIn} className="space-y-4">
                  <FloatingInput
                    label="Email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    icon={<PersonIcon />}
                    required
                    autoFocus
                  />
                  <FloatingInput
                    label="Password"
                    type={showPw ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    icon={<LockIcon />}
                    required
                    rightSlot={
                      <button type="button" onClick={() => setShowPw((v) => !v)} className="text-blue-500 hover:text-blue-700">
                        <EyeIcon open={showPw} />
                      </button>
                    }
                  />

                  {signInError && !signInOk && (
                    <p className="text-sm text-red-600">{signInError}</p>
                  )}
                  {signInOk && signInError && (
                    <p className="text-sm text-green-600 bg-green-50 border border-green-200 rounded-lg px-3 py-2">{signInError}</p>
                  )}

                  <button
                    type="submit"
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-xl transition-colors disabled:opacity-50"
                    disabled={signingIn}
                  >
                    {signingIn ? "Signing in…" : "Sign In"}
                  </button>

                  <p className="text-xs text-center text-gray-400">Sign in with your registered email and password</p>

                  <div className="border-t border-gray-100 pt-4">
                    <div className="flex items-center justify-center gap-2 mb-3">
                      <div className="flex-1 h-px bg-gray-200" />
                      <span className="text-xs text-gray-400 px-2">Auth method</span>
                      <div className="flex-1 h-px bg-gray-200" />
                    </div>
                    <div className="flex items-center justify-center px-4 py-2.5 border border-gray-200 rounded-xl bg-gray-50 text-sm font-medium text-gray-600">
                      <span className="mr-2">☁️</span> AWS Cognito (ACTIVE)
                    </div>
                  </div>
                </form>
              )}

              {/* ── SIGN UP ── */}
              {tab === "signup" && signupStep === "form" && (
                <form onSubmit={handleSignUp} className="space-y-4">
                  <FloatingInput label="Full Name" value={suName} onChange={(e) => setSuName(e.target.value)} icon={<PersonIcon />} required autoFocus />
                  <FloatingInput label="Email" type="email" value={suEmail} onChange={(e) => setSuEmail(e.target.value)} icon={<PersonIcon />} required />
                  <FloatingInput
                    label="Password"
                    type={showSuPw ? "text" : "password"}
                    value={suPw}
                    onChange={(e) => setSuPw(e.target.value)}
                    icon={<LockIcon />}
                    required
                    rightSlot={
                      <button type="button" onClick={() => setShowSuPw((v) => !v)} className="text-blue-500 hover:text-blue-700">
                        <EyeIcon open={showSuPw} />
                      </button>
                    }
                  />
                  <FloatingInput label="Confirm Password" type="password" value={suConfirm} onChange={(e) => setSuConfirm(e.target.value)} icon={<LockIcon />} required />

                  {suError && <p className="text-sm text-red-600">{suError}</p>}

                  <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-xl transition-colors disabled:opacity-50" disabled={suLoading}>
                    {suLoading ? "Creating account…" : "Create Account"}
                  </button>
                  <p className="text-xs text-center text-gray-400">New accounts are registered as students</p>
                </form>
              )}

              {tab === "signup" && signupStep === "verify" && (
                <form onSubmit={handleVerify} className="space-y-4">
                  <div className="text-center py-2">
                    <div className="text-4xl mb-3">📧</div>
                    <p className="text-sm text-gray-600">
                      We sent a 6-digit code to<br />
                      <span className="font-semibold text-gray-800">{suEmail}</span>
                    </p>
                  </div>
                  <FloatingInput
                    label="Verification Code"
                    value={suCode}
                    onChange={(e) => setSuCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                    required
                    autoFocus
                  />
                  {suError && <p className="text-sm text-red-600">{suError}</p>}
                  {resent && <p className="text-sm text-green-600">Code resent — check your inbox.</p>}
                  <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-xl transition-colors disabled:opacity-50" disabled={suLoading || suCode.length < 6}>
                    {suLoading ? "Verifying…" : "Verify Email"}
                  </button>
                  <button type="button" onClick={handleResend} className="w-full text-sm text-gray-400 hover:text-blue-600 transition-colors">
                    Didn't receive a code? Resend
                  </button>
                </form>
              )}

              {tab === "signup" && signupStep === "done" && (
                <div className="text-center py-4 space-y-4">
                  <div className="text-5xl">✅</div>
                  <div>
                    <p className="font-semibold text-gray-900">Account verified!</p>
                    <p className="text-sm text-gray-500 mt-1">Your student account is ready.</p>
                  </div>
                  <button onClick={switchToSignIn} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-xl transition-colors">
                    Sign In Now →
                  </button>
                </div>
              )}

            </div>
          </div>
        </div>
      </div>

      {/* Active session warning modal */}
      {sessionWarning && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-sm p-6">
            <div className="text-center mb-4">
              <div className="text-4xl mb-3">⚠️</div>
              <h3 className="text-lg font-semibold text-gray-900">Active Session Detected</h3>
              <p className="text-sm text-gray-500 mt-2">
                You already have an active session on another device or tab.
                Continuing will sign out that session and create a new one here.
              </p>
            </div>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleForceSignIn}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 rounded-xl transition-colors"
                disabled={signingIn}
              >
                {signingIn ? "Signing in…" : "Continue & Replace Session"}
              </button>
              <button
                onClick={() => setSessionWarning(false)}
                className="w-full border border-gray-200 text-gray-600 hover:bg-gray-50 font-medium py-2.5 rounded-xl transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <Footer />
    </div>
  );
}
