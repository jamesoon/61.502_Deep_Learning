import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import { isAuthenticated, getRole } from "../auth.js";

/**
 * Wraps a route requiring authentication.
 * If requiredRole is given, also checks the Cognito group role.
 *
 * Usage:
 *   <ProtectedRoute requiredRole="Admin"><AdminPanel /></ProtectedRoute>
 *   <ProtectedRoute><StudentExam /></ProtectedRoute>
 */
export default function ProtectedRoute({ children, requiredRole = null }) {
  const [state, setState] = useState("loading"); // "loading" | "ok" | "unauth" | "forbidden"

  useEffect(() => {
    (async () => {
      const authed = await isAuthenticated();
      if (!authed) { setState("unauth"); return; }
      if (requiredRole) {
        const role = await getRole();
        if (role !== requiredRole) { setState("forbidden"); return; }
      }
      setState("ok");
    })();
  }, [requiredRole]);

  if (state === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-400 text-sm">Checking session…</p>
      </div>
    );
  }
  if (state === "unauth")    return <Navigate to="/" replace />;
  if (state === "forbidden") return <Navigate to="/exam" replace />;
  return children;
}
