import { useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Amplify } from "aws-amplify";
import { awsConfig } from "./aws-config.js";

import Login           from "./components/Login.jsx";
import SignUp           from "./components/SignUp.jsx";
import AdminPanel       from "./components/AdminPanel.jsx";
import StudentDashboard from "./components/StudentDashboard.jsx";
import StudentExam      from "./components/StudentExam.jsx";
import Results          from "./components/Results.jsx";
import ProtectedRoute   from "./components/ProtectedRoute.jsx";
import Leaderboard      from "./components/Leaderboard.jsx";

Amplify.configure(awsConfig);

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/leaderboard" element={<Leaderboard />} />
      <Route path="/signup" element={<Navigate to="/" replace />} />

      <Route path="/admin" element={
        <ProtectedRoute requiredRole="Admin">
          <AdminPanel />
        </ProtectedRoute>
      } />

      <Route path="/dashboard" element={
        <ProtectedRoute>
          <StudentDashboard />
        </ProtectedRoute>
      } />

      <Route path="/exam" element={
        <ProtectedRoute>
          <StudentExam />
        </ProtectedRoute>
      } />

      <Route path="/results" element={
        <ProtectedRoute>
          <Results />
        </ProtectedRoute>
      } />

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
