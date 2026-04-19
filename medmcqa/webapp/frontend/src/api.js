/**
 * API client for the AWS API Gateway backend.
 * Automatically attaches the Cognito ID token as Bearer auth.
 */

import axios from "axios";
import { getIdToken, signOut } from "./auth.js";
import { API_BASE } from "./aws-config.js";

const api = axios.create({ baseURL: API_BASE });

// Attach Cognito JWT on every request
api.interceptors.request.use(async (config) => {
  const token = await getIdToken();
  if (token) {
    config.headers["Authorization"] = `Bearer ${token}`;
  } else {
    // No token — force sign-out so user gets redirected to login
    await signOut().catch(() => {});
    window.location.href = "/";
    return Promise.reject(new Error("Session expired. Please sign in again."));
  }
  return config;
});

// Handle 401 responses (token accepted by interceptor but rejected by API Gateway)
api.interceptors.response.use(
  (res) => res,
  async (err) => {
    if (err?.response?.status === 401) {
      await signOut().catch(() => {});
      window.location.href = "/?reason=session_expired";
    }
    return Promise.reject(err);
  }
);

// ── Questions ─────────────────────────────────────────────────────────────────
export const getQuestions    = ()          => api.get("/questions");
export const createQuestion  = (data)      => api.post("/questions", data);
export const updateQuestion  = (id, data)  => api.put(`/questions/${id}`, data);
export const deleteQuestion  = (id)        => api.delete(`/questions/${id}`);

// ── Image upload (admin) ──────────────────────────────────────────────────────
/** Read a File as base64 (without the data: prefix). */
const _readAsBase64 = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => {
    const [, b64] = (reader.result || "").toString().split(",", 2);
    resolve(b64 || "");
  };
  reader.onerror = () => reject(reader.error);
  reader.readAsDataURL(file);
});

/** Upload an image file; returns { url, key }. */
export const uploadQuestionImage = async (file) => {
  const data_base64 = await _readAsBase64(file);
  const { data } = await api.post("/upload-image", {
    filename:     file.name,
    content_type: file.type,
    data_base64,
  });
  return data;
};

// ── Exam groups (admin) ───────────────────────────────────────────────────────
export const getGroups      = ()              => api.get('/groups');
export const createGroup    = (data)          => api.post('/groups', data);
export const getGroup       = (id)            => api.get(`/groups/${id}`);
export const updateGroup    = (id, data)      => api.put(`/groups/${id}`, data);
export const deleteGroup    = (id)            => api.delete(`/groups/${id}`);
export const assignGroup    = (id, data)      => api.post(`/groups/${id}/assign`, data);
export const getGroupUsers  = (id)            => api.get(`/groups/${id}/users`);

// ── User management (admin) ───────────────────────────────────────────────────
export const getAdminUsers      = ()              => api.get('/admin/users');
export const setUserRole        = (uid, role)     => api.post(`/admin/users/${uid}/role`, { role });
export const revokeUserSession  = (uid)           => api.post(`/admin/users/${uid}/revoke`);
export const assignUserGroup    = (uid, groupId)  => api.post(`/admin/users/${uid}/group`, { groupId });

// ── Student exam ──────────────────────────────────────────────────────────────
/** GET /my-exam — returns assigned group with questions + duration */
export const getMyExam = () => api.get('/my-exam');

/** GET /my-submissions — returns all past submissions for the current user (no results array) */
export const getMySubmissions = () => api.get('/my-submissions');

// ── Exam submission ───────────────────────────────────────────────────────────
/** POST /submit — returns immediately with score (status=GRADING) */
export const submitExam = (answers, groupId) =>
  api.post("/submit", { answers, groupId });

/** GET /submissions/{id} — poll until status=COMPLETE */
export const getSubmission = (submissionId) =>
  api.get(`/submissions/${submissionId}`);
