import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  getQuestions, createQuestion, updateQuestion, deleteQuestion,
  getGroups, createGroup, updateGroup, deleteGroup,
  getAdminUsers, setUserRole, revokeUserSession, assignUserGroup,
  uploadQuestionImage,
} from "../api.js";
import { signOut, getCurrentUserInfo } from "../auth.js";
import Footer from "./Footer.jsx";

// ─── Constants ────────────────────────────────────────────────────────────────

const EMPTY_QUESTION = {
  question: "", option_a: "", option_b: "", option_c: "", option_d: "",
  correct_answer: "A", subject: "", topic: "", explanation: "", image_url: "",
};

const EMPTY_GROUP = {
  name: "", description: "", topics: [], duration_minutes: 60,
  question_ids: [], models: [],
};

const MEDMCQA_TOPICS = [
  "Anatomy", "Physiology", "Biochemistry", "Pharmacology", "Pathology",
  "Microbiology", "Medicine", "Surgery", "Obstetrics & Gynaecology",
  "Paediatrics", "Psychiatry", "Ophthalmology", "ENT", "Radiology",
  "Forensic Medicine", "Social & Preventive Medicine", "Dental",
];

// ─── Question Modal ───────────────────────────────────────────────────────────

function QuestionModal({ initial, onSave, onClose }) {
  const [form, setForm] = useState({ ...EMPTY_QUESTION, ...(initial || {}) });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);

  const set = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }));

  const handleImageFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError("");
    setUploading(true);
    try {
      const { url } = await uploadQuestionImage(file);
      setForm((f) => ({ ...f, image_url: url }));
    } catch (err) {
      setError(err.response?.data?.error || "Image upload failed.");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const required = ["question", "option_a", "option_b", "option_c", "option_d", "correct_answer"];
    if (required.some((k) => !form[k].trim())) {
      setError("Please fill in all required fields.");
      return;
    }
    setSaving(true);
    setError("");
    try {
      await onSave(form);
      onClose();
    } catch (err) {
      setError(err.response?.data?.detail || "Save failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-xl font-semibold">{initial?.questionId ? "Edit Question" : "Add Question"}</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">&times;</button>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="label">Question <span className="text-red-500">*</span></label>
            <textarea className="input" rows={3} value={form.question} onChange={set("question")} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            {["a", "b", "c", "d"].map((letter) => (
              <div key={letter}>
                <label className="label">Option {letter.toUpperCase()} <span className="text-red-500">*</span></label>
                <input className="input" value={form[`option_${letter}`]} onChange={set(`option_${letter}`)} />
              </div>
            ))}
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="label">Correct Answer <span className="text-red-500">*</span></label>
              <select className="input" value={form.correct_answer} onChange={set("correct_answer")}>
                {["A", "B", "C", "D"].map((l) => <option key={l}>{l}</option>)}
              </select>
            </div>
            <div>
              <label className="label">Subject</label>
              <input className="input" value={form.subject} onChange={set("subject")} placeholder="e.g. Physiology" />
            </div>
            <div>
              <label className="label">Topic</label>
              <input className="input" value={form.topic} onChange={set("topic")} placeholder="e.g. Hormones" />
            </div>
          </div>
          <div>
            <label className="label">Figure (optional)</label>
            {form.image_url ? (
              <div className="flex items-start gap-3">
                <img
                  src={form.image_url}
                  alt="Question figure"
                  className="h-32 w-auto rounded border border-gray-200 object-contain bg-gray-50"
                />
                <button
                  type="button"
                  className="text-sm text-red-600 hover:text-red-800"
                  onClick={() => setForm((f) => ({ ...f, image_url: "" }))}
                >
                  Remove
                </button>
              </div>
            ) : (
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp,image/gif"
                onChange={handleImageFile}
                disabled={uploading}
                className="block text-sm text-gray-700"
              />
            )}
            {uploading && <p className="text-xs text-gray-500 mt-1">Uploading…</p>}
          </div>
          <div>
            <label className="label">Explanation</label>
            <textarea className="input" rows={4} value={form.explanation} onChange={set("explanation")} placeholder="Explanation for the correct answer (used for AI grading context)" />
          </div>
          {error && <p className="text-sm text-red-600">{error}</p>}
          <div className="flex gap-3 pt-2">
            <button type="submit" className="btn-primary flex-1" disabled={saving}>
              {saving ? "Saving…" : "Save Question"}
            </button>
            <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Group Modal ──────────────────────────────────────────────────────────────

function GroupModal({ initial, questions, onSave, onClose }) {
  const [form, setForm] = useState(() =>
    initial
      ? { ...initial, topics: initial.topics || [], question_ids: initial.question_ids || [], models: initial.models || [] }
      : { ...EMPTY_GROUP }
  );
  const [saving, setSaving] = useState(false);
  const [error, setError]   = useState("");
  const [modelInput, setModelInput] = useState("");
  const [qSearch, setQSearch]       = useState("");

  const set    = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }));
  const setNum = (field) => (e) => setForm((f) => ({ ...f, [field]: Number(e.target.value) }));

  const toggleTopic = (t) => setForm((f) => ({
    ...f,
    topics: f.topics.includes(t) ? f.topics.filter((x) => x !== t) : [...f.topics, t],
  }));

  const toggleQuestion = (qid) => setForm((f) => ({
    ...f,
    question_ids: f.question_ids.includes(qid)
      ? f.question_ids.filter((x) => x !== qid)
      : [...f.question_ids, qid],
  }));

  const addModel = () => {
    const m = modelInput.trim();
    if (m && !form.models.includes(m)) {
      setForm((f) => ({ ...f, models: [...f.models, m] }));
      setModelInput("");
    }
  };

  const removeModel = (m) => setForm((f) => ({ ...f, models: f.models.filter((x) => x !== m) }));

  const filteredQs = questions.filter((q) => {
    const s = qSearch.toLowerCase();
    return !s || q.question?.toLowerCase().includes(s) || q.subject?.toLowerCase().includes(s) || q.topic?.toLowerCase().includes(s);
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.name.trim()) { setError("Group name is required."); return; }
    setSaving(true);
    setError("");
    try {
      await onSave(form);
      onClose();
    } catch (err) {
      setError(err.response?.data?.detail || "Save failed.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-3xl max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-xl font-semibold">{initial?.groupId ? "Edit Exam Group" : "New Exam Group"}</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">&times;</button>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-5">

          {/* Basic info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Group Name <span className="text-red-500">*</span></label>
              <input className="input" value={form.name} onChange={set("name")} placeholder="e.g. Batch 2026 Exam 1" />
            </div>
            <div>
              <label className="label">Duration (minutes)</label>
              <input className="input" type="number" min={5} max={300} value={form.duration_minutes} onChange={setNum("duration_minutes")} />
            </div>
          </div>
          <div>
            <label className="label">Description</label>
            <textarea className="input" rows={2} value={form.description} onChange={set("description")} placeholder="Brief description of this exam group" />
          </div>

          {/* Topics */}
          <div>
            <label className="label">Medical Topics</label>
            <div className="flex flex-wrap gap-2 mt-1">
              {MEDMCQA_TOPICS.map((t) => (
                <button
                  type="button" key={t}
                  onClick={() => toggleTopic(t)}
                  className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors
                    ${form.topics.includes(t)
                      ? "bg-blue-600 text-white border-blue-600"
                      : "bg-white text-gray-600 border-gray-300 hover:border-blue-400"}`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* HF Models */}
          <div>
            <label className="label">HuggingFace Models for AI Grading</label>
            <div className="flex gap-2">
              <input
                className="input flex-1"
                placeholder="e.g. jamezoon/gemma-3-4b-it-medmcqa-lora"
                value={modelInput}
                onChange={(e) => setModelInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addModel(); } }}
              />
              <button type="button" className="btn-secondary" onClick={addModel}>Add</button>
            </div>
            {form.models.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {form.models.map((m, i) => (
                  <span key={m} className="flex items-center gap-1.5 px-3 py-1 bg-purple-50 text-purple-700 border border-purple-200 rounded-full text-xs font-mono">
                    <span className="text-purple-400">{i + 1}.</span> {m}
                    <button type="button" onClick={() => removeModel(m)} className="text-purple-400 hover:text-purple-700 leading-none ml-0.5">&times;</button>
                  </span>
                ))}
              </div>
            )}
            <p className="text-xs text-gray-400 mt-1">Models are tried in order. Falls back to default model if all fail.</p>
          </div>

          {/* Question assignment */}
          <div>
            <label className="label">Questions
              <span className="ml-2 text-blue-600 font-semibold">{form.question_ids.length} selected</span>
            </label>
            <input
              className="input mb-2"
              placeholder="Search questions by text, subject, or topic…"
              value={qSearch}
              onChange={(e) => setQSearch(e.target.value)}
            />
            <div className="border border-gray-200 rounded-lg max-h-52 overflow-y-auto divide-y divide-gray-100">
              {filteredQs.length === 0 ? (
                <p className="text-sm text-gray-400 p-4 text-center">No questions found.</p>
              ) : filteredQs.map((q) => {
                const selected = form.question_ids.includes(q.questionId);
                return (
                  <label
                    key={q.questionId}
                    className={`flex items-start gap-3 px-4 py-2.5 cursor-pointer hover:bg-gray-50 transition-colors
                      ${selected ? "bg-blue-50" : ""}`}
                  >
                    <input
                      type="checkbox"
                      className="mt-0.5 flex-shrink-0"
                      checked={selected}
                      onChange={() => toggleQuestion(q.questionId)}
                    />
                    <div className="min-w-0">
                      <p className="text-sm text-gray-800 leading-snug line-clamp-2">{q.question}</p>
                      <div className="flex gap-1.5 mt-0.5">
                        {q.subject && <span className="text-xs text-blue-600">{q.subject}</span>}
                        {q.topic && <span className="text-xs text-gray-400">{q.topic}</span>}
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
            <div className="flex gap-2 mt-1.5">
              <button
                type="button"
                className="text-xs text-blue-600 hover:underline"
                onClick={() => setForm((f) => ({ ...f, question_ids: questions.map((q) => q.questionId) }))}
              >
                Select all ({questions.length})
              </button>
              <span className="text-gray-300">|</span>
              <button
                type="button"
                className="text-xs text-gray-500 hover:underline"
                onClick={() => setForm((f) => ({ ...f, question_ids: [] }))}
              >
                Clear
              </button>
            </div>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}
          <div className="flex gap-3 pt-2">
            <button type="submit" className="btn-primary flex-1" disabled={saving}>
              {saving ? "Saving…" : "Save Group"}
            </button>
            <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Assign Group Modal ───────────────────────────────────────────────────────

function AssignGroupModal({ targetUser, groups, onSave, onClose }) {
  const [selectedGroup, setSelectedGroup] = useState(targetUser.groupId || "");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave(targetUser.sub, selectedGroup || null);
      onClose();
    } catch {
      alert("Failed to assign group.");
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Assign Exam Group</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">&times;</button>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-sm text-gray-600">
            Assigning exam group to:{" "}
            <span className="font-medium text-gray-900">{targetUser.name || targetUser.email}</span>
          </p>
          <div>
            <label className="label">Exam Group</label>
            <select className="input" value={selectedGroup} onChange={(e) => setSelectedGroup(e.target.value)}>
              <option value="">— No group (unassign) —</option>
              {groups.map((g) => (
                <option key={g.groupId} value={g.groupId}>
                  {g.name} ({(g.question_ids || []).length} questions, {g.duration_minutes || 60} min)
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-3 pt-2">
            <button className="btn-primary flex-1" onClick={handleSave} disabled={saving}>
              {saving ? "Saving…" : "Save"}
            </button>
            <button className="btn-secondary" onClick={onClose}>Cancel</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main AdminPanel ──────────────────────────────────────────────────────────

export default function AdminPanel() {
  const navigate = useNavigate();
  const [user, setUser]           = useState(null);
  const [activeTab, setActiveTab] = useState("questions");

  // Questions
  const [questions, setQuestions] = useState([]);
  const [qLoading, setQLoading]   = useState(true);
  const [qModal, setQModal]       = useState(null);
  const [qSearch, setQSearch]     = useState("");

  // Groups
  const [groups, setGroups]     = useState([]);
  const [gLoading, setGLoading] = useState(false);
  const [gFetched, setGFetched] = useState(false);
  const [gModal, setGModal]     = useState(null);

  // Users
  const [users, setUsers]         = useState([]);
  const [uLoading, setULoading]   = useState(false);
  const [uFetched, setUFetched]   = useState(false);
  const [assignModal, setAssignModal] = useState(null);

  useEffect(() => {
    (async () => {
      const info = await getCurrentUserInfo();
      setUser(info);
      fetchQuestions();
    })();
  }, []);

  // Lazy-load tab data on first visit
  useEffect(() => {
    if (activeTab === "groups" && !gFetched) fetchGroups();
    if (activeTab === "users" && !uFetched) {
      fetchUsers();
      if (!gFetched) fetchGroups();
    }
  }, [activeTab]);

  // ── Questions ──────────────────────────────────────────────────────────────

  const fetchQuestions = async () => {
    setQLoading(true);
    try {
      const { data } = await getQuestions();
      setQuestions(data);
    } catch {
      alert("Failed to load questions.");
    } finally {
      setQLoading(false);
    }
  };

  const handleQSave = async (form) => {
    if (qModal?.questionId) await updateQuestion(qModal.questionId, form);
    else await createQuestion(form);
    await fetchQuestions();
  };

  const handleQDelete = async (q) => {
    if (!confirm(`Delete: "${q.question.slice(0, 60)}…"?`)) return;
    await deleteQuestion(q.questionId);
    setQuestions((qs) => qs.filter((x) => x.questionId !== q.questionId));
  };

  // ── Groups ─────────────────────────────────────────────────────────────────

  const fetchGroups = async () => {
    setGLoading(true);
    try {
      const { data } = await getGroups();
      setGroups(data);
      setGFetched(true);
    } catch {
      alert("Failed to load exam groups.");
    } finally {
      setGLoading(false);
    }
  };

  const handleGSave = async (form) => {
    if (gModal?.groupId) await updateGroup(gModal.groupId, form);
    else await createGroup(form);
    await fetchGroups();
  };

  const handleGDelete = async (g) => {
    if (!confirm(`Delete group "${g.name}"? Students assigned to this group will be unassigned.`)) return;
    await deleteGroup(g.groupId);
    setGroups((gs) => gs.filter((x) => x.groupId !== g.groupId));
  };

  // ── Users ──────────────────────────────────────────────────────────────────

  const fetchUsers = async () => {
    setULoading(true);
    try {
      const { data } = await getAdminUsers();
      setUsers(data);
      setUFetched(true);
    } catch {
      alert("Failed to load users.");
    } finally {
      setULoading(false);
    }
  };

  const handleRoleChange = async (u, newRole) => {
    if (!confirm(`Change ${u.email} role to ${newRole}?`)) return;
    try {
      await setUserRole(u.sub, newRole);
      setUsers((us) => us.map((x) => x.sub === u.sub ? { ...x, role: newRole } : x));
    } catch {
      alert("Failed to change role.");
    }
  };

  const handleRevoke = async (u) => {
    if (!confirm(`Revoke all active sessions for ${u.email}?\nThey will be signed out immediately.`)) return;
    try {
      await revokeUserSession(u.sub);
      alert(`Sessions revoked for ${u.email}.`);
    } catch {
      alert("Failed to revoke session.");
    }
  };

  const handleAssignGroup = async (sub, groupId) => {
    await assignUserGroup(sub, groupId);
    const grp = groups.find((g) => g.groupId === groupId);
    setUsers((us) =>
      us.map((u) =>
        u.sub === sub ? { ...u, groupId: groupId || null, groupName: grp?.name || null } : u
      )
    );
  };

  const handleSignOut = async () => { await signOut(); navigate("/"); };

  const filteredQs = questions.filter((q) => {
    const s = qSearch.toLowerCase();
    return !s || q.question?.toLowerCase().includes(s) || q.subject?.toLowerCase().includes(s);
  });

  const TABS = [
    { key: "questions", label: "Questions", count: questions.length },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">

      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🔧</span>
          <div>
            <h1 className="text-xl font-bold text-gray-900">Admin Panel</h1>
            <p className="text-xs text-gray-500">{user?.name || user?.email || "Admin"}</p>
          </div>
        </div>
        <button className="btn-secondary" onClick={handleSignOut}>Sign Out</button>
      </header>

      {/* Tab bar */}
      <div className="bg-white border-b border-gray-200 px-6">
        <nav className="max-w-6xl mx-auto flex">
          {TABS.map(({ key, label, count }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors
                ${activeTab === key
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"}`}
            >
              {label}
              {count !== null && (
                <span className={`ml-2 px-1.5 py-0.5 rounded-full text-xs
                  ${activeTab === key ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-500"}`}>
                  {count}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      <main className="flex-1 max-w-6xl mx-auto px-6 py-6 w-full">

        {/* ══ Questions Tab ══════════════════════════════════════════════════ */}
        {activeTab === "questions" && (
          <>
            <div className="flex items-center justify-between mb-5">
              <input
                className="input max-w-sm"
                placeholder="Search questions or subjects…"
                value={qSearch}
                onChange={(e) => setQSearch(e.target.value)}
              />
              <button className="btn-primary" onClick={() => setQModal("new")}>+ Add Question</button>
            </div>

            {qLoading ? (
              <div className="text-center py-20 text-gray-400">Loading questions…</div>
            ) : filteredQs.length === 0 ? (
              <div className="text-center py-20">
                <div className="text-5xl mb-4">📋</div>
                <p className="text-gray-500">
                  {qSearch ? "No questions match your search." : "No questions yet. Add your first question!"}
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {filteredQs.map((q, idx) => (
                  <div key={q.questionId} className="card hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between gap-4">
                      {q.image_url && (
                        <img
                          src={q.image_url}
                          alt="Figure"
                          className="h-20 w-20 object-contain rounded border border-gray-200 bg-gray-50 flex-shrink-0"
                        />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="text-xs font-mono text-gray-400">#{idx + 1}</span>
                          {q.subject && (
                            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">{q.subject}</span>
                          )}
                          {q.topic && (
                            <span className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">{q.topic}</span>
                          )}
                          {q.image_url && (
                            <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded-full">🖼 figure</span>
                          )}
                        </div>
                        <p className="text-gray-900 font-medium leading-snug line-clamp-3">{q.question}</p>
                        <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-gray-600">
                          {["a", "b", "c", "d"].map((l) => (
                            <span key={l} className={q.correct_answer?.toUpperCase() === l.toUpperCase() ? "text-green-700 font-semibold" : ""}>
                              {l.toUpperCase()}. {q[`option_${l}`]}
                              {q.correct_answer?.toUpperCase() === l.toUpperCase() && " ✓"}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="flex gap-2 flex-shrink-0">
                        <button className="btn-secondary text-sm px-3 py-1" onClick={() => setQModal(q)}>Edit</button>
                        <button className="btn-danger text-sm px-3 py-1" onClick={() => handleQDelete(q)}>Delete</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* ══ Exam Groups Tab ════════════════════════════════════════════════ */}
        {activeTab === "groups" && (
          <>
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-base font-semibold text-gray-700">Exam Groups</h2>
                <p className="text-xs text-gray-400 mt-0.5">Create exam sets and assign them to students with a timer and AI model</p>
              </div>
              <button className="btn-primary" onClick={() => {
                if (questions.length === 0) fetchQuestions();
                setGModal("new");
              }}>
                + New Group
              </button>
            </div>

            {gLoading ? (
              <div className="text-center py-20 text-gray-400">Loading groups…</div>
            ) : groups.length === 0 ? (
              <div className="text-center py-20">
                <div className="text-5xl mb-4">📁</div>
                <p className="text-gray-500">No exam groups yet. Create one to get started.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {groups.map((g) => (
                  <div key={g.groupId} className="card hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                          <h3 className="font-semibold text-gray-900 text-base">{g.name}</h3>
                          <span className="px-2 py-0.5 bg-amber-100 text-amber-700 text-xs rounded-full font-medium">
                            {g.duration_minutes || 60} min
                          </span>
                          <span className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">
                            {(g.question_ids || []).length} question{(g.question_ids || []).length !== 1 ? "s" : ""}
                          </span>
                        </div>
                        {g.description && (
                          <p className="text-sm text-gray-500 mb-2">{g.description}</p>
                        )}
                        {(g.topics || []).length > 0 && (
                          <div className="flex flex-wrap gap-1.5 mb-2">
                            {g.topics.map((t) => (
                              <span key={t} className="px-2 py-0.5 bg-blue-50 text-blue-600 text-xs rounded-full border border-blue-100">{t}</span>
                            ))}
                          </div>
                        )}
                        {(g.models || []).length > 0 && (
                          <div className="flex flex-wrap gap-1.5">
                            {g.models.map((m, i) => (
                              <span key={m} className="px-2 py-0.5 bg-purple-50 text-purple-600 text-xs rounded-full font-mono border border-purple-100">
                                {i + 1}. {m}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex gap-2 flex-shrink-0">
                        <button
                          className="btn-secondary text-sm px-3 py-1"
                          onClick={() => {
                            if (questions.length === 0) fetchQuestions();
                            setGModal(g);
                          }}
                        >
                          Edit
                        </button>
                        <button className="btn-danger text-sm px-3 py-1" onClick={() => handleGDelete(g)}>Delete</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* ══ User Management Tab ════════════════════════════════════════════ */}
        {activeTab === "users" && (
          <>
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-base font-semibold text-gray-700">User Management</h2>
                <p className="text-xs text-gray-400 mt-0.5">Manage roles, exam group assignments, and active sessions</p>
              </div>
              <button className="btn-secondary text-sm" onClick={fetchUsers}>
                ↻ Refresh
              </button>
            </div>

            {uLoading ? (
              <div className="text-center py-20 text-gray-400">Loading users…</div>
            ) : users.length === 0 ? (
              <div className="text-center py-20 text-gray-400">No users found.</div>
            ) : (
              <div className="card p-0 overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 border-b border-gray-200">
                    <tr>
                      <th className="text-left px-5 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">User</th>
                      <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Role</th>
                      <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Exam Group</th>
                      <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Status</th>
                      <th className="px-4 py-3 text-xs uppercase tracking-wide text-right font-medium text-gray-500">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {users.map((u) => (
                      <tr key={u.sub} className="hover:bg-gray-50 transition-colors">
                        <td className="px-5 py-3">
                          <p className="font-medium text-gray-900">{u.name || u.email}</p>
                          {u.name && <p className="text-xs text-gray-400">{u.email}</p>}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium
                            ${u.role === "Admin"
                              ? "bg-red-100 text-red-700"
                              : "bg-blue-100 text-blue-700"}`}>
                            {u.role || "Student"}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-600">
                          {u.groupName
                            ? <span className="font-medium">{u.groupName}</span>
                            : <span className="text-gray-300 italic">Unassigned</span>}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium
                            ${u.status === "CONFIRMED"
                              ? "bg-green-100 text-green-700"
                              : "bg-gray-100 text-gray-500"}`}>
                            {u.status || "—"}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex gap-1.5 justify-end flex-wrap">
                            <button
                              className="text-xs px-2.5 py-1 rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50 transition-colors"
                              onClick={() => setAssignModal(u)}
                            >
                              Assign Group
                            </button>
                            {u.role !== "Admin" ? (
                              <button
                                className="text-xs px-2.5 py-1 rounded-lg border border-amber-300 text-amber-700 hover:bg-amber-50 transition-colors"
                                onClick={() => handleRoleChange(u, "Admin")}
                              >
                                Make Admin
                              </button>
                            ) : (
                              <button
                                className="text-xs px-2.5 py-1 rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50 transition-colors"
                                onClick={() => handleRoleChange(u, "Student")}
                              >
                                Make Student
                              </button>
                            )}
                            <button
                              className="text-xs px-2.5 py-1 rounded-lg border border-red-300 text-red-600 hover:bg-red-50 transition-colors"
                              onClick={() => handleRevoke(u)}
                            >
                              Revoke Session
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </main>

      {/* Modals */}
      {qModal && (
        <QuestionModal
          initial={qModal === "new" ? null : qModal}
          onSave={handleQSave}
          onClose={() => setQModal(null)}
        />
      )}
      {gModal && (
        <GroupModal
          initial={gModal === "new" ? null : gModal}
          questions={questions}
          onSave={handleGSave}
          onClose={() => setGModal(null)}
        />
      )}
      {assignModal && (
        <AssignGroupModal
          targetUser={assignModal}
          groups={groups}
          onSave={handleAssignGroup}
          onClose={() => setAssignModal(null)}
        />
      )}

      <Footer />
    </div>
  );
}
