/**
 * Leaderboard — public page showcasing MedMCQA model benchmark results.
 * No auth required. Accessible at /leaderboard.
 */

import { useState } from "react";
import Footer from "./Footer.jsx";

// ── Static benchmark data ─────────────────────────────────────────────────────

const HF_BASE = "https://huggingface.co/jamezoon";

const MCAT_MODELS = [
  {
    key: "qwen35_9b_cot",
    name: "Qwen3.5-9B",
    tag: "Baseline • CoT",
    type: "baseline",
    size: "9B",
    quant: "Q4_K_M GGUF",
    mcat_acc: 0.7841,
    mcat_f1: 0.7631,
    mcat_std: 0.1309,
    latency_s: 104.9,
    hf_repo: null,
    note: "Chain-of-thought prompting",
  },
  {
    key: "qwen35_9b_base",
    name: "Qwen3.5-9B",
    tag: "Baseline",
    type: "baseline",
    size: "9B",
    quant: "Q4_K_M GGUF",
    mcat_acc: 0.7738,
    mcat_f1: 0.7720,
    mcat_std: 0.0248,
    latency_s: 2.8,
    hf_repo: null,
  },
  {
    key: "qwen35_9b_lora",
    name: "Qwen3.5-9B MedMCQA",
    tag: "Fine-tuned LoRA",
    type: "finetuned",
    size: "9B",
    quant: "Q4_K_M GGUF",
    mcat_acc: 0.7725,
    mcat_f1: 0.7727,
    mcat_std: 0.0252,
    latency_s: 16.8,
    hf_repo: `${HF_BASE}/qwen3-5-9b-medmcqa-gguf`,
  },
  {
    key: "qwen3_14b_lora",
    name: "Qwen3-14B MedMCQA",
    tag: "Fine-tuned LoRA",
    type: "finetuned",
    size: "14B",
    quant: "BF16",
    mcat_acc: 0.7626,
    mcat_f1: 0.7630,
    mcat_std: 0.0246,
    latency_s: 23.3,
    hf_repo: `${HF_BASE}/qwen3-14b-medmcqa-lora`,
  },
  {
    key: "qwen3_8b_cot",
    name: "Qwen3-8B",
    tag: "Baseline • CoT",
    type: "baseline",
    size: "8B",
    quant: "Q4_K_M GGUF",
    mcat_acc: 0.7303,
    mcat_f1: 0.7274,
    mcat_std: 0.2056,
    latency_s: 38.7,
    hf_repo: null,
    note: "Chain-of-thought prompting",
  },
  {
    key: "qwen3_8b_base",
    name: "Qwen3-8B",
    tag: "Baseline",
    type: "baseline",
    size: "8B",
    quant: "Q4_K_M GGUF",
    mcat_acc: 0.7178,
    mcat_f1: 0.7256,
    mcat_std: 0.0259,
    latency_s: 9.8,
    hf_repo: null,
  },
  {
    key: "gemma3_4b_base",
    name: "Gemma-3-4B-it",
    tag: "Baseline",
    type: "baseline",
    size: "4B",
    quant: "Q4_0 GGUF",
    mcat_acc: 0.5494,
    mcat_f1: 0.5462,
    mcat_std: 0.0202,
    latency_s: 0.17,
    hf_repo: null,
  },
];

const MEDMCQA_MODELS = [
  {
    key: "gemma3_4b_lora",
    name: "Gemma-3-4B-it MedMCQA",
    tag: "Fine-tuned LoRA",
    type: "finetuned",
    size: "4B",
    quant: "BF16",
    dev_acc: 0.4542,
    dev_macro_acc: 0.4327,
    total: 4183,
    hf_repo: `${HF_BASE}/gemma-3-4b-it-medmcqa-lora`,
  },
  {
    key: "deberta_cross",
    name: "DeBERTa-v3-large",
    tag: "Cross-Encoder Classifier",
    type: "discriminative",
    size: "0.4B",
    quant: "FP32",
    dev_acc: 0.3223,   // val accuracy at best checkpoint
    dev_macro_acc: null,
    total: 4183,
    hf_repo: `${HF_BASE}/medmcqa-pubmedbert-mcqa`,
    note: "4-way MCQ classification head. Val acc at best checkpoint (4 epochs).",
  },
  {
    key: "qwen3_14b_base",
    name: "Qwen3-14B",
    tag: "Zero-shot Baseline",
    type: "baseline",
    size: "14B",
    quant: "BF16",
    dev_acc: 0.2742,
    dev_macro_acc: 0.3001,
    total: 4183,
    hf_repo: null,
  },
  {
    key: "qwen35_9b_base_med",
    name: "Qwen3-9B",
    tag: "Zero-shot Baseline",
    type: "baseline",
    size: "9B",
    quant: "BF16",
    dev_acc: 0.2663,
    dev_macro_acc: 0.2954,
    total: 4183,
    hf_repo: null,
  },
];

// ── Helper components ─────────────────────────────────────────────────────────

function AccBar({ value, max = 1, color = "bg-blue-500" }) {
  const pct = Math.round(value * 100);
  const width = Math.round((value / max) * 100);
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 bg-gray-100 rounded-full h-2.5 overflow-hidden">
        <div
          className={`h-2.5 rounded-full ${color} transition-all duration-500`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className="w-12 text-right text-sm font-semibold text-gray-700">
        {pct}%
      </span>
    </div>
  );
}

function TypeBadge({ type }) {
  const styles = {
    finetuned:      "bg-blue-50 text-blue-700 border-blue-200",
    baseline:       "bg-gray-100 text-gray-600 border-gray-200",
    discriminative: "bg-violet-50 text-violet-700 border-violet-200",
  };
  const labels = {
    finetuned:      "Fine-tuned",
    baseline:       "Baseline",
    discriminative: "Discriminative",
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${styles[type] ?? styles.baseline}`}>
      {labels[type] ?? type}
    </span>
  );
}

function HFLink({ repo }) {
  if (!repo) return <span className="text-gray-400 text-xs">—</span>;
  return (
    <a
      href={repo}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 hover:underline font-medium"
    >
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
      </svg>
      HuggingFace
    </a>
  );
}

// ── MCAT table ────────────────────────────────────────────────────────────────

function MCATTable() {
  const [sortKey, setSortKey] = useState("mcat_acc");
  const sorted = [...MCAT_MODELS].sort((a, b) => b[sortKey] - a[sortKey]);
  const best = sorted[0]?.mcat_acc ?? 1;

  const cols = [
    { key: "mcat_acc", label: "Accuracy" },
    { key: "mcat_f1",  label: "Macro F1" },
    { key: "latency_s", label: "Latency (s)" },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
            <th className="text-left px-4 py-3 font-semibold text-gray-600 w-8">#</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">Model</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">Size</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">Type</th>
            {cols.map(c => (
              <th
                key={c.key}
                className={`text-left px-4 py-3 font-semibold cursor-pointer select-none whitespace-nowrap
                  ${sortKey === c.key ? "text-blue-600" : "text-gray-600 hover:text-gray-900"}`}
                onClick={() => setSortKey(c.key)}
              >
                {c.label} {sortKey === c.key ? "↓" : ""}
              </th>
            ))}
            <th className="text-left px-4 py-3 font-semibold text-gray-600">HF Repo</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m, i) => (
            <tr
              key={m.key}
              className={`border-b border-gray-100 hover:bg-blue-50/30 transition-colors
                ${i === 0 ? "bg-yellow-50/50" : ""}`}
            >
              <td className="px-4 py-3.5 font-bold text-gray-400">
                {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : i + 1}
              </td>
              <td className="px-4 py-3.5">
                <div className="font-semibold text-gray-800">{m.name}</div>
                <div className="text-xs text-gray-500 mt-0.5">{m.tag}</div>
                {m.note && <div className="text-xs text-amber-600 mt-0.5 italic">{m.note}</div>}
              </td>
              <td className="px-4 py-3.5">
                <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs font-mono">
                  {m.size} · {m.quant}
                </span>
              </td>
              <td className="px-4 py-3.5"><TypeBadge type={m.type} /></td>
              <td className="px-4 py-3.5 min-w-[180px]">
                <AccBar value={m.mcat_acc} max={best * 1.05}
                  color={m.type === "finetuned" ? "bg-blue-500" : "bg-gray-400"} />
              </td>
              <td className="px-4 py-3.5 text-gray-700 font-medium">{(m.mcat_f1 * 100).toFixed(1)}%</td>
              <td className="px-4 py-3.5 text-gray-600 font-mono text-xs">
                {m.latency_s < 10 ? m.latency_s.toFixed(2) : Math.round(m.latency_s)}s
              </td>
              <td className="px-4 py-3.5"><HFLink repo={m.hf_repo} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-400 mt-3 px-4">
        MCAT benchmark: 1,609 questions across 7 test sets (BB, CARS, CP, PS). Text-only (647 image questions skipped). Evaluated on DGX Spark (NVIDIA GB10 Blackwell).
      </p>
    </div>
  );
}

// ── MedMCQA table ─────────────────────────────────────────────────────────────

function MedMCQATable() {
  const sorted = [...MEDMCQA_MODELS].sort((a, b) => b.dev_acc - a.dev_acc);
  const best = sorted[0]?.dev_acc ?? 1;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
            <th className="text-left px-4 py-3 font-semibold text-gray-600 w-8">#</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">Model</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">Size</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">Type</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">Dev Accuracy</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">Macro Acc</th>
            <th className="text-left px-4 py-3 font-semibold text-gray-600">HF Repo</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m, i) => (
            <tr
              key={m.key}
              className={`border-b border-gray-100 hover:bg-blue-50/30 transition-colors
                ${i === 0 ? "bg-yellow-50/50" : ""}`}
            >
              <td className="px-4 py-3.5 font-bold text-gray-400">
                {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : i + 1}
              </td>
              <td className="px-4 py-3.5">
                <div className="font-semibold text-gray-800">{m.name}</div>
                <div className="text-xs text-gray-500 mt-0.5">{m.tag}</div>
                {m.note && <div className="text-xs text-amber-600 mt-0.5 italic">{m.note}</div>}
              </td>
              <td className="px-4 py-3.5">
                <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs font-mono">
                  {m.size} · {m.quant}
                </span>
              </td>
              <td className="px-4 py-3.5"><TypeBadge type={m.type} /></td>
              <td className="px-4 py-3.5 min-w-[180px]">
                <AccBar
                  value={m.dev_acc}
                  max={best * 1.05}
                  color={
                    m.type === "finetuned" ? "bg-blue-500"
                    : m.type === "discriminative" ? "bg-violet-500"
                    : "bg-gray-400"
                  }
                />
              </td>
              <td className="px-4 py-3.5 text-gray-600 font-mono text-xs">
                {m.dev_macro_acc != null ? `${(m.dev_macro_acc * 100).toFixed(1)}%` : "—"}
              </td>
              <td className="px-4 py-3.5"><HFLink repo={m.hf_repo} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-400 mt-3 px-4">
        MedMCQA dev split: 4,183 questions across 21 medical subjects (Indian PGMEE). Random baseline ≈ 25%.
      </p>
    </div>
  );
}

// ── Model cards ───────────────────────────────────────────────────────────────

const MODEL_CARDS = [
  {
    name: "Qwen3-14B MedMCQA LoRA",
    desc: "Qwen3-14B fine-tuned on MedMCQA train split (182K questions) with LoRA (r=16, α=32) in BF16. Best generative model for MedMCQA.",
    hf: `${HF_BASE}/qwen3-14b-medmcqa-lora`,
    badge: "14B · LoRA · BF16",
    color: "from-blue-500 to-blue-700",
  },
  {
    name: "Qwen3.5-9B MedMCQA GGUF",
    desc: "Qwen3.5-9B LoRA merged and quantized to Q4_K_M GGUF. Best throughput for inference. Runs on CPU or any llama.cpp backend.",
    hf: `${HF_BASE}/qwen3-5-9b-medmcqa-gguf`,
    badge: "9B · Q4_K_M GGUF",
    color: "from-indigo-500 to-indigo-700",
  },
  {
    name: "Qwen3-14B MedMCQA GGUF",
    desc: "Qwen3-14B LoRA merged and quantized to Q4_K_M GGUF. Higher accuracy than 9B with moderate memory requirements.",
    hf: `${HF_BASE}/qwen3-14b-medmcqa-gguf`,
    badge: "14B · Q4_K_M GGUF",
    color: "from-cyan-500 to-cyan-700",
  },
  {
    name: "Gemma-3-4B MedMCQA LoRA",
    desc: "Google Gemma-3-4B-it fine-tuned on MedMCQA. Best accuracy-per-parameter: 45.4% on MedMCQA dev vs 25% random baseline.",
    hf: `${HF_BASE}/gemma-3-4b-it-medmcqa-lora`,
    badge: "4B · LoRA · BF16",
    color: "from-emerald-500 to-emerald-700",
  },
  {
    name: "DeBERTa-v3-large Cross-Encoder",
    desc: "Discriminative cross-encoder: all 4 options in a single sequence → 4-way classification head. Fastest inference, no generation needed.",
    hf: `${HF_BASE}/medmcqa-pubmedbert-mcqa`,
    badge: "0.4B · FP32 · Classifier",
    color: "from-violet-500 to-violet-700",
  },
];

function ModelCards() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {MODEL_CARDS.map(c => (
        <div key={c.name} className="rounded-xl border border-gray-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
          <div className={`bg-gradient-to-r ${c.color} px-4 py-3`}>
            <span className="text-white/90 text-xs font-mono font-medium">{c.badge}</span>
          </div>
          <div className="p-4">
            <h3 className="font-semibold text-gray-800 text-sm mb-1.5">{c.name}</h3>
            <p className="text-xs text-gray-500 leading-relaxed mb-3">{c.desc}</p>
            <a
              href={c.hf}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs font-medium text-white bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded-lg transition-colors"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
              View on HuggingFace
            </a>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Leaderboard() {
  const [tab, setTab] = useState("mcat");

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">

      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div>
            <h1 className="font-bold text-gray-900 text-sm leading-tight">MedMCQA Model Leaderboard</h1>
            <p className="text-xs text-gray-500">SUTD MSTR-DAIE · Deep Learning Project · 2026</p>
          </div>
        </div>
        <a
          href="/"
          className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
          </svg>
          Sign in to take exam
        </a>
      </header>

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 space-y-8">

        {/* Hero */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-700 rounded-2xl p-6 text-white">
          <h2 className="text-2xl font-bold mb-2">Medical MCQ Benchmark</h2>
          <p className="text-blue-100 text-sm max-w-2xl">
            Comparing generative LLMs (fine-tuned vs baseline) and discriminative classifiers on
            MedMCQA (182K Indian PGMEE questions) and MCAT (1,609 questions). All models evaluated
            on NVIDIA DGX Spark (GB10 Blackwell, 121 GB unified memory).
          </p>
          <div className="mt-4 flex flex-wrap gap-3">
            {[
              { label: "Training samples", value: "182,822" },
              { label: "MCAT test questions", value: "1,609" },
              { label: "MedMCQA dev questions", value: "4,183" },
              { label: "Models evaluated", value: "8" },
            ].map(s => (
              <div key={s.label} className="bg-white/10 rounded-lg px-3 py-2">
                <div className="text-lg font-bold">{s.value}</div>
                <div className="text-blue-200 text-xs">{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Leaderboard tabs */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <div className="border-b border-gray-200 flex">
            {[
              { id: "mcat",    label: "MCAT Benchmark" },
              { id: "medmcqa", label: "MedMCQA Dev" },
            ].map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-6 py-3.5 text-sm font-medium border-b-2 transition-colors
                  ${tab === t.id
                    ? "border-blue-600 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700"}`}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div className="py-4">
            {tab === "mcat"    && <MCATTable />}
            {tab === "medmcqa" && <MedMCQATable />}
          </div>
        </div>

        {/* Model cards */}
        <section>
          <h2 className="text-lg font-bold text-gray-800 mb-4">Available Models on HuggingFace</h2>
          <ModelCards />
        </section>

        {/* Key findings */}
        <section className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
          <h2 className="text-lg font-bold text-gray-800 mb-4">Key Findings</h2>
          <div className="grid sm:grid-cols-2 gap-4 text-sm">
            {[
              {
                icon: "🎯",
                title: "Fine-tuning helps on MedMCQA",
                desc: "Gemma-3-4B LoRA achieves 45.4% on MedMCQA dev (vs 25% random). Larger models benefit less on this discriminative task.",
              },
              {
                icon: "📊",
                title: "CoT boosts MCAT accuracy",
                desc: "Chain-of-thought prompting improves Qwen3.5-9B from 77.4% → 78.4% on MCAT — at the cost of 37× slower inference.",
              },
              {
                icon: "⚡",
                title: "GGUF best for throughput",
                desc: "Q4_K_M quantization (5–8 GB) enables deployment on consumer hardware with ~2.8s/question, vs 23s for BF16 LoRA.",
              },
              {
                icon: "🔬",
                title: "Discriminative vs Generative",
                desc: "DeBERTa cross-encoder (0.4B) achieves 32% dev accuracy in FP32 — competitive with 14B zero-shot LLMs at 100× lower parameter count.",
              },
            ].map(f => (
              <div key={f.title} className="flex gap-3">
                <span className="text-2xl">{f.icon}</span>
                <div>
                  <h3 className="font-semibold text-gray-700">{f.title}</h3>
                  <p className="text-gray-500 text-xs mt-0.5 leading-relaxed">{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

      </main>

      <Footer />
    </div>
  );
}
