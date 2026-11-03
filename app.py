"""
app.py
Production FastAPI Web Server & Interactive Dashboard for Content Moderation.
Serves REST API endpoints and a modern web UI at http://localhost:8000.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database
from moderation_service import moderate_comment

# Initialize FastAPI application (Swagger/OpenAPI docs disabled)
app = FastAPI(
    title="Automated Content Moderation System",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Enable CORS for external frontend or mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class ModerateRequest(BaseModel):
    text: str = Field(..., example="This article is poorly written and full of misinformation.")

class BatchModerateRequest(BaseModel):
    comments: List[str] = Field(..., example=["Have a great day!", "You are a complete idiot."])


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.post("/api/moderate", summary="Moderate a single comment")
def api_moderate(request: ModerateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Comment text cannot be empty.")
    return moderate_comment(request.text)


@app.post("/api/moderate/batch", summary="Moderate a batch of comments")
def api_moderate_batch(request: BatchModerateRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="Comment list cannot be empty.")
    results = [moderate_comment(text) for text in request.comments]
    return {
        "count": len(results),
        "results": results
    }


@app.get("/api/logs", summary="Get recent moderation audit logs")
def api_get_logs(limit: int = 50):
    return database.get_recent_logs(limit=limit)


@app.get("/api/stats", summary="Get moderation statistics and analytics")
def api_get_stats():
    return database.get_statistics()


@app.post("/api/logs/clear", summary="Clear all moderation audit logs")
def api_clear_logs():
    return database.clear_all_logs()


# -------------------------------------------------------------
# Interactive Web Dashboard UI
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, summary="Interactive Web Dashboard")
def serve_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Content Moderation System</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @keyframes pulse-subtle { 0%, 100% { opacity: 1; } 50% { opacity: 0.85; } }
    .animate-subtle { animation: pulse-subtle 3s infinite ease-in-out; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen antialiased">
  
  <!-- Navigation Header -->
  <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <i class="fa-solid fa-shield-halved text-white text-lg"></i>
        </div>
        <div>
          <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
            AI Content Moderation System
            <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">DistilBERT</span>
          </h1>
          <p class="text-xs text-slate-400">Automated Multi-Label Toxicity Detection & Audit Engine</p>
        </div>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-8">

    <!-- Real-time Stats Cards -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div class="bg-slate-800/60 border border-slate-700/60 rounded-xl p-4">
        <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total Evaluated</div>
        <div id="stat-total" class="text-2xl font-bold text-white mt-1">0</div>
        <div class="text-[11px] text-slate-500 mt-1">Comments processed</div>
      </div>
      <div class="bg-slate-800/60 border border-emerald-900/40 rounded-xl p-4">
        <div class="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Approved (Clean)</div>
        <div id="stat-approved" class="text-2xl font-bold text-emerald-400 mt-1">0%</div>
        <div class="text-[11px] text-slate-500 mt-1">Passed automated safety</div>
      </div>
      <div class="bg-slate-800/60 border border-amber-900/40 rounded-xl p-4">
        <div class="text-xs font-semibold text-amber-400 uppercase tracking-wider">Flagged For Review</div>
        <div id="stat-flagged" class="text-2xl font-bold text-amber-400 mt-1">0%</div>
        <div class="text-[11px] text-slate-500 mt-1">Borderline toxicity</div>
      </div>
      <div class="bg-slate-800/60 border border-rose-900/40 rounded-xl p-4">
        <div class="text-xs font-semibold text-rose-400 uppercase tracking-wider">Auto-Blocked</div>
        <div id="stat-blocked" class="text-2xl font-bold text-rose-400 mt-1">0%</div>
        <div class="text-[11px] text-slate-500 mt-1">High severity violation</div>
      </div>
    </div>

    <!-- Live Testing & Evaluation Section -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Left Column: Input Form -->
      <div class="lg:col-span-6 bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 shadow-xl space-y-5">
        <div>
          <h2 class="text-base font-semibold text-white flex items-center gap-2">
            <i class="fa-solid fa-magnifying-glass text-indigo-400"></i> Live Comment Inspector
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">Type or choose a test comment below to evaluate toxicity probabilities in real time.</p>
        </div>

        <!-- Quick Preset Buttons -->
        <div>
          <label class="text-xs font-medium text-slate-400 mb-1.5 block">Quick Test Presets:</label>
          <div class="flex flex-wrap gap-2">
            <button onclick="setComment('Thank you so much for helping improve this Wikipedia article! Have a wonderful day.')" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700 transition">
              🌸 Clean & Helpful
            </button>
            <button onclick="setComment('You are an absolute idiot and know nothing. Stop editing my pages!')" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700 transition">
              😠 Targeted Insult
            </button>
            <button onclick="setComment('I will track down where you live and find you and destroy you.')" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700 transition">
              🚨 Violent Threat
            </button>
            <button onclick="setComment('This entire page is complete f***ing garbage and total nonsense.')" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700 transition">
              🤬 Obscene / Vulgar
            </button>
          </div>
        </div>

        <!-- Textarea -->
        <div>
          <textarea id="comment-input" rows="4" class="w-full bg-slate-900 border border-slate-700 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition" placeholder="Enter comment to evaluate..."></textarea>
        </div>

        <!-- Action Button -->
        <button id="btn-moderate" onclick="runModeration()" class="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-sm py-3 px-4 rounded-xl shadow-lg shadow-indigo-600/30 transition flex items-center justify-center gap-2">
          <i class="fa-solid fa-wand-magic-sparkles"></i> Evaluate Comment
        </button>
      </div>

      <!-- Right Column: Results & Score Breakdown -->
      <div class="lg:col-span-6 bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between pb-4 border-b border-slate-700/60">
            <div>
              <h2 class="text-base font-semibold text-white flex items-center gap-2">
                <i class="fa-solid fa-chart-simple text-purple-400"></i> Moderation Verdict
              </h2>
              <span id="latency-display" class="text-xs text-slate-400">Ready for evaluation</span>
            </div>
            <div id="verdict-badge" class="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-slate-800 text-slate-400 border border-slate-700">
              Awaiting Input
            </div>
          </div>

          <!-- Probability Bars -->
          <div class="mt-6 space-y-4">
            <!-- Labels list generated dynamically -->
            <div id="scores-container" class="space-y-3">
              <!-- Default blank state -->
              <div class="text-center py-10 text-slate-500 text-sm">
                Enter a comment on the left and click <strong>Evaluate Comment</strong> to view multi-label toxicity scores.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom Section: Database Audit History Table -->
    <div class="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 shadow-xl">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h2 class="text-base font-semibold text-white flex items-center gap-2">
            <i class="fa-solid fa-database text-sky-400"></i> Database Audit Logs (SQLite)
          </h2>
          <p class="text-xs text-slate-400">Permanent record of all automated moderation events and policy actions.</p>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="clearLogs()" class="text-xs bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 px-3 py-1.5 rounded-lg border border-rose-800/60 transition flex items-center gap-1.5">
            <i class="fa-solid fa-trash-can"></i> Clear All Logs
          </button>
          <button onclick="loadLogs()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center gap-1.5">
            <i class="fa-solid fa-rotate-right"></i> Refresh
          </button>
        </div>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="border-b border-slate-700 text-[11px] font-semibold text-slate-400 uppercase tracking-wider bg-slate-900/50">
              <th class="py-2.5 px-3">ID</th>
              <th class="py-2.5 px-3">Timestamp</th>
              <th class="py-2.5 px-3">Comment Preview</th>
              <th class="py-2.5 px-3">Decision</th>
              <th class="py-2.5 px-3">Primary Factor</th>
              <th class="py-2.5 px-3">Latency</th>
            </tr>
          </thead>
          <tbody id="logs-table-body" class="text-xs divide-y divide-slate-800/60">
            <tr>
              <td colspan="6" class="py-6 text-center text-slate-500">Loading historical audit logs...</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

  </main>

  <script>
    const LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"];

    function setComment(text) {
      document.getElementById('comment-input').value = text;
      runModeration();
    }

    async function runModeration() {
      const text = document.getElementById('comment-input').value.trim();
      if (!text) return;

      const btn = document.getElementById('btn-moderate');
      btn.disabled = true;
      btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...`;

      try {
        const response = await fetch('/api/moderate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text: text})
        });

        const data = await response.json();
        renderResult(data);
        loadStats();
        loadLogs();
      } catch (err) {
        alert("Error during moderation: " + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Evaluate Comment`;
      }
    }

    function renderResult(data) {
      // Verdict Badge
      const badge = document.getElementById('verdict-badge');
      if (data.verdict === 'APPROVED') {
        badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
        badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Approved`;
      } else if (data.verdict === 'FLAGGED_FOR_REVIEW') {
        badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/30";
        badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Flagged for Review`;
      } else {
        badge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-500/10 text-rose-400 border border-rose-500/30";
        badge.innerHTML = `<i class="fa-solid fa-ban"></i> Auto-Blocked`;
      }

      document.getElementById('latency-display').innerText = `Processed in ${data.latency_ms} ms (${data.primary_category})`;

      // Render 6 bars
      const container = document.getElementById('scores-container');
      container.innerHTML = "";

      LABELS.forEach(label => {
        const score = data.scores[label] || 0.0;
        const pct = (score * 100).toFixed(1);

        let color = "bg-emerald-500";
        if (pct >= 70) color = "bg-rose-500";
        else if (pct >= 40) color = "bg-amber-500";

        const row = `
          <div>
            <div class="flex justify-between text-xs font-medium mb-1">
              <span class="text-slate-300 capitalize">${label.replace('_', ' ')}</span>
              <span class="text-slate-400 font-mono">${pct}%</span>
            </div>
            <div class="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden">
              <div class="${color} h-2.5 rounded-full transition-all duration-500" style="width: ${Math.max(pct, 2)}%"></div>
            </div>
          </div>
        `;
        container.innerHTML += row;
      });
    }

    async function loadStats() {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('stat-total').innerText = data.total || 0;
        document.getElementById('stat-approved').innerText = (data.approved_pct || 0) + '%';
        document.getElementById('stat-flagged').innerText = (data.flagged_pct || 0) + '%';
        document.getElementById('stat-blocked').innerText = (data.blocked_pct || 0) + '%';
      } catch (e) {
        console.error(e);
      }
    }

    async function loadLogs() {
      try {
        const res = await fetch('/api/logs?limit=15');
        const logs = await res.json();
        const tbody = document.getElementById('logs-table-body');
        tbody.innerHTML = "";

        if (logs.length === 0) {
          tbody.innerHTML = `<tr><td colspan="6" class="py-6 text-center text-slate-500">No comments moderated yet.</td></tr>`;
          return;
        }

        logs.forEach(log => {
          let badgeClass = "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
          if (log.verdict === 'AUTO_BLOCKED') badgeClass = "text-rose-400 bg-rose-500/10 border-rose-500/20";
          else if (log.verdict === 'FLAGGED_FOR_REVIEW') badgeClass = "text-amber-400 bg-amber-500/10 border-amber-500/20";

          const preview = log.comment_text.length > 55 ? log.comment_text.substring(0, 55) + "..." : log.comment_text;

          const row = `
            <tr class="hover:bg-slate-800/40 transition">
              <td class="py-2.5 px-3 font-mono text-slate-500">#${log.id}</td>
              <td class="py-2.5 px-3 text-slate-400 whitespace-nowrap">${log.timestamp.split(' ')[1]}</td>
              <td class="py-2.5 px-3 text-slate-200 font-medium">${preview}</td>
              <td class="py-2.5 px-3">
                <span class="px-2 py-0.5 rounded text-[10px] font-semibold uppercase border ${badgeClass}">
                  ${log.verdict.replace('_', ' ')}
                </span>
              </td>
              <td class="py-2.5 px-3 text-slate-400 capitalize">${log.primary_category}</td>
              <td class="py-2.5 px-3 font-mono text-slate-400">${log.latency_ms}ms</td>
            </tr>
          `;
          tbody.innerHTML += row;
        });
      } catch (e) {
        console.error(e);
      }
    }

    async function clearLogs() {
      if (!confirm("Are you sure you want to clear all moderation logs?")) return;
      try {
        await fetch('/api/logs/clear', { method: 'POST' });
        loadStats();
        loadLogs();
      } catch (e) {
        alert("Error clearing logs: " + e);
      }
    }

    // Initial load on page open
    loadStats();
    loadLogs();
  </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🚀 Starting Content Moderation Server at http://localhost:8000")
    print("=" * 65 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
