
from pathlib import Path
import os
import html
import json
import random
import sqlite3
import pickle
from datetime import datetime

import requests
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# ---------------- Configuration ----------------
ROOT = Path(__file__).parent
DATABASE_PATH = ROOT / "tickets.db"
CSV_FILE = ROOT / "tickets_with_solutions.csv"
EMBED_FILE = ROOT / "ticket_embeddings.pkl"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
FINETUNED_MODEL_DIR = ROOT / "model" / "checkpoint-5807"

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")
ADMIN_PASSWORD = "mypassword123"  # must match frontend

# ---------------- App init ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Load models & data ----------------
print("[startup] loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL_ID)

print("[startup] loading ticket CSV...")
df_main = pd.read_csv(CSV_FILE)

# Load precomputed embeddings (try torch pickle first, fallback to plain pickle)
if EMBED_FILE.exists():
    print("[startup] loading saved embeddings...")
    with open(EMBED_FILE, "rb") as f:
        try:
            ticket_embeddings = torch.load(f, map_location=torch.device("cpu"))
        except Exception:
            f.seek(0)
            ticket_embeddings = pickle.load(f)
    print("[startup] embeddings ready.")
else:
    print("[startup] computing embeddings (first-run)...")
    ticket_embeddings = embedder.encode(
        df_main["ticket_text"].astype(str).tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    with open(EMBED_FILE, "wb") as f:
        torch.save(ticket_embeddings.cpu(), f)
    print("[startup] embeddings saved.")

# Load fine-tuned priority classifier
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
priority_classifier = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_DIR)

# ---------------- Database helpers ----------------
def ensure_db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                content TEXT,
                priority TEXT,
                embedding BLOB
            );
            """
        )
        conn.commit()

def store_ticket(filename: str, content: str, priority: str, embedding):
    blob = pickle.dumps(embedding)
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute(
            "INSERT INTO tickets (filename, content, priority, embedding) VALUES (?, ?, ?, ?)",
            (filename, content, priority, blob)
        )
        conn.commit()

def query_tickets(search: str = "", priority: str = "All"):
    sql = "SELECT id, filename, content, priority FROM tickets WHERE 1=1"
    params = []
    if search:
        sql += " AND content LIKE ?"
        params.append(f"%{search}%")
    if priority != "All":
        sql += " AND priority = ?"
        params.append(priority)
    with sqlite3.connect(DATABASE_PATH) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [{"id": r[0], "filename": r[1], "content": r[2], "priority": r[3]} for r in rows]

# ---------------- Prediction & similarity helpers ----------------
def predict_ticket_priority(text: str) -> str:
    toks = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = priority_classifier(**toks).logits
        probs = torch.softmax(logits, dim=1)
        label_idx = int(torch.argmax(probs))
    labels = ["Low", "Medium", "High"]
    return labels[label_idx] if label_idx < len(labels) else "Unknown"

def find_similarity_and_status(text: str, threshold: float = 0.25):
    q_emb = embedder.encode(text, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, ticket_embeddings).squeeze()
    best_idx = int(torch.argmax(sims).item())
    best_score = float(sims[best_idx])
    status = "Knowledge Gap Detected" if best_score < threshold else "Covered by Existing Knowledge"
    return best_idx, best_score, status

# ---------------- Slack notifications ----------------
GAP_CSV = ROOT / "content_gap_tickets.csv"

def post_slack(ticket_id, ticket_text, similarity, timestamp, webhook_url, solved=False):
    if not webhook_url:
        print("[slack] webhook URL not configured.")
        return

    safe_text = html.escape(ticket_text)
    if len(safe_text) > 2000:
        safe_text = safe_text[:2000] + "…"

    title = ":white_check_mark: *Content Gap Resolved*" if solved else ":warning: *New Content Gap Ticket*"
    status_text = "Solved" if solved else "Pending"

    payload = {
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": title}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Ticket ID:*\n{ticket_id}"},
                {"type": "mrkdwn", "text": f"*Similarity:*\n{similarity:.2f}"},
                {"type": "mrkdwn", "text": f"*Status:*\n{status_text}"},
                {"type": "mrkdwn", "text": f"*Updated:*\n{timestamp}"}
            ]},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Content:*\n>{safe_text}"}}
        ]
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        print("[slack] status", resp.status_code, resp.text)
    except Exception as e:
        print("[slack] error:", e)

# ---------------- Content-gap CSV management ----------------
def add_or_get_gap_ticket(ticket_text: str, similarity_score: float):
    text_clean = ticket_text.strip()
    ts = datetime.utcnow().isoformat() + "Z"
    if GAP_CSV.exists():
        df_gap = pd.read_csv(GAP_CSV, encoding="utf-8-sig")
        df_gap["ticket_text"] = df_gap["ticket_text"].astype(str).str.strip()
    else:
        df_gap = pd.DataFrame(columns=["ticket_id", "ticket_text", "similarity_score", "status", "created_at"])

    exists = df_gap["ticket_text"].str.lower().eq(text_clean.lower()).any()
    if not exists:
        ticket_id = int(datetime.utcnow().timestamp())
        new_row = {
            "ticket_id": ticket_id,
            "ticket_text": text_clean,
            "similarity_score": similarity_score,
            "status": "Pending",
            "created_at": ts
        }
        df_gap = pd.concat([df_gap, pd.DataFrame([new_row])], ignore_index=True)
        df_gap.to_csv(GAP_CSV, index=False, encoding="utf-8-sig")
        print("[gap] added:", text_clean)
    else:
        ticket_id = int(df_gap.loc[df_gap["ticket_text"].str.lower() == text_clean.lower(), "ticket_id"].values[0])
        print("[gap] already exists:", text_clean)

    post_slack(ticket_id, ticket_text, similarity_score, ts, SLACK_WEBHOOK, solved=False)
    return ticket_id

# ---------------- Routes ----------------
@app.route("/upload", methods=["POST"])
def route_upload():
    try:
        if request.is_json:
            payload = request.get_json()
            fname = payload.get("filename", "unknown")
            content = payload.get("content", "").strip()
        elif "file" in request.files:
            f = request.files["file"]
            fname = f.filename
            content = f.read().decode("utf-8", errors="ignore").strip()
        else:
            return jsonify({"error": "No file or JSON provided"}), 400

        if not content:
            return jsonify({"error": "Empty content"}), 400

        priority = predict_ticket_priority(content)
        emb = embedder.encode(content)

        store_ticket(fname, content, priority, emb)

        return jsonify({
            "message": "Ticket saved",
            "filename": fname,
            "priority": priority,
            "embedding_dim": len(emb)
        }), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/analyze", methods=["POST"])
def route_analyze():
    try:
        data = request.get_json(force=True)
        text = data.get("content", "")
        if not text:
            return jsonify({"error": "content required"}), 400

        priority = predict_ticket_priority(text)
        emb = embedder.encode(text)
        return jsonify({"priority": priority, "embedding_dim": len(emb)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/tickets", methods=["GET"])
def route_list_tickets():
    s = request.args.get("search", "")
    p = request.args.get("priority", "All")
    return jsonify(query_tickets(s, p)), 200

@app.route("/recommend", methods=["POST"])
def route_recommend():
    try:
        ticket_text = ""
        if request.is_json:
            ticket_text = request.get_json().get("ticket_text", "").strip()

        if not ticket_text and request.files:
            uploaded = next(iter(request.files.values()))
            ticket_text = uploaded.read().decode("utf-8", errors="ignore").strip()

        if not ticket_text:
            return jsonify({"error": "No ticket text provided"}), 400

        best_idx, best_score, status = find_similarity_and_status(ticket_text, threshold=0.25)
        matched_row = df_main.iloc[best_idx]
        solution = str(matched_row.get("solution", ""))
        solution = solution.replace("<name>", "customer").replace("<NAME>", "customer") \
                           .replace("<tel_num>", "your contact number").replace("<acc_num>", "your account number")

        is_gap = best_score < 0.25
        if is_gap:
            add_or_get_gap_ticket(ticket_text, best_score)

        resp = {
            "uploaded_ticket_text": ticket_text,
            "recommended_solution": solution,
            "similarity_score": round(best_score, 3),
            "gap_status": status,
            "content_gap": is_gap,
            "show_generate_button": is_gap
        }
        return jsonify(resp), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/detect-gap", methods=["POST"])
def route_detect_gap():
    try:
        data = request.get_json()
        text = data.get("ticket_text", "").strip()
        if not text:
            return jsonify({"error": "No ticket text provided"}), 400
        idx, score, status = find_similarity_and_status(text)
        return jsonify({"gap_status": status, "max_similarity": round(score, 3)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/add-solution", methods=["POST"])
def route_add_solution():
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
        new_solution = data.get("solution", "")

        if not ticket_text:
            return jsonify({"error": "No ticket text provided"}), 400

        if not new_solution:
            new_solution = random.choice([
                "Issue acknowledged, our team is investigating.",
                "Reset configuration and retry after clearing cache.",
                "Escalate this issue to Tier 2 support for deeper investigation.",
                "Perform a clean reinstallation of the application.",
                "Restart the system and check network connectivity before retrying."
            ])

        global df_main, ticket_embeddings
        added = pd.DataFrame([{"ticket_text": ticket_text, "solution": new_solution}])
        df_main = pd.concat([df_main, added], ignore_index=True)
        df_main.to_csv(CSV_FILE, index=False)

        new_emb = embedder.encode([ticket_text], convert_to_tensor=True)
        ticket_embeddings = torch.cat((ticket_embeddings, new_emb), dim=0)
        with open(EMBED_FILE, "wb") as f:
            torch.save(ticket_embeddings.cpu(), f)

        return jsonify({"message": "✅ New solution added successfully!", "solution": new_solution}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/generate_new_solution", methods=["POST"])
def route_generate_solution():
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
        if not ticket_text:
            return jsonify({"error": "No ticket text provided"}), 400

        new_solution = (
            f"Dear customer, thank you for reaching out regarding the issue: '{ticket_text}'. "
            "We are sorry for the inconvenience caused. Our team will review your request and "
            "provide a suitable resolution shortly. Please ensure you have shared your account "
            "details or order reference for faster support. Thank you for your patience."
        )
        with open(ROOT / "generated_solutions.csv", "a", encoding="utf-8") as fh:
            fh.write(f"{ticket_text},{new_solution}\n")

        return jsonify({"message": "New solution generated successfully.", "new_solution": new_solution}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/update_solution_manual", methods=["POST"])
def route_update_solution_manual():
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
        new_solution = data.get("new_solution", "").strip()
        password = data.get("password", "").strip()

        if password != ADMIN_PASSWORD:
            return jsonify({"error": "Invalid admin password"}), 403

        if not ticket_text or not new_solution:
            return jsonify({"error": "ticket_text and new_solution are required"}), 400

        global df_main, ticket_embeddings
        matches = df_main.index[df_main["ticket_text"].str.strip().str.lower() == ticket_text.lower()].tolist()
        if matches:
            df_main.at[matches[0], "solution"] = new_solution
        else:
            new_entry = pd.DataFrame([{"ticket_text": ticket_text, "solution": new_solution}])
            df_main = pd.concat([df_main, new_entry], ignore_index=True)
            new_emb = embedder.encode([ticket_text], convert_to_tensor=True)
            ticket_embeddings = torch.cat((ticket_embeddings, new_emb), dim=0)

        df_main.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        with open(EMBED_FILE, "wb") as f:
            torch.save(ticket_embeddings.cpu(), f)

        # mark solved in gap CSV if present
        if GAP_CSV.exists():
            df_gap = pd.read_csv(GAP_CSV, encoding="utf-8-sig")
            if "status" not in df_gap.columns:
                df_gap["status"] = "Pending"
            df_gap["ticket_text_norm"] = df_gap["ticket_text"].astype(str).str.lower().str.strip()

            mask = df_gap["ticket_text_norm"] == ticket_text.lower().strip()
            if mask.any():
                df_gap.loc[mask, "status"] = "Solved"
                ticket_id = int(df_gap.loc[mask, "ticket_id"].values[0])
                df_gap = df_gap.drop(columns=["ticket_text_norm"])
                df_gap.to_csv(GAP_CSV, index=False, encoding="utf-8-sig")
                post_slack(ticket_id, ticket_text, 0, datetime.utcnow().isoformat() + "Z", SLACK_WEBHOOK, solved=True)
                print("[gap] marked solved:", ticket_text)
            else:
                print("[gap] not found in CSV.")

        return jsonify({"message": "✅ Solution updated successfully and ticket marked as Solved!", "solution": new_solution}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/content-gap-tickets", methods=["GET"])
def route_content_gap_tickets():
    try:
        if GAP_CSV.exists():
            df_gap = pd.read_csv(GAP_CSV)
            return jsonify(df_gap.fillna("").to_dict(orient="records")), 200
        return jsonify([]), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/mark_ticket_solved", methods=["POST"])
def route_mark_ticket_solved():
    try:
        data = request.get_json()
        ticket_text = data.get("ticket_text", "").strip()
        password = data.get("password", "").strip()

        if password != ADMIN_PASSWORD:
            return jsonify({"error": "Invalid admin password"}), 403

        if not ticket_text:
            return jsonify({"error": "ticket_text is required"}), 400

        if not GAP_CSV.exists():
            return jsonify({"error": "content_gap_tickets.csv not found"}), 404

        df_gap = pd.read_csv(GAP_CSV, encoding="utf-8-sig")
        df_gap["ticket_text_clean"] = df_gap["ticket_text"].astype(str).str.strip().str.lower()
        cleaned = ticket_text.lower().strip()

        mask = df_gap["ticket_text_clean"] == cleaned
        if mask.any():
            df_gap.loc[mask, "status"] = "Solved"
            df_gap.drop(columns=["ticket_text_clean"], inplace=True, errors="ignore")
            df_gap.to_csv(GAP_CSV, index=False, encoding="utf-8-sig")
            print("[gap] manually marked solved:", ticket_text)
            return jsonify({"message": f"✅ '{ticket_text}' marked as Solved!"}), 200
        return jsonify({"error": "Ticket not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ---------------- Startup ----------------
print("[startup] initializing DB and starting server...")
if __name__ == "__main__":
    ensure_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
