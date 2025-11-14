import os
from pathlib import Path
from io import StringIO
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

sns.set(style="whitegrid")
load_dotenv()

# -------------------- Config --------------------
API_BASE = "http://127.0.0.1:5000"
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
GAP_CSV_PATH = Path(r"C:/Users/POOJITHA/Documents/Knowledge Engine/backend/content_gap_tickets.csv")
st.set_page_config(page_title="🎫 Ticket Uploader & Analyzer", layout="wide")

# -------------------- Helper Functions --------------------
def notify_slack(message: str, rich_payload: dict = None):
    if not SLACK_WEBHOOK_URL:
        st.warning("⚠️ Slack webhook URL not configured.")
        return
    try:
        payload = rich_payload if rich_payload else {"text": message}
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        if resp.ok:
            st.info("📢 Slack notified.")
        else:
            st.warning(f"⚠️ Slack webhook returned: {resp.status_code}")
    except Exception as e:
        st.error(f"⚠️ Slack request failed: {e}")

def load_csv(file_path: Path, default_columns=None):
    if not file_path.exists():
        df = pd.DataFrame(columns=default_columns)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return pd.read_csv(file_path, encoding="utf-8-sig")

def upload_ticket(content: str, filename: str) -> bool:
    payload = {"filename": filename, "content": content}
    try:
        resp = requests.post(f"{API_BASE}/upload", json=payload, timeout=30)
        return resp.ok
    except Exception as e:
        st.error(f"Error uploading ticket: {e}")
        return False

def analyze_ticket(content: str) -> dict:
    try:
        resp = requests.post(f"{API_BASE}/analyze", json={"content": content}, timeout=30)
        return resp.json() if resp.ok else {}
    except Exception as e:
        st.error(f"Error analyzing ticket: {e}")
        return {}

def recommend_solution(ticket_text: str) -> dict:
    try:
        resp = requests.post(f"{API_BASE}/recommend", json={"ticket_text": ticket_text})
        return resp.json() if resp.status_code == 200 else {}
    except Exception as e:
        st.error(f"Error contacting backend: {e}")
        return {}

def show_content_gap_tickets():
    try:
        resp = requests.get(f"{API_BASE}/content-gap-tickets", timeout=30)
        if resp.ok:
            df_gap = pd.DataFrame(resp.json())
            if not df_gap.empty:
                st.dataframe(df_gap, use_container_width=True)
                st.write(f"Total tickets with content gaps: {len(df_gap)}")
            else:
                st.info("✅ No content gap tickets found.")
        else:
            st.error(f"Failed to fetch content-gap tickets: {resp.text}")
    except Exception as e:
        st.error(f"Error fetching content-gap tickets: {e}")

# -------------------- UI Components --------------------
st.title("🎫 Ticket Upload & Analysis")

# Sidebar Filters
st.sidebar.header("Filters & Search")
search_query = st.sidebar.text_input("Search tickets (keyword)")
priority_filter = st.sidebar.selectbox("Priority filter", ["All", "High", "Medium", "Low"])
search_button = st.sidebar.button("🔎 Search Tickets")

# File Upload
st.subheader("1️⃣ Upload Ticket")
uploaded_file = st.file_uploader("Choose a .txt or .csv file", type=["txt", "csv"])
preview_content = ""

if uploaded_file:
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith(".txt"):
        preview_content = uploaded_file.read().decode("utf-8")
        st.text_area("Ticket preview", preview_content, height=200)
    else:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(StringIO(uploaded_file.read().decode("utf-8")))
            st.dataframe(df)
            preview_content = df["content"].iloc[0] if "content" in df.columns else ""
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# Upload Button
if st.button("📤 Upload Ticket"):
    if not uploaded_file:
        st.warning("Please choose a file first.")
    else:
        uploaded_df = pd.DataFrame()
        if uploaded_file.name.lower().endswith(".csv"):
            uploaded_file.seek(0)
            df = pd.read_csv(StringIO(uploaded_file.read().decode("utf-8")))
            column_name = "content" if "content" in df.columns else "ticket_text"
            successes, failures = 0, 0
            for _, row in df.iterrows():
                content = str(row[column_name]).strip()
                if not content:
                    failures += 1
                    continue
                if upload_ticket(content, uploaded_file.name):
                    successes += 1
                    uploaded_df = pd.concat([uploaded_df, pd.DataFrame([{"ticket_text": content}])], ignore_index=True)
                else:
                    failures += 1
            st.success(f"✅ Uploaded {successes} rows. ❌ Failed: {failures}.")
        else:  # txt file
            uploaded_file.seek(0)
            content = uploaded_file.read().decode("utf-8").strip()
            if upload_ticket(content, uploaded_file.name):
                st.success(f"✅ Ticket '{uploaded_file.name}' saved successfully!")
                uploaded_df = pd.DataFrame([{"ticket_text": content}])
        
        # Update session state
        if not uploaded_df.empty:
            if "df_tickets" not in st.session_state:
                st.session_state["df_tickets"] = uploaded_df
            else:
                st.session_state["df_tickets"] = pd.concat([st.session_state["df_tickets"], uploaded_df], ignore_index=True)

# Manual Ticket Analysis
st.subheader("2️⃣ Analyze a Ticket Text")
ticket_text = st.text_area("Paste ticket text here (or leave empty to use uploaded preview):", value=preview_content, height=200)
if st.button("🔍 Analyze Text (Predict Priority)"):
    if ticket_text.strip():
        result = analyze_ticket(ticket_text.strip())
        st.success(f"Predicted Priority: {result.get('priority', 'Unknown')}")
        st.json(result)
    else:
        st.warning("Add some text to analyze.")

# Browse Stored Tickets
if search_button:
    try:
        params = {"search": search_query.strip(), "priority": priority_filter}
        resp = requests.get(f"{API_BASE}/tickets", params=params, timeout=30)
        if resp.ok:
            tickets = resp.json()
            if tickets:
                df = pd.DataFrame(tickets)
                st.dataframe(df, use_container_width=True)
                st.write(f"Total: {len(df)} tickets")
                if "priority" in df.columns:
                    st.bar_chart(df["priority"].value_counts())
            else:
                st.info("No tickets matched your criteria.")
        else:
            st.error(f"Failed to fetch tickets: {resp.text}")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")

# Recommendation System
st.markdown("---")
st.title("🎯 Ticket Solution Recommendation System")
if "uploaded_ticket_text" not in st.session_state:
    st.session_state["uploaded_ticket_text"] = preview_content

ticket_text = st.text_area("Enter your ticket description:", value=st.session_state.get("uploaded_ticket_text", ""), height=200, key="ticket_text_area")
if st.button("🔍 Recommend Solution"):
    if ticket_text.strip():
        data = recommend_solution(ticket_text)
        st.session_state["recommendation"] = data
    else:
        st.warning("⚠️ Please enter a ticket description first.")

if "recommendation" in st.session_state:
    data = st.session_state["recommendation"]
    if "recommended_solution" in data:
        st.subheader("🧠 Recommendation Result")
        st.caption(f"Similarity Score: {data.get('similarity_score', 'N/A')}")
        if not data.get("content_gap", False):
            st.success("✅ Covered by existing knowledge base.")
            st.info(data['recommended_solution'])
        else:
            st.warning("🚨 Knowledge Gap Detected!")
            st.info("No similar issue found in the database.")
            if st.button("✨ Generate New Solution"):
                gen_response = requests.post(f"{API_BASE}/generate_new_solution", json={"ticket_text": ticket_text})
                if gen_response.status_code == 200:
                    result = gen_response.json()
                    st.session_state["new_solution"] = result["new_solution"]
                    st.success("✅ New Solution Generated Successfully!")
                    st.info(result["new_solution"])
                else:
                    st.error("❌ Failed to generate new solution.")

# Content Gap Tickets
st.markdown("---")
st.title("📌 Content Gap Tickets")
if st.button("Refresh Content Gap Tickets"):
    show_content_gap_tickets()
else:
    show_content_gap_tickets()

# Update Solution Form
st.title(" Update Ticket Solution (Admin Only)")
with st.form("update_solution_form"):
    ticket_text = st.text_area("Ticket Text", height=200, key="ticket_text")
    new_solution = st.text_area("New Solution", height=150, key="new_solution")
    password = st.text_input("Admin Password", type="password", key="password")
    submitted = st.form_submit_button("💾 Update Solution")

if submitted:
    if ticket_text.strip() and new_solution.strip() and password.strip():
        try:
            payload = {"ticket_text": ticket_text.strip(), "new_solution": new_solution.strip(), "password": password.strip()}
            response = requests.post(f"{API_BASE}/update_solution_manual", json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                st.success(data.get("message", "✅ Solution updated successfully!"))
                st.info(data.get("solution"))
                st.session_state.ticket_text = ""
                st.session_state.new_solution = ""
                st.session_state.password = ""
            elif response.status_code == 403:
                st.error("❌ Invalid admin password.")
            else:
                st.error(f"Failed to update solution: {response.status_code}")
        except Exception as e:
            st.error(f"Error contacting backend: {e}")

# Content Gap CSV Manager
st.title(" Content Gap Ticket Manager")
df_gap = load_csv(GAP_CSV_PATH, default_columns=["ticket_id","ticket_text","similarity_score","status","created_at"])
pending = df_gap[df_gap["status"].str.lower() == "pending"]

if pending.empty:
    st.success("🎉 No pending tickets.")
else:
    for idx, row in pending.iterrows():
        ticket_id = row["ticket_id"]
        ticket_text = row["ticket_text"]
        similarity = row.get("similarity_score", "")
        created_at = row.get("created_at", "")
        st.markdown(f"**Ticket ID:** {ticket_id}  \n**Content:** {ticket_text}  \n**Similarity:** {similarity}  \n**Created:** {created_at}")
        if st.button(f"✅ Mark as Solved - {ticket_id}", key=f"solve_{idx}"):
            solved_at = datetime.utcnow().isoformat() + "Z"
            df_gap.loc[idx, ["status", "solved_by", "solved_at"]] = ["Solved", "Admin", solved_at]
            df_gap.to_csv(GAP_CSV_PATH, index=False, encoding="utf-8-sig")

            slack_payload = {
                "blocks": [
                    {"type":"section","text":{"type":"mrkdwn","text":":white_check_mark: *Ticket Solved!*"}},
                    {"type":"section","fields":[
                        {"type":"mrkdwn","text":f"*Ticket ID:*\n{ticket_id}"},
                        {"type":"mrkdwn","text":f"*Solved by:*\nAdmin"},
                        {"type":"mrkdwn","text":f"*Solved at (UTC):*\n{solved_at}"},
                        {"type":"mrkdwn","text":f"*Similarity:*\n{similarity}"}
                    ]},
                    {"type":"section","text":{"type":"mrkdwn","text":f"*Content:*\n>{ticket_text}"}}
                ]
            }
            notify_slack("", slack_payload)
            st.rerun()
