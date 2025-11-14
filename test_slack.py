import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SLACK_WEBHOOK_URL")

if not url:
    print("❌ SLACK_WEBHOOK_URL not found in environment variables.")
else:
    payload = {
        "text": ":bell: Slack integration test successful!"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Message sent successfully to Slack!")
        else:
            print(f"⚠️ Slack returned {response.status_code}: {response.text}")
    except Exception as e:
        print("❌ Error sending message:", e)
