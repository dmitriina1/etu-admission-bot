import os
import sys
import requests

if len(sys.argv) != 2:
    raise SystemExit("Использование: python set_webhook.py https://имя-сервиса.onrender.com")

base_url = sys.argv[1].rstrip("/")
token = os.environ["BOT_TOKEN"]
secret = os.environ["WEBHOOK_SECRET"]
url = f"{base_url}/telegram/{secret}"
response = requests.post(
    f"https://api.telegram.org/bot{token}/setWebhook",
    json={"url": url, "drop_pending_updates": True},
    timeout=30,
)
print(response.status_code, response.text)
response.raise_for_status()
