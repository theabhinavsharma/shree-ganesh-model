"""One-time Telegram setup for SGM phone messages (2026-09-29).

Plain English: asks for the bot token without showing it on screen, checks it with Telegram, finds your chat ID from
the message you sent the bot, saves both to ~/.config/sgm/telegram.env (only your user can read it), and sends a test
message. The token is never printed, logged or committed. From then on notify.py sends every alert to Telegram.
Usage (in Terminal): /usr/bin/python3 ~/Code/Zoom/src/agentic/setup_telegram.py
"""
from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

CFG = Path.home() / ".config/sgm"
ENV = CFG / "telegram.env"


def api(tok: str, method: str, *data: str) -> dict:
    args = ["/usr/bin/curl", "-s", "-m", "30", f"https://api.telegram.org/bot{tok}/{method}"]
    for d in data:
        args += ["--data-urlencode", d]
    r = subprocess.run(args, capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {"ok": False, "description": f"no answer from Telegram (curl exit {r.returncode}; a proxy may block it)"}


def main() -> None:
    tok = getpass.getpass("Paste the SGM alerts bot token (it stays hidden), then press Enter: ").strip()
    if ":" not in tok or " " in tok:
        sys.exit("That does not look like a bot token (it has one colon, no spaces). Nothing saved.")
    me = api(tok, "getMe")
    if not me.get("ok"):
        sys.exit(f"Telegram did not accept the token: {me.get('description')}. Nothing saved.")
    bot = me["result"]["username"]
    print(f"Token works for @{bot}.")
    chats = {}
    for u in api(tok, "getUpdates").get("result", []):
        c = (u.get("message") or u.get("edited_message") or {}).get("chat", {})
        if c.get("type") == "private":
            chats[c["id"]] = c.get("first_name", "")
    if not chats:
        sys.exit(f"@{bot} has no messages yet. Open it in Telegram, send 'hi', then run this again. Nothing saved.")
    chat_id = list(chats)[-1]
    CFG.mkdir(parents=True, exist_ok=True)
    old = os.umask(0o077)
    try:
        ENV.write_text(f"TELEGRAM_BOT_TOKEN={tok}\nTELEGRAM_CHAT_ID={chat_id}\n")
    finally:
        os.umask(old)
    ENV.chmod(0o600)
    print(f"Saved {ENV} (readable only by you). Chat: {chats[chat_id]}.")
    r = api(tok, "sendMessage", f"chat_id={chat_id}",
            "text=SGM test message · if you can read this on your phone, alerts work.")
    print("✅ Test message sent. Check your phone." if r.get("ok") else f"❌ Test message failed: {r.get('description')}")


if __name__ == "__main__":
    main()
