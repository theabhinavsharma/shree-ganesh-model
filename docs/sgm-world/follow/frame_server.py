"""Receives preview frames from the local Data World page (POST /frame?n=<k>&dir=<name>) and writes them as JPEGs."""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
OUT = Path(sys.argv[1])
class H(BaseHTTPRequestHandler):
    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "http://localhost:8795")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
    def do_OPTIONS(self):
        self.send_response(204); self.cors(); self.end_headers()
    def do_POST(self):
        q = parse_qs(urlparse(self.path).query); n = int(q["n"][0]); d = OUT / q.get("dir", ["frames"])[0]
        d.mkdir(parents=True, exist_ok=True)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        (d / f"{n:05d}.jpg").write_bytes(body)
        self.send_response(200); self.cors(); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
ThreadingHTTPServer(("127.0.0.1", 8797), H).serve_forever()
