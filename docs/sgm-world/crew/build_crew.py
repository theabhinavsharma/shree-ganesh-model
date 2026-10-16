"""Build the crew page from crew_template.html: the artifact preview (crew.html here) and the website page
(shreeganeshmodel-deploy/crew.html, served at /crew.html). The world's content (world/data.js) is inlined so both
versions show the same bots and plain-language text. Models load from assets/ next to the page."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
tpl = (HERE / "crew_template.html").read_text()
data = (ROOT / "shreeganeshmodel-deploy/world/data.js").read_text()
page = tpl.replace("<script>__DATA__</script>", "<script>\n" + data + "\n</script>")
# artifact preview: claude.ai pages don't serve .glb, so each model is also written as glTF JSON with its data embedded
import base64, json, struct
MODELS = HERE / "models"
for glb in sorted((ROOT / "shreeganeshmodel-deploy/assets").glob("*/*.glb")):
    raw = glb.read_bytes(); assert raw[:4] == b"glTF"
    jlen = struct.unpack("<I", raw[12:16])[0]; gj = json.loads(raw[20:20 + jlen])
    off = 20 + jlen; blen = struct.unpack("<I", raw[off:off + 4])[0]; binc = raw[off + 8: off + 8 + blen]
    gj["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(binc).decode()
    out = MODELS / glb.parent.name / (glb.stem + ".gltf.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gj, separators=(",", ":")))
(HERE / "crew.html").write_text(page.replace('MODEL_EXT = ".glb"', 'MODEL_EXT = ".gltf.json"'))
site = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="The SGM crew: every bot, ship and machine that carries the data behind the Friday list, and its journey.">\n')
(ROOT / "shreeganeshmodel-deploy/crew.html").write_text(site + page + "\n</html>\n")
print("built crew.html (artifact) and shreeganeshmodel-deploy/crew.html (site)")
