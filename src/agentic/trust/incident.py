"""Incident writer (2026-09-30): turn every failure into a permanent check and a skill lesson, without losing anything.

Plain English: when something breaks (a feed goes silently empty, prices carry fake crashes, a regex matches the wrong
words), this records it once and makes the system remember it:
  1. appends the incident to evals/incidents.yaml (what broke, how it was found, root cause, blast radius, fix, guard)
  2. appends a new eval to evals/registry.yaml when one is given (text append, comments and order kept)
  3. appends a dated lesson to each named skill's "## Lessons from incidents" section (.claude/skills/<name>/SKILL.md)
LOSSLESS by construction: only appends; before writing it backs up, after writing it re-reads and checks that every eval
id, every incident id and every lesson line that existed before still exists, else it restores the backups. `check`
compares the files with the ledger evals/.lessons_ledger.json (every eval id / incident / lesson ever written); the
daily eval trust.lessons_intact runs it, so a later edit that deletes a lesson or an eval blocks the run.
Usage:
  incident.py add --id INC-YYYY-MM-DD-slug --title T --found-by F --root-cause R --blast-radius B --fix F
                  [--guard-eval EVAL_ID] [--eval-file path/to/eval_block.yaml] [--skills a,b] [--lesson "..."] [--commit SHA]
  incident.py check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
REG = ROOT / "evals/registry.yaml"
INC = ROOT / "evals/incidents.yaml"
LEDGER = ROOT / "evals/.lessons_ledger.json"
SKILLS = ROOT / ".claude/skills"
BACKUP = ROOT / "evals/.backups"
HEAD = "## Lessons from incidents"


def _ids() -> set[str]:
    return {e["id"] for e in yaml.safe_load(REG.read_text())["evals"]}


def _incident_ids() -> set[str]:
    return {i["id"] for i in (yaml.safe_load(INC.read_text()) or {}).get("incidents", [])} if INC.exists() else set()


def _lessons(skill: str) -> list[str]:
    f = SKILLS / skill / "SKILL.md"
    if not f.exists() or HEAD not in f.read_text():
        return []
    return [l.strip() for l in f.read_text().split(HEAD, 1)[1].splitlines() if l.strip().startswith("- ")]


def _ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"evals": [], "incidents": [], "lessons": {}}


def check() -> tuple[bool, list[str]]:
    """Everything the ledger has ever recorded must still be in the files."""
    L, problems = _ledger(), []
    ids = _ids()
    problems += [f"eval removed: {e}" for e in L["evals"] if e not in ids]
    problems += [f"incident removed: {i}" for i in L["incidents"] if i not in _incident_ids()]
    for skill, lines in L["lessons"].items():
        have = set(_lessons(skill))
        problems += [f"lesson removed from {skill}: {h}" for h in lines if h not in {hashlib.sha1(x.encode()).hexdigest()[:12] for x in have}]
    return (not problems), problems


def _snapshot_ledger() -> None:
    L = _ledger()
    L["evals"] = sorted(set(L["evals"]) | _ids())
    L["incidents"] = sorted(set(L["incidents"]) | _incident_ids())
    for d in SKILLS.iterdir() if SKILLS.exists() else []:
        hs = {hashlib.sha1(x.encode()).hexdigest()[:12] for x in _lessons(d.name)}
        if hs:
            L["lessons"][d.name] = sorted(set(L["lessons"].get(d.name, [])) | hs)
    LEDGER.write_text(json.dumps(L, indent=1))


def add(a) -> None:
    ok, probs = check()
    if not ok:
        sys.exit("refusing to write: the files already lost something the ledger remembers:\n  " + "\n  ".join(probs))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKUP.mkdir(parents=True, exist_ok=True)
    touched = [REG, INC] + [SKILLS / s / "SKILL.md" for s in (a.skills.split(",") if a.skills else [])]
    for f in touched:
        if f.exists():
            shutil.copy2(f, BACKUP / f"{ts}__{str(f.relative_to(ROOT)).replace('/', '__')}")
    before_ids, before_inc = _ids(), _incident_ids()
    before_lessons = {s: set(_lessons(s)) for s in (a.skills.split(",") if a.skills else [])}
    try:
        if a.id in before_inc:
            sys.exit(f"{a.id} already recorded")
        if a.eval_file:
            block = Path(a.eval_file).read_text().rstrip() + "\n"
            new_ids = {e["id"] for e in yaml.safe_load("evals:\n" + block)["evals"]}
            if new_ids & before_ids:
                sys.exit(f"eval id already exists: {new_ids & before_ids} (edit it with a history note instead)")
            REG.write_text(REG.read_text().rstrip() + "\n\n" + block)
        inc = dict(id=a.id, date=str(date.today()), title=a.title, found_by=a.found_by, root_cause=a.root_cause,
                   blast_radius=a.blast_radius, fix=a.fix, guard_eval=a.guard_eval, skills=a.skills.split(",") if a.skills else [],
                   lesson=a.lesson, commit=a.commit)
        txt = INC.read_text() if INC.exists() else "# Incidents: what broke, how we found it, and what now guards it (append-only; src/agentic/trust/incident.py)\nincidents:\n"
        dump = yaml.safe_dump([inc], sort_keys=False, allow_unicode=True, width=120)
        INC.write_text(txt.rstrip() + "\n" + "\n".join("  " + l for l in dump.splitlines()) + "\n")
        if a.lesson:
            for s in a.skills.split(",") if a.skills else []:
                f = SKILLS / s / "SKILL.md"
                if not f.exists():
                    continue
                body = f.read_text().rstrip()
                if HEAD not in body:
                    body += f"\n\n{HEAD}\n(appended by src/agentic/trust/incident.py; never edit or delete these lines)"
                f.write_text(body + f"\n- {date.today()} [{a.id}] {a.lesson}\n")
        # verify: nothing that existed before is gone, and the files still parse
        assert before_ids <= _ids(), "an eval disappeared"
        assert before_inc <= _incident_ids(), "an incident disappeared"
        for s, ls in before_lessons.items():
            assert ls <= set(_lessons(s)), f"a lesson disappeared from {s}"
        if a.guard_eval:
            assert a.guard_eval in _ids(), f"guard eval {a.guard_eval} is not in the registry"
    except (AssertionError, yaml.YAMLError) as x:
        for f in touched:
            b = BACKUP / f"{ts}__{str(f.relative_to(ROOT)).replace('/', '__')}"
            if b.exists():
                shutil.copy2(b, f)
        sys.exit(f"rolled back: {x}")
    _snapshot_ledger()
    print(f"recorded {a.id}" + (f" · eval {a.guard_eval}" if a.guard_eval else "") + (f" · lessons in {a.skills}" if a.lesson and a.skills else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    for k in ("id", "title", "found-by", "root-cause", "blast-radius", "fix"):
        p.add_argument(f"--{k}", required=True)
    for k in ("guard-eval", "eval-file", "skills", "lesson", "commit"):
        p.add_argument(f"--{k}", default="")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "add":
        add(a)
    else:
        ok, probs = check()
        print("lossless: OK" if ok else "LOST:\n  " + "\n  ".join(probs))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
