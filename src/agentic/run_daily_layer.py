"""Cron wrapper for daily_data_layer.sh.

macOS TCC blocks /bin/bash from reading ~/Documents when launched by cron/launchd
("Operation not permitted" — killed the daily layer Aug-31, Sep-1, Sep-2), while
/usr/bin/python3 has disk access (proof: the cian watcher cron runs daily).
Launching bash as a child of python3 makes python the TCC-responsible process,
so the pipeline runs. Cron calls THIS file; it just execs the bash script.
"""
import subprocess
import sys

r = subprocess.run(["/bin/bash", "/Users/abhinavs./Documents/Zoom/src/agentic/daily_data_layer.sh"])
sys.exit(r.returncode)
