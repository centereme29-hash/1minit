import os, time, subprocess, sys

ROOT = r"c:\Users\user\Desktop\1minit"
marker = os.path.join(ROOT, "_DONE.marker")
mon = os.path.join(ROOT, "_mon.py")
py = sys.executable

while True:
    subprocess.run([py, mon], cwd=ROOT)
    if os.path.exists(marker):
        break
    time.sleep(60)
