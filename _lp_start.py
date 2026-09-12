import os, time, datetime

# Get start time of pid 1512
pids = os.popen('tasklist /FI "PID eq 1512" /FO CSV /V /NH').read().strip()
lines = [l for l in pids.split('\n') if '1512' in l]
if lines:
    # CSV: "python.exe","1512","Console","1","537,060 K","...",...,"9/12/2026 3:27:04 PM","...",...
    parts = lines[0].split(',')
    # Find the start time field (usually one of the later fields)
    for i, p in enumerate(parts):
        p = p.strip().strip('"')
        if '/' in p and ':' in p and '2026' in p:
            start_time = p
            break
    else:
        start_time = 'UNKNOWN'
else:
    start_time = 'NOT_FOUND'

mtime = time.ctime(os.path.getmtime('c:/Users/user/Desktop/1minit/data/live_prediction.html'))

with open('c:/Users/user/Desktop/1minit/_lp_start_result.txt', 'w') as f:
    f.write(f'PID=1512\n')
    f.write(f'START_TIME={start_time}\n')
    f.write(f'MTIME={mtime}\n')
    f.write(f'PID_CHECK={pids}\n')
print('done')
