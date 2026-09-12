import os, time

# Check if pid 1512 is alive
pids = os.popen('tasklist /FI "PID eq 1512" /FO CSV /NH').read().strip()
alive = '1512' in pids

# Check live_prediction.html mtime
mtime = time.ctime(os.path.getmtime('c:/Users/user/Desktop/1minit/data/live_prediction.html'))

with open('c:/Users/user/Desktop/1minit/_lp_result.txt', 'w') as f:
    f.write(f'ALIVE={alive}\n')
    f.write(f'PID_CHECK={pids}\n')
    f.write(f'MTIME={mtime}\n')
    f.write(f'SIZE={os.path.getsize("c:/Users/user/Desktop/1minit/data/live_prediction.html")}\n')
print('done')
