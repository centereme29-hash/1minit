import os, time, re

# Check process
pids = os.popen('tasklist /FI "PID eq 1512" /FO CSV /NH').read().strip()
alive = '1512' in pids

# Check HTML
html_path = 'c:/Users/user/Desktop/1minit/data/live_prediction.html'
mtime = time.ctime(os.path.getmtime(html_path))
size = os.path.getsize(html_path)

with open(html_path, encoding='utf-8') as f:
    html = f.read()

# Extract timestamp from HTML header
ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC', html)
latest_ts = ts_match.group(1) if ts_match else 'NOT_FOUND'

# Extract action
action_match = re.search(r'<div style="font-size:46px[^"]*"[^>]*>([^<]+)</div>', html)
action = action_match.group(1).strip() if action_match else 'NOT_FOUND'

# Extract confidence
conf_match = re.search(r'confidence ([\d.]+)%', html)
conf = conf_match.group(1) if conf_match else 'NOT_FOUND'

# Extract price
price_match = re.search(r'price ([\d.]+)', html)
price = price_match.group(1) if price_match else 'NOT_FOUND'

result = f"""ALIVE={alive}
PID_CHECK={pids}
MTIME={mtime}
SIZE={size}
LATEST_TS={latest_ts}
ACTION={action}
CONFIDENCE={conf}
PRICE={price}
"""
with open('c:/Users/user/Desktop/1minit/_live_result.txt', 'w') as f:
    f.write(result)
print(result)
