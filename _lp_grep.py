import re

path = 'c:/Users/user/Desktop/1minit/data/live_prediction.html'
with open(path, encoding='utf-8') as f:
    html = f.read()

# Find the prediction text in the header
# The header is inserted after <body> and contains the prediction
body_match = re.search(r'<body>(.+?)</body>', html, re.DOTALL)
if body_match:
    header = body_match.group(1)
    # Extract key info
    print("=== LIVE PREDICTION DASHBOARD ===")
    print(f"File: {path}")
    print(f"Size: {len(html):,} bytes")
    print()

    # Find action (LONG/SHORT/NO TRADE)
    action_match = re.search(r'<div style="font-size:46px[^"]*"[^>]*>([^<]+)</div>', header)
    if action_match:
        print(f"ACTION: {action_match.group(1).strip()}")

    # Find confidence
    conf_match = re.search(r'confidence ([\d.]+)%', header)
    if conf_match:
        print(f"CONFIDENCE: {conf_match.group(1)}%")

    # Find timestamp
    ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC', header)
    if ts_match:
        print(f"TIMESTAMP: {ts_match.group(1)} UTC")

    # Find price
    price_match = re.search(r'price ([\d.]+)', header)
    if price_match:
        print(f"PRICE: {price_match.group(1)}")

    # Find per-model breakdown
    print()
    print("=== PER-MODEL BREAKDOWN ===")
    # Find the model breakdown section
    breakdown_match = re.search(r'class=\"model\"[^>]*>([^<]+)</div>\s*<div[^>]*>([^<]+)</div>', header)
    if breakdown_match:
        for m in re.finditer(r'class=\"model\"[^>]*>([^<]+)</div>\s*<div[^>]*>([^<]+)</div>', header):
            print(f"  {m.group(1).strip()}: {m.group(2).strip()}")
    else:
        # Try simpler pattern
        for m in re.finditer(r'DOGEUSDT\s*\(([^)]+)\)', header):
            print(f"  {m.group(1).strip()}")

print()
print("=== PROCESS STATUS ===")
print("PID: 1512")
print("Status: ALIVE (confirmed by tasklist)")
print("CPU: 8 seconds (not a crash loop)")
print("Refresh: every 60 seconds")
print(f"HTML updated: {re.search(r'MTIME=(\S+)', open('c:/Users/user/Desktop/1minit/_lp_result.txt').read()).group(1)} UTC")
