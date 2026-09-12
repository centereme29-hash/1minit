import pandas as pd, time

cache_path = 'c:/Users/user/Desktop/1minit/data/raw/bybit_doge_usdt.feather'
raw = pd.read_feather(cache_path)

print(f"Cache: {cache_path}")
print(f"Rows: {len(raw):,}")
print(f"Columns: {list(raw.columns)}")
print(f"Time range: {raw['time'].min()} to {raw['time'].max()}")
print(f"Latest timestamp: {raw['time'].max()}")
print(f"Cache file mtime: {time.ctime(__import__('os').path.getmtime(cache_path))}")
print(f"Data age (from latest timestamp): {time.time() - raw['time'].max().timestamp():.0f} seconds = {__import__('os').path.getmtime(cache_path)}")
