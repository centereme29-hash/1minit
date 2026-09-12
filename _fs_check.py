import ast
import io

for f in (r"c:/Users/user/Desktop/1minit/src/full_system.py",
          r"c:/Users/user/Desktop/1minit/src/run_live.py"):
    src = io.open(f, encoding="utf-8").read()
    ast.parse(src)
    print("AST OK", f.split("/")[-1], len(src.splitlines()), "lines")