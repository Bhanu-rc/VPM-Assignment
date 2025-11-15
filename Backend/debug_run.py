print('DEBUG: starting debug_run.py')
import sys, runpy, traceback, pathlib
print('DEBUG: sys.executable =', sys.executable)
p = pathlib.Path('build_index.py')
print('DEBUG: build_index.py exists =', p.exists())
if p.exists():
    print('DEBUG: build_index.py head (first 400 chars):')
    print(p.read_text()[:400].replace('\n','\\n'))
try:
    runpy.run_path('build_index.py', run_name='__main__')
except Exception:
    traceback.print_exc()
print('DEBUG: debug_run.py finished')
