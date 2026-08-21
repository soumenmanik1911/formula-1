import pathlib
content = '''
import requests
import time
import subprocess
import sys

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=r"D:\\final show\\f1",
)
time.sleep(3)

try:
    r1 = requests.get("http://127.0.0.1:8000/predict/race/2025/1")
    print("Status:", r1.status_code)
    print(r1.json())
    print()
    r2 = requests.get("http://127.0.0.1:8000/predict/wdc/2025")
    print("Status:", r2.status_code)
    print(r2.json())
    print()
    r3 = requests.get("http://127.0.0.1:8000/predict/race/2025/999")
    print("Status:", r3.status_code)
    print(r3.text)
finally:
    proc.terminate()
    proc.wait()
'''
pathlib.Path(r'D:\\final show\\f1\\app\\ml\\test_endpoints.py').write_text(content)
