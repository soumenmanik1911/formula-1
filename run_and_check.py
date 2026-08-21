import subprocess, time, requests, json

p = subprocess.Popen(['python', '-m', 'uvicorn', 'app.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'])
time.sleep(4)
base = 'http://localhost:8000'

r = requests.get(f'{base}/features/2023/10')
print('2023 R10 features:', r.status_code, len(r.json()), 'rows')
print('Top 3:')
for row in r.json()[:3]:
    print(f'  {row["driver_name"]}: elo={row["driver_elo"]:.1f}, finish={row["finish_position"]}')

r = requests.get(f'{base}/features/2024/1')
print('\n2024 R1 features:', r.status_code, len(r.json()), 'rows')
print('Top 3:')
for row in r.json()[:3]:
    print(f'  {row["driver_name"]}: elo={row["driver_elo"]:.1f}, finish={row["finish_position"]}')
