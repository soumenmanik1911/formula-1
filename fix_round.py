import re
path = r'D:\final show\f1\app\routers\predict.py'
with open(path, 'r') as f:
    content = f.read()
content = content.replace('def predict_race(season: int, round: int):', 'def predict_race(season: int, race_round: int):')
content = content.replace('.filter(DriverRaceFeature.round == round)', '.filter(DriverRaceFeature.round == race_round)')
with open(path, 'w') as f:
    f.write(content)
print('updated')
