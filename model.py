import json
import requests


url = "https://recruitment.aimtechnologies.co/ai-tasks"
response = requests.post(url, data = json.dumps(['1175358310087892992']))
print(response.text)
