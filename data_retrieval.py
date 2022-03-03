import json
import requests
import pandas as pd

url = 'https://recruitment.aimtechnologies.co/ai-tasks'
data_with_text = pd.DataFrame(columns=['id','text','dialect'])
with pd.read_csv("dialect_dataset.csv",chunksize=1000) as reader:
    for chunk in reader:
        response = requests.post(url, data = json.dumps(list(map(str,chunk['id']))))
        chunk['text'] = dict(response.json()).values()
        data_with_text = pd.concat([data_with_text , chunk])
print(data_with_text.info())
