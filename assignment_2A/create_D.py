import requests
import json
import pandas as pd
from collections import defaultdict
import numpy as np
from geopy.distance import geodesic


response_API = requests.get('https://restcountries.com/v3.1/all')

data = response_API.text
parse_json = json.loads(data)
d = defaultdict(lambda: "Not present")

# Create a dictionary with all the nessecary information about the capitals
for country in parse_json:
    d_temp = dict()
    try: 
        # if (country['population'] < 1e6) or country['continents'][0] == 'Oceania': continue  
        if (country['population'] < 1e6): continue  # All capitals
        name = country['name']['common']
        if name == 'Hong Kong': continue
        d_temp['capital'] = country['capital'][0]
        d_temp['latlng'] = country['capitalInfo']['latlng']  
        d_temp['continent'] = country['continents'][0]
        d[name] = d_temp    
    except:
        continue

# Create the D matrix 
n = len(d)
D = np.zeros((n,n))
df = pd.DataFrame.from_dict(d, orient='index')
df['country'] = df.index
df.reset_index(inplace=True,drop=True)

# Calculate distance pairwise between all cities
for country_1 in df.iterrows():
    id1 = country_1[0]
    for country_2 in df[id1:].iterrows():
        id2 = country_2[0]
        if id1 == id2:
            continue
        else:
            distance = geodesic(country_1[1]['latlng'],country_2[1]['latlng'])
            D[id1][id2] = distance.km
            D[id2][id1] = distance.km
# Save the matrix and the dataframe
np.save('D',D)
df.to_pickle('df.pkl')

