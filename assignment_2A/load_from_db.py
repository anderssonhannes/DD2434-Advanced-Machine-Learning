import pandas as pd
import numpy as np
from geopy.distance import geodesic
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('worldcities.csv')
df = df[df.capital == 'primary']
df = df.dropna(subset=['population'],axis=0)

min_pop = 100000
df = df[df.population>min_pop]
df.reset_index(inplace=True,drop=True)

n = len(df)
D = np.zeros((n,n))

for country_1 in df.iterrows():
    id1 = country_1[0]
    for country_2 in df[id1:].iterrows():
        id2 = country_2[0]
        if id1 == id2:
            continue
        else:
            distance = geodesic([country_1[1]['lat'],country_1[1]['lng']],[country_2[1]['lat'],country_2[1]['lng']])
            D[id1][id2] = distance.km
            D[id2][id1] = distance.km
print('Done building D matrix')

# Start to calculate
N = len(D)

# ddf = pd.DataFrame(D,columns=df.country)
# ddf = ddf.set_index(df.country,drop=True)

centering_matrix = lambda n: np.identity(n) -(np.ones((n, 1)) @ np.ones((1, n)))/n

S = -(centering_matrix(N) @ D @ centering_matrix(N))/2

# S = -0.5*( D - D.mean(axis=1) - D.mean(axis=0) + D.mean() )
W,U = LA.eig(S)
idx = W.argsort()[::-1]
W = W[idx]
U = U[:,idx]
# Remove negative eigenvalues
W[W<0] = 0

lamb = np.diag(W)
I_kn = np.eye(2,N)
tmp = np.matmul(I_kn,np.sqrt(lamb))
X = -np.matmul(tmp,U.T).T
# X[:,0] = -X[:,0]
scaler1 = MinMaxScaler()
X_scaled = scaler1.fit_transform(X)

for i, cord in enumerate(X_scaled):
    element = df.iloc[i]
    color = 'red'
    # color = color_dict[element.continent]
    txt = element.country[:6]

    plt.plot(cord[0],cord[1],marker='.',color=color)
    plt.text(cord[0],cord[1],txt)

    # plt.plot(exact[i][0],exact[i][1],marker='*',color=color)
    # plt.text(exact[i][0],exact[i][1],txt)
plt.show()


print(df.to_string())