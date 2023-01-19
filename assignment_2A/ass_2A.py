import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load matrix and data set
D = np.load('D.npy')
df = pd.read_pickle('df.pkl')
N = len(D)

# Create the centering matrix
centering_matrix = lambda n: np.identity(n) -(np.ones((n, 1)) @ np.ones((1, n)))/n

# Create the similarity matrix
S = -(centering_matrix(N) @ D @ centering_matrix(N))/2

# Eigen-decomposition
W,U = LA.eig(S)
# Sort them
idx = W.argsort()[::-1]
W = W[idx]
U = U[:,idx]
# Remove negative eigenvalues
W[W<0] = 0

# Calculate 2-dimensional representation
lamb = np.diag(W)
I_kn = np.eye(2,N)
tmp = np.matmul(I_kn,np.sqrt(lamb))
X = np.matmul(tmp,U.T).T

# Scale the data between 0-1. 
scaler1 = MinMaxScaler()
X_scaled = scaler1.fit_transform(X)

# Preparations for the plot
conts = df.continent.unique()
color_dict = dict()
colors = ['red','orange','blue','yellow','magenta','sienna','cyan']
colors = colors[:len(conts)]
for k, cont in enumerate(conts): color_dict[cont] = colors[k]

# Plot the x,y coordinates
for i, cord in enumerate(X_scaled):
    element = df.iloc[i]
    color = color_dict[element.continent]
    txt = element.capital[0:2]

    plt.scatter(cord[0],cord[1],marker='.',color=color,label=element.continent)
    plt.text(cord[0],cord[1],txt)
    plt.title('Capitals over the world')
# Save figures
plt.savefig('capitals.png')
plt.show()



        
        