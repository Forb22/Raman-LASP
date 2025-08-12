import numpy as np #file manipulation
import pandas as pd #data manipulation
from sklearn.preprocessing import StandardScaler #scales data so columns that vary by 100's do not overshadow columns that vary by 0.1's
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt #save plot
import seaborn as sns #pair plot

datafile = '/Users/guy/Desktop/Sherbrooke_Lab_Data/Raman Data/ML DATA All.csv'
data = np.genfromtxt(datafile,delimiter=',',unpack=False, skip_header = 1, usecols=(range(1,22)),dtype = float)

X = data[:,1:] #features
Y = data[:,0] #target

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

pca = PCA(n_components = 4) #n_components decides how much of the variance is explained (if between 0 and 1),
                            #or decides how many principal components there are (if >1).

principal_components = pca.fit_transform(data_scaled)
df = pd.DataFrame(principal_components)
df['Temperature'] = Y

#create a pair plot
sns.pairplot(df, hue = 'Temperature', palette='viridis',plot_kws={'alpha': 0.3})
#save pair plot
plt.savefig('pca_pairplot_all.png')

#PCA Metrics
print(f'Original data shape: {np.shape(data)}')
print(f"Transformed data shape: {principal_components.shape}")
print(f"Number of components chosen: {pca.n_components_}")
print(f"Explained variance per component: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")