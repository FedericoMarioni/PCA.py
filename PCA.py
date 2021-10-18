# How to make up some data that we can apply PCA to
# How to do the PCA() function from sklearn to do PCA
# How to determine how much variation each principal component accounts for
# How to draw a PCA graph using matplotlib
# How to determine the loading scores to determine what variables have the largest effect on the graph

import pandas as pd  # used for loading and manipulating data
import numpy as np  # data manipulation and generate random numerical data
import random as rd  # useful for generating random example data set. If you are working with real data
#  you do not need this
from sklearn.decomposition import PCA  # importing the PCA algorithm to perform dimensionality reduction
from sklearn import preprocessing  # give us functions for scaling (normalize) the data before performing PCA
import matplotlib.pyplot as plt  # graphs

# generate a sample data set

genes = ['gene' + str(i) for i in range(1, 101)]
wt = ['wt' + str(i) for i in range(1, 6)]   # wild type samples   [1,2,3,4,5]
ko = ['ko' + str(i) for i in range(1, 6)]   # knock out samples   [1,2,3,4,5]

data = pd.DataFrame(columns=[*wt, *ko], index=genes)

for gene in data.index:
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)

#  print(data.head())

# before we do PCA we need to center and scale the data (normalize)
#  after centering the avg. value for each gene will be 0 and after scaling the std.dev for the values of each
#  gene will be 1
scaled_data = preprocessing.scale(data.T)

# alternative method: Standard.Scaler().fit_transform(data,T) [more common in Machine Learning][Using sklearn]

pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# now we draw a Scree plot to see how many principal components should go into the final plot

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal component')
plt.title('Scree Plot')
plt.show()

#  almost all of the variation is along the first PC, so a 2-D graph, using PC 1 and PC 2, should do a good
#  job representing the original data

#  draw a PCA plot

pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('PCA graph')
plt.xlabel('PC1 -{0}%'.format(per_var[0]))
plt.ylabel('PC2 -{0}%'.format(per_var[1]))

#  this loop adds samples names to the graph

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

#  the 'wt' samples are clustered on the left side, suggesting that they are correlated with each other
#  the 'ko' samples are clustered on the right side, suggesting that they are correlated with each other
#  the separation of the two clusters along the x-axis suggests that 'wt' samples are very different
#  from 'ko' samples

# Now let´s look at the loading scores from PC1 to determine which genes had the largest influence
# on separating the two clusters along the x-axis

loading_scores = pd.Series(pca.components_[0], index=genes)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_genes = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_genes])

# This values are super similar, so a lot of genes play a role in separating the samples
# rather than just one or two























