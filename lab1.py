from numpy.core.fromnumeric import mean
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import statistics

CLUSTER_NUMBER = 3

X, y = make_blobs(n_samples=100, centers=CLUSTER_NUMBER, n_features=2, random_state=0)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_title("Input")
ax1.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')

kmeans = KMeans(n_clusters=CLUSTER_NUMBER, random_state=0).fit(X)

ax2.set_title("Output")
ax2.scatter(X[:, 0], X[:, 1], marker='o', c=kmeans.labels_, s=25, edgecolor='k')

methods = [statistics.mean, statistics.median, statistics.quantiles]

columnNames = ['Cluster']

for method in methods:
    columnNames.append(method.__name__ + " X")
    columnNames.append(method.__name__ + " Y")

t = PrettyTable(columnNames)

for i in range(CLUSTER_NUMBER):
    cluster = [X[j] for j in range(len(X)) if kmeans.labels_[j] == i]
    clusterX = [cluster[j][0] for j in range(len(cluster))]
    clusterY = [cluster[j][1] for j in range(len(cluster))]
    rowValues = [i]
    for method in methods:
        rowValues.append(method(clusterX))
        rowValues.append(method(clusterY))
    t.add_row(rowValues)

print(t)

plt.show()