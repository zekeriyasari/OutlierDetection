import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from scipy.io import loadmat

# Get the data
dataset = loadmat("../Ionosphere/ionosphere.mat")
data = dataset["X"][:, [1, 5]]
contamination = 0.36
num_samples = np.shape(data)[0]
num_neighbors = 35

# Construct the outlier detector
clf = LocalOutlierFactor(n_neighbors=num_neighbors, contamination=contamination)

# Perform outlier detection# clf = IsolationForest()
predicted_data = clf.fit_predict(data)
inlier_predicted_data = data[predicted_data == 1]
outlier_predicted_data = data[predicted_data == -1]
num_inliers_predicted = inlier_predicted_data.shape[0]
num_outliers_predicted = outlier_predicted_data.shape[0]

# Calculate outlier scores
xr = np.linspace(-2, 2, 500)
yr = np.linspace(-2, 2, 500)
xx, yy = np.meshgrid(xr, yr)
zz = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)
scores = clf.negative_outlier_factor_
threshold = stats.scoreatpercentile(scores, 100 * contamination)
plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), threshold, 7), cmap=plt.cm.Blues_r)  # Outlier
plt.contour(xx, yy, zz, levels=np.array([threshold]), linewidths=2, colors="red")  # The frontier
plt.contourf(xx, yy, zz, levels=np.linspace(threshold, zz.max(), 7), colors="orange")  # Inlier

# Plot the sets
plt.scatter(inlier_predicted_data[:, 0], inlier_predicted_data[:, 1], c="white", s=10, edgecolors="black",
            label="Inliers")
plt.scatter(outlier_predicted_data[:, 0], outlier_predicted_data[:, 1], c="black", s=10, edgecolors="black",
            label="Outliers")
plt.title("Inliers={} Outliers={}".format(num_inliers_predicted, num_outliers_predicted))
plt.xlabel("Local Outlier Factor. n_neighbors={}, contamination={}".format(num_neighbors, contamination))
plt.legend()
plt.show()
