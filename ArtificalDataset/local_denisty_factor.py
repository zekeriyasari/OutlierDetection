# Outlier detection using LocalOutlierFactor and artificial dataset.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# Set rng seed
np.random.seed(42)

# Example settings
num_samples = 200
contamination = 0.25
num_outliers = int(num_samples * contamination)
num_inliers = num_samples - num_outliers
num_neighbors = 35

# Construct the data set
offset = 2
data_inliers_1 = 0.3 * np.random.randn(num_inliers // 2, 2) - offset
data_inliers_2 = 0.3 * np.random.randn(num_inliers // 2, 2) + offset
data_inliers = np.r_[data_inliers_1, data_inliers_2]
data_outliers = np.random.uniform(low=-5, high=5, size=(num_outliers, 2))
data = np.r_[data_inliers, data_outliers]

# Construct the outlier detector
clf = LocalOutlierFactor(n_neighbors=num_neighbors, contamination=contamination)

# Perform outlier detection# clf = IsolationForest()
predicted_data = clf.fit_predict(data)
inlier_predicted_data = data[predicted_data == 1]
outlier_predicted_data = data[predicted_data == -1]
num_inliers_predicted = inlier_predicted_data.shape[0]
num_outliers_predicted = outlier_predicted_data.shape[0]

# Calculate outlier scores
xr = np.linspace(-6, 6, 600)
yr = np.linspace(-6, 6, 600)
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
