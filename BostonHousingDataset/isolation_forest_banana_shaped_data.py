# OneClassSVM outlier detection

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_boston
from scipy import stats

# Get the data
dataset = load_boston()
data = dataset["data"][:, [5, 12]]  # Two cluster data
contamination = 0.261
num_samples = data.shape[0]

# Construct the outlier detector
clf = IsolationForest(contamination=contamination, max_samples=num_samples)
clf.fit(data)

# Perform outlier detection# clf = IsolationForest()
predicted_data = clf.predict(data)
inlier_predicted_data = data[predicted_data == 1]
outlier_predicted_data = data[predicted_data == -1]
num_inliers_predicted = inlier_predicted_data.shape[0]
num_outliers_predicted = outlier_predicted_data.shape[0]

# # Plot decision function values
xr = np.linspace(3, 10, 500)
yr = np.linspace(-5, 45, 500)
xx, yy = np.meshgrid(xr, yr)
zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)
scores = clf.decision_function(data)
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
plt.xlabel("Isolation Forest. contamination={}, max_samples={}".format(contamination, num_samples))
plt.legend()
plt.show()
