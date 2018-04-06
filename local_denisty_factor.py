# OneClassSVM outlier detection

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

# Construct the data set
offset = 2
x_inliers_1 = 0.3 * np.random.randn(num_inliers // 2, 2) - offset
x_inliers_2 = 0.3 * np.random.randn(num_inliers // 2, 2) + offset
x_inliers = np.r_[x_inliers_1, x_inliers_2]
x_outliers = np.random.uniform(low=-5, high=5, size=(num_outliers, 2))
x = np.r_[x_inliers, x_outliers]

# Construct the outlier detector
clf = LocalOutlierFactor(n_neighbors=35, contamination=contamination)
y = clf.fit_predict(x)
scores = clf.negative_outlier_factor_
threshold = stats.scoreatpercentile(scores, 100 * contamination)

# Calculate outlier scores
xr = np.linspace(-5, 5, 500)
yr = np.linspace(-5, 5, 500)
xx, yy = np.meshgrid(xr, yr)
zz = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)

# # Plot decision function values
plt.contourf(xx, yy, zz, levels=np.linspace(zz.min(), threshold, 7), cmap=plt.cm.Blues_r)  # Outlier
plt.contour(xx, yy, zz, levels=np.array([threshold]), linewidths=2, colors="red")  # The frontier
plt.contourf(xx, yy, zz, levels=np.linspace(threshold, zz.max(), 7), colors="orange")  # Inlier

# Plot the set
plt.scatter(x_inliers[:, 0], x_inliers[:, 1], c="white", s=20, edgecolors="black", label="Inliers")
plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c="black", s=20, edgecolors="black", label="Outliers")
plt.legend()
plt.show()
