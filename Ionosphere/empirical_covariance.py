# One-Class SVM for real dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from scipy.io import loadmat
from scipy import stats

# Get the data
dataset = loadmat("../Ionosphere/ionosphere.mat")
data = dataset["X"][:, [1, 5]]
contamination = 0.36

# Fit the model
clf = EllipticEnvelope(support_fraction=1., contamination=contamination)
clf.fit(data)

# Perform outlier detection
predicted_data = clf.predict(data)
inlier_predicted_data = data[predicted_data == 1]
outlier_predicted_data = data[predicted_data == -1]
num_inliers_predicted = inlier_predicted_data.shape[0]
num_outliers_predicted = outlier_predicted_data.shape[0]

# Plot decision function values
xr = np.linspace(-2, 2, 500)
yr = np.linspace(-2, 2, 500)
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
plt.xlabel("Elliptic Envelope(Empirical Covariance Estimate). contamination={}".format(contamination))
plt.legend()
plt.show()
