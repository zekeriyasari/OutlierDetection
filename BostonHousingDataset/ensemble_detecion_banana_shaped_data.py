# Outlier detection using ensemble of outlier detection techniques and Boston housing dataset(banana-shaped data)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from sklearn.datasets import load_boston

# Set rng seed
np.random.seed(42)

# Example settings
num_samples = 200
contamination = 0.25
num_outliers = int(num_samples * contamination)
num_inliers = num_samples - num_outliers
gamma = 0.1
num_neighbors = 35

# Get the data
dataset = load_boston()
data = dataset["data"][:, [5, 12]]  # Two cluster data
contamination = 0.261

# Construct the classifiers.
ensemble = dict(oneclasssvm=OneClassSVM(kernel="rbf", gamma=gamma, nu=contamination),
                elliptic_envelope=EllipticEnvelope(contamination=contamination),
                isolation_forest=IsolationForest(contamination=contamination, max_samples=num_samples),
                local_outlier_factor=LocalOutlierFactor(n_neighbors=num_neighbors, contamination=contamination))
ensemble_predicted_data = dict()

# Fit the data for different classifiers
for name, clf in ensemble.items():
    if name.startswith("local_outlier_factor"):
        predicted_data = clf.fit_predict(data)
    else:
        clf.fit(data)
        predicted_data = clf.predict(data)
    ensemble_predicted_data[name] = predicted_data

    # Perform outlier detection
    inlier_predicted_data = data[predicted_data == 1]
    outlier_predicted_data = data[predicted_data == -1]
    num_inliers_predicted = inlier_predicted_data.shape[0]
    num_outliers_predicted = outlier_predicted_data.shape[0]

    # Plot decision function values
    xr = np.linspace(3, 10, 500)
    yr = np.linspace(-5, 45, 500)
    xx, yy = np.meshgrid(xr, yr)
    if name.startswith("local_outlier_factor"):
        zz = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
        scores = clf.negative_outlier_factor_
    else:
        zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        scores = clf.decision_function(data)
    zz = zz.reshape(xx.shape)
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

scores = np.vstack([predicted_data for predicted_data in ensemble_predicted_data.values()]).sum(axis=0)
first_degree_inlier_data = data[scores == 4]
second_degree_inlier_data = data[scores == 2]
undetermined_data = data[scores == 0]
second_degree_outlier_data = data[scores == -2]
first_degree_outlier_data = data[scores == -4]
num_inliers_predicted = first_degree_inlier_data.shape[0] + second_degree_inlier_data.shape[0]
num_outliers_predicted = first_degree_outlier_data.shape[0] + second_degree_outlier_data.shape[0]
num_undetermined = data.shape[0] - num_inliers_predicted - num_outliers_predicted

# Plot the frontiers
colors = {"oneclasssvm": "r", "elliptic_envelope": "m", "isolation_forest": "g", "local_outlier_factor": "k"}
xr = np.linspace(3, 10, 500)
yr = np.linspace(-5, 45, 500)
legends = {}
xx, yy = np.meshgrid(xr, yr)
for name, clf in ensemble.items():
    # Plot decision function values
    xr = np.linspace(3, 10, 500)
    yr = np.linspace(-5, 45, 500)
    xx, yy = np.meshgrid(xr, yr)
    if name.startswith("local_outlier_factor"):
        zz = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
        scores = clf.negative_outlier_factor_
    else:
        zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        scores = clf.decision_function(data)
    zz = zz.reshape(xx.shape)
    threshold = stats.scoreatpercentile(scores, 100 * contamination)
    legends[name] = plt.contour(xx, yy, zz, levels=np.array([threshold]), linewidths=2,
                                colors=colors[name])  # The frontier

legends["1st Inl"] = plt.scatter(first_degree_inlier_data[:, 0], first_degree_inlier_data[:, 1], s=10,
                                 edgecolors="black")
legends["2nd Inl"] = plt.scatter(second_degree_inlier_data[:, 0], second_degree_inlier_data[:, 1], s=10,
                                 edgecolors="black")
legends["Und"] = plt.scatter(undetermined_data[:, 0], undetermined_data[:, 1], s=10, edgecolors="black")
legends["2nd Outl"] = plt.scatter(second_degree_outlier_data[:, 0], second_degree_outlier_data[:, 1],
                                  s=10, edgecolors="black")
legends["1st Outl"] = plt.scatter(first_degree_outlier_data[:, 0], first_degree_outlier_data[:, 1], s=10,
                                  edgecolors="black")
legend1_values_list = list(legends.values())
legend1_keys_list = list(legends.keys())
plt.legend((legend1_values_list[0].collections[0],
            legend1_values_list[1].collections[0],
            legend1_values_list[2].collections[0],
            legend1_values_list[3].collections[0]),
           (legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2], legend1_keys_list[3]))
plt.title("1st Inl={}, 2nd Inl={}, Und={}, "
          "2nd Outl={}, 1st Outl={}".format(first_degree_inlier_data.shape[0],
                                            second_degree_inlier_data.shape[0],
                                            undetermined_data.shape[0],
                                            second_degree_outlier_data.shape[0],
                                            first_degree_outlier_data.shape[0]))
plt.xlabel("Ensemble outlier detector".format(num_neighbors, contamination))
plt.show()
