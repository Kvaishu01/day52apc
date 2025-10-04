# day52_apc.py
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation

# Title
st.title("Day 52 - Affinity Propagation Clustering (APC) 🚀")

st.write("""
### What is Affinity Propagation?
- Unlike k-means, you don't need to specify the number of clusters.
- It finds 'exemplars' among the data points and forms clusters around them.
- Great for datasets where the number of clusters is unknown.
""")

# Generate synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Run Affinity Propagation
ap = AffinityPropagation(random_state=42)
ap.fit(X)

labels = ap.labels_
cluster_centers = ap.cluster_centers_

# Number of clusters found
n_clusters = len(cluster_centers)
st.subheader(f"✅ Number of clusters found: {n_clusters}")

# Plot clusters
fig, ax = plt.subplots()
for cluster_id in range(n_clusters):
    cluster_points = X[labels == cluster_id]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c="red", marker="X", label="Centers")
ax.legend()
st.pyplot(fig)
