from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import torch

def feature_plot(features, ground_truth_labels):
    # azimuth converted to hue, elevation converted to brightness
    h = (ground_truth_labels[:, 0] % 360) / 360
    s = torch.ones_like(h)
    v = (ground_truth_labels[:, 0] + 90) / 180
    
    colors = hsv_to_rgb(torch.stack([h, s, v], axis=1))
    
    plot = plt.scatter(features[:, 0], features[:, 1], color=colors)
    plt.show()
    
    return plot

def tsne_plot(extracted_features, ground_truth_labels, n_components=2, perplexity=30.0, **kwargs):
    tsne_map = TSNE(n_components=n_components, perplexity=perplexity, **kwargs).fit_transform(extracted_features)
    plot = feature_plot(tsne_map, ground_truth_labels)
    
    return tsne_map, plot

def umap_plot(extracted_features, ground_truth_labels, n_components=2, n_neighbors=15, **kwargs):
    umap_map = UMAP(n_neighbors=n_neighbors, n_components=n_components, **kwargs).fit_transform(extracted_features)
    plot = feature_plot(umap_map, ground_truth_labels)
    
    return umap_map, plot