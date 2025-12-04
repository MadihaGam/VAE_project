import numpy as np
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import matplotlib
matplotlib.use("Agg") 

def plot_2d(xy, labels, dis_type, vis_type):
    xy = np.asarray(xy)
    labels = np.asarray(labels)
    
    plt.figure(figsize=(7,7))
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels, palette="tab10", s=12, linewidth=0)
    plt.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{dis_type}/{vis_type}", dpi=300)
    plt.close()
    
    print(f"{dis_type}_{vis_type} plot saved")

def latent_visualization(zs, labels, dis_type):
    zs_cpu  = zs.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()

    # TSNE
    xy = TSNE(n_components=2, random_state=44).fit_transform(zs_cpu)
    plot_2d(xy, labels_cpu, dis_type, "TSNE")

    # UMAP
    xy = umap.UMAP(n_components=2, random_state=44).fit_transform(zs_cpu)
    plot_2d(xy, labels_cpu, dis_type, "UMAP")

    # MDS
    xy = MDS(n_components=2, n_init=1, random_state=44).fit_transform(zs_cpu)
    plot_2d(xy, labels_cpu, dis_type, "MDS")

