import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import EdgeConvModel
from sklearn.manifold import TSNE
from utils import plot_3d_shape, sim_matrix
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ShapeNet


dataset = ShapeNet(root=".", categories=["Airplane", "Chair", "Lamp", "Table"]).shuffle()[:4000]
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('clr.pt')

# Get sample batch
sample = next(iter(data_loader))
input_data = sample.to(device)

# Get representations
h = model(input_data, train=False)
h = h.cpu().detach()
labels = sample.category.cpu().detach().numpy()

# Get low-dimensional t-SNE Embeddings
# h_embedded = TSNE(n_components=2, learning_rate='auto',
#                    init='random').fit_transform(h.numpy())

# # Plot
# ax = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1], hue=labels, 
#                     alpha=0.5, palette="tab10")

# # Add labels to be able to identify the data points
# annotations = list(range(len(h_embedded[:,0])))

# def label_points(x, y, val, ax):
#     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
#     for i, point in a.iterrows():
#         ax.text(point['x']+.02, point['y'], str(int(point['val'])))

# label_points(pd.Series(h_embedded[:,0]), 
#             pd.Series(h_embedded[:,1]), 
#             pd.Series(annotations), 
#             plt.gca()) 

similarity = sim_matrix(h, h)
max_indices = torch.topk(similarity, k=2)[1][:, 1]
max_vals  = torch.topk(similarity, k=2)[0][:, 1]

# Select index
idx = 2
similar_idx = max_indices[idx]
print(f"Most similar data point in the embedding space for {idx} is {similar_idx}")

plot_3d_shape(sample[idx].cpu())
plot_3d_shape(sample[similar_idx].cpu())