import tqdm
import torch
import torch.optim as optim
import torch_geometric.transforms as T
from model import EdgeConvModel
from utils import plot_3d_shape
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ShapeNet
from pytorch_metric_learning.losses import NTXentLoss

dataset = ShapeNet(root=".", categories=["Airplane", "Chair", "Lamp", "Table"]).shuffle()

print("Number of Samples: ", len(dataset))
# print("Sample: ", dataset[0])

# # Visualize samples
# sample_idx = 3
# plot_3d_shape(dataset[sample_idx])

cat_dict = {key: 0 for key in dataset.categories}
for datapoint in dataset: cat_dict[dataset.categories[datapoint.category.int()]]+=1
print(cat_dict)

data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

augmentation = T.Compose([T.RandomJitter(0.02), T.RandomFlip(1), T.RandomShear(0.2)])

loss_func = NTXentLoss(temperature=0.10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeConvModel(augmentation=augmentation).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(1, 4):
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(data_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        
        h_1, h_2, compact_h_1, compact_h_2 = model(data)
        
        embeddings = torch.cat((compact_h_1, compact_h_2))
        
        indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    
    cumulative_loss =  total_loss / len(dataset)
    print(f'Epoch {epoch:03d}, Loss: {cumulative_loss:.4f}')
    scheduler.step()


torch.save(model, "clr.pt")