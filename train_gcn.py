import torch
# from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from model import GCN_pyT
import os

# Print information
def print_dataset_info(dataset, data):
    print(dataset)
    print('------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Graph: {data}')
    print(f'x = {data.x.shape}')
    print(data.x)
    print(data.x.dtype)
    print(f'edge_index = {data.edge_index.shape}')
    print(data.edge_index)
    print(data.edge_index.dtype)

    A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
    print(f'A = {A.shape}')
    print(A)
    print(f'y = {data.y.shape}')
    print(data.y)
    print(f'train_mask = {data.train_mask.shape}')
    print(data.train_mask)
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')


# Define accuracy calculation
def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()


def train(model, data, optimizer, criterion, epochs=100, model_type="default"):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        
        # Calculate training loss and accuracy using train_mask
        loss = criterion(z[data.train_mask], data.y[data.train_mask])
        acc = accuracy(z[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        val_loss = criterion(z[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(z[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.2f} | Train Acc: {acc*100:.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')
    return model

# Testing function
def test(model, data, model_type="default"):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        acc = accuracy(z.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    print(f'Test Accuracy: {acc * 100:.2f}%')
    return acc

# Save model function
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Main execution
if __name__ == "__main__":

    dataset_name = 'citeseer' # Can be 'cora', 'citeseer', or 'pubmed'

    # Load dataset
    if dataset_name == 'cora':
        dataset = Planetoid(root='.', name="Cora")
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root='.', name="CiteSeer")
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='.', name="PubMed")

    data = dataset[0]
    print("Before update:")
    print_dataset_info(dataset, data)
    
    # Define models
    models = {
        "GCN_pyT": GCN_pyT(dataset.num_features, dataset.num_classes)
    }

    # Define optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_params = {'lr': 0.01, 'weight_decay': 5e-4}  # Define common optimizer parameters

    # Train, save, and test each model
    for model_name, model in models.items():
        
        print(f"\nTraining {model_name}...")
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        trained_model = train(model, data, optimizer, criterion, epochs=101, model_type=model_name)
        os.makedirs("torch_models/", exist_ok=True)
        save_model(trained_model, path=f"torch_models/{model_name}_{dataset_name}.pth")
        test(trained_model, data, model_type=model_name)

        # Print final embeddings and outputs
        z = trained_model(data.x, data.edge_index)
        print(f'Final outputs = {z.shape}')
        print(z.argmax(dim=1))
        print(f'Golden classes = {data.y.shape}')
        print(data.y)
