import torch
# from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
import openvino as ov
from pathlib import Path
from model import GCN_pyT

# Load model function
def load_model(model_name, path, num_features, num_classes, data):
    # Import the model dynamically based on the model name
    if model_name == "GCN_pyT":
        from model import GCN_pyT as GCNModel
        model = GCNModel(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.load_state_dict(torch.load(path))  # Load the saved state dictionary
    return model

# Main execution
if __name__ == "__main__":
    # Load dataset
    dataset_name = 'cora' # Can be 'cora', 'citeseer', or 'pubmed'

    # Load dataset
    if dataset_name == 'cora':
        dataset = Planetoid(root='.', name="Cora")
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root='.', name="CiteSeer")
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='.', name="PubMed")

    data = dataset[0]
    
    # Specify model details
    model_name = "GCN_pyT"
    model_path = f'torch_models/{model_name}_{dataset_name}.pth'
    
    # Load the trained model
    model = load_model(model_name, model_path, dataset.num_features, dataset.num_classes, data)
    print(model)
    
    # Convert model to OpenVINO format
    ov_model = ov.convert_model(model, input=[("x", ov.Shape([data.num_nodes, dataset.num_features]), ov.Type.f32), ("edge_index", ov.Shape([2, data.edge_index.size(1)]), ov.Type.i32)])

    # Save the model as IR (xml and bin files)
    output_dir = Path("ov_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ov.save_model(ov_model, str(output_dir / f"{model_name}_{dataset_name}_fp32.xml"), compress_to_fp16=False)
    ov.save_model(ov_model, str(output_dir / f"{model_name}_{dataset_name}_fp16.xml"), compress_to_fp16=True)

    print(f"Model {model_name} converted and saved successfully.")
