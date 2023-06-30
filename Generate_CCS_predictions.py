'''
Generates predictions for a CSV that includes embeddings with a 
trained CCS probe. Inputs from a companion config file that expects:
    -dataset to make predictions about: "prediction_dataset"
    -directory of dataset: "processed_dataset_dir"
    -where to save predictions: "csv_save_path"
    -trained probe name: "probe"
    -accompanying .npz file with parameters for probe: "params"
    -directory for probes: "probes_dir"
    
'''

import torch
import pandas as pd
from Train_CCSProbe import load_csv, MLPProbe
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda"


def get_embeddings(data):
    '''Get the embeddings in the csv'''
    embeddings = []
    for idx in tqdm(range(0, len(data)), desc="Getting embeddings"):
        hs = data.iloc[idx]['embeddings']  
        embeddings.append(hs)
    return embeddings

def prepare_data_for_mlp(embeddings):
    # Convert lists to PyTorch tensors
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings

def record_predictions(probe, data, filename, np_filename):
    """Record model predictions for each example in the dataframe when treated as positive or negative example.
      If the probe learns reversed labels, then we record the opposite"""
    pos_predictions = []
    neg_predictions = []
    device = next(probe.parameters()).device  # Get the device of the probe
    with np.load(np_filename) as model_data:
        pos_mean = model_data['pos_mean']
        neg_mean = model_data['neg_mean']
        pos_std = model_data['pos_std']
        neg_std = model_data['neg_std']
        reverse_predictions = model_data['reverse_predictions']
    for _, row in data.iterrows():
        try:
            embedding = torch.tensor(row['embeddings'], dtype=torch.float32, device=device)  # Move data to the device
            pos_embedding = (embedding - pos_mean) / pos_std
            neg_embedding = (embedding - neg_mean) / neg_std
            with torch.no_grad():
                pos_prediction = probe(pos_embedding.unsqueeze(0)).item()
                neg_prediction = probe(neg_embedding.unsqueeze(0)).item()
            if reverse_predictions:
                pos_prediction = 1 - pos_prediction
                neg_prediction = 1 - neg_prediction
            pos_predictions.append(pos_prediction)
            neg_predictions.append(neg_prediction)
        except Exception as e:
            print(f"Error occurred while predicting label for example: {e}")
            pos_predictions.append(None)  # Append None or some default value in case of error
            neg_predictions.append(None)
    # Add the predictions as a new column to the dataframe
    data['pos_predictions'] = pos_predictions
    data['neg_predictions'] = neg_predictions
    # Save the dataframe to a new csv file
    data.to_csv(filename, index=False)

def main():

    with open('CCS_config.json') as f:
        config = json.load(f)

    dataset = config["prediction_dataset"]
    dataset_dir = Path(config["processed_dataset_dir"])
    probes_dir = Path(config["probes_dir"])
    probe = config["probe"]
    output_path = Path(config["csv_save_path"])
    params_file = config["params"]


    data = load_csv(dataset_dir / dataset)
    embeddings = get_embeddings(data)
    embeddings = prepare_data_for_mlp(embeddings)
    
    # Load the model
    d = embeddings.shape[-1]
    model = MLPProbe(d)
    model.load_state_dict(torch.load(probes_dir/probe))
    model.eval()

    with torch.no_grad():
        name = f"{dataset[:-4]}_CCSpreds.csv"
        record_predictions(model, data, output_path / name, probes_dir / params_file)
    
if __name__ == "__main__":
    main()
 
                                                
