"""
This script loads embeddings and labels from CSV files, splits the data into 
training and test sets, trains an MLP probe on the training data, predicts 
labels for the test data using the trained probe, and records the probe's predictions in a new CSV file.
Input CSV files and paths are all set in a companion config file. 

It is based off of Burns et al.'s code, found here: https://github.com/collin-burns/discovering_latent_knowledge.

CAUTION: The "model", "layer", and "rmv_period" parameters in the 
companion config file need to be set correctly to ensure the trained 
probe and accompanying data are saved with a sensible name. (This could
be automated better if you're good at strings.)
"""

from tqdm import tqdm
import pandas as pd
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import json
import os


logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Add logging statements in your functions, e.g., when saving the CSV:
logger.info(f"Saving dataframe with predictions to CCS_eval.log")


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda"


def load_csv(filename):
    """
    Load a CSV file and convert the 'embeddings' column from string to list.

    Args:
    filename: str, name of the CSV file to load

    Returns:
    df: pandas DataFrame, the loaded data
    """
    df = pd.read_csv(filename)
    df['embeddings'] = df['embeddings'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    return df


def get_embeddings(data):
    '''Get the embeddings and labels for all contrast pairs'''
    all_neg_hs, all_pos_hs, all_labels = [], [], []

    # Ensure data length is even
    assert len(data) % 2 == 0

    for idx in tqdm(range(0, len(data), 2), desc="Getting embeddings"):
        pos_hs = data.iloc[idx]['embeddings']  # Embedding for positive statement
        neg_hs = data.iloc[idx+1]['embeddings']  # Embedding for negation of positive statement

        pos_label = data.iloc[idx]['label']  # Label for positive statement
        neg_label = data.iloc[idx+1]['label']  # Label for negation of positive statement
        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_labels.append(pos_label)

    return all_neg_hs, all_pos_hs, all_labels

def prepare_data_for_mlp(all_pos_hs, all_neg_hs):
    # Convert lists to PyTorch tensors
    pos_embeddings = torch.tensor(all_pos_hs, dtype=torch.float32)
    neg_embeddings = torch.tensor(all_neg_hs, dtype=torch.float32)
    return pos_embeddings, neg_embeddings

def record_predictions(probe, data, filename, pos_mean, pos_std, neg_mean, neg_std, reverse_predictions=False):
    """Record model predictions for each example in the dataframe. If the probe learns reversed labels, then we record the opposite"""
    predictions = []
    device = next(probe.parameters()).device  # Get the device of the probe

    # Convert mean and std to tensor 
    pos_mean, pos_std = torch.tensor(pos_mean, device=device), torch.tensor(pos_std, device=device)
    neg_mean, neg_std = torch.tensor(neg_mean, device=device), torch.tensor(neg_std, device=device)

    for i, (index, row) in enumerate(data.iterrows()):
        try:
            embedding = torch.tensor(row['embeddings'], dtype=torch.float32, device=device)  # Move data to the device

            # Normalize the embedding separately for positive and negative examples
            if i % 2 == 0:  # If the example is positive
                embedding = (embedding - pos_mean) / pos_std
            else:  # If the example is negative
                embedding = (embedding - neg_mean) / neg_std

            with torch.no_grad():
                prediction = probe(embedding.unsqueeze(0)).item()

            if reverse_predictions:
                prediction = 1 - prediction

            predictions.append(prediction)

        except Exception as e:
            print(f"Error occurred while predicting label for example: {e}")
            predictions.append(None)  # Append None or some default value in case of error

    # Add the predictions as a new column to the dataframe
    data['predictions'] = predictions

    # Save the dataframe to a new csv file
    data.to_csv(filename, index=False)





def save_model(probe, path):
    torch.save(probe.state_dict(), path)


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)
    
class CCS(object):
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1, 
                 verbose=False, device=device, linear=False, weight_decay=0.01, var_normalize=True):
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # probe
        self.linear = linear
        self.probe = self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

        
    def initialize_probe(self):
        """
        Initialize the model probe. If self.linear is True, the probe is a linear layer followed by a sigmoid activation function; otherwise, it is a MLPProbe.

        Returns:
        None
        """
        try:
            if self.linear:
                self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
            else:
                self.probe = MLPProbe(self.d)
            self.probe.to(self.device)
            logger.info(f'Probe initialized and moved to device: {self.device}')
        except Exception as e:
            logger.error(f'Error in initialize_probe: {e}')
            raise 


    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

        
    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1
    

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss

    def get_raw_acc(self, x0_test, x1_test, y_test):
        '''
        Computes accuracy for current parameters without reversing.
        '''
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5*(p0 + (1-p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        return acc

    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        acc = self.get_raw_acc(x0_test, x1_test, y_test)
        acc = max(acc, 1-acc)
        return acc
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]
        
        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j*batch_size:(j+1)*batch_size]
                x1_batch = x1[j*batch_size:(j+1)*batch_size]
            
                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss

def main():

    try:
        logging.basicConfig(filename='CCS_eval.log', level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return
    
    logger.info("Excecution started.")
    # Load data from csv files
    with open('CCS_config.json') as f:
        config = json.load(f)
    
    dataset_names = config["datasets"]
    dataset_dir = Path(config["processed_dataset_dir"])
    probes_dir = Path(config["probes_dir"])
    output_path = Path(config["csv_save_path"])
    llm_name = config["model"]
    layer = config["layer"]
    rmv_period = config["rmv_period"]

    # Load data from csv files
    dataframes = []
    for dataset_file in dataset_names:
        dataset_path = dataset_dir / dataset_file  # Combine directory and file name
        df = load_csv(dataset_path)
        dataframes.append(df)

    # Combine data and get embeddings
    combined_data = pd.concat(dataframes)
    all_neg_hs, all_pos_hs, all_labels = get_embeddings(combined_data)

    # Shuffle the data
    data_size = len(all_labels)
    # shuffled_indices = np.random.permutation(data_size)
    # all_neg_hs = [all_neg_hs[i] for i in shuffled_indices]
    # all_pos_hs = [all_pos_hs[i] for i in shuffled_indices]
    # all_labels = [all_labels[i] for i in shuffled_indices]


    # Split data into train and test
    train_ratio = 0.8
    train_size = int(train_ratio * len(all_labels))  # Size of the training set

    neg_hs_train, neg_hs_test = all_neg_hs[:train_size], all_neg_hs[train_size:]
    pos_hs_train, pos_hs_test = all_pos_hs[:train_size], all_pos_hs[train_size:]
    y_test = all_labels[train_size:]

    # Prepare data for MLP
    pos_embeddings_train, neg_embeddings_train = prepare_data_for_mlp(pos_hs_train, neg_hs_train)
    pos_embeddings_test, neg_embeddings_test = prepare_data_for_mlp(pos_hs_test, neg_hs_test)

    #Capture means and stds of training data to save later
    pos_mean = pos_embeddings_train.numpy().mean(axis=0)
    neg_mean = neg_embeddings_train.numpy().mean(axis=0)
    pos_std = pos_embeddings_train.numpy().std(axis=0)
    neg_std = neg_embeddings_train.numpy().std(axis=0)


    # Initialize the CCS with the embeddings from positive and negative statements
    ccs = CCS(pos_embeddings_train.numpy(), neg_embeddings_train.numpy(), nepochs=1000, ntries=10, lr=1e-3)

    # Train the CCS
    best_loss = ccs.repeated_train()
    print(f"Best loss achieved after training: {best_loss}")

    pos_embeddings_test_np = pos_embeddings_test.numpy()
    neg_embeddings_test_np = neg_embeddings_test.numpy()
    raw_acc = ccs.get_raw_acc(pos_embeddings_test_np, neg_embeddings_test_np, y_test)
    if raw_acc < .5:
        reverse_predictions = True
    else:
        reverse_predictions = False
    
    for df, dataset_file in zip(dataframes, dataset_names):
        output_filename = dataset_file.rsplit('.', 1)[0] + "CSSpreds.csv"  # Modify the file name
        output_path_name = output_path / output_filename  # Combine directory and file name
        record_predictions(ccs.best_probe, df, output_path_name, pos_mean, pos_std, neg_mean, neg_std, reverse_predictions)

    # Test accuracy
    ccs_acc = ccs.get_acc(pos_embeddings_test_np, neg_embeddings_test_np, y_test)
    print("CCS accuracy: {}".format(ccs_acc))

    # Save mean and std to a file
    # Save the normalization parameters
    layer = str(abs(layer))
    suffix = ""
    if rmv_period:
        suffix = "rp"
    if not os.path.exists(probes_dir):
        os.makedirs(probes_dir)
    normalization_params_filename = f"norm_params_{llm_name}_{layer}_{suffix}.npz"
    normalization_params_path = probes_dir / normalization_params_filename  # Combine directory and file name
    np.savez(normalization_params_path, pos_mean=pos_mean, neg_mean=neg_mean, pos_std=pos_std,
             neg_std=neg_std, reverse_predictions=reverse_predictions)

    # Save the best probe 
    save_model(ccs.best_probe, probes_dir /f"CCSprobe_{llm_name}_{layer}_{suffix}.pth")



if __name__ == "__main__":
    main()
