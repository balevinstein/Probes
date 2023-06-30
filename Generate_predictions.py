"""
This script generates predictions from a set of trained models on new input data, and 
appends the predictions to the original input data. The script also calculates and appends 
the average prediction from all models. The models and input data are specified in a 
configuration file, or, optionally with command-line arguments for some parameters. 

Expected configuration parameters include:
- dataset_path: The path to the input data.
- dataset_name: The name of the input data file.
- model: The name of the language model used to generate the embeddings.
- layer: The layer of the language model used to generate the embeddings.
- probes_path: The path to the trained models.
- suffix_list: The suffixes of the trained models to be used.

The script expects the input data to be in CSV format, with one column labeled "statements", 
another labeled "embeddings", and potentially other columns. The script generates a new CSV 
file that includes the original data as well as a new column for the average prediction and 
a new column for each model's prediction.

The script uses error and exception handling to account for potential issues with the 
configuration file, data loading, and model loading. 

Date: 5/20/23
"""


import pandas as pd
from keras.models import load_model
import numpy as np
from pathlib import Path
import json
import logging
from Generate_Embeddings import load_data
from TrainProbes import correct_str
import argparse


def make_model_list(probes_path, llm_name, layer, suffix_list):
    models = []
    for suffix in suffix_list:
        try:  
            model = probes_path / f"{llm_name}_{abs(int(layer))}_{suffix}.h5"
            models.append(model)
        except FileNotFoundError:
            print(f"Probe '{llm_name}_{abs(int(layer))}_{suffix}.h5' not found in {probes_path}")
    return models

def main():
    try:
        with open("config.json") as config_file:
            config_parameters = json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found. Please ensure the file exists and the path is correct.")
        return
    except PermissionError:
        logging.error("Permission denied. Please check your file permissions.")
        return
    except json.JSONDecodeError:
        logging.error("Configuration file is not valid JSON. Please check the file's contents.")
        return
    
    parser = argparse.ArgumentParser(description="Makes predictions using already trained probes.")
    parser.add_argument("--model", 
                       help="Name of the language model to use: '6.7b', '2.7b', '1.3b', '350m'")
    parser.add_argument("--dataset_name", help="Name of dataset to make predictions about without 'csv' extension.")
    parser.add_argument("--layer", help="Layer for probe")
    parser.add_argument("--probe_suffixes", nargs='*', 
                        help="Part of the probe name after the layer without the h5 extension. For example, '350_-4_facts_rp.h5' has the suffix 'facts_rp'.")
    args = parser.parse_args()
    
    dataset_name = args.dataset_name if args.dataset_name is not None else config_parameters["gen_predictions_dataset"]
    llm_name = args.model if args.model is not None else config_parameters["model"]
    layer = args.layer if args.layer is not None else config_parameters["gen_predictions_layer"]
    suffix_list = args.probe_suffixes if args.probe_suffixes is not None else config_parameters["suffix_list"]

    probes_path = Path(config_parameters["probes_dir"])
    dataset_path = Path(config_parameters["processed_dataset_path"])


    df = load_data(dataset_path, dataset_name, true_false=False)
    models = make_model_list(probes_path, llm_name, layer, suffix_list)

    # Prepare a container for the predictions
    all_preds = []

    start_index = df.columns.get_loc('embeddings') + 1

    for idx, model_name in enumerate(models):
        # Load the model
        try:
            model = load_model(model_name)
        except OSError as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            return
        except ValueError as e:
            logging.error(f"Invalid model {model_name}: {str(e)}")
            return
        
        # Get the embeddings
        embeddings = np.array([np.fromstring(correct_str(embedding), sep=',') for embedding in df['embeddings'].tolist()])

        # Make predictions and round to 4 decimal places
        preds = model.predict(embeddings).round(4)

        # Add to the list of all predictions
        all_preds.append(preds)

        # Add the predictions as a new column in the DataFrame, or overwrite the existing one
        column_name = str(suffix_list[idx])
        assert column_name in str(model_name)
        #column_name = str(model_name)
        if column_name in df:
            df[column_name] = preds
        else:
            df.insert(start_index, column_name, preds)
            start_index += 1

    # Calculate the average prediction
    avg_preds = np.mean(all_preds, axis=0).round(4)

    # Add the average prediction as a new column in the DataFrame, or overwrite the existing one
    if 'average_prediction' in df:
        df['average_prediction'] = avg_preds
    else:
        df.insert(df.columns.get_loc('embeddings') + 1, 'average_prediction', avg_preds)

    # Save the DataFrame to a new CSV file
    path = config_parameters["processed_dataset_path"]
    df.to_csv(Path(path) / f"{dataset_name}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
