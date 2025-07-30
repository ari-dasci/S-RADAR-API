from inspect import signature
from fastapi import APIRouter, HTTPException,Request
from typing import Dict, Any
import networkx as nx  # Use for topological sorting
from SADL.static_data.static_datasets_uci import global_load as global_load_static
from SADL.time_series.time_series_datasets_uci import global_load as global_load_ts
from sklearn.model_selection import train_test_split

#Algoritm lists
from SADL.static_data.algorithms.pyod import pyod_algorithms
from SADL.static_data.algorithms.sklearn import sklearn_algorithms
from SADL.time_series.algorithms.tsfedl import tsfedl_algorithms
from SADL.federated_data.algorithms.flexanomalies import flexanomalies_algorithms
from SADL.time_series.algorithms.transformers import transformers_algorithms

#Models
from SADL.static_data.algorithms import pyod
from SADL.static_data.algorithms import sklearn
from SADL.time_series.algorithms import tsfedl
from SADL.federated_data.algorithms import flexanomalies
from SADL.time_series.algorithms import transformers

#Preprocessing
from SADL.static_data.preprocessing.preprocessing_static import preprocessing_static_algorithms
from SADL.time_series.preprocessing.preprocessing_ts import preprocessing_ts_algorithms

# Visualization
from SADL.visualization_module import DataVisualization


from SADL.time_series.preprocessing.preprocessing_ts import StandardScalerPreprocessing
from SADL.time_series.time_series_utils import TimeSeriesProcessor

import numpy as np
import torch


from torch.utils.data import Dataset

class PermuteDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].permute(1, 0)
        y = self.y[idx]
        return x, y


class topModuleTSFEDL(torch.nn.Module):
    def __init__(self, in_features=103, out_features=103, npred=1):
        super(topModuleTSFEDL, self).__init__()
        self.npred = npred
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=in_features, out_features=50),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=50, out_features=npred*out_features)
        )

    def forward(self, x):
        out = self.model(x)
        if len(out.shape)>2:
            out = out[:, -1, :]
        if self.npred > 1:
            # Reshape to (batch_size, npred, out_features)
            out = out.reshape(out.shape[0], self.npred, -1)
        return out 


router = APIRouter()

def get_predecessors(node_id: str, edges: list):
    return [edge["source"] for edge in edges if edge["target"] == node_id]

@router.post("/pipelines/run_pipeline", tags=["pipelines"])
async def run_pipeline(request: Request):
    data = await request.json()
    print(f"Received data: {data}")

    nodes = {node["id"]: node for node in data["nodes"]}
    edges = data["edges"]

    # Step 1: Build graph and sort topologically
    G = nx.DiGraph()
    for node_id in nodes:
        G.add_node(node_id)
    for edge in edges:
        G.add_edge(edge["source"], edge["target"])

    try:
        sorted_node_ids = list(nx.topological_sort(G))
    except Exception as e:
        return {"error": f"Cycle detected in graph: {e}"}

    # Step 2: Execute nodes in order
    results_dict: Dict[str, Any] = {
        "message": "Pipeline execution finished with an unknown error",
    } # Dictionary to store final results
    context: Dict[str, Any] = {}
    visualizations = {}  # Dictionary to store node_id: plotly_json
    results = None  # Variable to store numerical results from the pipeline

    for node_id in sorted_node_ids:
        node = nodes[node_id]
        node_category = node.get("category", "")
        node_type = node["op_type"]
        model_type = node.get("model", "")
        params = node.get("params", {})
        print(f"Processing node: {node}")

        if node_type == "Load Dataset":
            print(f"Loading dataset: {params}")

            if(node_category == "static_data"):
                data = global_load_static(params["dataset"])

                if isinstance(data, tuple):
                    if len(data) == 2:
                        X, y = data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
                    elif len(data) == 4:
                        X_train, X_test, y_train, y_test = data
                context[node_id] = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

            elif(node_category == "time_series"):
                # Placeholder for time series dataset loading logic
                if params["dataset"] == "ai4i_2020_predictive_maintenance_dataset": #TODO: Change for different datasets
                    X, y = global_load_ts(params["dataset"])
                    y = y["Machine failure"]
                    X = X.drop('Type',axis=1)  # remove Type or apply one hot encoding
                    
                context[node_id] = {"X_train": X, "y_train": y}
           

            
            print(f"Dataset loaded: {params['dataset']}")


        elif node_type == "Preprocessing":
            print(f"Preprocessing...{model_type}")

            predecessors = get_predecessors(node_id, edges)
            if not predecessors:
                return {"message": f" Error: (Preprocessing) No predecessor found for Preprocessing node."}
            
            prev_node_id = predecessors[0]
            prev_output = context.get(prev_node_id)

            if prev_output is None:
                return {"message": f"Error: (Preprocessing) No data to preprocess from previous node."}

            # Infer what kind of data it is
            if ("X_train" in prev_output):
                X_train = prev_output["X_train"]
            if ("X_test" in prev_output):
                X_test = prev_output["X_test"]
            if ("y_train" in prev_output):
                y_train = prev_output["y_train"]
            if ("y_test" in prev_output):
                y_test = prev_output["y_test"]

            if X_train is None:
                return {"message": f"Error: (Preprocessing) No dataset loaded before preprocessing."}

            ScalerClass = None
            if (model_type in preprocessing_static_algorithms):
                ScalerClass = preprocessing_static_algorithms.get(model_type)
            elif (model_type in preprocessing_ts_algorithms):
                ScalerClass = preprocessing_ts_algorithms.get(model_type)
                
            if ScalerClass is None:
                return {"message": f"Error: (Preprocessing) Unknown preprocessing model_type: {model_type}"}
            
            scaler_instance = ScalerClass()
            X_scaled = scaler_instance.fit_transform(X_train)
            X_test_scaled = scaler_instance.transform(X_test)
            
            context[node_id] = {**prev_output, "X_scaled": X_scaled, "X_test_scaled": X_test_scaled}
            print(f"Preprocessing done: {scaler_instance}")

        elif node_type == "Model Setup":
            print(f"Setting up model: {params['algorithm_']}")

            predecessors = get_predecessors(node_id, edges)
            prev_node_id = predecessors[0]
            prev_output = context.get(prev_node_id)

            context[node_id] = {}
            if prev_output is None:
                return {"message": f"Error: (Model Fitting) No predecessor found for model fitting node."}
            
            # Infer data for training
            if ("X_train" in prev_output):
                X = prev_output["X_train"]
            #if ("X_test" in prev_output):
            #    X = prev_output["X_test"]
            if ("X_scaled" in prev_output):
                X = prev_output["X_scaled"]
                print("Tengo X_scaled")
            if X is None:
                return {"message": f"Error: (Model Fitting) No valid data found in previous node."}

            if(params['algorithm_'] in pyod_algorithms):
                kwargs = params
                model = pyod.PyodAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                model.fit(X)

            elif(params['algorithm_'] in sklearn_algorithms):
                kwargs = params
                model = sklearn.SkLearnAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                model.fit(X)

            elif(params['algorithm_'] in tsfedl_algorithms):
                kwargs = params

                #Set TSFEDL specific parameters
                in_features_ = int(kwargs["in_features_topmodule"])
                out_features_ = int(kwargs["out_features_topmodule"])
                if "in_features_topmodule" in kwargs and "out_features_topmodule" in kwargs:
                    kwargs["top_module"] = topModuleTSFEDL(in_features=in_features_, out_features=out_features_)
                if "in_features" in kwargs:
                    kwargs["in_features"] = int(kwargs["in_features"])
                kwargs.pop("in_features_topmodule", None)
                kwargs.pop("out_features_topmodule", None)
                if "loss" in kwargs:
                    kwargs["loss"] = torch.nn.MSELoss()
                if "input_shape" in kwargs:
                    kwargs["input_shape"] = (126,126)
                
                print(f"kwargs before initialization: {kwargs}")

                model = tsfedl.TsfedlAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                model.fit(prev_output.get("train_loader", None))

            elif(params['algorithm_'] in flexanomalies_algorithms):
                kwargs = params
                print(  f"kwargs before initialization flexanomalies: {kwargs}")
                # Set FlexAnomalies specific parameters
                #kwargs["n_clients"] = 15
                kwargs["n_rounds"] = 5

                if "input_shape" in kwargs:
                    kwargs["input_dim"] = X.shape[1]

                model = flexanomalies.FlexAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                model.fit(X, prev_output.get("y_train", None))

            elif(params['algorithm_'] in transformers_algorithms):
                kwargs = params
                # Set Transformers specific parameters
                kwargs["label_parser"] = None
                kwargs['train_epochs'] = 1
                kwargs['batch_size']= 16
                kwargs['lr']= 0.001
                # Set Tranformers train loader
                scaler = StandardScalerPreprocessing()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, prev_output["y_train"], test_size=0.2, random_state=42)

                processor = TimeSeriesProcessor(window_size=kwargs["seq_len"], step_size=1,future_prediction=False)
                X_train_windows, y_train_windows, X_test_windows, y_test_windows = processor.process_train_test(X_train, y_train, X_test, y_test)

                model = transformers.TransformersAnomalyDetection(**kwargs)
                model.fit(X_train_windows)

                
                context[node_id]["X_train_windows"] = X_train_windows
                context[node_id]["X_test_windows"] = X_test_windows
                context[node_id]["y_train_windows"] = y_train_windows
                context[node_id]["y_test_windows"] = y_test_windows
                context[node_id]["X_test"] = X_test
                context[node_id]["y_test"] = y_test

            
            context[node_id]["model"] = model
            context[node_id]["X_train"] = prev_output.get("X_train", None)
            if context[node_id].get("X_test") is None:
                context[node_id]["X_test"] = prev_output.get("X_test", None)
            context[node_id]["X_test_scaled"] = prev_output.get("X_test_scaled", None)
            context[node_id]["y_train"] = prev_output.get("y_train", None)
            context[node_id]["y_scaled"] = prev_output.get("y_scaled", None)
            if context[node_id].get("y_test") is None:
                context[node_id]["y_test"] = prev_output.get("y_test", None)
            context[node_id]["y_test_scaled"] = prev_output.get("y_test_scaled", None)
            context[node_id]["train_loader"] = prev_output.get("train_loader", None)

            print(context[node_id])
            print(f"Model fitted: {model}")
            

        elif node_type == "Decision Function Model":
            predecessors = get_predecessors(node_id, edges)
            if not predecessors:
                return {"message": f"Error: (Decision Function) No predecessor found for Decision Function node."}
            
            prev_node_id = predecessors[0]
            prev_output = context.get(prev_node_id)

            model = prev_output.get("model")
            if "X_train" in prev_output:
                X_train = prev_output.get("X_train")
            if "X_test_windows" in prev_output:
                X_train = prev_output.get("X_test_windows")

            scores_pred = model.decision_function(X_train)* -1
            print("Scores",scores_pred)
            # Add scores_pred to previous context, preserving other keys
            context[node_id] = {**prev_output, "data": scores_pred}

            results = scores_pred.tolist()
            results_dict["results_dec"] = results
            results_str = str(results)
            results_dict["message"] = "Pipeline executed successfully! Results are: " + results_str[:100] + "...\n If you want to see the complete output, please export the results."
            

        elif node_type == "Predict Model":
            predecessors = get_predecessors(node_id, edges)
            prev_node_id = predecessors[0]
            prev_output = context.get(prev_node_id)

            if prev_output is None:
                return {"message": f"Error: (Predict) No predecessor found for predict node {node_id}"}
            
            # Infer data for prediction
            if ("X_test" in prev_output and prev_output["X_test"] is not None):
                X = prev_output["X_test"]
            if ("X_test_scaled" in prev_output and prev_output["X_test_scaled"] is not None):
                X = prev_output["X_test_scaled"]
            if ("X_test_windows" in prev_output and prev_output["X_test_windows"] is not None):
                X = prev_output["X_test_windows"]
            if X is None:
                return {"message": f"Error: (Predict) No valid data found in previous node."}
           
            if ("model" in prev_output and prev_output["model"] is not None):
                model = prev_output["model"]
            if model is None:
                return {"message": f"Error: (Predict) Model not set up before prediction."}

            pred = model.predict(X)
            # Add scores_pred to previous context, preserving other keys
            context[node_id] = {**prev_output, "data": pred}

            #print(f"Prediction context : {context[node_id]}")
            results = pred.labels_.tolist() if hasattr(pred, 'labels_') else pred.tolist()
            results_dict["results_pred"] = results
            results_str = str(results)
            results_dict["message"] = "Pipeline executed successfully! Results are: " + results_str[:100] + "...\n If you want to see the complete output, please export the results."

        elif node_type == "Visualization":

            predecessors = get_predecessors(node_id, edges)
            if not predecessors:
                return {"message": f"Error: (Visualization) No predecessor found for visualization node."}            
            prev_node_id = predecessors[0]
            prev_output = context.get(prev_node_id)

            if prev_output is None:
                return {"message": f"Error: (Visualization) No data to visualize from previous node."}

            # Check previous node type
            if prev_node["op_type"] == "Load Dataset":
                data_to_plot = prev_output["X_train"]

                # Create visualization
                clean_params = {k: v for k, v in params.items() if v not in [None, "", [], "plot"]}
                clean_params.pop("plot", None)
                # Optional: Cast known numeric parameters
                if "n_components" in clean_params:
                    try:
                        clean_params["n_components"] = int(clean_params["n_components"])
                    except ValueError:
                        pass 

                
                vis = DataVisualization(data_to_plot, **clean_params) 
                vis.fit()  
                json_fig = vis.to_json()    
                visualizations[node_id] = json_fig

                # Store visualizations in results
                results_dict["visualizations"] = visualizations

            elif prev_node["op_type"] == "Preprocessing":
                data_to_plot = prev_output["X_scaled"]

                # Create visualization
                clean_params = {k: v for k, v in params.items() if v not in [None, "", [], "plot"]}
                clean_params.pop("plot", None)
                # Optional: Cast known numeric parameters
                if "n_components" in clean_params:
                    try:
                        clean_params["n_components"] = int(clean_params["n_components"])
                    except ValueError:
                        pass 

                
                vis = DataVisualization(data_to_plot, **clean_params) 
                vis.fit()  
                json_fig = vis.to_json()    
                visualizations[node_id] = json_fig

                # Store visualizations in results
                results_dict["visualizations"] = visualizations

            elif prev_node["op_type"] == "Predict Model":
                data_to_plot = prev_output["X_test"]
                clean_params = {k: v for k, v in params.items() if v not in [None, "", [], "plot"]}
                if prev_output.get("y_train_windows") is not None: #Transformer values
                    print("MME MEETOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                    print(prev_output["X_test"])
                    model = prev_output["model"]
                    model.evaluate(prev_output.get("X_test_windows"),prev_output.get("y_test_windows"))
                    pred = model.labels_preds
                    true = np.array(prev_output.get("y_test")).ravel()
                else:
                    true = np.array(prev_output["y_test"]).flatten()
                    pred = np.array(prev_output["data"]).astype(int)
                
                
                clean_params["y_true"] = true
                clean_params["y_pred"] = pred
                clean_params.pop("plot", None)
            
                #Cast known numeric parameters
                if "n_components" in clean_params:
                    clean_params["n_components"] = int(clean_params["n_components"])
                if "subset_size_percent" in clean_params:       
                    clean_params["subset_size_percent"] = float(clean_params["subset_size_percent"])
                
                print(f"Creating visualization for node {node_id} with params: {clean_params}")
                vis = DataVisualization(data_to_plot, **clean_params) 
                vis.fit()  
                json_fig = vis.to_json()    
                visualizations[node_id] = json_fig
                # Store visualizations in results
                results_dict["visualizations"] = visualizations
                
            else: 
                return {"message": f"Error: (Visualization) Previous node is not a valid source for visualization."}
            
            results_dict["message"] = "Visualization created successfully!"

        else:
            return {"message": f"Unknown node type: {node_type}"}
        
        #Once node is processed, store its data for the next iteration
        prev_node = node
    return results_dict