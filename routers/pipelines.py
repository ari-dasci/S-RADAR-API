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

#Models
from SADL.static_data.algorithms import pyod
from SADL.static_data.algorithms import sklearn
from SADL.time_series.algorithms import tsfedl
from SADL.federated_data.algorithms import flexanomalies

#Preprocessing
from SADL.static_data.preprocessing.preprocessing_static import preprocessing_static_algorithms
from SADL.time_series.preprocessing.preprocessing_ts import preprocessing_ts_algorithms

# Visualization
from SADL.visualization_module import DataVisualization

import numpy as np
import torch


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
    context: Dict[str, Any] = {}
    visualizations = {}  # Dictionary to store node_id: plotly_json
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
                X, y = global_load_static(params["dataset"])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
                #context["X_train"] = X_train
                #context["y_train"] = y_train
                #context["X_test"] = X_test
                #context["y_test"] = y_test
                context[node_id] = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
                print(f"Dataset loaded: {params['dataset']}")

            elif(node_category == "time_series"):
                # Placeholder for time series dataset loading logic
                X, y = global_load_ts(params["dataset"])
                #TODO: what to do with this data division
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
                #context["X_train"] = X_train
                #context["y_train"] = y_train
                #context["X_test"] = X_test
                #context["y_test"] = y_test
                context[node_id] = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
                print(f"Dataset loaded: {params['dataset']}")

        elif node_type == "Preprocessing":
            print(f"Preprocessing...{model_type}")

            predecessors = get_predecessors(node_id, edges)
            if not predecessors:
                return {"error": f"No predecessor found for visualization node {node_id}"}
            
            prev_node_id = predecessors[0]
            prev_output = context.get(prev_node_id)

            if prev_output is None:
                return {"error": f"No data to visualize from previous node {prev_node_id}"}

            # Infer what kind of data it is
            if ("X_train" in prev_output):
                X_train = prev_output["X_train"]
            if ("X_test" in prev_output):
                X_test = prev_output["X_test"]
            else:
                return {"error": f"No valid data found in previous node {prev_node_id}"}

            if X_train is None:
                raise ValueError("No dataset loaded before preprocessing")

            ScalerClass = None
            if (model_type in preprocessing_static_algorithms):
                ScalerClass = preprocessing_static_algorithms.get(model_type)
            elif (model_type in preprocessing_ts_algorithms):
                ScalerClass = preprocessing_ts_algorithms.get(model_type)
                
            if ScalerClass is None:
                return {"error": f"Unknown preprocessing model_type: {model_type}"}
            
            scaler_instance = ScalerClass()
            X_scaled = scaler_instance.fit_transform(X_train)
            X_test_scaled = scaler_instance.transform(X_test)
            #context["X_train"] = X_scaled
            #context["X_test"] = X_test_scaled

            context[node_id] = {"X_scaled": X_scaled, "X_test_scaled": X_test_scaled}
            print(f"Preprocessing done: {scaler_instance}")

        elif node_type == "Model Setup":
            print(f"Setting up model: {params['algorithm_']}")
            if(params['algorithm_'] in pyod_algorithms):
                kwargs = params
                model = pyod.PyodAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                context["model"] = model

            elif(params['algorithm_'] in sklearn_algorithms):
                kwargs = params
                model = sklearn.SkLearnAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                context["model"] = model

            elif(params['algorithm_'] in tsfedl_algorithms):
                kwargs = params

                #Set top_module class manually
                in_features_ = int(kwargs["in_features_topmodule"])
                out_features_ = int(kwargs["out_features_topmodule"])
                kwargs["top_module"] = topModuleTSFEDL(in_features=in_features_, out_features=out_features_)
                kwargs["in_features"] = int(kwargs["in_features_topmodule"])
                kwargs.pop("in_features_topmodule", None)
                kwargs.pop("out_features_topmodule", None)
                print(f"kwargs before initialization: {kwargs}")

                model = tsfedl.TsfedlAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                context["model"] = model

            elif(params['algorithm_'] in flexanomalies_algorithms):
                kwargs = params
                model = flexanomalies.FlexAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                context["model"] = model

            model.fit(X)
            print(f"Model fitted: {model}")
            

        elif node_type == "Decision Function Model":
            model = context.get("model")
            X_train = context.get("X_train")
            scores_pred = model.decision_function(X_train)* -1
            print("Scores",scores_pred)

        elif node_type == "Predict Model":
            model = context.get("model")
            X_test = context.get("X_test")

            if model is None:
                raise ValueError("Model not set up before prediction")

            pred = model.predict(X_test)
            print(f"Prediction: {pred}")

        elif node_type == "Visualization":

            predecessors = get_predecessors(node_id, edges)
            if not predecessors:
                return {"error": f"No predecessor found for visualization node {node_id}"}

            
            prev_node_id = predecessors[0]
            prev_output = context.get(prev_node_id)

            if prev_output is None:
                return {"error": f"No data to visualize from previous node {prev_node_id}"}
            print(prev_output)
            # Infer what kind of data it is
            if ("X_train" in prev_output):
                data_to_plot = prev_output["X_train"]
            elif ("X_scaled" in prev_output):
                data_to_plot = prev_output["X_scaled"]
            else:
                return {"error": f"No valid data found in previous node {prev_node_id}"}
            
            # Create visualization
            clean_params = {k: v for k, v in params.items() if v not in [None, "", [], "plot"]}
            clean_params.pop("plot", None)
            
            # Optional: Cast known numeric parameters
            if "n_components" in clean_params:
                try:
                    clean_params["n_components"] = int(clean_params["n_components"])
                except ValueError:
                    pass  # Keep as string if it's e.g., 'mle'
            print(f"Creating visualization for node {node_id} with params: {clean_params}")
            vis = DataVisualization(data_to_plot, **clean_params) 
            vis.fit()  
            json_fig = vis.to_json()    
            visualizations[node_id] = json_fig
            print(f"Visualization for node {node_id} created from {prev_node_id}")
            
        else:
            return {"error": f"Unknown node type: {node_type}"}

    return {"message": "Pipeline executed","visualizations": visualizations}