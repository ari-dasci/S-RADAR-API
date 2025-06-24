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

import numpy as np

router = APIRouter()

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

    for node_id in sorted_node_ids:
        node = nodes[node_id]
        node_category = node.get("category", "")
        node_type = node["op_type"]
        model_type = node.get("model_type", "")
        params = node.get("params", {})
        print(f"Processing node: {node}")

        if node_type == "Load Dataset":
            print(f"Loading dataset: {params}")

            if(node_category == "static_data"):
                X, y = global_load_static(params["dataset"])
            elif(node_category == "time_series"):
                # Placeholder for time series dataset loading logic
                X, y = global_load_ts(params["dataset"])
                #TODO: what to do with this data division
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
                print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            #context["X"] = X
            #context["y"] = y
        elif node_type == "ModelSetup":
            if(model_type in pyod_algorithms):
                kwargs = params
                model = pyod.PyodAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
            elif(model_type in sklearn_algorithms):
                kwargs = params
                model = sklearn.SkLearnAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
            elif(model_type in tsfedl_algorithms):
                kwargs = params
                model = tsfedl.TsfedlAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
            elif(model_type in flexanomalies_algorithms):
                kwargs = params
                model = flexanomalies.FlexAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")

        elif node_type == "FitModel":
            model = context.get("model")
            X = context.get("X")

            # Clean X
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            context["X"] = X  # update X after cleaning
            context["model"].fit(X)
            scores = context["model"].decision_function(X) * -1
            context["scores"] = scores.tolist()

        elif node_type == "Visualize":
            # Dummy output — in real use, return a plot or metrics
            return {
                "message": "Pipeline executed successfully",
                "scores": context.get("scores", [])
            }

        else:
            return {"error": f"Unknown node type: {node_type}"}

    return {"message": "Pipeline executed, but no visualize node found"}