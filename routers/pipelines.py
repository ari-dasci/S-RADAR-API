from inspect import signature
from fastapi import APIRouter, HTTPException,Request
from typing import Dict, Any
import networkx as nx  # Use for topological sorting
from RADAR.static_data.static_datasets_uci import global_load as global_load_static
from RADAR.static_data.anomaly_dataset_utils import (
    build_loaded_uci_anomaly_dataset,
    build_kddcup99_anomaly_dataset,
    build_har_anomaly_dataset,
)
from RADAR.time_series.time_series_datasets_uci import global_load as global_load_ts
from sklearn.model_selection import train_test_split

#Algoritm lists
from RADAR.static_data.algorithms.pyod import pyod_algorithms
from RADAR.static_data.algorithms.sklearn import sklearn_algorithms
from RADAR.time_series.algorithms.tsfedl import tsfedl_algorithms
from RADAR.federated_data.algorithms.flexanomalies import flexanomalies_algorithms
from RADAR.time_series.algorithms.transformers import transformers_algorithms

#Models
from RADAR.static_data.algorithms import pyod
from RADAR.static_data.algorithms import sklearn
from RADAR.time_series.algorithms import tsfedl
from RADAR.federated_data.algorithms import flexanomalies
from RADAR.time_series.algorithms import transformers

#Preprocessing
from RADAR.static_data.preprocessing.preprocessing_static import preprocessing_static_algorithms
from RADAR.time_series.preprocessing.preprocessing_ts import preprocessing_ts_algorithms

# Visualization
from RADAR.visualization_module import DataVisualization,DataVisualizationScoresTS


from RADAR.time_series.preprocessing.preprocessing_ts import StandardScalerPreprocessing
from RADAR.time_series.time_series_utils import TimeSeriesProcessor

import numpy as np
import torch
from torch.utils.data import Dataset
from TSFEDL.models_pytorch import (
    OhShuLih_Forecaster, YiboGao_Forecaster, LihOhShu_Forecaster, YaoQihang_Forecaster,
    HtetMyetLynn_Forecaster, YildirimOzal_Forecaster, CaiWenjuan_Forecaster, ZhangJin_Forecaster,
    KongZhengmin_Forecaster, WeiXiaoyan_Forecaster, GaoJunLi_Forecaster, KhanZulfiqar_Forecaster,
    ZhengZhenyu_Forecaster, WangKejun_Forecaster, ChenChen_Forecaster, KimTaeYoung_Forecaster,
    GenMinxing_Forecaster, FuJiangmeng_Forecaster, ShiHaotian_Forecaster, HuangMeiLing_Forecaster,
    HongTan_Forecaster, SharPar_Forecaster, DaiXiLi_Forecaster
)


def _resolve_stratify_labels(y, test_size):
    labels = np.asarray(y).reshape(-1)
    if labels.size == 0:
        return None, "Stratified split skipped because target labels are empty."

    _, counts = np.unique(labels, return_counts=True)
    min_class_count = counts.min()
    if min_class_count < 2:
        return None, (
            "Stratified split skipped because at least one class has fewer than 2 samples."
        )

    n_samples = labels.shape[0]
    if isinstance(test_size, float):
        n_test = int(np.ceil(n_samples * test_size))
    else:
        n_test = int(test_size)
    n_train = n_samples - n_test
    n_classes = len(counts)

    if n_test < n_classes or n_train < n_classes:
        return None, (
            "Stratified split skipped because the requested train/test sizes cannot include all classes."
        )

    return y, None

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
top_modules_map = {
    "ohshulih": OhShuLih_Forecaster,
    "yibogao": YiboGao_Forecaster,
    "liohshu": LihOhShu_Forecaster,
    "yaoqihang": YaoQihang_Forecaster,
    "htetmyetlynn": HtetMyetLynn_Forecaster,
    "yildirimozal": YildirimOzal_Forecaster,
    "caiwenjuan": CaiWenjuan_Forecaster,
    "zhangjin": ZhangJin_Forecaster,
    "kongzhengmin": KongZhengmin_Forecaster,
    "weixiaoyan": WeiXiaoyan_Forecaster,
    "gaojunli": GaoJunLi_Forecaster,
    "khanzulfiqar": KhanZulfiqar_Forecaster,
    "zhengzhenyu": ZhengZhenyu_Forecaster,
    "wangkejun": WangKejun_Forecaster,
    "chenchen": ChenChen_Forecaster,
    "kimtaeyoung": KimTaeYoung_Forecaster,
    "genminxing": GenMinxing_Forecaster,
    "fujiangmeng": FuJiangmeng_Forecaster,
    "shihaotian": ShiHaotian_Forecaster,
    "huangmeiling": HuangMeiLing_Forecaster,
    "hongtan": HongTan_Forecaster,
    "sharpar": SharPar_Forecaster,
    "daixili": DaiXiLi_Forecaster
}    

router = APIRouter()

def get_predecessors(node_id: str, edges: list):
    return [edge["source"] for edge in edges if edge["target"] == node_id]

@router.post("/pipelines/run_pipeline", tags=["pipelines"])
async def run_pipeline(request: Request):
    data = await request.json()
    print(f"Received data: {data}")

    timeseries_visualization = any([n['category']=='time_series' for n in data['nodes']])
    federated_visualizacion = any([n['category']=='federated_data' for n in data['nodes']])
    nodes = {node["id"]: node for node in data["nodes"]}
    edges = data["edges"]
    deepModel = False
    
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
                context[node_id] = process_dataset_static(params["dataset"])

            # elif(node_category == "time_series"):
            #     # Placeholder for time series dataset loading logic
            #     if params["dataset"] == "ai4i_2020_predictive_maintenance_dataset": #TODO: Change for different datasets
            #         X, y = global_load_ts(params["dataset"])
            #         y = y["Machine failure"]
            #         X = X.drop('Type',axis=1)  # remove Type or apply one hot encoding
                    
            #     context[node_id] = {"X_train": X, "y_train": y}
            
            elif node_category == "time_series":
                data = global_load_ts(params["dataset"])
                context[node_id] = process_dataset_ts(data, params["dataset"])

            
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
            # Ensure algorithm_ is present in params (frontend may put it in node.model instead)
            if "algorithm_" not in params and model_type:
                params["algorithm_"] = model_type.strip().lower()

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
                model.fit(X.astype(np.float64) )

            elif(params['algorithm_'] in sklearn_algorithms):
                kwargs = params
                model = sklearn.SkLearnAnomalyDetection(**kwargs)
                print(f"Model initialized: {model}")
                model.fit(X)

            elif(params['algorithm_'] in tsfedl_algorithms):
                kwargs = params
                # NEWWWWWWWW
                # Obtain top module class according to the algorithm
                top_class = top_modules_map.get(params['algorithm_'])

                # Extraer parámetros del top module
                in_features_top = int(kwargs.pop("in_features_topmodule", 0))
                out_features_top = int(kwargs.pop("out_features_topmodule", 0))
                n_pred = 1 #int(kwargs.pop("n_pred_topmodule", 1))  # configurable desde frontend

                # Assign top_module dynamically
                if top_class and in_features_top and out_features_top:
                    kwargs["top_module"] = top_class(out_features=out_features_top, n_pred=n_pred)

                # Ajustar parámetros generales
                kwargs["in_features"] = int(kwargs.get("in_features", 0)) if "in_features" in kwargs else None
                kwargs["loss"] = torch.nn.MSELoss()
                kwargs["max_epochs"] = kwargs.get("max_epochs", 1)  # configurable desde frontend
                

                print(f"kwargs before initialization: {kwargs}")
                                 
                # Create temporal windows
                processor = TimeSeriesProcessor(window_size=24, step_size=1, future_prediction=False, n_pred=n_pred)
                X_train_w, y_train_w, X_test_w, y_test_w = processor.process_train_test(prev_output["X_train"], prev_output["y_train"], prev_output["X_test"], prev_output["y_test"])

                # Convert tensors 
                X_train_w = torch.tensor(X_train_w, dtype=torch.float32)
                y_train_w = torch.tensor(y_train_w, dtype=torch.float32).unsqueeze(-1)  # -> (N, 24, 1)
                X_test_w = torch.tensor(X_test_w, dtype=torch.float32)
                y_test_w = torch.tensor(y_test_w, dtype=torch.float32).unsqueeze(-1)

                # Initialize and train model
                model = tsfedl.TsfedlAnomalyDetection(**kwargs)
                model.fit(X_train_w, y_train_w)

                # save results in context
                context[node_id].update({
                    "X_train_windows": X_train_w,
                    "X_test_windows": X_test_w,
                    "y_train_windows": y_train_w,
                    "y_test_windows": y_test_w,
                  
                })



            elif(params['algorithm_'] in flexanomalies_algorithms):
                kwargs = params
                print(  f"kwargs before initialization flexanomalies: {kwargs}")
                # Set FlexAnomalies specific parameters
                #kwargs["n_clients"] = 15
                #kwargs["n_rounds"] = 10
                
                if "input_dim" in kwargs:
                    kwargs["input_dim"] = X.shape[1]
                    
                if timeseries_visualization:
                    # Set Tranformers train loader
                    if "preprocess" in kwargs:
                        kwargs["preprocess"] = False
                        processor = TimeSeriesProcessor(window_size= kwargs["w_size"], step_size=1, future_prediction=False)
                        X_train_windows, y_train_windows, X_test_windows, y_test_windows = processor.process_train_test(prev_output["X_train"], prev_output["y_train"], prev_output["X_test"], prev_output["y_test"])
                    else:
                        processor =   TimeSeriesProcessor(window_size= kwargs["w_size"], step_size=1, future_prediction=True, n_pred=kwargs["n_pred"])
                        X_train_windows, y_train_windows, X_test_windows, y_test_windows, l_test_windows = processor.process_train_test(prev_output["X_train"], prev_output["y_train"], prev_output["X_test"], prev_output["y_test"],l_test=prev_output["y_test"])
                        deepModel = True
                    
                    model = flexanomalies.FlexAnomalyDetection(**kwargs)
                    print(f"Model initialized: {model}")
                    model.fit(X_train_windows,y_train_windows)
                    
                    context[node_id].update({
                    "X_train_windows": X_train_windows,
                    "X_test_windows": X_test_windows,
                    "y_train_windows": y_train_windows,
                    "y_test_windows": y_test_windows,
                    })
                    
                else:
                                 
                    model = flexanomalies.FlexAnomalyDetection(**kwargs)
                    print(f"Model initialized: {model}")
                    model.fit(X, prev_output.get("y_train", None))



            elif(params['algorithm_'] in transformers_algorithms):
                kwargs = params
                
                # Set Transformers specific parameters
                kwargs["label_parser"] = None
                kwargs['lr']= 0.001
                kwargs['train_epochs'] = kwargs.get("train_epochs", 1)
                kwargs['batch_size']= kwargs.get("batch_size", 16) #16
                
                # Set Tranformers train loader
                scaler = StandardScalerPreprocessing()
                X_scaled = scaler.fit_transform(X)
                
                processor = TimeSeriesProcessor(window_size=kwargs["seq_len"], step_size=1,future_prediction=False)
                #X_train_windows, y_train_windows, X_test_windows, y_test_windows = processor.process_train_test(X_train, y_train, X_test, y_test)
                X_train_windows, y_train_windows, X_test_windows, y_test_windows = processor.process_train_test(prev_output["X_train"], prev_output["y_train"], prev_output["X_test"], prev_output["y_test"])
                model = transformers.TransformersAnomalyDetection(**kwargs)
                model.fit(X_train_windows)

                
                context[node_id]["X_train_windows"] = X_train_windows
                context[node_id]["X_test_windows"] = X_test_windows
                context[node_id]["y_train_windows"] = y_train_windows
                context[node_id]["y_test_windows"] = y_test_windows
                # context[node_id]["X_test"] = X_test
                # context[node_id]["y_test"] = y_test

            else:
                return {"message": f"Error: (Model Setup) Unknown algorithm: {params['algorithm_']}"}

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
                
            if deepModel:
                    scores_pred = model.decision_function(X_train, prev_output.get("y_test_windows"))
            else:        
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
            
            
            if deepModel:
                pred = model.predict(X,prev_output["y_test_windows"]) 
            else:   
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
               
                # if prev_output.get("y_train_windows") is not None: #Transformer values
                #     print("MME MEETOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                #     print(prev_output["X_test"])
                #     model = prev_output["model"]
                #     model.evaluate(prev_output.get("X_test_windows"),prev_output.get("y_test_windows"))
                #     pred = model.labels_preds
                #     true = np.array(prev_output.get("y_test")).ravel()
                # else:
                
                if federated_visualizacion:
                     pred =  prev_output["model"].model.labels_       #labels 
                     true = np.array(prev_output["y_test"]).flatten()   #labels
                     print("Entreeeeeee")
                      
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
                
                #NUEVO
               
                if timeseries_visualization:
                    print(f"Creating visualization for node {node_id} in  {node_category}  with params: {clean_params}")
                    model = prev_output["model"]
                    if deepModel:
                        scores = model.decision_function(prev_output.get("X_test_windows"),prev_output.get("y_test_windows"))
                    else:   
                        scores = model.decision_function(prev_output.get("X_test_windows"))
                        
                    vis = DataVisualizationScoresTS(scores.ravel())
                    json_fig = vis.to_json()    
                    visualizations[node_id] = json_fig
                    # Store visualizations in results
                    results_dict["visualizations"] = visualizations 
                    
                else:    
                    vis = DataVisualization(data_to_plot, **clean_params) 
                    vis.fit()  
                    json_fig = vis.to_json()    
                    visualizations[node_id] = json_fig
                    # Store visualizations in results
                    print(f"Created visualization for node {node_id}  in  {node_category} with params: {clean_params}")
                    results_dict["visualizations"] = visualizations
                
            else: 
                return {"message": f"Error: (Visualization) Previous node is not a valid source for visualization."}
            
            results_dict["message"] = "Visualization created successfully!"

        else:
            return {"message": f"Unknown node type: {node_type}"}
        
        #Once node is processed, store its data for the next iteration
        prev_node = node
    print(timeseries_visualization)
    return results_dict



# ── Per-dataset normal label definitions ─────────────────────────────────
_DATASET_NORMAL_LABELS = {
    "shuttle": 1,                            # Class 1 = Rad Flow (~80 %)
    "spambase": 0,                           # 0 = not spam
    "mammographic_mass": 0,                  # 0 = benign
    "arrhythmia": 1,                         # 1 = normal rhythm
    "default_of_credit_card_clients": 0,     # 0 = no default
}


class _IdentityScaler:
    """No-op scaler so build_* functions skip scaling (the Preprocessing node does it)."""
    def fit_transform(self, X):
        return X
    def transform(self, X):
        return X


def process_dataset_static(dataset_name):
    """Load and prepare a static dataset for anomaly detection.

    Delegates to the build functions in ``anomaly_dataset_utils`` which handle:
    - Loading the raw dataset
    - Binarising labels (0 = normal, 1 = anomaly)
    - Stratified train/test split
    - Filtering training set to only normal samples
    - Imputing NaN values

    Scaling is skipped here (identity scaler) because the pipeline's
    Preprocessing node handles it.
    """
    if dataset_name == "kddcup99":
        result = build_kddcup99_anomaly_dataset(scaler_cls=_IdentityScaler)
    elif dataset_name == "human_activity_recognition":
        result = build_har_anomaly_dataset(scaler_cls=_IdentityScaler)
    else:
        normal_label = _DATASET_NORMAL_LABELS.get(dataset_name, 1)
        result = build_loaded_uci_anomaly_dataset(
            dataset_name=dataset_name,
            normal_label=normal_label,
            scaler_cls=_IdentityScaler,
        )

    # The build functions return y_test but not y_train;
    # y_train is all zeros because training only contains normal samples.
    result["y_train"] = np.zeros(result["X_train"].shape[0], dtype=int)

    # Wrap extra stats into a metadata sub-dict for the pipeline context
    meta_keys = [k for k in result if k not in ("X_train", "X_test", "y_train", "y_test")]
    result["metadata"] = {k: result.pop(k) for k in meta_keys}

    return result

def process_dataset_ts(data, dataset_name, test_size=0.2, random_state=42):
    """
   Adjusts X and y according to the specific dataset and returns the split in train/test.
    """
    X, y = data

    # Specific rules per dataset
    rules = {
        "ai4i_2020_predictive_maintenance_dataset": lambda X, y: (X.drop("Type", axis=1), y["Machine failure"]),
        "power_consumption_of_tetouan_city": lambda X, y: (X.drop("DateTime", axis=1), y["Zone 1 Power Consumption"]),
        "individual_household_electric_power_consumption": lambda X, y: (X.drop(["Date", "Time"], axis=1),X["Global_active_power"]),
        "metro_interstate_traffic_volume": lambda X, y: (X.drop(["date_time", "holiday","weather_main","weather_description"], axis=1),y["traffic_volume"]),
        
        }



    if dataset_name in rules:
        X, y = rules[dataset_name](X, y)

    # Split
    if y is None:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": None, "y_test": None
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test
        }