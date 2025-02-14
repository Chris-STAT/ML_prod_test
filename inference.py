import json
import os
import xgboost as xgb

def model_fn(model_dir):
    model_file = os.path.join(model_dir, 'xgboost-model')
    model = xgb.Booster()
    model.load_model(model_file)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return xgb.DMatrix(data)
    else:
        raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction.tolist())
    else:
        raise ValueError("Unsupported content type")

def handler(data, context):
    model = context.model
    return predict_fn(input_fn(data, context.request_content_type), model)