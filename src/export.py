import os
import onnx
import pickle

from skl2onnx.common.data_types import FloatTensorType

from onnxmltools.convert.xgboost import convert as convert_xgboost

from settings import load_config

general, _ = load_config()


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, model_path=general["outputs"]["pickle"]):
    ensure_directory_exists(model_path)

    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def save_model_onnx(model, input_shape, model_path=general["outputs"]["onnx"]):
    ensure_directory_exists(model_path)

    initial_types = [("input", FloatTensorType([None, input_shape[1]]))]
    onnx_model = convert_xgboost(model, initial_types=initial_types)

    onnx.save_model(onnx_model, model_path)


def save_model_json(model, model_path=general["outputs"]["raw"]):
    ensure_directory_exists(model_path)
    model.save_model(model_path)
