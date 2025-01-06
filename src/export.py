import os
import pickle
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


def save_model_json(model, model_path=general["outputs"]["raw"]):
    ensure_directory_exists(model_path)
    model.save_model(model_path)
