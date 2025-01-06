import json

def load_config():
    with open("settings/general.json", "rb") as f1:
        general = json.load(f1)
    with open("settings/hyperparameters.json", "r") as f2:
        hyperparameters = json.load(f2)
    return general, hyperparameters