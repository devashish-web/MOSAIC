import os
import importlib

BACKBONE_DIR = os.path.dirname(__file__)


def get_all_models():
    models = []
    for fname in os.listdir(BACKBONE_DIR):
        full_path = os.path.join(BACKBONE_DIR, fname)

        if not os.path.isfile(full_path):
            continue
        if not fname.endswith(".py"):
            continue
        if fname.startswith("__"):
            continue

        model_name = fname[:-3].strip()
        if model_name:
            models.append(model_name)

    return models


names = {}
for model in get_all_models():
    mod = importlib.import_module("backbone." + model)
    class_map = {x.lower(): x for x in dir(mod)}
    class_name = class_map[model.lower()]
    names[model] = getattr(mod, class_name)


def get_model(model_name, input_dim, hidden_dim, out_dim, num_layers, dropout):
    return names[model_name](input_dim, hidden_dim, out_dim, num_layers, dropout)