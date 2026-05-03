import os
import importlib


def get_all_algorithms():
    alg_dir = os.path.dirname(__file__)
    algorithms = []

    for filename in os.listdir(alg_dir):
        # keep only real python files, skip __init__.py and hidden/special files
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3].strip()
            if module_name:   # avoid empty names
                algorithms.append(module_name)

    return algorithms


fed_servers = {}
fed_clients = {}

for algorithm in get_all_algorithms():
    mod = importlib.import_module('algorithm.' + algorithm)
    server_class_name = algorithm + 'Server'
    client_class_name = algorithm + 'Client'

    fed_servers[algorithm] = getattr(mod, server_class_name)
    fed_clients[algorithm] = getattr(mod, client_class_name)


def get_server(model_name, args, clients, model, data, logger):
    return fed_servers[model_name](args, clients, model, data, logger)


def get_client(model_name, args, model, data):
    return fed_clients[model_name](args, model, data)


def load_client(args, model, data):
    pass


def load_server(args, clients, model, data, logger):
    pass