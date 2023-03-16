import yaml

def yaml_loader(filepath):
    """Loads yaml file from filepath"""
    with open(filepath, "r") as file_descriptor:
        yaml_file = yaml.safe_load(file_descriptor)
    return yaml_file

def yaml_dump(filepath, data):
    """Dump data to yaml file"""
    with open(filepath, "w") as file_descriptor:
        yaml.dump(data, file_descriptor)

if __name__ == "__main__":
    filepath = 'config_distcovertree.yaml'
    data = yaml_loader(filepath)
    print(data)
    items = data.get('general')
    print(items)