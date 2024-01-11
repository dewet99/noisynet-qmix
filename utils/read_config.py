import yaml

def merge_yaml_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)

    # Merge the data
    merged_data = merge_dicts(data1, data2)

    # Write the merged data to a new YAML file
    return merged_data

def merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.
    Values from dict2 overwrite corresponding values in dict1.
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged