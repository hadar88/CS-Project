import json

def convert_to_dict(data):
    days = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    meals = ["breakfast", "lunch", "dinner"]
    
    nested_data = data.get("output", [])
    structured_dict = {}
    
    for i, group in enumerate(nested_data):
        group_key = days[i]
        structured_dict[group_key] = {}
    
        for j, sub_group in enumerate(group):
            sub_group_key = meals[j]
            structured_dict[group_key][sub_group_key] = {}
    
            for k, pair in enumerate(sub_group):
                structured_dict[group_key][sub_group_key][pair[0]] = round(pair[1], 2)


# Example usage
input_file = "temp.json"   # Replace with actual file path
output_file = "temp2.json"
with open(input_file, "r") as file:
    data = json.load(file)
convert_to_dict(data)
