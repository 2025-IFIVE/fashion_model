from grouping_map import MERGE_MAP

def merge_attribute_value(attr_name, value):
    
    if attr_name not in MERGE_MAP:
        return value
    for group, keywords in MERGE_MAP[attr_name].items():
        if value in keywords:
            return group
    return value

def merge_attribute_list(attr_name, values):
    
    if attr_name not in MERGE_MAP:
        return values
    merged = set()
    for v in values:
        for group, keywords in MERGE_MAP[attr_name].items():
            if v in keywords:
                merged.add(group)
    return list(merged) if merged else values