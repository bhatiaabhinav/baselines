# import numpy as np


def convert_to_constraints_dict(nbases, nresources, min_constraints, max_constraints):
    constraints = {
        "name": "root_node",
        "equals": nresources,
        "max": nresources,
        "min": nresources,
        "children": []
    }
    for i in range(nbases):
        child = {
            "name": "zone{0}".format(i),
            "zone_id": i,
            "equals": None,
            "min": min_constraints[i],
            "max": max_constraints[i],
            "children": []
        }
        constraints["children"].append(child)
    return constraints


def normalize_constraints(constraints, nresources=None):
    if constraints['equals'] is not None:
        nresources = constraints['equals']
        constraints['equals'] = constraints['equals'] / nresources
    assert nresources is not None, "Number of resources not specified"
    constraints['min'] = constraints['min'] / nresources
    constraints['max'] = constraints['max'] / nresources
    if 'children' in constraints:
        for child_constraints in constraints['children']:
            normalize_constraints(child_constraints, nresources)


def count_leaf_nodes_in_constraints(constraints):
    if 'children' not in constraints or len(constraints['children']) == 0:
        return 1
    else:
        count = 0
        for child_constraints in constraints['children']:
            count += count_leaf_nodes_in_constraints(child_constraints)
        return count


def count_nodes_in_constraints(constraints):
    count = 1
    if 'children' in constraints:
        for child_constraints in constraints['children']:
            count += count_nodes_in_constraints(child_constraints)
    return count


if __name__ == '__main__':
    nbases = 25
    nresources = 32
    c = convert_to_constraints_dict(
        nbases, nresources, min_constraints=[1] * nbases, max_constraints=[4] * nbases)
    normalize_constraints(c)
    print(c)
    print(count_leaf_nodes_in_constraints(c))
    print(count_nodes_in_constraints(c))
