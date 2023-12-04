from rdflib import Graph, RDF, RDFS

# Create an RDF graph and parse your RDF file
g = Graph()
g.parse("rdf_schema.rdf", format="xml")  # Replace with your RDF file and format

# Create dictionaries to store the whitelist and blacklist
whitelist = {}
blacklist = {}


# Create dictionaries to store classes and properties
classes = {}
properties = {}

for x in g.predicates():
    print("-->", x)
    # break

# Iterate through the classes in the RDF graph
for class_uri in g.subjects(RDF.type, RDFS.Class):
    print(class_uri)
    class_label = g.label(class_uri)
    class_comment = g.comment(class_uri)
    classes[class_uri] = {
        "label": class_label,
        "comment": class_comment,
    }

# Iterate through the properties in the RDF graph
for predicate in g.predicates():
    pred_label = g.label(predicate)
    pred_comment = g.comment(predicate)
    properties[predicate] = {
        "label": pred_label,
        "comment": pred_comment,
    }

# Create a list to store the triples with directions
triples_with_directions = []

# Iterate through the triples in the RDF graph
for s, p, o in g:
    # Check if the subject and object are classes or resources
    if s in classes and o in classes:
        triples_with_directions.append((s, p, o, "left to right"))
        triples_with_directions.append((o, p, s, "right to left"))

# Print classes and properties
print("Classes:")
for class_uri, class_info in classes.items():
    print(f"URI: {class_uri}")
    print(f"Label: {class_info['label']}")
    print(f"Comment: {class_info['comment']}")
    print()

print("\nProperties:")
for pred_uri, pred_info in properties.items():
    print(f"URI: {pred_uri}")
    print(f"Label: {pred_info['label']}")
    print(f"Comment: {pred_info['comment']}")
    print()

# Print triples with directions
print("\nTriples with Directions:")
for triple in triples_with_directions:
    s, p, o, direction = triple
    print(f"Subject: {s}")
    print(f"Predicate: {p}")
    print(f"Object: {o}")
    print(f"Direction: {direction}")
    print()
