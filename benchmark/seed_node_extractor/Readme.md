The steps followed are:
1. script `retrieve_node_per_type`: Retrieve All nodes
   1. Get all types of the knowledge graph. We are only considering types of the KG domain
   2. For each type, we retrieve all nodes for that type using multithreading, and different offsets
2. script `get_predicates_count_per_node`: For all nodes of a specific type, get number of unique predicates connected to it
   1. Get the types and their distribution
   2. Use the nodes saved in step 1 and get the unique predicate count for each of them