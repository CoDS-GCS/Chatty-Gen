import seed_node_extractor.utils as utils
import seed_node_extractor.sampling as sampling
from benchmark.kg.kg.kg import Node
from rdflib import URIRef
import json
import random


class SeedNodeSelector:

    def __init__(self, kg_name, seed_nodes_file=None, is_random=False):
        self.sampled_nodes = list()
        self.is_random = is_random
        self.type_to_index = dict()
        self.knowledge_graph_uri = utils.knowledge_graph_to_uri[kg_name][0]
        self.knowledge_graph_prefix = utils.knowledge_graph_to_uri[kg_name][1]
        self.seed_nodes_file = seed_nodes_file
        if seed_nodes_file is not None:
            with open(seed_nodes_file, 'r') as file:
                self.file_seeds = file.read().splitlines()
            self.seed_file_index = 0

    def return_seed_nodes(self, samples_per_type, prefix):
        node_samples = list()
        for key, value in samples_per_type.items():
            if value > 0:
                node_type_samples = self.get_samples_for_type(key, value, prefix)
                node_samples.extend(node_type_samples)
        return node_samples


    def get_samples_for_type(self, node_type, num_samples, prefix):
        file_name = utils.get_file_name_from_type(node_type)
        human_readable_type = f"index_data/{prefix}/{file_name}.txt"
        with open(human_readable_type, "r") as file:
            lines = file.readlines()
        data = [line.strip().split('\t') for line in lines]
        data = [(type_, int(count)) for type_, count in data]
        node_samples = list()

        if self.is_random:
            samples = list()
            sample_count = num_samples
            while len(samples) < num_samples:
                sampled_entities = random.sample(data, k=sample_count)
                for entity in sampled_entities:
                    if entity[0] not in samples and int(
                            entity[1]) >= 2:
                        samples.append(entity[0])
                sample_count = num_samples - len(samples)
            for sample in samples:
                node_samples.append(Node(uri=URIRef(sample), nodetype=URIRef(type.strip())))
            return node_samples
        else:
            sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
            sorted_count = 0 if node_type not in self.type_to_index else self.type_to_index[node_type]
            for i in range(num_samples):
                while sorted_data[sorted_count][0] in self.sampled_nodes:
                    sorted_count += 1
                node_samples.append(Node(uri=URIRef(sorted_data[sorted_count][0]), nodetype=URIRef(node_type.strip())))
                self.sampled_nodes.append(sorted_data[sorted_count][0])
                sorted_count += 1
            self.type_to_index[node_type] = sorted_count
            return node_samples

    def retrieve_initial_list_top_k(self, num_samples=100):
        if self.seed_nodes_file is None:
            return self.retrieve_initial_list_top_k_from_kg(num_samples)
        else:
            seralized_nodes = ""
            for i in range(num_samples):
                if self.seed_file_index >= len(self.file_seeds):
                    break
                seralized_nodes += f"<{self.file_seeds[self.seed_file_index].strip()}> "
                self.seed_file_index += 1
            query = ("select distinct ?entity, ?type where {VALUES ?entity {"
                             f"{seralized_nodes}"
                             "} ?entity rdf:type ?type}")
            result = utils.send_sparql_query(self.knowledge_graph_uri, query)
            processed_nodes = set()
            node_samples = list()
            distribution = dict()
            for binding in result["results"]["bindings"]:
                node = binding.get('entity', {}).get('value', None)
                node_type = binding.get('type', {}).get('value', None).strip()
                if node not in processed_nodes:
                    node_samples.append(
                        Node(uri=URIRef(node), nodetype=URIRef(node_type)))
                    processed_nodes.add(node)
                    if node_type in distribution:
                        distribution[node_type] += 1
                    else:
                        distribution[node_type] = 1

            return node_samples, distribution


    def retrieve_initial_list_top_k_from_kg(self, num_samples=100):
        average_richness_file = f"index_data/{self.knowledge_graph_prefix}/average_per_type.txt"
        filtered_df = sampling.remove_low_richness(average_richness_file)
        percentage_df = sampling.calculate_class_importance(filtered_df)
        update_df = sampling.remove_rare_types(percentage_df)
        cleaned_df = sampling.eliminate_dominated_parents(update_df, self.knowledge_graph_uri)
        json_object = json.loads(cleaned_df)
        sample_distribution = sampling.get_sample_distribution(json_object, num_samples)
        print(sample_distribution)
        seed_nodes = self.return_seed_nodes(sample_distribution, self.knowledge_graph_prefix)
        return seed_nodes, sample_distribution

    def sample_node(self, node_type):
        if self.seed_nodes_file is None:
            return self.sample_node_from_kg(node_type)
        else:
            if self.seed_file_index >= len(self.file_seeds):
                raise Exception("Seed file index is out of range")
            node = self.file_seeds[self.seed_file_index].strip()
            self.seed_file_index += 1
            query = f"SELECT ?type where {{ <{node}> rdf:type ?type }}"
            result = utils.send_sparql_query(self.knowledge_graph_uri, query)
            node_type = result["results"]["bindings"][0].get('type', {}).get('value', None)
            return Node(uri=URIRef(node), nodetype=URIRef(node_type.strip()))

    def sample_node_from_kg(self, node_type):
        node = self.get_samples_for_type(node_type, 1, self.knowledge_graph_prefix)
        return node[0]


if __name__ == '__main__':
    sampler = SeedNodeSelector('dblp')
    initial, sample = sampler.retrieve_initial_list_top_k(10)
    for seed in initial:
        print(seed)