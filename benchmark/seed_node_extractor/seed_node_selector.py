from rdflib import URIRef
import json
import pdb
import random
import pandas as pd
import concurrent.futures
import os
import re
import seed_node_extractor.utils as utils
import seed_node_extractor.sampling as sampling
from kg.kg.kg import Node
from llm.prompt_chains import get_prompt_chains
from llm.llms import llms_dict

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

llm = llms_dict["question_generation_model"]

prompt_chains = get_prompt_chains()
representative_label_for_type = prompt_chains.get("get_representative_label_for_type")(llm)
max_label_length = 10

def trim_after_first_occurrence(text, pattern):
    # Find the first occurrence of the pattern
    match = re.search(pattern, text)
    
    # If the pattern is found, return the text up to the first occurrence
    if match:
        return text[:match.end()]
    else:
        # If the pattern is not found, return the original text
        return text

class NodeType:
    sample_index: int
    current_data: list
    offset: int
    node_type: str
    predicate_label: str

    def __init__(self, node_type, predicate_label):
        self.node_type = node_type
        self.offset = 0
        self.current_data = []
        self.sample_index = 0
        self.predicate_label = predicate_label

    def get_graph_size(self, node, knowledge_graph_uri):
        sparql = f"""SELECT count(*) as ?count WHERE {{ {{?s ?p <{node}>}} Union {{<{node}> ?p ?o}} }}"""
        sub_result = utils.send_sparql_query(knowledge_graph_uri, sparql)
        sub_count = sub_result["results"]["bindings"][0].get('count', {}).get('value', None)
        return int(sub_count)

    def get_distinct_predicates(self, node, knowledge_graph_uri):
        sparql = f"""SELECT count(distinct ?p) as ?count WHERE {{ {{?s ?p <{node}>}} Union {{<{node}> ?p ?o}} }}"""
        sub_result = utils.send_sparql_query(knowledge_graph_uri, sparql)
        sub_count = sub_result["results"]["bindings"][0].get('count', {}).get('value', None)
        return int(sub_count)


    def get_data(self, knowledge_graph_uri, knowledge_graph_prefix):
        if 'schema' == knowledge_graph_prefix:
            query = f"""
                   Select ?entity ?label
                   where {{ ?entity rdf:type <{self.node_type}> . ?entity <{self.predicate_label}> ?label . FILTER (lang(?label) = "en")}}

                   limit 10000
                   offset {self.offset}
                       """
        else:
            query = f"""
                Select ?entity ?label
                where {{ ?entity rdf:type <{self.node_type}> . ?entity <{self.predicate_label}> ?label . }}
                
                limit 10000
                offset {self.offset}
            """
        result = utils.send_sparql_query(knowledge_graph_uri, query)
        data = []
        for binding in result['results']['bindings']:
            entity = binding.get('entity', {}).get('value', None)
            label = binding.get('label', {}).get('value', None)
            if label.isascii() and len(label.split(' ')) <= max_label_length:
                data.append({'entity': entity, 'label': label})
        self.offset += 10000
        self.current_data = data
        self.sample_index = 0

    def get_one_sample(self, knowledge_graph_uri, knowledge_graph_prefix, sampled_nodes):
        while True:
            if self.sample_index == len(self.current_data):
                self.get_data(knowledge_graph_uri, knowledge_graph_prefix)

            node = self.current_data[self.sample_index]
            self.sample_index += 1
            graph_size = self.get_graph_size(node["entity"], knowledge_graph_uri)
            distinct_predicates = self.get_distinct_predicates(node["entity"], knowledge_graph_uri)
            min_distinct_predicates = 7 if knowledge_graph_prefix == 'makg' else 10
            if node["entity"] not in sampled_nodes and 20 < graph_size < 1000 and distinct_predicates > min_distinct_predicates:
                sampled_nodes.append(node["entity"])
                return Node(uri=URIRef(node["entity"]), nodetype=URIRef(self.node_type.strip()))

    def get_samples(self, num_samples, knowledge_graph_uri, knowledge_graph_prefix, sampled_nodes):
        node_samples = list()
        for i in range(num_samples):
            node_samples.append(self.get_one_sample(knowledge_graph_uri, knowledge_graph_prefix, sampled_nodes))
        return node_samples


class SeedNodeSelector:

    def __init__(self, kg_name, seed_nodes_file=None, is_random=False):
        self.sampled_nodes = list()
        self.is_random = is_random
        self.type_to_index = dict()
        self.type_to_offset = dict()
        self.type_toNodetype = dict()
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

    def retrieve_initial_list_top_k(self, kg_name, num_samples=100):
        # def retrieve_initial_list_top_k(self, num_samples=100):
        if self.seed_nodes_file is None:
            return self.retrieve_initial_list_top_k_from_kg_new(kg_name, num_samples)
            # return self.retrieve_initial_list_top_k_from_kg(num_samples)
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

            file_name = f"{kg_name}_types_representative.json"
            filepath = os.path.join(CURR_DIR, file_name)
            type_to_predicate_map = dict()
            if os.path.exists(filepath):
                file = open(filepath, 'r')
                type_to_predicate_map = json.load(file)

            processed_nodes = set()
            node_samples = list()
            distribution = dict()
            for binding in result["results"]["bindings"]:
                node = binding.get('entity', {}).get('value', None)
                node_type = binding.get('type', {}).get('value', None).strip()
                if node not in processed_nodes and node_type in type_to_predicate_map:
                    node_samples.append(
                        Node(uri=URIRef(node), nodetype=URIRef(node_type)))
                    processed_nodes.add(node)
                    if node_type in distribution:
                        distribution[node_type] += 1
                    else:
                        distribution[node_type] = 1

            nodetype_to_label = self.get_representative_label_per_node_type(distribution, type_to_predicate_map,filepath)
            return node_samples, distribution, nodetype_to_label

    def get_node_for_label_extraction(self, node_type):
        query = f""" select ?entity where {{ ?entity rdf:type <{node_type}>}} Limit 1"""
        result = utils.send_sparql_query(self.knowledge_graph_uri, query)
        for binding in result["results"]["bindings"]:
            node = binding.get('entity', {}).get('value', None)
            return node

    def get_representative_label_per_node_type(self, sampling_distribution, exisiting_map, file_name):
        type_per_label = exisiting_map
        for key, value in sampling_distribution.items():
            if value == 0:
                continue
            key = key.strip()
            if key not in type_per_label:
                sample_node = self.get_node_for_label_extraction(key)
                query = f"select distinct ?p, ?ent where {{ <{sample_node}> ?p ?ent}}"
                result = utils.send_sparql_query(self.knowledge_graph_uri, query)
                predicates = list()
                for binding in result["results"]["bindings"]:
                    entity_type = binding.get('ent', {}).get('type', None)
                    predicate = binding.get('p', {}).get('value', None)
                    if (entity_type == 'literal' or entity_type == 'typed-literal') and predicate not in predicates:
                        predicates.append(predicate)

                try:
                    # pdb.set_trace()
                    ch = representative_label_for_type.get("chain")
                    llm_result = ch.generate([{"node_type": key, "predicates": ', '.join(predicates)}], None)
                    print(llm_result)
                    for generation in llm_result.generations:
                        generation[0].text = "```json" + trim_after_first_occurrence(generation[0].text, "```")
                        print("gen-text: ", generation[0].text)
                    output = [
                        # Get the text of the top generated string.
                        {
                            ch.output_key: ch.output_parser.parse_result(generation),
                            "full_generation": generation,
                        }
                        for generation in llm_result.generations
                    ]
                    if ch.return_final_only:
                        output = [{ch.output_key: r[ch.output_key]} for r in output]
                    output = output[0][ch.output_key]
                    print(output)
                    type_per_label[key] = output["predicate"].strip()
                except Exception as e:
                    response = str(e)
                    if response.startswith("Got invalid return object. Expected key"):
                        start_index = response.index('got')
                        type_per_label[key] = response[start_index + 3: len(response)].strip()
                with open(file_name, 'w') as file:
                    json.dump(type_per_label, file, indent=4)
        return type_per_label

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
            return self.sample_node_from_kg_new(node_type)
            # return self.sample_node_from_kg(node_type)
        else:
            if self.seed_file_index >= len(self.file_seeds):
                print("Seed file index is out of range")
                return None
            node = self.file_seeds[self.seed_file_index].strip()
            self.seed_file_index += 1
            query = f"SELECT ?type where {{ <{node}> rdf:type ?type }}"
            result = utils.send_sparql_query(self.knowledge_graph_uri, query)
            node_type = result["results"]["bindings"][0].get('type', {}).get('value', None)
            return Node(uri=URIRef(node), nodetype=URIRef(node_type.strip()))

    def sample_node_from_kg(self, node_type):
        node = self.get_samples_for_type(node_type, 1, self.knowledge_graph_prefix)
        return node[0]

    def sample_node_from_kg_new(self, node_type):
        # pdb.set_trace()
        node_type = str(node_type)
        nodetype_obj = self.type_toNodetype[node_type]
        return nodetype_obj.get_one_sample(self.knowledge_graph_uri, self.knowledge_graph_prefix, self.sampled_nodes)

    def return_seed_nodes_new(self, samples_per_type, type_to_label):
        node_samples = list()
        for key, value in samples_per_type.items():
            if value > 0:
                if key in self.type_toNodetype:
                    nodeTypeObj = self.type_toNodetype[key]
                else:
                    nodeTypeObj = NodeType(key, type_to_label[key])
                    self.type_toNodetype[key] = nodeTypeObj
                node_type_samples = nodeTypeObj.get_samples(value, self.knowledge_graph_uri, self.knowledge_graph_prefix, self.sampled_nodes)
                node_samples.extend(node_type_samples)
        return node_samples

    def get_distinct_sample(self, node_type):
        query = f"""
        SELECT (AVG(?count) as ?average_count)
        WHERE {{
          {{
            SELECT ?entity (COUNT(DISTINCT ?p) as ?count)
            WHERE {{
              {{ ?entity rdf:type <{node_type}>. ?entity ?p ?o }}
              UNION
              {{ ?entity rdf:type <{node_type}>. ?s ?p ?entity }}
            }}
            GROUP BY ?entity
            LIMIT 1000
          }}
        }}
        """
        result = utils.send_sparql_query(self.knowledge_graph_uri, query)
        for binding in result['results']['bindings']:
            average = binding.get('average_count', {}).get('value', None)
            average = int(average)
        return {"Type": node_type, "average": average}

    def remove_poor_types(self, input_df):
        types = input_df['Type'].values
        num_threads = 10
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.get_distinct_sample, type) for type in types]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.extend(result)

        excluded_list = list()
        for result in results:
            if result['average'] < 5:
                excluded_list.append(result['Type'])
        df_cleaned = input_df[~input_df['Type'].isin(excluded_list)]
        print(df_cleaned)
        return df_cleaned.to_json(orient='records')

    def eliminate_dominated_parents(self, df, knowledge_graph_uri):
        df['Type'] = df['Type'].str.strip()
        children_names = [x.strip() for x in df['Type'].values]
        parents = sampling.get_parents(children_names, knowledge_graph_uri)
        types_to_remove = list()
        for child, parent in zip(children_names, parents):
            if parent is not None:
                count_child = df[df['Type'] == child]['Count'].values[0]
                count_parent = df[df['Type'] == parent]['Count'].values[0]
                if count_child / (count_parent * 1.0) > 0.99:
                    types_to_remove.append(parent)
        df_cleaned = df[~df['Type'].isin(types_to_remove)]
        print(df_cleaned)
        return df_cleaned.to_json(orient='records')
        # return df_cleaned

    def remove_rare_types(self, input_df):
        threshold = 1
        filtered_df = input_df[input_df['percentage'] > threshold]
        filtered_df = filtered_df.drop(columns=['percentage'])
        total_count = filtered_df['Count'].sum()
        percentage_df = filtered_df.copy()
        percentage_df['percentage'] = (percentage_df['Count'] / total_count) * 100
        # return percentage_df.to_json(orient='records')
        return percentage_df

    def calculate_class_importance(self, input_df):
        input_df['Count'] = input_df['Count'].astype(int)
        total_count = input_df['Count'].sum()
        percentage_df = input_df.copy()
        percentage_df['percentage'] = (percentage_df['Count'] / total_count) * 100
        return percentage_df

    def retrieve_initial_list_top_k_from_kg_new(self, kg_name, num_samples):
        # pdb.set_trace()
        kg_type_distribution = utils.get_type_distrubution(self.knowledge_graph_uri, self.knowledge_graph_prefix)
        distribution = pd.DataFrame(kg_type_distribution)
        percentage_df = self.calculate_class_importance(distribution)
        update_df = self.remove_rare_types(percentage_df)
        cleaned_df = self.eliminate_dominated_parents(update_df, self.knowledge_graph_uri)
        # cleaned_df = self.remove_poor_types(cleaned_df)
        json_object = json.loads(cleaned_df)
        sample_distribution = sampling.get_sample_distribution(json_object, num_samples)
        print(sample_distribution)
        file_name = f"{kg_name}_types_representative.json"
        filepath = os.path.join(CURR_DIR, file_name)
        type_to_predicate_map = dict()
        if os.path.exists(filepath):
            file = open(filepath, 'r')
            type_to_predicate_map = json.load(file)
        nodetype_to_label = self.get_representative_label_per_node_type(sample_distribution, type_to_predicate_map,filepath)
        seed_nodes = self.return_seed_nodes_new(sample_distribution, nodetype_to_label)
        return seed_nodes, sample_distribution, nodetype_to_label



if __name__ == '__main__':
    kg_name = 'makg'
    sampler = SeedNodeSelector(kg_name)
    # initial, sample = sampler.retrieve_initial_list_top_k(10)
    initial, sample, _ = sampler.retrieve_initial_list_top_k_from_kg_new(kg_name, 20)
    for seed in initial:
        print(f"{seed.uri}\t{seed.nodetype}")
