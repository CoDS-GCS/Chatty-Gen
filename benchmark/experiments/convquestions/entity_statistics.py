from SPARQLWrapper import SPARQLWrapper, JSON

entity_list = ["http://dbpedia.org/resource/Moby-Dick", "http://dbpedia.org/resource/Carrie_(novel)",
               "http://dbpedia.org/resource/The_Wonderful_Wizard_of_Oz",
               "http://dbpedia.org/resource/Gone_with_the_Wind_(film)", "http://dbpedia.org/resource/Grease_(film)",
               "http://dbpedia.org/resource/Kindergarten_Cop",
               "http://dbpedia.org/resource/Night_of_the_Living_Dead", "http://dbpedia.org/resource/Camp_Rock",
               "http://dbpedia.org/resource/Janis_Joplin", "http://dbpedia.org/resource/Deadmau5",
               "http://dbpedia.org/resource/Avril_Lavigne", "http://dbpedia.org/resource/Gloryhammer",
               "http://dbpedia.org/resource/Friends", "http://dbpedia.org/resource/Seinfeld",
               "http://dbpedia.org/resource/Monty_Python's_Flying_Circus",
               "http://dbpedia.org/resource/Saved_by_the_Bell", "http://dbpedia.org/resource/Tim_Howard",
               "http://dbpedia.org/resource/Lionel_Messi", "http://dbpedia.org/resource/Son_Heung-min",
               "http://dbpedia.org/resource/Major_League_Soccer"]

entity_wikidata = ["Q174596", "Q623394", "Q130295", "Q2875", "Q267721", "Q257630", "Q623051", "Q217250", "Q1514",
                   "Q49009", "Q30449", "Q11838832", "Q79784", "Q23733", "Q16401", "Q1026823", "Q200785", "Q615",
                   "Q439722", "Q18543"]

# Function to run a SPARQL query
def run_sparql_query(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()
    return result

def get_predicate_count_dbpedia(entity):
    query = f"""
        SELECT (COUNT( Distinct ?predicate) AS ?predicateCount)
        WHERE {{
          <{entity}> ?predicate ?object .
        }}
        """
    endpoint_url = "http://206.12.95.86:8890/sparql"
    result = run_sparql_query(endpoint_url, query)
    count = int(result["results"]["bindings"][0]["predicateCount"]["value"])
    return count

# Function to get the number of predicates (triples) surrounding an entity
def get_predicate_count(entity):
    query = f"""
    SELECT (COUNT(Distinct ?predicate) AS ?predicateCount)
    WHERE {{
      wd:{entity} ?predicate ?object .
    }}
    """
    endpoint_url = "https://query.wikidata.org/sparql"
    result = run_sparql_query(endpoint_url, query)
    count = int(result["results"]["bindings"][0]["predicateCount"]["value"])
    return count

# Main function to get average predicate count for a list of entities
def average_predicate_count(entities, kg):
    total_predicates = 0
    for entity in entities:
        if kg == 'dbpedia':
            count = get_predicate_count_dbpedia(entity)
        else:
            count = get_predicate_count(entity)
        print(f"Entity: {entity}, Predicate Count: {count}")
        total_predicates += count
    average = total_predicates / len(entities)
    return average

# # List of Wikidata entities (replace with your own list)
# entities = ['Q42', 'Q64', 'Q159']

# Calculate the average number of predicates
average_count_dbpedia = average_predicate_count(entity_list, 'dbpedia')
average_count_wikidata = average_predicate_count(entity_wikidata, 'wikidata')
print(f"\nAverage number of predicates DBpedia: {average_count_dbpedia}")
print(f"\n\nAverage number of predicates Wikidata: {average_count_wikidata}")
