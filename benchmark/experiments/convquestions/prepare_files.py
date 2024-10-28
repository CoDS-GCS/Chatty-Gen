import json

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
               "http://dbpedia.org/resource/Major_League_Soccer", "http://dbpedia.org/resource/Harry_Potter",
               "http://dbpedia.org/resource/Fight_Club", "http://dbpedia.org/resource/The_Beatles",
               "http://dbpedia.org/resource/Manchester_United_F.C.", "http://dbpedia.org/resource/Game_of_Thrones"]

def get_chatty_gen_file():
    # files = [
    #     '/home/rehamomar/Downloads/dbpedia_subgraph_summ/subgraph_summ_books/gpt-4o/dbpedia_subgraph_summarized_20_5_original.json',
    #     '/home/rehamomar/Downloads/dbpedia_subgraph_summ/subgraph_summ_movies/gpt-4o/dbpedia_subgraph_summarized_20_5_original.json',
    #     '/home/rehamomar/Downloads/dbpedia_subgraph_summ/subgraph_summ_music/gpt-4o/dbpedia_subgraph_summarized_20_5_original.json',
    #     '/home/rehamomar/Downloads/dbpedia_subgraph_summ/subgraph_summ_soccer/gpt-4o/dbpedia_subgraph_summarized_20_5_original.json',
    #     '/home/rehamomar/Downloads/dbpedia_subgraph_summ/subgraph_summ_tv_series/gpt-4o/dbpedia_subgraph_summarized_20_5_original.json',
    #          ]
    files = [
        '/home/rehamomar/Project/Chatbot-Resources/benchmark/seed_files_output/gpt/dbpedia_subgraph_summarized_25_5_original.json'
        # '/home/rehamomar/Downloads/subgraph_summ_latest/subgraph_summ_books/gemini-15pro/dbpedia_subgraph_summarized_20_5_original.json',
        # '/home/rehamomar/Downloads/subgraph_summ_latest/subgraph_summ_books/gemini-15pro/dbpedia_subgraph_summarized_20_5_original.json',
        # '/home/rehamomar/Downloads/subgraph_summ_latest/subgraph_summ_books/gemini-15pro/dbpedia_subgraph_summarized_20_5_original.json',
        # '/home/rehamomar/Downloads/subgraph_summ_latest/subgraph_summ_books/gemini-15pro/dbpedia_subgraph_summarized_20_5_original.json',
        # '/home/rehamomar/Downloads/subgraph_summ_latest/subgraph_summ_books/gemini-15pro/dbpedia_subgraph_summarized_20_5_original.json',
             ]
    final_objs = list()
    ordered_entity_list = list()
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
        for obj in data["data"]:
            entity = obj["seed_entity"]
            if entity in entity_list:
                if entity not in ordered_entity_list:
                    ordered_entity_list.append(obj["seed_entity"])
                    final_objs.append(obj['dialogue'])


    print(len(ordered_entity_list))
    with open('chatty_gen_data.json', 'w') as f:
        json.dump(final_objs, f, indent=4)

    # Handle difference between wikidata and Dbpedia labels
    return ordered_entity_list

def get_conv_file(ordered_entity_urls):
    file_name = '/home/rehamomar/Downloads/test_set_ALL_new.json'
    with open(file_name, 'r') as file:
        data = json.load(file)
    processed_entities = list()
    entity_to_questions = dict()
    final_objs = list()
    for obj in data:
        dbpedia_entity = obj['dbpedia_url']
        if dbpedia_entity not in processed_entities and dbpedia_entity in ordered_entity_urls:
            processed_entities.append(dbpedia_entity)
            questions = list()
            for q in obj['questions']:
                # question = q['completed_question'] if "completed_question" in q else q['question']
                question = q['question']
                questions.append(question)
            entity_to_questions[dbpedia_entity] = questions
    for entity in ordered_entity_urls:
        final_objs.append(entity_to_questions[entity])
    with open('conv_data.json', 'w') as f:
        json.dump(final_objs, f, indent=4)

if __name__ == '__main__':
    entity_labels = get_chatty_gen_file()
    get_conv_file(entity_labels)
