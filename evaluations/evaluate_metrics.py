import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from collections import Counter
import json
from typing import List, Any


def read_json(filename: str):
    with open(filename, "r") as f:
        json_data = json.load(f)
        return json_data


def read_jsonl(filename: str) -> List[Any]:
    with open(filename, "r") as f:
        data = f.readlines()
        json_data = [json.loads(d) for d in data]
        return json_data

nltk.download("punkt")

def compute_bleu_score(reference, hypothesis):
    return sentence_bleu([reference.split()], hypothesis.split())

def compute_rouge_scores(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores

def compute_ngram_similarity(reference, hypothesis, n=1):
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    reference_ngrams = [tuple(reference_tokens[i:i + n]) for i in range(len(reference_tokens) - n + 1)]
    hypothesis_ngrams = [tuple(hypothesis_tokens[i:i + n]) for i in range(len(hypothesis_tokens) - n + 1)]
    
    reference_ngram_counts = Counter(reference_ngrams)
    hypothesis_ngram_counts = Counter(hypothesis_ngrams)
    
    intersection = sum((reference_ngram_counts & hypothesis_ngram_counts).values())
    union = sum(reference_ngram_counts.values()) + sum(hypothesis_ngram_counts.values())
    
    return intersection / union

def evaluate_text_tuples(text_tuples):
    bleu_scores = []
    rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}
    ngram_similarities = {"1-gram": [], "2-gram": [], "3-gram": []}
    
    for reference, hypothesis in text_tuples:
        bleu = compute_bleu_score(reference, hypothesis)
        bleu_scores.append(bleu)
        
        rouge = compute_rouge_scores(reference, hypothesis)
        for metric in rouge.keys():
            rouge_scores[metric].append(rouge[metric]["f"])
        
        for n in range(1, 4):
            similarity = compute_ngram_similarity(reference, hypothesis, n)
            ngram_similarities[f"{n}-gram"].append(similarity)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}
    avg_ngram = {ngram: sum(similarities) / len(similarities) for ngram, similarities in ngram_similarities.items()}
    
    return avg_bleu, avg_rouge, avg_ngram

if __name__ == "__main__":
    data = read_jsonl("output_data.jsonl")
    test_data = []
    for d in data:
        rewrite = d["Rewrite"]
        ai_rewrite = d["AiRewrite"]
        test_data.append((rewrite, ai_rewrite))

    # text_tuples = [
    #     ("This is a reference sentence.", "This is a generated sentence."),
    #     ("The cat is on the mat.", "A cat sits on the mat."),
    #     ("Hello, world!", "Hi, there!"),
    # ]
    
    avg_bleu, avg_rouge, avg_ngram = evaluate_text_tuples(test_data)
    
    print(f"Average BLEU Score: {avg_bleu}")
    print(f"Average ROUGE Scores: {avg_rouge}")
    print(f"Average N-gram Similarities: {avg_ngram}")


### output_data_large.jsonl ###
"""
Average BLEU Score: 0.3419926371952984
Average ROUGE Scores: {'rouge-1': 0.6376527938954858, 'rouge-2': 0.4806196491328122, 'rouge-l': 0.6221112998162439}
Average N-gram Similarities: {'1-gram': 0.3176190465533474, '2-gram': 0.24031803691393022, '3-gram': 0.18905196467657898}
"""

### output_data.jsonl ###
"""
Average BLEU Score: 0.3826841327213166
Average ROUGE Scores: {'rouge-1': 0.6649033137939901, 'rouge-2': 0.5188024958759003, 'rouge-l': 0.6459553544170027}
Average N-gram Similarities: {'1-gram': 0.3310718979228084, '2-gram': 0.25940125013939336, '3-gram': 0.2111427577591092}
"""
