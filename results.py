from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize, sent_tokenize
from torchmetrics.text.bert import BERTScore
from nltk.translate.meteor_score import meteor_score
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import spacy
import numpy as np
from rouge import Rouge

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
import os
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import bert_score
import matplotlib.pyplot as plt



nlp = spacy.load("en_core_web_md")
rouge = Rouge()


def get_scores(ref, can):
    smoothie = SmoothingFunction().method4
    r_scores = rouge.get_scores(ref, can)
    r_scores = {k: v['f'] for k,v in r_scores[0].items()}
    ref_tokens = word_tokenize(ref)
    can_tokens = word_tokenize(can)

    bleu1 = sentence_bleu([ref_tokens], can_tokens, weights=(1,0,0,0), smoothing_function=smoothie)
    bleu2 = sentence_bleu([ref_tokens], can_tokens, weights=(0,1,0,0), smoothing_function=smoothie)
    complete_bleu = sentence_bleu([ref_tokens], can_tokens, smoothing_function=smoothie)

    meteor = meteor_score([ref_tokens], can_tokens)

    b_scores = {"bleu1": bleu1, "bleu2": bleu2, "bleu":complete_bleu}
    scores = r_scores.copy()
    for key, val in b_scores.items():
        scores[key] = val

    scores['meteor'] = meteor
    return scores


def avg_scores(score_list):
    sample = score_list[0]
    final_score = {key: 0 for key in sample.keys()}

    for score in score_list:
        for key, val in score.items():
            final_score[key] += val

    n = len(score_list)
    final_score = {k:v/n for k,v in final_score.items()}
    return final_score


def get_score(model, sentence_1, sentence_2):
    """
    here, sentence_2 is not necessarily a single sentence, it can be a bunch of sentences
    """
    emb_1 = model.encode(sentence_1,  convert_to_tensor = True)
    emb_2 = model.encode(sentence_2,  convert_to_tensor = True)
    cosine_scores = util.pytorch_cos_sim(emb_1, emb_2)
    return cosine_scores


def similarity(sim_model1, sentence_1, sentence_2, select=True):
    scores = get_score(sim_model1, sentence_1, sentence_2)[0]
    scores = [score.item()*0.5+0.5 for score in scores]
    return scores


def get_category_score(sentence, code, model, tokenizer):
    model_inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to('cuda')
    output = model(**model_inputs).logits[0]
    pred = torch.argmax(output).item()
    return int(pred == code), pred


def get_jaccard_sim(str1, str2):   
    if isinstance(str1, float) or isinstance(str2, float):
        return (-1)
    try:
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        print((str1))
        print(type(str2))
        return 0


def get_novelty(sent,training_corpus):
    max_overlap = 0
    for instance in training_corpus:
        max_overlap = max(max_overlap,get_jaccard_sim(instance,sent))
    return 1-max_overlap


def avg_novelty(sentences,training_corpus):
    avg = 0
    for sent in sentences:
        avg += get_novelty(sent,training_corpus)
    avg = (avg/float(len(sentences)))
    return avg


def get_diversity(sentences):
    avg = 0.0
    for i in range(len(sentences)):
        max_overlap = 0
        for j in range(len(sentences)):
            if i!=j:
                max_overlap = max(max_overlap,get_jaccard_sim(sentences[i],sentences[j]))
        avg = avg + (1-max_overlap)
    avg = (avg/len(sentences))
    return avg
    
    
def diversity_and_novelty(training_corpus, gen_replies):
    diversity = get_diversity(gen_replies)
    novelty   = avg_novelty(gen_replies,training_corpus)
    return diversity, novelty


def compute_metrics(fpath):
    df = pd.read_csv(fpath)
    df['code'] = df['Category'].apply(lambda x: mapper[x])
    
    category_outputs = []
    score_list = []
    bert_scores = []
    sem_scores = []
    preds = []
    trgs = []
    individual_category_scores = {i: [] for i in range(5)}
    pred_classes = []
    actual_classes = []

    for idx, row in df.iterrows():
        hs, cs, code, gen = row['Hate Speech'], row['Counterspeech'], row['code'], row['Generated']
        preds.append(gen)
        trgs.append(cs)
        score_list.append(get_scores(cs, gen))
        category_score, pred_class = get_category_score(gen, code, model, tokenizer)
        individual_category_scores[code].append(category_score)
        pred_classes.append(pred_class); actual_classes.append(code)
        category_outputs.append(category_score)
        sim_score = similarity(sim_model, cs, gen)[0]
        sem_scores.append(sim_score)

        if (idx+1)%50 == 0:
            print(idx+1, 'done!')
            
    bert_scores = bert_score.score(preds, trgs, lang='en')
    diversity, novelty = diversity_and_novelty(training_corpus, preds)
    
    syn_scores = avg_scores(score_list)
    r1, r2, r3, meteor = syn_scores['rouge-1'], syn_scores['rouge-2'], syn_scores['rouge-l'], syn_scores['meteor']
    ca = np.mean(category_outputs)
    sem_sim = np.mean(sem_scores)
    bs = torch.mean(bert_scores[2]).item()
    
    return r1, r2, r3, meteor, sem_sim, bs, ca, diversity, novelty



if __name__ == "__main__":
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("models/cs_category_roberta_model")
    model = AutoModelForSequenceClassification.from_pretrained("models/cs_category_roberta_model").to('cuda')
    
    training_corpus = set(pd.read_csv('train.csv')['Counterspeech'].tolist())
    mapper = {'Facts': 0, 'Question': 1, 'Denouncing': 2, 'Humor': 3, 'Positive': 4}
    rev_mapper = {v:k for k,v in mapper.items()}
    
    results_data = []
    indexes = []
    columns = ['R1', 'R2', 'RL', 'M', 'SS', 'BS', 'CA', 'D', 'N']
    
    
    for file in os.listdir('outputs/'):
        fpath = os.path.join('outputs', file)
        scores = compute_metrics(fpath)
        results_data.append(scores)
        indexes.append(file)
        
        
    result_df = pd.DataFrame(data=results_data, index=indexes, columns=columns)
    result_df.to_csv('results.csv')