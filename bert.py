from bert_embedding import BertEmbedding

bert_abstract = "He went to New York City"
sentences = bert_abstract.split('\n')
bert_embedding = BertEmbedding()
result = bert_embedding(sentences);

wordVectors=result[0][1];

# print(wordVectors);

import spacy
import textacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")


import json

with open('./train-v2.0.json') as squadFile:
  squadQuestions = json.load(squadFile)


q=squadQuestions['data'][0]['paragraphs'][0]['qas'][1]['question'];


# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj","acomp"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}


def sov_triplets(text):
  parsed_text = nlp(text)
  subj=""
  verb=""
  obj=""
  for token in parsed_text:
    if (token.dep_=="xcomp") : verb=token.lemma_
    if (token.dep_=="nsubj") : subj=token.lemma_
    if token.dep_ in OBJECTS: obj=token.lemma_
    print(token,token.dep_,token.lemma_)
  return (subj,verb,obj)


# Process whole documents
text = (q)
parsed_text = nlp(text)


temp=sov_triplets(text)
print(temp)