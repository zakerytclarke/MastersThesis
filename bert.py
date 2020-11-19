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





# Process whole documents
text = (q)
parsed_text = nlp(text)


temp=sov_triplets(text)
print(temp)
