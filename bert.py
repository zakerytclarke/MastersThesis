from bert_embedding import BertEmbedding

bert_abstract = "He went to New York City"
sentences = bert_abstract.split('\n')
bert_embedding = BertEmbedding()
result = bert_embedding(sentences);

wordVectors=result[0][1];

print(wordVectors);