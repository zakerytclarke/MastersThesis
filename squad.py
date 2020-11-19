import json

with open('./datasets/squad/train-v2.0.json') as squadFile:
  squadQuestions = json.load(squadFile)


q=squadQuestions['data'][0]['paragraphs'][0]['qas'][1]['question'];
