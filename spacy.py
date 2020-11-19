import spacy
import textacy

nlp = spacy.load("en_core_web_sm")

## Parser to convert questions to Subject, Verb, Object triplets
# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl", "subj"}
# dependency markers for objects
OBJECTS = {"dobj","acomp","iobj"}
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
    if (token.dep_=="nsubj") : subj=token.lemma_
    if (token.dep_=="xcomp") : verb=token.lemma_
    if token.dep_ in OBJECTS: obj=token.lemma_
    print(token,token.dep_,token.lemma_)
  return (subj,verb,obj)
