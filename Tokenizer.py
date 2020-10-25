"""
Code By Hrituraj Singh
"""
from tqdm import tqdm
from torch.autograd import Variable
import torch
# Default word tokens
UNK_token = 0  # Unknown or OOV word
PAD_token = 1  # Used for padding short sentences
SOS_token = 2  # Start of the sequence
EOS_token = 3  # End of the sequence


class Tokenizer:
	"""
	Tokenizer class to convert vocabulary words to numeric indices/tokens
	"""

	def __init__(self, vocab, maxSeqLen = 30, mode='input', vocabType = 'non-types'):
		"""
		Initializer for the tokenizer class

		[Input]
		vocab: Object of Vocab class to be used a vocabulary while tokenizing
		maxSeqLen: Maximum length of the sequence Trims the long, pads the short
		mode: whether the tokenizer is for input tokens or output target tokens
		"""

		self.vocab = vocab
		self.maxSeqLen = maxSeqLen
		self.mode =mode
		self.vocabType = vocabType



	def sent2tokens(self, sentence):
		"""
		Converts a raw text sentence to tokenized version

		[Input]
		sentence: A list of tokens in the sentence

		[Output]
		tokSentence: Single sentence in tokenized form
		"""

		tokSentence = []
		for word in sentence:
			if word in self.vocab.word2index:
				tokSentence.append(self.vocab.word2index[word])
			else:
				tokSentence.append(UNK_token)

	
		return tokSentence


	def tokens2sent(self, tokSentence):
		"""
		Converts a tokenized sentence to raw text

		[Input]
		tokSentence: Single sentence in tokenized form

		[Output]
		sentence: Single sentence in raw text form
		"""
		sentence = ""
		
		#Finding where sentence ends so that padding can be accordingly removed
		try:
			lastIndex = self.unPad(tokSentence)
		except:
			print(tokSentence)
			exit()
		for index in tokSentence[:lastIndex+1]:
			sentence += self.vocab.index2word[index.item()] + " "

		return sentence


	def tokenize(self, instances):
		"""
		Tokenizes the dataset and returns the tokenized version in appropriate format

		[Input]
		instances: Text dataset as a list where every element in a list is an instance

		[Output]
		tokens: Tokenized form of the sentences in a list
		mode: Mode selects what is being tokenized: input or output (unary, binary, varunary, var1binary, var2binary)

		"""
		tokens = []
		teachertokens = [] # For Teacher Forcing. Gets populated only in output mode

		if self.mode == 'input': # While tokenizing input sentences
			for instance in instances:
				tokens.append(self.trimPad(self.sent2tokens(instance)))

		else: # While tokenizing output, predicates etc
			for instance in instances:
				tokens_ = self.sent2tokens(instance)

				tokens.append(self.trimPad(tokens_ + [EOS_token]))
				teachertokens.append(self.trimPad([SOS_token] + tokens_, mode='teacher'))

		tensorizedTokens = []
		tensorizedTeacherTokens = []
		for token in tokens:
			tensorizedTokens.append(Variable(torch.Tensor(token)))
		for token in teachertokens:
			tensorizedTeacherTokens.append(Variable(torch.Tensor(token)))


		return tensorizedTeacherTokens, tensorizedTokens


	def detokenize(self, tokTextData):
		"""
		Detokenizes the dataset and returns the detokenized version in appropriate format.
		Uses the target vocabulary for performing detokenization

		[Input]
		tokTextData: Tokenized text dataset as a list of tokenized sentences

		[Output]
		textData: Text dataset as a list of sentences
		"""

		textData = []


		for tokSentence in tokTextData:
			textSentence = self.tokens2sent(tokSentence)
			textData.append(textSentence)

		return textData


	def unPad(self, sentence):
		"""
		Finds the pad in the sentence to remove the unnessary clutter so as to make it print ready

		[Input]
		sentence: padded sentence in raw text form
		"""
		if self.mode == 'input':
			length = len(sentence) - 1
		else:
			length = len(sentence) - 2 # Since last will be EOS anyway

		while(length >= 0):
			if sentence[length] != PAD_token:
				return length
			length = length - 1

		return length




	def trimPad(self, sentence, mode='non-teacher'):
		"""
		Trims the sentence to the maxSeqLen if it is larger and pads it if it is shorter

		[Input] 
		sentence: Single sentence in tokenized form
		mode: if teacher forcing mode, don't append EOS_token

		[Output]
		tpSentence: trim padded sentence in tokenized form
		"""
		tpsentence = []
	
		if len(sentence) >= self.maxSeqLen:
			if mode == 'teacher':
				tpsentence = sentence[:self.maxSeqLen]
				return tpsentence 
			if self.vocabType =='types':
				tpsentence = sentence[:self.maxSeqLen-1] + [self.vocab.word2index['unaryPredicate']]
			else:
				tpsentence = sentence[:self.maxSeqLen-1] + [EOS_token]

		elif len(sentence) < self.maxSeqLen:
			pads = [PAD_token] *(self.maxSeqLen-len(sentence))
			tpsentence = sentence + pads

		return tpsentence


