"""
Code By Hrituraj Singh
"""
import operator
from tqdm import tqdm
from utils import cleanText

# Default word tokens
UNK_token = 0  # Unknown or OOV word
PAD_token = 1  # Used for padding short sentences
SOS_token = 2  # Start of the sequence
EOS_token = 3  # End of the sequence
class Vocab:
	"""
	Vocabulary class for mapping words to Indices and vice-versa
	"""
	def __init__(self, maxWords = 30000):
		"""
		Initializer for the vocabulary class

		[Input]
		maxWords: Maximum number of words allowed in the vocabulary
		"""

		self.word2index = {"PAD": PAD_token, "UNK": UNK_token, "<SOS>": SOS_token, "<EOS>": EOS_token}
		self.word2count = {"PAD": 0, "UNK": 0, "<SOS>": 0, "<EOS>": 0}
		self.index2word = {PAD_token: "PAD", UNK_token: "UNK", SOS_token: "<SOS>", EOS_token: "<EOS>"}
		self.numWords = 4  # Count CLS, UNK, SOS, EOS
		self.maxWords  = maxWords


	def addSentence(self, sentence):
		"""
		Takes as input a sentence and adds the words in the sentence to the vocabulary

		[Input]
		sentence : single sentence as a list of tokens
		"""
	
		for word in sentence:
				self.addWord(word)
	




	def addWord(self, word):
		"""
		Adds a word to vocabulary

		[Input]
		word : word to be added
		"""
		if word in ['', ' ', '\t']: # Possible formatting emanating useless words
			return

		if word not in self.word2index:
			self.word2index[word] = self.numWords
			self.word2count[word] = 1
			self.index2word[self.numWords] = word
			self.numWords += 1
		else:
			self.word2count[word] += 1


	def addInstance(self, instance):
		"""
		Adds an instance to vocabulary

		[Input]
		instance: an instance which contains a tuple of list of tokens and mask
		"""
		sentence = instance[0]
		masks = instance[1]

		for index, mask in enumerate(masks):
			if mask==1:
				self.addWord(sentence[index]) # Typecasting to string to accomodate integer words/tokens




	def create(self, instances):
		"""
		Forms the vocabulary from instances. It forms only one vocabulary at a time.
		So either pass instances as the input instances or the target instances

		[Input]
		instances : list of the instances where each instance contains both masking as well as list of tokens
		"""



		for instance in tqdm(instances):
			self.addInstance(instance)

		

		self.trim() #Restrict the vocabulary to max size





	def trim(self):
		"""
		Restricts the vocabulary to max words by deleting the entries
		which have less count than the top "self.maxWords" no of entries
		"""


		if self.numWords <= self.maxWords:
			print("Vocabulary size within the restricted range. Vocab created with " +str(self.numWords) + " unique words!")
			return
		else:

			sorted_ = sorted(self.word2count.items(), key= lambda item : item[1])
			count = 0

			while(count < self.numWords - self.maxWords - 5): #We do not consider 4 words as part of text
				self.word2count.pop(sorted_[count][0])
				count += 1


			#Override the index to word and word to index dictionaries after deletions
			self.index2word = {PAD_token: "PAD", UNK_token: "UNK", SOS_token: "<SOS>", EOS_token: "<EOS>"}
			self.word2index = {"PAD": PAD_token, "UNK": UNK_token, "<SOS>": SOS_token, "<EOS>": EOS_token}


			index = 4
			print("Vocabulary size exceeded restricted range. Trimmed " + str(self.numWords - self.maxWords) + " number of words")

			for word in self.word2count.keys():
				self.word2index[word] = index
				self.index2word[index] = word
				index += 1
			print("Vocab Created with " + str(self.maxWords) +" unique words!")





	def writeToText(self, file):
		"""
		Writes the vocabulary to a file in the following format
		word --tab separated space-- count--tab seperated space-- index

		[Input]
		file : Address of the file where vocab is to be written
		"""
		print("Writing the vocabulary to text")
		vocabFile = open(file, 'w')

		for word, count in self.word2count.items():
			vocabFile.write(word + "\t" + str(count) + "\t" + str(self.word2index[word]) + "\n")



	def readFromText(self, file):
		"""
		Reads the vocabulary from a file written in the following format
		word --tab separated space-- count--tab seperated space-- index

		[Input]
		file : Address of the file where vocab is written
		"""
		print("Reading the vocabulary from text")
		vocabFile = open(file, 'r')
		vocabText = vocabFile.readlines()
		vocabFile.close()
		vocabText = [entry.strip() for entry in vocabText]
		self.numWords = 0

		for wordEntry in vocabText:
			word, count, index = wordEntry.split("\t")
			self.word2count[word] = count
			self.word2index[word] = int(index)
			self.index2word[int(index)] = word
			self.numWords += 1
		print("Finished reading the vocabulary file! Found " + str(self.numWords) + " words!")





















	






