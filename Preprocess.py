"""
Code By Hrituraj Singh
"""
import os
import random
import re
from glob import glob
from utils import cleanText, parsePredicates		
from tqdm import tqdm
import json
import torch
import numpy as np

pad = 'PAD' # Global variable to keep padding token handy. Keep it same ideally as in vocab and tokenizer
eos = '<EOS>' # Global variable for EOS Token
class Instance(object):
	"""
	An instance
	"""
	def __init__(self, inp, target, maxSeqlenOut):
		self.inputSentence = inp
		self.target = target
		self.maxSeqlenOut = maxSeqlenOut
		
		# Cleaning and analyzing the instance
		self.types = []
		self.clean()
		self.introduceAdversary()

		# Data
		self.binaryPredicates = []
		self.unaryPredicates = []
		self.variables = []
		self.pointers = []

		# Masks
		self.binaryPredicatesMask = torch.zeros(maxSeqlenOut)
		self.unaryPredicatesMask = torch.zeros(maxSeqlenOut)
		self.variablesMask = torch.zeros(maxSeqlenOut)
		self.pointersMask = torch.zeros(maxSeqlenOut)
		self.typesMask = torch.zeros(maxSeqlenOut)

		# Copy Mechanism
		self.copyDecisions = torch.zeros(maxSeqlenOut)
		self.copyIndices = torch.zeros(maxSeqlenOut)


		self.create()
        
		if len(self.unaryPredicates) >= maxSeqlenOut:
			self.fix() # Making sure that the last mask is 1 only for unaryPredicates as it is what contains EOS



	def __print__(self):
		print("Input: ", self.inputSentence)
		print("Target: ", self.target)
		print("Types: ", self.types)
		print("Unary Predicates: ", self.unaryPredicates)
		print("Binary Predicates: ", self.binaryPredicates)
		print("Variables: ", self.variables)
		print("Pointers: ", self.pointers)
		print("Unary Predicates Mask: ", self.unaryPredicatesMask)
		print("Binary Predicates Mask: ", self.binaryPredicatesMask)
		print("Variables Mask: ", self.variablesMask)
		print("Pointers Mask: ", self.pointersMask)
		print("Copy Decisions: ", self.copyDecisions)
		print("copyIndices: ", self.copyIndices)

	def clean(self, lower = True):
		self.inputSentence = cleanText(self.inputSentence, lower)
		self.inputSentence = [x.strip() for x in self.inputSentence.split(' ')]
		self.inputSentenceMask = torch.ones(len(self.inputSentence)) # Creating only to make it compatible with vocabulary creation

		newTarget = []

		for predicate in self.target:

			# Check what type of predicate it is
			if len(predicate) == 3: # It is a unary or binary thing
				predicateFun  = predicate[1]
				predicateVar  = predicate[2]
				predicateVar  = predicateVar.replace(u'(', '')
				predicateVar  = predicateVar.replace(u')', '')
				predicateVars = predicateVar.split(',')

				if len(predicateVars) == 1:
					newTarget.append(predicateFun) 
					self.types.append('unaryPredicate') # Unary Predicate
					newTarget.append(predicateVars[0])
					self.types.append('variable') # Argument of Unary Predicate
				if len(predicateVars) == 2:
					newTarget.append(predicateFun) 
					self.types.append('binaryPredicate') # Binary Predicates
					newTarget.append(predicateVars[0])
					self.types.append('variable') # Argument 1 of Binary Predicate
					newTarget.append(predicateVars[1]) 
					self.types.append('variable') # Argument 2 of Binary Predicate
			else: # Or is it a pointer
				if predicate == ')':
					newTarget.append(')')
					self.types.append('pointer')
				else:
					newTarget.append(predicate)
					self.types.append('pointer')

		self.types.append('unaryPredicate') # This is for EOS Token. We assume EOS token is part of Unary Dictionary
		newTarget.append(eos) # Appending the EOS token
		self.target = newTarget

	def introduceAdversary(self):
		"""
		Introducing Adversaries to check the performance of models
		"""
		varNameMapping = []

		# Collecting all variables
		for index, tokenType in enumerate(self.types):
			if index>=self.maxSeqlenOut:
				break # We don't want to create sequences longer than max seq len, So trimming them
			if tokenType == 'variable':
				if self.target[index] not in varNameMapping:
					varNameMapping.append(self.target[index])

		# Randomize the variables
		newVarNameMapping = np.random.permutation(varNameMapping)
		var2varMapping = {}
		for varA, varB in zip(varNameMapping, newVarNameMapping):
			var2varMapping[varA] = varB


		# Randomizing the variables
		for index, tokenType in enumerate(self.types):
			if index>=self.maxSeqlenOut:
				break # We don't want to create sequences longer than max seq len, So trimming them
			if tokenType == 'variable':
				self.target[index] = var2varMapping[self.target[index]]
	def fix(self):
		self.binaryPredicatesMask[-1] = 0
		self.unaryPredicatesMask[-1] = 1
		self.variablesMask[-1] = 0
		self.pointersMask[-1] = 0
		self.typesMask[-1] = 1


	def create(self):
		"""
		Creates the full fetched instance which has masks, binary, unary etc
		"""
		varNameIndexMapping = {}
		for index, tokenType in enumerate(self.types):
			if index>=self.maxSeqlenOut:
				break # We don't want to create sequences longer than max seq len, So trimming them
			if tokenType == 'unaryPredicate': # Unary Predicate
				# Adding the values to unary predicate
				self.unaryPredicates.append(self.target[index])
				self.unaryPredicatesMask[index] = 1
				self.typesMask[index] = 1

				# Masking and adding PAD to others. We can actually add
				# Anything instead of PAD as well. It won't matter because
				# It will have no contribution whatever in deciding the network weights
				self.binaryPredicates.append(pad)
				self.variables.append(pad)
				self.pointers.append(pad)

			elif tokenType == 'binaryPredicate': # Binary Predicate
				# Adding the values to Binary predicate
				self.binaryPredicates.append(self.target[index])
				self.binaryPredicatesMask[index] = 1
				self.typesMask[index] = 1

				# Masking and adding PAD to others. We can actually add
				# Anything instead of PAD as well. It won't matter because
				# It will have no contribution whatever in deciding the network
				self.unaryPredicates.append(pad)
				self.variables.append(pad)
				self.pointers.append(pad)

			elif tokenType == 'variable': # Argument of unary predicate
				# Adding the values to variable of unary predicate
				self.variables.append(self.target[index])
				self.variablesMask[index] = 1
				self.typesMask[index] = 1
				if self.target[index] in varNameIndexMapping:
					self.copyDecisions[index] = 1
					self.copyIndices[index] = varNameIndexMapping[self.target[index]]
				else:
					varNameIndexMapping[self.target[index]] = index
				# Masking and adding PAD to others. We can actually add
				# Anything instead of PAD as well. It won't matter because
				# It will have no contribution whatever in deciding the network
				self.unaryPredicates.append(pad)
				self.binaryPredicates.append(pad)
				self.pointers.append(pad)
			elif tokenType == 'pointer': # Pointer beginning or end
				self.pointers.append(self.target[index])
				self.pointersMask[index] = 1
				self.typesMask[index] = 1

				# Masking and adding PAD to others. We can actually add
				# Anything instead of PAD as well. It won't matter because
				# It will have no contribution whatever in deciding the network
				self.unaryPredicates.append(pad)
				self.binaryPredicates.append(pad)
				self.variables.append(pad)


	def isEqualto(self, anotherInstance):
		"""
		Compares the Instance to another Instance
		"""
		sentence = anotherInstance.inputSentence
		if sentence == self.inputSentence:
			return True
		else:
			return False





class Preprocess:
	"""
	Preprocess class to perform cleaning and aggregation of dataset
	"""
	def __init__(self, filespath, clean = True, seed = 0, maxSeqlenOut = 30):
		"""
		Initializer for the preprocessor class

		[Input]
		clean: Whether to clean the text data
		filespath: path where all the .txt files are stored
		"""

		self.clean = clean
		self.sep = True
		self.rand = random.Random(seed)
		self.filespath = filespath
		self.maxSeqlenOut = maxSeqlenOut


	def makeInstances(self, string):
		"""
		Converts a string in the .txt input file to instance. Please note that this is very
		specific function working only for the files give in the dataset format
		"""
		dictionary = json.loads(string)
		instances = []


		# Getting the instances
		instance = Instance(dictionary['premise'], parsePredicates(dictionary['premiseFol']), self.maxSeqlenOut)
		instances.append(instance)
		
		# Getting the instances
		instance = Instance(dictionary['hypothesis'], parsePredicates(dictionary['hypothesisFol']), self.maxSeqlenOut)
		instances.append(instance)


		return instances







	def perform(self):
		"""
		Performs the whole preprocessing part to produce the dataset to be used in tokenizer
		It needs to do N number of things depending on how you want to load data


		[Output]
		textData: Text dataset as a list where every element in a list is an instance
		"""

		files = glob(self.filespath)


		instances = []
		for file in files:
			textFile = open(file, 'r').readlines()
			textFile = [x.strip() for x in textFile]


			for sentenceInstance in tqdm(textFile, ncols = 100, dynamic_ncols=False, desc = 'Loading ' + file.split('/')[-1]):
				try:
					instance_ = self.makeInstances(sentenceInstance)
				except:
					continue

				if len(instance_) == 0: # Both complex sentence
					#Do Nothing
					continue
				elif len(instance_) == 1:
					Duplicate = False

					# Eliminating Duplications
					for instance in instances:
						if instance.isEqualto(instance_[0]):
							Duplicate = True

					if not Duplicate:
						instances.append(instance_[0])


				elif len(instance_) == 2:

					# Eliminating Duplications
					Duplicate = False
					for instance in instances:
						if instance.isEqualto(instance_[0]):
							Duplicate = True

					if not Duplicate:
						instances.append(instance_[0])

					# Eliminating Duplications
					Duplicate = False
					for instance in instances:
						if instance.isEqualto(instance_[1]):
							Duplicate = True

					if not Duplicate:
						instances.append(instance_[1])

				

		print("No of instances Found: ",len(instances))
		return instances