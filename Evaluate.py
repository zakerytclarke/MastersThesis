"""
Code By Hrituraj Singh
"""

import torch
import torch.nn as nn

# Default word tokens
UNK_token = 0  # Unknown or OOV word
PAD_token = 1  # Used for padding short sentences
SOS_token = 2  # Start of the sequence
EOS_token = 3  # End of the sequence
class FOLEvaluator:
	"""
	FOLEvaluator Class for First Order Logic Parser
	"""
	def __init__(self, vocabs, mode = 'standard', outputMaxLen = 50):
		"""
		Initializer for the Evaluator Class

		[Input]
		mode: selected mode for evaluation, 'standard' or 'graph'. Currently only standard works
		vocabs: list of vocabularies in the order unaryP, binaryP, variable, types
		"""
		self.mode = mode
		self.outputMaxLen = outputMaxLen

		# Loading the vocabularies
		self.vocabUnary = vocabs[0]
		self.vocabBinary = vocabs[1]
		self.vocabVariables = vocabs[2]
		self.vocabtypes = vocabs[3]

		# Keeping track of batches
		self.unaryScores = {'precision':0.0, 'recall':0.0, 'f1':0.0}
		self.binaryScores = {'precision':0.0, 'recall':0.0, 'f1':0.0}
		self.numInstances = 0


	def evaluateBatch(self, outputs, targets):
		"""
		Evaluates all the outputs of a batch against targets and gives the precision and recall scores 
		for both unary and binary predicates
		"""
		numInstances = len(outputs)
		assert numInstances==len(targets)
		self.numInstances += numInstances

		totalUnaryPrecision = 0.0
		totalBinaryPrecision = 0.0
		totalUnaryRecall = 0.0
		totalBinaryRecall = 0.0

		if self.mode == 'standard':
			for index in range(numInstances):
				unaryPrecision, unaryRecall, binaryPrecision, binaryRecall = self.evaluateInstanceStandard(outputs[index], targets[index])
				self.unaryScores['precision'] += unaryPrecision
				self.binaryScores['precision']+= binaryPrecision
				self.unaryScores['recall'] += unaryRecall
				self.binaryScores['recall'] += binaryRecall
				try:
					self.unaryScores['f1'] += (2*unaryPrecision*unaryRecall)/(unaryPrecision+unaryRecall)
				except ZeroDivisionError:
					self.unaryScores['f1'] += 0
				try:
					self.binaryScores['f1'] += (2*binaryPrecision*binaryRecall)/(binaryPrecision+binaryRecall)
				except ZeroDivisionError:
					self.binaryScores['f1'] += 0
		elif self.mode == 'alignment':
			for index in range(numInstances):
				unaryPrecision, unaryRecall, binaryPrecision, binaryRecall = self.evaluateInstanceAlignment(outputs[index], targets[index])
				self.unaryScores['precision'] += unaryPrecision
				self.binaryScores['precision']+= binaryPrecision
				self.unaryScores['recall'] += unaryRecall
				self.binaryScores['recall'] += binaryRecall
				try:
					self.unaryScores['f1'] += (2*unaryPrecision*unaryRecall)/(unaryPrecision+unaryRecall)
				except ZeroDivisionError:
					self.unaryScores['f1'] += 0
				try:
					self.binaryScores['f1'] += (2*binaryPrecision*binaryRecall)/(binaryPrecision+binaryRecall)
				except ZeroDivisionError:
					self.binaryScores['f1'] += 0

		else:
			raise NotImplementedError

	def showScores(self):
		"""
		Show the scores over the whole test set
		"""
		print("=============== Unary Scores ===============")
		print("Precision: ", self.unaryScores['precision']/self.numInstances)
		print("Recall: ", self.unaryScores['recall']/self.numInstances)
		print("F1-Score: ",self.unaryScores['f1']/self.numInstances)
		print("=============== Binary Scores ===============")
		print("Precision: ", self.binaryScores['precision']/self.numInstances)
		print("Recall: ", self.binaryScores['recall']/self.numInstances)
		print("F1-Score: ", self.binaryScores['f1']/self.numInstances)




	def evaluateInstanceStandard(self, output, target):
		"""
		Performs the evaluation over the output of a single Instance and returns
		the precision and recall scores [Standard Mode only]

		[Input]
		output: Output produces for one instance of shape (noTypes, maxSeqLen)
		target: target FOL for the current instance of shape (noTypes, maxSeqLen)

		[Output]
		tuple: a tuple of unaryPrecision, unaryRecall, binaryPrecision, binaryRecall
		"""
		# Getting the outputs
		unary = output[0]
		binary = output[1]
		variables = output[2]
		types = output[3]



		# Getting the targets
		unaryT = target[0]
		binaryT = target[1]
		variablesT = target[2]
		typesT = target[3]

	
		
		unaryVariablesDict = {} # Dictionary which maps unary variables to their corresponding unary predicates
		unaryVariablesInvDict = {} # Dictionary which maps uunary predicates to their corresponding unary variables
		for idx in range(self.outputMaxLen-1):
			if unary[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[types[idx].item()] == 'unaryPredicate':
				unaryVariablesDict[self.vocabVariables.index2word[variables[idx+1].item()]] = unary[idx]
				unaryVariablesInvDict[unary[idx]] = self.vocabVariables.index2word[variables[idx+1].item()]




		unaryVariablesDictT = {} # Dictionary which maps unary variables to their corresponding unary predicates
		unaryVariablesInvDictT = {} # Dictionary which maps uunary predicates to their corresponding unary variables
		for idx in range(self.outputMaxLen-1):
			if unaryT[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[typesT[idx].item()] == 'unaryPredicate':
				unaryVariablesDictT[self.vocabVariables.index2word[variablesT[idx+1].item()]] = unaryT[idx]
				unaryVariablesInvDict[unaryT[idx]] = self.vocabVariables.index2word[variables[idx+1].item()]

		# Getting the output Unaries and Binaries
		unariesOutput = []
		binariesOutput = []

		for idx in range(self.outputMaxLen-2):
			if unary[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[types[idx].item()] == 'unaryPredicate':
				pred = []
				pred.append(unary[idx])
				try:
					pred.append(unaryVariablesDict[self.vocabVariables.index2word[variables[idx+1].item()]])
				except:
					pred.append(torch.append(PAD_token))
				unariesOutput.append(pred)
			if self.vocabtypes.index2word[types[idx].item()] == 'binaryPredicate':
				pred = []
				pred.append(binary[idx])
				try:
					var1 = unaryVariablesDict[self.vocabVariables.index2word[variables[idx+1].item()]]
					var2 = unaryVariablesDict[self.vocabVariables.index2word[variables[idx+2].item()]]
				except:
					var1 = torch.Tensor([PAD_token]).long()
					var2 = torch.Tensor([PAD_token]).long()
				pred.append(var1)
				pred.append(var2)
				binariesOutput.append(pred)

		# Getting the Target Unaries and Binaries
		unariesTarget = []
		binariesTarget = []
		
		for idx in range(self.outputMaxLen-2):
			if unaryT[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[typesT[idx].item()] == 'unaryPredicate':
				pred = []
				pred.append(unaryT[idx])
				try:
					pred.append(unaryVariablesDict[self.vocabVariables.index2word[variablesT[idx+1].item()]])
				except:
					pred.append(torch.Tensor(PAD_token).long())
				unariesTarget.append(pred)
			if self.vocabtypes.index2word[typesT[idx].item()] == 'binaryPredicate':
				pred = []
				pred.append(binaryT[idx])

				try:
					var1 = unaryVariablesDictT[self.vocabVariables.index2word[variablesT[idx+1].item()]]
					var2 = unaryVariablesDictT[self.vocabVariables.index2word[variablesT[idx+2].item()]]
				except:
					var1 = torch.Tensor([PAD_token]).long()
					var2 = torch.Tensor([PAD_token]).long()

				pred.append(var1)
				pred.append(var2)
				binariesTarget.append(pred)



		# Calculating Precision
		unaryPrecision = self.getPrecision(unariesOutput, unariesTarget)
		binaryPrecision = self.getPrecision(binariesOutput, binariesTarget)

		#Calculating Recall
		unaryRecall = self.getRecall(unariesOutput, unariesTarget)
		binaryRecall= self.getRecall(binariesOutput, binariesTarget)

		return (unaryPrecision, unaryRecall, binaryPrecision, binaryRecall)

	def getPrecision(self, output, target):
		"""
		Gets the precision score for given output and target patters
		"""
		matched = 0.0
		total = 0.0

		for oneFunction in output:
			if oneFunction in target:
				matched +=1.0
			total+=1.0
		return matched/total


	def getRecall(self, output, target):
		"""
		Gets the Recall score for given output and target patters
		"""
		matched = 0.0
		total = 0.0

		for oneFunction in target:
			if oneFunction in output:
				matched +=1.0
			total+=1.0
		return matched/total

	def evaluateInstanceAlignment(self, output, target):
		"""
		Performs the evaluation over the output of a single Instance and returns
		the precision and recall scores [Alignment Mode only]

		[Input]
		output: Output produces for one instance of shape (noTypes, maxSeqLen)
		target: target FOL for the current instance of shape (noTypes, maxSeqLen)

		[Output]
		tuple: a tuple of unaryPrecision, unaryRecall, binaryPrecision, binaryRecall
		"""
		# Getting the outputs
		unary = output[0]
		binary = output[1]
		variables = output[2]
		types = output[3]



		# Getting the targets
		unaryT = target[0]
		binaryT = target[1]
		variablesT = target[2]
		typesT = target[3]

	
		# Forming the output dictionary
		unaryVariablesDict = {} # Dictionary which maps unary variables to their corresponding unary predicates
		for idx in range(self.outputMaxLen-1):
			if unary[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[types[idx].item()] == 'unaryPredicate':
				key = self.vocabVariables.index2word[variables[idx+1].item()]
				value = self.vocabUnary.index2word[unary[idx].item()]
				if key not in unaryVariablesDict:
					unaryVariablesDict[key] = [value]
				else:
					unaryVariablesDict[key].append(value)

		# Forming the target dictionary
		unaryVariablesDictT = {} # Dictionary which maps unary variables to their corresponding unary predicates
		for idx in range(self.outputMaxLen-1):
			if unaryT[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[typesT[idx].item()] == 'unaryPredicate':
				key = self.vocabVariables.index2word[variablesT[idx+1].item()]
				value = self.vocabUnary.index2word[unaryT[idx].item()]
				if key not in unaryVariablesDictT:
					unaryVariablesDictT[key] = [value]
				else:
					unaryVariablesDictT[key].append(value)

		# Aligning the dictionaries
		alignmentDictionary = self.alignment(unaryVariablesDict, unaryVariablesDictT)

		# Replacing the variables in output with aligned target variables
		for idx in range(self.outputMaxLen):
			if unary[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[types[idx].item()] == 'variable':
				varOutput = self.vocabVariables.index2word[variables[idx].item()]
				if varOutput in alignmentDictionary: # If at all the variable is aligned
					varTarget = alignmentDictionary[varOutput]
				else:
					varTarget = 'PAD'
				variables[idx] = self.vocabVariables.word2index[varTarget]

		# Replacing those variables with pad in target which have no unary description (may be results of eq relation)
		for idx in range(self.outputMaxLen):
			if unaryT[idx] == EOS_token:
				break

			if self.vocabtypes.index2word[typesT[idx].item()] == 'variables':
				var = self.vocabVariables.index2word[variablesT[idx].item()]
				if var not in alignmentDictionary.values():
					variablesT[idx] = PAD_token


		# Forming Unary and Binaries now
		# Getting the output Unaries and Binaries
		unariesOutput = []
		binariesOutput = []

		for idx in range(self.outputMaxLen-2):
			if unary[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[types[idx].item()] == 'unaryPredicate':
				pred = []
				pred.append(unary[idx])
				pred.append(variables[idx+1])
				unariesOutput.append(pred)
			if self.vocabtypes.index2word[types[idx].item()] == 'binaryPredicate':
				pred = []
				pred.append(binary[idx])
				
				var1 = variables[idx+1]
				var2 = variables[idx+2]
				pred.append(var1)
				pred.append(var2)
				binariesOutput.append(pred)

		# Getting the Target Unaries and Binaries
		unariesTarget = []
		binariesTarget = []

		for idx in range(self.outputMaxLen-2):
			if unaryT[idx] == EOS_token:
				break
			if self.vocabtypes.index2word[typesT[idx].item()] == 'unaryPredicate':
				pred = []
				pred.append(unaryT[idx])
				pred.append(variablesT[idx+1])
				unariesTarget.append(pred)
			if self.vocabtypes.index2word[types[idx].item()] == 'binaryPredicate':
				pred = []
				pred.append(binaryT[idx])
				
				var1 = variablesT[idx+1]
				var2 = variablesT[idx+2]
				pred.append(var1)
				pred.append(var2)
				binariesTarget.append(pred)

		# Calculating Precision
		unaryPrecision = self.getPrecision(unariesOutput, unariesTarget)
		binaryPrecision = self.getPrecision(binariesOutput, binariesTarget)

		#Calculating Recall
		unaryRecall = self.getRecall(unariesOutput, unariesTarget)
		binaryRecall= self.getRecall(binariesOutput, binariesTarget)

		return (unaryPrecision, unaryRecall, binaryPrecision, binaryRecall)


	def alignment(self, outputDict, targetDict):
		"""
		Find the alignment between the variables of output and target 
		Dictionary
		"""

		alignmentDictionary = {}


		for key, value in outputDict.items():
			for keyT, valueT in targetDict.items():
				if self.isEquivalent(value, valueT):
					alignmentDictionary[key] = keyT

		return alignmentDictionary

	def isEquivalent(self, value1, value2):
		"""
		Helper function for checking equivalence while creating an
		alignment dictionary
		"""

		for value in value1:
			if value in value2:
				return True
		return False











