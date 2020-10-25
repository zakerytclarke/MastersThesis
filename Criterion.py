"""
Code By Hrituraj Singh
"""
import operator
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F



# Default word tokens
UNK_token = 0  # Unknown or OOV word
PAD_token = 1  # Used for padding short sentences
SOS_token = 2  # Start of the sequence
EOS_token = 3  # End of the sequence


class FOLCriterion(nn.Module):
	"""
	FOLCriterion function used while training
	"""
	def __init__(self):
		"""
		Initializer for the criterion class
		"""

		super(FOLCriterion, self).__init__()

		# Initializing the loss functions
		self.unaryLoss = nn.CrossEntropyLoss(reduction = 'none')
		self.binaryLoss = nn.CrossEntropyLoss(reduction = 'none')
		self.variablesLoss = nn.CrossEntropyLoss(reduction = 'none')
		self.pointersLoss = nn.CrossEntropyLoss(reduction = 'none') 
		self.typesLoss = nn.CrossEntropyLoss(reduction = 'none')
		self.copyDecisionLoss = nn.BCELoss(reduction = 'none')
		self.copyIndicesLoss = nn.CrossEntropyLoss(reduction = 'none')

	def forward(self, output, copy, target, masks, copyDecisions, copyIndices):
		"""
		Calculates the loss and returns the loss scalar

		[Input]
		output: lost of size 6 where each element is an output for unary, binary, etc
		copy: a tuple of copy mechanism outputs
		target: list/Tensor for targets of unary, binary etc
		masks: list/Tensor for masks of unary, binary etc
		copyDecisions: list/Tensor for copying decisions
		copyIndices: list/Tensor for copying Indices
		"""

		# Getting the outputs
		unaryOutput = output[0].squeeze(1)
		binaryOutput = output[1].squeeze(1)
		variablesOutput = output[2].squeeze(1)
		pointersOutput = output[3].squeeze(1)
		typesOutput = output[4].squeeze(1)

		# Getting Copy Parameters
		copyDecisionsOutput = copy[1].squeeze(1)
		copyIndicesOutput = copy[0]


		# Getting the targets
		unaryTarget = target[:, 0]
		binaryTarget = target[:, 1]
		variablesTarget = target[:, 2]
		pointersTarget = target[:, 3]
		typesTarget = target[:, 4]

		# Getting the masks
		unaryMask = masks[:, 0]
		binaryMask = masks[:, 1]
		variablesMask = masks[:, 2]
		pointersMask = masks[:, 3]
		typesMask = masks[:, 4]

		#Printing the different values for verification
		# print("unary output: ", torch.argmax(unaryOutput, dim=-1))
		# print("unary target: ", unaryTarget)
		# print("binary output: ", torch.argmax(binaryOutput, dim=-1))
		# print("binary target: ", binaryTarget)
		# print("variable output: ", torch.argmax(variablesOutput, dim=-1))
		# print("variable target: ", variablesTarget)
		# print("Pointers Output: ", torch.argmax(pointersOutput, dim=-1))
		# print("Pointers Target: ", pointersTarget)
		# print("types output: ", torch.argmax(typesOutput, dim=-1))
		# print("types target: ", typesTarget)
		# print("unary Mask: ", unaryMask)
		# print("binary Mask: ", binaryMask)
		# print("variables Mask: ", variablesMask)
		# print("Pointers Mask: ", pointersMask)
		# print("types Mask", typesMask)
		# print("copy decisions", copyDecisions)
		# print("copy indices", copyIndices)
		# print("---------------------------------------------")
		# exit()
		

		# Getting different loss values
		unaryLoss = unaryMask* self.unaryLoss(unaryOutput, unaryTarget) # Loss on unary predicates
		binaryLoss = binaryMask * self.binaryLoss(binaryOutput, binaryTarget) # Loss on binary predicates
		variablesLoss = variablesMask * self.variablesLoss(variablesOutput, variablesTarget) # Loss on variables
		pointersLoss = pointersMask * self.pointersLoss(pointersOutput, pointersTarget) # Loss on Pointers
		typesLoss = typesMask * self.typesLoss(typesOutput, typesTarget) # Loss on types
		copyDecisionLoss = variablesMask * self.copyDecisionLoss(copyDecisionsOutput, copyDecisions) # Loss on copy decisions
		copyIndicesLoss = copyDecisions * self.copyIndicesLoss(copyIndicesOutput, copyIndices) # Loss on copy Indices


		# Total loss averaged
		totalLoss = (unaryLoss + binaryLoss + variablesLoss + pointersLoss + typesLoss + copyDecisionLoss + copyIndicesLoss).mean()

		return totalLoss