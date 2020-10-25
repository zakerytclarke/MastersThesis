"""
Code By Hrituraj Singh
"""
from tqdm import tqdm
import torch
import torch.utils.data




class FOLDataset(torch.utils.data.Dataset):
	"""
	Dataset Class for the First Order Logic Dataset
	"""
	def __init__(self, instances):
		"""
		Initializer for the Dataset

		[Input]
		maxSeqlenOut: Maximum length of the output sequence. Should be passed
		instances: list of objects of class instance as declared in preprocess.py
		"""

		super(FOLDataset, self).__init__()
		self.instances = self.prepare(instances)

	def __len__(self):
		"""
		Returns the length of the dataset
		"""
		return len(self.instances)

	def __getitem__(self, index):
		"""
		Returns the indexed element
		"""
		return self.instances[index]

	def prepare(self, instances):
		"""
		Since the Dataset can only have dicts/lists etc and not something like Instance, make the instances compatible
		"""
		newInstances = []

		for instance in instances:
			newInstance = {'input': instance.InputSentence, 'decoderInput': instance.DecoderInput, 'masks': instance.Masks, 'target': instance.Target, 'copyDecisions': instance.CopyDecisions, 'copyIndices':instance.copyIndices}
			newInstances.append(newInstance)

		return newInstances






