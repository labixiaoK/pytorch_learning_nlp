import os
from io import open
import torch


class Dictionary():
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []
		
	def addWord(self, word):
		if word not in self.word2idx:
			self.word2idx[word] = len(self.word2idx)
			self.idx2word.append(word)
		return self.word2idx[word]
		
	def __len__(self):
		return len(self.word2idx)
		
class Corpus():
	def __init__(self, path):
		self.dictionary = Dictionary()
		self.train = self.tokenize(os.path.join(path, 'train.txt'))
		self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
		self.test = self.tokenize(os.path.join(path, 'test.txt'))
		
	def tokenize(self, path):
		"""Tokenizes a text file."""
		with open(path, 'r', encoding='utf-8') as fin:
			n_tokens = 0
			for line in fin:
				tokens = line.strip().split() + ['<eos>']
				n_tokens += len(tokens)
				for tk in tokens:
					self.dictionary.addWord(tk)
					
		# Tokenize file content
		with open(path, 'r', encoding='utf-8') as fin:
			ids = torch.LongTensor(n_tokens)
			tk_id = 0
			for line in fin:
				tokens = line.strip().split() + ['<eos>']
				for tk in tokens:
					ids[tk_id] = self.dictionary.word2idx[tk]
					tk_id += 1
		
		return ids
				
				
			
			
		
		