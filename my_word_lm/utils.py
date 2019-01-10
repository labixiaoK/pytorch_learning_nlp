import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def batchify(data, bsz, device='cpu'):
    # Work out how cleanly we can divide the dataset into bsz parts.
	nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
	data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
	data = data.view(bsz, -1).t().contiguous()#if tensor is contiguous in memory in C order
	
	return data.to(device)

def repackageHidden(h):
	"""Wraps hidden states in new Tensors, to detach them from their history."""
	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackageHidden(v) for v in h)

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def getBatch(source, i, bptt_limit):
	sent_len = min(bptt_limit, len(source)-1-i)
	data = source[i : i+sent_len]
	target = source[i+1 : i+1+sent_len].view(-1)
	return data, target

def plotLoss(train_losses, valid_losses):
	plt.figure()
	fig, ax = plt.subplots()
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(range(len(train_losses)), train_losses, label='train_loss')
	plt.plot(range(len(train_losses)//len(valid_losses) - 1, len(train_losses), len(train_losses)//len(valid_losses)), valid_losses, label='valid_loss')
	plt.legend()
	plt.show()
	plt.savefig('losses.png')
	
	



