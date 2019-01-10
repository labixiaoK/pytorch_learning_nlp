#encoding=utf-8
import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.onnx

import data
import utils
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/wikitext-2', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', help='rnn type')
parser.add_argument('--emsize', type=int, default=200, help='embedding size')
parser.add_argument('--nhid', type=int, default=200, help='rnn hidden size')
parser.add_argument('--nlayers', type=int, default=2, help='layer nums')
parser.add_argument('--lr', type=float, default=20, help='init learning rate')
parser.add_argument('--epochs', type=int, default=40, help='training epochs')
parser.add_argument('--batch_size', type=int, default=20, help='batch sizes')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clip')
parser.add_argument('--bptt', type=int, default=35, help='bptt limit')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--tie', action='store_true', help='tie weights between embeddings and softmax')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='print log each iter interval')
parser.add_argument('--save', type=str, default='model.pt', help='path of save model')
parser.add_argument('--onnx_export', type=str, default='', help='path of export model in onnx format')

args = parser.parse_args()
print('args.tie', args.tie)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	if not args.cuda:
		print('WARNNING: ur have CUDA device, so u probably run with --cuda')


device = torch.device('cuda' if args.cuda else 'cpu')
###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
#print('len corpus:', len(corpus.dictionary))
eval_batch_size = 10
train_data = utils.batchify(corpus.train, args.batch_size)
valid_data = utils.batchify(corpus.valid, eval_batch_size)
test_data = utils.batchify(corpus.test, eval_batch_size)
#print('train data size', train_data.size())

###############################################################################
# Build the model
###############################################################################

n_tks = len(corpus.dictionary)
model = model.RNNModel(args.model, n_tks, args.emsize, args.nhid, args.nlayers, args.dropout, args.tie).to(device)

creterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def train(epoch):
    # Turn on training mode which enables dropout.
	model.train()
	start_time = time.time()
	total_loss = 0
	n_tks = len(corpus.dictionary)
	hidden = model.initHidden(args.batch_size)
	losses = []
	for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
		data, target = utils.getBatch(train_data, i, args.bptt)
		#print(data.size(), target.size())
		# Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
		#注意，只是detach,并没有initHidden,值还会继承下来，评估时候也是，注意一致
		hidden = utils.repackageHidden(hidden)
		model.zero_grad()	
		
		output, hidden = model(data, hidden)
		loss = creterion(output.view(-1, n_tks), target)
		loss.backward()
		total_loss += loss.item()
		
		# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
		nn.utils.clip_grad_norm_(model.parameters(), args.clip)
		for p in model.parameters():
			p.data.add_(-lr, p.grad.data)
			
		if batch % args.log_interval == 0 and batch > 0:
			cur_loss = total_loss / args.log_interval
			losses.append(cur_loss)
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | s/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data)//args.bptt, lr,elapsed/args.log_interval, cur_loss, math.exp(cur_loss)))
			total_loss = 0
			start_time = time.time()
	return losses

#evaluate
def evaluate(source):
    # Turn on evaluation mode which disables dropout.
	model.eval()
	total_loss = 0
	hidden = model.initHidden(eval_batch_size)
	n_tks = len(corpus.dictionary)
	with torch.no_grad():
		for i in range(0, source.size(0) - 1, args.bptt):
			data, target = utils.getBatch(source, i, args.bptt)
			output, hidden = model(data, hidden)
			output_flat = output.view(-1, n_tks)
			total_loss += creterion(output_flat, target).item() * len(data)
			hidden = utils.repackageHidden(hidden)#已经no_grad()了，此步多余?
			
	return total_loss / (source.size(0) - 1)
	
def export_onnx(path, batch_size, seq_len):
	print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
	model.eval()
	dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
	hidden = model.initHidden(batch_size)
	torch.onnx.export(model, (dummy_input, hidden), path)	
	
#loop over epochs
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
	total_elapsed_time = 0
	train_losses = []
	valid_losses = []
	for epoch in range(1, args.epochs+1):
		epoch_start_time = time.time()
		train_losses.extend(train(epoch))
		valid_loss = evaluate(valid_data)
		valid_losses.append(valid_loss)
		elapsed_time = (time.time() - epoch_start_time) / 60
		total_elapsed_time += elapsed_time
		total_need_time = total_elapsed_time / (epoch / args.epochs)
		print('-' * 89)
		print('| end of epoch {:3d} | has elapsed time: {:5.2f}mins | total need time: {:5.2f}mins | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, total_elapsed_time, total_need_time, valid_loss, math.exp(valid_loss)))
		print('-' * 89)
		# Save the model if the validation loss is the best we've seen so far.
		if not best_val_loss or valid_loss < best_val_loss:
			with open(args.save, 'wb') as fo:
				torch.save(model, fo)
			best_val_loss = valid_loss
		else:
			# Anneal the learning rate if no improvement has been seen in the validation dataset.
			lr /= 4.0
	utils.plotLoss(train_losses, valid_losses)
except KeyboardInterrupt:
	print('-' * 89)
	print('exiting from training early')

#load the best saved model
with open(args.save, 'rb') as fin:
	model = torch.load(fin)
	# after load the rnn params are not a continuous chunk of memory
	# this makes them a continuous chunk, and will speed up forward pass
	model.rnn.flatten_parameters()

#run on test data
test_loss = evaluate(test_data)
print('=' * 89)
print('| end of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
	export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)






