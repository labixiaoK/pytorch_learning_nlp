import torch
import torch.nn as nn

class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""
	def __init__(self, rnn_type, n_tks, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
		super(RNNModel, self).__init__()
		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers
		self.tie_weights = tie_weights
		self.dropout = nn.Dropout(dropout)
		self.encoder = nn.Embedding(n_tks, ninp)
		
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
				
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		
		self.decoder = nn.Linear(nhid, n_tks)
		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016) https://arxiv.org/abs/1611.01462
		#Linear weight shape(out_features,in_features)!!!所以直接赋值，不用转置
		if tie_weights:
			if ninp != nhid:
				raise ValueError("if tie_weights is true, need embed_size==hidden_size")
			self.decoder.weight = self.encoder.weight#reference, self.decoder.weight is self.encoder.weight
			

		#print('encoder:', self.encoder)
		#print('decoder:', self.decoder)

		self.initWeights()
	
	def initWeights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange) #if tie_weight, will be reassign by decoder
		self.decoder.bias.data.zero_()
		if not self.tie_weights:
			self.decoder.weight.data.uniform_(-initrange, initrange)
		#print('encoder wei:', self.encoder.weight)
		#print('decoder wei:', self.decoder.weight)
		#print('encoder wei:', self.encoder.weight.size())
		#print('decoder wei:', self.decoder.weight.size())
	
	def forward(self, input_, hidden):
		#print('input size', input_.size())
		embed = self.dropout(self.encoder(input_))
		output, hidden = self.rnn(embed, hidden)
		output = self.dropout(output)
		#output:shape (seq_len, batch, num_directions * hidden_size):tensor containing
		#the output features (h_t) from the last layer of the LSTM, for each t.
		#hidden:h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing 
		#the hidden state for t = seq_len.
		decoder_out = self.decoder(output.view(output.size(0) * output.size(1), -1))
		#print('decoder output size:', decoder_out.size())
		return decoder_out.view(output.size(0), output.size(1), decoder_out.size(1)), hidden
	
	def initHidden(self, bsz):
		#parameter of self.encoder, dropout has not paras.
		weight = next(self.parameters())
		#Returns a Tensor of size size filled with 0. By default, the returned Tensor has 
		#the same torch.dtype and torch.device as this tensor.
		if self.rnn_type == 'LSTM':
			return weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid)
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)		
		
		
	
	
	
	
	
	
	
	
	
	
	
	