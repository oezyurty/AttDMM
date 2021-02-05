import argparse
import time
import os
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from pyro.infer import Predictive

from util import Emitter, GatedTransition, Combiner, Predicter_Attention
from util import batchify, reverse_sequences, get_mini_batch, pad_and_reverse

import logging
import json
from collections import namedtuple

#You may comment out this later on
pyro.set_rng_seed(1234)

def get_logger(log_file):
	logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	logging.getLogger('').addHandler(console)

	def log(s):
		logging.info(s)

	return log

class AttDMM(nn.Module):

	def __init__(self, input_dim=12, z_dim=8, static_dim=5, min_x_scale=0.2, emission_dim=16,
				 transition_dim=32, linear_gain=False, att_dim=16, MLP_dims="12-3", guide_GRU=False, rnn_dim=16, rnn_num_layers=1, rnn_dropout_rate=0.0,
				 num_iafs=0, iaf_dim=50, use_feature_mask=False, use_cuda=False):
		super().__init__()

		#Keep the flag of use_feature_mask
		self.use_feature_mask = use_feature_mask
		# instantiate PyTorch modules used in the model and guide below
		# For now, we limit that emitter will never use feature mask
		self.emitter = Emitter(input_dim, z_dim, emission_dim, False, min_x_scale)
		self.trans = GatedTransition(z_dim, static_dim, transition_dim)
		self.combiner = Combiner(z_dim, static_dim, rnn_dim)
		self.predicter = Predicter_Attention(z_dim, att_dim, MLP_dims, batch_first=True, use_cuda=use_cuda)
		#Save if the linear gain will be used (important for model)
		self.linear_gain = linear_gain
		# dropout just takes effect on inner layers of rnn
		rnn_dropout_rate = 0. if rnn_num_layers == 1 else rnn_dropout_rate
		if not guide_GRU:
			self.rnn = nn.RNN(input_size=input_dim + use_feature_mask*input_dim, hidden_size=rnn_dim, nonlinearity='relu',
								batch_first=True, bidirectional=False, num_layers=rnn_num_layers,
								dropout=rnn_dropout_rate)
		else:
			self.rnn = nn.GRU(input_size=input_dim + use_feature_mask*input_dim, hidden_size=rnn_dim, 
								batch_first=True, bidirectional=False, num_layers=rnn_num_layers,
								dropout=rnn_dropout_rate)

		# if we're using normalizing flows, instantiate those too
		self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
		self.iafs_modules = nn.ModuleList(self.iafs)

		# define a (trainable) parameters z_0 and z_q_0 that help define the probability
		# distributions p(z_1) and q(z_1)
		# (since for t = 1 there are no previous latents to condition on)
		self.z_0 = nn.Parameter(torch.zeros(z_dim))
		self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
		# define a (trainable) parameter for the initial hidden state of the rnn
		self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

		self.use_cuda = use_cuda
		# if on gpu cuda-ize all PyTorch (sub)modules
		if use_cuda:
			self.cuda()

	# HERE WE DEFINE model AND guide functions

	def model(self, mini_batch_static, mini_batch, mini_batch_reversed, mini_batch_mask,
				mini_batch_seq_lengths, mini_batch_feature_mask=None, y_mini_batch=None, y_mask_mini_batch=None, annealing_factor=1.0, regularizer=1.0, renormalize_y_weight=False):

		# this is the number of time steps we need to process in the mini-batch
		T_max = mini_batch.size(1)

		# register all PyTorch (sub)modules with pyro
		# this needs to happen in both the model and guide
		pyro.module("attdmm", self)

		# set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
		z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

		#Inıtialize the empty list to keep hidden states of every time step
		all_hidden_states = []

		# we enclose all the sample statements in the model in a plate.
		# this marks that each datapoint is conditionally independent of the others
		#Yilmazcan Update: Changed dim to -2 (it was not specified before!)
		for t in pyro.markov(range(1, T_max + 1)):
			# sample the latents z and observed x's one time step at a time
			# we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
			with pyro.plate("z_minibatch_"+str(t), len(mini_batch), dim=-1):
				# the next chunk of code samples z_t ~ p(z_t | z_{t-1})
				# note that (both here and elsewhere) we use poutine.scale to take care
				# of KL annealing. we use the mask() method to deal with raggedness
				# in the observed data (i.e. different sequences in the mini-batch
				# have different lengths)

				# first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
				z_loc, z_scale = self.trans(z_prev, mini_batch_static)

				# then sample z_t according to dist.Normal(z_loc, z_scale)
				# note that we use the reshape method so that the univariate Normal distribution
				# is treated as a multivariate Normal distribution with a diagonal covariance.
				with poutine.scale(scale=annealing_factor*regularizer/T_max):
					z_t = pyro.sample("z_%d" % t,
										dist.Normal(z_loc, z_scale)
											.mask(mini_batch_mask[:, t - 1:t])
											.to_event(1))

			# compute the probabilities that parameterize the bernoulli likelihood
			# For now, we limit that emitter will never use feature mask
			# Previously, if not self.use_feature_mask:
			if True:
				x_loc, x_scale  = self.emitter(z_t)
			else:
				x_loc, x_scale  = self.emitter(z_t, mini_batch_feature_mask[:,t-1,:])

				# the next statement instructs pyro to observe x_t according to the
				# bernoulli distribution p(x_t|z_t)
				'''
				pyro.sample("obs_x_%d" % t,
								dist.OneHotCategorical(emission_probs_t)
									.mask(mini_batch_mask[:, t - 1])
									.to_event(1),
								obs=mini_batch[:, t - 1, :])
				'''
			with pyro.plate("x_minibatch_"+str(t), len(mini_batch), dim=-2):
				with pyro.plate("x_feature_"+str(t), mini_batch.shape[2], dim=-1):
					poutine_mask = torch.ones((mini_batch.shape[0], mini_batch.shape[2])).cuda() == 1 if mini_batch_feature_mask is None else mini_batch_feature_mask[:,t-1,:] == 1
					with poutine.mask(mask=poutine_mask):
						#If there is mask, only the real obs. will be counted for log prob.
						#To make scaling correctly, we count the number of observations per datum via mask.
						#At the end, the total weight of all observations will be => num_data_points * regularizer
						num_obs_per_datum = mini_batch.shape[2]*T_max if mini_batch_feature_mask is None else mini_batch_feature_mask.sum().item()/mini_batch_feature_mask.shape[0]
						with poutine.scale(scale=regularizer/num_obs_per_datum):
							pyro.sample("obs_x_%d" % t,
											dist.Normal(x_loc, x_scale)
												.mask(mini_batch_mask[:, t - 1].unsqueeze(-1).expand(x_loc.shape[0], x_loc.shape[1])),
											obs=mini_batch[:, t - 1, :])
				
				
			#If linear gain is used, then prediction is done at every time step with linear weighting (i.e. y_1 to y_24)
			#For batch_size being M, sum of weights of all y's will be equal to M again to assure proportional weights between reconstruction and prediction losses
			if self.linear_gain:
				#The last time step's z_t, which is z_prev, will be used to predict the mortality label
				mortality_probs = self.predicter(z_t)
				#Adjust the scaling weights for each class such that 
				#Total weight of y's will be => num_data_points
				if y_mini_batch is not None:
					weights = torch.zeros_like(y_mini_batch)
					weights[y_mini_batch == 0] = 0.56
					weights[y_mini_batch == 1] = 5
				else:
					weights = 1
				#Calculate multiplier for linear gain
				multiplier_lin_gain = t / (T_max * (T_max + 1) / 2)
				weights *= multiplier_lin_gain
				with pyro.plate("y_minibatch_"+str(t), len(mini_batch)):
					poutine_y_mask = torch.ones((len(mini_batch))).cuda() == 1 if y_mask_mini_batch is None else y_mask_mini_batch == 1
					#Re-normalize weights to cover missing y values
					if y_mask_mini_batch is not None and renormalize_y_weight:
						weights = weights/y_mask_mini_batch.mean()
					with poutine.mask(mask=poutine_y_mask): 
						with poutine.scale(scale=weights):
							pyro.sample("y_%d" % t, dist.Bernoulli(mortality_probs), obs=y_mini_batch)

			# the latent sampled at this time step will be conditioned upon
			# in the next time step so keep track of it
			z_prev = z_t

			#Add hidden states at time t to all_hidden_states
			all_hidden_states.append(z_prev)

		#The predicter (Attention+MLP) will use all the hidden states {z_t}'s to make prediction
		all_hidden_states = torch.stack(all_hidden_states).transpose(0,1)


		#If linear gain is not used, then prediction is done only at the last time step
		if not self.linear_gain:
			#The last time step's z_t, which is z_prev, will be used to predict the mortality label
			mortality_probs = self.predicter(all_hidden_states, mini_batch_mask)
			#Adjust the scaling weights for each class such that 
			#Total weight of y's will be => num_data_points
			if y_mini_batch is not None:
				weights = torch.zeros_like(y_mini_batch)
				weights[y_mini_batch == 0] = 0.56
				weights[y_mini_batch == 1] = 5
			else:
				weights = 1
			with pyro.plate("y_minibatch", len(mini_batch)):
				poutine_y_mask = torch.ones((len(mini_batch))).cuda() == 1 if y_mask_mini_batch is None else y_mask_mini_batch == 1
				#Re-normalize weights to cover missing y values
				if y_mask_mini_batch is not None and renormalize_y_weight:
					weights = weights/y_mask_mini_batch.mean()
				with poutine.mask(mask=poutine_y_mask): 
					with poutine.scale(scale=weights):
						#DEBUG
						'''
						if len(mini_batch) == 191:
							#Just show the hidden states of 12-16th data points 
							print("------ Here is z_24 of 12-16h data points: ------")
							print(z_prev[12:17])
							print("------ Here is y_mini_batch: ------")
							print(y_mini_batch[12:17])
							print("------ Here is the mortality probs: ------")
							print(mortality_probs[12:17])
							print("------ Here is the weights: ------")
							print(weights[12:17])
							print("------ Here is the poutine_y_mask: ------")
							print(poutine_y_mask[12:17])
						'''
						pyro.sample("y", dist.Bernoulli(mortality_probs), obs=y_mini_batch)

	# the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
	def guide(self, mini_batch_static, mini_batch, mini_batch_reversed, mini_batch_mask,
				mini_batch_seq_lengths, mini_batch_feature_mask=None, y_mini_batch=None, y_mask_mini_batch=None, annealing_factor=1.0, regularizer=1.0, renormalize_y_weight=False):
		
		# this is the number of time steps we need to process in the mini-batch
		T_max = mini_batch.size(1)
		# register all PyTorch (sub)modules with pyro
		pyro.module("attdmm", self)

		# if on gpu we need the fully broadcast view of the rnn initial state
		# to be in contiguous gpu memory
		h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
		#h_0_contig = h_0_contig.type('torch.DoubleTensor')
		# push the observed x's through the rnn;
		# rnn_output contains the hidden state at each time step
		rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
		# reverse the time-ordering in the hidden state and un-pack it
		rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths)
		# set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
		z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

		# we enclose all the sample statements in the guide in a plate.
		# this marks that each datapoint is conditionally independent of the others.
		with pyro.plate("z_minibatch", len(mini_batch), dim=-1):
			# sample the latents z one time step at a time
			# we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
			for t in pyro.markov(range(1, T_max + 1)):
				# the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
				z_loc, z_scale = self.combiner(z_prev, mini_batch_static, rnn_output[:, t - 1, :])

				# if we are using normalizing flows, we apply the sequence of transformations
				# parameterized by self.iafs to the base distribution defined in the previous line
				# to yield a transformed distribution that we use for q(z_t|...)
				if len(self.iafs) > 0:
					z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
					assert z_dist.event_shape == (self.z_q_0.size(0),)
					assert z_dist.batch_shape[-1:] == (len(mini_batch),)
				else:
					z_dist = dist.Normal(z_loc, z_scale)
					assert z_dist.event_shape == ()
					assert z_dist.batch_shape[-2:] == (len(mini_batch), self.z_q_0.size(0))

				# sample z_t from the distribution z_dist
				with pyro.poutine.scale(scale=annealing_factor*regularizer/T_max):
					if len(self.iafs) > 0:
						# in output of normalizing flow, all dimensions are correlated (event shape is not empty)
						z_t = pyro.sample("z_%d" % t,
											z_dist.mask(mini_batch_mask[:, t - 1]))
					else:
						# when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
						z_t = pyro.sample("z_%d" % t,
											z_dist.mask(mini_batch_mask[:, t - 1:t])
											.to_event(1))
				# the latent sampled at this time step will be conditioned upon in the next time step
				# so keep track of it
				z_prev = z_t


def main(args):

	# setup logging
	if not args.eval_mode:
		log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, args.log))
	else:
		log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, "eval_"+args.log))
	log(args)

	#log(parser.get_default('z_dim'))

	#If in training mode, save the argparse arguments for eval mode to be used later on
	if not args.eval_mode:
		if args.load_model == '':
			with open(os.path.join(args.experiments_main_folder, args.experiment_folder,'commandline_args.txt'), 'w') as f:
				json.dump(args.__dict__, f, indent=2)
		else:
			pretrained_model_path = '/'.join(args.load_model.split('/')[:-1])
			with open(os.path.join(args.experiments_main_folder, args.experiment_folder, pretrained_model_path, 'commandline_args.txt'), 'r') as f:
				prev_args_dict_ = json.load(f)
			curr_args = {}
			for attr, value in prev_args_dict_.items():
				curr_args[attr] = value
			# filter_label will be forced to be False for our current setting (It can be overwritten by current args)
			curr_args['filter_label'] = False
			curr_args['filter_no_label'] = False
			#Now overwrite curr_args with args
			for attr, value in args.__dict__.items():
				if not attr in curr_args.keys() or value != parser.get_default(attr):
					curr_args[attr] = value
			#Now we can replace args with curr_args
			args = namedtuple("Args", curr_args.keys())(*curr_args.values())
			#Save the current args to 'commandline_args.txt'
			with open(os.path.join(args.experiments_main_folder, args.experiment_folder,'commandline_args.txt'), 'w') as f:
				json.dump(curr_args, f, indent=2)

			#For Debug
			log("Last situtation of args after mergin previous and current settings:")
			log(args)


	#If in eval mode, load the previously saved argparse arguments to load all models correctly
	else:
		with open(os.path.join(args.experiments_main_folder, args.experiment_folder,'commandline_args.txt'), 'r') as f:
			saved_args_dict_ = json.load(f)
		#If the model was saved before the latest updates, some arguments can be missing in saved_args. We fill them with default arguments
		for attr, value in args.__dict__.items():
			if not attr in saved_args_dict_.keys():
				saved_args_dict_[attr] = value
		saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

	data_folder = args.data_folder

	#Choose static features to be added to AttDMM model
	#TODO: get argument for that 	
	training_static = np.load(os.path.join(data_folder, 'nontime_series_training.npy'))

	val_static = np.load(os.path.join(data_folder, 'nontime_series_val.npy'))

	test_static = np.load(os.path.join(data_folder, 'nontime_series_test.npy'))

	#Now load the time series dataset (Extreme values are clipped for our purposes)
	clip_val = args.clip_obs_val if not args.eval_mode else saved_args.clip_obs_val
	def do_clipping(x, clip_val):
		x[x>clip_val] = clip_val
		x[x<-clip_val] = -clip_val
		return x

	training_data_sequences = np.load(os.path.join(data_folder, 'time_series_training.npy'), allow_pickle=True)
	training_data_sequences = np.array(list(map(lambda x: do_clipping(x, clip_val), training_data_sequences)))

	training_seq_lengths = np.array(list(map(lambda x: len(x), training_data_sequences)))

	val_data_sequences = np.load(os.path.join(data_folder, 'time_series_val.npy'), allow_pickle=True)
	val_data_sequences = np.array(list(map(lambda x: do_clipping(x, clip_val), val_data_sequences)))

	val_seq_lengths = np.array(list(map(lambda x: len(x), val_data_sequences)))

	test_data_sequences = np.load(os.path.join(data_folder, 'time_series_test.npy'), allow_pickle=True)
	test_data_sequences = np.array(list(map(lambda x: do_clipping(x, clip_val), test_data_sequences)))

	test_seq_lengths = np.array(list(map(lambda x: len(x), test_data_sequences)))

	#Load the mortality labels for each split
	y_train = np.load(os.path.join(data_folder, "y_mor_training.npy"))
	y_train = y_train.flatten()
	y_val = np.load(os.path.join(data_folder, "y_mor_val.npy"))
	y_val = y_val.flatten()
	y_test = np.load(os.path.join(data_folder, "y_mor_test.npy"))
	y_test = y_test.flatten()

	#Now load the feature-level maskings to be added to the model if flag_load_feature_mask
	#Idea of flag:
	#If feature mask will be used for "emitter and guide" or "ELBO loss masking", we will load them.
	#Current Approach: 
		#If only args.use_feature_mask_ELBO : mask will only be used during ELBO computation
		#If args.use_feature_mask: It will be used for both "emitter and guide" and "ELBO computation" (Basically we don't check args.use_feature_mask_ELBO at all. Still make it true for convenience!)
	flag_load_feature_mask = (not args.eval_mode and (args.use_feature_mask or args.use_feature_mask_ELBO)) or (args.eval_mode and (saved_args.use_feature_mask or saved_args.use_feature_mask_ELBO))
	print(flag_load_feature_mask)
	if flag_load_feature_mask:
		training_data_sequences_mask = np.load(os.path.join(data_folder, 'time_series_training_masking.npy'), allow_pickle=True)

		val_data_sequences_mask = np.load(os.path.join(data_folder, 'time_series_val_masking.npy'), allow_pickle=True)

		test_data_sequences_mask = np.load(os.path.join(data_folder, 'time_series_test_masking.npy'), allow_pickle=True)

		#EXPERIMENTAL PART: Convert all imputed values to -4 (or something out of current range)
		if (not args.eval_mode and args.convert_imputations) or (args.eval_mode and saved_args.convert_imputations):
			training_data_sequences[training_data_sequences_mask == 0] = -4
			val_data_sequences[val_data_sequences_mask == 0] = -4
			test_data_sequences[test_data_sequences_mask == 0] = -4
	else:
		training_data_sequences_mask = None
		val_data_sequences_mask = None
		test_data_sequences_mask = None

	#Load the label masking if provided (for SEMI-SUPERVISED experiments)
	flag_load_label_mask = (not args.eval_mode and args.label_masking != '') or (args.eval_mode and saved_args.label_masking != '')
	if flag_load_label_mask:
		mask_name = args.label_masking if not args.eval_mode else saved_args.label_masking
		y_train_mask = np.load(os.path.join(data_folder, "y_training_masking_"+mask_name+".npy"))
	else:
		y_train_mask = None
	#We won't use any mask for val or test at all!
	y_val_mask = None
	y_test_mask = None

	#FOR SEMI-SUPERVISED EXPERIMENTS
	#If filter_no_label is on, filter out all the training points that are masked out by the label
	#To make filter_no_label flag on: filter_no_label must be true and a valid label_masking must be entered
	#Else if filter_label is on, filter out all the training points having the label (Basically, AttDMM works like autoencoder (i.e. unsupervised))
	#To make filter_label flag on: filter_label must be true and a valid label_masking must be entered

	flag_filter_no_label = flag_load_label_mask and ((not args.eval_mode and args.filter_no_label) or (args.eval_mode and saved_args.filter_no_label)) 
	flag_filter_label = flag_load_label_mask and ((not args.eval_mode and args.filter_label) or (args.eval_mode and saved_args.filter_label)) 

	if flag_filter_no_label:
		training_static = training_static[y_train_mask == 1]
		training_data_sequences = training_data_sequences[y_train_mask == 1]
		training_seq_lengths = training_seq_lengths[y_train_mask == 1]
		y_train = y_train[y_train_mask == 1]
		if training_data_sequences_mask is not None:
			training_data_sequences_mask = training_data_sequences_mask[y_train_mask == 1]
		y_train_mask = y_train_mask[y_train_mask == 1]

	elif flag_filter_label:
		training_static = training_static[y_train_mask == 0]
		training_data_sequences = training_data_sequences[y_train_mask == 0]
		training_seq_lengths = training_seq_lengths[y_train_mask == 0]
		y_train = y_train[y_train_mask == 0]
		if training_data_sequences_mask is not None:
			training_data_sequences_mask = training_data_sequences_mask[y_train_mask == 0]
		y_train_mask = y_train_mask[y_train_mask == 0]
		#Additional to filter_no_label, mask out all the y_val and y_test to assure consistent validation and test losses
		y_val_mask = torch.zeros_like(y_val)
		y_test_mask = torch.zeros_like(y_test)

	#BURADA KALDIM
	use_feature_mask_mini_batch = (not args.eval_mode and args.use_feature_mask) or (args.eval_mode and saved_args.use_feature_mask)
	batches_training = batchify(training_data_sequences, training_seq_lengths, training_data_sequences_mask, training_static, y_train, y_train_mask, max_len=args.max_ICU_length, batch_size=args.mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)
	batches_val = batchify(val_data_sequences, val_seq_lengths, val_data_sequences_mask, val_static, y_val, y_val_mask, max_len=args.max_ICU_length, batch_size=args.mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)
	batches_test = batchify(test_data_sequences, test_seq_lengths, test_data_sequences_mask, test_static, y_test, y_test_mask, max_len=args.max_ICU_length, batch_size=args.mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)


	N_train_data = args.mini_batch_size * (len(batches_training)-1) + len(batches_training[-1][0])
	N_train_time_slices = float(np.sum(training_seq_lengths))
	N_mini_batches = len(batches_training)

	log("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
		(N_train_data, training_seq_lengths.mean(), N_mini_batches))

	# how often we do validation/test evaluation during training
	val_test_frequency = args.eval_freq
	# the number of samples we use to do the evaluation
	n_eval_samples = 1

	# package repeated copies of val/test data for faster evaluation
	# (i.e. set us up for vectorization)
	'''
	def rep(x):
		rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
		repeat_dims = [1] * len(x.size())
		repeat_dims[0] = n_eval_samples
		return x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape)

	# get the validation/test data ready for the attdmm: pack into sequences, etc.
	val_seq_lengths = rep(val_seq_lengths)
	test_seq_lengths = rep(test_seq_lengths)
	'''
	'''
	val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths, val_feature_mask = get_mini_batch(
		torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
		val_seq_lengths, val_data_sequences_mask, cuda=args.cuda)
	test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths, test_feature_mask = get_mini_batch(
		torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
		test_seq_lengths, test_data_sequences_mask, cuda=args.cuda)
	'''

	# instantiate the attdmm (if eval_mode, then use the previous setting)
	# Same idea to setup optimizer (if eval_mode, then use the previous setting)
	if not args.eval_mode:
		attdmm = AttDMM(input_dim=batches_training[0][1].shape[2], static_dim=batches_training[0][0].shape[1], z_dim=args.z_dim, min_x_scale=args.min_x_scale, emission_dim=args.emission_dim, transition_dim=args.transition_dim, linear_gain=args.linear_gain, att_dim = args.att_dim, MLP_dims=args.MLP_dims,
					guide_GRU=args.guide_GRU, rnn_dropout_rate=args.rnn_dropout_rate, rnn_dim=args.rnn_dim, rnn_num_layers=args.rnn_num_layers, 
					num_iafs=args.num_iafs, iaf_dim=args.iaf_dim, use_feature_mask=args.use_feature_mask, use_cuda=args.cuda)
		adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
						"clip_norm": args.clip_norm, "lrd": args.lr_decay,
						"weight_decay": args.weight_decay}
	else:
		attdmm = AttDMM(input_dim=batches_training[0][1].shape[2], static_dim=batches_training[0][0].shape[1], z_dim=saved_args.z_dim, min_x_scale=saved_args.min_x_scale, emission_dim=saved_args.emission_dim, transition_dim=saved_args.transition_dim, linear_gain=saved_args.linear_gain, att_dim = saved_args.att_dim, MLP_dims=saved_args.MLP_dims,
					guide_GRU=saved_args.guide_GRU, rnn_dropout_rate=saved_args.rnn_dropout_rate, rnn_dim=saved_args.rnn_dim, rnn_num_layers=saved_args.rnn_num_layers,
					num_iafs=saved_args.num_iafs, iaf_dim=saved_args.iaf_dim, use_feature_mask=saved_args.use_feature_mask, use_cuda=saved_args.cuda)
		adam_params = {"lr": saved_args.learning_rate, "betas": (saved_args.beta1, saved_args.beta2),
						"clip_norm": saved_args.clip_norm, "lrd": saved_args.lr_decay,
						"weight_decay": saved_args.weight_decay}

	adam = ClippedAdam(adam_params)

	# setup inference algorithm
	if args.tmc:
		if args.jit:
			raise NotImplementedError("no JIT support yet for TMC")
		tmc_loss = TraceTMC_ELBO()
		attdmm_guide = config_enumerate(attdmm.guide, default="parallel", num_samples=args.tmc_num_samples, expand=False)
		svi = SVI(attdmm.model, attdmm_guide, adam, loss=tmc_loss)
	elif args.tmcelbo:
		if args.jit:
			raise NotImplementedError("no JIT support yet for TMC ELBO")
		elbo = TraceEnum_ELBO()
		attdmm_guide = config_enumerate(attdmm.guide, default="parallel", num_samples=args.tmc_num_samples, expand=False)
		svi = SVI(attdmm.model, attdmm_guide, adam, loss=elbo)
	else:
		elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
		svi = SVI(attdmm.model, attdmm.guide, adam, loss=elbo)

	# now we're going to define some functions we need to form the main training loop

	# saves the model and optimizer states to disk
	def save_checkpoint(is_best=False):
		save_model = os.path.join(args.experiments_main_folder, args.experiment_folder, args.save_model)
		save_opt = os.path.join(args.experiments_main_folder, args.experiment_folder, args.save_opt)
		if is_best:
			save_model+='_best'
			save_opt+='_best'
		log("saving model to %s..." % save_model)
		torch.save(attdmm.state_dict(), save_model)
		log("saving optimizer states to %s..." % save_opt)
		adam.save(save_opt)
		log("done saving model and optimizer checkpoints to disk.")

	# loads the model and optimizer states from disk
	def load_checkpoint():
		load_model = os.path.join(args.experiments_main_folder, args.experiment_folder, args.load_model)
		load_opt = os.path.join(args.experiments_main_folder, args.experiment_folder, args.load_opt)
		assert exists(load_opt) and exists(load_model), \
			"--load-model and/or --load-opt misspecified"
		log("loading model from %s..." % load_model)
		attdmm.load_state_dict(torch.load(load_model))
		#if model is loaded in training mode, randomly initialize the weights of predicter 
		#so that it will be ready for supervised learning after attdmm being trained with unsupervised methods
		#If this step is not done, predicter starts with weigths quite close to 0, and cannot learn anything from it
		#SPECIAL NOTE: Adam won't be loaded for below scenario, instead it's created by current args (so that we can change learning rate etc.)
		if not args.eval_mode and args.filter_no_label:
			for i in range(len(attdmm.predicter.lin_layers_nn)):
				print("Re-initalization will be done for Linear layer " + str(i))
				activation = 'sigmoid' if i == len(attdmm.predicter.lin_layers_nn)-1 else 'relu'
				nn.init.xavier_uniform_(attdmm.predicter.lin_layers_nn[i].weight.data, gain=nn.init.calculate_gain(activation))
				#nn.init.uniform_(attdmm.predicter.lin_layers_nn[i].weight.data, a=-0.1, b=0.1)
				nn.init.zeros_(attdmm.predicter.lin_layers_nn[i].bias.data)
				#print(attdmm.predicter.lin_layers_nn[i].weight.data)
				#print(attdmm.predicter.lin_layers_nn[i].bias.data)
			log("Predicter weights are re-initialized!")
			log("Adam optimizer will not be loaded from previous model, instead it's created by current args")
		else:
			log("loading optimizer states from %s..." % load_opt)
			adam.load(load_opt)
		log("done loading model and optimizer states.")

	# prepare a mini-batch and take a gradient step to minimize -elbo
	def process_minibatch(epoch, which_mini_batch):
		if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
			# compute the KL annealing factor approriate for the current mini-batch in the current epoch
			min_af = args.minimum_annealing_factor
			max_af = args.maximum_annealing_factor
			annealing_factor = min_af + (max_af - min_af) * \
				(float(which_mini_batch + epoch * N_mini_batches + 1) /
					float(args.annealing_epochs * N_mini_batches))
		else:
			# by default the KL annealing factor is unity
			annealing_factor = args.maximum_annealing_factor

		#Calculate if renormalize_y_weight will be done for svi.step
		renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)

		# grab a fully prepped mini-batch using the helper function in the data loader
		mini_batch_static, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, mini_batch_feature_mask, y_mini_batch, y_mask_mini_batch, _ = batches_training[which_mini_batch]
		
		# do an actual gradient step
		loss = svi.step(mini_batch_static,mini_batch, mini_batch_reversed, mini_batch_mask,
						mini_batch_seq_lengths, mini_batch_feature_mask, y_mini_batch, y_mask_mini_batch, annealing_factor, args.regularizer, renormalize_y_weight)

		return loss

	# helper function for doing evaluation
	def do_evaluation():
		'''
		Do also evaulation batch by batch (otherwise we have GPU memory problem)
		'''
		attdmm.rnn.eval()

		#During the evaluation, we will always use maximum annealing factor
		annealing_factor = args.maximum_annealing_factor
		#Calculate regularizer depending on which mode we are running (train or eval mode)
		regularizer = args.regularizer if not args.eval_mode else saved_args.regularizer
		#Calculate if renormalize_y_weight will be done for svi.evaluate_loss
		renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)


		#EVALUATION FOR VALIDATION SET
		N_val_data = args.mini_batch_size * (len(batches_val)-1) + len(batches_val[-1][0])
		N_val_time_slices = float(np.sum(val_seq_lengths))
		eval_N_mini_batches = len(batches_val)

		val_nll = 0
		for i in range(eval_N_mini_batches):

			val_batch_static, val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_batch_feature_mask, y_val_batch, y_mask_val_batch, _ = batches_val[i]

			val_nll_batch = svi.evaluate_loss(val_batch_static, val_batch, val_batch_reversed, val_batch_mask,
									val_batch_seq_lengths, val_batch_feature_mask, y_val_batch, y_mask_val_batch, annealing_factor, regularizer, renormalize_y_weight)
			val_nll+=val_nll_batch

		val_nll = val_nll/float(N_val_data)

		#EVALUATION FOR TEST SET
		N_test_data = args.mini_batch_size * (len(batches_test)-1) + len(batches_test[-1][0])
		N_test_time_slices = float(np.sum(test_seq_lengths))
		eval_N_mini_batches = len(batches_test)

		test_nll = 0
		for i in range(eval_N_mini_batches):

			test_batch_static, test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_batch_feature_mask, y_test_batch, y_mask_test_batch, _ = batches_test[i]

			test_nll_batch = svi.evaluate_loss(test_batch_static, test_batch, test_batch_reversed, test_batch_mask,
										test_batch_seq_lengths, test_batch_feature_mask, y_test_batch, y_mask_test_batch, annealing_factor, regularizer, renormalize_y_weight)
			test_nll+=test_nll_batch

		test_nll = test_nll/float(N_test_data)

		attdmm.rnn.train()

		return val_nll, test_nll

	def do_evaluation_rocauc(mini_batch_size=args.mini_batch_size, num_samples=10, verbose=False):
		'''
		Do also evaulation batch by batch (otherwise we have GPU memory problem)
		additional_mop -> including the mean and std of prediction for different samples of num_samples
		'''
		attdmm.rnn.eval()

		#During the evaluation, we will always use maximum annealing factor
		annealing_factor = args.maximum_annealing_factor
		#Calculate regularizer depending on which mode we are running (train or eval mode)
		regularizer = args.regularizer if not args.eval_mode else saved_args.regularizer
		#Calculate if renormalize_y_weight will be done for pred
		renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)

		#Initialize the predicter module
		pred = Predictive(model=attdmm.model, guide=attdmm.guide, num_samples=num_samples)


		#EVALUATION FOR VALIDATION SET
		N_val_data = args.mini_batch_size * (len(batches_val)-1) + len(batches_val[-1][0])
		N_val_time_slices = float(np.sum(val_seq_lengths))
		eval_N_mini_batches = len(batches_val)

		all_y_val = []
		all_mortality_probs_val = []
		for i in range(eval_N_mini_batches):
			val_batch_static, val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_batch_feature_mask, y_val_batch, y_mask_val_batch, _ = batches_val[i]

			pred_dict_val = pred(val_batch_static, val_batch, val_batch_reversed, val_batch_mask,
									val_batch_seq_lengths, val_batch_feature_mask, None, None, annealing_factor, regularizer, renormalize_y_weight)

			#Do Accuracy and ROCAUC analysis for mortality prediction
			#If no linear gain, then we will only use last hidden state for prediction
			if not attdmm.linear_gain:
				list_mortality_probs = []
				z_name = 'z_'
				for j in range(num_samples):
					all_z_vals = []
					# I added [0] here since the original dim was (1,6379,8)
					for w in range(1, val_batch.shape[1]+1):
						z_i_name = z_name + str(w)
						z_i_val = pred_dict_val[z_i_name][j][0]
						all_z_vals.append(z_i_val)
					all_z_vals = torch.stack(all_z_vals).transpose(0,1)
					mortality_probs = attdmm.predicter(all_z_vals, val_batch_mask)
					list_mortality_probs.append(mortality_probs.detach().cpu().numpy())

				mortality_probs = np.mean(np.array(list_mortality_probs), axis=0)
				all_mortality_probs_val = all_mortality_probs_val + list(mortality_probs)

			#If there is linear gain, then we will only all the hidden states (with certain weights) for prediction
			else:
				mortality_probs = np.zeros(val_batch.shape[0])
				for w in range(1,val_batch.shape[1]+1):
					list_mortality_probs_w = []
					week = str(w)
					z_name = 'z_' + week
					for j in range(num_samples):
						# I added [0] here since the original dim was (1,6379,8)
						z_w_test = pred_dict_val[z_name][j][0]
						mortality_probs_w = attdmm.predicter(z_w_test)
						list_mortality_probs_w.append(mortality_probs_w.detach().cpu().numpy())
					mortality_probs_w = np.mean(np.array(list_mortality_probs_w), axis=0)

					w_max = val_batch.shape[1]
					weight_w = w / (w_max * (w_max+1)/2)
					mortality_probs = mortality_probs + mortality_probs_w * weight_w

				all_mortality_probs_val = all_mortality_probs_val + list(mortality_probs)


			all_y_val = all_y_val + list(y_val_batch.detach().cpu().numpy())

			if verbose:
				print("In Validation split: %04d/%04d" % (i+1, eval_N_mini_batches), end="\r", flush=True)

		if verbose:
			print("Validation Done!                   ")

		roc_auc_val = roc_auc_score(np.array(all_y_val), np.array(all_mortality_probs_val))

		#EVALUATION FOR TEST SET
		N_test_data = args.mini_batch_size * (len(batches_test)-1) + len(batches_test[-1][0])
		N_test_time_slices = float(np.sum(test_seq_lengths))
		eval_N_mini_batches = len(batches_test)

		all_y_test = []
		all_mortality_probs_test = []
		all_mortality_probs_test_std = []
		all_test_batch_indices = []
		for i in range(eval_N_mini_batches):
			test_batch_static, test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_batch_feature_mask, y_test_batch, y_mask_test_batch, index_test_batch = batches_test[i]

			all_test_batch_indices = all_test_batch_indices + list(index_test_batch)

			pred_dict_test = pred(test_batch_static, test_batch, test_batch_reversed, test_batch_mask,
										test_batch_seq_lengths, test_batch_feature_mask, None, None, annealing_factor, regularizer, renormalize_y_weight)
			
			#Do Accuracy and ROCAUC analysis for mortality prediction
			#If no linear gain, then we will only use last hidden state for prediction
			if not attdmm.linear_gain:
				list_mortality_probs = []
				z_name = 'z_'
				for j in range(num_samples):
					all_z_tests = []
					for w in range(1, test_batch.shape[1]+1):
						z_i_name = z_name + str(w)
						z_i_test = pred_dict_test[z_i_name][j][0]
						all_z_tests.append(z_i_test)
					all_z_tests = torch.stack(all_z_tests).transpose(0,1)
					mortality_probs = attdmm.predicter(all_z_tests, test_batch_mask)
					list_mortality_probs.append(mortality_probs.detach().cpu().numpy())

				mortality_probs = np.mean(np.array(list_mortality_probs), axis=0)
				mortality_probs_std = np.std(np.array(list_mortality_probs), axis=0)

				all_mortality_probs_test = all_mortality_probs_test + list(mortality_probs)
				all_mortality_probs_test_std = all_mortality_probs_test_std + list(mortality_probs_std)

			#If there is linear gain, then we will only all the hidden states (with certain weights) for prediction
			else:
				mortality_probs = np.zeros(test_batch.shape[0])
				for w in range(1,test_batch.shape[1]+1):
					list_mortality_probs_w = []
					week = str(w)
					z_name = 'z_' + week
					for j in range(num_samples):
						# I added [0] here since the original dim was (1,6379,8)
						z_w_test = pred_dict_test[z_name][j][0]
						mortality_probs_w = attdmm.predicter(z_w_test)
						list_mortality_probs_w.append(mortality_probs_w.detach().cpu().numpy())
					mortality_probs_w = np.mean(np.array(list_mortality_probs_w), axis=0)

					w_max = test_batch.shape[1]
					weight_w = w / (w_max * (w_max+1)/2)
					mortality_probs = mortality_probs + mortality_probs_w * weight_w

				all_mortality_probs_test = all_mortality_probs_test + list(mortality_probs)

			all_y_test = all_y_test + list(y_test_batch.detach().cpu().numpy())
			
			if verbose:
				print("In Test split: %04d/%04d" % (i+1, eval_N_mini_batches), end="\r", flush=True)

		if verbose:
			print("Test Done!                    ")

		roc_auc_test = roc_auc_score(np.array(all_y_test), np.array(all_mortality_probs_test))

		attdmm.rnn.train()

		return roc_auc_val, roc_auc_test, np.array(all_y_test), np.array(all_mortality_probs_test), np.array(all_mortality_probs_test_std), np.array(all_test_batch_indices)


	def do_evaluation_rocauc_custom_time(mini_batch_size=args.mini_batch_size, num_samples=10, cropped_t_from_last=1, do_eval_for_val=True ,verbose=False):
		'''
		Do also evaulation batch by batch (otherwise we have GPU memory problem)
		'''
		attdmm.rnn.eval()

		#Calculate if feature mask is used for get_mini_batch(...):
		use_feature_mask_mini_batch = (not args.eval_mode and args.use_feature_mask) or (args.eval_mode and saved_args.use_feature_mask)
		#During the evaluation, we will always use maximum annealing factor
		annealing_factor = args.maximum_annealing_factor
		#Calculate regularizer depending on which mode we are running (train or eval mode)
		regularizer = args.regularizer if not args.eval_mode else saved_args.regularizer
		#Calculate if renormalize_y_weight will be done for pred
		renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)

		#Initialize the predicter module
		pred = Predictive(model=attdmm.model, guide=attdmm.guide, num_samples=num_samples)

		if do_eval_for_val:

			#MODIFY VALIDATION DATA TO CROP LATEST t TIME STEPS
			custom_val_data_sequences = np.array(list(map(lambda x: x[:-cropped_t_from_last,:] , val_data_sequences)))
			custom_val_seq_lengths = val_seq_lengths - cropped_t_from_last
			custom_val_data_sequences_mask = np.array(list(map(lambda x: x[:-cropped_t_from_last,:] , val_data_sequences_mask)))


			batches_val = batchify(custom_val_data_sequences, custom_val_seq_lengths, custom_val_data_sequences_mask, val_static, y_val, y_val_mask, max_len=args.max_ICU_length - cropped_t_from_last, batch_size=mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)
			

			#EVALUATION FOR VALIDATION SET
			N_val_data = mini_batch_size * (len(batches_val)-1) + len(batches_val[-1][0])
			N_val_time_slices = float(np.sum(custom_val_seq_lengths))
			eval_N_mini_batches = len(batches_val)

			all_y_val = []
			all_mortality_probs_val = []
			for i in range(eval_N_mini_batches):
				val_batch_static, val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_batch_feature_mask, y_val_batch, y_mask_val_batch, _ = batches_val[i]

				pred_dict_val = pred(val_batch_static, val_batch, val_batch_reversed, val_batch_mask,
										val_batch_seq_lengths, val_batch_feature_mask, None, None, annealing_factor, regularizer, renormalize_y_weight)

				#Do Accuracy and ROCAUC analysis for mortality prediction
				#If no linear gain, then we will only use last hidden state for prediction
				if not attdmm.linear_gain:
					list_mortality_probs = []
					z_name = 'z_'
					for j in range(num_samples):
						all_z_vals = []
						# I added [0] here since the original dim was (1,6379,8)
						for w in range(1, val_batch.shape[1]+1):
							z_i_name = z_name + str(w)
							z_i_val = pred_dict_val[z_i_name][j][0]
							all_z_vals.append(z_i_val)
						all_z_vals = torch.stack(all_z_vals).transpose(0,1)
						mortality_probs = attdmm.predicter(all_z_vals, val_batch_mask)
						list_mortality_probs.append(mortality_probs.detach().cpu().numpy())

					mortality_probs = np.mean(np.array(list_mortality_probs), axis=0)
					all_mortality_probs_val = all_mortality_probs_val + list(mortality_probs)

				#If there is linear gain, then we will only all the hidden states (with certain weights) for prediction
				else:
					mortality_probs = np.zeros(val_batch.shape[0])
					for w in range(1,val_batch.shape[1]+1):
						list_mortality_probs_w = []
						week = str(w)
						z_name = 'z_' + week
						for j in range(num_samples):
							# I added [0] here since the original dim was (1,6379,8)
							z_w_test = pred_dict_val[z_name][j][0]
							mortality_probs_w = attdmm.predicter(z_w_test)
							list_mortality_probs_w.append(mortality_probs_w.detach().cpu().numpy())
						mortality_probs_w = np.mean(np.array(list_mortality_probs_w), axis=0)

						w_max = val_batch.shape[1]
						weight_w = w / (w_max * (w_max+1)/2)
						mortality_probs = mortality_probs + mortality_probs_w * weight_w

					all_mortality_probs_val = all_mortality_probs_val + list(mortality_probs)


				all_y_val = all_y_val + list(y_val_batch.detach().cpu().numpy())

				if verbose:
					print("In Validation split: %04d/%04d" % (i+1, eval_N_mini_batches), end="\r", flush=True)

			if verbose:
				print("Validation Done!                   ")

			roc_auc_val = roc_auc_score(np.array(all_y_val), np.array(all_mortality_probs_val))
		else:
			roc_auc_val = -1

		#MODIFY Test DATA TO CROP LATEST t TIME STEPS
		custom_test_data_sequences = np.array(list(map(lambda x: x[:-cropped_t_from_last,:] , test_data_sequences)))
		custom_test_seq_lengths = test_seq_lengths - cropped_t_from_last
		custom_test_data_sequences_mask = np.array(list(map(lambda x: x[:-cropped_t_from_last,:] , test_data_sequences_mask)))

		batches_test = batchify(custom_test_data_sequences, custom_test_seq_lengths, custom_test_data_sequences_mask, test_static, y_test, y_test_mask, max_len=args.max_ICU_length - cropped_t_from_last, batch_size=mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)

		N_test_data = mini_batch_size * (len(batches_test)-1) + len(batches_test[-1][0])
		N_test_time_slices = float(np.sum(custom_test_seq_lengths))
		eval_N_mini_batches = len(batches_test)


		all_y_test = []
		all_mortality_probs_test = []
		all_mortality_probs_test_std = []
		all_test_batch_indices = [] 
		for i in range(eval_N_mini_batches):
			test_batch_static, test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_batch_feature_mask, y_test_batch, y_mask_test_batch, index_test_batch = batches_test[i]

			all_test_batch_indices = all_test_batch_indices + list(index_test_batch)

			pred_dict_test = pred(test_batch_static, test_batch, test_batch_reversed, test_batch_mask,
										test_batch_seq_lengths, test_batch_feature_mask, None, None, annealing_factor, regularizer, renormalize_y_weight)
			
			#Do Accuracy and ROCAUC analysis for mortality prediction
			#If no linear gain, then we will only use last hidden state for prediction
			if not attdmm.linear_gain:
				list_mortality_probs = []
				z_name = 'z_'
				for j in range(num_samples):
					all_z_tests = []
					for w in range(1, test_batch.shape[1]+1):
						z_i_name = z_name + str(w)
						z_i_test = pred_dict_test[z_i_name][j][0]
						all_z_tests.append(z_i_test)
					all_z_tests = torch.stack(all_z_tests).transpose(0,1)
					mortality_probs = attdmm.predicter(all_z_tests, test_batch_mask)
					list_mortality_probs.append(mortality_probs.detach().cpu().numpy())

				mortality_probs = np.mean(np.array(list_mortality_probs), axis=0)
				mortality_probs_std = np.std(np.array(list_mortality_probs), axis=0)

				all_mortality_probs_test = all_mortality_probs_test + list(mortality_probs)
				all_mortality_probs_test_std = all_mortality_probs_test_std + list(mortality_probs_std)

			#If there is linear gain, then we will only all the hidden states (with certain weights) for prediction
			else:
				mortality_probs = np.zeros(test_batch.shape[0])
				for w in range(1,test_batch.shape[1]+1):
					list_mortality_probs_w = []
					week = str(w)
					z_name = 'z_' + week
					for j in range(num_samples):
						# I added [0] here since the original dim was (1,6379,8)
						z_w_test = pred_dict_test[z_name][j][0]
						mortality_probs_w = attdmm.predicter(z_w_test)
						list_mortality_probs_w.append(mortality_probs_w.detach().cpu().numpy())
					mortality_probs_w = np.mean(np.array(list_mortality_probs_w), axis=0)

					w_max = test_batch.shape[1]
					weight_w = w / (w_max * (w_max+1)/2)
					mortality_probs = mortality_probs + mortality_probs_w * weight_w

				all_mortality_probs_test = all_mortality_probs_test + list(mortality_probs)

			all_y_test = all_y_test + list(y_test_batch.detach().cpu().numpy())
			
			if verbose:
				print("In Test split: %04d/%04d" % (i+1, eval_N_mini_batches), end="\r", flush=True)

		if verbose:
			print("Test Done!                    ")

		roc_auc_test = roc_auc_score(np.array(all_y_test), np.array(all_mortality_probs_test))

		attdmm.rnn.train()

		return roc_auc_val, roc_auc_test, np.array(all_y_test), np.array(all_mortality_probs_test), np.array(all_mortality_probs_test_std), np.array(all_test_batch_indices)





	# if checkpoint files provided, load model and optimizer states from disk before we start training
	if args.load_opt != '' and args.load_model != '':
		load_checkpoint()

	times = [time.time()]
	#################
	# TRAINING LOOP #
	#################
	if not args.eval_mode: 
		best_val_nll = np.inf
		best_test_nll = np.inf
		val_nll = np.inf
		test_nll = np.inf
		for epoch in range(args.num_epochs):
			# if specified, save model and optimizer states to disk every checkpoint_freq epochs
			if args.save_model != '':
				if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
					save_checkpoint()

			# accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
			epoch_nll = 0.0

			# process each mini-batch; this is where we take gradient steps
			for which_mini_batch in range(N_mini_batches):
				epoch_nll += process_minibatch(epoch, which_mini_batch)

			#Update Beta of attdmm.predicter if necessary:
			if attdmm.predicter.Beta <= 5:
				attdmm.predicter.Beta = attdmm.predicter.Beta.add(0.03)

			# report training diagnostics
			times.append(time.time())
			epoch_time = times[-1] - times[-2]

			log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
					(epoch, epoch_nll / float(N_train_data), epoch_time))
			# do evaluation on test and validation data and report results
			if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
				val_nll, test_nll = do_evaluation()
				log("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))
				if val_nll < best_val_nll:
					save_checkpoint(is_best=True)
					best_val_nll = val_nll

				roc_auc_val, roc_auc_test, _, _, _, _ = do_evaluation_rocauc()
				log("ROCAUC [val/test epoch %04d]  %.4f  %.4f" % (epoch, roc_auc_val, roc_auc_test))

	#################
	### EVALUATION ##
	#################
	else:
		#IMPORTANT NOTE: Below part under if clause is just for exploration purposes. If you are only interested in prediction performance, you can safely ignore it.
		#In short, flag_obs_analysis is turned on to see how well AttDMM can reconstruct the observations from the hidden states
		flag_obs_analysis = False
		if flag_obs_analysis:
			pred = Predictive(model=attdmm.model, guide=attdmm.guide, num_samples=args.num_samples_eval)
			#Process all val and test set here
			use_feature_mask_mini_batch = (not args.eval_mode and args.use_feature_mask) or (args.eval_mode and saved_args.use_feature_mask)
			#During the evaluation, we will always use maximum annealing factor
			annealing_factor = saved_args.maximum_annealing_factor
			#Calculate if renormalize_y_weight will be done for pred
			renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)

			#Instead of y_val and y_test, we entered None so that the model won't see them during the prediction
			val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_feature_mask, y_val_batch, y_mask_val_batch = get_mini_batch(
				torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
				val_seq_lengths, val_data_sequences_mask, None, y_val_mask, np.arange(len(val_seq_lengths)), use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)
			test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_feature_mask, y_test_batch, y_mask_test_batch = get_mini_batch(
				torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
				test_seq_lengths, test_data_sequences_mask, None, y_test_mask, np.arange(len(test_seq_lengths)), use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)

			#Hidden states can be found in pred_dict
			pred_dict_val = pred(val_static, val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_feature_mask, y_val_batch, y_mask_val_batch, annealing_factor, saved_args.regularizer, renormalize_y_weight)
			pred_dict_test = pred(test_static, test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_feature_mask, y_test_batch, y_mask_test_batch, annealing_factor, saved_args.regularizer, renormalize_y_weight)

			#Create new dataframe to save the results 
			x_val_org_df = pd.DataFrame()
			x_val_pred_df = pd.DataFrame()
			x_test_org_df = pd.DataFrame()
			x_test_pred_df = pd.DataFrame()

			#Take the prediction over time steps
			num_time_steps = val_batch.shape[1]

			val_hidden_states = np.zeros((val_batch.shape[0], val_batch.shape[1], saved_args.z_dim))
			test_hidden_states = np.zeros((test_batch.shape[0], test_batch.shape[1], saved_args.z_dim))

			for w in range(1,num_time_steps+1):
				week = str(w)
				z_i_name = 'z_' + week
				x_i_name = 'obs_x_' + week

				#For Validation Set
				list_z_val = []
				list_locs_val = []
				list_scales_val = []
				for i in range(args.num_samples_eval):
					# I added [0] here since the original dim was (1,6379,8)
					z_i_val = pred_dict_val[z_i_name][i][0]
					#Append i'th sampled hidden states to the list
					list_z_val.append(z_i_val.detach().cpu().numpy())
					#If no feature mask
					# For now, we limit that emitter will never use feature mask
					# Previously, if not self.use_feature_mask:
					if True:
						x_loc, x_scale = attdmm.emitter(z_i_val)
					#If there is feature mask
					else:
						x_loc, x_scale = attdmm.emitter(z_i_val, val_feature_mask[:,w-1,:])
					x_loc = x_loc.detach().cpu().numpy()
					x_scale = x_scale.detach().cpu().numpy()
					list_locs_val.append(x_loc)
					list_scales_val.append(x_scale)

				pred_val = np.mean(np.array(list_locs_val), axis=0)
				#x_val_pred_df[x_i_name] = pred_val

				x_val_org = pred_dict_val[x_i_name][0]
				x_val_org = x_val_org.detach().cpu().numpy()
				#x_val_org = np.argmax(x_val_org, axis=1)
				#x_val_org_df[x_i_name] = x_val_org

				#Save the w'th time step of hidden states
				z_val_mean = np.mean(np.array(list_z_val), axis=0)
				val_hidden_states[:,w-1,:] = z_val_mean

				#For Test Set
				list_locs_test = []
				list_scales_test = []
				list_z_test = []
				for i in range(args.num_samples_eval):
					z_i_test = pred_dict_test[z_i_name][i][0]
					#Append i'th sampled hidden states to the list
					list_z_test.append(z_i_test.detach().cpu().numpy())
					#If no feature mask
					# For now, we limit that emitter will never use feature mask
					# Previously, if not self.use_feature_mask:
					if True:
						x_loc, x_scale = attdmm.emitter(z_i_test)
					#If there is feature mask
					else:
						x_loc, x_scale = attdmm.emitter(z_i_test, test_feature_mask[:,w-1,:])
					x_loc = x_loc.detach().cpu().numpy()
					x_scale = x_scale.detach().cpu().numpy()
					list_locs_test.append(x_loc)
					list_scales_test.append(x_scale)

				pred_test = np.mean(np.array(list_locs_test), axis=0)
				#x_test_pred_df[x_i_name] = pred_test

				x_test_org = pred_dict_test[x_i_name][0]
				x_test_org = x_test_org.detach().cpu().numpy()
				#x_test_org = np.argmax(x_test_org, axis=1)
				#x_test_org_df[x_i_name] = x_test_org

				#Save the w'th time step of hidden states
				z_test_mean = np.mean(np.array(list_z_test), axis=0)
				test_hidden_states[:,w-1,:] = z_test_mean

				#Calculate the performance at each time step 
				#If the mask is used, then make the evaluation only based on original (non-imputed) observations
				diff_arr = np.abs(pred_test - x_test_org)
				if test_feature_mask is not None:
					mask_of_week = test_feature_mask[:,w-1,:].detach().cpu().numpy()
					diff_arr[mask_of_week == 0] = np.nan 		
				#Evaluation Based on Loc values
				acc = np.nanmean(diff_arr)
				log("For week %02d : the mean absolute difference is %.4f " % (w, acc))
				max_diff = np.nanmax(diff_arr)
				min_diff = np.nanmin(diff_arr)
				log("For week %02d : the max and min absolute differences are %.4f and %.4f" % (w, max_diff, min_diff))

				#Evaluation Based on Scale values
				arr_scales_test = np.mean(np.array(list_scales_test), axis=0)
				if test_feature_mask is not None:
					mask_of_week = test_feature_mask[:,w-1,:].detach().cpu().numpy()
					arr_scales_test[mask_of_week == 0] = np.nan 
				max_scale = np.nanmax(arr_scales_test)
				min_scale = np.nanmin(arr_scales_test)
				log("For week %02d : the max and min scales predicted are %.4f and  %.4f" % (w, max_scale, min_scale))

			del val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_feature_mask
			del test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_feature_mask
			del pred_dict_val, pred_dict_test
			import gc
			gc.collect()
			torch.cuda.empty_cache()

		#By cropping last T time steps, we will make predictions of mortality
		#We will keep all the predictions in a dataframe

		mortality_predictions_df = pd.DataFrame(columns=["ID_test", "y", "cropped_t", "y_prob", "y_prob_std"])	
		df_save_path = os.path.join(args.experiments_main_folder, args.experiment_folder, "mortality_predictions_test.csv")

		for cropped_t in range(0,args.max_ICU_length):
			if cropped_t == 0:
				roc_auc_val, roc_auc_test, y_test_np, y_prob_np, y_prob_std_np, index_test = do_evaluation_rocauc(mini_batch_size=args.mini_batch_size, num_samples=20, verbose=True)
			else:
				roc_auc_val, roc_auc_test, y_test_np, y_prob_np, y_prob_std_np, index_test = do_evaluation_rocauc_custom_time(mini_batch_size=args.mini_batch_size, num_samples=20, cropped_t_from_last=cropped_t, do_eval_for_val=False, verbose=True)

			new_df = pd.DataFrame({"ID_test":index_test, "y":y_test_np, "cropped_t":cropped_t, "y_prob":y_prob_np, "y_prob_std":y_prob_std_np})
			mortality_predictions_df = mortality_predictions_df.append(new_df)

			#Save the latest data frame
			mortality_predictions_df.to_csv(df_save_path, index=False)

			log("For cropped_t=%d Number of Test Samples: %d" % (cropped_t, len(y_test_np)))
			log("For cropped_t=%d Test ROC AUC score is : %.4f " % (cropped_t, roc_auc_test))
			for c in np.arange(0.1,1,0.1):
				pred_label = np.zeros(len(y_prob_np))
				pred_label[y_prob_np>c] = 1

				acc = accuracy_score(y_test_np, pred_label)

				log("Test Accuracy for threshold %.2f : %.4f " % (c,acc))




# parse command-line arguments and execute the main method
if __name__ == '__main__':
	assert pyro.__version__.startswith('1.3.0')
	torch.set_default_tensor_type('torch.DoubleTensor')

	parser = argparse.ArgumentParser(description="parse args")
	#AttDMM settings
	parser.add_argument('-zd', '--z_dim', type=int, default=8)
	parser.add_argument('-ed', '--emission_dim', type=int, default=16)
	parser.add_argument('-td', '--transition_dim', type=int, default=32)
	parser.add_argument('-ad', '--att_dim', type=int, default=48)
	parser.add_argument('-md', '--MLP_dims', type=str, default='12-3')

	parser.add_argument('-r', '--regularizer', type=float, default=1.0)
	parser.add_argument('--use_feature_mask', action='store_true')
	#Below is only used for masking during ELBO computation (doesn't use mask for emitter or guide!)
	parser.add_argument('--use_feature_mask_ELBO', action='store_true')
	parser.add_argument('--convert_imputations', action='store_true')
	#Label masking stands for semi-supervised experiments
	parser.add_argument('-lm', '--label_masking', type=str, default='')
	#An option to filter out training points that are masked out by label (i.e. label_masking)
	# filter_no_label -> removes data points having no mortality label
	# filter_label -> removes datap points having the mortality label
	# Usage Note: DON'T ACTIVATE both filter_no_label and filter_label
	parser.add_argument('--filter_no_label', action='store_true')
	parser.add_argument('--filter_label', action='store_true')
	#An option to renormalize the weights of y_labels to cover label_masking
	parser.add_argument('--renormalize_y_weight', action='store_true')
	#Add some minimum constant scale to ensure pdf of x will be upper-bounded
	parser.add_argument('-mxs', '--min_x_scale', type=float, default=0.2)
	parser.add_argument('-co', '--clip_obs_val', type=float, default=3.0)
	#Default version is RNN guide. Use below argumment to use GRU instead
	parser.add_argument('--guide_GRU', action='store_true')
	#Linear gain is to predict 'y' at each time step t with linear weight t/T
	#Overall y weights are renormalied to keep balance between reconstruction and prediction loss
	#Renormalization weight is -> (t/T) / (T*(T+1)/(2T))
	parser.add_argument('--linear_gain', action='store_true')

	parser.add_argument('-maxlen', '--max_ICU_length', type=int, default=360)

	parser.add_argument('-n', '--num_epochs', type=int, default=200)
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
	parser.add_argument('-b1', '--beta1', type=float, default=0.96)
	parser.add_argument('-b2', '--beta2', type=float, default=0.999)
	parser.add_argument('-cn', '--clip_norm', type=float, default=10.0)
	parser.add_argument('-lrd', '--lr_decay', type=float, default=0.99996)
	parser.add_argument('-wd', '--weight_decay', type=float, default=2.0)
	parser.add_argument('-mbs', '--mini_batch_size', type=int, default=16)
	parser.add_argument('-ae', '--annealing_epochs', type=int, default=100)
	parser.add_argument('-maf', '--minimum_annealing_factor', type=float, default=0.2)
	parser.add_argument('-maxaf', '--maximum_annealing_factor', type=float, default=1.0)
	parser.add_argument('-rdr', '--rnn_dropout_rate', type=float, default=0.0)
	parser.add_argument('-rnl', '--rnn_num_layers', type=int, default=1)
	parser.add_argument('-rd', '--rnn_dim', type=int, default=16)
	parser.add_argument('-iafs', '--num_iafs', type=int, default=0)
	parser.add_argument('-id', '--iaf_dim', type=int, default=100)
	parser.add_argument('-cf', '--checkpoint_freq', type=int, default=20)
	parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments')
	parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
	parser.add_argument('-lopt', '--load_opt', type=str, default='')
	parser.add_argument('-lmod', '--load_model', type=str, default='')
	parser.add_argument('-sopt', '--save_opt', type=str, default='opt')
	parser.add_argument('-smod', '--save_model', type=str, default='model')
	parser.add_argument('-l', '--log', type=str, default='attmm.log')
	parser.add_argument('-efreq', '--eval_freq', type=int, default=20)
	parser.add_argument('--data_folder', type=str, default='/home/Data_AttDMM/fold0')
	parser.add_argument('--eval_mode', action='store_true')
	parser.add_argument('-nse', '--num_samples_eval', type=int, default=1)
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--jit', action='store_true')
	parser.add_argument('--tmc', action='store_true')
	parser.add_argument('--tmcelbo', action='store_true')
	parser.add_argument('--tmc_num_samples', default=10, type=int)
	
	args = parser.parse_args()

	if not exists(args.experiments_main_folder):
		os.mkdir(args.experiments_main_folder)
	if not exists(os.path.join(args.experiments_main_folder, args.experiment_folder)):
		os.mkdir(os.path.join(args.experiments_main_folder, args.experiment_folder))

	#TO DO: Add more sanity checks to make sure consistent program running
	#SOME SANITY CHECKTS before main starts
	assert not (args.filter_no_label and args.filter_label)

	main(args)


