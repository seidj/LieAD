import jax
import os
import sys
from functools import partial
from tqdm import trange
import torch.utils.data as data
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.experimental.jet import jet
from jax import random, vmap, jit, jacrev, grad
from jax.lax import scan
import flax
from flax import linen as nn
from flax.training import train_state

import os
import numpy as np

import optax
from pyDOE import lhs

from lie_derivs import make_derivs_func, iterated_brackets

# sample from multivariate uniform distribution on hyper-rectangle
def multivariate_uniform(key, dim, N, bounds):
	dilations = np.asarray([interval[1]-interval[0] for interval in bounds])
	lbs = np.asarray([interval[0] for interval in bounds])
	data = random.uniform(key, minval=0, maxval=1, shape=(dim, N))
	data = dilations[:,None]*data # check broadcasting
	data = data + lbs[:,None]	# check broadcasting
	return data.T

class MLP(nn.Module):
	num_hidden: int # Neurons per hidden layer
	num_layers: int # Number of hidden layers
	num_outputs: int # Output dimension

	@nn.compact
	def __call__(self, x):
		# Hidden Layers
		for _ in range(self.num_layers):
			x = nn.Dense(features=self.num_hidden)(x)
			x = nn.gelu(x)
		# Final dense layer
		x = nn.Dense(features=self.num_outputs)(x)
		return x



class FL_PINN(data.Dataset):
	def __init__(self, f, g, dim, order, size, bounds, key, batch_size=64):
		super().__init__()
		self.f = f
		self.g = g
		self.dim = dim
		self.bounds = bounds
		self.size = size
		self.key = key
		self.batch_size = batch_size
		self.order = order
		print('Generating collocation points')
		self.generate_inputs()
		print('Generating brackets at points')
		self.generate_lie_brackets()
		print('Brackets done')

	def generate_inputs(self):
		self.key, subkey = random.split(self.key)
		# data = multivariate_uniform(self.key, self.dim, self.size, self.bounds)
		data = lhs(self.dim,  self.size)
		self.data = jnp.asarray(data)

	def generate_lie_brackets(self):
		brackets_fun = lambda x: iterated_brackets(self.f, self.g, x, self.order, self.dim)
		brackets = vmap(brackets_fun)(self.data)
		self.brackets = brackets

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		self.key, subkey = random.split(self.key)
		batch = self.__select_batch(subkey)
		return batch

	@partial(jit, static_argnums=(0,))
	def __select_batch(self, key):
		idx = random.choice(key, self.size, (self.batch_size,), replace=False)
		return (self.data[idx], self.brackets[idx])
