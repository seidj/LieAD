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
import os
import numpy as np

import flax
from flax import linen as nn
from flax.training import train_state

import optax

from lie_derivs import make_derivs_func, iterated_brackets

from jax.config import config
config.update("jax_enable_x64", True)

from models import MLP, FL_PINN
from examples import single_link_man_f_g
from lie_derivs import make_derivs_func, iterated_brackets
from control_test import test_linkman_controller


def get_freer_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	return str(np.argmin(np.asarray(memory_available)))

os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"


def main():

	# Setting up RNG keys
	key = random.PRNGKey(13)
	key, input_key, init_key = random.split(key, 3)

	d = 4

	# Creating inputs to test
	inputs = random.normal(input_key, (d,))

	# Initialize model for measurement function h:R^d -> R
	model = MLP(num_hidden=256, num_layers=4, num_outputs=1)

	# Initializing Parameters
	params = model.init(init_key, inputs)

	# Initialize optimizer
	optimizer = optax.adam(learning_rate=0.0001)

	lr = optax.exponential_decay(init_value=1e-3,
                                     transition_steps=1e3,
                                     decay_rate=0.9)
	optimizer = optax.adam(learning_rate=lr)

	# Initialize train state
	model_state = train_state.TrainState.create(apply_fn=model.apply,
											params=params,
											tx=optimizer)

	# initialize dynamics functions for \dot x = f(x) + g(x) u
	f,g = single_link_man_f_g(1.,1.,1.,1.)
	# here order is relative degree - 1
	order = 3
	bounds = [(0,2), (0,2), (0,2), (0,2)]
	num_points = 10000

	# Make dataset
	# note lie bracket condition only computes iterated brackets up to rel degree - 2, 
	# thus we give this order - 1
	train_data = FL_PINN(f, g, d, order-1, num_points, bounds, key, batch_size=128)

	# training loss
	def calculate_loss(state, params, batch, lam=1e-3):

		points = batch[0]
		brackets = batch[1]

		h = lambda x : state.apply_fn(params, x)[0] # turns output into a scalar
		h_x = vmap(grad(h))(points)

		# compute 1/||h_x||^2
		h_x_mags = jnp.einsum('ij, ij -> i', h_x, h_x)
		mag_mean = jnp.mean(h_x_mags)
		penalty = 1.0/mag_mean

		# compute lie bracket condition h_x [g(x), ..., adj_f^{n-2}g(x)]
		mults = vmap(lambda deriv, bracket: jnp.matmul(deriv, bracket.T))(h_x, brackets)
		losses = jnp.einsum('ij, ij -> i', mults, mults)

		# add nonconstant regularizer
		loss = jnp.mean(losses) + lam*penalty
		return loss	

	@jit
	def train_step(state, batch):
		grad_fn = jax.value_and_grad(calculate_loss, # loss function
									argnums=1, # parameters are second argument
									has_aux=False) # Signals loss function has no additional outputs
		loss, grads = grad_fn(state, state.params, batch)

		# Update parameters
		state = state.apply_gradients(grads=grads)

		return state, loss
	
	def train_model(state, data_loader, num_iter=4000):
		loss_hist = []
		generator = iter(data_loader)

		pbar = trange(num_iter)
		for it in pbar:
			batch = next(generator)
			state, loss = train_step(state, batch)
			if it % 5 == 0:
				loss_hist.append(loss)
			
			pbar.set_postfix({'Training loss': loss})
		return state, loss_hist

	trained_model_state, loss_hist = train_model(model_state,
											 train_data,
											 num_iter=100)
	trained_params = trained_model_state.params
	
	## check if h is constant (compare its max and min)
	# h = vmap(lambda x: trained_model_state.apply_fn(trained_params, x))(train_data.data)
	# print(jnp.max(h))
	# print(jnp.min(h))

	batch = train_data[0]
	test_loss = calculate_loss(trained_model_state, trained_params, batch, lam=0)
	print(test_loss)

	#test and plot stabilizing controller performance
	h = lambda x: trained_model_state.apply_fn(trained_params, x)[0]
	h_known = lambda x: x[0]
	x0 = jnp.asarray([.5,.5,.5,.5])
	test_linkman_controller(x0, h_known, "known control")
	test_linkman_controller(x0, h, "learned_control-100")


if __name__== "__main__":
	main()