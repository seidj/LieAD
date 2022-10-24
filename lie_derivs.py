import jax
import jax.numpy as jnp
from jax.experimental.jet import jet
from jax import random, vmap, jacrev
import os
import numpy as np
from math import factorial
from scipy.special import binom
from jax.lax import scan


def ODE_taylor_coeffs(x0, fun, n):
	# Returns (x_1, x_2, ..., x_n) the coefficients (excluding the initial 
	#                           conditon, x_0) of x(t) where \dot x = fun
	(y0, [*coeffs]) = jet(fun, (x0,), ((jnp.ones_like(x0),),))
	coeffs = []
	for i in range(n-1):
		newargs = ((y0,*coeffs),)
		(y0, [*coeffs]) = jet(fun, (x0,), newargs)
	coeffs = ((y0,*coeffs),)
	return coeffs

# returns function which computs L_g^k L_f h, k=0,...,n-1
def make_derivs_func(x0, f, g, n):
	f_xcoeffs = ODE_taylor_coeffs(x0, f, n)
	fg_xcoeffs = ODE_taylor_coeffs(x0, lambda x: f(x) + g(x), n)
	def lie_derivs(h):
		(y0, [*yis]) = jet(h, (x0,), f_xcoeffs)
		tildey_0, [*tilde_yis] = jet(h, (x0,), fg_xcoeffs)
		derivs = jnp.asarray(tilde_yis) - jnp.asarray(yis)
		return derivs
	return lie_derivs

def lie_feedback_derivs(x0, f, g, h, n):
	f_xcoeffs = ODE_taylor_coeffs(x0, f, n)
	fg_xcoeffs = ODE_taylor_coeffs(x0, lambda x: f(x) + g(x), n)
	(y0, [*yis]) = jet(h, (x0,), f_xcoeffs)
	tildey_0, [*tilde_yis] = jet(h, (x0,), fg_xcoeffs)
	Lfs = jnp.asarray(yis)
	Lfgs = jnp.asarray(tilde_yis)
	derivs = Lfgs - Lfs
	return h(x0), Lfs, derivs[-1]

# Computes factorial coefficients n!/(n-k)! for adj_f^k g 
# using the falling factorial formula: n!/(n-k)! = binom(n,k)*factorial(k)
# Returns an (n+1,n+1) matrix whose entries are ordered 
# for use in "coeff_matmul_conv".
def make_fact_coeffs(n):
	coeffs = jnp.ones((n+1,n+1))
	rows, cols = np.triu_indices(n+1)
	fs = jnp.array([factorial(i) for i in rows])
	kk = n+rows-cols
	ii = rows
	bs = binom(kk, ii)
	coeffs = coeffs.at[rows,cols].set(bs*fs)
	return coeffs

def matrix_commutator(A,B):
	return jnp.matmul(A,B) - jnp.matmul(B,A)

def iterated_linear_brackets(A,B,n):
	com = B
	for i in range(n):
		com = matrix_commutator(com,A)
	return com

# Returns (x_1, x_2, ..., x_n) the coefficients 
# (excluding the initial conditon, x_0) of x(t) where \dot x = fun
def ODE_taylor_coeffs_brack(x0, fun, n, d):
	def body_fn(args, i):
		(y0, [*coeffs]) = jet(fun, (x0,), (list(args),))
		args = args.at[i+1,:].set(jnp.array([*coeffs])[i,:])
		return args, None
	(y0, [*coeffs]) = jet(fun, (x0,), ((jnp.ones_like(x0),),))
	args = jnp.array([y0] + [jnp.zeros(d) for _ in range(n-1)])
	coeffs, _ = scan(body_fn, args, np.arange(n-1))
	return coeffs

# Helper function which makes inputs and outputs simple jnp arrays.  
# Takes x0 and [x1, ..., xn] separately. 
# Otherwise jacrev(jet) will return a complicated tuple of tuples of ...
def z(fun, xi, x0):
	(z0, [*coeffs]) = jet(fun, (x0,), ([*xi],))
	coeffs = jnp.asarray(coeffs)
	out = jnp.concatenate((z0[None], coeffs))
	return out

# Construct Zi matrices.  Returns shape n+1 x d x d
def get_Zi(Amats, d, n):
	Amats = jnp.flip(Amats, axis=0)
	def body_fn(carry, idx):
		Z, T, i = carry
		# Compute pairwise products Z_i A_i
		out = vmap(jnp.matmul, in_axes=(None, 0))(Z, Amats)
		# Update the i-th row of T
		T = T.at[i,...].set(out)
		# Take (off)-diagonal traces of T to sum the Z_k A_{k-i} for each k
		Z = -1.0/(i+1) * jnp.einsum('ijkl,ij->kl', T, idx)
		i += 1
		return (Z, T, i), Z
	# Recursively compute pairwise products Z_i A_i arranged with Z0 A0 
	# in upper right corner and indicies going up when moving to the left and down, by building the 
	# T tensor one row at a time.
	Z0 = jnp.eye(d)
	T = jnp.zeros((n+1,n+1,d,d))
	idx = jnp.array([jnp.eye(n+1, k=n-i) for i in range(n)])
	_, Zs = scan(body_fn, (Z0, T, 0), idx)
	# Concatenate Z0
	Zs = jnp.concatenate([Z0[None], Zs])
	return Zs
		
# Helper function taking two arrays with leading dimension n 
# and returning array "out" with out_i = sum_{j=0}^i A_i B_{i-j}
def coeff_matmul_conv(Zs, bis, fact_coeffs):
	assert Zs.shape[0] == bis.shape[0]
	n = Zs.shape[0] - 1

	# Make pairwise products Z_i b_j arranged with Z0 b0 
	# in upper right corner and indicies going up when moving to the left and down
	T = vmap(vmap(jnp.matmul, in_axes=(None,0)), in_axes=(0,None))(Zs, jnp.flip(bis, axis=0))

	# Multiply elementwise with accumulated factorials k!/(k-i)! for entry with Z_k b_{k-i}
	# This takes care of the 1/(k-i)! for b_{k-i} and the factor of k! for adj^k
	T = jnp.einsum('ijk,ij->ijk', T, fact_coeffs)

	# Take (off)-diagonal traces of T to sum the Z_k b_{k-i} for each k
	idx = jnp.array([jnp.eye(n+1, k=n-i) for i in range(n+1)])
	out = vmap(lambda t: jnp.einsum('ijk,ij->k', T, t))(idx)
	return out

def iterated_brackets(f, g, x0, n, d):
	# Get taylor coefficients of \dot x = f as single jnp array [x0, x1, ..., xn]
	xcoeffs = ODE_taylor_coeffs_brack(x0, f, n, d)

	# Construct input to z function as single jnp array [x0, x1, ..., xn]
	input = jnp.concatenate((x0[None], xcoeffs))

	# Compute derivative tensor of shape n x d x n x d
	Amats = jacrev(lambda ic: z(f, input[1:], ic))(input[0])

	# Compute Zis
	Zs = get_Zi(Amats, d, n)

	# Get lie derivs L_f^k g(x_0)/k! = b_k.
	# Note: jet returns just lie deriv part without factorials
	(b0, [*bis]) = jet(g, (x0,), (list(xcoeffs),))
	bis = jnp.concatenate((b0[None], jnp.asarray(bis)))

	# Compute factorial coefficients
	fact_coeffs = make_fact_coeffs(n)
	
	# Power series mult of coeffs Zis and bis to get (adj_f^k g)(x_0)/k!, k=0,..,n
	# Then multiply by factorials to get answer
	adjs = coeff_matmul_conv(Zs, bis, fact_coeffs)
	return adjs