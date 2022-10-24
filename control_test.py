import jax
import jax.numpy as jnp
import os
import sys
from functools import partial
from jax import jit, vmap
import matplotlib.pyplot as plt

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt

import os
import numpy as np


from lie_derivs import make_derivs_func, iterated_brackets

from jax.config import config
config.update("jax_enable_x64", True)

from examples import single_link_man_f_g
from lie_derivs import make_derivs_func, iterated_brackets, lie_feedback_derivs


def get_freer_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	return str(np.argmin(np.asarray(memory_available)))

os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"

def test_linkman_controller(x0, h, plot_name):
    # get polynomial coefficients for roots s1,...sn
    r=4
    s1=-1
    s2 = -2
    s3 = -3
    s4 = -4
    nomz = np.poly1d([s1,s2,s3,s4], True)
    coeffs = jnp.asarray(nomz.c)

    f,g = single_link_man_f_g(1.,1.,1.,1.)
    h_derivs = lambda x: lie_feedback_derivs(x, f, g, h, r)

    @jit
    def control_func(x):
        hx0, Lfs, Lgf = h_derivs(x)
        sum = -jnp.dot(jnp.flip(Lfs),coeffs[:-1]) - coeffs[-1]*hx0
        out = (1.0/Lgf)*sum
        return out

    @jit
    def odefunc(t, x, args):
        u = control_func(x)
        return f(x) + g(x)*u

    term = ODETerm(odefunc)
    solver = Dopri5()
    t0 = 0
    t1 = 10
    ts = jnp.linspace(t0,t1,100)
    saveat = SaveAt(ts=ts)
    print('integrating control system')
    solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=0.1, y0=x0, saveat=saveat)
    print('finished integrating')
    # print(solution.ys)
    xs = solution.ys

    plt.figure()
    plt.plot(ts, xs[:,0], 'r')  
    plt.plot(ts, xs[:,1], 'b')  
    plt.plot(ts, xs[:,2], 'g') 
    plt.plot(ts, xs[:,3], 'm') 
    plt.savefig(plot_name)

def main():

    n = 4
    r = n

    
    # get polynomial coefficients for roots s1,...sn
    s1=-1
    s2 = -2
    s3 = -3
    s4 = -4
    nomz = np.poly1d([s1,s2,s3,s4], True)
    coeffs = jnp.asarray(nomz.c)
    # print(coeffs)

    f,g = single_link_man_f_g(1.,1.,1.,1.)
    xtest = jnp.asarray([.2,.3,.4,.2])

    h = lambda x: x[0]

    # test_derivs = make_derivs_func(xtest, f, g, r)
    # print(test_derivs(h))
    # print(lie_feedback_derivs(xtest, f, g, h, r))
    # print(h(xtest))

    h_derivs = lambda x: lie_feedback_derivs(x, f, g, h, r)

    @jit
    def control_func(x):
        hx0, Lfs, Lgf = h_derivs(x)
        sum = -jnp.dot(jnp.flip(Lfs),coeffs[:-1]) - coeffs[-1]*hx0
        out = (1.0/Lgf)*sum
        return out

    @jit
    def odefunc(t, x, args):
        u = control_func(x)
        return f(x) + g(x)*u

    term = ODETerm(odefunc)
    solver = Dopri5()
    t0 = 0
    t1 = 10
    ts = jnp.linspace(t0,t1,100)
    saveat = SaveAt(ts=ts)
    solution = diffeqsolve(term, solver, t0=t0, t1=t1, dt0=0.1, y0=xtest, saveat=saveat)
    # print(solution.ys)
    xs = solution.ys

    plot=True
    if plot:
        plt.plot(ts, xs[:,0], 'r') # plotting t, a separately 
        plt.plot(ts, xs[:,1], 'b') # plotting t, b separately 
        plt.plot(ts, xs[:,2], 'g') # plotting t, c separately
        plt.plot(ts, xs[:,3], 'm') # plotting t, c separately 
        plt.savefig('testpic.png')


if __name__== "__main__":
	main()