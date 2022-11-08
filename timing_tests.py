import jax
import jax.numpy as jnp
import os
import sys
from functools import partial
from jax import jit, vmap, grad, jacrev, jacfwd, jvp
import matplotlib.pyplot as plt
import timeit
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

@partial(jit, static_argnums=(0,1,))
def bracket_with_f_revAD(f, g, x):
    df = jacrev(f)(x)
    dg = jacrev(g)(x)
    out = jnp.matmul(df, g(x)) - jnp.matmul(dg, f(x))
    return out


def bracket_with_f_fwdAD(f, g, x):
    df = jacfwd(f)(x)
    dg = jacfwd(g)(x)
    out = jnp.matmul(df, g(x)) - jnp.matmul(dg, f(x))
    return out


# def revAD_bracket(f, g, x0, order):
#     bracket_fun = lambda x: bracket_with_f_revAD(f,g,x)
#     for i in range(order-1):
#         rightarg = bracket_fun
#         bracket_fun = lambda x: bracket_with_f_revAD(f, rightarg, x)
#     out = bracket_with_f_revAD(f, rightarg, x0)
#     return out

def reg_AD_bracket_1(f, g, x0, direction):
    if direction=='Forward':
        bracket_fun = bracket_with_f_fwdAD
    elif direction=='Reverse':
        bracket_fun = bracket_with_f_revAD
    else:
        print('Last argument must be \'Forward\' or \'Reverse\'')
    bracketlast = bracket_fun(f, g, x0)
    out = bracketlast
    return out

def reg_AD_bracket_2(f, g, x0, direction):
    if direction=='Forward':
        bracket_fun = bracket_with_f_fwdAD
    elif direction=='Reverse':
        bracket_fun = bracket_with_f_revAD
    else:
        print('Last argument must be \'Forward\' or \'Reverse\'')
    bracket1 = lambda x: bracket_fun(f, g, x)
    bracketlast = bracket_fun(f, bracket1, x0)
    out = bracketlast
    return out

def reg_AD_bracket_3(f, g, x0, direction):
    if direction=='Forward':
        bracket_fun = bracket_with_f_fwdAD
    elif direction=='Reverse':
        bracket_fun = bracket_with_f_revAD
    else:
        print('Last argument must be \'Forward\' or \'Reverse\'')
    bracket1 = lambda x: bracket_fun(f, g, x)
    bracket2 = lambda x: bracket_fun(f, bracket1, x)
    bracketlast = bracket_fun(f, bracket2, x0)
    out = bracketlast
    return out

def reg_AD_bracket_4(f, g, x0, direction):
    if direction=='Forward':
        bracket_fun = bracket_with_f_fwdAD
    elif direction=='Reverse':
        bracket_fun = bracket_with_f_revAD
    else:
        print('Last argument must be \'Forward\' or \'Reverse\'')
    bracket1 = lambda x: bracket_fun(f, g, x)
    bracket2 = lambda x: bracket_fun(f, bracket1, x)
    bracket3 = lambda x: bracket_fun(f, bracket2, x)
    bracketlast = bracket_fun(f, bracket3, x0)
    out = bracketlast
    return out

def reg_AD_bracket_5(f, g, x0, direction):
    if direction=='Forward':
        bracket_fun = bracket_with_f_fwdAD
    elif direction=='Reverse':
        bracket_fun = bracket_with_f_revAD
    else:
        print('Last argument must be \'Forward\' or \'Reverse\'')
    bracket1 = lambda x: bracket_fun(f, g, x)
    bracket2 = lambda x: bracket_fun(f, bracket1, x)
    bracket3 = lambda x: bracket_fun(f, bracket2, x)
    bracket4 = lambda x: bracket_fun(f, bracket3, x)
    bracketlast = bracket_fun(f, bracket4, x0)
    out = bracketlast
    return out

def reg_AD_bracket_6(f, g, x0, direction):
    if direction=='Forward':
        bracket_fun = bracket_with_f_fwdAD
    elif direction=='Reverse':
        bracket_fun = bracket_with_f_revAD
    else:
        print('Last argument must be \'Forward\' or \'Reverse\'')
    bracket1 = lambda x: bracket_fun(f, g, x)
    bracket2 = lambda x: bracket_fun(f, bracket1, x)
    bracket3 = lambda x: bracket_fun(f, bracket2, x)
    bracket4 = lambda x: bracket_fun(f, bracket3, x)
    bracket5 = lambda x: bracket_fun(f, bracket4, x)
    bracketlast = bracket_fun(f, bracket5, x0)
    out = bracketlast
    return out

def reg_AD_bracket_7(f, g, x0, direction):
    if direction=='Forward':
        bracket_fun = bracket_with_f_fwdAD
    elif direction=='Reverse':
        bracket_fun = bracket_with_f_revAD
    else:
        print('Last argument must be \'Forward\' or \'Reverse\'')
    bracket1 = lambda x: bracket_fun(f, g, x)
    bracket2 = lambda x: bracket_fun(f, bracket1, x)
    bracket3 = lambda x: bracket_fun(f, bracket2, x)
    bracket4 = lambda x: bracket_fun(f, bracket3, x)
    bracket5 = lambda x: bracket_fun(f, bracket4, x)
    bracket6 = lambda x: bracket_fun(f, bracket5, x)
    bracketlast = bracket_fun(f, bracket6, x0)
    out = bracketlast
    return out

def reg_AD_bracket_8(f, g, x0, direction):
    if direction=='Forward':
        bracket_fun = bracket_with_f_fwdAD
    elif direction=='Reverse':
        bracket_fun = bracket_with_f_revAD
    else:
        print('Last argument must be \'Forward\' or \'Reverse\'')
    bracket1 = lambda x: bracket_fun(f, g, x)
    bracket2 = lambda x: bracket_fun(f, bracket1, x)
    bracket3 = lambda x: bracket_fun(f, bracket2, x)
    bracket4 = lambda x: bracket_fun(f, bracket3, x)
    bracket5 = lambda x: bracket_fun(f, bracket4, x)
    bracket6 = lambda x: bracket_fun(f, bracket5, x)
    bracket7 = lambda x: bracket_fun(f, bracket6, x)
    bracketlast = bracket_fun(f, bracket7, x0)
    out = bracketlast
    return out

def jvp_lie(f, h, x):
	_, out = jvp(h, (x,), f(x))
	return out

def jvp_lie_1(f, h, x):
	out = jvp_lie(f, jvp_lie(f, h, x), x)
	return out

def jvp_lie_2(f, h, x):
	out = jvp_lie(f, jvp_lie_1(f, h, x))
	return out

def main():

    f, g = single_link_man_f_g(1.,1.,1.,1.)
    x0 = jnp.asarray([.5,.5,.5,.5])
    order = 1


    ans = jvp_lie(f, g, x0)
    print(ans)

    sys.exit()
    # test = revAD_bracket(f, g, x0, 3)    
    # print(test)
    # bracket1 = lambda x: bracket_with_f_revAD(f, g, x)
    # bracket2 = lambda x: bracket_with_f_revAD(f, bracket1, x)
    # test = bracket_with_f_revAD(f, bracket2, x0)

    # rightarg = g
    # bracket_fun = lambda x: bracket_with_f_revAD(f, rightarg, x)
    # test = bracket_fun(x0)

    # print(test)

    # taylor_brackets = iterated_brackets(f, g, x0, 4, 4)
    # print(taylor_brackets[-1])

    # funs = [reg_AD_bracket_1, reg_AD_bracket_2, reg_AD_bracket_3, reg_AD_bracket_4, reg_AD_bracket_5, reg_AD_bracket_6, 
    funs = [reg_AD_bracket_7, reg_AD_bracket_8]
    num_calls = 10

    fwds = []
    revs = []
    taylors = []

    for fun in funs:

        funname = fun.__name__
        order = fun.__name__[-1]

        revtime = timeit.timeit(funname+'(f,g,x0,dir)', 
                                setup="f, g = single_link_man_f_g(1.,1.,1.,1.); x0 = jnp.asarray([.5,.5,.5,.5]); dir='Reverse'", 
                                number=num_calls, globals=globals())/num_calls
        revs.append(revtime)

        taylortime = timeit.timeit('iterated_brackets(f, g, x0, n, 4)', 
                                    setup="f, g = single_link_man_f_g(1.,1.,1.,1.); x0 = jnp.asarray([.5,.5,.5,.5]); n="+order, 
                                    number=num_calls, globals=globals())/num_calls
        taylors.append(taylortime)

        fwdtime = timeit.timeit(funname+'(f,g,x0, dir)', 
                                setup="f, g = single_link_man_f_g(1.,1.,1.,1.); x0 = jnp.asarray([.5,.5,.5,.5]); dir='Forward'", 
                                number=num_calls, globals=globals())/num_calls
        fwds.append(fwdtime)
        
        print(fun.__name__[-1] + ' - Rev Mode: {}, Fwd Mode: {}, Taylor Mode: {}'.format(revtime, fwdtime, taylortime))


    
    revs = np.array(revs)
    taylors = np.array(taylors)
    fwds = np.array(fwds)
    data = {'revs': revs, 'fwds': fwds, 'taylors': taylors}
    np.save('single_link_man_times.npy', data)
    


    return





if __name__ == '__main__':
    main()