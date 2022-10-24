import jax.numpy as jnp

def get_VDP_f_g(eps):
	def f(x):
		x1, x2 = x[0], x[1]
		out = jnp.asarray([x2, -x1 + eps*(1-x1**2)*x2])
		return out
	def g(x):
		return jnp.asarray([0.0, 1.0])
	return f, g

# single link manipulator with flexible joints
# http://users.isr.ist.utl.pt/~pedro/NCS2012/07_FeedbackLinearization.pdf
def single_link_man_f_g(a, b, c, d):
	def f(x):
		x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
		out1 = x2
		out2 = -a*jnp.sin(x1) - b*(x1 - x3)
		out3 = x4
		out4 = c*(x1 - x3)
		return jnp.asarray([out1, out2, out3, out4])
	def g(x):
		return jnp.asarray([0,0,0,d])
	return f, g

def sastry_ex_9pt9():
    def f(x):
        x1, x2, x3 = x[0], x[1], x[2]
        out1 = 0
        out2 = x1 + x2**2
        out3 = x1 - x2
        return jnp.asarray([out1, out2, out3])
    def g(x):
        x1, x2 = x[0], x[1]
        out1 = jnp.exp(x2)
        out2 = jnp.exp(x2)
        out3 = 0.0
        return jnp.asarray([out1, out2, out3])
    return f, g