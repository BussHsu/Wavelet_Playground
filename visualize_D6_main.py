import sympy as sy
from my_scale_func import dx_func,sqrt2
import matplotlib.pyplot as plt
import numpy as np

# Problem 1 solve scale function coefficients
h0,h1,h2,h3,h4,h5=sy.symbols('h0,h1,h2,h3,h4,h5', real=True)
eq1=sy.Eq(h0**2+h1**2+h2**2+h3**2+h4**2+h5**2, 1)
eq2=sy.Eq(h0*h2+h1*h3+h2*h4+h3*h5, 0)
eq3=sy.Eq(h0+h1+h2+h3+h4+h5,sy.sqrt(2))
eq4=sy.Eq(h0-h1+h2-h3+h4-h5,0)
eq5=sy.Eq(h1-2*h2+3*h3-4*h4+5*h5,0)
eq6=sy.Eq(h1-4*h2+9*h3-16*h4+25*h5,0)
eqs = [eq1, eq2, eq3, eq4, eq5, eq6]

s = sy.solve(eqs)

sub_eqs = sy.solve([eqs[2],eqs[3],eqs[4],eqs[5]], ( h2, h3, h4, h5))
list_h = [float(sy.N(s[0][t])) for t in [h0,h1,h2,h3,h4,h5]]
# print(list_h)
# cir_e1 = eqs[0].subs(sub_eqs)
# cir_e2 = eqs[1].subs(sub_eqs)
#
# # plot the solution for h0, h1
# p1 = sy.plot_implicit(cir_e1,show=False)
# p2 = sy.plot_implicit(cir_e2,show=False)
# p1.extend(p2)
# # p1.extend(p3)
# p1.xlim = (-1.,2.)
# p1.ylim = (-1.,2.)
# p1.show()
#
# Plot symbol function
# create 200 sample intervals withing [0,1]

# x = np.linspace(0, 1, 201)
# p = (list_h[0]+list_h[1]*np.exp(1j*-2*np.pi*x[:, np.newaxis])+list_h[2]*np.exp(-1j*4*np.pi*x[:, np.newaxis])+list_h[3]*np.exp(1j*-6*np.pi*x[:, np.newaxis])+list_h[4]*np.exp(1j*-8*np.pi*x[:, np.newaxis])+list_h[5]*np.exp(1j*-10*np.pi*x[:, np.newaxis]))/sqrt2
# p_abs = np.absolute(p)
# plt.plot(x,p_abs)
# plt.show()
# # Problem 2 & 3
# # plot the scale function with cascading algorithm
# # plot the wavelet mother function
#
p = dx_func(x=6)
for _ in range(10):
    p = p.get_next_phi(list_h)

s = p.get_psi(list_h)
p.plot()
s.plot()
#
plt.show()