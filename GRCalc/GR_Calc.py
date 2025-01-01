from GR_Func import *

sym.init_printing()

t,r,theta,phi = sym.symbols("t,r,theta,phi",real=True)
c,G,M = sym.symbols("C,G,M",real=True)

'''
gc = sym.matrices.diag(-(1-(2*G*M)/(r*c**2)),\
                      1/(1-(2*G*M)/(r*c**2)),\
                      r**2,\
                      r**2 * sym.sin(theta)**2)

'''

xc = [t,r,theta,phi]
gc = sym.matrices.diag(-(1-(2*G*M)/(r)),\
                      1/(1-(2*G*M)/(r)),\
                      r**2,\
                      r**2 * sym.sin(theta)**2)

x = [theta, phi]
g = sym.matrices.diag(r**2, r**2 * sym.cos(theta)**2)
