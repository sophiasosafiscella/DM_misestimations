import math
from NG_timing_analysis import dmx_utils

def c(n,nulim):
    nulow,nuhigh = nulim
    return ((1.0/nulow**n) - (1.0/nuhigh**n))/n

def C(n,nulimA,nulimB):
    return c(n,nulimA) + c(n,nulimB)

def center_to_range(nu,B=0.064): #GHz bandwidth
    return (nu-B/2.0,nu+B/2.0)

def p(cx,alpha=4.4): #cx is the c convenience function, i.e., cx = lambda n: c(n,nulim)
    numer = cx(3)*cx(alpha-1) - cx(1)*cx(alpha+1)
    denom = cx(-1)*cx(3) - cx(1)**2
    return numer/denom
def q(cx,alpha=4.4):
    numer = cx(-1)*cx(alpha+1) - cx(1)*cx(alpha-1)
    denom = cx(-1)*cx(3) - cx(1)**2
    return numer/denom

