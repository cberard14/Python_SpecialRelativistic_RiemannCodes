import sys
import numpy as np
import matplotlib.pyplot as plt
from SRRiemann import Riemann
import scipy.integrate.quadrature as gauss
from scipy.optimize import brentq,fsolve

"""
This code generates the analytic solution
to the Riemann problem with sideflow
and is a python translation of
the Marti and Muller program that does the same
"""

xc,P,rho,ux,uy = np.loadtxt("64cellsnoppm_Sod0.99sideflow_t0.45.txt",skiprows=1,usecols=(0,1,2,3,4),unpack=True)

##### USEFUL FUNCTIONS #####
def ksiRf(gamma,K,A,P,u,sgn):
    """
    Called to determine Rf position.
    Computes self-similarity variable ksi (x/t)
    inside rarefaction wave
    "FLAMB" in Marti & Muller program
    """
    rho = (P/K)**(1.0/gamma)
    cs2 = gamma*(gamma-1)*P/(gamma*P+(gamma-1.0)*rho)
    h = 1.0/(1.0-cs2/(gamma-1.0))
    vT2 = (1.0-u**2)*(A**2)/(h**2+A**2)
    v2 = vT2+u**2
    beta = (1.0-v2)*cs2/(1.0-cs2)
    disc = np.sqrt(beta*(1.0+beta-u**2))
    ksival = (u+sgn*disc)/(1.0+beta)
    return ksival

def gaussfunc(Pval,gamma,K,const):
    """
    This function needs to be integrated
    to obtain x-direction velocity in rf state
    """
    rrho = (Pval/K)**(1.0/gamma)
    ccs2 = gamma*(gamma-1.0)*Pval/(gamma*Pval+(gamma-1.0)*rrho)
    hh = 1.0/(1.0-ccs2/(gamma-1.0))
    integrand = np.sqrt(hh**2+(const**2)*(1.0-ccs2))/(hh**2+const**2)\
                *(1.0/(rrho*np.sqrt(ccs2)))
    return integrand

def FP(gamma,P,xi,K,A,Pa,rhoa,ua,sgn):
    """
    Find 0's of this function
    """
    integral = gauss(lambda x: gaussfunc(x,gamma,K,A),Pa,P)[0]
    Bval = sgn*integral+0.5*np.log((1.0+ua)/(1.0-ua))
    uval = np.tanh(Bval) # VELOCITY AT TAIL OF WAVE
    ksival = ksiRf(gamma,K,A,P,uval,sgn)
    return ksival-xi

def rfstate(gamma,xi,Pguess,Psol,usol,rhoa,Pa,ea,csa,ua,vTa,sgn):
    """
    returns value of rarefaction state variables
    as a function of position (contained within 
    "flowv" variable (which is a SINGLE VALUE)
    this is "RAREF2" in the Marti & Muller program
    """
    ha = 1.0+ea+Pa/rhoa
    Wa = 1.0/np.sqrt(1.0-ua**2-vTa**2)
    const = ha*Wa*vTa
    Ka = Pa/(rhoa**gamma)
    xio = ksiRf(gamma,Ka,const,Pa,ua,sgn) #always same value
    Po = Pa
    Pnew = float(brentq(lambda x: FP(gamma,x,xi,Ka,const,Pa,rhoa,ua,sgn),0.0001,1100.0))
    Prf = max(Pnew,Psol)
    
    #get velocity
    integral = gauss(lambda x: gaussfunc(x,gamma,Ka,const),Pa,Pnew)[0]
    Bval = sgn*integral+0.5*np.log((1.0+ua)/(1.0-ua))
    urf = np.tanh(Bval) # VELOCITY AT TAIL OF WAVE
    
    rhorf = (Prf/Ka)**(1.0/gamma)
    erf = Prf/(rhorf*(gamma-1.0))
    hrf = 1.0+erf+Prf/rhorf
    vTrf = const*np.sqrt((1.0-urf**2)/(const**2+hrf**2))
    return Prf,rhorf,erf,urf,vTrf

def shockstate(gamma,Psol,ha,rhoa,Pa,ea,csa,ua,vTa,sgn):
    """
    Returns post-shock state
    """
    wa = (1.0-ua**2-vTa**2)**(-0.5)
    A = 1.0-(gamma-1.0)*(Psol-Pa)/gamma/Psol
    B = 1.0-A
    C = ha*(Pa-Psol)/rhoa-ha**2
    if C > (B**2/4.0/A):
        print "unphysical shock state"
        sys.exit(0)
    h = (-B+np.sqrt(B**2-4.0*A*C))/(2.0*A)
    rho = gamma*Psol/(gamma-1.0)/(h-1.0)
    e = Psol/(gamma-1.0)/rho
    j = sgn*np.sqrt((Psol-Pa)/(ha/rhoa-h/rho))
    
    A = j**2+(rhoa*wa)**2
    B = -2.0*ua*(rhoa**2)*(wa**2)
    C = (rhoa*ua*wa)**2-j**2

    vshock = (-B+sgn*np.sqrt(B**2-4.0*A*C))/(2.0*A)
    return vshock,rho
    

##### INITIAL CONDITIONS #####
gamma = 1.67
N = 5000 #number of points
length = 1.0
x0 = 0.0 #0.5*length #boundary location
tf = 0.4
PL,PR = 1000.0, 0.01
rhoL,rhoR = 1.0,1.0
uL,uR = 0.0,0.0 # x velocity
vTL,vTR = 0.99,0.99 # tangential velocity

KL = PL/(rhoL**gamma)
KR = PR/(rhoR**gamma)

wL = (1-uL**2-vTL**2)**(-0.5)
wR = (1-uR**2-vTR**2)**(-0.5)

eL = PL/(rhoL*(gamma-1.0))
eR = PR/(rhoR*(gamma-1.0))

hL = 1.0+eL+PL/rhoL
hR = 1.0+eR+PR/rhoR

csR = np.sqrt((gamma*PR)/(hR*rhoR))
csL = np.sqrt((gamma*PL)/(hL*rhoL))

constL = hL*wL*vTL
constR = hR*wR*vTR

##### GET RIEMANN SOLVER SOLUTION #####
P_star,u_star = Riemann(gamma,PL,rhoL,uL,vTL,PR,rhoR,uR,vTR)

##### GET REGION POSITIONS ######
x1 = x0+tf*ksiRf(gamma,KL,constL,PL,uL,-1)
x2 = x0+tf*ksiRf(gamma,KL,constL,P_star,u_star,-1)
x3 = x0+u_star*tf
x4 = x0+tf*shockstate(gamma,P_star,hR,rhoR,PR,eR,csR,uR,vTR,1)[0]
x5 = x4

##### SET UP SOLUTION ARRAYS #####
Psol = np.zeros(N)
rhosol = np.zeros(N)
usol = np.zeros(N)
vsol = np.zeros(N)
esol = np.zeros(N)
hsol = np.zeros(N)
wsol = np.zeros(N)
Rsol = np.zeros(N)
rad = np.zeros(N)

NRflist = []

##### ASSIGN VALUES TO REGIONS #####
for i in range(N):
    rad[i] = x0+float(i)/float(N)-0.5
    if (rad[i] > x1) and (rad[i] <= x2):
        NRflist.append(float(i))
NPs = len(NRflist)
Pints = np.arange(P_star,PL,(PL-P_star)/(NPs))
Pints = Pints[::-1]

k = 0
for i in range(N):
    if rad[i] <= x1:
        Psol[i] = PL
        rhosol[i] = rhoL
        usol[i] = uL
        vsol[i] = vTL
        esol[i] = eL
        hsol[i] = hL
        wsol[i] = wL
    if (rad[i] > x1) and (rad[i] <= x2):
        Aval = (rad[i]-x0)/tf
        Psol[i],rhosol[i],esol[i],usol[i],vsol[i] = \
             rfstate(gamma,Aval,Pints[k],P_star,u_star,rhoL,PL,eL,csL,uL,vTL,-1)
        hsol[i] = 1.0+esol[i]+Psol[i]/rhosol[i]
        wsol[i] = 1.0/np.sqrt(1.0-usol[i]**2-vsol[i]**2)
        k += 1
    if (rad[i] > x2) and (rad[i] <= x3):
        Psol[i] = P_star
        rhosol[i] = (P_star/KL)**(1.0/gamma)
        esol[i] = Psol[i]/((gamma-1.0)*rhosol[i])
        hsol[i] = 1.0+esol[i]+Psol[i]/rhosol[i]
        usol[i] = u_star
        vsol[i] = constL*np.sqrt((1.0-u_star**2)/(constL**2+hL**2))
        wsol[i] = 1.0/np.sqrt(1.0-usol[i]**2-vsol[i]**2)
    if (rad[i] > x3) and (rad[i] <= x4):
        Psol[i] = P_star
        rhosol[i] = shockstate(gamma,P_star,hR,rhoR,PR,eR,csR,uR,vTR,+1)[1]
        esol[i] = Psol[i]/((gamma-1.0)*rhosol[i])
        hsol[i] = 1.0+esol[i]+Psol[i]/rhosol[i]
        usol[i] = u_star
        vsol[i] = constR*np.sqrt((1.0-u_star**2)/(constR**2+hR**2))
        wsol[i] = 1.0/np.sqrt(1.0-usol[i]**2-vsol[i]**2)
    if (rad[i] > x4):   
        Psol[i] = PR
        rhosol[i] = rhoR
        usol[i] = uR
        vsol[i] = vTR
        esol[i] = eR
        hsol[i] = hR
        wsol[i] = wR

plt.figure(1)
plt.plot(rad+0.5,Psol,'b-',lw=2.0)
plt.title("no ppm 64 cells, gamma="+str(gamma)+", vTL="+str(vTL)+", vTR="+str(vTR)+" t="+str(tf))
plt.plot(xc,P,'r.-')
plt.legend(["Analytic","Python"],loc="best")
plt.ylabel("P")
plt.xlabel("x")

plt.figure(2)
plt.plot(rad+0.5,rhosol,'b-',lw=2.0)
plt.title("no ppm 64 cells, gamma="+str(gamma)+", vTL="+str(vTL)+", vTR="+str(vTR)+" t="+str(tf))
plt.plot(xc,rho,'r.-')
plt.legend(["Analytic","Python"],loc="best")
plt.ylabel("rho")
plt.xlabel("x")

plt.figure(3)
plt.plot(rad+0.5,usol,'b-',lw=2.0)
plt.title("no ppm 64 cells, gamma="+str(gamma)+", vTL="+str(vTL)+", vTR="+str(vTR)+" t="+str(tf))
plt.plot(xc,ux,'r.-')
plt.legend(["Analytic","Python"],loc="best")
plt.ylabel("u_x")
plt.xlabel("x")

plt.figure(4)
plt.plot(rad+0.5,vsol,'b-',lw=2.0)
plt.title("no ppm 64 cells, gamma="+str(gamma)+", vTL="+str(vTL)+", vTR="+str(vTR)+" t="+str(tf))
plt.plot(xc,uy,'r.-')
plt.legend(["Analytic","Python"],loc="best")
plt.ylabel("u_y")
plt.xlabel("x")
plt.show()


