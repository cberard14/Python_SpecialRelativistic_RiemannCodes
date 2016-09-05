import scipy.integrate as sint 
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import fsolve,brentq
import cmath as cm

#Exact Relativistic Riemann Solver

def getsign(N):
    if N < 0.0:
        return -1.0
    if N > 0.0:
        return 1.0
    if N == 0.0:
        return 0.0

def Riemann(gamma,PL,rhoL,uL,vTL,PR,rhoR,uR,vTR):

    if abs(PL-PR) < 1.0e-4 and abs(uL-uR) < 1.0e-4: #and abs(vTL-vTR) < 1.0e-7:
        #print "accessed"
        return PL,uL

    def shock(Pgs,s_sign,Ps,rhos,us,vs):
        """
        Get velocity behind shock wave as a function
        of Pg (Pguess)
        """
        
        es = Ps/((gamma-1.0)*rhos)
        hs = 1.0+es+Ps/rhos

        Ws = 1.0/np.real(cm.sqrt(1.0-(us**2+vs**2)))
        As = (1.0+(gamma-1.0)*(Ps-Pgs)/(gamma*Pgs))
        Bs = -(gamma-1.0)*(Ps-Pgs)/(gamma*Pgs)
        Cs = (Ps-Pgs)*hs/(rhos)-hs**2
        
        # Marti & Muller 2003, eq 31 - positive root is unique, so this approach works
        if (-Bs+np.real(cm.sqrt(Bs**2-4.0*As*Cs)))/(2.0*As) > 0.0:
            hb = (-Bs+np.real(cm.sqrt(Bs**2-4.0*As*Cs)))/(2.0*As)
        elif (-Bs-np.real(cm.sqrt(Bs**2-4.0*As*Cs)))/(2.0*As) > 0.0:
            hb = (-Bs-np.real(cm.sqrt(Bs**2-4.0*As*Cs)))/(2.0*As)
        else:
            hb = 0.0
        
        js = s_sign*np.real(cm.sqrt(-(Ps-Pgs)**2/((2.0*hs/rhos*(Ps-Pgs))-(hs**2-hb**2))))
        
        VSS = ((rhos*Ws)**2*us+s_sign*abs(js)*np.real(cm.sqrt(js**2+rhos**2*Ws**2*(1.0-us**2))))/(rhos**2*Ws**2+js**2)
        WSS = 1.0/np.real(cm.sqrt(1.0-VSS**2))
        v_shock = (hs*Ws*us+(WSS*(Pgs-Ps))/js)*(hs*Ws+(Pgs-Ps)*(WSS*us/js+1.0/(rhos*Ws)))**(-1)
      #  print "vshock",v_shock
        return v_shock

    
    def rf(V,Pi):
        """
        This function returns the form of the 
        rarefaction wave in differential form;
        function "rarefaction(...)" evaluates it
        using ODEINT.
        """
        ui = V[0]
        rhoi = V[1]
        vti = V[2]
        if rhoi <0.0:
            rhoi *= -1.0
        signi = V[3] 
        ei = Pi/((gamma-1.0)*rhoi)
        hi = 1.0+ei+Pi/rhoi 
        if rhoi <= 1.0e-15 or Pi <= 1.0e-15:
            csi = np.sqrt(gamma-1.0)
        else:
            csi = np.sqrt(gamma*Pi/(hi*rhoi)) 
        vi = np.real(cm.sqrt(ui**2+vti**2))
        Wi = 1.0/np.real(cm.sqrt(1.0-vi**2))
        Wx = 1.0/np.real(cm.sqrt(1.0-ui**2))
        ksi = (ui*(1.0-csi**2)+signi*csi*np.real(cm.sqrt((1.0-vi**2)*(1.0-vi**2*csi**2-ui**2*(1.0-csi**2)))))/(1.0-vi**2*csi**2) #x/t PMM3.6
        gi = (vti**2)*((ksi**2)-1.0)/(1.0-ksi*ui)**2 #PMM 3.11
        dvdp = signi/(rhoi*hi*Wi**2*csi*np.real(cm.sqrt(1.0+gi))) # 3.10 Pons Marti Muller 2000 (PMM 3.10)
        drhodp = 1.0/(hi*csi**2)  #PMM 3.5
        dvtdp = -(vti/hi*(1.0+1.0/(gamma-1.0))*(1.0/rhoi-Pi*drhodp/rhoi**2)+(Wi**2*ui*dvdp*vti))/(1.0+Wi**2*vti**2)    
        return [dvdp,drhodp,dvtdp,0.0]

    def rarefaction(Pgr,signr,Pr,rhor,ur,vr):
        """
        Get velocity behind rarefaction wave as a 
        function of Pg (Pguess) by solving an ODE
        using python built-in function ODEINT
        
        Left-moving rarefaction wave needs sign=-1
        
        *NOTE: can't change initial value of ODE solver
        """
        y0 = [ur,rhor,vr,signr] #Initial conditions + propagation sign
        Pvec = [Pr,Pgr] #Initial pressure, evolved pressure
        soln = sint.odeint(rf,y0,[Pr,Pgr],rtol=1e-16)
        v_rarefaction = np.array(soln[:,0])[1] # Return solution at Pg
        return v_rarefaction

  
    def FP(Pgf,PLf,rhoLf,uLf,vTLf,PRf,rhoRf,uRf,vTRf):
        """
        The 0 of this function will give us P*,u*
        """
        if Pgf <= PLf:
            v_int_L = rarefaction(Pgf,-1,PLf,rhoLf,uLf,vTLf)
        else:
            v_int_L = shock(Pgf,-1,PLf,rhoLf,uLf,vTLf)
        if Pgf <= PRf:
            v_int_R = rarefaction(Pgf,1,PRf,rhoRf,uRf,vTRf)
        else:
            v_int_R = shock(Pgf,1,PRf,rhoRf,uRf,vTRf)
        return v_int_L-v_int_R

    #### MAIN CODE #### 
    if rhoR == 0.0:
        return 0.0,float(rarefaction(0.0,-1,PL,rhoL,uL,vTL))
    else:
        # solve for P*,u*
        P1,P2 = 1.0e-5*min(PL,PR),1.0e4*max(PL,PR)
        if FP(P1,PL,rhoL,uL,vTL,PR,rhoR,uR,vTR) < 0.0:
            Pstar = 1e-15
            print "Pstar Min used in Riemann Solver"
        else:
            P1,P2=0.01,800.0
            Pstar = float(brentq(lambda x: FP(x,PL,rhoL,uL,vTL,PR,rhoR,uR,vTR),P1,P2,xtol=1e-16,maxiter=1000))

        if Pstar <= PL:
            ustar = rarefaction(Pstar,-1,PL,rhoL,uL,vTL)
            if ustar != ustar:
                if Pstar > PR:
                    ustar = shock(Pstar,1,PR,rhoR,uR,vTR)
                else:
                    ustar = rarefaction(Pstar,1,PR,rhoR,uR,vTR)
        else:
            ustar = shock(Pstar,-1,PL,rhoL,uL,vTL)
            if ustar != ustar:
                if Pstar > PR:
                    ustar = shock(Pstar,1,PR,rhoR,uR,vTR)
                else:
                    ustar = rarefaction(Pstar,1,PR,rhoR,uR,vTR)        

        if Pstar != Pstar:
            print "Riemann solver sees P* NaN"
        if ustar != ustar:
            print "Riemann solver sees u* NaN"        
        return Pstar,np.real(ustar)

if (__name__ == "__main__"):
    gamma = 5.0/3.0
    PL,rhoL,uL,vL= 1000.0 , 1.0 , 0.0 , 0.99
    PR,rhoR,uR,vR= 0.01, 1.0, 0.0 , 0.99
    print Riemann(gamma,PL,rhoL,uL,vL,PR,rhoR,uR,vR)

