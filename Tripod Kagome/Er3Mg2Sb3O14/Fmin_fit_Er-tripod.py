#!/usr/local/pacerepov1/anaconda/anaconda3/latest/envs/python-3.6


import os
import platform
import sys
import math
import multiprocessing as mp
from numpy import*
import numpy as np
from scipy import optimize

####################################################################################################   
####################################      Setting up         #######################################
####   1=Ce3+, 2=Pr3+, 3=Nd3+, 4=Pm3+, 5=Sm3+, 6=Tb3+, 7=Dy3+, 8=Ho3+, 9=Er3+, 10=Tm3+, 11=Yb3+,####
#################################################################################################### 
RE= 'Ce3+', 'Pr3+', 'Nd3+', 'Pm3+', 'Sm3+', 'Tb3+', 'Dy3+', 'Ho3+', 'Er3+', 'Tm3+', 'Yb3+'
J_total=[2.5, 4, 4.5, 4, 2.5, 6, 15/2, 8, 15/2, 6, 7/2]
g_J=[6/7,4/5,8/11,5/3, 2/7, 3/2, 4/3, 5/4, 6/5, 7/6, 8/7]
Ion=9

Jtotal=J_total[Ion-1]
Nlevels=int(2*Jtotal+1)
gJ=g_J[Ion-1]


# reading experimenal data 
N_set=6
N_point=[174,174,174,116,116,116]												                                        
Temperature=[5, 50, 100, 5, 50, 100]

Exp_30meV_5K=np.zeros((4,N_point[0]))
Exp_30meV_50K=np.zeros((4,N_point[1]))
Exp_30meV_100K=np.zeros((4,N_point[2]))
Exp_120meV_5K=np.zeros((4,N_point[3]))  
Exp_120meV_50K=np.zeros((4,N_point[4])) 
Exp_120meV_100K=np.zeros((4,N_point[5])) 
EXP=[Exp_30meV_5K, Exp_30meV_50K, Exp_30meV_100K, Exp_120meV_5K, Exp_120meV_50K,  Exp_120meV_100K]

with open('Exp_Er-tripod_30meV_5K.dat','r') as myfile:
	myfile.readline()
	data=myfile.read().split()
	for i in range(N_point[0]):
		Exp_30meV_5K[0][i]=float(data[3*i])                  #Energy
		Exp_30meV_5K[1][i]=float(data[3*i+1])				 #Intensity
		Exp_30meV_5K[2][i]=float(data[3*i+2])				 #Error
myfile.close
with open('Exp_Er-tripod_30meV_50K.dat','r') as myfile:
	myfile.readline()
	data=myfile.read().split()
	for i in range(N_point[1]):
		Exp_30meV_50K[0][i]=float(data[3*i])                  #Energy
		Exp_30meV_50K[1][i]=float(data[3*i+1])				 #Intensity
		Exp_30meV_50K[2][i]=float(data[3*i+2])				 #Error
myfile.close
with open('Exp_Er-tripod_30meV_100K.dat','r') as myfile:
	myfile.readline()
	data=myfile.read().split()
	for i in range(N_point[2]):
		Exp_30meV_100K[0][i]=float(data[3*i])                  #Energy
		Exp_30meV_100K[1][i]=float(data[3*i+1])				 #Intensity
		Exp_30meV_100K[2][i]=float(data[3*i+2])				 #Error
myfile.close
with open('Exp_Er-tripod_120meV_5K.dat','r') as myfile:
	myfile.readline()
	data=myfile.read().split()
	for i in range(N_point[3]):
		Exp_120meV_5K[0][i]=float(data[3*i])                  #Energy
		Exp_120meV_5K[1][i]=float(data[3*i+1])				 #Intensity
		Exp_120meV_5K[2][i]=float(data[3*i+2])				 #Error
myfile.close
with open('Exp_Er-tripod_120meV_50K.dat','r') as myfile:
	myfile.readline()
	data=myfile.read().split()
	for i in range(N_point[4]):
		Exp_120meV_50K[0][i]=float(data[3*i])                  #Energy
		Exp_120meV_50K[1][i]=float(data[3*i+1])				 #Intensity
		Exp_120meV_50K[2][i]=float(data[3*i+2])				 #Error
myfile.close
with open('Exp_Er-tripod_120meV_100K.dat','r') as myfile:
	myfile.readline()
	data=myfile.read().split()
	for i in range(N_point[5]):
		Exp_120meV_100K[0][i]=float(data[3*i])                  #Energy
		Exp_120meV_100K[1][i]=float(data[3*i+1])				 #Intensity
		Exp_120meV_100K[2][i]=float(data[3*i+2])				 #Error
myfile.close

#EXPEnergy=[0, 0, 6.5, 6.5, 10.5, 10.5, 21.4, 21.4, 48.5, 48.5, 50.8, 50.8,  65, 65, 67.5, 67.5]  #Er-tripod
EXPEnergy=[0, 0, 6.5, 6.5, 10.5, 10.5, 21.4, 21.4, 50.0, 50.0, 61, 61, 65, 65, 67.5, 67.5]  #Er-tripod

####################################################################################################   
################################ Run simpr.f and read simpr.out  ####################################
#################################################################################################### 
def simpre_job(R1, R2, R3, Theta2, Theta3, Phi, e1, e2, e3):
    if os.path.exists("simpre.dat"):	
        os.remove("simpre.dat")
        
    string="""1=Ce3+, 2=Pr3+, 3=Nd3+, 4=Pm3+, 5=Sm3+, 6=Tb3+, 7=Dy3+, 8=Ho3+, 9=Er3+, 10=Tm3+, 11=Yb3+, 12=user

 """+str(Ion)+"""    !ion code (from 1 to 12, see above)
  8    !number of effective charges
  1    """+str("{0:.7f}".format(R1))+"""    0.0000000    0.0000000    """+str("{0:.5f}".format(e1))+"""
  2    """+str("{0:.7f}".format(R1))+"""  180.0000000    0.0000000    """+str("{0:.5f}".format(e1))+"""
  3    """+str("{0:.7f}".format(R2))+"""   """+str("{0:.7f}".format(Theta2))+"""   """+str("{0:.7f}".format(Phi))+"""    """+str("{0:.5f}".format(e2))+"""
  4    """+str("{0:.7f}".format(R2))+"""   """+str("{0:.7f}".format(Theta2))+"""  """+str("{0:.7f}".format(360-Phi))+"""    """+str("{0:.5f}".format(e2))+"""
  5    """+str("{0:.7f}".format(R2))+"""  """+str("{0:.7f}".format(180-Theta2))+"""   """+str("{0:.7f}".format(180-Phi))+"""    """+str("{0:.5f}".format(e2))+"""
  6    """+str("{0:.7f}".format(R2))+"""  """+str("{0:.7f}".format(180-Theta2))+"""   """+str("{0:.7f}".format(180+Phi))+"""    """+str("{0:.5f}".format(e2))+"""
  7    """+str("{0:.7f}".format(R3))+"""  """+str("{0:.7f}".format(180-Theta3))+"""    0.0000000    """+str("{0:.5f}".format(e3))+"""
  8    """+str("{0:.7f}".format(R3))+"""   """+str("{0:.7f}".format(Theta3))+"""  180.0000000    """+str("{0:.5f}".format(e3))+"""
  
  
  """
    with open("simpre.dat", "w+") as f1:
        print(string, file=f1)
    f1.close
    os.system("./3")

#get the energy and wave-function from output_matrix file using string 'Eigenvalues','modulus'
def check(Energy, WF, str1, str2, Nlevels):   
    with open('simpre.out') as datafile:
        i=0 
        for line in datafile:
            if str(str1) in line:
                for line in datafile:
                    if str(str2) in line:
                        datafile.close()
                        return
                    ev=line.strip().split()
                    if len(ev)==Nlevels+1:
                        Energy[i]=float(ev[0])
                        #print(i,' ',Energy[i],' ')
                        for j in range(Nlevels):
                            WF[i][j]=float(ev[j+1])
                            # print(WF[i][j],' ')
                        i=i+1
    datafile.close

####################################################################################################   
####################################     Intenisty Matrix function     #############################
#################################################################################################### 
#define three angular momentum operator	
def Jz(vector):
    vector2=np.linspace(0,0,Nlevels)
    for j in range(Nlevels):
        m=j-Jtotal
        vector2[j]=m*vector[j]
    return vector2
def Jp(vector):
    vector2=np.linspace(0,0,Nlevels)
    for j in range(Nlevels-1):
        m=j-Jtotal
        vector2[j+1]=math.sqrt((Jtotal-m)*(Jtotal+m+1))*vector[j]
    return vector2
def Jm(vector):
    vector2=np.linspace(0,0,Nlevels)
    for j in range(1,Nlevels):
        m=j-Jtotal
        vector2[j-1]=math.sqrt((Jtotal+m)*(Jtotal-m+1))*vector[j]
    return vector2
def Jx(vector):
    return np.true_divide(np.add(Jp(vector),Jm(vector)),2)
def Jy(vector):
    return np.true_divide(np.subtract(Jp(vector),Jm(vector)),2*1j)

def Intensity(i,ip,WF, Energy): #each transition from ground state has 4 contributions with xyz component
    WFJzi=Jz(WF[i])
    WFJpi=Jp(WF[i])
    WFJmi=Jm(WF[i])
    Int=0
    # i to ip
    Int_Jx0=Int_Jy0=Int_Jz0=0
    for j in range(Nlevels):   
        Int_Jx0=Int_Jx0+WF[ip][j]*(WFJpi[j]+WFJmi[j])/2
        Int_Jy0=Int_Jy0+WF[ip][j]*(WFJpi[j]-WFJmi[j])/2/1j
        Int_Jz0=Int_Jz0+WF[ip][j]*WFJzi[j]
    Int=Int+abs(Int_Jx0)*abs(Int_Jx0)
    Int=Int+abs(Int_Jy0)*abs(Int_Jy0)
    Int=Int+abs(Int_Jz0)*abs(Int_Jz0)
    return abs(Int)

####################################################################################################   
####################################     Evaluate pattern  function     ###############################
#################################################################################################### 
def Resolution(Set, E):
	if Set<3:
		if E<5:
			return 1.3
		elif 5<E<9:
			return 1.5
		elif 9<E<18:
			return 1.1
		else:
			return 1.6
	else:
		if E<50.5:
			return 5.0
		else:
			return 3.2
"""		else:		
		if E<30:
			return 3.5	
		else:
			return 5.7-0.039*E
"""
def Evaluate_pattern(Set,Nlevels,Energy,IntensityMatrix,T):
    Calc=np.linspace(0,0,N_point[Set]) 

    for i in range(Nlevels):
        for j in range(Nlevels):
            if i<j and abs(Energy[i]-Energy[j])>0:      
                Convolution=np.linspace(0,0,N_point[Set])
                gamma=Resolution(Set, Energy[j]-Energy[i])      #resolution
                for k in range(N_point[Set]):
                    Convolution[k]=gamma/(math.pow(EXP[Set][0][k]-Energy[j]+Energy[i],2)+gamma*gamma/4)
                Calc=Calc+IntensityMatrix[i][j]*math.exp(-Energy[i]*11.602/T)*Convolution
                
    Max_Intensity=max(Calc)
    Calc=Calc/Max_Intensity
    
    return Calc




####################################################################################################   
####################################     Calculate and Write        ###############################
#################################################################################################### 
def simpre_Chi2(x0,Control):

    [R1,R2,R3,Theta2, Theta3, Phi, e1, e2, e3]=x0
    #[R1,R2,Theta,e2]=x0
    #[R1,R2,Theta]=x0
    #R2=R1
    #e1=0.3
    #e2=0.333

    simpre_job(R1, R2, R3, Theta2, Theta3, Phi, e1, e2, e3)                           #run the simpre.f

    Energy=np.linspace(0,0,Nlevels)   #Energy
    WF=np.zeros((Nlevels,Nlevels))    # Wave function
    IntensityMatrix=np.zeros((Nlevels,Nlevels))      # Transition Matrix
    
    Chi_total=0
    Chi_30meV=0
    Chi_120meV=0	
    Chi_energy=0
    Chi=linspace(0,0,N_set)
    
    
    check(Energy, WF, 'Eigenvalues', 'modulus', Nlevels)   #Input Wave function, Energy

    
    for i in range(Nlevels):                                #Calculate transition Matrix
        if EXPEnergy[i]>0:
            Chi_energy=Chi_energy+abs(Energy[i]-EXPEnergy[i])
        for j in range(Nlevels):
            IntensityMatrix[i][j]=Intensity(i,j,WF,Energy)
        
    for i in range(N_set):
        Calc=np.zeros((2,N_point[i]))
        Calc[0]=EXP[i][0]
        Calc[1]=Evaluate_pattern(i,Nlevels,Energy,IntensityMatrix,Temperature[i])

        temp=(Calc[1]-EXP[i][1])/abs(EXP[i][2])
        Chi[i]=1/N_point[i]*temp.dot(temp)/100

        #temp=(Calc[1]+EXP[i][2])/(abs(EXP[i][1])+EXP[i][2])
        #Chi[i]=1/N_point[i]*abs(Calc[1]-EXP[i][1]).dot(temp+1/temp-2)		
        
        if Control==1: 
            with open('output_Er-tripod.dat','a') as f:	     #Write I(E)
                f.write('\n')	
                f.write('Chi2=')
                f.write("{0:.3f}".format(Chi[i]))
                f.write('\n\n')	
                f.write(str(np.transpose(Calc)).replace('[','').replace(']',''))	
                f.write('\n')
            f.close()
        if i<3: 
            Chi_30meV=Chi_30meV+Chi[i]
        else: 
            Chi_120meV=Chi_120meV+Chi[i]

    Chi_total=(Chi_30meV+Chi_120meV)*(0*Chi_energy+1)/100
    
    if Control==1:                                         #Write Matrix
        with open('output_Er-tripod.dat','a') as f:
            f.write('\n\n')	
            r_list=["{0:.3f}".format(Chi_energy),"{0:.3f}".format(Chi_30meV),"{0:.3f}".format(Chi_120meV), RE[Ion-1],"{0:.3f}".format(R1), "{0:.3f}".format(R2), "{0:.3f}".format(R3), "{0:.3f}".format(Theta2),
                    "{0:.3f}".format(Theta3), "{0:.3f}".format(Phi), "{0:.3f}".format(e1),"{0:.3f}".format(e2), "{0:.3f}".format(e3),'Eigen-energy    ']
            f.write(' '.join(r_list))
            for i in range(Nlevels):
                f.write("{0:.2f}".format(Energy[i]))
                f.write(' ')
            f.write('\n\n')
            #Write Intensity Matrix 
            f.write('|<n/J/m>|2')
            for i in range(Nlevels):
                f.write("{0:8.3f}".format(Energy[i]))
            f.write('\n')
            for i in range(Nlevels):
                f.write("{0:7.3f}".format(Energy[i]))
                f.write('   ')
                for j in range(Nlevels):
                    IntensityMatrix[i][j]=Intensity(i,j,WF,Energy)
                    f.write(' ')
                    f.write("{0:7.3f}".format(IntensityMatrix[i][j]))
                f.write('\n')

            f.write('\n\n')
            np.set_printoptions(precision=5,suppress=True,threshold=sys.maxsize)  #use threshold to control max output length

            with open('simpre.out') as datafile:
                datasimpre = datafile.read()
            datafile.close()
            f.write(datasimpre)
        f.close()
        
    return Chi_total

####################################################################################################   
####################################     calc-gtensor            ###############################
#################################################################################################### 
def gtensor():
    
    Energy=np.linspace(0,0,Nlevels)   #Energy
    WF=np.zeros((Nlevels,Nlevels))    # Wave function
    IntensityMatrix=np.zeros((Nlevels,Nlevels))      # Transition Matrix
    check(Energy, WF, 'Eigenvalues', 'modulus', Nlevels)   #Input Wave function, Energy
    
    eigenvec1=WF[0]
    eigenvec2=WF[1]
    Sx=[[eigenvec1.dot(Jx(eigenvec1)),eigenvec1.dot(Jx(eigenvec2))],
        [eigenvec2.dot(Jx(eigenvec1)),eigenvec2.dot(Jx(eigenvec2))]]
    Sy=[[eigenvec1.dot(Jy(eigenvec1)),eigenvec1.dot(Jy(eigenvec2))],
        [eigenvec2.dot(Jy(eigenvec1)),eigenvec2.dot(Jy(eigenvec2))]] 
    Sz=[[eigenvec1.dot(Jz(eigenvec1)),eigenvec1.dot(Jz(eigenvec2))],
        [eigenvec2.dot(Jz(eigenvec1)),eigenvec2.dot(Jz(eigenvec2))]]
    g=np.multiply(np.matrix([[Sx[1][0].real, Sx[1][0].imag, Sx[0][0].real],
       [Sy[1][0].real, Sy[1][0].imag, Sy[0][0].real],
       [Sz[1][0].real, Sz[1][0].imag, Sz[0][0].real]]), 2*gJ)
    return g

def diagonalG():
    def Ry(theta):
        return array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    def Pesduospinrotation(theta):
        A=g.dot(np.linalg.inv(Ry(theta)))
        B=A-A.T
        return B.item((2, 0))
		
    g=gtensor()	
    x=optimize.fsolve(np.vectorize(Pesduospinrotation),0)      
    g2=dot(g,np.linalg.inv(Ry(x[0])))

    def Realspinrotation(theta):
        A=Ry(theta).dot(g2.dot(np.linalg.inv(Ry(theta)))) 
        return A.item((0, 1))+A.item((0, 2))
    Theta=optimize.fsolve(np.vectorize(Realspinrotation),0)[0]
    A=np.matrix(Ry(Theta))
    Theta_degree=Theta*180/pi
    g_diagonal=Ry(Theta).dot(g2.dot(np.linalg.inv(Ry(Theta))))

    print(' Rotational matrix = ', '\n', A,'\n', 'Rotation angle =',  Theta_degree,'\n')
    print('g_diagnal = ','\n', g_diagonal)
    
    return A,Theta_degree,g_diagonal
	
####################################################################################################   
####################################     Main        ###############################
#################################################################################################### 
#Distance
R1_0=1.72
R2_0=1.55
R3_0=1.47
#Angle  
Theta2_0=80.
Theta3_0=75.5
Phi_0=59
#Charge 
e1_0=0.52
e2_0=0.31
e3_0=0.18


def main():

    x0=[R1_0, R2_0, R3_0, Theta2_0, Theta3_0, Phi_0, e1_0, e2_0, e3_0] 

    #minimum=x0
    minimum=optimize.fmin(simpre_Chi2, x0,args=(0,))
    print(minimum)
    
    if os.path.exists("output_Er-tripod.dat"):
        os.remove("output_Er-tripod.dat")
    with open('output_Er-tripod.dat','a') as f:
        f.write('Chi2_total=') 
        f.write("{0:5.6f}".format(simpre_Chi2(minimum,0))) 
        f.write('      ') 
        f.write('values= ') 
        f.write(str(minimum)) 
        f.write('\n') 
        np.set_printoptions(precision=4,suppress=True)
        f.write(str(gtensor()))
        f.write('\n')
        A,Theta_degree,g_diagonal=diagonalG()
        f.write('g_diagonal=')
        f.write(str(g_diagonal))
        f.write('\n')
        f.write('roration_along_y=')
        f.write(str(Theta_degree))
        f.write('\n')
        
    f.close()
    simpre_Chi2(minimum,1)


main()

