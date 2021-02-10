# PointChargeCEF
A python3 program to analyze crystal field excitations using effective point charges.

Created by Zhiling Dun (dunzhiling@gmail.com)

PointChargeCEF calculates crystal electric field (CEF) excitations of rare earth systems and performs fits to experimental inelastic neutron scattering spectrum using adjustable effective point-charges. 

This program was originally developed to analyze the crystal field excitations and single-ion magnetism of the [tripod kagome magnets](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.157201), R3Mg2Sb3O14 (R = rare earth elements), where the determination of CEF parameters using conventional Stevens Operators approach is  challenging.

This repository contains all of the code and data that is used for our manuscirpt at [arXiv:2004.10957]( https://arxiv.org/abs/2004.10957). 


<img src="https://render.githubusercontent.com/render/math?math=     ">
## Mathematics & Logic of Program
1. The CEF Hamiltonian takes the form:  <img src="https://render.githubusercontent.com/render/math?math=\mathcal{H}_{\mathrm{CEF}} = \sum_{n,m} \left[ A^{m}_{n}  \theta_n \right] O^m_n= \sum_{n,m}B^{m}_{n} O^m_n.">

    
2. The point charge calculation is performed using existing software package [SIMPRE](https://pubmed.ncbi.nlm.nih.gov/24000391/), which calcualtes CEF paramers using  <img src="https://render.githubusercontent.com/render/math?math= B^{m}_{n} = -\sum_{i} C^{m}_{n}\theta_n \langle r^n \rangle \gamma^{nm}_i q_i">

3. We take number for <img src="https://render.githubusercontent.com/render/math?math=B^{m}_{n}"> from the output file simpre.out and calcualte the eigenstates (Ei) and eigenenergies (<img src="https://render.githubusercontent.com/render/math?math=\left|\Gamma_i\right\rangle">).

4. We calcualte the inelastic neutron scattering spectrum using: <img src="https://render.githubusercontent.com/render/math?math=I(\omega)$= $C \sum_{n, m} \frac{\sum_{\alpha=x,y,z}\left|\left\langle \Gamma_n\left|J_{\alpha}\right| \Gamma_{m}\right\rangle\right|^{2} \mathrm{e}^{-E_{n}/k_\mathrm{B}T}}{\sum_{j} \mathrm{e}^{-E_{j}/k_\mathrm{B}T}} \times\delta(\hbar\omega + E_n - E_m)     ">.

5. The $\delta$ function above is convoluted to a Voigt function 
<img src="https://render.githubusercontent.com/render/math?math= V(\omega ; \sigma_G, \gamma_L) \equiv   \int_{-\infty}^{\infty}   G\left(x ; \sigma_G\right) L\left(\hbar\omega + E_n - E_m-x ; \gamma_L\right) d x$,  "> 
where G is a Gaussian function to account for the  energy-resolution ($\sigma_G$)  of neutron scattering spectrometer, and $L$ is a Lorentzian function with $\gamma_L$ representing the intrinsic broadening (or finite lifetime) of CEF excitations.  We are now using the resolution for SEQUOIA, if you want to change the instrument resolution, one need to redefine function `PointchargeCEF.Instrument_resolution(x, Ei)`

6. By varying the point charge parameters, a least-squares fit is performed to minimize the difference between calculated and observed CEF spectra. The agreement is measured by a self-defined weighted profile factor: 
$R_{\mathrm{wp}} =  \frac{1}{N} \sqrt{\sum_{i} \left(\frac{I_i^{\text{calc}}-I_i^{\text{obs}}}{\sigma_i^\text{obs}}\right)^2}$

7. Static magnetic properties in an external magnetic field $\mathbf{H}$ can also be calculated from the single-ion CEF Hamiltonian,$\mathcal{H} =  \mathcal{H}_{\mathrm{CEF}} - \mu_{B} g_{J}  \mathbf{H} \cdot \mathbf{J}$.
    With the eigenstate ($E_{n}$) and eigenfunction ($\left|\Gamma_i\right\rangle$) of the CEF Hamiltonian available, the three components ($\alpha=x,y,z$) of the magnetization $\mathbf{M}^\text{CEF}(\mathbf{H}, T)$ in a Cartesian coordinate system are given by
$M^\text{CEF}_\alpha(\mathbf{H}, T) = g_J\sum_{n} e^{-\frac{E_n}{k_{\rm B} T}} \left\langle n\left|J_\alpha\right|n\right\rangle /\sum_{n} e^{-\frac{-E_n}{k_{\rm B} T}}, 
$, from which the DC magnetic susceptibility tensor can be calculated numerically following $\chi_{\alpha\beta}  =\frac{\partial M_{\alpha}}{\partial H_{\beta}}$. 

## How to Use
### (A) Usage
To run this program, you must have a Python3 environment.  A Jupyter notebook is required to open the examples in ".ipynb" format. The SIMPRE file has been compiled as Unix executable files, 'run_simpre_sphere' or 'run_simpre_cartesion', which run on Linux or MAC OS system. If you want to use this program, please ask José J. Baldoví (J.Jaime.Baldovi@uv.es)  for permission to use [SIMPRE](https://pubmed.ncbi.nlm.nih.gov/24000391/). 

### (B) Get started
1. add the downloaded master file 'PointChargeCEF.py' into your system path add import it, e.g. as 'CEF':

    `sys.path.append("[your path]") 
    import PointChargeCEF as CEF`

2. copy the 'run_simpre_sphere' file into your working directory.  

3. creat a ionic crystal field class using function PointChargeCEF.CEFmodel($R^{3+}$), e.g.

     `HTO=CEF.CEFmodel('Ho3+') `
     
### (C) Create a point charge model

1. define effective point charge variables. Take Ho$_2$Ti$_2$O$_7$ for example (
see R2Ti2O7/R2Ti2O7.ipynb), for the eight effective point charges, there are two distance rariables 'R1', 'R2' , one angluar variable 'Theta', and  two charge amount variable 'q1', 'q2': 

    `HTO.PC_variable = 'R1', 'R2', 'Theta', 'q1', 'q2'`

2. add each point charge in spherical coordination, for Ho$_2$Ti$_2$O$_7$, they are:

    `HTO.addPC('R1, 0, 0, q1')` <br />
    `HTO.addPC('R1, 180, 0, q1')` <br />
    `HTO.addPC('R2, 180-Theta,   0, q2')` <br />
    `HTO.addPC('R2,     Theta,  60, q2')` <br />
    `HTO.addPC('R2, 180-Theta, 120, q2')` <br />
    `HTO.addPC('R2,     Theta, 180, q2')` <br />
    `HTO.addPC('R2, 180-Theta, 240, q2')` <br />
    `HTO.addPC('R2,     Theta, 300, q2')`

3. give intinal values for the point charge variables, in the unit Å, degree, and electron charge:

    `HTO.PC_value_ini=np.array([1.594, 1.594, 79.4, 0.5, 0.333])`
    
### (D) CEF calculations
1. To perform a point charge calculation, go with function PointChargeCEF.simpre(), e.g.

    `HTO.simpre()`

    the progame will first creat a "simpre.dat" file based on the information you put in (C), and then run executable file "run_simpre_sphere". You will get the output file from "simpre.out" based on the standard processure of SIMPRE. 

2. One can read the values of CEF parameters for Stevens Operators from "simpre.out" or any specific file with function PointChargeCEF.readBkq(), e.g. 

    `HTO.readBkq()`
    
    About this function:
        PointChargeCEF.readBkq(filename='simpre.out', unit='meV', convention='Steven', printcontol='yes'), 
        Parameters
        [filename]: file to read. 
        [unit]:   unit of CEF parameters, either 'meV' or 'K' 
        [convetion]: either 'Steven' or 'Wybourne' 
        [printcontol]: if chose 'yes', print the values for k, q, Bkq
    
    
3. To calculate a inelastic neutron scattering pattern of CEF excitations, use function PointChargeCEF.Evaluate_pattern(), e.g. 

    `HTO.Evaluate_pattern(Ei=80, Temperature=5, Plotcontrol=True, UsePCini=True);`
    
    About this function:
        PointChargeCEF.Evaluate_pattern(Ei = 160, Temperature = 5, dataset='', Field=[0,0,0], FWHM=0.5, SpecialFWHM=[0,0,0], UsePCini=False, Plotcontrol=False, Chi2Method='linear'): 
        
        Parameters
        [Ei]: incident energy of neutron. 
        [Temperature]:   tempearture of experiment in the unit of Kelvin.
        [dataset]: index for experimental dataset. If given, use E$_i$, Temperature, and neutron energy transfer from that dataset. 
        [UsePCini]: control parameter, if this value = True, use intitial values of effective point charge parameters to evaluate the pattern; if this value = False, use the Stevens Opeartors (k, q, B$_{kq}$) from fitted to point charge parameters to evaluate the pattern.
        [Field]: vector, external magnetic field. 
        [FWHM]: full-width-half maximum of the Lorentian function, cossponding to the intrinsic broadening of CEF excitations.
        [plotcontol]: if this value = True, creat a plot for the scattering spectrum.
        [Chi2Method]:  control parameter with two options, 'linear', or 'log'. The differnce between calcualted and measured intensities. 
        
        Returns
        [xaxis,Calc]: ndarray, calculated energies and intensities
        


### (E) Effective point charge fit
1. Define experimental data. We take $\mathrm{Yb_3Mg_2Sb_3O_{14}}$  as an exmaple here (see Tripod Kagome/Yb3Mg2Sb3O14/YMSO_CEF.ipynb). <br />
    First we creat an CEFmodel:
    
    `YMSO=CEF.CEFmodel('Yb3+')`
    
    Then, load inelastic dataset, in the format of, [E, I(E), error]
    
    `Exp_1 = np.loadtxt("Exp_Yb-tripod_240meV_5K.dat", skiprows=10)`

    Add the file to CEFmodel with function PointChargeCEF.addINSdata() with specific incident neutron energy Ei, Temperature. If some of the observed exictations are too broad, you can define it with SpecialFWHM=[begin_energy, end_energy, FWHM], e.g.
    
    `YMSO.addINSdata(Exp_1, Ei = 240, Temperature = 5, SpecialFWHM=[88,92, 3])`
    
    One should add observed energy levels as well for fitting purpose. This is used to creat a smooth  $\chi^2$ function for quick converngence. One need to input an array with length of 2J+1 (J is the total angular momentum of R$^{3+}$ ion), use 0 for uncertain levels, e.g.
    
    `YMSO.levels_obs=[0, 0, 69, 69, 89, 89, 114, 114] `
    
2. To perform an effective point charge fit, go with function PointChargeCEF.PCfit(), one can specific what paramters to fit, Targeted $\chi^2$ for the eigenenergies, e.g. 

    `YMSO.PCfit(Fit_variable=['R1', 'R2', 'R3', 'FWHM'])`

    Here, PCfit uses a spefific Search Mearthod to evaluate function PointChargeCEF.Chi2_INS(), which is defined to evaluate the least square error between calcualted and observed pattern, the $\chi^2$ is weighted by a factor of  max(1,Chi2_energy/TargetChi2energy).

    About this function:
        PointChargeCEF.PCfit( Fit_variable = 'All', SearchMethod ='Nelder-Mead', Chi2Method='linear', Bonds= None, Tolfun=1e-3, TargetChi2energy = 1.0):  
        
        Parameters
        [Fit_variable]: define what paramters to fit, default= 'All', meaning all the PC_variable plus 'FWHM'. 
        [SearchMethod]:   'Nelder-Mead' or 'cma'
        [Chi2Method]: either 'linear' intensity or 'log' intensity 
        [Bonds]: control the range of Fit_variable, only applicable when Chi2Method = 'cma'
        [Tolfun]: Absolute error in Chi2_INS between iterations that is acceptable for convergence.
        [TargetChi2energy]: targeted 𝜒2 error for the CEF eigenenergies.

        Returns
        [OptimizeResult]: standard returns from scipy.optimize.minimize
   
   
3. check results

    To plot fitted pattern: `YMSO.Evaluate_pattern(dataset = 0, Plotcontrol=True);`
    
    To check eigenstates: `YMSO.eigensys()[0]`
    
    To chekc eigensystem: `YMSO.eigensys()[1]`
    
    To get g-tensor in defined xyz-axis: `YMSO.gtensor()`
    
    To get diagnalized g-tensor: `YMSO.diagonalG()`;
    
    To write all the fitted results to file: `YMSO.writefile("Filename.dat");` 

### (F) Calcualte susceptibility & magnetization 
1. To calculate a Magnetization at certian tempearture, and exeternal magnetic field H_ext=[Bx, By, Bz] in the unit of Tesla. Use function `PointchargeCEF.Magnetization(Temperature, H_ext, Weiss_field=0.0)`.

2. To calculate a powder-averaged Magnetization curve, we can define a field range and do e.g.

    `H = np.linspace(0,14,10)
YbMSO.Powder_Magnetization(dataset=0, B_range=H, intergration_step=10)`

3. To calculate a Susceptibility at certian tempearture, and exeternal magnetic field strength B, use function `PointchargeCEF.Susceptibility(B, Temperature, Weiss_field=0.0)`, it return the susceptibility in the format of [𝜒_powder, 𝜒_xx, 𝜒_yy, 𝜒_zz]. To calcualte the powder invere-susceipibility within a given range, one can go with PointchargeCEF.Powder_InverseSusceptibility(), e.g.

    `YMSO.Powder_InverseSusceptibility(B=0.1, Temperature_range=np.arange(2,300,2))`
    
4. To apply a Weiss-field correction (λ), one can add the option 'Weiss_field= λ' in the three functions above. Because the powder intergation with the Weiss field correction is slow, one is encouraged to define a intergration_step, e.g.

    `YMSO.Powder_Magnetization(dataset=0, B_range= H, Weiss_field = 0.3 ,intergration_step=10)`
    
### (G) Other tools
1. Conventional Stevens Operators fit <br />
    With intial values of CEF parameters give, the conventional Stevens Operator fit can be done within a single line, e.g. for the example of  TmMgGaO4/TMGO.ipynb
    
    `TMGO.StevenOpfit() `
    
2.  Convert Stevens Operators coeiffients from Stevens convention ($𝐵_𝑘^𝑞$) to Wybourne convention ($𝐵_q^k$), e.g.

    `PointchargeCEF.BkqtoAkq(TMGO.Bkq,'Tm3+')`
    
3. Convert between $𝐵_𝑘^𝑞$ and $A_𝑘^𝑞$
    
    use function `PointchargeCEF.BkqtoAkq(Bkq, Ion)` or `PointchargeCEF.AkqtoBkq(Akq, Ion)`
    
4. Convert CEF parameters from one ion to another rare earth ion

    use function `ConvertCEFparamters(Bkq, Ion_from, Ion_to)`
