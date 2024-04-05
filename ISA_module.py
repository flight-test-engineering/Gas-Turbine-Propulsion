import numpy as np



# constants
R = 287.053 # [m2/s2/K]
g_SL = 9.80665 # [m/s2]
m2ft = 3.28084
ft2m = 1 / m2ft

# define strata

# troposphere
L = -6.5 / 1000 #K/m
Hc_t_tropo = 36089.24 # [ft]
T_SL = 288.15 # [K]
p_SL = 101325 # [Pa]
rho_SL = 1.225 # [kg/m3]

# stratosphere
Hc_b_strato = Hc_t_tropo + 0.01 # [ft]
Hc_t_strato = 65616.8 # [ft]
T_b_strato = 216.65 # [K]
p_b_strato = 22632.06 # [Pa]
p_t_strato = 5474.88 # [Pa]
delta_b_strato = p_b_strato / p_SL
delta_t_strato = p_t_strato / p_SL
theta_b_strato = T_b_strato / T_SL
rho_b_strato = 0.36392 # [kg/m3]
rho_t_strato = 0.08803 #[kg/m3]
sigma_b_strato = rho_b_strato / rho_SL
sigma_t_strato = rho_t_strato / rho_SL

def delta_non_v(Hc:float)->float:
    '''
    this function calculates 'delta', the ISA pressure ratio, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        'delta'
    '''
   
    if Hc <= Hc_t_tropo:
        return (1 + (L / T_SL) * ((Hc)*ft2m))**(-g_SL / (L * R))
    elif Hc <= Hc_t_strato:
        return delta_b_strato * np.exp(-(g_SL / (R * T_b_strato))*((Hc - Hc_b_strato)*ft2m))
    else:
        raise ValueError("Altitude above stratospheric limit - outside bounds for this function")

delta = np.vectorize(delta_non_v)

        
def p(Hc:float)->float:
    '''
    this function calculates the ISA pressure, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        p: in Pascals
    '''

    return delta(Hc) * p_SL

def theta_non_v(Hc:float)->float:
    '''
    this function calculates 'theta', the ISA temperature ratio, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        'theta'
    
    '''

    if Hc <= Hc_t_tropo:
        return (1 + (L / T_SL) * ((Hc)*ft2m))
    elif Hc <= Hc_t_strato:
        return theta_b_strato
    else:
        raise ValueError("Altitude above stratospheric limit - outside bounds for this function")
        
theta = np.vectorize(theta_non_v)

def T(Hc:float)->float:
    '''
    this function calculates the ISA temperature, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        T: in Kelvin
    
    '''

    return theta(Hc) * T_SL

def sigma_non_v(Hc:float)->float:
    '''
    this function calculates 'sigma', the ISA density ratio, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        'sigma'
    
    '''
    
    if Hc <= Hc_t_tropo:
        return (1 + (L / T_SL) * ((Hc)*ft2m))**(-g_SL / (L * R) - 1)
    elif Hc <= Hc_t_strato:
        return sigma_b_strato * np.exp(-(g_SL / (R * T_b_strato))*((Hc - Hc_b_strato)*ft2m))
    else:
        raise ValueError("Altitude above stratospheric limit - outside bounds for this function")
        
sigma = np.vectorize(sigma_non_v)

def rho(Hc:float)->float:
    '''
    this function calculates the ISA density, for a given pressure altitude
    limited to top of stratosphere
    inputs:
        Hc: in feet
    outputs:
        rho: in kg/m3
    
    '''

    return sigma(Hc) * rho_SL

def inv_delta_non_v(delta:float)->float:
    '''
    this function calculates ISA pressure altitude for a given pressure ratio 'delta'
    limited to top of stratosphere
    inputs:
        delta  [non-dimensional]
    outputs:
        Hc: in feet
        
    
    '''
    
    if delta > delta_b_strato:
        return (T_SL / L) * ((delta)**(-(L * R) / g_SL) - 1) * m2ft
    elif delta >= delta_t_strato:
        return ((((-R * T_b_strato) / g_SL) / ft2m) * np.log(((delta) * np.exp((-g_SL / (R * T_b_strato)) * Hc_b_strato * ft2m)) / (p_b_strato / p_SL)))
    else:
        raise ValueError("Pressure/delta lower than stratospheric limit - outside bounds for this function")

inv_delta = np.vectorize(inv_delta_non_v)

def inv_p(p:float)->float:
    '''
    this function calculates the ISA pressure altitude, for a given pressure
    limited to top of stratosphere
    inputs:
        p: in Pascals
    outputs:
        Hc: in feet
    
    '''

    return inv_delta(p / p_SL)

def inv_sigma_non_v(sigma:float)->float:
    '''
    this function calculates ISA pressure altitude for a given density ratio
    limited to top of stratosphere
    inputs:
        sigma [non-dimensional]
    outputs:
        Hc: in feet
        
    
    '''
    
    if sigma > sigma_b_strato:
        return (T_SL / L)*(((sigma)**(1 / (-g_SL / (L * R) - 1)) - 1))*m2ft #validado
    elif sigma >= sigma_t_strato:
        return ((((R * T_b_strato) / -g_SL) / ft2m) * np.log(((sigma) * np.exp((-g_SL / (R * T_b_strato)) * Hc_b_strato * ft2m)) / (rho_b_strato / rho_SL))) #validado
    else:
        raise ValueError("Density/sigma below stratospheric limit - outside bounds for this function")

inv_sigma = np.vectorize(inv_sigma_non_v)

def inv_rho(rho:float)->float:
    '''
    this function calculates the ISA pressure altitude, for a given density
    limited to top of stratosphere
    inputs:
        rho: in kg/m3
    outputs:
        Hc: in feet
    
    '''

    return inv_sigma(rho / rho_SL)