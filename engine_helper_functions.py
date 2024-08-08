import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [12, 7]
import cantera as ct
import ISA_module as ISA

# helper functions -> move to separate file

def get_p(ps:float, 
          gamma:float, 
          M:float)->float:
    '''
    This function calculates the stagnation pressure for isentropic process, given:
    
    ps: static pressure
    gamma: cp/cv for the gas
    M: Mach number
    '''
    return ps * ((1 + ((gamma - 1) / 2) * M**2)**(gamma / (gamma - 1)))


def get_T(Ts:float, 
          gamma:float, 
          M:float)->float:
    '''
    This function calculates the stagnation temperature for isentropic process, given:
    
    Ts: static temperature
    gamma: cp/cv for the gas
    M: Mach number
    '''
    return Ts * (1 + ((gamma - 1) / 2) * M**2)



def get_Ts(T:float, 
           gamma:float, 
           M:float)->float:
    '''
    This function calculates the static temperature for isentropic process, given:
    
    T: stagnation or toal temperature
    gamma: cp/cv for the gas
    M: Mach number
    '''
    return T / (1 + ((gamma - 1) / 2) * M**2)



def get_ps(p:float, 
           Ts:float, 
           Tt:float, 
           gamma:float)->float:
    '''
    This function calculates the static pressure for isentropic process, given:
    
    p: stagnation pressure
    Ts: static temperature
    Tt: stagnation temperature
    gamma: cp/cv for the gas
    '''
    return p * (Ts / Tt)**(gamma / (gamma - 1))



def get_gamma(gas:ct.Solution)->float:
    '''
    This function calculates gamma, given:
    
    gas: Cantera Solution object
    '''
    return gas.cp / gas.cv



def get_R(gas:ct.Solution)->float:
    '''
    This function calculates the gas constant, given:
    
    gas: Cantera Solution object
    '''
    return gas.cp - gas.cv



def get_a(gas:ct.Solution)->float:
    '''
    This function calculates the local speed of sound, given:
    
    gas: Cantera Solution object
    '''
    return np.sqrt(get_R(gas) * get_gamma(gas) * gas.T)



def print_prop(gas:ct.Solution, endchar:str="\n"):
    '''
    This function prints the properties for a given station
    
    gas : Cantera object with gas conditions
    '''
    
    print(f'p = {(gas.P / ct.one_atm):2.2f} atm,',
      f'T = {(gas.T):2.0f} K,',
      f'h = {(gas.enthalpy_mass):2.0f} kJ/kg,',
      f's = {(gas.entropy_mass):2.0f} kJ/kg, ', end=endchar)
    
        
def print_total_prop(gas:ct.Solution, M:float):
    '''
    This function prints the total properties for a given station
    
    gas : Cantera object with gas conditions
    M : Mach number
    '''
    gamma = get_gamma(gas)
    print(f'p0 = {(get_p(gas.P, gamma, M) / ct.one_atm):2.2f} atm,',
      f'T0 = {(get_T(gas.T, gamma, M)):2.0f} K,')

    
def print_stations(st:list, station_names:dict, gas:dict, M:dict):
    '''
    This function prints the properties of all stations
    
    st : list with stations
    station_names : dictionary with station names
    gas : dictionary with gas objects
    M : dictionary with Mach numbers
    '''
    
    for station in st:
        print(f'station {station:<2} ({station_names[station]:<16}): Mach {M[station]:.3f}, ', end="")
        print_prop(gas[station])


def print_stations_total(st:list, station_names:dict, gas:dict, M:dict):
    '''
    This function prints the properties of all stations
    
    st : list with stations
    station_names : dictionary with station names
    gas : dictionary with gas objects
    M : dictionary with Mach numbers
    '''
    
    for station in st:
        print(f'station {station:<7} ({station_names[station]:<22}): Mach {M[station]:.3f}, ', end="")
        print_prop(gas[station], endchar="")
        print_total_prop(gas[station], M[station])
        
def plot_T_s(T:list, p:list, X:list, reaction_mechanism, phase_name):
    '''
    This function plots T-s states with isobars
    
    inputs
    T    : list with temperatures
    p    : list with pressures
    X    : list with gas composition

    
    returns
    fig : matplotlib figure
    '''

    to_st = len(T)
    dummy_gas = (ct.Solution(reaction_mechanism, phase_name))

    cycle_T = [0,0]
    cycle_s = [0,0]
    
    fig, ax = plt.subplots()

    for i in range(to_st):
        dummy_gas.TPX = T[i], p[i], X[i]
        
        # ISOBARS
        # create the T and s vectors for plotting the isobars
        curve_P = p[i]
        T_min = int(T[i]) - 100
        T_max = int(T[i]) + 400
    
        n = T_max - T_min
        s_isobar_data = []
        T_isobar_data = []

        for curve_T in range(T_min, T_max, 1):
            dummy_gas.TP = curve_T, curve_P
            s_isobar_data.append(dummy_gas.s)
            T_isobar_data.append(dummy_gas.T)
        ax.plot(s_isobar_data, T_isobar_data, linewidth=1, color='#31edd8', alpha=0.35)
        
        # State points
        # create point pairs to be able to trace a line
        dummy_gas.TPX = T[i], p[i], X[i]
        cycle_T[0] = cycle_T[1]
        cycle_s[0] = cycle_s[1]
        cycle_T[1] = dummy_gas.T
        cycle_s[1] = dummy_gas.entropy_mass
        if i != 0:
            ax.plot(cycle_s, cycle_T, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=4)
        

           
    ax.set_title('T-s Diagram')
    ax.set_xlabel('Entropy kJ/kg')
    ax.set_xlim(6800, 8500)
    ax.set_ylabel('Temperature K')
    ax.grid(visible=True, which='major', axis='both', alpha=0.1)
    
    return fig