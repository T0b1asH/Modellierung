import pandas as pd
import os
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% Speicherpfad der Datei einlesen
# muss noch angepasst werden, wenn Ordnerstruktur vorliegt
dir_path = os.path.dirname(os.path.realpath(__file__))  # Übergeordneter Ordner, in dem Programmcode liegt
input_dir = os.path.join(dir_path, 'input_data')        # Unterordner "input_data"
resample_dir = os.path.join(input_dir, 'resampled')     # weiterer Unterordner "resampled"

#%% 
# Start- und Enddatum, sowie die zeitliche Auflösung definieren
startdatum = "2024-01-01 00:00:00"
enddatum = "2024-12-31 23:45:00"
auflösung = '15min'                 # zeitliche Auflösung
stündl_simulationsschritte = 4      # Simulationsschritte pro Stunde 
#n_households = 36
#kwp_per_household = 10

# Konvertierung in pandas datetime-Objekte
startdatum = pd.to_datetime(startdatum)
enddatum   = pd.to_datetime(enddatum)
#jahr  = startdatum.year

# Gesamtzahl der Simulationsschritte
zeitstempel = len(pd.period_range(startdatum, enddatum, freq=auflösung))

#%% CSV-Dateien einlesen
"""
# Haushaltslastprofile
household_kw = pd.read_csv(
    os.path.join(resample_dir, "household_kw_" + str(timebase) + ".csv"), sep=";", index_col = 'Unnamed: 0'
)

# PV-Erzeugungsprofile
pv_kw = pd.read_csv(
    os.path.join(resample_dir, "pv_kw_2020_" + str(timebase) + ".csv"), sep=";", index_col = 'time'
).squeeze().clip(lower = 0)

# Reindex dataframes to have a continuous index based on total timestamps
household_kw.index              = range(0,timestamps)
pv_kw.index                     = range(0,timestamps)
"""
#%% Kosten
"""
#grid_supply_price = (electricity_price + umlagen_eus_per_kwh / timebase_quotient) + 500 #eur per 1/4 kWh

# konstanten Preis einsetzen (realistischen Wert) + Leistungspreis Generator
grid_supply_price = pd.Series([0.398 / 4] * 35040) # eur per 1/4 kWh
netz_leistungspreis_eur_per_kw = 27.32 # eur per kW

infeed_revenue = 0.10 / 4 # eur per 1/4 kWh
# eine Anlage 360kWp -> Direktvermarktung
"""

#%% Netzwerk
"""
net = pypsa.Network()
net.set_snapshots(range(len(household_kw)))
net.add("Carrier", name="electricity") # define carrier
net.add("Bus", name="re_bus", carrier="electricity")
net.add("Bus", name="electric_bus", carrier="electricity")

net.add(
    "Generator",
    name="pv",
    bus="re_bus",
    p_nom = n_households * kwp_per_household,
    p_set = pv_kw * n_households * kwp_per_household
)

net.add(
    "Generator",
    name="grid_demand",
    bus="electric_bus",
    #marginal_cost=grid_supply_price,
    #p_nom = 600,
    # Leistungsbezogenes Netzentgelt für Leistungsspitze
    # p_nom muss dafür auf 1 gesetzt werden
    # p_nom sonst = 600
    #capital_cost = netz_leistungspreis_eur_per_kw,
    #p_nom_extendable = True 
)

net.add(
    "Generator",
    name="grid_infeed",
    bus="re_bus",
    #p_nom=600,
    #marginal_cost=infeed_revenue,
    #capital_cost = netz_leistungspreis_eur_per_kw,
    sign=-1,
)

#Static household loads
net.add("Load", 
        name="Load_hh", 
        bus="electric_bus", 
        p_set=household_kw.sum(axis = 1))



#Add components for EVH flex
for i in range (1,37):
    hh_name = 'HH' + str(i).zfill(2)
    bus_str =  'evh_bus_' + str(i)
    link_str = 'evh_link_' + str(i)
    store_str = 'evh_store_' + str(i)
    load_str = 'evh_load_' + str(i)
    
    
    net.add("Bus", name=bus_str, carrier="electricity")
    
    net.add(
        "Link",
        name=link_str,
        bus1=bus_str,
        bus0="electric_bus",
        p_nom = 11,
        p_max_pu = (mobility_kw[hh_name] == 0).astype(int),
        carrier = "electricity"
        )
    
    net.add("Store", 
            name=store_str, 
            bus=bus_str,
            #e_nom = mobility_stats[hh_name] * 4,
            e_nom_extendable = False,
            e_cyclic=True,
            carrier = "electricity"
            )
    
    net.add("Load", 
            name=load_str, 
            bus=bus_str, 
            p_set = mobility_kw[hh_name]
            )


c_rate = 1  
batt_e_nom = 400 # 400 or 500 kWh capacity
efficiency_batt_load = 0.96
efficiency_batt_unload = 0.96


net.add(
    "StorageUnit",
    name="battery",
    bus="re_bus",
    p_nom = batt_e_nom * c_rate,
    #p_nom_extendable = opt_batt,
    #capital_cost = (capital_cost_battery / c_rate),
    #marginal_cost= (marginal_cost_battery/2) / timebase_quotient,
    state_of_charge_initial = 0.5,
    #e_cyclic=True, 
    max_hours= (1 / c_rate) * 4,
    efficiency_store = efficiency_batt_load, 
    efficiency_dispatch = efficiency_batt_unload, 
)


# RE-Bus->district
net.add(
    "Link",
    name="re_district",
    bus0="re_bus",
    bus1="electric_bus",
    p_nom=n_households * kwp_per_household,
    carrier="electricity"
)


net.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0})
"""