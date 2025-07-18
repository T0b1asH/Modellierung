import pandas as pd
import os
import numpy as np
import pypsa
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pvlib

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', None)

# %% Speicherpfad der Datei einlesen

# muss noch angepasst werden, wenn Ordnerstruktur vorliegt
dateipfad_code = os.path.dirname(os.path.realpath(__file__))  # Übergeordneter Ordner, in dem Codedatei liegt
ordner_input = os.path.join(dateipfad_code, 'Inputdaten')  # Unterordner "Inputdaten"

# %% Start- und Enddatum

# Start- und Enddatum, sowie die zeitliche Auflösung definieren
startdatum = "2023-01-01 00:00:00"
enddatum = "2023-12-31 23:45:00"
aufloesung = '1h'  # zeitliche Auflösung
stuendl_simulationsschritte = 4  # Simulationsschritte pro Stunde

# Konvertierung in pandas datetime-Objekte
startdatum = pd.to_datetime(startdatum)
enddatum = pd.to_datetime(enddatum)
# jahr  = startdatum.year

# Gesamtzahl der Simulationsschritte
simulationsschritte = len(pd.period_range(startdatum, enddatum, freq=aufloesung))
index_h = pd.date_range(start=startdatum, periods=simulationsschritte, freq=aufloesung)

# %% PV-Erzeugung einlesen

pv_erzeugung_1kWp = \
pd.read_csv(os.path.join(ordner_input, "PV_Erzeugung_1kWp.csv"), sep=";", decimal=',', skiprows=3, nrows=8764 - 3)[
    ["Leistung [kW]"]]
pv_erzeugung_1kWp.index = index_h

# print(pv_erzeugung_1kWp.sum())


"""
# Maske, um bestimmte zeiträume zu plotten
# Format: 2023-01-01 00:00:00
zeitraum_beginn = "2023-01-01 00:00:00"
zeitraum_ende = "2023-12-31 00:00:00"
maske = (index_h >= zeitraum_beginn) & (index_h <= zeitraum_ende) # Datum filtern

# Plot
plt.figure(figsize=(12, 6))
pv_erzeugung_1kWp[maske].plot()
plt.xlabel("Datum")
plt.ylabel("AC-Leistung [W]")
plt.title("PV-Erzeugung")
#plt.ylim(0, 300)
plt.grid(True)
plt.tight_layout()
plt.show()
"""
# %% Haushaltsprofile einlesen

# Haushaltsprofile mit LPG generieren
# E-Autos getrennt implementieren, damit Haushaltslasten in beiden Szenarien gleich sind

# Unterordner "Strompreise" einlesen
ordner_haushalte = os.path.join(ordner_input, 'Haushaltsprofile')
# leerer DataFrame
haushaltsprofile = pd.DataFrame()

# Schleife zum einlesen
for i, dateiname in enumerate(sorted(os.listdir(ordner_haushalte))):
    if dateiname.endswith(".csv"):
        pfad = os.path.join(ordner_haushalte, dateiname)
        spalte = os.path.splitext(dateiname)[0]

        # CSV einlesen
        haushaltsprofile[spalte] = pd.read_csv(pfad, sep=";", decimal=',', usecols=["Sum [kWh]"])

# Index für minütliche Werte einlesen
index_min = pd.read_csv(os.path.join(ordner_haushalte, "Haus_01.csv"), sep=";", usecols=["Time"])
index_min = pd.to_datetime(index_min["Time"].str.strip(), dayfirst=True)
haushaltsprofile.index = index_min  # DataFrame mit kWh pro Minute

haushaltsprofile_1h = haushaltsprofile.resample('1h').sum()  # aggregieren auf 1h
haushaltsprofile_1h.index = index_h  # Zeitindex ändern

# %% Kosten

# strompreis Zeitreihe einlesen
# variable Arbeitspreise?
# Leistungspreis?
# Zeitreihe für Direktvermarktung?

# Strompreise einlesen
strompreise = pd.read_csv(os.path.join(ordner_input, "Strompreise.csv"), sep=";", decimal=',')[["Strompreis [ct/kWh]"]]
strompreise.index = index_h

"""
netz_leistungspreis_eur_per_kw = 27.32 # eur per kW

infeed_revenue = 0.10 / 4 # eur per 1/4 kWh
# eine Anlage 360kWp -> Direktvermarktung
"""

# %% PyPSA-Netzwerk
"""
net = pypsa.Network() # Netzwerk erzeugen
net.set_snapshots(range(len(ausgangsleistung_PVAnlage))) # Snapshots festlegen
#net.add("Carrier", name="electricity")  # Carrier Strom hinzufügen
net.add("Bus", name="bus_erneuerbare", carrier="electricity") # EE / Erzeugungs-Bus hinzufügen
net.add("Bus", name="bus_verbrauch", carrier="electricity") # Verbrauchs-Bus hinzufügen

# Erzeugungs-Bus mit Verbrauchs-Bus verbinden
net.add(
    "Link",
    name="link_erzeugung_verbrauch",
    bus0="bus_erneuerbare",
    bus1="bus_verbrauch",
    # Begrenzung der Bezugsleistung nötig?
    #p_nom=n_households * kwp_per_household,
    carrier="electricity"
    )

# PV-Erzeugung
net.add(
    "Generator",
    name="pv",
    bus="bus_erneuerbare",
    #p_nom = max_erzeugung (Nennleistung),
    #p_max_pu = zeitreihe_erzeugung / max_erzeugung
    )

# Netzeinspeisung
net.add(
    "Generator",
    name="netzeinspeisung",
    bus="bus_erneuerbare",
    # Begrenzung der Einpeiseleistung? 
    #p_nom=600,
    #marginal_cost=infeed_revenue,
    #capital_cost = netz_leistungspreis_eur_per_kw,
    sign=-1,
    )

# Batteriespeicher
batt_c_rate = 1  
batt_kapazitaet = 400 
batt_ladeeffizienz = 0.96
batt_entladeeffizienz = 0.96
net.add(
    "StorageUnit",
    name="quartierspeicher",
    bus="bus_erneuerbare",
    p_nom = batt_kapazitaet * batt_c_rate,
    #p_nom_extendable = ,
    #capital_cost = (capital_cost_battery / c_rate),
    #marginal_cost= (marginal_cost_battery/2) / timebase_quotient,
    state_of_charge_initial = 0.5,
    e_cyclic=True, 
    max_hours = (1 / batt_c_rate), #* 4,
    efficiency_store = batt_ladeeffizienz, 
    efficiency_dispatch = batt_entladeeffizienz, 
    )

# Netzbezug
net.add(
    "Generator",
    name="netzbezug",
    bus="bus_verbrauch",
    #marginal_cost=grid_supply_price,
    # Bezugsleistung begrenzen? Gleicher Wert wie Einspeiseleistung
    #p_nom = 600,
    # Leistungsbezogenes Netzentgelt für Leistungsspitze
    # p_nom muss dafür auf 1 gesetzt werden
    # p_nom sonst = 600
    #capital_cost = netz_leistungspreis_eur_per_kw,
    #p_nom_extendable = True 
    )

# Haushaltslast
net.add("Load", 
        #name="Load_hh", 
        bus="bus_verbrauch", 
        #p_set=household_kw.sum(axis = 1)
        )
"""
"""
# Schleife zum hinzufügen der E-Autos
for i in range (1,37):
    haushalt_name = 'haushalt_' + str(i).zfill(2)
    bus_name =  'bus_e_auto_' + str(i)
    link_name = 'link_e_auto_' + str(i)
    store_name = 'store_e_auto_' + str(i)
    load_name = 'load_e_auto_' + str(i)

    # E-Auto-Bus 
    net.add("Bus", name=bus_name, carrier="electricity")

    # Link Verbrauchs-Bus -> E-Auto-Bus 
    net.add(
        "Link",
        name=link_name,
        bus1=bus_name,
        bus0="bus_verbrauch",
        #p_nom = 11,
        #p_max_pu = (mobility_kw[hh_name] == 0).astype(int),
        carrier = "electricity"
        )

    # Autobatterie
    net.add("Store", 
            name=store_name, 
            bus=bus_name,
            #e_nom = mobility_stats[hh_name] * 4,
            e_nom_extendable = False,
            e_cyclic=True,
            carrier = "electricity"
            )

    # Fahrtverbrauch
    net.add("Load", 
            name=load_name, 
            bus=bus_name, 
            #p_set = mobility_kw[hh_name]
            )


net.optimize(solver_name="gurobi", solver_options={"OutputFlag": 0})
"""
# %%


