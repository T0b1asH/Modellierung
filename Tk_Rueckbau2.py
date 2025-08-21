import pypsa
from pyomo.environ import Constraint, Var, Binary
#from pyomo.core import Constraint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#from sympy.physics.units import years


#%% Index für das erste Jahr erstellen; wird verwendet, um Zeitverschiebung der renewables Daten korrekt zu bereinigen

# Start- und Enddatum, sowie die zeitliche Auflösung definieren
startdatum = "2025-01-01 00:00:00"
enddatum = "2025-12-31 23:00:00"
aufloesung = '1h'                   # zeitliche Auflösung
stuendl_simulationsschritte = 1     # Simulationsschritte pro Stunde 

# Konvertierung in pandas datetime-Objekte
startdatum = pd.to_datetime(startdatum)
enddatum   = pd.to_datetime(enddatum)

# Gesamtzahl der Simulationsschritte
simulationsschritte = len(pd.period_range(startdatum, enddatum, freq=aufloesung))
index_h = pd.date_range(start=startdatum, periods=simulationsschritte, freq=aufloesung)

#%% Jahre definieren

years = [2025 + i for i in range(5)]
freq = 1

#%% CO2-Emissionen und Energiekosten definieren

# alte Eingaben, können vermutlich weg?
"""
co2_emissionen_gas = 358 #kg/MWh https://www.volker-quaschning.de/datserv/CO2-spez/index_e.php
co2_emissionen_strom = 407 #kg/MWh

aktuelle_co2_emissionen = 400000
#stahlproduktion = {2025:10000, 2026:11000, 2027:12000, 2028:13000, 2029:14000}
stromverbrauch_pro_kg_stahl = 0.5 #kWh/kg_stahl
"""


# CO2-Emissionen
co2_strommix = {# Angaben in Gramm / kWh
    2022 : 433, # historisch
    2023 : 386, 
    2024 : 363, 
    2030 : 261, # Studie
    2050 : 0    # Annahme für das Modell
    }           
co2_strommix = pd.Series(co2_strommix)
co2_strommix = co2_strommix.reindex(range(2022,2051))
co2_strommix = co2_strommix.interpolate(method="linear") # Interpolation für fehlende Jahre

co2_kohle = 769230 + 947690 # Gramm / Tonne Kohle; stofflich + energetisch


# Strompreisentwicklung
strompreise = { # Angaben in €/kWh
    2025 : 0.13,    # bekannt
    2026 : 0.128,   # Prognosen
    2031 : 0.076,
    2050 : 0.059
    }
strompreise = pd.Series(strompreise)
strompreise = strompreise.reindex(range(2025,2051))
strompreise = strompreise.interpolate(method="linear") # Interpolation für fehlende Jahre


# Wasserstoffpreisentwicklung
# bisher händisch, weil es schneller ging, können wir noch zu csv einlesen ändern
wasserstoffpreise = {
    2025 : 151,
    2026 : 137.83,
    2027 : 124.47,
    2028 : 111.94,
    2029 : 101.28,
    2030 : 93.5,
    2031 : 89.31,
    2032 : 88.12,
    2033 : 89,
    2034 : 91.35,
    2035 : 93.3,
    2036 : 95.04,
    2037 : 96.19,
    2038 : 96.82,
    2039 : 97.03,
    2040 : 96.9,
    2041 : 96.52,
    2042 : 95.91,
    2043 : 95.09,
    2044 : 94.08,
    2045 : 92.9,
    2046 : 91.56,
    2047 : 90.08,
    2048 : 88.49,
    2049 : 86.79,
    2050 : 85
    }

#%% Daten einlesen

def lade_daten(years):
    
    # Dateipfad einlesen
    dateipfad_code = os.path.dirname(os.path.realpath(__file__))  # Übergeordneter Ordner, in dem Codedatei liegt
    ordner_input = os.path.join(dateipfad_code, 'Inputdaten')     # Unterordner "Inputdaten"
    
    
    # Stahlproduktion
    df_stahl = pd.read_csv(os.path.join(ordner_input, "Stahlproduktion/Stahlproduktion.csv"), sep=";", decimal=",")
    df_stahl = df_stahl.set_index("Jahr")
    
    
    # PV-Erzeugung
    df_pv = pd.DataFrame(index=index_h)
    
    for name in ["sued", "ost", "west"]: # Schleife, um alle 3 Profile einzulesen
        profil = pd.read_csv(
            os.path.join(ordner_input, f"PV/{name}.csv"), skiprows=3, usecols=["electricity"]).shift(1, fill_value=0).to_numpy() # shift wegen Zeitverschiebung, 0 einsetzen
        df_pv[name] = profil
    
    df_pv["ost/west"] = df_pv[["ost", "west"]].mean(axis=1) # Mittelwert für Ost/West bilden
    # Leistungsdegradation berücksichtigen?
    
    
    # Wind-Erzeugung
    df_wind = pd.DataFrame(index=index_h)
    
    for name in ["Onshore", "Offshore"]: # Schleife, um beide Profile einzulesen
        profil = pd.read_csv(
            os.path.join(ordner_input, f"Wind/{name}.csv"), skiprows=3, usecols=["electricity"])["electricity"]
        df_wind[name] = profil.shift(1, fill_value = profil.iloc[-1]).to_numpy()  # shift wegen Zeitverschiebung, letzten Wert vorne einsetzen
    
    
    # Dummy Load
    df_netzlast = pd.read_csv(os.path.join(ordner_input, "Wind/Offshore.csv"), skiprows=3, usecols=["electricity"]) * 1000
    
    return df_stahl, df_netzlast, df_pv, df_wind


#%% Daten auf alle Jahre erweitern

def datenreihen_erweitern(df_netzlast, df_pv, df_wind):
    
    stromlast_liste = []
    pv_erzeugung_liste = []
    wind_erzeugung_liste = []
    
    for year in years:
        
        stromlast_liste.append(df_netzlast["electricity"])
        pv_erzeugung_liste.append(df_pv)
        wind_erzeugung_liste.append(df_wind)
        
    df_netzlast_alle_jahre = pd.concat(stromlast_liste, ignore_index = True)
    df_pv_alle_jahre = pd.concat(pv_erzeugung_liste, ignore_index = True)
    df_wind_alle_jahre = pd.concat(wind_erzeugung_liste, ignore_index = True)
    
    return df_netzlast_alle_jahre, df_pv_alle_jahre, df_wind_alle_jahre

# Ich weiß nicht genau, was die Funktion hier macht. Netzlast auf alle Jahre erweitern und liniearen Abbau der CO2-Emissionen modellieren, nehme ich an. 
# Das brauchen wir dann ja nicht mehr jetzt, oder? Daten erweitern wird hier drüber gemacht und CO2-Emissionen sind ganz oben
"""
def daten_anpassen(stahlproduktion,df_netzlast,stromverbrauch_pro_kg_stahl):
    alle_co2_emissionen = {}
    alle_co2_emissionen_strom = {}
    stromlast_liste = []
    df_netzlast_normiert = df_netzlast["electricity"] / df_netzlast["electricity"].sum()

    for year in years:
        funktion = (-(aktuelle_co2_emissionen/(len(years)-1)))*(year-2025) + aktuelle_co2_emissionen
        alle_co2_emissionen[year] = funktion

        funktion_strom = (-(co2_emissionen_strom/(len(years)-1)))*(year-2025) + co2_emissionen_strom
        alle_co2_emissionen_strom[year] = funktion_strom

        stromlast_pro_jahr = df_netzlast_normiert.copy() * int(stahlproduktion[year] * stromverbrauch_pro_kg_stahl)
        stromlast_liste.append(stromlast_pro_jahr)

    df_netzlast_alle_jahre = pd.concat(stromlast_liste, ignore_index = True)

    return df_netzlast_alle_jahre,alle_co2_emissionen,alle_co2_emissionen_strom
"""

#%% Snapshot-Zeiten erzeugen

#https://pypsa.readthedocs.io/en/stable/examples/multi-investment-optimisation.html
def erstelle_snapshots(years,freq):
    snapshots = pd.DatetimeIndex([])
    for year in years:
        period = pd.date_range(
            start="{}-01-01 00:00".format(year),
            freq="{}h".format(str(freq)),
            periods=int(8760 / freq),
        )
        snapshots = snapshots.append(period)
    return snapshots


'Das alles hier drunter war wahrscheinlich zur Kontrolle da, Ich brauche es nicht mehr also von mir aus kann es weg'
#df_stahl, df_netzlast, df_pv, df_wind = lade_daten(years)
#snapshots = erstelle_snapshots(years,freq)
#df_netzlast_alle_jahre,alle_co2_emissionen,alle_co2_emissionen_strom = daten_anpassen(df_stahl,df_netzlast,stromverbrauch_pro_kg_stahl)

#print(df_netzlast_alle_jahre)
#print(df_netzlast)
#type(snapshots)
#print(len(snapshots))


#%% Netzwerk aufbauen

#def erstelle_network(years, snapshots, df_netzlast, df_netzlast_alle_jahre, df_pv, all_co2_emissions, all_co2_emissions_strom, stahlproduktion, stromverbrauch_pro_kg_stahl):
def erstelle_network(years, snapshots, df_netzlast, df_pv, df_wind, df_stahl):
    network = pypsa.Network()
    network.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
    network.investment_periods = years

    
    # Carrier für CO2-Emissionen
    # müssen "emissions" hier eig. auf englisch stehen? Würde sonst zu deutsch ändern
    network.add("Carrier", name = "EE", co2_emissions = 0) # Erneuerbare
    network.add("Carrier", name = "Kohle", co2_emissions = co2_kohle) # Kohle
    
    
    # Busse
    #network.add("Bus", name = "elektrisches Netz", carrier = "electricity") # Einheit kWh   # Vorschlag Name: "strom_bus"
    network.add("Bus", name = "Wasserstoff", carrier = "H2")                # Einheit kWh   # Vorschlag Name: "H2_bus"
    network.add("Bus", name = "stahl_bus", carrier = "steel")               # Einheit t
    network.add("Bus", name = "kohle_bus", carrier = "coal")                # Einheit t
    
    
    # Dummy Load
    """
    network.add(
        "Load",
        name="el_verbrauch",
        bus="elektrisches Netz",
        p_set= df_netzlast
    )
    """
    
    # Konstante Stahlproduktion
    network.add(
        "Load",
        name = "Stahlproduktion",
        bus = "stahl_bus",
        p_set = 9500000 / 8760 # Tonnen Stahl pro Stunde
    )

    # jährlich veränderliche Komponenten
    for year in years:
        
        # Netzbezug
        #network.add("Carrier", name = "Stromnetz_{}".format(year), co2_emissions = co2_strommix[year]) # jährlich veränderte CO2-Emissionen an Carrier übergeben
        """
        network.add(
            "Generator",
            name = "Netzbezug_{}".format(year),
            bus = "elektrisches Netz",
            p_nom_extendable = True,
            capital_cost = 147.54, # Bereiche für Leistungspreis einfügen?, fixer Wert: https://www.netze-duisburg.de/fileadmin/user_upload/Netz_nutzen/Netzentgelte/Strom/241217_Netze_Duisburg_-_Endg%C3%BCltiges_Preisblatt_Strom_2025.pdf
            marginal_cost = strompreise[year],
            build_year = year,
            lifetime = 1,
            carrier = "Stromnetz_{}".format(year)
            )
        
        
        # Erneuerbare
        network.add(
            "Generator",
            name = "PV_Sued_{}".format(year),
            bus = "elektrisches Netz",
            p_nom_extendable = True,
            p_max_pu = df_pv["sued"],
            capital_cost = 1100,
            marginal_cost = 0.008,
            build_year = year,
            lifetime = 2, # eig. 20
            carrier = "EE"
        )
        
        network.add(
            "Generator",
            name = "PV_Ost_West_{}".format(year),
            bus = "elektrisches Netz",
            p_nom_extendable = True,
            p_max_pu = df_pv["ost/west"],
            capital_cost = 1100,
            marginal_cost = 0.008,
            build_year = year,
            lifetime = 2, # eig. 20
            carrier = "EE"
        )
        
        network.add(
            "Generator",
            name = "Wind_Onshore_{}".format(year),
            bus = "elektrisches Netz",
            p_nom_extendable = True,
            p_max_pu = df_wind["Onshore"],
            capital_cost = 1600,
            marginal_cost = 0.0128,
            build_year = year,
            lifetime = 2, # eig. 20
            carrier = "EE"
        )
        
        network.add(
            "Generator",
            name = "Wind_Offshore_{}".format(year),
            bus = "elektrisches Netz",
            p_nom_extendable = True,
            p_max_pu = df_wind["Offshore"],
            capital_cost = 2800,
            marginal_cost = 0.01775,
            build_year = year,
            lifetime = 2, # eig. 25
            carrier = "EE"
        )
        
        
        # Batteriespeicher
        network.add(
            "StorageUnit",
            name = "Batteriespeicher_{}".format(year),
            bus = "elektrisches Netz",
            p_nom_extendable = True,
            #p_nom_min = ,
            #p_nom_max = ,
            marginal_cost = 0.45,
            capital_cost = 1000,
            build_year = year,
            lifetime = 2, # eig. 15 
            #state_of_charge_initial = 0,
            max_hours = 4, # Stunden bei voller Leistung -> bestimmt Kapazität
            efficiency_store = 0.97,
            efficiency_dispatch = 0.97,
            standing_loss = 8.3335e-6 # 0,02%/Tag -> 0,0002/Tag/unit -> (1-x)^24 = 1 - 0,0002
            )
        """
        
        """
        # Load Lichtbogenöfen
        network.add(
            "Load",
            name = "Lichtbogenoefen_{}".format(year),
            bus = "strom_bus",
            p_set = (df_stahl.loc[year, "Produzierte Stahlmenge [t/a]"] / 8760) * 650 # kWh Strom pro Tonne Stahl / Stunde
            # järliche Stahlmenge durch 8760 für stündl. Wert; 650 als Faktor für Strombedarf
            # Problem: Load kann nicht hier definiert werden, weil die Installation eines Lichtbogenofens von dem Vorhandensein einer DRI-Anlage abhängt
            # Somit hängt auch die Anzahl der Öfen von der Anzahl der DRI ab, was bei p_set noch faktorisiert werden müsste
            # Hier kann man aber noch nicht auf das Netzwerk zugreifen
            # die Load muss also bei der Constraint für den Rückbau der Hochöfen hinzugefügt werden
            # Ich kümmer mich drum
            )
        """
        
        """
        # Elektrolysen
        # Hier könnte man noch Kostenreduktion in der Zukunft berücksichtigen
        network.add(
            "Link",
            name = "AEL_{}".format(year),
            bus0 = "elektrisches Netz",
            bus1 = "Wasserstoff",
            efficiency = 0.7,
            build_year = year,
            lifetime = 2, # eig. 25
            p_nom_extendable = True,
            p_nom_min = 2000,
            #p_nom_max = ,
            capital_cost = 0.875, # Alle Kosten der Elektrolysen könnte man nochmal prüfen, da wir einige verschiedene haben
            marginal_cost = 0.875 * 0.04,
            )
        
        network.add(
            "Link",
            name = "HTE_{}".format(year),
            bus0 = "elektrisches Netz",
            bus1 = "Wasserstoff",
            efficiency = 0.9,
            build_year = year,
            lifetime = 2, # eig. 25
            p_nom_extendable = True,
            p_nom_min = 9,
            #p_nom_max = ,
            capital_cost = 1.3, # nochmal prüfen
            marginal_cost = 1.3 * 0.12,
            ) # Annahme, dass Wärme durch Hochofen / Lichtbogenofen bereitgestellt werden kann, daher immer verfügbar
        """
        
        # Wasserstoff-Bus
        network.add(
            "Generator",
            name = "H2-Pipeline_{}".format(year),
            bus = "Wasserstoff",
            p_nom_extendable = True,
            capital_cost = 25, # https://www.bundesnetzagentur.de/SharedDocs/Pressemitteilungen/DE/2025/20250714_Hochlauf.html
            marginal_cost = wasserstoffpreise[year],
            build_year = year,
            lifetime = 1,
            carrier = "EE"
            )
        """
        network.add(
            "Store",
            name = "H2_Speicher_{}".format(year),
            bus = "Wasserstoff",
            e_nom_extendable = True,
            e_initial = 0,
            capital_cost = 9,
            marginal_cost = 0.45,
            standing_loss = 2.084e-5, # 0,05%/Tag -> 0,0005/Tag/unit -> (1-x)^24 = 1 - 0,0005
            build_year = year,
            lifetime = 2 # eig. 20
            )
        """
        
        network.add(
            "Link",
            name = "H2_stofflich_{}".format(year),
            bus0 = "Wasserstoff",
            bus1 = "stahl_bus",
            efficiency = 1 / (60 * 33.33), # 1t Stahl benötigt 60kg H2 mit 33,33kWh/kg
            p_nom_mod = ((9500000 / 8760) / 5) / (1 / (60 * 33.33)),# 1/5 der stündl. Stahlproduktion, geteilt durch die Effizienz, weil der Output des Links zählt
            p_nom_extendable = True, # eine DRI hat immer die festgelegte Kapazität
            capital_cost = 518.46e6 + 172.5855e6, # Invest. DRI + Rückbau Hochofen [Mio. €]
            build_year = year,
            lifetime = 25 # bleibt bis zum Ende der Simulation erhalten
            )
        
        """
        #p_nom_mod hinzufügen
        for i in range(1, 6): # Schleife, um bis zu 5 DRI zu bauen
            network.add(
                "Link",
                name = f"H2_stofflich_{i}_{year}",
                bus0 = "Wasserstoff",
                bus1 = "stahl_bus",
                efficiency = 1 / (60 * 33.33), # 1t Stahl benötigt 60kg H2 mit 33,33kWh/kg
                p_nom = ((9500000 / 8760) / 5) / (1 / (60 * 33.33)),# 1/5 der stündl. Stahlproduktion, geteilt durch die Effizienz, weil der Output des Links zählt
                p_nom_extendable = False, # eine DRI hat immer die festgelegte Kapazität
                capital_cost = 518.46e6 + 172.5855e6, # Invest. DRI + Rückbau Hochofen [Mio. €]
                build_year = year,
                lifetime = 25 # bleibt bis zum Ende der Simulation erhalten
                )
        """
        """
        network.add(
            "Load",
            name = "H2_energetisch_{}".format(year),
            bus = "Wasserstoff",
            p_set = (df_stahl.loc[year, "Produzierte Stahlmenge [t/a]"] / 8760) * 3000 # kWh H2 pro Stunde
        )
        """
        """
        # Stahl-Bus
        network.add(
            "Load",
            name = "Stahlproduktion_{}".format(year),
            bus = "stahl_bus",
            p_set = df_stahl.loc[year, "Produzierte Stahlmenge [t/a]"] / 8760 # Tonnen Stahl pro Stunde
        )
        """
        
        # Kohle-Bus
        network.add(
            "Generator",
            name = "Kohle_{}".format(year),
            bus = "kohle_bus",
            p_nom_extendable = True,
            marginal_cost = 90,
            build_year = year,
            lifetime = 1,
            carrier = "Kohle"
            )
        
        network.add(
            "Link",
            name = "Hochofen",
            bus0 = "kohle_bus",
            bus1 = "stahl_bus",
            efficiency = 1 / 1.6, # 1t Stahl benötigt 1,6t Kohle; 750kg energetisch und 850kg stofflich 
            p_nom = (9500000 / 8760) / (1 / 1.6),# stündl. Stahlproduktion (weil alle Anlagen zsm, werden einzeln über Constraint rückgebaut), geteilt durch die Effizienz
            p_nom_extendable = True,
            p_nom_max = (9500000 / 8760) / (1 / 1.6), # p_nom_max = p_nom, um weiteren Ausbau zu verhindern
            build_year = 2024, # schon vor Start der Simulation gebaut
            lifetime = 25 # bis zum Ende der Simulation
            )
        
        
        # CO2-Constraint
        # das hier vermutlich die finale Variante für 2050 = 0
        
        #for year in years[-3:]:
        network.add(
            "GlobalConstraint",
            name = f"emission_limit_2050_{year}",
            type = "primary_energy",
            carrier_attribute = "co2_emissions",
            sense = "<=",
            constant = 0.0,
            investment_period = years[-3]
        )
        
        
        # zunächst die hier verwenden, um System mit weniger Jahren zu testen
        """
        network.add(
            "GlobalConstraint",
            name = "emission_limit_{}".format(year),
            carrier_attribute="co2_emissions",
            sense="<=",
            #constant=all_co2_emissions[year],
            constant = 70000e9,
            investment_period = year
        )
        """
    return network


#%% Custom Constraints

def custom_constraint_rueckbau(network, snapshots):
    
    if "Hochofen" in network.links.index and "H2_stofflich" in network.links.index:
        model = network.model
    
        # Zähle DRI-Links
        dri_links = [l for l in network.links.index if "H2_stofflich" in l]
        
        # Effizienzen auslesen
        eff = network.links.efficiency.to_dict()
        
        # Rückbau-Constraint
        model.rueckbau = Constraint(expr = 
            model.link_p_nom["Hochofen"] * eff["Hochofen"] <=
            (9500000 / 8760) - sum(model.link_p_nom[l] * eff[l] for l in dri_links))
        # Kapazität der Hochöfen muss immer kleiner gleich der Anfangs-Kapazität 
        # minus der aktuellen Kapazität aller DRI sein


#%% Main

#def main():
        
df_stahl, df_netzlast, df_pv, df_wind = lade_daten(years) # Daten laden
snapshots = erstelle_snapshots(years, freq) # Snapshots erstellen
#df_netzlast_alle_jahre, alle_co2_emissionen, alle_co2_emissionen_strom = daten_anpassen(stahlproduktion, df_netzlast,stromverbrauch_pro_kg_stahl)
df_netzlast_alle_jahre, df_pv_alle_jahre, df_wind_alle_jahre = datenreihen_erweitern(df_netzlast, df_pv, df_wind) # Daten auf alle Jahre erweitern

# MultiIndex setzen
df_netzlast_alle_jahre.index = pd.MultiIndex.from_arrays(
    [snapshots.year, snapshots],
    names=["period", "snapshot"]
)
df_pv_alle_jahre.index = pd.MultiIndex.from_arrays(
    [snapshots.year, snapshots],
    names=["period", "snapshot"]
)
df_wind_alle_jahre.index = pd.MultiIndex.from_arrays(
    [snapshots.year, snapshots],
    names=["period", "snapshot"]
)

# Network erstellen
network = erstelle_network(years, snapshots, df_netzlast_alle_jahre, df_pv_alle_jahre, df_wind_alle_jahre, df_stahl)

"""
# Optimierung durchführen
network.optimize(
    solver_name = 'gurobi',
    multi_investment_periods=True,
    threads = 1)
"""
# Optimierung durchführen
network.optimize(
    solver_name = 'gurobi',
    multi_investment_periods=True,
    extra_functionality=custom_constraint_rueckbau,
    threads = 1)

c = "Generator"
df = pd.concat(
    {
        period: network.get_active_assets(c, period) * network.static(c).p_nom_opt
        for period in network.investment_periods
    },
    axis=1,
)
df.T.plot.bar(
    stacked=True,
    edgecolor="white",
    width=1,
    ylabel="Capacity",
    xlabel="Investment Period",
    rot=0,
    figsize=(10, 5),
)
plt.tight_layout()
plt.show()
"""
s = "StorageUnit"
df = pd.concat(
    {
        period: network.get_active_assets(s, period) * network.static(s).p_nom_opt
        for period in network.investment_periods
    },
    axis=1,
)
df.T.plot.bar(
    stacked=True,
    edgecolor="white",
    width=1,
    ylabel="Capacity",
    xlabel="Investment Period",
    rot=0,
    figsize=(10, 5),
)
plt.tight_layout()
plt.show()
"""
'''
df_g = network.generators_t.p.sum(axis=0).T
df_g.T.plot.bar(
    stacked=True,
    edgecolor="white",
    width=1,
    ylabel="Generation",
    xlabel="Investment Period",
    rot=0,
    figsize=(20, 5),
)
plt.tight_layout()
plt.show()
'''
#CO2 Emissionen ermittlen
df_carrier = network.carriers
df_generators = network.generators.carrier
standard_co2_emissions = round((network.generators_t.p.sum() / network.generators.efficiency *
                                pd.merge(df_carrier, df_generators, left_index=True, right_on='carrier')
                                ['co2_emissions'])).sum()

# CO2-Emissionen pro Jahr
gen_p = network.generators_t.p
carrier_co2 = network.carriers["co2_emissions"]
gen_carrier = network.generators["carrier"]
gen_emissions = gen_carrier.map(carrier_co2)
emissions_per_timestep = gen_p.multiply(gen_emissions, axis=1)
emissions_per_year = emissions_per_timestep.groupby(level=0).sum().sum(axis=1)
for year, emissions in emissions_per_year.items():
    print(f"CO2-Emissionen {year}: {emissions/1e9:.2f} * 1e9 g")  # wenn in g, sonst ggf. /1e6

"""
print(network.generators)
print(network.loads)
print(network.global_constraints)
print(network.generators_t.p)
print(network.generators[["marginal_cost", "carrier", "build_year"]])
print(network.carriers[["co2_emissions"]])
print("Gesamtkosten vorher: 15322.45 €/a")

print(f"Gesamtkosten: {network.objective:.2f} €/a")
"""
carrier_emissions = network.carriers["co2_emissions"]  # kg/MWh
emission_factors = network.generators["carrier"].map(carrier_emissions)
emissions_kg = (network.generators_t.p * emission_factors).sum().sum() # hatte die Emissionen bisher in Gramm angegeben, können wir aber auch auf kg oder tonnen ändern
#emissions_t = emissions_kg / 1000
print(f"Gesamte CO₂-Emissionen: {emissions_kg:.2f} kgCO₂")
"""
#print(f"PV_Deckungsgrad: {round(network.generators_t.p['PV_Sued_2025'].sum() + network.generators_t.p['PV_Ost_West_2025'].sum()/network.loads_t.p['el_verbrauch'].sum(),3)*100} %") # Hab hier den zweiten PV-Generator hinzugefügt
#print(f"Netz_Deckungsgrad: {round(network.generators_t.p['Netzbezug_2025'].sum()/network.loads_t.p['el_verbrauch'].sum(),3)*100} %") # hab hier Netzbezug richtig benannt

print(round(network.generators_t.p["PV_Sued_2025"].sum() + network.generators_t.p["PV_Ost_West_2025"].sum())) # hab hier beide PV-Generatoren aufaddiert
print(round(network.generators_t.p["Netzbezug_2025"].sum()))
#print(round(network.loads_t.p["el_verbrauch"].sum()))

# hab hier alle Generatoren eingefügt
network.generators_t.p["PV_Sued_2025"].plot()
plt.show()
network.generators_t.p["PV_Ost_West_2025"].plot()
plt.show()
network.generators_t.p["Wind_Onshore_2025"].plot()
plt.show()
network.generators_t.p["Wind_Offshore_2025"].plot()
plt.show()
network.generators_t.p["Netzbezug_2025"].plot()
plt.show()
network.loads_t.p.plot()
plt.show()
network.storage_units_t.state_of_charge.plot()
plt.show()
"""

network.links
network.links_t.p1


df_links_p1 = network.links_t.p1  # DataFrame mit den Ausgangsleistungen aller Links
spalten_gefiltert = [col for col in df_links_p1.columns if "Hochofen" in col or "H2_stofflich" in col] # alle interessanten Spalten filtern
df_links_p1_gefiltert = -df_links_p1[spalten_gefiltert] # df filtern; -1 weil alle p1 Werte negativ sind

# Alle "H2_stofflich" mit selbem Index addieren und in neuem df speichern
df_links_p1_gef_summ = pd.DataFrame(index = df_links_p1_gefiltert.index)
for i in range(1, 6):
    colname = f"H2_stofflich_{i}"
    df_links_p1_gef_summ[colname] = df_links_p1_gefiltert.filter(like=colname).sum(axis=1)

df_links_p1_gef_summ["Hochofen"] = df_links_p1_gefiltert["Hochofen"]

# Liniendiagramm // .loc[2029] für ein jahr
plt.figure(figsize=(14, 8))
df_links_p1_gef_summ.plot(ax=plt.gca())
plt.ylabel("Leistung an bus1 (p1) [kW]")
plt.xlabel("Zeit")
plt.title("Zeitverlauf der Stahlproduktion pro Link (p1)")
plt.legend(title="Link", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# Gruppieren und max. Wert bestimmen
df_yearly_max = df_links_p1_gef_summ.groupby(level="period").max()

# Balkendiagramm
df_yearly_max.plot(kind="bar", figsize=(12, 6))
plt.ylabel("Maximale Leistung an bus1 (p1) [kW]")
plt.xlabel("Jahr")
plt.title("Maximale Stahlproduktion pro Technologie und Jahr")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title="Link", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()



# 2029 monatsweise maximale Leistung pro Link 
df_2029 = df_links_p1_gef_summ.loc[2029]
df_2029_monat_max = df_2029.resample("M").max()
df_2029_monat_max.plot(kind="bar", figsize=(14, 6))

plt.ylabel("Monatliches Maximum der Leistung [kW]")
plt.xlabel("Monat")
plt.title("Monatliches Leistungsmaximum der Stahlproduktion pro Link – Jahr 2029")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(title="Link", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()



#if __name__ == "__main__":
 #   main()
    
 
    
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', None)
