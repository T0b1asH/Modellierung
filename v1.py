import pypsa
#from pyomo.core import Constraint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
#from networkx.algorithms.efficiency_measures import efficiency

#   ---------------Variablen einlesen----------------------
# CO2-Emissionen
co2_strommix = {  # Angaben in Gramm / kWh
    2022: 433,  # historisch
    2023: 386,
    2024: 363,
    2030: 261,  # Studie
    2050: 0  # Annahme für das Modell
}
co2_strommix = pd.Series(co2_strommix)
co2_strommix = co2_strommix.reindex(range(2022, 2051))
co2_strommix = co2_strommix.interpolate(method="linear")  # Interpolation für fehlende Jahre

co2_limits = {
    2025: np.inf,
    2026: 16350e5,
    2027: 0
    #2028: 26100e7,
    #2029: 26100e6,
    #2030: 26100e5
}
co2_limits = pd.Series(co2_limits)
co2_limits = co2_limits.reindex(range(2025, 2051))
co2_limits = co2_limits.interpolate(method="linear")  # Interpolation für fehlende Jahre

strompreis = 0.13 # €/kwh
''' 
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
'''
stahlproduktion = 9_500_000/8760 #kg in Format wie oben



co2_kohle = 769230 + 947690  # Gramm / Tonne Kohle; stofflich + energetisch
netzbezug_capitalcost = 147.54 # €/kWa

DRI_wasserstoffverbrauch_pro_kg_stahl_stofflich = 60  # kg H2 pro t Stahl
DRI_wasserstoffverbrauch_pro_kg_stahl_energetisch = 90  # kg H2 pro t Stahl
DRI_stromverbrauch_pro_kg_stahl = 3.65 # kWh/kg
DRI_baukosten = 518.46e6 + 172.5855e6 + 133.0714e6 # Invest. DRI + Rückbau Hochofen + Invest. Lichtbogenofen [Mio. €]
#betriebskosten_DRI = 586 + 488 #DRI + Lichtbogenofen [€ / t Stahl] hiermit sind KOsten für Wartung, Instandhaltung usw. gemeint


HO_betriebskosten = 558.25 #€ wie setzten die sich zusammen?
HO_kohleverbauch_pro_kg_Stahl = 1.6 #kg/kg_Stahl
#HO_baukosten = 133_071_400 # € kann raus


co2_strommix_2025 = co2_strommix[2025]
#wasserstoffpreise = 151 #€/kg kann raus, oder?
#------------------------Daten einlesen-------------------------------
def lade_daten(snapshots):
    # Dateipfad einlesen
    dateipfad_code = os.path.dirname(os.path.realpath(__file__))  # Übergeordneter Ordner, in dem Codedatei liegt
    ordner_input = os.path.join(dateipfad_code, 'Inputdaten')  # Unterordner "Inputdaten"

    # Stahlproduktion
    df_stahl = pd.read_csv(os.path.join(ordner_input, "Stahlproduktion/Stahlproduktion.csv"), sep=";", decimal=",")

    # PV-Erzeugung
    df_pv = pd.DataFrame(index = snapshots)
    for name in ["sued", "ost", "west"]:  # Schleife, um alle 3 Profile einzulesen
        profil = pd.read_csv(
            os.path.join(ordner_input, f"PV/{name}.csv"), skiprows=3, usecols=["electricity"]).shift(1, fill_value=0).to_numpy()  # shift wegen Zeitverschiebung, 0 einsetzen
        df_pv[name] = profil
    df_pv["ost/west"] = df_pv[["ost", "west"]].mean(axis=1)  # Mittelwert für Ost/West bilden

    # Wind-Erzeugung
    df_wind = pd.DataFrame(index = snapshots)
    for name in ["Onshore", "Offshore"]:  # Schleife, um beide Profile einzulesen
        profil = pd.read_csv(
            os.path.join(ordner_input, f"Wind/{name}.csv"), skiprows=3, usecols=["electricity"])["electricity"]
        df_wind[name] = profil.shift(1, fill_value=profil.iloc[
            -1]).to_numpy()  # shift wegen Zeitverschiebung, letzten Wert vorne einsetzen

    return df_stahl, df_pv, df_wind

#%% Netzwerkdefinition

def erstelle_network(df_pv, df_wind,snapshots):
    network = pypsa.Network()
    network.set_snapshots(snapshots)

    # Carrier für CO2-Emissionen
    network.add("Carrier", name="EE", co2_emissions=0)
    network.add("Carrier", name="Kohle", co2_emissions=co2_kohle)
    network.add("Carrier", name="H2")
    network.add("Carrier", name="Erdgas", co2_emissions=202) # Beispielwert; g CO2 / kWh Erdgas
    network.add("Carrier", name="Stromnetz", co2_emissions=co2_strommix_2025)

    # Busse
    network.add("Bus", name="elektrisches Netz", carrier="Stromnetz")
    network.add("Bus", name="Wasserstoff", carrier="H2")
    network.add("Bus", name="Erdgas", carrier="Erdgas")
    network.add("Bus", name="Stahl")
    network.add("Bus", name="Kohle", carrier="Kohle")
    network.add("Bus", name="DRI")
    """
    network.add(
        "Generator",
        name="Netzbezug",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        capital_cost=netzbezug_capitalcost,
        marginal_cost=strompreis,
        carrier="Stromnetz"
    )
    """
    # Erneuerbare
    network.add(
        "Generator",
        name="PV_Sued",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        p_max_pu=df_pv["sued"],
        capital_cost=1100, #in Variable rein
        marginal_cost=0.008,
        carrier="EE"
    )

    network.add(
        "Generator",
        name="PV_Ost_West",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        p_max_pu=df_pv["ost/west"],
        capital_cost=1100,
        marginal_cost=0.008,
        carrier="EE"
    )

    network.add(
        "Generator",
        name="Wind_Onshore",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        p_max_pu=df_wind["Onshore"],
        capital_cost=1600,
        marginal_cost=0.0128,
        carrier="EE"
    )

    network.add(
        "Generator",
        name="Wind_Offshore",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        p_max_pu=df_wind["Offshore"],
        capital_cost=2800,
        marginal_cost=0.01775,
        carrier="EE"
    )

    network.add(
        "StorageUnit",
        name="Batteriespeicher",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        # p_nom_min = ,
        # p_nom_max = ,
        marginal_cost=0.45,
        capital_cost=1000,
        # state_of_charge_initial = 0,
        max_hours=4,  # Stunden bei voller Leistung -> bestimmt Kapazität
        efficiency_store=0.97,
        efficiency_dispatch=0.97,
        standing_loss=8.3335e-6  # 0,02%/Tag -> 0,0002/Tag/unit -> (1-x)^24 = 1 - 0,0002
    )

    network.add(
        "Link",
        name="AEL",
        bus0="elektrisches Netz",
        bus1="Wasserstoff",
        efficiency=0.7/33.33, # 70% Effizienz energiebezogen
        p_nom_extendable=True,
        p_nom_min=2000,
        # p_nom_max = ,
        capital_cost=0.875,
        # Alle Kosten der Elektrolysen könnte man nochmal prüfen, da wir einige verschiedene haben
        marginal_cost=0.875 * 0.04
        )
    
    """
    network.add(
        "Link",
        name="HTE",
        bus0="elektrisches Netz",
        bus1="Wasserstoff",
        efficiency=0.9,
        p_nom_extendable=True,
        p_nom_min=9,
        # p_nom_max = ,
        capital_cost=1.3,  # nochmal prüfen
        marginal_cost=1.3 * 0.12,
    )  # Annahme, dass Wärme durch Hochofen / Lichtbogenofen bereitgestellt werden kann, daher immer verfügbar
    """    

    network.add(
        "Store",
        name="H2_Speicher",
        bus="Wasserstoff",
        e_nom_extendable=True,
        e_initial=0,
        capital_cost=9,
        marginal_cost=0.45,
        standing_loss=2.084e-5,  # 0,05%/Tag -> 0,0005/Tag/unit -> (1-x)^24 = 1 - 0,0005
    )
    """
    network.add(
        "Link",
        name="H2_stofflich",
        bus0="Wasserstoff",
        bus1="Stahl",
        efficiency=1 / (60 * 33.33),  # 1t Stahl benötigt 60kg H2 mit 33,33kWh/kg
        p_nom_extendable=True
    )
    """

    # Stahl-Bus
    network.add(
        "Load",
        name="Stahlproduktion",
        bus="Stahl",
        #p_set=df_stahl.loc[2025, "Produzierte Stahlmenge [t/a]"] / 8760  # Tonnen Stahl pro Stunde
        p_set=stahlproduktion
    )

    # Kohle-Bus
    network.add(
        "Generator",
        name="Kohle",
        bus="Kohle",
        p_nom_extendable=True,
        marginal_cost=90,
        carrier="Kohle"
    )

    network.add(
        "Link",
        name="Hochofen",
        bus0="Kohle",
        bus1="Stahl",
        efficiency = 1 / HO_kohleverbauch_pro_kg_Stahl, #1 / 1.6,  # 1t Stahl benötigt 1,6t Kohle; 750kg energetisch und 850kg stofflich
        p_nom_extendable=True,
        p_nom_mod = (stahlproduktion * HO_kohleverbauch_pro_kg_Stahl) / 5, # stündl. Stahlproduktion durch Effizienz durch Anz. Hochöfen (5)
        p_nom_max = stahlproduktion * HO_kohleverbauch_pro_kg_Stahl,
        p_min_pu = 0.75,
        marginal_cost = HO_betriebskosten * HO_kohleverbauch_pro_kg_Stahl,
    )

    
    network.add(
        "Link",
        name="H2_in_DRI",
        bus0="Wasserstoff",
        bus1="DRI",
        bus2 = "Wasserstoff",
        #bus3="elektrisches Netz",
        efficiency0 = 1 / DRI_wasserstoffverbrauch_pro_kg_stahl_stofflich,
        efficiency2 = - (1 / DRI_wasserstoffverbrauch_pro_kg_stahl_energetisch),
        #efficiency3 = - (stromverbrauch_pro_kg_stahl * 1000),
        p_nom_extendable=True,
        #marginal_cost = betriebskosten_DRI,
        #capital_cost = DRI_baukosten
    )

    """
    network.add(
        "Link",
        name="DRI",
        bus0="Wasserstoff",
        bus1="Stahl",
        bus2 = "Wasserstoff",
        bus3="elektrisches Netz",
        efficiency0 = 1 / DRI_wasserstoffverbrauch_pro_kg_stahl_stofflich,
        efficiency2 = - (1 / DRI_wasserstoffverbrauch_pro_kg_stahl_energetisch),
        efficiency3 = - (DRI_stromverbrauch_pro_kg_stahl * 1000),
        p_nom_extendable=True,
        #marginal_cost = betriebskosten_DRI,
        capital_cost = DRI_baukosten
    )
    """
    network.add(
        "Link",
        name="DRI",
        bus0="Wasserstoff",
        bus1="Stahl",
        bus2="elektrisches Netz",
        efficiency0 = 1 / DRI_wasserstoffverbrauch_pro_kg_stahl_stofflich,
        efficiency2 = - (DRI_stromverbrauch_pro_kg_stahl * 1000),
        p_nom_extendable=True,
        #marginal_cost = betriebskosten_DRI,
        capital_cost = DRI_baukosten
    )
    
    network.add(
        "Link",
        name = "H2_in_Brenner",
        bus0 = "Wasserstoff",
        bus1="Stahl",
        #bus1 = "DRI energetisch",
        efficiency = 1 / DRI_wasserstoffverbrauch_pro_kg_stahl_energetisch,
        p_nom_extendable = True
        )

    
    network.add(
        "Generator",
        name="Erdgas_Pipeline",
        bus="Erdgas",
        p_nom_extendable=True,
        capital_cost=100, # geschätzt; Leistungspreis nachgucken
        marginal_cost=0.08, # geschätzt; €/kWh 
        carrier="Erdgas"
    )
    
    network.add(
        "Link",
        name="Erdgas_in_DRI",
        bus0="Erdgas",
        bus1="DRI",
        bus2="Erdgas",
        efficiency0 = 1 / 2000, # Erdgas stofflich
        efficiency2 = - (1 / 3000), # Erdgas energetisch
        p_nom_extendable=True,
        #marginal_cost = betriebskosten_DRI,
        #capital_cost = DRI_baukosten
    )
    
    network.add(
        "Link",
        name="DRI",
        bus0="DRI",
        bus1="Stahl",
        bus2="elektrisches Netz",
        #bus1="DRI energetisch",
        efficiency2 = - (1 / 650),
        p_nom_extendable = True,
        capital_cost = DRI_baukosten
        )

    network.add(
        "Store",
        name="Stahllager",
        bus="Stahl",
        #e_nom_extendable=True,
        e_nom = 50000,
        e_initial = 0,
    )
    

    # Global Constraint
    network.add(
        'GlobalConstraint',
        name='co2-limit',
        sense='<=',
        carrier_attribute='co2_emissions',
        constant=np.inf
    )

    return network


#%% Custom Constraints

def custom_constraint_rueckbau(network, snapshots):
    
    model = network.model
    
    p_hochofen = model.variables['Link-p_nom']['Hochofen']
    eff_hochofen = network.links.at["Hochofen", "efficiency"]
    p_dri = model.variables['Link-p_nom']['DRI']
    eff_dri = network.links.at["DRI", "efficiency0"]
    
    constraint_expression = (p_hochofen*eff_hochofen + p_dri*eff_dri <= 
                            stahlproduktion)
    # Kapazität der Hochöfen muss immer kleiner gleich der Anfangs-Kapazität 
    # minus der aktuellen Kapazität aller DRI sein
    
    model.add_constraints(constraint_expression, name="Rückbau Hochofen")

"""
def custom_constraint_brenner(network, snapshots):
    model = network.model
    p = model.variables["Link-p"]

    eff_h2 = network.links.at["H2_in_Brenner", "efficiency"]
    eff_gas = network.links.at["Erdgas_in_Brenner", "efficiency"]
    eff_dri = network.links.at["DRI", "efficiency0"]

    for t in snapshots:
        model.add_constraints(
            p.at[t, "H2_in_Brenner"] * eff_h2 
          + p.at[t, "Erdgas_in_Brenner"] * eff_gas 
          - p.at[t, "DRI"] * eff_dri == 0,
            name=f"brenner_energy_balance_{t}"
        )


def alle_custom_constraints(network, snapshots):
    custom_constraint_rueckbau(network, snapshots)
    custom_constraint_brenner(network, snapshots)
"""
#%% Main

#def main():
snapshots = pd.RangeIndex(8760)
df_stahl, df_pv, df_wind = lade_daten(snapshots)
network = erstelle_network(df_pv, df_wind, snapshots)
#co2_reduktion(network, co2_limits)

# erste Optimierung, um Standard-Emissionen zu berechnen
# Berechnung aber falsch, daher bisher händisch eingetragen
"""
# Optimierung durchführen
network.optimize(
    solver_name='gurobi',
    threads=1)

"""

"""
standard_co2_emissions = round((network.generators_t.p.sum() / network.generators.efficiency *
                                pd.merge(df_carrier, df_generators, left_index=True, right_on='carrier')[
                                    'co2_emissions'])).sum()
"""

#%% Schleife mit Simulationen und Einlesen der Ergebnisse

# Initialisierung der DataFrames
df_emissionen = pd.DataFrame()
df_generators = pd.DataFrame()
df_links = pd.DataFrame()
df_stores = pd.DataFrame()
df_storage_units = pd.DataFrame()
hochofen_p1 = pd.DataFrame()
DRI_p1 = pd.DataFrame()
stahllager_e = pd.DataFrame()
generators_p = pd.DataFrame()

for co2_limit in np.flip(np.arange(0, 1.1, 0.2)):  # Inkl. 0 %
    col_name = f"{int(round(co2_limit * 100))}%"

    # CO₂-Constraint setzen
    network.global_constraints.loc['co2-limit', 'constant'] = co2_limit * 26097183999886
    
    """
    # Optimieren
    network.optimize(solver_name='gurobi', method=2, threads=4)
    """
    # Optimieren mit Constraint
    network.optimize(
    solver_name='gurobi',
    method=2,
    threads=4,
    extra_functionality=custom_constraint_rueckbau
    )


    # CO₂-Emissionen berechnen
    gen_p = network.generators_t.p
    carrier_co2 = network.carriers["co2_emissions"]
    gen_carrier = network.generators["carrier"]
    gen_emissions = gen_carrier.map(carrier_co2)
    emissions_per_timestep = gen_p.multiply(gen_emissions, axis=1)
    emissions_ges = emissions_per_timestep.sum(axis=1).sum()

    # DataFrames befüllen
    df_emissionen[col_name] = pd.Series({
        "CO2-Limit": co2_limit * 26097183999886,
        "CO2-Emissionen": emissions_ges
    })
    
    eff_links = network.links.efficiency.fillna(1.0).copy() # Effizienzen einlesen
    eff_links.loc["DRI"] = network.links.at["DRI", "efficiency0"] # Effizienz bei "DRI" ersetzen
    df_links[col_name] = network.links.p_nom_opt * eff_links # p_nom_opt * Effizienz für Output
    df_generators[col_name] = network.generators.p_nom_opt
    df_stores[col_name] = network.stores.e_nom_opt
    df_storage_units[col_name] = network.storage_units.p_nom_opt
    
    # Zeitreihen des Hochofens und der DRI speichern
    hochofen_p1[col_name] = -network.links_t.p1["Hochofen"] # p1 gibt bereits den Output an
    DRI_p1[col_name] = -network.links_t.p1["DRI"]
    stahllager_e[col_name] = network.stores_t.e["Stahllager"]
    
    for gen_name in network.generators_t.p.columns:
        generators_p[(gen_name, col_name)] = network.generators_t.p[gen_name]
    generators_p[('Gesamt', col_name)] = network.generators_t.p.sum(axis=1)


# Store- und StorageUnit-Ergebnisse kombinieren
df_storage_combined = pd.concat([df_stores, df_storage_units])


#%% Energiebedarf

# Kopie des Generators mit Leistungs-Zeitreihen erstellen
generators_p_energetisch = generators_p.copy()

# Heizwert Kohle
heizwert_kohle = 4.17e3 # kWh/t

kohle_spalten = [col for col in generators_p.columns if col[0] == "Kohle"] # alle Spalten mit "Kohle" einlesen
generators_p_energetisch.loc[:, kohle_spalten] = generators_p.loc[:, kohle_spalten] *heizwert_kohle # Umrechnung von t auf kWh
generators_p_energetisch.columns = pd.MultiIndex.from_tuples(generators_p_energetisch.columns) # Index zu MultiIndex machen

gesamt_spalten = [col for col in generators_p_energetisch.columns if col[0] == "Gesamt"] # Spalten "Gesamt" einlesen
generators_p_energetisch = generators_p_energetisch.drop(columns=gesamt_spalten) # Spalten "Gesamt" entfernen

prozentwerte = generators_p_energetisch.columns.get_level_values(1).unique() # alle Prozentwerte einlesen

# Schleife über Prozentwerte
for p in prozentwerte:
    spalten_ohne_gesamt = [col for col in generators_p_energetisch.columns if col[1] == p] # alle Spalten außer "Gesamt"
    gesamt_neu = generators_p_energetisch[spalten_ohne_gesamt].sum(axis=1) # Summieren
    generators_p_energetisch[("Gesamt", p)] = gesamt_neu # eintragen
    
    # dasselbe nur ohne Kohle
    spalten_ohne_gesamt_und_kohle = [
        col for col in generators_p_energetisch.columns
        if col[1] == p and col[0] != "Kohle" and col[0] != "Gesamt"]
    gesamt_ohne_kohle = generators_p_energetisch[spalten_ohne_gesamt_und_kohle].sum(axis=1)
    generators_p_energetisch[("Ges. ohne Kohle", p)] = gesamt_ohne_kohle

gesamt_spalten = generators_p_energetisch.filter(like="Gesamt") # Spalten mit "Gesamt" filtern
summen = gesamt_spalten.sum() # Summe berechnen
print("Energiebedarf mit Kohle") # ausgeben
for spalte, wert in summen.items():
    print(f"{spalte}: {round(wert/1e6)} GWh")

ges_ohne_kohle_spalten = generators_p_energetisch.filter(like="Ges.") # Spalten mit "Ges." filtern
summen_ohne_kohle = ges_ohne_kohle_spalten.sum() # Summe berechnen
print("\nEnergiebedarf ohne Kohle") # ausgeben
for spalte, wert in summen_ohne_kohle.items():
    print(f"{spalte}: {round(wert/1e6)} GWh")


#%% Diagramm Emissionen
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
df_emissionen.T.plot(subplots=True, ax=axs)

axs[0].set_ylabel('g')
axs[1].set_ylabel('g')

axs[0].set_title('CO2-Limit')
axs[1].set_title('Tatsächliche CO2-Emissionen')

fig.suptitle('CO2-Emissionen')
fig.tight_layout()
plt.show()

#%% Diagramm Generatoren

# Anzahl Generatoren
n_generators = len(df_generators)
ncols = 3
nrows = math.ceil(n_generators / ncols)

# Subplots erzeugen
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
axs = axs.flatten()

# Plot nur so viele wie Generatoren
df_generators_subset = df_generators.iloc[:len(axs)]
df_generators_subset.T.plot(subplots=True, ax=axs[:n_generators])

# Titel und y-Labels definieren (dynamisch oder manuell)
titles = df_generators.index.tolist()  # oder manuell wie vorher
ylabels = ['kW'] * (n_generators - 1) + ['t']  # Beispiel: letzter ist in Tonnen

# Beschriftung setzen
for i in range(n_generators):
    axs[i].set_title(titles[i])
    axs[i].set_ylabel(ylabels[i] if i < len(ylabels) else '')

# Überzählige Achsen deaktivieren
for j in range(n_generators, len(axs)):
    axs[j].axis("off")

fig.suptitle("Generatoren")
fig.tight_layout()
plt.show()



"""
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
df_generators.T.plot(subplots=True, ax=axs)

axs[0, 0].set_ylabel('kW')
axs[0, 1].set_ylabel('kW')
axs[0, 2].set_ylabel('kW')
axs[1, 0].set_ylabel('kW')
axs[1, 1].set_ylabel('kW')
axs[1, 2].set_ylabel('t')

axs[0, 0].set_title('Netzbezug')
axs[0, 1].set_title('PV_Sued')
axs[0, 2].set_title('PV_Ost_West')
axs[1, 0].set_title('Wind_Onshore')
axs[1, 1].set_title('Wind_Offshore')
axs[1, 2].set_title('Kohle')

fig.suptitle('Generatoren')
fig.tight_layout()
plt.show()
"""
#%% Diagramm Links

n_links = len(df_links)  # z. B. 5
nrows, ncols = 2, 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
axs = axs.flatten()

# Nur so viele Achsen wie Zeilen übergeben
df_links_subset = df_links.iloc[:len(axs)]
df_links_subset.T.plot(subplots=True, ax=axs[:len(df_links_subset)])

# Achsenbeschriftung und Titel setzen
titles = ['AEL', 'Hochofen', 'DRI', 'H2 in Brenner', 'Erdgas in Brenner', '']
ylabels = ['kg H2', 't Stahl', 't Stahl', 't Stahl', 't Stahl', '']

for ax, title, ylabel in zip(axs, titles, ylabels):
    ax.set_title(title)
    ax.set_ylabel(ylabel)

# Leere Achsen ausschalten (falls weniger Daten als Subplots)
for i in range(len(df_links), len(axs)):
    axs[i].axis("off")

fig.suptitle('Output der Links')
fig.tight_layout()
plt.show()

"""
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
df_links.T.plot(subplots=True, ax=axs)

axs[0, 0].set_ylabel('kg H2')
axs[0, 1].set_ylabel('t Stahl')
axs[0, 2].set_ylabel('t Stahl')
axs[1, 0].set_ylabel('t Stahl')
axs[1, 1].set_ylabel('t Stahl')
axs[1, 2].set_ylabel('')

axs[0, 0].set_title('AEL')
axs[0, 1].set_title('Hochofen')
axs[0, 2].set_title('DRI')
axs[1, 0].set_title('H2 in Brenner')
axs[1, 1].set_title('Erdgas in Brenner')
axs[1, 2].set_title('')

fig.suptitle('Output der Links')
fig.tight_layout()
plt.show()
"""
#%% Diagramm Speicher
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
df_storage_combined.T.plot(subplots=True, ax=axs)

axs[0].set_ylabel('kg H2')
axs[1].set_ylabel('t Stahl')
axs[2].set_ylabel('kW')

axs[0].set_title('H2_Speicher')
axs[1].set_title('Stahllager')
axs[2].set_title('Batterie Entladeleistung (Kapa. wäre geteilt durch 4')

fig.suptitle('Speicherauslegung')
fig.tight_layout()
plt.show()


#%% Analyse Hochofen (Anzahl / Rückgang)

# Anzahl Hochöfen und max. Produktion
# p_nom_mod auslesen und mit Effizienz multiplizieren, weil df_links auch Output ist
p_nom_mod_hochofen = network.links.at["Hochofen", "p_nom_mod"] * network.links.efficiency["Hochofen"]
# Schleife über alle Spalten von df_links und Einlesen des Wertes aus Zeile "Hochofen"
print("\nAnzahl Hochöfen")
for label, p_nom_opt_hochofen in df_links.loc["Hochofen"].items():
    
    anz_hochofen = round(p_nom_opt_hochofen / p_nom_mod_hochofen)
    
    max_prod = round(hochofen_p1[label].sum())
    print(f"{label}: {anz_hochofen} Hochöfen mit einer max. stündl. Produktion von {max_prod} t Stahl")


#%% Stahlproduktion nach Links / zeitlicher Verlauf alle Jahre

# alle Jahre in einem
szenarien = hochofen_p1.columns
spalten = 3
zeilen = 2  # entspricht math.ceil(anzahl / spalten)
# Subplots erstellen
fig, axs = plt.subplots(nrows=zeilen, ncols=spalten, figsize=(spalten * 4, zeilen * 3), sharex=True, sharey=True)
axs = axs.flatten()  # 2D zu 1D für einfacheres Indexing
# Plot für jede Spalte/Szenario
for i, sz in enumerate(szenarien):
    axs[i].plot(hochofen_p1.index, hochofen_p1[sz], label='Hochofen')
    axs[i].plot(DRI_p1.index, DRI_p1[sz], label='DRI')
    axs[i].set_title(sz)
    axs[i].set_xlabel("Stunde")
    axs[i].set_ylabel("t Stahl")
    axs[i].legend()
# Leere Achsen deaktivieren, falls nicht alle Subplots gebraucht werden
for j in range(i + 1, len(axs)):
    axs[j].axis("off")
fig.suptitle("Vergleich Hochofen / DRI zeitlich", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Platz für suptitle
plt.show()


# einzelne Jahre
for p in szenarien:
    fig, ax1 = plt.subplots(figsize=(14, 8))
    # Primärachse für Hochofen & DRI
    hochofen_p1[p].plot(ax=ax1, label="Hochofen", color="black", alpha=0.7)
    DRI_p1[p].plot(ax=ax1, label="DRI", color="orange")
    ax1.set_ylabel("Erzeugter Stahl [t]")
    ax1.set_xlabel("Zeit")
    ax1.grid(True, linestyle="--", alpha=0.5)
    # Sekundärachse für Stahllager
    ax2 = ax1.twinx()
    stahllager_e[p].plot(ax=ax2, label="Füllstand Stahllager", color="green", linestyle="dotted")
    ax2.set_ylabel("Füllstand Stahllager [t]")
    # Legenden zusammenführen
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, title="Link", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    # plotten
    plt.title(f"Zeitlicher Verlauf Output Hochofen & DRI und Füllstand Stahllager; {p}")
    plt.tight_layout()
    plt.show()


#%% Stahlproduktion
"""
# alle Links einlesen, die stahl_bus beliefern
links_zu_stahl = []

for i in range(10):  # bus0 bis bus9 durchlaufen
    bus_col = f"bus{i}" # Namen der zu durchsuchenden Spalte 
    if bus_col in network.links.columns: # Spalten durchlaufen
        matching = network.links[network.links[bus_col] == "Stahl"]
        for link in matching.index:
            links_zu_stahl.append((link, i))  # Linkname und Bus-Index


# DataFrame mit zeitlichen Flüssen aller Links zu stahl_bus
stahl_prod = pd.DataFrame()

for link, i in links_zu_stahl: # alle Einträge in den Links durchlaufen
    col_name = f"{link}_p{i}" # Spaltenname für neuen df
    link_prod = -network.links_t[f"p{i}"][link] # Erzeugung auslesen
    stahl_prod[col_name] = link_prod # Erzeugung eintragen

# Summe aller bilden aber ohne Spalte "Summe" + Runden
stahl_prod["Summe"] = stahl_prod.drop(columns=["Summe"], errors="ignore").sum(axis=1).round(6)


#.loc[2029] für ein jahr
plt.figure(figsize=(14, 8))
#stahl_prod["Summe"].plot(label="Stahlproduktion")
#network.stores_t.p["Stahllager"].clip(lower=0).plot(label="Stahllager-Entnahme")
(stahl_prod["Summe"]+network.stores_t.p["Stahllager"]).plot(label="Stahllager-Entnahme")
#network.links_t.p0.filter(like="DRI").sum(axis=1).plot(label="Wasserstoff-Verbrauch")
#network.links_t.p2.filter(like="DRI").sum(axis=1).plot(label="Stromverbrauch")
plt.ylabel("Erzeugter Stahl [t]")
plt.xlabel("Zeit")
#plt.ylim(1084.474886-1, 1084.474886+1)
plt.title("Zeitverlauf der gesamten Stahlproduktion")
plt.legend(title="Link", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
"""




#if __name__ == "__main__":
 #   main()
 
