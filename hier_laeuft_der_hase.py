import pypsa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from networkx.algorithms.efficiency_measures import efficiency

#   ---------------Variablen einlesen----------------------
# CO2-Emissionen
co2_strommix = {  # Angaben in Gramm / kWh
    2022: 433,  # historisch
    2023: 386,
    2024: 363,
    2030: 261,  # Studie
    2050: 0  # Annahme für das Modell
}
co2_limits = {
    2025: np.inf,
    2026: 16350e5,
    2027: 0
    #2028: 26100e7,
    #2029: 26100e6,
    #2030: 26100e5
}
co2_strommix = pd.Series(co2_strommix)
co2_strommix = co2_strommix.reindex(range(2022, 2051))
co2_strommix = co2_strommix.interpolate(method="linear")  # Interpolation für fehlende Jahre
co2_kohle = 769230 + 947690  # Gramm / Tonne Kohle; stofflich + energetisch

stahlproduktion = 9_500_000/8760# kg
strompreis = 0.13 # €/kwh
netzbezug_capitalcost = 147.54 # €/kWa
wasserstoffpreise = 151 # €/kg
wasserstoffverbrauch_pro_kg_stahl_stofflich = 60  # pro kg Stahl
wasserstoffverbrauch_pro_kg_stahl_energetisch = 90  # pro kg Stahl
stromverbrauch_pro_kg_stahl = 3.65 # kWh/kg
#betriebskosten_DRI = 586 + 488 # DRI + Lichtbogenofem [€ / t Stahl]
baukosten_DRI = 518.46e6 + 172.5855e6 + 133.0714e6 # Invest. DRI + Rückbau Hochofen + Invest. Lichtbogenofen [Mio. €]
betriebskosten_Hochofen = 558.25 # €
baukosten_Hochofen = 133_071_400 # €
kohleverbauch_pro_kg_Stahl = 1.6 #kg/kg_Stahl
co2_strommix_2025 = co2_strommix[2025]


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
    network.add("Carrier", name="Stromnetz", co2_emissions=co2_strommix_2025)

    # Busse
    network.add("Bus", name="elektrisches Netz", carrier="Stromnetz")
    network.add("Bus", name="Wasserstoff", carrier="H2")
    network.add("Bus", name="Stahl")
    network.add("Bus", name="Kohle", carrier="Kohle")

    network.add(
        "Generator",
        name="Netzbezug",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        capital_cost=147.54,
        marginal_cost=strompreis,
        carrier="Stromnetz"
    )

    # Erneuerbare
    network.add(
        "Generator",
        name="PV_Sued",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        p_max_pu=df_pv["sued"],
        capital_cost=1100,
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
        efficiency = 1 / kohleverbauch_pro_kg_Stahl, #1 / 1.6,  # 1t Stahl benötigt 1,6t Kohle; 750kg energetisch und 850kg stofflich
        p_nom_extendable=True,
        marginal_cost = betriebskosten_Hochofen * kohleverbauch_pro_kg_Stahl,
    )

    network.add(
        "Link",
        name="DRI",
        bus0="Wasserstoff",
        bus1="Stahl",
        bus2 = "Wasserstoff",
        bus3="elektrisches Netz",
        efficiency0 = 1 / wasserstoffverbrauch_pro_kg_stahl_stofflich,
        efficiency2 = - (1 / wasserstoffverbrauch_pro_kg_stahl_energetisch),
        efficiency3 = - (stromverbrauch_pro_kg_stahl * 1000),
        p_nom_extendable=True,
        #marginal_cost = betriebskosten_DRI,
        capital_cost = baukosten_DRI
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

#%% Simultaion
'''
def co2_reduktion(network, co2_limits):
    df_p_nom_opt = pd.DataFrame(index=network.generators.index)
    df_emissionen = pd.DataFrame(index=["CO2-Limit", "CO2-Emissionen"])

    for year in co2_limits:
        co2_limit = co2_limits[year]

        network.global_constraints.loc['co2-limit', 'constant'] = co2_limit

        network.optimize(
            solver_name='gurobi',
            # method = 2,
            # threads = 1
        )

        df_p_nom_opt[year] = network.generators.p_nom_opt

        gen_p = network.generators_t.p
        carrier_co2 = network.carriers["co2_emissions"]
        gen_carrier = network.generators["carrier"]
        gen_emissions = gen_carrier.map(carrier_co2)
        emissions_per_timestep = gen_p.multiply(gen_emissions, axis=1)
        emissions_ges = emissions_per_timestep.sum(axis=1).sum()
        df_emissionen.loc["CO2-Limit", year] = f"{co2_limit / 1e9:.2f} *1e9 g"
        df_emissionen.loc["CO2-Emissionen", year] = f"{emissions_ges / 1e9:.2f} * 1e9 g"

    print(df_p_nom_opt)
    print("\n", df_emissionen)
'''

#def main():
snapshots = pd.RangeIndex(8760)
df_stahl, df_pv, df_wind = lade_daten(snapshots)
network = erstelle_network(df_pv, df_wind, snapshots)
#co2_reduktion(network, co2_limits)
"""
# Optimierung durchführen
network.optimize(
    solver_name='gurobi',
    threads=1)

"""
"""
df_carrier = network.carriers
df_generators = network.generators.carrier

gen_p = network.generators_t.p
carrier_co2 = network.carriers["co2_emissions"]
gen_carrier = network.generators["carrier"]
gen_emissions = gen_carrier.map(carrier_co2)
emissions_per_timestep = gen_p.multiply(gen_emissions, axis=1)
emissions_ges = emissions_per_timestep.sum(axis=1).sum()
#df_emissionen.loc["CO2-Limit", year] = f"{co2_limit/1e9:.2f} *1e9 g"
#df_emissionen.loc["CO2-Emissionen", year] = f"{emissions_ges/1e9:.2f} * 1e9 g"
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

for co2_limit in np.flip(np.arange(0, 1.1, 0.1)):  # Inkl. 0 %
    col_name = f"{int(round(co2_limit * 100))}%"

    # CO₂-Constraint setzen
    network.global_constraints.loc['co2-limit', 'constant'] = co2_limit * 26097183999886

    # Optimieren
    network.optimize(solver_name='gurobi', method=2, threads=4)

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
    eff_links = network.links.efficiency.fillna(1.0) # Effizienzen der Links abrufen
    df_links[col_name] = network.links.p_nom_opt * eff_links # Leistung Mal Effizienz für Output
    df_generators[col_name] = network.generators.p_nom_opt
    df_stores[col_name] = network.stores.e_nom_opt
    df_storage_units[col_name] = network.storage_units.p_nom_opt
    
    # Zeitreihen des Hochofens und der DRI speichern
    hochofen_p1[col_name] = -network.links_t.p1["Hochofen"] * eff_links["Hochofen"]
    DRI_p1[col_name] = -network.links_t.p1["DRI"] * eff_links["DRI"]
    stahllager_e[col_name] = network.stores_t.e["Stahllager"]


# Store- und StorageUnit-Ergebnisse kombinieren
df_storage_combined = pd.concat([df_stores, df_storage_units])

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

#%% Diagramm Links
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
df_links.T.plot(subplots=True, ax=axs)

axs[0].set_ylabel('kg H2')
axs[1].set_ylabel('t Stahl')
axs[2].set_ylabel('t Stahl')

axs[0].set_title('AEL')
axs[1].set_title('Hochofen')
axs[2].set_title('DRI')

fig.suptitle('Output der Links')
fig.tight_layout()
plt.show()

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

#%% Stahlproduktion nach Links / zeitlicher Verlauf alle Jahre

# alle Jahre in einem
# Liste der Szenarien/Spalten
szenarien = hochofen_p1.columns
anzahl = len(szenarien)
# Dynamische Festlegung von Layout (z. B. 4 Spalten)
spalten = 4
zeilen = -(-anzahl // spalten)  # entspricht math.ceil(anzahl / spalten)
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
plt.figure(figsize=(14, 8))
hochofen_p1["0%"].plot(label="Hochofen")
DRI_p1["0%"].plot(label="DRI")
stahllager_e["0%"].plot(label="Füllstand Stahllager")
plt.ylabel("Erzeugter Stahl [t]")
plt.xlabel("Zeit")
#plt.ylim(1084.474886-1, 1084.474886+1)
plt.title("Zeitlicher Verlauf Output Hochofen & DRI")
plt.legend(title="Link", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
plt.grid(True, linestyle="--", alpha=0.5)
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
 
