import pypsa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

#%% Variablen einlesen

# betrachtete Jahre
years = [2025, 2030, 2035, 2040, 2045, 2050]

co2_limits = {
    2025: 26097183999886e-6,
    2050: 0
}

annuitaet_10a = 0.1113                                              # 2% Zinssatz
annuitaet_15a = 0.0778                                              # 2% Zinssatz
annuitaet_20a = 0.0612                                              # 2% Zinssatz
annuitaet_25a = 0.0512                                              # 2% Zinssatz
annuitaet_30a = 0.0446                                              # 2% Zinssatz
annuitaet_50a = 0.0318                                              # 2% Zinssatz

co2_limits = pd.Series(co2_limits)
co2_limits = co2_limits.reindex(range(2025, 2051))
co2_limits = co2_limits.interpolate(method="linear")                # Interpolation für fehlende Jahre


co2_ee = 0
co2_kohle = 769230e-6 + 947690e-6                                   # Tonnen CO2 / Tonne Kohle; stofflich + energetisch
co2_gas = 202e-6

# Heizwert Kohle
heizwert_kohle = 4.17e3                                             # kWh/t

# Direktreduktion
DRI_wasserstoffverbrauch_pro_t_stahl_stofflich = 60                 # kg H2 pro t Stahl
DRI_wasserstoffverbrauch_pro_t_stahl_energetisch = 90               # kg H2 pro t Stahl
DRI_stromverbrauch_pro_t_stahl = 650                                # kWh/t Stahl im Lichtbogenofen
#DRI_Lichtbogen_baukosten = 518.46e6 + 172.5855e6 + 133.0714e6      # Invest. DRI + Rückbau Hochofen + Invest. Lichtbogenofen [Mio. €]
DRI_Lichtbogen_baukosten = 2_185e6 + 172.6e6 + 1_605.5e6            # Invest. DRI + Rückbau Hochofen + Invest. Lichtbogenofen [Mio. €]
DRI_Lichtbogen_betriebskosten = 11.19 + 27.54                       # DRI €/t Stahl ohne Ressourcen + Lichtbogenofen €/t Stahl


# Hochofen
HO_betriebskosten = 77.47                                           # €/ t Stahl ohne Ressourcen
HO_kohleverbauch_pro_t_Stahl = 1.6                                  # t Kohle/t Stahl



#%% Daten einlesen

def lade_daten(snapshots):
    
    # Dateipfad einlesen
    dateipfad_code = os.path.dirname(os.path.realpath(__file__))    # Übergeordneter Ordner, in dem Codedatei liegt
    ordner_input = os.path.join(dateipfad_code, 'Inputdaten')       # Unterordner "Inputdaten"

    # Stahlproduktion
    df_stahl = pd.read_csv(os.path.join(ordner_input, "Stahlproduktion/Stahlproduktion.csv"), sep=";", decimal=",", index_col="Jahr")

    # PV-Erzeugung
    df_pv = pd.DataFrame(index = snapshots)
    for name in ["sued", "ost", "west"]:                            # Schleife, um alle 3 Profile einzulesen
        profil = pd.read_csv(
            os.path.join(ordner_input, f"PV/{name}.csv"), skiprows=3, usecols=["electricity"]).shift(1, fill_value=0).to_numpy()  # shift wegen Zeitverschiebung, 0 einsetzen
        df_pv[name] = profil
    df_pv["ost/west"] = df_pv[["ost", "west"]].mean(axis=1)         # Mittelwert für Ost/West bilden

    # Wind-Erzeugung
    df_wind = pd.DataFrame(index = snapshots)
    for name in ["Onshore", "Offshore"]:                            # Schleife, um beide Profile einzulesen
        profil = pd.read_csv(
            os.path.join(ordner_input, f"Wind/{name}.csv"), skiprows=3, usecols=["electricity"])["electricity"]
        df_wind[name] = profil.shift(1, fill_value=profil.iloc[
            -1]).to_numpy()                                         # shift wegen Zeitverschiebung, letzten Wert vorne einsetzen

    return df_stahl, df_pv, df_wind

#%% Netzwerkdefinition

def erstelle_network(df_stahl, df_pv, df_wind, snapshots, year):
    network = pypsa.Network()
    network.set_snapshots(snapshots)

    # Carrier für CO2-Emissionen
    network.add("Carrier", name="EE", co2_emissions=co2_ee)
    network.add("Carrier", name="Kohle", co2_emissions=co2_kohle)
    network.add("Carrier", name="H2")
    network.add("Carrier", name="Erdgas", co2_emissions=co2_gas)

    # Busse
    network.add("Bus", name="Strom")                                    # Einheit kWh
    network.add("Bus", name="Wasserstoff", carrier="H2")                # Einheit kg
    network.add("Bus", name="Erdgas", carrier="Erdgas")                 # Einheit kWh
    network.add("Bus", name="Stahl")                                    # Einheit t
    network.add("Bus", name="Kohle", carrier="Kohle")                   # Einheit t
    network.add("Bus", name="DRI")                                      # Einheit t Stahl
    network.add("Bus", name="Batterie_bus")                             # nur für Funktionalität
    network.add("Bus", name="Salzkaverne_bus")                          # nur für Funktionalität

    # Erneuerbare
    network.add(
        "Generator",
        name="PV_Sued",
        bus="Strom",
        p_nom_extendable=True,
        p_max_pu=df_pv["sued"],
        capital_cost=1100 * annuitaet_20a,
        marginal_cost=0.008,
        carrier="EE"
    )

    network.add(
        "Generator",
        name="PV_Ost_West",
        bus="Strom",
        p_nom_extendable=True,
        p_max_pu=df_pv["ost/west"],
        capital_cost=1100 * annuitaet_20a,
        marginal_cost=0.008,
        carrier="EE"
    )

    network.add(
        "Generator",
        name="Wind_Onshore",
        bus="Strom",
        p_nom_extendable=True,
        p_max_pu=df_wind["Onshore"],
        capital_cost=1600 * annuitaet_20a,
        marginal_cost=0.0128,
        carrier="EE"
    )

    network.add(
        "Generator",
        name="Wind_Offshore",
        bus="Strom",
        p_nom_extendable=True,
        p_max_pu=df_wind["Offshore"],
        capital_cost=2800 * annuitaet_25a,
        marginal_cost=0.01775,
        carrier="EE"
    )
    # Simulation Batt bestehend aus einem Bus, einem Store und zwei Links
    network.add(
        "Store",
        name="Batteriespeicher",
        bus="Batterie_bus",
        e_nom_extendable=True,
        capital_cost=1000 * annuitaet_15a,
        marginal_cost=0.45,
        standing_loss = 0.000056
    )
    network.add(
        "Link",
        name="Batterie_ein",
        bus0="Strom",
        bus1="Batterie_bus",
        p_nom_extendable=True,
        efficiency=0.9826
    )
    network.add(
        "Link",
        name="Batterie_aus",
        bus0="Batterie_bus",
        bus1="Strom",
        p_nom_extendable=True,
        efficiency=0.9826
    )
    '''
    # Batteriespeicher
    network.add(
        "StorageUnit",
        name="Batteriespeicher",
        bus="Strom",
        p_nom_extendable=True,
        capital_cost=1000,
        marginal_cost=0.45,
        max_hours=4,                                                                # Stunden bei voller Leistung -> bestimmt Kapazität
        efficiency_store=0.9826,
        efficiency_dispatch=0.9826,
        standing_loss=8.3335e-6  # 0,02%/Tag -> 0,0002/Tag/unit -> (1-x)^24 = 1 - 0,0002
    )
    '''
    # Elektrolyse
    network.add(
        "Link",
        name="AEL",
        bus0="Strom",
        bus1="Wasserstoff",
        efficiency=0.7/33.33,                                                       # 70% Effizienz energiebezogen
        p_nom_extendable=True,
        p_nom_min=2000,
        capital_cost=1200 * annuitaet_30a,
        marginal_cost=1200 * 0.04
        )

    # H2- Bus
    network.add(
        "Store",
        name="H2_Speicher",
        bus="Salzkaverne_bus",
        e_nom_extendable=True,
        capital_cost=9 * annuitaet_50a,
        marginal_cost=0.45
    )
    network.add(
        "Link",
        name="Salzkaverne_ein",
        bus0="Wasserstoff",
        bus1="Salzkaverne_bus",
        #p_nom_max = 30,
        p_nom_extendable=True,
        efficiency=0.98**0.5
    )
    network.add(
        "Link",
        name="Salzkaverne_aus",
        bus0="Salzkaverne_bus",
        bus1="Wasserstoff",
        p_nom_extendable=True,
        #p_nom_max = 30,
        efficiency=0.98**0.5
    )

    network.add(
        "Link",
        name="H2_in_DRI",
        bus0="Wasserstoff",
        bus1="DRI",
        efficiency = 1 / (DRI_wasserstoffverbrauch_pro_t_stahl_stofflich + DRI_wasserstoffverbrauch_pro_t_stahl_energetisch),
        p_nom_extendable=True
    )

    # Erdgas-Bus
    network.add(
        "Generator",
        name="Erdgas_Pipeline",
        bus="Erdgas",
        p_nom_extendable=True,
        capital_cost=100,                                                           # !!!ARSCHFICK!!! geschätzt; Leistungspreis nachgucken
        marginal_cost=0.08,                                                         # geschätzt; €/kWh
        carrier="Erdgas"
    )
    
    network.add(
        "Link",
        name="Erdgas_in_DRI",
        bus0="Erdgas",
        bus1="DRI",
        efficiency = 1 / (2777.77 + 3000),                                          # Erdgas stofflich + energetisch
        p_nom_extendable=True
    )
    
    
    # DRI-Bus
    network.add(
        "Link",
        name="Lichtbogenofen",
        bus0="DRI",
        bus1="Stahl",
        bus2="Strom",
        efficiency2 = - DRI_stromverbrauch_pro_t_stahl,
        p_nom_extendable = True,
        capital_cost = DRI_Lichtbogen_baukosten * annuitaet_30a,
        marginal_cost = DRI_Lichtbogen_betriebskosten
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
        efficiency = 1 / HO_kohleverbauch_pro_t_Stahl,                                                                #1 / 1.6,  # 1t Stahl benötigt 1,6t Kohle; 750kg energetisch und 850kg stofflich
        p_nom_extendable=True,
        p_nom_mod = ((df_stahl["Produzierte Stahlmenge [t/a]"].loc[2025] / 8760) * HO_kohleverbauch_pro_t_Stahl) / 4, # stündl. Stahlproduktion durch Effizienz durch Anz. Hochöfen (5)
        p_nom_max = (df_stahl["Produzierte Stahlmenge [t/a]"].loc[2025] / 8760) * HO_kohleverbauch_pro_t_Stahl,
        p_min_pu = 0.95,
        ramp_limit_up = 0.003,
        ramp_limit_down = 0.003,
        marginal_cost = HO_betriebskosten / HO_kohleverbauch_pro_t_Stahl,
    )
    
    
    # Stahl-Bus
    network.add(
        "Load",
        name="Stahlproduktion",
        bus="Stahl",
        p_set = df_stahl["Produzierte Stahlmenge [t/a]"].loc[year] / 8760
    )
    
    network.add(
        "Store",
        name="Stahllager",
        bus="Stahl",
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
    p_dri = model.variables['Link-p_nom']['Lichtbogenofen']
    
    constraint_expression = (p_hochofen*eff_hochofen + p_dri <= 
                            df_stahl["Produzierte Stahlmenge [t/a]"].loc[year])

    model.add_constraints(constraint_expression, name="Rückbau Hochofen")
#%% Schleife mit Simulationen und Einlesen der Ergebnisse

# Initialisierung der DataFrames
df_emissionen = pd.DataFrame()
df_generators = pd.DataFrame()
df_links = pd.DataFrame()
df_stores = pd.DataFrame()
#df_storage_units = pd.DataFrame()
hochofen_p1 = pd.DataFrame()
DRI_p0 = pd.DataFrame()
DRI_p1 = pd.DataFrame()
DRI_p2 = pd.DataFrame()
stahllager_e = pd.DataFrame()
generators_p = pd.DataFrame()
batterie_ladung_p = pd.DataFrame()
batterie_entladung_p = pd.DataFrame()
batterie_soc = pd.DataFrame()
elektrolyse_p0 = pd.DataFrame()
elektrolyse_p1 = pd.DataFrame()
H2_speicher_e = pd.DataFrame()
H2_speicher_p = pd.DataFrame()
H2_in_DRI_p0 = pd.DataFrame()
H2_in_DRI_p1 = pd.DataFrame()
Erdgas_in_DRI_p0 = pd.DataFrame()
Erdgas_in_DRI_p1 = pd.DataFrame()
stahlload_p = pd.DataFrame()

# Snapshots erstellen
snapshots = pd.RangeIndex(8760)
# Zeitreihen einlesen
df_stahl, df_pv, df_wind = lade_daten(snapshots)

network = erstelle_network(df_stahl, df_pv, df_wind, snapshots, year)

# Schleife über Jahre
for year in years:

    # CO₂-Limit setzen
    network.global_constraints.loc['co2-limit', 'constant'] = co2_limits[year]
    
    # Optimieren mit Constraint
    network.optimize(
    solver_name='gurobi',
    method=2,
    threads=4,
    extra_functionality=custom_constraint_rueckbau
    )

    # Ergebnisse berechnen
    gen_p = network.generators_t.p
    carrier_co2 = network.carriers["co2_emissions"]
    gen_carrier = network.generators["carrier"]
    gen_emissions = gen_carrier.map(carrier_co2)
    emissions_per_timestep = gen_p.multiply(gen_emissions, axis=1)
    emissions_ges = emissions_per_timestep.sum().sum()

    col_name = str(year)
    # DataFrames befüllen
    df_emissionen[col_name] = pd.Series({
        "CO2-Limit": co2_limits[year],
        "CO2-Emissionen": emissions_ges
    })

    # Effizienzen einlesen
    eff_links = network.links.efficiency.copy()
    df_links[col_name] = network.links.p_nom_opt * eff_links
    
    df_generators[col_name] = network.generators.p_nom_opt
    df_stores[col_name] = network.stores.e_nom_opt
    #df_storage_units[col_name] = network.storage_units.p_nom_opt
    
    
    # Zeitreihen auslesen
    # Generatoren
    for gen_name in network.generators_t.p.columns:
        generators_p[(gen_name, col_name)] = network.generators_t.p[gen_name]
    generators_p[('Gesamt', col_name)] = network.generators_t.p.sum(axis=1)
    
    # Batteriespeicher
    batterie_ladung_p[col_name] = network.storage_units_t.p_store["Batteriespeicher"] # Ladung am Bus
    batterie_entladung_p[col_name] = network.storage_units_t.p_dispatch["Batteriespeicher"] # Entladung am Bus
    batterie_soc[col_name] = network.storage_units_t.state_of_charge["Batteriespeicher"] # SoC
    
    # Elektrolyse
    elektrolyse_p0[col_name] = network.links_t.p0["AEL"] # Input Strom in AEL
    elektrolyse_p1[col_name] = network.links_t.p1["AEL"] # Output H2 aus AEL
    
    # H2-Speicher
    H2_speicher_e[col_name] = network.stores_t.e["H2_Speicher"]
    H2_speicher_p[col_name] = network.stores_t.p["H2_Speicher"] # positiv entspricht Entladung 
    
    # H2 in DRI
    H2_in_DRI_p0[col_name] = -network.links_t.p0["H2_in_DRI"] # Input H2
    H2_in_DRI_p1[col_name] = -network.links_t.p1["H2_in_DRI"] # Output Stahl
    
    # Erdgas in DRI
    Erdgas_in_DRI_p0[col_name] = -network.links_t.p0["Erdgas_in_DRI"] # Input Erdgas
    Erdgas_in_DRI_p1[col_name] = -network.links_t.p1["Erdgas_in_DRI"] # Output Stahl
    
    # DRI / Lichtbogenofen
    DRI_p0[col_name] = network.links_t.p0["Lichtbogenofen"] # Strom in Lichtbogenofen
    DRI_p1[col_name] = -network.links_t.p1["Lichtbogenofen"] # Output in Stahl-Bus
    DRI_p2[col_name] = network.links_t.p2["Lichtbogenofen"] # Stahl aus DRI-Bus
    
    # Hochofen
    hochofen_p1[col_name] = -network.links_t.p1["Hochofen"] # Output in Stahl-Bus
    
    # Stahllager
    stahllager_e[col_name] = network.stores_t.e["Stahllager"]
    
    # Stahl-Load
    stahlload_p[col_name] = network.loads_t.p["Stahlproduktion"]
    

# Store- und StorageUnit-Ergebnisse kombinieren
df_storage_combined = pd.concat([df_stores, df_storage_units])


#%% Energiebedarf

# Kopie des Generators mit Leistungs-Zeitreihen erstellen
generators_p_energetisch = generators_p.copy()

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


#%% Auslastung Anlagen

auslastung_anlagen = pd.DataFrame(index = ["Wind_Onshore", "Wind_Offshore", "AEL"], columns = years)

for year in years:
    auslastung_anlagen.at["Wind_Onshore", year] = np.nan_to_num(generators_p[('Wind_Onshore', str(year))].mean() / df_generators.at["Wind_Onshore", str(year)], nan=0)
    auslastung_anlagen.at["Wind_Offshore", year] = np.nan_to_num(generators_p[('Wind_Offshore', str(year))].mean() / df_generators.at["Wind_Offshore", str(year)], nan=0)
    auslastung_anlagen.at["AEL", year] = np.nan_to_num(elektrolyse_p0[str(year)].mean() / df_links.at["AEL", str(year)], nan=0)
    
    mask = [
    ("Wind" in name or "PV" in name) and str(year) in str(j)
    for (name, j) in generators_p.columns
    ]
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    # Primärachse für Hochofen & DRI
    (generators_p.loc[:, mask].sum(axis=1)).plot(ax=ax1, label="Stromerzeugung")
    #DRI_p2[str(year)].plot(ax=ax1, label="Strom in Lichtbogenofen")
    elektrolyse_p0[str(year)].plot(ax=ax1, label="Strom in AEL")
    ax1.set_ylabel("Stahl [t]")
    ax1.set_xlabel("Zeit")
    ax1.grid(True, linestyle="--", alpha=0.5)
    # Sekundärachse für Stahllager
    ax2 = ax1.twinx()
    #stahllager_e[p].plot(ax=ax2, label="Füllstand Stahllager", color="green", linestyle="dotted")
    #ax2.set_ylabel("Füllstand Stahllager [t]")
    # Legenden zusammenführen
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, title="Link", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    # plotten
    plt.title(f"Stromproduktion und -verbrauch; {year}")
    plt.tight_layout()
    plt.show()
    
    

#%% Diagramm Emissionen
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
df_emissionen.T.plot(subplots=True, ax=axs)

axs[0].set_ylabel('t CO2')
axs[1].set_ylabel('t CO2')

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


#%% Diagramm Links

n_links = len(df_links)  # z. B. 5
nrows, ncols = 2, 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
axs = axs.flatten()

# Nur so viele Achsen wie Zeilen übergeben
df_links_subset = df_links.iloc[:len(axs)]
df_links_subset.T.plot(subplots=True, ax=axs[:len(df_links_subset)])

# Achsenbeschriftung und Titel setzen
titles = ['AEL', 'H2 in DRI stofflich', 'Erdgas in DRI stofflich', 'DRI Stahlproduktion', 'Hochofen Stahlproduktion', '']
ylabels = ['kg H2', 't Stahl aus H2', 't Stahl aus Erdgas', 't Stahl', 't Stahl', '']

for ax, title, ylabel in zip(axs, titles, ylabels):
    ax.set_title(title)
    ax.set_ylabel(ylabel)

# Leere Achsen ausschalten (falls weniger Daten als Subplots)
for i in range(len(df_links), len(axs)):
    axs[i].axis("off")

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


#%% Analyse Hochofen (Anzahl / Rückgang)

# Anzahl Hochöfen und max. Produktion
# p_nom_mod auslesen und mit Effizienz multiplizieren, weil df_links auch Output ist
p_nom_mod_hochofen = network.links.at["Hochofen", "p_nom_mod"] * network.links.efficiency["Hochofen"]
# Schleife über alle Spalten von df_links und Einlesen des Wertes aus Zeile "Hochofen"
print("\nAnzahl Hochöfen")
for label, p_nom_opt_hochofen in df_links.loc["Hochofen"].items():
    
    anz_hochofen = round(p_nom_opt_hochofen / p_nom_mod_hochofen)
    
    max_prod = round(hochofen_p1[label].sum())
    print(f"{label}: {anz_hochofen} Hochöfen mit einer jährlichen Produktion von {max_prod} t Stahl")


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

"""
# einzelne Jahre gesamte Stahlproduktion
for p in szenarien:
    fig, ax1 = plt.subplots(figsize=(14, 8))
    # Primärachse für Hochofen & DRI
    (hochofen_p1[p]+DRI_p1[p]).plot(ax=ax1, label="ges. Stahproduktion", color="black", alpha=0.7)
    #stahlload_p[p].plot(ax=ax1, label="Stahlbedarf", color="orange")
    ax1.set_ylabel("Stahl [t]")
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
    plt.title(f"Zeitlicher Verlauf der gesamten Stahlproduktion und Füllstand Stahllager; {p}")
    plt.tight_layout()
    plt.show()
"""

#%% Analyse Hochofenroute
"""
for p in szenarien:
    fig, ax1 = plt.subplots(figsize=(14, 8))
    # Primärachse für Hochofen & DRI
    (H2_in_DRI_p1[p]+Erdgas_in_DRI_p1[p]).plot(ax=ax1, label="Stahl aus H2 und Erdgas", color="orange")
    #Erdgas_in_DRI_p1[p].plot(ax=ax1, label="Kohle-Generator normiert auf Stahl", color="black", alpha=0.7)
    ax1.set_ylabel("Stahl [t]")
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
    plt.title(f"Zeitlicher Verlauf der gesamten Stahlproduktion und Füllstand Stahllager; {p}")
    plt.tight_layout()
    plt.show()
"""



#if __name__ == "__main__":
 #   main()
 
