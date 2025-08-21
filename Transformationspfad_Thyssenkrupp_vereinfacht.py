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
co2_strommix = pd.Series(co2_strommix)
co2_strommix = co2_strommix.reindex(range(2022, 2051))
co2_strommix = co2_strommix.interpolate(method="linear")  # Interpolation für fehlende Jahre
co2_kohle = 769230 + 947690  # Gramm / Tonne Kohle; stofflich + energetisch

stahlproduktion = 9_500_000/8760# kg
strompreis = 0.13 # €/kwh
netzbezug_capitalcost = 147.54 # €/kWa
wasserstoffpreise = 151 # €/kg
wasserstoffverbrauch_pro_kg_stahl = 60  # pro kg Stahl
stromverbrauch_pro_kg_stahl = 3.65 # kWh/kg
betriebskosten_DRI = 1 # €
baukosten_DRI = 518_460_000 # €
betriebskosten_Hochofen = 1 # €
baukosten_Hochofen = 133_071_400 # €
kohleverbauch_pro_kg_Stahl = 60 #kg/kg_Stahl


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


def erstelle_network(df_pv, df_wind,snapshots):
    network = pypsa.Network()
    network.set_snapshots(snapshots)

    # Carrier für CO2-Emissionen
    network.add("Carrier", name="EE", co2_emissions=0)
    network.add("Carrier", name="Kohle", co2_emissions=co2_kohle)
    network.add("Carrier", name="H2")
    network.add("Carrier", name="Stromnetz", co2_emissions=co2_strommix)

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
        efficiency=0.7,
        p_nom_extendable=True,
        p_nom_min=2000,
        # p_nom_max = ,
        capital_cost=0.875,
        # Alle Kosten der Elektrolysen könnte man nochmal prüfen, da wir einige verschiedene haben
        marginal_cost=0.875 * 0.04,
    )

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

    network.add(
        "Link",
        name="H2_stofflich",
        bus0="Wasserstoff",
        bus1="stahl_bus",
        efficiency=1 / (60 * 33.33),  # 1t Stahl benötigt 60kg H2 mit 33,33kWh/kg
        p_nom_extendable=True
    )


    # Stahl-Bus
    network.add(
        "Load",
        name="Stahlproduktion",
        bus="stahl_bus",
        #p_set=df_stahl.loc[2025, "Produzierte Stahlmenge [t/a]"] / 8760  # Tonnen Stahl pro Stunde
        p_set=stahlproduktion/9760
    )

    # Kohle-Bus
    network.add(
        "Generator",
        name="Kohle",
        bus="kohle_bus",
        p_nom_extendable=True,
        marginal_cost=90,
        carrier="Kohle"
    )

    network.add(
        "Link",
        name="Hochofen",
        bus0="kohle_bus",
        bus1="stahl_bus",
        efficiency=kohleverbauch_pro_kg_Stahl, #1 / 1.6,  # 1t Stahl benötigt 1,6t Kohle; 750kg energetisch und 850kg stofflich
        p_nom_extendable=True,
        marginal_cost=betriebskosten_Hochofen,
        capital_cost=baukosten_Hochofen
    )

    network.add(
        "Link",
        name="DRI",
        bus0="Wasserstoff",
        bus1="stahl_bus",
        bus2="elektrisches Netz",
        efficiency=wasserstoffverbrauch_pro_kg_stahl,
        efficiency2=stromverbrauch_pro_kg_stahl,
        p_nom_extendable=True,
        marginal_cost = betriebskosten_DRI,
        capital_cost = baukosten_DRI

    )

    network.add(
        "StorageUnit",
        name="Stahllager",
        bus="Stahl",
        p_nom_extendable=True,
        # p_nom_min = ,
        # p_nom_max = ,
        #marginal_cost=Betriebskosten_Stahllager,
        #capital_cost=Baukosten_Stahllager,
        # state_of_charge_initial = 0,
        #max_hours=4,  # Stunden bei voller Leistung -> bestimmt Kapazität
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


def co2_reduktion(network, co2_limits):
    df_p_nom_opt = pd.DataFrame(index=network.generators.index)
    df_emissionen = pd.DataFrame(index=["CO2-Limit", "CO2-Emissionen"])

    for year in co2_limits():
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


def main():
    snapshots = pd.RangeIndex(8760)
    df_stahl, df_pv, df_wind = lade_daten(snapshots)
    network = erstelle_network(df_pv, df_wind, snapshots)

    # Optimierung durchführen
    network.optimize(
        solver_name='gurobi',
        multi_investment_periods=True,
        threads=1)
    '''
    df_results = []

    df_results[str(round(co2_limit * 100, 0)) + '%'] = [network.statistics()["Capital Expenditure"].sum(),
                                                        network.statistics()["Operational Expenditure"].sum()] + list(
        network.generators.p_nom_opt) + list(network.links.p_nom_opt) + list(network.stores.e_nom_opt)

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
    df_results.T.plot(subplots=True, ax=axs)

    axs[0, 0].set_ylabel('kW')
    axs[0, 1].set_ylabel('kW')
    axs[0, 2].set_ylabel('kW')
    axs[1, 0].set_ylabel('kW')
    axs[1, 1].set_ylabel('kW')
    axs[1, 2].set_ylabel('t')
    axs[2, 0].set_ylabel('kW')
    axs[2, 1].set_ylabel('€')
    axs[2, 2].set_ylabel('kW')
    axs[3, 0].set_ylabel('kW')
    axs[3, 1].set_ylabel('kWh')
    axs[3, 2].set_ylabel('kWh')

    axs[0, 0].set_title('PV_Sued')
    axs[0, 1].set_title('PV_Ost_West')
    axs[0, 2].set_title('Wind_Onshore')
    axs[1, 0].set_title('Wind_Offshore')
    axs[1, 1].set_title('Netzbezug')
    axs[1, 2].set_title('CO2 Emissionen')
    axs[2, 0].set_title('Hochofen')
    axs[2, 1].set_title('Investitionskosten')
    axs[2, 2].set_title('Wasserstoffverbrauch')
    axs[3, 0].set_title('Kohleverbauch')
    axs[3, 1].set_title('el. Speicherkapazität')
    axs[3, 2].set_title('H2 Speicherkapazität')

    fig.suptitle('Analyse des Transformationspfads eines Stahlherstellers')
    fig.tight_layout()

    # Generatorleistungen übereinander
    ax = network.generators_t.p.plot(alpha=0.5)

    # Nur die Jahre (Periods) als x-Tick-Labels anzeigen
    ax.set_xticks(range(0, len(network.generators_t.p), 2920))  # Setze Positionen
    ax.set_xticklabels(years)
    ax.set_xlabel('year')
    ax.set_ylabel('P [MW]')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    '''

if __name__ == "__main__":
    main()


