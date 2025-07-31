import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.physics.units import years

#Variablen definieren
years = [2025 + i for i in range(4)]
freq = 1

co2_emissions_gas = 358 #kg/MWh https://www.volker-quaschning.de/datserv/CO2-spez/index_e.php
co2_emissions_strom = 407 #kg/MWh

# Inputdaten laden und für alle Jahre duplizieren (aktuell ändern sich die verbräuche nicht)
def lade_daten(years):
    df_haus = pd.concat(
        [pd.read_csv("Inputdaten/htw_P.csv")] * len(years),
        ignore_index=True
    )

    df_netzlast = pd.concat(
        [pd.read_csv("Inputdaten/data_PyPSA_1.csv", nrows=8760)] * len(years),
        ignore_index=True
    )

    df_pv = pd.concat(
        [pd.read_csv("Inputdaten/PV_Erzeugung_1kWp.csv", sep=";", decimal=',', skiprows=3)] * len(years),
        ignore_index=True
    )
    return df_netzlast, df_pv

# Snapshot-Zeiten erzeugen
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

''' 
def strompreis(years):
    for year in years:
        anfangspreis = 0.35
        strompreis = anfangspreis * (1 + 0.03)**year
    return strompreis
'''
# Netzwerk aufbauen
def erstelle_network(years, snapshots, df_netzlast, df_pv):
    network = pypsa.Network()
    network.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
    network.investment_periods = years

    #----Komponenten----
    network.add("Carrier",
                name = "Gas",
                co2_emissions = co2_emissions_gas
                )

    network.add("Carrier",
                name = "PV",
                co2_emissions = 0
                )

    network.add(
        "Bus",
        name = "elektrisches Netz"
    )

    network.add(
        "Bus",
        name = "Wasserstoff"
    )

    network.add(
        "Load",
        name="el_verbrauch",
        bus="elektrisches Netz",
        p_set= list(df_netzlast["Netzlast [kW]"])
    )

    # Erzeuger pro Jahr hinzufügen
    for year in years:
        network.add("Carrier",
                    name="Strom_{}".format(year),
                    co2_emissions=407*(1-0.3)**(year-2025)
                    )

        network.add(
            "Generator",
            name="Netzanschluss_{}".format(year),
            bus="elektrisches Netz",
            #p_nom_max=100000,
            p_nom_extendable=True,
            #capital_cost=10000, sau schwer zu implementieren #GPT sagt häufig sind das Pausdchalpreise, in dem Fall zahlt man 10.000 Euro für 100MW
            marginal_cost=0.35 * (1 + 0.03)**(year-2025),
            build_year=year,
            lifetime=1,
            carrier="Strom_{}".format(year),
        )

        network.add(
            "Generator",
            name = "Gas_BHKW_{}".format(year),
            bus = "elektrisches Netz",
            #p_nom_max = 10,
            p_nom_extendable = True,
            marginal_cost = 0.09,
            capital_cost = 2000,
            efficiency = 0.3,
            build_year = year,
            lifetime = 1,
            carrier = "Gas"
        )

        # PV Aufdach
        network.add(
            "Generator",
            name="PV_Aufdach_{}".format(year),
            bus="elektrisches Netz",
            p_max_pu=list(df_pv["Leistung [kW]"]),
            p_nom_extendable=True,
            #p_nom_max=500,
            # maximal installierbare Leistung auf der verfügbaren Fläche; muss als Constraint, damit nicht jedes Jahr das p_nom_max neu installiert werden kann
            #p_nom_min=50,
            # mindestens zu installierende Leistung, weil winzige PV-Anlagen für den Maßstab keinen Sinn ergeben
            build_year=year,
            lifetime=2,
            capital_cost=100,
            marginal_cost=0.01
        )
        '''
        # PV extern via PPA
        network.add(
            "Generator",
            name="PV_extern_".format(year),
            bus="elektrisches Netz",
            p_max_pu=df_pv_ppa["pv_pu"].values,
            p_nom_extendable=True,
            build_year=year,
            lifetime=20,
            marginal_cost=0.02  # nur marginal_cost wegen PPA
        )

        # Wind onshore
        network.add(
            "Generator",
            name="Wind_onshore_".format(year),
            bus="elektrisches Netz",
            p_max_pu=df_wind_ppa["wind_pu"].values,
            p_nom_extendable=True,
            build_year=year,
            lifetime=25,
            marginal_cost=0.02  # nur marginal_cost wegen PPA
        )

        # Wind offshore
        network.add(
            "Generator",
            name="Wind_offshore_".format(year),
            bus="elektrisches Netz",
            p_max_pu=df_wind_ppa["wind_pu"].values,
            p_nom_extendable=True,
            build_year=year,
            lifetime=25,
            marginal_cost=0.02  # nur marginal_cost wegen PPA
        )
        '''
        # Batteriespeicher
        network.add(
            "StorageUnit",
            name="Batterie_{}".format(year),
            bus="elektrisches Netz",
            p_nom_extendable=True,
            build_year=year,
            lifetime=2,
            capital_cost=300,
            #efficiency_store=0.95,
            #efficiency_dispatch=0.95,
            #max_hours=4
        )

        # PEM-Elektrolyse
        network.add(
            "Link",
            name="PEM_".format(year),
            bus0="elektrisches Netz",
            bus1="Wasserstoff",
            p_nom_extendable=True,
            efficiency=0.7,
            build_year=year,
            lifetime=2,
            capital_cost=900
        )

        # HT-Elektrolyse (Strom + Wärme)
        network.add(
            "Link",
            name="HT_".format(year),
            bus0="elektrisches Netz",
            bus1="Wasserstoff",
            p_nom_extendable=True,
            efficiency=0.8,
            build_year=year,
            lifetime=2,
            capital_cost=1200
            # nur aktiv, wenn Lichbogenofen läuft (z.B. auf min. 80%)
        )

        # H2-Speicher
        network.add(
            "Store",
            name="wasserstoffspeicher",
            bus="Wasserstoff",
            e_nom_extendable=True,
            build_year=year,
            lifetime=2,
            capital_cost=20
            #e_cyclic=True
        )

        # H2-Verbraucher
        network.add(
            "Load",
            name="H2_brenner",
            bus="Wasserstoff",
            p_set=0
        )

        network.add(
            "Load",
            name="H2_dri",
            bus="Wasserstoff",
            p_set=0
        )
        '''
        # Last Lichtbogenofen ab in die Schleife
        network.add(
            "Load",
            name="Lichtbogenofen",
            bus="elektrisches Netz",
            p_set=df_load["Lichtbogenofen [kW]"].values
        )
        '''
        network.add(
            "GlobalConstraint",
            "emission_limit_{}".format(year),
            carrier_attribute="co2_emissions",
            sense="<=",
            #constant=2000000,
            constant=250000 * (1 - 0.1) ** (year - 2025),
            investment_period = year
        )
    return network


def main():
    years = [2025 + i for i in range(6)]
    freq = 1
    df_netzlast, df_pv = lade_daten(years)
    snapshots = erstelle_snapshots(years, freq)
    network = erstelle_network(years, snapshots, df_netzlast, df_pv)


    # Optimierung durchführen
    network.optimize(
        solver_name = 'gurobi',
        multi_investment_periods=True,
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

    print(network.generators)
    print(network.global_constraints)
    print(network.generators_t.p)
    print(network.generators[["marginal_cost", "carrier", "build_year"]])
    print(network.carriers[["co2_emissions"]])
    print("Gesamtkosten vorher: 15322.45 €/a")
    print(f"Gesamtkosten: {network.objective:.2f} €/a")

    carrier_emissions = network.carriers["co2_emissions"]  # kg/MWh
    emission_factors = network.generators["carrier"].map(carrier_emissions)
    emissions_kg = (network.generators_t.p * emission_factors).sum().sum()
    #emissions_t = emissions_kg / 1000
    print(f"Gesamte CO₂-Emissionen: {emissions_kg:.2f} kgCO₂")

    print(network.generators[["carrier", "marginal_cost", "build_year", "p_nom_opt"]])
    print(network.carriers[["co2_emissions"]])
    print(network.global_constraints[["sense", "carrier_attribute", "constant"]])

if __name__ == "__main__":
    main()



