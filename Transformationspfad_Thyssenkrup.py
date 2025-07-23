import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.physics.units import years


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

# Netzwerk aufbauen
def erstelle_network(years, snapshots, df_netzlast, df_pv):
    network = pypsa.Network()
    network.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
    network.investment_periods = years

    #----Komponenten----
    network.add(
        "Bus",
        name = "elektrisches Netz")

    network.add(
        "Load",
        name="el_verbrauch",
        bus="elektrisches Netz",
        p_set= list(df_netzlast["Netzlast [kW]"])
    )

    network.add(
        "Generator",
        name="Netzanschluss",
        bus="elektrisches Netz",
        p_set=np.inf,
        marginal_cost=0.3
    )


    # Erzeuger pro Jahr hinzufügen
    for year in years:
        network.add(
            "Generator",
            name = "PV_{}".format(year),
            bus = "elektrisches Netz",
            p_nom_max = 15,
            p_nom_extendable = True,
            p_max_pu = list(df_pv["Leistung [kW]"]),
            marginal_cost = 0.05,
            capital_cost = 1400,
            build_year=year,
            lifetime = 2
        )

        network.add(
            "Generator",
            name = "Gas_BHKW_{}".format(year),
            bus = "elektrisches Netz",
            p_nom_max = 10,
            p_nom_extendable = True,
            marginal_cost = 9,
            capital_cost = 3000,
            efficiency = 0.3,
            build_year = year,
            lifetime = 1
        )
    return network

def main():
    years = [2025 + i for i in range(4)]
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


if __name__ == "__main__":
    main()