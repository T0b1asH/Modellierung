import pypsa
import numpy as np
import pandas as pd
from sympy.physics.units import years

years = [2025 + i for i in range(2)]
df_haus = pd.concat([
                        pd.read_csv("Inputdaten/htw_P.csv")
                    ] * len(years), ignore_index=True)
df_pv = pd.concat([
                      pd.read_csv("Inputdaten/PV_Erzeugung_1kWp.csv",
                                  sep=";", decimal=',', skiprows=3)
                  ] * len(years), ignore_index=True)

#https://pypsa.readthedocs.io/en/stable/examples/multi-investment-optimisation.html
freq = 1

snapshots = pd.DatetimeIndex([])
for year in years:
    period = pd.date_range(
        start="{}-01-01 00:00".format(year),
        freq="{}h".format(str(freq)),
        periods=int(8760 / freq),
    )
    snapshots = snapshots.append(period)

network = pypsa.Network()
network.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
network.investment_periods = years

#----Komponenten----
network.add("Bus", name = "elektrisches Netz")
network.add("Load", name="el_verbrauch", bus="elektrisches Netz",
            p_set=list(df_haus['Haus_1']))

#network.add("Generator", name = "Gas_BHKW", bus = "elektrisches Netz",
            #p_nom_max = 10, p_nom_extendable = True,
            #marginal_cost = gaspreis, capital_cost = 3000, efficiency = 0.3)
#network.add("Generator", name = "PV", bus = "elektrisches Netz",
            #p_nom_max = 15, p_nom_extendable = True, p_max_pu = df_pv,
            #marginal_cost = pv_kosten, capital_cost = 1400)
#network.add("Load", name = "el_verbrauch", bus = "elektrisches Netz",
            #p_set = df_haus)

for year in years:
    network.add("Generator", name = "PV_{}".format(year), bus = "elektrisches Netz",
            p_nom_max = 15, p_nom_extendable = True, p_max_pu = list(df_pv["Leistung [kW]"]),
            marginal_cost = 0.05, capital_cost = 1400)
    network.add("Generator", name = "Gas_BHKW_{}".format(year), bus = "elektrisches Netz",
            p_nom_max = 10, p_nom_extendable = True,
            marginal_cost = 0.09, capital_cost = 3000, efficiency = 0.3)



network.optimize(solver_name = 'gurobi', multi_investment_periods=True, threads = 1)