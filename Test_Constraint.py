import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.physics.units import years
import pyomo.environ as pyo
from pyomo.environ import Constraint, summation



#Variablen definieren
years = [2025 + i for i in range(4)]
freq = 1

# Snapshot-Zeiten erzeugen
#https://pypsa.readthedocs.io/en/stable/examples/multi-investment-optimisation.html

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

network.add(
    "Carrier",
    name = "Gas",
    co2_emissions = 1 #kg/kWh
)

network.add(
    "Carrier",
    name = "EE",
    co2_emissions = 0.0 #kg/kWh
)



network.add(
    "Bus",
    name = "Strom"
)



co2_limits = {
    2025:20000000,
    2026:20000000,
    2027:0,
    2028:0
}

for year in years:

    network.add(
        "Load",
        name="Stromlast_{}".format(year),
        bus="Strom",
        p_set=2000,
        investment_periods=year
    )
    '''
    
    network.add(
        "Generator",
        name = "Gas-Generator",
        bus = "Strom",
        p_nom = 2000,
        p_nom_mod = 500,
        carrier = "Gas",
        lifetime = 2,
        build_year = 2024,
        marginal_cost = 0.06,
        capital_cost = 100
    )
    '''

    network.add(
        "Generator",
        name = "Gas-Generator_{}".format(year),
        bus = "Strom",
        p_nom_extendable = True,
        p_nom_mod = 500,
        p_nom_max = 2000,
        carrier = "Gas",
        lifetime = 2,
        build_year = year,
        marginal_cost = 0.06,
        capital_cost = 100
    )

    network.add(
        "Generator",
        name = "EE-Generator_{}".format(year),
        bus = "Strom",
        p_nom_extendable = True,
        p_nom_mod = 500,
        p_nom_max = 2000,
        carrier = "EE",
        lifetime = 2,
        build_year = year,
        marginal_cost = 0.06,
        capital_cost = 190
    )

    network.add(
        "GlobalConstraint",
        "emission_limit_{}".format(year),
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_limits[year],
        investment_period=year
    )


m = network.optimize.create_model()
for year in years:
    gens = network.generators
    active_gens = gens[
        (gens['build_year'] <= year) &
        (year < gens['build_year'] + gens['lifetime'])
    ]

    Gesamtkapaziteat = 0
    Gesamtenergie = 0
    for gens in active_gens.index:
        Gesamtkapaziteat += m.variables["Generator-p_nom"].loc[gens]
        constraint_name = f"Erzeugungslimit_{year}_{gens}"
        m.add_constraints(Gesamtkapaziteat <= 2000, name = constraint_name)


# Optimierung durchführen
network.optimize.solve_model(
    solver_name='gurobi',
    multi_investment_periods=True,
    threads=1,
)

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

#CO2 Emissionen ermittlen
df_carrier = network.carriers
df_generators = network.generators.carrier
standard_co2_emissions = round((network.generators_t.p.sum() / network.generators.efficiency *
                                pd.merge(df_carrier, df_generators, left_index=True, right_on='carrier')
                                ['co2_emissions'])).sum()
print(network.carriers)
print(network.generators[['carrier', 'p_nom_opt']])
print(network.loads[['p_set']])
'''
for year in years:
    energy_EE = network.generators_t.p.xs(year, level=0)['EE-Generator_{}'.format(year)].sum()
    print("Energieerzeugung_EE: {}".format(year), energy_EE)
    energy_Gas = network.generators_t.p.xs(year, level=0)['Gas-Generator_{}'.format(year)].sum()
    print("Energieerzeugung_Gas: {}".format(year), energy_Gas)
'''


print(network.global_constraints)
print(standard_co2_emissions)
print("-------------------------------------------------------------")
carrier_emissions = network.carriers["co2_emissions"]  # kg/MWh
emission_factors = network.generators["carrier"].map(carrier_emissions)
emissions_kg = (network.generators_t.p * emission_factors).sum().sum()

print(f"Gesamte CO₂-Emissionen: {emissions_kg:.2f} kg CO₂")
