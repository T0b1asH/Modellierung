
'Übung 03'
'b. BHKW'

network.add('Link', name = 'bhkw', bus0 = 'gas_bus', bus1 = 'electricity_grid', 
            bus2 = 'thermal', p_nom_extendable = True, capital_cost = bhkw_annuity,
            efficiency = bhkw_eff_el, efficiency2 = bhkw_eff_th)

'c. Wärmepumpe'

network.add('Link', name = 'heatpump', bus0 = 'electricity_grid', bus1 = 'thermal', 
            p_nom_extendable = True, capital_cost = heatpump_annuity, 
            p_max_pu = heatpump_p_max_pu, efficiency = heatpump_eff)

'd. CO2-Limit'

network.add('GlobalConstraint', name = 'co2-limt', 
            carrier_attribute = 'co2_emissions', sense = '<=', overwrite=True,
            constant = co2_1 * 0.3)

network.add('StorageUnit', name = 'Battery', bus = "electricity_grid", 
            p_nom_extendable = True, max_hours = 2, overwrite= True,
            capital_cost = battery_annuity / 2, cyclic_state_of_charge = True)


'Übung 04'
'b. Ergänzung Wärmepumpe'

# Netzwerk erweitern
network.add('Link', name = 'heatpump', 
            bus0 = 'electricity', 
            bus1 = 'thermal',
            p_nom = hp_p_nom, 
            committable = True,  
            p_max_pu = heatpump_p_max_pu,
            p_min_pu = heatpump_p_min_pu, 
            efficiency = heatpump_eff)


'Übung 05'
'MultiInvest'

# Code from https://pypsa.readthedocs.io/en/latest/examples/multi-investment-optimisation.html
snapshots = pd.DatetimeIndex([])
for year in years:
    period = pd.date_range(
        start="{}-01-01 00:00".format(year),
        freq="{}h".format(str(freq)),
        periods=int(8760 / freq),
    )
    snapshots = snapshots.append(period)
    
network.snapshots = pd.MultiIndex.from_arrays([snapshots.year, snapshots])
network.investment_periods = years

# Code from https://pypsa.readthedocs.io/en/latest/examples/multi-investment-optimisation.html
network.investment_period_weightings["years"] = list(np.diff(years)) + [10]


#%% ChatGPT

'1) Baue eine Brennstoffzelle ein, die den produzierten Wasserstoff nutzt'
'und die EE-Produktion am Strom-Bus unterstützt.'

# --- Brennstoffzelle (Fuel Cell) ---
LHV_H2 = 33.33                          # kWh/kg  (wie bei AEL genutzt)
fc_eta_el = 0.55                        # el. Wirkungsgrad (LHV-Basis)
fc_capex_eur_per_kWe = 1100             # €/kWel (Beispiel)
fc_vom_eur_per_kWhel = 0.01             # €/kWhel variable O&M

# Umrechnung: Link-p_nom ist am INPUT-Bus (kg H2/h).
# €/kWel -> €/ (kg/h H2)   via kWel = fc_eta_el * LHV_H2 * (kg/h)
fc_capital_cost = fc_capex_eur_per_kWe * (fc_eta_el * LHV_H2) * annuitaet_20a
fc_marginal_cost = fc_vom_eur_per_kWhel * (fc_eta_el * LHV_H2)

# Brennstoffzelle (H2 -> Strom)
network.add(
    "Link",
    name="Brennstoffzelle",
    bus0="Wasserstoff",                 # INPUT (kg/h)
    bus1="Strom",                       # OUTPUT (kW)
    efficiency=fc_eta_el * LHV_H2,      # kW_el pro (kg/h) H2
    p_nom_extendable=True,
    # p_nom_mod=5_000,                  # optional: Blockgröße in kg/h H2
    capital_cost=fc_capital_cost,
    marginal_cost=fc_marginal_cost
)


'2) Implementiere ein BHKW, das die EE-Produktion am Strom-Bus erweitert'
'und eine neue Wärmelast deckt.'

# --- Fernwärmebedarf ---
fernwaerme_last_MWh_a = 150_000        # z.B. 150 GWh/a
# --- BHKW ---
bhkw_eta_el = 0.35
bhkw_eta_th = 0.50                     # Gesamt ~0.85, Beispiel
bhkw_capex_eur_per_kWe = 800           # €/kWel
bhkw_vom_eur_per_kWhel = 0.005         # €/kWhel

# p_nom ist am INPUT-Bus (kW_fuel = kWh Erdgas/h).
# €/kWel -> €/kW_fuel via kWel = bhkw_eta_el * kW_fuel
bhkw_capital_cost = bhkw_capex_eur_per_kWe * bhkw_eta_el * annuitaet_20a
bhkw_marginal_cost = bhkw_vom_eur_per_kWhel * bhkw_eta_el

# Neuer Wärmenetz-Bus
network.add("Bus", name="Fernwaerme")

# Wärmelast (konstant über das Jahr; gern durch Zeitreihe ersetzen)
network.add(
    "Load",
    name="Fernwaermelast",
    bus="Fernwaerme",
    p_set = fernwaerme_last_MWh_a * 1e3 / 8760    # in kW_th
)

network.add(
    "Link",
    name="BHKW",
    bus0="Erdgas",                      # INPUT: kW_fuel (passt zu deinem Erdgas-Bus)
    bus1="Strom",                       # OUTPUT Strom
    bus2="Fernwaerme",                  # OUTPUT Wärme
    efficiency=bhkw_eta_el,             # kWel pro kW_fuel
    efficiency2=bhkw_eta_th,            # kWth pro kW_fuel
    p_nom_extendable=True,
    # p_nom_mod=1_000,                  # optional: blockweise Zubau in kW_fuel
    capital_cost=bhkw_capital_cost,
    marginal_cost=bhkw_marginal_cost
)

network.add(
    "Store",
    name="Fernwaermespeicher",
    bus="Fernwaerme",
    e_nom_extendable=True,
    capital_cost=15 * annuitaet_25a,    # €/kWh_th (Beispiel)
    marginal_cost=0.001,
    standing_loss=0.00008               # ~0.8%/Tag
)


'3) Ergänze eine Wärmepumpe, die die Abwärme des Lichtbogenofens'
'verwendet, um ein Fernwärmenetz zu speisen.'

# --- Abwärme vom EAF (pro t Stahl) ---
eaf_abwaerme_kWhth_pro_t = 200         # Beispielwert, kalibrierbar

# --- Wärmepumpe (EAF -> Fernwärme) ---
wp_cop_ref = 3.0                        # konst. COP; optional dynamisieren
wp_capex_eur_per_kWel = 600             # €/kWel
wp_vom_eur_per_kWhel = 0.003            # €/kWhel

wp_capital_cost = wp_capex_eur_per_kWel * annuitaet_20a
wp_marginal_cost = wp_vom_eur_per_kWhel

network.add("Bus", name="Abwaerme_EAF")

network.add(
    "Link",
    name="Lichtbogenofen",
    bus0="DRI",
    bus1="Stahl",
    bus2="Strom",                                   # zusätzlicher INPUT Strom
    bus3="Abwaerme_EAF",                            # OUTPUT: Abwärme (kW_th)
    efficiency2=-DRI_stromverbrauch_pro_t_stahl,    # kW_el pro t Stahl (Input)
    efficiency3= eaf_abwaerme_kWhth_pro_t,          # kW_th pro t Stahl (Output)
    p_nom_extendable=True,
    # p_nom_mod = 2_500_000,                        # falls gewünscht
    p_min_pu=0.9,
    ramp_limit_up=0.041,
    ramp_limit_down=0.041,
    capital_cost=DRI_Lichtbogen_baukosten * annuitaet_30a,
    marginal_cost=DRI_Lichtbogen_betriebskosten
)

network.add(
    "Link",
    name="WP_EAF",
    bus0="Strom",                       # INPUT: el. Leistung (kWel)
    bus1="Fernwaerme",                  # OUTPUT: kWth
    bus2="Abwaerme_EAF",                # zusätzlicher INPUT: kWth (Quelle)
    efficiency=wp_cop_ref,              # kWth_out pro kWel_in
    efficiency2=-(wp_cop_ref - 1.0),    # kWth_quelle pro kWel_in (negativ = weiterer Input)
    p_nom_extendable=True,
    capital_cost=wp_capital_cost,
    marginal_cost=wp_marginal_cost
)

# (Optional) Abwärme-„Dump“, damit überschüssige EAF-Abwärme abgeführt werden kann:
network.add(
    "Generator",
    name="Abwaerme_Dump",
    bus="Abwaerme_EAF",
    p_nom_extendable=True,
    sign=-1,                            # Senke
    marginal_cost=0.0
)


