import pypsa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


#---------------Variablen einlesen----------------------
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

# Strompreisentwicklung
strompreise = {  # Angaben in €/kWh
    2025: 0.13,  # bekannt
    2026: 0.128,  # Prognosen
    2031: 0.076,
    2050: 0.059
}
strompreise = pd.Series(strompreise)
strompreise = strompreise.reindex(range(2025, 2051))
strompreise = strompreise.interpolate(method="linear")  # Interpolation für fehlende Jahre

# Wasserstoffpreisentwicklung
wasserstoffpreise = {
    2025: 151,
    2026: 137.83,
    2027: 124.47,
    2028: 111.94,
    2029: 101.28,
    2030: 93.5,
    2031: 89.31,
    2032: 88.12,
    2033: 89,
    2034: 91.35,
    2035: 93.3,
    2036: 95.04,
    2037: 96.19,
    2038: 96.82,
    2039: 97.03,
    2040: 96.9,
    2041: 96.52,
    2042: 95.91,
    2043: 95.09,
    2044: 94.08,
    2045: 92.9,
    2046: 91.56,
    2047: 90.08,
    2048: 88.49,
    2049: 86.79,
    2050: 85
}


#------------------------Daten einlesen-------------------------------
def lade_daten(snapshots):
    # Dateipfad einlesen
    dateipfad_code = os.path.dirname(os.path.realpath(__file__))  # Übergeordneter Ordner, in dem Codedatei liegt
    ordner_input = os.path.join(dateipfad_code, 'Inputdaten')  # Unterordner "Inputdaten"

    # Stahlproduktion
    df_stahl = pd.read_csv(os.path.join(ordner_input, "Stahlproduktion/Stahlproduktion.csv"), sep=";", decimal=",")
    df_stahl = df_stahl.set_index("Jahr")


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

    # Dummy Load
    df_netzlast = pd.read_csv(os.path.join(ordner_input, "Wind/Offshore.csv"), skiprows=3,
                              usecols=["electricity"]) * 1000
    df_netzlast.index = snapshots

    return df_stahl, df_netzlast, df_pv, df_wind


# def erstelle_network(years, snapshots, df_netzlast, df_netzlast_alle_jahre, df_pv, all_co2_emissions, all_co2_emissions_strom, stahlproduktion, stromverbrauch_pro_kg_stahl):
def erstelle_network(df_netzlast, df_pv, df_wind, df_stahl,snapshots):
    network = pypsa.Network()
    network.set_snapshots(snapshots)

    # Carrier für CO2-Emissionen
    network.add("Carrier", name="EE", co2_emissions=0)
    network.add("Carrier", name="Kohle", co2_emissions=co2_kohle)

    # Busse
    network.add("Bus", name="elektrisches Netz", carrier="electricity")  # Einheit kWh   # Vorschlag Name: "strom_bus"
    network.add("Bus", name="Wasserstoff", carrier="H2")  # Einheit kWh   # Vorschlag Name: "H2_bus"
    network.add("Bus", name="stahl_bus", carrier="steel")  # Einheit t
    network.add("Bus", name="kohle_bus", carrier="coal")  # Einheit t

    # Dummy Load
    network.add(
        "Load",
        name="el_verbrauch",
        bus="elektrisches Netz",
        p_set= df_netzlast
    )


    network.add("Carrier", name="Stromnetz",
                co2_emissions=co2_strommix[2025])

    network.add(
        "Generator",
        name="Netzbezug",
        bus="elektrisches Netz",
        p_nom_extendable=True,
        capital_cost=147.54,
        # Bereiche für Leistungspreis einfügen?, fixer Wert: https://www.netze-duisburg.de/fileadmin/user_upload/Netz_nutzen/Netzentgelte/Strom/241217_Netze_Duisburg_-_Endg%C3%BCltiges_Preisblatt_Strom_2025.pdf
        marginal_cost=strompreise[2025],
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

    # Batteriespeicher
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

    # Elektrolysen
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

    # Wasserstoff-Bus
    network.add(
        "Generator",
        name="H2-Pipeline",
        bus="Wasserstoff",
        p_nom_extendable=True,
        capital_cost=25,
        # https://www.bundesnetzagentur.de/SharedDocs/Pressemitteilungen/DE/2025/20250714_Hochlauf.html
        marginal_cost=wasserstoffpreise[2025],
        carrier="EE"
    )

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

    network.add(
        "Load",
        name="H2_energetisch",
        bus="Wasserstoff",
        p_set=(df_stahl.loc[2025, "Produzierte Stahlmenge [t/a]"] / 8760) * 3000  # kWh H2 pro Stunde
    )

    # Stahl-Bus
    network.add(
        "Load",
        name="Stahlproduktion",
        bus="stahl_bus",
        p_set=df_stahl.loc[2025, "Produzierte Stahlmenge [t/a]"] / 8760  # Tonnen Stahl pro Stunde
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
        efficiency=1 / 1.6,  # 1t Stahl benötigt 1,6t Kohle; 750kg energetisch und 850kg stofflich
        build_year=1,
        p_nom_extendable=True
    )
    '''
    # CO2-Constraint
    network.add(
        "GlobalConstraint",
        "emission_limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        # constant=all_co2_emissions[year],
        constant=70000e9,
    )
    '''

    return network, snapshots


def main():
    snapshots = pd.RangeIndex(8760)
    df_stahl, df_netzlast, df_pv, df_wind = lade_daten(snapshots)
    network, snapshots = erstelle_network(df_stahl, df_netzlast, df_pv, df_wind, snapshots)

    # Optimierung durchführen
    network.optimize(
        solver_name='gurobi',
        multi_investment_periods=True,
        threads=1)


if __name__ == "__main__":
    main()


