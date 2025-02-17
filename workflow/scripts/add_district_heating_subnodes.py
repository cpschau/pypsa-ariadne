# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import sys
import os

paths = [
    "workflow/submodules/pypsa-eur/scripts",
    "pypsa-ariadne/workflow/submodules/pypsa-eur/scripts",
    "../submodules/pypsa-eur/scripts",
    "../submodules/pypsa-eur/",
]
for path in paths:
    sys.path.insert(0, os.path.abspath(path))
from prepare_network import maybe_adjust_costs_and_potentials
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

import shapely
import rasterio
from atlite.gis import ExclusionContainer
from atlite.gis import shape_availability


def add_ptes_limit(
    subnodes,
    corine,
    natura,
    groundwater,
    codes,
    max_groundwater_depth,
    ptes_potential_scalar,
):
    """
    Add PTES limit to subnodes according to land availability within city regions.
    """
    dh_systems = subnodes.copy()
    dh_systems["lau_shape"] = dh_systems["lau_shape"].apply(shapely.wkt.loads)
    dh_systems = dh_systems.set_geometry("lau_shape")
    dh_systems.crs = "EPSG:4326"
    dh_systems = dh_systems.to_crs(3035)

    excluder = ExclusionContainer(crs=3035, res=100)

    # Exclusion of unsuitable areas
    excluder.add_raster(corine, codes=codes, invert=True, crs=3035)

    # Exclusion of NATURA protected areas
    excluder.add_raster(natura, codes=[1], invert=True, crs=3035)

    # Calculation of shape availability and transformation of raster data to geodataframe
    band, transform = shape_availability(dh_systems.lau_shape, excluder)
    masked_data = band
    row_indices, col_indices = np.where(masked_data != corine.nodata)
    values = masked_data[row_indices, col_indices]

    x_coords, y_coords = rasterio.transform.xy(transform, row_indices, col_indices)
    eligible_areas = pd.DataFrame({"x": x_coords, "y": y_coords, "eligible": values})
    eligible_areas = gpd.GeoDataFrame(
        eligible_areas,
        geometry=gpd.points_from_xy(eligible_areas.x, eligible_areas.y),
        crs=corine.crs,
    )

    # Area calculation with buffer to match raster resolution of 100mx100m
    eligible_areas["geometry"] = eligible_areas.geometry.buffer(50, cap_style="square")
    merged_data = eligible_areas.union_all()
    eligible_areas = (
        gpd.GeoDataFrame(geometry=[merged_data], crs=eligible_areas.crs)
        .explode(index_parts=False)
        .reset_index(drop=True)
    )

    # Divide geometries with boundaries of dh_systems
    eligible_areas = gpd.overlay(eligible_areas, dh_systems, how="intersection")
    eligible_areas = gpd.sjoin(
        eligible_areas, dh_systems.drop("Stadt", axis=1), how="left", rsuffix=""
    )[["Stadt", "geometry"]].set_geometry("geometry")

    # filter for eligible areas that are larger than 10000 m^2
    eligible_areas = eligible_areas[eligible_areas.area > 10000]

    # Find closest value in groundwater dataset and kick out areas with groundwater level > threshold
    eligible_areas["groundwater_level"] = eligible_areas.to_crs("EPSG:4326").apply(
        lambda a: groundwater.sel(
            lon=a.geometry.centroid.x, lat=a.geometry.centroid.y, method="nearest"
        )["WTD"].values[0],
        axis=1,
    )
    eligible_areas = eligible_areas[
        eligible_areas.groundwater_level < max_groundwater_depth
    ]

    # Combine eligible areas by city
    eligible_areas = eligible_areas.dissolve("Stadt")

    # Calculate PTES potential according to Dronninglund and DEA parameters
    eligible_areas["area_m2"] = eligible_areas.area
    eligible_areas["nstorages_pot"] = eligible_areas.area_m2 / 10000
    eligible_areas["storage_pot_mwh"] = eligible_areas["nstorages_pot"] * 4500

    subnodes.set_index("Stadt", inplace=True)
    subnodes["ptes_pot_mwh"] = (
        eligible_areas.loc[subnodes.index.intersection(eligible_areas.index)][
            "storage_pot_mwh"
        ]
        * ptes_potential_scalar
    )
    subnodes["ptes_pot_mwh"] = subnodes["ptes_pot_mwh"].fillna(0)
    subnodes.reset_index(inplace=True)

    return subnodes


def add_subnodes(n, subnodes, cop, direct_heat_source_utilisation_profile, head=40):
    """
    Add subnodes to the network and adjust loads and capacities accordingly.
    """

    # If head is boolean set it to 40 for default behavior
    if isinstance(head, bool):
        head = 40

    # Keep only n largest district heating networks according to head parameter
    subnodes_head = subnodes.sort_values(
        by="Wärmeeinspeisung in GWh/a", ascending=False
    ).head(head)
    subnodes_head.to_file(snakemake.output.district_heating_subnodes, driver="GeoJSON")

    subnodes_rest = subnodes[~subnodes.index.isin(subnodes_head.index)]

    # Add subnodes to network
    for idx, row in subnodes_head.iterrows():
        name = f'{row["cluster"]} {row["Stadt"]} urban central'

        # Add buses
        buses = (
            n.buses.filter(like=f"{row['cluster']} urban central", axis=0)
            .reset_index()
            .replace({f"{row['cluster']} urban central": name}, regex=True)
            .set_index("Bus")
        )
        n.add("Bus", buses.index, **buses)

        # Add heat loads
        uch_load_cluster = (
            n.snapshot_weightings.generators
            @ n.loads_t.p_set[f"{row['cluster']} urban central heat"]
        )
        lti_load_cluster = (
            n.loads.loc[f"{row['cluster']} low-temperature heat for industry", "p_set"]
            * 8760
        )

        dh_load_cluster = uch_load_cluster + lti_load_cluster
        scalar = min(
            1,
            (row["yearly_heat_demand_MWh"] / dh_load_cluster),
        )

        dh_load_cluster_subnodes = subnodes.loc[
            subnodes.cluster == row["cluster"], "yearly_heat_demand_MWh"
        ].sum()
        lost_load = dh_load_cluster_subnodes - dh_load_cluster
        if dh_load_cluster_subnodes > dh_load_cluster:
            logger.warning(
                f"Aggregated district heating load of systems within {row['cluster']} exceeds load of cluster. {lost_load} MWh/a are disregarded."
            )
            scalar *= row["yearly_heat_demand_MWh"] / dh_load_cluster_subnodes

        uch_load = scalar * n.loads_t.p_set[
            f"{row['cluster']} urban central heat"
        ].rename(f"{row['cluster']} {row['Stadt']} urban central heat")
        n.add(
            "Load",
            f"{name} heat",
            bus=f"{name} heat",
            p_set=uch_load,
            carrier="urban central heat",
            location=f"{row['cluster']} {row['Stadt']}",
        )

        lti_load = (
            scalar
            * n.loads.loc[
                f"{row['cluster']} low-temperature heat for industry", "p_set"
            ]
        )
        n.add(
            "Load",
            f"{row['cluster']} {row['Stadt']} low-temperature heat for industry",
            bus=f"{name} heat",
            p_set=lti_load,
            carrier="low-temperature heat for industry",
            location=f"{row['cluster']} {row['Stadt']}",
        )

        # Adjust loads of cluster buses
        n.loads_t.p_set.loc[:, f'{row["cluster"]} urban central heat'] *= 1 - scalar

        n.loads.loc[f'{row["cluster"]} low-temperature heat for industry', "p_set"] *= (
            1 - scalar
        )

        # # Replicate district heating stores of mother node for subnodes
        # stores = (
        #     n.stores.filter(like=f"{row['cluster']} urban central", axis=0)
        #     .reset_index()
        #     .replace(
        #         {
        #             f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
        #         },
        #         regex=True,
        #     )
        #     .set_index("Store")
        # )

        # stores["e_nom_max"] = row["ptes_pot_mwh"]
        # n.add("Store", stores.index, **stores)

        # Replicate district heating storage units of mother node for subnodes
        storage_units = (
            n.storage_units.filter(like=f"{row['cluster']} urban central", axis=0)
            .reset_index()
            .replace(
                {
                    f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
                },
                regex=True,
            )
            .set_index("StorageUnit")
        )

        storage_units.loc[storage_units.carrier.str.contains("pits$"), "p_nom_max"] = (
            row["ptes_pot_mwh"] / storage_units["max_hours"]
        )
        n.add("StorageUnit", storage_units.index, **storage_units)

        # restrict PTES capacity in mother nodes
        mother_nodes_ptes_pot = subnodes_rest.groupby("cluster").ptes_pot_mwh.sum()
        # add " urban central water tanks" to the mother node name
        mother_nodes_ptes_pot.index = (
            mother_nodes_ptes_pot.index + " urban central water pits"
        )
        n.storage_units.loc[mother_nodes_ptes_pot.index, "p_nom_max"] = (
            mother_nodes_ptes_pot
            / n.storage_units.loc[mother_nodes_ptes_pot.index, "max_hours"]
        )

        # Replicate district heating generators of mother node for subnodes
        generators = (
            n.generators.filter(like=f"{row['cluster']} urban central", axis=0)
            .reset_index()
            .replace(
                {
                    f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
                },
                regex=True,
            )
            .set_index("Generator")
        )
        n.add("Generator", generators.index, **generators)

        # Replicate district heating links of mother node for subnodes with separate treatment for links with dynamic efficiencies
        links = (
            n.links.loc[~n.links.carrier.str.contains("heat pump|direct", regex=True)]
            .filter(like=f"{row['cluster']} urban central", axis=0)
            .reset_index()
            .replace(
                {
                    f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
                },
                regex=True,
            )
            .set_index("Link")
        )
        n.add("Link", links.index, **links)

        # Add heat pumps and direct heat source utilization to subnode
        for heat_source in snakemake.params.heat_pump_sources:
            cop_heat_pump = (
                cop.sel(
                    heat_system="urban central",
                    heat_source=heat_source,
                    name=f"{row['cluster']} {row['Stadt']}",
                )
                .to_pandas()
                .to_frame(name=f"{name} {heat_source} heat pump")
                .reindex(index=n.snapshots)
                if snakemake.params.sector["time_dep_hp_cop"]
                else n.links.filter(like=heat_source, axis=0).efficiency.mode()
            )

            heat_pump = (
                n.links.filter(
                    regex=f"{row['cluster']} urban central.*{heat_source}.*heat pump",
                    axis=0,
                )
                .reset_index()
                .replace(
                    {
                        f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
                    },
                    regex=True,
                )
                .drop(["efficiency", "efficiency2"], axis=1)
                .set_index("Link")
            )
            if heat_pump["bus2"].str.match("$").any():
                n.add("Link", heat_pump.index, efficiency=cop_heat_pump, **heat_pump)
            else:
                n.add(
                    "Link",
                    heat_pump.index,
                    efficiency=-(cop_heat_pump - 1),
                    efficiency2=cop_heat_pump,
                    **heat_pump,
                )

            if heat_source in snakemake.params.direct_utilisation_heat_sources:
                # Add direct heat source utilization to subnode
                efficiency_direct_utilisation = (
                    direct_heat_source_utilisation_profile.sel(
                        heat_source=heat_source,
                        name=f"{row['cluster']} {row['Stadt']}",
                    )
                    .to_pandas()
                    .to_frame(name=f"{name} {heat_source} heat direct utilisation")
                    .reindex(index=n.snapshots)
                )

                direct_utilization = (
                    n.links.filter(
                        regex=f"{row['cluster']} urban central.*{heat_source}.*direct",
                        axis=0,
                    )
                    .reset_index()
                    .replace(
                        {
                            f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
                        },
                        regex=True,
                    )
                    .set_index("Link")
                    .drop("efficiency", axis=1)
                )

                n.add(
                    "Link",
                    direct_utilization.index,
                    efficiency=efficiency_direct_utilisation,
                    **direct_utilization,
                )

            if heat_source in snakemake.params.heat_utilisation_potentials:
                # get potential
                p_max_source = pd.read_csv(
                    snakemake.input[heat_source],
                    index_col=0,
                ).squeeze()[f"{row['cluster']} {row['Stadt']}"]
                # add potential to generator
                n.generators.loc[
                    f"{row['cluster']} {row['Stadt']} urban central {heat_source} heat",
                    "p_nom_max",
                ] = p_max_source

        ###
        # heat_pumps = (
        #     n.links.filter(regex=f"{row['cluster']} urban central.*heat pump", axis=0)
        #     .reset_index()
        #     .replace(
        #         {
        #             f"{row['cluster']} urban central": f"{row['cluster']} {row['Stadt']} urban central"
        #         },
        #         regex=True,
        #     )
        #     .set_index("Link")
        # ).drop("efficiency", axis=1)
        # heat_pumps_t = n.links_t.efficiency.filter(
        #     regex=f"{row['cluster']} urban central.*heat pump"
        # )
        # heat_pumps_t.columns = heat_pumps_t.columns.str.replace(
        #     f"{row['cluster']} urban central",
        #     f"{row['cluster']} {row['Stadt']} urban central",
        # )
        # n.add("Link", heat_pumps.index, efficiency=heat_pumps_t, **heat_pumps)

        # Add heat vent to subnode
        # n.add(
        #     "Generator",
        #     f"{name} heat vent",
        #     bus=name,
        #     location=f"{row['cluster']} {row['Stadt']}",
        #     carrier="urban central heat vent",
        #     p_nom_extendable=True,
        #     p_min_pu=-1,
        #     p_max_pu=0,
        #     unit="MWh_th",
        # )

    return


def extend_cops(cops, subnodes):
    """
    Extend COPs by subnodes mirroring the timeseries of the corresponding
    mother node.
    """
    cops_extended = cops.copy()

    # Iterate over the DataFrame rows
    for _, row in subnodes.iterrows():
        cluster_name = row["cluster"]
        city_name = row["Stadt"]

        # Select the xarray entry where name matches the cluster
        selected_entry = cops.sel(name=cluster_name)

        # Rename the selected entry
        renamed_entry = selected_entry.assign_coords(name=f"{cluster_name} {city_name}")

        # Combine the renamed entry with the extended dataset
        cops_extended = xr.concat([cops_extended, renamed_entry], dim="name")

    # Change dtype of the name dimension to string
    cops_extended.coords["name"] = cops_extended.coords["name"].astype(str)

    return cops_extended


def extend_heating_distribution(existing_heating_distribution, subnodes):
    """
    Extend heating distribution by subnodes mirroring the distribution of the
    corresponding mother node.
    """
    # Merge the existing heating distribution with subnodes on the cluster name
    mother_nodes = (
        existing_heating_distribution.loc[subnodes.cluster.unique()]
        .unstack(-1)
        .to_frame()
    )
    cities_within_cluster = subnodes.groupby("cluster")["Stadt"].apply(list)
    mother_nodes["cities"] = mother_nodes.apply(
        lambda i: cities_within_cluster[i.name[2]], axis=1
    )
    # Explode the list of cities
    mother_nodes = mother_nodes.explode("cities")

    # Reset index to temporarily flatten it
    mother_nodes_reset = mother_nodes.reset_index()

    # Append city name to the third level of the index
    mother_nodes_reset["name"] = (
        mother_nodes_reset["name"] + " " + mother_nodes_reset["cities"]
    )

    # Set the index back
    mother_nodes = mother_nodes_reset.set_index(["heat name", "technology", "name"])

    # Drop the temporary 'cities' column
    mother_nodes.drop("cities", axis=1, inplace=True)

    # Reformat to match the existing heating distribution
    mother_nodes = mother_nodes.squeeze().unstack(-1).T

    # Combine the exploded data with the existing heating distribution
    existing_heating_distribution_extended = pd.concat(
        [existing_heating_distribution, mother_nodes]
    )
    return existing_heating_distribution_extended


if __name__ == "__main__":
    if "snakemake" not in globals():
        import os
        import sys

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        path = "../submodules/pypsa-eur/scripts"
        sys.path.insert(0, os.path.abspath(path))
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_district_heating_subnodes",
            simpl="",
            clusters=27,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2045",
            run="No_PTES",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    logger.info("Adding SysGF-specific functionality")

    n = pypsa.Network(snakemake.input.network)

    subnodes = gpd.read_file(snakemake.input.subnodes)

    # Add PTES limit to subnodes according to land availability within city regions
    corine = rasterio.open(snakemake.input.corine)
    natura = rasterio.open(snakemake.input.natura)
    groundwater = xr.open_dataset(snakemake.input.groundwater_depth).sel(
        lon=slice(subnodes["geometry"].x.min(), subnodes["geometry"].x.max()),
        lat=slice(subnodes["geometry"].y.min(), subnodes["geometry"].y.max()),
    )
    subnodes = add_ptes_limit(
        subnodes,
        corine,
        natura,
        groundwater,
        snakemake.params.district_heating["ptes_codes_corine"],
        snakemake.params.district_heating["max_groundwater_depth"],
        snakemake.params.district_heating["ptes_potential_scalar"],
    )

    add_subnodes(
        n,
        subnodes,
        cop=xr.open_dataarray(snakemake.input.cop_profiles),
        direct_heat_source_utilisation_profile=xr.open_dataarray(
            snakemake.input.direct_heat_source_utilisation_profiles
        ),
        head=snakemake.params.district_heating["add_subnodes"],
    )

    # if snakemake.config["foresight"] == "myopic":
    #     cops = xr.open_dataarray(snakemake.input.cop_profiles)
    #     cops_extended = extend_cops(cops, subnodes)
    #     cops_extended.to_netcdf(snakemake.output.cop_profiles_extended)

    if snakemake.wildcards.planning_horizons == str(snakemake.params["baseyear"]):
        existing_heating_distribution = pd.read_csv(
            snakemake.input.existing_heating_distribution,
            header=[0, 1],
            index_col=0,
        )
        existing_heating_distribution_extended = extend_heating_distribution(
            existing_heating_distribution, subnodes
        )
        existing_heating_distribution_extended.to_csv(
            snakemake.output.existing_heating_distribution_extended
        )
    else:
        # write empty file to output
        with open(snakemake.output.existing_heating_distribution_extended, "w") as f:
            pass

    maybe_adjust_costs_and_potentials(
        n, snakemake.params["adjustments"], snakemake.wildcards.planning_horizons
    )
    n.export_to_netcdf(snakemake.output.network)
