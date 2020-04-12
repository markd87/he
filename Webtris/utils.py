import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import contextily as ctx
import plotly.express as px


# Utility functions for working with the midas sites


def plot_report(df_report: pd.DataFrame) -> None:
    """
    Plot time series of the report dataframe obtained from Webtris API
    df_report is assumed to have the same columns as returned by 
    the Webtris.sites() function
    """
    _, ax = plt.subplots(figsize=(15, 4))
    sns.lineplot(data=df_report, x="time", y="Total Volume", hue="Site Name")
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Total Volume", fontsize=16)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    if len(df_report) == 1:
        ax.set_title(df_report["Site Name"].values[0], fontsize=18)


def show_sites(df_sites: pd.DataFrame, backend="geopandas") -> None:
    """
    Show midas sites on a map
    """
    if backend == "geopandas":
        if "geometry" in df_sites.columns:
            _, ax = plt.subplots(figsize=(10, 10))
            df_sites.to_crs("epsg:3857").plot(ax=ax, edgecolor="w", alpha=1)
            ax.grid(False)
            ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite)
        else:
            print("no geometry column")
    elif backend == "mapbox":
        mapbox_access_token = open(".mapbox").read()
        px.set_mapbox_access_token(mapbox_access_token)
        fig = px.scatter_mapbox(
            df_sites.dropna(),
            lat="Latitude",
            lon="Longitude",
            size_max=15,
            zoom=10,
            width=1000,
            height=550,
            hover_data=["Name"],
        )
        fig.show()
