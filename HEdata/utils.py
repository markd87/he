import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import contextily as ctx
import plotly.express as px
import matplotlib

# Utility functions for working with the midas sites


def plot_map(gdf, ax=None, backend="geopandas", **kwargs):
    """
    plot a geopandas data frame
    returns the axis to allow overlaying additional 
    layers
    """
    if backend == "geopandas":
        if not ax:
            _, ax = plt.subplots(figsize=(10, 10))
        gdf.to_crs("epsg:3857").plot(ax=ax, **kwargs)
        ax.grid(False)
        ax.set_axis_off()
        ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite, reset_extent=False)
        return ax

    elif backend == "mapbox":
        mapbox_access_token = open(".mapbox").read()
        px.set_mapbox_access_token(mapbox_access_token)
        fig = px.scatter_mapbox(
            gdf.dropna(),
            lat="Latitude",
            lon="Longitude",
            size_max=15,
            zoom=10,
            width=1000,
            height=550,
        )
        fig.show()
