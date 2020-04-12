import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import logging

logger = logging.getLogger()


class MIDAS:
    def __init__(self, version=1.0):
        self.version = version
        self.base_url = f"http://webtris.highwaysengland.co.uk/api/v{version}"
        self.gpd_sites = gpd.GeoDataFrame()

    def sites(self) -> gpd.GeoDataFrame:
        """
        Get MIDAS sites
        """
        url = f"{self.base_url}/sites"
        res = requests.get(url).json()
        sites = res["sites"]
        df = pd.DataFrame(sites)
        # Additional filtering on the sites
        # keep only active and with a name
        df = (
            df.query("Status != 'Inactive'").query("Name != ''").copy()
        ).copy()  # keep only active
        # get type MIDAS/.../
        df["type"] = df["Name"].apply(lambda x: x.split()[0].strip())
        # keep only MIDAS
        df = df.query("type=='MIDAS'").copy()
        # get link
        df["link"] = df["Name"].apply(lambda x: x.split(";")[0].split()[-1].strip())
        # get direction
        df["direction"] = df["Name"].apply(lambda x: x.split(";")[-1])
        # create point Geometry
        df["geometry"] = df.apply(
            lambda x: Point(x["Longitude"], x["Latitude"]), axis=1
        )
        gdf_sites = gpd.GeoDataFrame(df)
        # set CRS
        gdf_sites.crs = "epsg:4326"
        # convert to UK CRS
        gdf_sites = gdf_sites.to_crs("epsg:27700")

        return gdf_sites

    def areas(self) -> pd.DataFrame:
        """
        Get MIDAS areas
        """
        url = f"{self.base_url}/areas"
        res = requests.get(url).json()
        areas = res["areas"]
        df = pd.DataFrame(areas)
        return df

    def daily_report(self, start: str, end: str, sites: str) -> pd.DataFrame:
        """
        Get daily report between dates for sites
        """
        # parse dates
        try:
            start = pd.to_datetime(start, dayfirst=True).strftime("%d%m%Y")
            end = pd.to_datetime(end, dayfirst=True).strftime("%d%m%Y")
        except Exception:
            logger.error("Invalid dates")
        report_df = pd.DataFrame()
        page = 1
        url = f"{self.base_url}/reports/Daily?sites={sites}&start_date={start}&end_date={end}&page={page}&page_size=100"
        res = requests.get(url)
        # get pages while exist
        while res.status_code == 200:
            res_json = res.json()
            df = pd.DataFrame(res_json["Rows"])
            report_df = pd.concat([report_df, df])
            page += 1
            url = f"{self.base_url}/reports/Daily?sites={sites}&start_date={start}&end_date={end}&page={page}&page_size=100"
            res = requests.get(url)

        report_df = report_df.drop_duplicates(
            subset=["Site Name", "Report Date", "Time Period Ending"], keep="first",
        )

        if len(report_df) == 0:
            logger.warning("Empty Dataframe")
            return None

        # convert to datetime
        report_df["Report Date"] = pd.to_datetime(report_df["Report Date"])

        # create timestamp
        report_df["time"] = pd.to_datetime(
            report_df["Report Date"].dt.date.astype(str)
            + " "
            + report_df["Time Period Ending"]
        )

        report_df = report_df.replace("", 0)
        report_df["Total Volume"] = pd.to_numeric(report_df["Total Volume"])
        report_df["Avg mph"] = pd.to_numeric(report_df["Avg mph"])

        return report_df.reset_index(drop=True)
