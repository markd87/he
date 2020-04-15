from typing import List, Union, Optional
import requests
import logging
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import plotly.express as px

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.debug("test")


class MIDAS:
    def __init__(self, version=1.0):
        self.version = version
        self.base_url = f"http://webtris.highwaysengland.co.uk/api/v{self.version}"

    def sites(self) -> gpd.GeoDataFrame:
        """
        Get MIDAS sites
        """
        url = f"{self.base_url}/sites"
        res = requests.get(url).json()
        sites = res["sites"]
        logger.info(f"{len(sites)} sites received")
        df = pd.DataFrame(sites)
        # Additional filtering on the sites
        # keep only active and with a name
        df = (
            df.query("Status != 'Inactive'").query("Name != ''").copy()
        ).copy()  # keep only active
        logger.info(f"Filtered non-active sites, {len(df)} sites remaining")
        # get type MIDAS/.../
        df["type"] = df["Name"].apply(lambda x: x.split()[0].strip())
        # keep only MIDAS
        df = df.query("type=='MIDAS'").copy()
        logger.info(f"Filtered MIDAS only sites, {len(df)} sites remaining")
        # get link
        df["link_id"] = df["Name"].apply(lambda x: x.split(";")[0].split()[-1].strip())
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

    def daily_report(self, start: str, end: str, sites: list) -> pd.DataFrame:
        """
        Get daily report between dates for sites
        """
        # parse dates
        try:
            start = pd.to_datetime(start, dayfirst=True).strftime("%d%m%Y")
            end = pd.to_datetime(end, dayfirst=True).strftime("%d%m%Y")
        except Exception:
            logger.error("Invalid dates")
        sites_str = ",".join([str(s) for s in sites])
        logger.info(
            f"Getting report for date range {start} - {end} for sites {sites_str}"
        )
        report_df = pd.DataFrame()
        page = 1
        url = f"{self.base_url}/reports/Daily?sites={sites_str}&start_date={start}&end_date={end}&page={page}&page_size=100"
        res = requests.get(url)
        # get pages while exist
        while res.status_code == 200:
            res_json = res.json()
            df = pd.DataFrame(res_json["Rows"])
            report_df = pd.concat([report_df, df])
            page += 1
            url = f"{self.base_url}/reports/Daily?sites={sites_str}&start_date={start}&end_date={end}&page={page}&page_size=100"
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

    @staticmethod
    def sites_in_link(sites: gpd.GeoDataFrame, link_id: str) -> gpd.GeoDataFrame:
        return sites[sites["link_id"] == link_id]

    @staticmethod
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

    @staticmethod
    def plot_sites(
        df_sites: Union[pd.DataFrame, gpd.GeoDataFrame], backend="geopandas"
    ) -> None:
        """
        Show midas sites on a map
        can use either static or mapbox map
        for mapbox map requires Latitude and Longitude columns
        """
        if backend == "geopandas":
            if "geometry" in df_sites.columns:
                _, ax = plt.subplots(figsize=(10, 10))
                df_sites.to_crs("epsg:3857").plot(ax=ax)
                ax.grid(False)
                ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite)
            else:
                logger.info("no geometry column")
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


class NTIS:
    ns = {"ns": "http://datex2.eu/schema/2/2_0"}  # Namespace

    def __init__(self, network_file="ntis_model.xml"):
        self.root = ET.parse(network_file).getroot()

    @staticmethod
    def to_float(text: Optional[str]) -> float:
        try:
            return float(text)
        except:
            return np.nan

    @classmethod
    def get_text(cls, link: Element, tag: str) -> Union[list, Optional[str]]:
        elements = link.findall(f".//ns:{tag}", namespaces=cls.ns)
        if len(elements) > 1:
            return [el.text for el in elements]
        elif elements:
            return elements[0].text
        return None

    @classmethod
    def get_value(cls, link: Element, tag: str) -> Union[list, Optional[float]]:
        elements = link.findall(f".//ns:{tag}", namespaces=cls.ns)
        if len(elements) > 1:
            return [cls.to_float(el.text) for el in elements]
        elif elements:
            return cls.to_float(elements[0].text)
        return None

    @classmethod
    def get_location_description(cls, link: Element) -> dict:
        try:
            location = link.findall(
                f".//ns:predefinedLocationName/ns:values/ns:value", namespaces=cls.ns
            )[0].text
        except:
            location = None
        return {"location": location}

    @classmethod
    def get_area_cols(cls, link: Element) -> dict:
        """
        Get area information
        """
        area_descriptors = link.findall(f".//ns:areaDescriptor", namespaces=cls.ns)

        county = cls.get_text(area_descriptors[0], "value")
        area_team = cls.get_text(area_descriptors[1], "value")
        regional_control_centre = cls.get_text(area_descriptors[0], "value")

        return {
            "county": county,
            "area_team": area_team,
            "regional_control_centre": regional_control_centre,
        }

    @classmethod
    def start_end_nodes(cls, link: Element) -> dict:
        """
        Get start and end nodes IDs for link
        """
        start_node = cls.get_text(
            link, f"fromPoint/ns:fromReferent/ns:referentIdentifier"
        )
        end_node = cls.get_text(link, f"toPoint/ns:fromReferent/ns:referentIdentifier")
        return {"start_node": start_node, "end_node": end_node}

    def get_links(self) -> pd.DataFrame:
        """
        Get NTIS link metadata
        """

        # all link elements
        links = list(
            self.root.findall(
                f".//ns:predefinedLocationContainer[@id='NTIS_Network_Links']",
                namespaces=self.ns,
            )
        )[0]

        # mapping columns to individual tags
        text_cols = {
            "carriageway": "carriageway",
            "direction": "directionBoundOnLinearSection",
            "roadnumber": "roadNumber",
        }

        numeric_cols = {
            "length": "lengthAffected",
            "mid_point_capacity": "midPointStaticCapacity",
            "exit_point_capacity": "exitPointStaticCapacity",
        }

        links_vals = []

        for i in tqdm(range(len(links))):
            link = links[i]
            link_dict = {}

            # link_id
            link_dict.update({"id": link.get("id")})

            # location description
            link_dict.update(self.get_location_description(link))

            # area information
            link_dict.update(self.get_area_cols(link))

            # text columns tags
            for col in text_cols:
                link_dict.update({col: self.get_text(link, text_cols[col])})

            # numeric columns tags
            for col in numeric_cols:
                link_dict.update({col: self.get_value(link, numeric_cols[col])})

            # start and end node
            link_dict.update(self.start_end_nodes(link))

            # add to list of links
            links_vals.append(link_dict)

        print(f"Number of links found: {len(links_vals)}")

        return pd.DataFrame(links_vals)

    def get_links_shapes(self) -> gpd.GeoDataFrame:
        """
        Get NTIS links shapes
        - link_id
        - linestring geometry
        """

        ntis_links_shapes = []
        for elem in self.root.findall(
            f".//ns:predefinedLocationContainer", namespaces=self.ns
        ):
            if "NTIS_Link_Shape" in elem.get("id"):
                ntis_links_shapes.append(elem)

        link_vals = []
        for i in tqdm(range(len(ntis_links_shapes))):
            link = ntis_links_shapes[i]
            lats: List[float] = []
            longs: List[float] = []
            link_id = link.get("id").split("_")[-1]
            lats = self.get_value(link, "latitude")
            longs = self.get_value(link, "longitude")
            link_vals.append(
                {"link_id": link_id, "geometry": LineString(list(zip(longs, lats)))}
            )
        print(f"Number of links found: {len(link_vals)}")
        return gpd.GeoDataFrame(link_vals, crs="epsg:4326")

    def get_nodes(self) -> gpd.GeoDataFrame:
        """
        Get NTIS network nodes with point geometry
        for each node:
        - node_id
        - longitude
        - latitude
        - geometry
        """
        nodes = self.root.findall(
            f".//ns:predefinedLocationContainer[@id='NTIS_Network_Nodes']/ns:predefinedLocation",
            namespaces=self.ns,
        )

        nodes_vals = []
        for i in tqdm(range(len(nodes))):
            node = nodes[i]
            node_id = node.get("id")
            lat = self.get_value(node, "latitude")
            long = self.get_value(node, "longitude")
            nodes_vals.append(
                {
                    "node_id": node_id,
                    "Longitude": long,
                    "Latitude": lat,
                    "geometry": Point(long, lat),
                }
            )

        print(f"Number of nodes found: {len(nodes_vals)}")
        return gpd.GeoDataFrame(nodes_vals, crs="epsg:4326")

    def get_HATRIS_sections(self) -> pd.DataFrame:
        """
        Get hatris sections
        - section_id
        - NTIS links ids
        """
        hatris_sections = []
        for elem in self.root.findall(
            f".//ns:predefinedLocationContainer", namespaces=self.ns
        ):
            if "NTIS_HATRIS_Section" in elem.get("id"):
                hatris_sections.append(elem)

        hatris_vals = []
        for i in tqdm(range(len(hatris_sections))):
            sect = hatris_sections[i]
            section_id = sect.get("id").split("_")[-1]
            ntis_links = [
                link.get("id")
                for link in sect.findall(
                    f".//ns:predefinedLocationReference", namespaces=self.ns
                )
            ]
            hatris_vals.append({"section_id": section_id, "ntis_links": ntis_links})
        print(f"Number of sections found: {len(hatris_vals)}")
        return pd.DataFrame(hatris_vals)
