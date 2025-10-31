from io import BytesIO
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd

from vs_calc.constants import HammerType, SoilType # Assuming these are correctly imported


class SPT:
    """
    Contains the data from an SPT file and methods for N60 correction.
    """

    def __init__(
        self,
        name: str,
        depth: np.ndarray,
        n: np.ndarray,
        hammer_type: HammerType = HammerType.Auto,
        borehole_diameter: float = None,
        energy_ratio: float = None,
        soil_type: np.ndarray = None,
        gwl: float = None,
        lastdepth: float = None,
        rod_length: float = None,
    ):
        self.name = name
        self.depth = depth
        self.lastdepth = lastdepth
        self.N = n
        self.hammer_type = hammer_type
        self.borehole_diameter = borehole_diameter
        self.gwl = gwl
        self.energy_ratio = energy_ratio
        self.soil_type = (
            np.repeat(SoilType.Clay, len(depth)) if soil_type is None else soil_type
        )
        self.rod_length = rod_length
        self.info = {
            "z_min": depth[0],
            "z_max": depth[-1],
            "z_spread": depth[-1] - depth[0],
            "removed_rows": [],
            "lastdepth": lastdepth,
        }

        # spt parameter info init for lazy loading
        self._n60 = None
        
    def __str__(self):
        """
        Returns a human-readable string representation of the SPT object.
        """
        num_points = len(self.depth)
        
        depth_summary = f"{self.depth[:5].tolist()}... ( {num_points})"
        n_summary = f"{self.N[:5].tolist()}... ( {num_points})"
        
        try:
            soil_info = f"[{self.soil_type[0].name}]... ({num_points})"
        except (IndexError, AttributeError):
            soil_info = "None"
            
        return (
            f"---  SPT Profile: {self.name} ---\n"
            f"  Name: {self.name}, GWL: {self.gwl}, Last Depth : {self.lastdepth}\n"
            f"  Hammer: {self.hammer_type.name}, Diameter: {self.borehole_diameter}mm\n"
            f"  Soil: {soil_info}\n"
            f"  Depth (m): {depth_summary}\n"
            f"  N-Value:   {n_summary}\n" 
            f"-----------------------------------"
        )

    @property
    def N60(self):
        if self._n60 is None:
            N60_list = []
            for idx, N in enumerate(self.N):
                current_rod_length = self.rod_length if self.rod_length is not None else self.depth[idx]
                
                Ce, Cb, Cr = self.calc_n60_variables(
                    self.energy_ratio,
                    self.hammer_type,
                    self.borehole_diameter,
                    current_rod_length, # ⬅️ Passed the determined rod length
                )
                N60 = round(N * Ce * Cb * Cr, 2)
                N60_list.append(N60)
            self._n60 = np.asarray(N60_list)
        return self._n60

    def to_json(self):
        """
        Creates a json response dictionary from the SPT
        """
        return {
            "name": self.name,
            "depth": self.depth.tolist(),
            "N": self.N.tolist(),
            "hammer_type": self.hammer_type.name,
            "borehole_diameter": self.borehole_diameter,
            "energy_ratio": self.energy_ratio,
            "soil_type": [soil_type.name for soil_type in self.soil_type],
            "info": self.info,
            "N60": self.N60.tolist(),
        }

    @staticmethod
    def from_json(json: Dict):
        """
        Creates a SPT from a json dictionary string
        """
        spt = SPT(
            json["name"],
            np.asarray(json["depth"]),
            np.asarray(json["N"]),
            HammerType[json["hammer_type"]],
            float(json["borehole_diameter"]),
            None if json["energy_ratio"] is None else float(json["energy_ratio"]),
            [SoilType[soil_type] for soil_type in json["soil_type"]],
        )
        spt._n60 = None if json["N60"] is None else np.asarray(json["N60"])
        return spt

    @staticmethod
    def from_file(spt_ffp: str):
        """
        Creates an SPT from an SPT file
        """
        spt_ffp = Path(spt_ffp)
        data = pd.read_csv(spt_ffp)
        soil_type = (
            None
            if "Soil" not in data.columns
            else data["Soil"].map(lambda x: SoilType[x])
        )
        return SPT(
            spt_ffp.stem,
            data.iloc[:, 0].values,
            data.iloc[:, 1].values,
            soil_type=soil_type,
        )

    @staticmethod
    def from_file2(spt_ffp: str, spt_info_path: str = "SPT_info.txt"):
        """
        Creates an SPT object from an SPT data file.
        
        It attempts to apply HammerType, BoreholeDiameter, EnergyRatio, and 
        if a matching CODE is found in the SPT data filename.

        :param spt_ffp: File path of the SPT data CSV.
        :param spt_info_path: File path of the supplementary information CSV.
        :return: An SPT object with data and associated parameters.
        """
        spt_ffp = Path(spt_ffp)
        spt_info_path = Path(spt_info_path)
        
        # --- 1. Extract CODE from filename (e.g., 'PREFIX_CODE_SUFFIX.csv')
        filename = spt_ffp.stem 
        parts = filename.split("_")
    
        # Check if the format is BH_CODE
        if len(parts) == 2 and parts[0].upper() == "BH":
        # CODE is assumed to be the second part, stripped of whitespace
            code = parts[1].strip()
        else:
            # Fallback for unexpected formats (or if the original logic is sometimes needed)
            # Original logic: CODE is the second part if there are at least 3 parts (PREFIX_CODE_SUFFIX)
            if len(parts) >= 3:
                code = parts[1].strip()
            else:
                code = None
        
        # --- 2. Set default parameters (based on SPT.__init__ defaults)
        hammer_type = HammerType.Auto
        borehole_diameter = 100.0 
        energy_ratio = None
        gwl = 1
        lastdepth = None
        # --- 3. Search and override parameters using SPT_info.csv and CODE
        if code:
            try:
                info_df = pd.read_csv(spt_info_path)
                # Ensure CODE column is string type for comparison
                info_df["CODE"] = info_df["CODE"].astype(str).str.strip()
        
                # Find the row matching the extracted CODE (case-insensitive)
                row = info_df[info_df["CODE"].str.upper() == code.upper()]
                
                if not row.empty:
                    # Apply parameters if found in the info file
                    ld_val = row.iloc[0].get("Lastdepth", None)
                    if pd.notna(ld_val):
                        lastdepth = float(ld_val)    
                    #GWL                    
                    gwl_val = row.iloc[0].get("gwl", None)
                    if pd.notna(gwl_val):
                        gwl = float(gwl_val)
                        
                    # HammerType
                    ht_val = row.iloc[0].get("HammerType", None)
                    if pd.notna(ht_val):
                        try:
                            # Convert string to HammerType Enum
                            hammer_type = HammerType[str(ht_val).strip()]
                        except KeyError:
                            print(f"Warning: Invalid HammerType '{ht_val}' for CODE={code}. Using default.")
                            
                    # BoreholeDiameter
                    bd_val = row.iloc[0].get("BoreholeDiameter", None)
                    if pd.notna(bd_val):
                        borehole_diameter = float(bd_val)
        
                    # EnergyRatio
                    er_val = row.iloc[0].get("EnergyRatio", None)
                    if pd.notna(er_val):
                        energy_ratio = float(er_val)
                   
        
                    print(f"Applying info for CODE={code} (from {spt_info_path}): Last Depth = {lastdepth}, GWL ={gwl}, HT={hammer_type.name}, Dia={borehole_diameter}, Er={energy_ratio}")
                else:
                    print(f"CODE '{code}' not found in '{spt_info_path}', using defaults.")
                    
            except FileNotFoundError:
                print(f"'{spt_info_path}' not found, using defaults.")
            except Exception as e:
                print(f"Error processing info file: {e}. Using defaults.")
        
        # --- 4. Read Depth and N-values from the SPT data file
        data = pd.read_csv(spt_ffp)
        try:
            # Retrieve data using column headers 'depth' and 'n'
            depth_values = data["depth"].values
            n_values = data["n"].values
            
        except KeyError as e:
            # Raise an error if required columns are missing
            raise ValueError(f"CSV file must contain 'depth' and 'n' columns.") from e
        
        # --- 5. Create and return the SPT object with all gathered parameters
        return SPT(
            spt_ffp.stem,
            depth_values,
            n_values,
            hammer_type=hammer_type,         
            borehole_diameter=borehole_diameter, 
            energy_ratio=energy_ratio,       
            gwl=gwl,
            lastdepth =lastdepth            
        )
        
    @staticmethod
    def from_byte_stream_form(file_name: str, stream: bytes, form: Dict):
        """
        Creates an SPT from a file stream and form data
        """
        file_name = Path(file_name)
        file_data = (
            pd.read_csv(BytesIO(stream))
            if file_name.suffix == ".csv"
            else pd.read_excel(BytesIO(stream))
        )
        # Manage soil type from file first then form
        soil_type = (
            file_data["Soil"]
            if "Soil" in file_data.columns
            else (
                None
                if form["soilType"] == ""
                else np.repeat(form["soilType"], len(file_data.values))
            )
        )
        return SPT(
            form.get("sptName", file_name.stem),
            np.asarray(file_data["Depth"]),
            np.asarray(file_data["NValue"]),
            HammerType.Auto
            if form["hammerType"] == ""
            else HammerType[form["hammerType"]],
            form["boreholeDiameter"],
            None if form["energyRatio"] == "" else form["energyRatio"],
            None
            if soil_type is None
            else np.asarray([SoilType[soil] for soil in soil_type]),
        )

    @staticmethod
    def calc_n60_variables(
        energy_ratio: float,
        hammer_type: HammerType,
        borehole_diameter: float,
        rod_length: float,
    ):
        """
        Calculates the variables needed to get N60 from N
        Returns the variables Ce, Cr, Cb
        """
        # Calc Ce
        # In case the data is messed up and none of the following condition can be meet
        # assume a relative average Ce value of 0.8
        if energy_ratio is not None:
        #CHECK
            Ce = energy_ratio / 60
        else:
            if hammer_type == HammerType.Auto:
                Ce = 1.3
            elif hammer_type == HammerType.Safety:
                Ce = 1
            elif hammer_type == HammerType.Doughnut:
                Ce = 0.75

        # Calc Cr
        if rod_length< 3:
            Cr = 0.75
        elif 4 <= rod_length < 6:
            Cr = 0.85
        elif 6 <= rod_length < 10:
            Cr = 0.95
        else:
            Cr = 1

        # Calc Cb
        if 60 <= borehole_diameter <= 120:
            Cb = 1
        elif borehole_diameter == 200:
            Cb = 1.15
        else:
            Cb = 1.05

        return Ce, Cb, Cr