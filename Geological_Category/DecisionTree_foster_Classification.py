import pandas as pd


def classify_geounits(df):
    def grepl(pattern, text):
        if pd.isna(text):
            return False
        return pattern.lower() in str(text).lower()

    def apply_rules(row):
        category = ""

        mr = str(row.get("mr", "")).lower()
        sn = str(row.get("sn", "")).lower()
        de = str(row.get("de", "")).lower()
        mu = str(row.get("mu", "")).lower()
        uc = str(row.get("uc", "")).lower()

        checks = [
            ("peat", grepl("peat", mr)),
            ("fill", grepl("human-made", sn)),
            ("fluvial", grepl("fluvial", de)),
            ("estuarine", grepl("estu", sn) and ("sand" in mr or "mud" in mr)),
            ("alluvial", grepl("river", sn)),
            ("floodplain", grepl("flood", de) or grepl("flood", mu)),
            ("lacustrine", grepl("lake", sn)),
            ("beach", grepl("beach", mu)),
            ("dune", grepl("dune", mu)),
            ("fan", grepl("fan", mu)),
            ("loess", grepl("windblown", sn)),
            ("outwash", grepl("outwash", mu)),
            ("moraine", grepl("moraine", mu) or grepl("moraine", de)),
            ("till", grepl("till", mr)),
            ("terrace", grepl("terrace", mu)),
            ("volcanic", grepl("volcanic", mr)),
            ("igneous", grepl("igneous", sn)),
            ("metamorphic", grepl("metamorphic", sn)),
            ("undifSed", grepl("sediment", sn) and category == ""),
            ("water", grepl("water", uc)),
        ]

        for cat, cond in checks:
            if cond:
                category += cat + "_"

        # Fallback if no category was assigned
        if category == "":
            if any(x in mr for x in ["basalt", "gabbro", "rhyolite", "andesite"]):
                category += "igneous_"
            elif any(x in mr for x in ["mudstone", "sandstone", "siltstone", "shale"]):
                category += "undifSed_"
            elif "gravel" in mr and "till" in de:
                category += "till_"
            elif sn == "ice":
                category += "ICE_"

        # Final category mapping
        final_map = {
            "peat_": "01_peat",
            "fill_": "02_fill",
            "fluvial_": "03_fluvial",
            "estuarine_": "04_estuarine",
            "alluvial_": "05_alluvial",
            "floodplain_": "06_floodplain",
            "lacustrine_": "07_lacustrine",
            "beach_": "08_beach",
            "dune_": "09_dune",
            "fan_": "10_fan",
            "loess_": "11_loess",
            "outwash_": "12_outwash",
            "moraine_": "13_moraine",
            "till_": "14_till",
            "terrace_": "15_terrace",
            "volcanic_": "16_volcanic",
            "igneous_": "17_igneous",
            "metamorphic_": "18_metamorphic",
            "undifSed_": "19_undifSed",
            "water_": "20_water",
            "ICE_": "21_ICE",
        }

        for key in final_map:
            if category.startswith(key):
                return final_map[key]

        return "99_unknown"

    df["finalCategory"] = df.apply(apply_rules, axis=1)
    return df


# Example usage
df = pd.read_csv("GeologicalUnit_Addgeometry_4th3.csv")
classified_df = classify_geounits(df)

# Save the output
classified_df.to_csv("GeologicalUnit_with_Category_foster.csv", index=False)
print("Saved: GeologicalUnit_with_Category_foster.csv")
