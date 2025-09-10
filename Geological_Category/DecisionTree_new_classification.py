import pandas as pd


def classify_by_flowchart(row):
    def contains(text, *keywords):
        if pd.isna(text):
            return False
        return any(k.lower() in str(text).lower() for k in keywords)

    # Extract fields
    mainrock = row.get("MAINROCK", "")
    mapname = row.get("MAPNAME", "")
    simplename = row.get("SIMPLENAME", "")
    descr = row.get("DESCR", "")
    mapsymbol = row.get("MAPSYMBOL", "")
    keygrpname = row.get("KEYGRPNAME", "")

    # Decision logic following the flowchart
    if contains(mainrock, "peat") or contains(mapname, "swamp"):
        return "Group 1"  # Peat
    elif contains(mapname, "fill") or contains(simplename, "human"):
        return "Group 4"  # Artificial fill
    elif contains(simplename, "undifferentiated", "sedimentary rock") or contains(mapname, "landslide","undifferentiated"):
        return "Group 15"  # Undifferentiated sedimentary rock
    elif contains(mapname, "river", "valley", "fluvial", "estuary", "estuarine"):
        if contains(mapname, "lake") or contains(descr, "lacustrine"):
            return "Group 8"  # Lacustrine
        elif contains(mapname, "fan", "scree", 'avalanche') or contains(mapsymbol, "fan"):
            return "Group 10"
        elif contains(mainrock, "gravel","sandstone") or contains(mapname, "gravel","alluvium", "alluvial"):
            return "Group 6"  # Alluvium
        elif contains(mapname, "volc") or contains(mapsymbol, "bas") or contains(mainrock, "ignimbrite"):
            return "Group 17"  # Volcanic
        elif contains(mapname, "estuary", "estuarine", "fluvial", "fluvium") or contains(mainrock, "sand", "mud", "clay","pumice"):
            return "Group 5"  # Fluvial
        elif contains(simplename, "igneous", "metamorphic") or contains(keygrpname, "gnt", "qtz", "dio") or contains(mainrock, "amphibolite"):
            return "Group 18"  # Igneous/Metamorphic
    elif contains(mapname, "lake") or contains(descr, "lacustrine"):
        return "Group 8"  # Lacustrine
    elif contains(mapname, "marine", "ocean", "beach", "dune","bar") or contains(mapsymbol, "fan"):
        return "Group 09"  # Beach, bar, dune deposits
    elif contains(mapname, "loess"):
        return "Group 11"  # Loess
    elif contains(descr, "terrace"):
        return "Group 16"  # Terrace deposits
    elif contains(mapsymbol, "till") or contains(descr, "moraine","till") or contains(mainrock, "till") or contains(mapname,"till"):
        return "Group 14"  # Glacial till
    elif contains(mapname, "outwash","glacier"):
        return "Group 12"  # Glacigenic
    elif contains(descr, "floodplain"):
        return "Group 13"  # Flood deposits
    elif contains(mapname, "volc","lahar") or contains(mapsymbol, "bas") or contains(mapname, "basalt"):
        return "Group 17"  # Volcanic
    elif contains(simplename, "igneous", "metamorphic") or contains(keygrpname, "gnt", "qtz", "dio") or contains(mainrock, "peridotite","mylonite","serpentinite","amphibolite"):
        return "Group 18"  # Igneous/Metamorphic
    elif contains(mainrock, "melange") or contains(mapname, "melange") or contains(simplename, "Allochthonous"):
        return "Group 15" # Undifferentiated sedimentary rock
    elif contains(mapname, "fan", "scree", 'avalanche') or contains(mapsymbol, "fan"):
        return "Group 10"
    else:
        return "Manual Classification"


def apply_geological_classification(input_file, output_file):
    df = pd.read_csv(input_file, delimiter=":", encoding="utf-8")

    # Apply classification
    df["FlowchartGroup"] = df.apply(classify_by_flowchart, axis=1)

    # Reorder columns to ensure FlowchartGroup is the last column
    cols = [col for col in df.columns if col != "FlowchartGroup"] + ["FlowchartGroup"]
    df = df[cols]

    # Save to file
    df.to_csv(output_file, index=False)
    print(f"Classification results saved to: {output_file}")


# Example usage
apply_geological_classification("GeologicalUnit_Addgeometry_4th2.csv", "GeologicalUnit_Classified_new2.csv")
