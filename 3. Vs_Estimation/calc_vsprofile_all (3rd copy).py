import pandas as pd
from pathlib import Path
from vs_calc import CPT, VsProfile, vs30_correlations

examples_dir = Path('./CPT_rawdata_JK3')
out_dir = Path('./EstimatedVs_2509')
out_dir.mkdir(exist_ok=True)

log_file = out_dir / "error_log_2509.txt"

for file_path in examples_dir.glob('*.csv'):
    file_name = file_path.stem
    try:
        cpt = CPT.from_file(str(file_path))

        # Vs Correlations
        vs_correlations = "andrus_2007_pleistocene"
        vs_correlations2 = "mcgann_2015"
        vs_correlations3 = "mcgann_2018"

        # Create VsProfile objects
        vs_profile = VsProfile.from_cpt(cpt, vs_correlations)
        vs_profile2 = VsProfile.from_cpt(cpt, vs_correlations2)
        vs_profile3 = VsProfile.from_cpt(cpt, vs_correlations3)

        # Extract depth and Vs values
        depth, vs = vs_profile.depth, vs_profile.vs
        depth2, vs2 = vs_profile2.depth, vs_profile2.vs
        depth3, vs3 = vs_profile3.depth, vs_profile3.vs

        # Read input CSV
        df = pd.read_csv(file_path)
        df.iloc[:, 0] = df.iloc[:, 0].round(2)

        # Add Vs correlations
        vs_df = pd.DataFrame({
            'Depth': depth,
            'Andrus-P Vs': vs,
            'Mc15 Vs': vs2,
            'Mc18 Vs': vs3
        })
        df = pd.merge(df, vs_df, how='left', on='Depth')

        # Add Ic values
        ic_df = pd.DataFrame({
            'Depth': cpt.depth,
            'Ic': cpt.Ic
        })

        df = pd.merge_asof(df.sort_values('Depth'),
                           ic_df.sort_values('Depth'),
                           on='Depth')
        df.iloc[:, 0] = df.iloc[:, 0].round(2)

        # Save result
        df.to_csv(out_dir / f'{file_name}_result.csv', index=False)
        print(f'Done: {file_name}')

    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"Error in file {file_name}: {str(e)}\n")
        print(f"Skipped {file_name} due to error.")
