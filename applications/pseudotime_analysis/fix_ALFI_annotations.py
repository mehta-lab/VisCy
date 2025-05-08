# %%
import pandas as pd


def fix_annotations(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Create a copy to modify
    fixed_df = df.copy()

    # Group by FOV name first, then process each FOV separately
    for fov_name, fov_df in df.groupby("fov_name"):
        # Get unique track IDs within this FOV
        track_ids = fov_df["track_id"].unique()

        # Create a dictionary to store the last ID of each track
        last_ids = {}

        # First pass: collect the last ID of each track in this FOV
        for track_id in track_ids:
            track_data = fov_df[fov_df["track_id"] == track_id]
            last_ids[track_id] = track_data["id"].iloc[-1]

        # Second pass: fix parent_id and parent_track_id
        for track_id in track_ids:
            track_data = fov_df[fov_df["track_id"] == track_id]
            first_row = track_data.iloc[0]

            # If this is a child track (parent_track_id != -1)
            if first_row["parent_track_id"] != -1:
                parent_track_id = first_row["parent_track_id"]

                # Fix parent_id in the first row if it's -1
                if first_row["parent_id"] == -1:
                    parent_last_id = last_ids.get(parent_track_id)
                    if parent_last_id is not None:
                        # Find the index of this row in the fixed_df
                        idx = fixed_df[
                            (fixed_df["fov_name"] == fov_name)
                            & (fixed_df["track_id"] == track_id)
                            & (fixed_df["t"] == first_row["t"])
                        ].index[0]
                        fixed_df.at[idx, "parent_id"] = parent_last_id

                # Make sure all rows in this track have the same parent_track_id
                track_indices = fixed_df[
                    (fixed_df["fov_name"] == fov_name)
                    & (fixed_df["track_id"] == track_id)
                ].index
                fixed_df.loc[track_indices, "parent_track_id"] = parent_track_id

    # Save the fixed dataframe
    fixed_df.to_csv(output_file, index=False)

    # Return summary of changes
    return {
        "original_rows": len(df),
        "fixed_rows": len(fixed_df),
        "tracks_processed": len(df.groupby(["fov_name", "track_id"])),
        "fovs_processed": len(df["fov_name"].unique()),
    }


# Example usage
if __name__ == "__main__":
    input_file = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/test_annotations.csv"
    output_file = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/fixed_test_annotations.csv"

    results = fix_annotations(input_file, output_file)
    print(
        f"Processed {results['tracks_processed']} tracks across {results['fovs_processed']} FOVs"
    )
    print(f"Fixed annotations saved to {output_file}")
# %%
