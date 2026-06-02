"""Run RALPS as a subprocess for batch correction."""

import subprocess
from pathlib import Path
import pandas as pd

"""Run RALPS as a subprocess for batch correction."""

import subprocess
from pathlib import Path
import pandas as pd

def run_ralps_correction(
    ralps_input_dir: str,  # Directory containing config.csv, merged_data_for_ralps.csv, merged_batch_for_ralps.csv
) -> pd.DataFrame:
    """
    Run RALPS using the config file in ralps_input_dir.
    - Reads out_path from config.csv to find where RALPS saves normalized.csv.
    """
    ralps_input_dir = Path(ralps_input_dir).resolve()
    config_path = ralps_input_dir / "config.csv"

    # Path to RALPS script
    ralps_script = Path(__file__).parent.parent / "RALPS" / "src" / "ralps.py"

    # Run RALPS normalization (-n) with the config
    print("Running RALPS normalization...")
    command = ["python", str(ralps_script), "-n", str(config_path)]
    print("Command:", " ".join(command))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print("RALPS stderr:", result.stderr)
        raise RuntimeError(f"RALPS failed with exit code {result.returncode}")
    if result.stdout:
        print("RALPS stdout:", result.stdout)

    # Read out_path from config.csv to find where RALPS saved normalized.csv
    config_df = pd.read_csv(config_path, index_col=0)
    ralps_output_dir = Path(config_df.loc["out_path", "values"]).resolve()

    # Load the corrected data from the out_path specified in config.csv
    corrected_path = ralps_output_dir / "normalized.csv"
    if corrected_path.exists():
        return pd.read_csv(corrected_path, index_col=0)
    else:
        all_files = list(ralps_output_dir.rglob("*"))
        print(f"Files in {ralps_output_dir}: {all_files}")
        raise FileNotFoundError(f"RALPS output 'normalized.csv' not found in {ralps_output_dir}")