import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras
print("Backend set to TensorFlow successfully!")

import os
import argparse
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import mplhep as hep
import joblib

def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model directory, .keras only supported", type=str, required=True)
parser.add_argument("--pq_file", help="input pq file, with relative path", type=str, required=True)
parser.add_argument("--column_name", help="name of the output column of the pq file", type=str, required=False)
parser.add_argument("--pq_file_output", help="name of the output pq file", type=str, required=False)
parser.add_argument("--isSingleH", action="store_true", dest="isSingleH", default=False)
args = parser.parse_args()

isSingleH = args.isSingleH
if isSingleH:
    print("Model directory argument should be parent directory with subdirectories for singleH and VH, with trailing /.")


# Load Parquet file
parquet_file_incl = pq.read_table(args.pq_file)
df_incl = parquet_file_incl.to_pandas()
print("Loaded parquet from %s." % args.pq_file)

rename_vars = [
    "HHbbggCandidate_pt",
    "HHbbggCandidate_eta",
    "HHbbggCandidate_phi",
    "HHbbggCandidate_mass",
    "CosThetaStar_CS",
    "CosThetaStar_gg",
    "CosThetaStar_jj",
    "DeltaR_j1g1",
    "DeltaR_j2g1",
    "DeltaR_j1g2",
    "DeltaR_j2g2",
    "DeltaR_jg_min",
    "lead_bjet_pt",
    "lead_bjet_eta",
    "lead_bjet_btagPNetB",
    "sublead_bjet_pt",
    "sublead_bjet_eta",
    "sublead_bjet_btagPNetB",
    "dijet_mass",
    "dijet_pt",
    "dijet_eta",
    "DeltaPhi_j1MET",
    "DeltaPhi_j2MET",
    "pholead_PtOverM",
    "phosublead_PtOverM",
    "FirstJet_PtOverM",
    "SecondJet_PtOverM",
    "lead_bjet_pt_PNet_all",
    "sublead_bjet_pt_PNet_all",
    "dijet_pt_PNet_all",
    "dijet_mass_PNet_all"
]
if isSingleH:
    for var in rename_vars:
        df_incl = df_incl.rename(columns={"nonRes_" + var: var})

models = [""]
if isSingleH:
    models = ["VH", "SingleH"]
for modelname in models:
    model_dir = args.model + modelname

    # Ensure model directory exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

    # Search for YAML and .keras files in the model directory
    yaml_files = [f for f in os.listdir(model_dir) if f.endswith(".yaml")]
    mod_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]

    if not yaml_files or len(yaml_files) > 1:
        raise FileNotFoundError("No YAML file found or too many YAML files in the model directory.")
    if not mod_files or len(mod_files) > 1:
        raise FileNotFoundError("No .keras file found or too many .keras files in the model directory.")

    # Load model and variables
    model_path = os.path.join(model_dir, mod_files[0])
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    variables_file = os.path.join(model_dir, yaml_files[0])
    variables = load_yaml(variables_file)
    print(f"Loaded variables from: {variables_file}")


    input_variables = [var for var in variables.keys()]

    print("Checking for missing variables...")
    for var in input_variables:
        if var not in df_incl.columns:
            print(f"Variable '{var}' missing. Adding it...")

            # Add missing variables one by one with appropriate calculations
            if var == "pred_VHToGG":
                pass
            elif var == "gg_pT_OverHHcand_mass":
                df_incl["gg_pT_OverHHcand_mass"] = (
                    df_incl["pt"] / df_incl["HHbbggCandidate_mass"]
                )
            elif var == "jj_pT_OverHHcand_mass":
                df_incl["jj_pT_OverHHcand_mass"] = (
                    df_incl["dijet_pt"] / df_incl["HHbbggCandidate_mass"]
                )
            elif var == "pholead_PtOverM":
                df_incl["pholead_PtOverM"] = (
                    df_incl["lead_pt"] / df_incl["mass"]
                )
            elif var == "lead_j_pT_OverHbbcand_mass":
                df_incl["lead_j_pT_OverHbbcand_mass"] = (
                    df_incl["lead_bjet_pt"] / df_incl["dijet_mass"]
                )
            elif var == "phosublead_PtOverM":
                df_incl["phosublead_PtOverM"] = (
                    df_incl["sublead_pt"] / df_incl["mass"]
                )
            elif var == "sublead_j_pT_OverHbbcand_mass":
                df_incl["sublead_j_pT_OverHbbcand_mass"] = (
                    df_incl["sublead_bjet_pt"] / df_incl["dijet_mass"]
                )
            elif var == "lead_photon_EnergyErrOverE":
                df_incl["lead_photon_EnergyErrOverE"] = (
                    df_incl["lead_energyErr"] / df_incl["lead_energyRaw"]
                )
            elif var == "sublead_photon_EnergyErrOverE":
                df_incl["sublead_photon_EnergyErrOverE"] = (
                    df_incl["lead_energyErr"] / df_incl["lead_energyRaw"]
                )
            elif var == "HHbbggCandidate_ptoverMggjj":
                df_incl["HHbbggCandidate_ptoverMggjj"] = (
                    df_incl["HHbbggCandidate_pt"] / df_incl["HHbbggCandidate_mass"]
                )
            elif var == "lead_bjet_ptOverMjj_PNet_all":
                df_incl["lead_bjet_ptOverMjj_PNet_all"] = (
                    df_incl["lead_bjet_pt_PNet_all"] / df_incl["dijet_mass_PNet_all"]
                )
            elif var == "sublead_bjet_ptOverMjj_PNet_all":
                df_incl["sublead_bjet_ptOverMjj_PNet_all"] = (
                    df_incl["sublead_bjet_pt_PNet_all"] / df_incl["dijet_mass_PNet_all"]
                )
            elif var == "dijet_pt_PNet_all_OverHHcand_mass":
                df_incl["dijet_pt_PNet_all_OverHHcand_mass"] = (
                    df_incl["dijet_pt_PNet_all"] / df_incl["HHbbggCandidate_mass"]
                )
            else:
                raise ValueError(f"Missing variable '{var}' is not recognized or cannot be computed.")
            
            print(f"Variable '{var}' added successfully.")

    # Set the output column name for predictions
    if modelname == "VH":
        colname = "pred_VHToGG"
    elif modelname == "SingleH":
        colname = "pred_SingleH"
    else:
        "new_model"
    output_column_name = args.column_name if args.column_name else colname

    if isSingleH:
        transformer = joblib.load(model_dir + "/scaler.gz")
        df_trans = df_incl.copy(deep=True)
        df_trans = df_trans[input_variables]
        stand_vals = transformer.transform(df_trans.values)
        df_trans = pd.DataFrame(stand_vals, columns=input_variables)

        print("Making predictions...")
        prediction = model.predict(df_trans[input_variables])
    else:
        print("Making predictions...")
        prediction = model.predict(df_incl[input_variables])

    df_incl[output_column_name] = prediction
    print(f"Predictions added to column: '{output_column_name}'")

if args.pq_file_output:
    for var in rename_vars + ["dijet_PtOverMggjj_PNet_all", "sublead_bjet_ptOverMjj_PNet_all", "lead_bjet_ptOverMjj_PNet_all", "HHbbggCandidate_ptoverMggjj"]:
        df_incl = df_incl.rename(columns={var: "nonRes_" + var})

    df_incl.to_parquet(args.pq_file_output+'.parquet', index=False)
    print(f"Output file saved as: {args.pq_file_output}.parquet")

# Visualization
list_samples = df_incl["proc"].unique()
nodata = [sample for sample in list_samples if "Data" not in sample]

fig, ax = plt.subplots(figsize=(10, 10))
hep.style.use("CMS")
hep.cms.label("Preliminary", data=True, lumi=22, year=2022)

print(nodata)
for sample in nodata:
    plt.hist(
        df_incl.loc[df_incl["proc"] == sample][output_column_name],
        label=f"{sample} {output_column_name}",
        histtype="step",
        weights=df_incl.loc[df_incl["proc"] == sample]["weight"],
        bins=20,
        range=(0, 1),
        linewidth=2,
        density=True,
    )

plt.legend()
plt.show()
