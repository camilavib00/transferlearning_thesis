import os
import numpy as np
from simba_ml.simulation import (system_model, species, distributions, noisers, derivative_noiser, generators, kinetic_parameters as kinetic_parameters_module)
from schwen_setup import (params_linear, datafile_configs, get_observable_params, compute_observable, data_root)

# ensure numeric precision
DTYPE = np.float64

# simulation settings
TIME_TOTAL  = 200.0
N_STEPS     = 200
TIME_GRID   = np.linspace(0.0, TIME_TOTAL, N_STEPS)

# helper to compute observable for a generated signal (DataFrame-like)
def annotate_signal(sig, params):
    km, offset, scaleElisa, fragments = params
    return (
        sig
        .rename_axis("time")
        .reset_index()
        .assign(
            observable=lambda df: compute_observable(
                Ins=df["Ins"].values,
                InsulinFragments=df["InsulinFragments"].values,
                km=km,
                offset=offset,
                scaleElisa=scaleElisa,
                fragments=fragments,
            )
        )
    )

# model name
name = "Schwen"

# ODE derivative function
def deriv(t: float, y: list[float], p: dict[str, float]) -> tuple[float, ...]:
    Ins, Rec1, Rec2, IR1, IR2, IR1in, IR2in, BoundUnspec, InsulinFragments, Uptake1, Uptake2 = y
    dIns_dt = (
        BoundUnspec * p["koff_unspec"]
      + IR1         * p["kd1"]
      + IR2         * p["kd1"] * p["kd2fold"]
      - Ins         * p["kon_unspec"]
      - Ins * Rec1  * p["ka1"]
      - Ins * Rec2  * p["ka1"] * p["ka2fold"]
    )
    dRec1_dt  = IR1  * p["kd1"] + IR1in * p["kout_frag"] - Ins * Rec1 * p["ka1"]
    dRec2_dt  = IR2in * p["kout_frag"] + IR2 * p["kd1"] * p["kd2fold"] - Ins * Rec2 * p["ka1"] * p["ka2fold"]
    dIR1_dt   = IR1in * p["kout"]      - IR1 * p["kin"]   - IR1 * p["kd1"]   + Ins * Rec1 * p["ka1"]
    dIR2_dt   = IR2in * p["kout2"]     - IR2 * p["kin2"]  - IR2 * p["kd1"]*p["kd2fold"] + Ins * Rec2 * p["ka1"]*p["ka2fold"]
    dIR1in_dt = IR1  * p["kin"]        - IR1in * p["kout_frag"]
    dIR2in_dt = IR2  * p["kin2"]       - IR2in * p["kout2"] - IR2in * p["kout_frag"]
    dUpt1_dt  = Ins * Rec1 * p["ka1"]  - IR1   * p["kd1"]
    dUpt2_dt  = Ins * Rec2 * p["ka1"]*p["ka2fold"] - IR2 * p["kd1"]*p["kd2fold"]
    dFrag_dt  = IR1in * p["kout_frag"] + IR2in * p["kout_frag"]
    dBound_dt = Ins * p["kon_unspec"]  - BoundUnspec * p["koff_unspec"]
    return (
        dIns_dt, dRec1_dt, dRec2_dt, dIR1_dt, dIR2_dt,
        dIR1in_dt, dIR2in_dt, dBound_dt,  dFrag_dt,
        dUpt1_dt,  dUpt2_dt
    )

# Base kinetic parameters for reactions
base_kinetic_parameters = {
    name: kinetic_parameters_module.ConstantKineticParameter(distributions.Constant(val))
    for name, val in {
        "ka1": params_linear["ka1"],
        "ka2fold": params_linear["ka2fold"],
        "kd1": params_linear["kd1"],
        "kd2fold": params_linear["kd2fold"],
        "kin": params_linear["kin"],
        "kin2": params_linear["kin2"],
        "koff_unspec": params_linear["koff_unspec"],
        "kon_unspec": params_linear["kon_unspec"],
        "kout": params_linear["kout"],
        "kout2": params_linear["kout2"],
        "kout_frag": params_linear["kout_frag"]
    }.items()
}

# Noisers
noiser       = noisers.NoNoiser()
deriv_noiser = derivative_noiser.NoDerivNoiser()

# Output directory (single unified subfolder)
out_dir = os.path.join(data_root, "simba_simulation")
os.makedirs(out_dir, exist_ok=True)

# Loop over all datafile configurations
for label, cfg in datafile_configs.items():
    ins_val = cfg["Ins"]
    nExpID  = cfg["nExpID"]

    # prepare species list -> species are defined in the order of state_var_order
    # species = biochemical species (state variables)
    specieses = [
        species.Species("Ins", distributions.Constant(ins_val)),
        species.Species("Rec1", distributions.Constant(params_linear["ini_R1"])),
        species.Species("Rec2", distributions.Constant(params_linear["ini_R2fold"] * params_linear["ini_R1"])),
        species.Species("IR1", distributions.Constant(0.0)),
        species.Species("IR2", distributions.Constant(0.0)),
        species.Species("IR1in", distributions.Constant(0.0)),
        species.Species("IR2in", distributions.Constant(0.0)),
        species.Species("BoundUnspec", distributions.Constant(0.0)),
        species.Species("InsulinFragments", distributions.Constant(0.0)),
        species.Species("Uptake1", distributions.Constant(0.0), contained_in_output=False),
        species.Species("Uptake2", distributions.Constant(0.0), contained_in_output=False)
    ]

    # Per-config kinetic parameters (km/offset/scaleElisa + fragments)
    kp = base_kinetic_parameters.copy()
    kp.update({
        "km": kinetic_parameters_module.ConstantKineticParameter(distributions.Constant(params_linear[f"km_nExpID{nExpID}"])),
        "offset": kinetic_parameters_module.ConstantKineticParameter(distributions.Constant(params_linear[f"offset_nExpID{nExpID}"])),
        "scaleElisa": kinetic_parameters_module.ConstantKineticParameter(distributions.Constant(params_linear[f"scaleElisa_nExpID{nExpID}"])),
    })

    # Build system model with requested solver settings
    sm = system_model.SystemModel(
        name,
        specieses,
        kinetic_parameters=kp,
        deriv=deriv,
        noiser=noiser,
        deriv_noiser=deriv_noiser,
        timestamps=distributions.Constant(N_STEPS),
        solver_method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    # Generate one signal
    signals, = generators.TimeSeriesGenerator(sm).generate_signals(n=1)

    # Annotate observable and align time
    params = get_observable_params(params_linear, nExpID)
    df = annotate_signal(signals, params)
    if "time" in df.columns and len(df) == len(TIME_GRID):
        df["time"] = TIME_GRID

    # Save exactly one file per config into data_root/simba_simulation
    safe_label = label.replace(" ", "_")
    out_name = f"{label}_Exp{nExpID}_Ins{int(ins_val)}.xlsx"
    out_path = os.path.join(out_dir, out_name)
    if not os.path.exists(out_path):
        df.to_excel(out_path, index=False)
    else:
        print("file already exists:", out_path)
