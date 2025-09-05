import os
import pandas as pd
import numpy as np
from simba_ml.simulation import (system_model, species, distributions, noisers, derivative_noiser, generators, kinetic_parameters as kinetic_parameters_module)
from schwen_setup import params_linear, get_observable_params, data_root, compute_observable

'''
Reproducibility and default settings
To reproduce the exact same data as in the thesis:
Adapt to the following settings
Datafile_12: # Ins = 10.0, nExpID = 2
Datafile_13: # Ins = 100.0, nExpID = 2
num_signals = 1000 or 100
-> overall generated 4 datasets, 2 for each datafile with 100 and 1000 signals
adapt save_path accordingly:    f"1_Datafile12_size{num_signals}" to
                                f"2_Datafile12_size{num_signals}" or
                                f"3_Datafile13_size{num_signals}" or
                                f"4_Datafile13_size{num_signals}"
'''
np.random.seed(42)
default_ins     = 100.0                      
default_nExpID  = 2
num_signals = 1000
save_path = os.path.join(data_root, "simba_simulation", f"4_Datafile13_size{num_signals}")
os.makedirs(save_path, exist_ok=True)

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

'''
Beginning of model definition and signal generation
Implementation is very similar to schwen_simbaML.py
Added multiple noise sources to make it more realistic
Generates many signals to generate a dataset for training and testing
Generated signals are split into training and test sets (80% train, 20% test)
'''
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

# biochemical species list
specieses = [
    species.Species("Ins", distributions.NormalDistribution(default_ins, 0.1 * default_ins)),
    species.Species("Rec1", distributions.NormalDistribution(params_linear["ini_R1"], params_linear["ini_R1"] * 0.1)),
    species.Species("Rec2", distributions.NormalDistribution(params_linear["ini_R2fold"] * params_linear["ini_R1"], params_linear["ini_R2fold"] * params_linear["ini_R1"] * 0.1)),
    species.Species("IR1", distributions.Constant(0.0)),
    species.Species("IR2", distributions.Constant(0.0)),
    species.Species("IR1in", distributions.Constant(0.0)),
    species.Species("IR2in", distributions.Constant(0.0)),
    species.Species("BoundUnspec", distributions.Constant(0.0)),
    species.Species("InsulinFragments", distributions.Constant(0.0)),
    species.Species("Uptake1", distributions.Constant(0.0), contained_in_output=False),
    species.Species("Uptake2", distributions.Constant(0.0), contained_in_output=False)
]

base_kinetic_parameters = {
    name: kinetic_parameters_module.ConstantKineticParameter(
        distributions.NormalDistribution(val, 0.1 * val)
    )
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
        "kout_frag": params_linear["kout_frag"],
        "fragments":    params_linear["fragments"],  
    }.items()
}

# kinetic parameters specific to datafile
kp = base_kinetic_parameters.copy()
kp.update({
    "km": kinetic_parameters_module.ConstantKineticParameter(
        distributions.Constant(params_linear[f"km_nExpID{default_nExpID}"])
    ),
    "offset": kinetic_parameters_module.ConstantKineticParameter(
        distributions.Constant(params_linear[f"offset_nExpID{default_nExpID}"])
    ),
    "scaleElisa": kinetic_parameters_module.ConstantKineticParameter(
        distributions.Constant(params_linear[f"scaleElisa_nExpID{default_nExpID}"])
    )
})

# noisers
noiser = noisers.NoNoiser()
noiser = noisers.AdditiveNoiser(distributions.NormalDistribution(0, 0.01 * default_ins))
deriv_noiser = derivative_noiser.NoDerivNoiser()

# building system model
sm = system_model.SystemModel(
    name,
    specieses,
    kinetic_parameters=kp,
    deriv=deriv,
    noiser=noiser,
    deriv_noiser=deriv_noiser,
    timestamps=distributions.Constant(200)
)

# generate signals
params = get_observable_params(params_linear, default_nExpID)
signals = generators.TimeSeriesGenerator(sm).generate_signals(n=num_signals)
observables = np.array([annotate_signal(signal, params)['observable'] for signal in signals])
n_timepoints = observables.shape[1]
timepoints    = np.arange(n_timepoints)

# Split into training and test sets (80% train, 20% test)
idx = np.random.choice((True, False), num_signals, p=[0.8, 0.2])
train_signals = observables[idx]
test_signals = observables[~idx]

# Save training and test signals as .npy and .csv files
np.save(os.path.join(save_path, 'train_signals.npy'), train_signals)
np.save(os.path.join(save_path, 'test_signals.npy'),  test_signals)

cols = [f"t{t}" for t in timepoints]

train_df = pd.DataFrame(train_signals, columns=cols)
train_df.to_csv(os.path.join(save_path, "train_signals.csv"), index=False)
test_df = pd.DataFrame(test_signals, columns=cols)
test_df.to_csv(os.path.join(save_path, "test_signals.csv"), index=False)