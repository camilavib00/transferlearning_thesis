import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from scipy.integrate import solve_ivp
from schwen_setup import (data_root, params_linear, datafile_configs, get_observable_params, compute_observable, initial_conditions_map)

# use LaTeX fonts if available
try:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.unicode_minus": False,
        "font.size": 12, "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 11,
    })
except Exception:
    mpl.rcParams.update({"text.usetex": False, "font.family": "serif", "axes.unicode_minus": False})

# defining order of state variables for ODE function and solver
# so that initial conditions and the ODE system always agree about which value is which variable
state_var_order = [
    "Ins",
    "Rec1",
    "Rec2",
    "IR1",
    "IR2",
    "IR1in",
    "IR2in",
    "BoundUnspec",
    "InsulinFragments",
    "Uptake1",
    "Uptake2"
]

# ODE system function
def schwen_ode(t, y, params):

    Ins, Rec1, Rec2, IR1, IR2, IR1in, IR2in, BoundUnspec, InsulinFragments, Uptake1, Uptake2 = y
    
    # unpack linear parameters
    ka1 = params["ka1"]
    ka2fold = params["ka2fold"]
    kd1 = params["kd1"]
    kd2fold = params["kd2fold"]
    kin = params["kin"]
    kin2 = params["kin2"]
    koff_unspec = params["koff_unspec"]
    kon_unspec = params["kon_unspec"]
    kout = params["kout"]
    kout2 = params["kout2"]
    kout_frag = params["kout_frag"]
    fragments = params["fragments"]

    # system of 11 ODEs as defined in the general_info sheet
    dIns_dt = (
        BoundUnspec * koff_unspec
        + IR1 * kd1
        + IR2 * kd1 * kd2fold
        - Ins * kon_unspec
        - Ins * Rec1 * ka1
        #+ IR2 * kd1 * kd2fold # correction by umur: duplicate term, this line should be removed
        - Ins * Rec2 * ka1 * ka2fold
        #- Ins * Rec2 * ka1 * ka2fold # correction by umur: duplicate term, this line should be removed
    )
    dRec1_dt = IR1 * kd1 + IR1in * kout_frag - Ins * Rec1 * ka1
    dRec2_dt = IR2in * kout_frag + IR2 * kd1 * kd2fold - Ins * Rec2 * ka1 * ka2fold
    dIR1_dt = IR1in * kout - IR1 * kin - IR1 * kd1 + Ins * Rec1 * ka1
    dIR2_dt = IR2in * kout2 - IR2 * kin2 - IR2 * kd1 * kd2fold + Ins * Rec2 * ka1 * ka2fold
    dIR1in_dt = IR1 * kin - IR1in * kout_frag
    dIR2in_dt = IR2 * kin2 - IR2in * kout2 - IR2in * kout_frag
    dUptake1_dt = Ins * Rec1 * ka1 - IR1 * kd1
    dUptake2_dt = Ins * Rec2 * ka1 * ka2fold - IR2 * kd1 * kd2fold
    dInsulinFragments_dt = IR1in * kout_frag + IR2in * kout_frag
    dBoundUnspec_dt = Ins * kon_unspec - BoundUnspec * koff_unspec

    return [
        dIns_dt, dRec1_dt, dRec2_dt, dIR1_dt, dIR2_dt,
        dIR1in_dt, dIR2in_dt, dBoundUnspec_dt, dInsulinFragments_dt,
        dUptake1_dt, dUptake2_dt
    ]

# simulation with a time span of 60 timepoints
t_span = (0, 60)
t_eval = np.linspace(*t_span, 200)
plot_dir = os.path.join(".", "plots", "SciPy_Implementation")
os.makedirs(plot_dir, exist_ok=True)

# looping over each configuration and solving the ODE system
for label, config in datafile_configs.items():
    nExpID = config["nExpID"]
    Ins_init = config["Ins"]

    ic = initial_conditions_map.copy()
    ic["Ins"] = Ins_init
    initial_conditions = [ic[var] for var in state_var_order]

    #solve ODE-observable with scipy's solve_ivp
    sol = solve_ivp(
        fun=lambda t, y: schwen_ode(t, y, params_linear),
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval
    )

    # compute observable based on the solution and experiment-specific parameters
    km, offset, scaleElisa, fragments = get_observable_params(params_linear, nExpID)
    obs = compute_observable(
        Ins=sol.y[0],
        InsulinFragments=sol.y[8],
        km=km,
        offset=offset,
        scaleElisa=scaleElisa,
        fragments=fragments
    )

    '''
    Plotting and saving the results for each configuration
    Includes experimental and simulation data from Hass et al.
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, obs, label="Model Simulation", lw=2)

    # reading the Hass et al. experimental and simulation data
    file_num = ''.join(filter(str.isdigit, label))
    excel_path = os.path.join(data_root, "schwen_modeldata", f"model1_data{file_num}.xlsx")

    df_exp = pd.read_excel(excel_path, sheet_name="Exp Data")
    df_sim = pd.read_excel(excel_path, sheet_name="Simulation")

    plt.scatter(df_exp["time"], df_exp["Insulin_obs"], color='black', label="Exp Data", zorder=10)
    plt.scatter(df_sim["time"], df_sim["Insulin_obs"], color='red', marker='x', label="Provided Simulation", zorder=10)

    plt.xlabel("Time")
    plt.ylabel(r"Insulin Observable ($log_{10}$ scale)")
    plt.title(f"Schwen ODE Model: {label} (Ins={Ins_init}, nExpID={nExpID})")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"{label}_Ins{Ins_init}_Exp{nExpID}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()