import numpy as np
import os
'''
Central parameter and setup module for Schwen model including:
  - params_log, params_linear (converted to linear scale)
  - datafile_configurations
  - helper functions to compute the observable
  - root directory for generated data in and outputs
  - initial conditions map
'''

# Ensuring precision in numerical calculations
DTYPE = np.float64

'''
Root directories
'''
data_root = "data"
os.makedirs(data_root, exist_ok=True)


'''
Setup of datafile configurations and parameters dictionaries
Ins and nExpID vary between datafiles
Ins corresponds to the initial concentration of insulin in extracellular medium
nExpID corresponds to the experiment ID: determines km, offset and scaleElisa parameters

km:  Michaelis-Menten constant for the experiment. This can affect how quickly certain reactions saturate
offset: Adjusts the "zero point" or baseline for the observable. If, for example, a measurement device has a background signal, this compensates.
scaleElisa: A scaling factor for ELISA measurements (ELISA is a common lab technique to quantify proteins). It can convert model units to experimental units or vice versa.
fragments: A scaling factor for insulin fragments, which can affect the observable if the model includes insulin degradation or fragmentation.

parameters are implemented in log10 scale as provided by Hass et al authors
'''

datafile_configs: dict[str, dict[str, float | int]] = {
    "Datafile 9":   {"Ins": 10.0,    "nExpID": 1},
    "Datafile 10":  {"Ins": 10000.0, "nExpID": 1},
    "Datafile 12":  {"Ins": 10.0,    "nExpID": 2},
    "Datafile 13":  {"Ins": 100.0,   "nExpID": 2},
    "Datafile 15":  {"Ins": 10.0,    "nExpID": 3},
    "Datafile 16":  {"Ins": 100.0,   "nExpID": 3},
    "Datafile 18":  {"Ins": 100.0,   "nExpID": 4},
    "Datafile 19":  {"Ins": 10000.0, "nExpID": 4}
}

params_log = {
    "log10_IR_obs_std": -1.32618990031786,
    "log10_ini_R1": 1.76974916555649,
    "log10_ini_R2fold": 1.19337957381128,
    "log10_ka1": -2.42026674135759,
    "log10_ka2fold": 0.690404902719026,
    "log10_kd1": 0.961195911453211,
    "log10_kd2fold": 0.841519867982954,
    "log10_kin": -0.429611493204321,
    "log10_kin2": -0.268990429804171,
    "log10_koff_unspec": 1.00248730057157,
    "log10_kon_unspec": 1.30642041909865,
    "log10_kout": -1.34606420308761,
    "log10_kout2": -1.45062969370574,
    "log10_kout_frag": -1.95797187598835,
    # experiment-specific values (log10 or linear as appropriate)
    "log10_km_nExpID1": 7.99999999400177,
    "log10_km_nExpID2": 7.99999999967961,
    "log10_km_nExpID3": 7.99999999775497,
    "log10_km_nExpID4": 7.99999999899634,
    "log10_offset_nExpID1": -1.59989083921818,
    "log10_offset_nExpID2": -1.66264461812658,
    "log10_offset_nExpID3": -2.13567814524502,
    "log10_offset_nExpID4": 0.19509718712192,
    "log10_scale": -0.880061554890055,
    # scaleElisa and fragments already linear
    "scaleElisa_nExpID1": 0.538009450729947,
    "scaleElisa_nExpID2": 0.569911881220408,
    "scaleElisa_nExpID3": 0.100000000170598,
    "scaleElisa_nExpID4": 0.999999999951202,
    "fragments": 0.999999072086866
}

# converting parameters from log10 scale to linear scale and saving in a new dictionary
params_linear: dict[str, float] = {}
for k, v in params_log.items():
    if k.startswith("log10_"):
        params_linear[k.replace("log10_", "")] = DTYPE(10.0) ** DTYPE(v)
    else:
        params_linear[k] = DTYPE(v)

''' 
Definition of Initial Conditions
Ins is dependent on datafile and defined above

| **Ins**              | Free insulin in extracellular medium (outside the cell)                |
| -------------------- | ---------------------------------------------------------------------- |
| **Rec1**             | Surface insulin receptor type 1                                        |
| **Rec2**             | Surface insulin receptor type 2                                        |
| **IR1**              | Surface insulin–receptor 1 complex                                     |
| **IR2**              | Surface insulin–receptor 2 complex                                     |
| **IR1in**            | Internalized IR1 complex (inside the cell)                             |
| **IR2in**            | Internalized IR2 complex (inside the cell)                             |
| **BoundUnspec**      | Insulin bound to unspecific (non-receptor) sites                       |
| **InsulinFragments** | Degraded/fragmented insulin from internalized complexes                |
| **Uptake1**          | Cumulative insulin uptake via Rec1 pathway (bookkeeping/tracking only) |
| **Uptake2**          | Cumulative insulin uptake via Rec2 pathway (bookkeeping/tracking only) |
'''


_ini_R1 = params_linear["ini_R1"]
_ini_R2fold = params_linear["ini_R2fold"]

initial_conditions_map = {
    "Rec1": _ini_R1,
    "Rec2": _ini_R2fold * _ini_R1,
    "IR1": 0.0,
    "IR2": 0.0,
    "IR1in": 0.0,
    "IR2in": 0.0,
    "BoundUnspec": 0.0,
    "InsulinFragments": 0.0,
    "Uptake1": 0.0,
    "Uptake2": 0.0,
}

'''
Helper functions for observable
First retrieving all necessary parameters
Seond calculating observable as providedby Hass et al authors
'''
def get_observable_params(params: dict[str, float], nExpID: int) -> tuple[float, float, float, float]:
    km = params[f"km_nExpID{nExpID}"]
    offset = params[f"offset_nExpID{nExpID}"]
    scaleElisa = params[f"scaleElisa_nExpID{nExpID}"]
    fragments = params["fragments"]
    return km, offset, scaleElisa, fragments

def compute_observable(Ins, InsulinFragments, km, offset, scaleElisa, fragments):
    signal = Ins + InsulinFragments * fragments
    denominator = (signal / km) + 1
    return np.log10(np.maximum(offset + (scaleElisa * signal) / denominator, 1e-12))