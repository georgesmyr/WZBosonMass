import uproot
import pandas as pd
import numpy as np
import ROOT

import my_print as mp

inFileName = "SM.root"
treeName = "LHEF"
outFileName = "My_SM_hist.root"
plotFileName = "SM_plots.pdf"

mp.green("BEGINNING")
mp.blue("Reading from: {}\nWriting to: {}".format(inFileName,outFileName))
print("====================================")

outfile = ROOT.TFile.Open(outFileName,"RECREATE")
tree_df = uproot.open(inFileName)[treeName].pandas.df(flatten=False)






#----------- PT,ETA,PHI,E ----------

#---- PT ----
pT_values = tree_df["Particle.PT"].to_numpy()
prod_pT = pd.DataFrame.from_records(pT_values)
prod_pT = prod_pT.iloc[:,4:8]

#----- ETA -----
eta_values = tree_df["Particle.Eta"].to_numpy()
prod_eta = pd.DataFrame.from_records(eta_values)
prod_eta = prod_eta.iloc[:,4:8]

#----- PHI -----
phi_values = tree_df["Particle.Phi"].to_numpy()
prod_phi = pd.DataFrame.from_records(phi_values)
prod_phi = prod_phi.iloc[:,4:8]

#----- E -----
enrg_values = tree_df["Particle.E"].to_numpy()
prod_enrg = pd.DataFrame.from_records(enrg_values)
prod_enrg = prod_enrg.iloc[:,4:8]

# ----- Products_ID -----
prod_id = tree_df["Particle.PID"]
prod_id = pd.DataFrame.from_records(prod_id)
prod_id = prod_id.iloc[:,4:8]

# ----- Px, Py & Pz -----
prod_px = tree_df["Particle.Px"]
prod_px = pd.DataFrame.from_records(prod_px)
prod_px = prod_px.iloc[:,4:8]

prod_py = tree_df["Particle.Py"]
prod_py = pd.DataFrame.from_records(prod_py)
prod_py = prod_py.iloc[:,4:8]

prod_pz = tree_df["Particle.Pz"]
prod_pz = pd.DataFrame.from_records(prod_pz)
prod_pz = prod_pz.iloc[:,4:8]






# -------------- CREATION OF CHARGE DATAFRAME --------------
prod_chrg = prod_id
prod_chrg=prod_chrg.replace(-13,1)
prod_chrg=prod_chrg.replace(-12,0)
prod_chrg=prod_chrg.replace(13,-1)
prod_chrg=prod_chrg.replace(11,-1)
total_charge = prod_chrg.sum(axis=1)






# ----- PARTICLES AND CUTS ------

prods = pd.concat([prod_px,prod_py,prod_pz,prod_pT,prod_eta,prod_phi,prod_enrg,prod_id,prod_chrg,total_charge],axis=1)
prods.columns = ["px_1","px_2","px_3","px_4","py_1","py_2","py_3","py_4","pz_1","pz_2","pz_3","pz_4","pT_1","pT_2","pT_3","pT_4","eta_1","eta_2","eta_3","eta_4","phi_1","phi_2","phi_3","phi_4","E_1","E_2","E_3","E_4","ID_1","ID_2","ID_3","ID_4","Charge_1","Charge_2","Charge_3","Charge_4","Total Charge"]
prods = prods.dropna()

# ----- Baseline Selection Cut -----
baseline_pT_cut = (prods["pT_1"]>5) & (prods["pT_2"]>5) & (prods["pT_3"]>5) & (prods["pT_4"]>5)
baseline_eta_cut = (abs(prods["eta_1"])<2.7) & (abs(prods["eta_2"])<2.7) & (abs(prods["eta_3"])<2.7) & (abs(prods["eta_4"])<2.7)
prods_baseline = prods.loc[baseline_pT_cut & baseline_eta_cut].reset_index(drop=True)

# ----- Z Selection Cut -----
z_selection_pT_cut = (prods_baseline["pT_1"]>15) & (prods_baseline["pT_2"]>15) & (prods_baseline["pT_3"]>15) & (prods_baseline["pT_4"]>15)
z_selection_eta_cut = (abs(prods_baseline["eta_1"])<2.5) & (abs(prods_baseline["eta_2"])<2.5) & (abs(prods_baseline["eta_3"])<2.5) & (abs(prods_baseline["eta_4"])<2.5)
prods_z_selection = prods_baseline.loc[z_selection_pT_cut & z_selection_eta_cut].reset_index(drop=True)

# ----- W Selection Cut ------
w_selection_pT_cut = (prods_z_selection["pT_1"]>20) & (prods_z_selection["pT_2"]>20) & (prods_z_selection["pT_3"]>20) & (prods_z_selection["pT_4"]>20)
prods_w_selection = prods_z_selection.loc[w_selection_pT_cut].reset_index(drop=True)










# -------------- NUMBER OF W+/- OR Z -------------------
        
w_minus_counter = list(prods_w_selection["Total Charge"]).count(-1)
w_plus_counter = list(prods_w_selection["Total Charge"]).count(1)
z_counter = list(prods_z_selection["Total Charge"]).count(0)
lepton_counter = len(total_charge)
data_entries = len(tree_df["Particle.PID"])

print("============ INFO FROM LEPTON ANALYSIS =================")
print("Number of events with 3 leptons: {}/{}".format(lepton_counter,data_entries))
print("Number of W+ bosons produced: {}".format(w_plus_counter))
print("Number of W- bososn produced: {}".format(w_minus_counter))
if w_minus_counter != 0:
    print("Number of W+/W-: {}".format(w_plus_counter/w_minus_counter))
print("=============================")










#-------------- HIGH LEVEL VARIABLES CALCULATION -------------

# ----- Z TRANSVERSE MOMENTUM -----

prods_z_selection["pT_z"] = prods_z_selection["pT_1"]+prods_z_selection["pT_2"]

# ----- Z INVARIANT MASS -----

prods_z_selection["z_mass"] = np.sqrt((prods_z_selection["E_1"]+prods_z_selection["E_2"])**2 - ((prods_z_selection["px_1"]+prods_z_selection["px_2"])**2 + (prods_z_selection["py_1"]+prods_z_selection["py_2"])**2 + (prods_z_selection["pz_1"]+prods_z_selection["pz_2"])**2))

# ----- W TRANSVERSE MASS -----

prods_w_selection["w_transverse_mass"]=np.sqrt((prods_w_selection["E_3"]+prods_w_selection["E_4"])**2-(prods_w_selection["pT_3"]+prods_w_selection["pT_4"])**2)

# ----- WZ DIBOSON SYSTEM TRANSVERSE SYSTEM -----

prods_w_selection["wz_transverse_mass"] =np.sqrt((prods_w_selection["pT_1"]+prods_w_selection["pT_2"]+prods_w_selection["pT_3"]+prods_w_selection["pT_4"])**2 -(prods_w_selection["px_1"]+prods_w_selection["px_2"]+prods_w_selection["px_3"]+prods_w_selection["px_4"])**2
-(prods_w_selection["py_1"]+prods_w_selection["py_2"]+prods_w_selection["py_3"]+prods_w_selection["py_4"])**2)










# --------------- HIGH LEVEL VARIABLE HISTOGRAMS  ---------------
lumi0 = 36.1
new_lumi = 20

# ----- Z TRANSVERSE MOMENTUM -----
pT_z = pd.Series(prods_z_selection["pT_z"]).to_numpy()
bins_pT_z = np.linspace(0,200,num=200)
count, edges = np.histogram(pT_z,bins=bins_pT_z)

h_pTz = ROOT.TH1D("Z Transverse Momentum, {}".format(lumi0),";p_{T}^{Z};Count",len(edges)-1,edges)
h_new_pTz = ROOT.TH1D("Z Transverse Momentum, {}".format(new_lumi),";p_{T}^{Z};Count",len(edges)-1,edges)

for j in range(0,len(edges)-1):
    try:
        h_pTz.SetBinContent(j+1,count[j])
        h_new_pTz.SetBinContent(j+1,count[j]*new_lumi/lumi0)
    except Exception as e:
        mp.red(e)

# ----- Z MASS -----
z_mass = pd.Series(prods_z_selection["z_mass"]).to_numpy()
bins_z_mass = np.linspace(0,200,num=20)
count, edges = np.histogram(z_mass,bins=bins_z_mass)

h_mz = ROOT.TH1D("Z Invariant Mass, {} ".format(lumi0),";m_{z};Count",len(edges)-1,edges)
h_new_mz = ROOT.TH1D("Z Invariant Mass, {}".format(new_lumi),";m_{z};Count",len(edges)-1,edges)

for j in range(0,len(edges)-1):
    try:
        h_mz.SetBinContent(j+1,count[j])
        h_new_mz.SetBinContent(j+1,count[j]*new_lumi/lumi0)
    except Exception as e:
        mp.red(e)

# ----- W TRANSVERSE MASS -----
w_transverse_mass = pd.Series(prods_w_selection["w_transverse_mass"]).to_numpy()
bins_w_mass = np.linspace(0,200,num=20)
count, edges = np.histogram(w_transverse_mass,bins=bins_w_mass)

h_mw = ROOT.TH1D("W Transverse Mass, {}".format(lumi0),";m_{T}^{W};Count",len(edges)-1,edges)
h_new_mw = ROOT.TH1D("W Transverse Mass, {}".format(new_lumi),";m_{T}^{W};Count",len(edges)-1,edges)
for j in range(0,len(edges)-1):
    try:
        h_mw.SetBinContent(j+1,count[j])
        h_new_mw.SetBinContent(j+1,count[j]*new_lumi/lumi0)
    except Exception as e:
        mp.red(e)
    
# ----- WZ TRANSVERSE MASS -----
wz_transverse_mass = pd.Series(prods_w_selection["wz_transverse_mass"]).to_numpy()
bins_wz_mass = np.linspace(0,350,num=35)
count, edges = np.histogram(wz_transverse_mass,bins=bins_wz_mass)

h_mwz = ROOT.TH1D("WZ Diboson System Transverse Mass, {}".format(lumi0),";m_{T}^{WZ};Count",len(edges)-1,edges)
h_new_mwz = ROOT.TH1D("WZ Diboson System Transverse Mass, {}".format(new_lumi),";m_{T}^{WZ};Count",len(edges)-1,edges)

for j in range(0,len(edges)-1):
    try:
        h_mwz.SetBinContent(j+1,count[j])
        h_new_mwz.SetBinContent(j+1,count[j]*new_lumi/lumi0)
    except Exception as e:
        mp.red(e)

 
        
        
        
        
        
        
        
        
# ---------- NEW TREES ----------

# ----- MAKING ARRAYS THE SAME SIZE -----
# Otherwise, there is an error.
l1 = len(z_mass)
l2 = len(w_transverse_mass)
ls = np.array([l1,l2])
lmax = ls.max()
lmin = ls.min()

for i in range(lmin-1,lmax-1):
    try:
        w_transverse_mass = np.append(w_transverse_mass,np.nan)
        wz_transverse_mass = np.append(wz_transverse_mass,np.nan)
    except Exception as e:
        mp.red(e)



# ----- WRITING -----
        
with uproot.recreate("SM_out_2.root") as f:
    f["high_lvl"] = uproot.newtree({
    "pT_z": np.float64,
    "z_mass":uproot.newbranch(np.float64),
    "w_mass":uproot.newbranch(np.float64),
    "wz_mass":uproot.newbranch(np.float64)
                                 })
    f["high_lvl"].extend({"pT_z":pT_z,"z_mass":z_mass,"w_mass":w_transverse_mass,"wz_mass":wz_transverse_mass})
    

    

        
        
        
        
        
# ------------ SAVING ROOT FILE --------------
mp.blue("Saving ROOT File")
outfile.cd()

h_pTz.Write()
h_new_pTz.Write()
h_mz.Write()
h_new_mz.Write()
h_mw.Write()
h_new_mw.Write()
h_mwz.Write()
h_new_mwz.Write()


mp.blue("Saving Complete")
