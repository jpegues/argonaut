###FILE: argonaut_constants.py
###PURPOSE: Script of constants for use by argonaut software package.
###DATE FILE CREATED: 2022-05-31
###DEVELOPERS: (Jamila Pegues; using Nautilus from Nautilus developers and radmc-3D from radmc developers)


##Import necessary modules
import os
import numpy as np
import astropy.constants as const
from datetime import datetime
pi = np.pi
#Set astronomical constants
c0 = const.c.value #[m/s]
h0 = const.h.value #[J*s]
au0 = const.au.value #[m]
ergtoJ0 = 1E-7 #erg -> Joule
eVtoJ0 = const.e.value #Convert eV to Joules


##Below Section: Sets constants for use by argonaut package
##Base:
filepath_base = os.path.join(os.path.expanduser("~"), "Documents/STScI_Fellowship/Research/")
filepath_modules = os.path.join(filepath_base, "Modules/")
filepath_structure = os.path.join(filepath_base, "Project_ModelsSpectrum/structure/")
filepath_chemistry = os.path.join(filepath_base, "Project_ModelsSpectrum/chemistry/")
filepath_testing = os.path.join(filepath_base, "Project_ModelsSpectrum/testing")
#
filepath_chemistry_crosssecs_UV = os.path.join(filepath_chemistry, "all_cross_sections_text_continuum/")
filepath_chemistry_crosssecs_X_parametric = os.path.join(filepath_base, "Datasets", "table_crosssecX_bethell2011.txt")
filepath_pnautilus = os.path.join(filepath_modules, "pnautilus_argoradiation/pnautilus_base/")
#
scheme_crosssec_X = "parametric"
#

##Benchmarks
dict_control_radfields_energyflux = {"solar":0.098, "ISRF":2.6E-6, "TWHya":10.31} #W/m^2
energyflux_toscale_photoratecalc_UV = dict_control_radfields_energyflux["ISRF"] #Or None for no new scaling
#
dist_photorates_Leiden = 1 *au0
#

##Structure:
structure_value_maxnumscaleH = 5
#

##Spectra:
waveloc_Lya = 121.567E-9 #Lya line center [m]
profile_crossLya_oscillatorstrength = 0.4162 #From Laursen+2007 near Equations 1 and 2
profile_crossLya_naturallinewidth_infreq = 9.936E7 #[Hz]; from Laursen+2007
gastemperature_Lya = 1000 #[K]; see Bethell+2011; for upper gas layer
wavelength_range_rates_UV = [91.2E-9, 200E-9] #Typical UV range assumed; max of ISRF, Tw-Hya fields from Leiden database
wavelength_range_rates_Lya = [waveloc_Lya-2.5E-9, waveloc_Lya+2.5E-9] #Lya range assumed
wavelength_range_rates_X = [(c0/(((20E3)*eVtoJ0)/h0)), (c0/(((0.1E3)*eVtoJ0)/h0))] #0.1 to 20keV; DIANA project data down to ~0.1keV
disk_radiation_fields = ["UVcont", "Lya", "X"]
#
profile_crossLya_numpoints_persect = 500 #500
profile_crossLya_waverange = [100E-9, 140E-9] #[0, 10E-7] #500
#

##Chemistry:
chemistry_value_muval0 = 2.37 #Anders+1989
#
chemistry_ionrateX_deltaenergy = (37 * eVtoJ0) #See Glassgold+1997; also https://www.epj-conferences.org/articles/epjconf/pdf/2015/21/epjconf_ppd2014_00015.pdf for Eq. 58
#
chemistry_flux_ISRF = 1.6E-3 * ergtoJ0 * (100**2) #erg/s/cm^2 -> J/s/m^2; also known as Habing field? #x-Also Draine field
chemistry_reactant_withphotochemUV = "Photon" #Name of reactant that reacts with main species in Pnautilus UV photochemistry (photoionization, photodestruction) reactions
chemistry_reactant_withphotochemX = "CRP" #Name of reactant that reacts with main species in Pnautilus high-energy (cosmic-ray, X-ray) photochemistry (photoionization, photodestruction) reactions
chemistry_reactant_withcosmicray = "CR" #Name of reactant that reacts with main species in Pnautilus high-energy (cosmic-ray, X-ray) photochemistry (photoionization, photodestruction) reactions
chemistry_product_withphotoion = "e-" #Name of product produced from reactions with main species in Pnautilus photochemistry (photoionization, photodestruction) reactions
#
chemistry_thres_mindensgas_cgs = (-np.inf) #1E5 #cm^-3; Minimum allowed gas density for running Pnautilus in a disk cell
chemistry_thres_maxUV_uDraine = np.inf #5E6 #1E7 #cm^-3; Maximum allowed UV flux for running Pnautilus in a disk cell
chemistry_molecule_titles = {"H2CO":r"H$_2$CO", "CCH":r"C$_2$H"}
#


##Chemistry reactions
chem_formula_ids_UV = [2]

##Empirical Relationships:
empirical_filename_RcvsRout = os.path.join(filepath_structure, "data_litlupusRc.txt")
empirical_constant_temperature_atmosphere_prefactor_K = 55.0
empirical_constant_temperature_atmosphere_exponent = 0.5
empirical_constant_temperature_atmosphere_meters = 200*au0
empirical_constant_temperature_zq0_scaler = 2
empirical_constant_temperature_delta = 2
structure_str_precision = "{0:13.6e}" #Based on radmc3D wavelength_micron.inp level of precision
structure_gradient_HatomicandHmolec_lowscaleH = 3
structure_gradient_HatomicandHmolec_highscaleH = 4
#

##Terminal Commands:
command_radmc3d_main = os.path.join(filepath_modules, "radmc3d/src/radmc3d")
command_radmc3d_tempdust = "mctherm"
command_radmc3d_radiation = "mcmono"
command_radmc3d_setthreads = "setthreads"
command_radmc3d_numthreads = str(1) #str(10)
#

##Cores/Processes:
process_maxtime_radmc3d = np.inf #30*60 #100 #Maximum time allowed before forced termination

##RADMC-3D:
#File names
radmc3d_filename_amrgrid = "amr_grid.inp"
radmc3d_filename_stars = "stars.inp"
radmc3d_filename_radmc3dinp = "radmc3d.inp"
radmc3d_filename_wavelength_dust = "wavelength_micron.inp"
radmc3d_filename_wavelength_spectrum = "mcmono_wavelength_micron.inp"
radmc3d_filename_dustdensity = "dust_density.inp"
radmc3d_filename_dustopac = "dustopac.inp"
radmc3d_filename_extsource = "external_source.inp"
radmc3d_filename_dustkapparoot = "dustkappa"
#
radmc3d_filename_tempdust = "dust_temperature.dat"
radmc3d_filename_radmc3dout = "radmc3d.out"
radmc3d_filename_field_radiation = "mean_intensity.out"
#
radmc3d_numformat_dustkappa = ".6E"
#
radmc3d_dict_isextsource = {"UVcont":True, "Lya":False, "X":False}
radmc3d_dict_whichdustdens = {"dust":["dust"], "UVcont":["dust"], "Lya":["dust", "Hatomic"], "X":["gas", "dust"]}
radmc3d_dict_whichdustopac = {"dust":["silicate"], "UVcont":["silicate"], "Lya":["silicate", "crossLya"], "X":["crossX_gas", "crossX_dust"]}
#
#Base values
radmc3d_threshold_minval = 1E-80 #1E-95 #General minimum value allowed for radmc3d inputs
#Photon values
radmc3d_value_nphottherm = 15000#00
radmc3d_value_nphotscatt = 2500#00
radmc3d_value_nphotmono = 2500#00
#

##PNAUTILUS:
#Commands
pnautilus_command_main = os.path.join(filepath_pnautilus, "pnautilus")
pnautilus_command_output = os.path.join(filepath_pnautilus, "pnautilus_outputs")
pnautilus_process_maxtime = 40*60 #Seconds
pnautilus_ext_timeout = "_timeout"
#
#File bases
dir_reactions_base = "dir_reaction_files"
dir_reactions_formatperslice = "dir_reaction_files_slice{0}"
pnautilus_filebase_disk = ["activation_energies.in", "element.in",
        "gas_reactions_orig.in", "parameters_strip.in",
        "gas_species.in", "grain_species.in", "surface_parameters.in"]
pnautilus_filename_element = "element.in"
pnautilus_filename_static1D = "1D_static.dat"
pnautilus_filename_parameter = "parameters.in"
pnautilus_filename_parameterstrip = "parameters_strip.in"
pnautilus_filename_speciesgas = "gas_species.in"
pnautilus_filename_speciesgrain = "grain_species.in"
pnautilus_filename_reactiongas_formatpercell = "gas_reactions_{0}.in"
pnautilus_filename_reactiongasorig = "gas_reactions_orig.in"
pnautilus_filename_reactiongasstrip = "gas_reactions_strip.in"
pnautilus_filename_reactiongrain_formatpercell = "grain_reactions_{0}.in"
pnautilus_filename_reactiongrainorig = "grain_reactions_orig.in"
pnautilus_filename_abundances = "abundances.in"
#
pnautilus_fileout_loadedreactionsgas = "table_loaded_reactions_gas.txt"
pnautilus_fileout_loadedreactionsgrain = "table_loaded_reactions_grain.txt"
pnautilus_lenchar_1Dstatic = 14 #Character length of columns in 1D_static.dat file
#
pnautilus_disk_dir_working = "disk_working"
pnautilus_disk_dir_done = "disk_done"
pnautilus_disk_dir_discard = "disk_discard"
pnautilus_processed_output = "processed_output"
#
pnautilus_header_reactiongas = "!***!    Reactants                    Products                                                  A          B          C      xxxxxxxxxxxxxxxxxxxxx ITYPE Tmin   Tmax formula ID xxxxx\n!***! Created {0}\n".format(datetime.today().strftime('%Y-%m-%d'))
pnautilus_header_1Dstatic = "! Distance [AU] ; H Gas density [part/cm^3] ; Gas Temperature [K] ; Visual Extinction [mag] ; Diffusion coefficient [cm^2/s] ; Dust Temperature [K] ; 1/abundance of grains ; AV/NH conversion factor ; radius of grains (cm); J.P. ADD: UV flux (uDraine), X-ray ionization rate (s^-1)\n"
#
#Text character spans within gas_reactions.in files
pnautilus_text_span_reactant = [0,34]
pnautilus_text_span_product = [34,90]
pnautilus_text_span_splitreactant = [[0,10], [11,21], [22,32]]
pnautilus_text_span_splitproduct = [[34,44], [45,55], [56,66], [67,77], [78,88]]
pnautilus_text_span_coeffs = [[90,100], [101,111], [112,122]]
pnautilus_text_span_itype = [145,148]
pnautilus_text_span_temp = [[149,155], [156,162]]
pnautilus_text_span_formula = [162,165]
pnautilus_text_span_id = [166,171]
pnautilus_text_span_indoflastchar = 176 #Index of last character in lines
pnautilus_text_len_reactanteach = 11
pnautilus_text_len_reactantall = 34
pnautilus_text_len_producteach = 11
pnautilus_text_len_productall = 55
pnautilus_text_len_coeffseach = 11
pnautilus_text_len_itype = 3
pnautilus_text_len_blank1 = 23
pnautilus_text_len_tempeach = 7
pnautilus_text_len_formula = 3
pnautilus_text_len_id = 6
pnautilus_text_len_lineall = 176 #Length of full string
#
pnautilus_threshold_coeff0 = 1E-99 #Reactions with alpha < threshold are set to this threshold
pnautilus_threshold_minval = 1E-99 #General minimum value allowed for pnautilus inputs
#
special_species_names = {"Co_theelement":"Co", "C2H":"CCH", "CH2CN":"H2CCN", "C2H5OH":"CH3CH2OH", "C2H5OH+":"CH3OCH3+", "CH3CH2OH+":"CH3OCH3+", "C2H5O":"CH3OCH2", "CH3NH":"CH2NH2", "CCH+":"C2H+", "CH3CHO+":"C2H4O+", "H2O2":"HOOH", "HO2":"O2H", "O2H+":"HO2+", "SH":"HS", "SH+":"HS+", "l-C4":"C4", "l-C4H":"C4H", "l-C5H":"C5H"}#, "HC3H":"t-C3H2"} #, "HC3H":"l-C3H2"
pnautilus_banned_reaction_species = ["C3H7OH", "CH3SH", "NaCl", "CS2", "CS2+", "K", "HC3H"]
#
pnautilus_species_ignore = ["GRAIN0", "GRAIN-", "XH", "space"]


rtol_element_conservation = 5E-3














#
