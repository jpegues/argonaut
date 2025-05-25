###FILE: argonaut_tests.py
###PURPOSE: Container for all tests run for the argonaut software package.
###FILE CREATED: 2022-06-13
###DEVELOPERS: (Jamila Pegues)
#
#python -m unittest argonaut_tests
#


##Import necessary modules
import unittest
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as scipy_odr
import scipy.integrate as scipy_integrate
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.interpolate import griddata as scipy_interptogrid
import astropy.constants as const
import argonaut
import argonaut_math as math
import argonaut_rates as rates
import argonaut_utils as utils
import argonaut_constants as preset
pi = np.pi
np.random.seed(20)
#Set astronomical constants
muval0 = preset.chemistry_value_muval0
c0 = const.c.value #[m/s]
e0 = const.e.value #[C?]
h0 = const.h.value #[J*s]
G0 = const.G.value #[kg * m/s^2]
au0 = const.au.value #[m]
pc0 = const.pc.value #[m]
eps0 = const.eps0.value #Vacuum electric permittivity
k_B0 = const.k_B.value #[W/K]
me0 = const.m_e.value #Mass of electron; [kg]
mH0 = (const.m_p.value + me0) #Mass of H atom; [kg]
Lsun0 = const.L_sun.value #[J]
Msun0 = const.M_sun.value #[kg]
sigma_SB0 = const.sigma_sb.value #[W/(m^2 * K^4)]
sigma_T0 = const.sigma_T.value #[m^2]
#
filepath_testing = preset.filepath_testing
print("Loading test dictionary of UV cross-sections... Takes a few minutes...")
test_dict_mol0 = utils._load_data_crosssecs_UV(
	do_skip_elements=False,  do_ignore_JK=True, possible_elements=None,
	do_skip_bannedspecies=False,  wavelength_min=91.2E-9,
	wavelength_max=200E-9, which_mol=None, do_verbose=False
)
print("Dictionary of UV cross-sections loaded.")
#



#-------------------------------------------------------------------------------
#Setup: Helper Functions
#Helper function: Fetches all species from given files
def _helper_fetch_all_species(filename_gas, filename_grain, do_remove_phase_marker):
	#Load species files
	data_gas = np.genfromtxt(filename_gas, dtype=str)
	data_grain = np.genfromtxt(filename_grain, dtype=str)
	#Load species data
	if do_remove_phase_marker: #Remove case markers if requested
		act_gas = np.unique([
			re.sub("^(J|K)","",item) for item in data_gas[:,0]
		])
		act_grain = np.unique([
			re.sub("^(J|K)","",item) for item in data_grain[:,0]
		])
		act_all = np.unique(np.concatenate((act_gas, act_grain)))
	else: #Otherwise keep phase markers
		act_gas = data_gas[:,0]
		act_grain = data_grain[:,0]
		act_all = np.unique(np.concatenate((act_gas, act_grain)))

	#Return the extracted species
	return {"gas":act_gas, "grain":act_grain, "all":act_all}
#
#Helper function: Fetches all updatable UV reactions from dictionary
def _helper_fetch_updatable_reactions_UV(dict_orig, dict_mol):
	#Generate dictionary of only reactions with UV cross-sections
	dict_new = {}
	for curr_key in dict_orig:
		curr_lookup = dict_orig[curr_key]["reactant_main"]
		is_photoion = False
		is_photodiss = False
		if (curr_lookup in dict_mol):
			is_photoion = (
				(dict_orig[curr_key]["phototype"] == "photoion")
				and (dict_mol[curr_lookup]["wave_photoion_UV"] is not None)
			)
			is_photodiss = (
				(dict_orig[curr_key]["phototype"] == "photodiss")
				and (dict_mol[curr_lookup]["wave_photodiss_UV"] is not None)
			)
		if is_photoion:
			dict_new[curr_key] = {}
			dict_new[curr_key]["wave"] = (
				dict_mol[curr_lookup]["wave_photoion_UV"]
			)
			dict_new[curr_key]["cross"] = (
				dict_mol[curr_lookup]["cross_photoion_UV"]
			)
		elif is_photodiss:
			dict_new[curr_key] = {}
			dict_new[curr_key]["wave"] = (
				dict_mol[curr_lookup]["wave_photodiss_UV"]
			)
			dict_new[curr_key]["cross"] = (
				dict_mol[curr_lookup]["cross_photodiss_UV"]
			)
		else:
			pass
	#Return the subset dictionary
	return dict_new
#
#Helper function: Converts between string and dictionary reaction versions
def _helper_conv_reaction_stranddict(dict_orig, str_orig, which_conv):
	#Convert from dictionary to string
	if (which_conv == "dict_to_str"):
		#Extract information
		reactants = dict_orig["reactants"]
		products = dict_orig["products"]
		coeffs = dict_orig["coeffs"]
		itype = dict_orig["itype"]
		Tmin = dict_orig["Tmin"]
		Tmax = dict_orig["Tmax"]
		formula = dict_orig["formula"]
		orig_id = dict_orig["react_id"]

		#Generate string
		# Reactants
		str_reaction = ""
		for ii in range(0, len(reactants)):
			str_reaction += (reactants[ii].ljust(11))
		str_reaction = str_reaction.ljust(34)
		# Products
		tmp_str = ""
		for ii in range(0, len(products)):
			tmp_str += (products[ii].ljust(11))
		tmp_str = tmp_str.ljust(55)
		str_reaction += tmp_str
		# Coefficients
		for ii in range(0, len(coeffs)):
			tmp_str = f"{coeffs[ii]:.3e}".rjust(11)
			str_reaction += tmp_str
		# Blocked space
		str_reaction += (" "*23)
		# Itype
		str_reaction += f"{int(itype)}".rjust(3)
		# Temperature
		tmp_str = f"{Tmin}  {coeffs[1]:.3e}  {coeffs[2]:.3e} "
		str_reaction += (f"{Tmin}".rjust(7) + f"{Tmax}".rjust(7))
		# Formula and ID
		str_reaction += (f"{formula}".rjust(3) + f"{orig_id}".rjust(6))
		str_reaction = str_reaction.ljust(176)

		#Return the string
		return str_reaction
	#
	#Convert from string to dictionary
	elif (which_conv == "str_to_dict"):
		ind0 = 0
		ind1 = ind0 + 34
		ind2 = ind1 + 57
		str_right = str_orig[ind2:].split()
		#Build dictionary for current reaction
		dict_reaction = {}
		dict_reaction["reactants"] = str_orig[ind0:ind1].split()
		curr_react_low = [
			item.lower() for item in dict_reaction["reactants"]
		]
		dict_reaction["products"] = str_orig[ind1:ind2].split()
		curr_prod_low = [
			item.lower() for item in dict_reaction["products"]
		]
		dict_reaction["phototype"] = None
		is_photo = any([
			(item in curr_react_low) for item in ["photon", "crp"]
		])
		if is_photo:
			if ("e-" in curr_prod_low):
				dict_reaction["phototype"] = "photoion"
			else:
				dict_reaction["phototype"] = "photodiss"
		dict_reaction["reactant_main"] = None
		if (dict_reaction["phototype"] is not None):
			dict_reaction["reactant_main"] = (
				sorted([
					item for item in dict_reaction["reactants"]
					if (item.lower() not in ["photon", "crp", "cr"])
				])[0]
			)
		dict_reaction["coeffs"] = np.asarray(str_right[0:3]).astype(float)
		dict_reaction["prob_product"] = 1
		dict_reaction["itype"] = str_right[6]
		dict_reaction["Tmin"] = int(str_right[7])
		dict_reaction["Tmax"] = int(str_right[8])
		dict_reaction["formula"] = int(str_right[9])
		dict_reaction["react_id"] = int(str_right[10])
		dict_reaction["reaction"] = ("["+
			" + ".join(sorted(dict_reaction["reactants"]))
			+" -> "
			+" + ".join(sorted(dict_reaction["products"]))
			+"]"
		)
		#Return the dictionary
		return dict_reaction
	#
	#Otherwise throw error if unrecognized conversion scheme
	else:
		raise ValueError(f"Err: Unrecognized mode {which_conv}.")
	#
#
#Helper function: Fetches all reactions from given files
def _helper_dict_reaction(filename_input_gas, filename_input_grain, do_probability, do_keep_noprobability=False):
	#Load the data
	dict_data = {}
	with open(filename_input_gas,'r') as openfile:
		dict_data["reactions_gas"] = openfile.readlines()
	with open(filename_input_grain,'r') as openfile:
		dict_data["reactions_grain"] = openfile.readlines()
	#
	#Extract reactions
	dict_reaction = {"reactions_gas":{}, "reactions_grain":{}}
	i_counter = 0
	for curr_key in ["reactions_gas", "reactions_grain"]:
		curr_data = dict_data[curr_key]
		for ii in range(0, len(curr_data)):
			curr_row = curr_data[ii]
			if (curr_row.strip().startswith("!")):
				continue
			curr_dict = _helper_conv_reaction_stranddict(
				dict_orig=None, str_orig=curr_row, which_conv="str_to_dict"
			)

			#Store dictionary
			curr_store = i_counter
			dict_reaction[curr_key][f"#{curr_store}"] = curr_dict

			#Increment counter
			i_counter += 1
		#
	#
	# Set branching ratios, if requested
	if do_probability:
		#Record original reaction dictionaries if requested
		if do_keep_noprobability:
			dict_reaction["reactions_gas_orig"] = (
				dict_reaction["reactions_gas"].copy()
			)
			dict_reaction["reactions_grain_orig"] = (
				dict_reaction["reactions_grain"].copy()
			)
		#
		#Calculate probabilities
		is_checked = []
		for curr_key1 in ["reactions_gas", "reactions_grain"]:
			for curr_key2 in dict_reaction[curr_key1]:
				#Extract information
				curr_dict = dict_reaction[curr_key1][curr_key2]
				curr_reacts = curr_dict["reactants"]
				#Skip if not relevant reaction
				if (int(curr_dict["itype"]) not in [3]):
					continue
				#Skip if checked already
				if (curr_reacts in is_checked):
					continue
				#Fetch all ids with these reactants
				curr_ids = [
					key for key in dict_reaction[curr_key1]
					if (dict_reaction[curr_key1][key]["reactants"]==curr_reacts)
				]
				#Estimate branching ratios
				curr_coeffs0 = np.array([
					dict_reaction[curr_key1][key]["coeffs"][0]
					for key in curr_ids
				])
				curr_ratios = curr_coeffs0 / np.sum(curr_coeffs0)
				#Store branching ratios
				for ii in range(0, len(curr_ids)):
					dict_reaction[curr_key1][curr_ids[ii]]["probability"] = (
						curr_ratios[ii]
					)
				#Mark reactants as complete
				is_checked.append(curr_reacts)
		#
	#
	#Prepare other correct dictionary items
	dict_reaction["max_reaction_id"] = np.max([
		np.max([(dict_reaction["reactions_gas"][key]["react_id"])
		for key in dict_reaction["reactions_gas"]]),
		np.max([(dict_reaction["reactions_grain"][key]["react_id"])
		for key in dict_reaction["reactions_grain"]])
	])
	dict_reaction["reactions_UV"] = {
		key:dict_reaction["reactions_gas"][key]
		for key in dict_reaction["reactions_gas"]
		if (int(dict_reaction["reactions_gas"][key]["itype"]) in [3])
	}
	#
	#Return the dictionary
	return dict_reaction
#
#Helper function: Update individual reaction
def _helper_update_reaction_dict_indiv(arr_x_spec, arr_y_spec, dict_molinfo, react_orig, minval_coeff0):
	#Determine if reaction should be updated
	itype = react_orig["itype"]
	phototype = react_orig["phototype"]
	arr_x_cross = dict_molinfo["wave"]
	arr_y_cross = dict_molinfo["cross"]

	#Compute new photorate
	x_min = np.max([np.min(arr_x_spec), np.min(arr_x_cross)])
	x_max = np.min([np.max(arr_x_spec), np.max(arr_x_cross)])
	x_unified = np.sort(np.unique(np.concatenate((
		arr_x_spec[((arr_x_spec <= x_max) & (arr_x_spec >= x_min))],
		arr_x_cross[((arr_x_cross <= x_max) & (arr_x_cross >= x_min))]
	))))
	arr_spec_unified = scipy_interp1d(x=arr_x_spec, y=arr_y_spec)(x_unified)
	arr_cross_unified = scipy_interp1d(x=arr_x_cross, y=arr_y_cross)(x_unified)
	new_rate = np.trapz(x=x_unified, y=(arr_spec_unified*arr_cross_unified))

	#Copy over reaction dictionary
	react_new = react_orig.copy()

	#Store new coefficients in updated reaction dictionary
	prob = react_orig["probability"]
	new_coeffs = [(new_rate*prob), 0, 0]
	react_new["coeffs"] = new_coeffs

	#Return the updated reaction dictionary
	return react_new
#



#-------------------------------------------------------------------------------
#Class: Test_Math
#Purpose: Testing the functions in argonaut_math
class Test_Math(unittest.TestCase):
	print("\n> Running test suite for argonaut math functions.")
	##Test Planck function calculations
	def test_calc_bbody(self):
		print("\n> Running test: test_calc_bbody.")
		#Prepare input data
		num_points = 100
		arr_freq = np.random.uniform(0, 1000, size=num_points)
		arr_temp = np.random.uniform(0, 1000, size=num_points)
		#For spectrum per frequency
		# Prepare correct results
		arr_act = np.array([
			(2*h0*(arr_freq**3)/(c0**2))
			/ (np.exp(h0*arr_freq/(k_B0*arr_temp)) - 1)
			* (4*pi) #[W/m^2/Hz/str] -> [W/m^2/Hz]
		])
		# Calculate using package
		arr_calc = math._calc_bbody(
			temp=arr_temp, freq=arr_freq, wave=None,
			which_mode="spectrum", which_per="frequency"
		)
		# Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For spectrum per wave
		# Prepare correct results
		arr_act = np.array([
			(2*h0*(arr_freq**3)/(c0**2))
			/ (np.exp(h0*arr_freq/(k_B0*arr_temp)) - 1)
			* (4*pi) #[W/m^2/Hz/str] -> [W/m^2/Hz]
		]) * arr_freq / (c0 / arr_freq) #[per freq] -> [per wave]
		# Calculate using package
		arr_calc = math._calc_bbody(
			temp=arr_temp, freq=None, wave=(c0/arr_freq),
			which_mode="spectrum", which_per="wavelength"
		)
		# Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For intensity per frequency
		# Prepare correct results
		arr_act = np.array([
			(2*h0*(arr_freq**3)/(c0**2))
			/ (np.exp(h0*arr_freq/(k_B0*arr_temp)) - 1)
		]) #[W/m^2/Hz/str]
		# Calculate using package
		arr_calc = math._calc_bbody(
			temp=arr_temp, freq=arr_freq, wave=None,
			which_mode="spectrum", which_per="frequency"
		)
		# Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For intensity per wave
		# Prepare correct results
		arr_act = np.array([
			(2*h0*(arr_freq**3)/(c0**2))
			/ (np.exp(h0*arr_freq/(k_B0*arr_temp)) - 1)
		]) * arr_freq / (c0 / arr_freq) #[per freq] -> [per wave]
		# Calculate using package
		arr_calc = math._calc_bbody(
			temp=arr_temp, freq=None, wave=(c0/arr_freq),
			which_mode="spectrum", which_per="wavelength"
		)
		# Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_calc_bbody.")
	#
	##Test distance calculations
	def test_calc_cellcenters(self):
		print("\n> Running test: test_calc_cellcenters.")
		#Prepare input data
		num_points = 100
		arr_x = np.random.uniform(-100, 100, size=num_points)
		#Prepare correct results
		arr_act = np.array([
			((arr_x[ind]+arr_x[ind+1])/2)
			for ind in range(0, (len(arr_x)-1))
		])
		#Calculate using package
		arr_calc = math._calc_cellcenters(arr=arr_x)

		#Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		print("> Test complete: test_calc_cellcenters.")
	#
	##Test analytic surface density calculations
	def test_calc_densitysurf_analytic(self):
		print("\n> Running test: test_calc_densitysurf_analytic.")
		#Prepare input data
		num_s1 = 200
		num_s2 = 100
		matr_r = np.random.uniform(size=(num_s1,num_s2)) *1000
		norm_r = np.random.uniform(size=(num_s1,num_s2)) *1000
		mdiskgas = np.random.uniform(size=(num_s1,num_s2)) *1000

		#Prepare correct results
		matr_act = (
			(mdiskgas * (2-1) / (2*pi*(norm_r*norm_r)))
			* ((matr_r / norm_r)**-1)
			* np.exp(-((matr_r / norm_r)**(2-1)))
		)
		#Calculate using package
		matr_calc = math.calc_densitysurf_analytic(
			matr_x=matr_r, norm_x=norm_r, mdiskgas=mdiskgas
		)
		#Check that calculations are correct
		self.assertTrue(np.allclose(matr_act, matr_calc))
		print("> Test complete: test_calc_densitysurf_analytic.")
	#
	##Test 2D analytic density calculations
	def test_calc_density2D_analytic(self):
		print("\n> Running test: test_calc_density2D_analytic.")
		#Prepare input data
		num_s1 = 200
		num_s2 = 100
		num_s3 = 5
		matr_r = np.random.uniform(size=(num_s1,num_s2)) *1000
		matr_z = np.random.uniform(size=(num_s1,num_s2)) *1000
		norm_r = np.random.uniform(size=(num_s1,num_s2)) *1000
		mdiskgas = np.random.uniform(size=(num_s1,num_s2)) *1000
		arr_scaleH = np.random.uniform(size=num_s2) *1000
		matr_scaleH = arr_scaleH[np.newaxis,:]
		frac_dustovergasmass = np.random.uniform(size=(num_s1,num_s2))
		fracs_dustmdisk = np.random.uniform(size=(num_s3,num_s1,num_s2))
		fracs_dustscaleH = np.random.uniform(size=(num_s3,num_s1,num_s2))
		# User-set constants
		thres_lowscaleH = preset.structure_gradient_HatomicandHmolec_lowscaleH
		thres_highscaleH = preset.structure_gradient_HatomicandHmolec_highscaleH
		arr_lowscaleH = thres_lowscaleH*arr_scaleH
		arr_highscaleH = thres_highscaleH*arr_scaleH
		matr_lowscaleH = arr_lowscaleH[np.newaxis,:]
		matr_highscaleH = arr_highscaleH[np.newaxis,:]

		#Prepare correct results
		# Surface densities
		matr_sigma_gas = (
			(mdiskgas * (2-1) / (2*pi*(norm_r*norm_r)))
			* ((matr_r / norm_r)**(-1))
			* np.exp(-((matr_r / norm_r)**(2-1)))
		)
		matr_sigma_dust = matr_sigma_gas * frac_dustovergasmass
		# Hydrogen density
		matr_nH = (
			(1/np.sqrt(2*pi))
			* (matr_sigma_gas/matr_scaleH)
			* np.exp((-0.5*((matr_z/matr_scaleH)**2)))
		) / mH0
		# Atomic vs molecular hydrogen density
		matr_gradient = (
			(matr_z - matr_lowscaleH)
			/(matr_highscaleH - matr_lowscaleH)
		)
		matr_gradient[matr_z <= matr_lowscaleH] = 0
		matr_gradient[matr_z >= matr_highscaleH] = 1
		matr_nH_Hatomiconly = (matr_nH * matr_gradient)
		matr_nH_H2only = (matr_nH * (1 - matr_gradient))/2
		# Dust densities
		matr_dust_pergr = np.asarray([
			(fracs_dustmdisk[ii,:,:] * matr_sigma_dust)
			/ (np.sqrt(2*pi) * fracs_dustscaleH[ii,:,:] * matr_scaleH)
			* np.exp(
				(-1/2.0)
				*((matr_z/(fracs_dustscaleH[ii,:,:]*matr_scaleH))**2)
			)
			for ii in range(0, num_s3)
		])
		matr_dust_all = np.sum(matr_dust_pergr, axis=0)
		# Assemble into dictionary
		dict_act = {
			"volnumdens_nH":matr_nH, "volnumdens_nH_H2only":matr_nH_H2only,
			"volnumdens_nH_Hatomiconly":matr_nH_Hatomiconly,
			"volmassdens_dust_all":matr_dust_all,
			"volmassdens_dust_pergr":matr_dust_pergr, "matr_scaleH":matr_scaleH,
			"matr_sigma_dust":matr_sigma_dust,
			"matr_sigma_gas":matr_sigma_gas
		}
		#Calculate using package
		dict_calc = math.calc_density2D_analytic(
			matr_x=matr_r, matr_y=matr_z, norm_x=norm_r,
			mdiskgas=mdiskgas, matr_scaleH=matr_scaleH,
			frac_dustovergasmass=frac_dustovergasmass,
			fracs_dustmdisk=fracs_dustmdisk, fracs_dustscaleH=fracs_dustscaleH
		)
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key in dict_act:
			self.assertTrue(np.allclose(dict_act[curr_key], dict_calc[curr_key]))
		print("> Test complete: test_calc_density2D_analytic.")
	#
	##Test 2D analytic atomic and molecular hydrogen density calculations
	def test_calc_density2D_Hatomicandmolec(self):
		print("\n> Running test: test_calc_density2D_Hatomicandmolec.")
		#Prepare input data
		num_s1 = 200
		num_s2 = 100
		num_s3 = 5
		matr_z = np.random.uniform(size=(num_s1,num_s2)) *1000
		matr_nH = np.random.uniform(size=(num_s1,num_s2)) *1000
		arr_scaleH = np.random.uniform(size=num_s2) *1000
		matr_scaleH = arr_scaleH[np.newaxis,:]
		# User-set constants
		thres_lowscaleH = preset.structure_gradient_HatomicandHmolec_lowscaleH
		thres_highscaleH = preset.structure_gradient_HatomicandHmolec_highscaleH
		arr_lowscaleH = thres_lowscaleH*arr_scaleH
		arr_highscaleH = thres_highscaleH*arr_scaleH
		matr_lowscaleH = arr_lowscaleH[np.newaxis,:]
		matr_highscaleH = arr_highscaleH[np.newaxis,:]

		#Prepare correct results
		matr_gradient = (
			(matr_z - matr_lowscaleH)
			/(matr_highscaleH - matr_lowscaleH)
		)
		matr_gradient[matr_z <= matr_lowscaleH] = 0
		matr_gradient[matr_z >= matr_highscaleH] = 1
		matr_nHatomic = (matr_nH * matr_gradient)
		matr_nH2 = (matr_nH * (1 - matr_gradient))/2
		# Assemble into dictionary
		dict_act = {"nH2":matr_nH2, "nHatomic":matr_nHatomic}
		#Calculate using package
		dict_calc = math.calc_density2D_Hatomicandmolec(
			nHtotal=matr_nH, matr_y=matr_z,
			arr_yval_low=arr_lowscaleH, arr_yval_high=arr_highscaleH
		)
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key in dict_act:
			self.assertTrue(np.allclose(dict_act[curr_key], dict_calc[curr_key]))
		print("> Test complete: test_calc_density2D_Hatomicandmolec.")
	#
	##Test distance calculations
	def test_calc_distance(self):
		print("\n> Running test: test_calc_distance.")
		##Prepare input data and correct results
		num_points = 100
		scaler = 100
		arr_x1 = np.random.uniform(size=(num_points,2)) * scaler
		arr_x2 = np.random.uniform(size=(num_points,2)) * scaler
		arr_act = np.asarray([
			np.sqrt(
				((arr_x1[ind,0] - arr_x2[ind,0])**2)
				+ ((arr_x1[ind,1] - arr_x2[ind,1])**2)
			)
			for ind in range(0, num_points)
		])
		arr_calc = np.asarray([
			math.calc_distance(p1=arr_x1[ind,:], p2=arr_x2[ind,:], order=2)
			for ind in range(0, num_points)
		])

		#Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		print("> Test complete: test_calc_distance.")
	#
	##Test midplane temperature calculations
	def test_calc_temperature_midplane(self):
		print("\n> Running test: test_calc_temperature_midplane.")
		##Prepare input data and correct results
		num_points = 100
		arr_x = np.random.uniform(0, 1000, size=num_points) *au0
		arr_flareangle = np.random.uniform(size=num_points)
		arr_lstar = np.random.uniform(0, 1000, size=num_points) *Lsun0
		arr_act = (
			(arr_flareangle * arr_lstar)
			/ (sigma_SB0 * 8 * pi * (arr_x*arr_x))
		)**0.25
		arr_calc = math.calc_temperature_midplane(
			arr_x=arr_x, theta_flare=arr_flareangle, lstar=arr_lstar
		)

		#Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		print("> Test complete: test_calc_temperature_midplane.")
	#
	##Test midplane temperature radius calculations
	def test_calc_photorate_UV(self):
		print("\n> Running test: test_calc_photorate_UV.")
		#Prepare input data
		num_points = 1000
		arr_x = np.sort(np.random.uniform(size=num_points) *1000)
		arr_spec = np.random.uniform(size=num_points) *1000
		arr_cross = np.random.uniform(size=num_points) *1000
		arr_theta = np.random.uniform(size=num_points) *1000
		#Prepare correct results
		arr_rate = np.trapz(x=arr_x, y=(arr_spec*arr_cross*arr_theta))
		arr_accum = scipy_integrate.cumulative_trapezoid(
			x=arr_x, y=(arr_spec*arr_cross*arr_theta)
		)
		#Calculate using package
		# For rates
		arr_calc = math._calc_photorate_UV(
			y_spec=arr_spec, y_cross=arr_cross, y_prod_theta=arr_theta,
			x_unified=arr_x, axis=None, do_return_accum=False
		)
		self.assertTrue(np.allclose(arr_rate, arr_calc))
		# For rates
		arr_calc = math._calc_photorate_UV(
			y_spec=arr_spec, y_cross=arr_cross, y_prod_theta=arr_theta,
			x_unified=arr_x, axis=None, do_return_accum=True
		)
		self.assertTrue(np.allclose(arr_accum, arr_calc))
		print("> Test complete: test_calc_photorate_UV.")
	#
	##Test primary X-ray ionization rate calculations
	def test_calc_photorate_X_primary(self):
		print("\n> Running test: test_calc_photorate_X_primary.")
		#Prepare input data
		num_p1 = 1000
		num_p2 = 500
		num_p3 = 20
		arr_wave = np.sort(np.random.uniform(size=num_p1) *1000)
		arr_specwave = np.random.uniform(size=(num_p3,num_p2,num_p1)) *1000
		arr_crosswave = np.random.uniform(size=num_p1) *1000
		val_deltaenergy = 5.928E-18 #37 eV in Joules; from e.g. Glassgold+1997
		#Prepare correct results
		arr_energy = (c0 * h0 / arr_wave)
		arr_specenergy = (arr_specwave * arr_wave / arr_energy)
		arr_act = -1*np.trapz( #Negative flip applied since descending energy
			x=arr_energy,
			y=(arr_specenergy*arr_crosswave*(arr_energy/val_deltaenergy)),
			axis=-1
		) #-> ([photon/s/m^2/J] * (J/J) * m^2 * dJ) = [photon/s]
		#Calculate using package
		arr_calc = math._calc_photorate_X_primary(
			x_unified=arr_wave, y_spec_photon_perm=arr_specwave,
			y_cross_inm=arr_crosswave, delta_energy=val_deltaenergy, axis=-1
		)
		#Check calculations
		self.assertTrue(np.allclose(arr_act, arr_calc))
		print("> Test complete: test_calc_photorate_X_primary.")
	#
	##Test midplane temperature radius calculations
	def test_calc_radius_of_temperature_midplane(self):
		print("\n> Running test: test_calc_radius_of_temperature_midplane.")
		##Prepare input data and correct results
		num_points = 100
		arr_x = np.random.uniform(0, 1000, size=num_points) *au0
		arr_flareangle = np.random.uniform(size=num_points)
		arr_lstar = np.random.uniform(0, 1000, size=num_points) *Lsun0
		arr_act = np.sqrt(
			(arr_flareangle * arr_lstar)
			/ (sigma_SB0 * 8 * pi * (arr_x**4))
		)
		arr_calc = math.calc_radius_of_temperature_midplane(
			temp=arr_x, theta_flare=arr_flareangle, lstar=arr_lstar
		)

		#Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		print("> Test complete: test_calc_radius_of_temperature_midplane.")
	#
	##Test slope calculations
	def test_calc_slope(self):
		print("\n> Running test: test_calc_slope.")
		##Prepare input data and correct results
		num_points = 100
		scaler = 100
		arr_x1 = np.random.uniform(size=(num_points,2)) * scaler
		arr_x2 = np.random.uniform(size=(num_points,2)) * scaler
		arr_y1 = np.random.uniform(size=(num_points,2)) * scaler
		arr_y2 = np.random.uniform(size=(num_points,2)) * scaler
		arr_act = (arr_y2 - arr_y1) / (arr_x2 - arr_x1)
		arr_calc = math._calc_slope(
			x_1=arr_x1, x_2=arr_x2, y_1=arr_y1, y_2=arr_y2
		)

		#Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		print("> Test complete: test_calc_slope.")
	#
	##Test scale height calculations
	def test_calc_scaleheight(self):
		print("\n> Running test: test_calc_scaleheight.")
		##Prepare input data and correct results
		num_points = 100
		val_mu = 2.37
		arr_x = np.random.uniform(size=num_points) * 500 * au0
		arr_lstar = np.random.uniform(size=num_points) * 10 * Lsun0
		arr_mstar = np.random.uniform(size=num_points) * 10 * Msun0
		arr_temp_mid = np.random.uniform(size=num_points) * 50
		#
		arr_act = (
			np.sqrt((k_B0 * arr_temp_mid)/(val_mu * mH0))
			/ np.sqrt((G0*arr_mstar)/(arr_x**3))
		)
		arr_calc = math.calc_scaleheight(
			arr_x=arr_x, mstar=arr_mstar, temp_mid=arr_temp_mid,
			lstar=arr_lstar
		)

		#Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		print("> Test complete: test_calc_scaleheight.")
	#
	##Test Thomson scattering
	def test_calc_scatt_Thomson(self):
		print("\n> Running test: test_calc_scatt_Thomson.")
		#Prepare input data
		arr_act = [sigma_T0]
		arr_calc = [math._calc_scatt_Thomson(q=e0, m=me0)]

		#Check that calculations are correct
		self.assertTrue(np.allclose(arr_act, arr_calc))
		print("> Test complete: test_calc_scatt_Thomson.")
	#
	##Test synthetic parametric gas temperature calculations
	def test_calc_tempgas_parametric_synthetic(self):
		print("\n> Running test: test_calc_tempgas_parametric_synthetic.")
		#Prepare input data
		num_p1 = 100
		num_p2 = 80
		matr_r = np.random.uniform(size=(num_p1,num_p2)) * 1000
		matr_z = np.random.uniform(size=(num_p1,num_p2)) * 1000
		matr_tempdust = np.random.uniform(size=(num_p1,num_p2)) * 1000
		matr_lstar = np.random.uniform(size=(num_p1,num_p2)) * 1000
		matr_mstar = np.random.uniform(size=(num_p1,num_p2)) * 1000
		matr_theta_flare = np.random.uniform(size=(num_p1,num_p2)) * 1000
		#
		tempatm_T0 = preset.empirical_constant_temperature_atmosphere_prefactor_K
		tempatm_q0 = preset.empirical_constant_temperature_atmosphere_exponent
		tempatm_r0 = preset.empirical_constant_temperature_atmosphere_meters
		zq_scaler = preset.empirical_constant_temperature_zq0_scaler
		val_delta = preset.empirical_constant_temperature_delta
		val_mu = preset.chemistry_value_muval0
		#
		#Prepare correct results
		matr_tempatm = tempatm_T0*((matr_r/tempatm_r0)**(-tempatm_q0))
		matr_tempmid = (
			(matr_theta_flare * matr_lstar)
			/ (sigma_SB0 * 8 * pi * (matr_r*matr_r))
		)**0.25
		matr_scaleH = (
			np.sqrt((k_B0 * matr_tempmid)/(val_mu * mH0))
			/ np.sqrt((G0*matr_mstar)/(matr_r**3))
		)
		zq0 = (zq_scaler * matr_scaleH)
		matr_act = np.ones(shape=(num_p1,num_p2))*np.nan
		matr_act[matr_z < zq0] = (
			matr_tempatm[matr_z < zq0]
			+ ((matr_tempmid[matr_z < zq0] - matr_tempatm[matr_z < zq0])
				*((np.cos(pi*matr_z[matr_z < zq0]/(2*zq0[matr_z < zq0]))
				)**(2*val_delta))
			)
		)
		matr_act[matr_z >= zq0] = matr_tempatm[matr_z >= zq0]
		matr_act[matr_act <= matr_tempdust] = (
			matr_tempdust[matr_act <= matr_tempdust]
		)
		#Calculate using package
		matr_calc = math.calc_tempgas_parametric_synthetic(
			matr_tempdust=matr_tempdust, matr_r=matr_r, matr_z=matr_z,
			lstar=matr_lstar, mstar=matr_mstar, theta_flare=matr_theta_flare
		)
		#Check calculations are correct
		self.assertTrue(np.allclose(matr_act, matr_calc))
		#
		print("> Test complete: test_calc_tempgas_parametric_synthetic.")
	#
	##Test flux distance conversions
	def test_conv_distanddist(self):
		print("\n> Running test: test_conv_distanddist.")
		##Prepare input data and correct results
		num_points = 100
		arr_x = np.random.uniform(size=num_points) * 1000
		arr_distold = np.random.uniform(size=num_points) * 1000 * pc0
		arr_distnew = np.random.uniform(size=num_points) * 1000 * pc0
		#
		arr_act = (arr_x * ((arr_distold/arr_distnew)**2))
		arr_calc = math._conv_distanddist(
			flux=arr_x, dist_old=arr_distold, dist_new=arr_distnew
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_conv_distanddist.")
	#
	##Test luminosity and flux conversions
	def test_conv_lumandflux(self):
		print("\n> Running test: test_conv_lumandflux.")
		##Prepare input data and correct results
		num_points = 100
		arr_x = np.random.uniform(size=num_points) * 1000 * Lsun0
		arr_dist = np.random.uniform(size=num_points) * 1000 * pc0
		#
		#For luminosity -> flux
		arr_act = (arr_x / (4*pi*(arr_dist*arr_dist)))
		arr_calc = math._conv_lumandflux(
			which_conv="lumtoflux", lum=arr_x, dist=arr_dist
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For flux -> luminosity
		arr_act = (arr_x * 4 * pi * (arr_dist*arr_dist))
		arr_calc = math._conv_lumandflux(
			which_conv="fluxtolum", flux=arr_x, dist=arr_dist
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_conv_lumandflux.")
	#
	##Test luminosity and effective temperature conversions
	def test_conv_lumandTeff(self):
		print("\n> Running test: test_conv_lumandTeff.")
		##Prepare input data and correct results
		num_points = 100
		arr_lum = np.random.uniform(size=num_points) * 1000 * Lsun0
		arr_x = np.random.uniform(size=num_points) * 1000 * pc0
		#
		#For luminosity -> effective temperature
		arr_act = (sigma_SB0 * (arr_lum**4) * (4 * pi * (arr_x*arr_x)))
		arr_act = (arr_lum / (sigma_SB0 * (4 * pi * (arr_x*arr_x))))**(0.25)
		arr_calc = math._conv_lumandTeff(
			dist=arr_x, which_conv="lum_to_Teff", lstar=arr_lum
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_conv_lumandTeff.")
	#
	##Test spectrum conversions
	def test_conv_spectrum(self):
		print("\n> Running test: test_conv_spectrum.")
		##Prepare input data and correct results
		num_points = 100
		arr_spec = np.random.uniform(size=num_points) * 1000
		arr_wave = np.random.uniform(size=num_points) * 1000
		arr_freq = (c0 / arr_wave)
		#
		#For per wave -> per freq
		arr_act = arr_spec * arr_wave / arr_freq
		# For given wave
		arr_calc = math._conv_spectrum(
			which_conv="perwave_to_perfreq", spectrum=arr_spec, wave=arr_wave
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		# For given freq
		arr_calc = math._conv_spectrum(
			which_conv="perwave_to_perfreq", spectrum=arr_spec, freq=arr_freq
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For per freq -> per wave
		arr_act = arr_spec * arr_freq / arr_wave
		# For given wave
		arr_calc = math._conv_spectrum(
			which_conv="perfreq_to_perwave", spectrum=arr_spec, wave=arr_wave
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		# For given freq
		arr_calc = math._conv_spectrum(
			which_conv="perfreq_to_perwave", spectrum=arr_spec, freq=arr_freq
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For per wave -> per energy
		arr_act = arr_spec * arr_wave / (h0 * arr_freq)
		# For given wave
		arr_calc = math._conv_spectrum(
			which_conv="perwave_to_perenergy", spectrum=arr_spec, wave=arr_wave
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		# For given freq
		arr_calc = math._conv_spectrum(
			which_conv="perwave_to_perenergy", spectrum=arr_spec, freq=arr_freq
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For photon -> energy
		arr_act = arr_spec * (h0 * arr_freq)
		# For given wave
		arr_calc = math._conv_spectrum(
			which_conv="photon_to_energy", spectrum=arr_spec, wave=arr_wave
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		# For given freq
		arr_calc = math._conv_spectrum(
			which_conv="photon_to_energy", spectrum=arr_spec, freq=arr_freq
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For energy -> photon
		arr_act = arr_spec / (h0 * arr_freq)
		# For given wave
		arr_calc = math._conv_spectrum(
			which_conv="energy_to_photon", spectrum=arr_spec, wave=arr_wave
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		# For given freq
		arr_calc = math._conv_spectrum(
			which_conv="energy_to_photon", spectrum=arr_spec, freq=arr_freq
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_conv_spectrum.")
	#
	##Test Cartesian vs RADMC-3D coordinate system conversions
	def test_conv_sphericalandcartesian(self):
		print("\n> Running test: test_conv_sphericalandcartesian.")
		#Prepare input data
		num_points = 100
		arr_a = np.random.uniform(0, 1000, size=num_points)
		arr_b = np.random.uniform(0, 2*pi, size=num_points)
		#
		#Prepare correct results
		# For spherical -> Cartesian
		dict_act = {
			"x":(arr_a*np.cos(arr_b)),
			"y":(arr_a*np.sin(arr_b))
		}
		dict_calc = math._conv_sphericalandcartesian(
			which_conv="sphericaltocartesian", radius=arr_a, theta=arr_b
		)
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key in dict_act:
			self.assertTrue(np.allclose(dict_act[curr_key], dict_calc[curr_key]))
		#
		print("> Test complete: test_conv_sphericalandcartesian.")
	#
	##Test wavelength and frequency conversions
	def test_conv_waveandfreq(self):
		print("\n> Running test: test_conv_waveandfreq.")
		##Prepare input data and correct results
		num_points = 100
		arr_x = np.random.uniform(size=num_points) * 1000
		#
		#For wave -> frequency
		arr_act = (c0 / arr_x)
		arr_calc = math._conv_waveandfreq(
			which_conv="wavetofreq", wave=arr_x
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For frequency -> wave
		arr_act = (c0 / arr_x)
		arr_calc = math._conv_waveandfreq(
			which_conv="freqtowave", freq=arr_x
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For wave -> energy
		arr_act = (c0 * h0 / arr_x)
		arr_calc = math._conv_waveandfreq(
			which_conv="wavetoenergy", wave=arr_x
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		#For energy -> wave
		arr_act = (c0 / (arr_x / h0))
		arr_calc = math._conv_waveandfreq(
			which_conv="energytowave", energy=arr_x
		)
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_conv_waveandfreq.")
	#
	##Test search for outermost non-zero values
	def test_find_inds_outernonzeros(self):
		print("\n> Running test: test_find_inds_outernonzeros.")

		#Prepare input data
		arr_x = np.array([0, 0, 0, 1, 1, 0, 2, 1, 2, 1, 0])
		#
		#For case with inclusion of innermost outer-zeros
		# Set actual values
		dict_act = {"ind_left":2, "ind_right":10}
		# Calculate using package
		dict_calc = math._find_inds_outernonzeros(
			arr=arr_x, include_outerzeros=True
		)
		# Check calculations
		self.assertTrue(np.isclose(dict_act["ind_left"], dict_calc["ind_left"]))
		self.assertTrue(np.isclose(dict_act["ind_right"],dict_calc["ind_right"]))
		#
		#For case without inclusion of innermost outer-zeros
		# Set actual values
		dict_act = {"ind_left":3, "ind_right":9}
		# Calculate using package
		dict_calc = math._find_inds_outernonzeros(
			arr=arr_x, include_outerzeros=False
		)
		# Check calculations
		self.assertTrue(np.isclose(dict_act["ind_left"], dict_calc["ind_left"]))
		self.assertTrue(np.isclose(dict_act["ind_right"],dict_calc["ind_right"]))
		#
		print("> Test complete: test_find_inds_outernonzeros.")
	#
	##Test interpolation of given points to grid
	def test_interpolate_pointstogrid(self):
		print("\n> Running test: test_interpolate_pointstogrid.")
		#Prepare input data
		len_x = 100
		len_y = 85
		matr_x = np.random.uniform(-50, 100, size=(len_y,len_x))
		matr_y = np.random.uniform(-200, 80, size=(len_y,len_x))
		arr_x = matr_x.flatten()
		arr_y = matr_y.flatten()
		matr_x2 = np.random.uniform(-50, 100, size=(len_y,len_x))
		matr_y2 = np.random.uniform(-200, 80, size=(len_y,len_x))
		matr_val = np.random.uniform(0, 50, size=(len_y,len_x))
		#
		#Set actual values
		arr_act_raw = scipy_interptogrid(
			points=(arr_y, arr_x), values=matr_val.flatten(),
			xi=(matr_y2, matr_x2), method="linear", rescale=True,
			fill_value=np.nan
		)
		# Remove values beyond inputted range to disallow extrapolation
		tmpinds = np.isnan(arr_act_raw)
		matr_x2[tmpinds] = np.median(matr_x)
		matr_y2[tmpinds] = np.median(matr_y)
		# Redo the final interpolated grid
		arr_act = scipy_interptogrid(
			points=(arr_y, arr_x), values=matr_val.flatten(),
			xi=(matr_y2, matr_x2), method="linear", rescale=True,
			fill_value=np.nan
		)
		#
		#Calculate using package
		arr_calc = math._interpolate_pointstogrid(
			old_points_yx=np.asarray([arr_y, arr_x]).T, old_matr_values=matr_val,
			new_matr_x=matr_x2, new_matr_y=matr_y2, inds_valid=None
		)
		#
		#Check calculations
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_interpolate_pointstogrid.")
	#
	##Test trimming of repeating values within an array
	def test_remove_adjacent_redundancy(self):
		print("\n> Running test: test_remove_adjacent_redundancy.")

		##Prepare input data and correct results
		num_points = 100
		arr_x = np.array([
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
		])
		arr_y = np.array([
			0, 0, 1, 1, 0, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1
		])
		#
		#For case with trimming of leading zeros
		# Set actual values
		act_x = np.array([
			1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 15, 17, 18, 19
		])
		act_y = np.array([
			0, 1, 1, 0, 2, 1, 2, 2, 1, 1, 0, 0, 1, 1
		])
		dict_act = {"x":act_x, "y":act_y}
		# Calculate using package
		dict_calc = math._remove_adjacent_redundancy(
			y_old=arr_y, x_old=arr_x, do_trim_zeros=True, do_verbose_plot=False
		)
		# Check calculations
		self.assertTrue(np.allclose(dict_act["x"], dict_calc["x"]))
		self.assertTrue(np.allclose(dict_act["y"], dict_calc["y"]))
		#
		#For case without trimming of leading zeros
		# Set actual values
		act_x = np.array([
			0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 15, 17, 18, 19
		])
		act_y = np.array([
			0, 0, 1, 1, 0, 2, 1, 2, 2, 1, 1, 0, 0, 1, 1
		])
		dict_act = {"x":act_x, "y":act_y}
		# Calculate using package
		dict_calc = math._remove_adjacent_redundancy(
			y_old=arr_y, x_old=arr_x, do_trim_zeros=False, do_verbose_plot=False
		)
		# Check calculations
		self.assertTrue(np.allclose(dict_act["x"], dict_calc["x"]))
		self.assertTrue(np.allclose(dict_act["y"], dict_calc["y"]))
		#
		print("> Test complete: test_remove_adjacent_redundancy.")
	#
	##Test weighted linear fitting of given x and y data
	def test_fit_linear_weighted(self):
		print("\n> Running test: test_fit_linear_weighted.")
		#Prepare input data
		rtol = 0.01
		num_points = 100
		arr_x_raw = np.linspace(0, 100, num_points)
		arr_x = arr_x_raw + ((np.random.uniform(size=num_points)*10) - 5)
		arr_x_err = ((np.random.uniform(size=num_points)*15) - 7.5)
		arr_y = (15*arr_x) + 72 + ((np.random.uniform(size=num_points)*2)-1)
		arr_y_err = ((np.random.uniform(size=num_points)*30) - 15)

		#Prepare helper function
		def _func_helper(params, x):
			return ((params[0]*x) + params[1])

		#Set actual values
		model_odr = scipy_odr.Model(_func_helper)
		data_odr = scipy_odr.Data(
			arr_x, arr_y, wd=(1/(arr_x_err*arr_x_err)),
			we=(1/(arr_y_err*arr_y_err))
		)
		output = scipy_odr.ODR(data_odr, model_odr, beta0=[1,2]).run().beta
		dict_act = {"abest":output[0], "bbest":output[1]}

		#Calculate using package
		dict_calc = math._fit_linear_weighted(
			xdata=arr_x, ydata=arr_y, yerr=arr_y_err, xerr=arr_x_err
		)

		#Check calculations
		self.assertTrue(np.isclose(
			dict_act["abest"], dict_calc["abest"], rtol=rtol
		))
		self.assertTrue(np.isclose(
			dict_act["bbest"], dict_calc["bbest"], rtol=rtol
		))
		#
		print("> Test complete: test_fit_linear_weighted.")
	#
	##Test generation of scattering profile for Lyman-alpha photons
	def test_make_profile_kappa_scattLya(self):
		print("\n> Running test: test_make_profile_kappa_scattLya.")
		#Prepare input data
		num_points = 100
		arr_wave = (np.random.uniform(size=num_points) * 1000)
		#Prepare helper function
		def _func_integrand(y, x, a):
			numerator = np.exp(-(y*y))
			denominator = (((x+y)**2) + (a*a))
			return (numerator / denominator)
		#
		#Set actual values - e.g. Laursen+2007, Bethell+2011
		nu0 = (c0 / 121.567E-9) #[Hz]
		f12 = 0.4162
		T = 1000
		vT = np.sqrt((2*k_B0*T/mH0))
		delta_nuD = nu0*vT/c0
		delta_nuL = 9.936E7 #[Hz]
		val_prefac = (
			f12 * np.sqrt(pi) * e0*e0
			* (1/(4.0*pi*eps0)) #Weird statColoumb conversion work
			/ (me0 * c0 * delta_nuD)
		)
		val_a = delta_nuL / (2.0 * delta_nuD)
		arr_x = ((c0 / arr_wave) - nu0)/delta_nuD
		arr_crosssec = (
			(val_prefac * val_a / pi)
			* np.asarray([
				scipy_integrate.quad(
					_func_integrand, -np.inf, np.inf,
					args=(arr_x[ind], val_a)
				)[0]
				for ind in range(0, len(arr_x))
			])
		)
		arr_act = (arr_crosssec / mH0) #[m^2] -> [m^2/kg]
		#Calculate using package
		arr_calc = math.make_profile_kappa_scattLya(wave_arr=arr_wave)
		#Check calculations
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_make_profile_kappa_scattLya.")
	#
	##Test generation of scattering profile for X-rays (electron Thomson)
	def test_make_profile_kappa_scattX(self):
		print("\n> Running test: test_make_profile_kappa_scattX.")
		#Prepare input data
		num_points = 100
		arr_x = np.random.uniform(size=num_points) * 1000
		#Set actual values
		arr_crosssec = (
			np.ones(shape=len(arr_x)) * sigma_T0
		) #[m^2]; Thomson scattering
		arr_act = (arr_crosssec / mH0) #[m] -> [m^2/kg] per H
		#Calculate using package
		arr_calc = math.make_profile_kappa_scattX(x_arr=arr_x)
		#Check calculations
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_make_profile_kappa_scattX.")
	#
	##Test generation of Voigt profile
	def test_make_profile_voigt(self):
		print("\n> Running test: test_make_profile_voigt.")

		#Prepare input data
		num_points = 100
		arr_x = np.random.uniform(size=num_points) * 1000
		a_val = np.random.uniform()
		#
		#Prepare helper function
		def _func_integrand(y, x, a):
			numerator = np.exp(-(y*y))
			denominator = (((x+y)**2) + (a*a))
			return (numerator / denominator)
		#
		#Set actual values
		arr_act = (
			(a_val / pi)
			* np.asarray([
				scipy_integrate.quad(
					_func_integrand, -np.inf, np.inf,
					args=(arr_x[ind], a_val)
				)[0]
				for ind in range(0, len(arr_x))
			])
		)
		#Calculate using package
		arr_calc = math.make_profile_voigt(
			x_arr=arr_x, aval=a_val
		)
		#Check calculations
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_make_profile_voigt.")
	#
	##Test unification of y-arrays over same x-array
	def test_unify_spectra_wavelengths(self):
		print("\n> Running test: test_unify_spectra_wavelengths.")

		##Prepare input data and correct results
		num_points = 100
		arr_x1 = np.random.uniform(size=(num_points)) * 1000
		arr_x2 = np.random.uniform(size=(num_points)) * 1000
		arr_x3 = np.random.uniform(size=(num_points)) * 1000
		arr_y1 = np.random.uniform(size=(num_points)) * 1000
		arr_y2 = np.random.uniform(size=(num_points)) * 1000
		arr_y3 = np.random.uniform(size=(num_points)) * 1000
		fill_value = (np.random.uniform() * 1000) - 500
		x_list = [arr_x1, arr_x2, arr_x3]
		y_list = [arr_y1, arr_y2, arr_y3]
		set_axis = -1
		#
		#For minimum overlap
		# Determine x-values of minimum overlap
		x_min = np.max([arr_x1.min(), arr_x2.min(), arr_x3.min()]) #Largest min.
		x_max = np.min([arr_x1.max(), arr_x2.max(), arr_x3.max()]) #Smallest max.
		act_x = np.sort(np.unique(np.concatenate((
			arr_x1[((arr_x1 >= x_min) & (arr_x1 <= x_max))],
			arr_x2[((arr_x2 >= x_min) & (arr_x2 <= x_max))],
			arr_x3[((arr_x3 >= x_min) & (arr_x3 <= x_max))]
		))))
		# Interpolate y-arrays over minimum overlap range
		act_y1 = scipy_interp1d(
			x=arr_x1, y=arr_y1, fill_value=fill_value, bounds_error=False
		)(act_x)
		act_y2 = scipy_interp1d(
			x=arr_x2, y=arr_y2, fill_value=fill_value, bounds_error=False
		)(act_x)
		act_y3 = scipy_interp1d(
			x=arr_x3, y=arr_y3, fill_value=fill_value, bounds_error=False
		)(act_x)
		dict_act = {
			"x":act_x, "y":[act_y1, act_y2, act_y3]
		}
		# Calculate using package
		dict_calc = math._unify_spectra_wavelengths(
			x_list=x_list, y_list=y_list, which_overlap="minimum",
			fill_value=fill_value, axis=set_axis
		)
		# Check calculations
		self.assertTrue(np.allclose(dict_act["x"], dict_calc["x"]))
		self.assertTrue(np.allclose(dict_act["y"][0], dict_calc["y"][0]))
		self.assertTrue(np.allclose(dict_act["y"][1], dict_calc["y"][1]))
		self.assertTrue(np.allclose(dict_act["y"][2], dict_calc["y"][2]))
		#
		#For maximum overlap
		act_x = np.sort(np.unique(np.concatenate((arr_x1, arr_x2, arr_x3))))
		# Interpolate y-arrays over maximum overlap range
		act_y1 = scipy_interp1d(
			x=arr_x1, y=arr_y1, fill_value=fill_value, bounds_error=False
		)(act_x)
		act_y2 = scipy_interp1d(
			x=arr_x2, y=arr_y2, fill_value=fill_value, bounds_error=False
		)(act_x)
		act_y3 = scipy_interp1d(
			x=arr_x3, y=arr_y3, fill_value=fill_value, bounds_error=False
		)(act_x)
		dict_act = {
			"x":act_x, "y":[act_y1, act_y2, act_y3]
		}
		# Calculate using package
		dict_calc = math._unify_spectra_wavelengths(
			x_list=x_list, y_list=y_list, which_overlap="maximum",
			fill_value=fill_value, axis=set_axis
		)
		# Check calculations
		self.assertTrue(np.allclose(dict_act["x"], dict_calc["x"]))
		self.assertTrue(np.allclose(dict_act["y"][0], dict_calc["y"][0]))
		self.assertTrue(np.allclose(dict_act["y"][1], dict_calc["y"][1]))
		self.assertTrue(np.allclose(dict_act["y"][2], dict_calc["y"][2]))
		#
		print("> Test complete: test_unify_spectra_wavelengths.")
	#
	##Test generation of empirical profile
	def test_use_profile_empirical(self):
		print("\n> Running test: test_use_profile_empirical.")

		##Prepare input data and correct results
		num_points = 100
		arr_x = np.random.uniform(size=num_points) * 1000
		arr_x0 = np.random.uniform(size=num_points) * 1000
		arr_a = np.random.uniform(size=num_points) * 1000
		arr_q = np.random.uniform(size=num_points) * 10
		arr_act = arr_a*((arr_x/arr_x0)**arr_q)
		#Calculate using package
		arr_calc = math._use_profile_empirical(
			x_val=arr_x, x_c=arr_x0, a_c=arr_a, q_c=arr_q
		)
		#Check calculations
		self.assertTrue(np.allclose(arr_act, arr_calc))
		#
		print("> Test complete: test_use_profile_empirical.")
	#
#
#Purpose: Testing the functions in argonaut_rates
class Test_Rates(unittest.TestCase):
	print("\n> Running test suite for argonaut rate functions.")
	##Test extraction of subset of reactions
	def test_extract_subset_reactions(self):
		print("\n> Running test: test_extract_subset_reactions.")
		#Prepare input data
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		dict_set = _helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=True
		)
		#
		#Prepare correct results
		dict_act = {}
		for curr_key in dict_set["reactions_gas"]:
			if (int(dict_set["reactions_gas"][curr_key]["itype"]) in [3]):
				dict_act[curr_key] = dict_set["reactions_gas"][curr_key]
		#Calculate using package
		dict_calc = rates._extract_subset_reactions(
			mode="UV", dict_reactions_gas=dict_set["reactions_gas"],
            dict_reactions_gr=dict_set["reactions_grain"]
		)
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key1 in dict_act:
			self.assertTrue(
				(
				sorted(list(dict_act[curr_key1].keys()))
				== sorted(list(dict_calc[curr_key1].keys()))
				)
			)
			for curr_key2 in dict_act[curr_key1]:
				tmpval = dict_act[curr_key1][curr_key2]
				if (isinstance(tmpval, str)):
					self.assertTrue(
						(dict_act[curr_key1][curr_key2]
						== dict_calc[curr_key1][curr_key2])
					)
				elif (isinstance(tmpval, (float, int))):
					self.assertTrue(np.isclose(
						dict_act[curr_key1][curr_key2],
						dict_calc[curr_key1][curr_key2]
					))
				else:
					self.assertTrue(
						(np.array_equal(
							dict_act[curr_key1][curr_key2],
							dict_calc[curr_key1][curr_key2]
						))
					)
				#
			#
		#
		print("> Test complete: test_extract_subset_reactions.")
	#
	##Test extraction of species from chemical network files
	def test_fetch_all_species(self):
		print("\n> Running test: test_fetch_all_species.")
		#Prepare input data
		filename_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_speciesgas
		)
		filename_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_speciesgrain
		)
		#Prepare correct results
		# For case with phase markers included
		dict_act = _helper_fetch_all_species(
			filename_gas=filename_gas, filename_grain=filename_grain,
			do_remove_phase_marker=False
		)
		# Calculate using package
		dict_calc = rates._fetch_all_species(
			filename_species_gas=filename_gas,
			filename_species_grain=filename_grain,
			do_remove_phase_marker=False
		)
		# Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key in dict_act:
			self.assertTrue(np.array_equal(
				dict_act[curr_key], dict_calc[curr_key]
			))
		#
		#For case without phase markers
		dict_act = _helper_fetch_all_species(
			filename_gas=filename_gas, filename_grain=filename_grain,
			do_remove_phase_marker=True
		)
		# Calculate using package
		dict_calc = rates._fetch_all_species(
			filename_species_gas=filename_gas,
			filename_species_grain=filename_grain,
			do_remove_phase_marker=True
		)
		# Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key in dict_act:
			self.assertTrue(np.array_equal(
				dict_act[curr_key], dict_calc[curr_key]
			))
		#
		print("> Test complete: test_fetch_all_species.")
	#
	##Test extraction of updatable reactions
	def test_fetch_updatable_reactions(self):
		print("\n> Running test: test_fetch_updatable_reactions.")
		#Prepare input data
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		dict_orig = _helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=True
		)["reactions_UV"]
		#
		#Prepare correct results
		dict_act = _helper_fetch_updatable_reactions_UV(
			dict_orig=dict_orig, dict_mol=test_dict_mol0
		)
		#Calculate using package
		dict_calc = rates._fetch_updatable_reactions(
			dict_reactions_orig=dict_orig, dict_mol_UV=test_dict_mol0,
			mode="UV"
		)
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key1 in dict_act:
			self.assertTrue(
				(
				sorted(list(dict_act[curr_key1].keys()))
				== sorted(list(dict_calc[curr_key1].keys()))
				)
			)
			for curr_key2 in dict_act[curr_key1]:
				tmpval = dict_act[curr_key1][curr_key2]
				if (isinstance(tmpval, str)):
					self.assertTrue(
						(dict_act[curr_key1][curr_key2]
						== dict_calc[curr_key1][curr_key2])
					)
				elif (isinstance(tmpval, (float, int))):
					self.assertTrue(np.isclose(
						dict_act[curr_key1][curr_key2],
						dict_calc[curr_key1][curr_key2]
					))
				else:
					self.assertTrue(
						(np.array_equal(
							dict_act[curr_key1][curr_key2],
							dict_calc[curr_key1][curr_key2]
						))
					)
				#
			#
		#
		print("> Test complete: test_fetch_updatable_reactions.")
	#
	##Test extraction of species from chemical network files
	def test_generate_dict_reactions(self):
		print("\n> Running test: test_generate_dict_reactions.")
		#Prepare input species data
		filename_species_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_speciesgas
		)
		filename_species_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_speciesgrain
		)
		#
		#Prepare correct species results
		dict_species = _helper_fetch_all_species(
			filename_gas=filename_species_gas,
			filename_grain=filename_species_grain,
			do_remove_phase_marker=False
		)
		act_species_gas = dict_species["gas"]
		act_species_grain = dict_species["grain"]
		act_species_all = dict_species["all"]
		#
		#Prepare input reaction data
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		#
		#Prepare correct reaction results
		dict_reaction = _helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=True
		)
		#
		#Assemble full correct results
		dict_acts = {
			"list_species_gas":dict_species["gas"],
			"list_species_grain":dict_species["grain"],
			"list_species_all":dict_species["all"]
		}
		dict_acts.update(dict_reaction)
		#
		#Calculate using package
		dict_calcs = rates.generate_dict_reactions(
			mode="nautilus", filepath_chemistry=filepath_testing,
			filepath_save=filepath_testing
		)
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_acts.keys())) == sorted(list(dict_calcs.keys())))
		)
		for curr_key1 in dict_acts:
			# Internal dictionary comparisons
			if (isinstance(dict_acts[curr_key1], dict)):
				self.assertTrue(
					(
						sorted(list(dict_acts[curr_key1].keys()))
						== sorted(list(dict_calcs[curr_key1].keys()))
					)
				)
				for curr_key2 in dict_acts[curr_key1]:
					self.assertTrue(
						(
						sorted(list(dict_acts[curr_key1][curr_key2].keys()))
						== sorted(list(dict_calcs[curr_key1][curr_key2].keys()))
						)
					)
					for curr_key3 in dict_acts[curr_key1][curr_key2]:
						tmpval = dict_acts[curr_key1][curr_key2][curr_key3]
						if (isinstance(tmpval, str)):
							self.assertTrue(
								(dict_acts[curr_key1][curr_key2][curr_key3]
								== dict_calcs[curr_key1][curr_key2][curr_key3])
							)
						elif (isinstance(tmpval, (float, int))):
							self.assertTrue(np.isclose(
								dict_acts[curr_key1][curr_key2][curr_key3],
								dict_calcs[curr_key1][curr_key2][curr_key3]
							))
						else:
							self.assertTrue(
								(np.array_equal(
									dict_acts[curr_key1][curr_key2][curr_key3],
									dict_calcs[curr_key1][curr_key2][curr_key3]
								))
							)
						#
					#
				#
			#
			# Other comparisons
			else:
				if (isinstance(dict_acts[curr_key1], (int, float))):
					self.assertTrue(np.isclose(
						dict_acts[curr_key1], dict_calcs[curr_key1]
					))
				elif (isinstance(dict_acts[curr_key1], str)):
					self.assertTrue(
						(dict_acts[curr_key1] == dict_calcs[curr_key1])
					)
				else:
					self.assertTrue(np.array_equal(
						dict_acts[curr_key1], dict_calcs[curr_key1]
					))
			#
		#
		print("> Test complete: test_generate_dict_reactions.")
	#
	##Test extraction of reactions from chemical network files
	def test_read_pnautilus_reactions(self):
		print("\n> Running test: test_read_pnautilus_reactions.")
		#Prepare input data
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		filename_output_calc = os.path.join(
			filepath_testing, "test_output_reactions_calc.in"
		)
		#Prepare correct results
		dict_act = _helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=False
		)["reactions_gas"]
		#Calculate using package
		dict_calc = rates._read_pnautilus_reactions(
			text_filename=filename_input_gas,
			table_filename=filename_output_calc, last_counter=0
		)["result"]
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key1 in dict_act:
			self.assertTrue(
				(
				sorted(list(dict_act[curr_key1].keys()))
				== sorted(list(dict_calc[curr_key1].keys()))
				)
			)
			for curr_key2 in dict_act[curr_key1]:
				tmpval = dict_act[curr_key1][curr_key2]
				if (isinstance(tmpval, str)):
					self.assertTrue(
						(dict_act[curr_key1][curr_key2]
						== dict_calc[curr_key1][curr_key2])
					)
				elif (isinstance(tmpval, (float, int))):
					self.assertTrue(np.isclose(
						dict_act[curr_key1][curr_key2],
						dict_calc[curr_key1][curr_key2]
					))
				else:
					self.assertTrue(
						(np.array_equal(
							dict_act[curr_key1][curr_key2],
							dict_calc[curr_key1][curr_key2]
						))
					)
				#
			#
		#
		print("> Test complete: test_read_pnautilus_reactions.")
	#
	##Test calculation of branching ratios for reactions
	def test_set_branching_ratios(self):
		print("\n> Running test: test_set_branching_ratios.")
		#Prepare input data
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		filename_output_calc = os.path.join(
			filepath_testing, "test_output_reactions_calc.in"
		)
		#Prepare correct results
		dict_set = _helper_dict_reaction(
			filename_input_gas=filename_input_gas, do_keep_noprobability=True,
			filename_input_grain=filename_input_grain, do_probability=True
		)
		dict_orig = dict_set["reactions_gas_orig"]
		dict_act = dict_set["reactions_gas"]
		#Calculate using package
		dict_calc = rates._set_branching_ratios(dict_reactions_orig=dict_orig)
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key1 in dict_act:
			self.assertTrue(
				(
				sorted(list(dict_act[curr_key1].keys()))
				== sorted(list(dict_calc[curr_key1].keys()))
				)
			)
			for curr_key2 in dict_act[curr_key1]:
				tmpval = dict_act[curr_key1][curr_key2]
				if (isinstance(tmpval, str)):
					self.assertTrue(
						(dict_act[curr_key1][curr_key2]
						== dict_calc[curr_key1][curr_key2])
					)
				elif (isinstance(tmpval, (float, int))):
					self.assertTrue(np.isclose(
						dict_act[curr_key1][curr_key2],
						dict_calc[curr_key1][curr_key2]
					))
				else:
					self.assertTrue(
						(np.array_equal(
							dict_act[curr_key1][curr_key2],
							dict_calc[curr_key1][curr_key2]
						))
					)
				#
			#
		#
		print("> Test complete: test_set_branching_ratios.")
	#
	##Test update of individual reaction dictionary
	def test_update_reaction_dict_indiv(self):
		print("\n> Running test: test_update_reaction_dict_indiv.")
		#Prepare input data
		num_points = 70
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		dict_orig = _helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=True
		)["reactions_gas"]
		thres_coeff0 = preset.pnautilus_threshold_coeff0
		#
		#Iterate through reactions
		for curr_key in dict_orig:
			#Fetch current dictionaries of information
			curr_reactdict = dict_orig[curr_key]
			curr_phototype = dict_orig[curr_key]["phototype"]
			if (curr_reactdict["reactant_main"] not in test_dict_mol0):
				continue
			if (int(curr_reactdict["itype"]) not in [3]):
				continue
			#
			tmpdict = test_dict_mol0[curr_reactdict["reactant_main"]]
			curr_moldict = {"wave":None, "cross":None}
			if (curr_phototype == "photoion"):
				curr_moldict["wave"] = tmpdict["wave_photoion_UV"]
				curr_moldict["cross"] = tmpdict["cross_photoion_UV"]
			elif (curr_phototype == "photodiss"):
				curr_moldict["wave"] = tmpdict["wave_photodiss_UV"]
				curr_moldict["cross"] = tmpdict["cross_photodiss_UV"]
			#
			if (curr_moldict["wave"] is None):
				continue

			#Generate new spectrum values
			arr_x_spec = np.sort(np.unique(
				np.random.uniform(low=85E-9, high=205E-9, size=num_points)
			))
			arr_y_spec = np.random.uniform(low=1, high=100, size=len(arr_x_spec))

			#Prepare correct results
			dict_act = _helper_update_reaction_dict_indiv(
				arr_x_spec=arr_x_spec, arr_y_spec=arr_y_spec,
				dict_molinfo=curr_moldict, react_orig=curr_reactdict,
				minval_coeff0=None
			)

			#Calculate using package
			dict_calc = rates._update_reaction_dict_indiv(
				mode="UV", wavelength_radiation=arr_x_spec,
				spectrum_photon_radiation=arr_y_spec, dict_molinfo=curr_moldict,
				react_orig=curr_reactdict, minval_coeff0=thres_coeff0
			)

			#Check that calculations are correct
			self.assertTrue(
				(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
			)
			for curr_key1 in dict_act:
				tmpval = dict_act[curr_key1]
				if (isinstance(tmpval, str)):
					self.assertTrue(
						(dict_act[curr_key1] == dict_calc[curr_key1])
					)
				elif (isinstance(tmpval, (float, int))):
					self.assertTrue(np.isclose(
						dict_act[curr_key1], dict_calc[curr_key1]
					))
				else:
					self.assertTrue(
						(np.array_equal(
							dict_act[curr_key1], dict_calc[curr_key1]
						))
					)
				#
			#
		#
		print("> Test complete: test_update_reaction_dict_indiv.")
	#
	##Test update of reaction files for Nautilus
	def test_update_reaction_file_pnautilus(self):
		print("\n> Running test: test_update_reaction_file_pnautilus.")
		#Prepare input data
		num_points = 100
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		arr_x_spec = np.sort(np.unique(
			np.random.uniform(low=85E-9, high=205E-9, size=num_points)
		))
		arr_y_spec = np.random.uniform(low=1, high=100, size=len(arr_x_spec))
		thres_coeff0 = preset.pnautilus_threshold_coeff0
		#
		dict_orig_UV = _helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=True
		)["reactions_UV"]
		dict_updated = rates.update_reactions(
			wavelength_radiation=arr_x_spec,
			spectrum_photon_radiation=arr_y_spec,
			dict_reactions_orig=dict_orig_UV,
			minval_coeff0=thres_coeff0, do_return_byproducts=True,
			filesave_dictold=None, filesave_dictnew=None,
			dict_mol_UV=test_dict_mol0, mode="UV"
		)["reactions_new"]
		#
		filename_output_gas_calc = os.path.join(
			filepath_testing, "test_output_updates_gas_calc.in"
		)
		with open(filename_input_gas,'r') as openfile:
			lines_gas = openfile.readlines()

		#Prepare correct results
		str_gas_act = ""
		for curr_line in lines_gas:
			#Ensure a newline is included
			if (not curr_line.endswith("\n")):
				curr_line += "\n"

			#Store headers as they are
			if (curr_line.startswith("!")):
				str_gas_act += ("!***" + curr_line)
				continue
			#
			#Otherwise, process this line
			curr_dict = _helper_conv_reaction_stranddict(
				dict_orig=None, str_orig=curr_line, which_conv="str_to_dict"
			)
			list_keys = [
				key for key in dict_updated
				if (curr_dict["reaction"] == dict_updated[key]["reaction"])
			]
			if (len(list_keys) == 0): #Skip if not updatable
				str_gas_act += curr_line
				continue
			elif (len(list_keys) > 1): #Raise error if duplicate reactions
				raise ValueError(f"Err: Bad key! {curr_dict}\n{list_keys}")
			#
			#Fetch updated dictionary and convert to string
			curr_newdict = dict_updated[list_keys[0]]
			curr_newstr = _helper_conv_reaction_stranddict(
				dict_orig=curr_newdict, str_orig=None, which_conv="dict_to_str"
			)
			#
			#Update reaction file string
			str_gas_act += f"! {curr_line}"
			str_gas_act += f"{curr_newstr}\n"
		#
		lines_gas_act = str_gas_act.split("\n")
		#Remove last entry if empty string
		if (lines_gas_act[-1] == ""):
			lines_gas_act = lines_gas_act[0:(len(lines_gas_act)-1)]

		#Calculate using package
		tmpdict = {"UV":dict_updated}
		rates._update_reaction_file_pnautilus(
			dict_reactions_new_all=tmpdict,
			filename_orig=filename_input_gas,
			filename_save=filename_output_gas_calc
		)

		#Check that calculations are correct
		with open(filename_output_gas_calc,'r') as openfile:
			lines_gas_calc = [
				re.sub("\n$", "", item) for item in openfile.readlines()
				if (not item.startswith("!")) #Skip comments
			]
			#Remove last entry if empty string
			if (lines_gas_calc[-1] == ""):
				lines_gas_calc = lines_gas_calc[0:(len(lines_gas_calc)-1)]
		i_track = 0
		for ii in range(0, len(lines_gas_act)):
			#Skip comments
			if (lines_gas_act[ii].startswith("!")):
				continue
			self.assertTrue(
				(lines_gas_act[ii] == lines_gas_calc[i_track])
			)
			#Increment counter
			i_track += 1
		#
		self.assertTrue((i_track == len(lines_gas_calc))) #Verify count
		print("> Test complete: test_update_reaction_file_pnautilus.")
	#
	##Test update of all reactions in dictionary
	def test_update_reactions(self):
		print("\n> Running test: test_update_reactions.")
		#Prepare input data
		num_points = 70
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		dict_reactions = _helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=True
		)["reactions_UV"]
		dict_updater = _helper_fetch_updatable_reactions_UV(
			dict_orig=dict_reactions, dict_mol=test_dict_mol0
		)
		thres_coeff0 = preset.pnautilus_threshold_coeff0
		# Generate spectrum values
		arr_x_spec = np.sort(np.unique(
			np.random.uniform(low=85E-9, high=205E-9, size=num_points)
		))
		arr_y_spec = np.random.uniform(low=1, high=100, size=len(arr_x_spec))
		#
		#Prepare correct results
		dict_act = {}
		for curr_key in dict_reactions:
			#Fetch current dictionaries of information
			curr_reactdict = dict_reactions[curr_key]
			if (curr_reactdict["reactant_main"] not in test_dict_mol0):
				continue
			if (curr_key not in dict_updater):
				continue
			curr_moldict = dict_updater[curr_key]

			#Prepare correct results
			dict_act[curr_key] = _helper_update_reaction_dict_indiv(
				arr_x_spec=arr_x_spec, arr_y_spec=arr_y_spec,
				dict_molinfo=curr_moldict, react_orig=curr_reactdict,
				minval_coeff0=None
			)
		#
		#Calculate using package
		dict_calc = rates.update_reactions(
			wavelength_radiation=arr_x_spec,
			spectrum_photon_radiation=arr_y_spec,
			dict_reactions_orig=dict_reactions,
			minval_coeff0=thres_coeff0,
			do_return_byproducts=True,
			filesave_dictold=None, filesave_dictnew=None,
			dict_mol_UV=test_dict_mol0, mode="UV"
		)["reactions_new"]
		#
		#Check that calculations are correct
		self.assertTrue(
			(sorted(list(dict_act.keys())) == sorted(list(dict_calc.keys())))
		)
		for curr_key1 in dict_act:
			self.assertTrue(
				(
				sorted(list(dict_act[curr_key1].keys()))
				== sorted(list(dict_calc[curr_key1].keys()))
				)
			)
			tmpval = dict_act[curr_key1]
			if (isinstance(tmpval, str)):
				self.assertTrue(
					(dict_act[curr_key1] == dict_calc[curr_key1])
				)
			elif (isinstance(tmpval, (float, int))):
				self.assertTrue(np.isclose(
					dict_act[curr_key1], dict_calc[curr_key1]
				))
			else:
				self.assertTrue(
					(np.array_equal(
						dict_act[curr_key1], dict_calc[curr_key1]
					))
				)
			#
		#
		print("> Test complete: test_update_reactions.")
	#
#
#Purpose: Testing the functions in argonaut_utils
class Test_Utils(unittest.TestCase):
	print("\n> Running test suite for argonaut utility functions.")
	##Test conversion from reaction string to dictionary
	def test_conv_reactstr_to_reactdict(self):
		print("\n> Running test: test_conv_reactstr_to_reactdict.")
		#Prepare reaction data
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		with open(filename_input_gas,'r') as openfile:
			all_lines = openfile.readlines()
		with open(filename_input_grain,'r') as openfile:
			all_lines += openfile.readlines()

		#Iterate through reactions and convert
		for curr_line in all_lines:
			#Skip comments
			if (curr_line.strip().startswith("!")):
				continue

			#Convert from string to dictionary
			dict_act = _helper_conv_reaction_stranddict(
				dict_orig=None, str_orig=curr_line, which_conv="str_to_dict"
			)
			#Calculate using package
			dict_calc = utils._conv_reactstr_to_reactdict(
				line=curr_line, mode="nautilus"
			)

			#Check that calculations are correct
			self.assertTrue(
				(
					sorted(list(dict_act.keys()))
					== sorted(list(dict_calc.keys()))
				)
			)
			for curr_key in dict_act:
				tmpval = dict_act[curr_key]
				if (isinstance(tmpval, str)):
					self.assertTrue((dict_act[curr_key] == dict_calc[curr_key]))
				elif (isinstance(tmpval, (float, int))):
					self.assertTrue(np.isclose(
						dict_act[curr_key], dict_calc[curr_key]
					))
				else:
					self.assertTrue(
						(np.array_equal(dict_act[curr_key], dict_calc[curr_key]))
					)
				#
			#
		#
	#
		print("> Test complete: test_conv_reactstr_to_reactdict.")
	#
	##Test conversion from reaction dictionary to string
	def test_conv_reactdict_to_reactstr(self):
		print("\n> Running test: test_conv_reactdict_to_reactstr.")
		#Prepare reaction data
		filename_input_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongasorig
		)
		filename_input_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_reactiongrainorig
		)
		dict_reactions = _helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=True
		)["reactions_gas"]
		dict_reactions.update(_helper_dict_reaction(
			filename_input_gas=filename_input_gas,
			filename_input_grain=filename_input_grain, do_probability=True
		)["reactions_grain"])
		with open(filename_input_gas,'r') as openfile:
			all_lines = openfile.readlines()
		with open(filename_input_grain,'r') as openfile:
			all_lines += openfile.readlines()
		all_lines = [item for item in all_lines if (not item.startswith("!"))]

		#Iterate through reactions and convert
		tmpsort = np.argsort([
			int(item.replace("#","")) for item in dict_reactions.keys()
		])
		list_keys = np.asarray(list(dict_reactions.keys()))[tmpsort]
		for ii in range(0, len(list_keys)):
			curr_key = list_keys[ii]
			#Convert from dictionary to string
			curr_dict = dict_reactions[curr_key]
			str_act = _helper_conv_reaction_stranddict(
				dict_orig=curr_dict, str_orig=None, which_conv="dict_to_str"
			)
			#Prepare correct results
			str_calc = utils._conv_reactdict_to_reactstr(
				react_dict=curr_dict, mode="nautilus"
			)
			#Check that calculations are correct
			# Check against actual assembled string
			self.assertTrue((str_act == str_calc))
			# Check against original string from file with blank portion blocked
			str_orig_raw = all_lines[ii]
			str_orig_blocked = all_lines[ii][0:122]
			str_orig_blocked += (" "*23)
			str_orig_blocked += all_lines[ii][122+23:171]
			str_orig_blocked += (" "*5)
			#Unify treatment in original string of 'e' vs 'E' sci. notation
			str_orig_blocked = (
				str_orig_blocked.replace("E+","e+").replace("E-","e-")
					.replace("0.000e-00","0.000e+00")
			)
			#Handle weird edge cases in original file
			str_orig_blocked = (
				str_orig_blocked.replace("1.7000e+0","1.700e+00")
			)
			self.assertTrue((str_orig_blocked == str_calc))
		#
		print("> Test complete: test_conv_reactdict_to_reactstr.")
	#
	##Test extraction of elements from molecule
	def test_get_elements(self):
		print("\n> Running test: test_get_elements.")
		#Prepare input data
		filename_gas = os.path.join(
			filepath_testing, preset.pnautilus_filename_speciesgas
		)
		filename_grain = os.path.join(
			filepath_testing, preset.pnautilus_filename_speciesgrain
		)
		list_data = _helper_fetch_all_species(
			filename_gas=filename_gas, filename_grain=filename_grain,
			do_remove_phase_marker=False
		)["all"]
		#Iterate through species
		for curr_species in list_data:
			#Prepare correct results
			curr_split = re.findall(
				"[A-Z][a-z]*[0-9]*",
				re.sub("^(J|K)","",curr_species
					).replace("c-","").replace("l-","")
			)
			dict_act = {}
			for curr_item in curr_split:
				curr_name = re.sub("[0-9]*$","",curr_item)
				curr_val = re.findall("[0-9]*$", curr_item)[0]
				if (curr_val == ""):
					curr_val = 1
				if (curr_name not in dict_act):
					dict_act[curr_name] = 0
				dict_act[curr_name] += int(curr_val)
			#
			#Calculate using package
			dict_calc = utils._get_elements(
				curr_species, do_ignore_JK=True, do_strip_orientation=True
			)
			#Check that calculations are correct
			self.assertTrue((dict_act == dict_calc))
		#
		print("> Test complete: test_get_elements.")
	#
	##Test loading of elements from file
	def test_load_elements(self):
		print("\n> Running test: test_load_elements.")
		#Prepare input data
		filename = os.path.join(
			filepath_testing, preset.pnautilus_filename_element
		)
		#Prepare correct results
		data_el = np.genfromtxt(filename, comments="!", dtype=str)
		list_act = data_el[:,0]
		#Calculate using package
		list_calc = utils._load_elements(filename)
		#Check that calculations are correct
		self.assertTrue(np.array_equal(list_act, list_calc))
		#
		print("> Test complete: test_load_elements.")
	#
#



#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------





###















#
