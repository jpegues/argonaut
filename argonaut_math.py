###FILE: argonaut_math.py
###PURPOSE: Script for performing basic math, etc. calculations for argonaut software package.
###DATE FILE CREATED: 2022-06-08
###DEVELOPERS: (Jamila Pegues; using Nautilus from Nautilus developers)


##Import necessary modules
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scipy_integrate
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.interpolate import griddata as scipy_interptogrid
import astropy.constants as const
import argonaut_constants as preset
pi = np.pi
#Set astronomical constants
c0 = const.c.value #[m/s]
e0 = const.e.value #[C in S.I. units]
eps0 = const.eps0.value #[C] #Vacuum electric permittivity constant
e0_statSI = e0 * np.sqrt((1/(4*pi*eps0))) #Units should be SI version of esu = [kg * m^3 / s^2]^(1/2)
h0 = const.h.value #[J * s]
G0 = const.G.value #[kg * m/s^2]
au0 = const.au.value #[m]
k_B0 = const.k_B.value #[W/K]
me0 = const.m_e.value #[kg]
mH0 = (const.m_p.value + me0) #Mass of H atom; [kg]
sigma_SB0 = const.sigma_sb.value #[W/(m^2 * K^4)]
#
conv_energyfluxtoG0 = 2.7E-6 #[W/m^2]; From 1G0 = 2.7E-3 erg/s/cm^2, see e.g. Bruderer+2012 write-up
eVtoJ0 = e0
#


##Method: _assemble_crosssec_X_parametric()
##Purpose: Assemble x-ray cross-section using fitted parameters from Bethell+2011b
##References:
## - Bethell+2011b
def _assemble_crosssec_X_parametric(energy_inJ_orig, do_plot=False, filepath_plot=None):
	#Extract global variables
	filepath = preset.filepath_chemistry_crosssecs_X_parametric
	struct_str_precision = preset.structure_str_precision
	num_base = 500 #Number of points to use for assembling piece-wise function

	#Load and process data from file
	data = np.genfromtxt(filepath, dtype="str")
	spans_keV = data[:,0] #Energy spans in [keV]
	gas_c0s = data[:,4].astype(float) #c_0 for gas
	gas_c1s = data[:,5].astype(float) #c_1 for gas
	gas_c2s = data[:,6].astype(float) #c_2 for gas
	dust_c0s = data[:,7].astype(float) #c_0 for dust
	dust_c1s = data[:,8].astype(float) #c_1 for dust
	dust_c2s = data[:,9].astype(float) #c_2 for dust

	#Update last span to reach requested X-ray edge
	#For left edge
	fixed_min_X_inJ = _conv_waveandfreq(which_conv="wavetoenergy",
	 									wave=preset.wavelength_range_rates_X[1])
	fixed_min_X_inkeV = (fixed_min_X_inJ / eVtoJ0 * 1E-3)
	if (float(spans_keV[0].split("-")[0]) > fixed_min_X_inkeV):#If>target
		spans_keV[0] = ("{0}-{1}".format(fixed_min_X_inkeV,
										spans_keV[0].split("-")[1]))
	#
	#For right edge
	fixed_max_X_inJ = _conv_waveandfreq(which_conv="wavetoenergy",
	 									wave=preset.wavelength_range_rates_X[0])
	fixed_max_X_inkeV = (fixed_max_X_inJ / eVtoJ0 * 1E-3)
	if (float(spans_keV[-1].split("-")[1]) < fixed_max_X_inkeV): #If<target
		spans_keV[-1] = ("{0}-{1}".format(spans_keV[-1].split("-")[0],
							 				fixed_max_X_inkeV))
	#

	#Generate base for assembling the piece-wise function
	val_start_keV = float(spans_keV[0].split("-")[0]) #Start of fitted function
	val_end_keV = float(spans_keV[-1].split("-")[1]) #End of fitted function
	x_base_keV = np.logspace(np.log10(val_start_keV),
	 						np.log10(val_end_keV), num_base)
	y_base_gas_cgs = np.ones(num_base)*np.nan
	y_base_dust_cgs = np.ones(num_base)*np.nan

	#Assemble the piece-wise function
	num_inv = len(spans_keV) #Number of piece-wise intervals
	for ii in range(0, num_inv):
		#Extract energy bounds of current range
		curr_start_keV = float(spans_keV[ii].split("-")[0])
		curr_end_keV = float(spans_keV[ii].split("-")[1])
		curr_inds = ((x_base_keV >= curr_start_keV)
		 			& (x_base_keV <= curr_end_keV))
		#

		#Compute current segment of piecewise function
		curr_keV = x_base_keV[curr_inds]
		#For the gas
		y_base_gas_cgs[curr_inds] = (1E-24 * (curr_keV**(-3))
									* (gas_c0s[ii] + (gas_c1s[ii] * curr_keV)
										+ (gas_c2s[ii] * curr_keV * curr_keV))
									) #[cm^2/H nucleus]
		#
		#For the dust
		y_base_dust_cgs[curr_inds] = (1E-24 * (curr_keV**(-3))
									* (dust_c0s[ii] + (dust_c1s[ii] * curr_keV)
										+ (dust_c2s[ii] * curr_keV * curr_keV))
									) #[cm^2/H nucleus]
	#
	#Set values at ends of array explicitly (avoiding issues with precision)
	ii = None
	tmp_ind = 0
	curr_keV = x_base_keV[tmp_ind]
	y_base_gas_cgs[tmp_ind] = (1E-24 * (curr_keV**(-3))
							* (gas_c0s[tmp_ind] + (gas_c1s[tmp_ind] * curr_keV)
								+ (gas_c2s[tmp_ind] * curr_keV * curr_keV))
							) #[cm^2/H nucleus]
	y_base_dust_cgs[tmp_ind] = (1E-24 * (curr_keV**(-3))
						* (dust_c0s[tmp_ind] + (dust_c1s[tmp_ind] * curr_keV)
							+ (dust_c2s[tmp_ind] * curr_keV * curr_keV))
						) #[cm^2/H nucleus]
	tmp_ind = -1
	curr_keV = x_base_keV[tmp_ind]
	y_base_gas_cgs[tmp_ind] = (1E-24 * (curr_keV**(-3))
							* (gas_c0s[tmp_ind] + (gas_c1s[tmp_ind] * curr_keV)
								+ (gas_c2s[tmp_ind] * curr_keV * curr_keV))
							) #[cm^2/H nucleus]
	y_base_dust_cgs[tmp_ind] = (1E-24 * (curr_keV**(-3))
						* (dust_c0s[tmp_ind] + (dust_c1s[tmp_ind] * curr_keV)
							+ (dust_c2s[tmp_ind] * curr_keV * curr_keV))
						) #[cm^2/H nucleus]
	#

	#Throw error if any nans left in array
	if (any(np.isnan(y_base_gas_cgs)) or any(np.isnan(y_base_dust_cgs))):
		raise ValueError("Whoa! Still nans to fill in piecewise function!")
	#

	#Convert piece-wise function to cross-section in S.I. units
	y_base_gas = (y_base_gas_cgs / (100 * 100)) #[m^2/per H nucleus]
	y_base_dust = (y_base_dust_cgs / (100 * 100)) #[m^2/per H nucleus]
	x_base = (x_base_keV * 1E3 * eVtoJ0) #[keV] -> [J]
	energy_inJ_raw = np.sort(np.unique(
							np.concatenate((x_base, energy_inJ_orig)))) #J
	#
	#Keep energy values that are not duplicates within radmc3d precision
	tmp_inds_keep = [0]
	tmp_inds_keep += [
				ii for ii in range(1, len(energy_inJ_raw))
				if (struct_str_precision.format(energy_inJ_raw[ii])
					!= struct_str_precision.format(energy_inJ_raw[ii-1])
				)] #Keep only if unique at printed/.inp level of precision
	tmp_inds_keep = np.asarray(tmp_inds_keep)
	energy_inJ = energy_inJ_raw[tmp_inds_keep]
	#

	#Interpolate piece-wise function to requested energies
	#For gas
	sigma_gas = scipy_interp1d(x=x_base,
								y=y_base_gas,
								bounds_error=True,
								)(energy_inJ) #[cm^2/H nucleus]
	#
	#For dust
	sigma_dust = scipy_interp1d(x=x_base,
								y=y_base_dust,
								bounds_error=True,
								)(energy_inJ) #[cm^2/H nucleus]
	#

	#Plot the piecewise function as a check
	if do_plot:
		#Plot the profile in Bethell+2011b convention
		energy_inkeV = (energy_inJ / eVtoJ0 * 1E-3)
		plt.plot(energy_inkeV,
				(1E24 * (100*100) * sigma_gas * (energy_inkeV**3)),
				color="tomato", alpha=0.5, label="Gas")
		plt.plot(energy_inkeV,
				(1E24 * (100*100) * sigma_dust * (energy_inkeV**3)),
				color="dodgerblue", alpha=0.5, label="Dust")
		plt.plot(energy_inkeV,
				(1E24 * (100*100) * (sigma_gas+sigma_dust) * (energy_inkeV**3)),
				color="black", alpha=0.5, label="Gas+Dust")
		#
		#Scale and label the plot
		plt.xscale("log")
		plt.yscale("log")
		plt.xlabel(r"Energy [keV]")
		plt.ylabel(r"Cross-Section * E^3 (10^-24 cm$^2$ keV^3 / H nucleus)")
		plt.legend(loc="best", frameon=False)
		plt.title(r"X-Ray Absorption Cross-Section (e.g. Bethell+2011b)")
		plt.savefig(os.path.join(filepath_plot, "fig_math_crosssec_absX.png"))
		plt.close()
	#

	#Return assembled cross-section profile
	return {"x_inJ":energy_inJ, "cross_gas":sigma_gas, "cross_dust":sigma_dust,
			"cross_total":(sigma_gas + sigma_dust)}
#


##Method: calc_density2D_base()
##Purpose: Returns density, calculated using provided procedure
def calc_density2D_base(which_density, arr_x=None, arr_y=None, matr_x=None, matr_y=None, norm_x=None, lstar=None, mstar=None, mdiskgas=None, theta_flare=None, frac_dustovergasmass=None, fracs_dustmdisk=None, fracs_dustscaleH=None, matr_sigma_dust=None, matr_sigma_gas=None, matr_scaleH=None, matr_tempgas=None, do_verbose=False):
	#Print some notes
	if do_verbose:
		print("Running _calc_density2D_base()!")
		print("Scheme: {0}".format(which_density))
	#

	#Prepare 2D grid for values
	if (matr_x is None) and (matr_y is None):
		matr_x, matr_y = np.meshgrid(arr_x, arr_y)

	#For analytic density
	if which_density == "analytic":
		#Print some notes
		if do_verbose:
			print("Running analytic scheme.")
		#
		#Analytically estimate inner characteristics of disk from stellar char.
		temp_mid = calc_temperature_midplane(arr_x=matr_x, lstar=lstar,
										theta_flare=theta_flare) #[K]
		matr_scaleH = calc_scaleheight(arr_x=matr_x, temp_mid=temp_mid,
	 					mstar=mstar, lstar=lstar) #[m]
		#
		return calc_density2D_analytic(matr_x=matr_x, matr_y=matr_y,
		 			matr_scaleH=matr_scaleH, mdiskgas=mdiskgas,
					norm_x=norm_x, fracs_dustmdisk=fracs_dustmdisk,
					frac_dustovergasmass=frac_dustovergasmass,
					fracs_dustscaleH=fracs_dustscaleH, do_verbose=do_verbose)
	#
	#For simple density profile used in radmc3d benchmark
	elif which_density == "benchmark_radmc3d":
		#Print some notes
		if do_verbose:
			print("Running benchmark radmc3d scheme.")
		#
		RR = np.sqrt((matr_x**2) + (matr_y**2))
		#
		#Analytically estimate inner characteristics of disk from stellar char.
		temp_mid = calc_temperature_midplane(arr_x=matr_x, lstar=lstar,
										theta_flare=theta_flare) #[K]
		matr_scaleH = calc_scaleheight(arr_x=matr_x, temp_mid=temp_mid,
 						mstar=mstar, lstar=lstar) #[m]
		#
		dens_quick_raw = 1E-16 * np.exp(-(RR**2/(5*au0)**2)/2.0)
		dens_quick = dens_quick_raw / 1000 * (100**3) #[g/cm^3] -> [kg/m^3]
		conv_dustmasstonumH = 1.0*frac_dustovergasmass/mH0
		return {"volnumdens_nH":dens_quick*conv_dustmasstonumH,
		 		"volmassdens_dust_all":dens_quick,
				"volmassdens_dust_pergr":None, "matr_scaleH":matr_scaleH,
				"matr_sigma_gas":None, "matr_sigma_dust":None}
	#
	#Otherwise, throw error if density prescription not recognized
	else:
		raise ValueError("Whoa! Density {0} invalid!".format(which_density))
#


##Method: calc_density2D_analytic()
##Purpose: Calculate 2D gas,dust density using analytical equation
##Notes:
##	- Uses S.I. units
##	- chind = index of chosen grain population (the one with fixed given radius)
def calc_density2D_analytic(matr_x, matr_y, norm_x, mdiskgas, matr_scaleH, frac_dustovergasmass, fracs_dustmdisk, fracs_dustscaleH, do_verbose=False):
	num_popdust = len(fracs_dustscaleH) #Number of dust populations
	#Set global variables
	lowbound_numscaleH = preset.structure_gradient_HatomicandHmolec_lowscaleH
	highbound_numscaleH = preset.structure_gradient_HatomicandHmolec_highscaleH
	#
	#Print some notes
	if do_verbose:
		print("Running calc_density2D_analytic()!")
	#

	#Calculate mass density
	matr_sigma_gas = calc_densitysurf_analytic(mdiskgas=mdiskgas,
							matr_x=matr_x, norm_x=norm_x)
	matr_sigma_dust = (calc_densitysurf_analytic(mdiskgas=mdiskgas,
							matr_x=matr_x, norm_x=norm_x,
						) * frac_dustovergasmass)
	#

	#Calculate volume number-density for the gas [1/m^3]
	prefac_rho_gas = matr_sigma_gas/(np.sqrt(2.0*pi)*matr_scaleH)
	prefac_rho_dust = matr_sigma_dust/(np.sqrt(2.0*pi)*matr_scaleH)
	matr_rho_gas = prefac_rho_gas*np.exp(-0.5*((matr_y*1.0/matr_scaleH)**2))
	matr_nH = matr_rho_gas*1.0/mH0
	#
	#For atomic and molecular H distributions
	ylen = matr_nH.shape[0]
	xlen = matr_nH.shape[1]
	arr_lowscaleH = [(lowbound_numscaleH * matr_scaleH[0,xx])
					for xx in range(0, xlen)] #Min. scaleH bound for gradient
	arr_highscaleH = [(highbound_numscaleH * matr_scaleH[0,xx])
					for xx in range(0, xlen)] #Max. scaleH bound for gradient
	tmp_set = calc_density2D_Hatomicandmolec(nHtotal=matr_nH,
			 								arr_yval_low=arr_lowscaleH,
											arr_yval_high=arr_highscaleH,
											matr_y=matr_y,
											do_verbose=do_verbose)
	matr_nH_H2only = tmp_set["nH2"]
	matr_nH_Hatomiconly = tmp_set["nHatomic"]
	#

	#Calculate mass number-density for the dust [kg/m^3]
	prefac_dust = prefac_rho_dust #* frac_dustovergasmass Done earlier
	matr_dust_all = np.zeros(shape=matr_rho_gas.shape) #Hold combined dust pop.
	matr_dust_pergr = [None]*num_popdust #Hold individual dust populations
	for ii in range(0, num_popdust):
		curr_part = prefac_dust * fracs_dustmdisk[ii] / fracs_dustscaleH[ii]
		curr_exp = np.exp(-0.5*((matr_y*1.0/
								(fracs_dustscaleH[ii]*matr_scaleH))**2))
		matr_dust_pergr[ii] = curr_part * curr_exp
		matr_dust_all += matr_dust_pergr[ii]
	#

	#Return calculated densities
	if do_verbose:
		print("Run of calc_density2D_analytic() complete!")
	#
	return {"volnumdens_nH":matr_nH, "volnumdens_nH_H2only":matr_nH_H2only,
			"volnumdens_nH_Hatomiconly":matr_nH_Hatomiconly,
	 		"volmassdens_dust_all":matr_dust_all,
			"volmassdens_dust_pergr":matr_dust_pergr, "matr_scaleH":matr_scaleH,
			"matr_sigma_dust":matr_sigma_dust, "matr_sigma_gas":matr_sigma_gas}
#


##Method: calc_density2D_Hatomicandmolec()
##Purpose: Compute .
##Reference: See Bethell+2011, particularly Equation 2.
def calc_density2D_Hatomicandmolec(nHtotal, matr_y, arr_yval_low, arr_yval_high, do_verbose=False):
	#Print some notes
	if do_verbose:
		print("Running calc_density2D_Hatomicandmolec()!")
	#
	minval = 0 #Keep at 0
	maxval = 1
	#
	#Compute gradient from maxval to minval, going high to low
	matr_grad_tophigh = np.ones(shape=nHtotal.shape)*np.nan
	ylen = matr_grad_tophigh.shape[0]
	xlen = matr_grad_tophigh.shape[1]
	#Iterate through columns
	for xx in range(0, xlen):
		#Extract current y-bounds and gradient range
		yval_low = arr_yval_low[xx]
		yval_high = arr_yval_high[xx]
		tmp_inds = ((matr_y[:,xx] >= yval_low) & (matr_y[:,xx] <= yval_high))
		tmp_arr = np.ones(ylen)*np.nan
		tmp_arr[(matr_y[:,xx] < yval_low)] = minval
		tmp_arr[(matr_y[:,xx] >= yval_high)] = maxval
		tmp_arr[tmp_inds] = ((((matr_y[tmp_inds,xx] - yval_low)
								/(yval_high - yval_low)) * (maxval - minval))
							+ minval)
		#
		#Store the current gradient column
		matr_grad_tophigh[:,xx] = tmp_arr #tmparr
	#
	#Copy over and reverse direction of gradient
	matr_grad_toplow = (maxval - matr_grad_tophigh)
	#
	#Set gradients as final values
	nH2 = (matr_grad_toplow * nHtotal) / 2 #Factor of two since H2 = 2 H atoms
	nHatomic = (matr_grad_tophigh * nHtotal)
	#

	#Check for consistency
	is_outofbounds = ((matr_grad_toplow < minval) | (matr_grad_tophigh < minval)
				| (matr_grad_toplow > maxval) | (matr_grad_tophigh > maxval))
	if np.any(is_outofbounds):
		#Plot the results
		fig = plt.figure()
		ax0 = fig.add_subplot(2,2,1)
		ax1 = fig.add_subplot(2,2,2)
		ax2 = fig.add_subplot(2,2,3)
		ax3 = fig.add_subplot(2,2,4)
		cmap1 = plt.cm.bone_r
		cmap2 = plt.cm.PuBu
		ax0.imshow(matr_grad_toplow, origin="lower", cmap=cmap1)
		ax1.imshow(matr_grad_tophigh, origin="lower", cmap=cmap1)
		ax2.imshow(np.log10(nH2), origin="lower", cmap=cmap2)
		ax3.imshow(np.log10(nHatomic), origin="lower", cmap=cmap2)
		plt.show()
		#Raise the error
		raise ValueError("Whoa! Gradient beyond min,max range!".format(maxval))
	#
	elif (not np.allclose((matr_grad_toplow + matr_grad_tophigh), maxval)):
		#Plot the results
		fig = plt.figure()
		ax0 = fig.add_subplot(2,2,1)
		ax1 = fig.add_subplot(2,2,2)
		ax2 = fig.add_subplot(2,2,3)
		ax3 = fig.add_subplot(2,2,4)
		cmap1 = plt.cm.bone_r
		cmap2 = plt.cm.PuBu
		ax0.imshow(matr_grad_toplow, origin="lower", cmap=cmap1)
		ax1.imshow(matr_grad_tophigh, origin="lower", cmap=cmap1)
		ax2.imshow(np.log10(nH2), origin="lower", cmap=cmap2)
		ax3.imshow(np.log10(nHatomic), origin="lower", cmap=cmap2)
		plt.show()
		#Raise the error
		raise ValueError("Whoa! Rel split not conserved to {0}!".format(maxval))
	#
	elif (not np.allclose(((nH2*2) + nHatomic), nHtotal)):
		#Plot the results
		fig = plt.figure()
		ax0 = fig.add_subplot(2,2,1)
		ax1 = fig.add_subplot(2,2,2)
		ax2 = fig.add_subplot(2,2,3)
		ax3 = fig.add_subplot(2,2,4)
		cmap1 = plt.cm.bone_r
		cmap2 = plt.cm.PuBu
		tmp_image = ax0.imshow(matr_grad_toplow, origin="lower", cmap=cmap1)
		ax0.set_title("Gradient: Top Low")
		tmp_image = ax1.imshow(matr_grad_tophigh, origin="lower", cmap=cmap1)
		ax1.set_title("Gradient: Top High")
		tmp_image = ax2.imshow(np.log10(nH2), origin="lower", cmap=cmap2)
		ax2.set_title("H_molec")
		tmp_image = ax3.imshow(np.log10(nHatomic), origin="lower", cmap=cmap2)
		ax3.set_title("H_atomic")
		plt.show()
		#Raise the error
		raise ValueError("Whoa! Abs split not conserved to {0}!".format(maxval))
	#

	#Return the calculated atomic and molecular density distributions
	if do_verbose:
		print("Run of calc_density2D_Hatomicandmolec() complete!")
	#
	return {"nH2":nH2, "nHatomic":nHatomic}
#


##Method: calc_densitysurf_analytic()
##Purpose: Calculate analytic surface density profile.
def calc_densitysurf_analytic(mdiskgas, matr_x, norm_x):
	#Compute components of surface density profile
	prefac_sigma = mdiskgas/(2.0*pi*(norm_x**2))
	matr_sigma_base = (prefac_sigma * ((matr_x/norm_x)**-1)
		 				* np.exp(-1 * (matr_x/norm_x)))
	#
	matr_sigma_final = matr_sigma_base.copy()

	#Return completed surface density profile
	return matr_sigma_final
#


##Method: calc_distance()
##Purpose: Calculate distance between two points.
def calc_distance(p1, p2, order=None):
	return np.linalg.norm((np.asarray(p1) - np.asarray(p2)), ord=order)
#


##Method: calc_extinction_mag()
##Purpose: Calculate extinction in magnitudes for given spectra
def calc_extinction_mag(I_orig, I_new, do_mindenom):
	#If requested, set minimum denominator for spectra to avoid division by 0
	if do_mindenom:
		minval = preset.pnautilus_threshold_minval
		I_new_copy = I_new.copy()
		I_new_copy[I_new_copy <= minval] = minval
	else:
		I_new_copy = I_new
	#

	return ((100**(1/5)) * np.log10(I_orig/I_new_copy))
#


##Method: calc_tempgas_parametric_bruderer2012()
##Purpose: Calculate parametric gas temperature profile, with consistency with dust temperature profile, following Bruderer+2012 write-up
##Reference: Bruderer+2012 write-up
def calc_tempgas_parametric_bruderer2012(matr_tempdust, matr_nH, matr_energyflux_UVtot, arr_x=None, arr_y=None, matr_x=None, matr_y=None):
	##Extract global variables
	if (matr_x is None) and (matr_y is None):
		matr_x, matr_y = np.meshgrid(arr_x, arr_y)
	len_y = matr_y.shape[0]
	len_x = matr_y.shape[1]
	#

	##Calculate parametrized difference between gas and dust temperatures
	#Calculate components
	x = np.log10((matr_nH/(1E11))) #(n_H / 1E5 cm^-3), -> in m^-3
	matr_G0 = (matr_energyflux_UVtot / conv_energyfluxtoG0)
	prefac = 420.2 #[K]
	delta_maxval = 4200 #[K]; maximum allowed difference for parametrization
	#
	qval = (1.05 - (0.113*x))
	matr_tenpower = (1 / (10**((0.486*x) - (0.014*(x*x)))))
	#
	#Assemble temperature difference
	matr_deltaT = (prefac * ((matr_G0/1E3)**qval) * matr_tenpower)
	matr_deltaT[matr_deltaT > delta_maxval] = delta_maxval
	#

	##Compute and return the parametrized gas temperature
	matr_tempgas = (matr_tempdust + matr_deltaT)
	return matr_tempgas
#


##Method: calc_tempgas_parametric_synthetic()
##Purpose: Calculate synthetic parametric gas temperature profile, with consistency with dust temperature profile
##Reference: Rosenfeld+2013b,2013a
def calc_tempgas_parametric_synthetic(matr_r, matr_z, lstar, mstar, theta_flare, matr_tempdust):
	#Extract global variables
	tempatm_T0 = preset.empirical_constant_temperature_atmosphere_prefactor_K
	tempatm_q0 = preset.empirical_constant_temperature_atmosphere_exponent
	tempatm_r0 = preset.empirical_constant_temperature_atmosphere_meters
	zq_scaler = preset.empirical_constant_temperature_zq0_scaler
	val_delta = preset.empirical_constant_temperature_delta

	#Compute radial temperature and scale height profiles
	matr_tempmid = calc_temperature_midplane(
		arr_x=matr_r, lstar=lstar, theta_flare=theta_flare
	) #[K]
	matr_tempatm = tempatm_T0 * ((matr_r / tempatm_r0)**(-tempatm_q0)) #[K]
	matr_scaleH = calc_scaleheight(
		arr_x=matr_r, mstar=mstar, temp_mid=matr_tempmid, lstar=lstar
	)
	zq0 = (zq_scaler * matr_scaleH)

	#Compute full parametric gas temperature profile
	matr_tempgas = np.ones(shape=matr_r.shape)*np.nan
	matr_tempgas[matr_z < zq0] = (
		matr_tempatm[matr_z < zq0]
		+ (((matr_tempmid - matr_tempatm)[matr_z < zq0])
			*((np.cos(pi*matr_z[matr_z < zq0]/(2*zq0[matr_z < zq0]))
			)**(2*val_delta))
		)
	)
	matr_tempgas[matr_z >= zq0] = matr_tempatm[matr_z >= zq0]
	matr_tempgas[matr_tempgas < matr_tempdust] = (
		matr_tempdust[matr_tempgas < matr_tempdust]
	)

	#Return the computed parametric gas temperature
	return matr_tempgas
#


##Method: calc_scaleheight()
##Purpose: Calculate scale height using analytical equation
def calc_scaleheight(arr_x, mstar, temp_mid, lstar):
	muval0 = preset.chemistry_value_muval0
	return np.sqrt(k_B0*temp_mid*(arr_x**3)/(G0*mstar*muval0*mH0)) #[m]
#


##Method: _calc_normx()
##Purpose: Calculate normalization x-value
def _calc_normx(which_normx, R_out, frac=None):
	#For empirical approach
	if which_normx == "empirical":
		#Load empirical data
		dataset = np.genfromtxt(preset.empirical_filename_RcvsRout,
								comments="#").astype(float)
		lit_R_c = dataset[:,2]
		lit_R_c_err = dataset[:,3]
		lit_R_out = dataset[:,4]
		lit_R_out_err = dataset[:,5]
		#Fit empirical data
		res_fitRc = _fit_linear_weighted(xdata=np.log10(lit_R_out),
								ydata=np.log10(lit_R_c),
								yerr=(lit_R_c_err/(1.0*lit_R_c*np.log(10))),
								xerr=(lit_R_out_err/(1.0*lit_R_out*np.log(10))))
		#Linearly fit for R_c
		return (((1.0*R_out/au0)**res_fitRc["abest"])
				* (10**res_fitRc["bbest"])) * au0
	#
	#For fraction approach
	elif which_normx == "fraction":
		return (frac * R_out)
	else:
		raise ValueError("Whoa! Norm-x scheme {0} not recognized!"
							.format(which_normx))
#


##Method: _calc_photorates_base()
##Purpose: Base function for computing photorates for different radiation fields.
def _calc_photorates_base(mode, x_unified, y_spec, y_cross, dict_molinfo, ionrate_X_primary=None, react_dict=None, y_prod_theta=None, axis=-1, delta_energy=None):
	#Call photorate routines based on specified mode
	#For UV photorates
	if (mode == "UV"):
		return _calc_photorate_UV(y_spec=y_spec, y_cross=y_cross,
			 					y_prod_theta=y_prod_theta, x_unified=x_unified,
								axis=axis)
	#
	#For X-ray primary ionization rate and induced-UV ph.rates
	elif (mode in ["X_primary"]):
		#Compute primary X-ray ionization rate
		if (ionrate_X_primary is None):
			ionrate_X_primary = _calc_photorate_X_primary(x_unified=x_unified,
		 						y_spec_photon_perm=y_spec, y_cross_inm=y_cross,
								axis=axis, delta_energy=delta_energy)
		#
		if (mode == "X_primary"):
			return ionrate_X_primary
		#
	#
	#Otherwise, throw error if mode not recognized!
	else:
		raise ValueError("Whoa! Unrecognized mode {0}!".format(mode))
	#
#


##Method: _calc_photorate_UV()
##Purpose: Perform isolated calculation of UV photochemistry rate for spectrum and cross-section info
def _calc_photorate_UV(y_spec, y_cross, y_prod_theta, x_unified, axis=None, do_return_accum=False):
	#Integrate along axis, if specified
	if (axis is not None):
		val_rate = np.trapz(y=(y_spec*y_cross*y_prod_theta),
		 					x=x_unified, axis=axis)
		val_accum = scipy_integrate.cumtrapz(y=(y_spec*y_cross*y_prod_theta),
		 									x=x_unified, axis=axis)
	#Otherwise, integrate straight as given
	else:
		val_rate = np.trapz(y=(y_spec*y_cross*y_prod_theta), x=x_unified)
		val_accum = scipy_integrate.cumtrapz(y=(y_spec*y_cross*y_prod_theta),
		 									x=x_unified)
	#

	#Return the integrated value or accumulated integral
	if (do_return_accum):
		return val_accum
	else:
		return val_rate
	#
#


##Method: _calc_photorate_X_primary()
##Purpose: Perform isolated calculation of X-ray primary photochemistry rates for spectrum and cross-section info
##Notes: See also helpful lecture notes, Eq. 58, by Gudel+2015: https://www.epj-conferences.org/articles/epjconf/pdf/2015/21/epjconf_ppd2014_00015.pdf
##See Glassgold+1997 for derivation
##NOTE: Assuming average Auger energy ~ ionization energy
##Also requires input spectrum to have already been attenuated
def _calc_photorate_X_primary(x_unified, y_spec_photon_perm, y_cross_inm, delta_energy, axis=None):
	#Convert quantities to energy
	energy_unified = (h0 * c0 / x_unified) #[m] -> [J]
	y_spec_photon_perJ = _conv_spectrum(which_conv="perwave_to_perenergy",
									spectrum=y_spec_photon_perm, wave=x_unified)
	#

	#Compute primary ionization rate
	tmp_y_input = (y_spec_photon_perJ * y_cross_inm #y_cross_inJ
	 				* (energy_unified / delta_energy))
	rate_X_primary = np.trapz(x=energy_unified, y=tmp_y_input, axis=axis)
	#

	#Flip sign if descending energies
	if all([(energy_unified[ii-1] >= energy_unified[ii])
				for ii in range(1, len(energy_unified))]):
		rate_X_primary = -1*rate_X_primary
	#

	#Throw error if any rates negative
	if np.any((rate_X_primary < 0)):
		raise ValueError("Whoa! Negative primary X-ray rates?\n{0}"
						.format(rate_X_primary))
	#

	#Return the computed primary X-ray ionization rate
	return rate_X_primary
#


##Method: calc_temperature_midplane()
##Purpose: Calculate midplane temperature using analytical equation
def calc_temperature_midplane(arr_x, lstar, theta_flare):
	return ((theta_flare*lstar/(8*pi*(arr_x**2)*sigma_SB0))**0.25) #[K]
#


##Method: calc_radius_of_temperature_midplane()
##Purpose: Calculate radius corresponding to midplane temperature analytically
def calc_radius_of_temperature_midplane(temp, lstar, theta_flare):
	return np.sqrt((theta_flare*lstar/(8*pi*(temp**4)*sigma_SB0))) #[m]
#


##Method: _calc_bbody()
##Purpose: Calculate midplane temperature using analytical equation
def _calc_bbody(temp, freq, wave, which_mode, which_per):
	#Calculate bbody as function of given denominator
	if which_per == "wavelength":
		part_exp = (np.exp((h0*c0)/(k_B0*wave*temp)) - 1.0)
		res = (((2.0*(h0*(c0*c0)))/(wave**5)) / part_exp
				) #[W/(m^2 ster m)]
	elif which_per == "frequency":
		part_exp = (np.exp((h0*freq)/(k_B0*temp)) - 1.0)
		res = (((2.0*(freq*freq))/(c0*c0)) * ((h0*freq)/part_exp)
				) #[W/(m^2 ster Hz)]
	else:
		raise ValueError("Whoa! Denominator {0} invalid!".format(which_per))
	#
	#Return bbody in desired form
	if which_mode == "intensity": #[W/(m^2 ster Hz)]
		return res
	elif which_mode == "spectrum": # -> #[W/(m^2 Hz)]
		return (4*pi*res)
	else:
		raise ValueError("Whoa! Mode {0} not recognized!".format(which_mode))
#


##Method: _calc_slope()
##Purpose: Calculate the slope between two 2D points
def _calc_slope(x_1, x_2, y_1, y_2):
	return ((y_2 - y_1)/(x_2 - x_1))
#


##Method: _calc_cellcenters()
##Purpose: Determine centers of cells given their edges in array
def _calc_cellcenters(arr):
	return np.array([((arr[ii-1] + arr[ii])/2.0)
					for ii in range(1, len(arr))])
#


##Method: _calc_scatt_Thomson()
##Purpose: Calculate Thomson scattering for given particle
def _calc_scatt_Thomson(q, m):
	prefac = (8 * pi / 3.0)
	partsq_top = (q * q)
	partsq_bot = (4 * pi * eps0 * m * (c0 * c0))
	return (prefac * ((partsq_top / partsq_bot)**2))
#


##Method: _conv_distanddist()
##Purpose: Convert flux between two distances
def _conv_distanddist(flux, dist_old, dist_new):
	#Calculate and return flux at new distance
	return (flux * ((dist_old/dist_new)**2))
#


##Method: _conv_lumandflux()
##Purpose: Convert between wavelength and frequency
def _conv_lumandflux(which_conv, lum=None, flux=None, dist=None):
	part_dist = (4.0*pi*dist*dist)
	if which_conv == "fluxtolum":
		return (flux*part_dist)
	elif which_conv == "lumtoflux":
		return (lum/part_dist)
	else:
		raise ValueError("Whoa! Invalid conv={0} requested!".format(which_conv))
#


##Method: _conv_lumandTeff()
##Purpose: Convert between luminosity and effective temperature
def _conv_lumandTeff(which_conv, dist, lstar=None, Teff=None):
	#Convert from luminosity to effective temperature
	if which_conv == "lum_to_Teff":
		return ((lstar/(4*pi*(dist*dist)*sigma_SB0))**(0.25))
#


##Method: _conv_spectrum()
##Purpose: Convert spectrum between various units
def _conv_spectrum(which_conv, spectrum, wave=None, freq=None):
	#Throw an error if neither wavelength nor frequency given
	if (wave is None) and (freq is None):
		raise ValueError("Whoa! Please pass in either wavelength or frequency.")
	#
	#For conversion from 1/wavelength to 1/frequency
	if which_conv == "perwave_to_perfreq":
		if freq is None:
			freq = (c0 / wave)
		return (spectrum / (freq * freq / c0))
	#
	#For conversion from 1/frequency to 1/wavelength
	elif which_conv == "perfreq_to_perwave":
		if freq is None:
			freq = (c0 / wave)
		return (spectrum * (freq * freq / c0))
	#
	#For conversion from 1/wavelength to 1/energy
	elif which_conv == "perwave_to_perenergy":
		if wave is None:
			wave = (c0 / freq)
		return (spectrum * (wave / (h0 * c0 / wave)))
	#
	#For conversion from energy to photon spectrum
	elif which_conv == "energy_to_photon":
		if wave is not None:
			return (spectrum / (h0 * (c0 / wave)))
		elif freq is not None:
			return (spectrum / (h0 * freq))
	#
	#For conversion from energy to photon spectrum
	elif which_conv == "photon_to_energy":
		if wave is not None:
			return (spectrum * (h0 * (c0 / wave)))
		elif freq is not None:
			return (spectrum * (h0 * freq))
	#
	#Throw an error if conversion not recognized
	else:
		raise ValueError("Whoa! {0} not recognized!".format(which_conv))
#


##Method: _conv_sphericalandcartesian()
def _conv_sphericalandcartesian(which_conv, radius=None, theta=None, x=None, y=None):
	#Convert from spherical to Cartesian coordinates
	if which_conv == "sphericaltocartesian":
		return {"x":(radius * np.cos(theta)), "y":(radius * np.sin(theta))}
	#
	#Throw error if conversion not recognized
	else:
		raise ValueError("Whoa! Conversion method {0} not recognized!"
						.format(which_conv))
#


##Method: _conv_waveandfreq()
##Purpose: Convert between wavelength and frequency
def _conv_waveandfreq(which_conv, wave=None, freq=None, energy=None):
	#Do standard conversion ahead of time
	if ((wave is None) and (energy is None)):
		wave = (c0 / freq)
	#
	#Return requested conversion
	if which_conv == "wavetofreq":
		return (c0 / wave)
	elif which_conv == "freqtowave":
		return (c0 / freq)
	elif which_conv == "wavetoenergy":
		return (h0 * c0 / wave)
	elif which_conv == "energytowave":
		return (h0 * c0 / energy)
	else:
		raise ValueError("Whoa! Invalid conv={0} requested!".format(which_conv))
#


##Method: _interpolate_pointstogrid()
##Purpose: Interpolate set of unstructured points onto a regular grid
def _interpolate_pointstogrid(old_matr_values, old_points_yx, new_matr_x, new_matr_y, inds_valid, do_noextrap=True):
	#Interpolate values from unstructured points onto new grid
	new_matr_values = scipy_interptogrid(
					old_points_yx, old_matr_values.flatten(),
	 				(new_matr_y, new_matr_x), method="linear", rescale=True,
					fill_value=np.nan)
	#
	#Throw error if any extrapolated values, if so requested
	if do_noextrap:
		quick_check = new_matr_values
		if (inds_valid is not None):
			quick_check = quick_check[inds_valid]
		if any(np.isnan(quick_check.flatten())):
			print(inds_valid)
			print("")
			print(np.nanmin(new_matr_x))
			print(np.nanmax(new_matr_x))
			print("")
			print(np.nanmin(new_matr_y))
			print(np.nanmax(new_matr_y))
			print("")
			print(np.nanmin(old_points_yx))
			print(np.nanmax(old_points_yx))
			print("")
			tmp_x = np.zeros(shape=new_matr_x.shape)
			tmp_y = np.zeros(shape=new_matr_y.shape)
			tmp_x[inds_valid] = 1
			tmp_y[inds_valid] = 1
			plt.contourf(new_matr_x/au0, new_matr_y/au0, tmp_x, origin="lower")
			plt.scatter(new_matr_x/au0, new_matr_y/au0,color="tomato",alpha=0.5)
			plt.scatter(old_points_yx[:,1]/au0, old_points_yx[:,0]/au0,
						alpha=0.25, color="black")
			plt.show()
			plt.contourf(new_matr_x/au0, new_matr_y/au0, new_matr_values,
			 			origin="lower")
			plt.show()
			raise ValueError("Whoa! Extrapolated values in interpolated grid!")
	#
	#Return the interpolated values
	return new_matr_values
#


##Method: _find_inds_outernonzeros()
##Purpose: Find indices of outermost non-zero values (e.g., to use in removal of leading or trailing zeros).
def _find_inds_outernonzeros(arr, include_outerzeros):
	#Fetch indices of non-zero values
	arr_inds = np.arange(0, len(arr), 1)
	inds_nonzero = arr_inds[np.asarray(arr) != 0]
	#
	if len(inds_nonzero) == 0: #If no nonzero values
		return None
	#

	#Below includes or excludes outermost zeros
	if include_outerzeros:
		#NOTE: Below has +/-1 because must include leftmost/rightmost first trailing zero; otherwise incorrect integration over space between nonzero and zero values
		ind_left = np.max([(np.min(inds_nonzero)-1), 0])
		ind_right = np.min([(np.max(inds_nonzero)+1), (len(arr)-1)])
	else:
		ind_left = np.min(inds_nonzero)
		ind_right = np.max(inds_nonzero)
	#

	#Throw error if something went wrong/negative
	if ((ind_left < 0) or (ind_right < 0)):
		raise ValueError("Whoa! Bad removal of leading zeros!")
	#

	#Return the outermost indices
	return {"ind_left":ind_left, "ind_right":ind_right}
#


##Method: _fit_linear_weighted()
##Purpose: Calculate parameters for a linear fit (y=ax+b) to the given data; weighted given errors in y (and x if needed).  Follows approach of York+ 2004 (canvas.harvard.edu/courses/47513/files/7351005).  rerr=<correlation in errors>.
def _fit_linear_weighted(xdata, ydata, yerr=None, xerr=None, rerr=0.0, athres=1E-10, maxiter=100):
	#Assume uniform error, if none given
	if yerr is None:
		yerr = np.ones(len(ydata))

	if xerr is None:
		#Below prepares some summations
		xoe2 = np.sum(xdata/1.0/(yerr**2))
		x2oe2 = np.sum((xdata**2)/1.0/(yerr**2))
		yoe2 = np.sum(ydata/1.0/(yerr**2))
		xyoe2 = np.sum(xdata*ydata/1.0/(yerr**2))
		oe2 = np.sum(1.0/(yerr**2))

		#Below calculates best-fitting values based on error
		abest = ((xoe2*yoe2) - (xyoe2*oe2))/1.0/((xoe2**2) - (x2oe2*oe2)) #Slope
		bbest = (xyoe2 - (abest*x2oe2))/1.0/xoe2 #For y-intercept

		#Below calculates error in best-fitting values
		aerr = np.sqrt(oe2/1.0/((x2oe2*oe2) - (xoe2**2))) #Slope error
		berr = np.sqrt(x2oe2/1.0/((x2oe2*oe2) - (xoe2**2)))

		#Below returns the calculated values
		return {"abest":abest, "bbest":bbest, "aerr":aerr, "berr":berr}

	##Below Section: Sets up initial values for iteration if x-errors given
	#Calculate weights for x and y-data
	xweights = 1.0/(xerr**2)
	yweights = 1.0/(yerr**2)
	#Estimate initial value for slope (a)
	aold = np.polyfit(xdata, ydata, 1)[0]
	anew = np.inf

	#Iteratively estimate slope
	iteri = 0
	while (np.abs(aold - anew) > athres):
		#Update values
		if iteri > 0:
			aold = anew #Update slope estimate if not 0th iteration
		#
		Wmain = (xweights*yweights)/1.0/(xweights + ((aold**2)*yweights)
					- (2*rerr*aold*np.sqrt(xweights*yweights)))
		Xmain = np.sum(Wmain*xdata)/1.0/np.sum(Wmain)
		Ymain = np.sum(Wmain*ydata)/1.0/np.sum(Wmain)
		Umain = xdata - Xmain
		Vmain = ydata - Ymain
		#
		betavals = ((Umain/1.0/yweights) + (aold*Vmain/1.0/xweights)
			+ (((aold*Umain) + Vmain)*rerr/1.0/np.sqrt(xweights*yweights)))
		anew = np.sum(Wmain*betavals*Vmain)/1.0/np.sum(Wmain*betavals*Umain)
		#Update iteration count
		iteri += 1
		if iteri > maxiter:
			raise ValueError("Whoa!  Too many iterations!")
	#Estmate y-int. (b) and errors in a and b
	bval = Ymain - (Xmain*anew)
	xbar = np.sum(Wmain*(Xmain+betavals))/1.0/np.sum(Wmain)
	ubar = (Xmain + betavals) - xbar
	aerr = 1.0/np.sum(Wmain*(ubar**2))
	berr = np.sqrt((1.0/np.sum(Wmain)) + ((xbar**2)*(aerr**2)))

	#Return solutions
	return {"abest":anew, "bbest":bval, "aerr":aerr, "berr":berr}
#


##Method: make_profile_kappa_scattLya()
##Purpose: Generate kappa scattering profile for Lya propagation
##Reference: See Bethell+2011, particularly page 8; and see Laursen+2007, page 2
##	: See also for unit help (particularly with e in esu, aiyah: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S1323358014000332, or https://ui.adsabs.harvard.edu/abs/2014PASA...31...40D/abstract
def make_profile_kappa_scattLya(wave_arr, do_plot=False, filepath_plot=None, filename_plot=None):
	#Set routine variables
	if (filename_plot is None):
		filename_plot = "fig_math_crosssec_scattLya.png"
	wavecen_Lya = preset.waveloc_Lya
	temperature = preset.gastemperature_Lya
	f_12 = preset.profile_crossLya_oscillatorstrength
	natlinewidth_freq = preset.profile_crossLya_naturallinewidth_infreq
	nucen_Lya = _conv_waveandfreq(which_conv="wavetofreq", wave=wavecen_Lya)
	vtherm_H = np.sqrt(((2*k_B0*temperature)/(mH0)))
	delta_nuD = (nucen_Lya * vtherm_H)/c0
	#

	#Calculate and assemble profile components
	aval = natlinewidth_freq / (2*delta_nuD)
	numerator = (f_12 * (e0_statSI * e0_statSI) * np.sqrt(pi))
	denominator = (me0 * c0 * delta_nuD)
	xparam_arr = ((c0/wave_arr) - nucen_Lya)/delta_nuD
	voigt_raw = make_profile_voigt(x_arr=xparam_arr, aval=aval)
	voigt = voigt_raw
	#
	prefac = (numerator / denominator)
	y_arr_raw = (prefac * voigt)
	sigma_arr = y_arr_raw
	#

	#Convert the cross-section into kappa
	kappa_arr = (sigma_arr / mH0) #[m^2/per H nucleus] -> [m^2/(kg of H)]

	#Plot the profile, if so requested
	if do_plot:
		doppler_arr = xparam_arr
		#Plot the profile in Bethell+2011 convention
		fig = plt.figure(figsize=(10,5))
		axs = fig.subplots(1, 2)
		ax0 = axs[0]
		ax0.plot(doppler_arr, (sigma_arr*(100*100)), marker="o", alpha=0.5)
		#Scale and label the plot
		ax0.set_xlim([-400, 400])
		ax0.set_yscale("log")
		ax0.set_xlabel(r"Doppler Units (($\nu$ - $\nu_0$)/$\nu_\mathrm{D}$)")
		ax0.set_ylabel(r"Cross-Section (cm$^2$ / H nucleus)")
		ax0.set_title(
			r"Lya Scattering Cross-Section (e.g. Bethell+2011)"
			+f"\n:Prefac: {prefac:.2e}; aval: {aval:.2e}"
		)
		#
		ax0 = axs[1]
		ax0.plot(wave_arr*1E10, voigt)
		ax0.axvline(wavecen_Lya*1E10, color="purple", linestyle="--", alpha=0.5)
		ax0.set_xlim(
			[((wavecen_Lya-0.05E-10)*1E10), ((wavecen_Lya+0.05E-10)*1E10)]
		)
		ax0.set_xlabel("Wavelength (nm)")
		ax0.set_ylabel("Voigt")
		#
		plt.tight_layout()
		plt.savefig(os.path.join(filepath_plot, filename_plot))
		plt.close()
	#

	#Return assembled profile
	return kappa_arr
#


##Method: make_profile_kappa_scattX()
##Purpose: Generate kappa scattering profile for X-ray propagation
##Assumed to be electron Thomson scattering, flat with wavelength
def make_profile_kappa_scattX(x_arr):
	sigma_scatt_Thomson_electron = _calc_scatt_Thomson(q=e0, m=me0)
	kappa_scatt_Thomson_electron = (sigma_scatt_Thomson_electron / me0)
	kappa_scatt_Thomson_perH = (kappa_scatt_Thomson_electron * (me0/mH0))
	return (np.ones(shape=x_arr.shape) * kappa_scatt_Thomson_perH)
#


##Method: make_profile_voigt()
##Purpose: Generate Voigt profile using given input parameters
def make_profile_voigt(x_arr, aval):
	#Set global function variables
	bound_min = -np.inf
	bound_max = np.inf

	#Define the function to integrate
	def integrand(y, a, x):
		numerator = np.exp(-(y*y))
		denominator = (((x - y)**2) + (a*a))
		return (numerator / denominator)
	#

	#Compute the integral
	result_raw = np.array(
		[scipy_integrate.quad(
			integrand, bound_min, bound_max, args=(aval, item)
		) for item in x_arr]
	)[:,0]
	result = ((aval / pi) * result_raw)

	#Return the integrated result
	return result
#


##Method: _remove_adjacent_redundancy()
##Purpose: Reduce the unnecessary high-resolution of a given array by removing values that are in between identical values; meant to be used to, e.g., remove points in a wavelength-dependent cross-section that are identical to left,right neighboring points and are thus unnecessarily padding the resolution
def _remove_adjacent_redundancy(y_old, x_old, do_trim_zeros, do_verbose_plot=False):
	#Keep only indices of points that are unique with respect to both neighbors
	inds_mid = np.array([ii for ii in range(0, len(y_old))
			if ((ii == 0) or (ii == (len(y_old) - 1))
			 	or ((ii > 0) and (y_old[ii] != y_old[ii-1]))
				or ((ii < (len(y_old) - 1)) and (y_old[ii] != y_old[ii+1])))])
	#Apply the indices
	y_mid = y_old[inds_mid]
	x_mid = x_old[inds_mid]

	#Trim leading and trailing zeros, if requested
	if do_trim_zeros:
		#Fetch indices of outermost nonzeros
		tmp_res = _find_inds_outernonzeros(arr=y_mid, include_outerzeros=True)
		if (tmp_res is None):
			return None
		#
		ind_left = tmp_res["ind_left"]
		ind_right = tmp_res["ind_right"]
		#
		x_trim = x_mid[ind_left:ind_right+1]
		y_trim = y_mid[ind_left:ind_right+1]
	#
	else:
		x_trim = x_mid
		y_trim = y_mid
	#


	#Relabel variables (convention of old, removed section of this function)
	x_new = x_trim
	y_new = y_trim
	#

	#Make some verification plots
	if do_verbose_plot:
		plt.figure(figsize=(10,5))
		plt.scatter(x_old*1E9, y_old, marker="*", linewidth=1,
		 			color="salmon", s=300, alpha=0.75)
		plt.plot(x_new*1E9, y_new, marker="o", linewidth=3,
		 			linestyle="--", color="navy", alpha=0.5)
		plt.yscale("log")
		plt.show()
	#

	#Return the streamlined indices and associated arrays
	return {"x":x_new, "y":y_new}
#


##Method: _unify_spectra_wavelengths()
##Purpose: Unify a given list of y-arrays over a combined, unified x-array using given overlap scheme.
def _unify_spectra_wavelengths(x_list, y_list, which_overlap, fill_value=None, axis=-1):
	#Throw error if requested overlap scheme is not recognized
	if (which_overlap not in ["minimum", "maximum"]):
		raise ValueError("Whoa! {0} not accepted overlap scheme!"
						.format(which_overlap))
	#

	#Throw error if unequal number of wavelength vs spectra arrays
	if (len(x_list) != len(y_list)):
		raise ValueError("Whoa! Unequal count of wavelength vs spectra arrays!")
	#

	#Determine combined, unique wavelengths based on overlap scheme
	#For minimum overlap
	if (which_overlap == "minimum"):
		x_min = np.max([np.min(item) for item in x_list])
		x_max = np.min([np.max(item) for item in x_list])
		x_all = [item[((item >= x_min) & (item <= x_max))] for item in x_list]
		x_all = np.sort(np.unique(np.concatenate(x_all)))
		#Interpolate each spectrum over the combined wavelength set
		y_new = [scipy_interp1d(x=x_list[ii], y=y_list[ii],
								axis=axis, bounds_error=True)(x_all)
				for ii in range(0, len(x_list))]
	#
	#For maximum overlap with fill-in value
	elif (which_overlap == "maximum"):
		x_min = np.min([np.min(item) for item in x_list])
		x_max = np.max([np.max(item) for item in x_list])
		x_all = [item[((item >= x_min) & (item <= x_max))] for item in x_list]
		x_all = np.sort(np.unique(np.concatenate(x_all)))
		#Interpolate each spectrum over the combined wavelength set
		y_new = [scipy_interp1d(x=x_list[ii],y=y_list[ii], axis=axis,
	 						bounds_error=False, fill_value=fill_value)(x_all)
				for ii in range(0, len(x_list))]
	#

	#Return the unified wavelength points and trimmed, interpolated spectra
	return {"x":x_all, "y":y_new}
#


##Method: _use_profile_empirical()
##Purpose: Use generic empirical profile to scale given value
def _use_profile_empirical(x_val, x_c, a_c, q_c):
	return (a_c * ((1.0*x_val/x_c)**q_c))
#

















#
