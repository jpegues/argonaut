###FILE: argonaut.py
###PURPOSE: Script for preparing, generating, and running argonaut, a package for creating 2D astrochemical disk models using the radiative transfer code RADMC-3D and the gas-grain chemical network code Nautilus.
###DATE CREATED: 2022-05-24
###DEVELOPERS: (Jamila Pegues; using Nautilus from Nautilus developers and RADMC-3D from RADMC-3D developers)


##Below Section: Imports necessary functions
import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import cmasher
import time as timer
plt.close()
from sklearn.neighbors import NearestNeighbors as sklearn_nearestneighbors
from scipy.interpolate import interp1d as scipy_interp1d
try:
	import subprocess
except ModuleNotFoundError:
	import subprocess32 as subprocess
#Custom files
import argonaut_constants as preset
import argonaut_math as math
import argonaut_utils as utils
import argonaut_rates as reacts
#Set astronomical constants
e0 = const.e.value #[C in S.I. units]
au0 = const.au.value #[m]
pc0 = const.pc.value #[m]
me0 = const.m_e.value #[kg]
mH0 = (const.m_p.value + me0) #Mass of H atom; [kg]
Rsun0 = const.R_sun.value #[m]
conv_Jtoerg0 = 1E7 #[J] -> [erg]
conv_eVtoJ0 = e0
conv_specSItocgs0 = conv_Jtoerg0 / (100*100) #[W/(m^2*Hz)] -> [erg/(s*cm^2*Hz)]
conv_perm2topercm2 = 1.0 / (100**2) #[m^-2] -> [cm^-3]
conv_perm3topercm3 = 1.0 / (100**3) #[m^-3] -> [cm^-3]
conv_kgtog0 = 1000 #[kg] -> [g]
pi = np.pi
#


##Class: _Base
class _Base():
	"""
	Class: _Base
	Purpose:
		- Container for common underlying methods used in other classes
		- To be inherited by other classes
	Inputs:
		- None
	"""
	##Method: __init__
	##Purpose: Initialize this class instance
	def __init__(self):
		#Nothing to see here
		pass
	#

	##Method: get_info()
	##Purpose: Retrieve specified data via given key
	def get_info(self, key):
		#Attempt to retrieve data stored under the given key
		try:
			return self._storage[key]
		except KeyError:
			raise KeyError("Whoa there.  Looks like you requested data from a "
					+"key ({1}) that does not exist.  Available keys are:\n{0}"
					.format(sorted(self._storage.keys()), key))

		#Close method
		return
	#

	##Method: _store_info()
	##Purpose: Store given data into class instance in a unified way
	def _store_info(self, data, key):
		#Store the data into underlying dictionary
		self._storage[key] = data

		#Close method
		return
	#
#


##Class: Basket
##Purpose: Class for methods to plot traits and chemistry of set of disk models
class Basket(_Base):
	##Method: __init__()
	##Purpose: Initialization of this class instance
	def __init__(self, conchs, do_save_processed_chemistry_output=True, do_allow_load_processed_chemistry_output=True, do_verbose=False):
		#Load set of models, if locations given
		if isinstance(conchs[0], str):
			tmp_list = [None]*len(conchs)
			#Load each model
			for ii in range(0, len(conchs)):
				tmp_list[ii] = Conch_Disk(do_verbose=do_verbose, do_save=False,
				 			dict_grid=None, dict_names=None, dict_stars=None,
							dict_medium=None,
							dict_chemistry=None, loc_conch=conchs[ii],
							num_cores=None, mode_testing=False)
			#
			#Copy over model list
			conchs = tmp_list
		#

		#Store set of models
		self._storage = {}
		self._store_info(do_verbose, "do_verbose")
		self._store_info(conchs, "list_conchs")
		self._store_info(do_allow_load_processed_chemistry_output,
		 				"do_allow_load")
		self._store_info(do_save_processed_chemistry_output,
		 				"do_save_processed_chemistry_output")
		names_conchs = [item.get_info("dict_names")["model"]
						for item in conchs]
		self._store_info(names_conchs, "names_conchs")
		#
		#Print some notes
		if do_verbose:
			print("\n---\nInitialized Basket of conchs.")
			print("Stored disk models:\n{0}".format(names_conchs))
		#
		return
		#
	#


	##Function: plot_chemistry_1D()
	##Purpose: Plot comparison of disk chemistry across column densities
	def plot_chemistry_1D(self, which_year, which_dir, ybounds_zoverr, rtol_zoverr=None, cutoff_minval=None, rtol_time=0.1, model_titles=None, conch_norms=None, which_params=None, which_order=None, rmxticktitle=False, rmyticktitle=False, ncol=4, nxticks=6, xmin_val=None, xmin_frac=None, xmax=None, xlim=None, ylim=None, ymin=None, doxlog=False, doylog=False, doyspan_norm=False, yspan_norm=None, ylinecol_norm="black", ylinesty_norm="--", yspancol_norm="silver", yspanalpha_norm=0.5, dict_col_scaler=None, tbmargin=0.075, lrmargin=0.075, alpha=0.5, boxalpha=0.0, boxx=0.05, boxy=0.95, ticksize=16, textsize=18, titlesize=20, tickwidth=3, tickheight=5, tickspace=5, labelpad=1.5, filepath_save=None, figsize=(30,10), tbspace=0.25, lrspace=0.25, ext_name=None, do_normthenorm=False, inds_xticklabel=None, inds_yticklabel=None, fig_xlabel="Radius (au)", labels=None, colors="black", markers="", styles="-", markersizes=5, linewidths=3, legendind=None, legendloc="best", legendhandlelength=1, legendlabelspacing=0.25, do_verbose=None):
		##Extract global variables
		yspancol = yspancol_norm
		yspanalpha = yspanalpha_norm
		ylinecol = ylinecol_norm
		ylinesty = ylinesty_norm
		if (do_verbose is None):
			do_verbose = self.get_info("do_verbose")
		#
		#Print some notes
		if do_verbose:
			print("> Plotting with plot_chemistry_1D!")
			time_start = timer.time()
		#

		##Extract model characteristics and chemistry holders
		conchs = self.get_info("list_conchs")
		names = self.get_info("names_conchs")
		#
		do_allow_load = self.get_info("do_allow_load")
		do_save = self.get_info("do_save_processed_chemistry_output")
		conchs_chem = [conchs[ii].get_info("model_chemistry")
		 				for ii in range(0, len(conchs))]
		if (conch_norms is not None):
			if isinstance(conch_norms[0], str):
				conch_norms = [Conch_Disk(do_verbose=do_verbose, do_save=False,
				 			dict_grid=None, dict_names=None, dict_stars=None,
							dict_medium=None,
							dict_chemistry=None, loc_conch=item,
							num_cores=None, mode_testing=False)
							for item in conch_norms]
			#
			conch_norm_chems = [item.get_info("model_chemistry")
								for item in conch_norms]
		#
		if (model_titles is None):
			model_titles = names
		if (labels is None):
			labels = model_titles
		#
		chemistry_molecule_titles = preset.chemistry_molecule_titles
		#
		#Prepare some universal values
		num_conchs = len(conchs)
		#
		#Extract default parameters if no specific parameters given
		if (which_params is None):
			which_params = preset.plot_params_default_chemistry
		if (which_order is None):
			which_order = np.arange(0, num_conchs, 1)
		#
		num_panels = len(np.unique(which_order))

		##Prepare base grid to hold panels
		#Fill in some defaults
		nrow = (num_panels // ncol)
		if ((num_panels % ncol) > 0):
			nrow += 1
		if (isinstance(tbspace, float) or isinstance(tbspace, int)):
			tbspace = np.asarray([tbspace]*(nrow-1))
		if (isinstance(lrspace, float) or isinstance(lrspace, int)):
			lrspace = np.array([lrspace]*(ncol-1))
		if (isinstance(linewidths, float) or isinstance(linewidths, int)):
			linewidths = [linewidths]*num_conchs
		if (isinstance(markersizes, float) or isinstance(markersizes, int)):
			markersizes = [markersizes]*num_conchs
		if isinstance(colors, str):
			colors = [colors]*num_conchs
		if isinstance(markers, str):
			markers = [markers]*num_conchs
		if isinstance(styles, str):
			styles = [styles]*num_conchs
		#
		#Set the base grid
		fig = plt.figure(figsize=figsize)
		grid_base = utils._make_plot_grid(numsubplots=num_panels, numcol=ncol,
						numrow=nrow, tbspaces=tbspace, lrspaces=lrspace,
						tbmargin=tbmargin, lrmargin=lrmargin)
		#

		##Fetch all chemistry all at once first
		all_types_perconch = [item.split("_")[0] for item in which_params]
		all_species_perconch = [item.split("_")[1] for item in which_params]
		all_units_perconch = [item.split("_")[3] for item in which_params]
		#
		#Iterate through models
		set_dict_output_chem = [None]*num_conchs
		for ii in range(0, num_conchs):
			key_lookup =("1D_{0}_{1}".format(which_dir, all_units_perconch[ii]))
			if do_verbose:
				print("Fetching all chemistry for {0}...".format(names[ii]))
			#
			set_dict_output_chem[ii] = conchs_chem[ii]._fetch_disk_chemistry(
							cutoff_minval=cutoff_minval, xbounds_total=None,
							ybounds_zoverr=ybounds_zoverr,
							rtol_zoverr=rtol_zoverr,
							mode_species=all_types_perconch[ii],
							which_times_yr=[which_year], rtol_time=rtol_time,
							do_allow_load=do_allow_load, do_save=do_save,
							list_species_orig=[all_species_perconch[ii]]
							)[key_lookup]
		#
		if (conch_norms is not None):
			dict_output_norms = [item._fetch_disk_chemistry(
								cutoff_minval=cutoff_minval, xbounds_total=None,
								ybounds_zoverr=ybounds_zoverr,
								rtol_zoverr=rtol_zoverr,
								mode_species=all_types_perconch[ii],
								which_times_yr=[which_year],rtol_time=rtol_time,
								do_allow_load=do_allow_load, do_save=do_save,
								list_species_orig=[all_species_perconch[ii]]
								)[key_lookup]
								for item in conch_norm_chems]
		#

		##Plot aspect of 1D disk chemistry for each model
		#Prepare placeholders for aesthetic values
		group_s_vals = []
		group_arr_vals = []
		group_labels = []
		group_colors = []
		group_styles = []
		group_linewidths = []
		i_counter = 0
		#Iterate through models
		for ii in range(0, num_conchs):
			curr_conch = set_dict_output_chem[ii]
			curr_param = which_params[ii]
			curr_title = model_titles[ii]
			#If normalizing, then skip over normalizing model
			if (conch_norms is not None):
				curr_name_norm = conch_norms[ii].get_info("dict_names")["model"]
				if (names[ii] == curr_name_norm):
					#Print some notes
					if do_verbose:
						print("{0} is a normalizer. Skipping..."
								.format(names[ii]))
					continue
			#
			#Print some notes
			if do_verbose:
				print("Extracting chemistry for {0}...".format(names[ii]))
			#

			##Split parameters into components
			curr_split = curr_param.split("_")
			curr_spec = curr_split[1] #Species
			curr_phase = curr_split[2] #Phase (gas,gr,grsurf,grmant,tot)
			curr_unit = curr_split[3] #Absolute vs relative abundances
			#
			if (curr_spec in chemistry_molecule_titles):
				curr_spectitle = preset.chemistry_molecule_titles[curr_spec]
			else:
				curr_spectitle = curr_spec
			#
			#Prepare parameter title
			curr_paramtitle = (curr_spectitle
			 					+ (r"$_\mathrm{(" + curr_phase + r")}$" ))
			#

			##Extract values for normalizing disk model, if so requested
			if (conch_norms is not None):
				arr_norm = dict_output_norms[ii][curr_spec][curr_phase]
				#Unit conversion, if necessary
				if (curr_unit == "abs"):
					arr_norm *= conv_perm2topercm2 #[m^-2] -> [cm^-2]
			#

			##Extract value for scaling radial values, if so requested
			if (dict_col_scaler is not None):
				if (dict_col_scaler["type"] == "temperature_midplane"):
					curr_lstar = conchs[ii].get_info("dict_stars")["lstar"]
					curr_theta_flare = conchs[ii].get_info("dict_medium"
															)["theta_flare"]
					curr_col_scaler = math.calc_radius_of_temperature_midplane(
        	 							temp=dict_col_scaler["value"],
										lstar=curr_lstar,
										theta_flare=curr_theta_flare)
				else:
					raise ValueError("Err: {0} invalid scaler type."
									.format(dict_col_scaler))
			#

			##Extract 2D values for this disk model
			curr_set = set_dict_output_chem[ii][curr_spec]
			curr_time = curr_set["time"]
			if (which_dir == "col"):
				s_vals = curr_set["x"]
				if (dict_col_scaler is not None):
					s_vals = (s_vals / curr_col_scaler)
			elif (which_dir == "row"):
				s_vals = curr_set["zoverr"]
			#
			arr_vals = curr_set[curr_phase]
			#Unit conversion, if necessary
			if (curr_unit == "abs"):
				arr_vals *= conv_perm2topercm2 #[m^-2] -> [cm^-2]
			#
			#Throw error if actual time too far from target time
			if (not np.allclose([curr_time], [which_year], rtol=rtol_time)):
				raise ValueError("Time not in tol.!\n{0} vs {1}"
								.format(curr_time, which_year))
			#
			#Normalize, if so requested, and if not norm
			if (conch_norms is not None):
				curr_name_norm = conch_norms[ii].get_info("dict_names")["model"]
				if ((do_normthenorm) or (names[ii] != curr_name_norm)):
					arr_vals /= arr_norm
				#
			#

			#Set grid-based labels
			curr_xlabel = None
			curr_ylabel = None
			if ((which_order[ii] % ncol) == 0): #((ii % ncol) == 0):
				if (which_order[ii] >= (ncol*(nrow-1))):
					curr_xlabel = fig_xlabel
					curr_ylabel = "Column Density"
					if (conch_norms is not None):
						curr_ylabel += "\n(Approx. / Fiducial)"
					elif ("/" in curr_param):
						curr_ylabel += " Ratio"
					elif (curr_unit == "abs"):
						curr_ylabel += r" (cm$^{-2}$)"
					elif (curr_unit == "rel"):
						curr_ylabel += r" (Relative to 2 $\times$ H$_2$)"
			#
			if (curr_title is not None):
				curr_boxtext = "{0}\n{1}".format(curr_title, curr_paramtitle)
			else:
				curr_boxtext = "{1}".format(curr_title, curr_paramtitle)
			#
			rmxticklabel = False
			if ((inds_xticklabel is not None) and (ii not in inds_xticklabel)):
				rmxticklabel = True
			#
			rmyticklabel = False
			if ((inds_yticklabel is not None) and (ii not in inds_yticklabel)):
				rmyticklabel = True
			#

			##Gather latest values in current plot group
			group_s_vals.append(s_vals)
			group_arr_vals.append(arr_vals)
			group_labels.append(labels[ii])
			group_colors.append(colors[ii])
			group_styles.append(styles[ii])
			group_linewidths.append(linewidths[ii])
			#

			#Put in yspan around y=1 for normalized plots
			if (doyspan_norm and (conch_norms is not None)):
				doylines = True
				doyspans = True
				yspans = [yspan_norm]
				ylines = [[1.0]]
			else: #Otherwise, turn off yspan
				doylines = False
				doyspans = False
				ylines = None
				yspans = None
			#

			#Set unit scaler
			if (dict_col_scaler is not None):
				x_uniter = 1.0
			else:
				x_uniter = au0
			#

			##Plot results for this model+parameter, if group complete
			if ((ii == (num_conchs-1)) or (which_order[ii]!=which_order[ii+1])):
				#Set x-axis span
				if (xlim is None):
					curr_xlim = [0, (np.nanmax(group_s_vals)/x_uniter)]
					if (xmin_val is not None): #Fixed min. value
						curr_xlim[0] = xmin_val/x_uniter
					elif (xmin_frac is not None): #Fixed percentage of max
						curr_xlim[0] = (xmin_frac * curr_xlim[1])
				else:
					curr_xlim = xlim #None
				#
				if ((xmax is not None) and (xlim is None)):
					curr_xmax = None
					curr_xlim[1] = xmax[i_counter]
				#
				#Plot the 1D col. density
				utils.plot_lines(arr_xs=np.asarray(group_s_vals)/x_uniter,
				 		arr_ys=group_arr_vals, plotind=which_order[ii],
				 		doxlog=doxlog, doylog=doylog, textsize=textsize,
						ticksize=ticksize, titlesize=titlesize,
						legendfontsize=titlesize, colors=group_colors,
						labels=group_labels, styles=group_styles,
						nxticks=nxticks, tickwidth=tickwidth,
						tickheight=tickheight, markers=markers,
						markersizes=markersizes,
						legendlabelspacing=legendlabelspacing,
						legendhandlelength=legendhandlelength,
						dolegend=(legendind == which_order[ii]),dozeroline=True,
						boxtexts=[curr_boxtext], boxxs=[boxx], boxys=[boxy],
						boxalpha=boxalpha, ylabel=curr_ylabel,
						xlabel=curr_xlabel, doylines=doylines,
						doyspans=doyspans, yspans=yspans, ylines=ylines,
						yspancol=yspancol, yspanalpha=yspanalpha,
						ylinecol=ylinecol, ylinesty=ylinesty, xmax=curr_xmax,
						labelpad=labelpad, linewidth=group_linewidths,
						alpha=alpha, tickspace=tickspace, grid_base=grid_base,
						ymin=ymin, rmxticktitle=False, rmyticktitle=False,
						title=None, plotblank=False, rmxticklabel=rmxticklabel,
						rmyticklabel=rmyticklabel, xlim=curr_xlim, ylim=ylim,
						legendloc=legendloc, do_squareplot=True, fig=fig)
				#
				#Reset the group containers
				group_s_vals = []
				group_arr_vals = []
				group_labels = []
				group_colors = []
				group_styles = []
				group_linewidths = []
				i_counter += 1
			#
		#

		##Save and close the plot
		figname = ("figset_chemistry_1D{1}_time_{0:.1e}yr"
					.format(which_year, which_dir
							).replace(".","p").replace("+",""))
		if (conch_norms is not None):
			figname += "_norm"
		else:
			figname += "_unnorm"
		#
		if (ext_name is not None):
			figname += ("_"+ext_name)
		#
		plt.tight_layout()
		plt.savefig(os.path.join(filepath_save, figname+".png"))
		plt.close()
		#

		##Exit the function
		if do_verbose:
			time_total = (timer.time() - time_start)
			print("> Plot from plot_chemistry_1D complete in {0:.2f} minutes."
					.format(time_total/60))
		#
		return
	#


	##Function: plot_chemistry_2D()
	##Purpose: Plot comparison of disk chemistry across full disk
	def plot_chemistry_2D(self, which_year, cutoff_minval=None, rtol_time=0.1, model_titles=None, conch_norms=None, which_params=None, which_order=None, rmxticktitle=False, rmyticktitle=False, ncol=4, xlim=None, ymax_numscaleH=None, domatrlog=False, tbmargin=0.075, lrmargin=0.075, vmin=None, vmax=None, cmap_abs=plt.cm.PuBu, cmap_norm=None, list_contourcolors="black", list_contourwidths=3, contours_zoverr=None, ccolor_zoverr="gray", calpha_zoverr=0.5, cwidth_zoverr=1.5, cbarlabelpad=0, alpha=0.5, boxx=0.05, boxy=0.95, ticksize=16, textsize=18, titlesize=20, tickwidth=3, tickheight=5, filepath_save=None, figsize=(30,10), tbspace=0.25, lrspace=0.25, ext_name=None, do_zoverlines=False, do_normthenorm=False, subplot_norm_facecolor="white", inds_xticklabel=None, inds_yticklabel=None, inds_cbarlabel=None, inds_cbartitle=None, special_cbarlabel_norm=None, do_verbose=None):
		##Extract global variables
		if (do_verbose is None):
			do_verbose = self.get_info("do_verbose")
		#
		#Print some notes
		if do_verbose:
			print("> Plotting with plot_chemistry_2D!")
			time_start = timer.time()
		#

		##Extract model characteristics and chemistry holders
		conchs = self.get_info("list_conchs")
		names = self.get_info("names_conchs")
		do_allow_load = self.get_info("do_allow_load")
		do_save = self.get_info("do_save_processed_chemistry_output")
		conchs_chem = [conchs[ii].get_info("model_chemistry")
		 				for ii in range(0, len(conchs))]
		if (conch_norms is not None):
			if isinstance(conch_norms[0], str):
				conch_norms = [Conch_Disk(do_verbose=do_verbose, do_save=False,
				 			dict_grid=None, dict_names=None, dict_stars=None,
							dict_medium=None,
							dict_chemistry=None, loc_conch=item,
							num_cores=None, mode_testing=False)
							for item in conch_norms]
			#
			conch_norm_chems = [item.get_info("model_chemistry")
								for item in conch_norms]
		#
		if (model_titles is None):
			model_titles = names
		#
		chemistry_molecule_titles = preset.chemistry_molecule_titles
		#
		#Extract default parameters if no specific parameters given
		if (which_params is None):
			which_params = preset.plot_params_default_chemistry
		if (which_order is None):
			which_order = np.arange(0, len(which_params), 1)
		#
		#Prepare some universal values
		num_conchs = len(conchs)
		#num_params = len(which_params)
		num_panels = num_conchs
		#

		##Prepare base grid to hold panels
		#Fill in some defaults
		nrow = (num_panels // ncol)
		if ((num_panels % ncol) > 0):
			nrow += 1
		if (isinstance(tbspace, float) or isinstance(tbspace, int)):
			tbspace = np.asarray([tbspace]*(nrow-1))
		if (isinstance(lrspace, float) or isinstance(lrspace, int)):
			lrspace = np.array([lrspace]*(ncol-1))
		if isinstance(list_contourcolors, str):
			list_contourcolors = [list_contourcolors]*num_panels
		if (isinstance(list_contourwidths, float)
		 				or isinstance(list_contourwidths, int)):
			list_contourwidths = [list_contourwidths]*num_panels
		#Set the base grid
		fig = plt.figure(figsize=figsize)
		grid_base = utils._make_plot_grid(numsubplots=num_panels, numcol=ncol,
						numrow=nrow, tbspaces=tbspace, lrspaces=lrspace,
						tbmargin=tbmargin, lrmargin=lrmargin)
		#

		##Fetch all chemistry all at once first
		all_types_perconch = [item.split("_")[0] for item in which_params]
		all_species_perconch = [item.split("_")[1] for item in which_params]
		all_units_perconch = [item.split("_")[3] for item in which_params]
		#
		#Iterate through models
		set_dict_output_chem = [None]*num_conchs
		for ii in range(0, num_conchs):
			key_lookup = ("2D_{0}".format(all_units_perconch[ii]))
			if do_verbose:
				print("Fetching all chemistry for {0}...".format(names[ii]))
			#
			set_dict_output_chem[ii] = conchs_chem[ii]._fetch_disk_chemistry(
							cutoff_minval=cutoff_minval, xbounds_total=None,
							ybounds_zoverr=None,
							mode_species=all_types_perconch[ii],
							which_times_yr=[which_year], rtol_time=rtol_time,
							do_allow_load=do_allow_load, do_save=do_save,
							list_species_orig=[all_species_perconch[ii]]
							)[key_lookup]
		#
		if (conch_norms is not None):
			dict_output_norms = [item._fetch_disk_chemistry(
								cutoff_minval=cutoff_minval, xbounds_total=None,
								mode_species=all_types_perconch[ii],
								ybounds_zoverr=None,
								which_times_yr=[which_year],rtol_time=rtol_time,
								do_allow_load=do_allow_load, do_save=do_save,
								list_species_orig=[all_species_perconch[ii]]
								)[key_lookup]
								for item in conch_norm_chems]
		#

		##Plot aspect of 2D disk chemistry for each model
		for ii in range(0, num_conchs):
			matr_x = conchs[ii].get_info("dict_grid")["matr_x_chemistry"]
			matr_y = conchs[ii].get_info("dict_grid")["matr_y_chemistry"]
			curr_conch = set_dict_output_chem[ii]
			curr_param = which_params[ii]
			curr_title = model_titles[ii] #(names[ii]+"|"+curr_param)
			if do_verbose:
				print("Extracting chemistry for {0}...".format(names[ii]))
			#
			ylim = None
			if (ymax_numscaleH is not None):
				curr_ymax = (conchs[ii].get_info("dict_medium"
												)["matr_scaleH_structure"].max()
							* ymax_numscaleH)
				ylim = np.array([0, curr_ymax])
			#

			##Split parameters into components
			curr_split = curr_param.split("_")
			curr_type = curr_split[0] #Mol. or el.
			curr_spec = curr_split[1] #Species
			curr_phase = curr_split[2] #Phase (gas,gr,grsurf,grmant,tot)
			curr_unit = curr_split[3] #Absolute vs relative abundances
			#
			if (curr_spec in chemistry_molecule_titles):
				curr_spectitle = preset.chemistry_molecule_titles[curr_spec]
			else:
				curr_spectitle = curr_spec
			#
			#Prepare parameter title
			curr_paramtitle = (curr_spectitle
			 					+ (r"$_\mathrm{(" + curr_phase + r")}$" ))
			if (curr_unit == "abs"):
				curr_paramtitle += r" cm$^{-3}$"
			#
			#Prepare other unnorm. aesthetics
			curr_backcolor = "white"
			curr_ccolor = list_contourcolors[ii]
			#

			##Extract values for normalizing disk model, if so requested
			if (conch_norms is not None):
				#Throw error if physical structure not identical
				tmp_boolx = np.array_equal(matr_x,
				 	conch_norms[ii].get_info("dict_grid")["matr_x_chemistry"])
				tmp_booly = np.array_equal(matr_y,
				 	conch_norms[ii].get_info("dict_grid")["matr_y_chemistry"])
				if ((not tmp_boolx) or (not tmp_booly)):
					raise ValueError("Err: Cannot normalize over different "
									+"physical spatial grids.")
				#

				#Extract normalizing matrix
				matr_norm = dict_output_norms[ii][curr_spec][curr_phase
														].reshape(matr_y.shape)
				if (curr_unit == "abs"):
					matr_norm *= conv_perm3topercm3 #Convert [m^-3] -> [cm^-3]
				#
			#

			##Extract 2D values for this disk model
			curr_set = set_dict_output_chem[ii][curr_spec]
			curr_time = curr_set["time"]
			matr_vals = curr_set[curr_phase].reshape(matr_y.shape)
			curr_cmap = cmap_abs
			if (curr_unit == "abs"):
				matr_vals *= conv_perm3topercm3 #Convert [m^-3] -> [cm^-3]
			#
			#Throw error if actual time too far from target time
			if (not np.allclose([curr_time], [which_year], rtol=rtol_time)):
				raise ValueError("Time not in tol.!\n{0} vs {1}"
								.format(curr_time, which_year))
			#
			#Convert from rel. abund. to number density, if requested
			if (curr_unit == "abs"):
				curr_cbarlabel = "Abs. Abund."
			#
			#Set default vmin, vmax ranges
			vmintmp = np.nan
			vmaxtmp = np.nan
			if (curr_unit == "rel"):
				curr_cbarlabel = "Rel. Abund."
				#Range for ratio
				if ("/" in curr_spec):
					vmintmp = 1E-2 #1E-2 #1E-4
					vmaxtmp = 1E2 #1E2 #1E4
					#curr_contours = [1E-3, 1E-1, 1E1, 1E3]
					curr_contours = [1E-1, 1E1]
				#Range for singular element or molecule
				else:
					if (curr_type == "el"):
						vmintmp = 1E-10 #1E-11
						vmaxtmp = 1E-3
					elif (curr_type == "mol"):
						vmintmp = 1E-10 #1E-11
						vmaxtmp = 1E-5
					curr_contours = [1E-10, 1E-8, 1E-6, 1E-4]
					curr_ccolor = "white"
			#
			#Range in physical units (cm^-3)
			elif (curr_unit == "abs"):
				if (curr_type == "el"):
					vmintmp = 1E-4 #10**(10 - 10)
					vmaxtmp = 1E6 #10**(10 - 3)
					curr_contours = [1E-4, 1E-2, 1E0, 1E2, 1E4, 1E6]
				elif (curr_type == "mol"):
					vmintmp = 1E-4 #10**(6 - 10)
					vmaxtmp = 1E2 #10**(6 - 4)
					curr_contours = [1E-4, 1E-2, 1E0, 1E2]
				curr_ccolor = "white"
			#
			#Otherwise, throw error
			else:
				raise ValueError("Err: Unrecognized condition...")
			#
			#Normalize, if so requested, and if not norm
			if (conch_norms is not None):
				curr_name_norm = conch_norms[ii].get_info("dict_names")["model"]
				if ((do_normthenorm) or (names[ii] != curr_name_norm)):
					matr_vals /= matr_norm
					vmintmp = 1E-1 #1/3.0 #1E-1 #1E-2
					vmaxtmp = 1E1 #3 #1E1 #1E2
					curr_ccolor = "black"
					curr_contours = [0.5, 2]
					curr_cmap = cmap_norm
					curr_backcolor = subplot_norm_facecolor
					if (special_cbarlabel_norm is not None):
						curr_cbarlabel = special_cbarlabel_norm
					else:
						curr_cbarlabel = ("Norm. " + curr_cbarlabel)
				#
			#
			#Set vmin, vmax values to defaults if not given
			vminmod = vmin
			vmaxmod = vmax
			if (vminmod is None):
				vminmod = vmintmp
			if (vmaxmod is None):
				vmaxmod = vmaxtmp
			#
			#Take log, if so requested
			if domatrlog:
				matr_vals = np.log10(matr_vals)
				curr_contours = np.log10(curr_contours)
				if (vminmod is not None):
					vminmod = np.log10(vminmod)
				if (vmaxmod is not None):
					vmaxmod = np.log10(vmaxmod)
			#

			#Set grid-based labels
			curr_xlabel = None
			curr_ylabel = None
			if (((ii % ncol) == 0) and (ii >= (ncol*(nrow-1)))):
					curr_xlabel = "Radius (au)"
					curr_ylabel = "Height (au)"
			#
			curr_boxtext = "{0}\n{1}".format(curr_title, curr_paramtitle)
			if ((inds_cbarlabel is not None) and (ii not in inds_cbarlabel)):
				curr_cbarlabel = None
			elif ((inds_cbartitle is not None) and (ii not in inds_cbartitle)):
				curr_cbarlabel = ""
			#
			rmxticklabel = False
			if ((inds_xticklabel is not None) and (ii not in inds_xticklabel)):
				rmxticklabel = True
			#
			rmyticklabel = False
			if ((inds_yticklabel is not None) and (ii not in inds_yticklabel)):
				rmyticklabel = True
			#

			##Plot results for this model+parameter
			if domatrlog:
				if ((curr_cbarlabel is not None) and (curr_cbarlabel != "")):
					curr_cbarlabel = (r"$\log_{10}$(" + curr_cbarlabel + ")")
			#
			#Plot the 2D matrix
			utils.plot_diskslice(matr=matr_vals, arr_x=matr_x/au0,
			 					arr_y=matr_y/au0, backcolor=curr_backcolor,
								plotind=ii, cmap=curr_cmap,
								docbar=True, boxtexts=[curr_boxtext],
								boxxs=[boxx], boxys=[boxy],
								textsize=textsize, ticksize=ticksize,
								titlesize=titlesize, contours=curr_contours,
								contours_zoverr=contours_zoverr,
								ccolor=curr_ccolor, #list_contourcolors[ii],
								ccolor_zoverr=ccolor_zoverr,
								calpha_zoverr=calpha_zoverr,
								cwidth_zoverr=cwidth_zoverr,
								boxcolor="none", boxalpha=1.0, ylim=ylim/au0,
								tickwidth=tickwidth, tickheight=tickheight,
								grid_base=grid_base,
								xlabel=curr_xlabel, ylabel=curr_ylabel,
								docbarlabel=(curr_cbarlabel is not None),
								cbarlabel=curr_cbarlabel,
								cbarlabelpad=cbarlabelpad,
								rmxticklabel=rmxticklabel,
								rmyticklabel=rmyticklabel,
								vmin=vminmod, vmax=vmaxmod,
								do_return_panel=False)
			#
		#

		##Save and close the plot
		figname = ("figset_chemistry_2D_time_{0:.1e}yr"
					.format(which_year).replace(".","p").replace("+",""))
		if (conch_norms is not None):
			figname += "_norm"
		else:
			figname += "_unnorm"
		#
		if (ext_name is not None):
			figname += ("_"+ext_name)
		#
		plt.tight_layout()
		plt.savefig(os.path.join(filepath_save, figname+".png"))
		plt.close()
		#

		##Exit the function
		if do_verbose:
			time_total = (timer.time() - time_start)
			print("> Plot from plot_chemistry_2D complete in {0:.2f} minutes."
					.format(time_total/60))
		#
		return
	#


	##Function: plot_spectra()
	##Purpose: Plot comparison of disk spectra
	def plot_spectra(self, dist_norm, which_params=None, which_order=None, all_labels=None, model_titles=None, doxlog=True, doylog=True, xlim=None, ylim=None, do_squareplot=False, do_cornerlabel=False, do_shareyaxis=False, do_sharexaxis=False, figsize=(30,10), ncol=None, boxtexts=None, boxxs=[0.05], boxys=[0.95], ticksize=16, textsize=18, titlesize=20, legendfontsize=None, tickwidth=3, tickheight=5, tbspace=0.075, lrspace=0.075, tbmargin=0.1, lrmargin=0.1, filepath_save=None, alphas=0.5, list_colors="black", list_lstyles="-", list_lwidths=3, ext_name=None, ind_legend=None, legendloc="best", fin_ylabel=None, doxlines=False, xlines=None, xlinecol="gray", xlinesty="--", xlinewidth=1, doylines=False, ylines=None, ylinecol="gray", ylinesty="--", ylinewidth=1, doyspans=False, yspans=None, yspancol="gray", yspanalpha=0.5, ind_topline=None):
		##Extract model characteristics and chemistry holders
		conchs = self.get_info("list_conchs")
		names = self.get_info("names_conchs")
		do_verbose = self.get_info("do_verbose")
		#
		#Extract default parameters if no specific parameters given
		if (model_titles is None):
			model_titles = names
		if (which_params is None):
			which_params = preset.plot_params_default_spectrum
		if (which_order is None):
			which_order = [np.arange(0, len(which_params), 1)]
		#
		#Prepare some universal values
		num_conchs = len(conchs)
		num_order = len(which_order)
		num_panels = num_order
		#
		#Print some notes
		if do_verbose:
			print("\n> plot_spectra()!\nModels: {0}\nOrder: {1}\nNames: {2}"
				.format(which_order, model_titles, names))
		#

		##Prepare base grid to hold panels
		if (ncol is None):
			ncol = num_order
		nrow = (num_panels // ncol)
		if ((num_panels % ncol) > 0):
			nrow += 1
		if (isinstance(tbspace, float) or isinstance(tbspace, int)):
			tbspace = np.asarray([tbspace]*(nrow-1))
		if (isinstance(lrspace, float) or isinstance(lrspace, int)):
			lrspace = np.array([lrspace]*(ncol-1))
		if (isinstance(alphas, int) or isinstance(alphas, float)):
			alphas = [[alphas for kk in range(0,len(which_order[jj]))]
							for jj in range(0, len(which_order))]
		if isinstance(list_colors, str):
			list_colors = [[list_colors for kk in range(0,len(which_order[jj]))]
							for jj in range(0, len(which_order))]
		if (isinstance(list_lwidths, float) or isinstance(list_lwidths, int)):
			list_lwidths=[[list_lwidths for kk in range(0,len(which_order[jj]))]
							for jj in range(0, len(which_order))]
		if isinstance(list_lstyles, str):
			list_lstyles=[[list_lstyles for kk in range(0,len(which_order[jj]))]
							for jj in range(0, len(which_order))]
		if isinstance(doxlog, bool):
			doxlog = [doxlog]*num_panels
		if isinstance(doylog, bool):
			doylog = [doylog]*num_panels
		#
		#Set the base grid
		fig = plt.figure(figsize=figsize)
		grid_base = utils._make_plot_grid(numsubplots=(nrow*ncol), numcol=ncol,
						numrow=nrow, tbspaces=tbspace, lrspaces=lrspace,
						tbmargin=tbmargin, lrmargin=lrmargin)
		#

		##Prepare base figure and attributes
		conv_dist_fromRSun = (Rsun0 / dist_norm)**2
		conv_dist_fromau = (au0 / dist_norm)**2
		conv_wavelength_nm = 1E9
		conv_spectrum_pernm = (1E-9)
		conv_spectrum_percm2 = (1/((100)**2))
		xlabel = r"$\lambda$ (nm)"
		#

		##Plot each structure
		#Iterate through ordered parameters (overall subplots)
		for ii in range(0, len(which_order)):
			curr_set = which_order[ii]
			#Iterate through parameters within this subplot
			list_x_cgs = [None]*len(curr_set)
			list_y_cgs = [None]*len(curr_set)
			list_labels = [None]*len(curr_set)
			for jj in range(0, len(curr_set)):
				curr_imodel = which_order[ii][jj]
				dict_stars = conchs[curr_imodel].get_info("dict_stars")
				curr_param = which_params[ii][jj]
				if (all_labels is None):
					list_labels[jj] = curr_param
				else:
					list_labels[jj] = all_labels[ii][jj]
				#
				#Print some notes
				if do_verbose:
					print("\nPlotting {0} for {1} in subplot {2}..."
						.format(curr_param, model_titles[curr_imodel], ii))
				#

				#For total UV photon spectrum
				if (curr_param == "spectrum_photon_UVtot_struct"):
					try:
						x_cgs = (dict_stars["wavelength_spectrum_UVtot_struct"]
								* conv_wavelength_nm)
						y_cgs =(dict_stars["photon_spectrum_UVtot_1RSun_struct"]
								* conv_dist_fromRSun
								* conv_spectrum_pernm
								* conv_spectrum_percm2)
					except KeyError:
						range_UV = preset.wavelength_range_rates_UV
						tmp_inds = (
								(dict_stars["wavelength_spectrum_all_struct"]
										>= range_UV[0])
								& (dict_stars["wavelength_spectrum_all_struct"]
										<= range_UV[1]))
						#
						x_cgs = (dict_stars["wavelength_spectrum_all_struct"]
								* conv_wavelength_nm)[tmp_inds]
						y_cgs = (dict_stars["photon_spectrum_all_1RSun_struct"]
								* conv_dist_fromRSun
								* conv_spectrum_pernm
								* conv_spectrum_percm2)[tmp_inds]
					#
					ylabel = (r"F$_\mathrm{UV,tot.}$ at 1R$_\odot$"
					 			+r" (photon s$^{-1}$ cm$^{-2}$ nm$^{-1}$)")
				#
				elif (curr_param == "spectrum_energy_UVtot_struct"):
					try:
						x_cgs = (dict_stars["wavelength_spectrum_UVtot_struct"]
								* conv_wavelength_nm)
						y_cgs =(dict_stars["energy_spectrum_UVtot_1RSun_struct"]
								* conv_dist_fromRSun * conv_Jtoerg0
								* conv_spectrum_pernm
								* conv_spectrum_percm2)
					except KeyError:
						range_UV = preset.wavelength_range_rates_UV
						tmp_inds = (
								(dict_stars["wavelength_spectrum_all_struct"]
										>= range_UV[0])
								& (dict_stars["wavelength_spectrum_all_struct"]
										<= range_UV[1]))
						#
						x_cgs = (dict_stars["wavelength_spectrum_all_struct"]
								* conv_wavelength_nm)[tmp_inds]
						y_cgs = (dict_stars["energy_spectrum_all_1RSun_struct"]
								* conv_dist_fromRSun * conv_Jtoerg0
								* conv_spectrum_pernm
								* conv_spectrum_percm2)[tmp_inds]
					#
					ylabel = (r"F$_\mathrm{UV,tot.}$ at 1R$_\odot$"
					 			+r" (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)")
				#
				#For cont. UV energy spectrum
				elif (curr_param == "spectrum_energy_UVcont_struct"):
					x_cgs = (dict_stars["wavelength_spectrum_UVcont_struct"]
								* conv_wavelength_nm)
					y_cgs = (dict_stars["energy_spectrum_UVcont_1RSun_struct"]
								* conv_dist_fromRSun * conv_Jtoerg0
								* conv_spectrum_pernm
								* conv_spectrum_percm2)
					#
					ylabel = (r"F$_\mathrm{UV,cont.}$ at 1R$_\odot$"
					 			+r" (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)")
				#
				#For UV Lya energy spectrum
				elif (curr_param == "spectrum_energy_Lya_struct"):
					x_cgs = (dict_stars["wavelength_spectrum_Lya_struct"]
								* conv_wavelength_nm)
					y_cgs = (dict_stars["energy_spectrum_Lya_1RSun_struct"]
								* conv_dist_fromRSun * conv_Jtoerg0
								* conv_spectrum_pernm
								* conv_spectrum_percm2)
					#
					ylabel = (r"F$_\mathrm{UV,Ly\alpha}$ at 1R$_\odot$"
					 			+r" (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)")
				#
				#For X-ray energy spectrum
				elif (curr_param == "spectrum_energy_X_struct"):
					x_cgs = (dict_stars["wavelength_spectrum_X_struct"]
								* conv_wavelength_nm)
					y_cgs = (dict_stars["energy_spectrum_X_1RSun_struct"]
								* conv_dist_fromRSun * conv_Jtoerg0
								* conv_spectrum_pernm
								* conv_spectrum_percm2)
					#
					ylabel = (r"F$_\mathrm{X}$ at 1R$_\odot$"
					 			+r" (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)")
				#
				#For SED energy spectrum
				elif (curr_param == "spectrum_energy_SED_struct"):
					x_cgs = (dict_stars["wavelength_spectrum_all_struct"]
								* conv_wavelength_nm)
					y_cgs = (dict_stars["energy_spectrum_all_1RSun_struct"]
								* conv_dist_fromRSun * conv_Jtoerg0
								* conv_spectrum_pernm
								* conv_spectrum_percm2)
					#
					ylabel = (r"F$_\mathrm{SED}$ at 1R$_\odot$"
					 			+r" (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)")
				#
				#Otherwise, throw error if not recognized
				else:
					raise ValueError("Err: {0} inval.!".format(curr_param))
				#

				#Print some notes
				if do_verbose:
					print("Min. array value: {0}".format(np.nanmin(y_cgs)))
					print("Max. array value: {0}".format(np.nanmax(y_cgs)))
				#

				#Store current x,y values
				list_x_cgs[jj] = x_cgs
				list_y_cgs[jj] = y_cgs
			#

			#Set final y-label
			if (fin_ylabel is None):
				fin_ylabel = ylabel
			#

			#Wipe label if so requested for non-corner plots
			if (do_cornerlabel and (ii != (ncol*(nrow-1)))):
				curr_xlabel = None
				curr_ylabel = None
			else:
				curr_xlabel = xlabel
				curr_ylabel = fin_ylabel
			#
			curr_dolegend = ((ind_legend is not None) and (ind_legend == ii))
			#

			#Wipe axis if so requested for non-edge plots
			rmxticklabel = False
			rmyticklabel = False
			if (do_sharexaxis and (ii < (ncol*(nrow-1)))):
				rmxticklabel = True
			if (do_shareyaxis and ((ii % ncol) != 0)):
				rmyticklabel = True
			#

			#Plot the current set of spectra
			utils.plot_lines(arr_xs=list_x_cgs, arr_ys=list_y_cgs,
			 			plotind=ii, ylim=ylim, xlim=xlim,
						grid_base=grid_base,doxlog=doxlog[ii],doylog=doylog[ii],
						xlabel=curr_xlabel, ylabel=curr_ylabel,
						boxtexts=boxtexts[ii], boxxs=boxxs[ii], boxys=boxys[ii],
						colors=list_colors[ii], dolegend=curr_dolegend,
						labels=list_labels, linewidth=list_lwidths[ii],
						styles=list_lstyles[ii], alpha=alphas[ii],
						legendfontsize=legendfontsize,
						textsize=textsize, ticksize=ticksize,
						titlesize=titlesize, do_squareplot=do_squareplot,
						tickwidth=tickwidth, tickheight=tickheight,
						rmxticklabel=rmxticklabel, rmyticklabel=rmyticklabel,
						legendloc=legendloc, doxlines=doxlines, xlines=xlines,
						xlinecol=xlinecol, xlinesty=xlinesty,
						xlinewidth=xlinewidth,
						doylines=doylines, ylines=ylines,
						ylinecol=ylinecol, ylinesty=ylinesty,
						ylinewidth=ylinewidth, doyspans=doyspans,
						yspans=yspans, yspancol=yspancol, yspanalpha=yspanalpha,
						ind_topline=ind_topline)
			#
		#

		##Save and close the results
		figname = "figset_spectra"
		if (ext_name is not None):
			figname += ("_"+ext_name)
		#
		plt.tight_layout()
		plt.savefig(os.path.join(filepath_save, (figname+".png")))
		plt.close()
		#

		##Exit the method
		return
	#


	##Function: plot_structure()
	##Purpose: Plot comparison of disk structure
	def plot_structure(self, which_params=None, which_order=None, model_titles=None, ys_cell_au=None, xs_cell_au=None, point_colors="red", point_marker="x", point_alpha=0.75, point_size=100, figsize=(30,10), ncol=None, boxx=0.05, boxy=0.95, ticksize=16, textsize=18, titlesize=20, tickwidth=3, tickheight=5, tbspace=0.075, lrspace=0.075, tbmargin=0.1, lrmargin=0.1, filepath_save=None, list_cmaps=cmasher.rainforest_r, ext_name=None, do_cornerlabel=False):
		##Extract model characteristics and chemistry holders
		conchs = self.get_info("list_conchs")
		names = self.get_info("names_conchs")
		do_verbose = self.get_info("do_verbose")
		#
		#Extract default parameters if no specific parameters given
		if (model_titles is None):
			model_titles = names
		if (which_params is None):
			which_params = preset.plot_params_default_structure
		if (which_order is None):
			which_order = np.arange(0, len(which_params), 1)
		#
		#Prepare some universal values
		num_conchs = len(conchs)
		num_params = len(which_params)
		num_panels = (num_params * num_conchs)
		#

		##Prepare base grid to hold panels
		if (ncol is None):
			ncol = num_params
		nrow = (num_panels // ncol)
		if ((num_panels % ncol) > 0):
			nrow += 1
		if (isinstance(tbspace, float) or isinstance(tbspace, int)):
			tbspace = np.asarray([tbspace]*(nrow-1))
		if (isinstance(lrspace, float) or isinstance(lrspace, int)):
			lrspace = np.array([lrspace]*(ncol-1))
		if isinstance(list_cmaps, str):
			list_cmaps = [list_cmaps]*num_panels
		if ((ys_cell_au is not None) and isinstance(point_colors, str)):
			point_colors = [point_colors]*len(ys_cell_au)
		#
		#Set the base grid
		fig = plt.figure(figsize=figsize)
		grid_base = utils._make_plot_grid(numsubplots=(nrow*ncol), numcol=ncol,
						numrow=nrow, tbspaces=tbspace, lrspaces=lrspace,
						tbmargin=tbmargin, lrmargin=lrmargin)
		#

		##Prepare base figure and attributes
		cbarlabel = None
		conv_percm2 = conv_perm2topercm2
		conv_percm3 = conv_perm3topercm3
		#

		##Plot each structure
		#Iterate through parameters
		for ii in range(0, num_params):
			curr_param = which_params[ii]
			#Iterate through models
			for jj in range(0, num_conchs):
				dict_medium = conchs[jj].get_info("dict_medium")
				matr_x = conchs[jj].get_info("dict_grid")["matr_x_structure"]
				matr_y = conchs[jj].get_info("dict_grid")["matr_y_structure"]
				i_panel = ((jj*num_params) + ii)
				#
				if do_cornerlabel:
					if (i_panel == (ncol*(nrow-1))):
						xlabel = "Radius (au)"
						ylabel = "Height (au)"
					else:
						xlabel = None
						ylabel = None
					if (i_panel < (ncol*(nrow-1))):
						rmxticklabel = True
					else:
						rmxticklabel = False
					if ((i_panel % ncol) != 0):
						rmyticklabel = True
					else:
						rmyticklabel = False
				elif ((jj == (num_conchs - 1)) and (ii == 0)):
					xlabel = "Radius (au)"
					ylabel = "Height (au)"
					rmxticklabel = False
					rmyticklabel = False
				else:
					xlabel = None
					ylabel = None
					rmxticklabel = False
					rmyticklabel = True
				#
				#Print some notes
				if do_verbose:
					print("\nPlotting {0} for {1} in {2}x{3} grid..."
							.format(curr_param, model_titles[jj], nrow, ncol))
				#

				#For blank panel
				contours = None
				ccolor = None
				if curr_param is None:
					self.plot_diskslice(matr=None, arr_x=None, arr_y=None,
					 					grid_base=grid_base, plotind=i_panel,
										cmap=None, plotblank=True)
					#Skip ahead
					continue
				#
				#For nH density
				elif (curr_param == "dens_nH_vol"):
					matr_cgs = np.log10(dict_medium["volnumdens_nH_structure"]
										* conv_percm3)
					#
					vmin = 4
					vmax = 10 #4
					contours = [4, 5, 6, 7, 8, 9, 10] #, 10, 11, 12]
					ccolor = "black"
					cbarname = r"$\log_{10}$(n$_\mathrm{H}$)"
					cbarunit = r"(cm$^{-3})$"
				#
				#For total mass dust density
				elif (curr_param == "dens_dust_mass"):
					matr_cgs = np.log10(
								(dict_medium["volmassdens_dustall_structure"]
								* conv_percm3 * conv_kgtog0))
					#
					vmin = -24 #-24
					vmax = -16 #-14
					contours = [-23, -21, -19, -17] #, -19, -18, -17]
					ccolor = "black"
					cbarname = r"$\log_{10}$($\rho_\mathrm{dust}$)"
					cbarunit = r"(g cm$^{-3})$"
				#
				#For gas temperature
				elif (curr_param == "temp_gas"):
					tmp_islog = False
					if tmp_islog:
						matr_cgs=np.log10(dict_medium["matr_tempgas_structure"])
						#
						vmin = np.log10(10) #0
						vmax = np.log10(100) #300
						contours = [np.log10(25), np.log10(50), np.log10(75)]
						ccolor = "white"
						cbarname = r"T$_\mathrm{gas}$"
						cbarunit = r"(K)"
					else:
						matr_cgs = dict_medium["matr_tempgas_structure"]
						#
						vmin = 0
						vmax = 200 #150
						contours = [25, 50, 75, 100, 125, 150, 175, 200] #, 150, 200, 250]
						ccolor = "white"
						cbarname = r"T$_\mathrm{gas}$"
						cbarunit = r"(K)"
					#
				#
				#For dust temperature
				elif (curr_param == "temp_dust"):
					tmp_islog = False
					if tmp_islog:
						matr_cgs = np.log10(
										dict_medium["matr_tempdust_structure"])
						#
						vmin = np.log10(10) #0
						vmax = np.log10(100) #300
						contours = [np.log10(25), np.log10(50), np.log10(75)]
						ccolor = "white"
						cbarname = r"$\log_{10}$(T$_\mathrm{dust}$)"
						cbarunit = r"(K)"
					else:
						matr_cgs = dict_medium["matr_tempdust_structure"]
						#
						vmin = 0
						vmax = 200 #150
						contours = [25, 50, 75, 100, 125, 150, 175, 200] #, 150, 200, 250]
						ccolor = "white"
						#cbarlabel = r"T$_\mathrm{dust}$ (K)"
						cbarname = r"T$_\mathrm{dust}$"
						cbarunit = r"(K)"
				#
				#For integrated Lya photon radiation field
				elif (curr_param == "flux_photon_Lya"):
					matr_cgs = np.log10(
								dict_medium["matr_photonflux_Lya_structure"]
								* conv_percm2)
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_photonflux_Lya_structure"] == 0
							] = 0
					#
					vmin = 5
					vmax = 13
					ccolor = "cyan"
					contours = [6, 8, 10, 12]
					cbarname = r"$\log_{10}$(F$_\mathrm{UV,Ly\alpha}$)"
					cbarunit = r"(ph. s$^{-1}$ cm$^{-2}$)"
				#
				#For integrated Lya energy radiation field
				elif (curr_param == "flux_energy_Lya"):
					matr_cgs = np.log10(
								dict_medium["matr_energyflux_Lya_structure"]
								* conv_percm2 / ergtoJ0)
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_energyflux_Lya_structure"] == 0
							] = 0
					#
					vmin = -21
					vmax = -8
					cbarname = r"$\log_{10}$(F$_\mathrm{UV,Ly\alpha}$)"
					cbarunit = r"(erg s$^{-1}$ cm$^{-2}$)"
				#
				#For integrated UV cont. photon radiation field
				elif (curr_param == "flux_photon_UVcont"):
					matr_cgs = np.log10(
								dict_medium["matr_photonflux_UVcont_structure"]
								* conv_percm2)
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_photonflux_UVcont_structure"] ==0
							] = 0
					#
					vmin = 5
					vmax = 13
					ccolor = "cyan"
					contours = [6, 8, 10, 12]
					cbarname = r"$\log_{10}$(F$_\mathrm{UV,cont.}$)"
					cbarunit = r"(ph. s$^{-1}$ cm$^{-2}$)"
				#
				#For integrated UV cont. energy radiation field
				elif (curr_param == "flux_energy_UVcont"):
					matr_cgs = np.log10(
								dict_medium["matr_energyflux_UVcont_structure"]
								* conv_percm2 / ergtoJ0)
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_energyflux_UVcont_structure"] ==0
							] = 0
					#
					vmin = -21
					vmax = -8
					cbarlabel = (r"$\log_{10}$(F$_\mathrm{UV,cont.}$)"
					 			+"\n"+r"(erg s$^{-1}$ cm$^{-2}$)")
				#
				#For integrated total UV photon radiation field
				elif (curr_param == "flux_photon_UVtot"):
					matr_cgs = np.log10(
								dict_medium["matr_photonflux_UVtot_structure"]
								* conv_percm2)
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_photonflux_UVtot_structure"] == 0
							] = 0
					#
					vmin = 5
					vmax = 13
					ccolor = "cyan"
					contours = [6, 8, 10, 12]
					cbarname = r"$\log_{10}$(F$_\mathrm{UV,tot.}$)"
					cbarunit = r"(ph. s$^{-1}$ cm$^{-2}$)"
				#
				#For integrated total UV energy radiation field
				elif (curr_param == "flux_energy_UVtot"):
					matr_cgs = np.log10(
								dict_medium["matr_energyflux_UVtot_structure"]
								* conv_percm2 / ergtoJ0)
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_energyflux_UVtot_structure"] == 0
							] = 0
					#
					vmin = -21
					vmax = -8
					cbarname = r"F$\log_{10}$($_\mathrm{UV,tot.}$)"
					cbarunit = r"(erg s$^{-1}$ cm$^{-2}$)"
				#
				#For integrated X-ray photon radiation field
				elif (curr_param == "flux_photon_X"):
					matr_cgs = np.log10(
								dict_medium["matr_photonflux_X_structure"]
								* conv_percm2)
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_photonflux_X_structure"] == 0
							] = 0
					#
					vmin = 5
					vmax = 10
					ccolor = "white"
					contours = [6, 7, 8, 9]
					cbarname = r"$\log_{10}$(F$_\mathrm{X}$)"
					cbarunit = r"(ph. s$^{-1}$ cm$^{-2}$)"
				#
				#For integrated X-ray energy radiation field
				elif (curr_param == "flux_energy_X"):
					matr_cgs = np.log10(
								dict_medium["matr_energyflux_X_structure"]
								* conv_percm2 / ergtoJ0)
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_energyflux_X_structure"] == 0
							] = 0
					#
					vmin = -21
					vmax = -8
					cbarname = r"$\log_{10}$(F$_\mathrm{X}$)"
					cbarunit = r"(erg s$^{-1}$ cm$^{-2}$)"
				#
				#For X-ray ionization rate
				elif (curr_param == "ionrate_X_primary"):
					matr_x =conchs[jj].get_info("dict_grid")["matr_x_chemistry"]
					matr_y =conchs[jj].get_info("dict_grid")["matr_y_chemistry"]
					matr_cgs = np.log10(
								dict_medium["matr_ionrate_X_primary_chemistry"])
					#
					#Deal with log10(0) issue
					matr_cgs[dict_medium["matr_ionrate_X_primary_chemistry"]==0
							] = 0
					#
					vmin = -21
					vmax = -8
					ccolor = "white"
					contours = [-20, -18, -16, -14, -12, -10, -8]
					cbarname = r"$\log_{10}$($\zeta_\mathrm{X}$)"
					cbarunit = r"(s$^{-1}$)"
				#
				#Otherwise, throw error if not recognized
				else:
					raise ValueError("Err: {0} invalid!".format(curr_param))
				#

				#Print some notes
				if do_verbose:
					print("Min. matrix value: {0}".format(np.nanmin(matr_cgs)))
					print("Max. matrix value: {0}".format(np.nanmax(matr_cgs)))
				#

				#Plot the disk slice
				curr_boxtexts = [(model_titles[jj] + "\n"
				 					+ cbarname + "\n" + cbarunit)]
				cbarlabel = "" #cbarunit
				#
				if ("\n" in cbarlabel):
					cbarlabelpad = 45
				else:
					cbarlabelpad = 20
				#
				panel = utils.plot_diskslice(matr=matr_cgs,
				 					arr_x=matr_x/au0, arr_y=matr_y/au0,
				 					plotind=i_panel, cmap=list_cmaps[ii],
									docbar=True, boxtexts=curr_boxtexts,
									boxxs=[boxx], boxys=[boxy],
									rmxticklabel=rmxticklabel,
									rmyticklabel=rmyticklabel,
									textsize=textsize, ticksize=ticksize,
									titlesize=titlesize, contours=contours,
									ccolor=ccolor,boxcolor="none",boxalpha=1.0,
									tickwidth=tickwidth, tickheight=tickheight,
									grid_base=grid_base,
									xlabel=xlabel, ylabel=ylabel,
									docbarlabel=True, cbarlabel=cbarlabel,
									cbarlabelpad=cbarlabelpad,
									vmin=vmin, vmax=vmax, do_return_panel=True)
				#

				#Plot in points, if given
				if (ys_cell_au is not None):
					for kk in range(0, len(ys_cell_au[jj])):
						panel.scatter(xs_cell_au[jj][kk], ys_cell_au[jj][kk],
						 			color=point_colors[ii], marker=point_marker,
									alpha=point_alpha, s=point_size, zorder=100)
				#
			#
		#

		##Save and close the results
		figname = "figset_structure"
		if (ext_name is not None):
			figname += ("_"+ext_name)
		#
		plt.tight_layout()
		plt.savefig(os.path.join(filepath_save, (figname+".png")))
		plt.close()
		#

		##Exit the method
		return
	#


	##Function: table_chemistry()
	##Purpose: Generate table of disk chemistry output across full disk
	def table_chemistry(self, which_year, num_include, xbounds_total, ybounds_zoverr, cutoff_minval=None, rtol_time=0.1, rtol_zoverr=None, model_titles=None, filepath_save=None, do_verbose=None):
		##Extract global variables
		if (do_verbose is None):
			do_verbose = self.get_info("do_verbose")
		#
		#Print some notes
		if do_verbose:
			print("> Generating table with table_chemistry_2D!")
			time_start = timer.time()
		#

		##Extract model characteristics and chemistry holders
		conchs = self.get_info("list_conchs")
		names = self.get_info("names_conchs")
		do_allow_load = False
		do_save = False
		conchs_chem = [conchs[ii].get_info("model_chemistry")
		 				for ii in range(0, len(conchs))]
		#
		if (model_titles is None):
			model_titles = names
		#
		#Prepare some universal values
		num_conchs = len(conchs)
		num_panels = num_conchs
		#

		##Fetch all chemistry all at once first
		if True: #For easy minimization of this section
			str_table_gas_total = (
				"#Table: Top {1} gas species.\n#Time: {0:.5e} yr\n#Dim: Total\n"
						.format(which_year, num_include))
			str_table_gr_total = (
			"#Table: Top {1} grain species.\n#Time: {0:.5e} yr\n#Dim: Total\n"
						.format(which_year, num_include))
			str_table_gas_1Dcol = (("#Table: Top {1} gas species.\n"
						+"#Time: {0:.5e} yr\n#Dim: 1D_column (m^-2)\n")
						.format(which_year, num_include))
			str_table_gr_1Dcol = (("#Table: Top {1} grain species.\n"
						+"#Time: {0:.5e} yr\n#Dim: 1D_column (m^-2)\n")
						.format(which_year, num_include))
			str_table_gas_2D = (
				("#Table: Top {1} gas species.\n#Time: {0:.5e} yr\n"
				+"#Dim: 2D (m^-3)\n")
						.format(which_year, num_include))
			str_table_gr_2D = (
				("#Table: Top {1} grain species.\n#Time: {0:.5e} yr\n"
				+"#Dim: 2D (m^-3)\n")
						.format(which_year, num_include))
			#
			#Store strings for easy loops later
			dict_str = {"gas|total":str_table_gas_total,
						"gr|total":str_table_gr_total,
						"gas|1D_col":str_table_gas_1Dcol,
						"gr|1D_col":str_table_gr_1Dcol,
						"gas|2D":str_table_gas_2D,
						"gr|2D":str_table_gr_2D}
			#
			str_col = ("#Model\tPhase\tSpatial_Columns\tIndex_Abund_Order\t"
						+"Molecule\tAbund_Abs_perm3\tAbund_Rel\n")
			for curr_key in dict_str:
				dict_str[curr_key] += str_col
		#

		#Initialize files for saving table strings
		for curr_key in dict_str:
			#Extract current phase and dimension information
			tmpsplit = curr_key.split("|")
			curr_phase = tmpsplit[0]
			curr_dim = tmpsplit[1]

			#Set root filename for current table
			filename = ("tableset_chemistry_{1}_{2}_time_{0:.1e}yr"
							.format(which_year, curr_dim, curr_phase
								).replace(".","p").replace("+",""))
			#
			#Initialize current file
			tmppath = os.path.join(filepath_save, (filename+".txt"))
			with open(tmppath, 'w') as openfile:
				openfile.write(dict_str[curr_key])
			#

			#Clear the latest table string
			dict_str[curr_key] = ""
		#
		#Print some notes
		if do_verbose:
			print("\nFiles initialized.\nBeginning iteration through models.\n")
			time_prev = timer.time()
		#

		#Iterate through models
		for ii in range(0, num_conchs):
			#Print some notes
			if do_verbose:
				print("\n---")
				print("Fetching all chemistry for {0}...".format(names[ii]))
			#

			#Extract current structural and chemistry information
			matr_x = conchs[ii].get_info("dict_grid")["matr_x_chemistry"]
			matr_y = conchs[ii].get_info("dict_grid")["matr_y_chemistry"]
			output_chem = conchs_chem[ii]._fetch_disk_chemistry(
							cutoff_minval=cutoff_minval, mode_species="mol",
							which_times_yr=[which_year], rtol_time=rtol_time,
							do_allow_load=do_allow_load, do_save=do_save,
							xbounds_total=xbounds_total,
							ybounds_zoverr=ybounds_zoverr,
							rtol_zoverr=rtol_zoverr,
							list_species_orig=None) #["2D_abs"]
			#Print some notes
			if do_verbose:
				print("Chemistry fetched for {0}.".format(names[ii]))
			#
			list_species = np.sort(list(output_chem["total_abs"].keys()))
			num_species = len(list_species)
			num_x = matr_x.shape[1]
			num_y = matr_y.shape[0]
			#
			#Set number to include if none given
			if (num_include is None):
				num_include_mod = len(list_species)
			else:
				num_include_mod = num_include
			#

			#Iterate through phase and dimension combinations
			for curr_key in dict_str:
				tmpsplit = curr_key.split("|")
				curr_phase = tmpsplit[0]
				curr_dim = tmpsplit[1]
				#Print some notes
				if do_verbose:
					print("Adding to table: {0}:{1}..."
							.format(names[ii], curr_key))
				#

				#For dim 2D:
				if (curr_dim == "2D"):
					#Print some notes
					if do_verbose:
						print("Extracting 2D abundance information.")
					#
					#Iterate through spatial cells
					for yy in range(0, num_y):
						for xx in range(0, num_x):
							#Extract all abs. and rel. abundances
							curr_vals_abs = np.array([np.squeeze(
												output_chem["2D_abs"
												][item][curr_phase])[yy,xx]
												for item in list_species])
							curr_vals_rel = np.asarray([np.squeeze(
												output_chem["2D_rel"
												][item][curr_phase])[yy,xx]
												for item in list_species])
							#

							#Sort species list in abundance order
							curr_inds = np.argsort(curr_vals_abs)[::-1]
							curr_finspec =list_species[curr_inds][0:num_include_mod]
							curr_finabs =curr_vals_abs[curr_inds][0:num_include_mod]
							curr_finrel =curr_vals_rel[curr_inds][0:num_include_mod]
							#
							#Store these values in the table strings
							for zz in range(0, num_include_mod):
								curr_part = (
									("{0}\t{1}\tY={2:.5f},X={3:.5f}\t"
									+"{4}\t{5}\t{6:.5e}\t{7:.5e}\n")
									.format(names[ii], curr_phase,
											matr_y[yy,xx]/au0,
											matr_x[yy,xx]/au0,
											(zz+1), curr_finspec[zz],
											curr_finabs[zz], curr_finrel[zz]))
								dict_str[curr_key] += curr_part
							#
						#
					#
				#
				#For dim 1D_column:
				elif (curr_dim == "1D_col"):
					#Print some notes
					if do_verbose:
						print("Extracting 1D col. abundance information.")
					#
					#Iterate through spatial columns
					for xx in range(0, num_x):
						#Extract all abs. and rel. abundances
						curr_vals_abs = np.array([np.squeeze(
											output_chem["1D_col_abs"
											][item][curr_phase][xx])
											for item in list_species])
						curr_vals_rel = np.asarray([np.squeeze(
											output_chem["1D_col_rel"
											][item][curr_phase][xx])
											for item in list_species])
						#
						#Sort species list in abundance order
						curr_inds = np.argsort(curr_vals_abs)[::-1]
						curr_finspec =list_species[curr_inds][0:num_include_mod]
						curr_finabs =curr_vals_abs[curr_inds][0:num_include_mod]
						curr_finrel =curr_vals_rel[curr_inds][0:num_include_mod]
						#
						#Store these values in the table strings
						for zz in range(0, num_include_mod):
							curr_part = (
								("{0}\t{1}\tX={3:.5f}\t"
								+"{4}\t{5}\t{6:.5e}\t{7:.5e}\n")
								.format(names[ii], curr_phase,
										None, matr_x[0,xx]/au0,
										(zz+1), curr_finspec[zz],
										curr_finabs[zz], curr_finrel[zz]))
							dict_str[curr_key] += curr_part
						#
					#
				#
				#For dim total:
				elif (curr_dim == "total"):
					#Print some notes
					if do_verbose:
						print("Extracting total abundance information.")
					#
					#Extract all abs. and rel. abundances
					curr_vals_abs = np.array([np.squeeze(
											output_chem["total_abs"
											][item][curr_phase])
											for item in list_species])
					curr_vals_rel = np.asarray([np.squeeze(
											output_chem["total_rel"
											][item][curr_phase])
											for item in list_species])
					#
					#Sort species list in abundance order
					curr_inds = np.argsort(curr_vals_abs)[::-1]
					curr_finspec =list_species[curr_inds][0:num_include_mod]
					curr_finabs =curr_vals_abs[curr_inds][0:num_include_mod]
					curr_finrel =curr_vals_rel[curr_inds][0:num_include_mod]
					#
					#Store these values in the table strings
					for zz in range(0, num_include_mod):
						curr_part = (
							("{0}\t{1}\t{4}\t{5}\t{6:.5e}\t{7:.5e}\n")
								.format(names[ii], curr_phase,
										None, None,
										(zz+1), curr_finspec[zz],
										curr_finabs[zz], curr_finrel[zz]))
						dict_str[curr_key] += curr_part
					#
				#
				#Throw error if dimension not recognized
				else:
					raise ValueError("Err: Dim {0} invalid.".format(curr_dim))
				#
			#

			#Print some notes
			if do_verbose:
				print("Chemistry gathered for {0}.".format(names[ii]))
				print("Saving this latest progress to files... Do not cancel!")
			#

			##Append to, save, and close all table strings so far
			for curr_key in dict_str:
				#Extract current phase and dimension information
				tmpsplit = curr_key.split("|")
				curr_phase = tmpsplit[0]
				curr_dim = tmpsplit[1]

				#Set root filename for current table
				filename = ("tableset_chemistry_{1}_{2}_time_{0:.1e}yr"
								.format(which_year, curr_dim, curr_phase
									).replace(".","p").replace("+",""))
				#
				#Append the current table to existing file
				tmppath = os.path.join(filepath_save, (filename+".txt"))
				with open(tmppath, 'a') as openfile:
					openfile.write(dict_str[curr_key])
				#

				#Clear the latest table string
				dict_str[curr_key] = ""
			#

			#Print some notes
			if do_verbose:
				time_now = timer.time()
				time_stamp = (time_now - time_prev)
				print("Chemistry for {0} saved successfully.".format(names[ii]))
				print("{0} completed in {0:.2f} minutes.".format(time_stamp/60))
				print("\n---\n\n\n")
				time_prev = time_now
			#
		#

		##Exit the function
		if do_verbose:
			time_total = (timer.time() - time_start)
			print("> Tables from table_chemistry complete in {0:.2f} minutes."
					.format(time_total/60))
		#
		return
	#
#


##Class: Conch_Disk
##Purpose: Class for methods to prepare, manipulate, and run full disk model
class Conch_Disk(_Base):
	##Method: __init__()
	##Purpose: Initialization of this class instance
	def __init__(self, do_verbose, do_save, dict_grid, dict_names, dict_stars, dict_medium, dict_chemistry, loc_conch=None, num_cores=None, mode_testing=False):
		##Initialize timer
		time_start = timer.time()

		##Load conch, if so requested
		if (loc_conch is not None):
			#Print some notes
			if do_verbose:
				print("Loading conch data from: {0}".format(loc_conch))
			#
			#Load model output
			tmp_name = "{0}_chem.npy".format(loc_conch.split("/")[-1])
			dict_done = np.load(os.path.join(loc_conch, tmp_name),
			 					allow_pickle=True).item()
			#
			#Store loaded model output
			dict_names = dict_done["names"]
			dict_stars = dict_done["stars"]
			dict_grid = dict_done["grid"]
			dict_medium = dict_done["medium"]
			dict_chemistry = dict_done["chemistry"]
			#
			#Print some notes
			if do_verbose:
				print("Conch data loaded successfully. Exiting initialization.")
			#
			#Exit the method early
			#return
		#

		##Store model parameters
		self._storage = {}
		self._store_info(do_verbose, "do_verbose")
		self._store_info(dict_names, "dict_names")
		self._store_info(dict_stars, "dict_stars")
		self._store_info(dict_grid, "dict_grid")
		self._store_info(dict_medium, "dict_medium")
		self._store_info(dict_chemistry, "dict_chemistry")
		self._store_info(num_cores, "_num_cores_arg")
		#

		##Print some notes
		if do_verbose:
			print("Initializing disk model: {0}.".format(dict_names["model"]))
			print("Initializing instance of model's internal structure: {0}..."
					.format(dict_names["which_structure"]))
		#

		##Throw an error if chemistry loaded with new structure
		if ((dict_names["which_structure"] not in ["load"])
						and (dict_names["which_chemistry"] in ["load"])):
			raise ValueError("Err: Old chemistry+new structure not allowed!")
		#

		##Create model directory, if does not already exist
		if (loc_conch is not None): #Added 2024-05-22 patch to fix issue when model generated in different directory (e.g. on a remote node) before loading from different directory later
			dict_names["filepath_save_dictall"] = loc_conch
			dict_names["filepath_chemistry"] = os.path.join(loc_conch,
			 												"chemistry")
			dict_names["filepath_structure"] = os.path.join(loc_conch,
			 												"structure")
			tmpname = dict_names["filename_init_abund"].split("/")[-1]
			dict_names["filename_init_abund"] = os.path.join(
									dict_names["filepath_chemistry"], tmpname)
			fin_path = loc_conch
		else:
			fin_path = dict_names["filepath_save_dictall"]
		#
		fin_path_chem = os.path.join(fin_path, "chemistry")
		fin_path_struct = os.path.join(fin_path, "structure")
		fin_path_plots = os.path.join(fin_path, "plots")
		fin_path_discard = os.path.join(fin_path, "discard")
		if (not os.path.exists(fin_path)): #Main model folder
			if (dict_names["template_model_base"] is not None):
				comm = subprocess.call(["cp", "-r",
				 						dict_names["template_model_base"],
										fin_path])
				#Throw error if subprocess threw error
				if (comm != 0):
					raise ValueError("Err: Error in subprocess for cp!")
			#
			else:
				os.makedirs(fin_path)
			#
		#
		if (not os.path.exists(fin_path_chem)): #Model chemistry folder
			os.makedirs(fin_path_chem)
		if (not os.path.exists(fin_path_struct)): #Model structure folder
			os.makedirs(fin_path_struct)
		if (not os.path.exists(fin_path_plots)): #Model plots folder
			os.makedirs(fin_path_plots)
		if (not os.path.exists(fin_path_discard)): #Model discard folder
			os.makedirs(fin_path_discard)
		#

		##Run disk components
		if (not mode_testing):
			# For prerequisites
			self._do_disk_prerequisites() #For prerequisites
			time_prereq = timer.time()
			#Print some notes
			if do_verbose:
				print(
					"Prerequisites complete in: "
					+f"{(time_prereq-time_start)/60} min.")
			#

			# For structure
			self._do_disk_structure(do_save=do_save) #For structure
			time_structure = timer.time()
			#Print some notes
			if do_verbose:
				print(
					"Structure complete in: "
					+f"{(time_structure-time_prereq)/60} min.")
			#

			# For chemistry
			self._do_disk_chemistry(do_save=do_save) #For chemistry
			time_chemistry = timer.time()
			#Print some notes
			if do_verbose:
				print(
					"Chemistry complete in: "
					+f"{(time_chemistry-time_structure)/60} min.")
		#
		else:
			time_prereq = time_start
			time_structure = time_start
			time_chemistry = time_start
			if do_verbose:
				print("Operating in test mode. No further calculations done.")
		#

		##Exit the method
		time_done = timer.time()
		#Print some notes
		if do_verbose:
			print("Model instance successfully initialized.")
			print(
				"Time for prerequisite phase: "
				+f"{(time_prereq-time_start)/60} min.")
			print(
				"Time for structure phase: "
				+f"{(time_structure-time_prereq)/60} min.")
			print(
				"Time for chemistry phase: "
				+f"{(time_chemistry-time_structure)/60} min.")
			print(
				"Total model runtime: "
				+f"{(time_done-time_start)/60} min.")
		#
		return
		#
	#


	##Method: _do_disk_chemistry()
	##Purpose: Assemble disk's chemistry
	def _do_disk_chemistry(self, do_save):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		y_len = dict_grid["matr_x_chemistry"].shape[0]
		x_len = dict_grid["matr_x_chemistry"].shape[1]

		##Load previous structural model results, if so requested
		#Prepare file names
		do_update_photorates_UV = dict_names["do_update_photorates_UV"]
		filepath_chemistry = dict_names["filepath_chemistry"]
		filepath_save = os.path.join(dict_names["filepath_save_dictall"],
									"{0}_chem.npy".format(dict_names["model"]))
		#

		#Load existing chemistry record, if able
		if os.path.isfile(filepath_save):
			#Print some notes
			if do_verbose:
				print("Available chemistry data loaded from {0}."
						.format(filepath_save))
			#
			#Load previously saved dictionary and booleans
			dict_all_new = np.load(filepath_save, allow_pickle=True).item()
			is_done_reactions = dict_all_new["chemistry"
												]["flags"]["is_done_reactions"]
			is_done_chemmodel = dict_all_new["chemistry"
												]["flags"]["is_done_chemmodel"]
			#
			#Set status of data as loaded
			do_loaded = True
		#
		else:
			do_loaded = False
			is_done_reactions = False
			is_done_chemmodel = False
			dict_chemistry["flags"] = {"is_done_reactions":is_done_reactions,
										"is_done_chemmodel":is_done_chemmodel}
		#

		#Generate new reactions, if not yet completed
		if is_done_reactions:
			#Print some notes
			if do_verbose:
				print("Reactions recycled from previous run.")
			#
			#Nothing to do if reactions already generated
			pass
		else:
			#Print some notes
			if do_verbose:
				print("Calculating new reactions...")
			#
			#Initialize blank spot for new reactions
			dict_chemistry["dict_reactions_new_UV"] = None
			#
			#Initialize empty chemistry model
			if (dict_names["which_chemistry"] in ["pnautilus", "nautilus"]):
				#Print some notes
				if do_verbose:
					print("Generating new reactions in {0} using Nautilus..."
							.format(dict_names["filepath_chemistry"]))
				#Create and run pnautilus instance
				model_chemistry = Conch_Nautilus(do_new_input=True,
				 							do_new_output=True,
											do_verbose=do_verbose,
											dict_grid=dict_grid,
											dict_stars=dict_stars,
											dict_medium=dict_medium,
											dict_chemistry=dict_chemistry,
											dict_names=dict_names)
			#Throw an error if requested structure not recognized
			else:
				raise ValueError("Err: {0} chemistry not supported."
									.format(dict_names["which_chemistry"]))
			#
			#Calculate+Update UV and X-ray photorates
			if (do_update_photorates_UV):
				self._load_data_crosssecs_UV() #Load in mol. cross-section data
				self._set_photorates_UV()
			#
			#Generate new reaction file with updated reactions within
			if (dict_names["which_chemistry"] in ["pnautilus", "nautilus"]):
				#Generate model reaction base folder
				tmp_path = os.path.join(filepath_chemistry,
										preset.dir_reactions_base)
				if (not os.path.exists(tmp_path)):
					os.makedirs(tmp_path)
				#
				#Iterate through slices
				for xx in range(0, x_len):
					#Set directory path for current slice reactions
					curr_filepath_chem_slice = os.path.join(tmp_path,
							preset.dir_reactions_formatperslice.format(xx))
					#Generate model reaction folders per slice
					if (not os.path.exists(curr_filepath_chem_slice)):
						os.makedirs(curr_filepath_chem_slice)
					#
					#Iterate through points in slice
					for yy in range(0, y_len):
						#Fetch the calculated photorates
						dict_reactions_new_all = {"UV":None}
						if (dict_chemistry["dict_reactions_new_UV"] is not None):
							dict_reactions_new_all["UV"] = dict_chemistry[
										"dict_reactions_new_UV"][yy][xx]
						#

						#Fetch the name of the reaction file
						filename_gas_reactions_orig = os.path.join(
							filepath_chemistry,
							preset.pnautilus_filename_reactiongasorig)
						filesave_gas_reactions_updated = os.path.join(
							curr_filepath_chem_slice,
							preset.pnautilus_filename_reactiongas_formatpercell
							.format(yy))
						#Update the reaction file
						reacts._update_reaction_file_pnautilus(
								filename_orig=filename_gas_reactions_orig,
								filename_save=filesave_gas_reactions_updated,
								dict_reactions_new_all=dict_reactions_new_all,
								do_verbose=do_verbose)
						#

						#Copy over grain reactions to slice-reactions folder
						#Since currently no grain reaction changes, just copy
						filename_grain_reactions_orig = os.path.join(
							filepath_chemistry,
							preset.pnautilus_filename_reactiongrainorig)
						filesave_grain_reactions_newloc = os.path.join(
							curr_filepath_chem_slice,
							preset.pnautilus_filename_reactiongrain_formatpercell
							.format(yy))
						comm = subprocess.call(["cp",
				 						filename_grain_reactions_orig,
										filesave_grain_reactions_newloc])
					#
				#
			#
			#Otherwise, throw an error if requested structure not recognized
			else:
				raise ValueError("Err: {0} chemistry not supported."
									.format(dict_names["which_chemistry"]))
			#
			#Save the updated disk model
			if do_save:
				dict_chemistry["flags"]["is_done_reactions"] = True
				#Save combined dictionary
				dict_all = {"names":dict_names, "grid":dict_grid,
				 			"stars":dict_stars,
							"medium":dict_medium, "chemistry":dict_chemistry}
				np.save(filepath_save, dict_all)
				#Print some notes
				if do_verbose:
					print("Reactions calculated and saved: {0}."
							.format(filepath_save))
		#

		#Load previous chemistry model, if already complete
		if is_done_chemmodel:
			#Generate empty chemistry model for access to methods
			model_chemistry = Conch_Nautilus(do_new_input=False,
		 							do_new_output=False, do_verbose=do_verbose,
									dict_grid=dict_all_new["grid"],
									dict_stars=dict_all_new["stars"],
									dict_medium=dict_all_new["medium"],
									dict_chemistry=dict_all_new["chemistry"],
									dict_names=dict_names)
			self._store_info(model_chemistry, "model_chemistry")
		#
		#Otherwise, run new chemistry model
		else:
			#For chemistry mode: Nautilus
			if ((dict_names["which_chemistry"] in ["pnautilus", "nautilus"])
										or (not os.path.isfile(filepath_save))):
				#Print some notes
				if do_verbose:
					print("Generating new chemistry in {0} using Nautilus..."
							.format(dict_names["filepath_chemistry"]))
				#Create and run pnautilus instance
				model_chemistry = Conch_Nautilus(do_new_input=True,
				 							do_new_output=True,
											do_verbose=do_verbose,
											dict_grid=dict_grid,
											dict_stars=dict_stars,
											dict_medium=dict_medium,
											dict_chemistry=dict_chemistry,
											dict_names=dict_names)
			#
			#Otherwise, throw an error if requested structure not recognized
			else:
				raise ValueError("Err: {0} chemistry not supported."
									.format(dict_names["which_chemistry"]))
			#
			#Run the chemistry model
			model_chemistry.run_chemistry()
			#
			#Save the updated disk model
			if do_save:
				dict_chemistry["flags"]["is_done_chemmodel"] = True
				#Save combined dictionary
				dict_all = {"names":dict_names, "grid":dict_grid,
				 			"stars":dict_stars,
							"medium":dict_medium, "chemistry":dict_chemistry}
				np.save(filepath_save, dict_all)
				#Print some notes
				if do_verbose:
					print("Post-chem. model saved: {0}.".format(filepath_save))
		#

		##Exit the method
		if do_verbose:
			print("Chemistry complete.\n")
		#
		return
	#


	##Method: _do_disk_structure()
	##Purpose: Assemble disk's structure
	def _do_disk_structure(self, do_save):
		#Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		do_loaded = False #Boolean for whether or not previous data loaded


		##Load previous structural model results, if so requested
		filepath_save = os.path.join(dict_names["filepath_save_dictall"],
									"{0}.npy".format(dict_names["model"]))
		#Load and store previously saved structural information
		if os.path.isfile(filepath_save):
			#Print some notes
			if do_verbose:
				print("Structural data already loaded previously from {0}."
						.format(filepath_save))
			#
			do_loaded = True
		#
		#Otherwise, create new structure via radmc-3D
		elif ((dict_names["which_structure"] in ["radmc3d"])
									or (not os.path.isfile(filepath_save))):
			#Print some notes
			if do_verbose:
				print("Generating new structural grid and density...")
			#
			#Prepare disk grid values
			self._create_disk_structure_grid() #For disk grid
			self._create_disk_structure_density_dust() #Disk dust density
			#
			#Print some notes
			if do_verbose:
				print("Generating new structure in {0} using radmc-3D..."
						.format(dict_names["filepath_structure"]))
			#Create and run radmc3d instance
			model_structure = Conch_Radmc3d(do_load_input=False,
										do_load_output=False,
										do_verbose=do_verbose,
			 							dict_stars=dict_stars,
										dict_grid=dict_grid,
										dict_medium=dict_medium,
										dict_chemistry=dict_chemistry,
										dict_names=dict_names)
			#
			#Prepare gas temperature, density, and radiation distributions
			model_structure.run_radmc3d_dusttemperature()
			model_structure.run_radmc3d_radiationfields()
			self._create_disk_structure_temperature_gas()
			self._create_disk_structure_density_gas() #Disk gas density
			self._create_disk_structure_extinction() #Disk radiation extinction
			self._set_ionrate_X() #Calculate matrix of X-ray primary ion.rates
		#
		#Otherwise, throw an error if requested structure not recognized
		else:
			raise ValueError("Err: {0} structure not supported."
								.format(dict_names["which_structure"]))
		#


		##Save the disk model, if so requested
		if do_save and (not do_loaded):
			#Save combined dictionary
			dict_all = {"names":dict_names, "grid":dict_grid,
			 			"stars":dict_stars,
						"medium":dict_medium, "chemistry":dict_chemistry}
			np.save(filepath_save, dict_all)
			#Print some notes
			if do_verbose:
				print("Post-structure model saved: {0}.".format(filepath_save))
		#


		##Plot structural components for verification purposes
		if do_save:
			self._plot_disk_structure()
		#


		##Exit the method
		if do_verbose:
			print("Structure complete.\n")
		#
		return
	#


	##Method: _do_disk_prerequisites()
	##Purpose: Prepare disk prerequisites
	def _do_disk_prerequisites(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		num_cores_new = self.get_info("_num_cores_arg")
		#


		##Load previous structural model results, if so requested
		filepath_chemistry = dict_names["filepath_chemistry"]
		filepath_save_struct = os.path.join(dict_names["filepath_save_dictall"],
									"{0}.npy".format(dict_names["model"]))
		filepath_save_chem = os.path.join(dict_names["filepath_save_dictall"],
									"{0}_chem.npy".format(dict_names["model"]))
		#


		##Load previous structural model prerequisites, if so requested
		if os.path.isfile(filepath_save_struct):
			#Print some notes
			if do_verbose:
				print("Previous prerequisite structure data exists.")
				print("Loading structural data from {0}..."
						.format(filepath_save_struct))
			#
			dict_all_new =np.load(filepath_save_struct,allow_pickle=True).item()
			self._store_info(dict_all_new["stars"], "dict_stars")
			self._store_info(dict_all_new["grid"], "dict_grid")
			self._store_info(dict_all_new["medium"], "dict_medium")
			#
			#Print some notes
			if do_verbose:
				print("Previous structure data loaded.")
			#
		#
		#Otherwise, create new structure via radmc-3D
		elif ((dict_names["which_structure"] in ["radmc3d"])
								or (not os.path.isfile(filepath_save_struct))):
			#Print some notes
			if do_verbose:
				print("Generating new structural prerequisites...")
			#
			#Prepare disk grid values
			self._create_disk_time() #For disk time array (used for chemistry)
			self._load_spectrum_stars_base() #For loading stellar spectra
			#
		#
		#Otherwise, throw an error if requested structure not recognized
		else:
			raise ValueError("Err: {0} structure not supported."
								.format(dict_names["which_structure"]))
		#


		##Load previous chemistry model prerequisites, if so requested
		if os.path.isfile(filepath_save_chem):
			#Print some notes
			if do_verbose:
				print("Chemistry prerequisites previously computed.")
				print("Loading chemistry data from {0}..."
						.format(filepath_save_chem))
			#
			dict_all_new = np.load(filepath_save_chem, allow_pickle=True).item()
			self._store_info(dict_all_new["stars"], "dict_stars")
			self._store_info(dict_all_new["grid"], "dict_grid")
			self._store_info(dict_all_new["medium"], "dict_medium")
			self._store_info(dict_all_new["chemistry"], "dict_chemistry")
			#
			#Print some notes
			if do_verbose:
				print("Previous chemistry data loaded.")
			#
		#
		#Otherwise, set chemistry prerequisites
		elif ((dict_names["which_chemistry"] in ["pnautilus", "nautilus"])
								or (not os.path.isfile(filepath_save_chem))):
			#Print some notes
			if do_verbose:
				print("Running chemistry prerequisites...")
			#

			#Prepare elements
			self._load_elements() #Load all possible elements
			self._load_elemental_abundances() #Load elemental abundances
			#

			#Prepare cross-sections
			self._generate_crosssecs_X() #Generate X-ray molecular cross-section
			#
		#
		#Otherwise, throw an error if requested chemistry not recognized
		else:
			raise ValueError("Err: {0} chemistry not supported."
								.format(dict_names["which_chemistry"]))
		#

		#Update the number of cores to use, if any passed in
		if (num_cores_new is not None):
			dict_grid = self.get_info("dict_grid")
			dict_grid["num_cores"] = num_cores_new
		#

		#Exit the method
		return
	#


	##Method: _create_disk_time()
	##Purpose: Calculate disk's temporal evolution
	def _create_disk_time(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		#
		start_time_disk = dict_grid["time_start_yr"]
		end_time_disk = dict_grid["time_end_yr"]
		scheme_time_disk = dict_grid["scheme_time"]
		num_time_disk = dict_grid["num_time"]
		custom_time_disk = dict_grid["custom_time_arr_yr"]
		#
		#Print some notes
		if do_verbose:
			print("Running _create_disk_time...")
		#
		#Raise an error if already computed
		checklist = ["arr_time_yr"]
		if any([item in dict_grid for item in checklist]):
			raise ValueError("Err: Time array already computed!")
		#


		##Compute and store temporal array using specified scheme
		#For disk
		if scheme_time_disk == "linear":
			arr_time_disk = np.linspace(start_time_disk, end_time_disk,
			 						num_time_disk, endpoint=True)
		elif scheme_time_disk == "log":
			arr_time_disk = np.logspace(np.log10(start_time_disk),
			 						np.log10(end_time_disk),
									num_time_disk, endpoint=True)
		elif scheme_time_disk == "custom":
			arr_time_disk = custom_time_disk
		else:
			raise ValueError("Err: Temporal scheme {0} not recognized!"
								.format(scheme_time_disk))
		#
		#Cast time array to precision of Nautilus
		arr_time_disk = np.array([float("{0:.3E}".format(item))
									for item in arr_time_disk])
		#Store the temporal array
		dict_grid["arr_time_yr"] = arr_time_disk
		#


		##Exit the method
		if do_verbose:
			print("Run of _create_disk_time() complete!")
		#
		return
	#


	##Method: _create_disk_structure_density_dust()
	##Purpose: Calculate disk's base dust density
	def _create_disk_structure_density_dust(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_grid = self.get_info("dict_grid")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		which_fields = preset.disk_radiation_fields
		#
		which_normx = dict_names["which_normx"]
		frac_normx = dict_names["frac_normx"]
		matr_x_struct = dict_grid["matr_x_structure"]
		matr_y_struct = dict_grid["matr_y_structure"]
		out_x = dict_medium["out_dimx"]
		#
		matr_x_chem = dict_grid["matr_x_chemistry"]
		matr_y_chem = dict_grid["matr_y_chemistry"]
		matr_y_diskinds_chem = dict_grid["matr_y_diskinds_chemistry"]
		points_yx_struct = dict_grid["points_yx_structure"]
		#
		which_density_dust = dict_names["which_dens_dust"]
		which_density_gas = dict_names["which_dens_gas"]
		#
		lstar = dict_stars["lstar"]
		mstar = dict_stars["mstar"]
		mdiskgas = dict_medium["mdiskgas"]
		theta_flare = dict_medium["theta_flare"]
		frac_dustovergasmass = dict_medium["frac_dustovergasmass"]
		frac_freqdustovertotaldust = dict_medium["frac_freqdustovertotaldust"]
		fracs_dustmdisk = dict_medium["fracs_dustmdisk"]
		fracs_dustscaleH = dict_medium["fracs_dustscaleH"]
		num_speciesdust = len(dict_medium["fracs_dustspecies"]) #Num. dust spec.
		ind_chemgr = dict_medium["ind_chemistry_gr"]
		#

		#Count up and store number of solid density distributions
		tmp_dict = {}
		#Iterate through fields, including for dust
		for curr_field in (which_fields+["dust"]):
			num_distrs = 0
			#Iterate through distributions for this field
			for curr_distr in preset.radmc3d_dict_whichdustdens[curr_field]:
				if (curr_distr in ["dust"]):
					num_distrs += num_speciesdust
				elif (curr_distr in ["gas", "Hatomic"]):
					num_distrs += 1
				else:
					raise ValueError("Err: {0} not recognized!".format(curr_distr))
			#
			#Store this value for later use
			tmp_dict[curr_field] = num_distrs
		#
		dict_medium["num_distrs_dustdens_perfield"] = tmp_dict
		#

		#Compute and store the mass of the grain
		dict_medium["mass_pergr"] = (dict_medium["densmass_grain_cgs"]
							* (4.0/3 *pi *(dict_medium["radius_grain_cgs"]**3))
							) * 1E-3 #[g] -> [kg]
		#
		#Print some notes
		if do_verbose:
			print("Running _create_disk_structure_density_dust...")
		#


		##Compute gas and dust density and related values using given scheme
		#Print some notes
		if do_verbose:
			print("Preparing empirical inputs for density calculation...")
		#
		#For normalization x-value
		norm_x = math._calc_normx(which_normx=which_normx, R_out=out_x,
									frac=frac_normx)
		#
		#For density
		if do_verbose:
			print("Calculating 2D dust density distribution...")
		#
		res_dens = math.calc_density2D_base(do_verbose=do_verbose,
								which_density=which_density_dust,
								matr_x=matr_x_struct,
								matr_y=matr_y_struct,
								norm_x=norm_x, lstar=lstar,
								mstar=mstar, mdiskgas=mdiskgas,
								theta_flare=theta_flare,
								frac_dustovergasmass=frac_dustovergasmass,
								fracs_dustmdisk=fracs_dustmdisk,
								fracs_dustscaleH=fracs_dustscaleH)
		#


		##Store computed dust density [kg/m^3]
		dict_medium["norm_x"] = norm_x
		#Print some notes
		if do_verbose:
			print("Storing 2D dust density distribution and related output...")
		#
		#For structural versions
		dict_medium["matr_scaleH_structure"] = res_dens["matr_scaleH"]
		dict_medium["matr_sigma_gas_structure"] = res_dens["matr_sigma_gas"]
		dict_medium["matr_sigma_dust_structure"] = res_dens["matr_sigma_dust"]
		dict_medium["volmassdens_dustpergr_structure"] = res_dens[
													"volmassdens_dust_pergr"]
		dict_medium["volmassdens_dustpergr_chemistry"] = [
					math._interpolate_pointstogrid(
							old_matr_values=item,old_points_yx=points_yx_struct,
							new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
							inds_valid=matr_y_diskinds_chem)
					for item in dict_medium["volmassdens_dustpergr_structure"]]
		#
		volmassdens_dustall_struct = res_dens["volmassdens_dust_all"]
		dict_medium["volmassdens_dustall_structure"] =volmassdens_dustall_struct
		dict_medium["volmassdens_dustall_chemistry"] = (
						math._interpolate_pointstogrid(
							old_matr_values=volmassdens_dustall_struct,
							old_points_yx=points_yx_struct,
							new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
							inds_valid=matr_y_diskinds_chem))
		#


		##Store computed gas density [1/m^3] now, if scheme requests it
		if which_density_gas in ["analytic", "benchmark_radmc3d"]:
			#Print some notes
			if do_verbose:
				print("Storing byproduct 2D gas density output as well...")
			#
			#Store the gas density calculation output
			dict_medium["volnumdens_nH_structure"] = res_dens["volnumdens_nH"]
			#
			dict_medium["volnumdens_nH_Hatomic_structure"] = res_dens[
													"volnumdens_nH_Hatomiconly"]
			dict_medium["volnumdens_nH_Hmolec_structure"] = res_dens[
													"volnumdens_nH_H2only"]
			#
			dict_medium["volnumdens_nH_Hatomic_chemistry"] = (
					math._interpolate_pointstogrid(
						old_matr_values=res_dens["volnumdens_nH_Hatomiconly"],
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem))
			dict_medium["volnumdens_nH_Hmolec_chemistry"] = (
					math._interpolate_pointstogrid(
						old_matr_values=res_dens["volnumdens_nH_H2only"],
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem))
		#


		##Exit the method
		if do_verbose:
			print("Run of _create_disk_structure_density_dust() complete!")
		#
		return
	#


	##Method: _create_disk_structure_density_gas()
	##Purpose: Calculate disk's base gas density
	def _create_disk_structure_density_gas(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_grid = self.get_info("dict_grid")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		#
		matr_x_struct = dict_grid["matr_x_structure"]
		matr_y_struct = dict_grid["matr_y_structure"]
		matr_x_chem = dict_grid["matr_x_chemistry"]
		matr_y_chem = dict_grid["matr_y_chemistry"]
		matr_y_diskinds_chem = dict_grid["matr_y_diskinds_chemistry"]
		points_yx_struct = dict_grid["points_yx_structure"]
		#
		which_density_gas = dict_names["which_dens_gas"]
		mstar = dict_stars["mstar"]
		#
		#Print some notes
		if do_verbose:
			print("Running _create_disk_structure_density_gas...")
		#


		##Compute gas density using requested scheme
		#For cases where computed elsewhere
		if which_density_gas in ["analytic", "benchmark_radmc3d"]:
			#No need to recompute, so just pass ahead
			pass
		#
		#For case of vertical integration of hydrostatic equilibrium equation
		elif which_density_gas in ["exact"]:
			matr_tempgas = dict_medium["matr_tempgas_structure"]
			matr_sigma_gas = dict_medium["matr_sigma_gas_structure"]
			matr_sigma_dust = dict_medium["matr_sigma_dust_structure"]
			matr_scaleH = dict_medium["matr_scaleH_structure"]
			res_dens = math.calc_density2D_base(which_density=which_density_gas,
		 				matr_x=matr_x_struct, matr_y=matr_y_struct, mstar=mstar,
						matr_tempgas=matr_tempgas,
						matr_sigma_gas=matr_sigma_gas,
						matr_sigma_dust=matr_sigma_dust,
						matr_scaleH=matr_scaleH)
			dict_medium["volnumdens_nH_structure"] = res_dens["volnumdens_nH"]
		#
		#Otherwise, throw error if density scheme not recognized
		else:
			raise ValueError("Err: Invalid gas density scheme {0}!"
							.format(which_density_gas))
		#


		##Determine maximum y-point at each x-point within maximum scale height
		thres_mindensgas_cgs = preset.chemistry_thres_mindensgas_cgs
		matr_scaleH_struct = dict_medium["matr_scaleH_structure"]
		matr_scaleH_chem = math._interpolate_pointstogrid(
						old_matr_values=dict_medium["matr_scaleH_structure"],
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem)
		matr_nH_struct = dict_medium["volnumdens_nH_structure"]
		matr_nH_chem = math._interpolate_pointstogrid(
						old_matr_values=dict_medium["volnumdens_nH_structure"],
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem)
		#


		##Store computed density [1/m^3]
		#For chemistry versions
		dict_medium["matr_scaleH_chemistry"] = matr_scaleH_chem
		dict_medium["volnumdens_nH_chemistry"] = matr_nH_chem
		#


		##Exit the method
		if do_verbose:
			print("Run of _create_disk_structure_density_gas complete.")
		#
		return
	#


	##Method: _create_disk_structure_extinction()
	##Purpose: Calculate disk's extinction of radiation
	def _create_disk_structure_extinction(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		#
		matr_fieldrad_UV_perm_struct = dict_medium[
									"matr_fieldrad_photon_UVtot_perm_structure"]
		matr_fieldrad_UV_perm_chem = dict_medium[
									"matr_fieldrad_photon_UVtot_perm_chemistry"]
		#
		ylen_struct = matr_fieldrad_UV_perm_struct.shape[0]
		xlen_struct = matr_fieldrad_UV_perm_struct.shape[1]
		ylen_chem = matr_fieldrad_UV_perm_chem.shape[0]
		xlen_chem = matr_fieldrad_UV_perm_chem.shape[1]
		#
		#Print some notes
		if do_verbose:
			print("Running _create_disk_structure_extinction...")
		#


		##Estimate the UV extinction per cell
		matr_ext_UV_struct_raw =np.ones(shape=matr_fieldrad_UV_perm_struct.shape
										)*np.inf
		matr_ext_UV_chem_raw = np.ones(shape=matr_fieldrad_UV_perm_chem.shape
										)*np.inf
		#For structure matrix
		for yy in range(0, ylen_struct):
			for xx in range(0, xlen_struct):
				matr_ext_UV_struct_raw[yy,xx,:] = math.calc_extinction_mag(
						I_orig=matr_fieldrad_UV_perm_struct[(ylen_struct-1),xx],
						I_new=matr_fieldrad_UV_perm_struct[yy,xx],
						do_mindenom=True)
		#
		#For chemistry matrix
		for yy in range(0, ylen_chem):
			for xx in range(0, xlen_chem):
				matr_ext_UV_chem_raw[yy,xx] = math.calc_extinction_mag(
						I_orig=matr_fieldrad_UV_perm_chem[(ylen_chem-1),xx],
						I_new=matr_fieldrad_UV_perm_chem[yy,xx],
						do_mindenom=True)
		#


		##Extract the median and quartile values per disk cell
		#For UV
		matr_ext_UV_struct = np.median(matr_ext_UV_struct_raw, axis=2)
		matr_ext_UV_struct_err = np.quantile(matr_ext_UV_struct_raw,
		 									q=[0.25, 0.75], axis=2)
		matr_ext_UV_chem = np.median(matr_ext_UV_chem_raw, axis=2)
		matr_ext_UV_chem_err = np.quantile(matr_ext_UV_chem_raw,
		 									q=[0.25, 0.75], axis=2)
		#
		#Throw error if any invalid values
		if (np.any(np.isnan(matr_ext_UV_struct))
					or np.any(np.isinf(np.abs(matr_ext_UV_struct)))):
			#Plot spectra
			matr_x_struct = dict_grid["matr_x_structure"]
			matr_y_struct = dict_grid["matr_y_structure"]
			matr_x_chem = dict_grid["matr_x_chemistry"]
			matr_y_chem = dict_grid["matr_y_chemistry"]
			matr_outstruct = matr_fieldrad_UV_perm_struct[:,:,0].copy()
			plt.scatter(matr_x_struct, matr_y_struct, c=np.log10(matr_outstruct))
			plt.colorbar()
			plt.show()
			plt.scatter(matr_x_struct, matr_y_struct, c=matr_ext_UV_struct)
			plt.show()
			plt.scatter(matr_x_chem, matr_y_chem, c=matr_ext_UV_chem)
			plt.show()
			#
			#Raise the error
			raise ValueError("Err: Invalid extinction values!")
		#


		##Store the results
		dict_medium["matr_extinction_UV_structure"] = matr_ext_UV_struct
		dict_medium["matr_extinction_UV_chemistry"] = matr_ext_UV_chem
		#
		dict_medium["matr_extinction_UV_structure_err"] = matr_ext_UV_struct_err
		dict_medium["matr_extinction_UV_chemistry_err"] = matr_ext_UV_chem_err
		#


		##Exit the method
		if do_verbose:
			print("Run of _create_disk_structure_extinction complete.")
		#
		return
	#


	##Method: _create_disk_structure_grid()
	##Purpose: Create grid to use for computing disk structure
	def _create_disk_structure_grid(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		#
		is_dimx = dict_grid["is_dimx"] #Boolean for if dimension used
		is_dimy = dict_grid["is_dimy"]
		is_dimz = dict_grid["is_dimz"]
		thres_dist_au0 = dict_grid["thres_dist_structvschem_au"]
		#
		edge_struct_radius = dict_grid["edge_structure_radius"]
		edge_struct_theta = dict_grid["edge_structure_theta"]
		#
		matr_chem_x = dict_grid["matr_x_chemistry"]
		matr_chem_y = dict_grid["matr_y_chemistry"]
		#
		#Compute temporary scale height to use for disk height
		lstar = dict_stars["lstar"]
		mstar = dict_stars["mstar"]
		theta_flare = dict_medium["theta_flare"]
		matr_tempmid = math.calc_temperature_midplane(arr_x=matr_chem_x,
	 							lstar=lstar, theta_flare=theta_flare)
		matr_scaleH_chem = math.calc_scaleheight(arr_x=matr_chem_x, mstar=mstar,
	 							temp_mid=matr_tempmid, lstar=lstar)
		matr_maxscaleH_chem = (matr_scaleH_chem
		 						* preset.structure_value_maxnumscaleH)
		dict_grid["matr_y_diskinds_chemistry"] = (matr_chem_y
		 											<= matr_maxscaleH_chem)
		#
		#Throw error if overall y does not reach maximum scale height
		if (matr_chem_y.max() < matr_maxscaleH_chem.max()):
			raise ValueError("Err: y_array not long enough...\n{0} vs {1}"
						.format(matr_chem_y.max()/au0,
						 		matr_maxscaleH_chem.max()/au0))
		#


		##Generate coordinate values for spherical structure
		#For x-axis values
		cen_struct_radius = math._calc_cellcenters(edge_struct_radius)
		len_struct_radius = len(cen_struct_radius)
		#
		#For y-axis values
		cen_struct_theta = math._calc_cellcenters(edge_struct_theta)
		len_struct_theta = len(cen_struct_theta)
		#
		#For z-axis values
		if is_dimz:
			cen_struct_phi = dict_grid["cen_structure_phi"]
			edge_struct_phi = math._calc_celledges_fromcenters(cen_struct_phi)
		else:
			#Set default phi-axis resolution if not given
			edge_struct_phi = np.linspace(0, 2*pi, 1+1, endpoint=True)
			cen_struct_phi = np.array([
							(0.5 * (edge_struct_phi[1] + edge_struct_phi[0]))])
			dict_grid["cen_structure_phi"] = cen_struct_phi
		len_struct_phi = len(cen_struct_phi)
		#


		##Store dimensions and exit the function
		dict_grid["cen_structure_radius"] = cen_struct_radius #[m]
		dict_grid["cen_structure_theta"] = cen_struct_theta #[radians]
		dict_grid["edge_structure_phi"] = edge_struct_phi #[radians]
		#
		shape_raw_struct = (len_struct_theta, len_struct_radius, len_struct_phi)
		dict_grid["shape_structure"] = [item for item in shape_raw_struct
		 							if (item != 1)] #Compress
		dict_grid["num_points_structure"] = np.product(shape_raw_struct)
		#


		##Convert 1D radius,theta structure arrays to 2D Cartesian matrices
		matr_radius, matr_theta =np.meshgrid(cen_struct_radius,cen_struct_theta)
		dict_conv = math._conv_sphericalandcartesian(
										which_conv="sphericaltocartesian",
										radius=matr_radius, theta=matr_theta)
		matr_struct_x = dict_conv["x"]
		matr_struct_y = dict_conv["y"]
		list_struct_points_yx = np.asarray([
							matr_struct_y.flatten(), matr_struct_x.flatten()]
							).T #List of x,y structure points
		#Store these matrices
		dict_grid["matr_x_structure"] = matr_struct_x
		dict_grid["matr_y_structure"] = matr_struct_y
		dict_grid["points_yx_structure"] = list_struct_points_yx
		#


		##Verify that all chemistry points are near-enough to a structure point
		#Block out disk locations that are above disk
		inds_abovedisk = (matr_chem_y > matr_maxscaleH_chem)
		matr_chem_y_tmp = matr_chem_y.copy()
		matr_chem_y_tmp[inds_abovedisk] = np.nan
		#Convert x,y chemistry arrays into set of points
		list_chem_points_yx = np.asarray([
							matr_chem_y_tmp.flatten(), matr_chem_x.flatten()]
							).T #List of x,y structure points
		#Trim points that are above height of disk
		list_chem_points_yx_trim = np.asarray([item
		 								for item in list_chem_points_yx
										if (not np.isnan(item[0]))])
		#Compute distances of nearest neighbors for each valid point
		neighbors = sklearn_nearestneighbors(n_neighbors=1
											).fit(list_struct_points_yx/au0)
		distances_au0,indices=neighbors.kneighbors(list_chem_points_yx_trim/au0)
		#Raise error if any distances exceed threshold
		if any(distances_au0 >= thres_dist_au0):
			plt.scatter(matr_struct_x/au0, matr_struct_y/au0,
			 			alpha=0.5, color="black")
			plt.scatter(matr_chem_x/au0, matr_chem_y/au0,
			 			alpha=0.5, color="blue")
			plt.show()
			raise ValueError("Err: Distance between structure and chemistry "
							+"points exceed threshold of {0}au!\n"
							.format(thres_dist_au0)
							+"{0}".format(np.sort(distances_au0.flatten())))
		#


		##Exit the method
		if do_verbose:
			print("Run of _create_disk_structure_grid() complete.")
		#
		return
	#


	##Method: _create_disk_structure_temperature_gas()
	##Purpose: Calculate gas temperature distribution
	def _create_disk_structure_temperature_gas(self):
		##Extract global variables
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		#
		which_scheme = dict_names["which_scheme_tempgas"]
		matr_x_struct = dict_grid["matr_x_structure"]
		matr_y_struct = dict_grid["matr_y_structure"]
		matr_x_chem = dict_grid["matr_x_chemistry"]
		matr_y_chem = dict_grid["matr_y_chemistry"]
		matr_y_diskinds_chem = dict_grid["matr_y_diskinds_chemistry"]
		points_yx_struct = dict_grid["points_yx_structure"]
		#

		##Calculate gas temperature based on scheme
		#For equal-to-dust-temp. scheme
		if which_scheme == "equal_dust":
			matr_tempgas_structure = dict_medium["matr_tempdust_structure"]
			matr_tempgas_chemistry = dict_medium["matr_tempdust_chemistry"]
		#
		#For parametric synthesized gas-temperature scheme
		elif which_scheme == "parametric_synthetic":
			matr_tempgas_structure = math.calc_tempgas_parametric_synthetic(
						matr_y=matr_y_struct, matr_x=matr_x_struct,
						matr_tempdust=dict_medium["matr_tempdust_structure"],
						lstar=dict_stars["lstar"],
						matr_scaleH=dict_medium["matr_scaleH_structure"])
			matr_tempgas_chemistry = math._interpolate_pointstogrid(
							old_matr_values=matr_tempgas_structure,
							old_points_yx=points_yx_struct,
							new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
							inds_valid=matr_y_diskinds_chem)
		#
		#For parametric gas-temperature scheme from Bruderer+2012 write-up
		elif which_scheme == "parametric_bruderer2012":
			matr_tempgas_structure = math.calc_tempgas_parametric_bruderer2012(
						matr_y=matr_y_struct, matr_x=matr_x_struct,
						matr_tempdust=dict_medium["matr_tempdust_structure"],
						matr_nH=dict_medium["volnumdens_nH_structure"],
						matr_energyflux_UVtot=dict_medium[
											"matr_energyflux_UVtot_structure"])
			matr_tempgas_chemistry = math._interpolate_pointstogrid(
							old_matr_values=matr_tempgas_structure,
							old_points_yx=points_yx_struct,
							new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
							inds_valid=matr_y_diskinds_chem)
		#
		#Otherwise, throw error if scheme not recognized
		else:
			raise ValueError("Err: Gas temperature scheme not recognized!")
		#

		##Save the computed temperature and exit the method
		dict_medium["matr_tempgas_structure"] = matr_tempgas_structure
		dict_medium["matr_tempgas_chemistry"] = matr_tempgas_chemistry
		return
	#


	##Method: _generate_crosssecs_X()
	##Purpose: Generate X-ray radiation cross-section
	def _generate_crosssecs_X(self):
		##Extract global variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_chemistry = self.get_info("dict_chemistry")
		which_scheme = preset.scheme_crosssec_X #Which mode for X-ray crossec
		#
		filepath_plots = dict_names["filepath_plots"]
		possible_elements = dict_chemistry["chemistry_possible_elements"]
		dict_abunds_el = dict_chemistry["elemental_abundances"]
		#
		#Print some notes
		if do_verbose:
			print("\n> Calling _generate_crosssec_X() within argonaut...")
		#

		#Assemble and store the X-ray cross-sections per element
		arr_energy = math._conv_waveandfreq(which_conv="wavetoenergy",
								wave=dict_stars["wavelength_spectrum_X_struct"])
		if (which_scheme == "parametric"): #For parametric method
			dict_chemistry["dict_mol_X"] =math._assemble_crosssec_X_parametric(
												energy_inJ_orig=arr_energy,
		 										do_plot=True,
												filepath_plot=filepath_plots)
		else:
			raise ValueError("Err: Scheme {0} invalid!".format(which_scheme))
		#

		#Exit the method
		if do_verbose:
			print("\n> Run of _generate_crosssec_X() complete!")
		#
		return
	#


	##Method: _load_data_crosssecs_UV()
	##Purpose: Load data for molecular cross-sections
	def _load_data_crosssecs_UV(self, do_skip_elements=True):
		##Extract global variables
		do_verbose = self.get_info("do_verbose")
		dict_stars = self.get_info("dict_stars")
		dict_chemistry = self.get_info("dict_chemistry")
		#Initialize dictionary to store cross-sections
		chemistry_possible_elements = dict_chemistry[
											"chemistry_possible_elements"]
		#
		#Print some notes
		if do_verbose:
			print("\n> Calling _load_data_crosssecs_UV within argonaut...")
		#

		##Call external loading of UV cross-sectional data
		do_ignore_JK = True #For now; reflects Nautilus convention
		dict_chemistry["dict_mol_UV"] = utils._load_data_crosssecs_UV(
						do_skip_elements=do_skip_elements,
						do_skip_bannedspecies=True, do_ignore_JK=do_ignore_JK,
						possible_elements=chemistry_possible_elements,
						do_verbose=do_verbose,
						wavelength_min=preset.wavelength_range_rates_UV[0],
						wavelength_max=preset.wavelength_range_rates_UV[1])
		#

		##Exit the method
		if do_verbose:
			print("Call of _load_data_crosssecs_UV() complete!")
		#
		return
	#


	##Method: _load_elements()
	##Purpose: Load data for allowed chemical elements
	def _load_elements(self):
		##Extract global variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		filepath_chemistry = dict_names["filepath_chemistry"]
		filename_el = os.path.join(filepath_chemistry,
										preset.pnautilus_filename_element)
		#NOTE: Must be generated from elemental file, since minisculee abund...
		#		...of elements not shown in init. abund. possible
		#
		#Print some notes
		if do_verbose:
			print("\n> Running _load_elements...")
		#

		#Generate list of all possible elements
		list_el = utils._load_elements(filename=filename_el)
		dict_chemistry["chemistry_possible_elements"] = list_el
		#

		##Exit the method
		if do_verbose:
			print("Run of _load_elements() complete!")
		#
		return
	#


	##Method: _load_elemental_abundances()
	##Purpose: Load elemental abundances from initial abundances file.
	def _load_elemental_abundances(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		list_el = dict_chemistry["chemistry_possible_elements"]
		#
		#Generate dictionaries of molecular and elemental abundances
		filename_init_abunds = dict_names["filename_init_abund"]
		#
		#Load the elements from the file
		dict_abund_molec = {}
		dict_totabund_el = {key:0 for key in list_el}
		data = np.genfromtxt(filename_init_abunds, dtype=str, comments="!")
		for ii in range(0, data.shape[0]):
			curr_mol = data[ii,0]
			curr_eldict = utils._get_elements(mol=curr_mol, do_ignore_JK=True,
			 									do_strip_orientation=True)
			curr_val = float(data[ii,2].replace("D", "E"))
			#

			#Throw error if this molecule appears multiple times
			if (curr_mol in dict_abund_molec):
				raise ValueError("Err: Duplicate mol. {0} in init. abunds!"
								.format(curr_mol))
			#

			#Store molecular abundance
			dict_abund_molec[curr_mol] = curr_val
			#

			#Store elemental abundance breakdown of current molecule
			for curr_key in curr_eldict:
				#Store current abundance
				dict_totabund_el[curr_key] += (curr_val / curr_eldict[curr_key])
			#
		#

		#Take note of 0-abundance H, H2 if needed
		if ("H" not in dict_abund_molec):
			dict_abund_molec["H"] = 0
		#
		if ("H2" not in dict_abund_molec):
			dict_abund_molec["H2"] = 0
		#

		#Store the compiled dictionaries of abundances
		dict_chemistry["init_molecular_abundances"] = dict_abund_molec
		dict_chemistry["elemental_abundances"] = dict_totabund_el

		#Exit the method
		return
	#


	##Method: _load_spectrum_stars_base()
	##Purpose: Load stellar spectra into model
	def _load_spectrum_stars_base(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_stars = self.get_info("dict_stars")
		#
		datas_stars_UV = dict_stars["datas_spectra_stars_UV_1RSun"]
		datas_stars_X = dict_stars["datas_spectra_stars_X_1RSun"]
		datas_stars_SED = dict_stars["datas_spectra_stars_SED_1RSun"]
		#

		##Throw error if multiple forms of spectra given
		if (datas_stars_SED is None):
			raise ValueError("Please pass in SED data.")
		#

		##Call method for processing stellar spectra
		self._load_spectrum_stars_fromSED()
		#

		##Exit the method
		return
	#


	##Method: _load_spectrum_stars_fromSED()
	##Purpose: Load stellar spectra into model when single SED given
	def _load_spectrum_stars_fromSED(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_stars = self.get_info("dict_stars")
		do_verbose = self.get_info("do_verbose")
		#
		datas_stars_SED = dict_stars["datas_spectra_stars_SED_1RSun"]
		num_stars = len(datas_stars_SED)
		#
		wave_range_UVcont = preset.wavelength_range_rates_UV
		wave_range_Lya = preset.wavelength_range_rates_Lya
		wave_range_X = preset.wavelength_range_rates_X
		struct_str_precision = preset.structure_str_precision
		#
		#Print some notes
		if do_verbose:
			print("Running _load_spectrum_stars_fromSED!")
		#

		##Raise error (for now) if multiple stars given
		if (num_stars != 1):
			raise ValueError("Err: Only single stars supported at the moment!")
		#

		##Load full SED data
		set_wave_SED_struct = [None]*num_stars
		set_spectra_SED_energy_struct = [None]*num_stars
		set_wave_UVcont_struct = [None]*num_stars
		set_spectra_UVcont_energy_struct = [None]*num_stars
		set_wave_Lya_struct = [None]*num_stars
		set_spectra_Lya_energy_struct = [None]*num_stars
		set_wave_X_struct = [None]*num_stars
		set_spectra_X_energy_struct = [None]*num_stars
		#Iterate through stars
		for ii in range(0, num_stars):
			#Print some notes
			if do_verbose:
				print("Loading SED data...")
			#
			#Extract SED data for current star [energy spectrum!]
			data_input = datas_stars_SED[ii]
			if isinstance(data_input, str): #If string, read in
				curr_data = np.genfromtxt(data_input, comments="#", dtype=float)
			else: #If array, carry over
				curr_data = np.asarray(data_input)
			#

			curr_wave_origall_struct = curr_data[:,0]
			curr_spectrum_energy_origall_struct = curr_data[:,1]
			#

			#Extract UV, X-ray data for this star, if not given separately
			#For UV data
			#Print some notes
			if do_verbose:
				print("Extracting struct. UVcont, Lya from SED...")
			#
			curr_wave_UVcont_struct_raw = curr_wave_origall_struct[
						((curr_wave_origall_struct >= wave_range_UVcont[0])
	 					& (curr_wave_origall_struct <= wave_range_UVcont[1]))]
			curr_wave_UVcont_struct = np.sort(np.unique(np.concatenate(
							(curr_wave_UVcont_struct_raw, wave_range_UVcont))))
			#Ensure wavelength extends across full desired UV, X-ray ranges
			tmpenspec_tmpUV_struct_raw = scipy_interp1d(
									y=curr_spectrum_energy_origall_struct,
			 						x=curr_wave_origall_struct
									)(curr_wave_UVcont_struct)
			#Interpolate across full wavelength range
			tmp_set = utils.split_spectrum_into_emandcont(
							range_split=wave_range_Lya,
							spectrum_orig=tmpenspec_tmpUV_struct_raw,
							wave_cont=curr_wave_UVcont_struct, do_plot=False,
							window_phys=dict_stars["emextract_window_phys"],
							window_frac=dict_stars["emextract_window_frac"],
							poly_order=dict_stars["emextract_poly_order"],
							waveranges_exclude=[wave_range_Lya],
							do_keep_exclude=False, do_same_edges=True)
			curr_wave_Lya_struct = tmp_set["wave_em"]
			curr_spectrum_energy_UVcont_struct = tmp_set["spectrum_updated"]
			curr_spectrum_energy_Lya_struct = tmp_set["spectrum_em"]
			#
			#Print some notes
			if do_verbose:
				print("Extracting struct. X-ray reaction spectrum from SED...")
			#
			curr_wave_X_struct_raw = curr_wave_origall_struct[
							((curr_wave_origall_struct >= wave_range_X[0])
		 					& (curr_wave_origall_struct <= wave_range_X[1]))]
			#Ensure wavelength extends across full desired UV, X-ray ranges
			curr_wave_X_struct = np.sort(np.unique(np.concatenate(
								(curr_wave_X_struct_raw, wave_range_X))))
			#
			curr_spectrum_energy_X_struct = scipy_interp1d(
								y=curr_spectrum_energy_origall_struct,
		 						x=curr_wave_origall_struct)(curr_wave_X_struct)
			#

			#For all data
			curr_wave_modall_struct_raw = np.sort(np.unique(np.concatenate(
							(curr_wave_origall_struct, wave_range_UVcont,
								wave_range_Lya, wave_range_X))))
			#
			curr_spectrum_energy_modall_struct_raw = scipy_interp1d(
									y=curr_spectrum_energy_origall_struct,
			 						x=curr_wave_origall_struct
									)(curr_wave_modall_struct_raw)
			#
			#Ignore values below printed precision (to ensure monotonic struct.)
			tmp_inds_keep = [0]
			tmp_inds_keep += [
				ii for ii in range(1, len(curr_wave_modall_struct_raw))
				if (struct_str_precision.format(curr_wave_modall_struct_raw[ii])
				!=struct_str_precision.format(curr_wave_modall_struct_raw[ii-1])
					)] #Keep only if unique at printed/.inp level of precision
			tmp_inds_keep = np.asarray(tmp_inds_keep)
			curr_wave_modall_struct = curr_wave_modall_struct_raw[tmp_inds_keep]
			curr_spectrum_energy_modall_struct = (
	 					curr_spectrum_energy_modall_struct_raw[tmp_inds_keep])
			#

			#Store the current data
			set_wave_SED_struct[ii] = curr_wave_modall_struct
			set_spectra_SED_energy_struct[ii] = (
											curr_spectrum_energy_modall_struct)
			set_wave_UVcont_struct[ii] = curr_wave_UVcont_struct
			set_spectra_UVcont_energy_struct[ii] = (
			 								curr_spectrum_energy_UVcont_struct)
			set_wave_Lya_struct[ii] = curr_wave_Lya_struct
			set_spectra_Lya_energy_struct[ii] = curr_spectrum_energy_Lya_struct
			set_wave_X_struct[ii] = curr_wave_X_struct
			set_spectra_X_energy_struct[ii] = curr_spectrum_energy_X_struct
		#

		##Store the data and various converted and flux products
		#For all SED data
		dict_stars["num_stars"] = num_stars
		dict_stars["wavelength_spectrum_all_struct"] = set_wave_SED_struct[0]
		dict_stars["energy_spectrum_all_1RSun_struct"] = (
											set_spectra_SED_energy_struct[0])
		dict_stars["photon_spectrum_all_1RSun_struct"] = math._conv_spectrum(
									which_conv="energy_to_photon",
									spectrum=set_spectra_SED_energy_struct[0],
									wave=set_wave_SED_struct[0])
		#
		#For structural UV, X-ray data
		dict_stars["wavelength_spectrum_UVcont_struct"
									] = set_wave_UVcont_struct[0]
		dict_stars["energy_spectrum_UVcont_1RSun_struct"] = (
											set_spectra_UVcont_energy_struct[0])
		dict_stars["photon_spectrum_UVcont_1RSun_struct"] = (
					math._conv_spectrum(which_conv="energy_to_photon",
					spectrum=set_spectra_UVcont_energy_struct[0],
					wave=set_wave_UVcont_struct[0]))
		dict_stars["wavelength_spectrum_Lya_struct"] = set_wave_Lya_struct[0]
		dict_stars["energy_spectrum_Lya_1RSun_struct"] = (
											set_spectra_Lya_energy_struct[0])
		dict_stars["photon_spectrum_Lya_1RSun_struct"] = math._conv_spectrum(
						which_conv="energy_to_photon",
						spectrum=set_spectra_Lya_energy_struct[0],
						wave=set_wave_Lya_struct[0])
		dict_stars["wavelength_spectrum_X_struct"] = set_wave_X_struct[0]
		dict_stars["energy_spectrum_X_1RSun_struct"
											] = set_spectra_X_energy_struct[0]
		dict_stars["photon_spectrum_X_1RSun_struct"] = math._conv_spectrum(
										which_conv="energy_to_photon",
										spectrum=set_spectra_X_energy_struct[0],
										wave=set_wave_X_struct[0])
		#

		##Exit the method
		return
	#


	##Method: _plot_disk_structure()
	##Purpose: Plot aspects of disk structure for verification purposes
	def _plot_disk_structure(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_grid = self.get_info("dict_grid")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		#
		filepath_plot = dict_names["filepath_plots"]
		#
		#Load structural data
		if True: #For easy minimization of this section
			arr_time_yr = dict_grid["arr_time_yr"]
			matr_x_struct = dict_grid["matr_x_structure"]
			matr_y_struct = dict_grid["matr_y_structure"]
			ind_bot = 0 #Index along y-axis of midplane
			arr_x_midplane_struct = matr_x_struct[ind_bot,:] #x-val. of midplane
			len_y_struct = matr_y_struct.shape[0]
			len_x_struct = matr_y_struct.shape[1]
			matr_x_chem = dict_grid["matr_x_chemistry"]
			matr_y_chem = dict_grid["matr_y_chemistry"]
			#

			#
			wavelength_all = dict_stars["wavelength_spectrum_all_struct"]
			wavelength_UVcont = dict_stars["wavelength_spectrum_UVcont_struct"]
			wavelength_Lya = dict_stars["wavelength_spectrum_Lya_struct"]
			wavelength_UVtot = dict_stars["wavelength_spectrum_UVtot_struct"]
			wavelength_X = dict_stars["wavelength_spectrum_X_struct"]
			energy_spectrum_all_plot = (
							dict_stars["energy_spectrum_all_1RSun_struct"]
							* (conv_Jtoerg0 / (100*100) / (1E9))) #erg/cm^2/nm
			energy_spectrum_UVcont_plot = (
							dict_stars["energy_spectrum_UVcont_1RSun_struct"]
							* (conv_Jtoerg0 / (100*100) / (1E9))) #erg/cm^2/nm
			energy_spectrum_Lya_plot = (
							dict_stars["energy_spectrum_Lya_1RSun_struct"]
							* (conv_Jtoerg0 / (100*100) / (1E9))) #erg/cm^2/nm
			energy_spectrum_X_plot = (
							dict_stars["energy_spectrum_X_1RSun_struct"]
							* (conv_Jtoerg0 / (100*100) / (1E9))) #erg/cm^2/nm
			photon_spectrum_all_plot = (
							dict_stars["photon_spectrum_all_1RSun_struct"]
							* (1.0 / (100*100) / (1E9)))
			photon_spectrum_UVcont_plot = (
							dict_stars["photon_spectrum_UVcont_1RSun_struct"]
							* (1.0 / (100*100) / (1E9)))
			photon_spectrum_Lya_plot = (
							dict_stars["photon_spectrum_Lya_1RSun_struct"]
							* (1.0 / (100*100) / (1E9)))
			photon_spectrum_X_plot = (
							dict_stars["photon_spectrum_X_1RSun_struct"]
							* (1.0 / (100*100) / (1E9)))
			#

			#
			is_extsource = (
				dict_medium["energy_spectrum_perwave_external_radiation"]
				is not None
			) #If external radiation source included
			if is_extsource:
				x_extspectrum_all_plot = (
					dict_medium["wavelength_external_radiation"]
					* 1E9
				) #nm
				energy_extspectrum_all_plot = (
					dict_medium["energy_spectrum_perwave_external_radiation"]
					* (conv_Jtoerg0 / (100*100) / (1E9))
				) #erg/s/cm^2/nm
			#

			#
			matr_tempgas_struct = dict_medium["matr_tempgas_structure"]
			matr_tempgas_chem = dict_medium["matr_tempgas_chemistry"]
			matr_tempdust_struct = dict_medium["matr_tempdust_structure"]
			matr_tempdust_chem = dict_medium["matr_tempdust_chemistry"]
			#

			arr_sigma_gas_midplane_struct_plot = (
						dict_medium["matr_sigma_gas_structure"][ind_bot,:]
						* (1E3 / (100 * 100)))
			arr_sigma_dust_midplane_struct_plot = (
			 			dict_medium["matr_sigma_dust_structure"][ind_bot,:]
						* (1E3 / (100 * 100)))
			#

			matr_scaleH_struct_plot = (dict_medium["matr_scaleH_structure"]/au0)
			matr_scaleH_chem_plot = (dict_medium["matr_scaleH_chemistry"]/au0)
			#
			matr_nH_struct_plot = (dict_medium["volnumdens_nH_structure"]
									* conv_perm3topercm3)
			matr_nHatomic_struct_plot = (
								dict_medium["volnumdens_nH_Hatomic_structure"]
									* conv_perm3topercm3)
			matr_nHmolec_struct_plot = (
								dict_medium["volnumdens_nH_Hmolec_structure"]
									* conv_perm3topercm3)
			matr_rhodust_struct_plot = (
								dict_medium["volmassdens_dustall_structure"]
									* 1E3 * conv_perm3topercm3)
			matr_rhodust_pergr_struct_plot = [(item * 1E3 * conv_perm3topercm3)
								for item in
								dict_medium["volmassdens_dustpergr_structure"]]
			matr_nH_chem_plot = (dict_medium["volnumdens_nH_chemistry"]
									* conv_perm3topercm3)
			matr_nHatomic_chem_plot = (
								dict_medium["volnumdens_nH_Hatomic_chemistry"]
									* conv_perm3topercm3)
			matr_nHmolec_chem_plot = (
								dict_medium["volnumdens_nH_Hmolec_chemistry"]
									* conv_perm3topercm3)
			matr_rhodust_chem_plot = (
								dict_medium["volmassdens_dustall_chemistry"]
									* 1E3 * conv_perm3topercm3)
			matr_rhodust_pergr_chem_plot = [(item * 1E3 * conv_perm3topercm3)
								for item in
								dict_medium["volmassdens_dustpergr_chemistry"]]
			#
			photonflux_UVcont_struct_plot = (
								dict_medium["matr_photonflux_UVcont_structure"]
								* (1 / (100 * 100)))
			photonflux_Lya_struct_plot = (
								dict_medium["matr_photonflux_Lya_structure"]
								* (1 / (100 * 100)))
			photonflux_UVtot_struct_plot = (
								dict_medium["matr_photonflux_UVtot_structure"]
								* (1 / (100 * 100)))
			energyflux_UVcont_struct_plot = (
								dict_medium["matr_energyflux_UVcont_structure"]
								* (conv_Jtoerg0 / (100 * 100)))
			energyflux_Lya_struct_plot = (
								dict_medium["matr_energyflux_Lya_structure"]
								* (conv_Jtoerg0 / (100 * 100)))
			energyflux_UVtot_struct_plot = (
								dict_medium["matr_energyflux_UVtot_structure"]
								* (conv_Jtoerg0 / (100 * 100)))
			#
			photonflux_UVcont_chem_plot = (
								dict_medium["matr_photonflux_UVcont_chemistry"]
								* (1 / (100 * 100)))
			photonflux_Lya_chem_plot = (
								dict_medium["matr_photonflux_Lya_chemistry"]
								* (1 / (100 * 100)))
			photonflux_UVtot_chem_plot = (
								dict_medium["matr_photonflux_UVtot_chemistry"]
								* (1 / (100 * 100)))
			energyflux_UVcont_chem_plot = (
								dict_medium["matr_energyflux_UVcont_chemistry"]
								* (conv_Jtoerg0 / (100 * 100)))
			energyflux_Lya_chem_plot = (
								dict_medium["matr_energyflux_Lya_chemistry"]
								* (conv_Jtoerg0 / (100 * 100)))
			energyflux_UVtot_chem_plot = (
								dict_medium["matr_energyflux_UVtot_chemistry"]
								* (conv_Jtoerg0 / (100 * 100)))
			#
			standardflux_UVcont_struct_plot = (
								dict_medium["flux_UVcont_uDraine_structure"])
			standardflux_UVcont_chem_plot=(
								dict_medium["flux_UVcont_uDraine_chemistry"])
			standardflux_Lya_struct_plot = (
									dict_medium["flux_Lya_uDraine_structure"])
			standardflux_Lya_chem_plot=dict_medium["flux_Lya_uDraine_chemistry"]
			standardflux_UVtot_struct_plot = (
									dict_medium["flux_UVtot_uDraine_structure"])
			standardflux_UVtot_chem_plot = (
									dict_medium["flux_UVtot_uDraine_chemistry"])
			#
			photonflux_X_struct_plot=(dict_medium["matr_photonflux_X_structure"]
								* (1 / (100 * 100)))
			energyflux_X_struct_plot = (
								dict_medium["matr_energyflux_X_structure"]
								* (conv_Jtoerg0 / (100 * 100)))
			photonflux_X_chem_plot = (dict_medium["matr_photonflux_X_chemistry"]
								* (1 / (100 * 100)))
			energyflux_X_chem_plot = (dict_medium["matr_energyflux_X_chemistry"]
								* (conv_Jtoerg0 / (100 * 100)))
			#
			matr_ionrate_X = dict_medium["matr_ionrate_X_primary_chemistry"]
			matr_ext_UV_struct = dict_medium["matr_extinction_UV_structure"]
			matr_ext_UV_chem = dict_medium["matr_extinction_UV_chemistry"]
			matr_ext_UV_struct_err = (
								dict_medium["matr_extinction_UV_structure_err"])
			matr_ext_UV_chem_err = (
								dict_medium["matr_extinction_UV_chemistry_err"])
		#

		#Set some global parameters
		figsize_gen = np.array([20, 10])*0.75
		lookup_figsize = {1:(10,10), 2:figsize_gen, 3:(np.array([20, 10])*0.95)}
		#
		if do_verbose:
			print("Generating verification plots of structural aspects...")
		#


		##Generate plots of structural aspects
		#-----
		##For structural vs. chemistry grid
		if True: #For easy minimization of this section
			plt.figure(figsize=figsize_gen)
			#Plot structural x,y points
			plt.scatter(matr_x_struct/au0, matr_y_struct/au0,
						color="black", linewidth=2, alpha=0.5, label="Struct.")
			#
			#Plot chemistry x,y points
			plt.scatter(matr_x_chem/au0, matr_y_chem/au0,
						color="blue", linewidth=2, alpha=0.5, label="Chem.")
			#
			#Label and save the plot
			plt.suptitle("Structural and Chemical Grid")
			plt.xlabel("X (au)")
			plt.ylabel("Y (au)")
			leg = plt.legend(loc="best") #, frameon=False)
			#leg.set_alpha(0.5)
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_grid.png"))
			plt.close()
		#


		#-----
		##For time array
		if True: #For easy minimization of this section
			plt.figure(figsize=figsize_gen)
			#Plot time array
			plt.semilogx(arr_time_yr, np.zeros(len(arr_time_yr)), color="black",
						marker="o", linewidth=3)
			#Label and save the plot
			plt.suptitle("Chemistry Output Times")
			plt.xlabel("Time (yr)")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_time.png"))
			plt.close()
		#


		#-----
		##For scale height
		if True: #For easy minimization of this section
			plt.figure(figsize=figsize_gen)
			max_scaleH = 5
			#Plot multiples of scale height
			for ii in range(1, (max_scaleH+1)):
				if (ii == max_scaleH):
					labelstruct = "Struct."
					labelchem = "Chem."
				else:
					labelstruct = None
					labelchem = None
				#
				#Plot scale height (structure)
				plt.scatter(matr_x_struct/au0, (ii*matr_scaleH_struct_plot),
						color="black", marker="o",
						alpha=0.5, label=labelstruct)
				#Plot scale height (chemistry)
				plt.scatter(matr_x_chem/au0, (ii*matr_scaleH_chem_plot),
						color="blue", marker="o",
						alpha=0.5, label=labelchem)
			#
			#Label and save the plot
			plt.suptitle("Disk Scale Heights")
			plt.xlabel("Radius (au)")
			plt.ylabel("Scale Height (au)")
			plt.legend(loc="best")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_scaleH.png"))
			plt.close()
		#


		#-----
		##For structural spectrum
		if True: #For easy minimization of this section
			#Prepare plot base and information
			arr_xs_raw = [wavelength_all, wavelength_UVcont,
									wavelength_Lya, wavelength_X] #m
			arr_xs = [(item * 1E9) for item in arr_xs_raw] #nm
			arr_phots = [photon_spectrum_all_plot,
			 					photon_spectrum_UVcont_plot,
			 					photon_spectrum_Lya_plot,
			 					photon_spectrum_X_plot] #phot/cm^2/nm
			arr_energy = [energy_spectrum_all_plot,
								energy_spectrum_UVcont_plot,
								energy_spectrum_Lya_plot,
			 					energy_spectrum_X_plot] #erg/cm^2/nm
			#
			tot_integ_energy_plot = np.trapz(y=(energy_spectrum_all_plot*1E9),
										x=wavelength_all) #erg/cm^2
			tot_integ_photon_plot = np.trapz(y=(photon_spectrum_all_plot*1E9),
										x=wavelength_all) #phot/cm^2
			#
			ncol = 3
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(numsubplots=(ncol*nrow),
									numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.1]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			#Prepare aesthetics
			list_lab = ["All", "UVcont", "Lya", "X-Ray"]
			list_col = ["black", "purple", "dodgerblue", "green"]
			list_sty = ["-"]*len(list_col)
			#
			#For photon spectrum (all)
			utils.plot_lines(arr_xs=arr_xs, arr_ys=arr_phots, plotind=0,
			 			doxlog=True, doylog=True,
						colors=list_col,
						labels=list_lab, styles=list_sty,
						dolegend=True, boxtexts=None, boxxs=[0.05],
						boxys=[0.90], boxalpha=0.5, xlabel="Wavelength (nm)",
						ylabel=r"# photons/cm^2/nm", linewidth=3, alpha=0.5,
						grid_base=grid_base, ymin=1E2, xlim=None, ylim=None,
						legendloc="best",
						title="{0:.3e} phot/cm^2".format(tot_integ_photon_plot))
			#For energy spectrum (all)
			utils.plot_lines(arr_xs=arr_xs, arr_ys=arr_energy, plotind=3,
			 			doxlog=True, doylog=True,
						colors=list_col,
						labels=list_lab, styles=list_sty,
						dolegend=True, boxtexts=None, boxxs=[0.05],
						boxys=[0.90], boxalpha=0.5, xlabel="Wavelength (nm)",
						ylabel=r"erg/s/cm^2/nm", linewidth=3, alpha=0.5,
						grid_base=grid_base, ymin=1E2, xlim=None, ylim=None,
						legendloc="best",
						title="{0:.3e} erg/cm^2".format(tot_integ_energy_plot))
			#For photon spectrum (UV)
			utils.plot_lines(arr_xs=[arr_xs[1],arr_xs[2]],
			 			arr_ys=[arr_phots[1], arr_phots[2]],
			 			plotind=1, doxlog=True, doylog=True,
						colors=list_col[1:3],
						labels=list_lab[1:3], styles=list_sty[1:3],
						dolegend=True, boxtexts=None, boxxs=[0.05],
						boxys=[0.90], boxalpha=0.5, xlabel="Wavelength (nm)",
						ylabel=r"# photons/cm^2/nm", linewidth=3, alpha=0.5,
						grid_base=grid_base, ymin=None, xlim=None, ylim=None,
						legendloc="best")
			#For energy spectrum (UV)
			utils.plot_lines(arr_xs=[arr_xs[1],arr_xs[2]],
			 			arr_ys=[arr_energy[1], arr_energy[2]],
			 			plotind=4, doxlog=True, doylog=True,
						colors=list_col[1:3],
						labels=list_lab[1:3], styles=list_sty[1:3],
						dolegend=True, boxtexts=None, boxxs=[0.05],
						boxys=[0.90], boxalpha=0.5, xlabel="Wavelength (nm)",
						ylabel=r"erg/s/cm^2/nm", linewidth=3, alpha=0.5,
						grid_base=grid_base, ymin=None, xlim=None, ylim=None,
						legendloc="best")
			#For photon spectrum (X)
			utils.plot_lines(arr_xs=[arr_xs[3]], arr_ys=[arr_phots[3]],
			 			plotind=2, doxlog=True, doylog=True,
						colors=list_col[3],
						labels=list_lab[3], styles=list_sty[3],
						dolegend=True, boxtexts=None, boxxs=[0.05],
						boxys=[0.90], boxalpha=0.5, xlabel="Wavelength (nm)",
						ylabel=r"# photons/cm^2/nm", linewidth=3, alpha=0.5,
						grid_base=grid_base, ymin=None, xlim=None, ylim=None,
						legendloc="best")
			#For energy spectrum (X)
			utils.plot_lines(arr_xs=[arr_xs[3]], arr_ys=[arr_energy[3]],
			 			plotind=5, doxlog=True, doylog=True,
						colors=list_col[3],
						labels=list_lab[3], styles=list_sty[3],
						dolegend=True, boxtexts=None, boxxs=[0.05],
						boxys=[0.90], boxalpha=0.5, xlabel="Wavelength (nm)",
						ylabel=r"erg/s/cm^2/nm", linewidth=3, alpha=0.5,
						grid_base=grid_base, ymin=None, xlim=None, ylim=None,
						legendloc="best")
			#Label and save the plot
			plt.suptitle("Stellar Structural Spectrum")
			#plt.xlabel("X (au)")
			#plt.ylabel("Y (au)")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_specstruct_star.png"))
			plt.close()
		#


		#-----
		##For external radiation spectrum, if given
		if is_extsource: #For easy minimization of this section
			#Prepare plot base and information
			arr_xs = x_extspectrum_all_plot #nm
			arr_energy = energy_extspectrum_all_plot #erg/s/cm^2/nm
			tot_integ_energy_plot = np.trapz(y=energy_extspectrum_all_plot,
										x=x_extspectrum_all_plot) #erg/s/cm^2
			#
			ncol = 1
			nrow = 1
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(numsubplots=(ncol*nrow),
									numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.1]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			utils.plot_lines(arr_xs=[arr_xs], arr_ys=[arr_energy], plotind=3,
			 			doxlog=True, doylog=True,
						colors=["black"],
						labels=["External Radiation"], styles=["-"],
						dolegend=True, boxtexts=None, boxxs=[0.05],
						boxys=[0.90], boxalpha=0.5, xlabel="Wavelength (nm)",
						ylabel=r"erg/s/cm^2/nm", linewidth=3, alpha=0.5,
						grid_base=None, ymin=1E2, xlim=None, ylim=None,
						legendloc="best",
						title=f"{tot_integ_energy_plot:.3e} erg/s/cm^2"
			)
			#Label and save the plot
			plt.suptitle("External Radiation Source Spectrum")
			#plt.xlabel("X (au)")
			#plt.ylabel("Y (au)")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_specsource_ext.png"))
			plt.close()
		#


		#-----
		##For gas and dust temperature
		if True: #For easy minimization of this section
			#Prepare plot base
			ncol = 2
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(
									numsubplots=4, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			cmap_gas = cmasher.ember
			cmap_dust = cmap_gas
			contours_gas = None #[10, 20, 50, 100, 200]
			contours_dust = None #[10, 20, 50, 100, 200]
			vmin_gas = 1 #200
			vmax_gas = 3 #200 #np.log10(200) #200
			vmin_dust = vmin_gas
			vmax_dust = vmax_gas
			#
			#For gas temperature (structure)
			utils.plot_diskscatter(matr=np.log10(matr_tempgas_struct),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap_gas, contours=contours_gas,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=r"Struct. T$_\mathrm{gas}$ (K)",
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_gas, vmax=vmax_gas,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For gas temperature (chemistry)
			utils.plot_diskscatter(matr=np.log10(matr_tempgas_chem), plotind=1,
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						cmap=cmap_gas, contours=contours_gas,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=r"Chem. T$_\mathrm{gas}$ (K)",
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_gas, vmax=vmax_gas,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For dust temperature (structure)
			utils.plot_diskscatter(matr=np.log10(matr_tempdust_struct),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=2, cmap=cmap_dust, contours=contours_dust,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=r"Struct. T$_\mathrm{dust}$ (K)",
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_dust, vmax=vmax_dust,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For dust temperature (chemistry)
			utils.plot_diskscatter(matr=np.log10(matr_tempdust_chem), plotind=3,
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						cmap=cmap_dust, contours=contours_dust,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=r"Chem. T$_\mathrm{dust}$ (K)",
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_dust, vmax=vmax_dust,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#Label and save the plot
			plt.suptitle("Gas and Dust Temperature")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_temperature.png"))
			plt.close()
		#


		#-----
		##For surface densities
		if True: #For easy minimization of this section
			plt.figure(figsize=figsize_gen)
			#Plot gas surface density array
			plt.semilogy(arr_x_midplane_struct/au0,
			 			arr_sigma_gas_midplane_struct_plot, label="Gas",
						color="tomato", marker="o", linewidth=3)
			#Plot dust surface density array
			plt.semilogy(arr_x_midplane_struct/au0,
			 			arr_sigma_dust_midplane_struct_plot, label="Dust",
						color="navy", marker="o", linewidth=3)
			#Label and save the plot
			plt.suptitle("Midplane Surface Density Profiles")
			plt.xlabel("Midplane Radius (au)")
			plt.ylabel(r"$\Sigma$ (g cm$^{-2}$)")
			plt.legend(loc="best", frameon=False)
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_surfdensity.png"))
			plt.close()
		#


		#-----
		##For gas densities
		if True: #For easy minimization of this section
			#Prepare plot base
			ncol = 3
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(
									numsubplots=6, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin_gas = 4
			vmax_gas = 16
			cmap_gas = cmasher.lavender_r
			contours_gas = [4, 5, 6, 7, 8, 9, 10, 12, 14]
			#
			#For gas H total number density (structure)
			utils.plot_diskscatter(matr=np.log10(matr_nH_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap_gas, contours=contours_gas,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(n$_\mathrm{gas}$) (1/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_gas, vmax=vmax_gas,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For gas atomic H number density (structure)
			utils.plot_diskscatter(matr=np.log10(matr_nHatomic_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=1, cmap=cmap_gas, contours=contours_gas,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(n$_\mathrm{Hatomic}$) (1/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_gas, vmax=vmax_gas,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For gas molecular H number density (structure)
			utils.plot_diskscatter(matr=np.log10(matr_nHmolec_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=2, cmap=cmap_gas, contours=contours_gas,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(n$_\mathrm{Hmolec}$) (1/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_gas, vmax=vmax_gas,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For gas H total number density (chemistry)
			utils.plot_diskscatter(matr=np.log10(matr_nH_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=3, cmap=cmap_gas, contours=contours_gas,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(n$_\mathrm{gas}$) (1/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_gas, vmax=vmax_gas,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For gas atomic H number density (chemistry)
			utils.plot_diskscatter(matr=np.log10(matr_nHatomic_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=4, cmap=cmap_gas, contours=contours_gas,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(n$_\mathrm{Hatomic}$) (1/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_gas, vmax=vmax_gas,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For gas molecular H number density (chemistry)
			utils.plot_diskscatter(matr=np.log10(matr_nHmolec_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=5, cmap=cmap_gas, contours=contours_gas,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(n$_\mathrm{Hmolec}$) (1/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_gas, vmax=vmax_gas,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#Label and save the plot
			plt.suptitle("Gas Density")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_density_gas.png"))
			plt.close()
		#


		#-----
		##For dust densities
		if True: #For easy minimization of this section
			num_pops = len(matr_rhodust_pergr_struct_plot)
			#Prepare plot base
			ncol = 1 + num_pops
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(numsubplots=(ncol*nrow),
			 						numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin_dust = -24
			vmax_dust = -14
			cmap_dust = cmasher.lavender_r
			contours_dust = [-22, -21, -20, -19, -18, -17, -16]
			#
			#For total dust mass density (structure)
			utils.plot_diskscatter(matr=np.log10(matr_rhodust_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap_dust, contours=contours_dust,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
			 			+r"log$_{10}$($\rho_\mathrm{\Sigma dust}$) (g/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_dust, vmax=vmax_dust,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For dust mass density per grain (structure)
			for ii in range(0, num_pops):
				utils.plot_diskscatter(
						matr=np.log10(matr_rhodust_pergr_struct_plot[ii]),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=(1+ii), cmap=cmap_dust, contours=contours_dust,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=("Str. (i={0}) ".format(ii)
				 			+r"log$_{10}$($\rho_\mathrm{dust_i}$) (g/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_dust, vmax=vmax_dust,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For total dust mass density (chemistry)
			utils.plot_diskscatter(matr=np.log10(matr_rhodust_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=ncol, cmap=cmap_dust, contours=contours_dust,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
			 			+r"log$_{10}$($\rho_\mathrm{\Sigma dust}$) (g/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_dust, vmax=vmax_dust,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For dust mass density per grain (chemistry)
			for ii in range(0, num_pops):
				utils.plot_diskscatter(
						matr=np.log10(matr_rhodust_pergr_chem_plot[ii]),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=(ncol+1+ii),
						cmap=cmap_dust, contours=contours_dust,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=("Ch. (i={0}) ".format(ii)
				 			+r"log$_{10}$($\rho_\mathrm{dust_i}$) (g/cm^3)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_dust, vmax=vmax_dust,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#Label and save the plot
			plt.suptitle("Dust Density")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_density_dust.png"))
			plt.close()
		#


		#-----
		##For UV photon radiation fields
		if True: #For easy minimization of this section
			#Prepare plot base
			ncol = 3
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(
									numsubplots=6, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin_photon = 3 #4
			vmax_photon = 11 #5 #16
			cmap_photon = cmasher.rainforest#_r #torch
			contours_photon = None #[6, 8, 10, 12] #[10, 20, 50, 100, 200]
			#
			#For UVcont photon flux (structure)
			utils.plot_diskscatter(matr=np.log10(photonflux_UVcont_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
					 			+"log$_{10}$(F$_\mathrm{UVcont}$) "
								+"(#phot./s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For UVcont photon flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(photonflux_UVcont_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=3, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
					 			+"log$_{10}$(F$_\mathrm{UVcont}$) "
								+"(#phot./s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#For Lya photon flux (structure)
			utils.plot_diskscatter(matr=np.log10(photonflux_Lya_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=1, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
					 			+"log$_{10}$(F$_\mathrm{Lya}$) "
								+"(#phot./s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For Lya photon flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(photonflux_Lya_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=4, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
					 			+"log$_{10}$(F$_\mathrm{Lya}$) "
								+"(#phot./s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#For UVtot photon flux (structure)
			utils.plot_diskscatter(matr=np.log10(photonflux_UVtot_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=2, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
					 			+"log$_{10}$(F$_\mathrm{UVtot}$) "
								+"(#phot./s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For UVtot photon flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(photonflux_UVtot_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=5, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
					 			+"log$_{10}$(F$_\mathrm{UVtot}$) "
								+"(#phot./s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#Label and save the plot
			plt.suptitle("UV Photon Radiation Fields")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot,
			 						"fig_radiation_UVphoton.png"))
			plt.close()
		#


		#-----
		##For UV energy radiation fields
		if True: #For easy minimization of this section
			#Prepare plot base
			ncol = 3
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(
									numsubplots=6, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin_energy = (-4-4) #-24
			vmax_energy = (-4+4) #-14
			cmap_energy = cmasher.savanna_r
			contours_energy = None #[-22, -20, -18, -16] #[10, 20, 50, 100, 200]
			#
			#For UVcont energy flux (structure)
			utils.plot_diskscatter(matr=np.log10(energyflux_UVcont_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{UVcont}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For UVcont energy flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(energyflux_UVcont_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=3, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(F$_\mathrm{UVcont}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#For Lya energy flux (structure)
			utils.plot_diskscatter(matr=np.log10(energyflux_Lya_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=1, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{Lya}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For Lya energy flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(energyflux_Lya_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=4, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(F$_\mathrm{Lya}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#For UVtot energy flux (structure)
			utils.plot_diskscatter(matr=np.log10(energyflux_UVtot_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=2, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{UVtot}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For UVtot energy flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(energyflux_UVtot_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=5, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(F$_\mathrm{UVtot}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#Label and save the plot
			plt.suptitle("UV Energy Radiation Fields")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot,
			 						"fig_radiation_UVenergy.png"))
			plt.close()
		#


		#-----
		##For UV uDraine radiation fields
		if True: #For easy minimization of this section
			#Prepare plot base
			ncol = 3
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(
									numsubplots=6, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin_stand = (-1-4) #-24
			vmax_stand = (-1+4) #-14
			cmap_stand = cmasher.swamp #_r
			contours_stand = None #[-22, -20, -18, -16] #[10, 20, 50, 100, 200]
			#
			#For UVcont standard flux (structure)
			utils.plot_diskscatter(
						matr=np.log10(standardflux_UVcont_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap_stand, contours=contours_stand,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{UVcont}$) "
								+"(Rel. to Standard)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_stand, vmax=vmax_stand,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For UVcont standard flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(standardflux_UVcont_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=3, cmap=cmap_stand, contours=contours_stand,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(F$_\mathrm{UVcont}$) "
								+"(Rel. to Standard)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_stand, vmax=vmax_stand,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#For Lya standard flux (structure)
			utils.plot_diskscatter(
						matr=np.log10(standardflux_Lya_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=1, cmap=cmap_stand, contours=contours_stand,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{Lya}$) "
								+"(Rel. to Standard)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_stand, vmax=vmax_stand,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For Lya standard flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(standardflux_Lya_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=4, cmap=cmap_stand, contours=contours_stand,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(F$_\mathrm{Lya}$) "
								+"(Rel. to Standard)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_stand, vmax=vmax_stand,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#For UVtot standard flux (structure)
			utils.plot_diskscatter(
						matr=np.log10(standardflux_UVtot_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=2, cmap=cmap_stand, contours=contours_stand,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{UVtot}$) "
								+"(Rel. to Standard)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_stand, vmax=vmax_stand,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For UVtot standard flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(standardflux_UVtot_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=5, cmap=cmap_stand, contours=contours_stand,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+"log$_{10}$(F$_\mathrm{UVtot}$) "
								+"(Rel. to Standard)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_stand, vmax=vmax_stand,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#Label and save the plot
			plt.suptitle("UV Radiation Fields / Standard (uDraine)")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot,
			 						"fig_radiation_UVstandard.png"))
			plt.close()
		#


		#-----
		##For UV extinction fields
		if True: #For easy minimization of this section
			#Prepare plot base
			ncol = 3
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(
									numsubplots=6, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin = -1
			vmax = 3 #100 #2.5
			cmap = cmasher.lavender_r
			contours = np.arange(vmin, vmax, 0.5) #[2, 4, 6]
			#Generate the structural quartile and median plots
			utils.plot_diskscatter(matr=np.log10(matr_ext_UV_struct_err[0]),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap, contours=contours,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
					 			+r"log$_{10}$(Low. Quart. Error) (mag)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin, vmax=vmax,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			utils.plot_diskscatter(matr=np.log10(matr_ext_UV_struct),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=1, cmap=cmap, contours=contours,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
					 			+r"log$_{10}$(Median Extinction) (mag)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin, vmax=vmax,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			utils.plot_diskscatter(matr=np.log10(matr_ext_UV_struct_err[1]),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=2, cmap=cmap, contours=contours,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
					 			+r"log$_{10}$(Hi. Quart. Error) (mag)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin, vmax=vmax,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#Generate the chemical quartile and median plots
			utils.plot_diskscatter(matr=np.log10(matr_ext_UV_chem_err[0]),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=3, cmap=cmap, contours=contours,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
					 			+r"log$_{10}$(Low. Quart. Error) (mag)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin, vmax=vmax,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			utils.plot_diskscatter(matr=np.log10(matr_ext_UV_chem),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=4, cmap=cmap, contours=contours,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
					 			+r"log$_{10}$(Median Extinction) (mag)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin, vmax=vmax,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			utils.plot_diskscatter(matr=np.log10(matr_ext_UV_chem_err[1]),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=5, cmap=cmap, contours=contours,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
					 			+r"log$_{10}$(Hi. Quart. Error) (mag)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin, vmax=vmax,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#
			#Label and save the plot
			plt.suptitle("UV Extinction (Rel. to Column Tops)")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_extinction_UV.png"))
			plt.close()
		#


		#-----
		##For X-ray radiation fields
		if True: #For easy minimization of this section
			#Prepare plot base
			ncol = 2
			nrow = 2
			plt.figure(figsize=lookup_figsize[ncol])
			grid_base = utils._make_plot_grid(
									numsubplots=6, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin_photon = 1 #4
			vmax_photon = 7 #5 #16
			vmin_energy = -2 #-24
			vmax_energy = 12 #-14
			vmin_stand = -5 #-24
			vmax_stand = 5 #-14
			cmap_photon = cmasher.rainforest #_r #torch
			cmap_energy = cmasher.savanna_r
			cmap_stand = cmasher.swamp_r
			contours_photon = None
			contours_energy = None
			contours_stand = None
			#
			#For X photon flux (structure)
			utils.plot_diskscatter(matr=np.log10(photonflux_X_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
					 			+"log$_{10}$(F$_\mathrm{X}$) "
								+"(#phot./s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For X energy flux (structure)
			utils.plot_diskscatter(matr=np.log10(energyflux_X_struct_plot),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=1, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Str. "
					 			+"log$_{10}$(F$_\mathrm{X}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For X photon flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(photonflux_X_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=2, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
					 			+"log$_{10}$(F$_\mathrm{X}$) "
								+"(#phot./s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For X energy flux (chemistry)
			utils.plot_diskscatter(matr=np.log10(energyflux_X_chem_plot),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=3, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Ch. "
					 			+"log$_{10}$(F$_\mathrm{X}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#Label and save the plot
			plt.suptitle("X-Ray Radiation Fields")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_radiation_X.png"))
			plt.close()
		#


		#-----
		##Plot the primary X-ray ionization rate as a check
		if True: #For easy minimization of this section
			plt.figure(figsize=(15, 15))
			#Prepare plot base
			ncol = 1
			nrow = 1
			grid_base = utils._make_plot_grid(
									numsubplots=1, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin_ionrate = -7-8
			vmax_ionrate = -7
			cmap_ionrate = cmasher.lavender_r
			contours_ionrate = [6, 8, 10, 12]
			#Generate the plot
			utils.plot_diskscatter(matr=np.log10(matr_ionrate_X),
			 			matr_x=matr_x_chem/au0, matr_y=matr_y_chem/au0,
						plotind=0, cmap=cmap_ionrate, contours=contours_ionrate,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Chem. "
					 			+r"log$_{10}$(Ionization Rate) (s^-1)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_ionrate, vmax=vmax_ionrate,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#Label and save the plot
			plt.suptitle("X-Ray Primary Ionization Rate")
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot, "fig_ionrate_X_primary.png"))
			plt.close()
		#


		#-----
		##Exit the method
		return
	#


	##Method: _set_ionrate_X()
	##Purpose: Calculate matrix of X-ray ionization rates
	def _set_ionrate_X(self):
		##Fetch model parameters
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		dict_chemistry = self.get_info("dict_chemistry")
		filepath_plots = dict_names["filepath_plots"]
		#
		wavelengths_spec_X = dict_stars["wavelength_spectrum_X_struct"]
		matr_fieldrad_X_perm = dict_medium[
									"matr_fieldrad_photon_X_perm_chemistry"]
		#
		dict_mol_X = dict_chemistry["dict_mol_X"]
		wavelengths_cross_X = math._conv_waveandfreq(which_conv="energytowave",
											energy=dict_mol_X["x_inJ"])
		spectrum_cross_X = dict_mol_X["cross_total"] #["cross_fin_allabund"]
		delta_energy = preset.chemistry_ionrateX_deltaenergy
		#
		matr_x_chem = dict_grid["matr_x_chemistry"]
		matr_y_chem = dict_grid["matr_y_chemistry"]
		#
		#Print some notes
		if do_verbose:
			print("\n> Running _set_ionrate_X()...")
		#

		##Compute primary X-ray ionization rate at each point in matrix
		#NOTE: Cell-by-cell for now, for clarity and to avoid space issues
		ylen = matr_fieldrad_X_perm.shape[0]
		xlen = matr_fieldrad_X_perm.shape[1]
		matr_ionrate_X = np.ones(shape=(ylen,xlen))*np.inf
		for yy in range(0, ylen):
			for xx in range(0, xlen):
				##Unify the radiation and cross-section wavelengths and spectra
				tmp_set_unified = math._unify_spectra_wavelengths(
						x_list=[wavelengths_spec_X, wavelengths_cross_X],
						y_list=[matr_fieldrad_X_perm[yy,xx,:],spectrum_cross_X],
						which_overlap="minimum")
				tmp_x_unified = tmp_set_unified["x"]
				tmp_y_spec_unified = tmp_set_unified["y"][0]
				tmp_y_cross_unified = tmp_set_unified["y"][1]
				#

				##Calculate the primary X-ray ionization rates
				matr_ionrate_X[yy,xx] = math._calc_photorates_base(
							mode="X_primary", x_unified=tmp_x_unified, #axis=2,
						 	dict_molinfo=None, y_spec=tmp_y_spec_unified,
							y_cross=tmp_y_cross_unified,
							delta_energy=delta_energy)
		#

		##Store the primary X-ray ionization rates
		dict_medium["matr_ionrate_X_primary_chemistry"] = matr_ionrate_X
		#
		#Print some notes
		if do_verbose:
			print("X-ray primary ionization rate has been computed.")
			print("Plotting in {0}...".format(filepath_plots))
		#

		##Exit the method
		if do_verbose:
			print("Run of _set_ionrate_X() complete!")
		#
		return
	#


	##Method: _set_photorates_UV()
	##Purpose: Calculate set of UV photochemistry rate coefficients using spectrum and update chemistry file with them
	def _set_photorates_UV(self):
		##Fetch global variables and model parameters
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_grid = self.get_info("dict_grid")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		which_chemistry = dict_names["which_chemistry"]
		filepath_chemistry = dict_names["filepath_chemistry"]
		dict_mol = dict_chemistry["dict_mol_UV"]
		#
		dist_photorates = preset.dist_photorates_Leiden
		range_UV = preset.wavelength_range_rates_UV
		fixed_energyflux_UV = preset.energyflux_toscale_photoratecalc_UV
		minval_coeff0 = preset.pnautilus_threshold_coeff0
		#
		matr_y = dict_grid["matr_y_chemistry"]
		matr_x = dict_grid["matr_x_chemistry"]
		y_len = matr_x.shape[0]
		x_len = matr_x.shape[1]
		#
		#Fetch original reactions of chemistry model
		dict_reactions_UV_orig = dict_chemistry["dict_reactions_base_UV"]
		wavelength_radiation = dict_stars["wavelength_spectrum_UVtot_chem"]
		#

		#Print some notes
		if do_verbose:
			print("\nRunning _set_photorates_UV()!")
		#

		##Compute and store UV reactions per cell
		#Initialize container for reactions per cell
		dict_reactions_old_percell = [[None for xx in range(0, x_len)]
										for yy in range(0, y_len)]
		dict_reactions_new_percell = [[None for xx in range(0, x_len)]
										for yy in range(0, y_len)]
		#

		#Iterate along y-axis
		for yy in range(0, y_len):
			#Iterate along x-axis
			for xx in range(0, x_len):
				#Print some notes
				if do_verbose:
					print("Updating UV reactions for yy,xx={0},{1} ({2},{3})..."
							.format(yy, xx, y_len, x_len))
				#

				#Fetch energy SED at this disk cell
				curr_origspec_photon_UVtot_origdist = dict_medium[
						"matr_fieldrad_photon_UVtot_perm_chemistry"][yy,xx,:]

				#Scale the SED to the photorate target distance
				curr_celldist = np.sqrt((matr_y[yy,xx]**2) + (matr_x[yy,xx]**2))
				curr_origspec_photon_radiation = math._conv_distanddist(
						curr_origspec_photon_UVtot_origdist,
						dist_old=curr_celldist, dist_new=dist_photorates)
				#

				#Scale spectrum to fixed flux for photorate calculations
				if (fixed_energyflux_UV is not None): #Scale flux in range
					#Fetch flux boundaries
					ind_min = np.argmin(
									np.abs(wavelength_radiation - range_UV[0]))
					ind_max = np.argmin(
									np.abs(wavelength_radiation - range_UV[1]))
					#Throw error if boundary values not close to target bounds
					is_close = (np.isclose(range_UV[0],
					 						wavelength_radiation[ind_min])
				 			and np.isclose(range_UV[1],
							 				wavelength_radiation[ind_max]))
					if (not is_close):
						raise ValueError("Err: Wave. bounds not near target.\n"
										+"{0} vs. [{1}, {2}]"
										.format(range_UV,
										 		wavelength_radiation[ind_min],
												wavelength_radiation[ind_max]))
					#
					#Calculate current energy flux
					origspec_energy = math._conv_spectrum(
										which_conv="photon_to_energy",
										spectrum=curr_origspec_photon_radiation,
										wave=wavelength_radiation)
					old_energyflux_UV = np.trapz(
									y=origspec_energy[ind_min:(ind_max+1)],
									x=wavelength_radiation[ind_min:(ind_max+1)])
					#
					#Scale current photon spectrum to new energy flux
					if all((origspec_energy == 0)): #Avoid division by zero
						curr_scaledspec_photon_radiation = (
											curr_origspec_photon_radiation
											* 0
											* fixed_energyflux_UV)
					else:
						curr_scaledspec_photon_radiation = (
											curr_origspec_photon_radiation
											/ old_energyflux_UV
											* fixed_energyflux_UV)
				#
				#Otherwise, copy over original photon spectrum
				else:
					curr_scaledspec_photon_radiation = (
												curr_origspec_photon_radiation)
				#

				#Calculate + Update reaction file with new photorates
				dict_updates = reacts.update_reactions(
					wavelength_radiation=wavelength_radiation,
					spectrum_photon_radiation=curr_scaledspec_photon_radiation,
					dict_reactions_orig=dict_reactions_UV_orig,
					minval_coeff0=minval_coeff0,
					do_return_byproducts=True,
					filesave_dictold=None, filesave_dictnew=None,
					do_verbose=do_verbose, dict_mol_UV=dict_mol, mode="UV")
				#

				#Store the byproducts of the updates
				dict_reactions_old_percell[yy][xx] =dict_updates["reactions_old"]
				dict_reactions_new_percell[yy][xx] =dict_updates["reactions_new"]
			#
		#

		#Store the byproducts of the updates
		dict_chemistry["dict_reactions_old_UV"] = dict_reactions_old_percell
		dict_chemistry["dict_reactions_new_UV"] = dict_reactions_new_percell

		#Exit the method
		return
	#
#


##Class: Conch_Radmc3d
##Purpose: Class for methods to prepare, manipulate, and run radmc3d model
class Conch_Radmc3d(_Base):
	##Method: __init__()
	##Purpose: Initialization of this class instance
	def __init__(self, do_load_input, do_load_output, dict_stars=None, dict_grid=None, dict_medium=None, dict_chemistry=None, dict_names=None, do_verbose=False):
		##Store model parameters
		self._storage = {}
		self._store_info(do_verbose, "do_verbose")
		self._store_info(dict_names, "dict_names")
		self._store_info(dict_stars, "dict_stars")
		self._store_info(dict_grid, "dict_grid")
		self._store_info(dict_medium, "dict_medium")
		self._store_info(dict_chemistry, "dict_chemistry")
		self._store_info(do_load_output, "do_load_output")

		##Load previous radmc-3D structure input if requested
		if do_load_input:
			#Print some notes
			if do_verbose:
				print("Loading previous radmc3D input structure...")
			#
			#Load the previous structure input
			self._load_radmc3d_input()
			#Print some notes
			if do_verbose:
				print("Load of previous radmc3D input structure complete.")
			#
		#
		##Otherwise, create a new radmc-3D structure
		else:
			#Print some notes
			if do_verbose:
				print("Creating new radmc3D structure input...")
			#
			#Create a new structure
			self._create_radmc3d()
			#Print some notes
			if do_verbose:
				print("Creation of new radmc3D structure complete.")
			#
		#


		##Load previous radmc-3D structure output if requested
		if do_load_output:
			#Print some notes
			if do_verbose:
				print("Loading previous radmc3D structure output...")
			#
			#Load the previous structure output
			self._load_radmc3d_output()
			#Print some notes
			if do_verbose:
				print("Load of previous radmc3D structure output complete.")
			#
		#
		##Otherwise, pass for now #run the new radmc-3D structure
		else:
			pass
		#
		#Exit function
		return
		#
	#


	##Method: _load_radmc3d_input()
	##Purpose: Load existing radmc3d structure input into this class instance
	def _load_radmc3d_input(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		#Print some notes
		if do_verbose:
			print("Running _load_radmc3d_input()...")
		#


		##Run methods to read in data from input files
		self._read_input_wavelength()
		self._read_input_amrgrid()
		self._read_input_stars()
		self._read_input_radmc3d()
		self._read_input_dustdensity()
		self._read_input_dustopac()
		#


		##Exit the function
		if do_verbose:
			print("_load_radmc3d_input() complete.")
		#
		return
		#
	#


	##Method: _load_radmc3d_output()
	##Purpose: Load existing radmc3d structure output into this class instance
	def _load_radmc3d_output(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		which_fields = preset.disk_radiation_fields
		workingdir = dict_names["filepath_structure"]
		#Print some notes
		if do_verbose:
			print("Running _load_radmc3d_output()...")
		#


		##Run methods to read in data from output files
		self._read_output_dusttemperature()
		for curr_field in which_fields:
			curr_subdir = os.path.join(workingdir, ("structure_"+curr_field))
			self._read_output_fieldradiation(name_field=curr_field,
			 								subdir=curr_subdir)
		#


		##Exit the function
		if do_verbose:
			print("_load_radmc3d_output() complete.")
		#
		return
		#
	#


	##Method: _create_radmc3d()
	##Purpose: Create files for given model to run radmc3d
	def _create_radmc3d(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		which_fields = preset.disk_radiation_fields
		workingdir = dict_names["filepath_structure"]
		#Print some notes
		if do_verbose:
			print("Running _create_radmc3d()...")
		#

		##Run methods to write data to input files
		self._write_input_wavelength_dust()
		for curr_field in which_fields:
			#Create structure subdirectory for this field
			curr_subdir = os.path.join(workingdir, ("structure_"+curr_field))
			comm = subprocess.call(["mkdir", curr_subdir])
			#
			#Write radiation input file for this specific spectrum
			self._write_input_wavelength_spectrum(name_field=curr_field,
													subdir=curr_subdir)
			#
		#
		self._write_input_amrgrid()
		self._write_input_stars()
		self._write_input_externalsource(subdir=workingdir)#Ext. radiation field
		#
		#Write radmc3d input for dust
		self._write_input_radmc3d(name_field="dust", subdir=workingdir)
		self._write_input_dustkappa(name_field="dust", subdir=workingdir)
		self._write_input_dustopac(name_field="dust", subdir=workingdir)
		self._write_input_dustdensity(name_field="dust", subdir=workingdir)
		#Write radmc3d input for each radiation field
		for curr_field in which_fields:
			curr_subdir = os.path.join(workingdir, ("structure_"+curr_field))
			self._write_input_radmc3d(name_field=curr_field, subdir=curr_subdir)
			self._write_input_dustkappa(name_field=curr_field,
										subdir=curr_subdir)
			self._write_input_dustopac(name_field=curr_field,subdir=curr_subdir)
			#
			self._write_input_dustdensity(name_field=curr_field,
											subdir=curr_subdir)
		#

		##Exit the function
		if do_verbose:
			print("_create_radmc3d() complete.")
		#
		return
		#
	#


	##Method: _run_radmc3d()
	##Purpose: Run radmc3d to compute dust temperatures for given model
	def run_radmc3d_dusttemperature(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		do_load_output = self.get_info("do_load_output")
		#Print some notes
		if do_verbose:
			print("Running run_radmc3d_dusttemperature()...")
		#


		##Terminate early if previous data already loaded
		if do_load_output:
			#Print some notes
			if do_verbose:
				print("Previous radmc3D dust temperatures loaded already.")
				print("Exiting method...")
			#
			return
		#


		##Run terminal commands to operate radmc3d
		workingdir = dict_names["filepath_structure"]


		##For computing dust temperature
		if do_verbose:
			print("Computing dust temperature using radmc3d...")
		#
		commandlist = [preset.command_radmc3d_main,
						preset.command_radmc3d_tempdust,
						preset.command_radmc3d_setthreads,
						preset.command_radmc3d_numthreads]
		comm = subprocess.call(commandlist, cwd=workingdir,
						timeout=preset.process_maxtime_radmc3d)
		#Throw an error if subprocess returned an error
		if comm != 0:
			raise ValueError("Err: Above error returned from radmc3d temp.!")
		#
		#Run method to read output temperature data
		self._read_output_dusttemperature()
		#

		if do_verbose:
			print("Dust temperature complete.")
		#


		##Exit the function
		if do_verbose:
			print("run_radmc3d_dusttemperature() complete.")
		#
		return
		#
	#


	##Method: _run_radmc3d()
	##Purpose: Run radmc3d to compute radiation fields for given model
	def run_radmc3d_radiationfields(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		which_fields = preset.disk_radiation_fields
		do_load_output = self.get_info("do_load_output")
		is_extsource = (
			dict_medium["energy_spectrum_perwave_external_radiation"]
			is not None
		) #If external radiation source included
		dict_isextsource = preset.radmc3d_dict_isextsource
		#Print some notes
		if do_verbose:
			print("Running run_radmc3d_radiationfields()...")
		#


		##Terminate early if previous data already loaded
		if do_load_output:
			#Print some notes
			if do_verbose:
				print("Previous radmc3D radiation fields loaded already.")
				print("Exiting method...")
			#
			return
		#


		##Run terminal commands to operate radmc3d
		workingdir = dict_names["filepath_structure"]


		##For computing each radiation field
		for curr_field in which_fields:
			#Print some notes
			if do_verbose:
				print("Working on field: {0}".format(curr_field))
			#

			#Reference structure subdirectory for this field
			curr_subdir = os.path.join(workingdir, ("structure_"+curr_field))

			#Process the dust temperature for this radiation field
			self._write_midput_dusttemperature(name_field=curr_field,
			 									subdir=curr_subdir)
			#

			#Copy over relevant files for this field
			tmp_list = [preset.radmc3d_filename_wavelength_dust,
						preset.radmc3d_filename_amrgrid,
						preset.radmc3d_filename_stars]
			#
			#Include external radiation source file if present
			if (is_extsource and dict_isextsource[curr_field]):
				tmp_list += [preset.radmc3d_filename_extsource]
			#
			#Copy over the files
			for curr_file in tmp_list:
				comm = subprocess.call(["cp",
				 						os.path.join(workingdir, curr_file),
										os.path.join(curr_subdir, curr_file)])
				if (comm != 0): #Throw an error if subprocess error
					raise ValueError("Err: Above error returned from cp!")
			#

			#Print some notes
			if do_verbose:
				print("Subdir. created and filled at: {0}".format(curr_subdir))
				print("Running radmc3d in subdir...")
			#

			#Run radmc3d within the subdirectory
			commandlist = [preset.command_radmc3d_main,
							preset.command_radmc3d_radiation,
							preset.command_radmc3d_setthreads,
							preset.command_radmc3d_numthreads]
			comm = subprocess.call(commandlist, cwd=curr_subdir,
							timeout=preset.process_maxtime_radmc3d)
			if comm != 0: #Throw an error if subprocess returned an error
				raise ValueError("Err: Above error from radmc3d rad.!")
			#

			#Print some notes
			if do_verbose:
				print("Radiation field {0} complete.".format(curr_field))
			#
		#
		if do_verbose:
			print("All radiation fields complete.")
		#


		##Run method to read in radiation data from output files
		for curr_field in which_fields:
			curr_subdir = os.path.join(workingdir, ("structure_"+curr_field))
			self._read_output_fieldradiation(name_field=curr_field,
			 								subdir=curr_subdir)
		#


		##Combine flux radiation fields as applicable
		if all([(item in which_fields) for item in ["UVcont", "Lya"]]):
			#For radiation field matrices
			#For structure photon matrices
			tmp_wave1 = dict_stars["wavelength_spectrum_UVcont_struct"]
			tmp_wave2 = dict_stars["wavelength_spectrum_Lya_struct"]
			tmp_matr1 =dict_medium["matr_fieldrad_photon_UVcont_perm_structure"]
			tmp_matr2 = dict_medium["matr_fieldrad_photon_Lya_perm_structure"]
			tmp_set = math._unify_spectra_wavelengths(
								x_list=[tmp_wave1, tmp_wave2],
								y_list=[tmp_matr1, tmp_matr2],
								which_overlap="maximum", fill_value=0, axis=2)
			wave_struct = tmp_set["x"]
			sum_struct = np.sum(tmp_set["y"], axis=0)
			dict_stars["wavelength_spectrum_UVtot_struct"] = wave_struct
			dict_medium["matr_fieldrad_photon_UVtot_perm_structure"] =sum_struct
			#For chemistry photon matrices
			tmp_wave1 = dict_stars["wavelength_spectrum_UVcont_struct"]
			tmp_wave2 = dict_stars["wavelength_spectrum_Lya_struct"]
			tmp_matr1 =dict_medium["matr_fieldrad_photon_UVcont_perm_chemistry"]
			tmp_matr2 = dict_medium["matr_fieldrad_photon_Lya_perm_chemistry"]
			tmp_set = math._unify_spectra_wavelengths(
								x_list=[tmp_wave1, tmp_wave2],
								y_list=[tmp_matr1, tmp_matr2],
								which_overlap="maximum", fill_value=0, axis=2)
			wave_chem = tmp_set["x"]
			sum_chem = np.sum(tmp_set["y"], axis=0)
			dict_stars["wavelength_spectrum_UVtot_chem"] = wave_chem
			dict_medium["matr_fieldrad_photon_UVtot_perm_chemistry"] = sum_chem
			#
			#For photon fluxes
			dict_medium["matr_photonflux_UVtot_structure"] = np.trapz(
								y=sum_struct, x=wave_struct, axis=2) #phot/s/m^2
			dict_medium["matr_photonflux_UVtot_chemistry"] = np.trapz(
								y=sum_chem, x=wave_chem, axis=2) #phot/s/m^2
			#

			#For structure energy matrices
			tmp_wave1 = dict_stars["wavelength_spectrum_UVcont_struct"]
			tmp_wave2 = dict_stars["wavelength_spectrum_Lya_struct"]
			tmp_matr1 =dict_medium["matr_fieldrad_energy_UVcont_perm_structure"]
			tmp_matr2 = dict_medium["matr_fieldrad_energy_Lya_perm_structure"]
			tmp_set = math._unify_spectra_wavelengths(
								x_list=[tmp_wave1, tmp_wave2],
								y_list=[tmp_matr1, tmp_matr2],
								which_overlap="maximum", fill_value=0, axis=2)
			wave_struct = tmp_set["x"]
			sum_struct = np.sum(tmp_set["y"], axis=0)
			dict_medium["matr_fieldrad_energy_UVtot_perm_structure"] =sum_struct
			#For chemistry energy matrices
			tmp_wave1 = dict_stars["wavelength_spectrum_UVcont_struct"]
			tmp_wave2 = dict_stars["wavelength_spectrum_Lya_struct"]
			tmp_matr1 =dict_medium["matr_fieldrad_energy_UVcont_perm_chemistry"]
			tmp_matr2 = dict_medium["matr_fieldrad_energy_Lya_perm_chemistry"]
			tmp_set = math._unify_spectra_wavelengths(
								x_list=[tmp_wave1, tmp_wave2],
								y_list=[tmp_matr1, tmp_matr2],
								which_overlap="maximum", fill_value=0, axis=2)
			wave_chem = tmp_set["x"]
			sum_chem = np.sum(tmp_set["y"], axis=0)
			dict_medium["matr_fieldrad_energy_UVtot_perm_chemistry"] = sum_chem
			#
			#For energy fluxes
			tmp_eflux_struct=np.trapz(y=sum_struct,x=wave_struct,axis=2)#J/s/m^2
			tmp_eflux_chem=np.trapz(y=sum_chem,x=wave_chem,axis=2)#J/s/m^2
			dict_medium["matr_energyflux_UVtot_structure"] = tmp_eflux_struct
			dict_medium["matr_energyflux_UVtot_chemistry"] = tmp_eflux_chem
			#
			#For fluxes in uDraine units
			dict_medium["flux_UVtot_uDraine_structure"] = (tmp_eflux_struct
												/ preset.chemistry_flux_ISRF)
			dict_medium["flux_UVtot_uDraine_chemistry"] = (tmp_eflux_chem
												/ preset.chemistry_flux_ISRF)
		#
		#Otherwise, raise error if combination not recognized
		else:
			raise ValueError("Err: UVcont and Lya not in fields:\n{0}"
								.format(which_fields))
		#


		##Exit the function
		if do_verbose:
			print("run_radmc3d_radiationfields() complete.")
		#
		return
		#
	#


	##Method: _write_input_wavelength_dust()
	##Purpose: Write wavelength_micron.inp: dust wavelengths
	def _write_input_wavelength_dust(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		wavelengths_um =(self.get_info("dict_stars")[
								"wavelength_spectrum_all_struct"]*1E6)
		#Print some notes
		if do_verbose:
			print("Running _write_input_wavelength_dust()...")
		#


		##Build string to contain text of this new file
		#Line 1: #Num. of wavelength points
		#All other lines: wavelength points
		text = "{0:d}\n".format(len(wavelengths_um))
		for val in wavelengths_um:
			text += "{0:13.6e}\n".format(val) #Indiv. wavelengths
		#
		#Print some notes
		if do_verbose:
			print("Text complete.")
		#


		##Write this string to a new file
		filepath = dict_names["filepath_structure"]
		filename = os.path.join(filepath,
								preset.radmc3d_filename_wavelength_dust)
		utils._write_text(text=text, filename=filename)
		#


		##Exit the function
		if do_verbose:
			print("_write_input_wavelength_dust complete.")
		#
		return
		#
	#


	##Method: _write_input_wavelength_spectrum()
	##Purpose: Write mcmono_wavelength_micron.inp: star spectra
	def _write_input_wavelength_spectrum(self, name_field, subdir):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		wavelengths_field = dict_stars["wavelength_spectrum_{0}_struct"
										.format(name_field)]
		#
		wavelengths_um = (wavelengths_field *1E6)
		#Print some notes
		if do_verbose:
			print("Running _write_wavelength_spectrum() for {0}..."
					.format(name_field))
		#


		##Build string to contain text of this new file
		#Line 1: #Num. of wavelength points
		#All other lines: wavelength points
		text = "{0:d}\n".format(len(wavelengths_um))
		for val in wavelengths_um:
			text += "{0:13.6e}\n".format(val) #Indiv. wavelengths
		#
		#Print some notes
		if do_verbose:
			print("Text complete.")
		#


		##Write this string to a new file
		filename = os.path.join(subdir,
								preset.radmc3d_filename_wavelength_spectrum)
		utils._write_text(text=text, filename=filename)
		#


		##Exit the function
		if do_verbose:
			print("_write_input_wavelength_spectrum complete for {0}."
					.format(name_field))
		#
		return
		#
	#


	##Method: _write_input_amrgrid()
	##Purpose: Write amr_grid.inp
	def _write_input_amrgrid(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_grid = self.get_info("dict_grid")
		edges_struct_radius_cm = dict_grid["edge_structure_radius"] * 100
		edges_struct_theta_rad = (
			(pi/2)-np.flipud(dict_grid["edge_structure_theta"])
		) #Flip into radmc-3d coordinate system where midplane is at pi/2, not 0
		edges_struct_phi_rad = dict_grid["edge_structure_phi"]
		#
		is_dimx = dict_grid["is_dimx"]
		is_dimy = dict_grid["is_dimy"]
		is_dimz = dict_grid["is_dimz"]
		len_dimx = len(dict_grid["cen_structure_radius"])
		len_dimy = len(dict_grid["cen_structure_theta"])
		len_dimz = len(dict_grid["cen_structure_phi"])
		#Print some notes
		if do_verbose:
			print("Running _write_input_amrgrid()...")
		#


		##Build string to contain text of this new file
		#Line 1: #iformat
		#Line 2: AMR grid style (0=regular grid, no AMR)
		#Line 3: Coordinate system (0 = cartesian, 100 = spherical)
		#Line 4: gridinfo
		#Line 5: Include x,y,z coordinates (0=no, 1=yes)
		#Line 6: Size of grid
		#Line series: X  coordinates (cell walls)
		#Line series: Y coordinates (cell walls)
		#Line series: Z coordinates (cell walls)
		#
		text = "1\n" #iformat
		text += "0\n" #AMR grid style
		text += "100\n" #Spherical coordinates
		text += "0\n" #Grid info
		text += "{0} {1} {2}\n".format(int(is_dimx),int(is_dimy),int(is_dimz))
		text += "{0:d} {1:d} {2:d}\n".format(len_dimx, len_dimy, len_dimz)
		#
		for val in edges_struct_radius_cm:
			text += "{0:13.6e}\n".format(val) #Indiv. x-values
		#
		for val in edges_struct_theta_rad:
			text += "{0:13.6e}\n".format(val) #Indiv. y-values
		#
		for val in edges_struct_phi_rad:
			text += "{0:13.6e}\n".format(val) #Indiv. z-values
		#
		#Print some notes
		if do_verbose:
			print("Text complete.")
		#


		##Write this string to a new file
		filepath = dict_names["filepath_structure"]
		filename = os.path.join(filepath,
								preset.radmc3d_filename_amrgrid)
		utils._write_text(text=text, filename=filename)
		#


		##Exit the function
		if do_verbose:
			print("_write_input_amrgrid complete.")
		#
		return
		#
	#


	##Method: _write_input_stars()
	##Purpose: Write stars.inp
	def _write_input_stars(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		num_stars = dict_stars["num_stars"]
		#
		rstars_cm = [dict_stars["rstar"] * 100]
		mstars_g = [dict_stars["mstar"] * 1E3]
		xstars_cm = [dict_stars["xstar"] * 100]
		ystars_cm = [dict_stars["ystar"] * 100]
		zstars_cm = [dict_stars["zstar"] * 100]
		#
		wavelengths_um = dict_stars["wavelength_spectrum_all_struct"] * 1E6
		spectrum_SI_perfreq = math._conv_spectrum(
			which_conv="perwave_to_perfreq",
			spectrum=dict_stars["energy_spectrum_all_1RSun_struct"], #[W/m^2/m]
			wave=dict_stars["wavelength_spectrum_all_struct"]) #[W/m^2/Hz]
		spectrum_cgs_1RSun = (spectrum_SI_perfreq
	 						* conv_specSItocgs0) #[W/m^2/Hz] -> [erg/s/cm^2/Hz]
		spectrum_cgs_1pc = [math._conv_distanddist(spectrum_cgs_1RSun,
										dist_old=Rsun0, dist_new=pc0)]
		len_wavelengths = len(wavelengths_um)
		#Raise an error if unhandled number of stars
		if num_stars > 1:
			raise ValueError("Err: Unhandled number of stars {0}!"
								.format(num_stars))
		#Print some notes
		if do_verbose:
			print("Running _write_input_stars()...")
		#


		##Build string to contain text of this new file
		#Line 1: #iformat (2!)
		#Line 2: Number of stars; Num. freq. for star spec. (=wavelength_micron)
		#Line 4: Rstar, mstar, xstar, ystar, zstar (all in cgs units)
		#Line series: Each wavelength point of spectrum [micron (um)]
		# - (Same wavelength as in wavelength_micron.inp)
		#Line series: Flux at each wavelength point [erg/s/cm^2/Hz, at 1pc]
		#
		text = "2\n" #iformat
		text += "{0:d} {1:d}\n\n".format(num_stars, len_wavelengths)
		for ii in range(0, num_stars):
			#Stellar characteristics
			text += (("{0:13.6e} {1:13.6e} "
						.format(rstars_cm[ii], mstars_g[ii]))
					+ ("{0:13.6e} {1:13.6e} {2:13.6e}\n\n"
						.format(xstars_cm[ii], ystars_cm[ii], zstars_cm[ii])))
			#Wavelengths
			for jj in range(0, len_wavelengths):
				text += "{0:13.6e}\n".format(wavelengths_um[jj])#Indiv wavelen.
			#Spectrum [fluxes in erg/s/cm^2/Hz]
			text += "\n"
			for jj in range(0, len_wavelengths):
				text += "{0:13.6e}\n".format(spectrum_cgs_1pc[ii][jj])#Fluxes
		#
		#Print some notes
		if do_verbose:
			print("Text complete.")
		#


		##Write this string to a new file
		filepath = dict_names["filepath_structure"]
		filename = os.path.join(filepath, preset.radmc3d_filename_stars)
		utils._write_text(text=text, filename=filename)
		#


		##Exit the function
		if do_verbose:
			print("_write_input_stars complete.")
		#
		return
		#
	#


	##Method: _write_input_radmc3d()
	##Purpose: Write radmc3d.inp
	def _write_input_radmc3d(self, name_field, subdir):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		#Print some notes
		if do_verbose:
			print("Running _write_input_radmc3d()...")
		#


		##Build string to contain text of this new file
		#
		text = "nphot_therm = {0:d}\n".format(preset.radmc3d_value_nphottherm)
		text += "nphot_scat = {0:d}\n".format(preset.radmc3d_value_nphotscatt)
		text += "nphot_mono = {0:d}\n".format(preset.radmc3d_value_nphotmono)
		text += "incl_dust = {0:d}\n".format(1)
		text += ("scattering_mode_max = {0:d}\n"
				.format(dict_medium["mode_scattering_{0}".format(name_field)]))
		text += "iranfreqmode = 1\n"
		text += "istar_sphere = {0:d}\n".format(int(dict_stars["is_sphere"]))
		if dict_names["randomseed"] is not None:
			text += ("iseed = -{0:d}\n"
					.format(int(np.abs(dict_names["randomseed"]))))
		#
		#Print some notes
		if do_verbose:
			print("Text complete.")
		#


		##Write this string to a new file
		filepath = os.path.join(dict_names["filepath_structure"], subdir)
		filename = os.path.join(filepath,
								preset.radmc3d_filename_radmc3dinp)
		utils._write_text(text=text, filename=filename)
		#


		##Exit the function
		if do_verbose:
			print("_write_input_radmc3d complete.")
		#
		return
		#
	#


	##Method: _write_input_dustkappa()
	##Purpose: Write dustkappa_***.inp
	def _write_input_dustkappa(self, name_field, subdir):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		filepath_plots = dict_names["filepath_plots"]
		templatedir = os.path.join(dict_names["template_model_base"],
									"structure")
		struct_str_precision = preset.structure_str_precision
		#
		vol_gr_cgs = (4*pi/3)*(dict_medium["radius_grain_cgs"]**3)
		mass_gr_cgs = (dict_medium["densmass_grain_cgs"] * vol_gr_cgs)
		mass_gr = (mass_gr_cgs * 1E-3)
		dust_scaler = mH0 / mass_gr
		#
		#Print some notes
		if do_verbose:
			print("Running _write_input_dustkappa() for field {0}..."
					.format(name_field))
		#


		#Throw error if field request not recognized
		if (name_field.lower() not in ["dust", "uvcont", "lya", "x"]):
			raise ValueError("Err: Field {0} unrecognized!".format(name_field))
		#


		##Generate dustkappa file based on given field
		#Copy over any pre-existing dustkappa files; generate the rest later
		#Print some notes
		if do_verbose:
			print("Copying base kappa files for field {0}..."
					.format(name_field))
		#
		#Reference structure subdirectory for this field
		fileexts_dustopac = preset.radmc3d_dict_whichdustopac[name_field]
		#
		#Copy over relevant files for this field
		for curr_file in fileexts_dustopac:
			file_tocopy = "{0}_{1}.inp".format(
									preset.radmc3d_filename_dustkapparoot,
									curr_file)
			#
			comm = subprocess.call(["cp", os.path.join(templatedir,
			 											file_tocopy),
									os.path.join(subdir, file_tocopy)])
			if (comm != 0): #Throw an error if subprocess error
				pass #Failed cp is ok here; just means not pre-existing file
			else:
				#Print some notes
				if do_verbose:
					print("File {0} copied to: {1}."
							.format(file_tocopy, subdir))
			#
		#


		##Generate any new kappa profiles as needed
		#For Lya, generate additional Voigt scattering profile
		if (name_field.lower() in ["lya"]):
			list_exts = [""]
			#Generate generic wavelength array
			num_points_raw = preset.profile_crossLya_numpoints_persect
			waverange = preset.profile_crossLya_waverange
			wavecen = preset.waveloc_Lya
			profbuff = 5E-11
			x_cross_p1 = np.linspace(waverange[0], waverange[1],
										num_points_raw, endpoint=True)
			x_cross_p2 = np.linspace(wavecen-profbuff, wavecen+profbuff,
										num_points_raw, endpoint=True)
			x_cross_raw = np.sort(np.unique(np.concatenate(
										[[wavecen], x_cross_p1, x_cross_p2])))
			#
			#Ignore values below printed precision (to ensure monotonic struct.)
			ind_wavecen = x_cross_raw.tolist().index(wavecen)
			tmp_inds_keep = [0, ind_wavecen] #Ensure central point is kept
			tmp_inds_keep += [ii for ii in range(1, len(x_cross_raw))
					if (
						(struct_str_precision.format(x_cross_raw[ii])
							!= struct_str_precision.format(x_cross_raw[ii-1]))
						& (struct_str_precision.format(x_cross_raw[ii])
							!= (struct_str_precision.format(wavecen))))
					] #Keep only if unique at printed/.inp level of precision
			tmp_inds_keep = np.sort(tmp_inds_keep)
			x_cross = x_cross_raw[tmp_inds_keep]
			#
			num_points = len(x_cross)
			#

			#Generate absorption and scattering profiles
			kappas_scatt_perH = [math.make_profile_kappa_scattLya(
									wave_arr=x_cross,
									do_plot=True, filepath_plot=filepath_plots)]
			kappas_abs_perH = [(np.ones(num_points)
					 			* preset.radmc3d_threshold_minval
								)] #Abs. in [m^2]; very small number (~0)
			#
			#Convert per-H cross-sections to per-dust
			kappas_scatt_perfin = [(kappas_scatt_perH[0])] #per-H
			kappas_abs_perfin = [(kappas_abs_perH[0])] #per-H still
		#
		#For X-rays, generate profiles
		elif (name_field.lower() in ["x"]):
			list_exts = ["_gas", "_dust"]
			#Load X-ray cross-section information
			set_cross_X = dict_chemistry["dict_mol_X"]
			x_cross_rev = math._conv_waveandfreq(which_conv="energytowave",
			 								energy=set_cross_X["x_inJ"]
											) #Energy runs large to small values
			x_cross = x_cross_rev[::-1] #Switch to be small to large values
			sigmas_abs_perH = [set_cross_X["cross_gas"][::-1],
			 					set_cross_X["cross_dust"][::-1]] #[m^2/per H]
			kappas_abs_perH = [(item/mH0) for item in sigmas_abs_perH]
			#Above: -> [m^2/per kg of H]
			kappas_scatt_perH = [math.make_profile_kappa_scattX(x_arr=x_cross)
								]*len(kappas_abs_perH)
			sigmas_scatt_perH = [(item * mH0) for item in kappas_scatt_perH]
			#
			#Convert per-H to per-dust as needed
			kappas_abs_perfin = [kappas_abs_perH[0], #Keep per-H for gas
									(kappas_abs_perH[1] * dust_scaler)]#Per-dust
			kappas_scatt_perfin = [kappas_scatt_perH[0],
									(kappas_scatt_perH[1] * dust_scaler)]
		#


		##Save profiles to dustkappa files, if applicable
		if (name_field.lower() in ["lya", "x"]):
			for ii in range(0, len(list_exts)): #Make each dustkappa file
				#Prepare inputs and units for dustkappa files
				arrays = [(x_cross*1E6), #[m] -> [um]
						(kappas_abs_perfin[ii] * 100*100/1E3),
						(kappas_scatt_perfin[ii] * 100*100/1E3)
						] #[m^2/kg]->[cm^2/g]
				headers = ["File: dustkappa_cross{0}{1}.inp."
								.format(name_field, list_exts[ii]),
			 				"Purpose: radmc3d.", "Columns:",
							("Wavelength [um]\tKappa_abs [cm^2/(g of dust)]\t"
								+"Kappa_scatt [cm^2/(g of dust)]")]
				toplines = ["2", str(len(x_cross)), ""]
				filename = ("dustkappa_cross{0}{1}.inp"
							.format(name_field, list_exts[ii]))
				#

				#Generate and save profiles to dustkappa files
				utils.make_file_array(arrays=arrays, headers=headers,
								toplines=toplines,
								filepath=subdir, filename=filename,
								mode_format=preset.radmc3d_numformat_dustkappa)
		#


		##Plot profiles as a check, if applicable
		#For Lya
		if (name_field.lower() in ["lya"]):
			#Plot abs. if above 0
			if (kappas_abs_perfin[0].max() > preset.radmc3d_threshold_minval):
				plt.plot(x_cross*1E9, kappas_abs_perfin[0]*100*100/1E3,
				 			marker="o", label="Absorption", alpha=0.5,
							color="tomato", linewidth=3)
			else:
				plt.plot([], [], marker="o",
							label="(Absorption=0)", alpha=0.5,
							color="tomato", linewidth=3)
			#Plot scatter, if above 0
			if (kappas_scatt_perfin[0].max() >preset.radmc3d_threshold_minval):
				plt.plot(x_cross*1E9, kappas_scatt_perfin[0]*100*100/1E3,
				 			marker="o", label="Scattering", alpha=0.5,
							color="dodgerblue", linewidth=3)
			else:
				plt.plot([], [],
							label="(Scattering=0)", alpha=0.5, marker="o",
							color="dodgerblue", linewidth=3)
			#
			plt.xscale("log")
			plt.yscale("log")
			plt.xlabel("Wavelength (nm)")
			plt.ylabel("Kappa (cm^2/(g of H))")
			plt.title("{0} Kappas".format(name_field))
			plt.legend(loc="best")
			plt.savefig(os.path.join(filepath_plots,
			 					"fig_kappa_{0}.png".format(name_field)))
			plt.close()
		#
		#For X-ray
		elif (name_field.lower() in ["x"]):
			##For physical units
			fig = plt.figure(figsize=(20, 10))
			ax0 = fig.add_subplot(2, 1, 1)
			list_lwidths = [5, 1.5]
			list_markers = ["o", "*"]
			#Iterate through kappa components
			for ii in range(0, len(list_exts)):
				#Plot abs. if above 0
				if (kappas_abs_perfin[ii].max()
				 							> preset.radmc3d_threshold_minval):
					ax0.plot(x_cross*1E9,
					 		kappas_abs_perfin[ii]*100*100/1E3,
							label=("Absorption"+list_exts[ii]), alpha=0.5,
							color="tomato", marker=list_markers[ii],
							linewidth=list_lwidths[ii])
				else:
					ax0.plot([], [], marker=list_markers[ii],
							label="(Absorption=0)", alpha=0.5,
							color="tomato",
							linewidth=list_lwidths[ii])
				#
				#Plot scatter, if above 0
				if (kappas_scatt_perfin[ii].max()
											> preset.radmc3d_threshold_minval):
					ax0.plot(x_cross*1E9, kappas_scatt_perfin[ii]*100*100/1E3,
				 			label=("Scattering"+list_exts[ii]),
							alpha=0.5, color="dodgerblue",
							marker=list_markers[ii], linewidth=list_lwidths[ii])
				else:
					ax0.plot([], [],
							label="(Scattering=0)", alpha=0.5,
							marker=list_markers[ii],
							color="dodgerblue", linewidth=list_lwidths[ii])
				#
			#
			ax0.set_xscale("log")
			ax0.set_yscale("log")
			ax0.set_ylim(ymin=1E-10)
			ax0.set_xlabel("Wavelength (nm)")
			ax0.set_ylabel("Kappa (cm^2/(g of gas or dust))")
			ax0.set_title("{0} Kappas".format(name_field))
			ax0.legend(loc="best")
			#
			#For physical|energy units (comparison to Bethell+2011b
			ax0 = fig.add_subplot(2, 1, 2)
			tmpx_keV = math._conv_waveandfreq(which_conv="wavetoenergy",
		 									wave=x_cross)/conv_eVtoJ0*1E-3 #keV
			scaler = (1E24 * (100*100) * (tmpx_keV**3))
			#Iterate through kappa components
			for ii in range(0, len(list_exts)):
				#Plot abs. if above 0
				if (sigmas_abs_perH[ii].max() >preset.radmc3d_threshold_minval):
					ax0.plot(tmpx_keV, sigmas_abs_perH[ii]*scaler,
					 			label="Absorption"+list_exts[ii],
								alpha=0.5, color="tomato",
								marker=list_markers[ii],
								linewidth=list_lwidths[ii])
				else:
					ax0.plot([], [], marker=list_markers[ii],
								label="(Absorption=0)", alpha=0.5,
								color="tomato",
								linewidth=list_lwidths[ii])
				#Plot scatter, if above 0
				if (sigmas_scatt_perH[ii].max()
											> preset.radmc3d_threshold_minval):
					ax0.plot(tmpx_keV, sigmas_scatt_perH[ii]*scaler,
					 			label="Scattering"+list_exts[ii],
							 	alpha=0.5, color="dodgerblue",
								marker=list_markers[ii],
								linewidth=list_lwidths[ii])
				else:
					ax0.plot([], [],
								label="(Scattering=0)", alpha=0.5,
								color="dodgerblue", marker=list_markers[ii],
								linewidth=list_lwidths[ii])
			#
			ax0.set_xscale("log")
			ax0.set_yscale("log")
			ax0.set_ylim([1E-1, 1E3])
			ax0.set_xlabel("Energy (keV)")
			ax0.set_ylabel("Cross-Section * E^3 (1E-24 cm^2/H keV^3)")
			ax0.set_title("{0} Cross-Sections".format(name_field))
			ax0.legend(loc="best")
			#
			plt.savefig(os.path.join(filepath_plots,
			 					"fig_kappa_{0}.png".format(name_field)))
			plt.close()
		#


		##Exit the method
		if do_verbose:
			print("Run of _write_input_dustkappa() for field {0} complete!"
					.format(name_field))
		#
		return
	#


	##Method: _write_input_dustopac()
	##Purpose: Write dustopac.inp
	def _write_input_dustopac(self, name_field, subdir):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_medium = self.get_info("dict_medium")
		ext_dust = preset.radmc3d_dict_whichdustopac[name_field]
		num_distrs = dict_medium["num_distrs_dustdens_perfield"][name_field]
		comment_dust = "Extension of name of dustkappa_***.inp file\n"
		#Print some notes
		if do_verbose:
			print("Running _write_input_dustopac() for field {0}..."
					.format(name_field))
		#


		##Build string to contain text of this new file
		#
		text = "2"+(" "*15)+"Format number of this file\n"
		text += "{0}".format(num_distrs)+(" "*15)+"Nr of dust species\n"
		text += ("========================================================"
					+"====================\n")
		#if ext_dust is not None:
		#Write information for each dust species
		for ii in range(0, len(ext_dust)):
			text += "1"+(" "*15)+"Way in which this dust species is read\n"
			text += "0"+(" "*15)+"0=Thermal grain\n"
			text += "{0}".format(ext_dust[ii]).ljust(16) + comment_dust
			text += ("--------------------------------------------------------"
						+"--------------------\n")
		#
		#Print some notes
		if do_verbose:
			print("Text complete.")
		#


		##Write this string to a new file
		filepath = os.path.join(dict_names["filepath_structure"], subdir)
		filename = os.path.join(filepath,
								preset.radmc3d_filename_dustopac)
		utils._write_text(text=text, filename=filename)
		#


		##Exit the function
		if do_verbose:
			print("_write_input_dustopac complete for field {0}."
					.format(name_field))
		#
		return
		#
	#


	##Method: _write_input_externalsource()
	##Purpose: Write external_source.inp
	def _write_input_externalsource(self, subdir):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_medium = self.get_info("dict_medium")
		x_for_extrapolation_um = (
			self.get_info("dict_stars")["wavelength_spectrum_all_struct"]
			* 1E6 #-> [um]
		)
		is_extsource = (
			dict_medium["energy_spectrum_perwave_external_radiation"]
			is not None
		)
		extrap_fill_value = 0 #Values extrapolated beyond ext. spectrum = 0
		#Print some notes
		if do_verbose:
			print("Running _write_input_externalsource()...")
		#

		##Exit method if no external source given
		if (not is_extsource):
			if do_verbose:
				print("No external radiation source given. Exiting...")
			return
		#

		##Extract and convert external source spectrum
		wavelengths_um_raw = dict_medium["wavelength_external_radiation"] * 1E6
		extspectrum_SI_perwave = (
		 	dict_medium["energy_spectrum_perwave_external_radiation"]
		) #[W/m^2/m]
		extspectrum_SI_perfreq = math._conv_spectrum(
			which_conv="perwave_to_perfreq",
			spectrum=extspectrum_SI_perwave, #[W/m^2/m]
			wave=dict_medium["wavelength_external_radiation"]
		) #-> [W/m^2/Hz]
		extintensity_cgs_perfreq_perster_raw = (
			extspectrum_SI_perfreq #[W/m^2/Hz]
			/ (4*pi) #-> [W/m^2/Hz/ster]
			* conv_specSItocgs0 #-> [erg/s/cm^2/Hz/ster]
		)

		##Extrapolate external source over same wavelengths as star spectrum
		extintensity_cgs_perfreq_perster = scipy_interp1d(
			y=extintensity_cgs_perfreq_perster_raw,
			x=wavelengths_um_raw, bounds_error=False,
			fill_value=extrap_fill_value
		)(x_for_extrapolation_um)
		num_points = len(extintensity_cgs_perfreq_perster)

		##Build string to contain text of this new file
		text = "2\n"
		text += f"{num_points}\n"
		#Write information for wavelength points
		for ii in range(0, num_points):
			text += f"{x_for_extrapolation_um[ii]}\n"
		#
		#Write information for intensity points
		for ii in range(0, num_points):
			text += f"{extintensity_cgs_perfreq_perster[ii]}\n"
		#
		#Print some notes
		if do_verbose:
			print("Text complete.")
		#

		##Write this string to a new file
		filepath = os.path.join(dict_names["filepath_structure"], subdir)
		filename = os.path.join(filepath,
								preset.radmc3d_filename_extsource)
		utils._write_text(text=text, filename=filename)
		#

		##Exit the function
		if do_verbose:
			print("_write_input_externalsource complete.")
		#
		return
		#
	#


	##Method: _write_input_dustdensity()
	##Purpose: Write dust_density.inp
	def _write_input_dustdensity(self, name_field, subdir):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_medium = self.get_info("dict_medium")
		#
		which_distrs = preset.radmc3d_dict_whichdustdens[name_field]
		#
		num_points = self.get_info("dict_grid")["num_points_structure"]
		#
		fracs_dustspecies = dict_medium["fracs_dustspecies"]
		num_speciesdust = len(fracs_dustspecies) #Num. dust species
		num_distrs = dict_medium["num_distrs_dustdens_perfield"][name_field]
		#
		#For total gas density distribution
		matr_densgas_cgs = np.flipud((dict_medium["volnumdens_nH_structure"]
							* mH0 * conv_kgtog0
							* conv_perm3topercm3)) #-> radmc3d: midplane at -1
		matr_densgas_cgsflat = matr_densgas_cgs.ravel() #->1D
		#
		#For dust density distribution
		matr_densdust_cgs = np.flipud(
							(dict_medium["volmassdens_dustall_structure"]
							* conv_kgtog0
							* conv_perm3topercm3)) #-> radmc3d: midplane at -1
		matr_densdust_cgsflat_raw = matr_densdust_cgs.ravel() #->1D
		matr_densdust_cgsflat = [
						(matr_densdust_cgsflat_raw * fracs_dustspecies[dd])
						for dd in range(0, num_speciesdust)]
		#
		#For atomic H density distribution, if requested
		if ("Hatomic" in which_distrs):
			matr_densHatomic_cgs = np.flipud(
							(dict_medium["volnumdens_nH_Hatomic_structure"]
							* mH0 * conv_kgtog0
							* conv_perm3topercm3) #[x,y,z]
							) #-> radmc3d: midplane at -1
			matr_densHatomic_cgsflat = matr_densHatomic_cgs.ravel() #->1D
		#
		#Print some notes
		if do_verbose:
			print("\n> Running _write_input_dustdensity() for field {0}..."
					.format(name_field))
		#


		##Build string to contain text of this new file
		#Line 1: #iformat (1!)
		#Line 2: Number of cells
		#Line 3: Number of dust species
		#Line series: Density values per cell
		#
		#For header
		text = "1\n" #iformat
		text += "{0:d}\n".format(num_points)
		text += "{0:d}\n".format(num_distrs)
		#

		#Iterate through distributions
		i_track = 0
		for curr_distr in which_distrs:
			#For dust species
			if (curr_distr in ["dust"]):
				#Print some notes
				if do_verbose:
					print("Including {0} dust distribution(s)..."
							.format(num_speciesdust))
				#
				#Iterate through dust species
				for dd in range(0, num_speciesdust):
					for ii in range(0, num_points):
						#Dust density per point, per dust species
						text += ("{0:13.6e}\n"
									.format(matr_densdust_cgsflat[dd][ii]))
					#
					#Tack on extra space if more dust species to consider
					if (dd < (num_speciesdust - 1)):
						text += "\n"
					#
					#Increment count of overall distributions
					i_track += 1
				#
			#
			#For atomic H distribution
			elif (curr_distr in ["Hatomic"]):
				#Print some notes
				if do_verbose:
					print("Including Hatomic distribution...")
				#
				for ii in range(0, num_points):
					#Dust density per point
					text += ("{0:13.6e}\n"
								.format(matr_densHatomic_cgsflat[ii]))
				#
				#Split between distributions and increment distribution count
				if (i_track < (num_distrs - 1)):
					text += "\n"
				#
				i_track += 1
			#
			#For total gas distribution
			elif (curr_distr in ["gas"]):
				#Print some notes
				if do_verbose:
					print("Including gas distribution...")
				#
				for ii in range(0, num_points):
					#Dust density per point
					text += ("{0:13.6e}\n"
								.format(matr_densgas_cgsflat[ii]))
				#
				#Split between distributions and increment distribution count
				if (i_track < (num_distrs - 1)):
					text += "\n"
				#
				i_track += 1
			#
			#Otherwise, throw error if distribution not recognized
			else:
				raise ValueError("Err: {0} not recognized!".format(curr_distr))
			#
		#
		#Print some notes
		if do_verbose:
			print("Text complete for field {0}.".format(name_field))
		#


		##Write this string to a new file
		filename = os.path.join(subdir, preset.radmc3d_filename_dustdensity)
		utils._write_text(text=text, filename=filename)
		#


		##Exit the function
		if do_verbose:
			print("_write_input_dustdensity complete for field {0}."
					.format(name_field))
		#
		return
		#
	#


	##Method: _write_midput_dustdensity()
	##Purpose: Write dust_temperature.dat for radiation propagation
	def _write_midput_dusttemperature(self, name_field, subdir):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_medium = self.get_info("dict_medium")
		#
		which_distrs = preset.radmc3d_dict_whichdustdens[name_field]
		num_distrs = dict_medium["num_distrs_dustdens_perfield"][name_field]
		num_points = self.get_info("dict_grid")["num_points_structure"]
		#
		matr_tempdust_raw = np.flipud(dict_medium["matr_tempdust_structure"])
		matr_tempdust_flat = matr_tempdust_raw.ravel() #-> radmc3d: midplane at -1 #->1D
		#
		#Print some notes
		if do_verbose:
			print("\n> Running _write_midput_dusttemperature() for field {0}..."
					.format(name_field))
		#


		##Build string to contain text of this new file
		#Line 1: #iformat (1!)
		#Line 2: Number of cells
		#Line 3: Number of dust species
		#Line series: Temperature values per cell
		#
		#For header
		text = "1\n" #iformat
		text += "{0:d}\n".format(num_points)
		text += "{0:d}\n".format(num_distrs)
		#

		#Iterate through distributions
		i_track = 0
		for curr_distr in which_distrs:
			#For dust distribution
			if (curr_distr in ["dust", "gas"]):
				#Print some notes
				if do_verbose:
					print("Including {0} temperatures...".format(curr_distr))
				#
				for ii in range(0, num_points):
					#Dust temperature per point
					text += ("{0:17.14f}\n"
								.format(matr_tempdust_flat[ii]))
				#
				#Split between distributions and increment distribution count
				if (i_track < (num_distrs - 1)):
					text += "\n"
				#
				i_track += 1
			#
			#For atomic H distribution
			elif (curr_distr in ["Hatomic"]):
				#Print some notes
				if do_verbose:
					print("Including {0} temperatures...".format(curr_distr))
				#
				for ii in range(0, num_points):
					#H atomic temperature per point, per dust species
					text += ("{0:17.14f}\n"
								.format(matr_tempdust_flat[ii])
								)
					#Note: Bethell+2011 assumed coupled gas-dust temp
				#
				#Split between distributions and increment distribution count
				if (i_track < (num_distrs - 1)):
					text += "\n"
				#
				i_track += 1
			#
			#Otherwise, throw error if distribution not recognized
			else:
				raise ValueError("Err: {0} not recognized!".format(curr_distr))
			#
		#
		#Print some notes
		if do_verbose:
			print("Text complete for field {0}.".format(name_field))
		#


		##Write this string to a new file
		filename = os.path.join(subdir, preset.radmc3d_filename_tempdust)
		utils._write_text(text=text, filename=filename)
		#


		##Exit the function
		if do_verbose:
			print("_write_midput_dusttemperature complete for field {0}."
					.format(name_field))
		#
		return
		#
	#


	##Method: _read_output_dusttemperature()
	##Purpose: Read in dust_temperature.dat
	def _read_output_dusttemperature(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_medium = self.get_info("dict_medium")
		#
		num_points_struct = self.get_info("dict_grid")["num_points_structure"]
		new_shape_struct = self.get_info("dict_grid")["shape_structure"]
		#
		matr_x_chem = dict_grid["matr_x_chemistry"]
		matr_y_chem = dict_grid["matr_y_chemistry"]
		points_yx_struct = dict_grid["points_yx_structure"]
		matr_y_diskinds_chem = dict_grid["matr_y_diskinds_chemistry"]
		#
		#Print some notes
		if do_verbose:
			print("Running _read_output_dusttemperature()...")
		#
		if "matr_tempdust_structure" in dict_medium:
			raise ValueError("Err: Dust temperature data already exists.")
		#


		##Read from the existing file
		filepath = dict_names["filepath_structure"]
		filename = os.path.join(filepath, preset.radmc3d_filename_tempdust)
		with open(filename, 'r') as openfile:
			text_orig = openfile.readlines()
			header = int(text_orig[1])
			text = text_orig[3:len(text_orig)]
		#
		if header != num_points_struct:
			raise ValueError("Err: File length {0} != number of points {1}!"
								.format(header, num_points_struct))
		#


		##Process and store the read-in text
		matr_tempdust_struct = np.flipud(
									np.reshape(np.asarray(text).astype(float),
											new_shape_struct)) #-> argonaut conv.
		dict_medium["matr_tempdust_structure"] = matr_tempdust_struct
		matr_tempdust_chem = math._interpolate_pointstogrid(
							old_matr_values=matr_tempdust_struct,
							old_points_yx=points_yx_struct,
							new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
							inds_valid=matr_y_diskinds_chem)
		dict_medium["matr_tempdust_chemistry"] = matr_tempdust_chem
		#


		##Exit the function
		if do_verbose:
			print("_read_output_dusttemperature complete.")
		#
		return
		#
	#


	##Method: _read_output_fieldradiation()
	##Purpose: Read in mean_intensity.out
	def _read_output_fieldradiation(self, name_field, subdir, do_plot=False):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_grid = self.get_info("dict_grid")
		dict_stars = self.get_info("dict_stars")
		dict_medium = self.get_info("dict_medium")
		#
		num_points_struct = dict_grid["num_points_structure"]
		shape_struct = dict_grid["shape_structure"]
		#
		matr_x_struct = dict_grid["matr_x_structure"]
		matr_y_struct = dict_grid["matr_y_structure"]
		matr_x_chem = dict_grid["matr_x_chemistry"]
		matr_y_chem = dict_grid["matr_y_chemistry"]
		points_yx_struct = dict_grid["points_yx_structure"]
		matr_y_diskinds_chem = dict_grid["matr_y_diskinds_chemistry"]
		#
		wavelengths_field = dict_stars["wavelength_spectrum_{0}_struct"
										.format(name_field)]
		len_wavelength = len(wavelengths_field)
		new_shape_mid = np.concatenate(([len_wavelength], shape_struct))
		#
		#Print some notes
		if do_verbose:
			print("Running _read_output_fieldradiation() for field {0}..."
					.format(name_field))
		#
		if "matr_fieldrad_cgs_structure" in dict_medium:
			raise ValueError("Err: Radiation field data already exists.")
		#


		##Read from the existing file
		filename = os.path.join(subdir, preset.radmc3d_filename_field_radiation)
		with open(filename, 'r') as openfile:
			text_orig = openfile.readlines()
			header1 = int(text_orig[1]) #Number of cells
			header2 = int(text_orig[2]) #Number of frequencies
			text = text_orig[4:len(text_orig)]
		#
		if (header1 != num_points_struct) or (header2 != len_wavelength):
			raise ValueError("Err: File length={0}, number of points={1}; "
								.format(header1, num_points_struct)
								+"File # waves={0}, number of waves={1}!"
								.format(header2, len_wavelength))
		#


		##Process and store the read-in text
		matr_fieldrad_energy_perHz_raw = np.moveaxis(
							(np.reshape(np.asarray(text).astype(float),
										new_shape_mid)
									* (4*pi*(100.0*100.0)/conv_Jtoerg0)
							), 0, -1
							)#[erg/(s*cm^2*Hz*ster)->[W/(m^2*Hz)]; y,x,wave
		#
		matr_fieldrad_energy_perHz = np.flipud(matr_fieldrad_energy_perHz_raw)
		#
		#Convert to per wavelength: F_lambda = F_nu * (nu^2/c): [W/(m^2 * m)]
		matr_fieldrad_energy_perm = math._conv_spectrum(
								which_conv="perfreq_to_perwave",
								spectrum=matr_fieldrad_energy_perHz,
								wave=wavelengths_field[np.newaxis,np.newaxis,:])
		#Convert to photon spectrum: #[#phot/(s * m^2 * m)]
		matr_fieldrad_photon_perm = math._conv_spectrum(
								which_conv="energy_to_photon",
								spectrum=matr_fieldrad_energy_perm,
								wave=wavelengths_field[np.newaxis,np.newaxis,:])
		#Convert to photon spectrum per energy: #[#phot/(s * m^2 * J)]
		matr_fieldrad_photon_perJ = math._conv_spectrum(
								which_conv="perwave_to_perenergy",
								spectrum=matr_fieldrad_photon_perm,
								wave=wavelengths_field[np.newaxis,np.newaxis,:])
		#


		##Comput radiation products
		#For UV
		energyflux = np.trapz(y=matr_fieldrad_energy_perm,
		 						x=wavelengths_field, axis=2) #J/s/m^2
		photonflux = np.trapz(y=matr_fieldrad_photon_perm,
		 						x=wavelengths_field, axis=2)#phot/s/m^2
		#Prepare flux in standard units
		flux_uDraine = (energyflux / 1.0 / preset.chemistry_flux_ISRF
											) #J/s/m^2 -> units_Draine
		#


		##Store the radiation fields
		dict_medium["matr_fieldrad_photon_{0}_perm_structure".format(name_field)
						] = matr_fieldrad_photon_perm
		dict_medium["matr_fieldrad_energy_{0}_perm_structure".format(name_field)
						] = matr_fieldrad_energy_perm
		dict_medium["matr_fieldrad_photon_{0}_perJ_structure".format(name_field)
						] = matr_fieldrad_photon_perJ
		#
		dict_medium["flux_{0}_uDraine_structure"
						.format(name_field)] = flux_uDraine
		dict_medium["matr_energyflux_{0}_structure"
						.format(name_field)] = energyflux
		dict_medium["matr_photonflux_{0}_structure"
						.format(name_field)] = photonflux
		#
		dict_medium["matr_fieldrad_photon_{0}_perm_chemistry".format(name_field)
						] = np.moveaxis(
						np.asarray([math._interpolate_pointstogrid(
						old_matr_values=matr_fieldrad_photon_perm[:,:,ii],
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem)
						for ii in range(0, len_wavelength)]),
						0, -1) #Move wavelength axis back to last axis position
		dict_medium["matr_fieldrad_energy_{0}_perm_chemistry".format(name_field)
						] = np.moveaxis(
						np.asarray([math._interpolate_pointstogrid(
						old_matr_values=matr_fieldrad_energy_perm[:,:,ii],
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem)
						for ii in range(0, len_wavelength)]),
						0, -1) #Move wavelength axis back to last axis position
		#Throw error if incorrect shape of interpolated grid
		if (dict_medium["matr_fieldrad_photon_{0}_perm_chemistry"
						.format(name_field)].shape
				!= (matr_y_chem.shape[0], matr_x_chem.shape[1],len_wavelength)):
			print((matr_y_chem.shape, matr_x_chem.shape, len_wavelength))
			print(dict_medium["matr_fieldrad_photon_{0}_perm_chemistry"
							.format(name_field)
						].shape)
			raise ValueError("Err: Incorrect shape of interpolated grid!")
		#
		dict_medium["flux_{0}_uDraine_chemistry".format(name_field)
					] = math._interpolate_pointstogrid(
						old_matr_values=flux_uDraine,
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem)
		dict_medium["matr_energyflux_{0}_chemistry".format(name_field)
					] = math._interpolate_pointstogrid(
						old_matr_values=energyflux,
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem)
		dict_medium["matr_photonflux_{0}_chemistry".format(name_field)
					] = math._interpolate_pointstogrid(
						old_matr_values=photonflux,
						old_points_yx=points_yx_struct,
						new_matr_x=matr_x_chem, new_matr_y=matr_y_chem,
						inds_valid=matr_y_diskinds_chem)
		#


		##Plot the radiation field outputs, if requested
		do_plot = True
		if do_plot:
			filepath_plot = dict_names["filepath_plots"]
			plt.figure(figsize=(20, 10))
			#Prepare plot base
			ncol = 3
			nrow = 1
			grid_base = utils._make_plot_grid(
									numsubplots=6, numcol=ncol, numrow=nrow,
			 						tbspaces=np.asarray([0.1]*(nrow-1)),
									lrspaces=np.asarray([0.2]*(ncol-1)),
									tbmargin=0.1, lrmargin=0.1)
			#
			vmin_photon = 3
			vmax_photon = 11
			vmin_energy = -2
			vmax_energy = 12
			vmin_stand = -5
			vmax_stand = 5
			cmap_photon = cmasher.rainforest_r
			cmap_energy = cmasher.savanna_r
			cmap_stand = cmasher.swamp_r
			contours_photon = None
			contours_energy = None
			contours_stand = None
			#
			#For UV photon flux (structure)
			utils.plot_diskscatter(matr=np.log10(photonflux),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=0, cmap=cmap_photon, contours=contours_photon,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{UV}$) "
								+"(#photons/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_photon, vmax=vmax_photon,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For UV energy flux (structure)
			utils.plot_diskscatter(matr=np.log10(energyflux),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=1, cmap=cmap_energy, contours=contours_energy,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{UV}$) "
								+"(erg/s/cm^2)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_energy, vmax=vmax_energy,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#For UV standard flux (structure)
			utils.plot_diskscatter(matr=np.log10(flux_uDraine),
			 			matr_x=matr_x_struct/au0, matr_y=matr_y_struct/au0,
						plotind=2, cmap=cmap_stand, contours=contours_stand,
						ccolor="white", calpha=0.7, cwidth=2,
						xlabel="Radius (au)", ylabel="Height (au)",
						boxtexts=None, docbar=True,
						cbarlabel=(r"Struct. "
					 			+"log$_{10}$(F$_\mathrm{UV}$) "
								+"(Rel. to Standard)"),
						docbarlabel=True,
						boxalpha=0.5, vmin=vmin_stand, vmax=vmax_stand,
						xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90],
						grid_base=grid_base)
			#Label and save the plot
			plt.suptitle("Radiation Fields: {0}".format(name_field))
			plt.tight_layout()
			plt.savefig(os.path.join(filepath_plot,
		 			"fig_radmc3d_radiation_field_{0}.png".format(name_field)))
			plt.close()
		#


		##Exit the function
		if do_verbose:
			print("_read_output_fieldradiation for field {0} complete."
					.format(name_field))
		#
		return
		#
#


##Class: Conch_Nautilus
##Purpose: Class for methods to prepare, manipulate, and run Nautilus model
class Conch_Nautilus(_Base):
	##Method: __init__()
	##Purpose: Initialization of this class instance
	def __init__(self, do_new_input, do_new_output, dict_stars=None, dict_grid=None, dict_medium=None, dict_names=None, dict_chemistry=None, do_verbose=False):
		##Store model parameters
		self._storage = {}
		self._store_info(do_verbose, "do_verbose")
		self._store_info(dict_names, "dict_names")
		self._store_info(dict_stars, "dict_stars")
		self._store_info(dict_grid, "dict_grid")
		self._store_info(dict_medium, "dict_medium")
		self._store_info(dict_chemistry, "dict_chemistry")
		#

		##Process chemistry, if new input/output requested
		if (do_new_input or do_new_output):
			##Process reaction files into reaction dictionaries
			self._generate_dict_reactions()
		#

		##Exit method
		return
		#
	#


	##Method: _generate_dict_reactions()
	##Purpose: Generate dictionary of reactions from input reactions files.
	def _generate_dict_reactions(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		filepath_chemistry = dict_names["filepath_chemistry"]
		filepath_save = dict_names["filepath_save_dictall"]
		#
		filename_species_gas = os.path.join(filepath_chemistry,
										preset.pnautilus_filename_speciesgas)
		filename_species_grain = os.path.join(filepath_chemistry,
										preset.pnautilus_filename_speciesgrain)
		filename_reactions_gas = os.path.join(filepath_chemistry,
	 							preset.pnautilus_filename_reactiongasorig)
		filename_reactions_grain = os.path.join(filepath_chemistry,
	 							preset.pnautilus_filename_reactiongrainorig)
		#
		table_filename_gas = os.path.join(filepath_save,
								preset.pnautilus_fileout_loadedreactionsgas)
		table_filename_grain = os.path.join(filepath_save,
								preset.pnautilus_fileout_loadedreactionsgrain)
		#

		##Run function to process reactions into dictionaries
		dict_set = reacts.generate_dict_reactions(mode="nautilus",
							filepath_chemistry=filepath_chemistry,
							filepath_save=filepath_save,
							do_verbose=do_verbose)
		#
		#Store lists of chemical species
		dict_chemistry["list_species_chem_all"] = dict_set["list_species_all"]
		dict_chemistry["list_species_chem_gas"] = dict_set["list_species_gas"]
		dict_chemistry["list_species_chem_grain"]=dict_set["list_species_grain"]
		#
		#Store the reactions and maximum id
		dict_chemistry["dict_reactions_base_gas"] = dict_set["reactions_gas"]
		dict_chemistry["dict_reactions_base_gr"] = dict_set["reactions_grain"]
		dict_chemistry["dict_reactions_base_UV"] = dict_set["reactions_UV"]
		dict_chemistry["max_reaction_id_orig"] = dict_set["max_reaction_id"]
		#

		##Exit the method
		return
	#


	##Method: run_chemistry()
	##Purpose: Run the chemistry for this chemistry model instance.
	def run_chemistry(self):
		self._run_pnautilus()
		return
	#


	##Method: _run_pnautilus()
	##Purpose: Create files for given model to run pnautilus
	def _run_pnautilus(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		which_dim_disk = dict_names["which_disk_dim"]
		filepath_chemistry = dict_names["filepath_chemistry"]
		#
		#Print some notes
		if do_verbose:
			print("Running _run_pnautilus()...")
		#


		##DISK PHASE
		#Print some notes
		if do_verbose:
			print("\n"+("-"*60)+"\nBEGINNING PROTOPLANETARY DISK PHASE:")
			print("DISK DIMENSION: {0}\n".format(which_dim_disk))
		#
		##Run Nautilus for disk phase
		if (which_dim_disk == "0D"):
			self._run_pnautilus_disk0D()
		elif (which_dim_disk == "1D"):
			self._run_pnautilus_disk1D()
		else:
			raise ValueError("Err: Disk dimension {0} not recognized!"
							.format(which_dim_disk))
		#
		#Print some notes
		if do_verbose:
			print("PROTOPLANETARY DISK PHASE COMPLETE.\n"+("-"*60)+"\n")
		#


		##Exit the function
		if do_verbose:
			print("_run_pnautilus() complete.")
		#
		return
		#
	#


	##Method: _run_pnautilus_disk1D()
	##Purpose: Run pnautilus for each 1D section within given model
	def _run_pnautilus_disk1D(self):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_names = self.get_info("dict_names")
		dict_stars = self.get_info("dict_stars")
		dict_grid = self.get_info("dict_grid")
		dict_medium = self.get_info("dict_medium")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		which_dir = dict_names["which_direction_1D"]
		num_cores = dict_grid["num_cores"]
		filename_init_abunds = dict_names["filename_init_abund"]
		#
		matr_x_chem = dict_grid["matr_x_chemistry"]
		matr_y_chem = dict_grid["matr_y_chemistry"]
		y_len = matr_x_chem.shape[0]
		x_len = matr_x_chem.shape[1]
		if (which_dir == "x"):
			num_slices = y_len #Count of slices; from along y-dim
		elif (which_dir == "y"):
			num_slices = x_len #Count of slices; from along x-dim
		else:
			raise ValueError("Err: Direction {0} not recognized!"
							.format(which_dir))
		#
		time_start_yr = dict_grid["time_start_yr"]
		time_end_yr = dict_grid["time_end_yr"]
		num_time = dict_grid["num_time"]
		#
		ionrate_CR = dict_stars["ionrate_CR"]
		#
		matr_nH = dict_medium["volnumdens_nH_chemistry"]
		matr_tempgas = dict_medium["matr_tempgas_chemistry"]
		matr_tempdust = dict_medium["matr_tempdust_chemistry"]
		matr_extinction = dict_medium["matr_extinction_UV_chemistry"]
		AVoverNHconv_cgs = dict_medium["AVoverNH_factor_cgs"]
		radius_gr_cgs = dict_medium["radius_grain_cgs"]
		matr_fluxUVtotuDraine = dict_medium["flux_UVtot_uDraine_chemistry"]
		matr_ionratexray = dict_medium["matr_ionrate_X_primary_chemistry"]
		#
		frac_dustovergasmass = dict_medium["frac_dustovergasmass"]
		densmass_grain_cgs = dict_medium["densmass_grain_cgs"]
		#
		#Print some notes
		if do_verbose:
			print("\nRunning _run_pnautilus_disk1D!")
		#

		##Initialize storage for valid chemistry indices
		inds_kept = [None]*num_slices
		dict_chemistry["cells_perslice"] = inds_kept
		#

		##Prepare dictionary of information for each 1D slice
		dicts_perslice = {}
		#Iterate through 1D slices
		for i_slice in range(0, num_slices):
			#Initialize dictionary for current 1D slice
			curr_dict = {}
			curr_dict["dir_orig"] = dict_names["filepath_chemistry"]
			curr_dict["filename_init_abunds"] = filename_init_abunds
			curr_dict["filepath_trash_global"] = dict_names[
													"filepath_trash_global"]
			curr_dict["do_verbose"] = do_verbose
			curr_dict["time_start_yr"] = time_start_yr
			curr_dict["time_end_yr"] = time_end_yr
			curr_dict["num_time"] = num_time
			curr_dict["num_cores"] = num_cores
			curr_dict["num_slices"] = num_slices
			#
			curr_dict["ionrate_CR"] = ionrate_CR
			curr_dict["frac_dustovergasmass"] = frac_dustovergasmass
			curr_dict["densmass_grain_cgs"] = densmass_grain_cgs
			curr_dict["radius_grain_cgs"] = radius_gr_cgs
			#

			#Extract x,y indices based on requested 1D direction
			if (which_dir == "y"): #Slice along y-direction
				curr_xs_raw = [i_slice for ii in range(0, y_len)]
				curr_ys_initdir_raw = [ii for ii in range(0, y_len)]
			elif (which_dir == "x"): #Slice along x-direction
				curr_xs_raw = [ii for ii in range(0, x_len)]
				curr_ys_initdir_raw = [i_slice for ii in range(0, x_len)]
			else:
				raise ValueError("Err: Direction {0} not recognized!"
								.format(which_dir))
			#
			#Keep only indices with valid chemical conditions in this slice
			tmp_num_points = len(curr_xs_raw)
			curr_xs = []
			curr_ys_initdir = []
			for jj in range(0, tmp_num_points):
				#Fetch current pair of indices
				xx = curr_xs_raw[jj]
				yy = curr_ys_initdir_raw[jj]
				#Extract gas H density
				dens_gas_cgs = (matr_nH[yy,xx]
			 					* conv_perm3topercm3) #[1/m^3] -> [1/cm^-3]
				curr_flux_uDraine = matr_fluxUVtotuDraine[yy,xx] #[uDraine]
				#Verify density threshold
				if (dens_gas_cgs < preset.chemistry_thres_mindensgas_cgs):
					if do_verbose:
						print("Cell y{0},x{1} has gas dens. {2:.3e} cm^-3."
								.format(yy, xx, dens_gas_cgs)
								+" This is below threshold. Skipping...")
					continue
				#
				#Verify non-nan density
				elif np.isnan(dens_gas_cgs):
					if do_verbose:
						print("Cell y{0},x{1} has nan gas dens. {2:.3e} cm^-3."
								.format(yy, xx, dens_gas_cgs)
								+" This is invalid. Skipping...")
					continue
				#
				#Verify UV flux threshold
				elif (curr_flux_uDraine > preset.chemistry_thres_maxUV_uDraine):
					if do_verbose:
						print("Cell y{0},x{1} has UV flux {2:.3e} G0."
								.format(yy, xx, curr_flux_uDraine)
								+" This is above threshold. Skipping...")
					continue
				#
				else: #Store if valid cell
					curr_xs.append(xx)
					curr_ys_initdir.append(yy)
			#
			#Flip y direction
			curr_ys = curr_ys_initdir[::-1] #Flip to start at disk surf.
			#
			#Store the valid indices
			curr_dict["ind_xs"] = np.asarray(curr_xs)
			curr_dict["ind_ys"] = np.asarray(curr_ys)
			curr_num_points = len(curr_xs)
			curr_dict["num_points"] = curr_num_points

			#Compute distances along the slice, relative to top of slice
			tmp_refpoint = [matr_y_chem[curr_ys[0],curr_xs[0]],
							matr_x_chem[curr_ys[0],curr_xs[0]]]
			curr_dict["arr_dist_slice"] = np.asarray([
						math.calc_distance(p1=tmp_refpoint,
							p2=[matr_y_chem[curr_ys[ii],curr_xs[ii]],
								matr_x_chem[curr_ys[ii],curr_xs[ii]]])
							for ii in range(0, curr_num_points)])
			#

			#Fetch disk characteristics at these indices
			curr_dict["arr_volnumdens_nH"] = matr_nH[curr_ys, curr_xs]
			curr_dict["arr_tempgas"] = matr_tempgas[curr_ys, curr_xs]
			curr_dict["arr_tempdust"] = matr_tempdust[curr_ys, curr_xs]
			curr_dict["arr_extinction"] = matr_extinction[curr_ys, curr_xs]
			curr_dict["AVoverNHconv_cgs"] = AVoverNHconv_cgs
			curr_dict["radius_gr_cgs"] = radius_gr_cgs
			curr_dict["flux_UV_uDraine_slicetop"] = (
									matr_fluxUVtotuDraine[curr_ys, curr_xs][0])
			curr_dict["arr_flux_UV_uDraine"] = (
									matr_fluxUVtotuDraine[curr_ys, curr_xs])
			curr_dict["arr_ionrate_X"] = (
										matr_ionratexray[curr_ys, curr_xs])
			#

			#Store the compiled dictionary of info
			dicts_perslice[i_slice] = curr_dict

			#Store the indices kept for this slice
			inds_kept[i_slice] = {"ind_xs":curr_dict["ind_xs"],
									"ind_ys":curr_dict["ind_ys"]}
		#

		##Run process routine for 1D slices
		utils._run_pnautilus_diskslice_1D(dicts_perslice=dicts_perslice)
		#
		#Print some notes
		if do_verbose:
			print("Computations for all {0} 1D slices complete."
					.format(num_slices))
		#

		##Exit the method
		if do_verbose:
			print("_run_pnautilus_disk1D() complete.")
		return
	#


	##Method: _fetch_disk_chemistry()
	##Purpose: Convert collection of chemistry output into 0D, 1D, or 2D arrays
	def _fetch_disk_chemistry(self, mode_species, which_times_yr, list_species_orig, do_allow_load, do_save, rtol_time, xbounds_total, ybounds_zoverr, cutoff_minval=None, ys_cell_au=None, xs_cell_au=None, is_slice_au=None, dist_tol=(0.5*au0), rtol_zoverr=None):
		##Extract global variables
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dir_orig = dict_names["filepath_chemistry"]
		name_model = dict_names["model"]
		#
		cells_perslice = self.get_info("dict_chemistry")["cells_perslice"]
		matr_x = dict_grid["matr_x_chemistry"]
		matr_y = dict_grid["matr_y_chemistry"]
		matr_zoverr = (matr_y / matr_x)
		matr_nH = self.get_info("dict_medium")["volnumdens_nH_chemistry"]
		ylen = matr_y.shape[0]
		xlen = matr_y.shape[1]
		#
		list_keyext = ["gas", "gr", "grsurf", "grmant", "tot"]
		keyext_time = "time"
		arr_time = dict_grid["arr_time_yr"]
		num_time = len(which_times_yr)
		#
		filepath_base = os.path.join(dir_orig, preset.pnautilus_disk_dir_done)
		filepath_save = os.path.join(dir_orig,preset.pnautilus_processed_output)
		#
		#Prepare some calculations ahead of time
		#nH calculations
		coldens_nH = np.array([np.trapz(matr_nH[:,xx], x=matr_y[:,xx])
		 						for xx in range(0, xlen)])
		rowdens_nH = np.array([np.trapz(matr_nH[yy,:], x=matr_x[yy,:])
								for yy in range(0, ylen)])
		#z/r index calculations
		if (ybounds_zoverr is not None):
			#Raise an error if nearest points to z/r range are beyond tolerance
			near_bots = [np.min(np.abs(matr_zoverr[:,xx][
							(matr_zoverr[:,xx] >= ybounds_zoverr[0])]
							- ybounds_zoverr[0])) for xx in range(0, xlen)]
			near_tops = [np.min(np.abs(matr_zoverr[:,xx][
							(matr_zoverr[:,xx] <= ybounds_zoverr[1])]
							- ybounds_zoverr[1])) for xx in range(0, xlen)]
			if (ybounds_zoverr[0] == 0): #Edge case where boundary is zero
				if (np.min(near_bots) > rtol_zoverr):
					raise ValueError("Err: z/r point(s) beyond tolerance:\n{0}"
								.format(near_bots))
				elif (np.min(near_tops) > rtol_zoverr):
					raise ValueError("Err: z/r point(s) beyond tolerance:\n{0}"
								.format(near_tops))
			elif ((np.min(near_bots) / ybounds_zoverr[0]) > rtol_zoverr):
				raise ValueError("Err: z/r point(s) are beyond tolerance:\n{0}"
								.format(near_bots))
			elif ((np.min(near_tops) / ybounds_zoverr[0]) > rtol_zoverr):
				raise ValueError("Err: z/r point(s) are beyond tolerance:\n{0}"
								.format(near_tops))
			#
			#Extract indices for values within z/r range
			inds_zoverr = ((matr_zoverr >= ybounds_zoverr[0])
							& (matr_zoverr <= ybounds_zoverr[1]))
		#

		#Print some notes
		if do_verbose:
			print("Running _fetch_disk_chemistry...")
		#

		##If no species given, fetch all possible species
		if (list_species_orig is None):
			#Set the filepaths
			filename_species_gas = os.path.join(dir_orig,
										preset.pnautilus_filename_speciesgas)
			filename_species_grain = os.path.join(dir_orig,
										preset.pnautilus_filename_speciesgrain)
			#Load the species
			list_species_orig = reacts._fetch_all_species(
						filename_species_gas=filename_species_gas,
						filename_species_grain=filename_species_grain,
						do_remove_phase_marker=True)["all"]
		#

		##Fetch unique species from original species list (e.g., split ratios)
		list_species_unique = np.unique([item2 for item1 in list_species_orig
										for item2 in item1.split("/")
										if (item2 != "")]).tolist()
		#
		#Print some notes
		if do_verbose:
			print("Original requested species: {0}\nUnique species: {1}"
					.format(np.sort(list_species_orig),
					 		np.sort(list_species_unique)))
		#

		##Split original species list into preloaded vs new-to-load, as allowed
		#If loading is allowed
		if do_allow_load:
			#Create file for processed output chemistry, if not present yet
			#Below temporary until new models generated !!!
			if (not os.path.isdir(filepath_save)):
				comm = subprocess.call(["mkdir", filepath_save])
				if comm != 0:
					raise ValueError("Whoa! Above error returned from mkdir!")
			#
			#Split out species not saved yet
			list_species_unsaved = [
					item for item in list_species_unique
					if (not os.path.isfile(os.path.join(filepath_save,
					 	"{0}_{1}_{2}.npy"
						.format(name_model, mode_species, item)
						)))
			]
			#Split out species already saved
			list_species_alreadysaved = [
					item for item in list_species_unique
					if (os.path.isfile(os.path.join(filepath_save,
					 	"{0}_{1}_{2}.npy"
						.format(name_model, mode_species, item)
						)))
			]
		#
		#Otherwise, if no loading allowed
		else:
			list_species_unsaved = list_species_unique
			list_species_alreadysaved = []
		#
		#Print some notes
		if do_verbose:
			print("Species output previously saved: {0}\nOutput needed now: {1}"
					.format(list_species_alreadysaved, list_species_unsaved))
		#

		##Raise an error if unrecognized modes
		#For species mode
		tmp_list = ["mol", "el"]
		if (mode_species not in tmp_list):
			raise ValueError("Err: Invalid mode {0}!\nValid modes are: {1}"
							.format(mode_species, tmp_list))
		#

		##Initialize storage for chemistry per species, per timestep
		#Initialize dictionary of matrices to store chemistry
		dict_res = {key:None for key in list_species_unique}
		#Load any pre-processed output chemistry, if allowed
		for spec in list_species_alreadysaved:
			currname_save = ("{0}_{1}_{2}.npy"
							.format(name_model, mode_species, spec))
			currpath_save = os.path.join(filepath_save, currname_save)
			dict_res[spec] = np.load(currpath_save, allow_pickle=True).item()
			#Print some notes
			if do_verbose:
				print("{0}:{1} data loaded from {2}."
					.format(mode_species, spec, currpath_save))
		#
		#Load new output chemistry
		for spec in list_species_unsaved:
			#Store matrix
			dict_res[spec] = {keyext:(np.ones(shape=(num_time, ylen, xlen)
											)*np.nan) #T,Y,X
								for keyext in list_keyext} #Per phase (e.g. gas)
			#
			#Store temporal placeholder
			dict_res[spec]["time"] = [None]*num_time #For exact time points
		#

		##Determine slices that contain target chemistry
		target_dirs = sorted([item for item in os.listdir(filepath_base)
						if item.startswith(preset.pnautilus_disk_dir_done)])
		#Throw error if no slices found
		if (len(target_dirs) == 0):
			raise ValueError("Err: No target slices found?\n{0}"
							.format(target_dirs))
		#

		##Extract chemistry from target locations
		if (len(list_species_unsaved) > 0):
			num_targetslices = len(target_dirs)
			for ii in range(0, num_targetslices):
				#Extract current slice index
				curr_dir = target_dirs[ii]
				curr_islice = int((re.search("\_i([0-9]+)$",curr_dir)).group(1))
				curr_yxs = cells_perslice[curr_islice]
				#
				#Print some notes
				if do_verbose:
					print("Fetching chem. in dir. {0} for slice {1}..."
							.format(curr_dir, curr_islice))
				#

				#Fetch chemistry for species along current slice
				if (mode_species == "mol"):
					curr_dict = self._fetch_output1D_abundances(
								i_slice=curr_islice,
				 				which_env="disk",
								list_species=list_species_unsaved)
					curr_all_times = curr_dict[spec+"|"+keyext_time]
				elif (mode_species == "el"):
					curr_dict = self._fetch_output1D_elements(
							i_slice=curr_islice,
			 				which_env="disk",list_elements=list_species_unsaved)
					curr_all_times = arr_time
				#

				#Store this cell's chemistry in matrices
				for spec in list_species_unsaved:
					#Extract time points
					curr_tinds = np.array([np.argmin(
										np.abs(curr_all_times
										 		- which_times_yr[tt]))
										for tt in range(0, num_time)])
					curr_times = curr_all_times[curr_tinds]
					#
					#Throw error if actual times too far from target times
					if (not np.allclose(curr_times, which_times_yr,
					 					rtol=rtol_time)):
						raise ValueError("Time not in tol.!\n{0} vs {1}"
										.format(curr_times, which_times_yr))
					#
					#Store time points
					dict_res[spec]["time"][0:num_time] = curr_times
					#
					#Iterate through phases
					for keyext in list_keyext:
						#Extract and store chemistry for this species and times
						curr_vals = curr_dict[spec+"|"+keyext][curr_tinds]
						dict_res[spec][keyext][
										:,curr_yxs["ind_ys"],curr_yxs["ind_xs"]
										] = curr_vals
					#
				#
			#
		#
		#Otherwise, note nothing new to load
		else:
			if do_verbose:
				print("Nothing new to load. All chemistry data pre-loaded.")
			pass
		#

		##Save each unsaved species in dictionary as processed chemistry file
		if do_save:
			for ii in range(0, len(list_species_unsaved)):
				curr_spec = list_species_unsaved[ii]
				currname_save = ("{0}_{1}_{2}.npy"
								.format(name_model, mode_species, curr_spec))
				currpath_save = os.path.join(filepath_save, currname_save)
				np.save(currpath_save, dict_res[curr_spec])
				if do_verbose:
					print("{0}:{1} data saved to {2}."
						.format(mode_species, curr_spec, currpath_save))
		#

		##Copy over fetched abundances and apply any external calculations
		if do_verbose:
			print(("Externally computing ratios, applying cutoffs, etc."
					+"\nOriginal species: {0}\nMin. value cutoff: {1}")
					.format(list_species_orig, cutoff_minval))
		#

		##Compute dimensional abundances
		dict_comp = {}
		#Iterate through target species and ratios
		for curr_spec in list_species_unique:
			#Prepare storage for this species
			dict_comp[curr_spec] = {key:None for key in list_keyext}
			#Fetch abundances for this species for each phase
			for curr_key in list_keyext:
				#Copy over base numerator (or single) abundance
				new_matr = (dict_res[curr_spec][curr_key].copy()
							* matr_nH)
				#
				#Apply cutoff, if given
				if (cutoff_minval is not None):
					new_matr[new_matr < cutoff_minval] = np.nan
				#
				#Store singular entry
				dict_comp[curr_spec][curr_key] = new_matr.copy()
			#
		#

		##Extract abundances along target dimension
		tmplist = ["2D_abs", "2D_rel", "1D_col_abs", "1D_row_abs", "1D_col_rel",
		 			"1D_row_rel", "total_abs", "total_rel"]
		dict_fin_all = {key:{} for key in tmplist}
		#
		#---
		#For 2D point values
		#Iterate through target species and ratios
		for curr_set in list_species_orig:
			#Prepare storage for this set
			curr_abs = {key:None for key in list_keyext}
			curr_rel = {key:None for key in list_keyext}
			#
			#Split set into ratio components
			curr_split = curr_set.split("/")
			#
			#Store time and location information
			curr_time = dict_res[curr_split[0]][keyext_time].copy()
			curr_abs[keyext_time] = curr_time
			curr_rel[keyext_time] = curr_time
			curr_abs["x"] = matr_x[0,:]
			curr_rel["x"] = matr_x[0,:]
			curr_abs["y"] = matr_y[:,0]
			curr_rel["y"] = matr_y[:,0]
			#
			#Iterate through phases
			for curr_key in list_keyext:
				#Store base abundances
				curr_abs[curr_key] = dict_comp[curr_split[0]][curr_key].copy()
				curr_rel[curr_key] = (dict_comp[curr_split[0]][curr_key].copy()
												/ matr_nH)
				#
				#Accumulate ratios as applicable
				for ii in range(1, len(curr_split)): #Activates only if ratio
					curr_abs[curr_key] /= dict_comp[curr_split[ii]][curr_key
																	].copy()
					curr_rel[curr_key] /= (dict_comp[curr_split[ii]][curr_key
														].copy() / matr_nH)
			#
			#Store the 2D results
			dict_fin_all["2D_abs"][curr_set] = curr_abs
			dict_fin_all["2D_rel"][curr_set] = curr_rel
		#
		#---
		#For 1D column|row values
		#Iterate through target species and ratios
		for curr_set in list_species_orig:
			#Prepare storage for this set
			curr_abs_col = {key:None for key in list_keyext}
			curr_rel_col = {key:None for key in list_keyext}
			curr_abs_row = {key:None for key in list_keyext}
			curr_rel_row = {key:None for key in list_keyext}
			#
			#Split set into ratio components
			curr_split = curr_set.split("/")
			curr_time = dict_res[curr_split[0]][keyext_time].copy()
			curr_abs_col[keyext_time] = curr_time
			curr_rel_col[keyext_time] = curr_time
			curr_abs_row[keyext_time] = curr_time
			curr_rel_row[keyext_time] = curr_time
			curr_abs_col["x"] = matr_x[0,:]
			curr_rel_col["x"] = matr_x[0,:]
			curr_abs_row["x"] = matr_x[0,:]
			curr_rel_row["x"] = matr_x[0,:]
			#
			#Iterate through phases
			for curr_key in list_keyext:
				#For 1D column
				#Iterate through axis points and compute per 1D set
				curr_coldens_abs = np.ones(xlen)*np.nan
				curr_coldens_rel = np.ones(xlen)*np.nan
				for xx in range(0, xlen):
					curr_comp = dict_comp[curr_split[0]][curr_key].copy()
					#Erase values beyond y-bounds, if given
					if (ybounds_zoverr is not None):
						curr_comp[:,inds_zoverr] = 0
					#
					#Set base column density
					curr_coldens_abs[xx] = np.trapz(curr_comp[:,:,xx],
												x=matr_y[:,xx])
					curr_coldens_rel[xx] = (curr_coldens_abs[xx]
					 						/ coldens_nH[xx])
					#Iterate through components
					for ii in range(1, len(curr_split)):
						curr_comp = dict_comp[curr_split[ii]][curr_key].copy()
						#Erase values beyond y-bounds, if given
						if (ybounds_zoverr is not None):
							curr_comp[:,inds_zoverr] = 0
						#
						tmp_abs = np.trapz(curr_comp[:,:,xx], x=matr_y[:,xx])
						tmp_rel = (tmp_abs / coldens_nH[xx])
						curr_coldens_abs[xx] /= tmp_abs
						curr_coldens_rel[xx] /= tmp_rel
				#
				#Store 1D column densities
				curr_abs_col[curr_key] = curr_coldens_abs
				curr_rel_col[curr_key] = curr_coldens_rel
				#
				#---
				#
				#For 1D row
				#Iterate through axis points and compute per 1D set
				curr_coldens_abs = np.ones(ylen)*np.nan
				curr_coldens_rel = np.ones(ylen)*np.nan
				for yy in range(0, ylen):
					curr_comp = dict_comp[curr_split[0]][curr_key].copy()
					#Erase values beyond y-bounds, if given
					if (ybounds_zoverr is not None):
						curr_comp[:,inds_zoverr] = 0
					#
					#Set base column density
					curr_coldens_abs[yy] = np.trapz(curr_comp[:,yy,:],
												x=matr_x[yy,:])
					curr_coldens_rel[yy] = (curr_coldens_abs[yy]
					 						/ rowdens_nH[yy])
					#Iterate through components
					for ii in range(1, len(curr_split)):
						curr_comp = dict_comp[curr_split[ii]][curr_key].copy()
						#Erase values beyond y-bounds, if given
						if (ybounds_zoverr is not None):
							curr_comp[:,inds_zoverr] = 0
						#
						tmp_abs = np.trapz(curr_comp[:,yy,:], x=matr_x[yy,:])
						tmp_rel = (tmp_abs / rowdens_nH[yy])
						curr_coldens_abs[yy] /= tmp_abs
						curr_coldens_rel[yy] /= tmp_rel
				#
				#Store 1D row densities
				curr_abs_row[curr_key] = curr_coldens_abs
				curr_rel_row[curr_key] = curr_coldens_rel
			#
			#Store the 1D results
			#Calculation here, but not sure how row should be done for z/r disk
			dict_fin_all["1D_col_abs"][curr_set] = curr_abs_col
			dict_fin_all["1D_col_rel"][curr_set] = curr_rel_col
			dict_fin_all["1D_row_abs"][curr_set] = None #curr_abs_row
			dict_fin_all["1D_row_rel"][curr_set] = None #curr_rel_row
		#
		#---
		#For total disk-integrated values
		#Iterate through target species and ratios
		for curr_set in list_species_orig:
			#Prepare storage for this set
			curr_abs_total = {key:None for key in list_keyext}
			curr_rel_total = {key:None for key in list_keyext}
			#
			#Split set into ratio components
			curr_split = curr_set.split("/")
			curr_time = dict_res[curr_split[0]][keyext_time].copy()
			curr_abs_total[keyext_time] = curr_time
			curr_rel_total[keyext_time] = curr_time
			#
			#Iterate through phases
			for curr_key in list_keyext:
				tmpmatr = dict_comp[curr_split[0]][curr_key].copy()
				#Erase values beyond y-bounds, if given
				if (ybounds_zoverr is not None):
					tmpmatr[:,inds_zoverr] = 0
				#
				#Remove nan columns
				tmpnH = matr_nH.copy()
				tmpinds = [xx for xx in range(0, xlen)
							if any(np.isnan(tmpmatr[:,:,xx].flatten()))]
				tmpmatr = np.delete(tmpmatr, tmpinds, axis=2)
				tmpy = np.delete(matr_y[np.newaxis,:,:].copy(), tmpinds, axis=2)
				tmpx = np.delete(matr_x[np.newaxis,:,:].copy(), tmpinds, axis=2)
				tmpnH = np.delete(tmpnH, tmpinds, axis=1)
				#
				#If boundaries are given, interpolate then trim over bounds
				if (xbounds_total is not None):
					tmp_xadd = []
					linex_orig = tmpx[0,0,:].copy()
					#Fold in given boundaries along single array
					if ((xbounds_total[0] is not None)
				 				and (xbounds_total[0] > linex_orig.min())): #Left
						tmp_xadd.append(xbounds_total[0])
					if ((xbounds_total[1] is not None)
				 				and (xbounds_total[1] < linex_orig.max())):#Right
						tmp_xadd.append(xbounds_total[1])
					linex_int = np.sort(np.unique(
										np.concatenate((tmp_xadd, linex_orig))))
					#
					#Interpolate all matrices over new boundaries
					tmpx = scipy_interp1d(y=tmpx, x=linex_orig,
											axis=-1)(linex_int)
					tmpy = scipy_interp1d(y=tmpy, x=linex_orig,
					 						axis=-1)(linex_int)
					tmpmatr = scipy_interp1d(y=tmpmatr, x=linex_orig,
					 						axis=-1)(linex_int)
					tmpnH = scipy_interp1d(y=tmpnH, x=linex_orig,
					 						axis=-1)(linex_int)
					#
					#Cut at the boundaries
					tmp_bounds = [-np.inf, np.inf]
					if (xbounds_total[0] is not None):
						tmp_bounds[0] = xbounds_total[0]
					if (xbounds_total[1] is not None):
						tmp_bounds[1] = xbounds_total[1]
					tmpboundinds = ((linex_int >= tmp_bounds[0])
					 				& (linex_int <= tmp_bounds[1]))
					tmpx = tmpx[:,:,tmpboundinds]
					tmpy = tmpy[:,:,tmpboundinds]
					tmpmatr = tmpmatr[:,:,tmpboundinds]
					tmpnH = tmpnH[:,tmpboundinds]
				#
				#Compute base integral across full disk slice
				tmppart = np.trapz(tmpmatr, x=tmpy, axis=1)
				curr_integ_abs = np.trapz(tmppart, x=tmpx[0,0,:], axis=1)
				tmppart = np.trapz(tmpnH, x=tmpy[0,:,:], axis=0)
				curr_integ_denom = np.trapz(tmppart, x=tmpx[0,0,:])
				curr_integ_rel = (curr_integ_abs / curr_integ_denom)
				#
				#Iterate through components and accumulate ratios
				for ii in range(1, len(curr_split)):
					tmpmatr = dict_comp[curr_split[ii]][curr_key].copy()
					#Erase values beyond y-bounds, if given
					if (ybounds_zoverr is not None):
						tmpmatr[:,inds_zoverr] = 0
					#
					#Remove nan columns
					tmpmatr = np.delete(tmpmatr, tmpinds, axis=2)
					#Apply boundaries, if given
					if (xbounds_total is not None):
						tmpmatr = scipy_interp1d(y=tmpmatr, x=linex_orig,
						 						axis=-1)(linex_int)
						#
						#Cut at the boundaries
						tmpmatr = tmpmatr[:,:,tmpboundinds]
					#
					#Compute base integral across full disk slice
					tmppart = np.trapz(tmpmatr, x=tmpy, axis=1)
					tmp_abs = np.trapz(tmppart, x=tmpx[0,0,:], axis=1)
					tmppart = np.trapz(tmpnH, x=tmpy[0,:,:], axis=0)
					tmp_denom = np.trapz(tmppart, x=tmpx[0,0,:])
					tmp_rel = (tmp_abs / tmp_denom)
					#
					curr_integ_abs /= tmp_abs
					curr_integ_rel /= tmp_rel
				#

				#Store total integrated  values
				curr_abs_total[curr_key] = curr_integ_abs
				curr_rel_total[curr_key] = curr_integ_rel
			#
			#Store the total integrated results
			dict_fin_all["total_abs"][curr_set] = curr_abs_total
			dict_fin_all["total_rel"][curr_set] = curr_rel_total
		#

		##Return the dictionary of extracted abundances
		if do_verbose:
			print("Run of _fetch_disk_chemistry complete!\n")
		#
		return dict_fin_all
	#


	##Method: _fetch_output1D_abundances()
	##Purpose: Fetch abundances for given mol. from pnautilus 1D-slice output files
	def _fetch_output1D_abundances(self, list_species, which_env, i_slice):
		#Extract global variables
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		dir_orig = dict_names["filepath_chemistry"]
		list_species_all = dict_chemistry["list_species_chem_all"]
		num_time = dict_grid["num_time"]
		arr_time = dict_grid["arr_time_yr"]
		#
		#Fetch time based on phase
		if (which_env == "disk"):
			arr_time = dict_grid["arr_time_yr"]
			filepath_base = os.path.join(dir_orig,
						preset.pnautilus_disk_dir_done,
	 					"{0}_i{1}".format(preset.pnautilus_disk_dir_done,
												i_slice))
		#
		else:
			raise ValueError("Err: Environment {0} invalid!".format(which_env))
		#
		len_t = len(arr_time)
		#

		#Load filepath
		curr_path = filepath_base
		if os.path.isdir(os.path.join(curr_path, "ab")):
			curr_path = os.path.join(curr_path, "ab") #Tack on ab dir.
		#

		##Collect info from each chemical species
		dict_species = {}
		#Iterate through species
		for spec_raw in list_species:
			#Fetch base of this species
			spec_base = re.sub("^(J|K)", "", spec_raw)

			#Prepare names for chemical phases of this species
			currtxtg = "{0}.ab".format(spec_base) #Current gas-phase .ab file
			currtxts = "J{0}.ab".format(spec_base) #Curr. gr(s)-phase .ab file
			currtxtm = "K{0}.ab".format(spec_base) #Curr. gr(m)-phase .ab file
			#Note which chemical phases exist for this species
			has_gas = (currtxtg.replace(".ab","") in list_species_all)
			has_grsurf = (currtxts.replace(".ab","") in list_species_all)
			has_grmant = (currtxtm.replace(".ab","") in list_species_all)
			#Throw error if none exist
			if not any([has_gas, has_grsurf, has_grmant]):
				raise ValueError("Err: No gas,gr. phase for {0}?\n{1} {2} {3}"
							.format(spec_base, currtxtg, currtxts, currtxtm))
			#

			#Gather all phases into lists for easy iteration
			ind_g = 0
			ind_s = 1
			ind_m = 2
			list_txts = [currtxtg, currtxts, currtxtm]
			list_bools = [has_gas, has_grsurf, has_grmant]
			num_phases = len(list_txts)

			#Load slice chemistry length
			len_1D = None
			for ii in range(0, num_phases):
				if (list_bools[ii]):
					len_1D = (np.genfromtxt(os.path.join(curr_path,
			 						list_txts[ii]), comments="!", dtype=float
									).shape[1] - 1) #Slice length; exclude time
					#Stop early if length determined
					break
				#
			#

			#Load data for existing phases
			list_chems = [None]*num_phases
			for ii in range(0, num_phases):
				#If current phase exists
				if (list_bools[ii]):
					#Read in raw data, including times
					currload = np.genfromtxt(
										os.path.join(curr_path, list_txts[ii]),
										comments="!", dtype=float)
					#Split out abundances (first column is time)
					currtime = currload[:,0]
					currchem = currload[:,1:currload.shape[1]]
					#
					#Verify array lengths
					if (currchem.shape[1] != len_1D):
						raise ValueError("Err: Unequal shapes!\n{0}\n{1}\n{2}"
										.format(len_1D, currload, currchem))
				#
				#Otherwise, set 0-value placeholder
				else:
					currtime = arr_time
					currchem = np.zeros(shape=(num_time,len_1D))
				#
				#Verify latest time
				if (not np.allclose(currtime, arr_time)):
					raise ValueError("Err: Mismatching times!\n{0}\n{1}"
									.format(arr_time, currtime))
				#
				#Store latest chemistry
				list_chems[ii] = currchem
				#
			#

			#Throw error if any nan values
			if any([(any(np.isnan(item.flatten()))) for item in list_chems]):
				raise ValueError("Err: Nans!\n{0}\n{1}\n{2}"
						.format(list_chems[0], list_chems[1], list_chems[2]))
			#

			#Store compiled data for this species
			dict_species[spec_base+"|time"] = arr_time
			dict_species[spec_base+"|gas"] = list_chems[ind_g]
			dict_species[spec_base+"|grsurf"] = list_chems[ind_s]
			dict_species[spec_base+"|grmant"] = list_chems[ind_m]
			dict_species[spec_base+"|gr"] = list_chems[ind_s]+list_chems[ind_m]
			dict_species[spec_base+"|tot"] = (list_chems[ind_g]
			 							+ list_chems[ind_s] + list_chems[ind_m])
			#
		#


		##Return the completed dictionary
		return dict_species
	#


	##Method: _fetch_output1D_elements()
	##Purpose: Fetch elements for given mol. from pnautilus 1D-slice output files
	def _fetch_output1D_elements(self, which_env, i_slice, list_elements=None):
		raise ValueError("Err: This feature is not yet available!")

		#Extract global variables
		do_verbose = self.get_info("do_verbose")
		dict_grid = self.get_info("dict_grid")
		dict_names = self.get_info("dict_names")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		model_name = dict_names["model"]
		dir_orig = dict_names["filepath_chemistry"]
		list_species_all = dict_chemistry["list_species_chem_all"]
		list_possible_elements = dict_chemistry["chemistry_possible_elements"]
		list_species_ignore = preset.pnautilus_species_ignore
		num_time = dict_grid["num_time"]
		arr_time = dict_grid["arr_time_yr"]
		rtol_conservation = preset.rtol_element_conservation
		list_ext = ["|gas", "|grsurf", "|grmant", "|gr", "|tot"]
		#
		#Print some notes
		if do_verbose:
			print("\n> Running _fetch_output1D_elements() for {0}, slice {1}!"
					.format(model_name, i_slice))
		#
		#Fetch time based on phase
		if (which_env == "disk"):
			arr_time = dict_grid["arr_time_yr"]
			filepath_base = os.path.join(dir_orig,
						preset.pnautilus_disk_dir_done,
	 					"{0}_i{1}".format(preset.pnautilus_disk_dir_done,
												i_slice))
			filename_init_abunds = dict_names["filename_init_abund"]
		#
		else:
			raise ValueError("Err: Environment {0} invalid!".format(which_env))
		#
		len_t = len(arr_time)
		#
		#Fetch results for all elements if none specified
		if (list_elements is None):
			list_elements = list_possible_elements
		#
		#Fetch initial abundances
		dict_inits = self._fetch_writtenlist_elements(
											filename=filename_init_abunds,
											list_el=list_possible_elements)
		#

		##Build list of elements within any given ratios
		list_unique = []
		for curr_el in list_elements:
			#If ratio, split
			if ("/" in curr_el):
				list_unique += curr_el.split("/")
			#Otherwise, store as single element
			else:
				list_unique.append(curr_el)
		#
		#Remove any duplicates from base list of elements
		list_unique = list(set(list_unique))
		list_ratios = [item for item in list_elements if ("/" in item)]
		#
		#Throw error if any elements not recognized
		if any([(item not in list_possible_elements) for item in list_unique]):
			raise ValueError("Err: Unrecognized element in {0}!\n{1}\n{2}"
							.format(list_elements, list_unique,
							 		list_possible_elements))
		#
		#Print some notes
		if do_verbose:
			print("Fetching all molecular species in this slice...")
		#

		##Fetch all available species from this slice
		filepath_ab = filepath_base
		if os.path.isdir(os.path.join(filepath_ab, "ab")):
			filepath_ab = os.path.join(filepath_ab, "ab") #Tack on ab dir.
		filenames = [item for item in os.listdir(filepath_ab)
					if (item.endswith(".ab"))]
		all_species = [item.replace(".ab", "") for item in filenames]
		#Throw error if number of species != number of files
		if len(filenames) != len(os.listdir(filepath_ab)):
			raise ValueError("Err: Species missing:\n{0}\n{1}"
							.format(all_species, filepath_ab))
		#
		#Print some notes
		if do_verbose:
			print("Total of {0} molecular species fetched."
					.format(len(all_species)))
			print("Accumulating elemental abundances across all species...")
		#

		##Accumulate elemental abundances across all species
		done_species = []
		dict_abunds = {(key1+key2):None #np.zeros(shape=shape_matr)
		 					for key2 in list_ext
		 				for key1 in list_possible_elements} #For el. abundances
		for curr_spec_raw in all_species:
			#Fetch current base species (remove grain markers)
			curr_base = re.sub("^(J|K)", "", curr_spec_raw)
			if (curr_base in done_species):
				continue
			#
			#Skip if this species should be ignored
			if ((curr_spec_raw in list_species_ignore)
			 				or (curr_base in list_species_ignore)):
				if do_verbose:
					print("Ignoring {0} ({1})..."
						.format(curr_spec_raw,curr_base))
					continue
			#
			#Fetch current species abundances
			dict_currab = self._fetch_output1D_abundances(
								list_species=[curr_base], which_env=which_env,
								i_slice=i_slice)
			#Extract elemental breakdown of this species
			dict_breakdown = utils._get_elements(curr_base, do_ignore_JK=False)

			#

			#Update elemental abundances with latest species
			for curr_el in dict_breakdown:
				curr_stoic = dict_breakdown[curr_el] #Count of el. in species
				for curr_ext in list_ext:
					#Accumulate if previously stored
					if (dict_abunds[curr_el+curr_ext] is not None):
						dict_abunds[curr_el+curr_ext] += (curr_stoic
					 						* dict_currab[curr_base+curr_ext])
					#Otherwise, initialize
					else:
						dict_abunds[curr_el+curr_ext] = (curr_stoic
					 						* dict_currab[curr_base+curr_ext])
					#
				#
			#

			#Store this species as considered
			done_species.append(curr_base)
		#

		##Compute any ratios, if given
		if (len(list_ratios) > 0):
			#Iterate through ratios
			for curr_ratio in list_ratios:
				curr_split = curr_ratio.split("/")
				#Iterate through chemical phases
				for curr_ext in list_ext:
					#Initialize with values for first element of ratio
					dict_abunds[curr_ratio+curr_ext] = (
								dict_abunds[curr_split[0]+curr_ext].copy())
					#Iterate through remaining components of ratio
					for jj in range(1, len(curr_split)):
						dict_abunds[curr_ratio+curr_ext] /= (
								dict_abunds[curr_split[jj]+curr_ext])
			#
		#

		##Verify that elemental abundances sum up to initial abundances
		for curr_el in list_unique:
			if (not np.allclose(dict_abunds[curr_el+"|tot"],
								dict_inits[curr_el+"|tot"],
								rtol=rtol_conservation)):
				#
				#Print abundance output
				print("\n\n---\nElemental conservation error! Printing info:")
				print("Current slice: {0}".format(i_slice))
				print("Current element: {0}".format(curr_el))
				print("Initial elemental abundance: {0}"
						.format(dict_inits[curr_el+"|tot"]))
				#print("Modeled gas elemental abundance over time:\n{0}\n"
				#		.format(dict_abunds[curr_el+"|gas"]))
				#print("Modeled grain elemental abundance over time:\n{0}\n"
				#		.format(dict_abunds[curr_el+"|gr"]))
				print("Modeled total elemental abundance over time:")
				for zz in range(0, len_t):
					print("t={0:.2e}yr: X={1}"
						.format(arr_time[zz],dict_abunds[curr_el+"|tot"][zz,:]))
				#
				#Plot abundance output
				plt.close()
				plt.plot(arr_time, dict_abunds[curr_el+"|tot"], color="black",
							label="Total", alpha=0.6)
				plt.plot(arr_time, dict_abunds[curr_el+"|gas"], color="cyan",
							label="Gas", alpha=0.6)
				plt.plot(arr_time, dict_abunds[curr_el+"|gr"], color="tomato",
							label="Grain", alpha=0.6)
				plt.axhline(dict_inits[curr_el+"|tot"], color="gray",
							label="Initial", alpha=0.6)
				plt.xscale("log")
				plt.yscale("log")
				plt.xlabel("Time (yr)")
				plt.ylabel("Relative Abundance (n_X/n_H)")
				plt.legend(loc="best", frameon=False)
				plt.tight_layout()
				plt.show()
				#
				raise ValueError("Err: El-abund. not conserved!\n{0}\n{1}\n{2}"
								.format(curr_el, dict_abunds[curr_el+"|tot"],
										dict_inits[curr_el+"|tot"]))
				#
				#Raise the error
				#raise ValueError("Err: El-abund. not conserved!")
		#

		##Return the accumulated elemental abundances
		if do_verbose:
			print("\n> Run of _fetch_output1D_elements() for {0}:{1} complete!"
					.format(model_name, i_slice))
		#
		return dict_abunds
	#


	##Method: _fetch_writtenlist_abundances()
	##Purpose: Fetch abundances for given mol. from pnautilus abundance.in file
	def _fetch_writtenlist_abundances(self, filename, list_spec):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		list_species_pnautilus = dict_chemistry["list_species_chem_all"]
		list_species_pnautilus_gas = dict_chemistry["list_species_chem_gas"]
		list_species_pnautilus_gr = dict_chemistry["list_species_chem_grain"]
		zero_val = 0 #preset.pnautilus_threshold_minval
		#Print some notes
		if do_verbose:
			print("\n> Running _fetch_writtenlist_abundances()!")
		#


		##Load (relative) abundance data from file
		data_abund = np.genfromtxt(filename, comments="!", dtype=str,
		 							delimiter="=")
		names_abund = [item.strip() for item in data_abund[:,0]]#Rm. encomp. " "
		values_abund = [float(item.replace("D", "E")) #Fortran conversion
						for item in data_abund[:,1]]
		num_abund = data_abund.shape[0]


		##If no species given, compute for all species in file
		if list_spec is None:
			list_spec = names_abund
		#
		num_spec = len(list_spec)


		##Collect info from each chemical species
		list_keyext = ["|gas", "|grmant", "|grsurf"] #Key ext.
		dict_species = {}
		for ii in range(0, num_spec):
			dict_species.update({(list_spec[ii]+keyext):np.array([zero_val])
			 					for keyext in list_keyext})
		#
		#Iterate through species
		for ii in range(0, num_spec):
			curr_spec = list_spec[ii]

			#Determine species+phase label for this species
			#For gas-phase species
			if (((curr_spec in list_species_pnautilus_gas)
									and (not curr_spec.startswith("J"))
									and (not curr_spec.startswith("K")))
							or ((curr_spec in list_species_pnautilus_gr)
							 		and (not curr_spec.startswith("J"))
									and (not curr_spec.startswith("K")))):
				curr_label = curr_spec+"|gas"
			#For grain-phase species
			elif ((curr_spec in list_species_pnautilus_gr)
							and (curr_spec.startswith("J"))):
				curr_label = curr_spec+"|grsurf"
			elif ((curr_spec in list_species_pnautilus_gr)
							and (curr_spec.startswith("K"))):
				curr_label = curr_spec+"|grmant"
			#Otherwise, throw error
			else:
				raise ValueError("Err: {0} not recognized by pnautilus!"
								.format(curr_spec))
			#

			#Store initial abundance of this species
			#For present species
			if curr_spec in names_abund:
				dict_species[curr_label] = np.array([values_abund[ii]]) #One t
			#For non-present species
			else:
				dict_species[curr_label] = np.array([zero_val])
			#
		#


		##Compute totals across phases
		for ii in range(0, num_spec):
			curr_spec = list_spec[ii]
			#For total grain phase
			curr_gr = (dict_species[(curr_spec+"|grmant")]
			 			+ dict_species[(curr_spec+"|grsurf")])
			dict_species[(curr_spec+"|gr")] = curr_gr
			#For total of all phases
			curr_tot = (dict_species[(curr_spec+"|gas")]
						+ dict_species[(curr_spec+"|grmant")]
			 			+ dict_species[(curr_spec+"|grsurf")])
			dict_species[(curr_spec+"|tot")] = curr_tot
		#


		##Return the completed dictionary
		if do_verbose:
			print("> Run of _fetch_writtenlist_abundances() complete!\n")
		#
		return dict_species
	#


	##Method: _fetch_writtenlist_elements()
	##Purpose: Fetch abundances for given el. from pnautilus abundance.in file
	def _fetch_writtenlist_elements(self, filename, list_el):
		##Load instance variables
		do_verbose = self.get_info("do_verbose")
		dict_chemistry = self.get_info("dict_chemistry")
		#
		list_keyext = ["|gas", "|grmant", "|grsurf", "|gr", "|tot"] #Key ext.
		zero_val = 0 #preset.pnautilus_threshold_minval
		#
		#Print some notes
		if do_verbose:
			print("\n> Running _fetch_writtenlist_elements()!")
		#


		##Fetch relative abundances for every species in file
		dict_species = self._fetch_writtenlist_abundances(filename=filename,
									list_spec=None)
		num_spec = len(dict_species)
		list_species = np.unique([item.split("|")[0] for item in dict_species])
		#


		##Initialize dictionary of elements with empty abundances
		dict_el = {(key+keyext):np.array([zero_val]) for keyext in list_keyext
		 			for key in list_el}
		#
		#Reserve spots for atomic H and molecular H2
		dict_el.update({("H_atom"+keyext):zero_val for keyext in list_keyext})
		dict_el.update({("H_molec"+keyext):zero_val for keyext in list_keyext})
		#


		##Compute total elemental abundances from accumulated species abundances
		#Iterate through species
		for curr_spec in list_species:
			#Handle individual entries of species
			#For atomic H
			if (curr_spec == "H"):
				for keyext in list_keyext:
					dict_el["H_atom"+keyext] += (dict_species["H"+keyext] * 1)
			#For molecular H2
			elif (curr_spec == "H2"):
				for keyext in list_keyext:
					dict_el["H_molec"+keyext] += (dict_species["H2"+keyext] * 2)
			#Otherwise, carry on
			else:
				pass
			#

			#Handle all regular species
			#Extract elemental count for this species
			dict_res = utils._get_elements(curr_spec, do_ignore_JK=False)
			for inner_el in dict_res:
				curr_count = dict_res[inner_el]
				#Iterate through phases
				for keyext in list_keyext:
					dict_el[inner_el+keyext] = (
										dict_el[inner_el+keyext] +
											(dict_species[curr_spec+keyext]
											* curr_count))
			#
		#


		##Return the completed dictionary
		if do_verbose:
			print("> Run of _fetch_writtenlist_elements() complete!\n")
		#
		return dict_el
	#
#













#
