###FILE: argonaut_utils.py
###PURPOSE: Script for external, fundamental, class-instance-independent functions (e.g., text processing functions and functions for parallelization).
###DATE FILE CREATED: 2022-09-18
###DEVELOPERS: (Jamila Pegues)


##Import necessary modules
import re
import os
import sys
import glob
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
import astropy.constants as const
import matplotlib.gridspec as gridder
from mpl_toolkits.axes_grid1 import make_axes_locatable as axloc
plt.close()
import scipy.signal as scipy_signal
from scipy.interpolate import interp1d as scipy_interp1d
import multiprocessing as mp
try:
	import subprocess
except ModuleNotFoundError:
	import subprocess32 as subprocess
#
import argonaut_constants as preset
import argonaut_math as math
#
#Set some conversions
conv_perm3topercm3 = 1 / (100**3) #[W/(m^2*Hz)] -> [erg/(s*cm^2*Hz)]
au0 = const.au.value #[m]
#




##------------------------------------------------------------------------------
##UTILS: TEXT
#
##Function: _get_elements()
##Purpose: Break a chemical species/molecule/compound into its elements
def _get_elements(mol, do_ignore_JK, do_strip_orientation=True):
	#Cast the mol to a list, if individual molecule given
	if isinstance(mol, str):
		mol_list = [mol]
	#
	else:
		mol_list = mol.copy()
	#

	#Remove any ion markers
	mol_list = [re.sub("(\+|-)$", "", item) for item in mol_list]

	#Remove any grain markers
	if do_ignore_JK:
		mol_list = [item[1:len(item)]
					if (item.startswith("J") or item.startswith("K")) else item
					for item in mol_list]
	#

	#Strip off any orientation markers, if requested
	if do_strip_orientation:
		mol_list = [re.sub("^((c-)|(l-))", "", item) for item in mol_list]
	#

	#Join the molecules
	mol_mod = "".join(mol_list)

	#Locate all elemental strings (marked by upper and maybe lowercase)
	list_nameandnums = re.findall('[A-Z][a-z]?[0-9]*', mol_mod)

	#Determine each element and its counts
	dict_el = {}
	#Iterate through name+number combinations
	for nameandnum in list_nameandnums:
		is_num = re.search(r"[0-9]+", nameandnum)
		if bool(is_num):
			curr_count = int(is_num.group())
		else:
			curr_count = 1
		#
		#Store the name and count
		curr_el = re.sub("[0-9]+", "", nameandnum)
		if curr_el in dict_el: #Accumulate if already stored
			dict_el[curr_el] += curr_count
		else: #Otherwise, just store number
			dict_el[curr_el] = curr_count
		#
	#

	#Return the dictionary of elements and their counts
	return dict_el
#


##Function: _write_text()
##Purpose: Write given string to given file
def _write_text(text, filename):
	#Write to the file
	with open(filename, 'w') as openfile:
		openfile.write(text)
	#Close method
	return
#




##------------------------------------------------------------------------------
##UTILS: LOAD
#
##Function: _load_data_crosssecs_UV()
##Purpose: Load molecular cross-sectional UV data
def _load_data_crosssecs_UV(do_skip_elements, do_ignore_JK, possible_elements, do_skip_bannedspecies, wavelength_min, wavelength_max, which_mol=None, do_verbose=True):
	##Extract global variables
	if do_skip_bannedspecies:
		list_banned_reaction_species = preset.pnautilus_banned_reaction_species
	#Initialize dictionary to store cross-sections
	dict_mol = {}
	#
	#Print some notes
	if do_verbose:
		print("\n> Running _load_data_crosssecs_UV...")
	#


	##FOR UV CROSS-SECTIONS
	##Load cross-sectional data for each molecule
	#Extract all cross-section files
	list_filenames_raw = os.listdir(preset.filepath_chemistry_crosssecs_UV)
	list_filenames = [os.path.join(preset.filepath_chemistry_crosssecs_UV,
	 								item)
					for item in list_filenames_raw] #File paths+file names
	list_molnames = [item.replace(".txt", "")
	 				for item in list_filenames_raw] #Molecule names
	num_mol = len(list_molnames)
	#
	#Fix special cases where filename does not match chemical species name
	special_species_names = preset.special_species_names
	for key in special_species_names:
		if key in list_molnames: #If present, replace
			list_molnames[list_molnames.index(key)
							] = special_species_names[key]
	#

	#Prepare list of possible columns in file
	colnames = ["wavelength", "photoabsorption", "photodissociation",
				"photoionisation"]
	storeroots = [None, "photoabs", "photodiss", "photoion"]
	conv_toSI = [1E-9, (1.0/(100**2)), (1.0/(100**2)), (1.0/(100**2))]
	#NOTE: Conversion factors above are for: [nm -> m], [cm^2 -> m^2]
	#

	#Iterate through molecules and store cross-section information
	for ii in range(0, num_mol):
		#Extract current molecule and associated file
		curr_mol = list_molnames[ii]
		curr_filename = list_filenames[ii]

		#Skip this molecule if not a target molecule
		if ((which_mol is not None) and (curr_mol not in which_mol)):
			continue
		#

		#Print some notes
		if do_verbose:
			print("Loading UV cross-sectional data for molecule {0}..."
					.format(curr_mol))
		#

		#Skip this species if consists of elements not considered
		if do_skip_elements:
			curr_els = [key for key in _get_elements(curr_mol,
													do_ignore_JK=do_ignore_JK)]
			if any([(item not in possible_elements)
						for item in curr_els]):
				#Print some notes
				if do_verbose:
					print("{0} has invalid elements (in {1}). Skipping..."
							.format(curr_mol, curr_els))
				#Skip this species
				continue
		#

		#Skip this species if banned (unrecognized)
		if do_skip_bannedspecies and (curr_mol in list_banned_reaction_species):
			#Print some notes
			if do_verbose:
				print("Banned species (unrecognized by pnautilus) "
						+"in {0}. Skipping..."
						.format(curr_mol))
			#Skip ahead
			continue
		#

		#Initialize new dictionary to hold information
		curr_dict = {} #key:None for key in storenames}
		dict_mol[curr_mol] = curr_dict
		for jj in range(1, len(colnames)):
			curr_root = storeroots[jj]
			curr_dict["wave_{0}_UV".format(curr_root)] = None
			curr_dict["cross_{0}_UV".format(curr_root)] = None
		#

		#Determine table header of current file
		with open(curr_filename, 'r') as openfile:
			#Extract list of column titles for this file
			all_lines = openfile.readlines()
			header_str = [item for item in all_lines
						if item.startswith("# wavelength ")][0] #Col header
			header_str = header_str.replace("# ", "") #Remove comment marker
			header_col = header_str.split() #Convert to list of col. titles
		#Ensure that column headers match expected headers
		if any([item not in colnames for item in header_col]):
			raise ValueError("Whoa! Mismatch column titles?\n{0} vs. {1}"
							.format(colnames, header_col))
		#

		#Load the cross-section data
		curr_data = np.genfromtxt(curr_filename, comments="#", dtype=float)
		num_col = curr_data.shape[1]

		#Store each column of data
		for jj in range(1, num_col):
			curr_ind = colnames.index(header_col[jj])
			curr_root = storeroots[curr_ind] #Current phototype root
			#Print some notes
			if do_verbose:
				print("Working on {0}, {1}...".format(curr_mol, curr_root))
			#
			#Reduce redundant resolution
			curr_x_old = curr_data[:,0] *conv_toSI[0] #Wavelength;[m]
			curr_y_old = curr_data[:,jj] *conv_toSI[curr_ind]#-> S.I.
			dict_res = math._remove_adjacent_redundancy(
				y_old=curr_y_old, x_old=curr_x_old, do_trim_zeros=True
			)
			#Skip this phototype if no nonzero values
			if dict_res is None:
				if do_verbose:
					print("No nonzero values for: {0}, {1}. Skipping..."
						.format(curr_mol, curr_root))
				continue
			#
			#Otherwise, store streamlined values
			curr_x_mid = dict_res["x"]
			curr_y_mid = dict_res["y"]

			#Record blanks if cross-section outside of wavelength range
			if ((curr_x_mid.max() <= wavelength_min)
			 				or (curr_x_mid.min() >= wavelength_max)):
				#Print some notes
				if do_verbose:
					print("{0} cross-sec off-range. Skipping.".format(curr_mol)
							+"\nCross-sec: {0:.2e}|{1:.2e}"
							.format(curr_x_mid.min(), curr_x_mid.max())
							+"\nRange: {0:.2e}|{1:.2e}"
							.format(wavelength_min, wavelength_max))
				#
				curr_dict["wave_{0}_UV".format(curr_root)] = None
				curr_dict["cross_{0}_UV".format(curr_root)] = None
				#Skip ahead
				continue
			#

			#Trim the cross-section into given wavelength range as needed
			#For minimum cutoff
			if ((wavelength_min is not None)
			 				and (curr_x_mid.min() < wavelength_min)):
				curr_x_triml = np.sort(np.unique(np.concatenate(
									([wavelength_min],
									curr_x_mid[curr_x_mid > wavelength_min]))))
				curr_y_triml = scipy_interp1d(y=curr_y_mid, x=curr_x_mid
											)(curr_x_triml)
			#
			else:
				curr_x_triml = curr_x_mid
				curr_y_triml = curr_y_mid
			#
			#For maximum cutoff
			if ((wavelength_max is not None)
			 				and (curr_x_triml.max() > wavelength_max)):
				curr_x_trimr = np.sort(np.unique(np.concatenate(
								([wavelength_max],
								curr_x_triml[curr_x_triml < wavelength_max]))))
				curr_y_trimr = scipy_interp1d(y=curr_y_triml, x=curr_x_triml
											)(curr_x_trimr)
			#
			else:
				curr_x_trimr = curr_x_triml
				curr_y_trimr = curr_y_triml
			#
			#Cast the final trim results
			curr_x_new = curr_x_trimr
			curr_y_new = curr_y_trimr

			#Store the streamlined+trimmed cross-section
			curr_dict["wave_{0}_UV".format(curr_root)] = curr_x_new
			curr_dict["cross_{0}_UV".format(curr_root)] = curr_y_new
			#Print some notes
			if do_verbose:
				print("Redundant resolution has been reduced: {0} to {1}."
						.format(curr_x_old.shape, curr_x_new.shape))
				print("Original min, max wavelength: {0:.4e} - {1:.4e}."
						.format(curr_x_old.min(), curr_x_old.max()))
				print("New min, max wavelength: {0:.4e} - {1:.4e}."
						.format(curr_x_new.min(), curr_x_new.max()))
				print("")
		#
	#


	##Return the dictionary of molecular info and exit the method
	if do_verbose:
		print("Run of _load_data_crosssecs_UV() complete!")
	#
	return dict_mol
#


##Function: _load_elements()
##Purpose: Load elements from a given file
def _load_elements(filename):
	##Read in, process, and store unique elements
	data_el = np.genfromtxt(filename, delimiter="", comments="!",dtype=str)
	if (len(data_el.shape) > 1): #Compress to first column if necessary
		data_el = data_el[:,0]

	#Extract unique elements
	list_el = [mol.strip() for mol in data_el]
	return list_el
#




##------------------------------------------------------------------------------
##UTILS: PLOTS
#
##Function: _make_plot_grid()
##Purpose: Make base grid to hold subplots and panels
def _make_plot_grid(numsubplots, numcol, numrow, tbspaces, lrspaces, tbmargin, lrmargin):
	##Set up grid of subplots
	grid_base = [] #List to hold subplot data

	#Sum left-right and top-bot distances for normalization later
	sublength = (1.0/numcol)
	subwidth = (1.0/numrow)
	lrtot = (sublength*numcol) + lrspaces.sum() + (2*lrmargin)
	tbtot = (subwidth*numrow) + tbspaces.sum() + (2*tbmargin)

	#Iterate through subplots
	for gi in range(0, numsubplots):
		#Calculate current row and column indices
		ri = gi // numcol #Row index
		ci = gi % numcol #Column index

		#Calculate top, bottom, left, right positions of current subplot
		#For top and bottom
		tophere = (1 - (tbmargin/1.0/tbtot) - ri*(subwidth/1.0/tbtot)
						- (tbspaces[0:ri].sum()/1.0/tbtot))
		bothere = (1 - (tbmargin/1.0/tbtot) - (ri+1)*(subwidth/1.0/tbtot)
						- (tbspaces[0:ri].sum()/1.0/tbtot))
		#For left and right
		lefthere = ((lrmargin/1.0/lrtot) + ci*(sublength/1.0/lrtot)
						+ (lrspaces[0:ci].sum()/1.0/lrtot))
		righthere = ((lrmargin/1.0/lrtot) + (ci+1)*(sublength/1.0/lrtot)
						+ (lrspaces[0:ci].sum()/1.0/lrtot))

		#Append the calculated grid for this subplot
		grid_base.append(gridder.GridSpec(1, 1))
		grid_base[-1].update(top=tophere, bottom=bothere,
					left=lefthere, right=righthere)


	##Return the completed calculated grid
	return grid_base
#


##Function: plot_diskslice()
##Purpose: Plot a two-dimensional (2D) slice from a disk
def plot_diskslice(matr, arr_x, arr_y, plotind, cmap, contours=None, contours_zoverr=None, ccolor="black", calpha=0.7, cwidth=2, ccolor_zoverr="gray", calpha_zoverr=0.5, cwidth_zoverr=1.5, backcolor=None, rmxticklabel=False, rmyticklabel=False, xlabel="", ylabel="", tickwidth=3, tickheight=5, boxtexts=None, dolog=False, numcmapticks=100, numcbarticks=10, aspect="auto", title=None, docbar=False, cbarlabel="", docbarlabel=False, boxalpha=0.5, vmin=None, vmax=None, rmxticktitle=False, rmyticktitle=False, xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90], boxcolor="white", boxfontcolor="black", labelpad=0, cbarlabelpad=20, nbins=6, ticksize=14, textsize=16, titlesize=18, plotoutlinecolor=None, plotoutlinestyle=None, grid_base=None, plotblank=False, fig=None, do_return_panel=False):
	##Graph the given disk slice
	#Initialize a subplot assuming background figure
	if fig is None:
		if grid_base is None: #If single plot desired
			panel = plt.subplot(111, aspect=aspect)
		else: #If multiple subplots
			panel = plt.subplot(grid_base[plotind][0,0], aspect=aspect)
	#Otherwise, initialize a subplot within given figure
	else:
		if grid_base is None: #If single plot requested
			panel = fig.add_subplot(111, aspect=aspect)
		else: #If multiple subplots
			panel = fig.add_subplot(grid_base[plotind][0,0], aspect=aspect)

	#Set background color of plot, if so requested
	if backcolor is not None:
		panel.set_facecolor(backcolor)

	#Plot a blank square if no matrix requested
	if plotblank is True:
		panel.tick_params(labelbottom=False, labelleft=False)
		panel.tick_params(width=0, size=0, labelsize=0)
		#Removes outline of blank square
		panel.spines["bottom"].set_color("white")
		panel.spines["left"].set_color("white")
		panel.spines["right"].set_color("white")
		panel.spines["top"].set_color("white")
		#
		if (do_return_panel):
			return panel
		else:
			return
	#

	#Change box color if different color given
	if plotoutlinecolor is not None:
		panel.spines["bottom"].set_color(plotoutlinecolor)
		panel.spines["left"].set_color(plotoutlinecolor)
		panel.spines["right"].set_color(plotoutlinecolor)
		panel.spines["top"].set_color(plotoutlinecolor)
		panel.spines["bottom"].set_linestyle(plotoutlinestyle)
		panel.spines["left"].set_linestyle(plotoutlinestyle)
		panel.spines["right"].set_linestyle(plotoutlinestyle)
		panel.spines["top"].set_linestyle(plotoutlinestyle)
	#

	##Determine colorbar min, max, and ticks
	if vmin is None: #For colorbar minimum, if not given
		vmin = np.nanmin(matr)
		if np.isnan(vmin) or np.isinf(vmin):
			vmin = None
	if vmax is None: #For colorbar maximum, if not given
		vmax = np.ceil(np.nanmax(matr))
		if np.isnan(vmax) or np.isinf(vmax):
			vmax = None
	#Prepare colorbar caps (extended or exclusive)
	extend = "both"


	##Plot the matrix
	if ((vmin is not None) and (vmax is not None)):
		cmapticks = np.linspace(vmin, vmax, numcmapticks)
		maphere = panel.contourf(arr_x, arr_y, matr, cmapticks, cmap=cmap,
								vmin=vmin, vmax=vmax, extend=extend)
		cbarticks = np.linspace(vmin, vmax, numcbarticks)
	else:
		maphere = panel.contourf(arr_x, arr_y, matr, cmap=cmap, extend=extend)
		cbarticks = None


	##Plot the given contours
	if (contours is not None): #Plots contours explicitly, if given
		clinehere = panel.contour(arr_x, arr_y, matr, levels=contours,
					alpha=calpha, colors=ccolor, linewidths=cwidth)
		panel.clabel(clinehere, fontsize=textsize, fmt="%.1f")


	##Plot the given z/r contours as well
	if (contours_zoverr is not None): #Calculates and plots given z/r contours
		tmp_matr_zoverr = (arr_y / arr_x) #Matrix of z/r
		clinehere = panel.contour(arr_x, arr_y, tmp_matr_zoverr,
		 			levels=contours_zoverr, alpha=calpha_zoverr,
					colors=ccolor_zoverr, linewidths=cwidth_zoverr)
		panel.clabel(clinehere, fontsize=textsize, fmt="z/r=%.1f")


	##Format and label the axes
	#Below formats plot ticks
	panel.locator_params(nbins=nbins) #Reduces number of x-axis ticks
	panel.tick_params(width=tickwidth, size=tickheight, labelsize=ticksize,
							direction="in")

	#Remove tick labels, if so desired
	if rmxticklabel: #If no bottom axis labels desired
		panel.tick_params(axis="x", which="both", labelbottom=False)
	if rmyticklabel: #If no left axis labels desired
		panel.tick_params(axis="y", which="both", labelleft=False)

	#Label axes, if so desired
	if not rmyticktitle:
		panel.set_ylabel(ylabel, fontsize=titlesize,
							labelpad=labelpad) #y-axis label
	if not rmxticktitle:
		panel.set_xlabel(xlabel, fontsize=titlesize,
							labelpad=labelpad) #x-axis label
	if title is not None:
		panel.set_title(title)
	#

	#Zoom in on subplot, if so desired
	if xlim is not None: #For x-axis range
		panel.set_xlim(xlim)
	if ylim is not None: #For y-axis range
		panel.set_ylim(ylim)
	#

	#Add in in-graph labels, if so desired
	if boxtexts is not None:
		for ai in range(0, len(boxtexts)):
			panel.text(boxxs[ai], boxys[ai], boxtexts[ai],
				backgroundcolor=boxcolor, color=boxfontcolor,
				alpha=boxalpha,
				fontsize=textsize, transform=panel.transAxes,
				verticalalignment="top")


	##Generate a colorbar, if so desired
	if docbar:
		num_cbarlabels = 5
		if (np.abs(vmax) >= 1): #If above 1, allows sensible floor/ceiling calls
			fin_cbarticks = np.linspace(np.floor(vmin), np.ceil(vmax),
										num_cbarlabels)
			cbar_format = "{0}"
		else: #If below 1, do not apply floor/ceiling calls
			fin_cbarticks = np.linspace(vmin, vmax, num_cbarlabels)
			cbar_format = "{0:.2f}"
		#
		fin_cbarticklabels = [(int(item)) if (item.is_integer()) else item
								for item in fin_cbarticks]
		#
		cbar = plt.colorbar(maphere, ticks=cbarticks, extend=extend,
						cax=axloc(panel).append_axes("right", size="5%", pad=0))
		cbar.ax.tick_params(labelsize=ticksize)
		#
		cbar.set_ticks(fin_cbarticks)
		cbar.set_ticklabels([cbar_format.format(item)
		 					for item in fin_cbarticklabels])
		if docbarlabel: #Labels colorbar itself
			cbar.set_label(cbarlabel, labelpad=cbarlabelpad,
						fontsize=titlesize, rotation=270)
		else:
			cbar.set_ticklabels([])
		#
	#


	##Exit the method
	if (do_return_panel):
		return panel
	else:
		return
#


##Function: plot_diskscatter()
##Purpose: Plot a collection of two-dimensional (2D) points from a disk
def plot_diskscatter(matr, matr_x, matr_y, plotind, cmap, show_diskheights=False, x_diskheights=None, y_diskheights=None, color_diskheights="turquoise", linewidth_diskheights=5, linestyle_diskheights="-", alpha_diskheights=0.5, contours=None, ccolor="black", calpha=0.7, cwidth=2, backcolor=None, rmxticklabel=False, rmyticklabel=False, xlabel="", ylabel="", tickwidth=3, tickheight=5, boxtexts=None, dolog=False, numcmapticks=100, numcbarticks=10, aspect="auto", title=None, docbar=False, cbarlabel="", docbarlabel=False, boxalpha=0.5, vmin=None, vmax=None, rmxticktitle=False, rmyticktitle=False, xlim=None, ylim=None, boxxs=[0.05], boxys=[0.90], boxcolor="white", boxfontcolor="black", labelpad=-0, cbarlabelpad=20, nbins=6, ticksize=14, textsize=16, titlesize=18, plotoutlinecolor=None, plotoutlinestyle=None, grid_base=None, plotblank=False, fig=None):
	##Graph the given disk points
	#Initialize a subplot assuming background figure
	if fig is None:
		if grid_base is None: #If single plot desired
			panel = plt.subplot(111, aspect=aspect)
		else: #If multiple subplots
			panel = plt.subplot(grid_base[plotind][0,0], aspect=aspect)
	#Otherwise, initialize a subplot within given figure
	else:
		if grid_base is None: #If single plot requested
			panel = fig.add_subplot(111, aspect=aspect)
		else: #If multiple subplots
			panel = fig.add_subplot(grid_base[plotind][0,0], aspect=aspect)

	#Set background color of plot, if so requested
	if backcolor is not None:
		panel.set_facecolor(backcolor)

	#Plot a blank square if no matrix requested
	if plotblank is True:
		panel.tick_params(labelbottom=False, labelleft=False)
		panel.tick_params(width=0, size=0, labelsize=0)
		#Removes outline of blank square
		panel.spines["bottom"].set_color("white")
		panel.spines["left"].set_color("white")
		panel.spines["right"].set_color("white")
		panel.spines["top"].set_color("white")
		return
	#

	#Change box color if different color given
	if plotoutlinecolor is not None:
		panel.spines["bottom"].set_color(plotoutlinecolor)
		panel.spines["left"].set_color(plotoutlinecolor)
		panel.spines["right"].set_color(plotoutlinecolor)
		panel.spines["top"].set_color(plotoutlinecolor)
		panel.spines["bottom"].set_linestyle(plotoutlinestyle)
		panel.spines["left"].set_linestyle(plotoutlinestyle)
		panel.spines["right"].set_linestyle(plotoutlinestyle)
		panel.spines["top"].set_linestyle(plotoutlinestyle)
	#

	##Determine colorbar min, max, and ticks
	if vmin is None: #For colorbar minimum, if not given
		vmin = np.nanmin(matr)
		if np.isnan(vmin) or np.isinf(vmin):
			vmin = None
	if vmax is None: #For colorbar maximum, if not given
		vmax = np.ceil(np.nanmax(matr))
		if np.isnan(vmax) or np.isinf(vmax):
			vmax = None
	#Prepare colorbar caps (extended or exclusive)
	extend = "both"


	##Plot the matrix
	if (vmin is not None) and (vmax is not None):
		cmapticks = np.linspace(vmin, vmax, numcmapticks)
		maphere = panel.contourf(matr_x, matr_y, matr, cmapticks, cmap=cmap,
								vmin=vmin, vmax=vmax, extend=extend)
		cbarticks = np.linspace(vmin, vmax, numcbarticks)
	else:
		maphere = panel.contourf(matr_x, matr_y, matr, cmap=cmap, extend=extend)
		cbarticks = None
	#


	##Plot scale heights, if so desired
	if show_diskheights:
		for jj in range(0, len(x_diskheights)):
			panel.plot(x_diskheights[jj], y_diskheights[jj],
		 			color=color_diskheights, linewidth=linewidth_diskheights,
					linestyle=linestyle_diskheights, alpha=alpha_diskheights)
	#


	##Plot the given contours
	if contours is not None: #Plots contours explicitly, if given
		clinehere = panel.contour(matr_x, matr_y, matr, levels=contours,
					alpha=calpha, colors=ccolor, linewidths=cwidth)
		panel.clabel(clinehere, fontsize=textsize, fmt="%.0f")


	##Format and label the axes
	#Below formats plot ticks
	panel.locator_params(nbins=nbins) #Reduces number of x-axis ticks
	panel.tick_params(width=tickwidth, size=tickheight, labelsize=ticksize,
							direction="in")

	#Remove tick labels, if so desired
	if rmxticklabel: #If no bottom axis labels desired
		panel.tick_params(axis="x", which="both", labelbottom=False)
	if rmyticklabel: #If no left axis labels desired
		panel.tick_params(axis="y", which="both", labelleft=False)

	#Label axes, if so desired
	if not rmyticktitle:
		panel.set_ylabel(ylabel, fontsize=titlesize,
							labelpad=labelpad) #y-axis label
	if not rmxticktitle:
		panel.set_xlabel(xlabel, fontsize=titlesize,
							labelpad=labelpad) #x-axis label
	if title is not None:
		panel.set_title(title)
	#

	#Zoom in on subplot, if so desired
	if xlim is not None: #For x-axis range
		panel.set_xlim(xlim)
	if ylim is not None: #For y-axis range
		panel.set_ylim(ylim)
	#

	#Add in in-graph labels, if so desired
	if boxtexts is not None:
		for ai in range(0, len(boxtexts)):
			panel.text(boxxs[ai], boxys[ai], boxtexts[ai],
				backgroundcolor=boxcolor, color=boxfontcolor,
				alpha=boxalpha,
				fontsize=textsize, transform=panel.transAxes,
				verticalalignment="top")


	##Generate a colorbar, if so desired
	if docbar:
		cbar = plt.colorbar(maphere, ticks=cbarticks, extend=extend,
			cax=axloc(panel).append_axes("right", size="3.5%", pad=0))
		cbar.ax.tick_params(labelsize=ticksize)
		if docbarlabel: #Labels colorbar itself
			cbar.set_label(cbarlabel, labelpad=cbarlabelpad,
						fontsize=titlesize, rotation=270)


	##Exit the method
	return
#


##Function: plot_lines()
##Purpose: Plot a series of lines (e.g., spectra)
def plot_lines(arr_xs, arr_ys, plotind, doxlog=False, doylog=False, textsize=16, ticksize=14, titlesize=18, legendfontsize=None, legendcolspacing=1.0, legendlabelspacing=0.25, legendhandlelength=1.0, colors=["black"], labels=[""], styles=["-"], markers="", markersizes=5, nxticks=6, tickwidth=3, tickheight=5, dolegend=False, dozeroline=True, boxtexts=None, boxxs=[0.05], boxys=[0.90], boxalpha=0.5, ylabel="", xlabel=r"$\lambda$ [!]", labelpad=None, linewidth=3, alpha=0.8, tickspace=5, grid_base=None, aspect="auto", rmxticktitle=False, rmyticktitle=False, title=None, plotblank=False, rmxticklabel=False, rmyticklabel=False, xmin=None, xmax=None, xlim=None, ymin=None, ylim=None, legendloc="best", plotoutlinestyle="--", plotoutlinecolor=None, plotoutlinewidth=1, do_squareplot=True, doxlines=False, xlines=None, xlinecol="gray", xlinesty="--", xlinewidth=1, doylines=False, ylines=None, ylinecol="gray", ylinesty="--", ylinewidth=1, doyspans=False, yspans=None, yspancol="gray", yspanalpha=0.5, returnax=False, givenax=None, fig=None, ind_topline=None):
	##Initialize a subplot assuming background figure
	if (fig is None):
		if (grid_base is None): #If single plot desired
			panel = plt.subplot(111, aspect=aspect)
		else: #If multiple subplots
			panel = plt.subplot(grid_base[plotind][0,0], aspect=aspect)
	#Otherwise, initialize a subplot within given figure
	else:
		if (grid_base is None): #If single plot requested
			panel = fig.add_subplot(111, aspect=aspect)
		else: #If multiple subplots
			panel = fig.add_subplot(grid_base[plotind][0,0], aspect=aspect)
	#

	#Plot a blank square if no emission given
	if plotblank is True:
		panel.tick_params(labelbottom=False, labelleft=False)
		panel.tick_params(width=0, size=0, labelsize=0)
		#Remove outline of blank square
		panel.spines["bottom"].set_color("white")
		panel.spines["left"].set_color("white")
		panel.spines["right"].set_color("white")
		panel.spines["top"].set_color("white")
		return

	#Change box color if different color given
	if plotoutlinecolor is not None:
		panel.spines["bottom"].set_color(plotoutlinecolor)
		panel.spines["left"].set_color(plotoutlinecolor)
		panel.spines["right"].set_color(plotoutlinecolor)
		panel.spines["top"].set_color(plotoutlinecolor)
		panel.spines["bottom"].set_linestyle(plotoutlinestyle)
		panel.spines["left"].set_linestyle(plotoutlinestyle)
		panel.spines["right"].set_linestyle(plotoutlinestyle)
		panel.spines["top"].set_linestyle(plotoutlinestyle)
		panel.spines["bottom"].set_linewidth(plotoutlinewidth)
		panel.spines["left"].set_linewidth(plotoutlinewidth)
		panel.spines["right"].set_linewidth(plotoutlinewidth)
		panel.spines["top"].set_linewidth(plotoutlinewidth)

	#Turn spectra characteristics into lists, if necessary
	if isinstance(linewidth, int) or isinstance(linewidth, float):
		linewidth = [linewidth]*len(arr_ys)
	if isinstance(markersizes, int) or isinstance(markersizes, float):
		markersizes = [markersizes]*len(arr_ys)
	if isinstance(alpha, int) or isinstance(alpha, float):
		alpha = [alpha]*len(arr_ys)
	if (isinstance(labels, str) or (labels is None)):
		labels = [labels]*len(arr_ys)
	if isinstance(markers, str):
		markers = [markers]*len(arr_ys)
	#
	if (legendfontsize is None):
		legendfontsize = ticksize
	#


	#Iterate through and plots each spectra
	for ai in range(0, len(arr_ys)):
		if (ai == ind_topline):
			zorder = 200
		else:
			zorder = None
		#
		#Plot current spectrum
		panel.plot(arr_xs[ai], arr_ys[ai], #drawstyle="steps-mid",
			color=colors[ai], label=labels[ai], marker=markers[ai],
			linewidth=linewidth[ai], alpha=alpha[ai], linestyle=styles[ai],
			markersize=markersizes[ai], zorder=zorder)
	#
	#Legend labels, if so desired
	if dolegend:
		leg = panel.legend(loc=legendloc, labelspacing=legendlabelspacing,
							columnspacing=legendcolspacing,
							handlelength=legendhandlelength,
			prop={"size":legendfontsize}, frameon=False)
		leg.get_frame().set_alpha(0.5)
	#

	#Plot in a line at zero, if so desired
	if dozeroline:
		plt.axhline(0, linewidth=1, color="black", linestyle="--")

	#Plot in x-axis lines, if so desired
	if doxlines:
		for ai in range(0, len(xlines)):
			plt.axvline(xlines[ai], linewidth=xlinewidth, color=xlinecol,
					linestyle=xlinesty)

	#Plot in y-axis lines, if so desired
	if doylines:
		for ai in range(0, len(ylines)):
			plt.axhline(ylines[ai], linewidth=ylinewidth, color=ylinecol,
					linestyle=ylinesty)

	#Plot in y-axis spans, if so desired
	if doyspans:
		for ai in range(0, len(yspans)):
			plt.axhspan(yspans[ai][0], yspans[ai][1], color=yspancol,
					alpha=yspanalpha)


	##Format axes and labels of subplot
	#Format plot ticks
	panel.locator_params(axis="x", nbins=nxticks) #Reduce # of x-ticks
	panel.tick_params(width=tickwidth, size=tickheight,
	 					labelsize=ticksize, direction="in")

	#Remove tick labels, if so desired
	if rmxticklabel: #If no bottom axis labels desired
		panel.tick_params(axis="x", which="both", labelbottom=False)
	if rmyticklabel: #If no left axis labels desired
		panel.tick_params(axis="y", which="both", labelleft=False)

	#Label axes, if so desired
	if not rmyticktitle:
		panel.set_ylabel(ylabel, fontsize=titlesize, labelpad=labelpad)
	if not rmxticktitle:
		panel.set_xlabel(xlabel, fontsize=titlesize, labelpad=labelpad)
	if title is not None:
		panel.set_title(title)
	#

	#Zoom in on subplot, if so desired
	if xlim is not None: #For x-axis range
		panel.set_xlim(xlim)
	else:
		if xmin is not None: #For bottom of y-axis range
			panel.set_xlim(xmin=xmin)
		if xmax is not None: #For bottom of y-axis range
			panel.set_xlim(xmax=xmax)
	if ylim is not None: #For y-axis range
		panel.set_ylim(ylim)
	elif ymin is not None: #For bottom of y-axis range
		panel.set_ylim(ymin=ymin)
	#

	#Add in in-graph labels, if so desired
	if boxtexts is not None:
		for ai in range(0, len(boxtexts)):
			panel.text(boxxs[ai], boxys[ai], boxtexts[ai],
				horizontalalignment="left", verticalalignment="top",
				bbox=dict(facecolor="white", alpha=boxalpha,
								edgecolor="white"),
				fontsize=textsize, transform=panel.transAxes)
	#

	#Set log scale, if requested
	if doxlog:
		panel.set_xscale("log")
	if doylog:
		panel.set_yscale("log")
	#

	#Return plot axis, if so desired
	if returnax:
		return panel
#




##------------------------------------------------------------------------------
##UTILS: USEFUL FILE GENERATION
##Method: make_file_array()
##Purpose: Generate and save contents of array to file
def make_file_array(arrays, headers, toplines, filepath, filename, mode_format=".3E", mode_comment="#"):
	#Initialize with header and top-line strings
	str_fin = ""
	if (len(headers) > 0):
		str_fin += (mode_comment + (("\n"+mode_comment).join(headers)) + "\n")
	if (len(toplines) > 0):
		str_fin += (("\n".join(toplines)) + "\n") #Add any toplines given
	#

	#Fill in array values
	num_arrays = len(arrays)
	num_points = np.unique([len(item) for item in arrays])
	if (len(num_points) != 1):
		raise ValueError("Whoa! Unequal array lengths: {0}".format(num_points))
	else:
		num_points = num_points[0]
	#
	#Iterate through points
	for ii in range(0, num_points):
		#Iterate through arrays
		for zz in range(0, num_arrays):
			#Tack on latest data
			str_fin += (("{0:"+mode_format+"}").format(arrays[zz][ii]))
			#Ending character
			if (zz < (num_arrays-1)):
				str_fin += "\t"
			else:
				str_fin += "\n"
		#
	#

	#Save the string to file
	_write_text(text=str_fin, filename=os.path.join(filepath, filename))
#




##------------------------------------------------------------------------------
##UTILS: USEFUL ROUTINES
#
##Method: smooth_spectrum()
##Purpose: Smooth given spectrum, including and excluding given wavelength ranges
def smooth_spectrum(wave_fin, wave_all, spec_all, window_phys, window_frac, poly_order, waveranges_exclude, do_keep_exclude, do_same_edges, do_plot=False, name_star=None):
    #Print some notes
    print("Running smooth_spectrum for {0}...".format(name_star))

    #Throw error if both physical and fractional window lengths specified
    if ((window_phys is not None) and (window_frac is not None)):
        raise ValueError("Whoa! Both physical and fractional lengths given!")
    #

    #Exclude any wavelength ranges as given
    fin_all = spec_all.copy()
    list_inds_snippet = [None]*len(waveranges_exclude)
    if (waveranges_exclude is not None):
        #Iterate through ranges to exclude
        for ii in range(0, len(waveranges_exclude)):
            curr_exclude = waveranges_exclude[ii]
            #Replace current range with linear extension across boundaries
            inds_snippet = ((wave_all >= curr_exclude[0])
                                & (wave_all <= curr_exclude[1]))
            fin_all[inds_snippet] = np.logspace(
                                np.log10(fin_all[inds_snippet][0]),
                                np.log10(fin_all[inds_snippet][-1]),
                                len(fin_all[inds_snippet]))
            #Store indices for later use
            list_inds_snippet[ii] = inds_snippet
        #
    #

    #Set window length based on given physical or fractional length
    if (window_phys is not None):
        window_length = window_phys
    elif (window_frac is not None):
        #window_length = int(np.ceil(window_frac * len(fin_photon_UV)))
        window_length = int(np.ceil(window_frac * len(wave_fin)))
        #Add one to make odd, if necessary
        if ((window_length % 2) == 0):
            window_length += 1
        #Add two to exceed poly_order, if necessary
        if (window_length == poly_order):
            window_length += 2
    #

    #Smooth over the data with a filter
    y_smooth = 10**scipy_signal.savgol_filter(np.log10(fin_all),
                            window_length=window_length, polyorder=poly_order)
    #

    #Replace the edges of the smoothed spectrum with the original, if requested
    if do_same_edges:
        #Match the edges of the spectrum
        y_smooth[0:2] = fin_all[0:2]
        y_smooth[-2:len(y_smooth)] = (
                    fin_all[-2:len(y_smooth)])
        #
    #

    #Put back in excluded portions, if requested
    if do_keep_exclude and (waveranges_exclude is not None):
        for ii in range(0, len(waveranges_exclude)):
            y_smooth[list_inds_snippet[ii]] = (
                                        spec_all[list_inds_snippet[ii]])
    #

    #Interpolate smoothed spectra to be over final wavelengths
    y_fin = scipy_interp1d(x=wave_all, y=y_smooth)(wave_fin)
    #

    #Plot the data as a check, if so desired
    if do_plot: # and (name_star is not None):
        plt.close()
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(1, 1, 1)
        ax0.set_title("{0}: poly={3}\nwindow_frac={1}, window_phys={2}"
                    .format(name_star, window_frac, window_phys, poly_order))
        ax0.plot(wave_fin*1E9, y_fin, color="crimson", alpha=1,
                    linewidth=2)
        ax0.plot(wave_all*1E9, y_smooth, color="dodgerblue", alpha=1,
                    linewidth=2)
        ax0.plot(wave_all*1E9, spec_all, color="black", alpha=0.5,
                    linewidth=4)
        for ii in range(0, len(waveranges_exclude)):
            ax0.axvspan(waveranges_exclude[ii][0]*1E9,
                    waveranges_exclude[ii][1]*1E9, alpha=0.5, color="gray")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        plt.show()
    #

    #Return the fitted continuum
    return {"x":wave_fin, "y":y_fin}
#

##Method: split_spectrum_into_emandcont()
##Purpose: Split a spectrum into emission and continuum within given splitting range
def split_spectrum_into_emandcont(range_split, spectrum_orig, wave_cont, do_plot, window_phys, window_frac, poly_order, waveranges_exclude, do_keep_exclude, do_same_edges, name_star=None):
	#Extract emission wavelengths using given splitting range
	wave_em = wave_cont[((wave_cont >= range_split[0])
	 					& (wave_cont <= range_split[1]))]
	#

	#Smooth the given spectrum to estimate the continuum
	tmp_res = smooth_spectrum(wave_fin=wave_cont, wave_all=wave_cont,
			spec_all=spectrum_orig, window_phys=window_phys,
			window_frac=window_frac,
			poly_order=poly_order, waveranges_exclude=waveranges_exclude,
			do_keep_exclude=do_keep_exclude, do_same_edges=do_same_edges,
			do_plot=do_plot, name_star=name_star)
	#
	spectrum_full_smoothed = tmp_res["y"]
	#

	#Subtract smoothed spectrum from the original to estimate emission spectrum
	subtracted_spectrum = (spectrum_orig - spectrum_full_smoothed)

	#Zero out negative emission
	subtracted_spectrum[subtracted_spectrum < 0] = 0
	#

	#Split out the estimated emission within emission wavelength range
	spectrum_snippet_em = scipy_interp1d(y=subtracted_spectrum, x=wave_cont
												)(wave_em)
	#

	#Replace emission section in copy of original spectrum with smoothed version
	tmp_inds = ((wave_cont >= wave_em.min()) & (wave_cont <= wave_em.max()))
	spectrum_full_updated = spectrum_orig.copy()
	spectrum_full_updated[tmp_inds] = spectrum_full_smoothed[tmp_inds]
	#

	#Return the dictionary of results
	return {"spectrum_updated":spectrum_full_updated,
	 		"spectrum_em":spectrum_snippet_em, "wave_em":wave_em,
			"_spectrum_smoothed":spectrum_full_smoothed}
#




##------------------------------------------------------------------------------
##UTILS: PARALLELIZATION (NAUTILUS)
#
##Method: _conv_reactdict_to_reactstr()
##Purpose: Convert reaction-dictionary to reaction-string
def _conv_reactdict_to_reactstr(react_dict, mode):
	if (mode in ["nautilus"]):
		##Load global variables
		span_id = preset.pnautilus_text_len_id
		span_tempeach = preset.pnautilus_text_len_tempeach
		span_itype = preset.pnautilus_text_len_itype
		span_coeffseach = preset.pnautilus_text_len_coeffseach
		span_formula = preset.pnautilus_text_len_formula
		span_producteach = preset.pnautilus_text_len_producteach
		span_productall = preset.pnautilus_text_len_productall
		span_reactanteach = preset.pnautilus_text_len_reactanteach
		span_reactantall = preset.pnautilus_text_len_reactantall
		span_lineall = preset.pnautilus_text_len_lineall
		span_blank1 = preset.pnautilus_text_len_blank1
		thres_coeff0 = preset.pnautilus_threshold_coeff0

		##Write out reaction components
		new_str = "" #Initialize
		#Write out reactants
		tmp_str = ""
		for ii in range(0, len(react_dict["reactants"])):
			#Tack on current reactant
			tmp_str += react_dict["reactants"][ii].ljust(span_reactanteach)
		new_str += tmp_str.ljust(span_reactantall)
		#
		#Write out products
		tmp_str = ""
		for ii in range(0, len(react_dict["products"])):
			#Tack on current product
			tmp_str += react_dict["products"][ii].ljust(span_producteach)
		new_str += tmp_str.ljust(span_productall)
		#
		#Write out coefficients
		for ii in range(0, len(react_dict["coeffs"])):
			#Tack on current coefficient
			new_str += (
				f"{react_dict['coeffs'][ii]:.3e}".rjust(span_coeffseach)
			)
		#
		#Add in blank portion
		new_str += " ".ljust(span_blank1)
		#Write out itype
		new_str += str(int(react_dict["itype"])).rjust(span_itype)
		#
		#Write out temperature range
		tmp_temp = [react_dict["Tmin"], react_dict["Tmax"]]
		for ii in range(0, len(tmp_temp)):
			#Tack on current temperature
			new_str += str(int(tmp_temp[ii])).rjust(span_tempeach)
		#
		#Write out formula
		new_str += str(int(react_dict["formula"])).rjust(span_formula)
		#
		#Write out id
		new_str += str(int(react_dict["react_id"])).rjust(span_id)
		#

		##Pad out to end of string
		new_str = new_str.ljust(span_lineall)

		##Add any notes to end of string, if necessary
		if "note" in react_dict:
			new_str += "! ".format(react_dict["note"])

		##Raise error if alpha coeff. is below threshold or any nan coeffs
		if (react_dict["coeffs"][0] < thres_coeff0): #For very low alpha
			raise ValueError("Err: alpha_coeff below threshold {0:.3e}: {1}"
						.format(thres_coeff0, new_str))
		#
		elif any(np.isnan(react_dict["coeffs"])): #For nan coeffs.
			raise ValueError("Whoa! Reaction has nan coefficients:\n{0}"
							.format(react_dict))
		#
	#
	#Throw error if mode not recognized
	else:
		raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
	#

	##Return the string
	return new_str
#


##Method: _conv_reactstr_to_reactdict()
##Purpose: Convert reaction-string to reaction-dictionary
def _conv_reactstr_to_reactdict(line, mode):
	#For Nautilus reaction strings
	if (mode in ["pnautilus", "nautilus"]):
		#Load global variables
		span_id = preset.pnautilus_text_span_id
		span_temp = preset.pnautilus_text_span_temp
		span_itype = preset.pnautilus_text_span_itype
		span_coeffs = preset.pnautilus_text_span_coeffs
		span_formula = preset.pnautilus_text_span_formula
		span_products = preset.pnautilus_text_span_product
		span_reactants = preset.pnautilus_text_span_reactant
		#

		#Initialize holder for this reaction
		curr_id = int((line[span_id[0]:span_id[1]]).strip())
		res_dict = {"react_id":curr_id}
		#Process the reactants
		res_dict["reactants"] = line[
							span_reactants[0]:span_reactants[1]].split()
		#Process the products
		res_dict["products"] = line[
							span_products[0]:span_products[1]].split()
		#Tack in the probability of this reaction - always 1 originally
		res_dict["prob_product"] = 1.0
		#Process the reaction
		res_dict["reaction"] = ("["+" + ".join(sorted(res_dict["reactants"]))
								+" -> "
								+" + ".join(sorted(res_dict["products"]))+"]")
		#Process the minimum temperature
		curr_segment = float(line[span_temp[0][0]:span_temp[0][1]])
		if float(curr_segment) != int(curr_segment): #Raise error if not integer
			raise ValueError("Whoa! Float temperature?\n{0} from line:\n{1}"
								.format(curr_segment, line))
		res_dict["Tmin"] = int(curr_segment)
		#Process the maximum temperature
		curr_segment = float(line[span_temp[1][0]:span_temp[1][1]])
		if float(curr_segment) != int(curr_segment): #Raise error if not integer
			raise ValueError("Whoa! Float temperature?\n{0} from line:\n{1}"
								.format(curr_segment, line))
		res_dict["Tmax"] = int(curr_segment)
		#Process the formula codes
		res_dict["formula"] = int(line[span_formula[0]:span_formula[1]])
		res_dict["itype"] = line[span_itype[0]:span_itype[1]].strip()

		#Infer the phototype, if applicable
		#For photoionization or photodissociation
		if any([(item in [preset.chemistry_reactant_withphotochemUV,
								preset.chemistry_reactant_withphotochemX])
								for item in res_dict["reactants"]]):
			#Fetch elemental breakdowns
			tmp_e1 = _get_elements(mol=
						[item for item in res_dict["reactants"]
						if (item not in
							[preset.chemistry_reactant_withphotochemUV,
								preset.chemistry_reactant_withphotochemX])],
						do_ignore_JK=False)
			tmp_e2 = _get_elements(mol=res_dict["products"],
									do_ignore_JK=False)
			#Infer the phototype
			#For photoionization
			if (preset.chemistry_product_withphotoion in res_dict["products"]):
				res_dict["phototype"] = "photoion" #Photoionization
			#For photodissociation
			elif (tmp_e1 == tmp_e2):
				res_dict["phototype"] = "photodiss" #Photodissociation
			#Otherwise, throw error if not recognized reaction
			else:
				raise ValueError("Whoa! Not recognized:\n{0}".format(line)
								+"\nReact. dict so far: {0}\n".format(res_dict)
								+"Reactant elements: {0}\nProduct elements: {1}"
									.format(tmp_e1, tmp_e2))
		#
		#Otherwise, set blank if not a photochemical reaction
		else:
			res_dict["phototype"] = None
		#

		#Take note of main reactant for photochemistry reactions
		if (res_dict["phototype"] in ["photoion", "photodiss"]):
			reactant_main_raw = [item for item in res_dict["reactants"]
							if (item not in
							[preset.chemistry_reactant_withphotochemUV,
							preset.chemistry_reactant_withphotochemX,
							preset.chemistry_reactant_withcosmicray])]
			#
			#Throw error if more than one reactant extracted
			if (len(reactant_main_raw) != 1):
				raise ValueError("Whoa! Unexpected reactants?\n{0}"
								.format(reactant_main_raw))
			#
			#Else, fetch representative reaction
			reactant_main = reactant_main_raw[0]
			res_dict["reactant_main"] = reactant_main
		#
		else:
			res_dict["reactant_main"] = None
		#

		#Process the coefficients
		res_dict["coeffs"] = []
		for curr_span in span_coeffs:
			curr_segment = float(line[curr_span[0]:curr_span[1]])
			if curr_segment != "":
				res_dict["coeffs"].append(curr_segment)
	#
	#Throw error if mode not recognized
	else:
		raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
	#

	#Return the assembled reaction dictionary
	return res_dict
#


##Method: _run_pnautilus_diskslice_1D()
##Purpose: Run pnautilus for given 1D slice within model
def _run_pnautilus_diskslice_1D(dicts_perslice):
	##Prepare processes for this timestep
	tmp_key = list(dicts_perslice.keys())[0]
	do_verbose = dicts_perslice[tmp_key]["do_verbose"]
	num_cores = dicts_perslice[tmp_key]["num_cores"]
	num_slices = dicts_perslice[tmp_key]["num_slices"]
	dir_orig = dicts_perslice[tmp_key]["dir_orig"]
	dir_work = os.path.join(dir_orig, preset.pnautilus_disk_dir_working)
	dir_done = os.path.join(dir_orig, preset.pnautilus_disk_dir_done)
	dir_discard = os.path.join(dir_orig, preset.pnautilus_disk_dir_discard)
	dir_processed_output = os.path.join(dir_orig,
	 									preset.pnautilus_processed_output)
	dir_trash_global = dicts_perslice[tmp_key]["filepath_trash_global"]
	#
	#Print some notes
	if do_verbose:
		print("Running _run_pnautilus_diskslice_1D!")
		print("Generating working, complete, and discard directories...")
	#

	#Move any previous working, complete, discard directories to discard
	str_timestamp = (str(datetime.datetime.now()).split(".")[0]
					.replace(" ","_").replace(":","_").replace("-","_"))
	#For working directory
	if os.path.exists(dir_work):
		tmp_name = (preset.pnautilus_disk_dir_working
		 			+ "_" + str_timestamp + "/")
		comm = subprocess.call(["mv", dir_work, os.path.join(dir_trash_global,
																tmp_name)])
		if comm != 0:
			raise ValueError("Whoa! Above error returned from mv for:\n{0}\n{1}"
							.format(dir_work, dir_trash_global))
	#
	#For complete directory
	if os.path.exists(dir_done):
		tmp_name = (preset.pnautilus_disk_dir_done
		 			+ "_" + str_timestamp)
		comm = subprocess.call(["mv", dir_done, os.path.join(dir_trash_global,
																tmp_name)])
		if comm != 0:
			raise ValueError("Whoa! Above error returned from mv for:\n{0}\n{1}"
							.format(dir_done, dir_trash_global))
	#
	#For local discard directory
	if os.path.exists(dir_discard):
		tmp_name = (preset.pnautilus_disk_dir_discard
		 			+ "_" + str_timestamp)
		comm = subprocess.call(["mv", dir_discard,os.path.join(dir_trash_global,
																tmp_name)])
		if comm != 0:
			raise ValueError("Whoa! Above error returned from mv for:\n{0}\n{1}"
							.format(dir_discard, dir_trash_global))
	#
	#For processed output directory
	if os.path.exists(dir_processed_output):
		tmp_name = (preset.pnautilus_processed_output
		 			+ "_" + str_timestamp)
		comm = subprocess.call(["mv", dir_processed_output,
								os.path.join(dir_trash_global, tmp_name)])
		if comm != 0:
			raise ValueError("Whoa! Above error returned from mv for:\n{0}\n{1}"
							.format(dir_processed_output, dir_trash_global))
	#

	#Generate working, complete, and discard directories
	#For working directory
	comm = subprocess.call(["mkdir", dir_work]) #Make new working subdir.
	if comm != 0:
		raise ValueError("Whoa! Above error returned from mkdir for:\n{0}"
						.format(dir_work))
	#
	#For complete directory
	comm = subprocess.call(["mkdir", dir_done]) #Make new complete subdir.
	if comm != 0:
		raise ValueError("Whoa! Above error returned from mkdir for:\n{0}"
						.format(dir_done))
	#
	#For discard directory
	comm = subprocess.call(["mkdir", dir_discard]) #Make new discard subdir.
	if comm != 0:
		raise ValueError("Whoa! Above error returned from mkdir for:\n{0}"
						.format(dir_discard))
	#
	#For processed output later
	comm = subprocess.call(["mkdir", dir_processed_output])
	if comm != 0:
		raise ValueError("Whoa! Above error returned from mkdir for:\n{0}"
						.format(dir_processed_output))
	#

	#Print some notes
	if do_verbose:
		print("Running each slice for {0} total slices...".format(num_slices))
	#


	time_start = time.time()
	##Run method to operate pnautilus+outputs for each 1D slice within disk
	#Serial/Linear (non-parallelized) version
	if (num_cores == 1):
		for i_slice in range(0, num_slices):
			#Run chemistry for this 1D slice
			_core_pnautilus_diskslice_1D(i_slice=i_slice,
			 					dict_slice=dicts_perslice[i_slice])
	#
	#Parallelized version
	elif (num_cores > 1):
		#Consolidate inputs
		core_inputs = [(i_slice, dicts_perslice[i_slice])
						for i_slice in range(0, num_slices)]

		#Operate each 1D slice using pool of processes
		with mp.Pool(processes=num_cores) as pool:
			if do_verbose:
				print("Pool of {0} cores activated.".format(num_cores))
				print("Computing chemistry per 1D slice...")
			#
			#Compute chemistry per slice with pool of processes
			comm = pool.starmap(_core_pnautilus_diskslice_1D, core_inputs)
			#
			if do_verbose:
				print(comm)
				print("Chemistry per slice complete.")
			#
		#
		if do_verbose:
			print(comm)
			print("Pool of {0} cores now deactivated.".format(num_cores))
	#
	#Otherwise, throw error if invalid number of cores
	else:
		raise ValueError("Whoa! Invalid number of cores given: {0}"
						.format(num_cores))
	#
	time_end = time.time() - time_start
	#Print some notes
	if do_verbose:
		print("All {1} 1D slices complete in {1:.2f} minutes."
				.format(num_slices, (time_end/60)))
	#


	##Exit the method
	return
#


##Method: _core_pnautilus_diskslice_1D()
##Purpose: Run pnautilus for given 1D slice within model
def _core_pnautilus_diskslice_1D(i_slice, dict_slice):
	##Load global variables
	time_start_slice = time.time() #Start timer
	do_verbose = dict_slice["do_verbose"]
	dir_orig = dict_slice["dir_orig"]
	ind_xs = dict_slice["ind_xs"]
	ind_ys = dict_slice["ind_ys"]
	filename_init_abunds = dict_slice["filename_init_abunds"]
	max_allowed_time = preset.pnautilus_process_maxtime
	#
	num_slices = dict_slice["num_slices"]
	ext_slice_timeout = preset.pnautilus_ext_timeout
	#
	if len(mp.current_process()._identity) > 0:
		id_core = mp.current_process()._identity[0]
		is_parallel = True
	else:
		id_core = os.getpid()
		is_parallel = False
	#
	#Print some notes
	if do_verbose:
		print("- Prepping pnautilus for 1D slice {0} of {1}. Parallel={2}."
				.format(i_slice, num_slices-1, is_parallel))
		print("- y-pos: {0}\nx-pos: {1}."
				.format(ind_ys, ind_xs))
	#


	##Prepare directories for this slice at this timestep
	ext_slice_pos = "_i{0}".format(i_slice)
	ext_slice_work = ext_slice_pos #"_{0}".format(id_core)
	dir_done = os.path.join(dir_orig, preset.pnautilus_disk_dir_done)
	dir_work = os.path.join(dir_orig, preset.pnautilus_disk_dir_working)
	dir_discard = os.path.join(dir_orig, preset.pnautilus_disk_dir_discard)
	dir_work_slice = os.path.join(dir_work, ("disk_work"+ext_slice_work))
	dir_work_reactions = os.path.join(dir_work_slice, preset.dir_reactions_base)
	dir_done_slice = os.path.join(dir_done, ("disk_done"+ext_slice_pos))
	#
	#Skip if this slice has already been computed before
	if os.path.exists(dir_done_slice):
		#Print some notes
		if do_verbose:
			print_str = ("Slice {0} already in completed folder:\n{1}."
					.format(i_slice, dir_done_slice))
			print_str += ("\nWill not redo/overwrite. Exiting this slice.")
			print(print_str)
		#Skip this slice
		return
	#
	#Make the slice working directory
	comm = subprocess.call(["mkdir", dir_work_slice]) #Make new subdir.
	#Throw an error if subprocess returned an error
	if comm != 0:
		raise ValueError("Whoa! Above error returned from mkdir for:\n{0}"
						.format(dir_work_slice))
	#
	#Copy over base files
	_copy_pnautilus_filebase(dir_old=dir_orig, dir_new=dir_work_slice,
							which_phase="disk",
							filename_init_abunds=filename_init_abunds)
	#
	#Copy over reaction files for this slice
	dir_old_reactions = os.path.join(dir_orig, preset.dir_reactions_base,
						preset.dir_reactions_formatperslice.format(i_slice))
	comm = subprocess.call(["mkdir",  dir_work_reactions])
	if comm != 0:
		raise ValueError("Whoa! Above error returned from Nautilus!")
	#
	comm = subprocess.call(["cp", "-r",  (dir_old_reactions+"/."),
							(dir_work_reactions+"/")])
	if comm != 0:
		raise ValueError("Whoa! Above error returned from Nautilus!")
	#
	#Print some notes
	if do_verbose:
		print_str = ("Dir. has been prepared for slice {0}."
				.format(i_slice))
		print_str += ("\nBaseline pnautilus filebase was copied over.")
		print_str += ("\nReaction file dir. for this slice was copied over.")
		print_str += ("\nInitial abundances also renamed and copied over.")
		print_str += ("\nWriting input files next...\n")
		print(print_str)
	#


	##Prepare input files for this disk slice
	filename_strip_param = os.path.join(dir_work_slice,
								preset.pnautilus_filename_parameterstrip)
	filename_new_param = os.path.join(dir_work_slice,
	 							preset.pnautilus_filename_parameter)
	filename_new_static = os.path.join(dir_work_slice,
	 							preset.pnautilus_filename_static1D)
	#Update stripped parameter file for this slice
	_write_input_parameters_slice1D(filename_strip=filename_strip_param,
 							filename_new=filename_new_param,
							i_slice=i_slice, dict_slice=dict_slice)
	#Write static per-cell file for this slice
	_write_input_static_slice1D(filename=filename_new_static,
							i_slice=i_slice, dict_slice=dict_slice)
	#

	#Print some notes
	if do_verbose:
		print_str = ("Input params. written for slice {0}.".format(i_slice))
		print_str += ("\nRunning pnautilus for this slice...\n")
		print(print_str)
	#


	##Run Nautilus for this cell
	time_start_process = time.time() #Start a timer
	try:
		runner = subprocess.Popen([preset.pnautilus_command_main],
			cwd=dir_work_slice, stdout=subprocess.PIPE) #Run nautilus
		#Terminate early if NaNs in output
		do_halt = False
		for section in iter(runner.stdout.readline, b''):
			#Fetch, decode, and print latest line of output to terminal
			tmp_text = section.decode(sys.stdout.encoding)
			sys.stdout.write(tmp_text)
			#
			#Stop if this output contains any nans
			if ("R1=NaN" in tmp_text.replace(" ","")):
				print("Nans in pnautilus output for Slice {0}. Terminating..."
						.format(i_slice))
				do_halt = True
				break
			#
			#Raise an error if timer has expired
			#This is probably a poor way to do this, very bad form, but ah wells
			if ((time.time() - time_start_process) > max_allowed_time):
				print("Time exceeded. Exiting.")
				raise RuntimeError()
				print("***MAJOR ERROR: THIS STATEMENT SHOULD NOT PRINT.***")
		#
		if do_halt:
			runner.kill() #What a gruesome method name
			outs, errs = runner.communicate()
			print("Slice {0} terminated.".format(i_slice))
			return
		#
	#
	except RuntimeError:
		if is_parallel: #Return empty if parallel operation
			print("\n-\nSlice {0} took too long.\n-\n".format(i_slice))
			return
			print("***MAJOR ERROR: THIS STATEMENT SHOULD NOT PRINT.***")
		else: #Terminate entire operation if linear operation
			raise ValueError("\nSlice {0} took too long.\n-\n".format(i_slice))
			print("***MAJOR ERROR: THIS STATEMENT SHOULD NOT PRINT.***")
	#
	#Print some notes
	if do_verbose:
		print_str = ("Pnautilus run complete for slice {0}.".format(i_slice))
		print_str += ("\nProcessing outputs and cleaning up folders...\n")
		print(print_str)
	#


	##Process Nautilus outputs for this cell
	comm = subprocess.call([preset.pnautilus_command_output],
	 						cwd=dir_work_slice)
	#Throw an error if subprocess returned an error
	if comm != 0:
		raise ValueError("Whoa! Above error returned from Nautilus!")
	#
	#Remove cluttery .out files
	rmlist = (glob.glob(os.path.join(dir_work_slice, "abundances*out"))
				+ glob.glob(os.path.join(dir_work_slice, "col_dens*out"))
				+ glob.glob(os.path.join(dir_work_slice, "rates*out")))
	for tmpfile in rmlist:
		os.remove(tmpfile)
	#
	#Remove hefty copies of reaction files
	rmlist = [os.path.join(dir_work_reactions, item)
	 			for item in os.listdir(dir_work_reactions)]
	for tmpfile in rmlist:
		os.remove(tmpfile)
	#
	#Print some notes
	if do_verbose:
		print_str = ("Pnautilus output complete for slice {0}.".format(i_slice))
		print_str += ("\nMoving output of this slice into completed files...\n")
		print(print_str)
	#


	##Move slice output chemistry files to folder of completed output
	comm = subprocess.call(["mv", dir_work_slice, dir_done_slice])
	#Throw an error if subprocess returned an error
	if comm != 0:
		raise ValueError("Whoa! Above error returned from cp!")
	#
	#Print some notes
	if do_verbose:
		print_str = ("Output of slice {0} moved to completed files."
					.format(i_slice))
		print(print_str)
	#


	##Exit the method
	time_end_slice = (time.time() - time_start_slice)
	if do_verbose:
		print_str = ("Disk slice {0} of {1} complete in {2:.2f} minutes!"
					.format(i_slice, num_slices, (time_end_slice/60)))
		print_str += ("\nExiting the slice...\n")
		print(print_str)
	#
	return
#


##Method: _copy_pnautilus_filebase()
##Purpose: Copy over base files from old to new directory
def _copy_pnautilus_filebase(dir_old, dir_new, which_phase, filename_init_abunds):
	##Prepare and copy over base files from old to new directory
	if (which_phase == "disk"):
		copyfiles = preset.pnautilus_filebase_disk
	else:
		raise ValueError("Whoa! Requested phase {0} not recognized!"
							.format(which_phase))
	#

	##Copy over files
	for filename in copyfiles: #Copy over all unchanging files
		comm = subprocess.call(["cp", os.path.join(dir_old, filename),
								dir_new+"/"])
		#Throw an error if subprocess returned an error
		if comm != 0:
			raise ValueError("Whoa! Above error returned from cp for:\n{0}\n{1}"
							.format(os.path.join(dir_old, filename), dir_new))
	#

	##Copy over initial abundance file into new name in new directory
	comm = subprocess.call(["cp", os.path.join(dir_old, filename_init_abunds),
									os.path.join(dir_new,
									 	preset.pnautilus_filename_abundances)])
	#Throw an error if subprocess returned an error
	if comm != 0:
		raise ValueError("Whoa! Above error returned from cp for:\n{0}\n{1}"
						.format(os.path.join(dir_old, filename_init_abunds),
						 		dir_new))
	#

	##Exit the method
	return
#


##Method: _write_input_parameters_slice1D()
##Purpose: Write parameters.in file for 1D slice
def _write_input_parameters_slice1D(i_slice, dict_slice, filename_strip, filename_new):
	##Load instance variables
	do_verbose = dict_slice["do_verbose"]
	num_points = dict_slice["num_points"]
	time_start_yr = dict_slice["time_start_yr"]
	time_end_yr = dict_slice["time_end_yr"]
	num_time = dict_slice["num_time"]
	#
	flux_UV_uDraine = 0.0
	ionrate_CR = dict_slice["ionrate_CR"]
	frac_dustovergasmass = dict_slice["frac_dustovergasmass"]
	dens_pergrain_cgs = dict_slice["densmass_grain_cgs"]
	radius_grain_cgs = dict_slice["radius_grain_cgs"]
	#

	##Zero out any input values that are below threshold
	thres_minval = preset.pnautilus_threshold_minval
	#For integrated UV flux
	if (flux_UV_uDraine < thres_minval):
		if do_verbose:
			print("UV flux [uDraine] is tiny ({0})."
					.format(flux_UV_uDraine))
		#Zero out the value
		flux_UV_uDraine = 0
		if do_verbose:
			print("UV flux zeroed out to {0}.".format(flux_UV_uDraine))
		#
	#

	##Read in the stripped version of the parameters file
	with open(filename_strip, 'r') as openfile:
		text = openfile.read()

	##Update the stripped text of the parameters file
	text += "\nspatial_resolution = {0} ! \n".format(num_points) # # col. points
	text += "structure_type = 1D_no_diff ! \n" #1D (1D_no_diff)
	text += "is_dust_1D = 0 ! \n" #0 for constaint grain parameters=0D, 1 for 1D
	text += "grain_temperature_type = table_1D ! \n" #'table_1D' = 1D_static.dat
	#
	text += "uv_flux = {0:.3E} ! \n".format(flux_UV_uDraine) #Ref. unit
	text += "cr_ionisation_rate = {0:.3E} ! \n".format(ionrate_CR) #1/s
	#
	text += "initial_dtg_mass_ratio = {0:.3E} ! \n".format(
													frac_dustovergasmass)
	text += "grain_density = {0:.3E} ! \n".format(dens_pergrain_cgs) #g/cm3
	text += "grain_radius = {0:.3E} ! \n".format(radius_grain_cgs) #cm
	#
	text += "start_time = {0:.3E} ! \n".format(time_start_yr) #yr
	text += "stop_time = {0:.3E} ! \n".format(time_end_yr) #yr
	text += "nb_outputs = {0:d} ! \n".format(num_time) #Number
	text += "output_type = {0} ! \n".format("log") #log|linear
	#

	##Write the updated text to a new file
	_write_text(text=text, filename=filename_new)

	##Exit the method
	return
#


##Method: _write_input_static_slice1D()
##Purpose: Write 1D_static.dat file for 1D slice
def _write_input_static_slice1D(i_slice, dict_slice, filename):
	##Load instance variables
	do_verbose = dict_slice["do_verbose"]
	num_points = dict_slice["num_points"]
	thres_minval = preset.pnautilus_threshold_minval
	#
	distance_au = (dict_slice["arr_dist_slice"] / au0) #Distance along slice
	dens_gas_cgs = (dict_slice["arr_volnumdens_nH"]
	 					* conv_perm3topercm3) #[1/m^3] -> [1/cm^-3]
	temp_gas = dict_slice["arr_tempgas"] #[K]
	temp_dust = dict_slice["arr_tempdust"] #[K]
	extinction = dict_slice["arr_extinction"] #[mag]
	avnh_conv_factor_cgs = dict_slice["AVoverNHconv_cgs"]
	#
	#Below dust parameters no longer used! Since is_dust_1D -> 0 in param. file!
	gr_numdens_inv_cgs = np.zeros(shape=extinction.shape)
	radius_gr_cgs = 0.0
	#
	flux_UVtot_uDraine = dict_slice["arr_flux_UV_uDraine"].copy() #[uDraine]
	flux_UVtot_uDraine[flux_UVtot_uDraine<=thres_minval] = 0.0#Flatten small val.
	#
	ionrate_xray = dict_slice["arr_ionrate_X"] #[s^-1]
	#
	len_char = preset.pnautilus_lenchar_1Dstatic #Char. col. len. in 1D_static

	##Zero out any input values that are below threshold
	temp_gas[temp_gas < thres_minval] = 0
	temp_dust[temp_dust < thres_minval] = 0
	extinction[extinction < thres_minval] = 0
	dens_gas_cgs[dens_gas_cgs < thres_minval] = 0
	#

	##Generate string to hold contents of 1D static file
	text = preset.pnautilus_header_1Dstatic
	#Iterate through 1D slices
	for ii in range(0, num_points):
		text += "{0:.5e}".format(distance_au[ii]).ljust(len_char) #Distance
		text += "{0:.5e}".format(dens_gas_cgs[ii]).ljust(len_char) #H gas dens.
		text += "{0:.5e}".format(temp_gas[ii]).ljust(len_char) #Gas temp.
		text += "{0:.5e}".format(extinction[ii]).ljust(len_char) #Extinction
		text += "{0:.5e}".format(0).ljust(len_char) #Diffusion coeff.; not used
		text += "{0:.5e}".format(temp_dust[ii]).ljust(len_char) #Dust temp.
		text += "{0:.5e}".format(gr_numdens_inv_cgs[ii]
								).ljust(len_char)#Inv. gr.ab.
		text += "{0:.5e}".format(avnh_conv_factor_cgs).ljust(len_char)#AV/NH fac
		text += "{0:.5e}".format(radius_gr_cgs).ljust(len_char)#Radius of grains
		text += "{0:.5e}".format(flux_UVtot_uDraine[ii]
								).ljust(len_char) #Cell UV flux [uDraine]
		text += "{0:.5e}".format(ionrate_xray[ii]) #Cell x-ray ion. rate [s^-1]
		text += "\n" #New line
	#

	##Write the updated text to a new file
	_write_text(text=text, filename=filename)

	##Exit the method
	return
#





#


##------------------------------------------------------------------------------
#






#












#
