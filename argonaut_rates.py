###FILE: argonaut_rates.py
###PURPOSE: Script for computing parameters that represent the photochemical ultraviolet rate coefficients for molecules as a function of a given extincted spectrum.
###DATE FILE CREATED: 2023-03-01
###DEVELOPERS: (Jamila Pegues)


##Import necessary modules
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argonaut_math as math
import argonaut_utils as utils
import argonaut_constants as preset
plt.close()
#




##------------------------------------------------------------------------------
##REACTION PROCESSING
#
##Function: _extract_subset_reactions()
##Purpose: Extract subset of reactions from larger reaction dictionary.
def _extract_subset_reactions(mode, dict_reactions_gas, dict_reactions_gr):
    ##Extract subset of reactions according to requested mode
    #For UV photochemistry reactions
    if (mode == "UV"):
        dict_subset = {key:dict_reactions_gas[key]
                        for key in dict_reactions_gas
                        if (dict_reactions_gas[key]["formula"]
                            in preset.chem_formula_ids_UV)}
    #
    #Otherwise, throw error if mode not recognized
    else:
        raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
    #

    ##Return the subset dictionary
    return dict_subset
#

##Function: _fetch_all_species()
##Purpose: Fetch all species from given species lists.
def _fetch_all_species(filename_species_gas, filename_species_grain, do_remove_phase_marker):
    #Load the file data
    list_species_gas_raw = np.genfromtxt(filename_species_gas, delimiter="",
                                    comments="#", dtype=str)[:,0]
    list_species_grain_raw = np.genfromtxt(filename_species_grain, delimiter="",
                                    comments="#", dtype=str)[:,0]
    #Convert into list of species names
    list_species_all_raw = (np.unique(np.concatenate((list_species_gas_raw,
                                        list_species_grain_raw)))).tolist()
    #
    #Remove phase marker from species lists, if so requested
    if do_remove_phase_marker:
        list_species_gas = np.unique([re.sub("^(J|K)","",item)
                                    for item in list_species_gas_raw])
        list_species_grain = np.unique([re.sub("^(J|K)","",item)
                                    for item in list_species_grain_raw])
        list_species_all = np.unique([re.sub("^(J|K)","",item)
                                    for item in list_species_all_raw])
    else: #Otherwise just copy over
        list_species_gas = list_species_gas_raw
        list_species_grain = list_species_grain_raw
        list_species_all = list_species_all_raw
    #
    #Return the extracted species lists
    return {"gas":list_species_gas, "grain":list_species_grain,
            "all":list_species_all}
#

##Function: _fetch_updatable_reactions()
##Purpose: Fetch reactions from within a reaction dictionary that have cross-section data available for updating.
def _fetch_updatable_reactions(mode, dict_mol_UV, dict_reactions_orig, do_verbose=False):
    dict_updatable = {}
    for curr_key in dict_reactions_orig:
        curr_dict = dict_reactions_orig[curr_key]
        curr_mol = curr_dict["reactant_main"]
        curr_phototype = curr_dict["phototype"]
        #Extract current molecule for this mode of reaction
        #For UV reactions
        if (mode == "UV"):
            if (curr_mol is None):
                raise ValueError("Whoa! Invalid photochem. reaction!\n{0}"
                                .format(curr_dict))
        #
        #Otherwise, throw error if not recognized
        else:
            raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
        #

        #For UV mode:
        #Check if UV cross-sec data exists
        if (mode == "UV"):
            #Determine if any mol. data exists for this molecule
            if ((curr_mol not in dict_mol_UV) or
                    ((curr_mol in preset.special_species_names) and
                    (preset.special_species_names[curr_mol]
                        not in dict_mol_UV))):
                #Print some notes
                if do_verbose:
                    print("No molecular cross-section data for reactants: {0}"
                            .format(curr_dict["reactants"])+". Skipping.")
                #Skip ahead
                continue
            #

            #Determine if cross-sec. data exists for this molecule+phototype
            if (dict_mol_UV[curr_mol]["cross_{0}_{1}".format(curr_phototype,
                                    mode)] is not None):
                #Print some notes
                if do_verbose:
                    print("Molecular UV cross-section exists for reactants: {0}"
                            .format(curr_dict["reactants"])+". Keeping.")
                #Store this molecular info
                dict_updatable[curr_key] = {
                            "wave":dict_mol_UV[curr_mol][
                                "wave_{0}_{1}".format(curr_phototype, mode)],
                            "cross":dict_mol_UV[curr_mol][
                                "cross_{0}_{1}".format(curr_phototype, mode)]}
                #
                #Continue onward
                continue
            #
            #Otherwise, skip ahead
            else:
                #Print some notes
                if do_verbose:
                    print("No molecular cross-section data for reactants: {0}"
                            .format(curr_dict["reactants"])+". Skipping.")
                #Skip ahead
                continue
            #
        #
        #Otherwise, throw an error if mode not recognized
        else:
            raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
        #
    #

    #Return the compiled dictionary
    return dict_updatable
#

##Function: generate_dict_reactions()
##Purpose: Generate dictionary of reactions from input reactions files.
def generate_dict_reactions(mode, filepath_chemistry, filepath_save, do_verbose=False):
    ##Load instance variables
    if bool(re.search(mode, "nautilus", flags=re.IGNORECASE)):
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
    #Otherwise, throw error if mode not recognized
    else:
        raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
    #
    #Print some notes
    if do_verbose:
        print("Running generate_dict_reactions()...")
    #

    ##Load and store all possible species names
    tmpdict = _fetch_all_species(filename_species_gas=filename_species_gas,
                                filename_species_grain=filename_species_grain,
                                do_remove_phase_marker=False)
    data_species_gas = tmpdict["gas"]
    data_species_grain = tmpdict["grain"]
    list_species_all = tmpdict["all"]
    #

    ##Read in and store all base (original) gas and grain reactions
    #For Nautilus reaction format
    if bool(re.search(mode, "nautilus", flags=re.IGNORECASE)):
        #For gas reactions
        tmpdict = _read_pnautilus_reactions(
                                        text_filename=filename_reactions_gas,
                                        table_filename=table_filename_gas,
                                        last_counter=0)
        dict_reactions_gas = tmpdict["result"]
        tmpval = tmpdict["last_counter"]
        #For grain reactions
        tmpdict = _read_pnautilus_reactions(
                                        text_filename=filename_reactions_grain,
                                        table_filename=table_filename_grain,
                                        last_counter=tmpval)
        dict_reactions_grain = tmpdict["result"]
    #
    #Otherwise, throw error if mode not recognized
    else:
        raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
    #

    #Determine the maximum reaction id across all original reactions
    list_ids_gas = [dict_reactions_gas[key]["react_id"]
                    for key in dict_reactions_gas]
    list_ids_grain = [dict_reactions_grain[key]["react_id"]
                    for key in dict_reactions_grain]
    list_ids_all = list_ids_gas + list_ids_grain #Combined id list
    max_id = np.max(list_ids_all)

    #Extract all UV photochemistry reactions from base dict.
    dict_reactions_UV = _extract_subset_reactions(mode="UV",
                                    dict_reactions_gas=dict_reactions_gas,
                                    dict_reactions_gr=dict_reactions_grain)
    #

    ##Calculate branching ratios of each reaction based on original parameters
    #For UV
    if do_verbose:
        print("Setting branching ratios for UV reactions...")
    dict_reactions_UV = _set_branching_ratios(dict_reactions_UV)
    #

    ##Return the extracted dictionaries
    if do_verbose:
        print("Run of generate_dict_reactions() complete!\n")
    #
    return {"reactions_gas":dict_reactions_gas,
            "reactions_grain":dict_reactions_grain,
            "reactions_UV":dict_reactions_UV,
            "max_reaction_id":max_id, "list_species_all":list_species_all,
            "list_species_gas":data_species_gas,
            "list_species_grain":data_species_grain}
#

##Function: _read_pnautilus_reactions()
##Purpose: Read in reactions from pnautilus reaction file
def _read_pnautilus_reactions(text_filename, table_filename, last_counter, do_verbose=False):
    #Read lines from the file
    with open(text_filename, 'r') as openfile:
        text_reactions = openfile.readlines()

    #Initialize dictionary to hold reactions
    dict_reactions = {}

    #Iterate through lines of text and store each reaction
    i_counter = last_counter
    for line in text_reactions:
        #Skip if this line is a comment
        if line.startswith("!"):
            continue

        #Otherwise, process the reaction within this line
        curr_dict = utils._conv_reactstr_to_reactdict(line=line, mode="nautilus")
        curr_key = f"#{i_counter}"
        #Throw error if storage key already used
        if (curr_key in dict_reactions):
            raise ValueError(
                f"Err: Reaction key used:\n{curr_key}:{dict_reactions[curr_key]}"
            )
        #Otherwise, store reaction
        dict_reactions[curr_key] = curr_dict
        #Increment count of reaction ids
        i_counter += 1
    #

    #Save the reactions to a streamlined text file, if so desired
    if table_filename is not None:
        #Print some notes
        if do_verbose:
            print("Generating table of processed reactions...")
        #

        #Determine max number of reactants and products
        max_reactants = max([len(dict_reactions[item]["reactants"])
                            for item in dict_reactions])
        max_products = max([len(dict_reactions[item]["products"])
                            for item in dict_reactions])
        #Initialize string to hold table
        str_table = "#ID\t"
        #Set header of table
        for ii in range(0, max_reactants):
            str_table += "Reactant_{0}\t".format(ii+1)
        for ii in range(0, max_products):
            str_table += "Product_{0}\t".format(ii+1)
        #
        str_table += "Reaction\t"
        str_table += "Formula\tIType\tCoeff_A\tCoeff_B\tCoeff_C\tT_min\tT_max\n"
        #

        #Iterate through stored reactions
        for curr_key in dict_reactions:
            curr_dict = dict_reactions[curr_key]
            curr_str = "{0}\t".format(curr_key)
            #Write reactants
            num_reactants = 0
            for reactant in curr_dict["reactants"]:
                curr_str += "{0}\t".format(reactant)
                num_reactants += 1
            while num_reactants < max_reactants:
                curr_str += "\t"
                num_reactants += 1
            #Write products
            num_products = 0
            for product in curr_dict["products"]:
                curr_str += "{0}\t".format(product)
                num_products += 1
            while num_products < max_products:
                curr_str += "\t"
                num_products += 1
            #Write reaction
            curr_str += "{0}\t".format(curr_dict["reaction"])
            #Write formula info
            curr_str += "{0}\t".format(curr_dict["formula"])
            curr_str += "{0}\t".format(curr_dict["itype"])
            for coeff in curr_dict["coeffs"]:
                curr_str += "{0:.3e}\t".format(coeff)
            #Write temperature info
            curr_str+="{0}\t{1}\n".format(curr_dict["Tmin"],
                                            curr_dict["Tmax"])
            #Tack full row onto overall table
            str_table += curr_str
        #

        #Write the table string to given file
        utils._write_text(text=str_table, filename=table_filename)
    #

    #Return the stored dictionary
    return {"result":dict_reactions, "last_counter":i_counter}
#

##Function: _set_branching_ratios()
##Purpose: Update dictionary of reactions to include branching ratios.
def _set_branching_ratios(dict_reactions_orig):
    #Extract all unique reactant pairs
    unique_sets_reactants=list(set([tuple(dict_reactions_orig[key]["reactants"])
                                for key in dict_reactions_orig]))
    num_uniques = len(unique_sets_reactants)
    dict_reactions_new = dict_reactions_orig #Shallow instance copy
    #

    #Iterate through unique reactant pairs to compute branching ratios
    i_count = 0
    for ii in range(0, num_uniques):
        curr_unique = unique_sets_reactants[ii]
        #Fetch keys of all reactions with this set of reactants
        curr_keys = [key for key in dict_reactions_orig
                if (tuple(dict_reactions_orig[key]["reactants"])==curr_unique)]
        #

        #Calculate and store branching ratios for these reactions
        denom = sum([dict_reactions_orig[key]["coeffs"][0]
                    for key in curr_keys])
        dict_branchratios = {key:(dict_reactions_orig[key]["coeffs"][0]/denom)
                            for key in curr_keys}
        #

        #Throw error if branching ratios do not add to 1
        tmp_sum = sum([dict_branchratios[key] for key in dict_branchratios])
        if (not np.isclose(tmp_sum, 1)):
            raise ValueError("Whoa! Invalid branch ratio sum?\n{0}\n{1}"
                            .format(dict_branchratios, tmp_sum))
        #

        #Store the computed branching ratios for each key
        for key1 in dict_branchratios:
            dict_reactions_new[key1]["probability"] = dict_branchratios[key1]
        #

        #Accumulate count of reactions considered
        i_count += len(dict_branchratios)
    #

    #Throw error if total number of reactions not conserved
    if (i_count != len(dict_reactions_orig)):
        raise ValueError("Whoa! Unequal reaction count?\n{0} vs {1}"
                        .format(len(dict_reactions_orig), i_count))
    #

    #Return the updated dictionary of reactions
    return dict_reactions_new
#

##Function: _update_reaction_file_pnautilus()
##Purpose: Update Nautilus reaction file with given new reactions.
def _update_reaction_file_pnautilus(dict_reactions_new_all, filename_orig, filename_save, do_verbose=False):
    #Load in the lines for the original file
    with open(filename_orig, 'r') as openfile:
        all_lines = [item for item in openfile.readlines()
                    if (not item.strip().startswith("!"))] #Ignore comments
    #
    #Print some notes
    if do_verbose:
        print("\n> Running _update_reaction_file_pnautilus!")
    #

    #Initialize string to hold updated reaction file
    text = preset.pnautilus_header_reactiongas

    #Iterate through and copy/update original reactions
    dict_maps = {}
    list_used_new = []
    for curr_line in all_lines:
        #Ensure a newline is included
        if (not curr_line.endswith("\n")):
            curr_line += "\n"

        #Convert current line into reaction dictionary
        curr_dict = utils._conv_reactstr_to_reactdict(curr_line,mode="nautilus")
        curr_lookup_raw = [
            key for key in dict_reactions_new_all["UV"]
            if (
                (dict_reactions_new_all["UV"] is not None)
                and (dict_reactions_new_all["UV"][key]["react_id"]
                    == curr_dict["react_id"])
            )
        ]
        if (len(curr_lookup_raw) > 1):
            raise ValueError(f"Err: Bad id:\n{curr_dict}\n{curr_lookup_raw}")
        #
        is_UV = ((dict_reactions_new_all["UV"] is not None)
                    and (len(curr_lookup_raw) > 0)
                    and (curr_lookup_raw[0] in dict_reactions_new_all["UV"]))

        #Replace reaction in new file, if update available
        #For UV updates
        if is_UV:
            curr_lookup = curr_lookup_raw[0]
            #Comment out previous version
            text += "! {0}".format(curr_line)
            #Tack on new version
            text += "{0}\n".format(utils._conv_reactdict_to_reactstr(
                                        dict_reactions_new_all["UV"][curr_lookup],
                                        mode="nautilus"))
            #
            #Take note that this reaction was used
            list_used_new.append(curr_lookup)
        #
        #Otherwise, just copy over original line
        else:
            text += curr_line
    #

    #Throw an error if any updated reactions missed or duplicated
    tmp_list = [item for key in dict_reactions_new_all
                if (dict_reactions_new_all[key] is not None)
                for item in dict_reactions_new_all[key]]
    for ii in range(0, len(tmp_list)):
        if (tmp_list[ii] in dict_maps):
            tmp_key = tmp_list[ii]
            tmp_list[ii] = dict_maps[tmp_key]
            del dict_maps[tmp_key]
    #
    if (len(dict_maps) != 0): #If any conversions missed in verification
        raise ValueError("Whoa! Missed mappings!\n{0}".format(dict_maps))
    #
    if (not np.array_equal(np.sort(list(tmp_list)), np.sort(list_used_new))):
        raise ValueError("Whoa! Mismatch in new reactions vs written reactions!"
                        +"\nNew: {0}\n\nWritten: {1}"
                        .format(np.sort(tmp_list),
                                np.sort(list_used_new)))
    #

    #Save the new reaction file
    utils._write_text(text=text, filename=filename_save)
    #

    #Print some notes
    if do_verbose:
        print("\n{0} reactions updated and included in file {1}."
                .format(len(list_used_new), filename_save))
        print("Run of _update_reaction_file_pnautilus complete!")
    #

    #Exit the method
    return
#

##Function: _update_reaction_dict_indiv()
##Purpose: Update individual reaction with updated photorates, based on requested mode.
def _update_reaction_dict_indiv(mode, wavelength_radiation, spectrum_photon_radiation, dict_molinfo, react_orig, minval_coeff0, delta_energy=None, ionrate_X_primary=None, do_verbose=False):
    #Print some notes
    if do_verbose:
        print("---"*5)
        print("Updating reaction dictionary for: {0}:{1}"
                .format(react_orig["reactants"], react_orig["phototype"]))
    #
    #For updating individual photochemistry reaction
    if (mode in ["UV"]):
        #Unify the spectra wavelengths and values
        set_res = math._unify_spectra_wavelengths(
                    x_list=[wavelength_radiation, dict_molinfo["wave"]],
                    y_list=[spectrum_photon_radiation, dict_molinfo["cross"]],
                    which_overlap="minimum")
        #
        x_unified = set_res["x"]
        y_rad_trim = set_res["y"][0]
        y_crosssec_trim = set_res["y"][1]
        #
        #Calculate the UV photorate
        new_rate = math._calc_photorates_base(mode=mode,
                                    dict_molinfo=dict_molinfo,
                                    y_spec=y_rad_trim,
                                    y_cross=y_crosssec_trim, y_prod_theta=1,
                                    x_unified=x_unified,
                                    ionrate_X_primary=ionrate_X_primary)
        #
        #Generate a new reaction dictionary with new coefficients
        react_new = {key:react_orig[key] for key in react_orig}
    #
    #Otherwise, throw error if mode not recognized
    else:
        raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
    #

    #Fetch and incorporate branching ratio
    fin_rate = (react_new["probability"] * new_rate)

    #Set minimum coefficient value, if so requested
    if ((minval_coeff0 is not None) and (not np.isnan(fin_rate))):
        fin_rate = max([minval_coeff0, fin_rate]) #Sets minimum coeff0 threshold
    #

    #Store finalized coefficients
    react_new["coeffs"] = [fin_rate, 0.0, 0.0]

    #Print some notes
    if do_verbose:
        print("\nReaction dictionary updated for: {0}:{1}"
                .format(react_orig["reactants"], react_orig["phototype"]))
        print("Branching probability: {0}".format(react_new["probability"]))
        print("Old vs new coeffs: {0} vs {1}"
                .format(react_orig["coeffs"], react_new["coeffs"]))
        print("Old vs new equation formula: {0} vs {1}"
                .format(react_orig["formula"], react_new["formula"]))
        print("---"*5)
    #

    #Throw error if any nan values
    if any(np.isnan(react_new["coeffs"])):
        #Print spectra
        print(y_crosssec_trim)
        print(spectrum_photon_radiation)
        print(y_rad_trim)
        #Plot the spectra
        plt.close()
        plt.plot(x_unified, y_crosssec_trim)
        plt.plot(x_unified, y_rad_trim)
        plt.plot(wavelength_radiation, spectrum_photon_radiation)
        plt.yscale("log")
        plt.show()
        #Throw the error
        raise ValueError("Err: nans in coefficients???: {0}".format(new_rate))
    #

    #Return the new reaction dictionary
    return react_new
#

##Function: update_reactions()
##Purpose: Update reaction file with newly calculated+updated photorates as able.
def update_reactions(mode, dict_mol_UV, dict_reactions_orig, wavelength_radiation, spectrum_photon_radiation, minval_coeff0, delta_energy=None, ionrate_X_primary=None, filesave_dictold=None, filesave_dictnew=None, do_return_byproducts=False, do_verbose=False):
    #Print some notes
    if do_verbose:
        print("\nRunning update_reactions()!")
    #

    #Catalogue all reactions that can be updated with cross-sec. data
    #If replacements only, then fetch existing reactions to update
    dict_updatable = _fetch_updatable_reactions(mode=mode,
                            dict_mol_UV=dict_mol_UV,
                            dict_reactions_orig=dict_reactions_orig)
    #

    #Compute new photorates and reaction dicts. for each updatable reaction
    #For ultraviolet radiation
    if (mode == "UV"):
        dict_reactions_new = {key:_update_reaction_dict_indiv(mode=mode,
                            wavelength_radiation=wavelength_radiation,
                            spectrum_photon_radiation=spectrum_photon_radiation,
                            dict_molinfo=dict_updatable[key],
                            react_orig=dict_reactions_orig[key],
                            do_verbose=do_verbose, minval_coeff0=minval_coeff0)
                        for key in dict_updatable}
    #
    #Otherwise, throw error if mode not recognized
    else:
        raise ValueError("Whoa! Mode {0} not recognized!".format(mode))
    #

    #Take note of old versions of these updated reactions
    dict_reactions_old = {key:dict_reactions_orig[key]
                            for key in dict_updatable}
    #

    #Save dictionary, if requested
    if (filesave_dictold is not None):
        np.save(filesave_dictold, dict_reactions_old)
    #
    if (filesave_dictnew is not None):
        np.save(filesave_dictnew, dict_reactions_new)
    #

    #Return the byproducts of the calculations, if so requested
    if do_return_byproducts:
        return {"reactions_old":dict_reactions_old,
                "reactions_new":dict_reactions_new}
    #
    #Otherwise, just exit the function
    else:
        return
    #
#




##------------------------------------------------------------------------------
#
