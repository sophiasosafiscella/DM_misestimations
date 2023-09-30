#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:30:59 2020

@author: bshapiroalbert
"""
import subprocess

import matplotlib.pyplot as plt
# Import PINT
import pint.utils as pu

# import extra util functions brent wrote
from timing_analysis.utils import *

# color blind friends colors and markers?
# CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
# MARKERS = ['.', 'v', 's', 'x', '^', 'D', 'p', 'P', '*']

# Color scheme for consistent reciever-backend combos, same as published 12.5 yr
colorschemes = {'thankful_2':{
              "327_ASP":         "#BE0119",
              "327_PUPPI":       "#BE0119",
              "430_ASP":         "#FD9927",
              "430_PUPPI":       "#FD9927",
              "L-wide_ASP":      "#6BA9E2",
              "L-wide_PUPPI":    "#6BA9E2",
              "Rcvr1_2_GASP":    "#407BD5",
              "Rcvr1_2_GUPPI":   "#407BD5",
              "Rcvr_800_GASP":   "#61C853",
              "Rcvr_800_GUPPI":  "#61C853",
              "S-wide_ASP":      "#855CA0",
              "S-wide_PUPPI":    "#855CA0",
              "1.5GHz_YUPPI":    "#45062E",
              "3GHz_YUPPI":      "#E5A4CB",
              "6GHz_YUPPI":      "#40635F",
              "CHIME":           "#ECE133",
              }}
# marker dictionary to be used if desired, currently all 'x'
markers = {"327_ASP":        "x",
          "327_PUPPI":       "x",
          "430_ASP":         "x",
          "430_PUPPI":       "x",
          "L-wide_ASP":      "x",
          "L-wide_PUPPI":    "x",
          "Rcvr1_2_GASP":    "x",
          "Rcvr1_2_GUPPI":   "x",
          "Rcvr_800_GASP":   "x",
          "Rcvr_800_GUPPI":  "x",
          "S-wide_ASP":      "x",
          "S-wide_PUPPI":    "x",
          "1.5GHz_YUPPI":    "x",
          "3GHz_YUPPI":      "x",
          "6GHz_YUPPI":      "x",
          "CHIME":           "x",

            }
# Define the color map option
colorscheme = colorschemes['thankful_2']

def call(x):
    subprocess.call(x,shell=True)

def plot_residuals_time(fitter, restype = 'postfit', plotsig = False, avg = False, whitened = False, \
                        save = False, legend = True, title = True, axs = None, **kwargs):
    """
    Make a plot of the original_residuals vs. time


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of original_residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit original_residuals.
        'postfit' - plot the postfit original_residuals (default)
        'both' - overplot both the pre and post-fit original_residuals.
    plotsig [boolean] : If True plot number of measurements v. original_residuals/uncertainty, else v. original_residuals
        [default: False].
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average original_residuals [default: False].
    whitened [boolean] : If True will compute and plot whitened original_residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    mjds [list/array] : List or array of TOA MJDs to plot. Will override values from toa object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if wideband
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError("Cannot epoch average wideband original_residuals, please change 'avg' to False.")
    else:
        NB = True

    # Check if want epoch averaged original_residuals
    if avg == True and restype == 'prefit':
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'postfit':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'both':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)

    # Get original_residuals
    if 'res' in kwargs.keys():
        res = kwargs['res']
    else:
        if restype == 'prefit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                    res_pre = avg_dict_pre['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        else:
            raise ValueError("Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."\
                             %(restype))

    # Check if we want whitened original_residuals
    if whitened == True and ('res' not in kwargs.keys()):
        if avg == True:
            if restype != 'both':
                res = whiten_resids(avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype='prefit')
                res_pre = whiten_resids(avg_dict, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
        else:
            if restype != 'both':
                res = whiten_resids(fitter, restype=restype)
            else:
                res = whiten_resids(fitter, restype='prefit')
                res_pre = whiten_resids(fitter, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)

    # Get errors
    if 'errs' in kwargs.keys():
        errs = kwargs['errs']
    else:
        if restype == 'prefit':
            if avg == True:
                errs = avg_dict['errors'].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                    errs_pre = avg_dict_pre['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)

    # Get MJDs
    if 'mjds' in kwargs.keys():
        mjds = kwargs['mjds']
    else:
        mjds = fitter.toas.get_mjds().value
        if avg == True:
            mjds = avg_dict['mjds'].value
    # Convert to years
    years = (mjds - 51544.0)/365.25 + 2000.0

    # Get receiver backends
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
        if avg == True:
            avg_rcvr_bcknds = []
            for iis in avg_dict['indices']:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            rcvr_bcknds = np.array(avg_rcvr_bcknds)
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    if 'figsize' in kwargs.keys():
        figsize = kwargs['figsize']
    else:
        figsize = (10,4)
    if axs == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'fmt' in kwargs.keys():
            mkr = kwargs['fmt']
        else:
            mkr = markers[r_b_label]
            if restype == 'both':
                mkr_pre = '.'
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 0.5
        if plotsig:
            sig = res[inds]/errs[inds]
            ax1.errorbar(years[inds], sig, yerr=len(errs[inds])*[1], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha, picker=True)
            if restype == 'both':
                sig_pre = res_pre[inds]/errs_pre[inds]
                ax1.errorbar(years[inds], sig_pre, yerr=len(errs_pre[inds])*[1], fmt=mkr_pre, \
                         color=clr, label=r_b_label+" Prefit", alpha = alpha, picker=True)
        else:
            ax1.errorbar(years[inds], res[inds], yerr=errs[inds], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha, picker=True)
            if restype == 'both':
                ax1.errorbar(years[inds], res_pre[inds], yerr=errs_pre[inds], fmt=mkr_pre, \
                     color=clr, label=r_b_label+" Prefit", alpha = alpha, picker=True)

    # Set second axis
    ax1.set_xlabel(r'Year')
    ax1.grid(True)
    ax2 = ax1.twiny()
    mjd0  = ((ax1.get_xlim()[0])-2004.0)*365.25+53005.
    mjd1  = ((ax1.get_xlim()[1])-2004.0)*365.25+53005.
    ax2.set_xlim(mjd0, mjd1)
    if plotsig:
        if avg and whitened:
            ax1.set_ylabel('Average Residual/Uncertainty \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel('Average Residual/Uncertainty')
        elif whitened and not avg:
            ax1.set_ylabel('Residual/Uncertainty \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel('Residual/Uncertainty')
    else:
        if avg and whitened:
            ax1.set_ylabel('Average Residual ($\mu$s) \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel('Average Residual ($\mu$s)')
        elif whitened and not avg:
            ax1.set_ylabel('Residual ($\mu$s) \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel('Residual ($\mu$s)')
    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS)/2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, y_offset+1.0/figsize[1]), ncol=ncol)
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title("%s %s timing original_residuals" % (fitter.model.PSR.value, restype), y=y_offset + 1.0 / figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if whitened:
            ext += "_whitened"
        if avg:
            ext += "_averaged"
        if NB:
            ext += "_NB"
        else:
            ext += "_WB"
        if restype == 'prefit':
            ext += "_prefit"
        elif restype == 'postfit':
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_v_mjd%s.png" % (fitter.model.PSR.value, ext))
    
    if axs == None:
        # Define clickable points
        text = ax2.text(0,0,"")

        # Define point highlight color
        if "430_ASP" in RCVR_BCKNDS or "430_PUPPI" in RCVR_BCKNDS:
            stamp_color = "#61C853"
        else:
            stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data_res_avg
            xdata = mjds
            if plotsig:
                ydata = (res/errs).decompose().value
            else:
                ydata = res.value
            # Get x and y data_res_avg from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick)/10.0)**2 + (ydata - yclick)**2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax2.scatter(xdata[ind_close], ydata[ind_close], marker = 'x', c = stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            if plotsig:
                text.set_text("TOA Params:\n MJD: %s \n Res/Err: %.2f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))
            else:
                text.set_text("TOA Params:\n MJD: %s \n Res: %.2f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))

        fig.canvas.mpl_connect('button_press_event', onclick)

    return

def plot_dmx_time(fitter, savedmx = False, save = False, legend = True,\
                  axs = None, title = True, compare = False, **kwargs):
    """
    Make a plot of DMX vs. time


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    savedmx [boolean] : If True will save dmxparse values to output file with `"psrname"_dmxparse.nb/wb/.out`.
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    dmx [list/array] : List or array of DMX values to plot. Will override values from fitter object.
    errs [list/array] : List or array of DMX error values to plot. Will override values from fitter object.
    mjds [list/array] : List or array of DMX mjd epoch values to plot. Will override values from fitter object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Get pulsar name
    psrname = fitter.model.PSR.value

    # Check if wideband
    if fitter.is_wideband:
        NB = False
        dmxname = "%s_dmxparse.wb.out" % (psrname)
    else:
        NB = True
        dmxname = "%s_dmxparse.nb.out" % (psrname)

    # Get plotting dmx and error values for WB
    if 'dmx' in kwargs.keys():
        DMXs = kwargs['dmx']
    else:
        # get dmx dictionary from pint dmxparse function
        dmx_dict = pu.dmxparse(fitter, save=savedmx)
        DMXs = dmx_dict['dmxs'].value
        DMX_vErrs = dmx_dict['dmx_verrs'].value
        DMX_center_MJD = dmx_dict['dmxeps'].value
        DMX_center_Year = (DMX_center_MJD- 51544.0)/365.25 + 2000.0
        # move file name
        if savedmx:
            os.rename("dmxparse.out", dmxname)

    # Double check/overwrite errors if necessary
    if 'errs' in kwargs.keys():
        DMX_vErrs = kwargs['errs']
    # Double check/overwrite dmx mjd epochs if necessary
    if 'mjds' in kwargs.keys():
        DMX_center_MJD = kwargs['mjds']
        DMX_center_Year = (DMX_center_MJD- 51544.0)/365.25 + 2000.0

    # If we want to compare WB to NB, we need to look for the right output file
    if compare == True:
        # Look for other dmx file
        if NB:
            #log.log(1, "Searching for file: %s_dmxparse.wb.out" % (psrname))
            if not os.path.isfile("%s_dmxparse.wb.out"%(psrname)):
                raise RuntimeError("Cannot find Wideband DMX parse output file.")
            else:
                # Get the values from the DMX parse file
                dmx_epochs, nb_dmx, nb_dmx_var, nb_dmx_r1, nb_dmx_r2 = np.loadtxt("%s_dmxparse.wb.out"%(psrname),\
                                                    unpack=True, usecols=(0,1,2,3,4))
        else:
            #log.log(1, "Searching for file: %s_dmxparse.nb.out" % (psrname))
            if not os.path.isfile("%s_dmxparse.nb.out"%(psrname)):
                raise RuntimeError("Cannot find Narrowband DMX parse output file.")
            else:
                # Get the values from the DMX parse file
                dmx_epochs, nb_dmx, nb_dmx_var, nb_dmx_r1, nb_dmx_r2 = np.loadtxt("%s_dmxparse.nb.out"%(psrname),\
                                                    unpack=True, usecols=(0,1,2,3,4))
        dmx_mid_yr = (dmx_epochs- 51544.0)/365.25 + 2000.0
    
    # Define the plotting function
    if axs == None:
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (10,4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    # Get plot preferences
    if 'fmt' in kwargs.keys():
        mkr = kwargs['fmt']
    else:
        mkr = 's'
        if compare:
            mkr_nb = 'o'
    if 'color' in kwargs.keys():
        clr = kwargs['color']
    else:
        clr = 'gray'
        if compare:
            clr_nb = 'k'
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 1.0
    # Not actually plot
    if NB and not compare:
        ax1.errorbar(DMX_center_Year, DMXs*10**3, yerr=DMX_vErrs*10**3, fmt='.', c = clr, marker = mkr, \
                         label="Narrowband")
    elif not NB and not compare:
        ax1.errorbar(DMX_center_Year, DMXs*10**3, yerr=DMX_vErrs*10**3, fmt='.', c = clr, marker = mkr, \
                         label="Wideband")
    elif compare:
        if NB:
            ax1.errorbar(DMX_center_Year, DMXs*10**3, yerr=DMX_vErrs*10**3, fmt='.', c = clr, marker = mkr, \
                         label="Narrowband")
            ax1.errorbar(dmx_mid_yr, nb_dmx*10**3, yerr = nb_dmx_var*10**3, fmt = '.', color = clr_nb, marker = mkr_nb, \
                     label='Wideband')
        else:
            ax1.errorbar(DMX_center_Year, DMXs*10**3, yerr=DMX_vErrs*10**3, fmt='.', c = clr, marker = mkr, \
                         label="Wideband")
            ax1.errorbar(dmx_mid_yr, nb_dmx*10**3, yerr = nb_dmx_var*10**3, fmt = '.', color = clr_nb, marker = mkr_nb, \
                     label='Narrowband')

    # Set second axis
    ax1.set_xlabel(r'Year')
    ax1.grid(True)
    ax2 = ax1.twiny()
    mjd0  = ((ax1.get_xlim()[0])-2004.0)*365.25+53005.
    mjd1  = ((ax1.get_xlim()[1])-2004.0)*365.25+53005.
    ax2.set_xlim(mjd0, mjd1)
    ax1.set_ylabel(r"DMX ($10^{-3}$ pc cm$^{-3}$)")
    if legend:
        ax1.legend(loc='best')
    if title:
        if NB and not compare:
            plt.title("%s narrowband dmx" % (psrname), y=1.0+1.0/figsize[1])
        elif not NB and not compare:
            plt.title("%s wideband dmx" % (psrname), y=1.0+1.0/figsize[1])
        elif compare:
            plt.title("%s narrowband and wideband dmx" % (psrname), y=1.0+1.0/figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if NB and not compare:
            ext += "_NB"
        if not NB and not compare:
            ext += "_WB"
        if compare:
            ext += "_NB_WB_compare"
        plt.savefig("%s_dmx_v_time%s.png" % (psrname, ext))

    if axs == None:
        # Define clickable points
        text = ax1.text(0,0,"")
        # Define color for highlighting points
        stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data_res_avg
            xdata = DMX_center_Year
            ydata = DMXs*10**3
            # Get x and y data_res_avg from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick)/1000.0)**2 + (ydata - yclick)**2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax2.scatter(xdata[ind_close], ydata[ind_close], marker = 's', c = stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            text.set_text("DMX Params:\n MJD: %s \n DMX: %.2f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))

        fig.canvas.mpl_connect('button_press_event', onclick)

    return

def plot_dmxout(dmxout_files, labels, psrname=None, outfile=None, model = None):
    """ Make simple dmx vs. time plot with dmxout file(s) as input

    Parameters
    ==========
    dmxout_files: list/str
        list of dmxout files to plot (or str if only one)
    labels: list/str
        list of labels for dmx timeseries (or str if only one)
    psrname: str, optional
        pulsar name
    outfile: str, optional
        save figure and write to outfile if set
    model: `pint.model.TimingModel` object, optional
        if provided, plot excess DM based on a basic SWM (NE_SW = 10.0)

    Returns
    =======
    dmxDict: dictionary
        dmxout information (mjd, val, err, r1, r2) for each label
    """
    from astropy.time import Time
    if isinstance(dmxout_files, str): dmxout_files = [dmxout_files]
    if isinstance(labels, str): labels = [labels]
    
    figsize = (10,4)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Year')
    ax1.set_ylabel(r"DMX ($10^{-3}$ pc cm$^{-3}$)")
    ax1.grid(True)
    ax2 = ax1.twiny()
    ax2.set_xlabel('MJD')

    dmxDict = {}
    for ii,(df,lab) in enumerate(zip(dmxout_files,labels)):
        dmxmjd, dmxval, dmxerr, dmxr1, dmxr2 = np.loadtxt(df, unpack=True, usecols=range(0,5))
        idmxDict = {'mjd':dmxmjd,'val':dmxval,'err':dmxerr,'r1':dmxr1,'r2':dmxr2}        
        ax2.errorbar(dmxmjd, dmxval*10**3, yerr=dmxerr*10**3, label=lab, marker='o', ls='', markerfacecolor='none')
        dmxDict[lab] = idmxDict

    # set ax1 lims (year) based on ax2 lims (mjd)
    mjd_xlo, mjd_xhi = ax2.get_xlim()
    dy_xlo = Time(mjd_xlo,format='mjd').decimalyear
    dy_xhi = Time(mjd_xhi,format='mjd').decimalyear
    ax1.set_xlim(dy_xlo,dy_xhi)

    # capture ylim
    orig_ylim = ax2.get_ylim()

    if psrname: ax1.text(0.975,0.05,psrname,transform=ax1.transAxes,size=18,c='lightgray',
                         horizontalalignment='right', verticalalignment='bottom')
    if model:
        from pint.simulation import make_fake_toas_fromMJDs
        from timing_analysis.lite_utils import remove_noise
        fake_mjds = np.linspace(np.min(dmxmjd),np.max(dmxmjd),num=int(np.max(dmxmjd)-np.min(dmxmjd)))
        fake_mjdTime = Time(fake_mjds,format='mjd')

        # copy the model and add sw component
        mo_swm = copy.deepcopy(model)
        remove_noise(mo_swm)  # Not necessary here and avoids lots of warnings
        mo_swm.NE_SW.value = 10.0

        # generate fake TOAs and calculate excess DM due to solar wind
        fake_toas = make_fake_toas_fromMJDs(fake_mjdTime,mo_swm)
        sun_dm_delays = mo_swm.solar_wind_dm(fake_toas)*10**3  # same scaling as above
        ax2.plot(fake_mjds,sun_dm_delays,c='lightgray',label='Excess DM')

    # don't change ylim based on excess dm trace, if plotted
    ax2.set_ylim(orig_ylim)
    ax2.legend(loc='best')

    plt.tight_layout()
    if outfile: plt.savefig(outfile)
    return dmxDict

def plot_dmx_diffs_nbwb(dmxDict, show_missing=True, psrname=None, outfile=None):
    """ Uses output dmxDict from plot_dmxout() to plot diffs between simultaneous nb-wb values

    Parameters
    ==========
    dmxDict: dictionary
        dmxout information (mjd, val, err, r1, r2) for nb, wb, respectively (check both exist)
    show_missing: bool, optional
        if one value is missing (nb/wb), indicate the epoch and which one
    psrname: str, optional
        pulsar name
    outfile: str, optional
        save figure and write to outfile if set

    Returns
    =======
    None?
    """
    # should check that both nb/wb entries exist first...
    nbmjd = dmxDict['nb']['mjd']
    wbmjd = dmxDict['wb']['mjd']
    allmjds = set(list(nbmjd) + list(wbmjd))

    # May need slightly more curation if nb/wb mjds are *almost* identical
    wbonly = allmjds-set(nbmjd)
    nbonly = allmjds-set(wbmjd)
    both = set(nbmjd).intersection(set(wbmjd))

    # assemble arrays of common inds for plotting later; probably a better way to do this
    nb_common_inds = []
    wb_common_inds = []
    for b in both:
        nb_common_inds.append(np.where(nbmjd==b)[0][0])
        wb_common_inds.append(np.where(wbmjd==b)[0][0])

    nb_common_inds, wb_common_inds = np.array(nb_common_inds), np.array(wb_common_inds)

    nbdmx,nbdmxerr = dmxDict['nb']['val'],dmxDict['nb']['err']
    wbdmx,wbdmxerr = dmxDict['wb']['val'],dmxDict['wb']['err']

    # propagate errors as quadrature sum, though Michael thinks geometric mean might be better?
    nbwb_dmx_diffs = nbdmx[nb_common_inds]-wbdmx[wb_common_inds]
    nbwb_err_prop = np.sqrt(nbdmxerr[nb_common_inds]**2 + wbdmxerr[wb_common_inds]**2)

    # make the plot
    from astropy.time import Time
    figsize = (10,4)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Year')
    ax1.set_ylabel(r"$\Delta$DMX ($10^{-3}$ pc cm$^{-3}$)")
    ax1.grid(True)
    ax2 = ax1.twiny()
    ax2.set_xlabel('MJD')

    botharray = np.array(list(both))
    mjdbothTime = Time(botharray,format='mjd')
    dybothTime = mjdbothTime.decimalyear

    minmjd, maxmjd = np.sort(botharray)[0], np.sort(botharray)[-1]
    ax2.set_xlim(minmjd, maxmjd)

    ax1.errorbar(dybothTime,nbwb_dmx_diffs*1e3,yerr=nbwb_err_prop*1e3,
                marker='o', ls='', markerfacecolor='none',label='nb - wb')

    # want arrows indicating missing nb/wb DMX values to difference
    if show_missing:
        stddiffs = np.std(nbwb_dmx_diffs)
        mjdnbonlyTime = Time(np.array(list(nbonly)),format='mjd')
        dynbonlyTime = mjdnbonlyTime.decimalyear
        ax1.scatter(dynbonlyTime,np.zeros(len(nbonly))+stddiffs*1e3,marker='v',c='r',label='nb only')
        nbonlystr = [str(no) for no in nbonly]
        if nbonlystr: log.warning(f"nb-only measurements available for MJDs: {', '.join(nbonlystr)}")

        mjdwbonlyTime = Time(np.array(list(wbonly)),format='mjd')
        dywbonlyTime = mjdwbonlyTime.decimalyear
        ax1.scatter(dywbonlyTime,np.zeros(len(wbonly))-stddiffs*1e3,marker='^',c='r',label='wb only')
        wbonlystr = [str(wo) for wo in wbonly]
        if wbonlystr: log.warning(f"wb-only measurements available for MJDs: {', '.join(wbonlystr)}")

    if psrname: ax1.text(0.975,0.05,psrname,transform=ax1.transAxes,size=18,c='lightgray',
                        horizontalalignment='right', verticalalignment='bottom')
    plt.tight_layout()
    ax1.legend(loc='best')
    if outfile: plt.savefig(outfile)

    return None

# Now we want to make wideband DM vs. time plot, this uses the premade dm_resids from PINT
def plot_dm_residuals(fitter, restype = 'postfit', plotsig = False, save = False, legend = True, title = True,\
                      axs = None, mean_sub = True, **kwargs):
    """
    Make a plot of Wideband timing DM original_residuals v. time.


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of original_residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit original_residuals.
        'postfit' - plot the postfit original_residuals (default)
        'both' - overplot both the pre and post-fit original_residuals.
    plotsig [boolean] : If True plot number of measurements v. original_residuals/uncertainty, else v. original_residuals
        [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].
    mean_sub [boolean] : If False, will not mean subtract the DM original_residuals to be centered on zero [default: True]

    Optional Arguments:
    --------------------
    dmres [list/array] : List or array of DM residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of DM residual error values to plot. Will override values from fitter object.
    mjds [list/array] : List or array of DM residual mjds to plot. Will override values from fitter object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if wideband
    if not fitter.is_wideband:
        raise RuntimeError("Error: Narrowband TOAs have no DM original_residuals, use `plot_dmx_time() instead.")

    # Get the DM original_residuals
    if 'dmres' in kwargs.keys():
        dm_resids = kwargs['dmres']
    else:
        if restype == "postfit":
            dm_resids = fitter.resids.residual_objs['dm'].resids.value
        elif restype == 'prefit':
            dm_resids = fitter.resids_init.residual_objs['dm'].resids.value
        elif restype == 'both':
            dm_resids = fitter.resids.residual_objs['dm'].resids.value
            dm_resids_init = fitter.resids_init.residual_objs['dm'].resids.value

    # Get the DM residual errors
    if "errs" in kwargs.keys():
        dm_error = kwargs['errs']
    else:
        if restype == 'postfit':
            dm_error = fitter.resids.residual_objs['dm'].get_data_error().value
        elif restype == 'prefit':
            dm_error = fitter.resids_init.residual_objs['dm'].get_data_error().value
        elif restype == 'both':
            dm_error = fitter.resids.residual_objs['dm'].get_data_error().value
            dm_error_init = fitter.resids_init.residual_objs['dm'].get_data_error().value
        
    # Get the MJDs
    if 'mjds' in kwargs.keys():
        mjds = kwargs['mjds']
    else:
        mjds = fitter.toas.get_mjds().value
    years = (mjds - 51544.0)/365.25 + 2000.0
    
    # Get the receiver-backend combos
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    # If we don't want mean subtraced data_res_avg we add the mean
    if not mean_sub:
        if 'dmres' in kwargs.keys():
            dm_avg = dm_resids
        else:
            dm_avg = fitter.resids.residual_objs['dm'].dm_data
        if "errs" in kwargs.keys():
            dm_avg_err = dm_error
        else:
            dm_avg_err = fitter.resids.residual_objs['dm'].get_data_error().value
        DM0 = np.average(dm_avg, weights=(dm_avg_err)**-2)
        dm_resids += DM0.value
        if restype == 'both':
            dm_resids_init += DM0.value
        if plotsig:
            ylabel = r"DM/Uncertainty"
        else:
            ylabel = r"DM [cm$^{-3}$ pc]"
    else:
        if plotsig:
            ylabel = r"$\Delta$DM/Uncertainty"
        else:
            ylabel = r"$\Delta$DM [cm$^{-3}$ pc]"
    
    if axs == None:
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (10,4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'fmt' in kwargs.keys():
            mkr = kwargs['fmt']
        else:
            mkr = markers[r_b_label]
            if restype == 'both':
                mkr_pre = '.'
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 0.5

        # Do plotting command
        if restype == 'both':
            if plotsig:
                dm_sig = dm_resids[inds]/dm_error[inds]
                dm_sig_pre = dm_resids_init[inds]/dm_error[inds]
                ax1.errorbar(years[inds], dm_sig, yerr=len(dm_error[inds])*[1], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)
                ax1.errorbar(years[inds], dm_sig_pre, yerr=len(dm_error_init[inds])*[1], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label+" Prefit", alpha = 0.5)
            else:
                ax1.errorbar(years[inds], dm_resids[inds], yerr=dm_error[inds], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)
                ax1.errorbar(years[inds], dm_resids_init[inds], yerr=dm_error_init[inds], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label+" Prefit", alpha = 0.5)
        else:
            if plotsig:
                dm_sig = dm_resids[inds]/dm_error[inds]
                ax1.errorbar(years[inds], dm_sig, yerr=len(dm_error[inds])*[1], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)
            else:
                ax1.errorbar(years[inds], dm_resids[inds], yerr=dm_error[inds], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)

    # Set second axis
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(r'Year')
    ax1.grid(True)
    ax2 = ax1.twiny()
    mjd0  = ((ax1.get_xlim()[0])-2004.0)*365.25+53005.
    mjd1  = ((ax1.get_xlim()[1])-2004.0)*365.25+53005.
    ax2.set_xlim(mjd0, mjd1)

    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS)/2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, y_offset+1.0/figsize[1]), ncol=ncol)
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title("%s %s DM original_residuals" % (fitter.model.PSR.value, restype), y=y_offset + 1.0 / figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if restype == "postfit":
            ext += "_postfit"
        elif restype == "prefit":
            ext += "_prefit"
        elif restype == "both":
            ext += "_prefit_and_postfit"
        plt.savefig("%s_dm_resids_v_time%s.png" % (fitter.model.PSR.value, ext))

    if axs == None:
        # Define clickable points
        text = ax2.text(0,0,"")

        # Define point highlight color
        if "430_ASP" in RCVR_BCKNDS or "430_PUPPI" in RCVR_BCKNDS:
            stamp_color = "#61C853"
        else:
            stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data_res_avg
            xdata = mjds
            if plotsig:
                ydata = dm_resids/dm_error
            else:
                ydata = dm_resids
            # Get x and y data_res_avg from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick)/1000.0)**2 + (ydata - yclick)**2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax2.scatter(xdata[ind_close], ydata[ind_close], marker = 'x', c = stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            text.set_text("DM Params:\n MJD: %s \n Res: %.6f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))

        fig.canvas.mpl_connect('button_press_event', onclick)
    
    return

def plot_measurements_v_res(fitter, restype = 'postfit', plotsig = False, nbin = 50, avg = False, whitened = False, \
                        save = False, legend = True, title = True, axs = None, **kwargs):
    """
    Make a histogram of number of measurements v. original_residuals


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of original_residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit original_residuals.
        'postfit' - plot the postfit original_residuals (default)
        'both' - overplot both the pre and post-fit original_residuals.
    plotsig [boolean] : If True plot number of measurements v. original_residuals/uncertainty, else v. original_residuals
        [default: False].
    nbin [int] : Number of bins to use in the histogram [default: 50]
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average original_residuals [default: False].
    whitened [boolean] : If True will compute and plot whitened original_residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].
    
    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if wideband
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError("Cannot epoch average wideband original_residuals, please change 'avg' to False.")
    else:
        NB = True

    # Check if want epoch averaged original_residuals
    if avg == True and restype == 'prefit':
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'postfit':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'both':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)

    # Get original_residuals
    if 'res' in kwargs.keys():
        res = kwargs['res']
    else:
        if restype == 'prefit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                    res_pre = avg_dict_pre['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        else:
            raise ValueError("Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."\
                             %(restype))

    # Check if we want whitened original_residuals
    if whitened == True and ('res' not in kwargs.keys()):
        if avg == True:
            if restype != 'both':
                res = whiten_resids(avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype='prefit')
                res_pre = whiten_resids(avg_dict, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)    
        else:
            if restype != 'both':
                res = whiten_resids(fitter, restype=restype)
            else:
                res = whiten_resids(fitter, restype='prefit')
                res_pre = whiten_resids(fitter, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
    
    # Get errors
    if 'errs' in kwargs.keys():
        errs = kwargs['errs']
    else:
        if restype == 'prefit':
            if avg == True:
                errs = avg_dict['errors'].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                    errs_pre = avg_dict_pre['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)
    
    # Get receiver backends
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
        if avg == True:
            avg_rcvr_bcknds = []
            for iis in avg_dict['indices']:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            rcvr_bcknds = np.array(avg_rcvr_bcknds)
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)
    
    if axs == None:
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (10,4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    
    xmax=0
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        if plotsig:
            sig = res[inds]/errs[inds]
            ax1.hist(sig, nbin, histtype='step', color=colorscheme[r_b_label], label=r_b_label)
            xmax = max(xmax,max(sig),max(-sig))
            if restype == 'both':
                sig_pre = res_pre[inds]/errs_pre[inds]
                ax1.hist(sig_pre, nbin, histtype='step', color=colorscheme[r_b_label], linestyle = '--',\
                         label=r_b_label+" Prefit")
        else:
            ax1.hist(res[inds], nbin, histtype='step', color=colorscheme[r_b_label], label=r_b_label)
            xmax = max(xmax,max(res[inds]),max(-res[inds]))
            if restype == 'both':
                ax1.hist(res[inds], nbin, histtype='step', color=colorscheme[r_b_label], linestyle = '--',\
                         label=r_b_label+" Prefit")
            
    ax1.grid(True)
    ax1.set_ylabel("Number of measurements")
    if plotsig:
        if avg and whitened:
            ax1.set_xlabel('Average Residual/Uncertainty \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_xlabel('Average Residual/Uncertainty')
        elif whitened and not avg:
            ax1.set_xlabel('Residual/Uncertainty \n (Whitened)', multialignment='center')
        else:
            ax1.set_xlabel('Residual/Uncertainty')
    else:
        if avg and whitened:
            ax1.set_xlabel('Average Residual ($\mu$s) \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_xlabel('Average Residual ($\mu$s)')
        elif whitened and not avg:
            ax1.set_xlabel('Residual ($\mu$s) \n (Whitened)', multialignment='center')
        else:
            ax1.set_xlabel('Residual ($\mu$s)')
    ax1.set_xlim(-1.1*xmax,1.1*xmax)
    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS)/2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, y_offset+1.0/figsize[1]), ncol=ncol)
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title("%s %s residual measurements" % (fitter.model.PSR.value, restype), y=y_offset+1.0/figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if whitened:
            ext += "_whitened"
        if avg:
            ext += "_averaged"
        if NB:
            ext += "_NB"
        else:
            ext += "_WB"
        if restype == 'prefit':
            ext += "_prefit"
        elif restype == 'postfit':
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_measurements%s.png" % (fitter.model.PSR.value, ext))
        
    return

def plot_measurements_v_dmres(fitter, restype = 'postfit', plotsig = False, nbin = 50, \
                        save = False, legend = True, title = True, axs = None, mean_sub = True, **kwargs):
    """
    Make a histogram of number of measurements v. original_residuals


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of original_residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit original_residuals.
        'postfit' - plot the postfit original_residuals (default)
        'both' - overplot both the pre and post-fit original_residuals.
    plotsig [boolean] : If True plot number of measurements v. original_residuals/uncertainty, else v. original_residuals
        [default: False].
    nbin [int] : Number of bins to use in the histogram [default: 50]
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].
    mean_sub [boolean] : If False, will not mean subtract the DM original_residuals to be centered on zero [default: True]
    
    Optional Arguments:
    --------------------
    dmres [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    """
    # Check if wideband
    if not fitter.is_wideband:
        raise ValueError(
            "Narrowband Fitters have have no DM original_residuals, please use `plot_measurements_v_dmres` instead.")

    # Get the DM original_residuals
    if 'dmres' in kwargs.keys():
        dm_resids = kwargs['dmres']
    else:
        if restype == "postfit":
            dm_resids = fitter.resids.residual_objs['dm'].resids.value
        elif restype == 'prefit':
            dm_resids = fitter.resids_init.residual_objs['dm'].resids.value
        elif restype == 'both':
            dm_resids = fitter.resids.residual_objs['dm'].resids.value
            dm_resids_init = fitter.resids_init.residual_objs['dm'].resids.value
    
    # Get the DM residual errors
    if "errs" in kwargs.keys():
        dm_error = kwargs['errs']
    else:
        if restype == 'postfit':
            dm_error = fitter.resids.residual_objs['dm'].get_data_error().value
        elif restype == 'prefit':
            dm_error = fitter.resids_init.residual_objs['dm'].get_data_error().value
        elif restype == 'both':
            dm_error = fitter.resids.residual_objs['dm'].get_data_error().value
            dm_error_init = fitter.resids_init.residual_objs['dm'].get_data_error().value
    
    # Get the receiver-backend combos
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    # If we don't want mean subtraced data_res_avg we add the mean
    if not mean_sub:
        if 'dmres' in kwargs.keys():
            dm_avg = dm_resids
        else:
            dm_avg = fitter.resids.residual_objs['dm'].dm_data
        if "errs" in kwargs.keys():
            dm_avg_err = dm_error
        else:
            dm_avg_err = fitter.resids.residual_objs['dm'].get_data_error().value
        DM0 = np.average(dm_avg, weights=(dm_avg_err)**-2)
        dm_resids += DM0.value
        if restype == 'both':
            dm_resids_init += DM0.value
        if plotsig:
            xlabel = r"DM/Uncertainty"
        else:
            xlabel = r"DM [cm$^{-3}$ pc]"
    else:
        if plotsig:
            xlabel = r"$\Delta$DM/Uncertainty"
        else:
            xlabel = r"$\Delta$DM [cm$^{-3}$ pc]"
    
    if axs == None:
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (10,4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        
        if plotsig:
            sig = dm_resids[inds]/dm_error[inds]
            ax1.hist(sig, nbin, histtype='step', color=colorscheme[r_b_label], label=r_b_label)
            if restype == 'both':
                sig_pre = dm_resids_init[inds]/dm_error_init[inds]
                ax1.hist(sig_pre, nbin, histtype='step', color=colorscheme[r_b_label], linestyle = '--',\
                         label=r_b_label+" Prefit")
        else:
            ax1.hist(dm_resids[inds], nbin, histtype='step', color=colorscheme[r_b_label], label=r_b_label)
            if restype == 'both':
                ax1.hist(dm_resids_init[inds], nbin, histtype='step', color=colorscheme[r_b_label], linestyle = '--',\
                         label=r_b_label+" Prefit")
            
    ax1.grid(True)
    ax1.set_ylabel("Number of measurements")
    ax1.set_xlabel(xlabel)
    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS)/2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, y_offset+1.0/figsize[1]), ncol=ncol)
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title("%s %s DM residual measurements" % (fitter.model.PSR.value, restype), y=y_offset+1.0/figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if restype == 'prefit':
            ext += "_prefit"
        elif restype == 'postfit':
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_DM_resid_measurements%s.png" % (fitter.model.PSR.value, ext))
        
    return


def plot_residuals_orb(fitter, restype = 'postfit', plotsig = False, avg = False, whitened = False, \
                        save = False, legend = True, title = True, axs = None, **kwargs):
    """
    Make a plot of the original_residuals vs. orbital phase.


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of original_residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit original_residuals.
        'postfit' - plot the postfit original_residuals (default)
        'both' - overplot both the pre and post-fit original_residuals.
    plotsig [boolean] : If True plot number of measurements v. original_residuals/uncertainty, else v. original_residuals
        [default: False].
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average original_residuals [default: False].
    whitened [boolean] : If True will compute and plot whitened original_residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    orbphase [list/array] : List or array of orbital phases to plot. Will override values from model object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if wideband
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError("Cannot epoch average wideband original_residuals, please change 'avg' to False.")
    else:
        NB = True

    # Check if want epoch averaged original_residuals
    if avg == True and restype == 'prefit':
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'postfit':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'both':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)

    # Get original_residuals
    if 'res' in kwargs.keys():
        res = kwargs['res']
    else:
        if restype == 'prefit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                    res_pre = avg_dict_pre['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        else:
            raise ValueError("Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."\
                             %(restype))

    # Check if we want whitened original_residuals
    if whitened == True and ('res' not in kwargs.keys()):
        if avg == True:
            if restype != 'both':
                res = whiten_resids(avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype='prefit')
                res_pre = whiten_resids(avg_dict, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
        else:
            if restype != 'both':
                res = whiten_resids(fitter, restype=restype)
            else:
                res = whiten_resids(fitter, restype='prefit')
                res_pre = whiten_resids(fitter, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)

    # Get errors
    if 'errs' in kwargs.keys():
        errs = kwargs['errs']
    else:
        if restype == 'prefit':
            if avg == True:
                errs = avg_dict['errors'].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                    errs_pre = avg_dict_pre['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)

    # Get MJDs
    if 'orbphase' not in kwargs.keys():
        mjds = fitter.toas.get_mjds().value
        if avg == True:
            mjds = avg_dict['mjds'].value

    # Get receiver backends
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
        if avg == True:
            avg_rcvr_bcknds = []
            for iis in avg_dict['indices']:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            rcvr_bcknds = np.array(avg_rcvr_bcknds)
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    # Now we need to the orbital phases; start with binary model name
    if 'orbphase' in kwargs.keys():
        orbphase = kwargs['orbphase']
    else:
        orbphase = fitter.model.orbital_phase(mjds, radians = False)

    if 'figsize' in kwargs.keys():
        figsize = kwargs['figsize']
    else:
        figsize = (10,4)
    if axs == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'fmt' in kwargs.keys():
            mkr = kwargs['fmt']
        else:
            mkr = markers[r_b_label]
            if restype == 'both':
                mkr_pre = '.'
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 0.5
        if plotsig:
            sig = res[inds]/errs[inds]
            ax1.errorbar(orbphase[inds], sig, yerr=len(errs[inds])*[1], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha)
            if restype == 'both':
                sig_pre = res_pre[inds]/errs_pre[inds]
                ax1.errorbar(orbphase[inds], sig_pre, yerr=len(errs_pre[inds])*[1], fmt=mkr_pre, \
                         color=clr, label=r_b_label+" Prefit", alpha = alpha)
        else:
            ax1.errorbar(orbphase[inds], res[inds], yerr = errs[inds], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha)
            if restype == 'both':
                ax1.errorbar(orbphase[inds], res_pre[inds], yerr=errs_pre[inds], fmt=mkr_pre, \
                     color=clr, label=r_b_label+" Prefit", alpha = alpha)
    # Set second axis
    ax1.set_xlabel(r'Orbital Phase')
    ax1.grid(True)
    if plotsig:
        if avg and whitened:
            ax1.set_ylabel('Average Residual/Uncertainty \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel('Average Residual/Uncertainty')
        elif whitened and not avg:
            ax1.set_ylabel('Residual/Uncertainty \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel('Residual/Uncertainty')
    else:
        if avg and whitened:
            ax1.set_ylabel('Average Residual ($\mu$s) \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel('Average Residual ($\mu$s)')
        elif whitened and not avg:
            ax1.set_ylabel('Residual ($\mu$s) \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel('Residual ($\mu$s)')
    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS)/2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, y_offset+1.0/figsize[1]), ncol=ncol)
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title("%s %s timing original_residuals" % (fitter.model.PSR.value, restype), y=y_offset + 1.0 / figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if whitened:
            ext += "_whitened"
        if avg:
            ext += "_averaged"
        if NB:
            ext += "_NB"
        else:
            ext += "_WB"
        if restype == 'prefit':
            ext += "_prefit"
        elif restype == 'postfit':
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_v_orbphase%s.png" % (fitter.model.PSR.value, ext))

    if axs == None:
        # Define clickable points
        text = ax1.text(0,0,"")
        # Define color for highlighting points
        if "430_ASP" in RCVR_BCKNDS or "430_PUPPI" in RCVR_BCKNDS:
            stamp_color = "#61C853"
        else:
            stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data_res_avg
            xdata = orbphase
            if plotsig:
                ydata = (res/errs).decompose().value
            else:
                ydata = res.value
            # Get x and y data_res_avg from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt((xdata - xclick)**2 + ((ydata - yclick)/100.0)**2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax1.scatter(xdata[ind_close], ydata[ind_close], marker = 'x', c = stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            if plotsig:
                text.set_text("TOA Params:\n Phase: %.5f \n Res/Err: %.2f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))
            else:
                text.set_text("TOA Params:\n Phase: %.5f \n Res: %.2f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))

        fig.canvas.mpl_connect('button_press_event', onclick)

    return

def plot_toas_freq(fitter, save = False, legend = True, title = True, axs = None, **kwargs):
    """
    Make a plot of frequency v. toa


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    freqs [list/array] : List or array of TOA frequencies to plot. Will override values from toa object.
    mjds [list/array] : List or array of TOA MJDs to plot. Will override values from toa object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if wideband
    if fitter.is_wideband:
        NB = False
    else:
        NB = True

    # frequencies
    if 'freqs' in kwargs.keys():
        freqs = kwargs['freqs']
    else:
        freqs = fitter.toas.get_freqs().value
    # Get MJDs
    if 'mjds' in kwargs.keys():
        mjds = kwargs['mjds']
    else:
        mjds = fitter.toas.get_mjds().value
    # Convert to years
    years = (mjds - 51544.0)/365.25 + 2000.0

    # Get receiver backends
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    if 'figsize' in kwargs.keys():
        figsize = kwargs['figsize']
    else:
        figsize = (10,4)
    if axs == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'fmt' in kwargs.keys():
            mkr = kwargs['fmt']
        else:
            mkr = markers[r_b_label]
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0
        # Actually make the plot
        ax1.scatter(years[inds], freqs[inds], marker=mkr, color=clr,label=r_b_label, alpha = alpha)
    
    # Set second axis
    ax1.set_xlabel(r'Year')
    ax1.grid(True)
    ax2 = ax1.twiny()
    mjd0  = ((ax1.get_xlim()[0])-2004.0)*365.25+53005.
    mjd1  = ((ax1.get_xlim()[1])-2004.0)*365.25+53005.
    ax2.set_xlim(mjd0, mjd1)
    ax1.set_ylabel(r"Frequency (MHz)")
    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS)/2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, y_offset+1.0/figsize[1]), ncol=ncol)
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title("%s toa frequency v. time" % (fitter.model.PSR.value), y=y_offset+1.0/figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if NB:
            ext += "_NB"
        else:
            ext += "_WB"
        plt.savefig("%s_freq_v_time%s.png" % (fitter.model.PSR.value, ext))
    
    if axs == None:
        # Define clickable points
        text = ax2.text(0,0,"")

        # Define point highlight color
        if "430_ASP" in RCVR_BCKNDS or "430_PUPPI" in RCVR_BCKNDS:
            stamp_color = "#61C853"
        else:
            stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data_res_avg
            xdata = mjds
            ydata = freqs
            # Get x and y data_res_avg from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick)/10.0)**2 + (ydata - yclick)**2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax2.scatter(xdata[ind_close], ydata[ind_close], marker = 'x', c = stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            text.set_text("TOA Params:\n MJD: %s \n Freq: %.2f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))

        fig.canvas.mpl_connect('button_press_event', onclick)
    
    return

def plot_fd_res_v_freq(fitter, plotsig = False, comp_FD = True, avg = False, whitened = False, save = False, \
                        legend = True, title = True, axs = None, **kwargs):
    """
    Make a plot of the original_residuals vs. frequency, can do WB as well. Note, if WB fitter, comp_FD may not work.
    If comp_FD is True, the panels are organized as follows:
    Top: Residuals with FD parameters subtracted and the FD curve plotted as a dashed line.
    Middle: Best fit original_residuals with no FD parameters.
    Bottom: Residuals with FD correction included.
    Note - This function may take a while to run if there are many TOAs.
    
    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    plotsig [boolean] : If True plot number of measurements v. original_residuals/uncertainty, else v. original_residuals
        [default: False].
    comp_FD [boolean]: If True, will plot the original_residuals v. frequency with FD included, FD subtracted, and best fit
        without FD.
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average original_residuals [default: False].
    whitened [boolean] : If True will compute and plot whitened original_residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    freqs [list/array] : List or array of TOA frequencies to plot. Will override values from toa object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if fitter is wideband or not
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError("Cannot epoch average wideband original_residuals, please change 'avg' to False.")
    else:
        NB = True

    # Check if want epoch averaged original_residuals
    if avg:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)

    # Get original_residuals
    if 'res' in kwargs.keys():
        res = kwargs['res']
    else:
        if NB == True:
            if avg == True:
                res = avg_dict['time_resids'].to(u.us)
            else:
                res = fitter.resids.time_resids.to(u.us)
        else:
            res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)

    # Check if we want whitened original_residuals
    if whitened == True and ('res' not in kwargs.keys()):
        if avg == True:
            res = whiten_resids(avg_dict)
            res = res.to(u.us)
        else:
            res = whiten_resids(fitter)
            res = res.to(u.us)
    
    # Get errors
    if 'errs' in kwargs.keys():
        errs = kwargs['errs']
    else:
        if NB == True:
            if avg == True:
                errs = avg_dict['errors'].to(u.us)
            else:
                errs = fitter.resids.get_data_error().to(u.us)
        else:
            errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)

    # Get receiver backends
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
        if avg == True:
            avg_rcvr_bcknds = []
            for iis in avg_dict['indices']:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            rcvr_bcknds = np.array(avg_rcvr_bcknds)
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)
    
    # get frequencies
    if 'freqs' in kwargs.keys():
        freqs = kwargs['freqs']
    else:
        if avg == True:
            freqs = avg_dict['freqs'].value
        else:
            freqs = fitter.toas.get_freqs().value
    
    # Check if comparing the FD parameters
    if comp_FD:
        if axs != None:
            log.warn("Cannot do full comparison with three panels")
            axs = None
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (4,12)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(313)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(311)
    else:
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (4,4)
        if axs == None:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(111)
        else:
            ax1 = axs

    # Make the plot of residual vs. frequency
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'fmt' in kwargs.keys():
            mkr = kwargs['fmt']
        else:
            mkr = markers[r_b_label]
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 1.0
        if plotsig:
            sig = res[inds]/errs[inds]
            ax1.errorbar(freqs[inds], sig, yerr=len(errs[inds])*[1], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha)
        else:
            ax1.errorbar(freqs[inds], res[inds], yerr=errs[inds], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha)
        # assign axis labels
        ax1.set_xlabel(r'Frequency (MHz)')
        ax1.grid(True)
        if plotsig:
            if avg and whitened:
                ylabel = 'Average Residual/Uncertainty \n (Whitened)'
            elif avg and not whitened:
                ylabel = 'Average Residual/Uncertainty'
            elif whitened and not avg:
                ylabel ='Residual/Uncertainty \n (Whitened)'
            else:
                ylabel ='Residual/Uncertainty'
        else:
            if avg and whitened:
                ylabel = 'Average Residual ($\mu$s) \n (Whitened)'
            elif avg and not whitened:
                ylabel = 'Average Residual ($\mu$s)'
            elif whitened and not avg:
                ylabel = 'Residual ($\mu$s) \n (Whitened)'
            else:
                ylabel = 'Residual ($\mu$s)'
        ax1.set_ylabel(ylabel)

    # Now if we want to show the other plots, we plot them
    if comp_FD:
        # Plot the original_residuals with FD parameters subtracted
        cur_fd = [param for param in fitter.model.params if "FD" in param]
        # Get the FD offsets
        FD_offsets = np.zeros(np.size(res))
        # Also need FD line
        sorted_freqs = np.linspace(np.min(freqs), np.max(freqs), 1000)
        FD_line = np.zeros(np.size(sorted_freqs))
        for i, fd in enumerate(cur_fd):
            fd_val = getattr(fitter.model, fd).value * 10**6 # convert to microseconds
            FD_offsets += fd_val * np.log(freqs/1000.0)**(i+1)
            FD_line += fd_val * np.log(sorted_freqs/1000.0)**(i+1)
        # Now edit original_residuals
        fd_cor_res = res.value + FD_offsets

        # Now we need to redo the fit without the FD parameters
        psr_fitter_nofd = copy.deepcopy(fitter)
        try:
            psr_fitter_nofd.model.remove_component('FD')
        except:
            log.warning("No FD parameters in the initial timing model...")

        # Check if fitter is wideband or not
        if psr_fitter_nofd.is_wideband:
            resids = psr_fitter_nofd.resids.residual_objs['toa']
        else:
            resids = psr_fitter_nofd.resids

        psr_fitter_nofd.fit_toas(1)
        # Now we need to figure out if these need to be whitened and/or averaged
        if avg:
            avg = psr_fitter_nofd.resids.ecorr_average(use_noise_model=True)
            avg_rcvr_bcknds = rcvr_bcknds
            if whitened:
                # need to whiten and average
                wres_avg = whiten_resids(avg)
                res_nofd = wres_avg.to(u.us).value
            else:
                # need to average
                res_nofd = avg['time_resids'].to(u.us).value
        elif whitened:
            # Need to whiten
            wres_nofd = whiten_resids(psr_fitter_nofd)
            res_nofd = wres_nofd.to(u.us).value
        else:
            res_nofd = resids.time_resids.to(u.us).value

        # Now plot
        for i, r_b in enumerate(RCVR_BCKNDS):
            inds = np.where(rcvr_bcknds==r_b)[0]
            if not inds.tolist():
                r_b_label = ""
            else:
                r_b_label = rcvr_bcknds[inds][0]
            # Get plot preferences
            if 'fmt' in kwargs.keys():
                mkr = kwargs['fmt']
            else:
                mkr = markers[r_b_label]
            if 'color' in kwargs.keys():
                clr = kwargs['color']
            else:
                clr = colorscheme[r_b_label]
            if 'alpha' in kwargs.keys():
                alpha = kwargs['alpha']
            else:
                alpha = 1.0
            if plotsig:
                sig = fd_cor_res[inds]/errs[inds]
                ax3.errorbar(freqs[inds], sig.value, yerr=len(errs[inds])*[1], fmt=mkr, \
                         color=clr, label=r_b_label, alpha = alpha)

                sig_nofd = res_nofd[inds]/errs[inds].value
                ax2.errorbar(freqs[inds], sig_nofd, yerr=len(errs[inds])*[1], fmt=mkr, \
                         color=clr, label=r_b_label, alpha = alpha)
            else:
                ax3.errorbar(freqs[inds], fd_cor_res[inds], yerr=errs[inds].value, fmt=mkr, \
                         color=clr, label=r_b_label, alpha = alpha)

                ax2.errorbar(freqs[inds], res_nofd[inds], yerr=errs[inds].value, fmt=mkr, \
                         color=clr, label=r_b_label, alpha = alpha)

            ax3.plot(sorted_freqs, FD_line, c = 'k', ls = '--')
            # assign axis labels
            ax3.set_xlabel(r'Frequency (MHz)')
            ax3.set_ylabel(ylabel)
            ax3.grid(True)
            ax2.set_xlabel(r'Frequency (MHz)')
            ax2.set_ylabel(ylabel)
            ax2.grid(True)

    if legend:
        if comp_FD:
            ax3.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=int(len(RCVR_BCKNDS)/2))
        else:
            ax1.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=int(len(RCVR_BCKNDS)/2))
    if title:
        plt.title("%s FD Paramter Check" % (fitter.model.PSR.value), y=1.0+1.0/figsize[1])
    plt.tight_layout()
    if save:
        ext = ""
        if whitened:
            ext += "_whitened"
        if avg:
            ext += "_averaged"
        if NB:
            ext += "_NB"
        else:
            ext += "_WB"
        plt.savefig("%s_FD_resid_v_freq%s.png" % (fitter.model.PSR.value, ext))
    return


"""
We also offer some options for convenience plotting functions, one that will show all possible summary plots, and
another that will show just the summary plots that are typically created in finalize_timing.py in that order.
"""
def summary_plots(fitter, title = None, legends = False, save = False, avg = True, whitened = True):
    """
    Function to make a composite set of summary plots for sets of TOAs.
    NOTE - This is noe the same set of plots as will be in the pdf writer

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    title [boolean] : If True, will add titles to ALL plots [default: False].
    legend [boolean] : If True, will add legends to ALL plots [default: False].
    save [boolean] : If True will save plot with the name "psrname_summary.png" Will add averaged/whitened
         as necessary [default: False].
    avg [boolean] : If True and not wideband fitter, will make plots of epoch averaged original_residuals [default: True].
    whitened [boolean] : If True will make plots of whitened original_residuals [default: True].
    """

    if fitter.is_wideband:
        if avg == True:
            raise ValueError("Cannot epoch average wideband original_residuals, please change 'avg' to False.")
    # Determine how long the figure size needs to be
    figlength = 18
    gs_rows = 6
    if whitened:
        figlength += 18
        gs_rows += 4
    if avg:
        figlength += 18
        gs_rows += 4
    if whitened and avg:
        figlength += 18
        gs_rows += 4
    # adjust size if not in a binary
    if not hasattr(fitter.model, 'binary_model_name'):
        sub_rows = 1
        sub_len = 3
        if whitened:
            sub_rows += 1
            sub_len += 3
        if avg:
            sub_rows += 1
            sub_len += 3
        if whitened and avg:
            sub_rows += 1
            sub_len += 3
        figlength -= sub_len
        gs_rows -= sub_rows

    fig = plt.figure(figsize = (12,figlength)) # not sure what we'll need for a fig size
    if title != None:
        plt.title(title, y = 1.015, size = 16)
    gs = fig.add_gridspec(gs_rows, 2)

    count = 0
    k = 0
    # First plot is all original_residuals vs. time.
    ax0 = fig.add_subplot(gs[count, :])
    plot_residuals_time(fitter, title = False, axs = ax0, figsize=(12,3))
    k += 1

    # Plot the original_residuals divided by uncertainty vs. time
    ax1 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(fitter, title = False, legend = False, plotsig = True, axs = ax1, figsize=(12,3))
    k += 1

    # Second plot is residual v. orbital phase
    if hasattr(fitter.model, 'binary_model_name'):
        ax2 = fig.add_subplot(gs[count+k, :])
        plot_residuals_orb(fitter, title = False, legend = False, axs = ax2, figsize=(12,3))
        k += 1

    # Now add the measurement vs. uncertainty
    ax3_0 = fig.add_subplot(gs[count+k, 0])
    ax3_1 = fig.add_subplot(gs[count+k, 1])
    plot_measurements_v_res(fitter, nbin = 50, plotsig=False, title = False, legend = False, axs = ax3_0, \
                            figsize=(6,3),)
    plot_measurements_v_res(fitter, nbin = 50, plotsig=True, title = False, legend = False, axs = ax3_1, \
                            figsize=(6,3),)
    k += 1

    # and the DMX vs. time
    ax4 = fig.add_subplot(gs[count+k, :])
    plot_dmx_time(fitter, savedmx = True, legend = False, title = False, axs = ax4, figsize=(12,3))
    k += 1

    # And residual vs. Frequency
    ax5 = fig.add_subplot(gs[count+k, :])
    plot_toas_freq(fitter, title = False, legend = False, axs =ax5,  figsize=(12,3))
    k += 1

    # Now if whitened add the whitened residual plots
    if whitened:
        ax6 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, title = False, whitened = True, axs = ax6, figsize=(12,3))
        k += 1

        # Plot the original_residuals divided by uncertainty vs. time
        ax7 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, title = False, legend = False, plotsig = True, whitened = True, axs = ax7, figsize=(12,3))
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(fitter.model, 'binary_model_name'):
            ax8 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, whitened = True, axs = ax8,  figsize=(12,3))
            k += 1

        # Now add the measurement vs. uncertainty
        ax9_0 = fig.add_subplot(gs[count+k, 0])
        ax9_1 = fig.add_subplot(gs[count+k, 1])
        plot_measurements_v_res(fitter, nbin = 50, plotsig=False, title = False, legend = False, whitened = True,\
                           axs = ax9_0, figsize=(6,3),)
        plot_measurements_v_res(fitter, nbin = 50, plotsig=True, title = False, legend = False, whitened = True,\
                           axs = ax9_1, figsize=(6,3),)
        k += 1

    # Now plot the average original_residuals
    if avg:
        ax10 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, title = False, avg = True, axs = ax10, figsize=(12,3))
        k += 1

        # Plot the original_residuals divided by uncertainty vs. time
        ax11 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, title = False, legend = False, plotsig = True, avg = True, axs = ax11, figsize=(12,3))
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(fitter.model, 'binary_model_name'):
            ax12 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, avg = True, axs = ax12, figsize=(12,3))
            k += 1

        # Now add the measurement vs. uncertainty
        ax13_0 = fig.add_subplot(gs[count+k, 0])
        ax13_1 = fig.add_subplot(gs[count+k, 1])
        plot_measurements_v_res(fitter, nbin = 50, plotsig=False, title = False, legend = False,\
                                avg = True, axs = ax13_0, figsize=(6,3))
        plot_measurements_v_res(fitter, nbin = 50, plotsig=True, title = False, legend = False, \
                                avg = True, axs = ax13_1, figsize=(6,3))
        k += 1

    # Now plot the whitened average original_residuals
    if avg and whitened:
        ax14 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, avg = True, whitened = True, axs = ax14, figsize=(12,3))
        k += 1

        # Plot the original_residuals divided by uncertainty vs. time
        ax15 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, title = False, legend = False, plotsig = True, avg = True, whitened = True,\
                            axs = ax15, figsize=(12,3))
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(fitter.model, 'binary_model_name'):
            ax16 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, avg = True, whitened = True, axs = ax16, \
                               figsize=(12,3))
            k += 1

        # Now add the measurement vs. uncertainty
        ax17_0 = fig.add_subplot(gs[count+k, 0])
        ax17_1 = fig.add_subplot(gs[count+k, 1])
        plot_measurements_v_res(fitter, nbin = 50, plotsig=False, title = False, legend = False, avg = True, whitened = True, \
                                axs = ax17_0, figsize=(6,3))
        plot_measurements_v_res(fitter, nbin = 50, plotsig=True, title = False, legend = False, avg = True, whitened = True, \
                                axs = ax17_1, figsize=(6,3))
        k += 1

    plt.tight_layout()
    if save:
        plt.savefig("%s_summary_plots.png" % (fitter.model.PSR.value))

    return

"""We also define a function to output the summary plots exactly as is done in finalize_timing.py (for now)"""
def summary_plots_ft(fitter, title = None, legends = False, save = False):
    """
    Function to make a composite set of summary plots for sets of TOAs
    NOTE - This is note the same set of plots as will be in the pdf writer

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    title [boolean] : If True, will add titles to ALL plots [default: False].
    legend [boolean] : If True, will add legends to ALL plots [default: False].
    save [boolean] : If True will save plot with the name "psrname_summary.png" Will add averaged/whitened
         as necessary [default: False].
    """
    # Define the figure
    # Determine how long the figure size needs to be
    figlength = 18*3
    gs_rows = 13
    if not hasattr(fitter.model, 'binary_model_name'):
        figlength -= 9
        gs_rows -= 3
    if fitter.is_wideband:
        figlength -= 9
        gs_rows -= 3

    fig = plt.figure(figsize = (12,figlength)) # not sure what we'll need for a fig size
    if title != None:
        plt.title(title, y = 1.015, size = 16)
    gs = fig.add_gridspec(gs_rows, 2)

    count = 0
    k = 0
    # First plot is all original_residuals vs. time.
    ax0 = fig.add_subplot(gs[count, :])
    plot_residuals_time(fitter, title = False, axs = ax0, figsize=(12,3))
    k += 1

    # Then the epoch averaged original_residuals v. time
    if not fitter.is_wideband:
        ax10 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, title = False, legend = False, avg = True, axs = ax10, figsize=(12,3))
        k += 1

    # Epoch averaged vs. orbital phase
    if hasattr(fitter.model, 'binary_model_name'):
        if not fitter.is_wideband:
            ax12 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, avg = True, axs = ax12, figsize=(12,3))
            k += 1
        else:
            ax12 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, axs = ax12, figsize=(12,3))
            k += 1

    # And DMX vs. time
    ax4 = fig.add_subplot(gs[count+k, :])
    plot_dmx_time(fitter, savedmx = True, legend = False, title = False, axs = ax4, figsize=(12,3))
    k += 1

    # Whitened original_residuals v. time
    ax6 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(fitter, whitened = True, axs = ax6, figsize=(12,3))
    k += 1

    # Whitened epoch averaged original_residuals v. time
    if not fitter.is_wideband:
        ax15 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, title = False, legend = False, plotsig = False, avg = True, \
                            whitened = True, axs = ax15, figsize=(12,3))
        k += 1

    # Whitened epoch averaged original_residuals v. orbital phase
    if hasattr(fitter.model, 'binary_model_name'):
        if not fitter.is_wideband:
            ax16 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, \
                                 avg = True, whitened = True, axs = ax16, figsize=(12,3))
            k += 1
        else:
            ax16 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, \
                                 avg = False, whitened = True, axs = ax16, figsize=(12,3))
            k += 1

    # Now add the measurement vs. uncertainty for both all reaiduals and epoch averaged
    ax3_0 = fig.add_subplot(gs[count+k, 0])
    ax3_1 = fig.add_subplot(gs[count+k, 1])
    plot_measurements_v_res(fitter, nbin = 50, title = False, legend = False, plotsig=False, \
                            whitened = True, axs = ax3_0, figsize=(6,3))
    if not fitter.is_wideband:
        plot_measurements_v_res(fitter, nbin = 50, title = False, legend = False, avg = True, \
                                whitened = True, axs = ax3_1, figsize=(6,3))
        k += 1
    else:
        plot_measurements_v_res(fitter, nbin = 50, title = False, legend = False, avg = False, \
                                whitened = False, axs = ax3_1, figsize=(6,3))
        k += 1

    # Whitened residual/uncertainty v. time
    ax26 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(fitter, plotsig = True, title = False, legend = False, whitened = True,\
                        axs = ax26, figsize=(12,3))
    k += 1

    # Epoch averaged Whitened residual/uncertainty v. time
    if not fitter.is_wideband:
        ax25 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(fitter, title = False, legend = False, plotsig = True, \
                            avg = True, whitened = True, axs = ax25, figsize=(12,3))
        k += 1

    # Epoch averaged Whitened residual/uncertainty v. orbital phase
    if hasattr(fitter.model, 'binary_model_name'):
        if not fitter.is_wideband:
            ax36 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, plotsig = True, avg = True, \
                               whitened = True, axs = ax36,  figsize=(12,3))
            k += 1
        else:
            ax36 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(fitter, title = False, legend = False, plotsig = True, avg = False, \
                               whitened = True, axs = ax36,  figsize=(12,3))
            k += 1

    # Now add the measurement vs. uncertainty for both all reaiduals/uncertainty and epoch averaged/uncertainty
    ax17_0 = fig.add_subplot(gs[count+k, 0])
    ax17_1 = fig.add_subplot(gs[count+k, 1])
    plot_measurements_v_res(fitter, nbin = 50, plotsig=True, title = False, legend = False, whitened = True,\
                           axs = ax17_0, figsize=(6,3))
    if not fitter.is_wideband:
        plot_measurements_v_res(fitter, nbin = 50, title = False, plotsig=True, \
                            legend = False, avg = True, whitened = True, axs = ax17_1, figsize=(6,3))
        k += 1
    else:
        plot_measurements_v_res(fitter, nbin = 50, title = False, plotsig=True, \
                            legend = False, avg = False, whitened =False, axs = ax17_1, figsize=(6,3))
        k += 1

    # Now plot the frequencies of the TOAs vs. time
    ax5 = fig.add_subplot(gs[count+k, :])
    plot_toas_freq(fitter, title = False, legend = False, axs =ax5, figsize=(12,3))
    k += 1

    plt.tight_layout()
    if save:
        plt.savefig("%s_summary_plots_FT.png" % (fitter.model.PSR.value))

    return

# JUST THE PLOTS FOR THE PDF WRITERS LEFT
def plots_for_summary_pdf_nb(fitter, title = None, legends = False):
    """
    Function to make a composite set of summary plots for sets of TOAs to be put into a summary pdf.
    This is for Narrowband timing only. For Wideband timing, use `plots_for_summary_pdf_wb`.
    By definition, this function will save all plots as "psrname"_summary_plot_#.nb.png, where # is
    and integer from 1-4. 

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    title [boolean] : If True, will add titles to ALL plots [default: False].
    legend [boolean] : If True, will add legends to ALL plots [default: False].
    """
    
    if fitter.is_wideband:
        raise ValueError("Cannot use this function with WidebandTOAFitter, please use `plots_for_summary_pdf_wb` instead.")
    # Need to make four sets of plots
    for ii in range(4):
        if ii != 3:
            fig = plt.figure(figsize=(8,10.0),dpi=100)
        else:
            fig = plt.figure(figsize=(8,5),dpi=100)
        if title != None:
            plt.title(title, y = 1.08, size = 14)
        if ii == 0:
            gs = fig.add_gridspec(nrows = 4, ncols = 1)

            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            ax2 = fig.add_subplot(gs[2,:])
            ax3 = fig.add_subplot(gs[3,:])
            # Plot original_residuals v. time
            plot_residuals_time(fitter, title = False, axs = ax0, figsize=(8, 2.5))
            # Plot averaged original_residuals v. time
            if 'ecorr_noise' in fitter.model.get_components_by_category().keys():
                plot_residuals_time(fitter, avg = True, axs = ax1, title = False, legend = False, figsize=(8,2.5))
            else:
                log.warning(
                    "ECORR not in model, cannot generate epoch averaged original_residuals. Plots will show all original_residuals.")
                plot_residuals_time(fitter, avg=False, axs=ax1, title=False, legend=False, figsize=(8, 2.5))
            # Plot original_residuals v orbital phase
            if hasattr(fitter.model, 'binary_model_name'):
                if 'ecorr_noise' in fitter.model.get_components_by_category().keys():
                    plot_residuals_orb(fitter, title = False, legend = False, avg = True, axs = ax2, figsize=(8,2.5))
                else:
                    plot_residuals_orb(fitter, title = False, legend = False, avg = False, axs = ax2, figsize=(8,2.5))
            # plot dmx v. time
            if 'dispersion_dmx' in fitter.model.get_components_by_category().keys():
                plot_dmx_time(fitter, savedmx = True, legend = False, title = False, axs = ax3,  figsize=(8,2.5))
            else:
                log.warning("No DMX bins in timing model, cannot plot DMX v. Time.")
            plt.tight_layout()
            plt.savefig("%s_summary_plot_1_nb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 1:
            if hasattr(fitter.model, 'binary_model_name'):
                gs = fig.add_gridspec(4,2)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,0])
                ax4 = fig.add_subplot(gs[3,1])
            else:
                gs = fig.add_gridspec(3,2)
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            # plot whitened original_residuals v time
            plot_residuals_time(fitter, title = False, whitened = True, axs = ax0, figsize=(8,2.5))
            # plot whitened, epoch averaged original_residuals v time
            if 'ecorr_noise' in fitter.model.get_components_by_category().keys():
                plot_residuals_time(fitter, title = False, legend = False, avg = True, \
                            whitened = True, axs = ax1, figsize=(8,2.5))
            else:
                plot_residuals_time(fitter, title = False, legend = False, avg = False, \
                            whitened = True, axs = ax1, figsize=(8,2.5))
            # Plot whitened, epoch averaged original_residuals v orbital phase
            if hasattr(fitter.model, 'binary_model_name'):
                if 'ecorr_noise' in fitter.model.get_components_by_category().keys():
                    plot_residuals_orb(fitter, title = False, legend = False, avg = True, whitened = True, \
                                   axs = ax2, figsize=(8,2.5))
                else:
                    plot_residuals_orb(fitter, title = False, legend = False, avg = False, whitened = True, \
                                   axs = ax2, figsize=(8,2.5))
            # plot number of whitened original_residuals histogram
            plot_measurements_v_res(fitter, nbin = 50, title = False, legend = False, whitened = True,\
                           axs = ax3, figsize=(4,2.5))
            # plot number of whitened, epoch averaged original_residuals histogram
            if 'ecorr_noise' in fitter.model.get_components_by_category().keys():
                plot_measurements_v_res(fitter, nbin = 50, title = False, legend = False, avg = True, whitened = True, \
                                    axs = ax4, figsize=(4,2.5))
            else:
                plot_measurements_v_res(fitter, nbin = 50, title = False, legend = False, avg = False, whitened = True, \
                                    axs = ax4, figsize=(4,2.5))
            plt.tight_layout()
            plt.savefig("%s_summary_plot_2_nb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 2:
            if hasattr(fitter.model, 'binary_model_name'):
                gs = fig.add_gridspec(4,2)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,0])
                ax4 = fig.add_subplot(gs[3,1])
            else:
                gs = fig.add_gridspec(3,2)
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            # plot whitened original_residuals/uncertainty v. time
            plot_residuals_time(fitter, plotsig = True, title = False, whitened = True, axs = ax0, figsize=(8,2.5))
            # plot whitened, epoch averaged original_residuals/uncertainty v. time
            if 'ecorr_noise' in fitter.model.get_components_by_category().keys():
                plot_residuals_time(fitter, title = False, legend = False, plotsig = True, avg = True,\
                                whitened = True, axs = ax1, figsize=(8,2.5))
            else:
                plot_residuals_time(fitter, title = False, legend = False, plotsig = True, avg = False,\
                                whitened = True, axs = ax1, figsize=(8,2.5))
            # plot whitened, epoch averaged original_residuals/uncertainty v. orbital phase
            if hasattr(fitter.model, 'binary_model_name'):
                if 'ecorr_noise' in fitter.model.get_components_by_category().keys():
                    plot_residuals_orb(fitter, title = False, legend = False, plotsig = True, \
                            avg = True, whitened = True, axs = ax2, figsize=(8,2.5))
                else:
                    plot_residuals_orb(fitter, title = False, legend = False, plotsig = True, \
                            avg = False, whitened = True, axs = ax2, figsize=(8,2.5))
            # plot number of whitened original_residuals/uncertainty histogram
            plot_measurements_v_res(fitter, nbin = 50, plotsig=True, title = False, legend = False, whitened = True,\
                           axs = ax3, figsize=(4,2.5))
            # plot number of whitened, epoch averaged original_residuals/uncertainties histogram
            if 'ecorr_noise' in fitter.model.get_components_by_category().keys():
                plot_measurements_v_res(fitter, nbin = 50, plotsig=True, title = False, legend = False, \
                                    avg = True, whitened = True, axs = ax4, figsize=(4,2.5))
            else:
                plot_measurements_v_res(fitter, nbin = 50, plotsig=True, title = False, legend = False, \
                                    avg = False, whitened = True, axs = ax4, figsize=(4,2.5))
            plt.tight_layout()
            plt.savefig("%s_summary_plot_3_nb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 3:
            gs = fig.add_gridspec(1,1)
            ax0 = fig.add_subplot(gs[0])
            plot_toas_freq(fitter, title = False, legend = True, axs =ax0, figsize=(8,4))
            plt.tight_layout()
            plt.savefig("%s_summary_plot_4_nb.png" % (fitter.model.PSR.value))
            plt.close()

def plots_for_summary_pdf_wb(fitter, title = None, legends = False):
    """
    Function to make a composite set of summary plots for sets of TOAs to be put into a summary pdf.
    This is for Wideband timing only. For Narrowband timing, use `plots_for_summary_pdf_nb`.
    By definition, this function will save all plots as "psrname"_summary_plot_#.wb.png, where # is
    and integer from 1-4. 

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    title [boolean] : If True, will add titles to ALL plots [default: False].
    legend [boolean] : If True, will add legends to ALL plots [default: False].
    """
    if not fitter.is_wideband:
        raise ValueError("Cannot use this function with non-WidebandTOAFitter, please use `plots_for_summary_pdf_nb` instead.")
    # Need to make four sets of plots
    for ii in range(4):
        if ii != 3:
            fig = plt.figure(figsize=(8,10.0),dpi=100)
        else:
            fig = plt.figure(figsize=(8,5),dpi=100)
        if title != None:
            plt.title(title, y = 1.08, size = 14)
        if ii == 0:
            if hasattr(fitter.model, 'binary_model_name'):
                gs = fig.add_gridspec(nrows = 4, ncols = 1)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,:])
            else:
                gs = fig.add_gridspec(nrows = 3, ncols = 1)
                ax3 = fig.add_subplot(gs[2,:])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            # Plot time original_residuals v. time
            plot_residuals_time(fitter, title = False, axs = ax0, figsize=(8, 2.5))
            # Plot DM original_residuals v. time
            plot_dm_residuals(fitter, save = False, legend = False, title = False, axs = ax1, figsize=(8, 2.5))
            # Plot time original_residuals v. orbital phase
            if hasattr(fitter.model, 'binary_model_name'):
                plot_residuals_orb(fitter, title = False, legend = False, axs = ax2, figsize=(8,2.5))
            plot_dmx_time(fitter, savedmx = True, legend = False, title = False, axs = ax3, figsize=(8,2.5))
            plt.tight_layout()
            plt.savefig("%s_summary_plot_1_wb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 1:
            if hasattr(fitter.model, 'binary_model_name'):
                gs = fig.add_gridspec(3,2)
                ax2 = fig.add_subplot(gs[1,:])
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
            else:
                gs = fig.add_gridspec(2,2)
                ax3 = fig.add_subplot(gs[1,0])
                ax4 = fig.add_subplot(gs[1,1])
            ax0 = fig.add_subplot(gs[0,:])
            # ax1 = fig.add_subplot(gs[1,:])
            # Plot whitened time original_residuals v. time
            plot_residuals_time(fitter, title = False, whitened = True, axs = ax0, figsize=(8,2.5))
            # Plot whitened time original_residuals v. time
            if hasattr(fitter.model, 'binary_model_name'):
                plot_residuals_orb(fitter, title = False, legend = False, whitened = True, axs = ax2, figsize=(8,2.5))
            # Plot number of whitened original_residuals histograms
            plot_measurements_v_res(fitter, nbin = 50, title = False, plotsig=False, legend = False, whitened = True,\
                           axs = ax3, figsize=(4,2.5))
            # plot number of DM original_residuals histograms
            plot_measurements_v_dmres(fitter, nbin = 50, legend = False, title = False, axs = ax4)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_2_wb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 2:
            if hasattr(fitter.model, 'binary_model_name'):
                gs = fig.add_gridspec(4,2)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,0])
                ax4 = fig.add_subplot(gs[3,1])
            else:
                gs = fig.add_gridspec(3,2)
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            # plot whitened time original_residuals/uncertainty v time
            plot_residuals_time(fitter, plotsig = True, title = False, whitened = True, axs = ax0, figsize=(8,2.5))
            # Plot DM original_residuals/uncertainty v. time
            plot_dm_residuals(fitter, plotsig = True, save = False, legend = False, title = False, axs = ax1, figsize=(8, 2.5))
            # Plot whitened time original_residuals/uncertainty v orbital phase
            if hasattr(fitter.model, 'binary_model_name'):
                plot_residuals_orb(fitter, title = False, legend = False, \
                           plotsig = True, whitened = True, axs = ax2,  figsize=(8,2.5))
            # plot number of whitened time original_residuals/uncertainty histograms
            plot_measurements_v_res(fitter, nbin = 50, title = False, plotsig=True, legend = False, whitened = True,\
                           axs = ax3,  figsize=(4,2.5))
            # plot number of DM original_residuals/uncertainty histograms
            plot_measurements_v_dmres(fitter, plotsig = True, nbin = 50, legend = False, title = False, \
                                      axs = ax4)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_3_wb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 3:
            gs = fig.add_gridspec(1,1)
            ax0 = fig.add_subplot(gs[0])
            plot_toas_freq(fitter, title = False, legend = True, axs =ax0, figsize=(8,4))
            plt.tight_layout()
            plt.savefig("%s_summary_plot_4_wb.png" % (fitter.model.PSR.value))
            plt.close()
